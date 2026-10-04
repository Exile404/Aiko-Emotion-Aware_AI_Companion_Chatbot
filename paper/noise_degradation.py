#!/usr/bin/env python3
"""Controlled degradation: does added noise reproduce the clean-to-wild voice collapse?

Table C1-11 attributes the IEMOCAP-to-MELD drop to "clean -> degraded audio", but that
step is a comparison between two corpora that differ in many ways besides the audio.
This script MANIPULATES the audio instead: it takes held-out IEMOCAP session 5 (clean,
1,241 common-four utterances) and adds noise at controlled SNRs, holding speakers,
labels and content fixed. If audio degradation is what defeats the voice modality, voice
accuracy should fall toward MELD's level as SNR drops.

The full deployed path is run on every degraded clip, because in deployment noise hurts
BOTH modalities: the words reach the text model only through speech recognition.

    degraded audio -> emotion2vec+                          (deployed voice)
                   -> WavLM-large + probe                   (clean voice)
                   -> Whisper small -> DistilRoBERTa        (deployed text, on ASR words)
                                    -> MiniLM + probe       (clean text, on ASR words)

Clean probes are trained on sessions 1-4 only, with the layer and regularisation frozen
in paper/clean_probe_config.json (selected on MELD dev), so session 5 is never seen.

Noise types, each mixed at whole-utterance SNR in {20, 10, 5, 0, -5} dB:
  babble    six utterances from sessions 1-4 summed; never a session-5 speaker
  laughter  ESC-50 "laughing" clips, resampled to 16 kHz; the laugh-track analogue
  steady    synthetic pink noise
The noise realisation for a clip is seeded by clip and type, so across SNR levels only
its scale changes and the dose-response is not confounded by noise sampling.

Whisper runs as a batched greedy decode with language fixed to English. This matched
aiko/stt.py's transcribe() on spot checks but omits its temperature fallback, which can
fire more often under heavy noise; the paper should state this.

Two calibrations tie the controlled noise back to MELD:
  - the blind SNR proxy (snr_proxy.py) on each degraded condition, which also checks
    the proxy is monotone in added noise;
  - Whisper word error rate on clean MELD test against the same measure under each
    degradation, an independent read of acoustic difficulty.

Caches (with ASR text, which reproduces licensed corpus content) go to the gitignored
paper/noise_cache/. Only aggregate numbers reach paper/noise_results.md.

funasr leaks RAM, so run in chunks, then MELD's word error rate, then the report:

    while :; do .venv/bin/python paper/noise_degradation.py run --max-new 1300; \\
        [ $? -ne 3 ] && break; done
    .venv/bin/python paper/noise_degradation.py meld-wer
    .venv/bin/python paper/noise_degradation.py report
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

from snr_proxy import snr_proxy

COMMON4 = ["angry", "happy", "neutral", "sad"]
IEMOCAP_ROOT = "IEMOCAP_full_release/IEMOCAP_full_release"
ESC50 = "data/ESC-50-master"
TEST_SESSION = 5
NOISES = ["babble", "laughter", "steady"]
SNRS = [20, 10, 5, 0, -5]
CACHE = "paper/noise_cache/iemocap_s5.jsonl"
MELD_CACHE = "paper/noise_cache/meld_test_asr.jsonl"
OUT = "paper/noise_results.md"
BATCH = 32
LAYERS = [3, 6, 9, 12, 15, 18, 21, 24]          # must match clean_extract_features.py


# ----------------------------------------------------------------------------- data
def seed_of(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def load_wav(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{path}: {sr} Hz")
    return x


def iemocap_rows() -> list[dict]:
    """Common-four IEMOCAP rows with wav paths and gold transcripts (local backup)."""
    rows = []
    for r in (json.loads(l) for l in open("paper/.text_backup/iemocap_scores.jsonl",
                                          encoding="utf-8")):
        if r.get("gold") not in COMMON4:
            continue
        r["wav"] = os.path.join(IEMOCAP_ROOT, f"Session{r['session']}", "sentences", "wav",
                                r["dialog"], f"{r['clip']}.wav")
        rows.append(r)
    return rows


def conditions() -> list[tuple[str, float | None]]:
    return [("clean", None)] + [(n, s) for n in NOISES for s in SNRS]


# ---------------------------------------------------------------------------- noise
class NoiseBank:
    def __init__(self, pool_rows: list[dict]):
        rng = np.random.default_rng(0)
        pick = rng.choice(len(pool_rows), size=min(600, len(pool_rows)), replace=False)
        self.babble_pool = [load_wav(pool_rows[i]["wav"]) for i in pick]
        self.laughs = self._laughs()

    @staticmethod
    def _laughs() -> list[np.ndarray]:
        from scipy.signal import resample_poly
        meta = open(os.path.join(ESC50, "meta", "esc50.csv"), encoding="utf-8").read().splitlines()
        files = [l.split(",")[0] for l in meta[1:] if l.split(",")[3] == "laughing"]
        out = []
        for fn in files:
            x, sr = sf.read(os.path.join(ESC50, "audio", fn), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            out.append(resample_poly(x, 160, 441).astype(np.float32) if sr == 44100 else x)
        return out

    @staticmethod
    def _fill(segments: list[np.ndarray], n: int, rng) -> np.ndarray:
        buf, have = [], 0
        while have < n + 16000:
            s = segments[rng.integers(len(segments))]
            buf.append(s)
            have += len(s)
        x = np.concatenate(buf)
        start = rng.integers(0, len(x) - n)
        return x[start:start + n]

    def make(self, kind: str, n: int, clip: str) -> np.ndarray:
        rng = np.random.default_rng(seed_of(clip, kind))
        if kind == "babble":
            mix = np.zeros(n, dtype=np.float64)
            for _ in range(6):
                s = self._fill(self.babble_pool, n, rng).astype(np.float64)
                mix += s / (np.sqrt(np.mean(s ** 2)) + 1e-9)
            return mix.astype(np.float32)
        if kind == "laughter":
            return self._fill(self.laughs, n, rng)
        if kind == "steady":
            spec = np.fft.rfft(rng.standard_normal(n))
            f = np.arange(len(spec), dtype=np.float64)
            f[0] = 1.0
            return np.fft.irfft(spec / np.sqrt(f), n).astype(np.float32)
        raise ValueError(kind)


def mix_at(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps = np.mean(x.astype(np.float64) ** 2) + 1e-12
    pn = np.mean(noise.astype(np.float64) ** 2) + 1e-12
    y = x + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    peak = np.max(np.abs(y))
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


# --------------------------------------------------------------------------- models
class Models:
    def __init__(self):
        import torch
        import whisper
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel

        from aiko.emotion import EmotionDetector

        self.torch, self.whisper = torch, whisper
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = json.load(open("paper/clean_probe_config.json", encoding="utf-8"))
        self.layer = cfg["layer_index"]

        with contextlib.redirect_stderr(io.StringIO()):
            self.det = EmotionDetector()
        self.asr = whisper.load_model("small", device=self.dev)
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(self.dev).eval()
        self.minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                          device=self.dev)

        # Clean probes trained on sessions 1-4 from the cached clean features, so the
        # training representation is byte-identical to clean_probe_eval.py's.
        z = np.load("paper/clean_feats/iemocap_all.npz")
        meta = [json.loads(l) for l in open("paper/clean_feats/iemocap_all.meta.jsonl",
                                            encoding="utf-8")]
        keep = np.array([m["gold"] in COMMON4 and m["session"] != TEST_SESSION for m in meta])
        y = np.array([m["gold"] for m in meta])[keep]

        def probe(c):
            return make_pipeline(StandardScaler(), LogisticRegression(
                C=c, max_iter=3000, class_weight="balanced"))
        self.pv = probe(cfg["C_voice"]).fit(z["wavlm"][keep][:, self.layer, :].astype(np.float32), y)
        self.pt = probe(cfg["C_text"]).fit(z["minilm"][keep], y)

    def voice_deployed(self, x: np.ndarray) -> dict[str, float]:
        with contextlib.redirect_stderr(io.StringIO()):
            return self.det.voice_scores(x)

    def voice_clean_feat(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = x[: 30 * 16000]
        if len(x) < 1600:
            x = np.pad(x, (0, 1600 - len(x)))
        inp = self.fe(x, sampling_rate=16000, return_tensors="pt").input_values.to(self.dev)
        with torch.inference_mode(), torch.autocast(self.dev, dtype=torch.float16,
                                                    enabled=self.dev == "cuda"):
            hs = self.wavlm(inp, output_hidden_states=True).hidden_states
        pooled = torch.stack([h[0].float().mean(0) for h in hs])
        sel = torch.cat([pooled[LAYERS], pooled.mean(0, keepdim=True)])
        # Round-trip through float16 exactly as the cached training features were stored.
        return sel[self.layer].cpu().numpy().astype(np.float16).astype(np.float32)

    def transcribe(self, xs: list[np.ndarray]) -> list[str]:
        torch, whisper = self.torch, self.whisper
        mels = torch.stack([whisper.log_mel_spectrogram(
            whisper.pad_or_trim(torch.from_numpy(x[: 30 * 16000])),
            n_mels=self.asr.dims.n_mels) for x in xs]).to(self.dev)
        res = whisper.decode(self.asr, mels, whisper.DecodingOptions(
            language="en", fp16=self.dev == "cuda", without_timestamps=True))
        return [r.text.strip() for r in res]


# ---------------------------------------------------------------------------- stages
def done_keys(path: str) -> set[tuple]:
    keys = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                keys.add((r["clip"], r["noise"], r["snr"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def run(max_new: int) -> None:
    rows = iemocap_rows()
    test = [r for r in rows if r["session"] == TEST_SESSION]
    pool = [r for r in rows if r["session"] != TEST_SESSION]
    done = done_keys(CACHE)
    todo = [(r, n, s) for n, s in conditions() for r in test if (r["clip"], n, s) not in done]
    total = len(conditions()) * len(test)
    print(f"{len(done)}/{total} cached; {len(todo)} to go", flush=True)
    if not todo:
        return

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    bank = NoiseBank(pool)
    m = Models()
    clean_audio: dict[str, np.ndarray] = {}
    n_new = 0
    with open(CACHE, "a", encoding="utf-8", buffering=1) as out:
        for b0 in range(0, len(todo), BATCH):
            batch = todo[b0:b0 + BATCH]
            audios = []
            for r, noise, snr in batch:
                x = clean_audio.get(r["clip"])
                if x is None:
                    x = clean_audio[r["clip"]] = load_wav(r["wav"])
                audios.append(x if noise == "clean" else
                              mix_at(x, bank.make(noise, len(x), r["clip"]), snr))
            asr = m.transcribe(audios)
            emb = m.minilm.encode(asr, batch_size=64, normalize_embeddings=True,
                                  show_progress_bar=False)
            tclean = m.pt.predict_proba(emb)
            vfeat = np.stack([m.voice_clean_feat(a) for a in audios])
            vclean = m.pv.predict_proba(vfeat)
            for k, ((r, noise, snr), a) in enumerate(zip(batch, audios)):
                rec = {
                    "clip": r["clip"], "noise": noise, "snr": snr, "gold": r["gold"],
                    "dialog": r["dialog"], "kind": r["kind"], "speaker": r["speaker"],
                    "voice_dep": m.voice_deployed(a), "text_dep": m.det.text_scores(asr[k]),
                    "voice_clean": dict(zip(COMMON4, map(float, vclean[k]))),
                    "text_clean": dict(zip(COMMON4, map(float, tclean[k]))),
                    "asr": asr[k], "snr_proxy": round(snr_proxy(a), 3),
                }
                out.write(json.dumps(rec) + "\n")
                n_new += 1
            print(f"  +{n_new} ({batch[-1][1]} {batch[-1][2]})", flush=True)
            if max_new and n_new >= max_new and b0 + BATCH < len(todo):
                print("chunk limit reached; exiting for a fresh process", flush=True)
                sys.exit(3)


def meld_wer() -> None:
    """Whisper on clean MELD test, for the cross-corpus acoustic-difficulty comparison."""
    import torch
    import whisper
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    asr = whisper.load_model("small", device=dev)
    rows = [json.loads(l) for l in open("paper/.text_backup/meld_test_scores.jsonl",
                                        encoding="utf-8")]
    rows = [r for r in rows if r["gold"] in COMMON4]
    os.makedirs(os.path.dirname(MELD_CACHE), exist_ok=True)
    with open(MELD_CACHE, "w", encoding="utf-8") as out:
        for b0 in range(0, len(rows), BATCH):
            batch = rows[b0:b0 + BATCH]
            xs = [load_wav(f"paper/meld_wavs/dia{r['dia']}_utt{r['utt']}.wav") for r in batch]
            mels = torch.stack([whisper.log_mel_spectrogram(
                whisper.pad_or_trim(torch.from_numpy(x[: 30 * 16000])),
                n_mels=asr.dims.n_mels) for x in xs]).to(dev)
            res = whisper.decode(asr, mels, whisper.DecodingOptions(
                language="en", fp16=dev == "cuda", without_timestamps=True))
            for r, h, x in zip(batch, res, xs):
                out.write(json.dumps({"clip": f"test_dia{r['dia']}_utt{r['utt']}",
                                      "asr": h.text.strip(), "ref": r.get("text") or "",
                                      "snr_proxy": round(snr_proxy(x), 3)}) + "\n")
    print(f"{len(rows)} MELD test clips -> {MELD_CACHE}")


# ---------------------------------------------------------------------------- report
_NORM = re.compile(r"[^a-z0-9' ]+")


def words(s: str) -> list[str]:
    return _NORM.sub(" ", (s or "").lower().replace("’", "'")).split()


def wer_counts(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance, reference length) at the word level."""
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def corpus_wer(pairs) -> float:
    e = n = 0
    for ref, hyp in pairs:
        a, b = wer_counts(ref, hyp)
        e, n = e + a, n + b
    return 100 * e / n if n else float("nan")


def crossing(xs: list[float], a: list[float], b: list[float]) -> float | None:
    """First SNR (descending) where curve a falls below curve b, linearly interpolated."""
    for i in range(len(xs) - 1):
        d0, d1 = a[i] - b[i], a[i + 1] - b[i + 1]
        if d0 >= 0 > d1:
            return xs[i] + (xs[i + 1] - xs[i]) * d0 / (d0 - d1)
    return None


def interp_level(xs: list[float], ys: list[float], target: float) -> float | None:
    """SNR at which a monotone-ish curve ys(xs) reaches target, interpolated."""
    for i in range(len(xs) - 1):
        lo, hi = sorted((ys[i], ys[i + 1]))
        if lo <= target <= hi and ys[i] != ys[i + 1]:
            return xs[i] + (xs[i + 1] - xs[i]) * (target - ys[i]) / (ys[i + 1] - ys[i])
    return None


def fmt(title: str, rows: list[list], cols: list[str]) -> str:
    w = [max([len(str(c))] + [len(str(r[i])) for r in rows]) for i, c in enumerate(cols)]
    out = [title, "", "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
    return "\n".join(out)


def report() -> None:
    from sklearn.metrics import balanced_accuracy_score

    import meld_eval as me

    def bal(g, p):
        return 100 * balanced_accuracy_score(g, p)

    def am(s, labels=COMMON4):
        return max(labels, key=lambda e: (s or {}).get(e, 0.0))

    refs = {r["clip"]: r.get("text") or "" for r in iemocap_rows()}
    starts = {}
    from iemocap_extract_scores import read_labels
    for k, v in read_labels(IEMOCAP_ROOT).items():
        starts[k] = v["start"]
    recs = []
    for line in open(CACHE, encoding="utf-8"):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:            # a write cut off by a killed process
            continue
    by = defaultdict(list)
    for r in recs:
        by[(r["noise"], r["snr"])].append(r)
    n_clips = len({r["clip"] for r in by[("clean", None)]})

    def evaluate(rs: list[dict]) -> dict:
        g = [r["gold"] for r in rs]
        out = {
            "voice_dep": bal(g, [am(r["voice_dep"]) for r in rs]),
            "text_dep": bal(g, [am(r["text_dep"]) for r in rs]),
            "voice_clean": bal(g, [am(r["voice_clean"]) for r in rs]),
            "text_clean": bal(g, [am(r["text_clean"]) for r in rs]),
            "wer": corpus_wer((refs[r["clip"]], r["asr"]) for r in rs),
            "proxy": float(np.median([r["snr_proxy"] for r in rs])),
            "n": len(rs),
        }
        # Deployed heuristic on the degraded inputs, chronological within each dialogue.
        ordered = sorted(rs, key=lambda r: (r["dialog"], starts.get(r["clip"], 0.0)))
        feed = [{"voice": r["voice_dep"], "text_scores": r["text_dep"],
                 "n_words": len(r["asr"].split()), "dia": r["dialog"]} for r in ordered]
        hp = me.run_condition(feed, me.make_fuse())
        out["heur_dep"] = bal([r["gold"] for r in ordered],
                              [p if p in COMMON4 else "other" for p in hp])
        return out

    clean = evaluate(by[("clean", None)])
    tables, curves = [], {}
    for noise in NOISES:
        rows = [["clean", f"{clean['proxy']:.1f}", f"{clean['wer']:.1f}",
                 f"{clean['voice_dep']:.2f}", f"{clean['text_dep']:.2f}",
                 f"{clean['heur_dep']:.2f}",
                 f"{clean['voice_clean']:.2f}", f"{clean['text_clean']:.2f}"]]
        cur = {"snr": [], "voice_dep": [], "text_dep": [], "voice_clean": [],
               "text_clean": [], "wer": [], "proxy": [], "heur_dep": []}
        for snr in SNRS:
            rs = by.get((noise, snr), [])
            if len(rs) < n_clips:
                continue
            e = evaluate(rs)
            rows.append([f"{snr:+d} dB", f"{e['proxy']:.1f}", f"{e['wer']:.1f}",
                         f"{e['voice_dep']:.2f}", f"{e['text_dep']:.2f}", f"{e['heur_dep']:.2f}",
                         f"{e['voice_clean']:.2f}", f"{e['text_clean']:.2f}"])
            cur["snr"].append(snr)
            for k in cur:
                if k != "snr":
                    cur[k].append(e[k])
        curves[noise] = cur
        tables.append(fmt(f"## Table N-{len(tables) + 1} - {noise} noise, IEMOCAP session 5 "
                          f"(n = {n_clips}, balanced accuracy)", rows,
                          ["SNR", "SNR proxy dB", "ASR WER %", "deployed voice",
                           "deployed text (ASR)", "deployed heuristic", "clean voice",
                           "clean text (ASR)"]))

    meld = [json.loads(l) for l in open(MELD_CACHE, encoding="utf-8")] \
        if os.path.exists(MELD_CACHE) else []
    meld_w = corpus_wer((r["ref"], r["asr"]) for r in meld) if meld else float("nan")
    meld_p = float(np.median([r["snr_proxy"] for r in meld])) if meld else float("nan")

    # MELD's own accuracy on the same four classes, deployed and clean, for the anchors.
    md = [json.loads(l) for l in open("paper/meld_test_scores.jsonl", encoding="utf-8")]
    md = [r for r in md if r["gold"] in COMMON4]
    meld_voice_dep = bal([r["gold"] for r in md], [am(r["voice"]) for r in md])

    cross, anchor = [], []
    for noise, c in curves.items():
        xs = [99.0] + c["snr"]                                  # 99 stands in for clean
        vd, td = [clean["voice_dep"]] + c["voice_dep"], [clean["text_dep"]] + c["text_dep"]
        vc, tc = [clean["voice_clean"]] + c["voice_clean"], [clean["text_clean"]] + c["text_clean"]
        xd, xc = crossing(xs, vd, td), crossing(xs, vc, tc)
        cross.append([noise, "never above -5 dB" if xd is None else f"{xd:+.1f} dB",
                      "never above -5 dB" if xc is None else f"{xc:+.1f} dB"])
        lw = interp_level(c["snr"], c["wer"], meld_w)
        # The proxy anchor is only meaningful where the proxy falls as noise rises; bursty
        # noise such as laughter leaves quiet gaps that the proxy reads as a clean floor.
        p_seq = [clean["proxy"]] + c["proxy"]
        monotone = all(b <= a + 0.5 for a, b in zip(p_seq, p_seq[1:]))
        lp = interp_level(c["snr"], c["proxy"], meld_p) if monotone else None

        def at(level, ys):
            return "-" if level is None else f"{np.interp(-level, [-s for s in c['snr']], ys):.2f}"
        anchor.append([noise,
                       "beyond tested range" if lw is None else f"{lw:+.1f} dB",
                       at(lw, c["voice_dep"]), at(lw, c["voice_clean"]),
                       ("proxy not monotone: invalid" if not monotone else
                        f"milder than {SNRS[0]:+d} dB"
                        if lp is None and min(p_seq[:2]) <= meld_p <= max(p_seq[:2]) else
                        "beyond tested range" if lp is None else f"{lp:+.1f} dB"),
                       at(lp, c["voice_dep"])])

    head = [
        "# Controlled degradation of clean speech (IEMOCAP session 5)\n",
        "Held-out session 5, common four classes, the full deployed path on every degraded "
        "clip. Clean probes trained on sessions 1-4 with settings frozen on MELD dev. "
        "`deployed heuristic` is Aiko's fusion rule fed the degraded voice scores and the "
        "text scores of the ASR transcript; its seven-way output counts as wrong when it "
        "names a class outside the four.\n",
    ]
    body = tables + [
        fmt("## Table N-4 - Where voice falls below text (ASR words)", cross,
            ["Noise", "deployed encoders", "clean encoders"]),
        "\nInterpolated SNR at which voice balanced accuracy first drops below text "
        "balanced accuracy, where the text model reads Whisper's transcript of the same "
        "degraded audio.\n",
        fmt("## Table N-5 - How much noise makes IEMOCAP look like MELD acoustically?", anchor,
            ["Noise", "SNR matching MELD's ASR WER", "deployed voice there",
             "clean voice there", "SNR matching MELD's SNR proxy",
             "deployed voice there"]),
        f"\nMELD test (common four, clean): Whisper WER {meld_w:.1f} %, median SNR proxy "
        f"{meld_p:.1f} dB, deployed voice balanced accuracy {meld_voice_dep:.2f}. "
        f"IEMOCAP session 5 clean: WER {clean['wer']:.1f} %, proxy {clean['proxy']:.1f} dB. "
        "Each anchor finds the added-noise level at which degraded IEMOCAP matches MELD on "
        "that measure, then reads the voice accuracy IEMOCAP still has at that level. If "
        "that accuracy stays far above MELD's, noise of a severity matching MELD's acoustic "
        "difficulty does NOT reproduce MELD's voice collapse, and the IEMOCAP-to-MELD step "
        "must be attributed to something other than the audio (for example how MELD was "
        "labelled).\n",
    ]
    text = "\n".join(head + body) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"(written to {OUT})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "meld-wer", "report"])
    ap.add_argument("--max-new", type=int, default=0)
    args = ap.parse_args()
    {"run": lambda: run(args.max_new), "meld-wer": meld_wer, "report": report}[args.stage]()


if __name__ == "__main__":
    main()

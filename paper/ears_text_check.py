#!/usr/bin/env python3
"""Gap 2, part 1: does "voice beats text only in improvised speech" replicate on EARS?

The headline pattern (clean_spontaneity_results.md, Table S-4) rests on IEMOCAP: voice
minus text is about +26 points in improvised dialogue and about zero in scripted dialogue,
and about zero in scripted television (MELD). EARS offers an independent test with the
same contamination-free encoders: 107 speakers, each producing every emotion once READING
fixed sentences and once speaking FREEFORM. It is still acted speech, so it cannot answer
whether the pattern holds for natural speech; mosei_eval.py addresses that.

Unit of analysis is the RECORDING, not the 3 s window: text needs context, and a window
cuts sentences mid-way. Voice is the mean of the recording's window features; text is the
Whisper-small transcript of the whole recording (decoded as in noise_degradation.py), so
both modalities see the same audio, and text reaches the model only through speech
recognition, as in deployment. (IEMOCAP's Table S-4 used gold transcripts; EARS ships none
for freeform speech, so ASR is used for both conditions here.)

Classes angry / neutral / sad. Probes grouped 5-fold by speaker with the MELD-dev
settings (clean_probe_config.json). CIs resample speakers, 1000 times.

Transcripts are cached in data/ears/asr.jsonl (gitignored); only aggregates are written.

    .venv/bin/python paper/ears_text_check.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
from sklearn.model_selection import GroupKFold

from clean_probe_eval import FEATS, bal, fmt, probe

MAP = {"anger": "angry", "neutral": "neutral", "sadness": "sad"}
ASR = "data/ears/asr.jsonl"
OUT = "paper/ears_text_results.md"
B_BOOT = 1000


def transcribe(recs: list[dict]) -> dict[str, str]:
    done = {}
    if os.path.exists(ASR):
        for line in open(ASR, encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r["asr"]
    todo = [r for r in recs if r["id"] not in done]
    if todo:
        import torch
        import whisper
        model = whisper.load_model("small", device="cuda")
        with open(ASR, "a", encoding="utf-8") as f:
            for b0 in range(0, len(todo), 16):
                batch = todo[b0:b0 + 16]
                xs = [sf.read(f"data/ears/wav/{r['id']}.wav", dtype="float32")[0][: 30 * 16000]
                      for r in batch]
                mels = torch.stack([whisper.log_mel_spectrogram(
                    whisper.pad_or_trim(torch.from_numpy(x)), n_mels=model.dims.n_mels)
                    for x in xs]).cuda()
                res = whisper.decode(model, mels, whisper.DecodingOptions(
                    language="en", fp16=True, without_timestamps=True))
                for r, h in zip(batch, res):
                    done[r["id"]] = h.text.strip()
                    f.write(json.dumps({"id": r["id"], "asr": done[r["id"]]}) + "\n")
        print(f"transcribed {len(todo)} recordings", flush=True)
    return done


def main() -> None:
    cfg = json.load(open("paper/clean_probe_config.json", encoding="utf-8"))
    recs = [json.loads(l) for l in open("data/ears/manifest.jsonl", encoding="utf-8")]
    recs = sorted((r for r in recs if r["emotion"] in MAP), key=lambda r: r["id"])
    asr = transcribe(recs)

    # recording-level voice: mean of the recording's window features
    z = np.load(f"{FEATS}/ears.npz")
    wmeta = [json.loads(l) for l in open(f"{FEATS}/ears.meta.jsonl", encoding="utf-8")]
    by_rec = defaultdict(list)
    for i, m in enumerate(wmeta):
        by_rec[m["recording"]].append(i)
    recs = [r for r in recs if r["id"] in by_rec]          # recordings with >= 1 window
    Xv = np.stack([z["wavlm"][by_rec[r["id"]], cfg["layer_index"], :].astype(np.float32).mean(0)
                   for r in recs])

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    Xt = st.encode([asr[r["id"]] for r in recs], batch_size=64, normalize_embeddings=True,
                   show_progress_bar=False).astype(np.float32)

    y = np.array([MAP[r["emotion"]] for r in recs])
    spk = np.array([r["speaker"] for r in recs])
    cond = np.array([r["condition"] for r in recs])
    pv, pt = np.empty(len(y), dtype=object), np.empty(len(y), dtype=object)
    for a, b in GroupKFold(n_splits=5).split(Xv, y, spk):
        pv[b] = probe(cfg["C_voice"]).fit(Xv[a], y[a]).predict(Xv[b])
        pt[b] = probe(cfg["C_text"]).fit(Xt[a], y[a]).predict(Xt[b])

    by_spk = defaultdict(list)
    for i, s in enumerate(spk):
        by_spk[s].append(i)
    keys = list(by_spk)
    rng = np.random.default_rng(0)
    samples = [np.concatenate([by_spk[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
               for _ in range(B_BOOT)]

    def ci(fn):
        vals = [fn(s) for s in samples]
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    def margin(sel, c):
        m = sel[cond[sel] == c]
        return bal(y[m], pv[m]) - bal(y[m], pt[m])

    allidx = np.arange(len(y))
    rows = []
    for c in ("read", "freeform"):
        m = cond == c
        lo, hi = ci(lambda s: margin(s, c))
        rows.append([c, int(m.sum()), f"{bal(y[m], pv[m]):.2f}", f"{bal(y[m], pt[m]):.2f}",
                     f"{margin(allidx, c):+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    d = margin(allidx, "freeform") - margin(allidx, "read")
    dlo, dhi = ci(lambda s: margin(s, "freeform") - margin(s, "read"))

    report = ["# Gap 2, part 1: the voice-text pattern on EARS (read versus freeform)\n",
              f"{len(recs)} recordings, {len(keys)} speakers, angry / neutral / sad. Voice: "
              "WavLM probe on the mean of each recording's 3 s windows. Text: MiniLM probe on "
              "the Whisper-small transcript of the whole recording. Probes grouped 5-fold by "
              "speaker, settings from MELD dev.\n",
              fmt("## Table T-1 - Voice minus text, by condition", rows,
                  ["Condition", "recordings", "voice bal acc", "text bal acc",
                   "voice - text [95 % CI]"]),
              f"\nFreeform margin minus read margin: {d:+.2f} points, 95 % CI "
              f"[{dlo:+.2f}, {dhi:+.2f}] (speakers resampled; every speaker contributes to "
              "both conditions, so the comparison is paired).\n",
              "For reference, clean_spontaneity_results.md Table S-4 (IEMOCAP, gold "
              "transcripts): scripted +0.40, improvised +25.76, improvised minus scripted "
              "+25.36 [+19.65, +30.59]; MELD (scripted television) +0.03.\n"]
    words = {c: float(np.mean([len(asr[r["id"]].split()) for r, cc in zip(recs, cond) if cc == c]))
             for c in ("read", "freeform")}
    import re

    def norm(s):
        return " ".join(re.sub(r"[^a-z' ]+", " ", s.lower()).split())
    read_by_text = defaultdict(set)
    for r, c in zip(recs, cond):
        if c == "read":
            read_by_text[norm(asr[r["id"]])].add(r["emotion"])
    shared = sum(1 for t, e in read_by_text.items() if len(e) > 1)
    report.append(f"Mean transcript length: read {words['read']:.1f} words, freeform "
                  f"{words['freeform']:.1f} words. Of {len(read_by_text)} distinct read "
                  f"transcripts, {shared} occur under more than one emotion. If that share "
                  "were high, the read sentences would be carrier sentences (as in CREMA-D) "
                  "and text would be uninformative by design; it is reported so the read "
                  "condition's text accuracy can be interpreted.\n")
    ceiling = {c: bal(y[cond == c], pt[cond == c]) >= 98.0 for c in ("read", "freeform")}
    if all(ceiling.values()):
        report.append("**Verdict: UNINFORMATIVE, text is at ceiling in both conditions.** "
                      "The read sentences are emotion-specific and shared across all "
                      "speakers, so a text probe evaluated on held-out speakers still sees "
                      "held-in SENTENCES and memorises sentence -> emotion; and freeform "
                      "monologues, recorded on the instruction to speak in an emotion, name "
                      "or describe it. Neither condition can show whether voice or text "
                      "carries more emotion, so EARS does not test the Table S-4 pattern. "
                      "The margins above are reported for completeness and must not be cited "
                      "as evidence for or against it.\n")
    elif any(ceiling.values()):
        report.append("**Caution:** text is at ceiling in "
                      + ", ".join(c for c, v in ceiling.items() if v)
                      + "; margins in that condition are bounded by the ceiling.\n")
    text = "\n".join(report) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"(written to {OUT})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cache emotion2vec voice scores and DistilRoBERTa text scores over IEMOCAP.

Why IEMOCAP, and why it is the decisive corpus for the C1 boundary condition:

CREMA-D against MELD is a TWO-point comparison and it confounds every variable at once.
CREMA-D is acted, isolated, studio-clean, and built on twelve fixed carrier sentences;
MELD is scripted TV performance, multi-party, broadcast audio, and open vocabulary. The
55.67-point balanced-accuracy gap between them therefore cannot be attributed to any one
cause, and the paper can only say "in-the-wild conditions" as an undifferentiated blob.

IEMOCAP sits precisely in the middle: studio-recorded with head-mounted microphones, so
acoustically clean like CREMA-D, but conversational, like MELD. That
splits the blob in two:

    CREMA-D -> IEMOCAP   varies spontaneity and dialogue context, recording held clean
    IEMOCAP -> MELD      varies recording conditions, but also corpus, annotation and
                         structure; see noise_results.md before attributing it to audio

Better still, IEMOCAP contains BOTH improvised and scripted dialogues from the same ten
actors in the same sessions through the same recording chain. Script against impro varies
spontaneity ALONE, within corpus, within speaker. That is a controlled contrast rather
than a cross-corpus difference, which is what lets Section 4.5 name the responsible
variable instead of gesturing at a bundle of them.

FIVE label traps this script exists to handle.

1. `xxx` means the annotators did not reach agreement. 2,507 of 10,039 utterances carry
   it. They are dropped, not folded into neutral.
2. `fru` (frustrated) has no counterpart in the deployed seven-class scheme. It is the
   second largest category, 1,849 utterances. It is RETAINED in the cache with
   gold=null so it is available as a diagnostic (where does the encoder put frustration?)
   but it cannot enter any accuracy table. If frustrated speech is harder than the
   retained classes, dropping it flatters IEMOCAP: conservative for a CREMA-D-to-IEMOCAP
   drop, but it ENLARGES an IEMOCAP-to-MELD drop. (An earlier version of this note had
   the second direction backwards.)
3. `exc` (excited) is conventionally merged into happy, and the deployed scheme has no
   excited category, so the merge is forced if those 1,041 utterances are kept. The merge
   is recorded per row (`merged_exc`) so iemocap_eval.py can report it both ways rather
   than burying a consequential choice.
4. `dis` has TWO utterances and `fea` has 40. The six-class scheme shared by CREMA-D and
   MELD is therefore unusable here. The common scheme across all three corpora is the
   four classes with real support everywhere: angry, happy, neutral, sad. iemocap_eval.py
   rescores CREMA-D and MELD on those four from their existing caches, so the ladder is
   self-consistent and no re-extraction is needed.
5. Transcripts carry bracketed non-speech annotation ([LAUGHTER], [BREATHING], [GARBAGE]).
   The deployed pipeline sees Whisper output, which contains no such tags, so they are
   stripped before the text model runs. Gold transcripts still make the text pathway more
   favourable than deployment, exactly as noted for MELD in methods.md 3.8.

The macOS tarball ships an AppleDouble `._<name>` sibling for every real file, 10,190 of
them under sentences/wav. They are not audio and are filtered by basename everywhere.

funasr leaks RAM, so run in chunks, each in a fresh process:

    while :; do .venv/bin/python paper/iemocap_extract_scores.py \
        --root IEMOCAP_full_release/IEMOCAP_full_release --resume --max-new 500; \
        [ $? -ne 3 ] && break; done

Validate the manifest first, which loads no model and takes a second:

    .venv/bin/python paper/iemocap_extract_scores.py \
        --root IEMOCAP_full_release/IEMOCAP_full_release --dry-run
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IEMOCAP tag -> deployed seven-class scheme. `fru`/`oth` deliberately absent: they have
# no counterpart and must not be silently coerced into a neighbouring class.
IEMOCAP_TO_UNIFIED = {
    "ang": "angry", "dis": "disgusted", "fea": "fearful",
    "hap": "happy", "exc": "happy", "neu": "neutral",
    "sad": "sad", "sur": "surprised",
}
OUT_OF_SCHEME = {"fru": "frustrated", "oth": "other"}   # cached, gold=null
DROP = {"xxx"}                                          # no annotator agreement

# [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
EMO_LINE = re.compile(
    r"^\[(?P<start>\d+\.\d+)\s*-\s*(?P<end>\d+\.\d+)\]\s+"
    r"(?P<utt>\S+)\s+(?P<tag>\S+)\s+\[(?P<vad>[^\]]*)\]"
)
# Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
TRANS_LINE = re.compile(r"^(?P<utt>\S+)\s+\[(?P<span>[\d.\-]+)\]:\s*(?P<text>.*)$")
BRACKETED = re.compile(r"\[[^\]]*\]")       # [LAUGHTER], [BREATHING], [GARBAGE]


def is_junk(path: str) -> bool:
    """AppleDouble resource forks from the macOS tarball are not data."""
    return os.path.basename(path).startswith("._")


def dialog_of(utt: str) -> str:
    """Ses01F_impro01_F000 -> Ses01F_impro01;  Ses01F_script01_1_F000 -> Ses01F_script01_1."""
    return utt.rsplit("_", 1)[0]


def session_of(utt: str) -> int:
    """Ses01F_... -> 1. The two digits after 'Ses' are the session, 1 to 5."""
    return int(utt[3:5])


def speaker_of(utt: str) -> str:
    """Speaker identity is session plus the gender in the FINAL group.

    Note the trap: the 'F' in `Ses01F_impro01` names the session's target actor, not the
    speaker of this utterance. The speaker is the F/M on the final group (`_F000`). Five
    sessions times two actors gives exactly ten distinct speakers.
    """
    return f"Ses{utt[3:5]}_{utt.rsplit('_', 1)[1][0]}"


def kind_of(dialog: str) -> str:
    if "impro" in dialog:
        return "impro"
    if "script" in dialog:
        return "script"
    return "unknown"


def clean_text(raw: str) -> str:
    """Strip non-speech annotation so the text model sees what Whisper would emit."""
    return " ".join(BRACKETED.sub(" ", raw).split()).strip()


def read_labels(root: str) -> dict[str, dict]:
    """Parse every dialog/EmoEvaluation/*.txt into {utt_id: {...}}."""
    out: dict[str, dict] = {}
    pattern = os.path.join(root, "Session*", "dialog", "EmoEvaluation", "*.txt")
    for path in sorted(glob.glob(pattern)):
        if is_junk(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = EMO_LINE.match(line)
                if not m:
                    continue
                utt, tag = m.group("utt"), m.group("tag").lower()
                try:
                    vad = [float(x) for x in m.group("vad").split(",")]
                except ValueError:
                    vad = []
                out[utt] = {
                    "utt": utt,
                    "tag": tag,
                    "start": float(m.group("start")),
                    "end": float(m.group("end")),
                    "valence": vad[0] if len(vad) == 3 else None,
                    "arousal": vad[1] if len(vad) == 3 else None,
                    "dominance": vad[2] if len(vad) == 3 else None,
                }
    return out


def read_transcripts(root: str) -> dict[str, str]:
    out: dict[str, str] = {}
    pattern = os.path.join(root, "Session*", "dialog", "transcriptions", "*.txt")
    for path in sorted(glob.glob(pattern)):
        if is_junk(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = TRANS_LINE.match(line.strip())
                if m:
                    out[m.group("utt")] = clean_text(m.group("text"))
    return out


def build_manifest(root: str) -> tuple[list[dict], Counter]:
    """Join labels, transcripts and per-utterance wav paths. Returns (rows, stats)."""
    labels = read_labels(root)
    trans = read_transcripts(root)
    stats: Counter[str] = Counter()
    rows: list[dict] = []

    for utt, meta in sorted(labels.items()):
        stats[f"tag:{meta['tag']}"] += 1
        if meta["tag"] in DROP:
            stats["dropped_no_agreement"] += 1
            continue
        if meta["tag"] in IEMOCAP_TO_UNIFIED:
            gold = IEMOCAP_TO_UNIFIED[meta["tag"]]
        elif meta["tag"] in OUT_OF_SCHEME:
            gold = None                      # cached as diagnostic, never scored
        else:
            stats["dropped_unknown_tag"] += 1
            continue

        dialog = dialog_of(utt)
        wav = os.path.join(root, f"Session{session_of(utt)}", "sentences", "wav",
                           dialog, f"{utt}.wav")
        if not os.path.exists(wav):
            stats["missing_wav"] += 1
            continue
        text = trans.get(utt)
        if text is None:
            stats["missing_transcript"] += 1
            text = ""

        rows.append({
            "clip": utt,
            "wav": wav,
            "dialog": dialog,
            "session": session_of(utt),
            "speaker": speaker_of(utt),
            "kind": kind_of(dialog),
            "tag": meta["tag"],
            "gold": gold,
            "merged_exc": meta["tag"] == "exc",
            "duration": round(meta["end"] - meta["start"], 4),
            "valence": meta["valence"],
            "arousal": meta["arousal"],
            "dominance": meta["dominance"],
            "text": text,
        })
        stats["kept"] += 1
        stats[f"kind:{kind_of(dialog)}"] += 1
    return rows, stats


def is_uniform(scores: dict[str, float]) -> bool:
    """voice_scores() swallows exceptions and returns a uniform distribution, which would
    otherwise be silently counted as a real prediction."""
    if not scores:
        return True
    vals = list(scores.values())
    return max(vals) - min(vals) < 1e-9


def rss_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def summarise(rows: list[dict], stats: Counter) -> None:
    gold = Counter(r["gold"] or f"({r['tag']})" for r in rows)
    print(f"manifest: {len(rows)} utterances with audio")
    print(f"  gold distribution: {dict(gold.most_common())}")
    print(f"  by kind: impro={stats['kind:impro']} script={stats['kind:script']} "
          f"unknown={stats['kind:unknown']}")
    print(f"  dropped: no_agreement(xxx)={stats['dropped_no_agreement']} "
          f"unknown_tag={stats['dropped_unknown_tag']} "
          f"missing_wav={stats['missing_wav']} "
          f"missing_transcript={stats['missing_transcript']}")
    raw = {k[4:]: v for k, v in stats.items() if k.startswith("tag:")}
    print(f"  raw IEMOCAP tags: {dict(sorted(raw.items(), key=lambda kv: -kv[1]))}")
    scoreable = sum(1 for r in rows if r["gold"])
    print(f"  scoreable (gold in deployed scheme): {scoreable}; "
          f"diagnostic-only (fru/oth): {len(rows) - scoreable}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="IEMOCAP_full_release dir containing Session1..Session5")
    ap.add_argument("--out", default="paper/iemocap_scores.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N (smoke)")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out, skipping clips already cached")
    ap.add_argument("--max-new", type=int, default=0,
                    help="exit (code 3) after this many new records so a fresh process "
                         "releases funasr's leaked RAM")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and summarise the manifest, load no model, write nothing")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"! --root not a directory: {args.root}")
        sys.exit(1)
    if not glob.glob(os.path.join(args.root, "Session*")):
        print(f"! no Session* under {args.root}. The tarball nests one level: point --root "
              f"at the inner IEMOCAP_full_release/IEMOCAP_full_release")
        sys.exit(1)

    rows, stats = build_manifest(args.root)
    if not rows:
        print("! empty manifest")
        sys.exit(1)
    summarise(rows, stats)
    if args.dry_run:
        print("\n(dry run: no model loaded, nothing written)")
        return
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done: set[str] = set()
    mode = "w"
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["clip"])
                except (json.JSONDecodeError, KeyError):
                    continue
        mode = "a"
        print(f"resuming: {len(done)} clips already cached", flush=True)

    import torch
    from aiko.emotion import EmotionDetector

    print("Loading EmotionDetector...", flush=True)
    det = EmotionDetector()

    n_ok = n_fail = n_notext = 0
    stopped_early = False
    with open(args.out, mode, encoding="utf-8", buffering=1) as out:
        for i, r in enumerate(rows):
            if r["clip"] in done:
                continue
            with torch.inference_mode():
                voice = det.voice_scores(r["wav"])
                # Unlike CREMA-D's 12 carrier sentences there is nothing to cache here:
                # IEMOCAP transcripts are open vocabulary and near all distinct.
                text_scores = det.text_scores(r["text"]) if r["text"] else None
            if is_uniform(voice):
                n_fail += 1          # model failed on this clip; do not cache as data
                continue
            if text_scores is None:
                n_notext += 1

            rec = {k: v for k, v in r.items() if k != "wav"}
            rec["n_words"] = len(r["text"].split())
            rec["voice"] = voice
            rec["text_scores"] = text_scores
            out.write(json.dumps(rec) + "\n")
            n_ok += 1

            del voice
            if (i + 1) % 25 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if (i + 1) % 200 == 0:
                print(f"  [{i + 1}/{len(rows)}] new={n_ok} rss={rss_mb():.0f}MB", flush=True)

            if args.max_new and n_ok >= args.max_new:
                stopped_early = True
                print(f"  chunk limit {args.max_new} reached at [{i + 1}/{len(rows)}] "
                      f"(rss={rss_mb():.0f}MB); exiting for a fresh process", flush=True)
                break

    cached: Counter[str] = Counter()
    total = 0
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                cached[rec["gold"] or f"({rec['tag']})"] += 1
                total += 1
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"\n{'Chunk done' if stopped_early else 'Done'}. cache holds {total} -> {args.out}")
    print(f"  this run: new={n_ok} model_fail={n_fail} empty_transcript={n_notext}")
    print(f"  cached distribution: {dict(cached.most_common())}")
    if stopped_early:
        sys.exit(3)


if __name__ == "__main__":
    main()

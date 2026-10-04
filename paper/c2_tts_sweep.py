#!/usr/bin/env python3
"""Sweep XTTS first-chunk length: the time-to-first-audio / faithfulness tradeoff.

tts_first_chunk is ~58 % of Aiko's time-to-first-audio, but in c2_efficiency.py it is
measured on whatever the LLM happened to say (temperature 0.8), so its mean moved
825 -> 929 ms and its median 747 -> 931 ms between otherwise identical runs. This script
takes the LLM out of the loop: fixed texts at controlled lengths, so the figure is
reproducible and the length curve becomes measurable.

TTS_CHUNK_MIN_CHARS = 60 exists because XTTS rambles on very short inputs. Milliseconds
of audio per character detects that objectively -- roughly flat when synthesis is
faithful, inflated where the model invents speech the text does not contain.

Needs tts_server.py up on :5123. Touches no GPU model of its own.

    .venv/bin/python paper/c2_tts_sweep.py --reps 5
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiko import config
from aiko.tts import VoiceClient, trim_pauses

# In-domain reply sentences, so the sweep measures the voice on the text it deployed for.
POOL = [
    "I missed you today.",
    "Come home soon and tell me everything.",
    "I have the tea ready for you.",
    "You sound tired, love.",
    "Just keep breathing, you are almost there.",
    "I am proud of how hard you worked.",
    "We can watch a movie when you get in.",
    "Do not worry about the dishes tonight.",
    "I was thinking about you all afternoon.",
    "Let me know when you are on your way.",
    "That sounds exhausting, and I am sorry.",
    "You always make me laugh, even now.",
]


def build(target: int, offset: int) -> str:
    """Concatenate pool sentences until >= target chars; overshoots by at most one."""
    out, n, i = [], 0, offset
    while n < target:
        s = POOL[i % len(POOL)]
        out.append(s)
        n += len(s) + 1
        i += 1
    return " ".join(out)


def stats(xs: list[float]) -> dict:
    """No p95 here: at these rep counts a 95th percentile is just the maximum, so the
    tail is reported honestly as max rather than dressed up as a percentile."""
    if not xs:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "p50": 0.0, "max": 0.0}
    s = sorted(xs)
    return {"n": len(xs), "mean": st.fmean(xs),
            "sd": st.stdev(xs) if len(xs) > 1 else 0.0,
            "p50": s[len(s) // 2], "max": s[-1]}


def fmt(title: str, rows: list[list], cols: list[str]) -> str:
    w = [max([len(str(c))] + [len(str(r[i])) for r in rows]) for i, c in enumerate(cols)]
    out = [title, "", "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="20,30,45,60,80,120,160,200",
                    help="target first-chunk character lengths to sweep")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="paper/c2_tts_sweep.md")
    args = ap.parse_args()

    if not VoiceClient._port_open():
        print(f"! TTS server not on :{config.TTS_PORT}. Start it first:\n"
              f"    {config.TTS_VENV_PYTHON} {config.TTS_SERVER_SCRIPT}")
        sys.exit(1)

    voice = VoiceClient()
    targets = [int(x) for x in args.lengths.split(",") if x.strip()]
    bench = os.path.join(config.VOICE_OUTPUT_DIR, "c2_sweep.wav")
    print(f"XTTS first-chunk sweep: {targets} chars x {args.reps} reps "
          f"({args.warmup} warmup discarded)\n", flush=True)

    rows, curve = [], []
    for target in targets:
        lat, trm, dur, dpost, chars = [], [], [], [], []
        for r in range(args.warmup + args.reps):
            text = build(target, offset=r * 3)   # vary text per rep, defeat any caching
            t0 = time.perf_counter()
            wav = voice.synthesize(text, bench)
            t1 = time.perf_counter()
            if not wav:
                print(f"  ! synthesis failed at target {target}", flush=True)
                break
            info = sf.info(wav)                  # pre-trim: rambling is extra SPEECH
            t2 = time.perf_counter()
            trim_pauses(wav)                     # on the deployed critical path
            t3 = time.perf_counter()
            post = sf.info(wav)                  # post-trim: silence gone, speech kept
            if r < args.warmup:
                continue
            lat.append((t1 - t0) * 1000.0)
            trm.append((t3 - t2) * 1000.0)
            dur.append(info.frames / info.samplerate)
            dpost.append(post.frames / post.samplerate)
            chars.append(len(text))
        if not lat:
            continue
        s, td = stats(lat), stats(trm)
        ch, ad = st.fmean(chars), st.fmean(dur)
        adp = st.fmean(dpost)
        ms_per_char, ms_per_char_post = 1000 * ad / ch, 1000 * adp / ch
        rows.append([target, f"{ch:.0f}", s["n"], f"{s['mean']:8.1f}", f"{s['sd']:7.1f}",
                     f"{s['p50']:8.1f}", f"{s['max']:8.1f}", f"{td['mean']:7.1f}",
                     f"{ad:7.2f}", f"{ms_per_char:7.1f}", f"{ms_per_char_post:8.1f}",
                     f"{s['mean'] / (1000 * ad):6.3f}"])
        curve.append((target, s["mean"], ms_per_char, ms_per_char_post))
        print(f"  {target:>4} chars (actual {ch:>4.0f})  synth {s['mean']:7.1f} ms  "
              f"trim {td['mean']:5.1f} ms  audio {ad:5.2f} s  "
              f"{ms_per_char:5.1f} ms/char -> {ms_per_char_post:5.1f} post-trim", flush=True)

    report = [fmt("## Table C2-6 - XTTS first-chunk length sweep", rows,
                  ["target", "chars", "n", "synth ms", "sd", "p50", "max", "trim ms",
                   "audio s", "ms/char", "ms/char post", "RTF"])]
    if curve:
        tail = [c[2] for c in curve if c[0] >= 80]
        tailp = [c[3] for c in curve if c[0] >= 80]
        plateau = st.fmean(tail) if tail else curve[-1][2]
        plateau_p = st.fmean(tailp) if tailp else curve[-1][3]
        report.append(f"\n**Reading.** Deployed default is `TTS_CHUNK_MIN_CHARS = "
                      f"{config.TTS_CHUNK_MIN_CHARS}`. Audio-per-character plateau "
                      f"(targets >= 80 chars) = {plateau:.1f} ms/char pre-trim, "
                      f"{plateau_p:.1f} post-trim. Inflation above the plateau is EITHER "
                      "XTTS speaking words the text does not contain (the rambling the "
                      "floor was introduced to prevent) OR leading/trailing silence. The "
                      "post-trim column separates them: if inflation survives trimming it "
                      "is invented speech; if it collapses to the plateau it was silence "
                      "and costs nothing.\n")
        for t, ms, mc, mcp in curve:
            pre = mc > 1.25 * plateau
            post_hot = mcp > 1.25 * plateau_p
            flag = ("  <-- INVENTED SPEECH (survives trim)" if pre and post_hot else
                    "  <-- silence only (trim removes it)" if pre else "")
            report.append(f"- {t:>4} chars: {ms:7.1f} ms to first audio, "
                          f"{mc:5.1f} -> {mcp:5.1f} ms/char{flag}")
        cheap = [c for c in curve if c[3] <= 1.25 * plateau_p]
        if cheap:
            best = min(cheap, key=lambda c: c[1])
            report.append(f"\n**Shortest faithful chunk (post-trim): {best[0]} chars at "
                          f"{best[1]:.0f} ms** to first audio.")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# C2 - XTTS first-chunk length sweep\n\n"
                f"XTTS-v3 fine-tune via tts_server.py on :{config.TTS_PORT}. "
                f"{args.reps} reps per length, {args.warmup} warmup discarded. "
                "Text is fixed and LLM-independent, unlike Table C2-2's tts_first_chunk.\n\n"
                + "\n".join(report) + "\n")
    print(f"\n(written to {args.out})")


if __name__ == "__main__":
    main()
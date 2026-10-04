#!/usr/bin/env python3
"""C2: efficiency of the fully-local Aiko pipeline on a single consumer GPU.

Measures the DEPLOYED code path (aiko/*), not a reimplementation:

  Phase 1  cold start   - load wall-time + VRAM delta per component, in Companion order
  Phase 2  warm latency - per-stage ms over N real utterances (mean/sd/p50/p95)
  Phase 3  conditional-reasoning ablation - emotional turn (3B thinker runs) vs neutral (skipped)
  Phase 4  VRAM attribution - per-process GPU memory (this proc / ollama / tts_server)
  Phase 5  Ollama internals - prompt-eval vs generation, tokens/sec; optional model comparison

Inputs are cached MELD test clips + their transcripts (real speech, already on disk).
Filtered to --min-sec..--max-sec so they resemble conversational user turns.

memory.recall() runs against the REAL ChromaDB (read-only, so recall time reflects the
live collection size). memory.store() is deliberately NEVER called: it is a write, and it
happens after the reply is produced, so excluding it keeps the user's memory store clean
without affecting time-to-reply.

Prerequisites: `ollama serve` running. TTS server optional (skipped with a note if :5123
is closed, or pass --no-tts).

    .venv/bin/python paper/c2_efficiency.py --reps 20
    .venv/bin/python paper/c2_efficiency.py --reps 20 --models aiko-v4,aiko-v4-f16
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics as st
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf
import torch

from aiko import config

OLLAMA = "http://localhost:11434"


# --------------------------------------------------------------------------- #
# Instrumentation helpers
# --------------------------------------------------------------------------- #
def sync() -> None:
    """GPU work is async; without this we would time kernel launches, not compute."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def timed(store: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sync()
        store.setdefault(key, []).append((time.perf_counter() - t0) * 1000.0)


def nvsmi(query: str, extra: str = "--format=csv,noheader,nounits") -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-{query}", extra],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def gpu_used_mb() -> float:
    rows = nvsmi("gpu=memory.used")
    try:
        return float(rows[0].split(",")[0])
    except (IndexError, ValueError):
        return 0.0


def gpu_total_mb() -> float:
    rows = nvsmi("gpu=memory.total")
    try:
        return float(rows[0].split(",")[0])
    except (IndexError, ValueError):
        return 0.0


def torch_mb() -> float:
    return torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "p50": 0.0, "p95": 0.0}
    s = sorted(xs)
    return {
        "n": len(xs),
        "mean": st.fmean(xs),
        "sd": st.stdev(xs) if len(xs) > 1 else 0.0,
        "p50": s[len(s) // 2],
        "p95": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
    }


def ollama_post(path: str, payload: dict, timeout: int = 180) -> dict | None:
    req = urllib.request.Request(
        f"{OLLAMA}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        print(f"  ! ollama {path} failed: {exc}", flush=True)
        return None


def ollama_tags() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
            return sorted(m["name"] for m in json.loads(r.read().decode()).get("models", []))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_inputs(scores_path, wav_dir, n, lo, hi):
    """(wav, transcript, duration) triples from the cached MELD clips."""
    picked = []
    for line in open(scores_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        wav = os.path.join(wav_dir, f"dia{r['dia']}_utt{r['utt']}.wav")
        if not os.path.exists(wav) or not r.get("text"):
            continue
        try:
            dur = sf.info(wav).duration
        except (RuntimeError, OSError):
            continue
        if lo <= dur <= hi:
            picked.append((wav, r["text"], dur))
        if len(picked) >= n:
            break
    return picked


# --------------------------------------------------------------------------- #
# The deployed reply path, instrumented (faithful to AikoChat.reply minus store)
# --------------------------------------------------------------------------- #
def reply_stages(chat, memory, text, emotion, conf, store: dict):
    with timed(store, "memory_recall"):
        memories = memory.recall(text) or "(nothing yet)"
    with timed(store, "think_3b"):
        thoughts = chat._reason(text, emotion, conf, memories)

    tagged = (f"[voice_emotion: {emotion}, confidence: {conf:.2f}] {text}"
              if emotion and conf > config.EMOTION_TAG_MIN_CONF else text)
    read = (f"\n\n[Privately, your read on this moment - let it shape your reply, "
            f"never mention it: {thoughts}]" if thoughts else "")
    system = config.AIKO_SYSTEM + read + f"\n[Things you remember about them:\n{memories}]"

    with timed(store, "chat_7b_q4"):
        reply = chat._chat.invoke([("system", system), ("human", tagged)]).content.strip()
    return reply, bool(thoughts)


def fmt(title, rows, cols):
    w = [max(len(c), *(len(str(r[i])) for r in rows)) if rows else len(c)
         for i, c in enumerate(cols)]
    out = [title, "", "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="paper/meld_test_scores.jsonl")
    ap.add_argument("--wavs", default="paper/meld_wavs")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=6.0)
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--models", default="", help="comma-separated ollama models to compare")
    ap.add_argument("--out", default="paper/c2_efficiency.md")
    args = ap.parse_args()

    report = []
    gpu_name = (nvsmi("gpu=name", "--format=csv,noheader") or ["unknown GPU"])[0]
    total_vram = gpu_total_mb()
    print(f"GPU: {gpu_name} ({total_vram:.0f} MB)   torch {torch.__version__}")
    tags = ollama_tags()
    if not tags:
        print("! Ollama not reachable at :11434 — start `ollama serve` first.")
        sys.exit(1)
    print(f"ollama models: {', '.join(tags)}\n")

    def have(m: str) -> bool:
        return (m in tags or f"{m}:latest" in tags
                or any(t.split(":")[0] == m for t in tags))

    # Preflight the deployed models: fail in a second rather than after ~20s of torch loads.
    missing = [m for m in (config.CHAT_MODEL, config.THINK_MODEL) if not have(m)]
    if missing:
        print(f"! missing from ollama: {', '.join(missing)}\n  restore with:")
        if not have(config.CHAT_MODEL):
            print(f"    ollama create {config.CHAT_MODEL} -f Modelfile --quantize q4_K_M")
        if not have(config.THINK_MODEL):
            print(f"    ollama pull {config.THINK_MODEL}")
        sys.exit(1)

    inputs = load_inputs(args.scores, args.wavs, args.reps + args.warmup,
                         args.min_sec, args.max_sec)
    if len(inputs) < args.warmup + 1:
        print(f"! only {len(inputs)} usable clips in {args.min_sec}-{args.max_sec}s")
        sys.exit(1)
    mean_dur = st.fmean(d for _, _, d in inputs)
    print(f"{len(inputs)} clips, mean duration {mean_dur:.2f}s "
          f"({args.warmup} warmup + {len(inputs) - args.warmup} measured)\n")

    # ---------------- Phase 1: cold start ---------------- #
    print("== Phase 1: cold start ==", flush=True)
    baseline_gpu = gpu_used_mb()
    cold = []

    def load(name, fn):
        g0, t0_mb = gpu_used_mb(), torch_mb()
        t0 = time.perf_counter()
        obj = fn()
        sync()
        dt = time.perf_counter() - t0
        cold.append([name, f"{dt:6.2f}", f"{torch_mb() - t0_mb:8.0f}",
                     f"{gpu_used_mb() - g0:8.0f}", f"{gpu_used_mb():8.0f}"])
        print(f"  {name:<22} {dt:6.2f}s  (+{gpu_used_mb() - g0:.0f} MB GPU)", flush=True)
        return obj

    from aiko.emotion import EmotionDetector
    from aiko.llm import AikoChat
    from aiko.memory import Memory
    from aiko.stt import Transcriber
    from aiko.tts import VoiceClient, split_into_chunks

    memory = load("Memory (ChromaDB)", Memory)
    det = load("EmotionDetector", EmotionDetector)
    stt = load("Whisper-small", Transcriber)
    chat = load("AikoChat (handles)", lambda: AikoChat(memory))
    voice = load("VoiceClient", VoiceClient)

    tts_up = (not args.no_tts) and VoiceClient._port_open()
    if not args.no_tts and not tts_up:
        print("  ! TTS server not on :5123 — TTS stages skipped")

    report.append(fmt("## Table C2-1 - Cold start (load time and VRAM per component)", cold,
                      ["Component", "Load s", "torch MB", "GPU dMB", "GPU MB"]))
    report.append(f"\nBaseline GPU before loading: {baseline_gpu:.0f} MB of "
                  f"{total_vram:.0f} MB. GPU: {gpu_name}.\n")

    # ---------------- Phase 2: warm per-stage latency ---------------- #
    print("\n== Phase 2: warm per-stage latency ==", flush=True)
    T: dict[str, list[float]] = {}
    for i, (wav, text, _dur) in enumerate(inputs):
        warm = i < args.warmup
        S: dict[str, list[float]] = {} if warm else T
        with timed(S, "stt_whisper"):
            hyp = stt.transcribe(wav)
        # detect() re-runs voice_scores internally, so time the TOTAL first, on cold audio.
        # The component splits below then run warm: they are diagnostic only and are never
        # summed into end-to-end, which uses emotion_detect_total.
        with timed(S, "emotion_detect_total"):
            res = det.detect(wav, hyp or text)
        with timed(S, "emotion_voice"):
            det.voice_scores(wav)
        with timed(S, "emotion_text"):
            det.text_scores(hyp or text)

        reply, did_think = reply_stages(chat, memory, hyp or text,
                                        res.emotion, res.confidence, S)

        if tts_up:
            chunks = split_into_chunks(reply)
            with timed(S, "tts_first_chunk"):
                voice.synthesize(chunks[0], os.path.join(
                    config.VOICE_OUTPUT_DIR, "c2_bench_first.wav"))
            if len(chunks) > 1:
                with timed(S, "tts_remaining_chunks"):
                    for c in chunks[1:]:
                        voice.synthesize(c, os.path.join(
                            config.VOICE_OUTPUT_DIR, "c2_bench_rest.wav"))
            S.setdefault("tts_n_chunks", []).append(len(chunks))

        tag = "warmup" if warm else f"{i - args.warmup + 1}/{len(inputs) - args.warmup}"
        print(f"  [{tag}] think={'yes' if did_think else 'no ':<3} "
              f"emotion={res.emotion:<9} reply={reply[:44]!r}", flush=True)

    order = ["stt_whisper", "emotion_voice", "emotion_text", "emotion_detect_total",
             "memory_recall", "think_3b", "chat_7b_q4", "tts_first_chunk",
             "tts_remaining_chunks"]
    rows = []
    for k in order:
        if k in T and T[k]:
            s = stats(T[k])
            rows.append([k, s["n"], f"{s['mean']:9.1f}", f"{s['sd']:7.1f}",
                         f"{s['p50']:8.1f}", f"{s['p95']:8.1f}"])
    report.append("\n" + fmt("## Table C2-2 - Warm per-stage latency (ms)", rows,
                             ["Stage", "n", "mean", "sd", "p50", "p95"]))

    # user-perceived latency: transcript -> first audible audio
    perceived = ["stt_whisper", "emotion_detect_total", "memory_recall", "think_3b",
                 "chat_7b_q4"] + (["tts_first_chunk"] if tts_up else [])
    e2e = sum(stats(T[k])["mean"] for k in perceived if k in T)
    to_text = sum(stats(T[k])["mean"] for k in perceived[:-1] if k in T) if tts_up else e2e
    report.append(f"\n**End-to-end (mic-ready audio -> first spoken audio): "
                  f"{e2e / 1000:.2f} s**; to reply text: {to_text / 1000:.2f} s. "
                  f"Excludes VAD hang ({config.VAD_SILENCE_HANG:.1f}s) and memory.store "
                  f"(post-reply write).\n")
    print(f"\n  end-to-end {e2e / 1000:.2f}s   (to text {to_text / 1000:.2f}s)")

    # ---------------- Phase 3: conditional reasoning ---------------- #
    print("\n== Phase 3: conditional-reasoning ablation ==", flush=True)
    abl_rows = []
    for label, emo, conf in (("emotional (thinker runs)", "sad", 0.80),
                             ("neutral (thinker skipped)", "neutral", 0.90)):
        A: dict[str, list[float]] = {}
        for j, (_w, text, _d) in enumerate(inputs[: args.warmup + 8]):
            S = {} if j < args.warmup else A
            reply_stages(chat, memory, text, emo, conf, S)
        tot = sum(stats(A[k])["mean"] for k in ("memory_recall", "think_3b", "chat_7b_q4")
                  if k in A)
        abl_rows.append([label, stats(A.get("think_3b", []))["n"],
                         f"{stats(A.get('think_3b', []))['mean']:8.1f}",
                         f"{stats(A.get('chat_7b_q4', []))['mean']:9.1f}",
                         f"{tot:9.1f}"])
        print(f"  {label:<26} reply-path {tot:7.1f} ms", flush=True)
    report.append("\n" + fmt("## Table C2-3 - Conditional reasoning (reply path only, ms)",
                             abl_rows, ["Condition", "n", "think 3B", "chat 7B", "total"]))

    # ---------------- Phase 4: VRAM attribution ---------------- #
    print("\n== Phase 4: VRAM attribution ==", flush=True)
    procs, mine, other = [], 0.0, 0.0
    ours = {"python", "llama-server", "ollama"}
    for row in nvsmi("compute-apps=pid,process_name,used_gpu_memory"):
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 3:
            continue
        # nvidia-smi reports full argv for some processes; basename alone keeps the flags
        name = os.path.basename(parts[1].split(" ", 1)[0]) or parts[1]
        try:
            mb = float(parts[2])
        except ValueError:
            mb = 0.0
        tag = "aiko" if name in ours else "other"
        mine, other = (mine + mb, other) if tag == "aiko" else (mine, other + mb)
        procs.append([parts[0], name[:24], tag, parts[2]])
        print(f"  pid {parts[0]:<8} {name[:24]:<24} {tag:<5} {parts[2]} MB")
    report.append("\n" + fmt("## Table C2-4 - GPU memory by process (MB)", procs,
                             ["PID", "Process", "Owner", "VRAM MB"]))
    report.append(f"\n**Aiko's own footprint: {mine:.0f} MB of {total_vram:.0f} MB "
                  f"({100 * mine / total_vram:.0f} % of the card).** Desktop/IDE overhead "
                  f"present during measurement but not part of the system: {other:.0f} MB. "
                  f"Total in use: {gpu_used_mb():.0f} MB "
                  f"({100 * gpu_used_mb() / total_vram:.0f} %).\n")

    # ---------------- Phase 5: Ollama internals / model comparison ---------------- #
    print("\n== Phase 5: Ollama internals ==", flush=True)
    probe = inputs[args.warmup][1] if len(inputs) > args.warmup else "hey, how was your day?"
    models = [m for m in (args.models.split(",") if args.models else [config.CHAT_MODEL]) if m]
    mrows = []
    for m in models:
        if not have(m):
            print(f"  ! {m} not in ollama — skipped")
            continue
        got = []
        for j in range(args.warmup + 5):
            r = ollama_post("/api/chat", {
                "model": m, "stream": False,
                "messages": [{"role": "system", "content": config.AIKO_SYSTEM},
                             {"role": "user", "content": probe}],
                "options": {k: v for k, v in config.CHAT_DECODE.items() if k != "keep_alive"},
                "keep_alive": -1,
            })
            if r and j >= args.warmup:
                got.append(r)
        if not got:
            continue
        ns = 1e6  # ns -> ms

        def avg(field):
            vals = [g.get(field, 0) for g in got if g.get(field)]
            return st.fmean(vals) if vals else 0.0

        tps = st.fmean([g["eval_count"] / (g["eval_duration"] / 1e9)
                        for g in got if g.get("eval_duration")])
        # everything total_duration does not attribute to load/prompt/generate
        residual = (avg("total_duration") - avg("load_duration")
                    - avg("prompt_eval_duration") - avg("eval_duration")) / ns
        mrows.append([m, len(got), f"{avg('total_duration') / ns:9.0f}",
                      f"{avg('load_duration') / ns:8.0f}",
                      f"{avg('prompt_eval_duration') / ns:10.0f}",
                      f"{avg('eval_duration') / ns:9.0f}", f"{residual:7.0f}",
                      f"{avg('eval_count'):6.0f}", f"{tps:7.1f}"])
        print(f"  {m:<18} total {avg('total_duration') / ns:7.0f} ms   "
              f"load {avg('load_duration') / ns:5.0f}   other {residual:5.0f}   "
              f"{tps:.1f} tok/s")
    report.append("\n" + fmt("## Table C2-5 - Ollama generation internals (ms, mean)", mrows,
                             ["Model", "n", "total", "load", "prompt eval", "generate",
                              "other", "tokens", "tok/s"]))

    text = "\n".join(["# C2 - Efficiency of the fully-local Aiko pipeline\n",
                      f"GPU {gpu_name}, {total_vram:.0f} MB. torch {torch.__version__}. "
                      f"{len(inputs) - args.warmup} measured turns, mean input "
                      f"{mean_dur:.2f}s, {args.warmup} warmup discarded.\n"] + report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n(written to {args.out})")


if __name__ == "__main__":
    main()
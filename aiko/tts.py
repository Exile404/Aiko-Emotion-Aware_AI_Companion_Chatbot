"""Client for the persistent XTTS voice server (``tts_server.py``).

The server loads the fine-tuned XTTS model once and synthesizes over a socket, so the
notebook/app never pays the model-load cost per reply. This module also handles the two
text-side fixes XTTS needs: splitting a reply into chunks long enough that the model does
not ramble, and trimming over-long silent gaps from the generated wav.
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import time
import uuid

import numpy as np
import soundfile as sf

from . import config

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_into_chunks(text: str) -> list[str]:
    """Group sentences into chunks of >= ``TTS_CHUNK_MIN_CHARS`` characters.

    XTTS hallucinates on very short inputs, so short replies stay whole and only long
    replies split — each chunk staying long enough to synthesize cleanly.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split((text or "").strip()) if s.strip()]
    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        buf = f"{buf} {sentence}".strip() if buf else sentence
        if len(buf) >= config.TTS_CHUNK_MIN_CHARS:
            chunks.append(buf)
            buf = ""
    if buf:
        if chunks and len(buf) < config.TTS_CHUNK_MIN_TAIL:
            chunks[-1] += " " + buf
        else:
            chunks.append(buf)
    return chunks or [(text or "").strip() or "Mhm."]


def trim_pauses(path: str) -> str:
    """Compress silent gaps longer than ``TRIM_MAX_PAUSE`` so the voice doesn't stall."""
    x, sr = sf.read(path)
    mono = x.mean(axis=1) if x.ndim > 1 else x
    hop = max(1, int(sr * config.TRIM_WIN))
    silent = np.zeros(len(mono), dtype=bool)
    for s in range(0, len(mono), hop):
        seg = mono[s:s + hop]
        if len(seg) and np.sqrt(np.mean(seg ** 2)) < config.TRIM_THRESH:
            silent[s:s + len(seg)] = True

    keep = np.ones(len(mono), dtype=bool)
    cap = int(config.TRIM_MAX_PAUSE * sr)
    i = 0
    while i < len(mono):
        if silent[i]:
            j = i
            while j < len(mono) and silent[j]:
                j += 1
            if (j - i) > cap:
                keep[i + cap // 2: j - cap // 2] = False
            i = j
        else:
            i += 1
    sf.write(path, x[keep], sr)
    return path


class VoiceClient:
    """Synthesizes text by talking to the running XTTS server."""

    def __init__(self) -> None:
        os.makedirs(config.VOICE_OUTPUT_DIR, exist_ok=True)

    @staticmethod
    def _port_open() -> bool:
        with socket.socket() as s:
            s.settimeout(0.5)
            return s.connect_ex((config.TTS_HOST, config.TTS_PORT)) == 0

    def start_server(self, wait: int = 180) -> bool:
        """Launch ``tts_server.py`` (in its own venv) and wait until it accepts connections."""
        if self._port_open():
            log.info("TTS server already running on :%d", config.TTS_PORT)
            return True
        log.info("Launching %s (loading XTTS, ~20-40s)...", config.TTS_SERVER_SCRIPT)
        log_file = open(config.TTS_SERVER_LOG, "w")
        self._proc = subprocess.Popen(
            [config.TTS_VENV_PYTHON, config.TTS_SERVER_SCRIPT],
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        start = time.time()
        while time.time() - start < wait:
            if self._proc.poll() is not None:
                log.error("TTS server exited (code %s); see %s", self._proc.returncode, config.TTS_SERVER_LOG)
                return False
            if self._port_open():
                log.info("TTS server ready in %.0fs", time.time() - start)
                return True
            time.sleep(1)
        log.error("Timed out waiting for the TTS server; see %s", config.TTS_SERVER_LOG)
        return False

    def synthesize(self, text: str, output_path: str | None = None) -> str | None:
        """Synthesize ``text`` to a wav file. Returns the path, or None if the server is down."""
        if output_path is None:
            output_path = os.path.join(config.VOICE_OUTPUT_DIR, f"aiko_{uuid.uuid4().hex[:8]}.wav")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(120)
                s.connect((config.TTS_HOST, config.TTS_PORT))
                s.send(json.dumps({"text": text, "output": os.path.abspath(output_path)}).encode())
                resp = s.recv(1024).decode()
            if resp.startswith("OK"):
                return output_path
            log.error("TTS error: %s", resp)
            return None
        except OSError as exc:
            log.error("TTS server unreachable: %s (is tts_server.py running?)", exc)
            return None

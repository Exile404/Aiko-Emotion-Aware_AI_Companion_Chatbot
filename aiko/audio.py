"""Microphone capture: fixed-duration recording and energy-based VAD.

Each captured utterance is written at two sample rates — high quality for the emotion
model and 16 kHz for Whisper — and returned as ``(emotion_wav, whisper_wav)`` paths.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Iterator

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from . import config

log = logging.getLogger(__name__)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)) + 1e-9)


def _write_pair(audio: np.ndarray) -> tuple[str, str]:
    """Write ``audio`` (mono, SR_HIGH) to temp wavs for the emotion model and Whisper."""
    emotion_wav = os.path.join(tempfile.gettempdir(), "aiko_emotion.wav")
    whisper_wav = os.path.join(tempfile.gettempdir(), "aiko_whisper.wav")
    sf.write(emotion_wav, audio, config.SR_HIGH)
    resampled = librosa.resample(audio, orig_sr=config.SR_HIGH, target_sr=config.SR_WHISPER)
    sf.write(whisper_wav, resampled, config.SR_WHISPER)
    return emotion_wav, whisper_wav


def record_fixed(duration: int = 5) -> tuple[str, str]:
    """Record a fixed window (push-to-talk style) and return the two wav paths."""
    log.info("Recording for %ds...", duration)
    audio = sd.rec(int(duration * config.SR_HIGH), samplerate=config.SR_HIGH, channels=1, dtype="float32")
    sd.wait()
    return _write_pair(audio.flatten())


class VoiceActivityDetector:
    """Hands-free capture: yields one ``(emotion_wav, whisper_wav)`` per detected utterance."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold

    def calibrate(self, seconds: float = 1.0) -> float:
        """Measure the ambient noise floor and set the speech threshold."""
        log.info("Calibrating ambient noise — stay quiet...")
        rec = sd.rec(int(seconds * config.SR_HIGH), samplerate=config.SR_HIGH, channels=1, dtype="float32")
        sd.wait()
        floor = _rms(rec.flatten())
        self.threshold = max(floor * 3.5, config.VAD_THRESHOLD_FLOOR)
        log.info("Noise floor %.4f -> speech threshold %.4f", floor, self.threshold)
        return self.threshold

    def utterances(self) -> Iterator[tuple[str, str]]:
        """Yield captured utterances forever (stop by breaking out / KeyboardInterrupt)."""
        threshold = self.threshold or self.calibrate()
        hang = int(config.VAD_SILENCE_HANG * config.SR_HIGH / config.VAD_FRAME)
        with sd.InputStream(samplerate=config.SR_HIGH, channels=1, dtype="float32",
                            blocksize=config.VAD_FRAME) as stream:
            while True:
                # wait for speech to start
                voiced = 0
                blk = None
                while voiced < config.VAD_START_FRAMES:
                    blk, _ = stream.read(config.VAD_FRAME)
                    voiced = voiced + 1 if _rms(blk[:, 0]) > threshold else 0
                # capture until the silence-hang elapses
                frames, quiet = [blk[:, 0].copy()], 0
                while quiet < hang:
                    blk, _ = stream.read(config.VAD_FRAME)
                    amp = blk[:, 0]
                    frames.append(amp.copy())
                    quiet = quiet + 1 if _rms(amp) < threshold else 0

                audio = np.concatenate(frames)
                if len(audio) / config.SR_HIGH < config.VAD_MIN_SPEECH:
                    continue

                yield _write_pair(audio)

                # drain buffered audio so Aiko's own voice / the tail doesn't re-trigger
                time.sleep(0.15)
                pending = stream.read_available
                if pending:
                    stream.read(pending)

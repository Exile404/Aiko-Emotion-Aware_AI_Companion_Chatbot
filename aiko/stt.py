"""Speech-to-text via OpenAI Whisper."""
from __future__ import annotations

import logging

from . import config

log = logging.getLogger(__name__)


class Transcriber:
    """Thin wrapper around a loaded Whisper model."""

    def __init__(self) -> None:
        import torch
        import whisper

        self._fp16 = torch.cuda.is_available()
        device = "cuda" if self._fp16 else "cpu"
        log.info("Loading Whisper (%s) on %s...", config.WHISPER_MODEL, device)
        self._model = whisper.load_model(config.WHISPER_MODEL, device=device)

    def transcribe(self, audio_path: str) -> str:
        """Return the transcript of a 16 kHz wav file (empty string on silence)."""
        result = self._model.transcribe(audio_path, language="en", fp16=self._fp16)
        return result["text"].strip()

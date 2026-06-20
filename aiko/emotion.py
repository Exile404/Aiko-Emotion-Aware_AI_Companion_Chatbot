"""Hybrid emotion detection: late fusion of a voice model and a text model.

The voice model (emotion2vec+) reads tone from the audio; the text model (DistilRoBERTa)
reads sentiment from the transcript. They are fused with per-emotion reliability weights
plus two corrections learned empirically:

* short utterances (<= a few words) are unreliable for the voice model, so its weight is
  cut and the text model leads;
* a winning emotion below a confidence floor collapses to ``neutral`` — better a neutral
  read than a confidently wrong one.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import NamedTuple

from . import config

log = logging.getLogger(__name__)


class EmotionResult(NamedTuple):
    emotion: str
    confidence: float
    voice_top: str
    text_top: str
    short: bool
    top3: list[tuple[str, float]]


class EmotionDetector:
    """Loads the voice + text emotion models and fuses their predictions."""

    def __init__(self) -> None:
        import torch
        from funasr import AutoModel
        from transformers import pipeline as hf_pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading text emotion model (CPU)...")
        self._text = hf_pipeline(
            "text-classification",
            model=config.TEXT_EMOTION_MODEL,
            top_k=None,
            device=-1,
        )
        log.info("Loading voice emotion model (%s)...", device)
        self._voice = AutoModel(model=config.VOICE_EMOTION_MODEL, device=device)
        self._history: deque[str] = deque(maxlen=config.EMOTION_HISTORY)

    # ----------------------------- per-modality ---------------------------- #
    def _uniform(self) -> dict[str, float]:
        return {e: 1 / len(config.UNIFIED_LABELS) for e in config.UNIFIED_LABELS}

    def voice_scores(self, audio_path: str) -> dict[str, float]:
        try:
            result = self._voice.generate(audio_path, granularity="utterance")
            if not result:
                return self._uniform()
            raw = result[0]["scores"]
            scores: dict[str, float] = {}
            for i, label in enumerate(config.VOICE_LABELS):
                mapped = "neutral" if label in ("other", "unknown") else label
                scores[mapped] = scores.get(mapped, 0.0) + float(raw[i])
            total = sum(scores.values())
            return {k: v / total for k, v in scores.items()} if total else scores
        except Exception as exc:  # pragma: no cover
            log.warning("voice emotion failed: %s", exc)
            return self._uniform()

    def text_scores(self, text: str) -> dict[str, float]:
        try:
            if not text or len(text.strip()) < 2:
                return self._uniform()
            results = self._text(text)[0]
            return {
                config.TEXT_TO_UNIFIED.get(r["label"], "neutral"): float(r["score"])
                for r in results
            }
        except Exception as exc:  # pragma: no cover
            log.warning("text emotion failed: %s", exc)
            return self._uniform()

    # -------------------------------- fusion -------------------------------- #
    def _fuse(self, voice: dict[str, float], text: dict[str, float], n_words: int) -> EmotionResult:
        short = n_words <= config.SHORT_UTTERANCE_WORDS
        voice_weight = config.VOICE_SHORT_WEIGHT if short else 1.0

        fused = {
            emo: voice.get(emo, 0.0) * config.VOICE_RELIABILITY[emo] * voice_weight
            + text.get(emo, 0.0) * config.TEXT_RELIABILITY[emo]
            for emo in config.UNIFIED_LABELS
        }

        voice_top = max(voice, key=voice.get)
        text_top = max(text, key=text.get)
        text_conf = text.get(text_top, 0.0)

        if voice_top == text_top:
            fused[voice_top] *= 1.20
        if text_top in ("fearful", "sad") and text_conf > 0.40 and voice_top in ("neutral", "happy"):
            fused[text_top] *= 1.35
            fused[voice_top] *= 0.80
        if not short and voice_top in ("angry", "happy") and voice.get(voice_top, 0.0) > 0.70 and text_top == "neutral":
            fused[voice_top] *= 1.25
        for recent in list(self._history)[-2:]:
            if recent != "neutral" and recent in fused:
                fused[recent] *= 1.05

        total = sum(fused.values())
        if total:
            fused = {k: v / total for k, v in fused.items()}

        best = max(fused, key=fused.get)
        if best != "neutral" and fused[best] < config.NEUTRAL_CONF_FLOOR:
            best = "neutral"  # too uncertain to risk a wrong mood tag
        self._history.append(best)

        top3 = sorted(fused.items(), key=lambda kv: -kv[1])[:3]
        return EmotionResult(best, fused[best], voice_top, text_top, short, top3)

    def detect(self, audio_path: str, text: str) -> EmotionResult:
        """Detect the user's emotion from their audio + transcript."""
        n_words = len((text or "").split())
        return self._fuse(self.voice_scores(audio_path), self.text_scores(text), n_words)

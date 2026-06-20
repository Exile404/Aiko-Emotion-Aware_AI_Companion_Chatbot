"""Drives the browser Live2D avatar over WebSocket and maps replies to expressions.

The avatar's face uses its own emotion read — derived from the *text of Aiko's reply*
(warm palette), separate from the user-emotion detection in :mod:`aiko.emotion`.
"""
from __future__ import annotations

import json
import logging
import re

from websockets.sync.client import connect

from . import config
from .emotion import EmotionDetector

log = logging.getLogger(__name__)


class Avatar:
    """Sends play/emotion commands to the browser viewer and picks facial expressions."""

    def __init__(self, emotion_detector: EmotionDetector) -> None:
        self._emotion = emotion_detector
        self._warm = re.compile(config.AVATAR_WARM_CUES, re.IGNORECASE)

    def expression_for(self, reply_text: str) -> str:
        """Choose the avatar's facial emotion from the text of Aiko's reply."""
        scores = self._emotion.text_scores(reply_text)
        if self._warm.search(reply_text or "") and scores.get("sad", 0.0) < config.AVATAR_SAD_GATE:
            return "happy"
        folded: dict[str, float] = {}
        for emo, score in scores.items():
            target = config.AVATAR_REMAP.get(emo, emo)
            folded[target] = folded.get(target, 0.0) + score
        return max(folded, key=folded.get)

    def play(self, audio_url: str, emotion: str = "neutral") -> bool:
        """Tell the browser to play ``audio_url`` (relative to the page) with an expression."""
        try:
            with connect(config.AVATAR_WS_URL, open_timeout=3) as ws:
                ws.send(json.dumps({"type": "speak", "audio": audio_url, "emotion": emotion}))
            return True
        except Exception as exc:  # avatar is optional; voice still plays without it
            log.warning("avatar not reachable: %s", exc)
            return False

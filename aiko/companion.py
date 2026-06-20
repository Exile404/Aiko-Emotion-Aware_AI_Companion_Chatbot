"""The Companion ties every component into one conversation loop.

    mic / text -> STT -> emotion -> reply (memory + reasoning) -> voice -> avatar

It owns the loaded models and exposes the two conversation modes: hands-free voice
(:meth:`converse`) and an interactive text console (:meth:`chat_loop`).
"""
from __future__ import annotations

import logging
import os
import uuid

from . import audio, config
from .avatar import Avatar
from .emotion import EmotionDetector
from .llm import AikoChat
from .memory import Memory
from .stt import Transcriber
from .tts import VoiceClient, split_into_chunks, trim_pauses

log = logging.getLogger(__name__)


class Companion:
    """Loads all models and runs the end-to-end conversation pipeline."""

    def __init__(self) -> None:
        self.memory = Memory()
        self.emotion = EmotionDetector()
        self.stt = Transcriber()
        self.chat = AikoChat(self.memory)
        self.voice = VoiceClient()
        self.avatar = Avatar(self.emotion)
        os.makedirs(config.AVATAR_VOICE_DIR, exist_ok=True)

    def start(self) -> None:
        """Bring up the TTS server (models are already loaded in __init__)."""
        if not self.voice.start_server():
            log.warning("TTS server failed to start — Aiko will reply in text only.")

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def say(self, text: str, emotion: str | None = None) -> str | None:
        """Speak ``text`` (sentence-chunked) and drive the avatar. Returns the last wav path."""
        emotion = emotion or self.avatar.expression_for(text)
        last_wav = None
        for chunk in split_into_chunks(text):
            fname = f"aiko_{uuid.uuid4().hex[:8]}.wav"
            wav = self.voice.synthesize(chunk, os.path.join(config.AVATAR_VOICE_DIR, fname))
            if wav:
                trim_pauses(wav)
                self.avatar.play(f"voice_output/{fname}", emotion)  # relative to the avatar page
                last_wav = wav
        return last_wav

    # ------------------------------------------------------------------ #
    # A single voice turn
    # ------------------------------------------------------------------ #
    def handle_turn(self, emotion_wav: str, whisper_wav: str) -> str | None:
        """Transcribe, detect emotion, reply, and speak. Returns the reply (or None if silent)."""
        text = self.stt.transcribe(whisper_wav)
        if not text or len(text.strip()) < 2:
            return None
        result = self.emotion.detect(emotion_wav, text)
        reply = self.chat.reply(text, result.emotion, result.confidence)
        print(f"You [{result.emotion} {result.confidence:.0%}]: {text}")
        print(f"Aiko: {reply}\n")
        self.say(reply)
        return reply

    # ------------------------------------------------------------------ #
    # Conversation modes
    # ------------------------------------------------------------------ #
    def converse(self) -> None:
        """Hands-free voice conversation via energy-based VAD."""
        vad = audio.VoiceActivityDetector()
        print("Hands-free — just talk. Press Ctrl+C to stop.\n")
        try:
            for emotion_wav, whisper_wav in vad.utterances():
                self.handle_turn(emotion_wav, whisper_wav)
        except KeyboardInterrupt:
            print("\nAiko: Talk to you later.")

    def chat_loop(self) -> None:
        """Interactive text console. Commands: !clear, quit."""
        print("Text chat — type a message. Commands: !clear, quit\n")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAiko: Talk to you later.")
                return
            if text.lower() in ("quit", "exit", "bye"):
                print("Aiko: Talk to you later.")
                return
            if text.lower() == "!clear":
                self.memory.clear()
                continue
            if not text:
                continue
            scores = self.emotion.text_scores(text)
            emotion = max(scores, key=scores.get)
            reply = self.chat.reply(text, emotion, scores[emotion])
            print(f"Aiko: {reply}\n")
            self.say(reply)

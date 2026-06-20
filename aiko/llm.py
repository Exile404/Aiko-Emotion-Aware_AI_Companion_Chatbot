"""Aiko's reply generation: memory recall -> conditional reasoning -> persona reply.

Two Ollama models are used. A small "strategist" (``THINK_MODEL``) produces private notes
only on emotional turns; the fine-tuned persona model (``CHAT_MODEL``) writes the actual
reply. Plain/neutral turns skip the reasoning call entirely for speed.
"""
from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from . import config
from .memory import Memory

log = logging.getLogger(__name__)


class AikoChat:
    """Generates Aiko's reply for a user turn, using memory and optional reasoning."""

    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        self._chat = ChatOllama(model=config.CHAT_MODEL, **config.CHAT_DECODE)
        think_llm = ChatOllama(model=config.THINK_MODEL, **config.THINK_DECODE)
        think_prompt = ChatPromptTemplate.from_messages([
            ("system", config.THINK_SYSTEM),
            ("human", config.THINK_HUMAN),
        ])
        self._think = think_prompt | think_llm | StrOutputParser()
        log.info("Chat model=%s, reasoning model=%s", config.CHAT_MODEL, config.THINK_MODEL)

    def _reason(self, user_msg: str, emotion: str, conf: float, memories: str) -> str:
        """Private notes from the strategist model — only for genuinely emotional turns."""
        if not (emotion and emotion != "neutral" and conf > config.THINK_MIN_CONF):
            return ""
        return self._think.invoke({
            "msg": user_msg,
            "emotion": emotion,
            "conf": f"{conf:.0%}",
            "memories": memories,
        }).strip()

    def reply(self, user_msg: str, emotion: str | None = None, conf: float = 0.0,
              show_thinking: bool = True) -> str:
        """Produce Aiko's reply and persist the user's message to memory."""
        tagged = (
            f"[voice_emotion: {emotion}, confidence: {conf:.2f}] {user_msg}"
            if emotion and conf > config.EMOTION_TAG_MIN_CONF else user_msg
        )
        memories = self.memory.recall(user_msg) or "(nothing yet)"

        thoughts = self._reason(user_msg, emotion or "", conf, memories)
        if show_thinking and thoughts:
            print(f"\n[thinking]\n{thoughts}\n" + "-" * 44)

        read = (
            f"\n\n[Privately, your read on this moment - let it shape your reply, "
            f"never mention it: {thoughts}]"
            if thoughts else ""
        )
        system = config.AIKO_SYSTEM + read + f"\n[Things you remember about them:\n{memories}]"
        reply = self._chat.invoke([("system", system), ("human", tagged)]).content.strip()

        self.memory.store(user_msg, emotion)
        return reply

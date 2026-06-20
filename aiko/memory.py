"""Long-term conversational memory backed by ChromaDB.

Only the user's side of each turn is stored: recalling Aiko's own past replies caused a
self-echo loop where she kept repeating themes. Storing what the *user* said surfaces
context about them instead.
"""
from __future__ import annotations

import datetime
import logging
import uuid

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from . import config

log = logging.getLogger(__name__)


class Memory:
    """Persistent semantic memory of the user's messages."""

    def __init__(self) -> None:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            model_kwargs={"device": "cpu"},  # keep VRAM free for the language models
        )
        self._db = Chroma(
            collection_name=config.MEMORY_COLLECTION,
            embedding_function=embeddings,
            persist_directory=config.MEMORY_DIR,
        )
        log.info("Memory ready (%d stored)", self.count())

    def store(self, user_msg: str, emotion: str | None = None) -> None:
        """Persist a single user message with its detected emotion."""
        self._db.add_texts(
            texts=[f"User said: {user_msg}"],
            metadatas=[{
                "emotion": emotion or "neutral",
                "time": datetime.datetime.now().isoformat(),
            }],
            ids=[str(uuid.uuid4())],
        )

    def recall(self, query: str, k: int = config.MEMORY_TOP_K) -> str:
        """Return up to ``k`` relevant past messages as a bulleted block (empty if none)."""
        try:
            hits = self._db.similarity_search(query, k=k)
        except Exception:  # pragma: no cover - empty store / backend hiccup
            return ""
        return "\n".join(f"- {h.page_content}" for h in hits)

    def clear(self) -> int:
        """Delete all stored memories. Returns how many were removed."""
        ids = self._db._collection.get()["ids"]
        if ids:
            self._db._collection.delete(ids=ids)
        log.info("Cleared %d memories", len(ids))
        return len(ids)

    def count(self) -> int:
        return self._db._collection.count()

"""Aiko — a fully-local, emotion-aware conversational companion.

Pipeline: speech -> emotion -> fine-tuned LLM (+ memory & reasoning) -> voice -> avatar.

Typical use::

    from aiko import Companion
    aiko = Companion()
    aiko.start()
    aiko.converse()        # hands-free voice

``Companion`` is imported lazily (PEP 562) so that ``import aiko`` and ``aiko.config``
stay light — useful for the training scripts, which need only the prompt/constants.
"""
from __future__ import annotations

__all__ = ["Companion"]
__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "Companion":
        from .companion import Companion
        return Companion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

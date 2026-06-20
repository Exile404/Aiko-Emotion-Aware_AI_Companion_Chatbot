"""Command-line entry point.

    python -m aiko            # hands-free voice conversation (default)
    python -m aiko --mode text
    python -m aiko --no-tts   # skip launching the voice server (text replies only)

Requires Ollama running with the ``aiko-v4`` and ``qwen2.5:3b`` models. For voice + avatar,
also run ``avatar_ws.py`` and serve ``avatar/`` in the browser (see README).
"""
from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="aiko", description="Run the Aiko companion.")
    parser.add_argument("--mode", choices=["vad", "text"], default="vad",
                        help="vad = hands-free voice (default); text = typed console")
    parser.add_argument("--no-tts", action="store_true", help="don't launch the TTS voice server")
    parser.add_argument("--log", default="INFO", help="logging level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper(), format="%(levelname)s %(name)s: %(message)s")

    from .companion import Companion  # imported here so --help is instant (no model load)

    aiko = Companion()
    if not args.no_tts:
        aiko.start()

    if args.mode == "text":
        aiko.chat_loop()
    else:
        aiko.converse()


if __name__ == "__main__":
    main()

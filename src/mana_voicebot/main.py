from __future__ import annotations

import argparse

from dotenv import load_dotenv
from openai import OpenAI

from .config import BotConfig
from .core.orchestrator import MultiDomainBot
from .io.voice_io import VoiceIO


def main() -> None:
    load_dotenv()
    client = OpenAI()
    config = BotConfig()

    parser = argparse.ArgumentParser(description="Mana multi-domain voice bot.")
    parser.add_argument("--voice", action="store_true", help="Use voice mode.")
    args = parser.parse_args()

    if args.voice:
        voice = VoiceIO(client, config)
        bot = MultiDomainBot(client, config, voice_io=voice)
        bot.loop_voice()
    else:
        bot = MultiDomainBot(client, config, voice_io=None)
        bot.loop_text_only()


if __name__ == "__main__":
    main()

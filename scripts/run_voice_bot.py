from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI
import sys
from pathlib import Path

# --- add src to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ----------------------------

from mana_voicebot.config import BotConfig
from mana_voicebot.core.orchestrator import MultiDomainBot
from mana_voicebot.io.voice_io import VoiceIO


def main() -> None:
    load_dotenv()
    client = OpenAI()
    cfg = BotConfig()
    voice = VoiceIO(client, cfg)
    bot = MultiDomainBot(client, cfg, voice_io=voice)
    bot.loop_voice()


if __name__ == "__main__":
    main()

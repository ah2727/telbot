from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import sys
from pathlib import Path

# --- add src to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# ----------------------------


from mana_voicebot.config import PROMPTS_DIR


def main() -> None:
    load_dotenv()
    target = PROMPTS_DIR / "main_system_prompt.txt"
    print("Training prompt (text mode). Paste your new system prompt, end with EOF (Ctrl+D).")
    print("Current file:", target)
    print("-" * 40)
    try:
        content = "".join(iter(input, None))  # will raise EOFError
    except EOFError:
        pass

    if not content.strip():
        print("No content captured. Prompt unchanged.")
        return

    target.write_text(content.strip(), encoding="utf-8")
    print(f"Prompt updated at {target}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
VAR_DIR = BASE_DIR / "var"

LOG_DIR = VAR_DIR / "logs"
SESSIONS_DIR = VAR_DIR / "sessions"
AUDIO_DEBUG_DIR = VAR_DIR / "audio"
PROMPTS_DIR = SRC_DIR / "mana_voicebot" / "prompts"
CLIENTS_FILE = VAR_DIR / "clients.json"

for p in (DATA_DIR, VAR_DIR, LOG_DIR, SESSIONS_DIR, AUDIO_DEBUG_DIR, PROMPTS_DIR):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class BotConfig:
    sample_rate: int = 16000
    record_seconds: float = 8.0

    push_chunk_seconds: float = 0.3
    push_silence_timeout: float = 1.5
    push_energy_threshold: float = 80.0
    silence_trim_threshold: float = 40.0

    realtime_chunk_seconds: float = 0.25
    realtime_silence_timeout: float = 0.35
    realtime_energy_threshold: float = 200.0

    vad_aggressiveness: int = 1
    use_vad_for_filtering: bool = False

    history_limit: int = 16

    tts_model: str = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    tts_voice: str = os.getenv("OPENAI_TTS_VOICE", "alloy")
    response_model: str = os.getenv("OPENAI_RESPONSE_MODEL", "gpt-4o-mini")
    transcription_model: str = os.getenv(
        "OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"
    )
    transcription_fallback: str = os.getenv(
        "OPENAI_TRANSCRIBE_FALLBACK", "gpt-4o-mini-transcribe"
    )

    data_dir: Path = DATA_DIR
    var_dir: Path = VAR_DIR
    log_dir: Path = LOG_DIR
    sessions_dir: Path = SESSIONS_DIR
    audio_debug_dir: Path = AUDIO_DEBUG_DIR
    prompts_dir: Path = PROMPTS_DIR
    clients_file: Path = CLIENTS_FILE

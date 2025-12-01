from __future__ import annotations

from openai import OpenAI

from ..config import BotConfig, AUDIO_DEBUG_DIR
from .audio_capture import AudioCapture
from .asr import ASR
from .tts import TTS


class VoiceIO:
    """
    High-level voice I/O:
    - record() -> transcript
    - speak(text)
    """

    def __init__(self, client: OpenAI, config: BotConfig):
        self.client = client
        self.config = config

        self.capture = AudioCapture(config, AUDIO_DEBUG_DIR)
        self.asr = ASR(client, config)
        self.tts = TTS(client, config)

    def record(self) -> str:
        audio = self.capture.record_push_to_talk()
        if audio.size == 0:
            return ""
        return self.asr.transcribe(audio)

    def speak(self, text: str) -> None:
        self.tts.speak(text)

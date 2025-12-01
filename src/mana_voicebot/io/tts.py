from __future__ import annotations

import io
import wave

import numpy as np
import pyttsx3
import sounddevice as sd
from openai import OpenAI

from ..config import BotConfig


class TTS:
    def __init__(self, client: OpenAI, config: BotConfig):
        self.client = client
        self.config = config
        self.tts_engine = pyttsx3.init()
        self._select_persian_voice()

    def speak(self, text: str) -> None:
        if not text:
            return
        audio_bytes = self._synthesize_with_openai(text)
        self._play_wav_bytes(audio_bytes)

    def _synthesize_with_openai(self, message: str) -> bytes:
        resp = self.client.audio.speech.create(
            model=self.config.tts_model,
            voice=self.config.tts_voice,
            input=message,
            response_format="wav",
            instructions="Speak naturally in Persian.",
        )
        return resp.read()

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        if not wav_bytes:
            return
        sd.stop()
        buffer = io.BytesIO(wav_bytes)
        with wave.open(buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width == 1:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if channels > 1:
            audio = audio.reshape(-1, channels)

        sd.play(audio, sample_rate)
        sd.wait()

    def _select_persian_voice(self) -> None:
        try:
            voices = self.tts_engine.getProperty("voices")
        except Exception:
            return

        def _match(voice, keyword: str) -> bool:
            name = (getattr(voice, "name", "") or "").lower()
            langs = ",".join(
                str(lang).lower() for lang in getattr(voice, "languages", []) or []
            )
            return keyword in name or keyword in langs

        for keyword in ("persian", "farsi", "iran"):
            for voice in voices:
                if _match(voice, keyword):
                    self.tts_engine.setProperty("voice", voice.id)
                    return

        for voice in voices:
            if _match(voice, "fa"):
                self.tts_engine.setProperty("voice", voice.id)
                return

from __future__ import annotations

import base64
import io
import wave
from typing import List

import numpy as np
from openai import OpenAI, OpenAIError

from ..config import BotConfig
from ..data.names import build_name_prompt


class ASR:
    """ASR wrapper around OpenAI audio + responses endpoints."""

    def __init__(self, client: OpenAI, config: BotConfig):
        self.client = client
        self.config = config
        self._warned = False

    def transcribe(self, audio: np.ndarray) -> str:
        buf = self._to_wav_bytes(audio)
        buf.name = "speech.wav"

        use_audio_endpoint = any(
            marker in self.config.transcription_model.lower()
            for marker in ("transcribe", "whisper")
        )
        if use_audio_endpoint:
            return self._via_audio_endpoint(buf, self.config.transcription_model)

        try:
            return self._via_responses(buf)
        except OpenAIError as exc:
            if not self._warned:
                print(
                    f"[asr] responses transcription failed ({exc}), falling back to {self.config.transcription_fallback}"
                )
                self._warned = True
            return self._via_audio_endpoint(buf, self.config.transcription_fallback)

    def _via_audio_endpoint(self, audio_buffer: io.BytesIO, model: str) -> str:
        clone = io.BytesIO(audio_buffer.getvalue())
        clone.name = "speech.wav"
        result = self.client.audio.transcriptions.create(
            model=model,
            file=clone,
            language="fa",
            prompt=build_name_prompt(),
        )
        return (result.text or "").strip()

    def _via_responses(self, audio_buffer: io.BytesIO) -> str:
        payload = base64.b64encode(audio_buffer.getvalue()).decode("ascii")
        instruction = (
            "Transcribe the following Persian speech. The audio is a base64-encoded WAV "
            "string. Decode it and reply with only the transcript.\n"
            f"{payload}"
        )
        resp = self.client.responses.create(
            model=self.config.transcription_model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": instruction}],
                }
            ],
        )
        return self._extract_text(resp)

    def _extract_text(self, response) -> str:
        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    chunks.append(getattr(content, "text", ""))
        return "".join(chunks).strip()

    def _to_wav_bytes(self, audio: np.ndarray) -> io.BytesIO:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(audio.tobytes())
        buf.seek(0)
        return buf

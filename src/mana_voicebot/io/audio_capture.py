from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Optional, List

import numpy as np
import sounddevice as sd
import webrtcvad

from ..config import BotConfig


class AudioCapture:
    def __init__(self, config: BotConfig, debug_dir: Path):
        self.config = config
        self.sample_rate = config.sample_rate
        self.debug_dir = debug_dir
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)

        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def record_push_to_talk(self) -> np.ndarray:
        print(
            f"Recording (max {self.config.record_seconds:.1f}s)... "
            "stop speaking to send sooner."
        )
        sd.stop()
        chunk_frames = max(1, int(self.sample_rate * self.config.push_chunk_seconds))
        max_frames = int(self.sample_rate * self.config.record_seconds)
        audio_queue: "queue.Queue[np.ndarray]"  # type: ignore[name-defined]

        import queue  # lazy import to keep top light
        audio_queue = queue.Queue()

        def _callback(indata, frames, time_info, status):
            if status:
                print(f"Audio warning: {status}")
            audio_queue.put(indata.copy().flatten())

        buffer: List[np.ndarray] = []
        total_frames = 0
        silence_since: Optional[float] = None

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=chunk_frames,
                callback=_callback,
            ):
                while True:
                    chunk = audio_queue.get()
                    buffer.append(chunk)
                    total_frames += len(chunk)
                    energy = float(np.mean(np.abs(chunk)))
                    now = time.time()
                    if energy >= self.config.push_energy_threshold:
                        silence_since = None
                    else:
                        silence_since = silence_since or now
                        elapsed_silence = now - silence_since
                        min_duration = 0.5
                        if (
                            elapsed_silence >= self.config.push_silence_timeout
                            and total_frames >= int(self.sample_rate * min_duration)
                        ):
                            break
                    if total_frames >= max_frames:
                        break
        except Exception as exc:
            print(f"Streaming record failed ({exc}); falling back to fixed window.")
            frames = max_frames
            audio = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="int16")
            sd.wait()
            return audio.flatten()

        if not buffer:
            return np.array([], dtype=np.int16)

        audio = np.concatenate(buffer)
        audio = self._trim_trailing_silence(audio)
        self._save_debug_wav(audio, "last_raw.wav")
        return audio

    def _trim_trailing_silence(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        threshold = self.config.silence_trim_threshold
        abs_audio = np.abs(audio)
        idx = len(audio) - 1
        while idx >= 0 and abs_audio[idx] <= threshold:
            idx -= 1
        if idx <= 0:
            return audio
        return audio[: idx + 1]

    def _save_debug_wav(self, audio: np.ndarray, name: str) -> None:
        path = self.debug_dir / name
        try:
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self.sample_rate)
                f.writeframes(audio.tobytes())
            print(f"[debug] saved raw audio to {path}")
        except Exception as exc:
            print(f"[debug] failed to save raw audio: {exc}")

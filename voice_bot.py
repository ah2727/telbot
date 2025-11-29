"""Voice-based doctor reservation assistant powered by OpenAI."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyttsx3
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

SYSTEM_PROMPT = (
    "You are an empathetic Persian-speaking medical office assistant who books doctor "
    "appointments. Always respond with a compact JSON object that contains at least "
    "the key 'reply'. Optional keys include 'name', 'address', 'appointment', and "
    "'notes'. Use null for name/address if you did not learn them in the latest user "
    "utterance. Gather the caller's name and address early in the conversation and "
    "confirm them. Keep the 'reply' friendly, concise, and entirely in Persian so it "
    "can be spoken aloud."
)


class VoiceDoctorBot:
    """Conversational loop that records audio, transcribes, reasons, and speaks."""

    def __init__(self, record_seconds: float = 8.0, sample_rate: int = 16000) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment or .env file.")

        self.client = OpenAI(api_key=api_key)
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.profile: Dict[str, Optional[str]] = {"name": None, "address": None}
        self.notes: list[str] = []
        self.previous_snapshot: Optional[Dict[str, Any]] = None
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session_log = self.data_dir / "session_log.txt"
        self.profile_file = self.data_dir / "last_session.json"
        self.clients_file = self.data_dir / "clients.json"
        self.session_meta_file = self.data_dir / "session_meta.json"
        self.prompt_file = self.data_dir / "custom_prompt.txt"
        self.names_file = self.data_dir / "iranian_names.txt"
        self.tts_engine = pyttsx3.init()
        self.tts_model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self.tts_voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
        self.response_model = os.getenv("OPENAI_RESPONSE_MODEL", "gpt-4o-mini")
        self.transcription_model = os.getenv(
            "OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"
        )
        self.transcription_fallback = os.getenv(
            "OPENAI_TRANSCRIBE_FALLBACK", "gpt-4o-mini-transcribe"
        )
        self._transcribe_warned = False
        self.history_limit = 16
        self.history: list[Dict[str, str]] = []
        self.push_chunk_seconds = 0.3
        self.push_silence_timeout = 0.6
        self.push_energy_threshold = 180.0
        self.silence_trim_threshold = 65.0
        self.session_name = self._generate_session_name()
        self.known_clients: set[str] = self._load_known_clients()
        self.iranian_name_list: set[str] = self._load_iranian_names()
        self.system_prompt = self._load_system_prompt()
        self._load_last_session()
        self._load_history_from_log()
        self._select_persian_voice()
        self._save_session_meta()
        self._log_session_start()

    def run(self) -> None:
        print(
            f"Doctor Voice Assistant ready. Session '{self.session_name}'. "
            "Press Enter to speak, 'q' to quit."
        )
        while True:
            user_input = input("Press Enter to speak (q to quit): ").strip().lower()
            if user_input == "q":
                print("Session ended. See data/last_session.json for captured details.")
                break

            try:
                audio_np = self._record_audio()
                transcript = self._transcribe(audio_np)
            except Exception as exc:  # broad to keep loop alive
                print(f"Recording or transcription failed: {exc}")
                continue

            if not transcript:
                print("No speech detected. Try again.")
                continue

            print(f"You said: {transcript}")
            self._log("user", transcript)

            reply, payload = self._reason(transcript)
            if not reply:
                print("The assistant could not create a response. Try again.")
                continue

            print(f"Assistant: {reply}")
            self._speak(reply)
            self._log("assistant", reply)
            self._update_profile(payload)

    def train_prompt(self) -> None:
        print(
            "Prompt training mode.\n"
            f"Speak for up to {self.record_seconds} seconds to describe how the assistant should behave."
        )
        try:
            audio_np = self._record_audio()
            transcript = self._transcribe(audio_np)
        except Exception as exc:
            print(f"Training failed: {exc}")
            return
        if not transcript:
            print("Did not capture any speech. Prompt unchanged.")
            return
        print(f"Captured prompt:\n{transcript}")
        self._save_system_prompt(transcript)
        print(f"Custom prompt saved to {self.prompt_file}.")

    def run_realtime(
        self,
        chunk_seconds: float = 0.25,
        silence_timeout: float = 0.35,
        energy_threshold: float = 200.0,
    ) -> None:
        """Continuously listen for speech and answer as soon as silence is detected."""
        print(
            "Realtime Doctor Voice Assistant listening.\n"
            f"Session '{self.session_name}' at DrX clinic.\n"
            "Speak naturally; pause for a second to let the assistant reply.\n"
            "Press Ctrl+C to end the session."
        )
        chunk_frames = int(self.sample_rate * chunk_seconds)
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(indata, frames, time_info, status) -> None:  # type: ignore[override]
            if status:
                print(f"Audio warning: {status}")
            audio_queue.put(indata.copy().flatten())

        buffer: list[np.ndarray] = []
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
                    energy = float(np.mean(np.abs(chunk)))
                    if energy >= energy_threshold:
                        buffer.append(chunk)
                        silence_since = None
                    elif buffer:
                        now = time.time()
                        silence_since = silence_since or now
                        if (now - silence_since) >= silence_timeout:
                            segment = np.concatenate(buffer)
                            buffer.clear()
                            silence_since = None
                            self._process_segment(segment)
        except KeyboardInterrupt:
            print("\nSession ended. See data/last_session.json for captured details.")

    def _process_segment(self, audio_np: np.ndarray) -> None:
        """Transcribe, reason, speak, and log a block of audio."""
        transcript = self._transcribe(audio_np)
        if not transcript:
            return
        print(f"You said: {transcript}")
        self._log("user", transcript)
        reply, payload = self._reason(transcript)
        if not reply:
            print("The assistant could not create a response. Try again.")
            return
        print(f"Assistant: {reply}")
        self._speak(reply)
        self._log("assistant", reply)
        self._update_profile(payload)

    def _record_audio(self) -> np.ndarray:
        print(
            f"Recording (max {self.record_seconds:.1f}s)... "
            "stop speaking to send sooner."
        )
        sd.stop()
        chunk_frames = max(1, int(self.sample_rate * self.push_chunk_seconds))
        max_frames = int(self.sample_rate * self.record_seconds)
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(indata, frames, time_info, status) -> None:  # type: ignore[override]
            if status:
                print(f"Audio warning: {status}")
            audio_queue.put(indata.copy().flatten())

        buffer: list[np.ndarray] = []
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
                    if energy >= self.push_energy_threshold:
                        silence_since = None
                    else:
                        silence_since = silence_since or now
                        elapsed_silence = now - silence_since
                        min_duration = 0.5
                        if (
                            elapsed_silence >= self.push_silence_timeout
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
        return self._trim_trailing_silence(audio)

    def _transcribe(self, audio: np.ndarray) -> str:
        audio_buffer = self._to_wav_bytes(audio)
        audio_buffer.name = "speech.wav"
        use_audio_endpoint = any(
            marker in self.transcription_model.lower()
            for marker in ("transcribe", "whisper")
        )
        if use_audio_endpoint:
            return self._transcribe_via_audio_endpoint(audio_buffer, self.transcription_model)

        try:
            return self._transcribe_via_responses(audio_buffer)
        except OpenAIError as exc:
            if not self._transcribe_warned:
                print(
                    f"Advanced transcription with '{self.transcription_model}' failed "
                    f"({exc}). Falling back to '{self.transcription_fallback}'."
                )
                self._transcribe_warned = True
            return self._transcribe_via_audio_endpoint(audio_buffer, self.transcription_fallback)

    def _reason(self, transcript: str) -> tuple[str, Dict[str, Optional[str]]]:
        profile_json = json.dumps(
            {"name": self.profile.get("name"), "address": self.profile.get("address")}
        )
        history_context = self._history_context()
        previous_snapshot_json = json.dumps(self.previous_snapshot or {})
        known_clients_list = sorted(self.known_clients)
        client_json = json.dumps(known_clients_list[-20:])
        possible_return = self._find_similar_client(transcript) or "none"
        prompt = (
            f"Session name: {self.session_name}\n"
            f"Known returning clients: {client_json}\n"
            f"Possible returning client mentioned: {possible_return}\n"
            "Previous session snapshot (for reference only—confirm before reuse): "
            f"{previous_snapshot_json}\n"
            "If a possible returning client is noted, do not assume; politely ask whether "
            "they are the same person and only keep the name if confirmed in the latest "
            "caller statement.\n"
            "Conversation so far:\n"
            f"{history_context}\n"
            f"Caller statement: {transcript}\n"
            f"Known data: {profile_json}\n"
            "Return only JSON as described earlier."
        )
        try:
            response = self.client.responses.create(
                model=self.response_model,
                temperature=0.4,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": self.system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    },
                ],
            )
        except OpenAIError as exc:
            print(f"OpenAI request failed: {exc}")
            return "", {}

        raw_text = self._extract_text(response)
        payload = self._parse_json(raw_text)
        reply = payload.get("reply", raw_text)
        return reply, payload

    def _speak(self, message: str) -> None:
        if not message:
            return
        try:
            audio_bytes = self._synthesize_with_openai(message)
            self._play_wav_bytes(audio_bytes)
            return
        except Exception as exc:
            print(f"TTS service failed ({exc}); falling back to local voice.")
        self.tts_engine.say(message)
        self.tts_engine.runAndWait()

    def _select_persian_voice(self) -> None:
        """Pick a Persian-capable TTS voice if the system has one."""
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

    def _synthesize_with_openai(self, message: str) -> bytes:
        response = self.client.audio.speech.create(
            model=self.tts_model,
            voice=self.tts_voice,
            input=message,
            response_format="wav",
            instructions="Speak naturally in Persian.",
        )
        return response.read()

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        if not wav_bytes:
            return
        sd.stop()  # stop any lingering playback before starting a new clip
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

    def _load_last_session(self) -> None:
        """Seed profile and notes from the previous session if available."""
        self.previous_snapshot = None
        if not self.profile_file.exists():
            return
        try:
            data = json.loads(self.profile_file.read_text())
        except Exception:
            return
        if isinstance(data, dict):
            self.previous_snapshot = data

    def _load_system_prompt(self) -> str:
        if self.prompt_file.exists():
            try:
                text = self.prompt_file.read_text(encoding="utf-8").strip()
                if text:
                    return text
            except Exception:
                pass
        return SYSTEM_PROMPT

    def _save_system_prompt(self, prompt: str) -> None:
        content = prompt.strip() or SYSTEM_PROMPT
        self.prompt_file.write_text(content, encoding="utf-8")
        self.system_prompt = content

    def _load_history_from_log(self) -> None:
        """Warm conversation memory from the existing session log."""
        entries = []
        for _, role, text in self._iter_log_entries():
            if role in ("user", "assistant"):
                entries.append({"role": role, "text": text})
        if entries:
            self.history = entries[-self.history_limit :]

    def _generate_session_name(self) -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = os.urandom(2).hex()
        return f"drx-{timestamp}-{suffix}"

    def _save_session_meta(self) -> None:
        meta = {
            "session_name": self.session_name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.session_meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _log_session_start(self) -> None:
        line = f"[{self.session_name}] session: started at {time.ctime()}\n"
        with self.session_log.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _load_known_clients(self) -> set[str]:
        if self.clients_file.exists():
            try:
                data = json.loads(self.clients_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return {str(item).strip() for item in data if str(item).strip()}
            except Exception:
                pass
        names: set[str] = set()
        for _, role, text in self._iter_log_entries():
            if role != "assistant":
                continue
            payload = self._parse_json(text)
            name = payload.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
        return names

    def _load_iranian_names(self) -> set[str]:
        if self.names_file.exists():
            try:
                return {
                    line.strip()
                    for line in self.names_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
            except Exception:
                pass
        # Minimal fallback set
        fallback = {
            "علی",
            "مریم",
            "رضا",
            "سارا",
            "محمد",
            "حمید",
            "نازنین",
            "آرزو",
            "پریسا",
            "فرهاد",
        }
        return fallback

    def _persist_known_clients(self) -> None:
        sorted_names = sorted(self.known_clients)
        self.clients_file.write_text(
            json.dumps(sorted_names, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _add_known_client(self, name: str) -> None:
        clean = name.strip()
        if not clean:
            return
        if clean not in self.known_clients:
            self.known_clients.add(clean)
            self._persist_known_clients()

    def _to_wav_bytes(self, audio: np.ndarray) -> io.BytesIO:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())
        buffer.seek(0)
        return buffer

    def _transcribe_via_audio_endpoint(self, audio_buffer: io.BytesIO, model: str) -> str:
        clone = io.BytesIO(audio_buffer.getvalue())
        clone.name = "speech.wav"
        result = self.client.audio.transcriptions.create(
            model=model,
            file=clone,
        )
        return (result.text or "").strip()

    def _trim_trailing_silence(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        threshold = self.silence_trim_threshold
        abs_audio = np.abs(audio)
        idx = len(audio) - 1
        while idx >= 0 and abs_audio[idx] <= threshold:
            idx -= 1
        if idx <= 0:
            return audio
        return audio[: idx + 1]

    def _transcribe_via_responses(self, audio_buffer: io.BytesIO) -> str:
        payload = base64.b64encode(audio_buffer.getvalue()).decode("ascii")
        instruction = (
            "Transcribe the following Persian speech. The audio is a base64-encoded WAV "
            "string. Decode it and reply with only the transcript.\n"
            f"{payload}"
        )
        response = self.client.responses.create(
            model=self.transcription_model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": instruction}],
                }
            ],
        )
        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    chunks.append(getattr(content, "text", ""))
        return "".join(chunks).strip()

    def _parse_json(self, blob: str) -> Dict[str, Optional[str]]:
        try:
            data = json.loads(blob)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return {"reply": blob}

    def _update_profile(self, payload: Dict[str, Optional[str]]) -> None:
        updated = False
        for key in ("name", "address"):
            value = payload.get(key)
            if value:
                if self.profile.get(key) != value:
                    self.profile[key] = value
                    updated = True
                if key == "name" and isinstance(value, str):
                    self._add_known_client(value)
        note = payload.get("notes") or payload.get("appointment")
        if note:
            self.notes.append(str(note))
            updated = True
        if updated:
            snapshot = {
                "session": self.session_name,
                "profile": self.profile,
                "notes": self.notes,
            }
            self.profile_file.write_text(json.dumps(snapshot, indent=2))
            self.previous_snapshot = snapshot
            print("Profile updated:", snapshot)

    def _log(self, role: str, text: str) -> None:
        line = f"[{self.session_name}] {role}: {text}\n"
        with self.session_log.open("a", encoding="utf-8") as handle:
            handle.write(line)
        self._remember(role, text)

    def _remember(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text})
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit :]

    def _history_context(self) -> str:
        if not self.history:
            return "No prior conversation."
        return "\n".join(f"{item['role']}: {item['text']}" for item in self.history)

    def _find_similar_client(self, transcript: str) -> Optional[str]:
        lower_transcript = transcript.lower()
        for name in sorted(self.known_clients):
            normalized = name.lower()
            if normalized and normalized in lower_transcript:
                return name
        return None

    def _detect_iranian_names(self, transcript: str) -> list[str]:
        matches: list[str] = []
        lower_transcript = transcript.lower()
        for name in sorted(self.iranian_name_list):
            normalized = name.lower()
            if normalized and normalized in lower_transcript:
                matches.append(name)
            if len(matches) >= 5:
                break
        return matches

    def _parse_log_line(self, line: str):
        stripped = line.rstrip("\n")
        if not stripped:
            return None
        session = None
        remainder = stripped
        if stripped.startswith("["):
            closing = stripped.find("]")
            if closing != -1:
                session = stripped[1:closing]
                remainder = stripped[closing + 1 :].lstrip()
        if ": " not in remainder:
            return None
        role, text = remainder.split(": ", 1)
        return session, role.strip(), text

    def _iter_log_entries(self):
        if not self.session_log.exists():
            return
        current_session: Optional[str] = None
        current_role: Optional[str] = None
        current_lines: list[str] = []
        with self.session_log.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                parsed = self._parse_log_line(raw_line)
                if parsed:
                    if current_role:
                        yield (
                            current_session,
                            current_role,
                            "\n".join(current_lines).strip(),
                        )
                    current_session, current_role, text = parsed
                    current_lines = [text]
                else:
                    if current_role is not None:
                        current_lines.append(raw_line.rstrip("\n"))
        if current_role:
            yield current_session, current_role, "\n".join(current_lines).strip()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Doctor voice assistant powered by OpenAI."
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Continuously listen with automatic speech detection.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.25,
        help="Realtime chunk size (seconds) for faster speech detection.",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=0.35,
        help="Silence duration (seconds) that triggers a response in realtime mode.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=200.0,
        help="Minimum average energy to treat audio as speech in realtime mode.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=8.0,
        help="Max duration for push-to-talk recordings.",
    )
    parser.add_argument(
        "--train-prompt",
        action="store_true",
        help="Capture a new system prompt from voice input and exit.",
    )
    args = parser.parse_args()

    bot = VoiceDoctorBot(record_seconds=args.record_seconds)
    if args.train_prompt:
        bot.train_prompt()
    elif args.realtime:
        bot.run_realtime(
            chunk_seconds=args.chunk_seconds,
            silence_timeout=args.silence_timeout,
            energy_threshold=args.energy_threshold,
        )
    else:
        bot.run()

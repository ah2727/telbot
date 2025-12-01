"""Voice-based doctor reservation assistant powered by OpenAI (refactored)."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import re
import numpy as np
import pyttsx3
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from data.names import IRANIAN_DEFAULT_NAMES


# ---------- Helpers & constants ----------

def _normalize_persian_name(name: str) -> str:
    """نرمال‌سازی ی و ک عربی، و فاصله‌ها."""
    if not isinstance(name, str):
        name = str(name)
    s = name.strip()
    # ي -> ی ، ك -> ک
    s = s.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    # حذف فاصله‌های اضافه
    s = re.sub(r"\s+", " ", s)
    return s


SYSTEM_PROMPT = """
شما «ManaCare Voice Concierge» هستید؛ دستیار تلفنی مهربان و حرفه‌ای سازمان Mana در کلینیک DrX که فقط به زبان فارسی صحبت می‌کند و وظیفه‌تان رزرو وقت پزشک و ثبت اطلاعات مراجعان است.

قوانین کلی:
- همیشه و بدون استثنا فقط یک شیء JSON معتبر برگردان.
- بیرون از JSON هیچ متن دیگری ننویس (بدون توضیح، اموجی، یا متن اضافی).
- همهٔ کلیدها باید با حروف کوچک انگلیسی باشند.

ساختار JSON:
- کلید اجباری: "reply"
- کلیدهای توصیه‌شده: "name", "address", "appointment", "notes"
- اگر هرکدام از این مقادیر را نمی‌دانی، مقدارش را null قرار بده.
- اگر اطلاعات را در مکالمه‌های قبلی همین جلسه یاد گرفته‌ای، می‌توانی مقدار فعلی را دوباره در JSON تکرار کنی.

تعریف هر فیلد:
- "reply": پاسخ کوتاه و محترمانهٔ تو، فقط به زبان فارسی، مناسب پخش صوتی (۱ تا ۲ جملهٔ کوتاه).
- "name": نام و نام خانوادگی تماس‌گیرنده به فارسی، یا null اگر هنوز مشخص نشده.
- "address": آدرس نسبتاً دقیق (شهر، محله و اگر ممکن بود خیابان/پلاک) به فارسی، یا null اگر هنوز مشخص نشده.
- "appointment": خلاصهٔ زمان/بازهٔ پیشنهادی و نوع ویزیت (مثلاً «سه‌شنبه عصر برای ویزیت حضوری»)، یا null اگر هنوز مشخص نشده.
- "notes": توضیحات مهم دیگر مثل دلیل مراجعه، علائم، ترجیحات (پزشک خانم/آقا، حضوری/آنلاین و…)، یا null اگر نکته‌ای ثبت نشده است.

رفتار و لحن:
- در اولین پاسخ، یک‌بار بگو: «من دستیار ManaCare هستم از کلینیک DrX.»
- در پاسخ‌های بعدی فقط بگو «من دستیار ManaCare هستم» و نام کلینیک را تکرار نکن مگر در جمع‌بندی نهایی.
- همیشه محترمانه، گرم، همدلانه و حرفه‌ای صحبت کن.
- از جملات کوتاه و واضح استفاده کن که برای شنیدن تلفنی مناسب باشند.
- در "reply" از کلمات ساده و کاملاً فارسی استفاده کن.

هدف مکالمه:
- نام کامل تماس‌گیرنده را بگیر و در "name" ذخیره کن.
- آدرس را هرچه زودتر بگیر و در "address" ذخیره کن و در صورت نیاز به‌صورت کوتاه تأیید کن.
- دلیل مراجعه (چکاپ، درد خاص، پیگیری آزمایش، مشاوره و…) را بپرس و در "notes" ثبت کن.
- بازهٔ زمانی یا روز و ساعت پیشنهادی برای نوبت را بپرس و در "appointment" ثبت کن.
- اگر کاربر ترجیحات خاصی مثل پزشک خانم/آقا یا حضوری/آنلاین دارد، آن را در "notes" یادداشت کن.

کار با نام‌ها (خیلی مهم):
- روی نام‌ها بسیار دقت کن و تا حد امکان آن‌ها را به فارسی برگردان (مثلاً "Mohammad Reza" → «محمدرضا»).
- اگر احتمال اشتباه در املای فارسی وجود دارد، در "reply" با یک سؤال کوتاه و مودب املای دقیق را تأیید کن.
- اگر نامی شبیه یکی از مراجعان قبلی یا «known clients» که در پیام سیستم آمده به نظر رسید، فقط با احترام اشاره کن که «احتمالاً با این نام قبلاً پرونده‌ای داریم» و حتماً بپرس آیا خودِ اوست یا برای شخص دیگری نوبت می‌گیرد؛ هرگز خودبه‌خود فرض نکن.

یادگیری از گذشته (session log):
- از تاریخچهٔ مکالمهٔ همین جلسه که در پیام سیستم به‌صورت متن «Conversation so far» می‌آید استفاده کن.
- اگر در تاریخچه نام یا آدرس یا ترجیح زمانی قبلاً گفته شده، بدون نیاز به پرسیدن دوباره می‌توانی آن‌ها را در JSON نگه داری، مگر این‌که کاربر خودش آن را اصلاح کند.
- اگر کاربر چیزی را تغییر داد (مثلاً زمان یا آدرس)، مقدار جدید را در JSON بنویس و مقدار قبلی را در نظر نگیر.

مدیریت مکالمه:
- "reply" همیشه باید مرحلهٔ بعد را روشن کند (مثلاً: «حالا لطفاً آدرس کامل را هم بفرمایید.» یا «بسیار خوب، برای چه مشکلی می‌خواهید مراجعه بفرمایید؟»).
- پرسش‌ها را ساده و سریالی نگه دار: ابتدا سلام و معرفی، سپس نام، بعد آدرس، بعد دلیل مراجعه، بعد زمان و نوع ویزیت.
- اگر کاربر موضوع نامرتبط مطرح کرد، محترمانه به او توضیح بده که تو برای رزرو نوبت هستی و سپس سریعاً مکالمه را به گرفتن زمان/دلیل مراجعه برگردان.

پایان مکالمه:
- وقتی نام، حداقل یک سطح از آدرس، دلیل مراجعه و ترجیح زمانی را دانستی، در "reply" خلاصه‌ای بسیار کوتاه از آنچه ثبت شده بگو.
- سپس اضافه کن که «تیم ManaCare در کلینیک DrX تأیید نهایی نوبت را برای شما ارسال می‌کند.»

نمونهٔ JSON خروجی:
{"reply":"سلام، من دستیار ManaCare هستم از کلینیک DrX. لطفاً نام کامل شما را بفرمایید.","name":null,"address":null,"appointment":null,"notes":null}

به‌یاد داشته باش: همیشه فقط یک شیء JSON برگردان، بدون متن اضافی.
""".strip()


@dataclass
class BotConfig:
    sample_rate: int = 16000
    record_seconds: float = 8.0

    push_chunk_seconds: float = 0.3
    push_silence_timeout: float = 0.6
    push_energy_threshold: float = 80.0
    silence_trim_threshold: float = 40.0

    realtime_chunk_seconds: float = 0.25
    realtime_silence_timeout: float = 0.35
    realtime_energy_threshold: float = 200.0

    vad_aggressiveness: int = 1
    use_vad_for_filtering: bool = False  # اگر خواستی فقط speech-frameها ترنسکرایب شوند، True کن.

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

    data_dir: Path = Path("data")


class VoiceDoctorBot:
    """Conversational loop that records audio, transcribes, reasons, and speaks."""

    def __init__(self, config: Optional[BotConfig] = None) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment or .env file.")

        self.config = config or BotConfig()
        self.client = OpenAI(api_key=api_key)

        self.is_speaking = False
        self.sample_rate = self.config.sample_rate
        self.record_seconds = self.config.record_seconds

        # state
        self.profile: Dict[str, Optional[str]] = {"name": None, "address": None}
        self.notes: List[str] = []
        self.previous_snapshot: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, str]] = []
        self._transcribe_warned = False

        # files
        self.data_dir = self.config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session_log = self.data_dir / "session_log.txt"
        self.profile_file = self.data_dir / "last_session.json"
        self.clients_file = self.data_dir / "clients.json"
        self.session_meta_file = self.data_dir / "session_meta.json"
        self.prompt_file = self.data_dir / "custom_prompt.txt"
        self.names_file = self.data_dir / "iranian_names.txt"

        # tts
        self.tts_engine = pyttsx3.init()
        self.tts_model = self.config.tts_model
        self.tts_voice = self.config.tts_voice

        # models
        self.response_model = self.config.response_model
        self.transcription_model = self.config.transcription_model
        self.transcription_fallback = self.config.transcription_fallback

        # misc config
        self.push_chunk_seconds = self.config.push_chunk_seconds
        self.push_silence_timeout = self.config.push_silence_timeout
        self.push_energy_threshold = self.config.push_energy_threshold
        self.silence_trim_threshold = self.config.silence_trim_threshold
        self.history_limit = self.config.history_limit

        # session
        self.session_name = self._generate_session_name()
        self.known_clients: set[str] = self._load_known_clients()
        self.iranian_name_list: set[str] = self._load_iranian_names()
        self.system_prompt = self._load_system_prompt()
        self._load_last_session()
        self._load_history_from_log()
        self._select_persian_voice()
        self._save_session_meta()
        self._log_session_start()

        # VAD
        self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)

    # ---------- Public entrypoints ----------

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
                if audio_np.size == 0:
                    print("No audio captured. Try again.")
                    continue
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
        chunk_seconds: Optional[float] = None,
        silence_timeout: Optional[float] = None,
        energy_threshold: Optional[float] = None,
    ) -> None:
        """Continuously listen for speech and answer as soon as silence is detected."""
        chunk_seconds = chunk_seconds or self.config.realtime_chunk_seconds
        silence_timeout = silence_timeout or self.config.realtime_silence_timeout
        energy_threshold = energy_threshold or self.config.realtime_energy_threshold

        print(
            "Realtime Doctor Voice Assistant listening.\n"
            f"Session '{self.session_name}' at DrX clinic.\n"
            "Speak naturally; pause for a second to let the assistant reply.\n"
            "Press Ctrl+C to end the session."
        )
        chunk_frames = int(self.sample_rate * chunk_seconds)
        audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

        def _callback(indata, frames, time_info, status):
            if status:
                print(f"Audio warning: {status}")
            if self.is_speaking:
                return  # discard mic frames while assistant voice is playing
            audio_queue.put(indata.copy().flatten())

        buffer: List[np.ndarray] = []
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

    # ---------- Audio capture & processing ----------

    def _record_audio(self) -> np.ndarray:
        print(
            f"Recording (max {self.record_seconds:.1f}s)... "
            "stop speaking to send sooner."
        )
        sd.stop()
        chunk_frames = max(1, int(self.sample_rate * self.push_chunk_seconds))
        max_frames = int(self.sample_rate * self.record_seconds)
        audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

        def _callback(indata, frames, time_info, status) -> None:  # type: ignore[override]
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

        # debug save
        debug_file = self.data_dir / "last_raw.wav"
        try:
            with wave.open(str(debug_file), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self.sample_rate)
                f.writeframes(audio.tobytes())
            print(f"[debug] saved raw audio to {debug_file}")
        except Exception as exc:
            print(f"[debug] failed to save raw audio: {exc}")

        return self._trim_trailing_silence(audio)

    def _process_segment(self, audio_np: np.ndarray) -> None:
        """Realtime: trim, optionally VAD-filter, transcribe, reason, speak."""
        if audio_np.size == 0:
            return

        audio_np = self._trim_trailing_silence(audio_np)

        # اختیاری: فقط قطعات حاوی گفتار را نگه داریم
        if self.config.use_vad_for_filtering:
            speech_only = self._keep_speech_only(audio_np)
            if speech_only.size > 0:
                audio_np = speech_only

        if audio_np.size == 0:
            print("[audio] no usable speech after filtering.")
            return

        transcript = self._transcribe(audio_np)
        if not transcript:
            print("No transcript from model.")
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

    def _keep_speech_only(self, audio: np.ndarray, frame_ms: int = 20) -> np.ndarray:
        if audio.size == 0:
            return audio

        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        sample_rate = self.sample_rate
        frame_len = int(sample_rate * frame_ms / 1000)
        raw = audio.tobytes()

        speech_bytes = bytearray()
        for offset in range(0, len(raw), frame_len * 2):
            chunk = raw[offset: offset + frame_len * 2]
            if len(chunk) < frame_len * 2:
                break
            if self.vad.is_speech(chunk, sample_rate):
                speech_bytes.extend(chunk)

        if not speech_bytes:
            return np.array([], dtype=np.int16)

        return np.frombuffer(bytes(speech_bytes), dtype=np.int16)

    # ---------- Transcription ----------

    def _to_wav_bytes(self, audio: np.ndarray) -> io.BytesIO:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())
        buffer.seek(0)
        return buffer

    def _build_name_prompt(self) -> str:
        known = sorted(self.known_clients)
        tail = known[-10:]
        base = (
            "این تماس برای رزرو نوبت است. نام و نام‌خانوادگی فارسی مراجعه‌کننده را دقیق بنویس. "
            "اگر در فایل فقط موسیقی، نویز یا صداهای مبهم شنیدی و گفتار واضح فارسی وجود نداشت، "
            "خروجی را خالی بگذار و هیچ متنی تولید نکن."
        )
        if not tail:
            return base
        joined = "، ".join(tail)
        return base + f" برخی نام‌های قبلی: {joined}."

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

    def _transcribe_via_audio_endpoint(self, audio_buffer: io.BytesIO, model: str) -> str:
        clone = io.BytesIO(audio_buffer.getvalue())
        clone.name = "speech.wav"
        result = self.client.audio.transcriptions.create(
            model=model,
            file=clone,
            language="fa",
            prompt=self._build_name_prompt(),
        )
        return (result.text or "").strip()

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
        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    chunks.append(getattr(content, "text", ""))
        return "".join(chunks).strip()

    # ---------- Reasoning / JSON orchestration ----------

    def _is_pure_test_utterance(self, transcript: str) -> bool:
        """
        Heuristic: detect when user is clearly just testing audio,
        not really booking an appointment.
        """
        txt = transcript.replace("🎤", "").strip().lower()

        test_keywords = [
            "تست صدا",
            "تست ضبط",
            "آزمایش صدا",
            "آزمایش میکروفون",
            "آزمایش میکروفن",
            "کالیبره",
            "کالیبره‌کردن",
            "برای تست",
            "فقط تست",
            "فقط برای آزمایش",
        ]

        booking_keywords = [
            "نوبت",
            "ویزیت",
            "ويزيت",
            "وقت",
            "مشاوره",
            "دکتر",
            "دكتر",
            "پزشک",
            "کلینیک",
            "كلينيك",
        ]

        if any(k in txt for k in booking_keywords):
            return False
        return any(k in txt for k in test_keywords)

    def _reason(self, transcript: str) -> Tuple[str, Dict[str, Optional[str]]]:
        # تست ساده بدون رفتن به LLM
        if self._is_pure_test_utterance(transcript):
            payload: Dict[str, Optional[str]] = {
                "intent": "test",
                "reply": (
                    "من دستیار ManaCare هستم. این بخش فقط برای تست صدا ثبت شد؛ "
                    "هر زمان آماده نوبت واقعی بودید، نام و درخواست‌تان را بفرمایید."
                ),
                "name": self.profile.get("name"),
                "address": self.profile.get("address"),
                "appointment": None,
                "notes": None,
            }
            return payload["reply"], payload

        profile_json = json.dumps(
            {"name": self.profile.get("name"), "address": self.profile.get("address")},
            ensure_ascii=False,
        )
        history_context = self._history_context()
        previous_snapshot_json = json.dumps(self.previous_snapshot or {}, ensure_ascii=False)
        known_clients_list = sorted(self.known_clients)
        client_json = json.dumps(known_clients_list[-20:], ensure_ascii=False)
        possible_return = self._find_similar_client(transcript) or "none"

        prompt = (
            f"Session name: {self.session_name}\n"
            f"Known returning clients: {client_json}\n"
            f"Possible returning client mentioned: {possible_return}\n"
            "Previous session snapshot (for reference only—confirm before reuse): "
            f"{previous_snapshot_json}\n"
            "Conversation so far:\n"
            f"{history_context}\n"
            f"Caller statement: {transcript}\n"
            f"Known data: {profile_json}\n"
            "وظیفه تو:\n"
            "- فیلد \"intent\" را یکی از این مقادیر قرار بده: "
            "\"booking\" (رزرو نوبت)، \"test\" (تست صدا/سیستم)، "
            "\"noise\" (نویز یا محتوای نامربوط)، \"other\" (سایر موارد).\n"
            "- فقط اگر intent = \"booking\" بود نام، آدرس، نوبت و notes را به‌صورت جدی به‌روز کن.\n"
            "- اگر intent برابر با \"test\" یا \"noise\" بود، name و address و appointment را تغییر نده و فقط یک reply مودب بده.\n"
            "- همیشه فقط یک JSON برگردان مثل:\n"
            "{\"intent\":\"booking\",\"reply\":\"...\",\"name\":null,\"address\":null,\"appointment\":null,\"notes\":null}\n"
        )

        try:
            response = self.client.responses.create(
                model=self.response_model,
                temperature=0.1,
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
            # fallback: جواب خیلی ساده
            fallback_reply = (
                "من دستیار ManaCare هستم. در دریافت پاسخ فنی مشکل پیش آمد؛ "
                "لطفاً یک بار دیگر به‌صورت کوتاه نام و دلیل مراجعه را بفرمایید."
            )
            payload = {
                "intent": "other",
                "reply": fallback_reply,
                "name": self.profile.get("name"),
                "address": self.profile.get("address"),
                "appointment": None,
                "notes": None,
            }
            return fallback_reply, payload

        raw_text = self._extract_text(response)
        payload = self._normalize_payload(raw_text)

        intent = payload.get("intent")

        # clamp behavior for non-booking
        if intent in ("test", "noise"):
            payload["reply"] = (
                "من دستیار ManaCare هستم. صدای شما را برای تست دریافت کردم؛ "
                "هر زمان برای نوبت واقعی آماده بودید، فقط نام و درخواست‌تان را بفرمایید."
                if intent == "test"
                else "من دستیار ManaCare هستم. در این بخش صدای مناسب برای رزرو نوبت دریافت نکردم؛ "
                     "اگر می‌خواهید وقت بگیرید، لطفاً نام و دلیل مراجعه را بفرمایید."
            )
            payload["appointment"] = None
            payload["notes"] = None

        reply = payload.get("reply") or raw_text
        return reply, payload

    def _normalize_payload(self, blob: str) -> Dict[str, Optional[str]]:
        """
        Ensure we always return a dict with standard keys:
        intent, reply, name, address, appointment, notes
        """
        data: Dict[str, Any]
        try:
            data_raw = json.loads(blob)
            if isinstance(data_raw, dict):
                data = data_raw
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}

        intent = str(data.get("intent") or "booking")
        reply = data.get("reply")
        name = data.get("name")
        address = data.get("address")
        appointment = data.get("appointment")
        notes = data.get("notes")

        # normalize nulls
        def _clean(x):
            if x is None:
                return None
            s = str(x).strip()
            return s or None

        return {
            "intent": _clean(intent),
            "reply": _clean(reply),
            "name": _clean(name),
            "address": _clean(address),
            "appointment": _clean(appointment),
            "notes": _clean(notes),
        }

    # ---------- TTS ----------

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

    def _speak(self, message: str) -> None:
        if not message:
            return
        self.is_speaking = True
        try:
            audio_bytes = self._synthesize_with_openai(message)
            self._play_wav_bytes(audio_bytes)
        finally:
            self.is_speaking = False

    # ---------- Session / persistence ----------

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
        entries: List[Dict[str, str]] = []
        for _, role, text in self._iter_log_entries():
            if role in ("user", "assistant"):
                entries.append({"role": role, "text": text})
        if entries:
            self.history = entries[-self.history_limit :]

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
        """
        Load a set of common Iranian first names.
        """
        if self.names_file.exists():
            try:
                raw = self.names_file.read_text(encoding="utf-8")
                tokens = re.split(r"[\n,;]+", raw)
                names: set[str] = set()
                for t in tokens:
                    t = t.strip()
                    if not t:
                        continue
                    if re.search(r"[A-Za-z]", t):
                        continue
                    names.add(_normalize_persian_name(t))
                if names:
                    return names
            except Exception:
                pass
        return {_normalize_persian_name(n) for n in IRANIAN_DEFAULT_NAMES}

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

        raw_name = payload.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            name = _normalize_persian_name(raw_name)
            if self.profile.get("name") != name:
                self.profile["name"] = name
                updated = True
            self._add_known_client(name)

        raw_addr = payload.get("address")
        if isinstance(raw_addr, str) and raw_addr.strip():
            addr = raw_addr.strip()
            if self.profile.get("address") != addr:
                self.profile["address"] = addr
                updated = True

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
            self.profile_file.write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.previous_snapshot = snapshot
            print("Profile updated:", snapshot)

    # ---------- History / logging ----------

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
        current_lines: List[str] = []
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


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Doctor voice assistant powered by OpenAI (refactored)."
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Continuously listen with automatic speech detection.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Realtime chunk size (seconds) for faster speech detection.",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=None,
        help="Silence duration (seconds) that triggers a response in realtime mode.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=None,
        help="Minimum average energy to treat audio as speech in realtime mode.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=None,
        help="Max duration for push-to-talk recordings.",
    )
    parser.add_argument(
        "--train-prompt",
        action="store_true",
        help="Capture a new system prompt from voice input and exit.",
    )
    args = parser.parse_args()

    cfg = BotConfig()
    if args.record_seconds is not None:
        cfg.record_seconds = args.record_seconds

    bot = VoiceDoctorBot(config=cfg)

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

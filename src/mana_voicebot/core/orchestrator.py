from __future__ import annotations

from typing import Dict, Any

from openai import OpenAI

from ..config import BotConfig, LOG_DIR, SESSIONS_DIR, CLIENTS_FILE
from .state import ConversationState, SkillResult
from .brain import MultiDomainBrain
from ..skills.base import BaseSkill
from ..skills.reservation import ReservationSkill
from ..skills.sales import SalesSkill
from ..skills.smalltalk import SmallTalkSkill
from ..skills.produce import ProduceSkill
from ..io.voice_io import VoiceIO
from ..persistence.file_store import SessionStore
from ..persistence.clients_store import ClientsStore
from ..skills.produce import ProduceSkill
from ..skills.visitor import VisitorSkill

class MultiDomainBot:
    """
    Orchestrator:
    - routes each user turn to the correct skill (reservation/sales/smalltalk)
    - remembers last profile/state via SessionStore
    - remembers known client names via ClientsStore
    - logs every turn into a text log
    """

    def __init__(self, client: OpenAI, config: BotConfig, voice_io: VoiceIO | None = None):
        self.client = client
        self.config = config

        self.state = ConversationState()
        self.brain = MultiDomainBrain(client, model=config.response_model)
        self.voice = voice_io

        self.skills: Dict[str, BaseSkill] = {
            "reservation": ReservationSkill(client),
            "sales": SalesSkill(client),
            "smalltalk": SmallTalkSkill(client),
            "produce": ProduceSkill(client),  
            "visitor": VisitorSkill(client), 
        }

        # persistence
        self.session_store = SessionStore(LOG_DIR, SESSIONS_DIR)
        self.clients_store = ClientsStore(CLIENTS_FILE)

        # seed from last session if exists
        last = self.session_store.load_last_snapshot()
        if last:
            self._seed_state_from_snapshot(last)

        # start new session (and log header)
        self.session_store.start_new_session(self.state)

    # ---------- core handling ----------

    def handle_turn(self, user_text: str) -> SkillResult:
        # remember + log user
        self.state.append_history("user", user_text)
        self._clamp_history()
        self.session_store.log_turn("user", user_text)

        # ask the brain
        brain_json, usage = self.brain.infer(user_text, self.state)
        domain = str(brain_json.get("domain") or "smalltalk")

        # 🔹 اینجا raw_json برای هر دامین ساخته می‌شود:
        if domain == "reservation":
            domain_payload = brain_json.get("reservation", {})
        elif domain == "sales":
            domain_payload = brain_json.get("sales", {})
        elif domain == "smalltalk":
            domain_payload = brain_json.get("smalltalk", {})
        elif domain == "produce": 
            domain_payload = brain_json.get("produce", {})
        elif domain == "visitor": 
            domain_payload = brain_json.get("visitor", {})
        else:
            domain_payload = {}

        # این دو خط باعث می‌شوند داخل raw_json مقدار intent و reply هم همیشه باشد
        domain_payload.setdefault("intent", brain_json.get("intent"))
        domain_payload.setdefault("reply", brain_json.get("reply"))

        # حالا skill مناسب را بردار و raw_json را بهش بده
        skill = self.skills.get(domain, self.skills["smalltalk"])
        result = skill.handle(user_text, self.state, domain_payload)  # 👈 raw_json همین domain_payload است

        # log + snapshot مثل قبل ...
        self.state.append_history("assistant", result.reply)
        self._clamp_history()
        self.session_store.log_turn("assistant", result.reply, domain=result.domain, intent=result.intent,usage=usage)

        return result


    # ---------- loops ----------

    def _start_visitor_intro(self, product_name: str = "TeleBot AI") -> None:
        """
        یک بار در شروع visitor mode صدا زده می‌شود.
        بدون کمک مغز مرکزی، مستقیم VisitorSkill را برای intent = intro فراخوانی می‌کنیم
        تا مکالمه را با یک معرفی و گرفتن اجازه شروع کند.
        """
        visitor_skill = self.skills.get("visitor")
        if not isinstance(visitor_skill, BaseSkill):
            # اگر به هر دلیل visitor ثبت نشده بود، هندل ساده:
            intro = (
                f"سلام، من دستیار فروش {product_name} هستم. "
                "اگر موافق باشید، در چند جمله توضیح می‌دهم این ربات چه کمکی می‌کند، "
                "بعد می‌توانید سوال‌های‌تان را بپرسید."
            )
            self.state.append_history("assistant", intro)
            self._clamp_history()
            self.session_store.log_turn("assistant", intro, domain="visitor", intent="intro")
            if self.voice:
                self.voice.speak(intro)
            else:
                print(f"[visitor/intro] {intro}")
            return

        # raw_json اولیه برای intro
        raw_json = {
            "intent": "intro",
            "product_name": product_name,
            "visitor_name": self.state.profile.get("name"),
        }

        result = visitor_skill.handle(
            turn_text="",
            state=self.state,
            raw_json=raw_json,
        )

        # history + log
        self.state.append_history("assistant", result.reply)
        self._clamp_history()
        self.session_store.log_turn(
            "assistant",
            result.reply,
            domain=result.domain,
            intent=result.intent,
            usage=None,  # این intro بدون call به مغز است
        )

        # خروجی به کاربر
        if self.voice:
            self.voice.speak(result.reply)
        else:
            print(f"[visitor/intro] {result.reply}")


    def loop_text_only(self) -> None:
        print("Multi-domain bot (text mode). Type 'q' to quit.")
        while True:
            txt = input("You: ").strip()
            if txt.lower() == "q":
                break
            if not txt:
                continue
            result = self.handle_turn(txt)
            print(f"[{result.domain}/{result.intent}] Bot:", result.reply)

    def loop_voice(self) -> None:
        if self.voice is None:
            raise RuntimeError("VoiceIO is not configured.")
        print("Multi-domain bot (voice mode). Ctrl+C to exit.")

        while True:
            try:
                text = self.voice.record()
            except KeyboardInterrupt:
                break

            if not text:
                continue

            result = self.handle_turn(text)
            self.voice.speak(result.reply)
    def loop_visitor_text(self, product_name: str = "TeleBot AI") -> None:
        """
        حالت visitor برای متن: ابتدا خودش مکالمه را شروع می‌کند،
        بعد هر ورودی را مثل همیشه از مغز و skillها عبور می‌دهد.
        """
        print(f"Visitor mode (text) for {product_name}. Type 'q' to quit.")
        # شروع مکالمه
        self._start_visitor_intro(product_name)

        while True:
            user_text = input("You: ").strip()
            if user_text.lower() == "q":
                break
            if not user_text:
                continue
            result = self.handle_turn(user_text)
            print(f"[{result.domain}/{result.intent}] Bot:", result.reply)

    def loop_visitor_voice(self, product_name: str = "TeleBot AI") -> None:
        """
        حالت visitor برای voice: اول خودش معرفی می‌کند و اجازه می‌گیرد،
        سپس هر بار از کاربر صدا می‌گیرد و جواب می‌دهد.
        """
        if self.voice is None:
            raise RuntimeError("VoiceIO is not configured for visitor voice mode.")

        print(f"Visitor mode (voice) for {product_name}. Ctrl+C to exit.")
        # شروع مکالمه
        self._start_visitor_intro(product_name)

        while True:
            try:
                text = self.voice.record()
            except KeyboardInterrupt:
                break

            if not text:
                continue

            result = self.handle_turn(text)
            self.voice.speak(result.reply)

    # ---------- helpers ----------

    def _clamp_history(self) -> None:
        """Keep only the last N turns to limit context size."""
        limit = self.config.history_limit
        if len(self.state.history) > limit:
            self.state.history = self.state.history[-limit:]

    def _seed_state_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Use last_session.json to prefill profile, notes, and domain states.
        """
        profile = snapshot.get("profile")
        if isinstance(profile, dict):
            # only keys we understand
            name = profile.get("name")
            address = profile.get("address")
            if isinstance(name, str):
                self.state.profile["name"] = name
            if isinstance(address, str):
                self.state.profile["address"] = address

        notes = snapshot.get("notes")
        if isinstance(notes, list):
            self.state.notes = [str(n) for n in notes]

        visitor = snapshot.get("visitor")
        if isinstance(visitor, dict):
            self.state.visitor_state = visitor
      
        reservation = snapshot.get("reservation")
        if isinstance(reservation, dict):
            self.state.reservation_state = reservation

        sales = snapshot.get("sales")
        if isinstance(sales, dict):
            self.state.sales_state = sales

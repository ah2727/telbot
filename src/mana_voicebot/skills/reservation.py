from __future__ import annotations

from typing import Dict, Any

from openai import OpenAI

from ..core.state import ConversationState, SkillResult
from ..data.names import normalize_persian_name


class ReservationSkill(BaseSkill := object):
    name = "reservation"

    def __init__(self, client: OpenAI):
        self.client = client

    def handle(
        self,
        turn_text: str,
        state: ConversationState,
        raw_json: Dict[str, Any],
    ) -> SkillResult:
        intent = str(raw_json.get("intent") or "booking")
        reply = str(raw_json.get("reply") or "").strip()

        name = raw_json.get("name")
        address = raw_json.get("address")
        appointment = raw_json.get("appointment")
        notes = raw_json.get("notes")

        updated = False

        if isinstance(name, str) and name.strip():
            norm = normalize_persian_name(name)
            if state.profile.get("name") != norm:
                state.profile["name"] = norm
                updated = True

        if isinstance(address, str) and address.strip():
            addr = address.strip()
            if state.profile.get("address") != addr:
                state.profile["address"] = addr
                updated = True

        if isinstance(notes, str) and notes.strip():
            state.notes.append(notes.strip())
            updated = True

        if isinstance(appointment, str) and appointment.strip():
            state.reservation_state["appointment"] = appointment.strip()
            updated = True

        # Fallback reply if model returned nothing
        if not reply:
            if state.profile.get("name") is None:
                reply = "من دستیار ManaCare هستم از کلینیک DrX. لطفاً نام کامل شما را بفرمایید."
            elif state.profile.get("address") is None:
                reply = "من دستیار ManaCare هستم. لطفاً آدرس شهر و محله خود را هم بفرمایید."
            else:
                reply = "برای چه زمانی مایل هستید نوبت بگیرید؟ روز و بازه زمانی را بفرمایید."

        payload = {
            "intent": intent,
            "name": state.profile.get("name"),
            "address": state.profile.get("address"),
            "appointment": state.reservation_state.get("appointment"),
            "notes": notes,
        }

        return SkillResult(
            reply=reply,
            domain=self.name,
            intent=intent,
            payload=payload,
        )

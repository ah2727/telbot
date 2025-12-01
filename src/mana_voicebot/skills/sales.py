from __future__ import annotations

from typing import Dict, Any

from openai import OpenAI

from ..core.state import ConversationState, SkillResult


class SalesSkill(BaseSkill := object):
    name = "sales"

    def __init__(self, client: OpenAI):
        self.client = client

    def handle(
        self,
        turn_text: str,
        state: ConversationState,
        raw_json: Dict[str, Any],
    ) -> SkillResult:
        intent = str(raw_json.get("intent") or "product_question")
        reply = str(raw_json.get("reply") or "").strip()

        product = raw_json.get("product")
        quantity = raw_json.get("quantity")
        notes = raw_json.get("notes")

        if isinstance(product, str) and product.strip():
            state.sales_state["last_product"] = product.strip()
        if quantity is not None:
            state.sales_state["last_quantity"] = quantity
        if isinstance(notes, str) and notes.strip():
            state.sales_state.setdefault("notes", []).append(notes.strip())

        if not reply:
            reply = "در خدمت‌تان هستم. درباره کدام محصول یا خدمت می‌خواهید بیشتر بدانید؟"

        payload = {
            "intent": intent,
            "product": product,
            "quantity": quantity,
            "notes": notes,
        }

        return SkillResult(
            reply=reply,
            domain=self.name,
            intent=intent,
            payload=payload,
        )

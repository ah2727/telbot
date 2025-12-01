from __future__ import annotations

from typing import Dict, Any

from openai import OpenAI

from ..core.state import ConversationState, SkillResult


class SmallTalkSkill(BaseSkill := object):
    name = "smalltalk"

    def __init__(self, client: OpenAI):
        self.client = client

    def handle(
        self,
        turn_text: str,
        state: ConversationState,
        raw_json: Dict[str, Any],
    ) -> SkillResult:
        intent = str(raw_json.get("intent") or "smalltalk")
        reply = str(raw_json.get("reply") or "").strip()

        if not reply:
            reply = "سلام، من دستیار ManaCare هستم. چطور می‌توانم کمکتان کنم؟"

        return SkillResult(
            reply=reply,
            domain=self.name,
            intent=intent,
            payload={},
        )

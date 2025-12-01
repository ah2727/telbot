from __future__ import annotations

from typing import Dict, Any

from openai import OpenAI

from ..core.state import ConversationState, SkillResult


class BaseSkill:
    name: str = "base"

    def __init__(self, client: OpenAI):
        self.client = client

    def handle(
        self,
        turn_text: str,
        state: ConversationState,
        raw_json: Dict[str, Any],
    ) -> SkillResult:
        raise NotImplementedError

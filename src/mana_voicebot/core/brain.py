from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI, OpenAIError

from ..config import PROMPTS_DIR
from .state import ConversationState


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    # fallback minimal
    return "You are a JSON-only brain. Respond with a single valid JSON object."


MAIN_PROMPT_FILE = "main_system_prompt.txt"


class MultiDomainBrain:
    """LLM that produces multi-domain JSON for reservation/sales/smalltalk."""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.system_prompt = _load_prompt(MAIN_PROMPT_FILE)

    def infer(self, user_text: str, state: ConversationState) -> Dict[str, Any]:
        history_text = self._history_to_text(state.history)
        user_payload = (
            f"Conversation so far:\n{history_text}\n\n"
            f"User said:\n{user_text}\n\n"
            "Remember to respond ONLY with one JSON as described."
        )

        try:
            resp = self.client.responses.create(
                model=self.model,
                temperature=0.2,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": self.system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_payload}],
                    },
                ],
            )
        except OpenAIError as exc:
            print(f"[brain] OpenAI error: {exc}")
            return self._fallback_json()

        raw = self._extract_text(resp)
        return self._parse_json(raw)

    def _extract_text(self, response) -> str:
        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    chunks.append(getattr(content, "text", ""))
        return "".join(chunks).strip()

    def _parse_json(self, blob: str) -> Dict[str, Any]:
        try:
            data = json.loads(blob)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return self._fallback_json(blob)

    def _fallback_json(self, reply: str | None = None) -> Dict[str, Any]:
        if reply is None:
            reply = (
                "در دریافت پاسخ فنی مشکلی پیش آمد. لطفاً جمله‌تان را کوتاه‌تر تکرار کنید."
            )
        return {
            "domain": "smalltalk",
            "intent": "smalltalk",
            "reply": reply,
            "reservation": {
                "name": None,
                "address": None,
                "appointment": None,
                "notes": None,
            },
            "sales": {"product": None, "quantity": None, "notes": None},
            "smalltalk": {"intent": "smalltalk"},
        }

    def _history_to_text(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "No prior conversation."
        return "\n".join(f"{h['role']}: {h['text']}" for h in history)

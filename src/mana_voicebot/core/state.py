from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class SkillResult:
    reply: str
    domain: str
    intent: str
    payload: Dict[str, Any] = field(default_factory=dict)


from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ConversationState:
    """Global state shared between skills."""
    profile: Dict[str, Optional[str]] = field(
        default_factory=lambda: {"name": None, "address": None}
    )
    notes: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)

    reservation_state: Dict[str, Any] = field(default_factory=dict)
    sales_state: Dict[str, Any] = field(default_factory=dict)
    produce_state: Dict[str, Any] = field(default_factory=dict) 
    visitor_state: Dict[str, Any] = field(default_factory=dict)  

    def append_history(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text})

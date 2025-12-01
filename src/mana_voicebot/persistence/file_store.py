from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

from ..core.state import ConversationState


class SessionStore:
    def __init__(self, log_dir: Path, sessions_dir: Path):
        self.log_dir = log_dir
        self.sessions_dir = sessions_dir

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # fixed paths
        self.log_file: Path = self.log_dir / "session_log.txt"
        self.snapshot_file: Path = self.sessions_dir / "last_session.json"
        self.meta_file: Path = self.sessions_dir / "session_meta.json"

        self.session_name: Optional[str] = None

    # ---------- snapshot / remembering ----------

    def load_last_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Load last_session.json if available.
        Used to seed profile/notes/state on startup.
        """
        if not self.snapshot_file.exists():
            return None
        try:
            data = json.loads(self.snapshot_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def start_new_session(self, state: ConversationState) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.session_name = f"session-{timestamp}"

        line = f"[{self.session_name}] session: started at {time.ctime()}\n"
        with self.log_file.open("a", encoding="utf-8") as h:
            h.write(line)

        meta = {
            "session_name": self.session_name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.meta_file.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # also save initial snapshot
        self.save_snapshot(state)
        
    def save_snapshot(self, state: ConversationState) -> None:
        snapshot = {
            "session": self.session_name,
            "profile": state.profile,
            "notes": state.notes,
            "reservation": state.reservation_state,
            "sales": state.sales_state,
            "produce": state.produce_state,
            "visitor": state.visitor_state,  # 👈 این
        }
        self.snapshot_file.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    # ---------- logging text turns ----------
    def log_turn(self, role: str, text: str, domain: str | None = None, intent: str | None = None) -> None:
        """
        Append one line to session_log.txt:
        [session-name] role: text
        یا اگر domain/intent داده شد:
        [session-name] role(domain:intent): text
        """
        if not self.session_name:
            return

        if domain and intent:
            tag = f"{role}({domain}:{intent})"
        elif domain:
            tag = f"{role}({domain})"
        else:
            tag = role

        line = f"[{self.session_name}] {tag}: {text}\n"
        with self.log_file.open("a", encoding="utf-8") as h:
            h.write(line)

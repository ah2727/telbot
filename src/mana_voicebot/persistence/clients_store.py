from __future__ import annotations

import json
from pathlib import Path
from typing import Set


class ClientsStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._names: Set[str] = set()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not self.path.exists():
            self._loaded = True
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._names = {str(x).strip() for x in data if str(x).strip()}
        except Exception:
            pass
        self._loaded = True

    def all(self) -> Set[str]:
        self.load()
        return set(self._names)

    def add(self, name: str) -> None:
        self.load()
        clean = name.strip()
        if not clean or clean in self._names:
            return
        self._names.add(clean)
        self._persist()

    def _persist(self) -> None:
        self.path.write_text(
            json.dumps(sorted(self._names), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, cast


class Memory:
    def __init__(self, path: str = "data_memory.json", keep: int = 8) -> None:
        self.path = Path(path)
        self.keep = keep
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load(self) -> List[Dict[str, str]]:
        data: Any = json.loads(self.path.read_text(encoding="utf-8"))
        return cast(List[Dict[str, str]], data)

    def save(self, user: str, agent: str) -> None:
        data = self.load()
        data.append({"user": user, "agent": agent})
        self.path.write_text(json.dumps(data[-self.keep :], indent=2), encoding="utf-8")

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

    def append_trace(
        self, user: str, steps: List[Dict[str, Any]], final_answer: str
    ) -> None:
        tp = self.path.with_suffix(".traces.json")
        if not tp.exists():
            tp.write_text("[]", encoding="utf-8")
        traces: Any = json.loads(tp.read_text(encoding="utf-8"))
        traces = cast(List[Dict[str, Any]], traces)
        traces.append({"user": user, "steps": steps, "final": final_answer})
        tp.write_text(
            json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8"
        )

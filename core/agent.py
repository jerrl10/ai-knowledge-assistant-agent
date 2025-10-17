# core/agent.py — optimized for reliability & tool triggering
from __future__ import annotations
import json
import re
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, cast

from core.llm import call_llm
from core.memory import Memory
from core.tools import TOOL_REGISTRY
from core.logger import logger

MAX_STEPS = 4

# ---- updated DECISION_PROMPT ----
DECISION_PROMPT = """You are a reasoning agent that can THINK and ACT.

Available tools you can use: {tool_names}

When the user's question needs outside information or up-to-date facts,
ALWAYS use the 'search' tool first.

Respond ONLY in one of these JSON formats:

1) To act:
{{ "type": "action", "tool": "<tool_name>", "input": "<input text>" }}

2) To answer:
{{ "type": "final", "answer": "<your answer>" }}

User question: {user_question}
Context: {context}
"""

REFLECT_PROMPT = """Tool used: "{tool}" with input "{tool_input}"
Observation:
{observation}

Next step — return JSON only:
- "action" or "final" as before.
Available tools: {tool_names}
User question: {user_question}
"""


@dataclass
class Step:
    kind: str  # "action" | "final"
    thought: str
    tool: Optional[str] = None
    tool_input: Optional[str] = None
    observation: Optional[str] = None
    answer: Optional[str] = None


class Agent:
    def __init__(self) -> None:
        self.memory = Memory()

    # ---------- main loop ----------
    def run(self, prompt: str) -> str:
        history = self.memory.load()
        tool_names = ", ".join(TOOL_REGISTRY.keys())
        steps: List[Step] = []
        seen: set[Tuple[str, str]] = set()

        # initial decision
        decision = self._decide(
            DECISION_PROMPT.format(
                tool_names=tool_names,
                user_question=prompt,
                context=self._format_context(history),
            ),
            history,
        )
        logger.info(
            "Tool executed",
            extra={"tool": tool_names, "decision": decision},
        )
        steps.append(
            Step(
                kind=decision["type"],
                thought="initial",
                **{k: v for k, v in decision.items() if k != "type"},
            )
        )

        for _ in range(MAX_STEPS):
            if decision["type"] == "final":
                final = decision.get("answer", "I have no final answer.").strip()
                self.memory.save(prompt, final)
                return self._render_steps(steps, final)

            tool_name = (decision.get("tool") or "").strip()
            tool_input = (decision.get("input") or "").strip()
            tool = TOOL_REGISTRY.get(tool_name)
            if not tool:
                msg = f"Unknown tool '{tool_name}'."
                steps[-1].observation = msg
                self.memory.save(prompt, msg)
                return self._render_steps(steps, msg)

            if (tool_name, tool_input) in seen:
                msg = f"Stopped to avoid loop: repeated {tool_name}({tool_input})"
                steps[-1].observation = msg
                self.memory.save(prompt, msg)
                return self._render_steps(steps, msg)
            seen.add((tool_name, tool_input))

            # execute tool safely
            try:
                observation = str(tool.call(tool_input)) or "(no data returned)"
            except Exception as e:
                observation = f"[Tool error] {type(e).__name__}: {e}\n{traceback.format_exc(limit=1)}"

            # debug output for visibility
            print(
                f"[Agent] Tool executed: {tool_name}({tool_input}) → {observation[:100]}"
            )

            steps[-1].observation = observation

            # next reasoning step
            decision = self._decide(
                REFLECT_PROMPT.format(
                    tool=tool_name,
                    tool_input=tool_input,
                    observation=observation,
                    tool_names=tool_names,
                    user_question=prompt,
                ),
                history,
            )
            steps.append(
                Step(
                    kind=decision["type"],
                    thought="follow-up",
                    **{k: v for k, v in decision.items() if k != "type"},
                )
            )

        final = "⚠️ Step limit reached — stopped to avoid infinite loop."
        self.memory.save(prompt, final)
        return self._render_steps(steps, final)

    # ---------- helpers ----------

    def _decide(self, prompt: str, history: List[Dict[str, str]]) -> Dict[str, str]:
        raw = call_llm(prompt, history)
        parsed = self._parse_json_safe(raw)
        if not parsed:
            return {"type": "final", "answer": f"[Parse error] {raw}"}
        kind = parsed.get("type", "").lower()
        if kind == "action":
            return {
                "type": "action",
                "tool": parsed.get("tool", ""),
                "input": parsed.get("input", ""),
            }
        return {"type": "final", "answer": parsed.get("answer", "")}

    @staticmethod
    def _parse_json_safe(text: str) -> Optional[Dict[str, Any]]:
        # First, try to parse the whole string directly
        try:
            return cast(Dict[str, Any], json.loads(text))
        except Exception:
            pass

        # If that fails, try to extract the first JSON object with regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return cast(Dict[str, Any], json.loads(match.group(0)))
        except Exception:
            return None

    @staticmethod
    def _format_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return "(no prior turns)"
        return "\n---\n".join(
            f"User: {h['user']}\nAgent: {h['agent']}" for h in history[-6:]
        )

    @staticmethod
    def _render_steps(steps: List[Step], final_answer: str) -> str:
        out = []
        for i, s in enumerate(steps, 1):
            out += [
                f"### 🧠 Step {i}",
                f"Kind: {s.kind}",
                f"Tool: {s.tool or '-'}",
                f"Input: {s.tool_input or '-'}",
                f"Observation: {s.observation or '-'}",
                f"Answer: {s.answer or '-'}",
                "",
            ]
        out += ["✅ **Final Answer:**", final_answer]
        return "\n".join(out)

from __future__ import annotations
import json
import re
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterator, Protocol, cast

from core.llm import call_llm
from core.memory import Memory
from core.tools import TOOL_REGISTRY as _TOOLS  # import untyped registry


# ----- Type protocol for tools so mypy knows .call exists -----
class Tool(Protocol):
    def call(self, query: str) -> str: ...


# Cast the imported registry to a typed mapping
TOOL_REGISTRY: Dict[str, Tool] = cast(Dict[str, Tool], _TOOLS)

MAX_STEPS = 6

DECISION_PROMPT = """You are a reasoning agent that solves problems using THINK → ACT → OBSERVE.

Return STRICT JSON ONLY. Do not include text outside JSON.

If you need external info or current facts, prefer using 'search' first.
If you have raw long text, consider 'summarize' before the final answer.

Formats:

1) To ACT:
{
  "thought": "<why the action is needed>",
  "action": {
    "tool": "<one of: {tool_names}>",
    "input": "<input for the tool>"
  }
}

2) To ANSWER:
{
  "thought": "<why this is the final answer>",
  "final_answer": "<concise final answer>"
}

User question: {user_question}
Context: {context}
"""

REFLECT_PROMPT = """You used "{tool}" with input "{tool_input}".
OBSERVATION:
{observation}

Now decide the next step. 
If more info or compression is needed, ACT again.
Otherwise provide final_answer.

Return STRICT JSON ONLY, using the same schema.
User question: {user_question}
"""

CONFIDENCE_PROMPT = """You are a verifier. Score how relevant the OBSERVATION is to the USER QUESTION.

Return STRICT JSON, e.g.:
{ "relevance": 0.0 to 1.0, "reason": "<why>" }

USER QUESTION: {user_question}
OBSERVATION: {observation}
"""


@dataclass
class Step:
    idx: int
    thought: str
    tool: Optional[str] = None
    tool_input: Optional[str] = None
    observation: Optional[str] = None
    confidence: Optional[float] = None
    confidence_reason: Optional[str] = None
    final_answer: Optional[str] = None


class Agent:
    def __init__(self) -> None:
        self.memory = Memory()

    # ---------- public streaming ----------
    def run_stream(
        self,
        prompt: str,
        on_step: Optional[Callable[[Step], None]] = None,
    ) -> Iterator[Step]:
        steps, final_answer = self._reason(prompt, on_step=on_step)
        self.memory.save(prompt, final_answer)
        self.memory.append_trace(prompt, [asdict(s) for s in steps], final_answer)
        for s in steps:
            yield s

    # ---------- non-stream ----------
    def run(self, prompt: str) -> str:
        steps, final_answer = self._reason(prompt, on_step=None)
        self.memory.save(prompt, final_answer)
        self.memory.append_trace(prompt, [asdict(s) for s in steps], final_answer)
        return self._render(steps, final_answer)

    # ---------- core reasoning ----------
    def _reason(
        self,
        prompt: str,
        on_step: Optional[Callable[[Step], None]],
    ) -> tuple[List[Step], str]:
        history = self.memory.load()
        context = self._format_context(history)
        tool_names = ", ".join(TOOL_REGISTRY.keys())

        steps: List[Step] = []
        seen: set[Tuple[str, str]] = set()

        # Initial decision
        decision_raw = call_llm(
            DECISION_PROMPT.format(
                tool_names=tool_names,
                user_question=prompt,
                context=context,
            ),
            history,
        )
        decision = self._parse_json_safe(decision_raw)
        steps.append(Step(idx=1, thought=decision.get("thought", "(no thought)")))
        if on_step:
            on_step(steps[-1])

        # Reasoning loop
        for i in range(1, MAX_STEPS + 1):
            # Check final answer
            if "final_answer" in decision:
                steps[-1].final_answer = str(decision.get("final_answer", "")).strip()
                return steps, str(steps[-1].final_answer or "")

            # Parse action
            action = cast(Dict[str, Any], decision.get("action", {}))
            tool_name = str(action.get("tool", "")).strip()
            tool_input = str(action.get("input", "")).strip()

            # Tool validation
            if not tool_name or tool_name not in TOOL_REGISTRY:
                msg = f"⚠️ Unknown or missing tool '{tool_name}'."
                steps[-1].observation = msg
                steps[-1].final_answer = msg
                return steps, msg

            if (tool_name, tool_input) in seen:
                msg = f"⚠️ Repeated action {tool_name}({tool_input}) — stopping."
                steps[-1].observation = msg
                steps[-1].final_answer = msg
                return steps, msg
            seen.add((tool_name, tool_input))

            # Execute tool
            try:
                obs = str(TOOL_REGISTRY[tool_name].call(tool_input)) or "(no data)"
            except Exception as e:
                obs = f"[Tool Error] {type(e).__name__}: {e}\n{traceback.format_exc(limit=1)}"

            steps[-1].tool = tool_name
            steps[-1].tool_input = tool_input
            steps[-1].observation = obs

            # Confidence scoring
            score = self._score_observation(prompt, obs)
            steps[-1].confidence = cast(Optional[float], score.get("relevance"))
            steps[-1].confidence_reason = cast(Optional[str], score.get("reason"))
            if on_step:
                on_step(steps[-1])

            # Reflect → next reasoning step
            reflect_raw = call_llm(
                REFLECT_PROMPT.format(
                    tool=tool_name,
                    tool_input=tool_input,
                    observation=obs,
                    user_question=prompt,
                ),
                history,
            )
            decision = self._parse_json_safe(reflect_raw)
            steps.append(
                Step(idx=i + 1, thought=decision.get("thought", "(no thought)"))
            )
            if on_step:
                on_step(steps[-1])

        # Max steps reached
        steps[-1].final_answer = "⚠️ Max reasoning steps reached — stopping."
        return steps, str(steps[-1].final_answer or "")

    # ---------- helpers ----------
    def _score_observation(self, question: str, observation: str) -> Dict[str, Any]:
        """Ask the LLM to score how relevant a tool result is."""
        raw = call_llm(
            CONFIDENCE_PROMPT.format(user_question=question, observation=observation),
            history=None,
        )
        parsed = self._parse_json_safe(raw)
        try:
            r = float(parsed.get("relevance", 0.0))
            parsed["relevance"] = max(0.0, min(1.0, r))
        except Exception:
            parsed["relevance"] = None
        return parsed

    @staticmethod
    def _parse_json_safe(text: Any) -> Dict[str, Any]:
        """Try to extract and sanitize JSON even from messy LLM text."""
        if not isinstance(text, str):
            text = str(text)

        # Attempt direct JSON parse
        try:
            return cast(Dict[str, Any], json.loads(text))
        except Exception:
            pass

        # Try to extract first JSON-like block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            raw_json = m.group(0)
            # Try to fix single quotes or minor issues
            fixed = raw_json.replace("'", '"')
            try:
                return cast(Dict[str, Any], json.loads(fixed))
            except Exception:
                pass

        # Fallback: fabricate a minimal JSON structure so we don't crash
        return {
            "thought": "(model did not return valid JSON)",
            "final_answer": f"[Invalid JSON] {text.strip()[:400]}",
        }

    @staticmethod
    def _format_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return "(no prior turns)"
        return "\n---\n".join(
            f"User: {h['user']}\nAgent: {h['agent']}" for h in history[-6:]
        )

    @staticmethod
    def _render(steps: List[Step], final_answer: str) -> str:
        lines: List[str] = []
        for s in steps:
            lines.append(f"### Step {s.idx}")
            lines.append(f"Thought: {s.thought}")
            if s.tool:
                lines.append(f"Action: {s.tool}('{s.tool_input}')")
            if s.observation:
                lines.append(f"Observation: {s.observation}")
            if s.confidence is not None:
                reason = s.confidence_reason or ""
                lines.append(f"Relevance: {s.confidence:.2f} — {reason}")
            if s.final_answer:
                lines.append(f"✅ Final Answer: {s.final_answer}")
            lines.append("")
        lines.append(f"✅ Final Answer: {final_answer}")
        return "\n".join(lines)

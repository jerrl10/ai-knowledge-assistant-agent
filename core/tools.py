from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

from core.llm import call_llm


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    call: Callable[[str], str]


@retry(
    stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.3, min=0.3, max=3)
)
def _ddg_text(query: str, max_results: int = 5) -> List[dict]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def search(query: str) -> str:
    """
    DuckDuckGo text search (top results summarized).
    Returns a concise newline-separated list of title + url.
    """

    q = (query or "").strip()
    if not q:
        return "No query provided"
    try:
        results = _ddg_text(q, max_results=5)
        if not results:
            return "No results found"
        lines = []
        for r in results[:5]:
            title = r.get("title", "") or "Untitled"
            href = r.get("href", "") or r.get("url", "") or ""
            lines.append(f"- {title}: {href}".strip())
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching DuckDuckGo: {str(e)}"


def echo(text: str) -> str:
    return f"Echo: {text}"


@dataclass
class SummarizeTool:
    name: str = "summarize"
    description: str = "Summarize a given text into 3-5 bullet points."

    def call(self, text: str) -> str:
        prompt = (
            "Summarize the following text in 3-5 concise bullet points.\n\n"
            f"TEXT:\n{text}\n\n"
            "Return plain text bullets, no JSON."
        )
        return call_llm(prompt, history=None)


TOOL_REGISTRY = {
    "search": Tool(
        name="search", description="Search the web for information", call=search
    ),
    "echo": Tool(name="echo", description="Echo back the input text", call=echo),
    "summarize": SummarizeTool(),
}

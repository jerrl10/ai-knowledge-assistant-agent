from __future__ import annotations
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lmstudio"
MODEL = "tinyllama-1.1b-chat-v1.0"


def call_llm(prompt: str, history: List[Dict[str, str]] | None = None) -> str:
    """Call LM Studio or other local OpenAI-compatible endpoint."""

    LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
    LMSTUDIO_API_KEY = "lmstudio"
    MODEL = "tinyllama-1.1b-chat-v1.0"

    url = LMSTUDIO_BASE_URL.rstrip("/") + "/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
    }

    messages = [{"role": "system", "content": "You are a concise helpful AI."}]
    if history:
        for h in history:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["agent"]})
    messages.append({"role": "user", "content": prompt})

    prompt_text = flatten_messages(messages)

    payload = {
        "model": MODEL,
        "prompt": prompt_text,
        "temperature": 0.6,
        "max_tokens": 800,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text: str = str(data["choices"][0].get("text", "")).strip()
        return text
    except Exception as e:
        return f"[LLM error] {type(e).__name__}: {e}"


def flatten_messages(messages: list[dict[str, str]]) -> str:
    """Convert chat-style messages into a single text prompt string."""
    lines = []
    for m in messages:
        role = m.get("role", "").capitalize()
        content = m.get("content", "")
        if role and content:
            lines.append(f"{role}: {content}")
    # Encourage the model to output valid JSON
    lines.append(
        "\nAssistant: Please respond with ONLY a valid JSON object. "
        'Example: {"thought": "<reasoning>", "final_answer": "<answer>"}'
    )
    return "\n".join(lines)

from __future__ import annotations
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lmstudio"
MODEL = "tinyllama-1.1b-chat-v1.0"


def call_llm(prompt: str, history: List[Dict[str, str]] | None = None) -> str:
    """
    Sends a chat-completion request directly to an LM Studio local server.

    Args:
        prompt: user query
        history: optional list of {"user": str, "agent": str} pairs

    Returns:
        model reply text
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a concise, helpful AI agent."}
    ]

    # if history:
    #     for turn in history:
    #         messages.append({"role": "user", "content": turn["user"]})
    #         messages.append({"role": "assistant", "content": turn["agent"]})

    messages.append({"role": "user", "content": prompt})

    url = LMSTUDIO_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 800,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()  # type: ignore[no-any-return]
    except requests.exceptions.ConnectionError:
        return "[LLM error] Cannot reach LM Studio — is the local server running at 127.0.0.1:1234?"
    except Exception as e:
        return f"[LLM error] {type(e).__name__}: {e}"

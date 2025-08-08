from __future__ import annotations
import json
from typing import Any, Dict, Iterable, List, Optional
import requests


class OllamaClient:
    """Minimal Ollama REST client for chat and streaming.

    Methods
    - chat: Single-response convenience wrapper
    - chat_stream: Yields incremental content as the model responds
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def chat(
        self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0
    ) -> str:
        """Call `/api/chat` and return the final message content as a string.

        This is a non-streaming convenience wrapper around the Ollama API.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def chat_stream(
        self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0
    ):
        """Stream tokens from Ollama `/api/chat` with `stream=true`.

        Yields token text chunks.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": True,
        }
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                delta = obj.get("message", {}).get("content", "")
                if delta:
                    yield delta


def extract_action_or_final(text: str) -> Dict[str, Any]:
    """Parse model output to one JSON object with either action or final.

    Contract: model outputs a fenced JSON block or a raw JSON object like:
    {
      "thought": "...",
      "action": {"tool": "name", "args": {...}}
    }
    or
    {
      "thought": "...",
      "final": "answer"
    }
    """
    # Naive parsing: find the first {...} JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"final": text.strip()}
    snippet = text[start : end + 1]
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj
        return {"final": text.strip()}
    except Exception:
        return {"final": text.strip()}

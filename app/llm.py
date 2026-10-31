"""Llama generation through a local Ollama server (https://ollama.com).

    ollama pull llama3.2
    ollama serve

Returns None if Ollama isn't reachable, so the API still returns the
retrieved sources instead of failing.
"""
from typing import Optional

import httpx

from app import settings


def generate(prompt: str) -> Optional[str]:
    try:
        r = httpx.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={"model": settings.LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=settings.LLM_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception:
        return None

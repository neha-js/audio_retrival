"""Shared text utilities: tokenization and simple token counting.

A real deployment would use a proper tokenizer (tiktoken, HF
tokenizers) for the token-count estimate used in the token-savings
report; a whitespace-based approximation is used here to keep the
project dependency-free.
"""
from __future__ import annotations

import re
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric tokenization used by BM25 and TF-IDF."""
    return _TOKEN_RE.findall(text.lower())


def estimate_token_count(text: str) -> int:
    """Rough LLM-token estimate (~0.75 words/token is closer for GPT-ish
    tokenizers, but for reporting *relative* savings a word count is
    a fine, dependency-free proxy)."""
    return max(1, len(text.split()))

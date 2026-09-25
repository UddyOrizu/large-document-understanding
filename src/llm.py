"""Thin wrappers around the two API clients used in this demo.

Claude does all the reasoning: answering questions, writing section
summaries for the vectorless tree, and navigating that tree. OpenAI
does exactly one job: turning text into embedding vectors for the
classic vector-RAG baseline. Swap either provider out freely — nothing
else in this project cares which one does which job.
"""
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic
from openai import OpenAI

# If these defaults don't resolve for your account, check
# docs.claude.com/en/docs/about-claude/models for current model names.
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
FAST_MODEL = os.environ.get("CLAUDE_FAST_MODEL", "claude-haiku-4-5-20251001")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

_claude = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
_openai = OpenAI()     # reads OPENAI_API_KEY from the environment


@dataclass
class ClaudeCall:
    text: str
    latency_s: float
    input_tokens: int
    output_tokens: int


def ask_claude(system: str, user: str, max_tokens: int = 500,
                model: Optional[str] = None) -> ClaudeCall:
    start = time.time()
    resp = _claude.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return ClaudeCall(
        text=text,
        latency_s=time.time() - start,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )


def embed(texts: List[str]) -> List[List[float]]:
    resp = _openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

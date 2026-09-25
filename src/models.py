"""Shared result type every retrieval strategy returns, so compare.py
can display all three side by side."""
from dataclasses import dataclass
from typing import Optional

from .evidence import Evidence


@dataclass
class AnswerResult:
    method: str
    final_answer: str
    evidence: Optional[Evidence]
    source_section: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    retrieval_note: str

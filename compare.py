#!/usr/bin/env python3
"""Live-demo CLI: ask one question, see all three approaches answer it
side by side — final answer, exact quoted evidence (verified against
the source), where each one looked, and time/tokens spent.

Run build_index.py once first, then:

    python compare.py "What is the international relocation allowance?"

Good questions to try live (see README for why each one matters):
    python compare.py "What was FY2025 total revenue?"
    python compare.py "What is the settling-in allowance for a long-term international assignment?"
    python compare.py "When must the fourth quarterly privileged access review be completed?"
"""
import json
import os
import sys

from src import full_context, vector_rag, vectorless_rag
from src.parser import section_from_dict

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOC_PATH = os.path.join(DATA_DIR, "sample_report.md")
TREE_PATH = os.path.join(DATA_DIR, "index", "tree.json")

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


def load():
    if not os.path.exists(TREE_PATH):
        sys.exit("No index found. Run `python build_index.py` first.")
    with open(DOC_PATH) as f:
        full_text = f.read()
    with open(TREE_PATH) as f:
        root = section_from_dict(json.load(f))
    return full_text, root


def evidence_str(r) -> str:
    if not r.evidence:
        return "[no quote returned]"
    status = "VERIFIED" if r.evidence.verified else f"UNVERIFIED (match {r.evidence.match_score})"
    return f"{status}\n\u201c{r.evidence.quote}\u201d"


def run(question: str):
    full_text, root = load()

    results = [
        full_context.answer(question, full_text),
        vector_rag.answer(question, full_text),
        vectorless_rag.answer(question, root, full_text),
    ]

    if console:
        table = Table(title=f'Q: "{question}"', show_lines=True)
        for col in ("Method", "Answer", "Evidence", "Retrieval path", "Cost"):
            table.add_column(col)
        for r in results:
            table.add_row(
                r.method,
                r.final_answer,
                evidence_str(r),
                r.retrieval_note,
                f"{r.latency_s:.1f}s, {r.input_tokens + r.output_tokens} tok",
            )
        console.print(table)
    else:
        for r in results:
            print(f"\n=== {r.method} ===")
            print("Answer:   ", r.final_answer)
            print("Evidence: ", evidence_str(r).replace("\n", " "))
            print("Retrieval:", r.retrieval_note)
            print("Cost:     ", f"{r.latency_s:.1f}s, {r.input_tokens + r.output_tokens} tokens")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(" ".join(sys.argv[1:]))

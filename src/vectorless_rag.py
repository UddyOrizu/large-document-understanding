"""Vectorless RAG: no embeddings at all. Claude reads a tree of section
titles + one-line summaries and reasons about which branch to open —
the way a person uses a table of contents — then the exact content of
the chosen node(s) is fetched and handed back for the real answer.
Modelled on tools like PageIndex.
"""
from . import llm
from .evidence import locate_quote, split_quote_and_answer
from .models import AnswerResult
from .parser import Section


def _tree_outline(root: Section) -> str:
    lines = []

    def walk(node: Section):
        if node.level > 0:
            indent = "  " * (node.level - 1)
            lines.append(f"{indent}- [{node.id}] {node.title}: {node.summary or ''}")
        for c in node.children:
            walk(c)

    walk(root)
    return "\n".join(lines)


def _find_by_id(root: Section, node_id: str):
    for n in root.flatten():
        if n.id == node_id:
            return n
    return None


NAV_SYSTEM = (
    "You are navigating a document's table of contents to find where a "
    "question is answered. You see only titles and one-line summaries, "
    "never the full text. Pick the 1-2 section IDs most likely to contain "
    "the answer. Reply with ONLY the IDs, comma-separated — nothing else."
)

ANSWER_SYSTEM = (
    "Answer the question using ONLY the section text provided. First quote "
    "the exact sentence that supports your answer, verbatim, inside "
    "<quote></quote> tags. Then, on a new line, write 'Answer: ' followed "
    "by a one-sentence answer."
)


def answer(question: str, root: Section, full_doc_text: str) -> AnswerResult:
    outline = _tree_outline(root)

    # Phase 1: reasoning-based navigation over titles + summaries only —
    # cheap and fast, so route it to the fast model.
    nav_call = llm.ask_claude(
        NAV_SYSTEM, f"Table of contents:\n{outline}\n\nQuestion: {question}",
        max_tokens=64, model=llm.FAST_MODEL,
    )
    ids = [x.strip() for x in nav_call.text.split(",") if x.strip()]
    sections = [s for s in (_find_by_id(root, i) for i in ids) if s]
    if not sections:
        sections = [root]

    # Phase 2: fetch the exact content of the chosen node(s) and answer.
    content = "\n\n".join(f"[{s.title}]\n{s.full_text()}" for s in sections)
    ans_call = llm.ask_claude(ANSWER_SYSTEM, f"Section content:\n{content}\n\nQuestion: {question}")

    quote, final = split_quote_and_answer(ans_call.text)
    ev = locate_quote(full_doc_text, quote) if quote else None

    return AnswerResult(
        method="Vectorless RAG (tree navigation)",
        final_answer=final or ans_call.text,
        evidence=ev,
        source_section=", ".join(s.title for s in sections),
        latency_s=nav_call.latency_s + ans_call.latency_s,
        input_tokens=nav_call.input_tokens + ans_call.input_tokens,
        output_tokens=nav_call.output_tokens + ans_call.output_tokens,
        retrieval_note=(f"navigated to: {', '.join(s.id for s in sections)} "
                         f"(explicit path, no similarity score)"),
    )

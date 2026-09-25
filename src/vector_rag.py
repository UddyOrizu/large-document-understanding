"""Classic embed-and-retrieve RAG — the industry-standard baseline."""
import json
import os
from typing import List

import numpy as np

from . import llm
from .evidence import locate_quote, split_quote_and_answer
from .models import AnswerResult
from .parser import Section

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "embeddings.npy")


def _chunk_leaves(root: Section, max_words: int = 250) -> List[dict]:
    """Split each leaf section further if it's long, so no chunk is
    bigger than ~max_words — a simple stand-in for real chunking logic.
    This is also where 'chunking breaks context' becomes visible: a
    chunk boundary can land in the middle of a paragraph."""
    chunks = []
    for leaf in root.leaves():
        words = leaf.content.split()
        if not words:
            continue
        for i in range(0, len(words), max_words):
            piece = " ".join(words[i:i + max_words])
            chunks.append({"section_id": leaf.id, "section_title": leaf.title, "text": piece})
    return chunks


def build_index(root: Section) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    chunks = _chunk_leaves(root)
    vectors = llm.embed([c["text"] for c in chunks])
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)
    np.save(EMBEDDINGS_PATH, np.array(vectors, dtype=np.float32))
    print(f"vector_rag: embedded {len(chunks)} chunks -> {EMBEDDINGS_PATH}")


def _load_index():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    vectors = np.load(EMBEDDINGS_PATH)
    return chunks, vectors


def _top_k(question: str, k: int = 3):
    chunks, vectors = _load_index()
    q_vec = np.array(llm.embed([question])[0], dtype=np.float32)
    sims = vectors @ q_vec / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8)
    top_idx = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in top_idx]


SYSTEM_PROMPT = (
    "Answer the question using ONLY the excerpts provided. First quote "
    "the exact sentence that supports your answer, verbatim, inside "
    "<quote></quote> tags. Then, on a new line, write 'Answer: ' followed "
    "by a one-sentence answer. If the excerpts don't contain the answer, "
    "say so plainly instead of guessing."
)


def answer(question: str, full_doc_text: str) -> AnswerResult:
    top = _top_k(question, k=3)
    context = "\n\n".join(f"[Section: {c['section_title']}]\n{c['text']}" for c, _ in top)
    call = llm.ask_claude(SYSTEM_PROMPT, f"Excerpts:\n{context}\n\nQuestion: {question}")

    quote, final = split_quote_and_answer(call.text)
    ev = locate_quote(full_doc_text, quote) if quote else None

    return AnswerResult(
        method="Vector RAG",
        final_answer=final or call.text,
        evidence=ev,
        source_section=top[0][0]["section_title"] if top else "?",
        latency_s=call.latency_s,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        retrieval_note=(f"top chunk: \"{top[0][0]['section_title']}\" "
                         f"(cosine sim {top[0][1]:.2f})") if top else "no match",
    )

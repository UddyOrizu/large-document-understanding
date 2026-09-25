#!/usr/bin/env python3
"""Run once before the demo.

Parses the sample document, asks Claude for a one-line summary of every
section (that's the entire "index" vectorless_rag.py needs — no
embeddings), and builds chunk embeddings for vector_rag.py. Both get
cached under data/index/, so the live demo itself only makes the small
number of calls needed to actually answer a question — nothing gets
rebuilt mid-talk.

    python build_index.py
"""
import json
import os

from src import llm
from src.parser import Section, parse_markdown, section_to_dict
from src.vector_rag import build_index as build_vector_index

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOC_PATH = os.path.join(DATA_DIR, "sample_report.md")
TREE_PATH = os.path.join(DATA_DIR, "index", "tree.json")

SUMMARY_SYSTEM = (
    "Summarize this section of a document in ONE short sentence, under "
    "15 words, capturing what a reader would come here to find. "
    "Reply with the sentence only, nothing else."
)


def summarize_tree(root: Section) -> None:
    nodes = [n for n in root.flatten() if n.level > 0 and n.content.strip()]
    print(f"Summarizing {len(nodes)} sections with {llm.FAST_MODEL}...")
    for node in nodes:
        call = llm.ask_claude(SUMMARY_SYSTEM, node.content[:1500],
                               max_tokens=40, model=llm.FAST_MODEL)
        node.summary = call.text.strip()
        print(f"  [{node.id}] {node.title} -> {node.summary}")


def main():
    with open(DOC_PATH) as f:
        text = f.read()
    root = parse_markdown(text)

    summarize_tree(root)
    os.makedirs(os.path.dirname(TREE_PATH), exist_ok=True)
    with open(TREE_PATH, "w") as f:
        json.dump(section_to_dict(root), f, indent=2)
    print(f"\nSaved tree index -> {TREE_PATH}")

    print(f"\nBuilding vector RAG chunk embeddings with {llm.EMBED_MODEL}...")
    build_vector_index(root)

    print('\nDone. Try: python compare.py "your question"')


if __name__ == "__main__":
    main()

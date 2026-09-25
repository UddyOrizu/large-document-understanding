# Document Intelligence Demo

A small, self-contained project to demo the ideas from the "AI for large document
analysis" talk: full-context vs. vector RAG vs. vectorless RAG, and why every answer
should come with an exact, verified quote rather than a paraphrase.

It runs the same question through three strategies and shows them side by side:

1. **Full-context baseline** — paste the whole document into the prompt and ask.
   This is the "why not just use a bigger context window" strawman.
2. **Vector RAG** — chunk the document, embed the chunks (OpenAI), retrieve the
   top-k by cosine similarity, answer from those.
3. **Vectorless RAG** — no embeddings at all. AI reads a tree of section
   titles + one-line summaries (built once, up front) and reasons about which
   branch to open, the way a person uses a table of contents — modelled on
   tools like [PageIndex](https://pageindex.ai). Retrieval path is explicit
   and inspectable, rather than a similarity score.

Every answer, from all three strategies, is required to return an exact quoted
sentence. `src/evidence.py` then checks that the quote actually appears in the
source document (near-verbatim match, tolerant of minor paraphrase) before it's
shown as "verified" — a minimal version of the citation-resolvability checks
described in the talk.

## The document

`data/sample_report.md` is a synthetic ~12,000-word company annual report and
policy manual, structured with real headings (`#`/`##`/`###`) so the tree in
strategy 3 has genuine hierarchy to navigate. It deliberately buries three
specific, checkable facts at different depths and surrounds them with
similar-sounding distractor numbers — the same setup "lost in the middle" and
Chroma's "context rot" research use to measure degradation:

| Fact | Where | Difficulty |
|---|---|---|
| FY2025 total revenue: **£412.6 million** | Executive Summary (near the start) | Easy — all three strategies should nail this |
| Long-term international settling-in allowance: **£4,750** | Buried in §5.3, surrounded by 6+ other allowance figures (per diems, housing supplements, tuition, domestic relocation) | Hard — the interesting one |
| Fourth quarterly privileged-access review deadline: **15 December** | Buried in §6.3, among several other review cadences (90-day password rotation, twice-yearly recertification) | Hard, and near the end |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

Build the index once (this is the only step that touches the whole document up
front — a handful of cheap Haiku calls for section summaries, plus one OpenAI
embeddings call):

```bash
python build_index.py
```

## Running the demo

```bash
python compare.py "What was FY2025 total revenue?"
python compare.py "What is the settling-in allowance for a long-term international assignment?"
python compare.py "When must the fourth quarterly privileged access review be completed?"
```

Each prints a table: **Answer**, **Evidence** (the exact quote, verified or
flagged), **Retrieval path** (which chunk/section was used and why), and
**Cost** (latency + tokens).

## Suggested live-demo flow (maps to the talk)

1. Run the **revenue** question first. All three get it right — it's near the
   start of the document, so it's not a fair test yet.
2. Run the **settling-in allowance** question. This is where it gets
   interesting: the full-context baseline is the one most likely to blend it
   with a nearby distractor figure (the £1,200 domestic allowance, the £1,650
   housing supplement, or the £85 per diem) — a live example of lost-in-the-
   middle. Vector RAG and vectorless RAG should both isolate the right chunk.
3. Run the **privileged access review** question and point at the
   **Retrieval path** column: vector RAG shows a similarity score, vectorless
   RAG shows an actual navigated path (`doc.1.6.3`, i.e. "§6.3 Privileged
   Access Review") — that's the traceability argument from the talk.
4. Point at the **Evidence** column throughout: this is the "don't just
   paraphrase, quote and verify" pattern — the same shape as the multi-agent
   extract-then-verify pattern used in production hallucination-detection
   pipelines.
5. Point at **Cost**: vectorless RAG makes two Claude calls (a cheap Haiku
   navigation call, then a Sonnet answer call) versus one call each for the
   others — a real tradeoff worth naming, not just a strictly-better story.

## Extending this

- **Swap in a real 300-page PDF**: convert it to Markdown with headings first
  (e.g. `pypdf`/`unstructured`, or Claude itself) so `src/parser.py` has
  structure to work with — vectorless RAG depends on the document actually
  having a hierarchy to navigate.
- **Swap the embedding provider**: everything embedding-related lives in
  `src/llm.py::embed()` — point it at Voyage, Cohere, or a local model without
  touching `vector_rag.py`.
- **Add GraphRAG/RAPTOR-style summarization**: `build_index.py` already
  generates one summary per section; recursively summarizing summaries up the
  tree is a small extension if you want to demo "what's the overall argument"
  style questions, not just fact lookup.
- **Add a UI**: `compare.py`'s `run()` function returns nothing today but
  easily could — wrapping it in a small Streamlit app is a natural next step
  if you want something other than a terminal on the projector.

## Project layout

```
data/sample_report.md      the demo document
data/index/                 cached tree summaries + chunk embeddings (generated)
src/parser.py                markdown -> hierarchical Section tree
src/llm.py                    Claude + OpenAI API wrappers
src/evidence.py               quote location, verification, quote/answer parsing
src/models.py                  shared AnswerResult type
src/full_context.py             strategy 1
src/vector_rag.py                strategy 2
src/vectorless_rag.py             strategy 3
build_index.py               one-time index build
compare.py                    live-demo entry point
```

"""Naive baseline: paste the whole document into the prompt and ask.
This is the 'why not just use a bigger context window' strawman — run
it next to the other two strategies to show context rot live, especially
on the question whose answer is buried in the middle of the document.
"""
from . import llm
from .evidence import locate_quote, split_quote_and_answer
from .models import AnswerResult

SYSTEM = (
    "Answer the question using ONLY the document provided. First quote "
    "the exact sentence that supports your answer, verbatim, inside "
    "<quote></quote> tags. Then, on a new line, write 'Answer: ' followed "
    "by a one-sentence answer."
)


def answer(question: str, full_doc_text: str) -> AnswerResult:
    call = llm.ask_claude(SYSTEM, f"Document:\n{full_doc_text}\n\nQuestion: {question}",
                           max_tokens=500)
    quote, final = split_quote_and_answer(call.text)
    ev = locate_quote(full_doc_text, quote) if quote else None
    return AnswerResult(
        method="Full-context baseline",
        final_answer=final or call.text,
        evidence=ev,
        source_section="(whole document, no retrieval)",
        latency_s=call.latency_s,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        retrieval_note="none — entire document sent as context",
    )

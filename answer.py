"""Threshold gate + Groq answer generation on top of the existing retriever.

Pipeline for one question:
  1. Run the existing retriever (embedding model, chunking, and ranking are
     all unchanged - see ``retriever.py`` / ``loader.py``).
  2. Look at the top-1 cosine similarity.
       - top-1 <  SIMILARITY_THRESHOLD -> do NOT call Groq; return a clear
         "not covered by the NimbusNote docs" message. The top retrieved
         result is still exposed for debugging.
       - top-1 >= SIMILARITY_THRESHOLD -> send the question + top-3 chunks to
         Groq (Llama-3.3-70B) and ask it to answer ONLY from those chunks.
  3. The citation (source filename, section, passage text, score) always comes
     straight from the retriever - never from anything the LLM says.

No UI, no vector DB, no LangChain, no reranking, no conversation memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from groq import Groq

from retriever import RetrievalResult, Retriever

# Heuristic relevance floor chosen after evaluating 14 covered + 14 not-covered
# questions with the real all-MiniLM-L6-v2 model. It is a "is this query even in
# scope" gate, NOT a factual-confidence score.
SIMILARITY_THRESHOLD = 0.50

# Groq model id for Llama-3.3-70B. Overridable via the GROQ_MODEL env var
# (some API keys don't have access to every model on Groq).
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

NOT_COVERED_MESSAGE = (
    "That question does not appear to be covered by the provided NimbusNote "
    "documents, so no answer was generated."
)

_SYSTEM_PROMPT = (
    "You are a support assistant for NimbusNote. Answer the user's question "
    "using ONLY the context passages provided in the user message. Do not use "
    "any outside or prior knowledge. If the passages do not contain enough "
    "information to answer the question, reply that the NimbusNote "
    "documentation does not cover it - do not guess. Do not mention the "
    "passages, numbering, or context in your answer."
)


@dataclass
class AnswerResult:
    """Everything a caller needs to display an answer with its evidence."""

    question: str
    top_score: float
    supported: bool          # True if top-1 >= SIMILARITY_THRESHOLD
    groq_called: bool
    answer: str              # generated answer, or the not-covered message
    top_result: RetrievalResult          # top-1 hit, always present (debugging)
    retrieved: list[RetrievalResult] = field(default_factory=list)  # top-k


def format_evidence(result: RetrievalResult) -> str:
    """Displayed citation evidence: the retrieved section heading followed by
    its body text.

    Both parts come straight from the retriever's ``RetrievalResult`` - this
    only controls how the evidence is shown, not what was retrieved, ranked,
    or scored.
    """
    return f"{result.section}\n\n{result.text}"


def _format_context(results: list[RetrievalResult]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] ({r.source} - {r.section})\n{r.text}")
    return "\n\n".join(blocks)


def _generate_with_groq(question: str, results: list[RetrievalResult]) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    user_message = (
        f"Question: {question}\n\n"
        f"Context passages:\n{_format_context(results)}\n\n"
        "Answer:"
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


def answer_query(query: str, retriever: Retriever, top_k: int = 3) -> AnswerResult:
    """Retrieve, apply the threshold gate, and (only if it passes) call Groq."""
    results = retriever.retrieve(query, top_k=top_k)
    top = results[0]

    # Below the relevance floor: skip Groq entirely, but still hand back the
    # best retrieved chunk so it can be inspected during evaluation.
    if top.score < SIMILARITY_THRESHOLD:
        return AnswerResult(
            question=query,
            top_score=top.score,
            supported=False,
            groq_called=False,
            answer=NOT_COVERED_MESSAGE,
            top_result=top,
            retrieved=results,
        )

    generated = _generate_with_groq(query, results)
    return AnswerResult(
        question=query,
        top_score=top.score,
        supported=True,
        groq_called=True,
        answer=generated,
        top_result=top,
        retrieved=results,
    )

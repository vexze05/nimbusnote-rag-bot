"""Demo for the threshold gate + Groq generation stage.

Runs one SUPPORTED and one UNSUPPORTED question through ``answer_query`` and
prints, for each: the question, top retrieval score, whether Groq was called,
the answer/result, and the retriever-produced citation (source, section,
passage) when available.

Usage:
    python test_answer.py
"""

from __future__ import annotations

import sys

from answer import SIMILARITY_THRESHOLD, answer_query, format_evidence
from retriever import build_retriever

# The bundled docs use em-dashes; keep them readable on a cp1252 Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

CASES = [
    ("SUPPORTED", "How much does the Pro plan cost?"),
    ("UNSUPPORTED", "What programming language is NimbusNote written in?"),
]


def main() -> None:
    retriever = build_retriever()
    print(f"SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}")
    print("=" * 72)

    for expected, question in CASES:
        result = answer_query(question, retriever, top_k=3)
        print(f"Expected case : {expected}")
        print(f"Question      : {result.question}")
        print(f"Top score     : {result.top_score:.4f}")
        print(f"Groq called   : {'yes' if result.groq_called else 'no (skipped)'}")
        print(f"Supported     : {result.supported}")
        print(f"Answer/result : {result.answer}")
        print("Citation (from the retrieval system, not the LLM):")
        print(f"  source      : {result.top_result.source}")
        print(f"  section     : {result.top_result.section}")
        print(f"  score       : {result.top_result.score:.4f}")
        print("  displayed evidence/passage:")
        for line in format_evidence(result.top_result).splitlines():
            print(f"    {line}")
        print("=" * 72)


if __name__ == "__main__":
    main()

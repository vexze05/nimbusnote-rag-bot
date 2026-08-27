"""Minimal interactive entry point for the NimbusNote RAG Q&A bot.

Builds the existing retriever from ./data/, then loops: read a question,
call the existing answer_query(), and print the answer plus the
retriever-produced evidence. Type "exit" or "quit" to stop.

Usage:
    python main.py
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

EXIT_WORDS = {"exit", "quit"}


def show(result) -> None:
    print()
    print(f"Answer                     : {result.answer}")
    print(f"Retrieval passed threshold : {result.supported}")
    print(f"Groq called                : {'Yes' if result.groq_called else 'No'}")
    print(f"Source                     : {result.top_result.source}")
    print(f"Section                    : {result.top_result.section}")
    print(f"Similarity                 : {result.top_score:.4f}")
    print("Evidence / retrieved passage:")
    for line in format_evidence(result.top_result).splitlines():
        print(f"  {line}")
    print()


def main() -> None:
    retriever = build_retriever()
    print("NimbusNote RAG Q&A bot")
    print(f"(similarity threshold = {SIMILARITY_THRESHOLD}; type 'exit' or 'quit' to leave)")

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("Bye.")
            break

        result = answer_query(question, retriever, top_k=3)
        show(result)


if __name__ == "__main__":
    main()

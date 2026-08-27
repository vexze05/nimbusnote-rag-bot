"""Demo/eval for the embeddings + retrieval stage.

Runs three sample questions through the retriever and prints the ranked
top-3 chunks with their cosine-similarity scores, source, section, and a
short passage preview.

Usage:
    python eval_retrieval.py
"""

from __future__ import annotations

import sys

from retriever import MODEL_NAME, build_retriever

# The bundled docs use em-dashes; keep them readable on a cp1252 Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

QUESTIONS = [
    "How often does NimbusNote sync while the app is in the foreground?",
    "How much does the Pro plan cost?",
    "What happens if two devices edit the same note?",
]

PREVIEW_CHARS = 200


def main() -> None:
    retriever = build_retriever()
    print(f"Model: {MODEL_NAME}")
    print(f"Chunks indexed: {len(retriever.chunks)}")
    print("=" * 72)

    for q_num, question in enumerate(QUESTIONS, start=1):
        results = retriever.retrieve(question, top_k=3)
        print(f"Q{q_num}: {question}")
        print("-" * 72)
        for rank, result in enumerate(results, start=1):
            preview = " ".join(result.text.split())
            if len(preview) > PREVIEW_CHARS:
                preview = preview[:PREVIEW_CHARS] + " ..."
            print(f"  #{rank}  score={result.score:.4f}")
            print(f"      source : {result.source}")
            print(f"      section: {result.section}")
            print(f"      preview: {preview}")
        print("=" * 72)


if __name__ == "__main__":
    main()

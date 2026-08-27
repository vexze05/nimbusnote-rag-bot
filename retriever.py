"""Embeddings + retrieval for the NimbusNote RAG bot.

This stage adds a small semantic search layer on top of the existing local
loader and section-based chunker (see ``loader.py``). It embeds every chunk
once with ``all-MiniLM-L6-v2`` and answers queries by cosine similarity.

Not in this stage: similarity thresholds, unsupported-question fallback,
LLM generation, citations, a vector database, or any UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from loader import Chunk, build_chunks

# Small, fast, widely-used sentence embedding model (384-dim vectors).
MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class RetrievalResult:
    """One retrieved chunk plus its similarity score to the query."""

    source: str   # source filename, carried over from the chunk
    section: str  # section name, carried over from the chunk
    text: str     # chunk text, carried over from the chunk
    score: float  # cosine similarity to the query, in [-1, 1]


class Retriever:
    """Holds the chunk embeddings and does similarity search over them."""

    def __init__(self, chunks: list[Chunk], model_name: str = MODEL_NAME) -> None:
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        # Embed every chunk once. We embed the section title together with its
        # body, because the title often carries key facts (e.g. the heading
        # "Pro plan - $6/month per workspace" holds the price, not the body).
        # normalize_embeddings=True makes each vector unit length, so a plain
        # dot product between two vectors equals their cosine similarity.
        self.embeddings = self.model.encode(
            [f"{chunk.section}\n{chunk.text}" for chunk in chunks],
            normalize_embeddings=True,
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return the ``top_k`` chunks most similar to ``query``, best first."""
        # Encode the query with the SAME model and the SAME normalization as the
        # chunks, so query and chunk vectors live in the same unit-length space.
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        # Both sides are unit vectors, so this dot product IS the cosine
        # similarity: cos(a, b) = (a . b) / (|a| |b|) = a . b when |a| = |b| = 1.
        similarities = self.embeddings @ query_embedding

        # Indices of the highest scores first.
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            RetrievalResult(
                source=self.chunks[i].source,
                section=self.chunks[i].section,
                text=self.chunks[i].text,
                score=float(similarities[i]),
            )
            for i in ranked_indices
        ]


def build_retriever() -> Retriever:
    """Convenience: build chunks from ./data/ and wrap them in a Retriever."""
    return Retriever(build_chunks())

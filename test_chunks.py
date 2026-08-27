"""Run/verify path for the local loader + section chunker.

Usage:
    python test_chunks.py

It loads the three local NimbusNote documents, builds the chunks, and prints
the total count plus each chunk's source, section, and a text preview so the
chunks can be eyeballed for correctness. Also runs under pytest.
"""

from __future__ import annotations

import sys

from loader import DOC_FILENAMES, build_chunks

# The bundled docs use em-dashes; keep them readable on a cp1252 Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

PREVIEW_CHARS = 200
EXPECTED_CHUNK_COUNT = 15


def main() -> None:
    chunks = build_chunks()

    print(f"Loaded documents: {', '.join(DOC_FILENAMES)}")
    print(f"Total chunks: {len(chunks)}")
    print("=" * 70)

    for i, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.text.split())
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS] + " ..."
        print(f"[{i:02d}] source : {chunk.source}")
        print(f"     section: {chunk.section}")
        print(f"     text   : {preview}")
        print(f"     (chars : {len(chunk.text)})")
        print("-" * 70)

    print(f"Total chunks: {len(chunks)} (expected {EXPECTED_CHUNK_COUNT})")


def test_expected_chunk_count() -> None:
    assert len(build_chunks()) == EXPECTED_CHUNK_COUNT


def test_every_chunk_has_all_fields() -> None:
    for chunk in build_chunks():
        assert chunk.source in DOC_FILENAMES
        assert chunk.section.strip()
        assert chunk.text.strip()


if __name__ == "__main__":
    main()

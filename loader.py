"""Local document loading and section-based chunking for the NimbusNote RAG bot.

This stage only covers loading the bundled Markdown files from ``./data/`` and
splitting each file into section-level chunks by its ``##`` headings. No
embeddings, retrieval, or LLM calls live here yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# The three bundled starter documents, in a stable order.
DOC_FILENAMES = [
    "01-getting-started.md",
    "02-pricing-and-plans.md",
    "03-troubleshooting.md",
]

# Matches a level-2 heading ("## Section name") but not "#" or "###".
_H2_RE = re.compile(r"^##(?!#)\s+(.*\S)\s*$")


@dataclass
class Chunk:
    """One section-level chunk of a source document."""

    source: str   # source filename, e.g. "01-getting-started.md"
    section: str  # section name taken from the "##" heading
    text: str     # the chunk body text under that heading


def chunk_markdown_by_sections(markdown: str, source: str) -> list[Chunk]:
    """Split one Markdown string into chunks, one per ``##`` heading.

    Any content before the first ``##`` heading (the ``#`` title and intro
    paragraph) is intentionally dropped, since it is not a section.
    """
    chunks: list[Chunk] = []
    current_section: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_section is not None:
            text = "\n".join(current_lines).strip()
            chunks.append(Chunk(source=source, section=current_section, text=text))

    for line in markdown.splitlines():
        heading = _H2_RE.match(line)
        if heading:
            flush()
            current_section = heading.group(1).strip()
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    flush()
    return chunks


def load_documents(data_dir: Path | str = DATA_DIR) -> dict[str, str]:
    """Load the raw text of every bundled document from ``data_dir``."""
    data_dir = Path(data_dir)
    docs: dict[str, str] = {}
    for filename in DOC_FILENAMES:
        path = data_dir / filename
        docs[filename] = path.read_text(encoding="utf-8")
    return docs


def build_chunks(data_dir: Path | str = DATA_DIR) -> list[Chunk]:
    """Load all local documents and return their combined section-level chunks."""
    chunks: list[Chunk] = []
    for filename, markdown in load_documents(data_dir).items():
        chunks.extend(chunk_markdown_by_sections(markdown, filename))
    return chunks

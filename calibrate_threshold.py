"""Threshold calibration data collection (expanded).

This does NOT set, recommend, or auto-select a similarity threshold or any
decision rule. It only runs the existing all-MiniLM-L6-v2 retrieval over two
labeled question groups (COVERED / NOT COVERED) and reports, per question:
top-1 score, top-2 score, top-1 source, top-1 section, and the top1-top2
margin. It then reports per-group score/margin ranges and any overlap.

Usage:
    python calibrate_threshold.py
"""

from __future__ import annotations

import sys

from retriever import MODEL_NAME, build_retriever

# The bundled docs use em-dashes; keep them readable on a cp1252 Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

COVERED = [
    # --- first calibration batch ---
    "How much does the Pro plan cost?",
    "How often does NimbusNote sync while the app is in the foreground?",
    "Can I use NimbusNote notes without an internet connection?",
    "What happens when two devices edit the same note?",
    "How long do password reset emails remain valid?",
    "How many notebooks can a Free workspace have?",
    "How large can an image attachment be?",
    "When does a downgrade from Pro or Team take effect?",
    # --- expansion batch ---
    "What is the maximum image size I can attach to a note?",
    "How long is a password reset link valid?",
    "Can I keep using notes while offline?",
    "What happens to my Pro features after downgrading?",
    "How many collaborators can the Pro plan have?",
    "What happens if an image upload is larger than 20MB?",
]

NOT_COVERED = [
    # --- first calibration batch ---
    "Does NimbusNote have an Android app?",
    "Does NimbusNote support two-factor authentication?",
    "What programming language is NimbusNote written in?",
    "Does NimbusNote have a public API?",
    "Does NimbusNote support dark mode?",
    "Where are NimbusNote's servers located?",
    "Does NimbusNote offer a desktop application?",
    "Can NimbusNote be integrated with Slack?",
    # --- expansion batch ---
    "Does NimbusNote have an iPhone app?",
    "Does NimbusNote support biometric login?",
    "Can NimbusNote integrate with Google Drive?",
    "Does NimbusNote encrypt notes?",
    "Does NimbusNote support dark mode?",
    "Can I export notes to PDF?",
]


def rng(values: list[float]) -> tuple[float, float, float]:
    return min(values), max(values), sum(values) / len(values)


def run_group(retriever, label: str, questions: list[str]) -> list[dict]:
    print(f"=== {label} ===")
    rows: list[dict] = []
    for i, question in enumerate(questions, start=1):
        results = retriever.retrieve(question, top_k=3)
        top1, top2 = results[0], results[1]
        margin = top1.score - top2.score
        rows.append(
            {
                "question": question,
                "label": label,
                "top1": top1.score,
                "top2": top2.score,
                "margin": margin,
                "source": top1.source,
                "section": top1.section,
            }
        )
        print(f"{i}. {question}")
        print(f"   expected label : {label}")
        print(f"   top-1 score    : {top1.score:.4f}")
        print(f"   top-2 score    : {top2.score:.4f}")
        print(f"   top-1 source   : {top1.source}")
        print(f"   top-1 section  : {top1.section}")
        print(f"   top1 - top2    : {margin:.4f}")
    print()
    return rows


def overlap_report(name: str, covered: list[float], uncovered: list[float]) -> None:
    c_min, c_max, _ = rng(covered)
    u_min, u_max, _ = rng(uncovered)
    print(f"--- {name} overlap ---")
    print(f"COVERED range     : [{c_min:.4f}, {c_max:.4f}]")
    print(f"NOT COVERED range : [{u_min:.4f}, {u_max:.4f}]")
    # Two closed ranges intersect iff the larger lower bound <= the smaller upper bound.
    lo = max(c_min, u_min)
    hi = min(c_max, u_max)
    if lo <= hi:
        print(f"OVERLAP in [{lo:.4f}, {hi:.4f}] "
              f"(no single cutoff on this metric separates the groups).")
    else:
        print(f"NO OVERLAP: clean gap between the ranges.")
    print()


def main() -> None:
    retriever = build_retriever()
    print(f"Model: {MODEL_NAME}")
    print(f"Chunks indexed: {len(retriever.chunks)}")
    print(f"COVERED questions: {len(COVERED)} | NOT COVERED questions: {len(NOT_COVERED)}")
    print()

    covered_rows = run_group(retriever, "COVERED", COVERED)
    uncovered_rows = run_group(retriever, "NOT COVERED", NOT_COVERED)

    c_top1 = [r["top1"] for r in covered_rows]
    u_top1 = [r["top1"] for r in uncovered_rows]
    c_margin = [r["margin"] for r in covered_rows]
    u_margin = [r["margin"] for r in uncovered_rows]

    c1 = rng(c_top1)
    u1 = rng(u_top1)
    cm = rng(c_margin)
    um = rng(u_margin)

    print("=== Summary ===")
    print(f"COVERED     top-1 score : min {c1[0]:.4f} | max {c1[1]:.4f} | mean {c1[2]:.4f}")
    print(f"NOT COVERED top-1 score : min {u1[0]:.4f} | max {u1[1]:.4f} | mean {u1[2]:.4f}")
    print(f"COVERED     margin      : min {cm[0]:.4f} | max {cm[1]:.4f} | mean {cm[2]:.4f}")
    print(f"NOT COVERED margin      : min {um[0]:.4f} | max {um[1]:.4f} | mean {um[2]:.4f}")
    print()

    overlap_report("top-1 score", c_top1, u_top1)
    overlap_report("margin (top1 - top2)", c_margin, u_margin)


if __name__ == "__main__":
    main()

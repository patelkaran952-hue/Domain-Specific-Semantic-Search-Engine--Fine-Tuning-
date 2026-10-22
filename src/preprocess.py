"""Document/query text construction and deterministic corpus ordering."""
from __future__ import annotations

from typing import Any


def build_doc_text(title: str, text: str, separator: str = "\n") -> str:
    """Title + separator + abstract. No stopword removal or term stripping (CLAUDE.md section 5.2)."""
    return f"{title.strip()}{separator}{text.strip()}"


def build_processed_corpus(records: list[dict], separator: str = "\n") -> list[dict[str, Any]]:
    """Return records sorted by string document ID (lexicographic), the ordering that FAISS rows
    must follow in Phase 2. Original `title`/`text` are kept alongside the joined `doc_text`."""
    out = []
    for rec in sorted(records, key=lambda r: r["_id"]):
        out.append(
            {
                "_id": rec["_id"],
                "title": rec["title"].strip(),
                "text": rec["text"].strip(),
                "doc_text": build_doc_text(rec["title"], rec["text"], separator),
            }
        )
    return out


def build_processed_queries(records: list[dict]) -> list[dict[str, str]]:
    """Queries sorted by string ID; fact-verification metadata is intentionally dropped
    (it is not a relevance label)."""
    return [{"_id": r["_id"], "text": r["text"].strip()} for r in sorted(records, key=lambda r: r["_id"])]

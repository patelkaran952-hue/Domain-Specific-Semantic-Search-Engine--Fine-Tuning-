"""Strict, ID-as-string readers for the SciFact JSONL / TSV files. Never silently drops rows."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

QRELS_HEADER = ["query-id", "corpus-id", "score"]


class DataFormatError(Exception):
    """Raised for malformed or missing input files (message names the file and line)."""


def _require_file(path: Path, hint: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise DataFormatError(f"Missing input file: {path}. {hint}")
    return path


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield one dict per non-empty line; raises with the line number on bad JSON."""
    path = _require_file(path, "Place the original upload in data/raw/ (see README).")
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataFormatError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(rec, dict):
                raise DataFormatError(f"{path}:{lineno}: expected a JSON object, got {type(rec).__name__}")
            yield rec


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def read_qrels(path: Path) -> list[tuple[str, str, int]]:
    """Read a qrels TSV (header: query-id, corpus-id, score). IDs stay strings; CRLF is handled.

    Raises DataFormatError on wrong header, wrong column count, blank IDs or non-integer scores.
    """
    path = _require_file(path, "Place the original upload in data/raw/ (see README).")
    rows: list[tuple[str, str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration as exc:
            raise DataFormatError(f"{path}: file is empty") from exc
        if header != QRELS_HEADER:
            raise DataFormatError(f"{path}:1: expected header {QRELS_HEADER}, got {header}")
        for lineno, row in enumerate(reader, start=2):
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) != 3:
                raise DataFormatError(f"{path}:{lineno}: expected 3 tab-separated columns, got {len(row)}: {row}")
            qid, did, score = (c.strip() for c in row)
            if not qid or not did:
                raise DataFormatError(f"{path}:{lineno}: blank query-id or corpus-id")
            try:
                score_i = int(score)
            except ValueError as exc:
                raise DataFormatError(f"{path}:{lineno}: non-integer score {score!r}") from exc
            rows.append((qid, did, score_i))
    return rows


def write_qrels(rows: list[tuple[str, str, int]], path: Path) -> None:
    """Write qrels with LF endings and the standard header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(QRELS_HEADER)
        writer.writerows(rows)

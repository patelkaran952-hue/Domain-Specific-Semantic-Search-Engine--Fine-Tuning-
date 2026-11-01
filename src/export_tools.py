"""Build a publishable copy of the project WITHOUT touching the working repo.
Whitelist-based: dataset text, model weights, embeddings, indexes, .env and review sheets never enter the export. Measured outputs are
placed under results/ (not artifacts/) so a reproducer's pipeline starts clean and the test-evaluation ledger cannot block their run.
Local absolute paths are scrubbed and notebook stderr streams are dropped. The result is checked by the strict hygiene scan."""
from __future__ import annotations

import fnmatch
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .hygiene import scan
from .utils import sha256_file, utc_now

TEXT_EXPORT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".ipynb", ".cff", ""}
_WIN_USER = re.compile(r"(?i)[A-Za-z]:(?:\\\\|\\|/)+Users(?:\\\\|\\|/)+[^\\/\s\"']+")
_WIN_ANY = re.compile(r"\b[A-Za-z]:(?:\\\\|\\|/)+[A-Za-z0-9._ \-\\/()]{2,}?(?=[\"'\s<>|]|$)")
_POSIX_HOME = re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+")


def scrub_text(text: str, root: Path) -> tuple[str, int]:
    """Replace the repo root, home directories and any remaining drive-letter paths with placeholders. Returns (text, n_replacements)."""
    n = 0
    variants = {str(root), Path(root).as_posix(), str(root).replace("\\", "\\\\")}
    for v in sorted((x for x in variants if len(x) > 3), key=len, reverse=True):
        n += text.count(v)
        text = text.replace(v, "<REPO_ROOT>")
    for rx, rep in ((_WIN_USER, "<HOME>"), (_POSIX_HOME, "<HOME>"), (re.compile(r"file:///[^\s\"')]+"), "file:///<REDACTED>"), (_WIN_ANY, "<LOCAL_PATH>")):
        text, k = rx.subn(rep, text)
        n += k
    return text, n


def scrub_notebook(raw: str, root: Path) -> tuple[str, int]:
    nb = json.loads(raw)
    dropped = 0
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            keep = [o for o in c.get("outputs", []) if not (o.get("output_type") == "stream" and o.get("name") == "stderr")]
            dropped += len(c.get("outputs", [])) - len(keep)
            c["outputs"] = keep
    text, n = scrub_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", root)
    json.loads(text)                                   # must still be a valid notebook
    return text, n + dropped


def collect(root: Path, include: list[str], exclude: list[str]) -> list[Path]:
    root = Path(root)
    found: set[Path] = set()
    for pat in include:
        for p in root.glob(pat):
            if p.is_file():
                found.add(p)
    keep = []
    for p in sorted(found):
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, ex) for ex in exclude):
            continue
        keep.append(p)
    return keep


def remap(rel: str, rules: list[list[str]]) -> str:
    for prefix, new in rules:
        if rel.startswith(prefix):
            return new + rel[len(prefix):]
    return rel


def export(root: Path, dest: Path, spec: dict[str, Any], make_zip: bool = False) -> dict[str, Any]:
    root, dest = Path(root).resolve(), Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} already exists and is not empty; exports never overwrite. Move or delete it yourself.")
    files = collect(root, spec["include"], spec["exclude"])
    if not files:
        raise SystemExit("export whitelist matched no files; check configs/export.yaml and run from the repository root.")
    dest.mkdir(parents=True, exist_ok=True)
    manifest_files, scrubbed_files, replacements = [], 0, 0
    for p in files:
        rel = p.relative_to(root).as_posix()
        out_rel = remap(rel, spec.get("remap", []))
        out = dest / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".py":
            shutil.copyfile(p, out)                        # source code is never rewritten; a private path in code fails the strict scan instead
        elif p.suffix.lower() in TEXT_EXPORT_SUFFIXES and p.stat().st_size < 20_000_000:
            raw = p.read_text(encoding="utf-8", errors="replace")
            new, n = scrub_notebook(raw, root) if p.suffix.lower() == ".ipynb" else scrub_text(raw, root)
            if p.suffix.lower() == ".json":
                json.loads(new)                          # scrubbing must not break JSON
            out.write_text(new, encoding="utf-8", newline="\n")
            scrubbed_files += int(n > 0)
            replacements += n
        else:
            shutil.copyfile(p, out)
        manifest_files.append({"path": out_rel, "from": rel, "size": out.stat().st_size, "sha256": sha256_file(out)})
    hyg = scan(dest, mode="export")
    manifest = {"generated_utc": utc_now(), "n_files": len(manifest_files), "files": manifest_files,
                "scrubbed_files": scrubbed_files, "scrub_replacements": replacements, "hygiene_status": hyg["status"], "hygiene_counts": hyg["counts"],
                "remap": spec.get("remap", []),
                "never_exported": ["data/raw/*", "data/processed/corpus_processed.jsonl", "data/processed/queries_processed.jsonl", "artifacts/models, embeddings, indexes, service",
                                   ".env", "review sheets (contain abstract sentences)", "per-query top-100 run files", "failure-case files (contain claim text)"],
                "notes": "Author's measured outputs are under results/. Running the pipeline writes fresh outputs to artifacts/ (git-ignored)."}
    (dest / "PUBLISH_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if hyg["status"] != "PASS":
        rejected = dest.with_name(dest.name + ".REJECTED")
        if rejected.exists():
            shutil.rmtree(rejected)
        dest.rename(rejected)
        raise SystemExit("Export REJECTED by the strict hygiene scan (moved to " + rejected.name + "):\n" +
                         "\n".join(f"  {f['severity']} {f['kind']} {f['path']}:{f['line']} {f['detail']}" for f in hyg["findings"] if f["severity"] == "FAIL")[:2000])
    if make_zip:
        shutil.make_archive(str(dest), "zip", root_dir=dest)
    return manifest

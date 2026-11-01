"""Pre-publication hygiene scan: secrets, private absolute paths, oversized files, licensed dataset text, tracked .env.
`mode="repo"` scans what Git would commit from the working repo (licensed data = WARN: fine for a private repo).
`mode="export"` is strict: anything that must not be published is a FAIL. Secret values are never printed in full."""
from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv", "venv", "node_modules"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".ipynb", ".ps1", ".sh", ".cff", ""}
LICENSED_NAMES = {"corpus.jsonl", "queries.jsonl", "corpus_processed.jsonl", "queries_processed.jsonl", "train.tsv", "test.tsv"}
BINARY_ASSET_SUFFIXES = {".faiss", ".npy", ".safetensors", ".bin", ".pt", ".onnx"}
STRONG_SECRETS = {
    "api_key_sk": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
}
ASSIGNED_SECRET = re.compile(r"(?i)\b(api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*['\"]?([A-Za-z0-9_\-/+=]{16,})")
PLACEHOLDER = re.compile(r"(?i)(your|example|placeholder|changeme|xxxx|<|\.\.\.|test|dummy|fake|\$\{|env)")
PATH_PATTERNS = [re.compile(r"\b[A-Za-z]:[\\/]{1,2}[A-Za-z0-9._ -]{2,}[\\/]{1,2}"), re.compile(r"/home/[A-Za-z0-9._-]+/"),
                 re.compile(r"/Users/[A-Za-z0-9._-]+/"), re.compile(r"file:///[A-Za-z]:")]
ALLOW_MARK = "hygiene:ok"


@dataclass
class Finding:
    severity: str          # FAIL | WARN
    kind: str
    path: str
    line: int
    detail: str


def candidate_files(root: Path) -> list[Path]:
    """Files that would be published: Git's tracked + untracked-not-ignored set if a repo exists, else everything under root."""
    root = Path(root)
    if (root / ".git").exists():
        try:
            out = subprocess.run(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"], capture_output=True, check=True).stdout
            return sorted(root / p for p in out.decode("utf-8", errors="replace").split("\0") if p and (root / p).is_file())
        except (OSError, subprocess.CalledProcessError):
            pass
    return sorted(p for p in root.rglob("*") if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS))


def _mask(s: str) -> str:
    return f"{s[:4]}…[{len(s)} chars]"


def scan(root: Path, mode: str = "repo", max_mb: float = 50.0, warn_mb: float = 25.0, files: list[Path] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    strict = mode == "export"
    findings: list[Finding] = []
    flist = files if files is not None else candidate_files(root)
    for p in flist:
        rel = p.relative_to(root).as_posix()
        size_mb = p.stat().st_size / 1e6
        if size_mb > max_mb:
            findings.append(Finding("FAIL", "file_too_large", rel, 0, f"{size_mb:.1f} MB > {max_mb} MB (GitHub rejects files over 100 MB)"))
        elif size_mb > warn_mb:
            findings.append(Finding("WARN", "file_large", rel, 0, f"{size_mb:.1f} MB"))
        if p.name == ".env" or (p.name.startswith(".env.") and p.name != ".env.example"):
            findings.append(Finding("FAIL", "env_file_present", rel, 0, "environment file with possible secrets"))
        if p.name in LICENSED_NAMES or p.suffix in BINARY_ASSET_SUFFIXES:
            findings.append(Finding("FAIL" if strict else "WARN", "licensed_or_binary_asset", rel, 0,
                                    "dataset text (SciFact is CC BY-NC 2.0) or a large model/index asset: do not publish unless you have verified the terms"))
        if p.suffix.lower() not in TEXT_SUFFIXES or size_mb > 5:
            continue
        try:
            head = p.read_bytes()
            if b"\0" in head[:4096]:
                continue
            text = head.decode("utf-8", errors="replace")
        except OSError:
            continue
        in_tests = rel.startswith("tests/")
        for i, line in enumerate(text.splitlines(), 1):
            if ALLOW_MARK in line:
                continue
            for kind, rx in STRONG_SECRETS.items():
                m = rx.search(line)
                if m:
                    findings.append(Finding("FAIL", f"secret:{kind}", rel, i, _mask(m.group(0))))
            if not in_tests and not rel.endswith(".env.example"):
                m = ASSIGNED_SECRET.search(line)
                if m and not PLACEHOLDER.search(m.group(2)) and not PLACEHOLDER.search(line):
                    findings.append(Finding("FAIL", "secret:assigned_value", rel, i, f"{m.group(1)}=" + _mask(m.group(2))))
            for rx in PATH_PATTERNS:
                m = rx.search(line)
                if m:
                    findings.append(Finding("FAIL" if strict else "WARN", "private_absolute_path", rel, i, "absolute local path (text redacted from this report)"))
                    break
    names = {Path(f).name.upper() for f in (p.name for p in flist)}
    if not any(n.startswith("LICENSE") for n in names):
        findings.append(Finding("WARN", "no_license_file", "(repo)", 0, "no LICENSE: others have no permission to reuse your code by default"))
    counts: dict[str, int] = {}
    for f in findings:
        counts[f"{f.severity}:{f.kind}"] = counts.get(f"{f.severity}:{f.kind}", 0) + 1
    return {"mode": mode, "status": "FAIL" if any(f.severity == "FAIL" for f in findings) else "PASS", "n_files_scanned": len(flist),
            "counts": counts, "findings": [asdict(f) for f in findings]}

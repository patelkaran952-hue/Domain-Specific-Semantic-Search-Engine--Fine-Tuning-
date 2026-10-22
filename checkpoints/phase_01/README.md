# Phase 01 checkpoint — Setup, ingestion, and dataset audit

**Status: PASS** | generated 2026-10-22T18:27:55+00:00

## Quality gates (real executions)
| Gate | Result |
|---|---|
| pytest | PASS |
| smoke_test.py | PASS |
| data audit | PASS |

## File changes vs previous checkpoint
{'added': 42, 'changed': 0, 'unchanged': 0, 'removed': 0}

## Resume / reproduce (Windows PowerShell, repo root)
```powershell
conda activate scifact-search
python scripts/prepare_data.py
python scripts/inspect_dataset.py
python scripts/smoke_test.py
python -m pytest -q
python scripts/build_notebook_phase01.py --execute
```
Verify integrity: compare SHA256 values in `inventory.json` (`Get-FileHash <file> -Algorithm SHA256`).

## Contents
`manifest.json` (versions, hashes, regeneration), `inventory.json` (every project file: path/size/SHA256),
`configs/`, `reports/` (audit, sample searches, test outputs, environment freeze), `snapshot/` (split IDs and qrels).

## Next phase
Phase 2 — requires explicit user approval ("Proceed to Phase 2").

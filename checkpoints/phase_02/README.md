# Phase 02 checkpoint — Baseline MiniLM embeddings and exact retrieval

**Status: PARTIAL** | generated 2026-10-22T18:49:03+00:00

## Quality gates (real executions)
| Gate | Result |
|---|---|
| pytest | PASS |
| smoke_test.py | PASS |
| data audit | PASS |
| baseline_val_metrics | NOT_RUN |

## File changes vs previous checkpoint
{'added': 11, 'removed': 0, 'changed': 5, 'unchanged': 37}

## Resume / reproduce (Windows PowerShell, repo root)
```powershell
conda activate scifact-search
python scripts/check_env.py --download-model
python scripts/run_baseline.py --smoke
python scripts/run_baseline.py
python scripts/make_checkpoint.py --phase 2 --name "Baseline MiniLM embeddings and exact retrieval" --git --force --require-report artifacts/reports/baseline_val_metrics.json
```
Verify integrity: compare SHA256 values in `inventory.json` (`Get-FileHash <file> -Algorithm SHA256`).

## Limitations
The real baseline run has NOT executed: the build sandbox cannot reach huggingface.co, so no baseline metrics, embeddings or FAISS index exist yet. Code paths were verified only with unit tests on synthetic data and a throwaway dry run with a fake encoder (results discarded).

## Contents
`manifest.json` (versions, hashes, regeneration), `inventory.json` (every project file: path/size/SHA256),
`configs/`, `reports/` (audit, sample searches, test outputs, environment freeze), `snapshot/` (split IDs and qrels).

## Next phase
Phase 3 — requires explicit user approval ("Proceed to Phase 3").

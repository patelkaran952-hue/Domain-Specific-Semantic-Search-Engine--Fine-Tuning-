# Phase 05 checkpoint — Model comparison and held-out test

**Status: PASS** | generated 2026-10-25T19:36:36+00:00

## Quality gates (real executions)
| Gate | Result |
|---|---|
| pytest | PASS |
| smoke_test.py | PASS |
| data audit | PASS |
| model_comparison_val | PASS |
| test_results | PASS |

## File changes vs previous checkpoint
{'added': 33, 'removed': 0, 'changed': 5, 'unchanged': 136}

## Resume / reproduce (Windows PowerShell, repo root)
```powershell
conda activate scifact-search
python scripts/evaluate_models.py --stage validation
python scripts/evaluate_models.py --stage test --confirm-single-test-evaluation
```
Verify integrity: compare SHA256 values in `inventory.json` (`Get-FileHash <file> -Algorithm SHA256`).

## Limitations
None recorded.

## Contents
`manifest.json` (versions, hashes, regeneration), `inventory.json` (every project file: path/size/SHA256),
`configs/`, `reports/` (audit, sample searches, test outputs, environment freeze), `snapshot/` (split IDs and qrels).

## Next phase
Phase 6 — requires explicit user approval ("Proceed to Phase 6").

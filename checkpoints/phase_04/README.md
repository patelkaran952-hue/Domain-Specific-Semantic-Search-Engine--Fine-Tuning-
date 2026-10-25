# Phase 04 checkpoint — MiniLM fine-tuning

**Status: PARTIAL** | generated 2026-10-25T08:38:14+00:00

## Quality gates (real executions)
| Gate | Result |
|---|---|
| pytest | PASS |
| smoke_test.py | PASS |
| data audit | PASS |
| finetune_verification | PARTIAL |

## File changes vs previous checkpoint
{'added': 55, 'removed': 0, 'changed': 12, 'unchanged': 41}

## Resume / reproduce (Windows PowerShell, repo root)
```powershell
conda activate scifact-search
python scripts/finetune.py --experiment positive_only
python scripts/finetune.py --experiment hard_negative
python scripts/verify_finetune.py
```
Verify integrity: compare SHA256 values in `inventory.json` (`Get-FileHash <file> -Algorithm SHA256`).

## Limitations
None recorded.

## Contents
`manifest.json` (versions, hashes, regeneration), `inventory.json` (every project file: path/size/SHA256),
`configs/`, `reports/` (audit, sample searches, test outputs, environment freeze), `snapshot/` (split IDs and qrels).

## Next phase
Phase 5 — requires explicit user approval ("Proceed to Phase 5").

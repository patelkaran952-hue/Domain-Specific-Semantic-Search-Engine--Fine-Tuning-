# Domain-Specific Scientific Semantic Search Engine (SciFact)

> **Status: Phase 1 of 9 complete (setup, ingestion, dataset audit).** No model has been trained or evaluated yet;
> this README is a stub that grows each phase. Project rules live in [`CLAUDE.md`](CLAUDE.md).

## Goal
Fine-tune `sentence-transformers/all-MiniLM-L6-v2` on SciFact claim→abstract relevance pairs, compare against the
un-fine-tuned baseline on held-out queries, index with exact FAISS (`IndexFlatIP`, cosine), and serve via Streamlit.

## Setup (Windows PowerShell, VS Code terminal)
```powershell
conda create -n scifact-search python=3.11 -y
conda activate scifact-search
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```
Place the four original files (immutable) in `data/raw/`: `corpus.jsonl`, `queries.jsonl`, `train.tsv`, `test.tsv`.

## Phase 1 commands
```powershell
python scripts/prepare_data.py        # validate, build data/processed/*, split, write artifacts/reports/data_audit.json
python scripts/inspect_dataset.py     # statistics + TF-IDF sample searches (training queries only)
python scripts/smoke_test.py          # artifact consistency gates
python -m pytest -q
python scripts/build_notebook_phase01.py --execute   # notebooks/01_dataset_audit.ipynb
```

## Data design (summary)
- 5,183 documents, 1,109 queries; official `train.tsv` (919 positives / 809 queries), `test.tsv` (339 / 300).
- Validation = seeded (42) query-level 80/20 split of the official training queries → 647 train / 162 val queries.
- `test.tsv` is reserved for one final evaluation (Phase 5). Qrels are positive-only; unjudged pairs are *unlabeled*.
- IDs are strings everywhere; corpus order is lexicographic by string ID (`data/processed/document_ids.json`).

## Licenses / citations (to complete before publication)
SciFact (Wadden et al., 2020), BEIR (Thakur et al., 2021), and the MiniLM model each carry their own license and
citation requirements; these will be verified and documented in Phase 9.

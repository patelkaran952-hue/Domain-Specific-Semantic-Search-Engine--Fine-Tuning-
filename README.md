# Domain-Specific Scientific Semantic Search Engine (SciFact)

> **Status: Phase 3 code complete; dense mining pending on the user's machine (see below).** No fine-tuning yet.
> This README is a stub that grows each phase. Project rules live in [`CLAUDE.md`](CLAUDE.md).

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

## Phase 2 commands (baseline MiniLM + exact FAISS, validation only)
```powershell
# optional smaller CPU-only torch first:  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/check_env.py --download-model   # confirms every dependency imports and the model downloads
python scripts/run_baseline.py --smoke         # 64 docs / 8 queries, no metrics
python scripts/run_baseline.py                 # full run -> artifacts/reports/baseline_val_metrics.json
python scripts/build_notebook_phase02.py --execute
```
Outputs: `artifacts/embeddings/baseline_docs.npy`, `artifacts/indexes/baseline_flatip.faiss` (+ document-ID map),
`artifacts/reports/baseline_val_*`, and a row in `experiments/runs.csv`. Re-running refuses to overwrite an existing
report unless `--force` is given; the embedding cache is reused only if corpus/model/config hashes match.

## Phase 3 commands (positive pairs + hard-negative mining, training queries only)
```powershell
python scripts/mine_negatives.py --source bm25     # lexical negatives (seconds)
python scripts/mine_negatives.py --source dense    # baseline-MiniLM negatives (needs Phase 2 embeddings + index)
python scripts/verify_mining.py                    # re-checks all exit gates from disk; PASS only if every source exists
python scripts/build_notebook_phase03.py --execute
```
Outputs in `data/mining/`: `train_positive_pairs.jsonl`, `hard_negatives_<src>.jsonl` (with rank/score/provenance),
`train_triplets_<src>.jsonl`, `mining_settings_<src>.json`, `samples_<src>.md` (read these for false negatives).
`configs/finetune.yaml` prepares two experiments that share every hyperparameter: positive-only vs positive + mined negative.
**Mined negatives are unjudged top-ranked documents; some are relevant.** Only `train_qrels.tsv` is read.

## Phase 3 commands (training pairs + hard-negative mining; training queries only)
```powershell
python scripts/mine_negatives.py --source bm25    # lexical negatives
python scripts/mine_negatives.py --source dense   # baseline-MiniLM negatives (needs Phase 2 embeddings + index)
python scripts/verify_mining.py                   # exit gates (PASS / PARTIAL / FAIL)
python scripts/build_notebook_phase03.py --execute
```
Outputs in `data/mining/`: `train_positive_pairs.jsonl`, `hard_negatives_<src>.jsonl` (with rank/score/provenance),
`train_triplets_<src>.jsonl`, `mining_settings_<src>.json`, `samples_<src>.md`. Mined negatives are *unjudged* top-ranked
documents; false-negative risk is flagged (`false_negative_risk_flag`) and optionally filtered with
`mining.max_score_ratio`. `configs/finetune.yaml` prepares two comparable Phase 4 experiments (positive-only vs. mined
negatives); nothing has been trained.

## Phase 4 commands (fine-tuning; CPU, validation-only selection)
```powershell
python -m pip install "sentence-transformers[train]"
python scripts/finetune.py --experiment positive_only --smoke   # pipeline check on 32 rows (no claims made)
python scripts/finetune.py --experiment positive_only           # (anchor, positive) pairs, MNRL
python scripts/finetune.py --experiment hard_negative           # (anchor, positive, mined dense negative), MNRL consumes the negative
python scripts/verify_finetune.py
python scripts/build_notebook_phase04.py --execute
```
Both runs share every hyperparameter in `configs/finetune.yaml` (`training:` block); they differ only in data/columns.
Each run saves `artifacts/models/<run_id>/final` (best epoch on validation), `training_config_resolved.json`,
`train_log_history.json`, `val_metrics_by_epoch.json`, and `artifacts/reports/finetune_<run_id>_report.json`.
Existing model directories are never overwritten. `test.tsv` is not read.

## Phase 5 commands (comparison, validation-only selection, one held-out test evaluation)
```powershell
python scripts/evaluate_models.py --stage validation
python scripts/evaluate_models.py --stage test --confirm-single-test-evaluation   # ONE shot per frozen model
python scripts/build_notebook_phase05.py --execute
```
All three models (pretrained, `ft_pos_only`, `ft_hardneg_dense`) are scored with the same code (`src/evaluate.py`), corpus (5,183 docs),
queries and top-100 depth. The validation stage selects ONE model with a pre-registered rule (highest validation nDCG@10, ties -> simplest),
reports paired bootstrap CIs, and writes `artifacts/reports/model_selection.json` (weights hashes + config fingerprint). The test stage
refuses to run unless that freeze is intact, evaluates only the frozen model and the baseline, and records the evaluation in
`test_evaluation_ledger.json`; after that, validation/selection can no longer be re-run (that would be test-guided iteration).

## Phase 6 commands (packaged search service)
```powershell
python scripts/build_search_package.py        # copies the FROZEN model + embeddings + FAISS index + ID map into artifacts/service/<name>/ (hash-checked, never overwrites)
python scripts/validate_search_service.py     # reload vs Phase 5 rankings (tie-aware), input handling, cosine check, latency AFTER startup
python scripts/search_cli.py "Wnt signaling in stem cells" --top-k 5
python scripts/build_notebook_phase06.py --execute
```
```python
from src.search_service import SearchService
svc = SearchService.load("artifacts/service/scifact_search_v1")      # verifies SHA-256 of every packaged file
hits = svc.search("statin therapy and cardiovascular mortality", top_k=5)   # rank, corpus_id, title, snippet, similarity_score
```
Empty queries return no hits (status `empty_query`); `top_k` above the service limit or corpus size is clipped and flagged; invalid types/values
raise `TypeError`/`ValueError`. `similarity_score` is cosine similarity, **not** a probability of relevance or factual correctness.

## Phase 7 commands (Streamlit UI)
```powershell
python -m pip install -r requirements.txt      # adds streamlit
python scripts/smoke_test_app.py --serve-check   # headless run of the real app against the real package + `streamlit run` health check
streamlit run app/streamlit_app.py               # then open the printed local URL
python scripts/build_notebook_phase07.py --execute
```
The app shows a query box, a top-k slider, ranked results (paper ID, title, snippet, cosine similarity, optional full abstract) and a friendly
message for empty queries. The fine-tuned model is the default; the pretrained baseline is offered (and a side-by-side comparison) **only** when its
embeddings and index exist. Models load lazily and once per process (`st.cache_resource`). Dataset text is Markdown/LaTeX-escaped before display.
Scores are cosine similarities, not probabilities of relevance or truth.

## Data design (summary)
- 5,183 documents, 1,109 queries; official `train.tsv` (919 positives / 809 queries), `test.tsv` (339 / 300).
- Validation = seeded (42) query-level 80/20 split of the official training queries → 647 train / 162 val queries.
- `test.tsv` is reserved for one final evaluation (Phase 5). Qrels are positive-only; unjudged pairs are *unlabeled*.
- IDs are strings everywhere; corpus order is lexicographic by string ID (`data/processed/document_ids.json`).

## Licenses / citations (to complete before publication)
SciFact (Wadden et al., 2020), BEIR (Thakur et al., 2021), and the MiniLM model each carry their own license and
citation requirements; these will be verified and documented in Phase 9.

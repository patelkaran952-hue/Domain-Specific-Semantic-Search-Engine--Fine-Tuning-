# Reproducing the results

Every command below was executed by the author on Windows (Anaconda, CPU only, 8 GB RAM) while building the project; the test suite also runs on Linux.
Timings, metrics and file hashes are **not** quoted here: they are generated into the README from the measured reports so they cannot drift.

## 0. Environment
```powershell
conda create -n scifact-search python=3.11 -y
conda activate scifact-search
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu      # optional: smaller CPU-only build
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/check_env.py --download-model       # confirms imports and fetches the base model (~90 MB)
python -m pytest -q                                 # no GPU and no model download needed
python scripts/verify_fresh_env.py                  # optional: new virtualenv + tests (add --full for the complete install)
```
`requirements.txt` gives version floors. For the exact versions used for the reported results, run `python scripts/freeze_environment.py` in the environment that produced them (writes `requirements.lock.txt`).
Versions observed on the author's machine when the models were trained: sentence-transformers 5.6.0, transformers 5.12.1, torch 2.12.1, datasets 5.1.0, accelerate 1.15.0, huggingface-hub 1.33.0, numpy 2.3.5, faiss-cpu 1.15.0.

## 1. Data (Phase 1)
Download SciFact in BEIR format (see [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md) for the URL and expected hashes) and place `corpus.jsonl`, `queries.jsonl`, `train.tsv`, `test.tsv` in `data/raw/`.
```powershell
python scripts/prepare_data.py          # validates, audits, writes data/processed/*, the query-grouped split and data_audit.json
python scripts/inspect_dataset.py
python scripts/smoke_test.py
```
Check: `data/processed/corpus_processed.jsonl` must have the SHA-256 listed in `DATA_AND_LICENSES.md`, and the split files (`train_query_ids.json`, `val_query_ids.json`, `test_query_ids.json`) should be identical to the ones in `results/` of the published repository.
The published repository's `data/processed/` already contains these ID lists; `prepare_data.py` reports them as `unchanged` when your data matches.

## 2. Baseline (Phase 2)
```powershell
python scripts/run_baseline.py --smoke
python scripts/run_baseline.py
```

## 3. Hard negatives (Phase 3)
```powershell
python scripts/mine_negatives.py --source bm25
python scripts/mine_negatives.py --source dense
python scripts/verify_mining.py
```
Mined negatives are unjudged documents; inspect `data/mining/samples_*.md`.

## 4. Fine-tuning (Phase 4)
```powershell
python scripts/finetune.py --experiment positive_only --smoke
python scripts/finetune.py --experiment positive_only
python scripts/finetune.py --experiment hard_negative
python scripts/verify_finetune.py
```
Run one command at a time and do not interrupt training (a crash records `run_failed.json`; use `--archive-incomplete` to retry). CPU training is only approximately reproducible across library versions, so your numbers will differ slightly from the published ones.

## 5. Comparison and the one-shot test (Phase 5)
```powershell
python scripts/evaluate_models.py --stage validation
python scripts/evaluate_models.py --stage test --confirm-single-test-evaluation
```
The test stage reads `test.tsv` once per frozen model and refuses to repeat (ledger); after it runs, validation/selection cannot be re-run.
In the published repository the author's ledger and reports live under `results/`, so your run starts with an empty `artifacts/reports/` and its own ledger.

## 6. Service, UI and optional answers (Phases 6-8)
```powershell
python scripts/build_search_package.py
python scripts/validate_search_service.py
python scripts/smoke_test_app.py --serve-check
streamlit run app/streamlit_app.py

python scripts/calibrate_rag_gate.py                    # required before any answer panel is offered
python scripts/validate_rag.py
python scripts/rag_review_sheet.py --mode extractive    # then label the CSV by hand
python scripts/rag_review_summarize.py artifacts/reports/rag_review_sheet_extractive.csv
```
LLM answers are optional and off by default; see the comments in `configs/rag.yaml` and the privacy note in `src/rag_generators.py`.

## 7. Documentation and publishing (Phase 9)
```powershell
python scripts/generate_readme_results.py           # fills the README/model-card result blocks from the reports
python scripts/check_repo_hygiene.py
python scripts/export_portfolio.py
```
Then follow [PUBLISHING.md](PUBLISHING.md).

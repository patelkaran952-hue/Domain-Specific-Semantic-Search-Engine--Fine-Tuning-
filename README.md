# Scientific Semantic Search (SciFact): fine-tuned MiniLM + exact FAISS + Streamlit

<!-- HEADLINE:START -->
_Headline results are generated from the measured reports; run the pipeline, then `python scripts/generate_readme_results.py`._
<!-- HEADLINE:END -->

A reproducible, CPU-only semantic search engine over 5,183 scientific abstracts. Given a scientific claim or topic, it returns the most relevant papers
(paper ID, title, snippet, cosine similarity). A lightweight sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is fine-tuned on
claim→abstract relevance pairs from [SciFact](https://github.com/allenai/scifact), compared against the unmodified baseline under a leakage-controlled
protocol, indexed with exact FAISS search, and served through a Streamlit app. An optional, off-by-default layer adds citation-grounded answers.

Everything was built to run on a **Windows laptop with 8 GB RAM and no GPU**.

## Results

All numbers in this section are produced by `scripts/generate_readme_results.py` from the measured report files; none are typed by hand.

<!-- RESULTS:START -->
_Results are generated from `artifacts/reports/` (or `results/reports/` in the published repository)._
<!-- RESULTS:END -->

## Demo

<!-- SCREENSHOTS:START -->
<!-- SCREENSHOTS:END -->

```powershell
streamlit run app/streamlit_app.py
```
Query box, top-k slider, ranked results with paper IDs and cosine scores, an optional fine-tuned-vs-baseline comparison, and an optional evidence panel.
Screenshots appear above only if real images are placed in `docs/screenshots/`.

## How it works

```
 SciFact corpus (5,183 abstracts)         claims + judged positives
          │                                       │
   validate / audit / split  ───────────►  train │ val │ test   (split by QUERY; test sealed until Phase 5)
          │                                       │
   baseline MiniLM embeddings ──► hard-negative mining (train queries only; BM25 + dense)
          │                                       │
          └────────────► fine-tune (MultipleNegativesRankingLoss; positives-only vs +mined negatives)
                                                  │
        validation: shared metrics + paired bootstrap ──► pre-registered selection ──► FROZEN (hashes + fingerprint)
                                                  │
                      one-shot held-out test (ledger-enforced) ──► packaged, hash-verified search service
                                                  │
                  exact FAISS IndexFlatIP over L2-normalized vectors ──► Streamlit UI (+ optional cited answers)
```

| Phase | What was built | Main safeguard |
|---|---|---|
| 1 | Data audit, query-grouped train/val split | IDs as strings, no query straddles splits, raw files immutable |
| 2 | Baseline embeddings + exact FAISS index | cosine = inner product verified; cache keyed by corpus/model hashes |
| 3 | Positive pairs, hard-negative mining | negatives from training queries only; unjudged ≠ irrelevant (false-negative risk flagged) |
| 4 | Fine-tuning (two comparable runs) | identical hyperparameters; no existing run is ever overwritten |
| 5 | Comparison, selection, held-out test | selection rule fixed in advance; test read once (ledger); config fingerprint |
| 6 | Packaged `SearchService` | every packaged file SHA-256 verified; reload reproduces rankings |
| 7 | Streamlit UI | headless smoke test; dataset text escaped before display |
| 8 | Optional cited answers (RAG) | calibrated evidence gate; extractive default; LLM opt-in; output validation |
| 9 | Docs, hygiene, reproducibility tooling | numbers generated from reports; strict publish scan |

## Why the evaluation is trustworthy

- **No test leakage:** the validation split is carved from the *training queries* (query-grouped). Test *judgments* are used in exactly one place, the Phase 5 test stage, after the model is frozen, and a ledger blocks a second evaluation. Earlier phases open `test.tsv` only to list the reserved test query IDs and record its hash.
- **One metric implementation** (`src/evaluate.py`, unit-tested against hand-computed cases) scores every model on the same corpus, queries and retrieval depth. It agreed with the Sentence-Transformers evaluator on the pretrained model.
- **Uncertainty is reported:** paired bootstrap confidence intervals over per-query differences, not just point estimates.
- **Honest negatives:** training labels are positive-only. Mined "negatives" are unjudged documents; the pipeline flags those that look like false negatives.
- **Findings are reproducible across operating systems:** the processed corpus is byte-identical (SHA-256) between a Windows run and a Linux run.

## Quick start (Windows PowerShell)

```powershell
conda create -n scifact-search python=3.11 -y
conda activate scifact-search
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu     # optional: smaller CPU-only PyTorch
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/check_env.py --download-model
```
Put the four SciFact files in BEIR format into `data/raw/` (see [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md) for the official download and expected hashes), then follow
[docs/REPRODUCE.md](docs/REPRODUCE.md) phase by phase. The test suite needs no GPU and no model download:

```powershell
python -m pytest -q
python scripts/verify_fresh_env.py        # builds a brand-new virtual environment and runs the tests inside it
```

## Repository layout

```
src/        library code (data, retrieval, training, comparison, service, RAG, tooling)
scripts/    one script per step (prepare_data, run_baseline, finetune, evaluate_models, build_search_package, ...)
app/        Streamlit UI
configs/    every tunable value (YAML)
tests/      unit, protocol and headless-UI tests
docs/       reproduction guide, data & license notes, model card, publishing guide
notebooks/  thin viewers over the measured reports (one per phase)
results/    (published repo) the author's measured reports and checkpoint manifests
artifacts/  (created when you run the pipeline; git-ignored) models, embeddings, indexes, reports
```

## Limitations and responsible use

- **Sparse, positive-only judgments:** SciFact has about one judged paper per claim. Metrics measure agreement with known positives, not exhaustive relevance; a retrieved but unjudged paper may be relevant.
- **Small query sets and one training seed:** confidence intervals are wide and seed-to-seed variance was not measured.
- **Shared corpus:** all splits search the same 5,183 abstracts, so this benchmarks ranking within a fixed corpus, not retrieval over unseen literature.
- **Possible pretraining overlap:** the base model's published training mix includes S2ORC citation pairs (abstracts), and SciFact's abstracts are drawn from S2ORC. Overlap between the base model's training data and this corpus cannot be ruled out (claim→abstract labels are not part of it).
- **Similarity is not truth:** cosine scores are not probabilities of relevance or of a claim being correct. This is a research prototype, **not medical or clinical advice**.
- **Optional answers:** automatic checks verify structure (citations, numbers), not factual correctness; see the manual-review workflow in `docs/REPRODUCE.md`. A remote LLM would receive your query and retrieved sentences.

## Data, models, licenses and citations

SciFact is released under **CC BY-NC 2.0** (non-commercial); this repository therefore does **not** redistribute the dataset text. The base model is Apache-2.0.
Full details, verified sources, and BibTeX are in [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md). The **code in this repository is licensed under the Apache License 2.0** (see [LICENSE](LICENSE)); that licence does not change the separate terms of the dataset or model.

## Documentation

[REPRODUCE](docs/REPRODUCE.md) · [DATA_AND_LICENSES](docs/DATA_AND_LICENSES.md) · [MODEL_CARD](docs/MODEL_CARD.md) · [PUBLISHING](docs/PUBLISHING.md) · [CLAUDE.md](CLAUDE.md) (project specification and operating protocol)

# Scientific Semantic Search (SciFact): fine-tuned MiniLM + exact FAISS + Streamlit

<!-- HEADLINE:START -->
**On 300 held-out SciFact test queries, the fine-tuned retriever reaches nDCG@10 0.7131 [0.6694, 0.7567] versus 0.6451 [0.6010, 0.6900] for the pretrained `all-MiniLM-L6-v2` baseline. The paired difference is +0.0680 (95% CI [+0.0404, +0.0962]).** One training seed; sparse judgments (see Limitations).
<!-- HEADLINE:END -->

A reproducible, CPU-only semantic search engine over 5,183 scientific abstracts. Given a scientific claim or topic, it returns the most relevant papers
(paper ID, title, snippet, cosine similarity). A lightweight sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is fine-tuned on
claim→abstract relevance pairs from [SciFact](https://github.com/allenai/scifact), compared against the unmodified baseline under a leakage-controlled
protocol, indexed with exact FAISS search, and served through a Streamlit app. An optional, off-by-default layer adds citation-grounded answers.

Everything was built to run on a **Windows laptop with 8 GB RAM and no GPU**.

## Results

All numbers in this section are produced by `scripts/generate_readme_results.py` from the measured report files; none are typed by hand.

<!-- RESULTS:START -->
### Data

5,183 abstracts; 1,109 claims (queries): 647 train / 162 validation (seeded, query-grouped split of the official train queries) / 300 held-out test.

### Validation (model selection)

162 queries, 177 judgments, 5,183-document candidate corpus, top-100 retrieval, one shared metric implementation for every model.

| Model | nDCG@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 95% CI |
|---|---|---|---|---|---|---|
| Baseline MiniLM-L6 (pretrained) | 0.6816 | 0.5329 | 0.7634 | 0.7932 | 0.6536 | [0.6208, 0.7404] |
| Fine-tuned (+ mined dense hard negatives) | 0.7433 | 0.6183 | 0.8117 | 0.8323 | 0.7174 | [0.6830, 0.7994] |
| Fine-tuned (positive pairs) | 0.6974 | 0.5360 | 0.7973 | 0.8364 | 0.6584 | [0.6381, 0.7535] |


Paired per-query nDCG@10 differences (bootstrap):

| Comparison (A − B) | Mean diff | 95% CI | CI excludes 0 | Better/tied/worse |
|---|---|---|---|---|
| ft_hardneg_dense − baseline | +0.0616 | [+0.0205, +0.1043] | yes | 39/106/17 |
| ft_hardneg_dense − ft_pos_only | +0.0458 | [+0.0101, +0.0828] | yes | 38/107/17 |
| ft_pos_only − baseline | +0.0158 | [-0.0154, +0.0490] | no | 19/118/25 |

Selected on validation by the pre-registered rule: **ft_hardneg_dense**. The fine-tuned models' best epoch was also chosen on these queries, so their validation scores are optimistic.

### Held-out test (evaluated once)

300 queries, 339 judgments. The frozen model and the baseline were each evaluated exactly once (ledger-enforced); no tuning followed.

| Model | nDCG@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 95% CI |
|---|---|---|---|---|---|---|
| Baseline MiniLM-L6 (pretrained) | 0.6451 | 0.4823 | 0.7379 | 0.7833 | 0.6047 | [0.6010, 0.6900] |
| Fine-tuned (+ mined dense hard negatives) | 0.7131 | 0.5782 | 0.7743 | 0.8277 | 0.6815 | [0.6694, 0.7567] |


Paired nDCG@10 difference (ft_hardneg_dense − baseline): **+0.0680**, 95% CI [+0.0404, +0.0962]; queries better/tied/worse: 74/197/29.


Change in nDCG@10 from validation to test: baseline -0.0365, ft_hardneg_dense -0.0302.

### Training (CPU, 8 GB RAM class)

| Run | Training data | Epochs | Best stage | Val nDCG@10 before | Val nDCG@10 after | Train time (s) | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|
| ft_hardneg_dense | pairs + mined negatives | 2 | epoch_2 | 0.6816 | 0.7433 | 1272 | 2254 |
| ft_pos_only | pairs | 2 | epoch_2 | 0.6816 | 0.6974 | 974 | 1609 |

Validation numbers in this table come from the Sentence-Transformers evaluator; the comparison tables above use the project's own metric code (the two agreed on the pretrained model).

### Serving (packaged exact-FAISS service, CPU, sequential single-user)

| Setting | p50 ms | p95 ms | Encode ms (mean) | FAISS ms (mean) | Queries/s |
|---|---|---|---|---|---|
| top-k 1 | 15.5 | 22.9 | 15.5 | 0.86 | 60.4 |
| top-k 10 | 17.3 | 23.8 | 16.4 | 0.88 | 54.4 |
| top-k 100 | 23.3 | 30.5 | 14.4 | 0.80 | 41.7 |

Startup (excluded above): 17.8 s total (model load 17.1 s, hash verification 0.6 s). Package 115.6 MB; process RSS after load 500 MB. Reload reproduced Phase-5 rankings exactly for 162/162 validation queries.

### Optional RAG layer

_Not generated yet: run `python scripts/calibrate_rag_gate.py`._


### Provenance

Freeze fingerprint `d8e886cb5d080d5f…`; selected model weights SHA-256 `405b247000543b91…`; seed 42. Full hashes: `results/reports/model_selection.json`.
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
Full details, verified sources, and BibTeX are in [docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md). The license of *this* repository's code is chosen by its author (see [docs/PUBLISHING.md](docs/PUBLISHING.md)).

## Documentation

[REPRODUCE](docs/REPRODUCE.md) · [DATA_AND_LICENSES](docs/DATA_AND_LICENSES.md) · [MODEL_CARD](docs/MODEL_CARD.md) · [PUBLISHING](docs/PUBLISHING.md) · [CLAUDE.md](CLAUDE.md) (project specification and operating protocol)

# Model card: fine-tuned `all-MiniLM-L6-v2` for scientific claim → abstract retrieval

## Model details
- **Base model:** `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0), 384-dimensional embeddings, pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (see `configs/baseline.yaml`).
- **Variants trained:** (1) `ft_pos_only`: claim/positive-abstract pairs; (2) `ft_hardneg_dense`: the same pairs plus one mined hard negative per pair. The model carried forward is the one **selected on validation by a rule fixed in advance** and then frozen; the table below shows which.
- **Inference:** queries and abstracts are L2-normalized; relevance = cosine similarity; exact search with FAISS `IndexFlatIP`. Maximum sequence length 256 word-pieces (longer abstracts are truncated).
- **Weights are not published in this repository** (see licensing note in [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md)).

## Training
- **Data:** judged (claim, abstract) positives from the SciFact *training* queries only; validation queries are a seeded, query-grouped 20% carve-out of the official training queries; the official test queries were never used for training or selection.
- **Mined negatives (variant 2):** top-ranked non-positive abstracts retrieved by the *baseline* model for training queries only (candidate depth 50, pool of 5, one drawn per pair with a fixed seed). These are **unjudged**, not verified irrelevant; a substantial share look like false negatives and are flagged in `data/mining/`.
- **Objective and settings:** `MultipleNegativesRankingLoss` (in-batch negatives plus the mined negative as a third column), `no_duplicates` batch sampler, 2 epochs, batch size 8, learning rate 2e-5, warmup ratio 0.1, weight decay 0.01, seed 42, CPU. Best epoch within a run chosen on validation. Full resolved settings: `results/reports/finetune_*_report.json` (or `artifacts/reports/`).

## Evaluation
Metric code is in `src/evaluate.py` (nDCG@10 primary; Recall@1/5/10, MRR@10, Precision@10, MAP@10). All models are scored on the same 5,183-document corpus with top-100 retrieval. Selection used validation only; the held-out test split was evaluated once for the frozen model and the baseline.

<!-- MODEL_METRICS:START -->
Held-out test (300 queries); selected model: `ft_hardneg_dense`.

| Model | nDCG@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|
| baseline | 0.6451 | 0.4823 | 0.7379 | 0.7833 | 0.6047 |
| ft_hardneg_dense | 0.7131 | 0.5782 | 0.7743 | 0.8277 | 0.6815 |
<!-- MODEL_METRICS:END -->

Confidence intervals, paired differences, validation results and failure analysis are in the README Results section and `results/reports/`.

## Intended use
Ranking SciFact-style scientific abstracts for a claim or topic query; educational and research demonstration of fine-tuning, evaluation hygiene and exact vector search.

## Out of scope / not recommended
- Verifying whether a claim is **true** (the model measures semantic similarity, not support or contradiction).
- Medical, legal or other high-stakes decisions.
- Retrieval over other corpora or languages without re-evaluation; the model was tuned on a small biomedical-leaning claim set.
- Commercial use of anything derived from the SciFact data without checking its CC BY-NC 2.0 terms.

## Limitations and risks
Sparse positive-only judgments (about one per claim); small query sets (hundreds, not thousands) and a **single training seed**, so intervals are wide and seed variance is unmeasured; shared-corpus benchmark; possible overlap between the base model's training mix (it includes S2ORC citation-pair abstracts) and the SciFact corpus; hard negatives may teach the model to push away genuinely relevant papers; scores are not calibrated probabilities. The optional answer layer's automatic checks verify structure, not factual correctness.

## How to cite
Cite SciFact, BEIR, Sentence-BERT and MiniLM (BibTeX in [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md)) and link this repository.

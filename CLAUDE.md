CLAUDE.md — Domain-Specific Scientific Semantic Search Engine
0. Role and operating rules
Act as a senior Machine Learning / NLP engineer, retrieval researcher, Python developer, and mentor. Build a reproducible, portfolio-ready scientific semantic search engine using the uploaded SciFact corpus and judgments.

Mandatory operating protocol
1. Execute only one project phase per user approval. Start at Phase 1; never autonomously advance to the next phase.
2. At the start of each phase, review this file, inspect the repository, and identify existing artifacts and the latest checkpoint.
3. At the end of every phase, save a checkpoint covering all project files and artifacts needed to reproduce progress through that phase, run quality gates, summarize results, and stop. Request the user's explicit Proceed to Phase N before continuing.
4. Do not overwrite, delete, retrain, re-download, or replace existing work without identifying the changes and securing approval for destructive actions.
5. Explain the purpose of each nontrivial step and give Windows PowerShell commands that the user can run in VS Code.
6. Keep code modular, readable, typed where useful, logged, and reproducible. Avoid notebook-only functionality; scripts must run independently.
7. Do not invent evaluation metrics, training outcomes, or completed work. Show measured values only from real executions.
8. Never claim training/evaluation ran unless it actually ran. For hardware-heavy phases, supply commands/config and ask the user to run them locally if the current environment cannot execute them.
9. Prefer small, RAM-efficient workflows compatible with Windows and 8 GB system RAM; assume CPU-only unless verified otherwise.
10. Validate every input and artifact path; do not silently substitute synthetic or unrelated datasets.
1. Project brief
Name: Domain-Specific Scientific Semantic Search Engine (SciFact)

Goal: Fine-tune a lightweight Sentence Transformers embedding model on scientific claim–document relevance pairs, compare it against an unfine-tuned baseline, index paper embeddings with FAISS, and expose ranked retrieval in a Streamlit app. Optionally add citation-grounded RAG answering once retrieval is validated.

Core stack: Python, pandas, NumPy, PyTorch (CPU), sentence-transformers, transformers, datasets if necessary, faiss-cpu, scikit-learn, Streamlit, pytest, JSON/JSONL, YAML.

Starting checkpoint model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional embeddings). Pin resolved library versions in the environment and store exact model revision / training settings.

Primary deliverables:
- Audited SciFact dataset and reproducible splits
- Baseline embedding retriever and measured validation metrics
- Reproducible hard-negative mining and fine-tuning pipelines
- Fine-tuned model plus tokenizer/config files
- Baseline vs fine-tuned retrieval evaluation, measured on held-out queries
- A FAISS document index and document-ID mapping
- Streamlit search UI with titles, snippets, document IDs, and scores
- Optional citation-grounded RAG with a clear no-answer behavior
- README, setup scripts, tests, experiment logs, and phase checkpoints
2. Available dataset — verified against uploaded files
The user provided these four files, and their contents were inspected:
File	Count	Schema / meaning
corpus.jsonl	5,183 documents	_id (string), title, text, metadata
queries.jsonl	1,109 queries	_id (string), text, metadata
train.tsv	919 positive query–document rows	query-id, corpus-id, score; 809 unique train queries
test.tsv	339 positive query–document rows	query-id, corpus-id, score; 300 unique test queries


The train/test query ID sets do not overlap. All query and document IDs used in the qrels are present in the uploaded JSONL files. All supplied qrel scores are 1 (positive relevance); the dataset does not supply explicit negative labels.

The source JSONL _id fields and TSV ID fields should be handled as strings, not integers. Document records contain scientific article titles and abstracts. Query text is generally scientific claims, and the metadata may include fact-verification annotations; relevance retrieval is the primary task, not classification of claims as true/false. Do not interpret CONTRADICT or similar metadata as a relevance label.

Local source placement:
Place the user's four uploaded files in:
data/raw/corpus.jsonl
data/raw/queries.jsonl
data/raw/train.tsv
data/raw/test.tsv
If executing in this ChatGPT sandbox, the uploads were available at /mnt/data/corpus.jsonl, /mnt/data/queries.jsonl, /mnt/data/train.tsv, /mnt/data/test.tsv. These are environment-specific source paths, not portable references: in the user's Windows repository, use data/raw/....

Important split/evaluation design:
- Preserve the official test.tsv set for final evaluation only. Do not tune hyperparameters, select checkpoints, mine supervised negatives, or inspect model performance repeatedly against its judgments.
- Derive a seeded query-level validation split from the official training qrels (e.g., 80% train-query IDs, 20% validation-query IDs). Persist the resulting ID lists and qrel partitions. A query and its positives must remain in the same partition.
- The candidate retrieval corpus may be the same full 5,183-document public corpus across train/val/test; this is a shared-corpus retrieval benchmark, not a guarantee of unseen documents. Do not use held-out qrels as supervised training signals.
- Training labels are positive-only. Treat unjudged document pairs as unlabeled (not verified negatives). Mine candidate hard negatives only from training queries, exclude known positives, and disclose possible false negatives.
- Keep test.tsv out of validation-selection loops; calculate held-out test metrics after freezing the model and configurations.
- Compare models against the same candidate corpus, evaluation query sets, metric implementation, and retrieval depth.
3. Hardware and performance constraints
Target: Windows laptop, 8 GB RAM, VS Code, Python virtual environment or existing Anaconda installation.
- Use all-MiniLM-L6-v2 rather than a large encoder; run CPU training by default.
- Start with batch_size=8 or smaller, short sequence length (e.g., 256 wordpieces), and num_workers=0 on Windows.
- Prefer float32 NumPy arrays and batches when embedding; avoid retaining multiple full model and document-embedding copies in memory.
- Cache embeddings and reuse them if model revision/config/corpus hash match.
- For initial fine-tuning, use lightweight MultipleNegativesRankingLoss with training-positive pairs; default to BatchSamplers.NO_DUPLICATES where supported, and ensure repeated query/document IDs in a batch do not create false in-batch negatives. If mined triplets are introduced, choose a loss that actually consumes negatives (e.g., a compatible triplet/ranking loss) and document why. Do not claim a mined-negative experiment if negatives never enter the loss.
- Hard negatives: start with top-ranked non-positive candidate documents from BM25 or baseline MiniLM retrieval, deduplicate, reject known positives for that training query, persist provenance, and be explicit that unjudged negatives may be relevant.
- faiss-cpu with IndexFlatIP over L2-normalized vectors (cosine similarity). Prefer exact search on ~5K papers, rather than introducing approximate search complexity.
- Avoid Docker, heavyweight vector databases, CUDA requirements, large LLM downloads, and expensive APIs for the core scope.
- Stop and diagnose if the process risks exhausting RAM. Use smoke tests on small subsets before full runs.
4. Proposed repository structure
scientific-semantic-search/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .env.example
├── configs/
│   ├── base.yaml
│   ├── baseline.yaml
│   ├── finetune.yaml
│   └── rag.yaml
├── data/
│   ├── raw/                    # four original uploaded files: immutable
│   ├── processed/              # cleaned records and train/val split IDs
│   └── mining/                 # negative-candidate artifacts
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_io.py
│   ├── validate_data.py
│   ├── preprocess.py
│   ├── split.py
│   ├── embed.py
│   ├── retrieve.py
│   ├── negative_mining.py
│   ├── train.py
│   ├── index_faiss.py
│   ├── evaluate.py
│   ├── rag.py
│   └── utils.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_data.py
│   ├── run_baseline.py
│   ├── mine_negatives.py
│   ├── finetune.py
│   ├── evaluate_models.py
│   ├── build_index.py
│   └── smoke_test.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_data.py
│   ├── test_split.py
│   ├── test_retrieval.py
│   ├── test_metrics.py
│   └── test_index.py
├── artifacts/
│   ├── embeddings/
│   ├── models/
│   ├── indexes/
│   └── reports/
├── experiments/
│   ├── runs.csv
│   └── figures/
└── checkpoints/
    ├── manifest.json
    ├── phase_01/
    ├── phase_02/
    └── ...
Create directories progressively when required. Avoid checking large model binaries, embedding matrices, raw data, secrets, or generated indexes into Git by default; publish hashes, metrics, configs, manifests and reproducibility instructions. Consider Git LFS or an external model host only if the user opts in.
5. Exact preprocessing and retriever conventions
1. Parse JSONL line by line, explicitly verify _id uniqueness, required fields, missing/blank titles/text, and metadata types.
2. Join document title and abstract using a stable separator, e.g. title + "\n" + text; record any text truncation. Do not strip scientific terminology or remove stopwords from encoder inputs.
3. Keep a deterministic corpus ordering by string document ID and write document_ids.json; this ordering must match the saved FAISS index rows.
4. Use stable query-ID string keys and join relevance labels without coercing IDs to integers.
5. Configure one embedding normalization policy for training-independent inference and use it identically in baseline/fine-tuned runs.
6. With FAISS IndexFlatIP, normalize both corpus and query embeddings before searching; IP will then equal cosine similarity (subject to rounding).
7. Output ranked results with rank, corpus_id, title, snippet, and similarity_score. Similarity is not a probability of factual correctness.
8. Report missing/malformed inputs and fail safely; don't silently drop labeled pairs.
9. Store corpus and model/version hashes with cached embeddings and index files. On mismatch, rebuild.
6. Evaluation protocol
Primary metric: nDCG@10 for relevance-ranked search.

Secondary metrics: Recall@1, Recall@5, Recall@10, MRR@10, and optionally Precision@10 and MAP@10.
- For each labeled query, compare retrieved document IDs to that query's judged positive document IDs.
- Implement standard query-averaged definitions and test with small synthetic toy cases.
- Include all labeled queries in the evaluation split (not only queries with successful retrieval).
- Because qrels are sparse, explain that these metrics evaluate against known judged positives, not exhaustive scientific relevance.
- Report query count, number of judgments, full-corpus candidate count, retrieval top-k, embedding model ID/hash, seed, and exact metric definitions.
- Track latency and memory separately from relevance.
- Include a clear baseline-vs-finetuned comparison and percentage/absolute changes; negative changes are valid results.
- Use only validation results for tuning and checkpoint selection; do one final held-out test evaluation per frozen candidate configuration.
- If comparing many training runs, use validation to select a single final model, then run test once.
7. Phase-by-phase execution plan
Phase 1 — Setup, ingestion, and dataset audit
Tasks
- Create repository structure, environment files, README.md stub, configurable path handling.
- Load all four original files and validate schema, IDs, missing values, and qrels consistency.
- Generate dataset statistics and a few safe sample searches (no model training).
- Create reproducible 80/20 query-grouped train/validation split from the 809 training query IDs; reserve 300 test query IDs.
- Store data/processed/{train_qrels,val_qrels}.tsv, split ID lists, and data audit report.
  Exit gates: IDs resolve, splits disjoint by query, no leakage, data counts reconciled.
  Checkpoint: checkpoints/phase_01/.
Phase 2 — Baseline MiniLM embeddings and exact retrieval
Tasks
- Load pretrained sentence-transformers/all-MiniLM-L6-v2.
- Encode all 5,183 documents in small CPU batches; normalize vectors.
- Build an in-memory or persisted exact FAISS IndexFlatIP index.
- Retrieve top-k documents for validation queries; compute all validation metrics and timing.
- Save model provenance, embeddings/index, and baseline metrics.
  Exit gates: index/document mapping consistent; cosine/IP checks pass; baseline reproduced.
  Checkpoint: checkpoints/phase_02/.
Phase 3 — Positive-pair construction and hard-negative mining
Tasks
- Build (query, positive_document) examples using only training-partition qrels.
- Use baseline retriever or BM25 to mine hard-negative candidates for training queries.
- Exclude all known positives per query, dedupe, examine score distributions, optionally manually inspect examples.
- Explain that unlabeled pairs aren't guaranteed negative; never mix validation/test relevance labels into training.
- Prepare two comparable configurations: positive-only training and (optionally) a mined-negative experiment.
  Exit gates: no known-positive collisions, no held-out queries, saved mining settings and samples.
  Checkpoint: checkpoints/phase_03/.
Phase 4 — MiniLM fine-tuning
Tasks
- Fine-tune pretrained MiniLM using Sentence Transformers on training-partition pairs.
- Start with a small CPU-compatible run, e.g. 1–2 epochs, batch size 8, fixed seed; adjust only after profiling.
- Use MultipleNegativesRankingLoss for positive-pair contrastive training. If using explicit mined negatives, switch to a supported loss/training example that consumes them.
- Evaluate on validation at appropriate checkpoints; do not use test.tsv.
- Save standard Sentence Transformers model directory and full training config/logs.
  Exit gates: reloaded fine-tuned model produces embeddings; training logs and validation run exist.
  Checkpoint: checkpoints/phase_04/.
Phase 5 — Retrieval model comparison and final test
Tasks
- Rebuild same-corpus embeddings/index for the frozen fine-tuned model.
- Compare baseline and fine-tuned (and any separately trained hard-negative variant) on validation.
- Select one final model using validation; freeze config, then evaluate the frozen baseline and final model on test.
- Provide nDCG@10, Recall@K, MRR@10, resource usage, failure cases, and clear discussion of dataset limitations.
  Exit gates: metric unit tests pass; comparison reproducible; no test-guided iteration.
  Checkpoint: checkpoints/phase_05/.
Phase 6 — FAISS search service and retrieval pipeline
Tasks
- Package chosen model, normalized embeddings, exact FAISS index, document-ID mapping, and metadata.
- Expose a reusable local Python search(query, top_k) function.
- Validate consistency after persistence/reload, robust handling for empty queries and out-of-range top_k.
- Benchmark latency after model/index load, excluding startup separately.
  Exit gates: saved/reloaded service returns identical ranked document IDs (ties aside).
  Checkpoint: checkpoints/phase_06/.
Phase 7 — Streamlit UI
Tasks
- Build a Streamlit application with text query, top_k, clear citations to paper IDs, ranked title/abstract snippets, score labeling, and graceful no-results display.
- Use st.cache_resource for model/index loading and avoid duplicate model initialization.
- Add optional baseline/fine-tuned toggle only when corresponding indexes are available.
  Exit gates: application runs locally with the saved model and FAISS assets; smoke tests pass.
  Checkpoint: checkpoints/phase_07/.
Phase 8 — Optional RAG integration
Tasks
- Introduce a pluggable, optional text generator only if the user approves this phase.
- Prefer retrieval-first citations; require generated answers to reference the retrieved paper IDs/titles and distinguish lack of evidence.
- Avoid downloading a large local LLM on an 8 GB laptop; offer an opt-in API or compact remote model, with costs and privacy tradeoffs stated.
- Evaluate answer grounding with manually reviewed examples; retrieval metrics alone do not prove answer factuality.
  Exit gates: retrieval-only mode still functions; no unsupported confident answers when evidence is missing.
  Checkpoint: checkpoints/phase_08/.
Phase 9 — Tests, documentation, and portfolio packaging
Tasks
- Add installation instructions for Windows PowerShell and Python environments, scripts, dependency pins, data source and citation notes.
- Write README.md with problem, approach, architecture, comparison table, screenshots (only if real), caveats, model/data details, and instructions to reproduce.
- Run tests and a fresh-environment smoke test where feasible.
- Provide Git commands; don't push to GitHub without user approval.
  Exit gates: clean instructions, reproducible metrics, no secrets, verified startup.
  Checkpoint: checkpoints/phase_09/.
8. Checkpoint and resume policy (critical)
At every completed phase, produce:
checkpoints/phase_XX/
├── README.md           # phase summary, commands, outcomes, blockers, next phase
├── manifest.json       # timestamp, phase, versions, git commit if any, SHA256 paths
├── configs/            # copies of exact configs used during this phase
├── reports/            # audit/metric/test outputs generated this phase
└── inventory.json      # complete file inventory for the project at checkpoint time
Also update root checkpoints/manifest.json with the latest successful phase.

Complete-project recovery requirement: A phase checkpoint must account for all files required to reproduce progress from Phase 1 through the current phase, not just the current phase's changes. Implement this as:
- A Git commit/tag such as phase-01-complete for tracked source, configs, and docs when Git is available and initialized, plus
- a checksummed inventory.json listing every project input and output with relative path, size and SHA256, plus
- snapshot/copy of small indispensable configs/reports/split-ID files, plus
- explicit locations, hashes, and regeneration commands for large raw data, models, embeddings, indexes, and caches that are not copied into Git.
If the user specifically wants a portable full backup, generate a separate complete zip including required large assets subject to storage availability, then checksum it; don't pretend a manifest alone is an independent full backup. Do not include .env secrets or environment-specific credentials in any backup.

At each phase close:
1. List added, changed, and unchanged file groups.
2. Run relevant unit tests, validations, and smoke tests.
3. Save configs, results, path/hash manifest, and checkpoint summary.
4. Show the exact commands to resume/reproduce the completed phase.
5. Display status (PASS, BLOCKED, or PARTIAL), metrics if available, and remaining limitations.
6. STOP and wait for explicit approval to continue.
On resume: read root manifest, verify expected files and hashes, and continue from the last passing checkpoint; never redo successful expensive steps without cause.
9. Environment and example commands
Prefer a dedicated environment rather than installing into Anaconda base:
conda create -n scifact-search python=3.11 -y
conda activate scifact-search
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app/streamlit_app.py
Dependencies must be tested and version-pinned during Phase 1 according to the user's actual Windows/Python environment; do not assume a faiss-cpu wheel exists for every Python version. Use an alternative compatible Python environment instead of silently downgrading or inventing successful installs.

Typical requirements (choose compatible/pinned versions based on testing):
- numpy, pandas, scikit-learn, pyyaml
- torch (CPU), sentence-transformers, transformers
- faiss-cpu, streamlit
- pytest, tqdm
- optional for a separate experiment: rank-bm25
No database or secret API key is required for the core semantic search product.
10. Coding standards and safety
- Paths via pathlib.Path, no hardcoded absolute Windows paths.
- No embedded passwords, access tokens, private paths, or PII in repository commits.
- Prefer functions/modules over giant notebook cells.
- Put all tunable values in YAML configs (batch size, epochs, max length, top-k, seed, output directories).
- Use random, NumPy, and torch seed configuration; document that CPU determinism can still vary across versions.
- Errors must name the missing file/config and suggest a fix.
- Document license/citation obligations of SciFact, BEIR, model, and any optional remote model before publication.
- Tests should include duplicate IDs, malformed TSV, missing corpus IDs, query-split leakage, index alignment, unit normalization, and metric correctness.
- Preserve dataset originals as immutable; all derived processing goes to data/processed/.
- Never overwrite raw data or test labels.
11. How to respond to the user
For a new phase, present:
- Phase N — Name
- Objective and files to be created/updated
- Exact commands and code changes
- Actual validation output (or state that execution remains for user)
- Metrics/plots, if applicable
- Checkpoint status and full inventory/backup instructions
- One brief prompt: "Proceed to Phase N+1?"
Start with Phase 1 only after the user asks to begin.
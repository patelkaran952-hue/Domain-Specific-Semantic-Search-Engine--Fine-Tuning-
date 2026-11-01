# Data, models, licenses and citations

> I am not a lawyer. This page records what was verified from primary sources while preparing the project; re-check the linked pages before publishing or using anything commercially.

## Components and their terms

| Component | Terms (as verified) | Source |
|---|---|---|
| **SciFact** claims, abstracts and judgments | **CC BY-NC 2.0** (attribution, **non-commercial**). Abstracts originate from the S2ORC corpus. | Dataset card on Hugging Face (`allenai/scifact`), which points to the project's `LICENSE.md`; project: https://github.com/allenai/scifact |
| **BEIR** (the packaging/format of SciFact used here) | BEIR *code* is Apache-2.0. BEIR states it redistributes datasets "in a specific format" and does **not** claim you have a license to use the underlying data, so the SciFact terms above govern the data. | https://github.com/beir-cellar/beir |
| **`sentence-transformers/all-MiniLM-L6-v2`** | Apache-2.0. Fine-tuned from `nreimers/MiniLM-L6-H384-uncased` on over one billion sentence pairs; its card lists S2ORC citation pairs (abstracts) among the sources. | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| **This repository's code** (`src/`, `scripts/`, `app/`, `tests/`, `docs/`) | **Apache License 2.0** (see `LICENSE`). It applies to the code and documentation only, not to SciFact or any model weights. | `LICENSE` |
| Python libraries (PyTorch, Transformers, Sentence-Transformers, FAISS, Streamlit, NumPy, pandas, scikit-learn, ...) | Open-source licenses declared by each package. Print the license metadata of what you have installed with `python scripts/freeze_environment.py --licenses`. | package metadata |

## What this repository does and does not contain

- **Not included:** the SciFact abstracts/claims (`corpus.jsonl`, `queries.jsonl`, and the processed copies), the model weights, embeddings and FAISS indexes. `scripts/export_portfolio.py` enforces this and the strict hygiene scan fails the export if dataset text appears.
- **Included:** code, configs, tests, documentation, measured reports, document/query **IDs** with split membership and judged-positive pairs (derived annotations; they fall under the dataset's terms, so keep attribution and non-commercial use in mind), and a short hand-written off-topic query list.
- **Fine-tuned weights:** trained on non-commercially licensed data. Whether and how derived weights may be redistributed is a legal question this project has not resolved; they are not published here. If you share them, treat them as non-commercial, state the training data and consult the license holders.
- Previous commits in a working repository may contain the processed corpus (earlier phases committed it). Publish from the clean export described in [PUBLISHING.md](PUBLISHING.md), not from that history.

## Getting the data (official, BEIR format)

BEIR distributes SciFact as a zip: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip (if the host changes, follow the dataset links in the BEIR repository).
Place these four files in `data/raw/` (the BEIR archive keeps the qrels in a `qrels/` subfolder):

| File | Expected rows | SHA-256 of the files used for the reported results |
|---|---|---|
| `corpus.jsonl` | 5,183 documents | `dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6` |
| `queries.jsonl` | 1,109 queries | `8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313` |
| `train.tsv` | 919 judgments / 809 queries | `a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d` |
| `test.tsv` | 339 judgments / 300 queries | `0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c` |

If a hash differs (for example because of line endings in the TSVs), `python scripts/prepare_data.py` still validates the schema and IDs. The strongest cross-check is the **processed corpus**: after `prepare_data.py`,
`data/processed/corpus_processed.jsonl` should have SHA-256 `3b603fece11381ca339457929cd0c6c9e20ac08c08c2c7ab3494967bb53ee96f` (identical on a Windows run and a Linux run). Check with `Get-FileHash data\processed\corpus_processed.jsonl -Algorithm SHA256`.

## Citations

If you use this work, please cite the underlying resources. BibTeX below was assembled from the papers' public records (arXiv identifiers given); verify details against the publisher pages.

```bibtex
@inproceedings{wadden2020fact,
  title     = {Fact or Fiction: Verifying Scientific Claims},
  author    = {Wadden, David and Lin, Shanchuan and Lo, Kyle and Wang, Lucy Lu and van Zuylen, Madeleine and Cohan, Arman and Hajishirzi, Hannaneh},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2020},
  eprint    = {2004.14974},
  archivePrefix = {arXiv}
}

@inproceedings{thakur2021beir,
  title     = {BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  author    = {Thakur, Nandan and Reimers, Nils and R{\"u}ckl{\'e}, Andreas and Srivastava, Abhishek and Gurevych, Iryna},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems, Datasets and Benchmarks Track},
  year      = {2021}
}

@inproceedings{reimers2019sentencebert,
  title     = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year      = {2019},
  eprint    = {1908.10084},
  archivePrefix = {arXiv}
}

@inproceedings{wang2020minilm,
  title     = {MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers},
  author    = {Wang, Wenhui and Wei, Furu and Dong, Li and Bao, Hangbo and Yang, Nan and Zhou, Ming},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020},
  eprint    = {2002.10957},
  archivePrefix = {arXiv}
}

@article{johnson2019faiss,
  title   = {Billion-scale similarity search with GPUs},
  author  = {Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal = {IEEE Transactions on Big Data},
  year    = {2019},
  eprint  = {1702.08734},
  archivePrefix = {arXiv}
}

@inproceedings{lo2020s2orc,
  title     = {S2ORC: The Semantic Scholar Open Research Corpus},
  author    = {Lo, Kyle and Wang, Lucy Lu and Neumann, Mark and Kinney, Rodney and Weld, Daniel S.},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year      = {2020},
  eprint    = {1911.02782},
  archivePrefix = {arXiv}
}
```

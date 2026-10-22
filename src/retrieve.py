"""Retrieval helpers. Phase 1 only contains a training-free lexical sanity probe (TF-IDF);
dense retrieval / FAISS arrive in Phase 2."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfProbe:
    """Cosine TF-IDF search over processed documents. A sanity check only, NOT the project baseline."""

    def __init__(self, docs: list[dict]):
        self.doc_ids = [d["_id"] for d in docs]
        self.titles = [d["title"] for d in docs]
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, dtype=np.float32)
        self.matrix = self.vectorizer.fit_transform(d["doc_text"] for d in docs)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.lexsort((np.arange(len(scores)), -scores))[:top_k]  # stable tie-break
        return [
            {"rank": r + 1, "corpus_id": self.doc_ids[i], "title": self.titles[i], "score": float(scores[i])}
            for r, i in enumerate(order)
        ]

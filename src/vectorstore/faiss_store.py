"""FAISS vector store with JSON metadata sidecar and disk persistence.

Uses inner product on normalized vectors (= cosine similarity)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaissStore:
    def __init__(self, dim: int):
        import faiss
        self._faiss = faiss
        self.index = faiss.IndexFlatIP(dim)
        self.meta: list[dict] = []   # position-aligned with index rows

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        assert len(vectors) == len(metadatas)
        self.index.add(vectors)
        self.meta.extend(metadatas)

    def search(self, query_vec: np.ndarray, top_k: int = 6,
               applicant_id: str | None = None) -> list[dict]:
        """Search, optionally filtered to one applicant (post-filter with over-fetch)."""
        k = top_k * 10 if applicant_id else top_k
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(query_vec.reshape(1, -1), k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            m = self.meta[i]
            if applicant_id and m["applicant_id"] != applicant_id:
                continue
            results.append({**m, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    # ---- persistence ----
    def save(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(d / "index.faiss"))
        (d / "meta.json").write_text(json.dumps(self.meta))

    @classmethod
    def load(cls, dir_path: str | Path) -> "FaissStore":
        import faiss
        d = Path(dir_path)
        index = faiss.read_index(str(d / "index.faiss"))
        store = cls.__new__(cls)
        store._faiss = faiss
        store.index = index
        store.meta = json.loads((d / "meta.json").read_text())
        return store

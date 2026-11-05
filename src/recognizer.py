"""Embedding gallery and cosine-similarity matching.

Embeddings are L2-normalised, so cosine similarity is a dot product. For the
size of a school (hundreds to a few thousand students) a NumPy matmul is
sub-millisecond; FAISS (``IndexFlatIP``) is used optionally when installed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


@dataclass
class Match:
    roll_no: Optional[str]   # None when below threshold / ambiguous
    score: float             # best cosine similarity
    best_roll: Optional[str]  # best candidate regardless of threshold
    second_score: float = -1.0


class FaceGallery:
    """In-memory matrix of enrolled embeddings."""

    def __init__(self, roll_nos: Sequence[str], embeddings: np.ndarray, threshold: float = 0.45,
                 margin: float = 0.0, use_faiss: bool = False) -> None:
        if len(roll_nos) != len(embeddings):
            raise ValueError("roll_nos and embeddings length mismatch")
        self.roll_nos = list(roll_nos)
        self.matrix = l2norm(np.asarray(embeddings, dtype=np.float32).reshape(len(roll_nos), -1)) \
            if len(roll_nos) else np.zeros((0, 512), dtype=np.float32)
        self.threshold = float(threshold)
        self.margin = float(margin)
        self._index = None
        if use_faiss and len(self.roll_nos):
            try:
                import faiss  # type: ignore
                self._index = faiss.IndexFlatIP(self.matrix.shape[1])
                self._index.add(self.matrix)
                log.info("FAISS index built with %d embeddings", len(self.roll_nos))
            except ImportError:
                log.warning("use_faiss=true but faiss is not installed; using NumPy search.")

    @classmethod
    def from_db(cls, db, class_name: Optional[str], threshold: float, margin: float = 0.0,
                use_faiss: bool = False) -> "FaceGallery":
        students = db.list_students(class_name)
        if not students:
            return cls([], np.zeros((0, 512), dtype=np.float32), threshold, margin, use_faiss)
        return cls([s.roll_no for s in students], np.stack([s.embedding for s in students]),
                   threshold, margin, use_faiss)

    def __len__(self) -> int:
        return len(self.roll_nos)

    def similarities(self, emb: np.ndarray) -> np.ndarray:
        """Cosine similarity of one or more embeddings (N, D) against the gallery -> (N, G)."""
        q = l2norm(np.atleast_2d(emb))
        if len(self.roll_nos) == 0:
            return np.zeros((q.shape[0], 0), dtype=np.float32)
        return q @ self.matrix.T

    def match(self, emb: np.ndarray, threshold: Optional[float] = None) -> Match:
        return self.match_many(np.atleast_2d(emb), threshold)[0]

    def match_many(self, embs: np.ndarray, threshold: Optional[float] = None) -> list[Match]:
        thr = self.threshold if threshold is None else float(threshold)
        embs = np.atleast_2d(embs)
        if len(self.roll_nos) == 0 or embs.shape[0] == 0:
            return [Match(None, -1.0, None) for _ in range(embs.shape[0])]
        k = min(2, len(self.roll_nos))
        if self._index is not None:
            scores, idx = self._index.search(l2norm(embs), k)
        else:
            sims = self.similarities(embs)
            idx = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idx, axis=1)
        out: list[Match] = []
        for row_s, row_i in zip(scores, idx):
            best, best_i = float(row_s[0]), int(row_i[0])
            second = float(row_s[1]) if k > 1 else -1.0
            roll = self.roll_nos[best_i]
            accepted = best >= thr and (best - second) >= self.margin
            out.append(Match(roll if accepted else None, best, roll, second))
        return out

"""
Matching: given a probe embedding, find the best-matching enrolled identity.

All stored embeddings are unit-length (see engine.py), so cosine similarity
between the probe and any stored vector is just a dot product -- no need
for a vector database or approximate search at apartment scale (dozens to
low hundreds of samples). A full (N, 512) x (512,) matrix-vector product is
microseconds.

A match requires two things to both hold:
  1. best_score            >= THRESHOLD
  2. best_score - best_other_identity_score >= MARGIN_MIN

The margin is measured against the best score belonging to a DIFFERENT
identity than the winner -- not simply the second-highest sample overall,
which is almost always another photo of the same person and would make the
margin check meaningless. This margin is what catches confident-but-wrong
matches between similar-looking people.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from face import config
from face.store import FaceStore


@dataclass
class Candidate:
    name: str
    score: float


@dataclass
class MatchResult:
    decision: str                      # "match" | "no_match" | "empty_store"
    top: list[Candidate] = field(default_factory=list)  # ranked, best first
    margin: float = 0.0                # best.score - best-other-identity.score

    @property
    def best(self) -> Candidate | None:
        return self.top[0] if self.top else None


def _best_per_identity(names: list[str], scores: np.ndarray) -> list[Candidate]:
    """Collapse per-sample scores to one best score per identity, ranked
    best-first."""
    best: dict[str, float] = {}
    for name, score in zip(names, scores):
        if name not in best or score > best[name]:
            best[name] = float(score)
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return [Candidate(name=n, score=s) for n, s in ranked]


def query(
    store: FaceStore,
    probe_embedding: np.ndarray,
    threshold: float = config.THRESHOLD,
    margin_min: float = config.MARGIN_MIN,
    top_k: int = config.TOP_K,
) -> MatchResult:
    """Compare a probe embedding against every enrolled identity in the store."""
    if store.is_empty():
        return MatchResult(decision="empty_store", top=[])

    probe = probe_embedding.astype(np.float32)
    sample_scores = store.vectors @ probe  # (N,) cosine similarities (unit vectors)
    sample_names = [e.name for e in store.entries]

    ranked = _best_per_identity(sample_names, sample_scores)
    top = ranked[:top_k]

    best = ranked[0]
    margin = (best.score - ranked[1].score) if len(ranked) > 1 else best.score

    is_match = best.score >= threshold and margin >= margin_min
    return MatchResult(decision="match" if is_match else "no_match", top=top, margin=margin)


def compare(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Raw cosine similarity between two (assumed unit-normalized) embeddings."""
    a = embedding_a.astype(np.float32)
    b = embedding_b.astype(np.float32)
    return float(np.dot(a, b))

"""
Flat-file persistence for enrolled face embeddings.

Deliberately schema-light: this phase shouldn't commit to a database
schema before it's known what a "profile" actually needs to hold. This
module is the *only* place that touches disk, so swapping it for SQLite
or SQLAlchemy later changes nothing else in the package -- engine.py,
quality.py and matcher.py are unaffected.

Layout on disk:
    data/embeddings.npz   vectors: (N, 512) float32, L2-normalized
    data/manifest.json    schema_version, model info, entries[] (row-aligned
                           with vectors -- entries[i] describes vectors[i])
    data/crops/<id>.png   the aligned 112x112 crop for each sample
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from face import config
from face.io import save_bgr

SCHEMA_VERSION = 1


class StoreError(Exception):
    """Raised when the on-disk store is missing, corrupt, or incompatible."""


@dataclass
class SampleEntry:
    id: str
    name: str
    source: str            # original file path this sample was enrolled from
    crop_path: str          # relative path under data/crops/
    det_score: float
    quality: dict
    created_at: str

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "crop_path": self.crop_path,
            "det_score": self.det_score,
            "quality": self.quality,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_json(d: dict) -> "SampleEntry":
        return SampleEntry(
            id=d["id"],
            name=d["name"],
            source=d["source"],
            crop_path=d["crop_path"],
            det_score=d["det_score"],
            quality=d.get("quality", {}),
            created_at=d["created_at"],
        )


@dataclass
class FaceStore:
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32))
    entries: list[SampleEntry] = field(default_factory=list)
    model_name: str = config.MODEL_NAME
    model_version: str = config.MODEL_VERSION

    # -- invariants -----------------------------------------------------

    def _check_invariants(self) -> None:
        if len(self.entries) != self.vectors.shape[0]:
            raise StoreError(
                f"Corrupt store: {len(self.entries)} manifest entries but "
                f"{self.vectors.shape[0]} embedding rows -- these must match."
            )
        if self.vectors.shape[0] > 0 and self.vectors.shape[1] != config.EMBEDDING_DIM:
            raise StoreError(
                f"Corrupt store: embeddings have dim {self.vectors.shape[1]}, "
                f"expected {config.EMBEDDING_DIM}."
            )

    # -- load / save ------------------------------------------------------

    @classmethod
    def load(cls) -> "FaceStore":
        """Load the store from disk, or return an empty store if none exists yet."""
        if not config.MANIFEST_PATH.exists() or not config.EMBEDDINGS_PATH.exists():
            return cls()

        with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        npz = np.load(config.EMBEDDINGS_PATH)
        vectors = npz["vectors"].astype(np.float32)

        entries = [SampleEntry.from_json(e) for e in manifest.get("entries", [])]
        model = manifest.get("model", {})
        store = cls(
            vectors=vectors,
            entries=entries,
            model_name=model.get("name", config.MODEL_NAME),
            model_version=model.get("version", config.MODEL_VERSION),
        )
        store._check_invariants()

        # Vectors from a different model are mathematically incomparable --
        # refuse to silently match across them.
        if store.model_name != config.MODEL_NAME or store.model_version != config.MODEL_VERSION:
            raise StoreError(
                f"Store was built with model {store.model_name} v{store.model_version}, "
                f"but the configured model is {config.MODEL_NAME} v{config.MODEL_VERSION}. "
                "Re-enroll (crops are kept for exactly this reason) or update config.py."
            )
        return store

    def save(self) -> None:
        """Persist the store to disk (embeddings.npz + manifest.json)."""
        self._check_invariants()
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

        np.savez(config.EMBEDDINGS_PATH, vectors=self.vectors.astype(np.float32))

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model": {"name": self.model_name, "version": self.model_version},
            "entries": [e.to_json() for e in self.entries],
        }
        with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    # -- mutation ---------------------------------------------------------

    def add(
        self,
        name: str,
        embedding: np.ndarray,
        crop: np.ndarray,
        source: str,
        det_score: float,
        quality: dict,
    ) -> SampleEntry:
        """Add one sample, saving its crop to data/crops/. Does not write to disk
        by itself -- call save() when done adding (enrollment adds many at once)."""
        sample_id = uuid.uuid4().hex
        crop_rel = f"{sample_id}.png"
        save_bgr(config.CROPS_DIR / crop_rel, crop)

        entry = SampleEntry(
            id=sample_id,
            name=name,
            source=source,
            crop_path=crop_rel,
            det_score=det_score,
            quality=quality,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        vec = embedding.astype(np.float32).reshape(1, -1)
        self.vectors = np.vstack([self.vectors, vec]) if self.vectors.shape[0] else vec
        self.entries.append(entry)
        return entry

    def remove(self, name: str) -> int:
        """Remove all samples for an identity. Returns the number removed.
        Also deletes the corresponding crop files. Call save() to persist."""
        keep_mask = np.array([e.name != name for e in self.entries], dtype=bool)
        removed = [e for e in self.entries if e.name == name]

        for e in removed:
            crop_file = config.CROPS_DIR / e.crop_path
            if crop_file.exists():
                crop_file.unlink()

        self.entries = [e for e in self.entries if e.name != name]
        self.vectors = self.vectors[keep_mask] if self.vectors.shape[0] else self.vectors
        return len(removed)

    # -- queries ------------------------------------------------------------

    def names(self) -> list[str]:
        """Distinct enrolled identity names, in first-seen order."""
        seen: dict[str, None] = {}
        for e in self.entries:
            seen.setdefault(e.name, None)
        return list(seen.keys())

    def sample_count(self, name: str) -> int:
        return sum(1 for e in self.entries if e.name == name)

    def is_empty(self) -> bool:
        return self.vectors.shape[0] == 0

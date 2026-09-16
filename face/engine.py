"""
The face detection + alignment + embedding engine.

Wraps insightface's FaceAnalysis behind a narrow, stable interface
(DetectedFace) so nothing else in this package -- or in the future web
app -- depends on insightface's own types. If insightface's API changes,
only this file needs to change.

Loaded once as a process-wide singleton via get_engine(): model load takes
several seconds, so re-loading per call would make batch enrollment or a
web server unusable.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from face import config


@dataclass
class DetectedFace:
    """One detected face, aligned and embedded."""

    bbox: np.ndarray        # (4,) float: x1, y1, x2, y2
    kps: np.ndarray         # (5, 2) float: left eye, right eye, nose, mouth-left, mouth-right
    det_score: float
    embedding: np.ndarray   # (512,) float32, L2-normalized
    crop: np.ndarray        # (112, 112, 3) uint8 BGR, aligned face


class FaceEngine:
    """Thin, stable wrapper around insightface.app.FaceAnalysis."""

    def __init__(self, model_name: str = config.MODEL_NAME) -> None:
        # Imported lazily so importing face.engine doesn't require
        # insightface to be installed unless the engine is actually used
        # (keeps e.g. `face.config` importable in lightweight contexts).
        from insightface.app import FaceAnalysis

        self.model_name = model_name
        self._app = FaceAnalysis(name=model_name)
        # ctx_id=-1 forces CPU. This is a laptop/desktop tool, not a GPU
        # server -- CPU inference (~150-300ms/image) is fine for this phase.
        self._app.prepare(ctx_id=-1)

    def detect(self, bgr_image: np.ndarray) -> list[DetectedFace]:
        """Detect, align, and embed every face in a BGR image."""
        faces = self._app.get(bgr_image)
        results: list[DetectedFace] = []
        for f in faces:
            embedding = self._normalized_embedding(f)
            crop = self._aligned_crop(f, bgr_image)
            results.append(
                DetectedFace(
                    bbox=np.asarray(f.bbox, dtype=np.float32),
                    kps=np.asarray(f.kps, dtype=np.float32),
                    det_score=float(f.det_score),
                    embedding=embedding,
                    crop=crop,
                )
            )
        return results

    @staticmethod
    def _normalized_embedding(f) -> np.ndarray:
        # Prefer insightface's own normed_embedding if present; otherwise
        # normalize the raw embedding ourselves. The matcher assumes every
        # stored vector is unit-length -- never skip this.
        normed = getattr(f, "normed_embedding", None)
        if normed is not None:
            v = np.asarray(normed, dtype=np.float32)
        else:
            v = np.asarray(f.embedding, dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
        return v

    @staticmethod
    def _aligned_crop(f, bgr_image: np.ndarray) -> np.ndarray:
        # insightface's face_align module produces the same 112x112 warp
        # used internally to feed the embedder -- reuse it so the crop we
        # persist is exactly what was actually embedded.
        from insightface.utils import face_align

        return face_align.norm_crop(bgr_image, landmark=f.kps, image_size=112)


_engine_lock = threading.Lock()
_engine: FaceEngine | None = None


def get_engine() -> FaceEngine:
    """Return the process-wide FaceEngine singleton, creating it on first use."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = FaceEngine()
    return _engine

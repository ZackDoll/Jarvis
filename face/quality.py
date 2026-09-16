"""
Enrollment-time quality gates.

A bad enrollment image poisons recognition permanently -- a blurry, dark,
or off-angle sample sits in the store forever, pulling matches astray. So
images are rejected here, at enroll time, with a human-readable reason,
rather than silently accepted and dealt with later.

These same reason strings are what a future web UI shows as live capture
feedback ("Too dark", "Hold still") -- the browser never needs to run its
own model to know why a frame was rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from face import config
from face.engine import DetectedFace


@dataclass
class QualityResult:
    ok: bool
    reason: str = ""          # human-readable rejection reason, "" if ok
    metrics: dict = field(default_factory=dict)


def _face_width(bbox: np.ndarray) -> float:
    return float(bbox[2] - bbox[0])


def _blur_variance(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _yaw_ratio(kps: np.ndarray) -> float:
    """Rough yaw estimate: nose offset from the eye-midpoint, normalized
    by inter-eye distance. ~0 = facing camera, larger = turned away."""
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid = (left_eye + right_eye) / 2.0
    inter_eye = float(np.linalg.norm(right_eye - left_eye))
    if inter_eye < 1e-6:
        return 1.0  # degenerate landmarks -- treat as maximally bad
    offset = float(abs(nose[0] - eye_mid[0]))
    return offset / inter_eye


def check_frame(faces: list[DetectedFace]) -> QualityResult:
    """Run all quality gates against the detection result for one image.

    Takes the full list of detected faces (not just one) because "more
    than one face" is itself a rejection condition.
    """
    if len(faces) == 0:
        return QualityResult(ok=False, reason="No face detected")
    if len(faces) > 1:
        return QualityResult(
            ok=False,
            reason=f"Multiple faces detected ({len(faces)}) -- expected exactly one",
        )

    face = faces[0]
    metrics = {
        "det_score": face.det_score,
        "face_width_px": _face_width(face.bbox),
        "blur_variance": _blur_variance(face.crop),
        "brightness": _brightness(face.crop),
        "yaw_ratio": _yaw_ratio(face.kps),
    }

    if metrics["det_score"] < config.MIN_DET_SCORE:
        return QualityResult(False, "Low detection confidence", metrics)
    if metrics["face_width_px"] < config.MIN_FACE_WIDTH_PX:
        return QualityResult(False, "Face too small -- move closer", metrics)
    if metrics["blur_variance"] < config.MIN_BLUR_VARIANCE:
        return QualityResult(False, "Image is blurry -- hold still", metrics)
    if metrics["brightness"] < config.MIN_BRIGHTNESS:
        return QualityResult(False, "Too dark", metrics)
    if metrics["brightness"] > config.MAX_BRIGHTNESS:
        return QualityResult(False, "Too bright / overexposed", metrics)
    if metrics["yaw_ratio"] > config.MAX_YAW_RATIO:
        return QualityResult(False, "Not facing the camera -- turn towards it", metrics)

    return QualityResult(ok=True, reason="", metrics=metrics)

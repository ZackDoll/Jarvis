"""
Image loading, done the way that doesn't silently break.

Two gotchas this module exists to kill:

1. EXIF rotation -- phone photos are frequently stored "sideways" with an
   EXIF orientation tag. cv2.imdecode ignores that tag entirely, so the
   detector sees a rotated face and finds nothing. We apply the EXIF
   transpose via Pillow before any detection happens.

2. Non-ASCII paths -- cv2.imread silently returns None for paths containing
   non-ASCII characters on Windows. We never call cv2.imread directly;
   everything goes through Pillow (which handles paths fine) and is then
   converted to a BGR ndarray for OpenCV/insightface to consume.

Every entry point in this package that needs pixels from disk should call
load_bgr() -- nothing should call cv2.imread directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps


class ImageLoadError(Exception):
    """Raised when an image file can't be read or decoded."""


def load_bgr(path: Union[str, Path]) -> np.ndarray:
    """Load an image file as a BGR uint8 ndarray (OpenCV convention).

    Applies EXIF-based auto-rotation and handles non-ASCII paths correctly.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"File does not exist: {path}")

    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)  # normalize orientation
            img = img.convert("RGB")
            rgb = np.array(img)
    except Exception as exc:  # Pillow raises a variety of exception types
        raise ImageLoadError(f"Could not decode image: {path} ({exc})") from exc

    # RGB (Pillow/numpy) -> BGR (OpenCV/insightface convention)
    bgr = rgb[:, :, ::-1].copy()
    return bgr


def save_bgr(path: Union[str, Path], bgr: np.ndarray) -> None:
    """Save a BGR uint8 ndarray to disk as an image (PNG by extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = bgr[:, :, ::-1]
    Image.fromarray(rgb).save(path)

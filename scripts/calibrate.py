"""
Threshold calibration.

Don't ship a guessed THRESHOLD/MARGIN_MIN -- published cosine thresholds
for ArcFace vary by model pack, and the right value depends on your
camera and lighting. This script computes every pairwise similarity score
across a folder of test images and reports the genuine (same-person) vs.
impostor (different-person) score distributions, plus false-accept /
false-reject rates swept across a range of candidate thresholds.

Usage:
    python scripts/calibrate.py ./photos
    (expects photos/<name>/*.jpg, i.e. one subfolder per person)

The gap between the two distributions is the real output of this phase.
A wide, clean separation means the system is trustworthy; heavy overlap
means more or better enrollment images are needed before wrapping any of
this in a web app.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face.engine import get_engine
from face.io import ImageLoadError, load_bgr
from face.quality import check_frame


def collect_embeddings(photos_dir: Path) -> dict[str, list[np.ndarray]]:
    """photos_dir/<name>/*.jpg -> {name: [embedding, ...]}, skipping
    images that fail detection or quality gates."""
    engine = get_engine()
    by_name: dict[str, list[np.ndarray]] = {}

    person_dirs = sorted(p for p in photos_dir.iterdir() if p.is_dir())
    if not person_dirs:
        print(f"No subdirectories found under {photos_dir} -- expected photos/<name>/*.jpg")
        return by_name

    for person_dir in person_dirs:
        name = person_dir.name
        images = sorted(
            p for p in person_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        embeddings: list[np.ndarray] = []
        for img_path in images:
            try:
                bgr = load_bgr(img_path)
            except ImageLoadError as exc:
                print(f"  [SKIP] {img_path}: {exc}")
                continue
            faces = engine.detect(bgr)
            result = check_frame(faces)
            if not result.ok:
                print(f"  [SKIP] {img_path}: {result.reason}")
                continue
            embeddings.append(faces[0].embedding)

        print(f"{name}: {len(embeddings)}/{len(images)} usable images")
        if embeddings:
            by_name[name] = embeddings

    return by_name


def pairwise_scores(by_name: dict[str, list[np.ndarray]]) -> tuple[list[float], list[float]]:
    """Returns (genuine_scores, impostor_scores) -- all pairwise cosine
    similarities within the same identity, and across different identities."""
    genuine: list[float] = []
    impostor: list[float] = []

    names = list(by_name.keys())

    # genuine: all pairs within each identity
    for name in names:
        vecs = by_name[name]
        for a, b in combinations(vecs, 2):
            genuine.append(float(np.dot(a, b)))

    # impostor: all pairs across different identities
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            for a in by_name[name_a]:
                for b in by_name[name_b]:
                    impostor.append(float(np.dot(a, b)))

    return genuine, impostor


def summarize(label: str, scores: list[float]) -> None:
    if not scores:
        print(f"{label}: no pairs")
        return
    arr = np.array(scores)
    print(
        f"{label}: n={len(arr)}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
        f"min={arr.min():.4f}  max={arr.max():.4f}"
    )


def sweep_thresholds(genuine: list[float], impostor: list[float]) -> None:
    print("\nThreshold sweep (FAR = false-accept rate, FRR = false-reject rate):")
    print(f"{'threshold':>10} {'FAR':>8} {'FRR':>8}")
    g = np.array(genuine)
    i = np.array(impostor)
    for t in np.arange(0.30, 0.71, 0.02):
        far = float((i >= t).mean()) if len(i) else float("nan")
        frr = float((g < t).mean()) if len(g) else float("nan")
        print(f"{t:>10.2f} {far:>8.3f} {frr:>8.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate match threshold/margin")
    parser.add_argument("photos_dir", nargs="?", default="photos", help="Folder of photos/<name>/*.jpg")
    args = parser.parse_args()

    photos_dir = Path(args.photos_dir)
    if not photos_dir.exists():
        print(f"Directory not found: {photos_dir}")
        print("Expected layout: photos/<name>/*.jpg  (one subfolder per person)")
        return 1

    by_name = collect_embeddings(photos_dir)
    if len(by_name) < 2:
        print(
            "\nNeed at least 2 identities with usable images to compute an "
            "impostor distribution. Add more people/photos and re-run."
        )
        return 1

    genuine, impostor = pairwise_scores(by_name)
    print()
    summarize("Genuine (same person)", genuine)
    summarize("Impostor (different people)", impostor)

    if genuine and impostor:
        gap = min(genuine) - max(impostor)
        print(f"\nSeparation (min genuine - max impostor): {gap:+.4f}")
        print("  (positive and large = clean separation; negative = overlapping, unreliable)")

    sweep_thresholds(genuine, impostor)
    return 0


if __name__ == "__main__":
    sys.exit(main())

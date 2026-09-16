"""
Command-line driver for the face recognition core.

    python -m face.cli enroll --name "Zack" ./photos/zack/*.jpg
    python -m face.cli recognize ./test.jpg [--json]
    python -m face.cli compare a.jpg b.jpg
    python -m face.cli inspect ./img.jpg
    python -m face.cli list
    python -m face.cli remove --name "Zack"

This module is a thin driver, not the product: it only calls into
engine / quality / matcher / store, and contains no recognition logic of
its own. When the web app arrives, FastAPI imports those same modules
directly and this CLI stays as-is for offline testing and calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from face import config
from face.engine import get_engine
from face.io import ImageLoadError, load_bgr
from face.matcher import query
from face.quality import check_frame
from face.store import FaceStore


def _load_and_detect(path: str):
    """Load an image and run detection. Returns (bgr, faces) or (None, None)
    on load failure, printing a message."""
    try:
        bgr = load_bgr(path)
    except ImageLoadError as exc:
        print(f"  [SKIP] {path}: {exc}")
        return None, None
    faces = get_engine().detect(bgr)
    return bgr, faces


def cmd_enroll(args: argparse.Namespace) -> int:
    store = FaceStore.load()
    accepted = 0
    rejected = 0

    for path in args.images:
        bgr, faces = _load_and_detect(path)
        if bgr is None:
            rejected += 1
            continue

        result = check_frame(faces)
        if not result.ok:
            print(f"  [REJECT] {path}: {result.reason}")
            rejected += 1
            continue

        face = faces[0]
        store.add(
            name=args.name,
            embedding=face.embedding,
            crop=face.crop,
            source=str(Path(path).resolve()),
            det_score=face.det_score,
            quality=result.metrics,
        )
        print(f"  [OK] {path}  (det_score={face.det_score:.3f})")
        accepted += 1

    store.save()

    print(f"\nEnrolled '{args.name}': {accepted} accepted, {rejected} rejected.")
    total = store.sample_count(args.name)
    print(f"Total samples now on file for '{args.name}': {total}")
    if total < config.MIN_ENROLL_SAMPLES_WARN:
        print(
            f"  WARNING: fewer than {config.MIN_ENROLL_SAMPLES_WARN} samples -- "
            "accuracy will suffer. Add more images across varied angles/lighting."
        )
    return 0 if accepted > 0 else 1


def cmd_recognize(args: argparse.Namespace) -> int:
    bgr, faces = _load_and_detect(args.image)
    if bgr is None:
        return 1

    result = check_frame(faces)
    store = FaceStore.load()

    if not result.ok:
        payload = {"decision": "low_quality" if faces else "no_face", "reason": result.reason}
        _emit(payload, args.json)
        return 1

    face = faces[0]
    match = query(store, face.embedding)

    payload = {
        "decision": match.decision,
        "best": {"name": match.best.name, "score": match.best.score} if match.best else None,
        "margin": match.margin,
        "top": [{"name": c.name, "score": c.score} for c in match.top],
        "det_score": face.det_score,
    }
    _emit(payload, args.json)
    return 0 if match.decision == "match" else 1


def cmd_compare(args: argparse.Namespace) -> int:
    from face.matcher import compare

    _, faces_a = _load_and_detect(args.image_a)
    _, faces_b = _load_and_detect(args.image_b)
    if faces_a is None or faces_b is None:
        return 1

    ra, rb = check_frame(faces_a), check_frame(faces_b)
    if not ra.ok:
        print(f"[{args.image_a}] rejected: {ra.reason}")
        return 1
    if not rb.ok:
        print(f"[{args.image_b}] rejected: {rb.reason}")
        return 1

    score = compare(faces_a[0].embedding, faces_b[0].embedding)
    print(f"Cosine similarity: {score:.4f}")
    print(f"  (current THRESHOLD = {config.THRESHOLD})")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    bgr, faces = _load_and_detect(args.image)
    if bgr is None:
        return 1

    print(f"Detected {len(faces)} face(s) in {args.image}")
    result = check_frame(faces)
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"  quality: {'OK' if result.ok else 'REJECT - ' + result.reason}")
    return 0 if result.ok else 1


def cmd_list(args: argparse.Namespace) -> int:
    store = FaceStore.load()
    if store.is_empty():
        print("No identities enrolled yet.")
        return 0
    print(f"{len(store.names())} identity(ies), {store.vectors.shape[0]} total samples:")
    for name in store.names():
        print(f"  {name}: {store.sample_count(name)} sample(s)")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    store = FaceStore.load()
    removed = store.remove(args.name)
    store.save()
    print(f"Removed {removed} sample(s) for '{args.name}'.")
    return 0 if removed > 0 else 1


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    decision = payload["decision"]
    print(f"Decision: {decision}")
    if payload.get("best"):
        print(f"  Best match: {payload['best']['name']} (score={payload['best']['score']:.4f})")
        print(f"  Margin: {payload['margin']:.4f}")
    if payload.get("top"):
        print("  Top candidates:")
        for c in payload["top"]:
            print(f"    {c['name']}: {c['score']:.4f}")
    if payload.get("reason"):
        print(f"  Reason: {payload['reason']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m face.cli", description="Face recognition core CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll", help="Enroll one or more images for an identity")
    p_enroll.add_argument("--name", required=True, help="Identity name")
    p_enroll.add_argument("images", nargs="+", help="Image file paths")
    p_enroll.set_defaults(func=cmd_enroll)

    p_recognize = sub.add_parser("recognize", help="Recognize a face against the enrolled store")
    p_recognize.add_argument("image", help="Image file path")
    p_recognize.add_argument("--json", action="store_true", help="Output JSON")
    p_recognize.set_defaults(func=cmd_recognize)

    p_compare = sub.add_parser("compare", help="Compare two images directly (no store)")
    p_compare.add_argument("image_a")
    p_compare.add_argument("image_b")
    p_compare.set_defaults(func=cmd_compare)

    p_inspect = sub.add_parser("inspect", help="Show detection + quality metrics for one image")
    p_inspect.add_argument("image")
    p_inspect.set_defaults(func=cmd_inspect)

    p_list = sub.add_parser("list", help="List enrolled identities")
    p_list.set_defaults(func=cmd_list)

    p_remove = sub.add_parser("remove", help="Remove an enrolled identity")
    p_remove.add_argument("--name", required=True)
    p_remove.set_defaults(func=cmd_remove)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

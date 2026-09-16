"""
Minimal test harness for the face recognition core -- NOT the finalized
web app from the plan (no auth, no profiles/metadata, no persistence
strategy beyond what face/store.py already does). This exists to let you
click "enroll" / "recognize" in a browser against your own webcam instead
of shelling out image files to the CLI.

Imports face.engine / face.quality / face.store / face.matcher directly
and unchanged -- this file adds zero recognition logic of its own, only
HTTP plumbing around the same core the CLI drives.

Run (from the repo root):
    .venv/Scripts/python -m uvicorn webtest.server:app --reload --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face import config
from face.engine import get_engine
from face.matcher import query
from face.quality import check_frame
from face.store import FaceStore, StoreError

app = FastAPI(title="Jarvis Face Recognition -- Test Harness")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# The store is mutated by enroll/remove and read by recognize/identities.
# A single lock keeps concurrent requests (e.g. two browser tabs) from
# corrupting the in-memory state or racing on the on-disk files.
_store_lock = threading.Lock()


def _decode_upload(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes (JPEG from canvas.toBlob) into a BGR
    ndarray. Uses cv2.imdecode directly (not face.io.load_bgr) since this
    is an in-memory buffer, not a file path -- the EXIF/unicode-path
    gotchas that io.py exists for don't apply to a browser-captured frame."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode uploaded image")
    return bgr


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    with _store_lock:
        store = FaceStore.load()
        return {
            "model_name": config.MODEL_NAME,
            "model_version": config.MODEL_VERSION,
            "threshold": config.THRESHOLD,
            "margin_min": config.MARGIN_MIN,
            "identities": len(store.names()),
            "samples": int(store.vectors.shape[0]),
        }


@app.get("/api/identities")
def list_identities():
    with _store_lock:
        store = FaceStore.load()
        return {
            "identities": [
                {"name": n, "samples": store.sample_count(n)} for n in store.names()
            ]
        }


@app.delete("/api/identities/{name}")
def remove_identity(name: str):
    with _store_lock:
        store = FaceStore.load()
        removed = store.remove(name)
        store.save()
        return {"removed": removed}


@app.post("/api/enroll")
async def enroll(name: str = Form(...), image: UploadFile = File(...)):
    name = name.strip()
    if not name:
        return JSONResponse({"ok": False, "reason": "Name is required"}, status_code=400)

    try:
        bgr = _decode_upload(await image.read())
    except ValueError as exc:
        return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)

    faces = get_engine().detect(bgr)
    result = check_frame(faces)
    if not result.ok:
        return {"ok": False, "reason": result.reason, "metrics": result.metrics}

    face = faces[0]
    with _store_lock:
        try:
            store = FaceStore.load()
        except StoreError as exc:
            return JSONResponse({"ok": False, "reason": str(exc)}, status_code=500)
        store.add(
            name=name,
            embedding=face.embedding,
            crop=face.crop,
            source="webtest-upload",
            det_score=face.det_score,
            quality=result.metrics,
        )
        store.save()
        total = store.sample_count(name)

    return {
        "ok": True,
        "reason": "",
        "metrics": result.metrics,
        "total_samples": total,
        "warn_low_samples": total < config.MIN_ENROLL_SAMPLES_WARN,
    }


@app.post("/api/recognize")
async def recognize(image: UploadFile = File(...)):
    try:
        bgr = _decode_upload(await image.read())
    except ValueError as exc:
        return JSONResponse({"decision": "error", "reason": str(exc)}, status_code=400)

    faces = get_engine().detect(bgr)
    result = check_frame(faces)
    if not result.ok:
        decision = "low_quality" if faces else "no_face"
        return {"decision": decision, "reason": result.reason, "metrics": result.metrics}

    face = faces[0]
    with _store_lock:
        try:
            store = FaceStore.load()
        except StoreError as exc:
            return JSONResponse({"decision": "error", "reason": str(exc)}, status_code=500)
        match = query(store, face.embedding)

    return {
        "decision": match.decision,
        "best": {"name": match.best.name, "score": match.best.score} if match.best else None,
        "margin": match.margin,
        "top": [{"name": c.name, "score": c.score} for c in match.top],
        "det_score": face.det_score,
        "metrics": result.metrics,
    }


@app.on_event("startup")
def _warm_up_model():
    # Load the model at server startup rather than on the first request,
    # so the first browser action isn't the one eating the multi-second
    # model load.
    get_engine()

# Jarvis
Central facial recognition system for apartment profile tracking

## Setup

```
py -3.13 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

First run downloads the `buffalo_l` model pack (~280MB) to
`~/.insightface/models/`.

## Usage

```
python -m face.cli enroll --name "Zack" photos\zack\*.jpg
python -m face.cli recognize test.jpg
python -m face.cli recognize test.jpg --json
python -m face.cli compare a.jpg b.jpg
python -m face.cli inspect img.jpg
python -m face.cli list
python -m face.cli remove --name "Zack"
```

## Calibration

Before trusting any of this, tune the match threshold against your own
camera and lighting:

```
# photos/<name>/*.jpg, one folder per person, 5+ images each
python scripts/calibrate.py photos
```

This prints genuine-vs-impostor score distributions and a false-accept /
false-reject sweep. Update `THRESHOLD` / `MARGIN_MIN` in `face/config.py`
based on the results.

## Layout

```
face/
  config.py    tunables and paths
  io.py        EXIF-safe, unicode-safe image loading
  engine.py    detection + alignment + embedding (insightface wrapper)
  quality.py   enrollment-time rejection gates
  store.py     flat-file persistence (embeddings.npz + manifest.json + crops)
  matcher.py   cosine similarity matching against the store
  cli.py       command-line driver
scripts/
  calibrate.py threshold calibration
data/          embeddings, manifest, aligned crops — gitignored, never commit
photos/        your own test images — gitignored
```

## Status

Phase 1 (this): recognition core, CLI-driven, flat-file storage. Verified
end-to-end on live webcam captures.

Deferred to later phases: profiles with attached metadata, the web app
(FastAPI + React, importing this package unchanged), liveness/anti-spoofing
(a photo of a photo currently *will* match), door hardware integration.

`data/` holds biometric data (face embeddings, aligned crops) and must
never be committed — it's irrevocable if leaked, unlike a password.

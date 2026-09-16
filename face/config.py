"""
Central configuration for the face recognition core.

Nothing in engine.py / quality.py / matcher.py / store.py reads paths or
tunables from anywhere else -- change behavior here, not scattered through
the package.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CROPS_DIR = DATA_DIR / "crops"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "buffalo_l"
MODEL_VERSION = "1"  # bump manually if you re-download / change model files
EMBEDDING_DIM = 512

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
# Cosine similarity threshold an identity's best sample must clear to count
# as a match at all. These are starting points -- see scripts/calibrate.py.
THRESHOLD = 0.50

# The best score must beat the best score belonging to any OTHER identity
# by at least this much, or the match is rejected as ambiguous.
MARGIN_MIN = 0.05

# How many ranked candidates to report back on every query, for diagnosis.
TOP_K = 3

# ---------------------------------------------------------------------------
# Quality gates (enrollment-time)
# ---------------------------------------------------------------------------
MIN_DET_SCORE = 0.6
MIN_FACE_WIDTH_PX = 110
MIN_BLUR_VARIANCE = 60.0       # variance of Laplacian, higher = sharper
MIN_BRIGHTNESS = 50.0          # mean luma 0-255
MAX_BRIGHTNESS = 200.0
MAX_YAW_RATIO = 0.6            # nose offset from eye-midpoint / inter-eye distance

# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------
MIN_ENROLL_SAMPLES_WARN = 5    # warn (not block) if fewer accepted images than this

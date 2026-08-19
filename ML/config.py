"""Shared paths, capture settings and dataset targets for the ESLT pipeline."""
from pathlib import Path

# --- paths -------------------------------------------------------------
# Data lives outside the repo: video is far too large to commit.
DATA_ROOT = Path(r"C:\Users\habte\OneDrive\Desktop\ESLT-data")
RAW_DIR = DATA_ROOT / "raw"
LANDMARK_DIR = DATA_ROOT / "landmarks"
MANIFEST_DIR = DATA_ROOT / "manifests"
CLIPS_CSV = MANIFEST_DIR / "clips.csv"

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_MD = REPO_ROOT / "docs" / "vocabulary_list.md"

# --- capture -----------------------------------------------------------
FPS = 30
CLIP_SECONDS = 2.0
FRAMES_PER_CLIP = 30          # sequence length fed to the model
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
COUNTDOWN_SECONDS = 2

# --- landmarks ---------------------------------------------------------
# MediaPipe Holistic: 33 pose + 21 left hand + 21 right hand + 468 face
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FACE_LANDMARKS = 468
USE_FACE = False              # face adds 1404 dims for little gain on lexical signs
FEATURE_DIM = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 * 2  # 132 + 126 = 258

# --- dataset targets ---------------------------------------------------
# Calibrated against published EthSL work: 20 words x ~280 clips/word over
# 7 signers reached 94% signer-dependent but only 73% signer-invariant.
# Signer count is what buys generalisation, not clips per signer.
TAKES_PER_GLOSS_PER_SIGNER = 25
PILOT_SIGNERS = 3             # enough to prove the pipeline end to end
TARGET_SIGNERS = 10           # enough for a signer-invariant v1 model
MIN_SIGNERS_FOR_HELDOUT = 5   # below this, a held-out-signer split is meaningless

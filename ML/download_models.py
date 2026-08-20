"""Fetch the MediaPipe Tasks model bundles needed for landmark extraction.

MediaPipe >= 0.10.3x removed the legacy `mp.solutions.holistic` API that most
published EthSL work uses. The replacement is the Tasks API, which needs these
.task bundles on disk. Run once before extract_landmarks.py.
"""
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "holistic_landmarker.task":
        f"{BASE}/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
    "hand_landmarker.task":
        f"{BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_full.task":
        f"{BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}


def download(name: str, url: str) -> bool:
    dest = MODEL_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [have] {name} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        print(f"  [get ] {name} <- {url}")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(dest)
        print(f"  [ok  ] {name} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print(f"model dir: {MODEL_DIR}")
    ok = all([download(n, u) for n, u in MODELS.items()])
    if not ok:
        print(
            "\nSome downloads failed. If this machine blocks storage.googleapis.com,\n"
            "download the .task files on any machine that can reach it and copy them\n"
            "into the model dir above. URLs are listed in MODELS in this file."
        )
        sys.exit(1)

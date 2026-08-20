"""Fetch the MediaPipe Tasks model bundles needed for landmark extraction.

MediaPipe >= 0.10.3x removed the legacy `mp.solutions.holistic` API that most
published EthSL work uses. The replacement is the Tasks API, which needs these
.task bundles on disk. Run once before extract_landmarks.py.

The bundles are zip archives, so a truncated download is detectable. That
matters: on a flaky link the stream ends early with no error, and the corruption
surfaces much later as "Unable to open zip archive" during extraction. This
resumes with HTTP Range and only accepts a file that opens as a zip.

    python ML/download_models.py
    python ML/download_models.py --attempts 50
"""
import argparse
import sys
import urllib.error
import urllib.request
import zipfile
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


def valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def fetch(name: str, url: str, attempts: int) -> bool:
    dest = MODEL_DIR / name
    if valid(dest):
        print(f"  [have] {name} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    if dest.exists():
        print(f"  [bad ] {name} is truncated, resuming")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and not valid(dest):
        dest.replace(part)          # reuse the bytes we already have

    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                # A server ignoring Range restarts the file; do not append to it.
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(part, mode) as f:
                    while chunk := r.read(1 << 18):
                        f.write(chunk)
                        have += len(chunk)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"  [{attempt:>3} ] {name}: {type(e).__name__} at {have/1e6:.1f} MB")

        if valid(part):
            part.replace(dest)
            print(f"  [ok  ] {name} ({dest.stat().st_size/1e6:.1f} MB, "
                  f"{attempt} attempt{'s' if attempt > 1 else ''})")
            return True

    size = part.stat().st_size if part.exists() else 0
    print(f"  [FAIL] {name}: stalled at {size/1e6:.1f} MB after {attempts} attempts")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempts", type=int, default=40,
                    help="resume attempts per file (default 40)")
    ap.add_argument("--only", default="",
                    help="comma-separated bundle names to fetch")
    args = ap.parse_args()

    wanted = [n.strip() for n in args.only.split(",") if n.strip()] or list(MODELS)
    print(f"model dir: {MODEL_DIR}")
    ok = all(fetch(n, MODELS[n], args.attempts) for n in wanted if n in MODELS)
    if not ok:
        print(
            "\nSome bundles are incomplete. If this network keeps dropping the\n"
            "connection, download them on another machine and copy them into the\n"
            "model dir above — the URLs are in MODELS in this file."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

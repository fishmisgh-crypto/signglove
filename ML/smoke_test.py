"""Verify a machine can actually record and extract, before anyone records a session.

Runs headless — no camera operator, no signing required. Every signer's laptop
should pass this before their first session, because the failure modes here are
silent: a camera that ignores the requested resolution produces empty video
files, and a truncated model bundle fails only at extraction time.

    python ML/smoke_test.py
    python ML/smoke_test.py --no-camera     # skip the camera probe
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from config import (
    CAPTURE_HEIGHT, CAPTURE_WIDTH, CLIP_SECONDS, FEATURE_DIM, FPS,
    FRAMES_PER_CLIP,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = Path(__file__).resolve().parent / "models"
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def report(name, status, detail=""):
    results.append((name, status))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def check_vocabulary():
    try:
        from vocabulary import load_vocabulary
        entries = load_vocabulary()
        report("vocabulary parses", PASS, f"{len(entries)} glosses")
    except Exception as e:
        report("vocabulary parses", FAIL, f"{type(e).__name__}: {e}")


def check_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        report("camera opens", FAIL, "no camera at index 0")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        report("camera reads a frame", FAIL)
        return None
    h, w = frame.shape[:2]
    cap.release()
    if (w, h) == (CAPTURE_WIDTH, CAPTURE_HEIGHT):
        report("camera resolution", PASS, f"{w}x{h}")
    else:
        report("camera resolution", WARN,
               f"gave {w}x{h}, not {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} — "
               "clips will record at the smaller size")
    return (w, h)


def check_codec(size):
    """A VideoWriter can open happily and still write an unreadable file."""
    path = Path(tempfile.gettempdir()) / "eslt_codec_probe.mp4"
    n = int(CLIP_SECONDS * FPS)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not writer.isOpened():
        report("mp4v writer opens", FAIL)
        return None
    for i in range(n):
        writer.write(np.full((size[1], size[0], 3), (i * 4) % 255, np.uint8))
    writer.release()

    cap = cv2.VideoCapture(str(path))
    read = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        read += 1
    cap.release()
    if read == n:
        report("mp4 round-trip", PASS, f"{read}/{n} frames at {size[0]}x{size[1]}")
        return path
    report("mp4 round-trip", FAIL, f"wrote {n} frames, read back {read}")
    path.unlink(missing_ok=True)
    return None


def check_model_bundles():
    """Truncated .task files are the most common silent failure here."""
    import zipfile
    ok_any = False
    for name in ("holistic_landmarker.task", "hand_landmarker.task"):
        p = MODEL_DIR / name
        if not p.exists():
            report(f"model {name}", FAIL, "missing — run download_models.py")
        elif not zipfile.is_zipfile(p):
            report(f"model {name}", FAIL,
                   f"truncated ({p.stat().st_size} bytes) — delete and re-download")
        else:
            report(f"model {name}", PASS, f"{p.stat().st_size/1e6:.1f} MB")
            ok_any = True
    return ok_any


def check_extraction(clip_path):
    """Run the real Tasks API over a real video file and check the array shape."""
    import zipfile
    holistic = MODEL_DIR / "holistic_landmarker.task"
    if not (holistic.exists() and zipfile.is_zipfile(holistic)):
        report("landmark extraction", WARN,
               "skipped — holistic bundle not usable yet")
        return
    try:
        import extract_landmarks as ex
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        options = vision.HolisticLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(holistic)),
            running_mode=vision.RunningMode.VIDEO)
        with vision.HolisticLandmarker.create_from_options(options) as lm:
            seq, coverage, n = ex.extract_clip(lm, clip_path)
        if seq is None:
            report("landmark extraction", FAIL, "no frames decoded")
        elif seq.shape != (FRAMES_PER_CLIP, FEATURE_DIM):
            report("landmark extraction", FAIL,
                   f"shape {seq.shape}, expected ({FRAMES_PER_CLIP}, {FEATURE_DIM})")
        else:
            report("landmark extraction", PASS,
                   f"{seq.shape} from {n} frames "
                   f"(synthetic clip, so 0% hand coverage is expected)")
    except Exception as e:
        report("landmark extraction", FAIL, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-camera", action="store_true")
    args = ap.parse_args()

    print("ESLT pre-flight\n")
    check_vocabulary()
    size = None if args.no_camera else check_camera()
    clip = check_codec(size or (CAPTURE_WIDTH, CAPTURE_HEIGHT))
    check_model_bundles()
    if clip:
        check_extraction(clip)
        clip.unlink(missing_ok=True)

    failed = [n for n, s in results if s == FAIL]
    warned = [n for n, s in results if s == WARN]
    print(f"\n{len(results) - len(failed) - len(warned)} passed, "
          f"{len(warned)} warned, {len(failed)} failed")
    if failed:
        print("this machine is not ready to record: " + ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

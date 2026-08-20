"""Stage 1 — cleaning and feature extraction.

Walks the clip tree, runs MediaPipe Holistic over every video, and writes one
normalised (FRAMES_PER_CLIP, FEATURE_DIM) array per clip plus a manifest.

Cleaning rules, all recorded in the manifest so nothing is dropped silently:
  - a clip with no hand detected in any frame is rejected (framing/lighting)
  - a clip with hands in fewer than --min-coverage of frames is flagged
  - a gloss with zero usable clips is excluded from the label set

Landmarks are normalised per frame: translated so the shoulder midpoint is the
origin, then scaled by shoulder width. Without this the model keys on where the
signer stood and how close they were to the camera, which is exactly what breaks
on a new signer.

    python ML/build_dataset.py
    python ML/build_dataset.py --source external/Ethsl
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from config import DATA_ROOT, FEATURE_DIM, FRAMES_PER_CLIP, MANIFEST_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "holistic_landmarker.task"
OUT_X = DATA_ROOT / "dataset" / "X.npy"
OUT_Y = DATA_ROOT / "dataset" / "y.npy"
OUT_LABELS = DATA_ROOT / "dataset" / "labels.json"
REPORT = MANIFEST_DIR / "build_report.csv"

L_SHOULDER, R_SHOULDER = 11, 12          # MediaPipe pose indices


def frame_features(res):
    """pose(33x4) + left hand(21x3) + right hand(21x3) = 258 dims."""
    # Tasks API returns a flat list of NormalizedLandmark, not a list per person.
    if res.pose_landmarks:
        pose = np.array([[p.x, p.y, p.z, getattr(p, "visibility", 0.0)]
                         for p in res.pose_landmarks], dtype=np.float32)
    else:
        pose = np.zeros((33, 4), np.float32)

    def hand(lms):
        if lms:
            return np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
        return np.zeros((21, 3), np.float32)

    lh, rh = hand(res.left_hand_landmarks), hand(res.right_hand_landmarks)
    return pose, lh, rh


def normalise(pose, lh, rh):
    """Shoulder-centred, shoulder-width-scaled. Removes position and distance."""
    ls, rs = pose[L_SHOULDER, :2], pose[R_SHOULDER, :2]
    origin = (ls + rs) / 2.0
    width = float(np.linalg.norm(ls - rs))
    if width < 1e-6:                      # no pose this frame — leave as-is
        width = 1.0
    p = pose.copy()
    p[:, :2] = (p[:, :2] - origin) / width
    p[:, 2] = p[:, 2] / width
    out = [p.flatten()]
    for h in (lh, rh):
        h = h.copy()
        if h.any():
            h[:, :2] = (h[:, :2] - origin) / width
            h[:, 2] = h[:, 2] / width
        out.append(h.flatten())
    return np.concatenate(out)


def resample(frames, n=FRAMES_PER_CLIP):
    if not frames:
        return None
    arr = np.asarray(frames, dtype=np.float32)
    if len(arr) == n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return arr[idx]


def extract(landmarker, path):
    """Returns (sequence, hand_coverage, frame_count).

    Uses IMAGE mode, which is stateless per frame. VIDEO mode carries state
    between calls, and these clips come from different sources at different
    resolutions — the segmentation smoother asserts when consecutive frames
    change size, and its timestamps must also increase across the landmarker's
    whole lifetime rather than per clip. Stateless avoids both.
    """
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    frames, hand_hits = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(img)
        pose, lh, rh = frame_features(res)
        if lh.any() or rh.any():
            hand_hits += 1
        frames.append(normalise(pose, lh, rh))
    cap.release()
    seq = resample(frames)
    cov = hand_hits / len(frames) if frames else 0.0
    return seq, cov, len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="external/Ethsl",
                    help="clip tree relative to DATA_ROOT: <source>/<gloss>/*.mp4")
    ap.add_argument("--min-coverage", type=float, default=0.30)
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"missing {MODEL_PATH} — run ML/download_models.py")
    src = DATA_ROOT / args.source
    if not src.exists():
        raise SystemExit(f"no clip tree at {src}")

    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    options = vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_segmentation_mask=False)

    X, y, rows = [], [], []
    kept, rejected = 0, 0
    for gdir in sorted(p for p in src.iterdir() if p.is_dir()):
        gloss = gdir.name
        clips = sorted(gdir.glob("*.mp4"))
        if not clips:
            print(f"  skip  {gloss}: no clips")
            continue
        for clip in clips:
            # A fresh landmarker per clip. The holistic graph keeps a previous
            # segmentation mask internally, and these clips vary in resolution,
            # so a reused instance asserts on the first size change.
            try:
                with vision.HolisticLandmarker.create_from_options(options) as lm:
                    seq, cov, n = extract(lm, clip)
                status = "ok"
                if seq is None or seq.shape != (FRAMES_PER_CLIP, FEATURE_DIM):
                    status = "undecodable"
                elif cov == 0.0:
                    status = "no-hands"
                elif cov < args.min_coverage:
                    status = "low-coverage"
            except Exception as e:
                seq, cov, n = None, 0.0, 0
                status = type(e).__name__

            if status in ("ok", "low-coverage"):
                X.append(seq)
                y.append(gloss)
                kept += 1
            else:
                rejected += 1
            print(f"  {status:<12} {gloss:<12} {clip.name:<14} "
                  f"hands {cov:5.0%}  frames {n}")
            rows.append({"gloss": gloss, "clip": clip.name, "frames": n,
                         "hand_coverage": round(cov, 3), "status": status})

    if not X:
        raise SystemExit("no usable clips — nothing to train on")

    labels = sorted(set(y))
    yi = np.array([labels.index(g) for g in y], dtype=np.int64)
    Xa = np.stack(X).astype(np.float32)

    OUT_X.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_X, Xa)
    np.save(OUT_Y, yi)
    OUT_LABELS.write_text(json.dumps(labels, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    counts = {g: int((yi == i).sum()) for i, g in enumerate(labels)}
    print(f"\nkept {kept}   rejected {rejected}")
    print(f"X {Xa.shape}   classes {len(labels)}")
    print("per class:", counts)
    print(f"saved: {OUT_X.parent}")
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()

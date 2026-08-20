"""Turn recorded clips into fixed-length MediaPipe Holistic keypoint arrays.

Uses the MediaPipe Tasks API (HolisticLandmarker). The legacy
`mp.solutions.holistic` API used by most published EthSL work was removed in
MediaPipe 0.10.3x and does not exist on Python 3.13 — run download_models.py
once to fetch the .task bundle this needs.

Walks the raw clip tree, skips anything already extracted, and writes one
.npy of shape (FRAMES_PER_CLIP, FEATURE_DIM) per clip.

    python extract_landmarks.py
    python extract_landmarks.py --force
    python extract_landmarks.py --signer S01

Clips where neither hand is ever detected are reported as failures — those
are usually framing or lighting problems and should be re-recorded.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from config import (
    FEATURE_DIM, FRAMES_PER_CLIP, LANDMARK_DIR, MANIFEST_DIR, RAW_DIR,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUALITY_CSV = MANIFEST_DIR / "extraction_report.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "holistic_landmarker.task"


def frame_features(results):
    """Flatten one frame to pose(33x4) + left hand(21x3) + right hand(21x3)."""
    if results.pose_landmarks:
        pose = np.array([[p.x, p.y, p.z, getattr(p, "visibility", 0.0)]
                         for p in results.pose_landmarks[0]]).flatten()
    else:
        pose = np.zeros(33 * 4)

    def hand(landmarks):
        if landmarks:
            return np.array([[p.x, p.y, p.z] for p in landmarks[0]]).flatten()
        return np.zeros(21 * 3)

    return np.concatenate([pose, hand(results.left_hand_landmarks),
                           hand(results.right_hand_landmarks)])


def resample(frames, n=FRAMES_PER_CLIP):
    """Uniformly resample a variable-length sequence to exactly n frames."""
    if not frames:
        return None
    arr = np.asarray(frames, dtype=np.float32)
    if len(arr) == n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return arr[idx]


def extract_clip(landmarker, video_path):
    import mediapipe as mp
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, hand_hits, i = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        results = landmarker.detect_for_video(mp_image, int(i * 1000 / src_fps))
        i += 1
        if results.left_hand_landmarks or results.right_hand_landmarks:
            hand_hits += 1
        frames.append(frame_features(results))
    cap.release()
    seq = resample(frames)
    coverage = hand_hits / len(frames) if frames else 0.0
    return seq, coverage, len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-extract existing .npy files")
    ap.add_argument("--signer", default="", help="limit to one signer id")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="flag clips with hands visible in less than this fraction of frames")
    args = ap.parse_args()

    try:
        import mediapipe as mp
    except ImportError:
        raise SystemExit(
            "mediapipe is not installed in this interpreter.\n"
            "  .venv\\Scripts\\python.exe -m pip install -r ML/requirements.txt"
        )

    clips = sorted(RAW_DIR.rglob("take_*.mp4"))
    if args.signer:
        clips = [c for c in clips if c.parent.name == args.signer]
    if not clips:
        raise SystemExit(f"no clips found under {RAW_DIR} — record some first")

    rows, done, skipped, failed = [], 0, 0, 0
    options = vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_segmentation_mask=False,
    )
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:

        for clip in clips:
            gloss, signer = clip.parent.parent.name, clip.parent.name
            out = LANDMARK_DIR / gloss / signer / f"{clip.stem}.npy"
            if out.exists() and not args.force:
                skipped += 1
                continue

            seq, coverage, n_frames = extract_clip(landmarker, clip)
            if seq is None or seq.shape != (FRAMES_PER_CLIP, FEATURE_DIM):
                print(f"FAIL  {clip.name}  ({gloss}/{signer}) unreadable")
                failed += 1
                continue

            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, seq)
            done += 1
            flag = "LOW-HANDS" if coverage < args.min_coverage else "ok"
            if flag != "ok":
                failed += 1
                print(f"WARN  {gloss}/{signer}/{clip.name}  hands in {coverage:.0%} of frames")
            rows.append({"clip": str(clip.relative_to(RAW_DIR)), "gloss": gloss,
                         "signer_id": signer, "source_frames": n_frames,
                         "hand_coverage": round(coverage, 3), "status": flag})

    if rows:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        write_header = not QUALITY_CSV.exists()
        with QUALITY_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)

    print(f"\nextracted {done}   skipped {skipped}   needs attention {failed}")
    print(f"landmarks: {LANDMARK_DIR}")
    if rows:
        print(f"report:    {QUALITY_CSV}")


if __name__ == "__main__":
    main()

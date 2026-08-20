"""Webcam capture tool for building the ESLT clip corpus.

Records fixed-length clips of one signer performing glosses from the
vocabulary list, and appends every take to the manifest.

    python record_clips.py --signer S01
    python record_clips.py --signer S01 --glosses HELLO,WATER,HELP
    python record_clips.py --signer S01 --takes 10

Keys while running:  SPACE start take   r redo last   s skip gloss   q quit
Progress is stored on disk, so quitting and resuming later is safe.
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timezone

import cv2

from config import (
    CAPTURE_HEIGHT, CAPTURE_WIDTH, CLIPS_CSV, CLIP_SECONDS, COUNTDOWN_SECONDS,
    FPS, MANIFEST_DIR, RAW_DIR, TAKES_PER_GLOSS_PER_SIGNER,
)
from vocabulary import load_vocabulary

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANIFEST_FIELDS = [
    "clip_id", "gloss", "signer_id", "take", "path",
    "recorded_utc", "fps", "frames", "width", "height",
]


def existing_takes(gloss, signer_id):
    d = RAW_DIR / gloss / signer_id
    return sorted(d.glob("take_*.mp4")) if d.exists() else []


def append_manifest(row):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    new = not CLIPS_CSV.exists()
    with CLIPS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def open_camera(index):
    """Open the camera and report the resolution it actually negotiated.

    Cameras silently ignore a size they cannot do. Writing frames of one size
    into a VideoWriter declared at another produces empty files, so the caller
    must use the negotiated size, not the requested one.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
        print(f"note: camera gave {width}x{height}, not the requested "
              f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT} — recording at {width}x{height}")
    return cap, width, height


def overlay(frame, lines, colour=(255, 255, 255)):
    for i, text in enumerate(lines):
        y = 40 + i * 34
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    colour, 2, cv2.LINE_AA)


def record_take(cap, gloss, signer_id, take_no, size):
    """Record one clip. Returns the manifest row, or None if aborted."""
    out_dir = RAW_DIR / gloss / signer_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"take_{take_no:03d}.mp4"

    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                return None
            frame = cv2.flip(frame, 1)
            overlay(frame, [f"{gloss}  take {take_no}", f"starting in {remaining}"],
                    (0, 200, 255))
            cv2.imshow("ESLT recorder", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, size)
    target_frames = int(CLIP_SECONDS * FPS)
    written = 0
    while written < target_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        writer.write(frame)
        written += 1
        preview = frame.copy()
        overlay(preview, [f"{gloss}  take {take_no}",
                          f"REC {written}/{target_frames}"], (0, 0, 255))
        cv2.imshow("ESLT recorder", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    writer.release()

    if written < target_frames:
        path.unlink(missing_ok=True)
        return None

    return {
        "clip_id": f"{gloss}__{signer_id}__{take_no:03d}",
        "gloss": gloss,
        "signer_id": signer_id,
        "take": take_no,
        "path": str(path.relative_to(RAW_DIR.parent)),
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fps": FPS,
        "frames": written,
        "width": size[0],
        "height": size[1],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", required=True,
                    help="signer id, e.g. S01 (never a real name)")
    ap.add_argument("--glosses", default="",
                    help="comma-separated subset; default is the whole list")
    ap.add_argument("--takes", type=int, default=TAKES_PER_GLOSS_PER_SIGNER,
                    help="target takes per gloss for this signer")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    vocab = load_vocabulary()
    wanted = [g.strip().upper() for g in args.glosses.split(",") if g.strip()]
    if wanted:
        known = {e["gloss"] for e in vocab}
        unknown = [g for g in wanted if g not in known]
        if unknown:
            raise SystemExit(f"not in vocabulary list: {unknown}")
        vocab = [e for e in vocab if e["gloss"] in wanted]

    cap, cam_w, cam_h = open_camera(args.camera)
    size = (cam_w, cam_h)
    recorded = 0
    try:
        for entry in vocab:
            gloss = entry["gloss"]
            done = len(existing_takes(gloss, args.signer))
            if done >= args.takes:
                print(f"{gloss}: already has {done} takes, skipping")
                continue
            # Amharic cannot be drawn by cv2.putText, so show it in the console.
            print(f"\n{gloss}  ({entry['amharic']})  — {done}/{args.takes} takes")

            while done < args.takes:
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("camera read failed")
                frame = cv2.flip(frame, 1)
                overlay(frame, [
                    gloss,
                    f"signer {args.signer}   {done}/{args.takes} takes",
                    "SPACE record    s skip gloss    q quit",
                ], (0, 255, 120))
                cv2.imshow("ESLT recorder", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    return
                if key == ord("s"):
                    break
                if key == ord(" "):
                    row = record_take(cap, gloss, args.signer, done + 1, size)
                    if row:
                        append_manifest(row)
                        recorded += 1
                        done += 1
                        print(f"  saved {row['clip_id']}")
                    else:
                        print("  take aborted, not saved")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nrecorded {recorded} new clips this session")
        print(f"manifest: {CLIPS_CSV}")


if __name__ == "__main__":
    main()

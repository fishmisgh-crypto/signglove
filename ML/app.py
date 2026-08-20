"""Stage 3 — web demo.

The browser records a short clip from the webcam and posts it here. The server
runs the same MediaPipe Holistic extraction and the same pooled features used in
training, then returns the predicted gloss with its Amharic text.

Inference runs server-side deliberately: it reuses the exact preprocessing from
build_dataset.py, so what you see in the browser is what the model was actually
trained on. A browser-side port would be a second implementation to keep in sync.

    python ML/app.py
    then open http://127.0.0.1:5000
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from build_dataset import FRAMES_PER_CLIP, extract
from config import DATA_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "holistic_landmarker.task"
MODEL_JSON = DATA_ROOT / "dataset" / "model.json"

app = Flask(__name__, static_folder=str(HERE / "web"), static_url_path="")

M = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
W = np.array(M["W"], np.float32)
b = np.array(M["b"], np.float32)
MU = np.array(M["mean"], np.float32)
SD = np.array(M["std"], np.float32)
LABELS = M["labels"]
AMH = M["amharic"]


def featurise(X):
    d = np.diff(X, axis=1)
    return np.concatenate([
        X.mean(1), X.std(1), X[:, 0], X[:, -1], np.abs(d).mean(1)
    ], axis=1).astype(np.float32)


def predict(seq):
    f = (featurise(seq[None]) - MU) / SD
    z = f @ W + b
    z = z - z.max()
    p = np.exp(z) / np.exp(z).sum()
    order = p[0].argsort()[::-1][:3]
    return [{"gloss": LABELS[i], "amharic": AMH.get(LABELS[i], LABELS[i]),
             "confidence": round(float(p[0][i]), 4)} for i in order]


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/info")
def info():
    return jsonify({
        "classes": len(LABELS),
        "labels": LABELS,
        "amharic": AMH,
        "loo_top1": M.get("loo_top1"),
        "loo_top3": M.get("loo_top3"),
        "n_train": M.get("n_train"),
        "single_clip_classes": M.get("single_clip_classes", []),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "clip" not in request.files:
        return jsonify({"error": "no clip uploaded"}), 400

    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    suffix = Path(request.files["clip"].filename or "clip.webm").suffix or ".webm"
    tmp = Path(tempfile.mkdtemp()) / f"upload{suffix}"
    request.files["clip"].save(str(tmp))

    cap = cv2.VideoCapture(str(tmp))
    ok = cap.isOpened()
    cap.release()
    if not ok:
        return jsonify({"error": "could not decode the recording"}), 400

    options = vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_segmentation_mask=False)
    try:
        with vision.HolisticLandmarker.create_from_options(options) as lm:
            seq, cov, n = extract(lm, tmp)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    finally:
        tmp.unlink(missing_ok=True)

    if seq is None or seq.shape[0] != FRAMES_PER_CLIP:
        return jsonify({"error": "no usable frames in the recording"}), 400
    if cov == 0.0:
        return jsonify({"error": "no hands detected — check framing and lighting",
                        "hand_coverage": 0.0}), 200

    return jsonify({"predictions": predict(seq),
                    "hand_coverage": round(float(cov), 3),
                    "frames": int(n)})


if __name__ == "__main__":
    if not MODEL_JSON.exists():
        raise SystemExit(f"missing {MODEL_JSON} — run ML/train.py first")
    print(f"model: {len(LABELS)} classes, leave-one-out top-1 "
          f"{M.get('loo_top1', 0):.1%}")
    print("open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

"""Stage 2 — train the classifier and report honest accuracy.

The dataset is tiny (about two clips per class), so the choices here are driven
by that rather than by what would be best with real data:

  - Features are pooled over time (mean, std, first, last, and mean of frame
    deltas) instead of fed to an RNN. A sequence model has far too many
    parameters for this much data and would memorise it outright.
  - The model is L2-regularised softmax regression, trained in numpy. Small,
    convex, and exports to JSON for the browser without a deep learning runtime.
  - Evaluation is leave-one-out. With one to three clips per class there is no
    honest held-out split; anything else would overstate the result.

A class with a single clip can never be scored correctly under leave-one-out:
holding out its only example leaves the model with no instance of that class.
Those are counted separately so the headline number is not quietly inflated.

    python ML/train.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from config import DATA_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DS = DATA_ROOT / "dataset"
OUT_MODEL = DS / "model.json"

# Gloss -> Amharic. The recogniser's output language is Amharic; these strings
# are what the web app displays and speaks.
AMHARIC = {
    "hello": "ሰላም", "thank_you": "አመሰግናለሁ", "goodbye": "ደህና ሁን",
    "yes": "አዎ", "no": "አይ", "please": "እባክህ", "sorry": "ይቅርታ",
    "understand": "ገባኝ", "wait": "ጠብቅ", "help": "እርዳታ", "water": "ውሃ",
    "food": "ምግብ", "hungry": "ተርቦኛል", "tired": "ደክሞኛል", "sick": "ህመም",
    "hospital": "ሆስፒታል", "doctor": "ሐኪም", "pain": "ህመም", "police": "ፖሊስ",
    "where": "የት", "what": "ምን", "who": "ማን", "when": "መቼ", "cost": "ስንት",
    "me": "እኔ", "you": "አንተ", "mother": "እናት", "father": "አባት",
    "friend": "ጓደኛ",
}


def featurise(X):
    """(N, T, D) -> (N, 5D). Pooling over time, plus motion."""
    d = np.diff(X, axis=1)
    return np.concatenate([
        X.mean(1), X.std(1), X[:, 0], X[:, -1], np.abs(d).mean(1)
    ], axis=1).astype(np.float32)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit(Xf, y, n_classes, l2=1.0, epochs=600, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n, d = Xf.shape
    W = rng.normal(0, 0.01, (d, n_classes)).astype(np.float32)
    b = np.zeros(n_classes, np.float32)
    Y = np.zeros((n, n_classes), np.float32)
    Y[np.arange(n), y] = 1.0
    for _ in range(epochs):
        P = softmax(Xf @ W + b)
        gW = Xf.T @ (P - Y) / n + l2 * W / n
        gb = (P - Y).mean(0)
        W -= lr * gW
        b -= lr * gb
    return W, b


def standardise(Xf):
    mu = Xf.mean(0)
    sd = Xf.std(0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def main():
    X = np.load(DS / "X.npy")
    y = np.load(DS / "y.npy")
    labels = json.loads((DS / "labels.json").read_text(encoding="utf-8"))
    n_classes = len(labels)

    Xf = featurise(X)
    counts = np.bincount(y, minlength=n_classes)
    singles = {labels[i] for i in range(n_classes) if counts[i] == 1}

    print(f"samples {len(y)}   classes {n_classes}   features {Xf.shape[1]}")
    print(f"classes with a single clip: {len(singles)} "
          f"({', '.join(sorted(singles)) if singles else 'none'})")

    # ---- leave-one-out ----
    correct = wrong = 0
    correct_multi = total_multi = 0
    top3 = 0
    for i in range(len(y)):
        mask = np.ones(len(y), bool)
        mask[i] = False
        mu, sd = standardise(Xf[mask])
        W, b = fit((Xf[mask] - mu) / sd, y[mask], n_classes)
        p = softmax(((Xf[i] - mu) / sd)[None] @ W + b)[0]
        pred = int(p.argmax())
        hit = pred == y[i]
        correct += hit
        wrong += not hit
        if p.argsort()[-3:][::-1].tolist().count(int(y[i])):
            top3 += 1
        if counts[y[i]] > 1:
            total_multi += 1
            correct_multi += hit

    n = len(y)
    print(f"\nleave-one-out top-1 : {correct}/{n} = {correct/n:.1%}")
    print(f"leave-one-out top-3 : {top3}/{n} = {top3/n:.1%}")
    if total_multi:
        print(f"excluding single-clip classes: {correct_multi}/{total_multi} "
              f"= {correct_multi/total_multi:.1%}")
    print(f"random baseline     : {1/n_classes:.1%}")

    # ---- final model on everything, for the demo ----
    mu, sd = standardise(Xf)
    W, b = fit((Xf - mu) / sd, y, n_classes)
    train_acc = (softmax(((Xf - mu) / sd) @ W + b).argmax(1) == y).mean()
    print(f"\ntraining-set accuracy (memorisation, not skill): {train_acc:.1%}")

    OUT_MODEL.write_text(json.dumps({
        "labels": labels,
        "amharic": {g: AMHARIC.get(g, g) for g in labels},
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "W": W.tolist(),
        "b": b.tolist(),
        "feature_dim": int(Xf.shape[1]),
        "frames": int(X.shape[1]),
        "landmark_dim": int(X.shape[2]),
        "loo_top1": round(float(correct / n), 4),
        "loo_top3": round(float(top3 / n), 4),
        "n_train": int(n),
        "single_clip_classes": sorted(singles),
    }, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {OUT_MODEL}  ({OUT_MODEL.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

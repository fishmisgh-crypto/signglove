# ML — data collection and training

Python side of ESLT. Nothing here is imported by the Flutter app; the only
artifact that crosses over is the exported `.tflite` model.

## Where the data lives

Outside the repo, at `C:\Users\habte\OneDrive\Desktop\ESLT-data`. Change
`DATA_ROOT` in `config.py` if you move it.

```
ESLT-data/
  raw/        <GLOSS>/<SIGNER>/take_001.mp4      recorded clips
  landmarks/  <GLOSS>/<SIGNER>/take_001.npy      (30, 258) keypoint arrays
  external/                                       CESL etc. once access is granted
  manifests/  clips.csv, extraction_report.csv    index + quality flags
```

## Setup

```
.venv\Scripts\python.exe -m pip install -r ML/requirements.txt
```

## The loop

```
python record_clips.py --signer S01      # record a signer through the vocabulary
python extract_landmarks.py              # video -> keypoints, skips done work
python dataset_status.py --matrix        # how much is there, what is missing
```

All three are resumable. Stop and restart whenever; nothing is recomputed.

`dataset_status.py` is the one that answers "is there enough yet". Three gates:

| Gate | Condition | Meaning |
|---|---|---|
| PILOT | 3 signers × every gloss | pipeline proven, first model trainable |
| HELD-OUT | ≥5 signers | can evaluate on an unseen signer |
| TARGET | 10 signers × 25 takes × 30 glosses | train the real v1 |

TARGET is 7,500 clips. That is roughly 10 people × 2 sessions × 90 minutes.

## Why signer count matters more than clip count

Published EthSL work reached 94% accuracy when train and test shared signers,
but 73% when the test signer was unseen. Twenty more clips from a signer you
already have buys almost nothing; one new signer buys a lot. Recruit breadth
before depth.

## Recording protocol

Keep these constant or the model learns the room instead of the sign:

- Camera at chest height, signer framed head to waist, arms fully in frame at full extension.
- Plain background, light on the signer's face, no window behind them.
- Signer wears sleeves that contrast with the background, no reflective jewellery.
- One gloss per clip, hands starting and ending at rest.
- Vary deliberately across sessions: different rooms, clothing, distances, left- and right-handed signers. This variation is the point — it is what makes the model generalise.

Signer ids are `S01`, `S02`, ... — never real names in filenames or the manifest.

## Consent

Every signer is identifiable on video, so before recording get written consent
covering what the footage is used for, whether it may be published or shared
outside the team, and how to withdraw. File signed forms in
`manifests/consent/` and keep them out of git. Decide the sharing question
before session 1 — retrofitting consent onto existing footage means
re-contacting everyone, and any signer you cannot reach has to be deleted.

## External datasets

- **CESL** (1,320 videos, 22 signers, 30 sentences, 65-word vocabulary, 92.7 GB, CC-BY-4.0) — https://zenodo.org/records/10800699. Files are access-restricted; request through Zenodo. It is continuous sentence data, so it does not drop straight into an isolated-word pipeline, but it is the largest EthSL corpus that exists.
- **CESLR training code** — https://github.com/ethio-artifical/CESLR (code only, no data).

Do not train on ASL corpora such as WLASL and expect EthSL to work. They are
different languages, not dialects. WLASL is useful only as generic
pretraining for hand-motion features, never as a source of EthSL labels.

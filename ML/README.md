# ML — data collection and training

Python side of ESLT. Video data lives **outside** this repo, at the path set by
`DATA_ROOT` in `config.py` (default: `Desktop\ESLT-data`).

## Setup

    .venv\Scripts\python.exe -m pip install -r ML/requirements.txt
    .venv\Scripts\python.exe ML/download_models.py

`download_models.py` fetches the MediaPipe Tasks `.task` bundles. The legacy
`mp.solutions.holistic` API that most published EthSL work uses was removed in
MediaPipe 0.10.3x and is unavailable on Python 3.13 — this project uses the
Tasks `HolisticLandmarker` instead.

## Collect

    python ML/record_clips.py --signer S01        # record a session
    python ML/extract_landmarks.py                # video -> (30, 258) .npy
    python ML/dataset_status.py                   # what is still missing

Repeat with a new `--signer` id per person. `dataset_status.py` is the loop
controller: it names the exact gap to fill next and gates on PILOT / HELD-OUT /
TARGET.

## Targets

| Gate | Condition | Meaning |
|---|---|---|
| PILOT | 30 glosses x 3 signers | pipeline proven end to end |
| HELD-OUT | >= 5 signers | signer-invariant evaluation possible |
| TARGET | 10 signers x 25 takes | train the v1 model (7,500 clips) |

Signer count, not takes per signer, is what buys generalisation: published EthSL
work hit 94% signer-dependent but 73% signer-invariant with 7 signers.

## Consent

Every signer must sign a consent form before recording. Store it in
`ESLT-data/manifests/consent/`. This is face and body video of identifiable
people — treat it accordingly and do not commit it or publish it.

## Pipeline

    python ML/build_dataset.py    # clips -> normalised landmarks + manifest
    python ML/train.py            # train, leave-one-out eval, export model.json
    python ML/app.py              # web demo at http://127.0.0.1:5000

`build_dataset.py` normalises every frame to the shoulder midpoint and scales by
shoulder width, so the model cannot key on where the signer stood or how close
they were to the camera.

Notes on MediaPipe that cost real time to find:
- `pose_landmarks` in the Tasks API is a flat list, not one list per person.
- VIDEO mode needs timestamps increasing across the landmarker's whole lifetime,
  not per clip, and its segmentation node asserts when frame size changes. Use
  IMAGE mode with a fresh landmarker per clip when sources vary in resolution.

"""Fetch WLASL clips for the glosses that overlap the ESLT vocabulary.

WLASL distributes URLs, not video: the clips belong to third parties (YouTube
and a number of ASL dictionary sites), which is why the dataset cannot ship
them. Expect link rot — the maintainers acknowledge dead URLs and offer the
preprocessed videos through a request form instead. That route is cleaner and
worth pursuing in parallel.

Downloads are trimmed to each instance's frame range and written as

    external/wlasl/<gloss>/<video_id>.mp4

Nothing here says these signs are used in EthSL. See asl_bridge.py — every
gloss pairing needs a fluent signer's verification before it trains anything.

    python ML/fetch_wlasl.py                 # all overlapping glosses
    python ML/fetch_wlasl.py --gloss water   # one gloss
    python ML/fetch_wlasl.py --limit 5       # first N per gloss, for a trial
    python ML/fetch_wlasl.py --skip-youtube  # dictionary sites only
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import cv2

from config import DATA_ROOT
from vocabulary import load_vocabulary
from asl_bridge import candidates

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WLASL_JSON = DATA_ROOT / "external" / "WLASL_v0.3.json"
OUT_ROOT = DATA_ROOT / "external" / "wlasl"
REPORT = DATA_ROOT / "manifests" / "wlasl_fetch.csv"
YOUTUBE_HOSTS = {"youtube.com", "youtu.be", "m.youtube.com"}


def host_of(url):
    return urlparse(url).netloc.lower().replace("www.", "")


def download_direct(url, dest, timeout=90):
    """Plain HTTP fetch for the dictionary sites."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 18):
            f.write(chunk)


def download_youtube(url, dest):
    subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings",
         "-f", "mp4/best", "-o", str(dest), url],
        check=True, timeout=600,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def trim(src, dest, frame_start, frame_end):
    """Copy the instance's frame range out of the downloaded video."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or h == 0:
        cap.release()
        return 0
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx, written = 1, 0                      # WLASL frame numbers are 1-based
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= frame_start and (frame_end == -1 or idx <= frame_end):
            writer.write(frame)
            written += 1
        if frame_end != -1 and idx > frame_end:
            break
        idx += 1
    writer.release()
    cap.release()
    if written == 0:
        dest.unlink(missing_ok=True)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gloss", default="", help="single WLASL gloss")
    ap.add_argument("--limit", type=int, default=0, help="max instances per gloss")
    ap.add_argument("--skip-youtube", action="store_true")
    ap.add_argument("--hosts", default="",
                    help="comma-separated host substrings to allow; other hosts "
                         "are skipped without being contacted. Use this when most "
                         "sources are blocked or unroutable and a full run would "
                         "just accumulate timeouts")
    ap.add_argument("--timeout", type=int, default=90,
                    help="per-file timeout in seconds (default 90)")
    args = ap.parse_args()
    allow = [h.strip() for h in args.hosts.split(",") if h.strip()]

    if not WLASL_JSON.exists():
        raise SystemExit(f"missing {WLASL_JSON}")
    data = json.loads(WLASL_JSON.read_text(encoding="utf-8"))
    by_gloss = {g["gloss"].lower(): g["instances"] for g in data}

    if args.gloss:
        wanted = {args.gloss.lower()}
    else:
        wanted = set()
        for e in load_vocabulary():
            for c in candidates(e["gloss"]):
                if c in by_gloss:
                    wanted.add(c)
                    break

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    rows, ok, dead, skipped = [], 0, 0, 0
    tmpdir = Path(tempfile.mkdtemp(prefix="wlasl_"))

    for gloss in sorted(wanted):
        instances = by_gloss[gloss]
        if args.limit:
            instances = instances[:args.limit]
        outdir = OUT_ROOT / gloss.replace(" ", "_")
        outdir.mkdir(parents=True, exist_ok=True)

        for inst in instances:
            vid, url = str(inst["video_id"]), inst["url"]
            host = host_of(url)
            dest = outdir / f"{vid}.mp4"
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue
            if args.skip_youtube and host in YOUTUBE_HOSTS:
                continue
            if allow and not any(a in host for a in allow):
                continue

            raw = tmpdir / f"{vid}_raw.mp4"
            status = ""
            try:
                if host in YOUTUBE_HOSTS:
                    download_youtube(url, raw)
                else:
                    download_direct(url, raw, args.timeout)
                n = trim(raw, dest, inst.get("frame_start", 1),
                         inst.get("frame_end", -1))
                if n:
                    ok += 1
                    status = "ok"
                    print(f"  ok   {gloss:<12} {vid:<8} {n:>3} frames  ({host})")
                else:
                    dead += 1
                    status = "undecodable"
                    print(f"  bad  {gloss:<12} {vid:<8} undecodable ({host})")
            except Exception as e:
                dead += 1
                status = f"{type(e).__name__}"
                print(f"  dead {gloss:<12} {vid:<8} {status} ({host})")
            finally:
                raw.unlink(missing_ok=True)

            rows.append({"gloss": gloss, "video_id": vid, "host": host,
                         "url": url, "status": status,
                         "signer_id": inst.get("signer_id", ""),
                         "split": inst.get("split", "")})

    if rows:
        new = not REPORT.exists()
        with REPORT.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)

    attempted = ok + dead
    print(f"\ndownloaded {ok}   failed {dead}   already had {skipped}")
    if attempted:
        print(f"link rot: {dead}/{attempted} = {dead/attempted:.0%}")
    print(f"clips:  {OUT_ROOT}")
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()

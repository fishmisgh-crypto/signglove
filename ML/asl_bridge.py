"""Cross-reference the ESLT vocabulary against WLASL's gloss list.

EthSL developed in Ethiopian deaf schools under documented ASL and Nordic
influence, so some signs genuinely coincide. That makes WLASL a possible source
of extra training clips for the overlapping subset.

It is a possibility, not a fact. EthSL has independent lexical structures and
students coined local signs, so any given word may or may not share a form. This
script only reports which of our glosses HAVE WLASL clips available — it cannot
tell you whether the sign is the same. That judgement has to come from a fluent
EthSL signer, and until it does every row is `unverified`.

Shipping an unverified pair means the app confidently shows an Amharic word for
an ASL sign that Ethiopian signers do not use. For an assistive tool that is
worse than showing nothing.

    python ML/asl_bridge.py
    python ML/asl_bridge.py --write     # write the review sheet
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from config import DATA_ROOT
from vocabulary import load_vocabulary

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WLASL_JSON = DATA_ROOT / "external" / "WLASL_v0.3.json"
REVIEW_CSV = DATA_ROOT / "manifests" / "asl_bridge_review.csv"

# Our gloss labels are not always the English word WLASL uses.
ALIASES = {
    "THANK-YOU": ["thank you", "thanks"],
    "NOT-UNDERSTAND": ["don't understand", "not understand"],
    "HOW-MUCH": ["how much", "how many", "cost"],
    "I-ME": ["me", "i"],
    "PAIN": ["hurt", "pain"],
    "SICK": ["sick", "ill"],
    "POLICE": ["police", "cop"],
    "GOODBYE": ["goodbye", "bye"],
}


def candidates(gloss):
    base = gloss.lower().replace("-", " ")
    return [base] + [a for a in ALIASES.get(gloss, []) if a != base]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="write the review sheet for a signer to fill in")
    args = ap.parse_args()

    if not WLASL_JSON.exists():
        raise SystemExit(f"missing {WLASL_JSON}\n"
                         "  download WLASL_v0.3.json from the WLASL start_kit")
    data = json.loads(WLASL_JSON.read_text(encoding="utf-8"))
    counts = {g["gloss"].lower(): len(g["instances"]) for g in data}

    rows, hit, clips = [], 0, 0
    for e in load_vocabulary():
        match, n = "", 0
        for c in candidates(e["gloss"]):
            if c in counts:
                match, n = c, counts[c]
                break
        if match:
            hit += 1
            clips += n
        rows.append({
            "eslt_gloss": e["gloss"],
            "amharic": e["amharic"],
            "wlasl_gloss": match,
            "wlasl_clips": n,
            "same_sign_in_ethsl": "",      # signer fills: yes / no / partial
            "verified_by": "",
            "notes": "",
        })

    width = max(len(r["eslt_gloss"]) for r in rows)
    for r in rows:
        mark = f"{r['wlasl_clips']:>3} clips  ({r['wlasl_gloss']})" if r["wlasl_gloss"] \
            else "  — not in WLASL"
        print(f"  {r['eslt_gloss']:<{width}}  {mark}")

    total = len(rows)
    print(f"\n{hit}/{total} glosses have WLASL clips — {clips} clips in total")
    print(f"average {clips/hit:.1f} clips per matched gloss" if hit else "")
    print("\nEvery match above is UNVERIFIED. A WLASL entry existing says nothing")
    print("about whether Ethiopian signers use the same sign. Have a fluent EthSL")
    print("signer mark each row before any of it is used for training.")

    if args.write:
        REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
        with REVIEW_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nreview sheet: {REVIEW_CSV}")


if __name__ == "__main__":
    main()

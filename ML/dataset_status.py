"""Answer the only question that matters during collection:
is there enough data yet, and if not, what exactly is missing?

    python dataset_status.py
    python dataset_status.py --matrix

Readiness gates, in order:
  PILOT       every gloss covered by PILOT_SIGNERS   -> pipeline proven end to end
  HELD-OUT    >= MIN_SIGNERS_FOR_HELDOUT signers     -> signer-invariant eval possible
  TARGET      >= TARGET_SIGNERS at full take count   -> train the real v1 model
"""
import argparse
import sys
from collections import defaultdict

from config import (
    LANDMARK_DIR, MIN_SIGNERS_FOR_HELDOUT, PILOT_SIGNERS, RAW_DIR,
    TAKES_PER_GLOSS_PER_SIGNER, TARGET_SIGNERS,
)
from vocabulary import glosses

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def survey():
    """counts[gloss][signer] = n clips, plus the set of signers seen."""
    counts = defaultdict(lambda: defaultdict(int))
    signers = set()
    for clip in RAW_DIR.rglob("take_*.mp4"):
        gloss, signer = clip.parent.parent.name, clip.parent.name
        counts[gloss][signer] += 1
        signers.add(signer)
    return counts, sorted(signers)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", action="store_true", help="print the full gloss x signer grid")
    args = ap.parse_args()

    vocab = glosses()
    counts, signers = survey()
    n_landmarks = len(list(LANDMARK_DIR.rglob("*.npy")))
    total = sum(sum(v.values()) for v in counts.values())

    print("=" * 64)
    print("ESLT dataset status")
    print("=" * 64)
    print(f"vocabulary        {len(vocab)} glosses")
    print(f"signers recorded  {len(signers)}  {signers if signers else ''}")
    print(f"clips recorded    {total}")
    print(f"landmarks         {n_landmarks}"
          + ("" if n_landmarks == total else f"   ({total - n_landmarks} not yet extracted)"))

    full_target = len(vocab) * TARGET_SIGNERS * TAKES_PER_GLOSS_PER_SIGNER
    pct = (total / full_target * 100) if full_target else 0
    print(f"progress to v1    {total}/{full_target}  ({pct:.1f}%)")

    if args.matrix and signers:
        print("\ngloss x signer")
        print("  " + "".join(f"{s:>7}" for s in signers))
        for g in vocab:
            cells = "".join(f"{counts[g].get(s, 0):>7}" for s in signers)
            print(f"{g:<18}{cells}")

    # readiness gates
    print("\nreadiness")
    covered_by = {g: len([s for s, n in counts[g].items() if n >= TAKES_PER_GLOSS_PER_SIGNER])
                  for g in vocab}
    pilot_ok = all(covered_by[g] >= PILOT_SIGNERS for g in vocab)
    heldout_ok = len(signers) >= MIN_SIGNERS_FOR_HELDOUT
    target_ok = all(covered_by[g] >= TARGET_SIGNERS for g in vocab)
    for label, ok in (("PILOT", pilot_ok), ("HELD-OUT", heldout_ok), ("TARGET", target_ok)):
        print(f"  [{'x' if ok else ' '}] {label}")

    # what to do next
    print("\nnext")
    missing = [g for g in vocab if not counts.get(g)]
    if missing:
        print(f"  {len(missing)} glosses have zero clips: {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    thin = sorted(((sum(counts[g].values()), g) for g in vocab if counts.get(g)))[:5]
    for n, g in thin:
        print(f"  {g}: only {n} clips")
    if len(signers) < TARGET_SIGNERS:
        print(f"  recruit {TARGET_SIGNERS - len(signers)} more signers "
              f"— signer count, not takes per signer, is what drives generalisation")
    if total > n_landmarks:
        print("  run: python extract_landmarks.py")


if __name__ == "__main__":
    main()

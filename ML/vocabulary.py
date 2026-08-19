"""Parse the gloss list out of docs/vocabulary_list.md so the markdown table
stays the single source of truth for labels."""
import re
import sys
from config import VOCAB_MD

# Windows consoles default to cp1252 and raise on Amharic. Force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([A-Z][A-Z\-]*)\s*\|\s*(.+?)\s*\|\s*(\w+)\s*\|")


def load_vocabulary():
    """Return [{'index','gloss','amharic','category'}, ...] in table order."""
    if not VOCAB_MD.exists():
        raise FileNotFoundError(f"vocabulary list not found: {VOCAB_MD}")
    entries = []
    for line in VOCAB_MD.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if m:
            entries.append({
                "index": int(m.group(1)),
                "gloss": m.group(2),
                "amharic": m.group(3),
                "category": m.group(4),
            })
    if not entries:
        raise ValueError(f"no gloss rows parsed from {VOCAB_MD}")
    glosses = [e["gloss"] for e in entries]
    dupes = {g for g in glosses if glosses.count(g) > 1}
    if dupes:
        raise ValueError(f"duplicate glosses in vocabulary list: {sorted(dupes)}")
    return entries


def glosses():
    return [e["gloss"] for e in load_vocabulary()]


if __name__ == "__main__":
    for e in load_vocabulary():
        print(f"{e['index']:>3}  {e['gloss']:<16} {e['amharic']:<12} {e['category']}")

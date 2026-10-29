"""Verify a prose rewrite preserved every number, citation key, reference, label and macro.
  python scripts/check_tex_tokens.py sections/introduction.tex [...]   (compares with paper_backup_prerewrite/)"""
import re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
def tokens(t):
    t = re.sub(r"%.*", "", t)
    cites = Counter(k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", t) for k in m.split(","))
    refs = Counter(re.findall(r"\\(?:ref|label)\{([^}]*)\}", t))
    macros = Counter(re.findall(r"\\([A-Za-z]+)", t))
    prose = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    nums = Counter(re.findall(r"\d+(?:[.,]\d+)*", prose))
    return cites, refs, macros, nums
bad = False
for f in sys.argv[1:]:
    new = tokens((ROOT / "paper" / f).read_text()); old = tokens((ROOT / "paper_backup_prerewrite" / f).read_text())
    for name, a, b in zip(["cite", "ref/label", "macro", "number"], old, new):
        lost, gained = a - b, b - a
        # allow harmless formatting macros to differ
        if name == "macro":
            keep = {k for k in lost if k.startswith(("Pilot","Bench","R","C","P","Abl","Full","Human","Key")) or k in {"cite","ref","label","pd","promptA","promptB","promptBtwo","pending","authornote","input","includegraphics","caption"}}
            lost = Counter({k: v for k, v in lost.items() if k in keep}); gained = Counter({k: v for k, v in gained.items() if k in keep})
        if lost or gained:
            bad = True; print(f"{f}: {name}: lost {dict(lost)} gained {dict(gained)}")
print("TOKENS OK" if not bad else "TOKEN DIFFERENCES ABOVE")

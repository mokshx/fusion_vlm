"""Build a manifest of UP-Fall per-trial Google Drive file IDs from the official
HAR-UP Google Sites page (https://sites.google.com/up.edu.mx/har-up/).

Usage: python scripts/upfall_manifest.py <saved_page.html> data/raw/upfall/manifest.csv

Columns: subject, activity, trial, kind (DataSet | Camera1 | Camera2), drive_id
"""

import csv
import html
import re
import sys
from pathlib import Path

KINDS = {"DataSet", "Camera1", "Camera2"}


def main(page: str, out: str) -> None:
    text = html.unescape(html.unescape(Path(page).read_text(errors="replace")))
    rows, cur = [], None
    token = re.compile(
        r"Subject(\d+)Activity(\d+)Trial(\d+)</h5>"
        r"|uc\?id=([\w-]{20,})[^>]*>\s*([^<]+?)\s*</a>"
    )
    for m in token.finditer(text):
        if m.group(1):
            cur = tuple(int(g) for g in m.group(1, 2, 3))
        elif cur and m.group(5) in KINDS:
            rows.append((*cur, m.group(5), m.group(4)))
    rows = sorted(set(rows))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["subject", "activity", "trial", "kind", "drive_id"])
        w.writerows(rows)
    trials = {r[:3] for r in rows}
    print(f"{len(rows)} files across {len(trials)} trials, "
          f"subjects {sorted({t[0] for t in trials})}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

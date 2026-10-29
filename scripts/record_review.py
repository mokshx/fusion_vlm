"""Record visual-review decisions for one contact sheet.

The sheet's candidates are recovered exactly as review_sheets.py laid them out.
Every candidate on the sheet gets the default decision unless overridden.

  uv run python scripts/record_review.py SHEET --label L --subtype S \
      [--set c0123=hard_negative:crouching ...] [--reject c0456:reason ...]

Labels: person_down | hard_negative | normal. Rejected candidates get label
"reject" and are excluded from the benchmark. Re-running a sheet replaces its rows.
"""

import argparse
from pathlib import Path

import pandas as pd

BENCH = Path("data/interim/bench")
LABELS = {"person_down", "hard_negative", "normal"}
SUBTYPES = {"fall", "lying_floor_deliberate", "crouching", "kneeling", "bending", "sitting_chair",
            "sitting_furniture", "sitting_floor", "lying_furniture", "standing", "walking"}


def sheet_members(sheet: str, per_sheet: int = 30) -> list[str]:
    df = pd.read_csv(BENCH / "candidates.csv")
    for (ds, label, sub), g in df.groupby(["dataset", "proposed_label", "proposed_subtype"]):
        name = f"{ds}_{label}_{sub.rstrip('?')}"
        if sheet.startswith(name + "_"):
            k = int(sheet.rsplit("_", 1)[1].split(".")[0])
            g = g.sort_values(["sequence", "camera", "frame"])
            return list(g.cand_id.iloc[k * per_sheet:(k + 1) * per_sheet])
    raise SystemExit(f"unknown sheet {sheet}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--label", required=True)
    ap.add_argument("--subtype", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="cand=label:subtype")
    ap.add_argument("--reject", nargs="*", default=[], help="cand:reason")
    a = ap.parse_args()
    members = sheet_members(a.sheet)
    rows = {c: (a.label, a.subtype, "") for c in members}
    for s in a.set:
        c, ls = s.split("=")
        lab, sub = ls.split(":")
        rows[c] = (lab, sub, "relabelled at review")
    for s in a.reject:
        c, _, why = s.partition(":")
        rows[c] = ("reject", "", why)
    for c, (lab, sub, _) in rows.items():
        assert c in members, f"{c} is not on {a.sheet}"
        assert lab == "reject" or (lab in LABELS and sub in SUBTYPES), (c, lab, sub)
    path = BENCH / "review.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["cand_id", "label", "subtype", "note", "sheet"])
    old = old[~old.cand_id.isin(members)]
    new = pd.DataFrame([{"cand_id": c, "label": l, "subtype": s, "note": n, "sheet": a.sheet}
                        for c, (l, s, n) in rows.items()])
    pd.concat([old, new]).sort_values("cand_id").to_csv(path, index=False)
    print(f"{a.sheet}: {len(members)} reviewed, {sum(l == 'reject' for l, _, _ in rows.values())} rejected, "
          f"{len(a.set)} relabelled; total reviewed {len(old) + len(new)}")


if __name__ == "__main__":
    main()

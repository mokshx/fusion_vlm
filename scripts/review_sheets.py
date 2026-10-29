"""Benchmark step 2: contact sheets for visually reviewing every candidate.

Each tile shows the candidate's target frame captioned with its cand_id and
proposed subtype. Decisions are recorded in data/interim/bench/review.csv
(cand_id, label, subtype, note) by scripts/record_review.py.

Usage: uv run python scripts/review_sheets.py [--per-sheet 30 --cols 6 --width 290]
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

BENCH = Path("data/interim/bench")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sheet", type=int, default=30)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--width", type=int, default=290)
    a = ap.parse_args()
    df = pd.read_csv(BENCH / "candidates.csv")
    out = BENCH / "review_sheets"
    out.mkdir(parents=True, exist_ok=True)
    for (ds, label, sub), g in df.groupby(["dataset", "proposed_label", "proposed_subtype"]):
        g = g.sort_values(["sequence", "camera", "frame"])
        name = f"{ds}_{label}_{sub.rstrip('?')}"
        for k in range(0, len(g), a.per_sheet):
            part = g.iloc[k:k + a.per_sheet]
            tiles = []
            for _, r in part.iterrows():
                im = Image.open(r.image_path).convert("RGB")
                im = im.resize((a.width, round(im.height * a.width / im.width)))
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, a.width, 15], fill=(0, 0, 0))
                d.text((3, 2), f"{r.cand_id} {r.sequence} {r.camera}", fill=(255, 255, 0))
                tiles.append(im)
            th = max(t.height for t in tiles)
            rows = -(-len(tiles) // a.cols)
            sheet = Image.new("RGB", (a.cols * a.width, rows * th), (40, 40, 40))
            for i, t in enumerate(tiles):
                sheet.paste(t, ((i % a.cols) * a.width, (i // a.cols) * th))
            sheet.save(out / f"{name}_{k // a.per_sheet:02d}.jpg", quality=85)
    for p in sorted(out.glob("*.jpg")):
        print(p.name)


if __name__ == "__main__":
    main()

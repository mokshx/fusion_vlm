"""Tile images into a numbered grid for fast visual review.

Usage: uv run python scripts/contact_sheet.py out.jpg img1.png img2.png ... [--cols 4 --width 320]
Each tile is captioned with its index and file stem.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--captions", nargs="*", help="optional per-image caption, same order")
    args = ap.parse_args()

    tiles = []
    for i, p in enumerate(args.images):
        im = Image.open(p).convert("RGB")
        im = im.resize((args.width, round(im.height * args.width / im.width)))
        d = ImageDraw.Draw(im)
        cap = f"{i}: {p.stem}" + (f"  {args.captions[i]}" if args.captions else "")
        d.rectangle([0, 0, args.width, 16], fill=(0, 0, 0))
        d.text((3, 2), cap, fill=(255, 255, 0))
        tiles.append(im)
    th = max(t.height for t in tiles)
    rows = -(-len(tiles) // args.cols)
    sheet = Image.new("RGB", (args.cols * args.width, rows * th), (40, 40, 40))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % args.cols) * args.width, (i // args.cols) * th))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=88)


if __name__ == "__main__":
    main()

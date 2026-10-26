"""Fetch a pool of candidate frames from UR Fall for manual labelling.

Uses the official per-frame depth-feature CSVs to propose candidates, then pulls
only those PNGs from the remote zips (range requests). Every candidate must be
checked by eye before it becomes a label: the depth label says "lying on the
ground" but cannot tell floor from bed, and low height does not tell sitting
from crouching.

Candidate rules (cam0 frame numbers == CSV frame numbers):
  fall_lying     falls, middle frame of the longest label==1 run
  fall_overhead  falls, cam1 frame at 90% of the sequence (person usually down);
                 cam1 has no official labels, so these need visual confirmation
  adl_low        ADLs, label==-1 frame with minimum HHmaxRatio (sit/crouch/bend)
  adl_lying      ADLs, middle of the longest label==1 run (bed/sofa/floor?)
  adl_upright    ADLs, label==-1 frame with HHmaxRatio closest to 1.0

Usage:
  uv run python scripts/urfall_fetch_candidates.py --falls 1-10 --adls 1-40 \
      --out data/interim/urfall_candidates
"""

import argparse
import csv
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

from pdvlm.remote_zip import open_remote_zip

BASE = "http://fenix.ur.edu.pl/~mkepski/ds/data"
LABEL_DIR = Path("data/raw/urfall")


def read_labels(path: Path) -> dict[str, list[tuple[int, int, float]]]:
    """seq -> [(frame, label, HHmaxRatio)]

    Columns: seq, frame, label, HeightWidthRatio, MajorMinorRatio,
    BoundingBoxOccupancy, MaxStdXZ, HHmaxRatio, H[mm], D[mm], P40.
    """
    seqs = defaultdict(list)
    for r in csv.reader(open(path)):
        seqs[r[0]].append((int(r[1]), int(r[2]), float(r[7])))
    return seqs


def longest_run_middle(rows, label: int) -> int | None:
    best, cur = [], []
    for fr, lab, _ in rows:
        if lab == label:
            cur.append(fr)
            if len(cur) > len(best):
                best = cur[:]
        else:
            cur = []
    return best[len(best) // 2] if best else None


def span(s: str) -> list[int]:
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--falls", default="1-10")
    ap.add_argument("--adls", default="1-40")
    ap.add_argument("--out", default="data/interim/urfall_candidates")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    falls = read_labels(LABEL_DIR / "urfall-cam0-falls.csv")
    adls = read_labels(LABEL_DIR / "urfall-cam0-adls.csv")

    wanted = []  # (seq, cam, frame, rule, source_label, hh_ratio)
    for i in span(args.falls):
        seq = f"fall-{i:02d}"
        rows = falls[seq]
        fr = longest_run_middle(rows, 1)
        if fr:
            wanted.append((seq, 0, fr, "fall_lying", 1, dict((f, h) for f, _, h in rows)[fr]))
        wanted.append((seq, 1, None, "fall_overhead", "", ""))
    for i in span(args.adls):
        seq = f"adl-{i:02d}"
        rows = adls[seq]
        upright = [r for r in rows if r[1] == -1]
        lo = min(upright, key=lambda r: r[2])
        wanted.append((seq, 0, lo[0], "adl_low", -1, lo[2]))
        fr = longest_run_middle(rows, 1)
        if fr:
            wanted.append((seq, 0, fr, "adl_lying", 1, dict((f, h) for f, _, h in rows)[fr]))
        up = min(upright, key=lambda r: abs(r[2] - 1.0))
        wanted.append((seq, 0, up[0], "adl_upright", -1, up[2]))

    by_zip = defaultdict(list)
    for w in wanted:
        by_zip[(w[0], w[1])].append(w)

    meta_path = out / "candidates.csv"
    done = set()
    if meta_path.exists():
        done = {r["file"] for r in csv.DictReader(open(meta_path))}
    new_file = not meta_path.exists()
    lock = threading.Lock()
    with open(meta_path, "a", newline="") as mf:
        mw = csv.writer(mf, lineterminator="\n")
        if new_file:
            mw.writerow(["file", "sequence", "camera", "frame", "n_frames", "rule", "source_label", "hh_ratio"])

        def fetch(key) -> int:
            (seq, cam), items = key, by_zip[key]
            stem = f"{seq}-cam{cam}-rgb"
            if all(f"{stem}-{w[2]:03d}.png" in done for w in items if w[2]) and all(w[2] for w in items):
                return 0
            z, f = open_remote_zip(f"{BASE}/{stem}.zip")
            pngs = sorted(n for n in z.namelist() if n.endswith(".png"))
            for _, _, fr, rule, lab, hh in items:
                if fr is None:  # overhead: 90% through the sequence
                    fr = int(pngs[int(0.9 * (len(pngs) - 1))].rsplit("-", 1)[1][:3])
                fname = f"{stem}-{fr:03d}.png"
                if fname in done:
                    continue
                if not (out / fname).exists():
                    (out / fname).write_bytes(z.read(f"{stem}/{fname}"))
                with lock:
                    mw.writerow([fname, seq, cam, fr, len(pngs), rule, lab, hh])
                    mf.flush()
            print(f"{stem}: {len(items)} frames, {f.bytes_fetched / 1e6:.1f} MB fetched", flush=True)
            return f.bytes_fetched

        with ThreadPoolExecutor(args.workers) as ex:
            total = sum(ex.map(fetch, list(by_zip)))
    print(f"done; {total / 1e6:.1f} MB this run")

if __name__ == "__main__":
    main()

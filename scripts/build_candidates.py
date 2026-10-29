"""Benchmark step 1: propose candidate samples (with 8-frame clips) from the
source annotations, extract their frames, and write a candidate table for
visual review. Labels proposed here are NOT final; see scripts/review_candidates.py.

Spec (docs/labelling_policy.md):
  - at most 3 frames per label per sequence (MCF: per scenario-camera, then
    subsampled to a quota), at least 1 s apart, away from label boundaries;
  - each sample has the 7 preceding frames at 0.25 s spacing (8-frame clip);
  - transitional states are not sampled.

Sources:
  UR Fall  frame bank (data/raw/urfall/frames, one frame per 0.25 s) and the
           official per-frame cam0 labels (-1 not lying / 0 falling / 1 lying)
           with HHmaxRatio. cam1 frames take the label of the cam0 frame with
           the same index (both streams have the same frame count).
  MCF      OmniFall dense re-annotation (data/raw/omnifall/labels/mcfd.csv),
           timestamps on the true 30 fps timeline (AVI headers claim 120 fps).

Usage: uv run python scripts/build_candidates.py
Writes data/interim/bench/candidates.csv and data/interim/bench/frames/...
"""

import csv
import random
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/interim/bench")
CLIP_LEN, CLIP_STEP_S = 8, 0.25
MIN_GAP_S, PER_SEQ_LABEL = 1.0, 3
SEED = 20261026

# UR Fall ADL sequences whose "lying" (label 1) frames were checked by eye in the pilot.
UR_FLOOR_LYING = {30, 31, 32, 33, 34, 35, 38, 39, 40}  # deliberate lying on the floor
UR_FURNITURE_LYING = {10, 11, 21, 22, 23, 36, 37}  # lying on bed or sofa

# MCF candidate quotas (before review), per proposed class.
MCF_QUOTA = {"person_down": 150, "hard_negative": 220, "normal": 150}
OMNIFALL_MAP = {  # OmniFall class -> (proposed label, proposed subtype)
    "fallen": ("person_down", "fall"),
    "lying": ("hard_negative", "lying_furniture?"),
    "sitting": ("hard_negative", "sitting?"),
    "other": ("hard_negative", "other?"),  # mostly housekeeping/crouching; decided at review
    "standing": ("normal", "standing"),
    "walk": ("normal", "walking"),
}


def spread(items: list, k: int, key) -> list:
    """Up to k items, evenly spread, consecutive picks at least MIN_GAP_S apart."""
    items = sorted(items, key=key)
    picked = []
    for it in [items[round(i * (len(items) - 1) / max(1, k - 1))] for i in range(min(k, len(items)))] if k > 1 else items[len(items) // 2: len(items) // 2 + 1]:
        if all(abs(key(it) - key(p)) >= MIN_GAP_S for p in picked):
            picked.append(it)
    return picked


# ---------------------------------------------------------------- UR Fall
def urfall_candidates() -> list[dict]:
    lab = {}
    for f in ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]:
        for r in csv.reader(open(RAW / "urfall" / f)):
            lab[(r[0], int(r[1]))] = (int(r[2]), float(r[7]))
    bank = pd.read_csv(RAW / "urfall/frames/manifest.csv")
    bank["frame"] = bank.file.str.extract(r"-(\d+)\.png$").astype(int)[0]
    out = []
    for stem, g in bank.groupby("stem"):
        seq, cam = stem.rsplit("-cam", 1)[0], int(stem.split("-cam")[1][0])
        n_frames = int(g.n_frames.iloc[0])
        n0 = sum(1 for (s, _) in lab if s == seq)
        frames = sorted(g.frame)
        by_class = defaultdict(list)
        for pos in range(CLIP_LEN - 1, len(frames)):
            f = frames[pos]
            f0 = max(1, min(n0, round(f * n0 / n_frames)))  # cam1 -> cam0 index (same count in practice)
            if (seq, f0) not in lab:
                continue
            l, hh = lab[(seq, f0)]
            num = int(seq.split("-")[1])
            if l == 0:
                continue
            if seq.startswith("fall"):
                cls = ("person_down", "fall") if l == 1 else (("normal", "upright") if hh > 0.9 else None)
            elif l == 1:
                cls = (("person_down", "lying_floor_deliberate") if num in UR_FLOOR_LYING else
                       ("hard_negative", "lying_furniture") if num in UR_FURNITURE_LYING else None)
            else:
                cls = ("hard_negative", "low?") if hh < 0.65 else (("normal", "upright") if hh > 0.9 else None)
            if cls:
                by_class[cls].append((f, frames[pos - CLIP_LEN + 1: pos + 1], l, hh))
        for (label, sub), items in by_class.items():
            for f, clip, l, hh in spread(items, PER_SEQ_LABEL, key=lambda x: x[0] / 30):
                out.append({
                    "dataset": "urfall", "sequence": seq, "camera": f"cam{cam}",
                    "frame": f, "t_s": round((f - 1) / 30, 3),
                    "proposed_label": label, "proposed_subtype": sub,
                    "source_label": f"urfall:{l}", "hh_ratio": hh,
                    "viewpoint": "overhead" if cam == 1 else "side",
                    "clip_src": [str(RAW / "urfall/frames" / stem / f"{stem}-{c:03d}.png") for c in clip],
                })
    return out


# ---------------------------------------------------------------- MCF
def mcf_candidates() -> list[dict]:
    seg = pd.read_csv(RAW / "omnifall/labels/mcfd.csv")
    names = pd.read_csv(RAW / "omnifall/labels/label2id.csv").set_index("id").label
    seg["name"] = seg.label.map(names)
    pool = defaultdict(list)
    for (path, name), g in seg.groupby(["path", "name"]):
        if name not in OMNIFALL_MAP:
            continue
        label, sub = OMNIFALL_MAP[name]
        times = []
        for _, s in g.iterrows():
            lo = max(s.start + 0.25, (CLIP_LEN - 1) * CLIP_STEP_S)
            hi = s.end - 0.25
            t = lo
            while t <= hi:
                times.append(round(t, 3))
                t += MIN_GAP_S
        cap = 1 if name == "walk" else 2
        for t in spread(times, cap, key=lambda x: x):
            scen, cam = path.split("/")
            pool[label].append({
                "dataset": "mcf", "sequence": scen, "camera": cam,
                "frame": round(t * 30), "t_s": t,
                "proposed_label": label, "proposed_subtype": sub,
                "source_label": f"omnifall:{name}", "hh_ratio": "",
                "viewpoint": "elevated",
            })
    rng = random.Random(SEED)
    out = []
    for label, items in pool.items():
        rng.shuffle(items)
        out += items[: MCF_QUOTA[label]]
    return out


def extract_mcf_frames(cands: list[dict]) -> None:
    """Decode the needed frames (target + 7 preceding at 0.25 s) with ffmpeg."""
    need = defaultdict(set)
    for c in cands:
        if c["dataset"] != "mcf":
            continue
        idx = [round(30 * (c["t_s"] - j * CLIP_STEP_S)) for j in range(CLIP_LEN - 1, -1, -1)]
        c["clip_src"] = [str(OUT / "frames/mcf" / c["sequence"] / c["camera"] / f"{i:05d}.png") for i in idx]
        need[(c["sequence"], c["camera"])].update(idx)
    tmp = OUT / "tmp_video"
    tmp.mkdir(parents=True, exist_ok=True)
    for (scen, cam), idx in sorted(need.items()):
        dest = OUT / "frames/mcf" / scen / cam
        missing = sorted(i for i in idx if not (dest / f"{i:05d}.png").exists())
        if not missing:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RAW / f"mcf/zips/{scen}.zip") as z:
            video = tmp / f"{scen}_{cam}.avi"
            video.write_bytes(z.read(f"{scen}/{cam}.avi"))
        expr = "+".join(f"eq(n\\,{i})" for i in missing)
        # -frame_pts names each file by its pts, which equals the frame number for
        # these AVIs (time base = 1 tick per frame), so files map to frames exactly.
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='{expr}'",
                        "-fps_mode", "passthrough", "-frame_pts", "1", str(dest / "%05d.png")], check=True)
        video.unlink()
    tmp.rmdir()


def main() -> None:
    cands = urfall_candidates() + mcf_candidates()
    extract_mcf_frames(cands)
    kept, dropped = [], 0
    for c in cands:
        if all(Path(p).exists() for p in c["clip_src"]):
            kept.append(c)
        else:
            dropped += 1  # clip runs past the end of the video
    print(f"dropped {dropped} candidates with missing frames")
    cands = kept
    for k, c in enumerate(cands):
        c["cand_id"] = f"c{k:04d}"
        c["image_path"] = c["clip_src"][-1]
        c["clip_paths"] = ";".join(c.pop("clip_src"))
    df = pd.DataFrame(cands)
    cols = ["cand_id", "dataset", "sequence", "camera", "frame", "t_s", "proposed_label",
            "proposed_subtype", "source_label", "hh_ratio", "viewpoint", "image_path", "clip_paths"]
    OUT.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT / "candidates.csv", index=False)
    print(df.groupby(["dataset", "proposed_label", "proposed_subtype"]).size().to_string())
    print("total", len(df))


if __name__ == "__main__":
    main()

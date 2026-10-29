"""Non-VLM baseline: YOLO11 pose + a hand-written "person down" rule.

A person is flagged as down when
  (i)  the torso axis (hip midpoint -> shoulder midpoint) is within THETA degrees
       of horizontal (skipped if shoulder/hip keypoints are not confident), and
  (ii) the person's box is at least R times as wide as it is tall;
for clips, the condition must hold in at least a fraction Q of the frames.
THETA, R (and Q) are tuned on the dev split with the same criterion as the VLMs'
calibrated threshold: best dev F1 among settings with dev recall >= 95%.

  uv run --extra baseline --extra hf python scripts/pose_baseline.py extract
  uv run --extra baseline --extra hf python scripts/pose_baseline.py evaluate

Writes results/baseline/{features_<model>.csv, rows.csv, timing.json}. rows.csv
uses the same columns as the VLM run CSVs, so analyze_grid.py can include it.
"""

import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

OUT = Path("results/baseline")
LABELS = ["data/benchmark/v1/labels_test.csv", "data/benchmark/v1/labels_dev.csv"]
POSE_MODELS = ["models/yolo/yolo11n-pose.pt", "models/yolo/yolo11s-pose.pt"]
KP_CONF, BOX_CONF = 0.3, 0.25
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
THETAS = list(range(15, 65, 5))
RATIOS = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
QS = [0.25, 0.5, 0.75, 1.0]


def samples() -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in LABELS], ignore_index=True)


def torso_angle(kp: np.ndarray) -> float:
    """Degrees from horizontal of hip-midpoint -> shoulder-midpoint; NaN if unreliable."""
    sh = [kp[i, :2] for i in (L_SH, R_SH) if kp[i, 2] >= KP_CONF]
    hp = [kp[i, :2] for i in (L_HIP, R_HIP) if kp[i, 2] >= KP_CONF]
    if not sh or not hp:
        return math.nan
    v = np.mean(sh, axis=0) - np.mean(hp, axis=0)
    return math.degrees(math.atan2(abs(v[1]), abs(v[0]) + 1e-9))


def extract(model_name: str) -> None:
    from ultralytics import YOLO
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(model_name)
    df = samples()
    frames = sorted({p for c in df.clip_paths for p in c.split(";")})
    rows, times = [], []
    model.predict(frames[0], device=device, verbose=False)  # warm-up
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        r = model.predict(f, device=device, conf=BOX_CONF, verbose=False)[0]
        times.append((time.perf_counter() - t0) * 1000)
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            rows.append({"frame_path": f, "person": -1})
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        kps = r.keypoints.data.cpu().numpy()
        for j, (b, kp) in enumerate(zip(boxes, kps)):
            w, h = b[2] - b[0], b[3] - b[1]
            rows.append({"frame_path": f, "person": j, "aspect": w / max(h, 1e-6),
                         "angle": torso_angle(kp), "box_conf": float(r.boxes.conf[j])})
        if i % 500 == 0:
            print(f"{model_name}: {i}/{len(frames)} frames", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"features_{Path(model_name).stem}.csv", index=False)
    timing = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else {}
    timing[Path(model_name).name] = {"device": device, "median_ms": float(np.median(times)),
                          "p95_ms": float(np.percentile(times, 95)), "n": len(times),
                          "rss_gib": psutil.Process().memory_info().rss / 2**30}
    (OUT / "timing.json").write_text(json.dumps(timing, indent=2))


def frame_flags(feat: pd.DataFrame, theta: float, ratio: float) -> pd.Series:
    """frame_path -> True if any person satisfies the rule."""
    ok = (feat.aspect >= ratio) & (feat.angle.isna() | (feat.angle <= theta))
    return ok.groupby(feat.frame_path).any()


def decide(df: pd.DataFrame, flags: pd.Series, n: int, q: float) -> pd.Series:
    def one(clip: str) -> bool:
        fr = clip.split(";")[-n:]
        return np.mean([bool(flags.get(f, False)) for f in fr]) >= q
    return df.clip_paths.map(one)


def evaluate() -> None:
    df = samples()
    dev, test = df[df.split == "dev"], df[df.split == "test"]
    pos_dev = dev.label == "person_down"
    timing = json.loads((OUT / "timing.json").read_text())
    out_rows, chosen = [], {}
    for model_name in POSE_MODELS:
        feat = pd.read_csv(OUT / f"features_{Path(model_name).stem}.csv")
        feat = feat[feat.person >= 0]
        aspect_max = feat.groupby("frame_path").aspect.max()
        for n in [1, 4, 8]:
            best = None
            for theta, ratio, q in product(THETAS, RATIOS, QS if n > 1 else [1.0]):
                flags = frame_flags(feat, theta, ratio)
                y = decide(dev, flags, n, q)
                tp = (y & pos_dev).sum()
                rec = tp / pos_dev.sum()
                prec = tp / max(y.sum(), 1)
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
                key = (rec >= 0.95, f1, rec)
                if best is None or key > best[0]:
                    best = (key, theta, ratio, q)
            _, theta, ratio, q = best
            name = f"{Path(model_name).stem}-rule"
            chosen[f"{name}:{n}f"] = {"theta": theta, "ratio": ratio, "q": q,
                                      "dev_recall": best[0][2], "dev_f1": best[0][1]}
            flags = frame_flags(feat, theta, ratio)
            yes = decide(df, flags, n, q)
            score = df.image_path.map(lambda f: float(aspect_max.get(f, 0.0)))  # for AUROC only
            t = timing[Path(model_name).name]
            for (_, s), a, sc in zip(df.iterrows(), yes, score):
                out_rows.append({
                    "run_id": f"pose-{name}", "model": f"{name}:pose", "prompt": "rule",
                    "resolution": "native", "n_frames": n, "sample_id": s.sample_id, "split": s.split,
                    "label": s.label, "subtype": s.subtype, "viewpoint": s.viewpoint, "dataset": s.dataset,
                    "sequence": s.sequence, "camera": s.camera, "frame": s.frame,
                    "raw_text": "", "answer": "yes" if a else "no", "parse_mode": "strict",
                    "p_yes": sc, "e2e_ms": t["median_ms"] * n, "prompt_tokens": 0,
                })
    pd.DataFrame(out_rows).to_csv(OUT / "rows.csv", index=False)
    (OUT / "chosen_params.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        for m in POSE_MODELS:
            extract(m)
    else:
        evaluate()

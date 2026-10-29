"""Qualitative failure figure (UR Fall frames only: CC BY-NC-SA 4.0; MCF has no
licence, so its frames are not shown). Faces are blurred using YOLO11 face
keypoints (nose, eyes, ears); every tile is checked by eye afterwards.

  uv run --extra baseline --extra hf python scripts/failure_gallery.py
Writes paper/figures/failures.{pdf,png} and paper/generated/fig_failures_key.tex
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pdvlm.analysis import calibrate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# (config, resolution, sample_id): chosen from the calibrated failures of the
# best configuration (LFM2.5-VL-3B) and the smallest useful one (450M at 512 px).
TILES = [
    ("lfm-3b:Q8_0", "native", "urfall-adl-38-cam0-00279"),
    ("lfm-3b:Q8_0", "native", "urfall-adl-38-cam0-00219"),
    ("lfm-450m:Q8_0", "512", "urfall-adl-33-cam0-00143"),
    ("lfm-450m:Q8_0", "512", "urfall-fall-05-cam0-00151"),
    ("lfm-3b:Q8_0", "native", "urfall-adl-32-cam0-00121"),
    ("lfm-450m:Q8_0", "512", "urfall-adl-35-cam0-00083"),
    ("lfm-450m:Q8_0", "512", "urfall-adl-12-cam0-00181"),
    ("lfm-450m:Q8_0", "512", "urfall-adl-33-cam0-00173"),
]
NAMES = {"lfm-3b:Q8_0": "LFM-3B", "lfm-450m:Q8_0": "LFM-450M/512"}
# Faces the keypoint detector missed (people on the floor), located by eye:
# (x0, y0, x1, y1) in original 640x480 pixel coordinates.
MANUAL_BLUR = {
    "urfall-adl-33-cam0-00143": [(430, 385, 535, 480)],
    "urfall-fall-05-cam0-00151": [(365, 425, 465, 480)],
    "urfall-adl-32-cam0-00121": [(430, 370, 515, 455)],
    "urfall-adl-35-cam0-00083": [(395, 360, 490, 450)],
}
W = 400


def blur_faces(img: Image.Image, model) -> tuple[Image.Image, int]:
    r = model.predict(np.asarray(img), device="cpu", verbose=False)[0]
    out, n = img.copy(), 0
    if r.keypoints is None:
        return out, 0
    for kp, box in zip(r.keypoints.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        pts = kp[:5][kp[:5, 2] > 0.3][:, :2]
        if len(pts) == 0:
            continue
        cx, cy = pts.mean(axis=0)
        rad = max(18, 0.35 * (box[2] - box[0]) if len(pts) < 2 else 1.6 * np.ptp(pts, axis=0).max() + 12)
        x0, y0, x1, y1 = (int(max(0, cx - rad)), int(max(0, cy - rad)),
                          int(min(img.width, cx + rad)), int(min(img.height, cy + rad)))
        region = out.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(10))
        mask = Image.new("L", region.size, 0)
        ImageDraw.Draw(mask).ellipse([0, 0, region.width, region.height], fill=255)
        out.paste(region, (x0, y0), mask)
        n += 1
    return out, n


def main() -> None:
    from ultralytics import YOLO
    pose = YOLO(str(ROOT / "models/yolo/yolo11s-pose.pt"))
    d = pd.concat([pd.read_csv(ROOT / f"results/grid/{s}.csv") for s in ["stage1", "stage2"]])
    d["resolution"] = d.resolution.astype(str)
    labels = pd.read_csv(ROOT / "data/benchmark/v1/labels_test.csv").set_index("sample_id")
    tiles, key = [], []
    for i, (spec, res, sid) in enumerate(TILES, start=1):
        g = d[(d.model == spec) & (d.resolution == res) & (d.prompt == "A") & (d.n_frames == 1)]
        thr = calibrate(g[g.split == "dev"])
        row = g[g.sample_id == sid].iloc[0]
        alarm = row.p_yes >= thr
        kind = "false alarm" if alarm and row.label != "person_down" else "missed" if not alarm else "?"
        assert kind != "?", sid
        img = Image.open(ROOT / labels.loc[sid, "image_path"]).convert("RGB")
        img, nblur = blur_faces(img, pose)
        for box in MANUAL_BLUR.get(sid, []):
            region = img.crop(box).filter(ImageFilter.GaussianBlur(12))
            mask = Image.new("L", region.size, 0)
            ImageDraw.Draw(mask).ellipse([0, 0, region.width, region.height], fill=255)
            img.paste(region, box[:2], mask)
            nblur += 1
        img = img.resize((W, round(img.height * W / img.width)))
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, 26, 20], fill=(0, 0, 0))
        dr.text((6, 4), str(i), fill=(255, 255, 255))
        tiles.append(img)
        sub = labels.loc[sid, "subtype"].replace("_", " ").replace("lying floor deliberate", "deliberate lying on floor")
        key.append(f"({i}) {kind}, {sub}, {NAMES[spec]}, $p_\\text{{yes}}={row.p_yes:.2f}$ vs.\\ threshold {thr:.2f}")
        print(sid, kind, f"p_yes={row.p_yes:.3f} thr={thr:.3f} faces_blurred={nblur}")
    cols = 4
    th = max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * W + (cols - 1) * 4, 2 * th + 4), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (W + 4), (i // cols) * (th + 4)))
    out = ROOT / "paper/figures"
    sheet.save(out / "failures.png")
    sheet.save(out / "failures.pdf", resolution=200)
    (ROOT / "paper/generated/fig_failures_key.tex").write_text(
        "\\newcommand{\\KeyFailures}{" + "; ".join(key) + "}\n")


if __name__ == "__main__":
    main()

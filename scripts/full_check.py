"""Full by-eye check of every benchmark frame by the author (after the blind
100-frame check). Each frame is shown with its current label; the author
confirms it or changes / removes it.

  uv run python scripts/full_check.py            # build data/full_check/
  uv run python scripts/full_check.py --apply    # apply decisions.csv to the labels

--apply writes the checked labels back to data/benchmark/v1/labels_{test,dev}.csv
(column human_check: confirmed / changed; removed frames are dropped and listed in
removed.csv) and paper/generated/full_check.tex. scripts/analyze_grid.py reads
labels from those files, so results update without re-running any model.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/full_check"
BENCH = ROOT / "data/benchmark/v1"
SHORT = {"person_down": "down", "hard_negative": "hard", "normal": "normal"}
LONG = {v: k for k, v in SHORT.items()}

README = """FULL CHECK BY EYE - all benchmark frames (~30-40 min)
Do the blind 100-frame check in data/human_check FIRST.

Each frame shows its current label. Look at the frame and:
  Enter or Space   label is right  -> confirm, next frame
  D / H / N        label is wrong  -> set to down / hard / normal
  X                frame should not be in the benchmark (ambiguous, nobody
                   visible, mid-fall, ...) -> remove
  Left / Right     move between frames;  click the image to zoom

Definitions (same as the blind check):
  down    a person's torso is lying on the floor (mattress on the floor counts)
  hard    someone is low but torso not on the floor: crouching, kneeling,
          hands and knees, bending, sitting anywhere, lying on a bed or sofa
  normal  everyone visible is standing or walking
  Several people: down if anyone is down, else hard if anyone is low.

Progress is saved in the browser. At the end click "Download decisions" and
save the file into this folder as decisions.csv, then tell Claude.
"""

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Full check by eye</title>
<style>
  body { font: 15px -apple-system, system-ui, sans-serif; margin: 0; background: #fcfcfb; color: #0b0b0b; }
  header { display: flex; gap: 16px; align-items: center; padding: 10px 16px; border-bottom: 1px solid #e7e6e2; flex-wrap: wrap; }
  #prog { color: #52514e; }
  #lab { font-size: 22px; font-weight: 600; padding: 6px 14px; border: 2px solid #0b0b0b; border-radius: 6px; }
  #sub { color: #52514e; }
  #dec { font-weight: 600; }
  main { display: flex; justify-content: center; padding: 12px; }
  img { max-height: 72vh; max-width: 96vw; cursor: zoom-in; border: 1px solid #e7e6e2; }
  img.zoom { max-height: none; max-width: none; width: 200%; cursor: zoom-out; }
  .hint { text-align: center; color: #52514e; font-size: 13px; padding: 6px; }
  button { font: inherit; padding: 6px 12px; border: 1px solid #52514e; background: #fff; border-radius: 6px; cursor: pointer; }
  details { max-width: 820px; margin: 0 auto; color: #52514e; font-size: 13px; }
</style></head><body>
<header><strong>Full check by eye</strong>
<span>current label: <span id="lab"></span> <span id="sub"></span></span>
<span id="dec"></span><span id="prog"></span><button id="dl">Download decisions</button></header>
<details><summary>Instructions and definitions</summary><pre>__README__</pre></details>
<div class="hint">Enter/Space = correct &middot; D / H / N = change label &middot; X = remove &middot; &larr; &rarr; navigate &middot; click image to zoom</div>
<main><img id="im" alt="frame"></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "full_check_decisions_v1";
let dec = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;
function show() {
  const it = ITEMS[i];
  const im = document.getElementById("im");
  im.src = "images/" + it.file; im.classList.remove("zoom");
  document.getElementById("lab").textContent = it.label;
  document.getElementById("sub").textContent = "(" + it.subtype + ")";
  const d = dec[it.no];
  document.getElementById("dec").textContent = d ? ("your decision: " + d) : "";
  const done = Object.keys(dec).length;
  document.getElementById("prog").textContent = `frame ${i + 1} / ${ITEMS.length} \\u00b7 checked ${done} / ${ITEMS.length}`;
}
function decide(v) {
  dec[ITEMS[i].no] = v; localStorage.setItem(KEY, JSON.stringify(dec));
  if (i < ITEMS.length - 1) i++; show();
}
document.getElementById("im").onclick = e => e.target.classList.toggle("zoom");
document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (k === "enter" || k === " ") { e.preventDefault(); decide("ok"); }
  else if (k === "d") decide("down"); else if (k === "h") decide("hard");
  else if (k === "n") decide("normal"); else if (k === "x") decide("remove");
  else if (e.key === "ArrowRight" && i < ITEMS.length - 1) { i++; show(); }
  else if (e.key === "ArrowLeft" && i > 0) { i--; show(); }
});
document.getElementById("dl").onclick = () => {
  const missing = ITEMS.filter(it => !dec[it.no]);
  if (missing.length && !confirm(`${missing.length} frames are not checked yet (first: ${missing[0].no}). Download anyway?`)) return;
  const csv = "no,decision\\n" + ITEMS.map(it => `${it.no},${dec[it.no] || ""}`).join("\\n") + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "decisions.csv"; a.click();
};
show();
</script></body></html>
"""


def labels() -> pd.DataFrame:
    return pd.concat([pd.read_csv(BENCH / f"labels_{s}.csv") for s in ("test", "dev")], ignore_index=True)


def make() -> None:
    df = labels().sort_values(["dataset", "sequence", "camera", "frame"]).reset_index(drop=True)
    df["no"] = range(1, len(df) + 1)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    items = []
    for _, r in df.iterrows():
        f = f"{r.no:04d}.png"
        shutil.copy2(ROOT / r.image_path, OUT / "images" / f)
        items.append({"no": int(r.no), "file": f, "label": SHORT[r.label], "subtype": r.subtype.replace("_", " ")})
    (OUT / "README.txt").write_text(README)
    (OUT / "label.html").write_text(HTML.replace("__ITEMS__", json.dumps(items)).replace("__README__", README))
    df[["no", "sample_id"]].to_csv(OUT / "index.csv", index=False)
    print(f"wrote {len(df)} frames to {OUT}")


def apply() -> None:
    idx = pd.read_csv(OUT / "index.csv")
    dec = pd.read_csv(OUT / "decisions.csv").merge(idx, on="no")
    if dec.decision.isna().any():
        raise SystemExit(f"{int(dec.decision.isna().sum())} frames not checked yet")
    d = dict(zip(dec.sample_id, dec.decision.str.strip().str.lower()))
    n = {"confirmed": 0, "changed": 0, "removed": 0}
    removed = []
    for split in ("test", "dev"):
        path = BENCH / f"labels_{split}.csv"
        lab = pd.read_csv(path)
        keep = []
        for _, r in lab.iterrows():
            v = d[r.sample_id]
            if v == "remove":
                n["removed"] += 1
                removed.append(r)
                continue
            r = r.copy()
            if v == "ok" or LONG.get(v) == r.label:
                r["human_check"] = "confirmed"
                n["confirmed"] += 1
            else:
                r["notes"] = f"by-eye check: {SHORT[r.label]} -> {v}; " + str(r.get("notes", "") or "")
                r["label"], r["subtype"] = LONG[v], ("standing" if v == "normal" else "unspecified")
                r["human_check"] = "changed"
                n["changed"] += 1
            keep.append(r)
        pd.DataFrame(keep).to_csv(path, index=False)
    if removed:
        pd.DataFrame(removed).to_csv(BENCH / "removed_by_eye.csv", index=False)
    (ROOT / "paper/generated/full_check.tex").write_text(
        "% Generated by scripts/full_check.py --apply -- do not edit.\n\\fullchecktrue\n"
        + "".join(f"\\newcommand{{\\Full{k.capitalize()}}}{{{v}}}\n" for k, v in n.items()))
    print(n, "-> now re-run: scripts/analyze_grid.py, scripts/make_paper_tables.py, and rebuild the paper")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    apply() if ap.parse_args().apply else make()

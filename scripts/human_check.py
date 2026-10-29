"""Blind label spot-check by a human annotator.

Samples 100 test frames (stratified by label), copies them to
data/human_check/images/NNN.png in random order, and writes a browser labelling
page plus a blank form. The assigned labels go to a separate answer key outside
that folder, so the check is blind.

  uv run python scripts/human_check.py            # build data/human_check/
  uv run python scripts/human_check.py --score    # after labelling: agreement + Cohen's kappa
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/human_check"
KEY = ROOT / "data/interim/bench/human_check_key/answer_key.csv"
SEED = 7
PER_LABEL = {"person_down": 34, "hard_negative": 33, "normal": 33}

README = """HUMAN LABEL CHECK - 100 frames, about 30-45 minutes

For each frame decide ONE label about the whole image:

  D  down    a person's torso is lying on the floor or ground
             (after a fall, or lying down on the floor on purpose)
  H  hard    someone is low but their torso is NOT on the floor:
             crouching, kneeling (also on hands and knees), bending over,
             sitting (chair, sofa, bed or floor), lying on a bed or sofa
  N  normal  everyone visible is standing or walking
  U  unsure  you cannot tell (person too small, too dark, ambiguous)

Rules
  - If several people are visible, the image is "down" if ANY of them is down,
    otherwise "hard" if any of them is low, otherwise "normal".
  - Label what the frame shows, not what you think happened before or after.
  - Lying on a mattress that is on the floor counts as lying on the floor.
  - Don't overthink it: use U rather than guessing.

How
  1. Double-click label.html (opens in your browser).
  2. Press D / H / N / U for each image. Left/right arrows move between images;
     click the image to zoom (people are small in some wide-angle views).
  3. Answers are saved in the browser as you go. When all 100 are done, click
     "Download answers" and save the file into this folder as  answers.csv
  4. Tell Claude it is done.

Alternative without the browser: open images/ in Preview and fill in form.csv
(column your_label: down / hard / normal / unsure).

Do not open data/interim/bench/human_check_key/ until you have finished.
"""

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Human label check</title>
<style>
  body { font: 15px -apple-system, system-ui, sans-serif; margin: 0; background: #fcfcfb; color: #0b0b0b; }
  header { display: flex; gap: 16px; align-items: center; padding: 10px 16px; border-bottom: 1px solid #e7e6e2; }
  #prog { color: #52514e; }
  main { display: flex; justify-content: center; padding: 12px; }
  img { max-height: 74vh; max-width: 96vw; cursor: zoom-in; border: 1px solid #e7e6e2; }
  img.zoom { max-height: none; max-width: none; width: 200%; cursor: zoom-out; }
  .keys { display: flex; gap: 8px; justify-content: center; padding: 8px; flex-wrap: wrap; }
  button { font: inherit; padding: 8px 14px; border: 1px solid #52514e; background: #fff; border-radius: 6px; cursor: pointer; }
  button.on { background: #0b0b0b; color: #fff; }
  .hint { text-align: center; color: #52514e; font-size: 13px; padding-bottom: 12px; }
  details { max-width: 820px; margin: 0 auto 16px; color: #52514e; font-size: 13px; }
</style></head><body>
<header><strong>Human label check</strong><span id="prog"></span>
<button id="dl">Download answers</button></header>
<details><summary>Label definitions</summary><pre>__README__</pre></details>
<div class="keys">
  <button data-l="down">D &mdash; down</button><button data-l="hard">H &mdash; hard</button>
  <button data-l="normal">N &mdash; normal</button><button data-l="unsure">U &mdash; unsure</button>
</div>
<div class="hint">Keys: D H N U to label (moves to next) &middot; &larr; &rarr; to navigate &middot; click image to zoom</div>
<main><img id="im" alt="frame"></main>
<script>
const N = __N__;
const KEY = "human_check_answers_v1";
let ans = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 1;
const pad = n => String(n).padStart(3, "0");
function show() {
  document.getElementById("im").src = "images/" + pad(i) + ".png";
  document.getElementById("im").classList.remove("zoom");
  const done = Object.keys(ans).length;
  document.getElementById("prog").textContent = `Image ${i} / ${N} \\u00b7 labelled ${done} / ${N}` + (ans[i] ? ` \\u00b7 this one: ${ans[i]}` : "");
  document.querySelectorAll("button[data-l]").forEach(b => b.classList.toggle("on", b.dataset.l === ans[i]));
}
function label(l) { ans[i] = l; localStorage.setItem(KEY, JSON.stringify(ans)); if (i < N) i++; show(); }
document.querySelectorAll("button[data-l]").forEach(b => b.onclick = () => label(b.dataset.l));
document.getElementById("im").onclick = e => e.target.classList.toggle("zoom");
document.addEventListener("keydown", e => {
  const m = { d: "down", h: "hard", n: "normal", u: "unsure" }[e.key.toLowerCase()];
  if (m) label(m);
  if (e.key === "ArrowRight" && i < N) { i++; show(); }
  if (e.key === "ArrowLeft" && i > 1) { i--; show(); }
});
document.getElementById("dl").onclick = () => {
  const missing = [...Array(N).keys()].map(k => k + 1).filter(k => !ans[k]);
  if (missing.length && !confirm(`${missing.length} images are not labelled yet (first: ${missing[0]}). Download anyway?`)) return;
  const csv = "no,your_label\\n" + [...Array(N).keys()].map(k => `${k + 1},${ans[k + 1] || ""}`).join("\\n") + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "answers.csv"; a.click();
};
show();
</script></body></html>
"""


def make() -> None:
    df = pd.read_csv(ROOT / "data/benchmark/v1/labels_test.csv")
    pick = pd.concat([df[df.label == k].sample(n, random_state=SEED) for k, n in PER_LABEL.items()])
    pick = pick.sample(frac=1, random_state=SEED).reset_index(drop=True)
    pick["no"] = range(1, len(pick) + 1)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    for _, r in pick.iterrows():
        shutil.copy2(ROOT / r.image_path, OUT / "images" / f"{r.no:03d}.png")
    (OUT / "README.txt").write_text(README)
    (OUT / "label.html").write_text(HTML.replace("__N__", str(len(pick))).replace("__README__", README))
    pick[["no"]].assign(your_label="").to_csv(OUT / "form.csv", index=False)
    KEY.parent.mkdir(parents=True, exist_ok=True)
    pick[["no", "sample_id", "label", "subtype", "dataset", "viewpoint"]].to_csv(KEY, index=False)
    old = ROOT / "data/interim/bench/human_check"  # earlier version with sheets and the key side by side
    if old.exists():
        shutil.rmtree(old)
    print(f"wrote {len(pick)} images to {OUT}; answer key in {KEY}")
    print(pick.label.value_counts().to_string())


def score() -> None:
    key = pd.read_csv(KEY)
    path = OUT / "answers.csv" if (OUT / "answers.csv").exists() else OUT / "form.csv"
    form = pd.read_csv(path)
    m = key.merge(form, on="no")
    m["yours"] = m.your_label.fillna("").str.strip().str.lower().map(
        {"down": "person_down", "hard": "hard_negative", "normal": "normal", "unsure": "unsure"})
    sure = m[m.yours.isin(["person_down", "hard_negative", "normal"])]
    labels = ["person_down", "hard_negative", "normal"]
    po = (sure.label == sure.yours).mean()
    pe = sum((sure.label == k).mean() * (sure.yours == k).mean() for k in labels)
    kappa = (po - pe) / (1 - pe)
    binary = ((sure.label == "person_down") == (sure.yours == "person_down")).mean()
    res = {"file": path.name, "answered": len(sure), "unsure": int((m.yours == "unsure").sum()),
           "blank": int(m.yours.isna().sum()), "agreement": po, "kappa": kappa, "binary_agreement": binary}
    print(json.dumps(res, indent=2))
    print(pd.crosstab(sure.label, sure.yours, rownames=["assigned"], colnames=["human"]))
    dis = m[(m.yours != m.label) & m.yours.notna()]
    print("\ndisagreements / unsure:\n", dis[["no", "sample_id", "label", "subtype", "yours"]].to_string(index=False))
    (KEY.parent / "score.json").write_text(json.dumps(res, indent=2))
    (ROOT / "paper/generated/human_check.tex").write_text(
        "% Generated by scripts/human_check.py --score -- do not edit.\n"
        f"\\newcommand{{\\HumanAgree}}{{{100 * po:.0f}}}\n"
        f"\\newcommand{{\\HumanKappa}}{{{kappa:.2f}}}\n"
        f"\\newcommand{{\\HumanUnsure}}{{{res['unsure']}}}\n"
        f"\\newcommand{{\\HumanAnswered}}{{{len(sure)}}}\n"
        f"\\newcommand{{\\HumanBinaryAgree}}{{{100 * binary:.0f}}}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    score() if ap.parse_args().score else make()

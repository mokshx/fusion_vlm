"""Assemble the Step 0 pilot set (30 UR Fall frames) from reviewed candidates.

Every frame below was checked by eye (contact sheets in data/interim/review/).
Selection rules: one frame per sequence, no sequence reused across classes,
spread over all four ADL rooms and both lighting conditions; person_down split
5 side (cam0) / 5 overhead (cam1).

ADL sequences where the subject lies on the floor on purpose (adl-30..35,
38..40) are person_down / lying_floor_deliberate (docs/labelling_policy.md).
They go in a separate file so the balanced 30-frame set stays unchanged; being
in other rooms than the falls, they also probe the scene confound.

Output: data/benchmark/step0/{frames/*.png, labels.csv, labels_floor_deliberate.csv}
"""

import csv
import shutil
from pathlib import Path

SRC = Path("data/interim/urfall_candidates")
OUT = Path("data/benchmark/step0")

# (file, label, subtype, notes)
SELECTION = [
    ("fall-01-cam0-rgb-137.png", "person_down", "fall", ""),
    ("fall-02-cam0-rgb-090.png", "person_down", "fall", ""),
    ("fall-03-cam0-rgb-197.png", "person_down", "fall", ""),
    ("fall-04-cam0-rgb-070.png", "person_down", "fall", ""),
    ("fall-05-cam0-rgb-130.png", "person_down", "fall", "legs raised"),
    ("fall-06-cam1-rgb-090.png", "person_down", "fall", "cam1 frame at 90% of sequence; visually confirmed"),
    ("fall-07-cam1-rgb-140.png", "person_down", "fall", "cam1 frame at 90% of sequence; visually confirmed"),
    ("fall-08-cam1-rgb-082.png", "person_down", "fall", "cam1 frame at 90% of sequence; visually confirmed"),
    ("fall-09-cam1-rgb-166.png", "person_down", "fall", "cam1 frame at 90% of sequence; visually confirmed"),
    ("fall-10-cam1-rgb-117.png", "person_down", "fall", "cam1 frame at 90% of sequence; visually confirmed"),
    ("adl-01-cam0-rgb-144.png", "hard_negative", "crouching", ""),
    ("adl-04-cam0-rgb-098.png", "hard_negative", "bending", "bending over to pick up"),
    ("adl-07-cam0-rgb-161.png", "hard_negative", "sitting_chair", ""),
    ("adl-10-cam0-rgb-250.png", "hard_negative", "lying_furniture", "lying on bed"),
    ("adl-14-cam0-rgb-178.png", "hard_negative", "kneeling", "low light"),
    ("adl-22-cam0-rgb-205.png", "hard_negative", "lying_furniture", "lying on sofa; low light"),
    ("adl-24-cam0-rgb-031.png", "hard_negative", "crouching", "squatting, tying shoe"),
    ("adl-34-cam0-rgb-056.png", "hard_negative", "kneeling", "hands and knees"),
    ("adl-36-cam0-rgb-210.png", "hard_negative", "sitting_furniture", "sitting on bed edge"),
    ("adl-37-cam0-rgb-310.png", "hard_negative", "lying_furniture", "lying on bed"),
    ("adl-02-cam0-rgb-055.png", "normal", "standing", "small, in doorway"),
    ("adl-05-cam0-rgb-157.png", "normal", "standing", "close to camera"),
    ("adl-09-cam0-rgb-065.png", "normal", "walking", ""),
    ("adl-12-cam0-rgb-142.png", "normal", "walking", "low light"),
    ("adl-17-cam0-rgb-108.png", "normal", "standing", "low light"),
    ("adl-26-cam0-rgb-083.png", "normal", "walking", ""),
    ("adl-28-cam0-rgb-009.png", "normal", "standing", ""),
    ("adl-31-cam0-rgb-035.png", "normal", "standing", ""),
    ("adl-33-cam0-rgb-030.png", "normal", "standing", ""),
    ("adl-39-cam0-rgb-053.png", "normal", "standing", "in doorway"),
]


FLOOR_DELIBERATE = [
    ("adl-30-cam0-rgb-310.png", "person_down", "lying_floor_deliberate", "curled on side; partly out of frame"),
    ("adl-31-cam0-rgb-183.png", "person_down", "lying_floor_deliberate", ""),
    ("adl-32-cam0-rgb-154.png", "person_down", "lying_floor_deliberate", ""),
    ("adl-33-cam0-rgb-142.png", "person_down", "lying_floor_deliberate", "mostly out of frame / occluded"),
    ("adl-34-cam0-rgb-140.png", "person_down", "lying_floor_deliberate", "prone on elbows"),
    ("adl-35-cam0-rgb-134.png", "person_down", "lying_floor_deliberate", "prone"),
    ("adl-38-cam0-rgb-275.png", "person_down", "lying_floor_deliberate", "partly out of frame"),
    ("adl-39-cam0-rgb-232.png", "person_down", "lying_floor_deliberate", "partly out of frame"),
    ("adl-40-cam0-rgb-295.png", "person_down", "lying_floor_deliberate", "partly out of frame"),
]


def write_set(selection, cands, name: str, unique_sequences: bool) -> None:
    rows = []
    for fname, label, subtype, notes in selection:
        c = cands[fname]
        shutil.copy2(SRC / fname, OUT / "frames" / fname)
        rows.append({
            "sample_id": fname.removesuffix(".png"),
            "image_path": f"frames/{fname}",
            "label": label, "subtype": subtype,
            "viewpoint": "overhead" if c["camera"] == "1" else "side",
            "dataset": "urfall", "sequence": c["sequence"], "camera": c["camera"],
            "frame": c["frame"], "source_label": c["source_label"],
            "hh_ratio": c["hh_ratio"], "notes": notes,
        })
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    seqs = [r["sequence"] for r in rows]
    if unique_sequences:
        assert len(set(seqs)) == len(seqs), "a sequence is used twice"
    print(f"wrote {len(rows)} samples to {OUT / name}")


def main() -> None:
    cands = {r["file"]: r for r in csv.DictReader(open(SRC / "candidates.csv"))}
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    write_set(SELECTION, cands, "labels.csv", unique_sequences=True)
    # adl-34 also supplies a hard negative (hands and knees, frame 56) to the main set;
    # different frame, so not a duplicate sample, but keep it in mind for splits.
    write_set(FLOOR_DELIBERATE, cands, "labels_floor_deliberate.csv", unique_sequences=True)

if __name__ == "__main__":
    main()

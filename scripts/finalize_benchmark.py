"""Benchmark step 3: merge review decisions, split dev/test by sequence, write
the final label files.

Split: whole source sequences (MCF scenario with all 8 cameras; UR Fall
sequence with both cameras) go to dev or test, never both. About 25% of
sequences per dataset and sequence type go to dev; the dev set is used only to
choose decision thresholds. Deterministic (fixed seed).

Usage: uv run python scripts/finalize_benchmark.py
Writes data/benchmark/v1/{labels_test.csv, labels_dev.csv, summary.txt}
"""

import random
from pathlib import Path

import pandas as pd

BENCH = Path("data/interim/bench")
OUT = Path("data/benchmark/v1")
SEED = 20261026
DEV_FRAC = 0.25


def main() -> None:
    c = pd.read_csv(BENCH / "candidates.csv")
    r = pd.read_csv(BENCH / "review.csv")
    df = c.merge(r, on="cand_id", validate="one_to_one")
    df = df[df.label != "reject"].copy()

    # Sequence groups to stratify the split: MCF scenarios; UR falls; UR ADLs.
    def group(row) -> str:
        if row.dataset == "mcf":
            return "mcf"
        return "urfall-" + row.sequence.split("-")[0]

    df["group"] = df.apply(group, axis=1)
    rng = random.Random(SEED)
    dev_seqs = set()
    for g, sub in df.groupby("group"):
        seqs = sorted(sub.sequence.unique())
        rng.shuffle(seqs)
        dev_seqs.update(seqs[: round(DEV_FRAC * len(seqs))])
    df["split"] = df.sequence.map(lambda s: "dev" if s in dev_seqs else "test")

    df["sample_id"] = df.apply(lambda x: f"{x.dataset}-{x.sequence}-{x.camera}-{int(x.frame):05d}", axis=1)
    assert df.sample_id.is_unique
    df["notes"] = df.note.fillna("")
    cols = ["sample_id", "split", "label", "subtype", "viewpoint", "dataset", "sequence", "camera",
            "frame", "t_s", "source_label", "image_path", "clip_paths", "notes", "cand_id"]
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ["test", "dev"]:
        df[df.split == split][cols].sort_values("sample_id").to_csv(OUT / f"labels_{split}.csv", index=False)

    lines = []
    for split in ["test", "dev"]:
        s = df[df.split == split]
        lines.append(f"== {split}: {len(s)} samples, {s.sequence.nunique()} sequences")
        lines.append(s.groupby(["label", "subtype"]).size().to_string())
        lines.append(s.groupby(["dataset", "label"]).size().unstack(fill_value=0).to_string())
        lines.append(s.groupby(["viewpoint", "label"]).size().unstack(fill_value=0).to_string())
    leak = set(df[df.split == "dev"].sequence) & set(df[df.split == "test"].sequence)
    lines.append(f"sequences in both splits: {len(leak)}")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

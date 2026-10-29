"""Run the staged experiment grid on this machine (resumable; safe to re-run).

  stage1  5 models x prompts A/B/B2, Q8_0, native resolution, 1 frame
  stage2  per model: best prompt (dev AUROC, from stage 1) x {16-bit, Q8_0, Q4_K_M}
          x {native, 512, 256}  (Q8_0 native already done in stage 1)
  stage3  per model: best prompt x {4, 8} frames, Q8_0, native

All runs: Metal GPU (-ngl 99), --no-mmap (so peak RSS = memory actually used),
test + dev splits, greedy decoding. Waits for model files that are still
downloading. Usage: uv run python scripts/run_grid.py stage1 [stage2 stage3]
"""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from pdvlm.models import gguf_paths

OUT = Path("results/grid")
LABELS = ["data/benchmark/v1/labels_test.csv", "data/benchmark/v1/labels_dev.csv"]
MODELS = ["lfm-450m", "lfm-1.6b", "lfm-3b", "qwen3.5-2b", "qwen3.5-4b"]
SIXTEEN_BIT = {"lfm-450m": "F16", "lfm-1.6b": "F16", "lfm-3b": "F16",
               "qwen3.5-2b": "BF16", "qwen3.5-4b": "BF16"}  # Unsloth Qwen GGUFs have BF16 only


def wait_for(spec: str) -> None:
    model, mmproj = gguf_paths(spec)
    while not (model.exists() and mmproj.exists()):
        print(f"waiting for {model.name} / {mmproj.name} ...", flush=True)
        time.sleep(120)


def run(spec: str, prompts: list[str], out: str, resolution: int | None = None, n_frames: int = 1) -> None:
    wait_for(spec)
    cmd = ["uv", "run", "python", "scripts/run_eval.py", "--labels", *LABELS, "--models", spec,
           "--prompts", *prompts, "--ngl", "99", "--server-args=--no-mmap",
           "--n-frames", str(n_frames), "--out", str(OUT / out)]
    if resolution:
        cmd += ["--resolution", str(resolution)]
    print(time.strftime("%H:%M:%S"), " ".join(cmd[4:]), flush=True)
    log = open(OUT / f"{out.removesuffix('.csv')}.log", "a")
    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def auroc(g: pd.DataFrame) -> float:
    pos = g[g.label == "person_down"].p_yes.to_numpy()
    neg = g[g.label != "person_down"].p_yes.to_numpy()
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def best_prompt(model: str) -> str:
    """Prompt with the highest dev-split AUROC in stage 1 (ties -> A, the simplest)."""
    df = pd.read_csv(OUT / "stage1.csv")
    dev = df[(df.model == f"{model}:Q8_0") & (df.split == "dev")]
    scores = {p: auroc(g) for p, g in dev.groupby("prompt")}
    return max(sorted(scores), key=lambda p: (round(scores[p], 4), p == "A"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stages = sys.argv[1:] or ["stage1", "stage2", "stage3"]
    if "stage1" in stages:
        for m in MODELS:
            run(f"{m}:Q8_0", ["A", "B", "B2"], "stage1.csv")
    if "stage2" in stages:
        for m in MODELS:
            p = best_prompt(m)
            print(f"{m}: best prompt on dev = {p}", flush=True)
            for q in [SIXTEEN_BIT[m], "Q8_0", "Q4_K_M"]:
                for res in [None, 512, 256]:
                    if q == "Q8_0" and res is None:
                        continue  # stage 1
                    run(f"{m}:{q}", [p], "stage2.csv", resolution=res)
    if "stage3" in stages:
        for m in MODELS:
            p = best_prompt(m)
            for n in [4, 8]:
                run(f"{m}:Q8_0", [p], "stage3.csv", n_frames=n)
    print(time.strftime("%H:%M:%S"), "grid finished", flush=True)


if __name__ == "__main__":
    main()

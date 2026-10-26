"""Run model configs x prompts over a labelled sample set; append one row per
(sample, config) to a CSV and write a JSON sidecar per config with environment,
backend info and peak memory.

Example (Step 0, CPU-only on the Mac):
  uv run python scripts/run_eval.py --labels data/labels/step0.csv \
      --models lfm-450m:Q8_0 lfm-3b:Q8_0 --prompts A B --ngl 0 --threads 10 \
      --out results/step0/mac_cpu.csv

On Kaggle (transformers reference):
  python scripts/run_eval.py --labels data/labels/step0.csv \
      --models lfm-450m:hf lfm-3b:hf --prompts A B --out results/step0/kaggle.csv
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import time
import uuid
from pathlib import Path

from PIL import Image

from pdvlm import sysinfo
from pdvlm.backends.base import resize_longest
from pdvlm.models import make_backend
from pdvlm.prompts import build_prompt

LABEL_COLS = ["sample_id", "label", "subtype", "viewpoint", "dataset", "sequence", "camera", "frame"]


def load_samples(path: Path, root: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["frames"] = [root / p for p in r["image_path"].split(";")]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, default=Path("."))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", default=["A", "B"])
    ap.add_argument("--resolution", type=int, default=None, help="longest side px; omit for native")
    ap.add_argument("--ngl", type=int, default=0, help="llama.cpp GPU layers (0 = CPU only, 99 = all)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3, help="discarded requests before measuring")
    ap.add_argument("--server-args", default="", help='extra llama-server flags, e.g. "--no-mmap --no-repack"')
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if platform.system() == "Darwin":
        # Idle sleep pauses every process and silently ruins latency numbers
        # (seen in Step 0: one request "took" 1015 s). Hold an assertion while we run.
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    samples = load_samples(args.labels, args.images_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = sysinfo.collect()
    new = not args.out.exists()
    with open(args.out, "a", newline="") as f:
        writer = None
        for spec in args.models:
            run_id = uuid.uuid4().hex[:8]
            is_hf = spec.endswith(":hf")
            kw = {"max_tokens": args.max_tokens}
            if not is_hf:
                kw.update(n_gpu_layers=args.ngl, threads=args.threads,
                          extra_args=args.server_args.split(),
                          log_path=args.out.parent / f"server-{run_id}.log")
            backend = make_backend(spec, **kw)
            info = backend.info()
            print(f"[{run_id}] {spec} loaded in {info['load_s']:.1f}s", flush=True)

            warm = [resize_longest(Image.open(p), args.resolution) for p in samples[0]["frames"]]
            for _ in range(args.warmup):
                backend.predict(warm, build_prompt(args.prompts[0], len(warm)))

            t_start = time.time()
            for prompt_key in args.prompts:
                for s in samples:
                    imgs = [resize_longest(Image.open(p), args.resolution) for p in s["frames"]]
                    pred = backend.predict(imgs, build_prompt(prompt_key, len(imgs)))
                    row = {
                        "run_id": run_id, "model": spec, "prompt": prompt_key,
                        "resolution": args.resolution or "native", "n_frames": len(imgs),
                        "input_w": imgs[-1].width, "input_h": imgs[-1].height,
                        **{c: s.get(c, "") for c in LABEL_COLS},
                        **pred.to_row(),
                    }
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
                        if new:
                            writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                    mark = "ok " if (pred.answer == "yes") == (s["label"] == "person_down") else "ERR"
                    print(f"  {mark} {prompt_key} {s['sample_id']:<28} {s['label']:<14} -> "
                          f"{pred.answer:<8} p_yes={pred.p_yes if pred.p_yes is None else round(pred.p_yes, 3)} "
                          f"{pred.e2e_ms:.0f}ms  {pred.raw_text!r}", flush=True)
            mem = backend.close()
            side = {"run_id": run_id, "model": spec, "args": {k: str(v) for k, v in vars(args).items()},
                    "env": env, "backend": info, "memory": mem,
                    "wall_s": time.time() - t_start}
            (args.out.parent / f"run-{run_id}.json").write_text(json.dumps(side, indent=2))
            print(f"[{run_id}] done; memory {mem}", flush=True)


if __name__ == "__main__":
    main()

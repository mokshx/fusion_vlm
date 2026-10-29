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

LABEL_COLS = ["sample_id", "split", "label", "subtype", "viewpoint", "dataset", "sequence", "camera", "frame"]


def load_samples(paths: list[Path], root: Path, n_frames: int) -> list[dict]:
    """Rows from one or more label files. With n_frames > 1 the last n frames of
    the sample's clip (oldest first, ending at the labelled frame) are used."""
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    for r in rows:
        if n_frames > 1:
            clip = r["clip_paths"].split(";")
            if len(clip) < n_frames:
                raise ValueError(f"{r['sample_id']}: clip has {len(clip)} frames, need {n_frames}")
            r["frames"] = [root / p for p in clip[-n_frames:]]
        else:
            r["frames"] = [root / r["image_path"].split(";")[-1]]
    return rows


def done_keys(out: Path) -> set[tuple]:
    """(model, prompt, resolution, n_frames, sample_id) already in the output CSV."""
    if not out.exists():
        return set()
    return {(r["model"], r["prompt"], r["resolution"], r["n_frames"], r["sample_id"])
            for r in csv.DictReader(open(out))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--images-root", type=Path, default=Path("."))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", default=["A", "B"])
    ap.add_argument("--resolution", type=int, default=None, help="longest side px; omit for native")
    ap.add_argument("--n-frames", type=int, default=1, help="frames per decision (1, 4 or 8)")
    ap.add_argument("--limit", type=int, default=None, help="random subset of N samples (fixed seed), e.g. for CPU timing")
    ap.add_argument("--log-times", action="store_true", help="add wall-clock start/end of each request (power alignment)")
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

    samples = load_samples(args.labels, args.images_root, args.n_frames)
    if args.limit:
        import random
        samples = random.Random(0).sample(samples, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = sysinfo.collect()
    done = done_keys(args.out)  # resume: skip rows already written
    res_key = str(args.resolution or "native")
    header = next(csv.reader(open(args.out)), None) if args.out.exists() else None
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n") if header else None
        for spec in args.models:
            todo = [(p, s) for p in args.prompts for s in samples
                    if (spec, p, res_key, str(args.n_frames), s["sample_id"]) not in done]
            if not todo:
                print(f"{spec}: all {len(args.prompts) * len(samples)} rows already done", flush=True)
                continue
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
            print(f"[{run_id}] {len(todo)} requests to run", flush=True)
            for prompt_key, s in todo:
                if True:
                    imgs = [resize_longest(Image.open(p), args.resolution) for p in s["frames"]]
                    t_req0 = time.time()
                    pred = backend.predict(imgs, build_prompt(prompt_key, len(imgs)))
                    t_req1 = time.time()
                    row = {
                        "run_id": run_id, "model": spec, "prompt": prompt_key,
                        "resolution": args.resolution or "native", "n_frames": len(imgs),
                        "input_w": imgs[-1].width, "input_h": imgs[-1].height,
                        **{c: s.get(c, "") for c in LABEL_COLS},
                        **pred.to_row(),
                    }
                    if args.log_times:
                        row["t_start"], row["t_end"] = t_req0, t_req1
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
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

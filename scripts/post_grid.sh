#!/usr/bin/env bash
# Runs after scripts/run_grid.py finishes: re-time battery-affected configs, pose
# baseline, CPU-only timing of Pareto-optimal configs, then regenerate tables,
# figures and the PDF. Resumable; each step skips work already done.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "$(date +%H:%M:%S) $*"; }
while pgrep -f "scripts/run_grid.py" >/dev/null; do sleep 60; done
log "grid finished; starting post-grid steps"
T=data/benchmark/v1/labels_test.csv

uv run python scripts/analyze_grid.py > results/grid/analyze_pre.log 2>&1
log "re-timing battery-affected configurations"
uv run python - <<'PY' > results/grid/retime_list.txt
import pandas as pd
s = pd.read_csv("results/grid/summary.csv")
for _, r in s[s.battery & (s.prompt == "A")].iterrows():
    print(r.spec, r.resolution, r.n_frames)
PY
while read -r spec res nf; do
  extra=""; [ "$res" != "native" ] && extra="--resolution $res"
  uv run python scripts/run_eval.py --labels $T --models "$spec" --prompts A --ngl 99 \
    --server-args=--no-mmap --n-frames "$nf" $extra --out results/grid/retime.csv >> results/grid/retime.log 2>&1
done < results/grid/retime_list.txt

log "pose baseline"
uv run --extra baseline --extra hf python scripts/pose_baseline.py extract > results/baseline.log 2>&1
uv run --extra baseline --extra hf python scripts/pose_baseline.py evaluate >> results/baseline.log 2>&1

log "analysis for frontier"
uv run python scripts/analyze_grid.py > results/grid/analyze_mid.log 2>&1
log "CPU-only timing of Pareto-optimal configurations"
uv run python - <<'PY' > results/grid/cpu_list.txt
import json
for c in json.load(open("results/grid/frontier.json")):
    print(c["spec"], c["resolution"], c["n_frames"])
PY
while read -r spec res nf; do
  extra=""; [ "$res" != "native" ] && extra="--resolution $res"
  uv run python scripts/run_eval.py --labels $T --models "$spec" --prompts A --ngl 0 --threads 10 \
    --server-args=--no-mmap --n-frames "$nf" --limit 60 $extra --out results/grid/cpu.csv >> results/grid/cpu.log 2>&1
done < results/grid/cpu_list.txt

log "final analysis and paper build"
uv run python scripts/analyze_grid.py > results/grid/analyze_final.log 2>&1
uv run python scripts/make_paper_tables.py >> results/grid/analyze_final.log 2>&1
(cd paper && tectonic -X compile main.tex > ../results/grid/paper_build.log 2>&1)
log "post-grid done"

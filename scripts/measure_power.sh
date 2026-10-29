#!/usr/bin/env bash
# Power measurement for the Pareto-optimal configurations (GPU and CPU-only).
#
# Run this YOURSELF in a terminal (it needs sudo for powermetrics only):
#     cd ~/rnd/fusion_vlm && ./scripts/measure_power.sh
# It asks for your password once. Keep the Mac plugged in, lid open, and
# don't use it while this runs (~75 min). Safe to re-run: model runs resume.
#
# Writes results/power/{powermetrics.plist, gpu.csv, cpu.csv, markers.csv, run.log}
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/power
mkdir -p "$OUT"
T=data/benchmark/v1/labels_test.csv
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }

if ! pmset -g batt | head -1 | grep -q "AC Power"; then
  echo "Not on mains power. Plug in and re-run."; exit 1
fi

echo "powermetrics needs administrator rights; enter your password if asked."
sudo -v || exit 1
# keep the sudo timestamp fresh while we run
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
KEEPALIVE=$!

PM_START=$(python3 -c "import time; print(time.time())")
echo "$PM_START" > "$OUT/pm_start_epoch.txt"
sudo powermetrics --samplers cpu_power,gpu_power -i 200 -f plist -o "$OUT/powermetrics.plist" &
PM=$!
cleanup() { sudo kill -INT "$PM" 2>/dev/null; kill "$KEEPALIVE" 2>/dev/null; }
trap cleanup EXIT
sleep 5

mark() { echo "$1,$2,$(python3 -c 'import time; print(time.time())')" >> "$OUT/markers.csv"; }
[ -f "$OUT/markers.csv" ] || echo "event,label,epoch" > "$OUT/markers.csv"

log "idle baseline (60 s)"
mark idle_start before; sleep 60; mark idle_end before

uv run python - > "$OUT/configs.txt" <<'PY'
import json
for c in json.load(open("results/grid/frontier.json")):
    print(c["spec"], c["resolution"], c["n_frames"])
PY

for dev in gpu cpu; do
  if [ $dev = gpu ]; then flags="--ngl 99"; else flags="--ngl 0 --threads 10"; fi
  while read -r spec res nf; do
    extra=""; [ "$res" != "native" ] && extra="--resolution $res"
    log "$dev $spec res=$res"
    uv run python scripts/run_eval.py --labels $T --models "$spec" --prompts A $flags \
      --server-args=--no-mmap --n-frames "$nf" --limit 60 --log-times $extra \
      --out "$OUT/$dev.csv" >> "$OUT/run_eval.log" 2>&1
  done < "$OUT/configs.txt"
done

log "idle baseline after (60 s)"
mark idle_start after; sleep 60; mark idle_end after
log "done"

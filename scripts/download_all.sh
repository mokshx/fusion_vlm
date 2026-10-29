#!/usr/bin/env bash
# Start the source downloads (UR Fall frame bank, MCF, OmniFall labels) in parallel, detached, logging to data/raw/logs/.
# Safe to re-run: every step resumes. Keeps the Mac awake while anything runs.
# Check progress:  tail -f data/raw/logs/*.log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/logs data/raw/mcf/zips data/raw/mcf/docs

mcf() {
  local base=https://www.iro.umontreal.ca/~labimage/Dataset
  for f in technicalReport.pdf Erratum-MCF.pdf; do
    [ -f data/raw/mcf/docs/$f ] || curl -sfL --retry 5 -o data/raw/mcf/docs/$f "$base/$f"
  done
  seq -w 1 24 | xargs -P 4 -n 1 scripts/fetch_mcf_one.sh
  echo "$(date +%H:%M:%S) mcf finished"
}

nohup caffeinate -i uv run python scripts/download_sources.py urfall --workers 4 > data/raw/logs/urfall.log 2>&1 &
# Le2i (scripts/download_sources.py le2i) is not used: its repository was unavailable during this study.
nohup caffeinate -i uv run python scripts/download_sources.py omnifall > data/raw/logs/omnifall.log 2>&1 &
export -f mcf
nohup caffeinate -i bash -c mcf > data/raw/logs/mcf.log 2>&1 &
echo "started; logs in data/raw/logs/"

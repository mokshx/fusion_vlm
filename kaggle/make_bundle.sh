#!/usr/bin/env bash
# Package code + Step 0 data for upload as a private Kaggle dataset.
set -euo pipefail
cd "$(dirname "$0")/.."
out=dist/pdvlm_step0_bundle
rm -rf "$out" "$out.zip"
mkdir -p "$out/scripts" "$out/data/benchmark"
cp -r src "$out/"
cp scripts/run_eval.py scripts/summarize.py "$out/scripts/"
cp -r data/benchmark/step0 "$out/data/benchmark/"
cp kaggle/step0_notebook.py "$out/"
find "$out" -name __pycache__ -prune -exec rm -rf {} +
(cd dist && zip -qr pdvlm_step0_bundle.zip pdvlm_step0_bundle)
echo "wrote dist/pdvlm_step0_bundle.zip ($(du -h dist/pdvlm_step0_bundle.zip | cut -f1))"

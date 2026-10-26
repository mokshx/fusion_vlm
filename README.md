# fusion_vlm

Zero-shot study of edge vision-language models for person-down detection
("How Small Can You Go?"). Status: Step 0 pilot done, see `docs/step0_report.md`.

## Layout
- `src/pdvlm/` — common inference interface (`backends/`: llama.cpp server, HF transformers),
  prompts, yes/no parser, model registry, remote-zip reader, environment capture
- `scripts/` — data fetching, benchmark building, `run_eval.py`, `summarize.py`, `contact_sheet.py`
- `data/benchmark/<set>/labels.csv` — labels (frames are git-ignored; rebuild with the scripts)
- `results/` — one CSV row per (sample, config) + `run-*.json` with environment and peak memory
- `docs/datasets.md` — dataset access, licences, citations, publication blockers

## Reproduce Step 0
```bash
uv sync --extra hf                        # hf extra only needed for the transformers backend
brew install llama.cpp ffmpeg             # llama-server (tested b10090)
# GGUF weights -> models/<repo-name>/ (LFM2.5-VL-450M-GGUF, LFM2.5-VL-3B-GGUF; Q8_0 + mmproj F16)
# UR Fall labels -> data/raw/urfall/urfall-cam0-{falls,adls}.csv (from the official page)
uv run python scripts/urfall_fetch_candidates.py --falls 1-10 --adls 1-40
uv run python scripts/build_step0.py
uv run python scripts/run_eval.py --labels data/benchmark/step0/labels.csv \
    --images-root data/benchmark/step0 --models lfm-450m:Q8_0 lfm-3b:Q8_0 \
    --prompts A B --ngl 0 --threads 10 --out results/step0/mac_m4pro_cpu.csv
uv run python scripts/summarize.py results/step0/mac_m4pro_cpu.csv
```
Kaggle: `bash kaggle/make_bundle.sh`, upload `dist/pdvlm_step0_bundle.zip` as a **private**
dataset, run `kaggle/step0_notebook.py` in a GPU notebook.

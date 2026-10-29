# How Small Can You Go? Small vision-language models for on-device person-down detection

Code, labels and results for the paper *"How Small Can You Go? Accuracy-Efficiency Trade-offs of
Small Vision-Language Models for On-Device Person-Down Detection"* (Moksh Chadha, National
Institute of Technology Hamirpur).

A worker lying on the floor needs help within minutes. We ask how small, how heavily quantized
and how low-resolution a zero-shot vision-language model (VLM) can be and still detect a
**person down**, and what each choice costs in latency, memory and energy on a laptop-class
CPU and integrated GPU. Five open VLMs (LFM2.5-VL 450M / 1.6B / 3B, Qwen3.5 2B / 4B) are
evaluated across weight precision, input resolution, frames per decision and prompt, on a
benchmark of 711 reviewed frames from two public fall datasets, against a YOLO11-pose baseline.

## Headline results

Test set (494 frames), prompt A, one frame, threshold set on the development split for 95% recall,
integrated GPU of an Apple M4 Pro (48 GB). FAR = false-alarm rate on hard negatives (kneeling,
crouching, bending, sitting, lying on furniture). Full tables are in the paper.

| Model | Weights | Longest side | F1 | Recall | FAR | p95 latency | Peak RSS | Cameras at 1 decision/s |
|---|---|---|---|---|---|---|---|---|
| LFM2.5-VL-450M | Q8_0 | 512 px | 0.89 | 91% | 13% | 92 ms | 0.85 GiB | 10 |
| **LFM2.5-VL-1.6B** | **Q4_K_M** | **256 px** | **0.93** | 93% | 8% | 169 ms | 1.87 GiB | 5 |
| LFM2.5-VL-3B | Q8_0 | 512 px | 0.95 | 93% | 2% | 518 ms | 4.24 GiB | 1 |
| LFM2.5-VL-3B (best) | 16-bit | native | 0.97 | 98% | 5% | 672 ms | 7.07 GiB | 1 |
| YOLO11n-pose + rule | - | native | 0.50 | 43% | 26% | 11 ms | 0.61 GiB | - |

* Energy on the GPU is 3.2 J per decision for the recommended 1.6B configuration; the same
  configuration needs 10-26x more energy on the CPU alone.
* 4-bit weights save memory, not latency; 512 px costs little accuracy; 256 px breaks every model
  except LFM2.5-VL-1.6B; extra frames and long prompts hurt; the pose baseline's detector often
  misses a person lying on the floor.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | LaTeX source (`main.tex`, `sections/`), bibliography, generated tables/figures |
| `src/pdvlm/` | Inference backends (llama.cpp server, Hugging Face Transformers), prompts, yes/no parser, bootstrap statistics, metrics |
| `scripts/` | Data download, benchmark construction, experiment grid, analysis, power measurement |
| `data/benchmark/v1/` | **Released labels** (`labels_test.csv`, `labels_dev.csv`), review provenance and human-validation records; see its `README.md` |
| `results/` | Per-decision CSVs for every configuration, per-run JSON (environment, memory), summaries, power results |
| `docs/` | Labelling policy, dataset access and licences, pilot report |

## Data and licences

* **Code**: Apache-2.0 (`LICENSE`).
* **Labels** in `data/benchmark/v1/`: **CC BY-NC-SA 4.0**, because they derive from UR Fall (CC BY-NC-SA 4.0)
  and the OmniFall annotations of MCF (CC BY-NC-SA 4.0). Non-commercial research use only.
* **Frames are not redistributed.** UR Fall is non-commercial share-alike and the Multiple Cameras
  Fall dataset states no licence. The scripts below download the originals from the official sources.
* The UR Fall dataset requires citing Kwolek and Kepski (2014); MCF requires citing Auvinet et al. (2010).
  See `docs/datasets.md`.
* Model weights keep their own licences (LFM Open License 1.0; Apache-2.0 for Qwen3.5).

## Reproducing the study

Developed and run on macOS (Apple M4 Pro, 48 GB, llama.cpp build b10090). The pipeline has been
run end to end only on the author's machine; an independent run from a fresh clone has not yet
been done, so please open an issue if a step fails. Disk: about 41 GB for model files, 5 GB for raw data.

```bash
brew install llama.cpp ffmpeg tectonic
uv sync --extra hf --extra baseline

# 1. Data and models (resumable; the UR Fall and MCF servers are slow)
./scripts/download_all.sh          # UR Fall frame bank + labels, MCF videos, OmniFall labels
./scripts/download_models.sh       # GGUF weights, 5 models x 3 precisions

# 2. Benchmark. The candidate pool and labels rebuild exactly (verified byte-for-byte).
uv run python scripts/build_candidates.py
cp data/benchmark/v1/provenance/review.csv data/interim/bench/review.csv   # our visual-review decisions
uv run python scripts/finalize_benchmark.py                                # -> data/benchmark/v1/labels_{test,dev}.csv

# 3. Experiments (about 10 h on an M4 Pro; resumable)
uv run python scripts/run_grid.py stage1 stage2 stage3
./scripts/post_grid.sh             # re-timing, pose baseline, CPU-only timing, analysis, paper build

# 4. Power (needs sudo for powermetrics: run in your own terminal, on mains power, machine idle)
./scripts/measure_power.sh
uv run python scripts/power_analysis.py

# 5. Tables, figures and paper
uv run python scripts/analyze_grid.py
uv run python scripts/make_paper_tables.py
uv run --extra baseline --extra hf python scripts/failure_gallery.py
cd paper && tectonic main.tex
```

Notes for reproducers
* `scripts/analyze_grid.py` contains the 41-minute battery window of our own run (`BATTERY`) and
  the date of our run log; for a new run, set `BATTERY = []` or fill in your own window.
* Latency, memory and power depend on the machine; accuracy does not. Decoding is greedy, and CPU,
  GPU and bf16 runs gave identical answers on the pilot.
* The pose-baseline failure figure blurs faces with boxes placed by eye (`MANUAL_BLUR` in
  `scripts/failure_gallery.py`) and shows UR Fall frames only.
* `kaggle/` holds an unused helper for a Transformers cross-check on a cloud GPU; the paper's
  results do not depend on it.

## How the labels were made

Candidates were proposed from the official annotations, then **an AI assistant (Claude, Anthropic)
reviewed every candidate visually under the author's direction**, using the policy in
`docs/labelling_policy.md`. Because a VLM took part in labelling a VLM benchmark, the labels were
validated by the author in two ways (records in `data/benchmark/v1/validation/`):

1. a **blind check** of 100 random test frames: 92% agreement, Cohen's kappa 0.88, 2 frames marked unsure;
2. a **full pass** over all 711 frames with the label shown: 711 confirmed, 0 changed, 0 removed.
   Because the label is visible, this pass can anchor; the blind check is the unbiased measure.

## Limitations

Indoor, staged recordings only (no industrial footage); two datasets and few actors; one laptop-class
device, no embedded board; power is the CPU+GPU+Neural Engine package estimate from `powermetrics`,
not wall power; zero-shot only. See the paper's Limitations section.

## AI assistance

Claude (Anthropic) wrote most of the analysis code, drafted portions of the manuscript and carried
out the visual label review. The author directed the work, checked the labels, reviewed the results and
takes responsibility for the content.

## Citation

```bibtex
@misc{chadha2026howsmall,
  author = {Chadha, Moksh},
  title  = {How Small Can You Go? Accuracy--Efficiency Trade-offs of Small Vision--Language Models for On-Device Person-Down Detection},
  year   = {2026},
  url    = {https://github.com/mokshx/fusion_vlm}
}
```
Please also cite the datasets (UR Fall, MCF, OmniFall) listed in `docs/datasets.md`.

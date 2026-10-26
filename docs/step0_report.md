# Step 0 pilot report (2026-10-21)

30 UR Fall frames (10 person_down: 5 side cam0 + 5 overhead cam1; 10 hard_negative; 10 normal),
one frame per sequence, every frame checked by eye. Built by `scripts/build_step0.py`;
labels in `data/benchmark/step0/labels.csv`. Native 640×480, single frame, greedy decoding,
max 16 new tokens, prompt cache off.

## Accuracy (identical on llama.cpp CPU and Metal)

| model | prompt | strict parse | recall | FAR hard-neg | FAR normal | F1 | AUC (P(yes)) |
|---|---|---|---|---|---|---|---|
| LFM2.5-VL-450M Q8_0 | A | 100% | 10/10 | 5/10 | 0/10 | 0.80 | 1.00 |
| LFM2.5-VL-450M Q8_0 | B | 100% | 10/10 | 9/10 | **10/10** | 0.51 | 0.88 |
| LFM2.5-VL-3B Q8_0 | A | 100% | 10/10 | 2/10 | 0/10 | 0.91 | 1.00 |
| LFM2.5-VL-3B Q8_0 | B | 100% | 10/10 | 6/10 | 0/10 | 0.77 | 1.00 |
| LFM2.5-VL-450M HF bf16 (MPS) | A | 100% | 10/10 | 6/10 | 0/10 | 0.77 | — |

- Output format: all 240 llama.cpp (CPU + Metal) + 60 HF answers were exactly "Yes"/"No" (± full stop). No
  grammar constraint needed.
- GGUF Q8_0 vs HF bf16 (450M): 59/60 answers agree; the one flip is borderline (P(yes) 0.46 vs 0.56).
- Prompt B (detailed definition) *hurts*: 450M says "yes" to everything incl. people standing;
  3B false alarms rise 2 → 6. The long negative list ("kneeling, crouching, lying on a bed…") seems
  to prime the posture words rather than exclude them.
- With prompt A, P(yes) separates classes perfectly (AUC 1.0); errors come from the 0.5 cut-off.
  ⇒ report threshold-free metrics and pick operating thresholds on a dev split.
- Hard negatives that fool models: kneeling/hands-and-knees and crouching (both models), lying on bed (450M).

## Efficiency (Apple M4 Pro, 48 GB, macOS 26.5.2, llama.cpp b10090, AC power)

| config | e2e median | e2e p95 | prompt tok | peak RSS | peak footprint | streams @1 Hz |
|---|---|---|---|---|---|---|
| 450M Q8_0, CPU 10 threads | 0.93 s | 0.98 s | 261 | 1.11 GB | 0.72 GB | 1 |
| 3B Q8_0, CPU 10 threads | 4.32 s | 4.84 s | 261 | 6.58 GB | 3.81 GB | 0 (1 decision / 4.3 s) |
| 450M Q8_0, Metal | 0.11 s | 0.12 s | 261 | 0.72 GB | 0.42 GB | 8 |
| 3B Q8_0, Metal | 0.52 s | 0.53 s | 261 | 3.82 GB | 1.11 GB | 1 |

Prompt A, 30 requests after 3 discarded warm-ups. Streams = floor(1000 ms / p95), sequential.
On CPU, 97–99% of end-to-end time is prefill (image encode + prompt); on Metal prefill is 76% (450M) / 92% (3B) and the rest is client overhead (PNG encode, HTTP). Decoding the one-word answer takes a median of ~4–6 ms (450M) and ~21–33 ms (3B); one 3B CPU request took 210 ms (isolated outlier). Memory columns are GiB.

**Indicative only:** load average was 10–12 at run start (system services + my own earlier jobs);
first attempt was ruined by macOS idle sleep (one request "took" 1015 s; data in
`results/step0/aborted_sleep/`). Runner now holds `caffeinate`.

## Measurement issues to settle before Week 6
1. Footprint excludes Metal/unified buffers (3B Metal footprint 1.1 GB < 2.7 GB weights). On CPU,
   RSS ≈ 2× weights because llama.cpp repacks weights for ARM and keeps the mmap. Add system-level
   memory delta; test `--no-mmap` / `--no-repack`.
2. Thread count: 10 = P-cores on M4 Pro; sweep 4/6/8/10 on the real CPU device.
3. Run on AC, after reboot, nothing else running; record `pmset -g therm`.

## Limitations of the pilot
n = 30; all UR Fall falls happen in one office room while ADLs span other rooms (scene confound);
two actors; cam1 has positives only; Kaggle run still to do.

## Addendum 2026-10-22: prompt B2 and deliberate floor-lying (Metal; answers match CPU)
Decisions: see `docs/labelling_policy.md`. B2 = "A person is 'down' only if their torso is lying on
the floor or ground. Is any person down? Answer yes or no."

| model | prompt | recall | FA hard-neg | FA normal | F1 | AUC | median |
|---|---|---|---|---|---|---|---|
| 3B Q8_0 | A | 10/10 | 2/10 | 0/10 | 0.91 | 1.00 | 530 ms |
| 3B Q8_0 | B | 10/10 | 6/10 | 0/10 | 0.77 | 1.00 | 605 ms |
| 3B Q8_0 | B2 | 10/10 | 3/10 | 0/10 | 0.87 | 1.00 | 519 ms |
| 450M Q8_0 | A | 10/10 | 5/10 | 0/10 | 0.80 | 1.00 | 112 ms |
| 450M Q8_0 | B | 10/10 | 9/10 | 10/10 | 0.51 | 0.89 | 118 ms |
| 450M Q8_0 | B2 | 10/10 | 9/10 | 9/10 | 0.53 | 0.94 | 111 ms |

- B2 fixes most of B's damage for 3B (6 → 3 hard-neg false alarms), ≈ A within noise at n = 10.
- 450M keeps a "yes" bias whenever the prompt contains a definition (P(yes) on standing people
  ≈ 0.54–0.68 with B2 vs ≈ 0.99 with B): the smallest model works best with the plain question.
- Deliberate floor-lying (9 frames, rooms other than the fall room): detected 9/9 by both models
  under all three prompts. Together with correct rejections of standing people in those same rooms,
  this argues against the models merely recognising the fall room (still small n).

## Addendum 2026-10-26: Qwen3.5-2B (Unsloth GGUF Q8_0, reasoning off) and memory ablation
- llama.cpp b10090 runs Qwen3.5 vision; answers are a bare "yes"/"no".
- 30-frame set (Metal): A 10/10 recall, 2/10 hard FA, 0/10 normal; B 4/10 hard FA; **B2 0/10 hard FA** (F1 1.00).
  Best prompt differs by model: B2 best for Qwen, ≈A for LFM-3B, useless for LFM-450M.
- Floor-lying set: 27/27 detected. CPU vs Metal answers identical (210/210 across all three models).
- Latency A: CPU 4967 ms, Metal 523 ms; RSS 4.75 / 2.82 GiB. 329 prompt tokens per frame vs 261 for LFM.
- Memory ablation (results/memory_ablation/): --no-mmap makes RSS ≈ footprint (±0.06 GiB);
  --no-repack cuts 3B CPU RSS 6.42 → 3.92 GiB at +24% latency; no flag changed any answer;
  system-level (total − available) delta not reproducible on macOS → efficiency runs use --no-mmap and report RSS.

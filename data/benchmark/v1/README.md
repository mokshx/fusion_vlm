# Person-down benchmark v1: labels

711 frames (494 test, 217 development) from UR Fall and the Multiple Cameras Fall dataset (MCF).
**Frames are not included**; see the repository README for the scripts that download the sources and
rebuild them. Labels are released under **CC BY-NC-SA 4.0** (non-commercial research use), because they
derive from UR Fall (CC BY-NC-SA 4.0) and the OmniFall annotations of MCF (CC BY-NC-SA 4.0).

## Files
| File | Contents |
|---|---|
| `labels_test.csv`, `labels_dev.csv` | One row per sample. The development split is used only to choose decision thresholds |
| `provenance/candidates.csv` | The automatically proposed candidate pool (777 frames) with proposed labels |
| `provenance/review.csv` | The visual-review decision for every candidate (accept / relabel / reject, with reason) |
| `validation/human_check_*` | The blind 100-frame check by the author: answers, answer key, score |
| `validation/full_check_*` | The author's full pass over all 711 frames (decisions, and the first-pass file before one correction) |
| `summary.txt` | Counts by label, subtype, dataset and viewpoint |

## Columns of `labels_*.csv`
`sample_id`, `split`, `label` (`person_down` / `hard_negative` / `normal`), `subtype`, `viewpoint`
(`side`, `overhead`, `elevated`), `dataset`, `sequence`, `camera`, `frame`, `t_s`, `source_label`
(the source dataset's own annotation), `image_path` and `clip_paths` (paths where the build scripts
place the frames; clips are the labelled frame plus the seven preceding frames at 0.25 s), `notes`,
`cand_id`, `human_check`.

## Labels
* **person_down**: a person's torso lies on the floor or ground (falls, and deliberate lying on the floor
  as subtype `lying_floor_deliberate`; UR Fall counts the latter as a daily activity).
* **hard_negative**: low or horizontal posture with the torso off the floor: crouching, kneeling (including
  hands and knees), bending, sitting (chair, sofa or bed, floor), lying on a sofa or bed.
* **normal**: every visible person stands or walks.
* Several people in a frame: the most serious posture decides (down, then hard, then normal).
* Transitional, ambiguous or person-free frames are excluded. Full policy: `docs/labelling_policy.md`.

## Known properties
Split by sequence (an MCF scenario with all eight cameras, or a UR Fall sequence with both cameras, is wholly
in one split). UR Fall's overhead camera recorded only falls, so overhead frames are almost all person_down.
MCF has one actor, falls mostly onto a mattress. The labels were reviewed by an AI assistant and validated by the
author (blind agreement 92%, kappa 0.88).

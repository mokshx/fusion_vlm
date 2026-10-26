# Labelling policy and study decisions

## Classes
| label | definition | subtypes |
|---|---|---|
| `person_down` | a person whose torso is lying on the floor/ground, regardless of how they got there | `fall` (dataset-annotated fall), `lying_floor_deliberate` (lies down on the floor on purpose, e.g. UR Fall adl-30..35, 38..40) |
| `hard_negative` | low or horizontal posture that is *not* torso-on-floor | `crouching`, `kneeling` (incl. hands and knees), `bending`, `sitting_chair`, `sitting_furniture`, `sitting_floor`, `lying_furniture` (bed/sofa) |
| `normal` | upright person | `standing`, `walking` |

Rules: a frame is labelled on what is visible in it, not on intent or on the source dataset's
activity name. Transitional frames (mid-fall, getting down/up) are excluded. Frames where the person
is not visible are excluded. Every frame is checked by eye.

## Decision log
| date | decision | consequence |
|---|---|---|
| 2026-10-22 | Deliberate lying on the floor = `person_down` (subtype `lying_floor_deliberate`) | Departs from UR Fall's ADL convention; report this subtype separately and state it when comparing with prior UR Fall results |
| 2026-10-22 | Add prompt B2 (short criterion-only definition) alongside A and B | Grid uses prompts A, B, B2 |
| 2026-10-22 | Public datasets only; no industrial/company data | Domain gap (indoor, non-factory) stays a stated limitation |

## Benchmark construction parameters (fixed 2026-10-26; the builder must follow these)
- At most 3 frames per label per source sequence, at least 1 s apart.
- Clips: the labelled frame plus the 3 or 7 preceding frames, 0.25 s apart (8 frames = 1.75 s).
- Split by sequence (and by subject where available); dev set only for thresholds.
- Target ≥ 100 test samples per hard-negative subtype (±10 pp CI); report actual counts when fewer exist.
- Bootstrap CIs: 10,000 replicates, resampling whole sequences.
- Pose baseline: YOLO11-pose n and s; rule = torso within θ of horizontal AND box width ≥ r × height, for ≥ N frames on clips; θ, r, N tuned on dev.

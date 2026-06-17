# Milestone H

Milestone H is the next controlled experiment after Milestone G.

Current best baseline entering H:

```text
G1_schedule_extend, epoch 27
marking IoU: 0.542659
F1:          0.703538
precision:   0.635133
recall:      0.788456
mIoU:        0.823239
```

Source: `logs/milestone_g/run_analysis/G1_schedule_extend/summary.csv`

## Purpose

G1 is the best model so far. Milestone H started from the supervisor-requested
idea of making the training problem more road-marking-focused:

1. Reduce irrelevant `other` points using a vertical/z crop.
2. Add marking-aware sampling so training samples contain markings more often.
3. Train longer once the data/sampler setup is validated.

Those ideas were audited before training. The current evidence says:

- **Do not z-crop.** A safe crop exists, but it does not touch G1's dominant
  road-level road-to-marking false positives.
- **Do not change the sampler for H0.** Marking-aware and hard-negative
  patch-center changes have low leverage under the real 32768-neighbor patch
  geometry.
- **Target RGB brightness robustness next.** The frozen G1 model is measurably
  sensitive to absolute RGB brightness, and the dominant residual error is
  bright road predicted as marking.

Milestone H should test the next intervention without corrupting the frozen G1
baseline.

## Directory Layout

```text
logs/milestone_h/
  configs/
  dataset_analysis/
  launch_logs/
  notes/
  reports/
  run_analysis/
    analysis_code/
  runs/
```

## Rules

- G1 remains the frozen baseline.
- H0 must be compared against G1 epoch 27.
- Z-crop and sampler changes are rejected for H0 unless new evidence appears.
- The next H0 candidate is train-only RGB brightness/contrast jitter.
- Any H0 config must state exactly what changed from G1.
- Checkpoint selection remains max raw marking IoU unless explicitly changed.
- Keep all H outputs under `logs/milestone_h/`.

## Current Evidence

Resolved audits:

- `dataset_analysis/z_crop_audit/`: final decision **do not crop**.
- `dataset_analysis/marking_aware_sampling_audit/`: final decision **keep
  uniform sampling for H0**.
- `run_analysis/g1_rgb_brightness_sensitivity/`: frozen G1 brightness probe;
  final recommendation **run H0_rgb_jitter**.

The brightness probe scaled RGB at validation inference time for the frozen G1
epoch-27 checkpoint:

```text
scale 0.85: road->marking FP 454,798 (-10.32% vs scale 1.00)
scale 1.00: road->marking FP 507,112
scale 1.15: road->marking FP 556,667 (+9.77% vs scale 1.00)
```

Darkening improved precision but reduced recall, so fixed darkening is not the
solution. The result justifies a **train-only** RGB brightness/contrast jitter
experiment to reduce reliance on exact brightness while evaluating on unchanged
validation/test data.

## Immediate Work

Before training H0:

1. Implement train-only RGB brightness/contrast jitter.
2. Keep validation and test RGB unchanged.
3. Keep G1 model/loss/weights/sampler/scheduler unchanged.
4. Smoke-test the data path and verify jitter affects only RGB-valid training
   features.
5. Run H0 from scratch and compare against G1 epoch 27.

Primary source for the H0 rationale:

```text
logs/milestone_h/notes/rgb_jitter_context.md
```

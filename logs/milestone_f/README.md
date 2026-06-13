# Milestone F Context

Milestone F is the calibration follow-up to Milestone E. It keeps Milestone E's
front-camera RGB input pipeline **unchanged** and changes only the training
objective and the first-embedding width, to fix the failure that Run E exposed:
RGB improved marking recall but the strong marking class weight over-amplified
it into heavy overprediction (low precision).

```text
D0 (Milestone D): x y z intensity
Run E (Milestone E): x y z intensity r g b rgb_valid     marking weight 26.88, dim_features 8
F0 (Milestone F):    x y z intensity r g b rgb_valid     marking weight 15.00, dim_features 16
```

## Why Milestone F exists

Run E (`logs/milestone_e/runs/E0_rgb_front_v1`, best epoch 14) vs the D0
baseline (`logs/milestone_d/runs/D0_weighted_ce_25ep`, best epoch 18):

| metric | D0 | Run E | read |
| --- | ---: | ---: | --- |
| marking IoU | 0.440 | 0.438 | tied |
| marking precision | 0.515 | 0.470 | worse |
| marking recall | 0.753 | 0.866 | better |
| pred/true marking | 1.46x | 1.84x | worse (overpredicts) |

The Run E error analysis showed the overprediction concentrates exactly where
RGB is valid (rgb_valid precision 0.449 / 1.94x vs rgb_invalid 0.570 / 1.44x),
while RGB clearly helped detection (raw-8 lane-line recall 0.886 with RGB vs
0.669 without). Conclusion: RGB is useful but over-trusted under the rare-class
weight. The fix is calibration, not removing RGB.

Full Run E analysis: [`../milestone_e/run_analysis/E0_rgb_front_v1/`](../milestone_e/run_analysis/E0_rgb_front_v1/)

## What changed F vs Run E

Exactly two things change from Run E; everything else (optimizer, scheduler,
batch size, augmentations, epochs, `in_channels`, `num_layers`, `dim_output`,
the entire RGB projection/sampling/cache policy) is identical.

| field | Run E | F0 | reason |
| --- | ---: | ---: | --- |
| marking class weight (effective) | 26.88 | 15.00 | reduce overprediction pressure |
| `class_weights[marking]` (count form) | 5,129,328 | 14,344,000 | yields effective 15.00 via `1/(freq+0.02)` |
| `dim_features` | 8 | 16 | input doubled 4->8 vs D0; restore early embedding headroom |

Effective CE weights F0 produces: road `2.445`, marking `15.00`, other `1.711`.

## What is common with Milestone E (NOT duplicated here)

All pre-training dataset analysis is shared infrastructure and stays in
Milestone E. Milestone F reuses it as-is and does not copy it:

- front-camera inventory, projection contract + overlays, timestamp/ego-motion
  policy, stratified valid-RGB audit, occlusion screening, voxel-mixing audit,
  single-sample smoke:
  [`../milestone_e/dataset_analysis/`](../milestone_e/dataset_analysis/)
- pinned policy notes (projection, timestamp, cache):
  [`../milestone_e/notes/`](../milestone_e/notes/)
- end-to-end Milestone E context: [`../milestone_e/context.md`](../milestone_e/context.md)

The dataset code, adapter, and RGB cache policy live in `src/` and are shared by
both milestones. F builds its own cache (different `dim_features` does not change
preprocessing, but F uses a separate cache dir for clean isolation).

Class-count statistics are reused from Milestone D:
[`../milestone_d/road_marking3_training_statistics.json`](../milestone_d/road_marking3_training_statistics.json).

## Layout

```text
logs/milestone_f/
  README.md                  # this file
  configs/                   # F training configs
    f0_rgb_soft_weights.yml  # F0: RGB + softened marking weight (eff. 15) + dim_features 16
  cache/                     # F preprocessing caches (versioned), built at run time
  runs/                      # F training runs
  run_analysis/              # per-run analysis output (mirror Milestone E analysis)
  reports/                   # rolled-up reports
  launch_logs/               # nohup stdout logs
  notes/                     # F-specific notes (not the common E policy notes)
```

## Status

| step | what | status |
| --- | --- | --- |
| config | F0 config created (weight eff. 15, dim_features 16) | done |
| smoke | tiny 2-3 epoch smoke (cache builds, in_channels=8, dim_features=16 ok) | pending |
| full run | F0 full 25-epoch run | pending |
| analysis | F0 analysis mirroring Run E (vs D0 and vs Run E) | pending |

## How to run

Milestone F uses `tools/train_milestone_f.py`, a thin wrapper around the shared
road-marking runner that pins Milestone F defaults (config
`logs/milestone_f/configs/f0_rgb_soft_weights.yml`, runs dir
`logs/milestone_f/runs`, completion label `milestone_f_run_complete`).

Smoke (verify cache build + shapes + stability):

```bash
python tools/train_milestone_f.py \
  --config logs/milestone_f/configs/f0_rgb_soft_weights.yml \
  --run-name F0_smoke \
  --epochs 3 \
  --force
```

Full run (after smoke passes):

```bash
rm -rf logs/milestone_f/cache/F0_rgb_soft_weights_v1
rm -rf logs/milestone_f/runs/F0_rgb_soft_weights
mkdir -p logs/milestone_f/launch_logs

nohup python tools/train_milestone_f.py \
  --config logs/milestone_f/configs/f0_rgb_soft_weights.yml \
  --run-name F0_rgb_soft_weights \
  --epochs 25 \
  --seed 42 \
  --save-ckpt-freq 1 \
  --device cuda \
  --pin-memory \
  --force \
  > logs/milestone_f/launch_logs/F0_rgb_soft_weights_fresh_25.stdout.log 2>&1 &
```

## Decision rule when F0 lands

- clean win: marking precision >= 0.51 and recall >= 0.78 -> lock F0.
- still overpredicts (pred/true > 1.6x, precision < 0.50) -> next run marking
  weight ~12.
- recall drops below 0.75 -> next run marking weight ~18.

Report F0 against **both** D0 and Run E, using the same eval slices as Milestone
E (full val incl. 054, val excl. 054, per-distance bucket, raw 8/9/10 subtype,
rgb_valid vs rgb_invalid).

# Milestone E Context

Milestone E adds front-camera RGB features to the LiDAR input, on top of the
Milestone D `road_marking3` task.

The intent is to test whether per-point color helps where D0's residual error
lives: road/marking confusion when LiDAR intensities overlap.

```text
D0: x y z intensity
E0: x y z intensity r g b rgb_valid   (target)
```

This README is the working index. Each sub-directory has its own README.

## Layout

```text
logs/milestone_e/
  README.md                          # this file
  notes/                             # short curated notes and pinned facts
  dataset_analysis/                  # split-level analyses, not tied to a run
    front_camera_inventory/          # step 1: presence/load check
  (later)
  configs/                           # E0 training configs
  cache/                             # E0 preprocessing caches (versioned)
  runs/                              # training runs
  run_analysis/                      # per-run analysis output
  reports/                           # rolled-up reports
```

## Pre-Training Audit Status

Each step is from the agreed first-tasks list. "Done when" describes the gate.

| # | step | status | output |
| ---: | --- | --- | --- |
| 1 | Inventory check on all split sequences | done | `dataset_analysis/front_camera_inventory/` |
| 2 | Confirm `geometry.projection` contract | done | `notes/projection_contract.md` |
| 3 | Visual overlay sanity check (8 frames) | done | `dataset_analysis/projection_overlay/` |
| 4 | Timestamp + ego-motion policy | done | `notes/timestamp_policy.md`, `dataset_analysis/timestamp_policy/` |
| 5 | Stratified valid-RGB audit | done | `dataset_analysis/valid_rgb_audit/` |
| 6 | Occlusion screening measurement | done -- 0.03% marking occluded, skip for E0 | `dataset_analysis/occlusion_audit/` |
| 7 | Voxel-subsampling rgb_valid policy | done -- 0.03% mixed, pass-through | `dataset_analysis/voxel_mixing_audit/` |
| 8 | Cache versioning + projection-at-runtime decision | done | `notes/cache_policy.md` |
| 9 | Dataset extension + single-sample smoke | done | `dataset_analysis/sample_smoke/`, `configs/e0_rgb_front.yml` |
| 10 | 1-2 epoch tiny training smoke | pending | -- |

Only after step 10 passes should E0 launch a full training run.

## Training Runner Note

E0 uses `tools/train_milestone_e.py`. This is a thin Milestone E wrapper around
the shared road-marking training runner. It keeps the AdamW,
ReduceLROnPlateau, checkpoint, resume, and metric plumbing from Milestone D,
but pins Milestone E defaults:

```text
default config:   logs/milestone_e/configs/e0_rgb_front.yml
default runs dir: logs/milestone_e/runs
completion log:   milestone_e_run_complete
```

Use `tools/train_milestone_d.py` for D runs and `tools/train_milestone_e.py`
for E runs, so command names, configs, run directories, and logs stay aligned.

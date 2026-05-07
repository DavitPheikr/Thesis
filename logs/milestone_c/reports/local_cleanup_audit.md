# Local Cleanup Audit

Generated: 2026-05-04

Historical note, 2026-05-07: this audit predates the completed Milestone C medium runs and official full C0 run. The old run-free state reported below is no longer current. The completed official C0 run is `logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/`; see `logs/milestone_c/reports/c0_full_baseline_results.md`.

## 1. Audit Findings

### Milestone C Run Folders

At the time of this audit, `logs/milestone_c/runs/` was empty. This is no longer current after the C0 medium and full runs.

| Folder | Status | end_time.txt | training_log.txt | Size | Notes |
|---|---|---:|---:|---:|---|
| none | clean | n/a | n/a | 4.0K | No completed, partial, failed, or orphaned run folders remain. |

### Checkpoints Under `logs/`

All discovered `.pth` files are archived Milestone B checkpoints. No active checkpoint files remain under the Open3D run directory or Milestone C run directories.

| Path | Size | Modified | Category |
|---|---:|---|---|
| `logs/milestone_b_archive/checkpoints/ckpt_00000.pth` | 3,786,250 bytes | 2026-04-15 16:26 | archived |
| `logs/milestone_b_archive/checkpoints/ckpt_00002.pth` | 3,786,250 bytes | 2026-04-15 16:26 | archived |
| `logs/milestone_b_archive/open3d_checkpoint_archives/archived_day6_prererun_2026-04-15/ckpt_00000.pth` | 3,786,250 bytes | 2026-04-15 08:19 | archived |
| `logs/milestone_b_archive/open3d_checkpoint_archives/archived_day6_prererun_2026-04-15/ckpt_00002.pth` | 3,786,250 bytes | 2026-04-15 08:20 | archived |

Checkpoint references in current code/config:

- `configs/randlanet_pandaset_ff_lane3.yml` contains `ckpt_path:` but no active checkpoint path.
- `tools/train_milestone_c.py` forces `ckpt_path=None` and `is_resume=False`.
- `tools/sanity_train_check.py` also forces non-resume behavior for Milestone B sanity runs.

### Temporary Smoke Configs

`logs/milestone_c/configs/` is empty. No temporary or failed-smoke config files remain.

### Requirements Files

Only the working environment snapshot remains:

- `requirements_working_panda.txt`

Important pinned versions:

- `numpy==1.26.4`
- `open3d==0.19.0`
- `tensorboard==2.16.2`
- `torch==2.2.2+cu121`
- `torchvision==0.17.2+cu121`
- `torchaudio==2.2.2+cu121`

`requirements_local_full.txt` was not present at audit time. It had already been removed because it represented the broken local snapshot that included the bad NumPy 2.x environment. Current source of truth is `requirements_working_panda.txt`.

### `pandaset-devkit/`

- Total size: `30M`
- Patched `.pkl` fallback support is present in `pandaset-devkit/python/pandaset/utils.py`.
- The fallback helper explicitly searches alternate `.pkl` / `.pkl.gz` forms when the primary extension is not present.
- No remaining `__pycache__/` or `.pyc` artifacts were found after cleanup.

### Empty Or Near-Empty Files

These were found and intentionally not deleted because they are historical logs, simple marker files, or stale process-ID records that may still help reconstruct past work.

| Path | Recommendation |
|---|---|
| `logs/day2_chosen_frame.txt` | keep unless Day 2 history is no longer needed |
| `logs/day2_chosen_sequence.txt` | keep unless Day 2 history is no longer needed |
| `logs/day2_forward_sensor.txt` | keep unless Day 2 history is no longer needed |
| `logs/milestone_b_archive/old_sanity_reports/milestone_b_sanity_train_report.stale_empty_before_fresh_run.txt` | historical archive; keep or delete only if archive pruning is desired |
| `logs/milestone_b_dataset_init_probe.txt` | historical probe marker; keep or archive-prune later |
| `logs/milestone_b_dataset_stage_train.txt` | historical probe marker; keep or archive-prune later |
| `logs/milestone_b_sanity_train.pid` | stale PID file; safe to delete later if desired |
| `logs/milestone_b_sanity_train.pid.archived_day6_prererun_2026-04-15` | stale PID archive; safe to delete later if desired |
| `logs/milestone_b_sanity_train.pid.old.archived_day6_prererun_2026-04-15` | stale PID archive; safe to delete later if desired |
| `logs/RandLANet_PandaSetFFLane3_torch/archived_day6_prererun_2026-04-15/log_train_*.txt` | historical Open3D logs; leave unless archive pruning is desired |
| `logs/RandLANet_PandaSetFFLane3_torch/log_train_2026-04-15_15-35-20.txt` | stale Open3D log; safe to archive later, not deleted automatically |

### TODO / FIXME / XXX Search

No `TODO`, `FIXME`, or `XXX` comments were found in:

- `src/`
- `tools/`
- `datasets/`

## 2. Cleanup Actions Taken

### Removed Compiled Python Artifacts

Removed `__pycache__/` directories and `.pyc` files from:

- `src/`
- `tools/`
- `datasets/`
- `pandaset-devkit/`

Verification after cleanup: no `__pycache__/` or `.pyc` files remain in those paths.

### Milestone C Runs And Configs

No cleanup was needed during this audit because:

- `logs/milestone_c/runs/` was already empty.
- `logs/milestone_c/configs/` was already empty.

Earlier stale smoke artifacts had already been removed:

- `logs/milestone_c/runs/C0_local_smoke`
- `logs/milestone_c/configs/c0_local_smoke_1024.yml`
- `logs/milestone_c/configs/c0_local_smoke_cpu_tiny.yml`

### Requirements Cleanup

No action was taken during this audit because `requirements_local_full.txt` was already absent. The working requirements file remains:

- `requirements_working_panda.txt`

### Checkpoint Cleanup

No additional checkpoint movement was needed during this audit. All discovered `.pth` files are already under `logs/milestone_b_archive/`.

## 3. Items Flagged For User Decision

These are safe candidates for future pruning, but were not deleted automatically:

- Stale PID files: `logs/milestone_b_sanity_train.pid*`
- Near-empty Day 2 marker files: `logs/day2_chosen_*.txt`, `logs/day2_forward_sensor.txt`
- Historical empty/stale Open3D train logs under `logs/RandLANet_PandaSetFFLane3_torch/`
- Old empty sanity report already inside `logs/milestone_b_archive/old_sanity_reports/`

Recommended default: leave these alone until the project is pushed to GitHub or packaged for handoff. They are small and do not interfere with Milestone C.

## 4. Repository State Before Vs After

### Before

- Milestone C had previously accumulated failed local smoke artifacts.
- The environment had previously been confused by a broken `requirements_local_full.txt` / NumPy 2.x snapshot.
- Compiled Python caches existed in project code and the patched PandaSet devkit.
- Milestone B checkpoints had previously been visible in Open3D checkpoint locations, creating accidental-resume risk.

### After

- Historical state at audit time: `logs/milestone_c/runs/` was empty and ready for real C0 runs. Current state: it contains completed C0 artifacts.
- `logs/milestone_c/configs/` is empty; no dead smoke configs remain.
- Active checkpoint risk is cleared: all `.pth` files are archived under `logs/milestone_b_archive/`.
- Only `requirements_working_panda.txt` remains as the environment source of truth.
- `numpy==1.26.4` is the intended working NumPy version, not NumPy 2.x.
- No compiled Python cache artifacts remain in `src/`, `tools/`, `datasets/`, or `pandaset-devkit/`.
- Protected artifacts were not touched: raw intensity analysis, milestone archives, split files, PandaSet data, production config, and audit scripts remain intact.

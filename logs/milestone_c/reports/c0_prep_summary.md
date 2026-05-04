# C0 Prep Summary

## Training Script Status

- File: `tools/train_milestone_c.py`
- Line count: 255 lines
- Status: Created and `py_compile` passes.
- Purpose: Milestone C training driver, separate from `tools/sanity_train_check.py`.
- CLI implemented: `--config`, `--run-name`, `--epochs`, `--no-resume`, `--seed`, `--steps-per-epoch-train`, `--steps-per-epoch-valid`, `--force`.
- Startup artifacts implemented: `config_snapshot.yml`, `git_commit.txt`, `seed.txt`, `cli_args.json`, `start_time.txt`, `stdout.log`.
- Resume prevention: sets `model_cfg["ckpt_path"] = None` and `model_cfg["is_resume"] = False` before model/pipeline construction. Commented in code because Open3D-ML normally auto-resumes when `is_resume=True` and visible checkpoints exist.
- Run-local output routing: sets `pipeline.main_log_dir` to `logs/milestone_c/runs/<run-name>/open3d_logs` and `train_sum_dir` to `logs/milestone_c/runs/<run-name>/tensorboard`.
- Checkpoints: custom `MilestoneCPipeline.save_ckpt()` writes to `logs/milestone_c/runs/<run-name>/checkpoints/` every 10 human epochs and at final epoch.
- Epoch telemetry: custom `save_logs()` appends wall-clock seconds and peak GPU memory bytes to `training_log.txt` after each epoch.
- Deviation: script is slightly above the requested rough target (`255` lines vs `~250`) because it includes tee logging, run-artifact writing, checkpoint routing, and explicit progress logs.

## Smoke Test Result

- Exact production-config smoke attempt: started with `configs/randlanet_pandaset_ff_lane3.yml`, but was stopped because local CPU attempted full `num_points: 16384` work and did not reach a useful training step quickly.
- Temporary smoke config created: `logs/milestone_c/configs/c0_local_smoke_1024.yml`, later removed by user cleanup.
- Long tiny smoke attempt: `num_points: 1024`, `epochs: 1`, `steps_per_epoch_train: 2`, `steps_per_epoch_valid: 1`, `batch_size: 1`, `device: cpu`.
- Last observed smoke state: reached `run_train_start`, then stayed inside Open3D-ML `pipeline.run_train()` for more than an hour with high CPU usage and no Open3D training log output.
- Result: local full Open3D-ML CPU smoke did **not** complete; no `end_time.txt`, no checkpoint, no `training_log.txt`.
- User cleanup performed after laptop issue: removed `logs/milestone_c/runs/C0_local_smoke` and `logs/milestone_c/configs/c0_local_smoke_1024.yml`.
- Current local env check: imports pass for NumPy, Torch, Open3D, Open3D-ML `SemanticSegmentation`, and `PandaSetFFLane3Dataset`.
- Current versions: `numpy 1.26.4`, `torch 2.2.2+cu121`, `open3d 0.19.0`.
- Non-fatal environment note: local venv had drifted to `numpy 2.4.4`, which broke Open3D/TensorBoard imports. It was restored to the known working project pin `numpy==1.26.4` from `requirements_working_panda.txt`.

## Cleanup Summary

- YAML duplicate check: `configs/randlanet_pandaset_ff_lane3.yml` currently has only one `num_workers: 0` key; parsed value is `0`. No YAML edit was needed.
- Sanity script change: `tools/sanity_train_check.py` modified only to avoid the dual-writer issue. `_flush_report()` now skips script-side `write_text()` when stdout is already redirected to `logs/milestone_b_sanity_train_report.txt`.
- Milestone B active checkpoints moved out of active Open3D path:
  - From `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/ckpt_00000.pth`
  - From `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/ckpt_00002.pth`
  - To `logs/milestone_b_archive/checkpoints/`
- Remaining `.pth` files under `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/` are inside the old subfolder `archived_day6_prererun_2026-04-15/`, not directly visible as latest active checkpoints.

## Open Questions Or Risks

- Local CPU Open3D-ML training is not a useful smoke path for this pipeline: even tiny `1024`-point config reached `run_train_start` but did not produce a training step/log in over an hour.
- Recommended next smoke: run the same `tools/train_milestone_c.py` on the DigitalOcean GPU Droplet with a short GPU smoke before the real C0 baseline.
- Validation metric plumbing is still not implemented by design; next session should add mIoU/per-class IoU/precision/recall/confusion matrix logging.
- Split audit is still blocked for strict verdict rules because complete per-sequence class counts exist for train only, not val/test.

## Follow-Up Cleanup

- Removed dead Milestone C smoke config: `logs/milestone_c/configs/c0_local_smoke_cpu_tiny.yml`.
- Removed failed/incomplete smoke run artifacts; `logs/milestone_c/runs/` is currently empty.
- Removed misleading broken environment snapshot: `requirements_local_full.txt` because it pinned `numpy==2.4.4`, which broke Open3D/TensorBoard imports. Keep using `requirements_working_panda.txt` with `numpy==1.26.4`.
- Removed duplicated raw-intensity smoke outputs: `logs/raw_intensity_analysis_smoke/` and `logs/raw_intensity_analysis_fast_smoke/`. Full analysis remains at `logs/raw_intensity_analysis/`.
- Moved old invalid/stale Milestone B sanity report fragments to `logs/milestone_b_archive/old_sanity_reports/`.
- Moved the recursive archived Day 6 checkpoint folder out of the active Open3D checkpoint tree to `logs/milestone_b_archive/open3d_checkpoint_archives/`.
- Verified `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/` no longer contains `.pth` files directly or recursively.

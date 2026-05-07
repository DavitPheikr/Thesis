# C0 Prep Summary

## Current Server Update: 2026-05-05

- C0 sampler decision: use `SemSegRandomSampler`.
- Reason: `SemSegSpatiallyRegularSampler` was benchmarked on the server and found to eagerly preprocess the full split before epoch 1. One 80-frame training sequence took `80.98s`, about `1.01s/frame`, estimating `~1.30h` for the 4640-frame train split and roughly `~12min` for the 720-frame validation split.
- Server dataset root confirmed: `/home/coder/project/pandaset/PandaSet`.
- Dataset structure confirmed: 103 PandaSet sequences with per-sequence `annotations`, `camera`, `lidar`, and `meta` directories.
- Dataset class confirmed: `training len 4640`, `validation len 720`, `test len 720`, labels `[0, 1, 2, 3]`.
- GPU confirmed: `NVIDIA A100 80GB PCIe MIG 3g.40gb`, `torch.cuda.is_available() True`, `device_count 1`.
- Patched devkit import confirmed: `/home/coder/project/pandaset-devkit/python/pandaset/__init__.py`.
- Class weights confirmed active in server loss: effective CE weights are `[2.3753318786621094, 36.98638153076172, 1.6340690851211548]` for `road/lane/other`.
- Successful random-sampler smoke: `C0_gpu_tiny_smoke_random`, 1 train step + 1 validation step, `wall_clock=6.533s`, checkpoint saved.
- Successful full-model random-sampler smoke: `C0_gpu_fullmodel_1step_random`, 1 train step + 1 validation step, `wall_clock=6.135s`, checkpoint saved.
- Spatial sampler status: deferred for C0. It is class-blind and expensive at startup; the planned C2 lane-aware sampler remains the sampling contribution that directly targets lane rarity.
- Detailed current report: `logs/milestone_c/reports/c0_sampler_and_server_readiness.md`.

## C0 Medium-Run And Benchmark Update: 2026-05-05

- New detailed report: `logs/milestone_c/reports/c0_medium_runs_and_speed_benchmarks.md`.
- Batch-size-1 medium run: `C0_baseline_medium_10ep_random`, 10 epochs, `500` train steps, `200` validation steps, `batch_size=1`, `num_workers=0`, `pin_memory=false`.
- Batch-size-1 result: train loss improved `0.708701 -> 0.339711`, val loss improved `0.489297 -> 0.408176`, mIoU improved `0.502071 -> 0.558003`, lane IoU improved `0.077419 -> 0.160835`, lane F1 improved `0.143712 -> 0.277103`.
- Batch-size-1 best lane epoch: epoch 7, lane IoU `0.192446`, lane precision `0.250676`, lane recall `0.453093`, lane F1 `0.322776`.
- Batch-size-1 runtime: about `576s` per epoch after warmup, peak PyTorch GPU memory about `214MB`.
- Batch-size-2 medium run: `C0_baseline_medium_10ep_random_bs2`, 10 epochs, same train/validation step settings, `batch_size=2`, `num_workers=0`, `pin_memory=false`.
- Batch-size-2 result: completed and learned, but was slower in the realistic medium run and over-predicted lane by epoch 10. Final lane precision/recall/F1 was `0.104447 / 0.925006 / 0.187700`, with predicted lane about `5.88%` while true lane support was about `0.66%`.
- Batch-size-2 runtime: about `644s` per epoch after warmup, peak PyTorch GPU memory about `281MB`.
- Speed benchmarks showed that `num_workers=1`, `2`, and `4` all fail with DataLoader worker segmentation faults. Do not use worker subprocesses for official C0 on this environment.
- Speed benchmarks showed that `pin_memory=true` is slower than `pin_memory=false` for zero-worker runs.
- Official C0 runtime recommendation: `batch_size=1`, `val_batch_size=1`, `num_workers=0`, `pin_memory=false`, `device=cuda`.
- Official C0 step recommendation: `steps_per_epoch_train=4640`, `steps_per_epoch_valid=720`, with `SemSegRandomSampler`.
- Estimated official 30-epoch C0 runtime on the current server: about `36-42 hours`.

## C0 Full Baseline Completion: 2026-05-07

- New detailed report: `logs/milestone_c/reports/c0_full_baseline_results.md`.
- Official full C0 run: `C0_baseline_full_30ep_random_bs1`.
- Run directory: `logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/`.
- Completed on server workspace `/home/coder/project` with Conda env `panda312`.
- Runtime: about `41h 00m 31s`, consistent with the earlier `36-42h` estimate.
- Completed config: `SemSegRandomSampler`, `num_points=16384`, `steps_per_epoch_train=4640`, `steps_per_epoch_valid=720`, `batch_size=1`, `val_batch_size=1`, `num_workers=0`, `pin_memory=false`, `device=cuda`, seed `42`, 30 epochs.
- Artifact counts verified: `eval_history.csv` has 31 lines including header, 30 `eval_epoch_*.json` files, 30 `confusion_epoch_*.npy` files, and 30 checkpoint files.
- Integrity checks: `artifact_sha256.txt` and `checkpoints_sha256.txt` both verified successfully locally after GitHub pull.
- Selected C0 checkpoint: epoch 18 (`checkpoints/ckpt_epoch_00018.pth`), with lane IoU `0.310355`, lane precision `0.437552`, lane recall `0.516349`, lane F1 `0.473696`, and mIoU `0.703805`.
- Final checkpoint: epoch 30 (`checkpoints/ckpt_epoch_00030.pth`), with lane IoU `0.264658`, lane precision `0.321885`, lane recall `0.598172`, lane F1 `0.418545`, and mIoU `0.683506`.
- Interpretation: C0 learned lane and did not collapse. Epoch 18 is more balanced; epoch 30 finds more lane points but over-predicts lane relative to support (`1.487%` predicted lane vs `0.800%` true lane).
- Canonical final plots live in `logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/plots/`. Do not use `plots_pre_final/` for reporting; it contains mid-training plot outputs.

## Training Script Status

- File: `tools/train_milestone_c.py`
- Current line count: 625 lines.
- Status: Created, validation metric plumbing added, and earlier `py_compile` checks passed.
- Purpose: Milestone C training driver, separate from `tools/sanity_train_check.py`.
- CLI implemented: `--config`, `--run-name`, `--epochs`, `--no-resume`, `--seed`, `--steps-per-epoch-train`, `--steps-per-epoch-valid`, `--force`.
- Startup artifacts implemented: `config_snapshot.yml`, `git_commit.txt`, `seed.txt`, `cli_args.json`, `start_time.txt`, `stdout.log`.
- Resume prevention: sets `model_cfg["ckpt_path"] = None` and `model_cfg["is_resume"] = False` before model/pipeline construction. Commented in code because Open3D-ML normally auto-resumes when `is_resume=True` and visible checkpoints exist.
- Run-local output routing: sets `pipeline.main_log_dir` to `logs/milestone_c/runs/<run-name>/open3d_logs` and `train_sum_dir` to `logs/milestone_c/runs/<run-name>/tensorboard`.
- Checkpoints: custom `MilestoneCPipeline.save_ckpt()` writes to `logs/milestone_c/runs/<run-name>/checkpoints/` according to `cfg.save_ckpt_freq` on human epoch numbers and at final epoch. The official C0 run used `save_ckpt_freq=1`, so all 30 checkpoints were retained.
- Epoch telemetry: custom `save_logs()` appends wall-clock seconds and peak GPU memory bytes to `training_log.txt` after each epoch.
- Deviation: script is above the original lightweight-driver target because it now mirrors the Open3D training loop to persist thesis-critical validation artifacts and debug startup bottlenecks.

## Smoke Test Result

- Exact production-config smoke attempt: started with `configs/randlanet_pandaset_ff_lane3.yml`, but was stopped because local CPU attempted full `num_points: 16384` work and did not reach a useful training step quickly.
- Temporary smoke config created: `logs/milestone_c/configs/c0_local_smoke_1024.yml`. If this file exists in a current checkout, treat it as a stale local smoke config, not as a valid C0 training config.
- Long tiny smoke attempt: `num_points: 1024`, `epochs: 1`, `steps_per_epoch_train: 2`, `steps_per_epoch_valid: 1`, `batch_size: 1`, `device: cpu`.
- Last observed smoke state: reached `run_train_start`, then stayed inside Open3D-ML `pipeline.run_train()` for more than an hour with high CPU usage and no Open3D training log output.
- Result: local full Open3D-ML CPU smoke did **not** complete; no `end_time.txt`, no checkpoint, no `training_log.txt`.
- Later diagnosis: this local stall was consistent with the same `SemSegSpatiallyRegularSampler` eager initialization behavior measured on the server.
- User cleanup performed after laptop issue: removed `logs/milestone_c/runs/C0_local_smoke`; local smoke configs may reappear as tracked or scratch artifacts and should not be used for real runs.
- Current local env check: imports pass for NumPy, Torch, Open3D, Open3D-ML `SemanticSegmentation`, and `PandaSetFFLane3Dataset`.
- Current versions: `numpy 1.26.4`, `torch 2.2.2+cu121`, `open3d 0.19.0`.
- Non-fatal environment note: local venv had drifted to `numpy 2.4.4`, which broke Open3D/TensorBoard imports. It was restored to the known working project pin `numpy==1.26.4` from `requirements_working_panda.txt`.

## Server Smoke Test Result

- `C0_gpu_tiny_smoke_random`: `SemSegRandomSampler`, tiny model, 1 train step, 1 validation step. Completed in `6.533s`.
- `C0_gpu_fullmodel_1step_random`: `SemSegRandomSampler`, full model config, 1 train step, 1 validation step. Completed in `6.135s`.
- Both runs reached `epoch_start 0`, completed training and validation, and saved checkpoints.
- `nvidia-smi` after completion correctly showed no running process and MIG memory returned to idle.
- These results confirm that the server environment, dataset, full model construction, CUDA path, validation path, and checkpoint path are working.

## Cleanup Summary

- YAML duplicate check: `configs/randlanet_pandaset_ff_lane3.yml` currently has only one `num_workers: 0` key; parsed value is `0`. No YAML edit was needed.
- Sanity script change: `tools/sanity_train_check.py` modified only to avoid the dual-writer issue. `_flush_report()` now skips script-side `write_text()` when stdout is already redirected to `logs/milestone_b_sanity_train_report.txt`.
- Milestone B active checkpoints moved out of active Open3D path:
  - From `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/ckpt_00000.pth`
  - From `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/ckpt_00002.pth`
  - To `logs/milestone_b_archive/checkpoints/`
- Remaining `.pth` files under `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/` are inside the old subfolder `archived_day6_prererun_2026-04-15/`, not directly visible as latest active checkpoints.

## Open Questions Or Risks

- Local CPU Open3D-ML training is not a useful smoke path for this pipeline.
- Server smoke path is now confirmed with random sampling.
- Validation metric plumbing is implemented and has run through successful random-sampler smoke epochs.
- Do not use smoke configs with `steps_per_epoch_train: 1` / `steps_per_epoch_valid: 1` for real C0.
- Before future long runs, choose real step counts. Checkpoint cadence was patched so the custom saver uses `cfg.save_ckpt_freq` on human epoch numbers and still saves the final requested epoch.
- Split audit is still blocked for strict verdict rules because complete per-sequence class counts exist for train only, not val/test.

## Follow-Up Cleanup

- Removed dead Milestone C smoke config: `logs/milestone_c/configs/c0_local_smoke_cpu_tiny.yml`.
- Historical cleanup removed failed/incomplete smoke run artifacts. `logs/milestone_c/runs/` is no longer empty because it now contains completed medium runs and the official C0 full run.
- Removed misleading broken environment snapshot: `requirements_local_full.txt` because it pinned `numpy==2.4.4`, which broke Open3D/TensorBoard imports. Keep using `requirements_working_panda.txt` with `numpy==1.26.4`.
- Removed duplicated raw-intensity smoke outputs: `logs/raw_intensity_analysis_smoke/` and `logs/raw_intensity_analysis_fast_smoke/`. Full analysis remains at `logs/raw_intensity_analysis/`.
- Moved old invalid/stale Milestone B sanity report fragments to `logs/milestone_b_archive/old_sanity_reports/`.
- Moved the recursive archived Day 6 checkpoint folder out of the active Open3D checkpoint tree to `logs/milestone_b_archive/open3d_checkpoint_archives/`.
- Verified `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/` no longer contains `.pth` files directly or recursively.

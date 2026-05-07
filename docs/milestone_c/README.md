# Milestone C Notes

Generated: 2026-05-07

Milestone C is the real RandLA-Net experiment stage. It starts from the Milestone B dataset/class-weight foundation and moves toward C0-C4 training runs.

## Current C0 State

C0 is now defined as:

```text
RandLA-Net + xyz_ego + standardized intensity + SemSegRandomSampler + class-weighted CE
```

Important current decisions:

- Use `SemSegRandomSampler` for C0.
- Defer `SemSegSpatiallyRegularSampler` because it eagerly preprocesses the full split before epoch 1.
- Keep Open3D-native measured-count class weights; server verification confirmed they are active in the CE loss.
- Use the server dataset path `/home/coder/project/pandaset/PandaSet` for cloud runs.
- Official full C0 used `batch_size: 1`, `val_batch_size: 1`, `num_workers: 0`, and `pin_memory: false`.
- Do not use PyTorch DataLoader workers on the current server setup; `num_workers > 0` repeatedly segfaulted in worker subprocesses.
- Do not use `pin_memory: true` for the zero-worker C0 path; it was slower in server benchmarks.

## Current Evidence

The server now has full C0 baseline evidence:

- Random-sampler one-step smoke runs proved the CUDA, validation, and checkpoint paths.
- A 10-epoch medium C0 run with `batch_size=1` showed real learning and the best lane balance observed so far.
- A 10-epoch medium run with `batch_size=2` completed, but was slower in the realistic run and over-predicted lane by epoch 10.
- Speed benchmarks showed that `num_workers=1/2/4` are unstable and that `pin_memory=true` slows zero-worker runs.
- The official 30-epoch C0 run completed as `C0_baseline_full_30ep_random_bs1`.
- Epoch 18 is the selected C0 checkpoint by validation lane F1, lane IoU, and mIoU.

Completed full C0 settings:

```yaml
dataset:
  sampler:
    name: SemSegRandomSampler
  steps_per_epoch_train: 4640
  steps_per_epoch_valid: 720

pipeline:
  batch_size: 1
  val_batch_size: 1
  num_workers: 0
  pin_memory: false
  device: cuda
```

The completed 30-epoch run took about `41h 00m 31s` on the current A100 MIG server. `--save-ckpt-freq 1` was used, and all 30 epoch checkpoints are present.

Headline C0 result:

```text
run_dir: logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/
best_checkpoint: checkpoints/ckpt_epoch_00018.pth
best_lane_f1: 0.473696
best_lane_iou: 0.310355
best_mIoU: 0.703805
final_epoch_lane_f1: 0.418545
final_epoch_lane_iou: 0.264658
```

The final epoch has higher lane recall but worse lane precision than epoch 18, so epoch 18 is the selected C0 model for reporting and qualitative analysis.

## Key Reports

- `logs/milestone_c/reports/c0_sampler_and_server_readiness.md`
  - server readiness, dataset checks, sampler benchmark, smoke results, and C0 sampler decision.
- `logs/milestone_c/reports/c0_medium_runs_and_speed_benchmarks.md`
  - 10-epoch medium C0 results, batch-size comparison, DataLoader worker failures, pin-memory benchmarks, and final full-run settings.
- `logs/milestone_c/reports/c0_full_baseline_results.md`
  - official full C0 result, artifact inventory, selected checkpoint, plot guide, confusion matrices, and interpretation.
- `logs/milestone_c/reports/class_weight_decision.md`
  - class-weight policy and server loss verification.
- `logs/milestone_c/reports/validation_metrics_prep.md`
  - validation metric plumbing, class indexing, and artifact design.
- `logs/milestone_c/reports/c0_prep_summary.md`
  - historical C0 prep summary plus current server update.
- `docs/milestone_c_option_a_execution_plan.md`
  - full C0-C4 experiment plan.

## Post-Run Plots

After any Milestone C run has an `eval_history.csv`, generate thesis-friendly plots and a compact run summary with:

```bash
./panda/bin/python tools/plot_milestone_c_run.py --run-name C0_baseline_full_30ep_random_bs1
```

On the server, use the active Conda Python instead:

```bash
python tools/plot_milestone_c_run.py --run-name C0_baseline_full_30ep_random_bs1
```

The script writes `plots/` inside the run directory, including:

- `metrics_overview.png`
- `loss_curves.png`
- `per_class_iou.png`
- `lane_precision_recall_f1.png`
- `lane_recall_by_distance.png`
- `runtime_and_memory.png`
- `class_true_vs_predicted_share.png`
- final and best-lane-F1 confusion-matrix heatmaps
- `run_summary.md`

For the completed C0 run, the canonical final plots are in:

```text
logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/plots/
```

Do not use `plots_pre_final/` for reporting; it contains mid-training plot outputs generated before the final post-run plotting pass.

## Server-Specific Reminder

Server activation:

```bash
cd /home/coder/project
source envStart.sh
```

The server-specific configs created under `logs/milestone_c/configs/` may be generated and untracked. Before any future long run, inspect the config snapshot carefully and ensure it is not still a smoke config with `steps_per_epoch_train: 1` and `steps_per_epoch_valid: 1`.

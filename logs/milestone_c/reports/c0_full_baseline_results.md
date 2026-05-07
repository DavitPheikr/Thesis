# C0 Full Baseline Results

Generated: 2026-05-07

This report records the completed official C0 baseline run and explains the main artifacts, metrics, and interpretation.

## Run Identity

```text
run_name: C0_baseline_full_30ep_random_bs1
run_dir: logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/
server_workspace: /home/coder/project
server_env: panda312
server_dataset_root: /home/coder/project/pandaset/PandaSet
run_start: 2026-05-05T15:07:30.065620
run_end: 2026-05-07T08:08:01.203334
wall_clock_seconds: 147631.148
wall_clock_hms: about 41h 00m 31s
seed: 42
training_code_commit: b70a64d91857b49aeaf749e710f45999c81a3cff
artifact_commit: 9d36578 Add C0 full baseline run artifacts
```

The training code commit is the commit recorded by the run itself in `git_commit.txt`. The later artifact commit is the Git commit that added the completed run directory to the repository.

## C0 Configuration

The exact run config is preserved at:

```text
logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/config_snapshot.yml
```

The C0 baseline used:

```text
model: Open3D-ML RandLA-Net
input: xyz_ego + standardized intensity
classes: road / lane / other
ignore label: 0, excluded from active metrics and loss
sampler: SemSegRandomSampler
num_points: 16384
steps_per_epoch_train: 4640
steps_per_epoch_valid: 720
epochs: 30
batch_size: 1
val_batch_size: 1
num_workers: 0
pin_memory: false
device: cuda
save_ckpt_freq: 1
optimizer_lr: 0.001
scheduler_gamma: 0.99
class_weight_policy: Open3D-native measured-count class weights
```

Class weights were passed as measured training-set counts:

```text
road:  119562394.0
lane:    2098182.0
other: 176504630.0
```

Open3D transforms these internally before constructing the active-class weighted cross entropy loss.

## Artifact Inventory

Important files in the run directory:

```text
eval_history.csv
eval_epoch_001.json ... eval_epoch_030.json
confusion_epoch_001.npy ... confusion_epoch_030.npy
checkpoints/ckpt_epoch_00001.pth ... ckpt_epoch_00030.pth
config_snapshot.yml
cli_args.json
git_commit.txt
seed.txt
start_time.txt
end_time.txt
stdout.log
training_log.txt
artifact_manifest.txt
artifact_sha256.txt
checkpoints_sha256.txt
plots/
plots_pre_final/
```

Artifact counts verified locally after pulling from GitHub:

```text
eval_history.csv lines: 31 including header
eval_epoch_*.json: 30
confusion_epoch_*.npy: 30
checkpoints/ckpt_epoch_*.pth: 30
```

Hash verification passed for both result artifacts and checkpoints using:

```bash
sha256sum -c logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/artifact_sha256.txt
sha256sum -c logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints_sha256.txt
```

`plots/` contains the canonical final plots generated after the run finished. `plots_pre_final/` contains plots that were generated during training before the final post-run plotting pass; keep it only as historical context and do not use it for reporting.

## Headline Metrics

The best validation checkpoint is epoch 18, not the final epoch.

| metric | first epoch | final epoch | best epoch | best value |
| --- | ---: | ---: | ---: | ---: |
| train_loss | 0.433675 | 0.113191 | 30 | 0.113191 |
| val_loss | 0.461187 | 0.271130 | 20 | 0.243199 |
| mIoU | 0.613929 | 0.683506 | 18 | 0.703805 |
| lane_iou | 0.187062 | 0.264658 | 18 | 0.310355 |
| lane_f1 | 0.315167 | 0.418545 | 18 | 0.473696 |

Use this checkpoint for C0 model selection:

```text
logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints/ckpt_epoch_00018.pth
```

Keep this checkpoint for final-epoch comparison:

```text
logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints/ckpt_epoch_00030.pth
```

## Epoch 18: Selected C0 Checkpoint

Epoch 18 is the selected C0 checkpoint because it has the best lane F1, best lane IoU, and best mIoU.

```text
train_loss     0.133833
val_loss       0.260416
mIoU           0.703805
road_iou       0.881990
lane_iou       0.310355
other_iou      0.919072
lane_precision 0.437552
lane_recall    0.516349
lane_f1        0.473696
```

Epoch 18 class shares:

```text
true road/lane/other:      40.640% / 0.819% / 58.541%
predicted road/lane/other: 44.503% / 0.967% / 54.530%
```

The predicted lane share is close to the true lane share, so epoch 18 is more balanced than the final epoch.

Epoch 18 confusion matrix:

```text
              predicted road   predicted lane   predicted other
true road        4,696,484          44,326            42,531
true lane           44,563          49,775             2,060
true other         496,970          19,657         6,373,555
```

Row-normalized interpretation:

```text
road: 98.18% road, 0.93% lane, 0.89% other
lane: 46.23% road, 51.63% lane, 2.14% other
other: 7.21% road, 0.29% lane, 92.50% other
```

The main lane error at the selected checkpoint is lane being missed as road.

## Epoch 30: Final Checkpoint

Epoch 30 has the lowest training loss but not the best validation lane performance.

```text
train_loss     0.113191
val_loss       0.271130
mIoU           0.683506
road_iou       0.866600
lane_iou       0.264658
other_iou      0.919261
lane_precision 0.321885
lane_recall    0.598172
lane_f1        0.418545
```

Epoch 30 class shares:

```text
true road/lane/other:      38.358% / 0.800% / 60.841%
predicted road/lane/other: 41.804% / 1.487% / 56.709%
```

The predicted lane share is about `1.86x` the true lane share:

```text
1.487 / 0.800 = 1.86
```

Epoch 30 therefore finds more lane points than epoch 18, but it also creates more lane false positives.

Epoch 30 confusion matrix:

```text
              predicted road   predicted lane   predicted other
true road        4,380,056          88,801            45,572
true lane           35,604          56,346             2,247
true other         504,265          29,903         6,626,263
```

Row-normalized interpretation:

```text
road: 97.02% road, 1.97% lane, 1.01% other
lane: 37.80% road, 59.82% lane, 2.39% other
other: 7.04% road, 0.42% lane, 92.54% other
```

The final model has higher lane recall than epoch 18, but lower lane precision and lower lane F1.

## Plot Guide

Canonical final plots live under:

```text
logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/plots/
```

Use each plot as follows:

- `metrics_overview.png`: best single overview of loss, mIoU, lane IoU/F1, and per-class IoU.
- `loss_curves.png`: shows train loss decreasing while validation does not monotonically improve; supports best-checkpoint selection.
- `per_class_iou.png`: shows road/other are strong and lane remains the limiting class.
- `lane_precision_recall_f1.png`: shows the precision/recall tradeoff and why epoch 18 beats epoch 30.
- `lane_recall_by_distance.png`: shows lane recall by range bucket; interpret alongside precision because high recall can come from over-prediction.
- `runtime_and_memory.png`: shows per-epoch wall-clock and PyTorch peak memory telemetry.
- `class_true_vs_predicted_share.png`: compares class support against predicted class share; use it to diagnose class over- or under-prediction.
- `confusion_best_lane_f1_epoch_018.png`: preferred confusion heatmap for the selected C0 checkpoint.
- `confusion_epoch_030.png`: final-epoch confusion heatmap.
- `run_summary.md`: compact machine-generated summary of first/final/best values and final class shares.

## Validation Semantics

Validation does not update model weights. It measures the checkpoint after each training epoch on held-out validation frames.

For C0:

```text
training per epoch:   4640 steps
validation per epoch: 720 steps
batch_size:           1
val_batch_size:       1
```

Validation accumulates a confusion matrix over active classes after ignore-label filtering. Active-class index order is:

```text
0 = road
1 = lane
2 = other
```

The plot `class_true_vs_predicted_share.png` is derived from each epoch confusion matrix:

```text
true class percent      = row sum for class / all validation points
predicted class percent = column sum for class / all validation points
```

For epoch 30 lane:

```text
true lane count      = 94,197
predicted lane count = 175,050
total active points  = 11,769,057

true lane percent      = 94,197 / 11,769,057 = 0.800%
predicted lane percent = 175,050 / 11,769,057 = 1.487%
```

This is why epoch 30 is described as over-predicting lane.

## Interpretation

C0 is a valid baseline. It does not collapse to majority classes, and it learns a measurable lane signal.

The main findings are:

1. Road and other segmentation are strong relative to lane.
2. Lane remains the limiting class.
3. The best validation model is epoch 18.
4. More training after epoch 18 lowers training loss but does not improve lane validation quality.
5. The final epoch shifts toward higher lane recall but lower precision because it predicts lane too often.

For reporting and downstream qualitative analysis, use epoch 18 as the selected C0 checkpoint. Use epoch 30 only as the final-training checkpoint.

## Next Experiment Implications

C0 creates a defensible baseline for later Milestone C experiments.

Most relevant next levers:

- `C1`: engineered feature inputs, especially local contrast and range/intensity variants motivated by raw-intensity analysis.
- `C2`: lane-aware sampling, because lane is rare and C0 still misses many true lane points.
- `C3`: combined features plus lane-aware sampling.
- `C4`: conservative augmentation only after C1-C3 are understood.

Avoid judging later runs by mIoU alone. Lane IoU, lane precision, lane recall, lane F1, predicted lane share, and confusion matrices must be compared against this C0 result.


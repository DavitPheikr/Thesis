# C0 Medium Runs And Speed Benchmarks

Generated: 2026-05-05

## Purpose

This report records the non-smoke C0 evidence collected after server readiness was confirmed.

It answers four operational questions:

- Does the C0 baseline learn over more than a one-step smoke test?
- Is `batch_size=2` a safe speed optimization for the official baseline?
- Can PyTorch DataLoader workers be used to feed the GPU faster?
- What settings should be used for the full C0 baseline run?

## Common C0 Setup

All runs in this report used the same core C0 baseline:

```text
model: Open3D-ML RandLA-Net
dataset: PandaSet forward-facing LiDAR
classes: road, lane, other
ignored label: 0
input: xyz_ego + standardized intensity
num_points: 16384
sampler: SemSegRandomSampler
loss: Open3D SemSegLoss weighted CE
seed: 42
device: cuda
server dataset root: /home/coder/project/pandaset/PandaSet
```

Class weighting had already been verified on the server:

```text
effective CE weights:
road  = 2.3753318786621094
lane  = 36.98638153076172
other = 1.6340690851211548
```

## Medium Run A: Batch Size 1

Run:

```text
C0_baseline_medium_10ep_random
```

Config summary:

```text
epochs: 10
steps_per_epoch_train: 500
steps_per_epoch_valid: 200
batch_size: 1
val_batch_size: 1
num_workers: 0
pin_memory: false
```

Observed run artifacts:

```text
logs/milestone_c/runs/C0_baseline_medium_10ep_random/
```

Learning summary:

```text
train_loss: 0.708701 -> 0.339711
val_loss:   0.489297 -> 0.408176
mIoU:       0.502071 -> 0.558003
lane IoU:   0.077419 -> 0.160835
lane F1:    0.143712 -> 0.277103
```

Best observed epoch:

```text
best mIoU:           epoch 7, 0.584147
best lane IoU:       epoch 7, 0.192446
best lane precision: epoch 7, 0.250676
best lane F1:        epoch 7, 0.322776
best val loss:       epoch 9, 0.374263
```

Epoch 7 lane details:

```text
lane IoU:       0.192446
lane precision: 0.250676
lane recall:    0.453093
lane F1:        0.322776
true lane pct:  0.682%
pred lane pct:  1.232%
```

Epoch 10 lane details:

```text
lane IoU:       0.160835
lane precision: 0.185737
lane recall:    0.545379
lane F1:        0.277103
true lane pct:  0.664%
pred lane pct:  1.948%
```

Interpretation:

- Batch size 1 clearly learns.
- Road and other classes become strong.
- Lane class improves, but remains the limiting class.
- Lane precision/recall are reasonably balanced around the best epoch.
- Validation has random-sampling noise, including a noisy epoch 8 where lane recall rose but lane precision collapsed.

Runtime:

```text
average epoch wall-clock, all epochs:          593.301s
average epoch wall-clock, excluding warmup:    576.046s
peak PyTorch GPU memory:                       214071808 bytes
checkpoint saved:                              ckpt_epoch_00010.pth
```

## Medium Run B: Batch Size 2

Run:

```text
C0_baseline_medium_10ep_random_bs2
```

Config summary:

```text
epochs: 10
steps_per_epoch_train: 500
steps_per_epoch_valid: 200
batch_size: 2
val_batch_size: 2
num_workers: 0
pin_memory: false
```

Observed run artifacts:

```text
logs/milestone_c/runs/C0_baseline_medium_10ep_random_bs2/
```

Learning summary:

```text
train_loss: 0.736282 -> 0.339812
val_loss:   0.557304 -> 0.267007
mIoU:       0.509646 -> 0.595434
lane IoU:   0.063198 -> 0.103570
lane F1:    0.118883 -> 0.187700
```

Best observed epoch:

```text
best mIoU:           epoch 5, 0.619056
best lane IoU:       epoch 4, 0.184496
best lane precision: epoch 4, 0.219256
best lane recall:    epoch 10, 0.925006
best lane F1:        epoch 4, 0.311518
best val loss:       epoch 10, 0.267007
```

Epoch 7 lane details:

```text
lane IoU:       0.102329
lane precision: 0.104689
lane recall:    0.819507
lane F1:        0.185660
true lane pct:  0.682%
pred lane pct:  5.337%
```

Epoch 10 lane details:

```text
lane IoU:       0.103570
lane precision: 0.104447
lane recall:    0.925006
lane F1:        0.187700
true lane pct:  0.664%
pred lane pct:  5.877%
```

Interpretation:

- Batch size 2 also learns and produces strong mIoU/val-loss numbers.
- However, it shifts lane behavior toward very high recall and low precision.
- By epoch 10 it predicts lane on about `5.88%` of validation points while true lane support is only about `0.66%`.
- This is a lane false-positive problem and is less attractive for the official baseline than batch size 1.
- Batch size 2 changes optimization because Open3D batches the sampled items: observed training batches per epoch dropped from `500` to `250` and validation batches from `200` to `100`.

Runtime:

```text
average epoch wall-clock, all epochs:          643.442s
average epoch wall-clock, excluding warmup:    644.364s
peak PyTorch GPU memory:                       281381376 bytes
checkpoint saved:                              ckpt_epoch_00010.pth
```

Important result: in the realistic 10-epoch run, batch size 2 was slower than batch size 1. The earlier 1-epoch tiny speed benchmark was misleading.

## Speed Benchmark 1

Server benchmark file observed:

```text
logs/milestone_c/reports/c0_speed_benchmark_20260505T033313Z.csv
```

This benchmark used:

```text
epochs: 1
steps_per_epoch_train: 100
steps_per_epoch_valid: 50
```

Summary:

```text
bs1 nw0 pinfalse: 186s, status 0
bs2 nw0 pinfalse: 151s, status 0
bs4 nw0 pinfalse: 154s, status 0
bs8 nw0 pinfalse: 162s, status 0
bs16 nw0 pinfalse: 161s, status 0
```

Multiprocessing DataLoader worker runs failed:

```text
bs4  nw2 pintrue: status 1, DataLoader worker segmentation fault
bs8  nw2 pintrue: status 1, DataLoader worker segmentation fault
bs16 nw2 pintrue: status 1, DataLoader worker segmentation fault
bs4  nw4 pintrue: status 1, DataLoader worker segmentation fault
bs8  nw4 pintrue: status 1, DataLoader worker segmentation fault
bs16 nw4 pintrue: status 1, DataLoader worker segmentation fault
```

Initial takeaway:

- `num_workers > 0` looked unsafe.
- `batch_size=2` looked fastest in this very short benchmark.
- Because larger batches reduce optimizer steps per epoch, a medium learning check was still needed before adopting batch size 2.

## Speed Benchmark 2

Server benchmark file observed:

```text
logs/milestone_c/reports/c0_speed_benchmark_extra_20260505T035710Z.csv
```

This benchmark isolated pin memory and worker count:

```text
bs1 nw0 pintrue:  225s, status 0
bs2 nw0 pintrue:  169s, status 0
bs3 nw0 pinfalse: 153s, status 0
bs3 nw0 pintrue:  181s, status 0
bs4 nw0 pintrue:  175s, status 0
```

Worker isolation failures:

```text
bs2 nw1 pinfalse: status 1, DataLoader worker segmentation fault
bs2 nw1 pintrue:  status 1, DataLoader worker segmentation fault
bs2 nw2 pinfalse: status 1, DataLoader worker segmentation fault
```

Interpretation:

- `num_workers > 0` is the problem, independent of pin memory.
- `pin_memory=true` is slower than `pin_memory=false` when `num_workers=0`.
- Do not use DataLoader workers for official C0 until a separate engineering fix makes worker subprocesses stable.

## TensorBoard CLI Note

Repeated server activation/monitoring showed:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

when invoking the `tensorboard` CLI.

This did not block training. The training script's `SummaryWriter` path reached:

```text
run_train_tensorboard_done
```

in completed runs.

Interpretation:

- TensorBoard CLI viewing is broken in the current server environment because `pkg_resources` is missing.
- Training and artifact writing are not blocked.
- If TensorBoard viewing is needed later, install/fix `setuptools` in the `panda312` environment, but do not treat this as a C0 blocker.

## DataLoader Worker Failure Interpretation

All `num_workers > 0` tests failed with segmentation faults before the first training batch completed.

Representative error:

```text
ERROR: Unexpected segmentation fault encountered in worker.
RuntimeError: DataLoader worker (...) exited unexpectedly
```

Interpretation:

- This is a native/process-level crash, not a catchable Python exception.
- Likely causes include Open3D native code, PandaSet/devkit objects, pandas/native readers, or forked subprocess interactions.
- The warning around `np.uint32(torch.utils.data.get_worker_info().seed)` should be cleaned up eventually, but the crash occurs at the worker/native-code level.
- `num_workers=0` keeps loading in the main process and avoids this crash path.

## Final Recommendation For Official Full C0

Use the conservative setting that learned best and was stable:

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

Rationale:

- Batch size 1 medium run produced the best lane balance and best lane F1.
- Batch size 1 was faster than batch size 2 in the realistic 10-epoch medium comparison.
- Batch size 2 over-predicted lane substantially by epoch 10.
- `num_workers > 0` is unstable due to DataLoader worker segmentation faults.
- `pin_memory=true` slowed down zero-worker runs.
- `SemSegRandomSampler` is already the C0 sampler decision.

Estimated full C0 runtime with this setting:

```text
about 73 minutes per epoch
about 36-42 hours for 30 epochs
```

This estimate scales from the batch-size-1 medium run:

```text
500 train steps + 200 validation steps ~= 576s/epoch after warmup
4640 train steps + 720 validation steps ~= 73min/epoch
```

## Post-Run Plotting Workflow

The training run writes the data needed for analysis:

```text
eval_history.csv
eval_epoch_<NNN>.json
confusion_epoch_<NNN>.npy
training_log.txt
stdout.log
```

After a run finishes, generate plots and a compact summary with:

```bash
python tools/plot_milestone_c_run.py --run-name C0_baseline_full_30ep_random_bs1
```

The plotting script was tested locally on both medium runs. It writes a `plots/`
directory inside the run folder with:

```text
metrics_overview.png
loss_curves.png
per_class_iou.png
lane_precision_recall_f1.png
lane_recall_by_distance.png
runtime_and_memory.png
class_true_vs_predicted_share.png
confusion_epoch_<final>.png
confusion_best_lane_f1_epoch_<best>.png
run_summary.md
```

These artifacts are intended for both immediate debugging and thesis writing.

## Methodology Notes For Thesis

The C0 baseline is now well defined:

```text
RandLA-Net baseline with xyz_ego, standardized intensity, random patch sampling,
measured-count Open3D class weighting, batch size 1, and zero-worker loading.
```

The most important observed behavior is:

- The baseline learns lane, so the pipeline is valid.
- Lane remains the limiting class, so later C1/C2/C3 improvements have meaningful room.
- Batch size 2 improves val loss and recall but hurts lane precision/F1 by over-predicting lane.
- Workers would be an engineering optimization, not a thesis-method requirement, and are currently unstable.

For thesis writing, describe the worker/batch tests as a **training-systems calibration**:

```text
We benchmarked batch size, pinned memory, and DataLoader worker count on the target
server. Multiprocessing workers were unstable with Open3D/PandaSet loading, pinned
memory was slower in the zero-worker setting, and batch size 1 gave the best
representative lane precision/F1 in medium-run validation. Therefore all official
C0-Cx comparisons should use the same conservative, stable runtime settings.
```

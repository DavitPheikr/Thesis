# C0 Sampler And Server Readiness

Generated: 2026-05-05

## Decision

Milestone C C0 baseline will use:

```yaml
dataset:
  sampler:
    name: SemSegRandomSampler
```

This replaces the earlier `SemSegSpatiallyRegularSampler` default for C0.

The decision is operational and methodological:

- `SemSegRandomSampler` starts training immediately and matches RandLA-Net's random-sampling baseline design.
- `SemSegSpatiallyRegularSampler` is spatially even but class-blind; it does not directly solve the rare lane-class problem.
- Lane rarity is handled in C0 by confirmed class-weighted loss, then later by the planned C2 lane-aware sampling contribution.
- The spatial sampler is deferred, not deleted as an idea. It can still be used later as a separate ablation if we want to compare generic spatial coverage against random sampling.

## Server Environment Confirmed

Server workspace:

```text
/home/coder/project
```

Conda environment:

```text
panda312
```

Activation helper:

```bash
source envStart.sh
```

The server setup completed successfully with:

```text
python 3.12.13
torch 2.2.2+cu121
numpy 1.26.4
pandas 3.0.2
open3d 0.19.0
cuda_available True
```

GPU visible to PyTorch:

```text
NVIDIA A100 80GB PCIe MIG 3g.40gb
device_count 1
```

The harmless warning seen during imports is from Open3D:

```text
SyntaxWarning: invalid escape sequence '\d'
```

It does not block training.

## Dataset Confirmed

Correct dataset source:

```text
Kaggle dataset: pz19930809/pandaset
```

Server dataset root:

```text
/home/coder/project/pandaset/PandaSet
```

Confirmed structure:

```text
PandaSet/001/annotations
PandaSet/001/camera
PandaSet/001/lidar
PandaSet/001/meta
PandaSet/001/LICENSE.txt
```

Sequence count:

```text
103 PandaSet sequences
104 directories with find -maxdepth 1 because the root directory is included
```

Representative access checks passed:

```text
chosen_seq 001
chosen_frame 0
all_points 169171
front_points 62285
front_sensor_candidate 1
forward_sensor_usable True
semseg_loaded True
semantic alignment same_count True
intensity_column i
intensity_min 0.0
intensity_max 255.0
```

Dataset class check passed:

```text
training len 4640
validation len 720
test len 720
training labels [0, 1, 2, 3]
validation labels [0, 1, 2, 3]
test labels [0, 1, 2, 3]
dataset_class_ok True
```

## Patched PandaSet Devkit Import

The server uses the patched local PandaSet devkit:

```text
/home/coder/project/pandaset-devkit/python/pandaset/__init__.py
```

Important import rule:

```bash
export PYTHONPATH=/home/coder/project/pandaset-devkit/python:/home/coder/project/src:$PYTHONPATH
```

Reason: the extracted data folder is named `pandaset/`, so it can shadow the importable `pandaset` Python package unless the patched devkit path is placed first.

## Class Weights Confirmed

The C0 loss uses weighted cross entropy through Open3D's `SemSegLoss`.

Input stored in config/dataset:

```text
[119562394.0, 2098182.0, 176504630.0]
```

These are count-like values for active classes:

```text
road, lane, other
```

Open3D transforms them with `DataProcessing.get_class_weights(...)` before building `torch.nn.CrossEntropyLoss`.

Server verification:

```text
dataset_cfg_class_weights [119562394.0, 2098182.0, 176504630.0]
expected_effective_ce_weights [2.3753318786621094, 36.98638153076172, 1.6340690851211548]
actual_ce_weights [2.3753318786621094, 36.98638153076172, 1.6340690851211548]
```

Interpretation:

```text
road  = 2.38
lane  = 36.99
other = 1.63
```

So class weighting is not merely configured; it is actually active in the loss object.

Important caveat: do not replace `dataset.class_weights` with direct inverse-frequency CE weights unless the loss is patched. In this Open3D runtime, the configured list is transformed again.

## Spatial Sampler Diagnosis

Observed failure mode with `SemSegSpatiallyRegularSampler`:

```text
run_train_train_split_start
```

then the process stays CPU-bound before the first epoch starts.

Typical monitoring during the stall:

```text
CPU high, about 280-300%
RAM about 1.8-2.5 GB
GPU memory about 222 MiB
GPU otherwise idle
no epoch_start yet
```

Root cause:

```text
Open3D's SemSegSpatiallyRegularSampler.initialize_with_dataloader(...)
loops over every frame in the split and calls dataset.get_data(index)
and model.preprocess(...) before epoch 1.
```

For this project, that means scanning/preprocessing:

```text
4640 training frames
720 validation frames
```

This cost happens before training and is not reduced by:

```text
steps_per_epoch_train
steps_per_epoch_valid
num_points
batch_size
GPU availability
```

## Spatial Sampler Benchmark

Bounded server benchmark on one training sequence:

```text
bench_sequence 003
frames 80
dataset_split_seconds 0.1442215358838439
spatial_init_seconds 80.98244757112116
seconds_per_frame 1.0122805946390145
estimated_full_train_init_hours 1.3047172108680631
```

Estimated startup cost for full C0 with spatial sampler:

```text
training sampler init: about 1.3 hours
validation sampler init: about 12 minutes
total before epoch 1: about 1.5 hours
```

This explains the earlier "stuck" runs. They were alive, but doing eager CPU preprocessing.

## Random Sampler Smoke Results

Tiny random-sampler smoke:

```text
run_name C0_gpu_tiny_smoke_random
epochs 1
train steps 1
valid steps 1
wall_clock 6.533 seconds
checkpoint saved
```

Full-model one-step random-sampler smoke:

```text
run_name C0_gpu_fullmodel_1step_random
epochs 1
train steps 1
valid steps 1
training 1/1 in about 2.04 seconds
validation 1/1 in about 1.33 seconds
wall_clock 6.135 seconds
checkpoint saved
```

These confirm:

- training code builds the dataset, model, pipeline, optimizer, and dataloaders;
- CUDA path works;
- full RandLA-Net config can run one train and one validation step;
- validation metric plumbing can execute in a real epoch;
- checkpoint writing works;
- the bottleneck was sampler initialization, not the GPU, dataset path, or loss.

## Methodology Framing

C0 should be described as:

```text
RandLA-Net baseline with standardized intensity, random patch sampling, and class-weighted CE.
```

Do not describe C0 as spatially regular sampling.

The thesis sampling story remains:

```text
C0: random sampling + class weights
C1: engineered intensity/context features
C2: lane-aware patch sampling
C3: features + lane-aware sampling
C4: conservative augmentations
```

This is cleaner than trying to make Open3D's generic spatial sampler the sampling contribution. The planned C2 lane-aware sampler is the sampler that directly targets the rare lane class.

## Current Risks Before Long C0

- Server-generated configs under `logs/milestone_c/configs/` may still contain smoke values such as `steps_per_epoch_train: 1` and `steps_per_epoch_valid: 1`; do not use those unchanged for a real C0 run.
- Full validation with `steps_per_epoch_valid: 720` gives one sampled patch per validation frame, not exhaustive all-point validation.
- Random validation has sampling noise. Use enough validation steps to make curves meaningful.
- Checkpoint save cadence has been patched locally after this report's sampler diagnosis: `save_ckpt()` now uses `cfg.save_ckpt_freq` on human epoch numbers and saves the final requested epoch.
- The spatial sampler might become feasible with a custom lazy implementation or sequence-level caching, but it is not needed for C0.

## Post-Readiness Medium Runs And Benchmark Update

After the sampler/server-readiness work, two 10-epoch medium C0 runs and two speed-benchmark sweeps were completed on the server.

Detailed report:

```text
logs/milestone_c/reports/c0_medium_runs_and_speed_benchmarks.md
```

Key conclusions:

- `batch_size=1`, `val_batch_size=1`, `num_workers=0`, `pin_memory=false` is the recommended official C0 runtime setting.
- A 10-epoch batch-size-1 medium run learned successfully and produced the best lane balance observed so far.
- A 10-epoch batch-size-2 medium run completed, but was slower in the realistic medium run and over-predicted lane by epoch 10.
- `num_workers > 0` is currently unsafe. Worker counts 1, 2, and 4 all produced DataLoader worker segmentation faults.
- `pin_memory=true` was slower than `pin_memory=false` when `num_workers=0`.
- The TensorBoard CLI currently errors with missing `pkg_resources`; this does not block training because `SummaryWriter` still runs inside `tools/train_milestone_c.py`.

Official full C0 direction:

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

Estimated full C0 runtime:

```text
about 73 minutes per epoch
about 36-42 hours for 30 epochs
```

## Historical Recommended Next Run Config Direction

For a short pilot:

```yaml
dataset:
  sampler:
    name: SemSegRandomSampler
  steps_per_epoch_train: 100
  steps_per_epoch_valid: 50
```

For a more complete baseline after the pilot:

```yaml
dataset:
  sampler:
    name: SemSegRandomSampler
  steps_per_epoch_train: 4640
  steps_per_epoch_valid: 720
```

The second setting is closer to one sampled patch per frame per epoch, but it may take hours. Use the pilot first to estimate wall-clock time and GPU memory.

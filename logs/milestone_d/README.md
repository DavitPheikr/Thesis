# Milestone D Context

This is the canonical working context document for Milestone D. It records what
changed after C0, why it changed, which decisions were rejected, how D0 is
configured, and what caveats matter when interpreting the run. When D0 finishes,
update this file with the actual results before using it as thesis-writing
context.

## Current Status

As of the current update, the official D0 training run has completed:

```text
run_name: D0_weighted_ce_25ep
run_dir: logs/milestone_d/runs/D0_weighted_ce_25ep/
epochs: 25
wall_clock: 14,458.072 seconds (~4.02 hours)
best marking IoU epoch: 18
status: completed and pulled locally
```

Completed artifact sanity checks:

```text
eval_history.csv lines: 26 (header + 25 epochs)
checkpoints: 25
eval JSONs: 25
confusion matrices: 25
```

Analysis status:

```text
dataset remap intensity analysis: generated
D0 run-level plots/report: generated
D0 sampled epoch-18 error analysis: implemented and smoke-tested, full 2160-step GPU pass pending
```

## Why Milestone D Exists

C0 proved that the strict lane-line baseline is trainable, but it also exposed
a label-definition problem and an input-signal limitation.

C0 used this strict label definition:

```text
lane3:
  road  = PandaSet raw 7
  lane  = PandaSet raw 8
  other = all remaining non-ignored raw classes
```

The selected C0 checkpoint is epoch 18:

```text
run: logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/
checkpoint: checkpoints/ckpt_epoch_00018.pth
lane_iou: 0.310355
lane_precision: 0.437552
lane_recall: 0.516349
lane_f1: 0.473696
mIoU: 0.703805
```

C0's main failure mode was true lane predicted as road. The diagnostic analysis
showed that many missed lane points were intensity-wise similar to correctly
predicted road:

```text
lane predicted lane p25/p50/p75: 31 / 44 / 62
lane predicted road p25/p50/p75: 26 / 29 / 31
road predicted road p25/p50/p75: 27 / 29 / 33
```

That means the strict C0 problem is partly information-limited for LiDAR-only
input: some lane-line points do not look separable from road by intensity.

Separately, qualitative inspection found that PandaSet raw `9` and raw `10`
road-marking classes often look like lane-related road paint. Under C0 they were
mapped to `other`, so some visually reasonable model predictions were counted
as errors because the thesis label definition was narrower than the visual road
marking concept.

Milestone D changes the positive class from strict lane-line marking to broader
road-marking segmentation.

## Label Definition

C0 remains unchanged and reproducible. The default dataset behavior is still
`label_mode: lane3`.

Milestone D uses an opt-in label mode:

```text
road_marking3:
  road    = PandaSet raw 7
  marking = PandaSet raw 8 + raw 9 + raw 10
  other   = all remaining non-ignored raw classes
```

Ignored labels remain unchanged:

```text
PandaSet raw 1, 2, 3, 4 -> ignore label 0
```

The numeric label IDs remain unchanged:

```text
0 = ignore
1 = road
2 = positive class
3 = other
```

Important naming caveat: for `road_marking3`, numeric label `2` means
`marking`, but existing C-stage metric columns are still named `lane_iou`,
`lane_f1`, `lane_precision`, and `lane_recall`. In Milestone D, read those as:

```text
lane_iou       = marking_iou
lane_f1        = marking_f1
lane_precision = marking_precision
lane_recall    = marking_recall
```

This aliasing avoids breaking the existing metric and plotting pipeline before a
separate naming cleanup.

## Implemented Code Changes

Milestone D added label-mode support without changing C0 behavior:

```text
src/thesis_pipeline/adapters/pandaset_ff_lane3.py
src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py
```

Implemented label modes:

```text
LABEL_MODE_LANE3 = "lane3"
LABEL_MODE_ROAD_MARKING3 = "road_marking3"
```

New raw IDs used by the D remap:

```text
RAW_STOP_LINE_ID = 9
RAW_OTHER_ROAD_MARKING_ID = 10
RAW_ROAD_MARKING_IDS = {8, 9, 10}
```

The dataset constructor now accepts:

```python
label_mode: str = "lane3"
```

The default is still `lane3`, so C0 configs remain backward-compatible.

Milestone D also added:

```text
tools/compute_milestone_d_training_statistics.py
tools/train_milestone_d.py
logs/milestone_d/configs/d0_weighted_ce.yml
logs/milestone_d/road_marking3_training_statistics.json
logs/milestone_d/reports/road_marking3_training_statistics.md
```

`tools/train_milestone_d.py` is separate from `tools/train_milestone_c.py` so
C0 stays reproducible. The D runner adds run-local checkpoints, explicit resume,
AdamW support, ReduceLROnPlateau support, LR logging, and scheduler metric
smoothing.

## Training Statistics And Class Weights

Milestone D statistics were recomputed on the frozen training split with
`label_mode: road_marking3`.

Scope:

```text
split: training only
training sequences: 58
frames processed: 4640
frames failed: 0
forward sensor: PandaSet forward-facing LiDAR sensor 1
```

Class counts:

```text
ignore:     398,730
road:   119,562,394
marking:  5,129,328
other:  173,473,484
```

Active shares:

```text
road:    40.099378%
marking:  1.720297%
other:   58.180324%
```

Raw marking composition:

```text
raw 8 lane-line marking:       2,098,182
raw 9 stop-line marking:         162,729
raw 10 other road marking:     2,868,417
raw 8+9+10 total marking:      5,129,328
```

D0 uses Open3D-native count-style `class_weights`:

```text
[119562394.0, 5129328.0, 173473484.0]
```

Reason: local Open3D-ML transforms `dataset.cfg.class_weights` internally via
`DataProcessing.get_class_weights` before constructing `CrossEntropyLoss`. The
count-like list is the runtime-safe format for this codebase.

The effective CE weights produced by Open3D are:

```text
road:    2.3753321170806885
marking: 26.87957191467285
other:   1.6616727113723755
```

Do not replace the config list with these effective weights unless the loss
construction path is deliberately changed.

## Intensity Statistics

The Milestone D statistics script also recomputed intensity normalization values
on the training split:

```text
intensity_clip_low: 0.0
intensity_clip_high: 114.0
intensity_mean: 22.471752166748047
intensity_std: 14.902379989624023
```

The clip bounds are percentile-derived by the statistics script:

```text
clip_low_p0p5 = 0.0
clip_high_p99p5 = 114.0
```

These values are close to the C0 statistics but are recorded in the D stats file
so D0 has its own self-contained evidence.

## Raw Road-Marking Evidence

The raw road-marking analysis is:

```text
logs/raw_road_marking_analysis/
```

It compared raw classes `7`, `8`, `9`, and `10` over the frozen split.

Global medians:

```text
raw 7 road:                28
raw 8 lane-line marking:   33
raw 9 stop-line marking:   33
raw 10 other road marking: 33
```

This supports the D remap direction: raw `8`, `9`, and `10` are all road-paint
classes with similar intensity profiles. It also warns that road/marking
separation is still not trivial because the marking distributions overlap with
road, especially for low-intensity or distant markings.

Qualitative observations:

- raw `10` is mixed: some examples are long, thin, and lane-like; others are
  zebra-like, transverse, symbolic, or otherwise non-lane road paint.
- raw `9` is usually stop-line geometry: bright road paint but often transverse,
  so it is not the same shape as lane-line markings.
- merging raw `9` and `10` into the positive class changes the task from strict
  lane-line segmentation to road-marking segmentation.

This is a thesis-question shift. It is defensible, but it must be stated clearly.

## D0 Experiment Question

D0 asks:

```text
Can a LiDAR-only RandLA-Net baseline segment road markings
(raw 8 + raw 9 + raw 10) from road and other classes better and more
consistently than the strict lane-line C0 task?
```

D0 is not a direct label-identical comparison to C0 because the positive class
changed. Compare D0 to C0 as a change of task definition and training setup, not
as a strict same-label improvement.

The correct interpretation is:

- C0 answers strict lane-line segmentation.
- D0 answers expanded road-marking segmentation.
- Milestone E should test whether adding camera-derived color features improves
  this road-marking task.

## Analysis Structure

All Milestone D analysis code and outputs are contained under
`logs/milestone_d/`.

Dataset/remap analyses that are not tied to one model run live under:

```text
logs/milestone_d/dataset_analysis/
```

Per-run analyses live under:

```text
logs/milestone_d/run_analysis/<run_name>/
```

For D0:

```text
logs/milestone_d/dataset_analysis/road_marking3_intensity/
logs/milestone_d/run_analysis/D0_weighted_ce_25ep/
```

The analysis code itself is stored next to the outputs:

```text
logs/milestone_d/dataset_analysis/road_marking3_intensity/analysis_code/
logs/milestone_d/run_analysis/D0_weighted_ce_25ep/analysis_code/
```

This containment is deliberate. These scripts and outputs are thesis evidence
for Milestone D, not generic global tooling.

## D0 Final Config

Current config:

```text
logs/milestone_d/configs/d0_weighted_ce.yml
```

Final intended values:

```yaml
dataset:
  label_mode: road_marking3
  use_cache: true
  cache_dir: logs/milestone_d/cache/D0_weighted_ce
  class_weights: [119562394.0, 5129328.0, 173473484.0]
  intensity_clip_low: 0.0
  intensity_clip_high: 114.0
  intensity_mean: 22.471752166748047
  intensity_std: 14.902379989624023
  sampler: SemSegRandomSampler
  steps_per_epoch_train: 4640
  steps_per_epoch_valid: 720

model:
  name: RandLANet
  num_points: 32768
  num_neighbors: 24
  num_layers: 3
  num_classes: 3
  sub_sampling_ratio: [4, 4, 4]
  in_channels: 4
  dim_features: 8
  dim_output: [16, 64, 128]
  grid_size: 0.04
  augment:
    recenter: {dim: [0, 1]}
    scale: {scale_anisotropic: false, min_s: 0.95, max_s: 1.05}
    noise: {noise_std: 0.01}

pipeline:
  loss: weighted CE through Open3D SemSegLoss
  optimizer: AdamW
  lr: 0.0014
  weight_decay: 0.0001
  scheduler: ReduceLROnPlateau
  watch_metric: marking_iou
  smoothing_window: 3
  batch_size: 2
  val_batch_size: 2
  num_workers: 0
  pin_memory: true
  max_epoch: 24
  save_ckpt_freq: 1
```

`max_epoch: 24` means 25 actual epochs because the runner iterates epoch numbers
`0..24`.

## D0 Config Decisions

### Weighted CE Instead Of Focal Loss

Decision: keep weighted CE for D0.

Why:

- C0 diagnostics showed many lane-to-road errors are intensity-overlap cases:
  missed lane points often look like road.
- Focal loss would emphasize hard ambiguous points, but if those cases are
  information-limited rather than learnable, focal loss can increase marking
  overprediction without solving the underlying signal problem.
- C0 epoch 18 to epoch 30 already showed a precision-collapse pattern: recall
  improved while predicted-lane share rose too much.
- D0 should be a stable, interpretable road-marking baseline before adding RGB in
  Milestone E.

Pros:

- stable and comparable with C0's loss family
- less risk of marking overprediction
- fewer code patches before a long run

Cons:

- may leave some hard-case performance on the table
- does not explicitly focus gradient on difficult boundary cases

Focal loss is deferred to a later ablation, especially after RGB features change
which errors are learnable.

### AdamW

Decision: use AdamW with `weight_decay: 1e-4`.

Why:

- C0 showed overfitting after about epoch 15-20.
- AdamW decouples weight decay from Adam's adaptive gradient scaling.
- `1e-4` is a conservative regularization value.

Pros:

- cleaner regularization than coupled Adam weight decay
- low implementation risk in the D runner

Cons:

- not tuned
- stronger values such as `5e-4` could regularize more but risk starving rare
  marking learning

### ReduceLROnPlateau

Decision: use ReduceLROnPlateau on smoothed marking IoU.

Final scheduler:

```text
mode: max
factor: 0.5
patience: 6
threshold: 0.01
threshold_mode: abs
cooldown: 1
min_lr: 1e-6
watch_metric: marking_iou
smoothing_window: 3
```

Why:

- C0 validation loss and positive-class IoU did not peak at exactly the same
  epoch.
- The scheduler should serve the marking objective, not only total validation
  loss.
- Raw positive-class IoU is noisy, so the runner uses a 3-epoch moving average
  for scheduling.
- Best checkpoint selection remains separate and uses raw marking IoU after the
  run.

Pros:

- objective-aligned
- avoids reacting to single-epoch noise
- less arbitrary than fixed exponential decay

Cons:

- more moving parts than ExponentialLR
- scheduler behavior depends on validation metric noise

Cold-start behavior:

- epoch 1 scheduler value = raw epoch-1 marking IoU
- epoch 2 scheduler value = mean of epochs 1-2
- epoch 3+ scheduler value = mean of last 3 epochs

### Number Of Points

Decision: use `num_points: 32768`.

Why:

- D marking remains rare at about `1.72%` of active points.
- More sampled points gives more marking points per forward pass.
- New server has a full A100 80GB PCIe, so the old wall-clock constraint from the
  40GB MIG slice is relaxed.

Important caveat:

- `RandomDropout` was removed because it changes tensor lengths after sampling.
  With `batch_size=2`, that breaks the default batcher.
- Open3D's random sampler can pad short clouds by sampling with replacement, so
  `32768` itself is valid. The failed smoke was caused by dropout, not by short
  raw frames.

### Batch Size

Decision: use `batch_size: 2` and `val_batch_size: 2`.

Why:

- It reduces steps per epoch compared with batch size 1.
- It improves throughput on the full A100 while keeping the run relatively
  conservative.
- Learning rate uses sqrt scaling from C0: `0.001 * sqrt(2) ~= 0.0014`.

Pros:

- faster than batch size 1
- smoother gradients

Cons:

- C0 batch-size-2 medium run showed lane overprediction, so this is watched
  carefully.
- Larger batch sizes such as 4 or 8 are not used because they would reopen
  learning-rate and overprediction risk.

### DataLoader Workers

Decision: `num_workers: 0`.

Why:

- Worker subprocesses segfaulted in C0 tests.
- A fresh D0 server smoke with `num_workers=2` also segfaulted.
- The likely cause is native-library or devkit state after fork, not a normal
  Python exception.

Pros:

- stable
- avoids native multiprocessing failures

Cons:

- slower data loading

Future optimization:

- precompute frames into simple `.npz` files or make dataset objects worker-local
  after fork.

### Pin Memory

Decision: `pin_memory: true` for D0.

Why:

- D0 server smoke with pin memory off completed in `31.468s`.
- D0 server smoke with pin memory on completed in `28.142s`.
- Both were stable.

Pros:

- small speed improvement on the new full A100 server

Cons:

- only smoke-level evidence, not a full-epoch benchmark
- C0 on the old setup found pinning slower, so this is server/config-specific

### Augmentation

Final D0 augmentations:

```text
recenter x/y
scale 0.95-1.05
noise std 0.01m
```

Removed:

```text
RandomDropout 0.15
```

Why recenter:

- inherited from C0
- keeps random sampled patches in a stable local coordinate convention

Why scale:

- mild geometric scale regularization
- low risk because it is only `0.95-1.05`

Why noise:

- mild 1cm positional perturbation
- helps regularize without changing semantic labels

Why no rotation:

- Open3D's available rotation path is full yaw rotation, which is too aggressive
  for ego-frame driving scenes.

Why no RandomDropout:

- it reduces point count after sampling
- `DefaultBatcher` requires equal tensor shapes in a batch
- it caused smoke failures at both `32768` and `24576`

### Network Shape

Decision: keep C0's 3-layer network shape:

```text
num_layers: 3
sub_sampling_ratio: [4, 4, 4]
dim_features: 8
dim_output: [16, 64, 128]
grid_size: 0.04
```

Why:

- C0 did not show obvious underfitting; it learned and later overfit.
- Keeping network capacity close to C0 makes D0 easier to interpret.
- Adding a 4th layer would add context but also more compression of rare marking
  points.

At `num_points=32768`, the point counts through the 3 RandLA layers are roughly:

```text
32768 -> 8192 -> 2048 -> 512
```

This is a reasonable compromise for thin road markings.

### Number Of Neighbors

Decision: use `num_neighbors: 24`.

Why:

- C0 used `16`.
- Road markings need local contrast with surrounding road.
- `24` increases local context without jumping to the heavier and potentially
  oversmoothing `32`.

Status:

- reasoned but not proven
- D0 will provide evidence

## Cache Policy

D0 uses cache:

```text
use_cache: true
cache_dir: logs/milestone_d/cache/D0_weighted_ce
```

Before the first real D0 launch, the cache must be deleted:

```bash
rm -rf logs/milestone_d/cache/D0_weighted_ce
```

Do not delete the cache before resuming a compatible D0 run. A resume run should
use the same cache.

Delete the cache only if preprocessing-relevant settings change:

- label mode
- dataset path or split
- intensity stats
- grid size
- model transform behavior
- augmentations that affect cached preprocessing
- cache corruption

## Historical Run Commands

D0 was ultimately run fresh for all 25 epochs after smoke tests and cleanup:

```bash
rm -rf logs/milestone_d/cache/D0_weighted_ce
rm -rf logs/milestone_d/runs/D0_weighted_ce_25ep
mkdir -p logs/milestone_d/launch_logs

nohup python tools/train_milestone_d.py \
  --config logs/milestone_d/configs/d0_weighted_ce.yml \
  --run-name D0_weighted_ce_25ep \
  --epochs 25 \
  --save-ckpt-freq 1 \
  --force \
  > logs/milestone_d/launch_logs/D0_weighted_ce_25ep_fresh_25.stdout.log 2>&1 &
```

Monitor:

```bash
pgrep -af train_milestone_d
tail -f logs/milestone_d/launch_logs/D0_weighted_ce_25ep_fresh_25.stdout.log
```

Stop watching without stopping training:

```text
Ctrl+C
```

Check artifacts:

```bash
RUN=logs/milestone_d/runs/D0_weighted_ce_25ep

wc -l "$RUN/eval_history.csv"
tail -n 40 "$RUN/training_log.txt"
tail -n 10 "$RUN/eval_history.csv"
ls -1 "$RUN/checkpoints"
```

Resume support exists and was patched, but the official D0 run was completed as
a clean fresh 25-epoch run.

Resume command format for future compatible runs:

```bash
python tools/train_milestone_d.py \
  --config logs/milestone_d/configs/d0_weighted_ce.yml \
  --run-name D0_weighted_ce_25ep \
  --epochs 25 \
  --save-ckpt-freq 1 \
  --resume-latest
```

Do not use `--force` when resuming. Do not delete the cache when resuming.

## Analysis Commands

Dataset-level road_marking3 intensity analysis:

```bash
./panda/bin/python logs/milestone_d/dataset_analysis/road_marking3_intensity/analysis_code/analyze_road_marking3_intensity.py
```

D0 official run-level plots/report:

```bash
./panda/bin/python logs/milestone_d/run_analysis/D0_weighted_ce_25ep/analysis_code/plot_d0_run.py
```

D0 sampled epoch-18 error analysis, run on the A100 server:

```bash
python logs/milestone_d/run_analysis/D0_weighted_ce_25ep/analysis_code/analyze_d0_marking_errors.py \
  --steps 2160 \
  --device cuda

python logs/milestone_d/run_analysis/D0_weighted_ce_25ep/analysis_code/plot_d0_marking_error_analysis.py
```

Important: sampled error analysis is a new sampled validation inference pass. It
is not the exact validation sample used during the original epoch-18 training
evaluation.

## What Counted As Healthy After 5 Epochs

Check these before resuming:

- process completed all 5 epochs
- checkpoints exist through `ckpt_epoch_00005.pth`
- `eval_history.csv` has 5 metric rows plus header
- `training_log.txt` contains `scheduler_step` lines
- `lr` is logged in `eval_history.csv`
- train loss is finite and generally decreasing
- validation loss is finite
- marking precision and recall are not collapsed
- predicted marking share is not exploding far above true marking share
- wall-clock per epoch is acceptable

Do not expect high final-quality metrics after only 5 epochs. The purpose is
stability and sanity, not final selection.

The 5-epoch segment was healthy enough to continue, and the final official run
was later completed as a fresh 25-epoch run.

## Checkpoint Policy

D0 uses:

```text
save_ckpt_freq: 1
```

The official run retained:

```text
ckpt_epoch_00001.pth through ckpt_epoch_00025.pth
```

Best checkpoint selection is done after the run by maximum raw marking IoU
(`lane_iou` column), not by validation loss and not by the smoothed scheduler
metric.

For D0, that selected:

```text
best checkpoint: checkpoints/ckpt_epoch_00018.pth
marking IoU: 0.440294
marking F1: 0.611394
marking precision: 0.514788
marking recall: 0.752636
mIoU: 0.780829
```

Final epoch 25 had lower marking IoU but higher marking recall:

```text
marking IoU: 0.410337
marking F1: 0.581900
marking precision: 0.451201
marking recall: 0.819195
mIoU: 0.771793
```

Initial interpretation: D0 continues to learn recall late, but it becomes more
liberal about predicting marking, so precision drops and positive-class IoU
peaks earlier than the final epoch.

## Known Caveats

1. D0 is not a same-label comparison with C0. The positive class changed.
2. Metric columns still use `lane_*` names even though they mean marking in D.
3. D0 is LiDAR-only. It does not add RGB/color features yet.
4. Road/marking intensity overlap remains real; remapping does not magically
   solve LiDAR ambiguity.
5. `num_workers > 0` is unsafe in the current pipeline.
6. `RandomDropout` cannot be used with `batch_size=2` and the current default
   batcher because it changes tensor lengths.
7. The first cached run spends substantial time in preprocessing before training
   starts.
8. D0 sampled error analysis must not reuse the D0 training cache because the
   analysis carries raw subtype labels that were not present in training cache
   entries.

## Milestone E Direction

Milestone E is the likely RGB/color-feature extension:

```text
project LiDAR points into camera images
sample RGB values for valid projections
append color features to xyz + intensity
train/evaluate against D's road_marking3 task
```

Important E design points already identified:

- handle invalid projections explicitly, likely with an `rgb_valid` feature
- handle occlusion carefully
- do not silently fill invalid RGB with arbitrary colors without telling the
  model validity
- recompute feature normalization statistics after adding RGB
- keep D0 as the LiDAR-only baseline for the road-marking task

Focal loss can be reconsidered later, but it is intentionally not part of D0.

## Remaining Analysis Work

Completed:

- D0 full 25-epoch training
- D0 artifact count checks
- road_marking3 dataset intensity analysis
- D0 run-level plots and run summary

Pending:

- run the full 2160-step D0 sampled epoch-18 error analysis on the GPU server
- generate sampled diagnostic plots from that output
- update `logs/milestone_d/run_analysis/D0_weighted_ce_25ep/reports/d0_weighted_ce_results.md`
  with sampled failure-mode findings
- decide whether D0 is strong enough as the LiDAR-only road-marking baseline or
  whether another LiDAR-only D run is needed before Milestone E

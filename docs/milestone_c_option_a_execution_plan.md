# Milestone C Option A Execution Plan

Conservative end-to-end plan for moving from the current Milestone C server-readiness state to a defensible thesis result.

This plan follows **Option A**: stabilize a real RandLA-Net baseline first, then add low-risk dataset/input improvements that are directly motivated by the raw intensity analysis. It avoids architecture surgery, custom multi-head models, Lovasz loss, PCA geometry features, or other reach items until the simple path has produced a trustworthy baseline and ablation table.

---

## 1. Current Starting Point

These are the facts we should treat as the starting line.

- The project uses PandaSet forward-facing LiDAR and Open3D-ML RandLA-Net.
- The model is imported from Open3D-ML, not vendored or customized.
- The current active classes are `road`, `lane`, and `other`; `ignore=0` is excluded from the loss.
- The raw PandaSet label remap is `{1,2,3,4}->ignore`, `7->road`, `8->lane`, everything else -> `other`.
- Raw stop-line marking `9` and other road marking `10` are not separate model classes; they are inside `other`.
- The current input is `xyz_ego + standardized_intensity`, so `in_channels: 4`.
- The frozen split is sequence-level: train/val/test sequence lists live in `configs/splits/`.
- Milestone B sanity training passed mechanically, but it is not a performance baseline.
- The sanity run had finite/changing loss, but no IoU, no per-class metrics, and no confusion matrix.
- The intended config has `num_points: 16384`, but the successful sanity run used `4096` as an OOM concession.
- As of the current plan update, strong rented GPU training on DigitalOcean GPU Droplets is available, so local laptop VRAM should no longer define the final experiment shape.
- Lane points are rare: about `0.70%` of training points.
- Lane-aware sampling is not implemented yet.
- Raw intensity analysis is complete under `logs/raw_intensity_analysis/`.
- Server execution is now verified on `/home/coder/project` with Conda env `panda312`, PyTorch CUDA, Open3D, the patched PandaSet devkit, and the downloaded Kaggle `pz19930809/pandaset` dataset.
- The confirmed server GPU is `NVIDIA A100 80GB PCIe MIG 3g.40gb`.
- The server dataset root is `/home/coder/project/pandaset/PandaSet`.
- C0 sampler policy is now `SemSegRandomSampler`, not `SemSegSpatiallyRegularSampler`.
- `SemSegSpatiallyRegularSampler` was benchmarked and deferred because it eagerly preprocesses all split frames before epoch 1. The measured estimate was about `1.30h` for training sampler initialization plus about `12min` for validation initialization.
- Class-weighted CE was verified live on the server. Effective CE weights are `road=2.3753`, `lane=36.9864`, `other=1.6341`.
- Random-sampler GPU smoke tests completed for both tiny and full-model configs, including validation and checkpoint writing.
- C0 medium-run evidence now exists. A 10-epoch batch-size-1 medium run learned successfully and produced the best lane balance observed so far.
- Batch-size-2 was tested with a matching 10-epoch medium run. It completed and learned, but was slower in the realistic run and over-predicted lane by epoch 10.
- Runtime benchmarks showed `num_workers > 0` is unstable on the current server because DataLoader workers segfault. `pin_memory=true` was slower in the zero-worker setting.
- Official C0 full-run settings should be `batch_size=1`, `val_batch_size=1`, `num_workers=0`, `pin_memory=false`, `steps_per_epoch_train=4640`, and `steps_per_epoch_valid=720`.

Primary references:

- `docs/thesis_pipeline_context_report.md`
- `configs/randlanet_pandaset_ff_lane3.yml`
- `logs/milestone_b_training_statistics.json`
- `logs/milestone_b_sanity_train_report.txt`
- `logs/milestone_b_stop_conditions.txt`
- `logs/raw_intensity_analysis/reports/final/intensity_notes.md`
- `logs/raw_intensity_analysis/reports/final/plots.md`
- `logs/milestone_c/reports/c0_sampler_and_server_readiness.md`
- `logs/milestone_c/reports/c0_medium_runs_and_speed_benchmarks.md`

---

## 2. Main Thesis Strategy

The thesis story should be:

> I first built a correct PandaSet-to-RandLA-Net pipeline. Then I analyzed raw LiDAR intensity and found that lane markings are somewhat separable from asphalt globally, much more separable locally, and much less separable from other painted road markings. Based on that evidence, I improved the baseline using dataset-level changes: engineered intensity/context features and lane-aware sampling. I evaluated each change with controlled ablations.

This is clean because every modeling choice has a reason:

- **Keep intensity** because lane vs road has real signal.
- **Add local contrast** because Stage 3 showed lane points are often brighter than nearby road, even when global intensity overlaps.
- **Add range correction** because Stage 4 showed intensity signal weakens strongly with distance.
- **Add height-above-local-ground** because lane paint is on the road surface and this helps reject bright elevated objects.
- **Add lane-aware sampling** because the lane class is only about `0.70%` of training data.
- **Keep RandLA-Net architecture unchanged** because Open3D-ML makes dataset/input changes much cheaper and safer than architectural changes.

---

## 3. Foundational Reference

The raw intensity analysis should be understood as two separate problems.

**Problem A: lane vs road/asphalt.** This is meaningfully helped by intensity, especially local intensity. Globally, lane intensity is shifted above road intensity, but the stronger finding is local: Stage 3 showed that at `0.5m` radius, about `81.6%` of lane points are brighter than their local non-lane neighborhood mean, with average delta about `+12.46` raw intensity units. The model should therefore learn “bright relative to nearby surroundings,” not only “bright in absolute terms.”

**Problem B: lane vs other painted road markings.** This is not reliably solved by intensity alone. Raw lane marking, stop-line marking, and other road marking medians are all around `33`, so paint-vs-paint confusion is expected. The three-class remap handles this cleanly by keeping the thesis target as `road / lane / other`; raw stop-line and other road marking become `other`, not separate lane-like classes.

The remaining risk from Problem B is not that the model must identify every road-marking subtype. The real risk is that stop-line or other marking points are predicted as `lane`, which hurts **lane precision**. That is why the confusion matrix and lane precision are not optional metrics.

RandLA-Net can use extra feature channels without architecture changes. Its local feature aggregation uses each point’s coordinates, neighboring coordinates, relative offsets, distances, and feature vector inside local neighborhoods. So if we add a valid `local_contrast` feature, RandLA-Net can propagate and aggregate that context through its normal neighborhood machinery.

---

## 4. Non-Negotiable Rules

Follow these throughout Milestone C.

- Do not use the test split for tuning. Use validation for choices; use test only for final reporting.
- Do not compute model input features using semantic labels. That would leak the answer into training.
- Raw labels may be used for analysis, metrics, and sampling diagnostics, but not as input features.
- Every real run must save its config snapshot, git commit hash, random seed, metrics, checkpoint path, and peak GPU memory.
- Every ablation should change one major thing at a time.
- Keep the Open3D-ML RandLA-Net architecture frozen for Option A.
- Treat `num_points: 16384` as the default final experiment shape now that DigitalOcean GPU Droplets are available.
- If `num_points: 16384` still cannot run on a suitably sized DigitalOcean GPU Droplet, document the chosen concession explicitly instead of hiding it.
- Validation and test sampling must remain evaluation-realistic, not lane-centered.
- C0 must use `SemSegRandomSampler` unless a later report explicitly supersedes the sampler decision. The spatial sampler is deferred because it is class-blind and imposes a large eager startup cost.
- Do not use one-step smoke configs for real C0. Inspect `steps_per_epoch_train` and `steps_per_epoch_valid` before any long run.
- On the current server, official C0 must use `num_workers: 0`; worker subprocesses have repeatedly segfaulted.
- On the current server, official C0 should use `pin_memory: false`; pinned memory slowed zero-worker benchmarks.
- Official C0 should use `batch_size: 1` unless a later report supersedes this. Batch size 2 was slower in the realistic medium run and produced worse lane precision/F1 by over-predicting lane.

The label-leakage point is especially important. In the raw analysis, local contrast was allowed to compare lane points against raw road-surface classes because we were studying the dataset. In model training, the model will not know the true class labels at inference time, so features must be computed from geometry and intensity only.

---

## 5. Phase 0: Clean and Prepare the Experiment Ground

Goal: remove small issues that could make later runs confusing.

### 5.0 DigitalOcean GPU Droplet strategy

Use the local machine for code edits, short smoke tests, report writing, and artifact inspection. Use DigitalOcean GPU Droplets for long training, full validation, and ablations.

DigitalOcean changes the plan in three important ways:

- `num_points: 16384` should be the expected baseline target, not an aspirational laptop setting.
- Avoid spending thesis time on heavy memory workarounds like gradient checkpointing unless a large GPU Droplet also fails.
- Full ablations become more realistic, so C0-C4 should be treated as the target set rather than an optional luxury.

DigitalOcean operational facts checked from official docs:

- GPU Droplets are normal Linux VMs with GPU hardware, so SSH-based workflow and normal repo/environment setup apply.
- DigitalOcean recommends using its AI/ML-ready GPU images because NVIDIA/AMD drivers and GPU software are preinstalled.
- For NVIDIA GPU Droplets, the AI/ML-ready image is Ubuntu-based and includes CUDA/NVIDIA tooling; DigitalOcean documents image slugs such as `gpu-h100x1-base` for single-GPU NVIDIA Droplets and `gpu-h100x8-base` for 8-GPU NVIDIA Droplets.
- Current self-serve NVIDIA options include `gpu-h100x1-80gb`, `gpu-h200x1-141gb`, `gpu-l40sx1-48gb`, `gpu-6000adax1-48gb`, and `gpu-4000adax1-20gb`.
- H100/H200 class Droplets include a persistent boot disk plus a large non-persistent scratch disk; L40S/RTX 6000/RTX 4000 list no scratch disk in the current plan table.
- Scratch disks are useful for temporary training/cache staging, but they are not persistent and are not included in snapshots.
- Powered-off Droplets still bill because the compute resource remains reserved; destroy the Droplet when done with a training block, after syncing artifacts.
- DigitalOcean Volumes are persistent network-attached block storage; use one if the dataset/logs should survive Droplet destruction or move between Droplets.

Official references:

- GPU Droplet hardware/features: `https://docs.digitalocean.com/products/droplets/details/features/`
- Recommended GPU images/software: `https://docs.digitalocean.com/products/droplets/getting-started/recommended-gpu-setup/`
- GPU scratch disk behavior: `https://docs.digitalocean.com/products/droplets/how-to/gpu/use-scratch-disk/`
- Droplet pricing/billing behavior: `https://docs.digitalocean.com/products/droplets/details/pricing/`
- Volumes: `https://docs.digitalocean.com/products/volumes/how-to/create/`
- Snapshots: `https://docs.digitalocean.com/products/snapshots/how-to/snapshot-droplets/`

Recommended DigitalOcean setup checklist:

- Prefer a single NVIDIA H100 or H200 GPU Droplet for final C0-C4 training if budget allows.
- Use L40S or RTX 6000 only if cost matters and a `16384` smoke test proves it is stable.
- Current official pricing should be checked before launching; at the time this plan was updated, H100/H200 were about `$3.39-$3.44/hour`, while L40S/RTX 6000 were about `$1.57/hour`.
- Create the Droplet with an AI/ML-ready NVIDIA image.
- SSH in and verify `nvidia-smi` before installing project dependencies.
- Create a clean Python environment matching the local `panda` environment as closely as possible.
- Install the same Open3D/Open3D-ML/PyTorch/PandaSet dependencies.
- Sync the repo, `pandaset/`, `pandaset-devkit/`, config files, and frozen split files.
- Do not rely on local absolute paths if they differ on the Droplet; use config overrides or a DigitalOcean-specific config snapshot.
- Put persistent source code, final checkpoints, metrics, and reports on the boot disk or a mounted DigitalOcean Volume.
- Use the scratch disk, if available, only for temporary caches, copied datasets, and active training staging.
- Copy final logs/checkpoints/metrics off the Droplet before destroying it.
- After the environment is stable, optionally snapshot the boot disk so future GPU Droplets can start from the same setup; remember that scratch disk contents are not captured.
- Copy final logs/checkpoints/metrics back to the local machine after each major run.

Definition of done:

- A DigitalOcean smoke test can import the dataset, load one train sample, construct RandLA-Net, run `nvidia-smi`, and start one training step.
- DigitalOcean output artifacts are saved in the same structure as local artifacts.

### 5.1 Create Milestone C output structure

Use a clean directory layout:

```text
logs/milestone_c/
  runs/
  configs/
  eval/
  checkpoints/
  feature_cache/
  reports/
  figures/
```

Each run should get its own folder:

```text
logs/milestone_c/runs/C0_baseline/
logs/milestone_c/runs/C1_features/
logs/milestone_c/runs/C2_sampling/
logs/milestone_c/runs/C3_features_sampling/
logs/milestone_c/runs/C4_final_augmented/
```

Definition of done:

- `logs/milestone_c/` exists.
- There is a naming convention for every run before long training starts.

### 5.2 Decide how to handle old sanity checkpoints

The old sanity checkpoints are not final experiment checkpoints. They should not accidentally resume into Milestone C.

Recommended action:

- Archive the Milestone B sanity checkpoints or make the Milestone C entrypoint force fresh training.
- Ensure the Milestone C training script sets resume behavior explicitly.

Definition of done:

- A fresh Milestone C run cannot silently resume from Milestone B checkpoints.

### 5.3 Clean config/report issues

Check and clean:

- YAML duplicate keys, especially `num_workers`.
- The dual-writer issue in `tools/sanity_train_check.py`, where shell redirection and script writing can both target the same report.
- `real_training_allowed` sentinel behavior, so it is clear whether Milestone C is intentionally running.

Definition of done:

- The live YAML is easy to read and has no duplicate keys.
- Sanity/report scripts have one clear output path.
- The chosen Milestone C entrypoint is separate from the Day 6 sanity script or clearly marked as real training.

---

## 6. Phase 1: Build a Real Baseline

Goal: get the first real number to beat.

This is the most important phase. Right now the project has proof that training can start, but not proof that the model segments lanes.

### 6.1 Create a Milestone C training/evaluation entrypoint

Create a real training script, for example:

```text
tools/train_milestone_c.py
```

It should:

- Load `configs/randlanet_pandaset_ff_lane3.yml`.
- Instantiate the PandaSet dataset and Open3D-ML RandLA-Net pipeline.
- Force explicit resume behavior.
- Set seeds for Python, NumPy, and PyTorch.
- Save a run folder with config snapshot and git commit hash.
- Train for a configured number of epochs.
- Run validation and save metrics.

The command shape should eventually look like:

```bash
./panda/bin/python -u tools/train_milestone_c.py \
  --config configs/randlanet_pandaset_ff_lane3.yml \
  --run-name C0_baseline \
  --epochs 30 \
  --no-resume
```

Definition of done:

- A short 1-epoch run starts and exits cleanly.
- The run folder contains config snapshot, seed, and log file.

Current status:

- Done on server for random-sampler smoke runs.
- Done for `C0_baseline_medium_10ep_random` and `C0_baseline_medium_10ep_random_bs2`.
- The official full C0 run is still pending.

### 6.2 Add full validation metrics

The baseline must report more than loss.

Save these metrics every epoch:

- Training loss.
- Validation loss.
- mIoU.
- Per-class IoU for `road`, `lane`, `other`.
- Per-class precision, recall, and F1.
- Confusion matrix, active classes only: `road/lane/other`.
- Lane recall by distance bucket if feasible.
- Peak GPU memory.
- Wall-clock epoch time.

Minimum files:

```text
logs/milestone_c/runs/C0_baseline/eval_history.csv
logs/milestone_c/runs/C0_baseline/eval_epoch_001.json
logs/milestone_c/runs/C0_baseline/confusion_epoch_001.npy
```

After the run, generate human-readable plots and a compact report:

```bash
./panda/bin/python tools/plot_milestone_c_run.py --run-name C0_baseline
```

Expected plot/report outputs:

- `plots/metrics_overview.png`
- `plots/loss_curves.png`
- `plots/per_class_iou.png`
- `plots/lane_precision_recall_f1.png`
- `plots/lane_recall_by_distance.png`
- `plots/runtime_and_memory.png`
- `plots/class_true_vs_predicted_share.png`
- `plots/confusion_epoch_<NNN>.png`
- `plots/confusion_best_lane_f1_epoch_<NNN>.png`
- `plots/run_summary.md`

Why this matters:

- `mIoU` tells overall segmentation quality.
- `lane IoU` tells whether the thesis target is working.
- `lane precision` tells whether stop lines/other bright markings are being predicted as lane.

Current status:

- Implemented in `tools/train_milestone_c.py`.
- Verified through 10-epoch medium runs.
- Batch-size-1 medium run final values:
  ```text
  train_loss 0.339711
  val_loss   0.408176
  mIoU       0.558003
  lane_iou   0.160835
  lane_f1    0.277103
  ```
- Batch-size-1 best lane epoch was epoch 7:
  ```text
  lane_iou 0.192446
  lane_precision 0.250676
  lane_recall 0.453093
  lane_f1 0.322776
  ```
- `lane recall` tells whether the model is missing lanes.
- The confusion matrix tells which failure mode dominates.

Definition of done:

- One short run produces all metrics above.
- The confusion matrix can be inspected and matches the three active classes.

### 6.3 Revalidate `num_points: 16384` on the server

Train at the intended shape:

```yaml
model:
  num_points: 16384
```

Status:

- Done for the current A100 MIG server.
- Full-model random-sampler smoke runs completed at `16384`.
- Both 10-epoch medium C0 runs completed at `16384`.
- Official full C0 should therefore keep `num_points: 16384`.

If `16384` unexpectedly OOMs during the official full run, test concessions in this order:

1. Use a larger DigitalOcean GPU Droplet if practical, for example H200 instead of H100, or H100/H200 instead of L40S/RTX 6000.
2. Reduce validation/checkpoint overhead if that is causing memory spikes.
3. Try mixed precision if Open3D-ML integration is not too invasive.
4. Try `num_points: 8192`.
5. Use `num_points: 4096` only if absolutely necessary and document the limitation clearly.

Important:

- Do not spend thesis time optimizing laptop memory if DigitalOcean solves the problem.
- A full-size `16384` baseline is now the clean target.
- If `num_points` is reduced even on DigitalOcean, lane-aware sampling becomes even more important because fewer lane points survive per patch.

Definition of done:

- Server smoke and medium runs complete at `num_points: 16384`, or a concession is documented.
- Peak memory is logged.
- The chosen `num_points` is justified in `logs/milestone_c/reports/c0_medium_runs_and_speed_benchmarks.md`.

### 6.4 Lock class-weight policy for baseline

Recommended starting policy:

- Keep `open3d_native_from_measured_counts`, because it already passed the sanity run and is documented.
- Server verification on 2026-05-05 confirmed this policy is actually active in Open3D's `CrossEntropyLoss`.
- Expected and actual effective CE weights matched exactly: `[2.3753318786621094, 36.98638153076172, 1.6340690851211548]`.
- These weights correspond to active classes `road/lane/other`, after ignored label `0` is filtered out.

Important warning:

- Do not paste direct inverse-frequency or sqrt-inverse-frequency vectors into `dataset.class_weights` unless `SemSegLoss` is patched. This Open3D runtime transforms the configured list again through `DataProcessing.get_class_weights(...)`.

Do not tune class weights before the first baseline unless lane IoU is exactly zero and confusion matrix shows class collapse.

Definition of done:

- One class-weight policy is selected.
- Rationale is saved in `logs/milestone_c/reports/class_weight_decision.md`.
- Server loss verification output is recorded.

### 6.5 Run C0 baseline

C0 baseline:

- Input: `xyz + standardized_intensity`.
- `in_channels: 4`.
- Open3D `SemSegRandomSampler`.
- Existing recenter augmentation only.
- Weighted CE as currently configured.
- No engineered features.
- No lane-aware sampling.

Sampler rationale:

- `SemSegRandomSampler` is the C0 baseline sampler.
- `SemSegSpatiallyRegularSampler` was measured at about `1.01s/frame` for eager initialization, estimating roughly `1.5h` total startup before epoch 1 on the full train+validation split.
- The spatial sampler is class-blind and does not directly address the rare lane class.
- The thesis sampling contribution remains C2 lane-aware patch sampling.

Before launching a long C0:

- Create or inspect a server config that uses the server dataset path and `SemSegRandomSampler`.
- Ensure it is not a smoke config with `steps_per_epoch_train: 1` and `steps_per_epoch_valid: 1`.
- Use the calibrated official C0 settings: `steps_per_epoch_train: 4640`, `steps_per_epoch_valid: 720`, `batch_size: 1`, `val_batch_size: 1`, `num_workers: 0`, `pin_memory: false`.
- Run detached and expect about `36-42h` for 30 epochs on the current server.

Suggested run:

- Use 30 epochs as the initial planning target, not as a rule.
- Evaluate every epoch and inspect the validation IoU curves.
- If validation lane IoU and mIoU are still climbing at epoch 30, run longer.
- If the curves plateau earlier, future ablation runs can be shortened with that evidence.
- The 2-epoch pilot and 10-epoch medium stages are already complete; the next C0 step is the official full run.

Definition of done:

- C0 produces full validation metrics.
- You can state baseline `lane IoU`, `lane precision`, `lane recall`, and `mIoU`.

---

## 7. Phase 2: Measure Patch Lane Survival

Goal: verify the actual sampling problem before changing the sampler.

The global lane fraction is known, but what matters for training is how many lane points survive inside sampled patches and deeper RandLA-Net subsampling.

Create a diagnostic script, for example:

```text
tools/analyze_patch_lane_survival.py
```

It should sample training patches using the current pipeline behavior and log:

- Total patches inspected.
- Fraction of patches with at least 1 lane point.
- Fraction of patches with at least 10 lane points.
- Median lane count per patch.
- Mean lane count per patch.
- Lane fraction per patch.
- Approximate deepest-layer lane anchors using `/64` heuristic.
- Sequence/frame IDs of lane-rich and lane-poor patches.

Output:

```text
logs/milestone_c/reports/patch_lane_survival.md
logs/milestone_c/eval/patch_lane_survival.csv
```

Why this matters:

- If uniform patches already contain enough lane points, sampling may be less urgent.
- If many patches have almost no lane points, lane-aware sampling is strongly justified.

Definition of done:

- You know the actual lane count distribution inside training patches.
- You can justify lane-aware sampling with direct patch evidence, not only global class fraction.

---

## 8. Phase 3: Add Option A Engineered Features

Goal: increase model input from 4 channels to 7 channels without changing RandLA-Net architecture.

Current input:

```text
xyz + standardized_intensity
in_channels = 4
feat shape = [N, 1]
```

Option A feature input:

```text
xyz + standardized_intensity + local_contrast + range_corrected_intensity + height_above_ground
in_channels = 7
feat shape = [N, 4]
```

Important:

- Compute all feature statistics using the training split only.
- Apply the same train-split means/stds to val/test.
- Do not use semantic labels to compute model input features.

### 8.1 Build an offline feature cache

Recommended cache location:

```text
logs/milestone_c/feature_cache/features_v1/
```

Each frame cache can store:

```text
seq_id
frame_idx
local_contrast_raw
range_corrected_intensity_raw
height_above_ground_raw
```

Why offline cache:

- Local neighborhood queries are expensive.
- Training should not rebuild KD-trees every epoch if avoidable.
- Cached features make ablations reproducible.

Code-level no-label-leakage rule:

- The feature computation function should not accept `labels`, `raw_labels`, or `semseg_labels` as an argument.
- It should accept only geometry/intensity inputs such as `xyz_ego`, `raw_intensity`, and feature parameters.
- If class labels are needed for diagnostics, compute diagnostics in a separate function after the feature cache is written.
- This makes label leakage mechanically harder instead of relying on memory during implementation.

Definition of done:

- Cache exists for train/val/test frames.
- A random frame can be loaded and matched back to dataset point count.
- Cache metadata records feature version and parameters.

### 8.2 Feature 1: local contrast intensity

Purpose:

- Give the model the Stage 3 signal directly: “is this point brighter than nearby ground?”

Definition:

```text
local_contrast = raw_intensity - median(raw_intensity of nearby ground-like points)
```

Recommended default:

- XY radius: `0.5m`.
- Exclude the point itself.
- Neighbor set should be geometry-based ground-like points, not ground-truth road labels.
- If no valid neighbors exist, set contrast to `0` and flag count for diagnostics.

Why not use raw labels here:

- During inference, labels are unknown.
- If we use raw ID `7/9/10` to choose neighbors, the feature contains ground-truth information and invalidates the experiment.

Sanity checks:

- Lane points should have positive contrast more often than road points.
- Distribution should roughly agree with Stage 3 direction, though not exactly because model feature uses geometry-based neighbors instead of label-based analysis neighbors.

Definition of done:

- Local contrast channel is computed for all frames.
- Train-split mean/std are saved.
- Histograms for road/lane/other look plausible.

### 8.3 Feature 2: range-corrected intensity

Purpose:

- Address the Stage 4 distance decay where intensity signal weakens strongly after 10-20m.

Definition:

```text
range = sqrt(x_ego^2 + y_ego^2 + z_ego^2)
range_corrected = raw_intensity * (range / 5.0)^2
range_corrected = clip(range_corrected, 0, cap)
```

Initial cap:

- Start with `200`, then inspect train distribution.
- If many values saturate, adjust cap using training percentiles.

Important:

- Treat this as an ablation, not guaranteed physics truth.
- PandaSet intensity may already include sensor-specific calibration effects.

Definition of done:

- Range-corrected intensity stats are computed on train split.
- Distance-bucket distributions are inspected.
- Channel is standardized before feeding to model.

### 8.4 Feature 3: height-above-local-ground

Purpose:

- Help distinguish road-surface paint from bright elevated objects.

Definition:

```text
height_above_ground = z_i - min(z of points within 1.0m XY radius)
```

Recommended:

- Clip extreme values before standardization.
- Use geometry only.
- Expect lane points to cluster near zero.

Definition of done:

- Height channel is computed for all frames.
- Lane distribution is near ground level.
- Elevated objects have larger values.

### 8.5 Update dataset class and config

Modify the dataset so it can select feature modes:

```yaml
dataset:
  feature_set: intensity_only
```

and later:

```yaml
dataset:
  feature_set: features_v1
  feature_cache_dir: ./logs/milestone_c/feature_cache/features_v1

model:
  in_channels: 7
```

Implementation expectation:

- `intensity_only` keeps the current behavior.
- `features_v1` loads cached feature arrays and concatenates them with standardized intensity.
- Feature output remains `float32`.
- Returned `feat` has shape `[N, 4]`.

Definition of done:

- C0 still runs unchanged with `feature_set: intensity_only`.
- A 1-step run works with `feature_set: features_v1` and `in_channels: 7`.

### 8.6 Run C1 features

C1:

- Input: `xyz + standardized_intensity + local_contrast + range_corrected_intensity + height_above_ground`.
- `in_channels: 7`.
- Uniform/default sampling.
- Same loss and optimizer as C0.
- Same epochs as C0 if possible.

Definition of done:

- C1 full validation metrics exist.
- Compare C1 vs C0 on lane IoU, lane precision, lane recall, and confusion matrix.

---

## 9. Phase 4: Add Lane-Aware Sampling

Goal: make sure the model sees enough lane points during training.

This is the main class-imbalance intervention. Class weights change the loss after a point is sampled. Lane-aware sampling changes what the model sees in the first place.

### 9.1 Define sampling behavior

Recommended initial policy:

```text
50% lane-centered patches
50% default/uniform patches
```

Lane-centered means:

- Pick a training frame that contains lane points.
- Pick a lane point as the patch center.
- Use the normal patch selection around that center.

Do not apply lane-aware sampling to validation or test.

Code-level guard:

- The sampler must explicitly check `split == "train"` or equivalent before lane-aware behavior can run.
- Validation and test code paths should assert that lane-aware sampling is disabled.
- This should be enforced in code, not only by remembering to set a config flag correctly.

Definition of done:

- There is a clear train-only switch, for example:

```yaml
dataset:
  lane_aware_sampling:
    enabled: true
    fraction: 0.5
```

### 9.2 Implement least-invasive Open3D integration

Because Open3D-ML is a constraint, implementation should start with the least invasive path.

Try in this order:

1. Wrap or subclass the existing Open3D-ML sampler if it exposes center selection cleanly.
2. Add dataset-level support for lane-centered sample selection if sampler wrapping is messy.
3. Only vendor Open3D-ML sampler code if the first two paths fail.

Definition of done:

- Training can run with lane-aware sampling enabled.
- Validation uses the normal evaluation path.

### 9.3 Instrument the sampler

Before long training, log the first 100-500 sampled patches.

Metrics:

- Patch lane count.
- Patch road count.
- Patch other count.
- Whether selected center was lane.
- Sequence/frame ID.
- Approximate deepest-layer lane anchor count.

Expected result:

- Lane-centered patches should contain much more lane evidence than uniform patches.
- Overall training mix should still include lane-free and road-only context.

Definition of done:

- `logs/milestone_c/reports/lane_aware_sampling_check.md` exists.
- The sampler actually increases lane presence in training patches.

### 9.4 Run C2 sampling-only

C2:

- Input: current baseline features only.
- `in_channels: 4`.
- Lane-aware sampling enabled.
- Same loss/optimizer.

Why run sampling-only:

- It isolates the effect of class-imbalance handling.

Definition of done:

- C2 metrics exist.
- You can compare C2 vs C0 directly.

### 9.5 Run C3 features + sampling

C3:

- Input: features_v1.
- `in_channels: 7`.
- Lane-aware sampling enabled.
- Same loss/optimizer.

This is likely the most important Option A experiment.

Definition of done:

- C3 metrics exist.
- You can compare C3 vs C0 and C1.
- Confusion matrix shows whether lane precision/recall improved.

---

## 10. Phase 5: Add Conservative Augmentations

Goal: improve generalization without breaking lane geometry.

Current augmentation is only xy recentering. Add augmentations carefully.

### 10.1 Intensity jitter

Apply only during training.

Purpose:

- Prevent the model from overfitting exact intensity values.

Recommended:

- Small Gaussian noise on standardized intensity only.
- Do not blindly jitter all engineered channels, because they have different meanings and scales.

Definition of done:

- Feature histograms remain reasonable after augmentation.
- A 1-epoch run stays stable.

### 10.2 Small yaw rotation

Be conservative.

Recommended:

- Start with small z-axis rotation, for example `[-10°, +10°]` or `[-15°, +15°]`.

Why not full random rotation immediately:

- Lane geometry is meaningful relative to ego heading.
- Full arbitrary rotation can erase useful forward-road priors.

Definition of done:

- Train-time only.
- Validation/test not augmented.

### 10.3 Lateral flip

Consider a left-right flip if it matches the ego-coordinate convention.

Avoid careless forward-backward flips unless you confirm they make sense for forward-facing LiDAR.

Definition of done:

- The transformed point cloud still represents plausible driving geometry.

### 10.4 Run C4 final augmented

C4:

- features_v1.
- lane-aware sampling.
- conservative augmentations.
- Same loss/optimizer.

Definition of done:

- C4 metrics exist.
- If augmentations hurt lane IoU, report C3 as the final model instead.

---

## 11. Phase 6: Ablation Matrix

Minimum Option A ablation table:

| ID | Input Features | Sampling | Augmentation | Purpose |
|---|---|---|---|---|
| C0 | intensity only | default | recenter only | Baseline |
| C1 | features_v1 | default | recenter only | Feature effect |
| C2 | intensity only | lane-aware | recenter only | Sampling effect |
| C3 | features_v1 | lane-aware | recenter only | Combined effect |
| C4 | features_v1 | lane-aware | conservative augment | Final candidate |

If compute allows, add:

| ID | Input Features | Sampling | Augmentation | Purpose |
|---|---|---|---|---|
| C1a | local contrast only | default | recenter only | Isolate strongest analysis finding |
| C1b | local contrast + range-corrected | default | recenter only | Test distance feature separately |

For every config, report:

- mIoU.
- Road IoU.
- Lane IoU.
- Other IoU.
- Lane precision.
- Lane recall.
- Lane F1.
- Confusion matrix.
- Best epoch by validation lane IoU.
- Best epoch by validation mIoU.
- Peak GPU memory.
- Training time.

Definition of done:

- C0, C1, C2, C3, and C4 are complete unless DigitalOcean cost/time becomes unexpectedly limiting.
- Results table is ready for thesis writing.

DigitalOcean expectation:

- The full C0-C4 matrix should now be considered the planned experiment set.
- Individual feature ablations C1a/C1b are still optional, but more realistic if training time is acceptable.
- If DigitalOcean budget is tight, prioritize C0, C1, C3, C4, then C2.

---

## 12. Phase 7: Qualitative Evaluation

Goal: show what the model actually does.

Select a small set of frames:

- One easy lane-visible frame.
- One medium frame.
- One hard/low-intensity frame.
- One intersection or marking-heavy frame if available.

For each selected frame, save:

- Ground truth point cloud colored by class.
- C0 prediction colored by class.
- Best Option A prediction colored by class.
- Error view if possible: correct vs false lane vs missed lane.

Output:

```text
logs/milestone_c/figures/qualitative/
```

Why this matters:

- If lane IoU improves but visually predicts messy lane blobs, the thesis needs to say that.
- If stop lines are predicted as lane, this will show up visually and in lane precision.

Definition of done:

- At least 3 frames have GT vs prediction figures.
- Figures are usable in the thesis without rerendering.

---

## 13. What To Do With Intensity Specifically

This is the short operational answer.

Keep the current standardized intensity channel as the base input. Do not remove it.

Add three extra feature channels in Option A:

- `local_contrast`: because Stage 3 showed local brightness is stronger than global brightness.
- `range_corrected_intensity`: because Stage 4 showed intensity signal decays with distance.
- `height_above_ground`: because lane markings are road-surface points and this helps reject bright elevated objects.

Use raw intensity to compute these features, then standardize each feature using training-split statistics.

Do not use raw labels to compute model input features. For example, do not compute “median of nearby road points” using ground-truth road labels. Instead, compute “median of nearby ground-like points” using geometry.

Keep the original intensity preprocessing for the base channel:

- Clip using training stats.
- Standardize using training mean/std.

The new feature channels should have their own train-split means/stds.

---

## 14. What To Do With Class Imbalance

Use two mechanisms together.

### Loss weighting

Keep the current weighted CE policy for the first baseline because it already passed sanity training.

If lane IoU stays at zero:

- Inspect confusion matrix first.
- If the confusion matrix shows class collapse, test a stronger class-weight variant before adding features.
- The first stronger candidate is `raw_inverse_frequency`, because it directly increases the lane penalty.

Do not start by tuning loss weights blindly.

Do not add features to a collapsed baseline and call it an ablation. If C0 predicts road everywhere, fix the class-collapse problem first; otherwise C1/C2/C3 comparisons are not meaningful.

### Sampling

Implement lane-aware sampling after baseline and patch survival diagnostics.

Recommended start:

- `lane_fraction = 0.5`.

If overfitting or lane false positives increase badly:

- Try `lane_fraction = 0.3`.

Keep validation/test sampling normal.

---

## 15. What To Do With the Model

For Option A:

- Keep Open3D-ML RandLA-Net unchanged.
- Keep `num_classes: 3`.
- Keep ignore label excluded.
- Change only `in_channels` when feature channels are added.
- Avoid custom losses unless baseline completely fails.

Do not implement for Option A:

- Two-head architecture.
- PCA anisotropy features.
- Principal direction features.
- Lovasz-Softmax.
- Custom RandLA-Net fork.

Those are good reach ideas, but only after the conservative pipeline is stable.

---

## 16. Recommended Timeline

### Week 1: baseline and evaluation

Do:

- Clean experiment setup.
- Set up DigitalOcean GPU Droplet environment and verify one-sample dataset loading.
- Create Milestone C training script.
- Add full validation metrics.
- Revalidate `num_points: 16384` on DigitalOcean.
- Run C0 baseline.

End state:

- First real full-size lane IoU, lane precision, lane recall, confusion matrix.

### Week 2: features

Do:

- Build feature cache.
- Add local contrast, range-corrected intensity, height-above-ground.
- Compute train-split stats.
- Run C1 features.

End state:

- Know whether intensity-derived features improve over baseline.

### Week 3: sampling

Do:

- Measure patch lane survival.
- Implement lane-aware sampling.
- Run C2 sampling-only.
- Run C3 features + sampling.

End state:

- Know whether class-imbalance handling improves lane metrics.

### Week 4: augmentations

Do:

- Add conservative train-only augmentations.
- Run C4.

End state:

- Final Option A candidate exists.

### Week 5: finish ablations and figures

Do:

- Finish the full C0-C4 ablation matrix.
- Generate qualitative prediction figures.
- Create results tables.

End state:

- Results chapter has the numbers and visuals it needs.

### Week 6: writing buffer

Do:

- Write method, experiments, results, discussion.
- Use raw intensity analysis as dataset-analysis chapter.
- Ask advisor to review scope and result interpretation.

End state:

- Thesis draft is complete enough for revision.

---

## 17. Success Criteria

Milestone C is successful if you can answer these with evidence:

- What is the baseline lane IoU?
- Does local/context intensity feature engineering improve lane segmentation?
- Does lane-aware sampling improve lane recall or lane IoU?
- Does the final model confuse stop lines/other markings with lanes?
- Does performance degrade with distance?
- Is the final improvement supported by an ablation table?
- Can someone reproduce the result from saved config, seed, and checkpoint?

---

## 18. If Things Go Wrong

### Loss decreases but lane IoU is zero

Likely meaning:

- The model may have collapsed to predicting mostly the majority class, usually road.
- Loss can still decrease in this situation because road dominates the point count.

Check:

- Confusion matrix first. Confirm whether lane ground-truth points are predicted as road or other.
- Class label mapping.
- Whether `ignored_label_inds: [0]` is actually being applied.
- Whether lane points exist in sampled patches.
- Whether class weights are being interpreted correctly by Open3D-ML.

Then try:

- Try a stronger class-weight variant, starting with `raw_inverse_frequency`.
- Skip ahead to lane-aware sampling earlier than planned if patches have weak lane support.
- Re-run a short baseline before adding features.

Do not:

- Continue to C1 features if C0 is collapsed. That would make the feature ablation meaningless.

### Feature run performs worse

Likely meaning:

- The feature may be badly scaled, misaligned with point order, or not informative.
- A large-scale feature can dominate RandLA-Net's first input MLP and hurt training.

Check:

- Feature scale and standardization.
- Whether every feature is approximately zero-mean/unit-variance using train-split statistics.
- Feature cache alignment with point order.
- Whether validation is using train statistics.
- Whether any feature accidentally used labels.
- Histograms of each feature stratified by class.

Interpretation:

- If lane and road distributions look identical, the feature may be useless.
- If the distributions look impossible or extreme, the feature is probably broken.

Then try:

- Local contrast only.
- Disable range correction if it amplifies noise.
- Recompute feature statistics and rerun a short smoke training before committing to a full run.

### Augmentations hurt the lane class specifically

Likely meaning:

- The augmentation may be destroying task-relevant geometry.
- Lane geometry is anisotropic relative to ego heading, so generic full-rotation recipes can be harmful.

Check:

- Lane IoU, lane precision, and lane recall separately.
- Whether road/other improve while lane gets worse.
- Whether rotation range is too wide.

Then try:

- Reduce yaw rotation below `±10°`.
- Drop yaw rotation entirely.
- Keep intensity jitter and lateral flip only if they do not hurt lane metrics.

### Lane-aware sampling hurts precision

Likely meaning:

- The training distribution may be over-enriched with lane neighborhoods.
- Or validation/test sampling may accidentally be lane-aware, which invalidates metrics.

Check:

- Validation/test sampling code path explicitly.
- Whether `lane_fraction=0.5` is too high.
- Whether model predicts stop lines as lane.
- Whether lane recall improved but precision collapsed.

Then try:

- `lane_fraction=0.3`.
- Confirm validation/test use normal sampling.
- Keep C1 features as final if sampling is harmful.

### `16384` still OOMs

Do:

- First try a larger DigitalOcean GPU Droplet if practical.
- Document the failure with GPU type and peak memory.
- Choose `8192` only if `16384` is still not practical on rented hardware.
- Recompute expected deep anchors: `E = 8192 * 0.007 / 64 = 0.896`, approximately `0.9`.
- Treat lane-aware sampling as mandatory at `8192`, because `E < 1.0` means most deepest-layer patches have too little lane evidence.
- Document the concession transparently in methodology.

Do not:

- Hide the concession.
- Spend the whole thesis timeline on memory debugging.

---

## 19. Final Thesis Artifacts To Produce

Minimum final artifacts:

- `logs/milestone_c/runs/C0_baseline/eval_history.csv`
- `logs/milestone_c/runs/C1_features/eval_history.csv`
- `logs/milestone_c/runs/C2_sampling/eval_history.csv`
- `logs/milestone_c/runs/C3_features_sampling/eval_history.csv`
- `logs/milestone_c/runs/C4_final_augmented/eval_history.csv`
- Confusion matrices for every run.
- Config snapshots for every run.
- Final checkpoint for best run.
- Ablation table CSV.
- Qualitative figures.
- Distance-binned lane recall plot.
- Short final report summarizing baseline vs final model.

Minimum thesis table:

| Run | mIoU | Road IoU | Lane IoU | Other IoU | Lane Precision | Lane Recall | Lane F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 baseline | | | | | | | |
| C1 features | | | | | | | |
| C2 sampling | | | | | | | |
| C3 features + sampling | | | | | | | |
| C4 final | | | | | | | |

---

## 20. First Concrete Next Steps

Do these next, in this order:

1. Create `logs/milestone_c/` run structure.
2. Set up DigitalOcean GPU Droplet and run a one-sample dataset/model smoke test there.
3. Create or adapt a Milestone C training script separate from the Day 6 sanity script.
4. Add validation metric logging: mIoU, per-class IoU, precision/recall/F1, confusion matrix.
5. Run a short DigitalOcean baseline at `num_points: 16384` and log memory.
6. Run C0 baseline long enough to get the first real lane IoU.
7. Only after that, build the feature cache and run C1.

The key discipline is: **baseline first, then improvements**. Without the baseline, we cannot prove that the intensity work, sampling work, or augmentations helped.

---

## 21. Glossary

**AUC:** Area under the ROC curve. In the intensity analysis, it can be read as the probability that a randomly chosen lane point receives a higher intensity score than a randomly chosen comparison point.

**E, expected deep anchors:** A rough heuristic for how many lane points survive to the deepest RandLA-Net layer: `E = num_points * lane_fraction / total_downsampling`. With `16384`, lane fraction about `0.007`, and `/64` downsampling, `E` is about `1.8`; with `8192`, it is about `0.9`.

**LFA, Local Feature Aggregation:** RandLA-Net's neighborhood encoding mechanism. It aggregates each point with nearby points using coordinates, relative offsets, distances, and feature vectors, so added feature channels can influence local neighborhood learning.

**Lane-aware sampling:** A training-time sampling strategy where some patches are centered on lane points instead of being selected uniformly. It addresses the fact that lane points are rare and may otherwise barely appear in training patches.

**mIoU:** Mean Intersection-over-Union across active classes. For this project, active-class mIoU averages road IoU, lane IoU, and other IoU, while ignore is excluded.

**Range correction:** A feature transformation that scales raw intensity by distance, for example `raw_intensity * (range / 5.0)^2`. It tests whether the measured distance decay in lane intensity is partly caused by geometric LiDAR falloff.

**Local contrast:** A feature measuring how much brighter or darker a point is than nearby ground-like points. It captures “bright relative to surroundings,” which the raw intensity analysis showed is more useful than relying only on absolute intensity.

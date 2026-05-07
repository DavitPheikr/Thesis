# Thesis Pipeline Context Report

A concrete grounding document for the LiDAR lane-segmentation thesis pipeline: PandaSet + RandLA-Net.

---

## 1. Project Structure and Pipeline Shape

### Top-level layout

```
configs/                          # YAML configuration files
configs/randlanet_pandaset_ff_lane3.yml
configs/splits/                   # Frozen train/val/test sequence lists

datasets/                         # Simple import wrapper (delegates to src/)

docs/                             # Milestone docs organized by context, milestones A/B
docs/context/
docs/milestone_a/
docs/milestone_b/
docs/milestone_c/

logs/                             # Artifacts, statistics, sanity-run reports, analysis
logs/raw_intensity_analysis/      # Multi-stage intensity signal analysis
logs/milestone_c/runs/            # Milestone C run artifacts, including completed C0
logs/milestone_c/reports/         # Milestone C operational reports
logs/milestone_b_training_statistics.json
logs/milestone_b_sanity_train_report.txt

pandaset/                         # Local dataset directory (PandaSet sequences)
pandaset-devkit/                  # Local patched PandaSet devkit
panda/                            # Local Python venv (python3.12)

src/thesis_pipeline/              # Core reusable logic
src/thesis_pipeline/adapters/     # PandaSet remap logic
src/thesis_pipeline/datasets/     # Open3D-ML dataset class
src/thesis_pipeline/core/         # Compatibility and path helpers
src/thesis_pipeline/checks/       # Small reusable check modules
src/thesis_pipeline/analysis/     # Analysis modules (e.g., intensity signal)

tools/                            # Runnable milestone scripts
tools/sanity_train_check.py       # Day 6 sanity training entrypoint
tools/compute_training_statistics.py
tools/freeze_split.py
tools/audit_semseg_sequences.py
tools/[20+ other check/audit scripts]

README.md                         # Global orientation doc
MILESTONE_B_DAY6_PREP_AND_STATUS.md  # Day 6 preparation and status
requirements_working_panda.txt
```

### Training entrypoints

**Milestone B sanity entrypoint:** `tools/sanity_train_check.py`

**Milestone C real-training entrypoint:** `tools/train_milestone_c.py`

Milestone C uses a custom `MilestoneCPipeline` wrapper around Open3D-ML's torch semantic-segmentation pipeline. The wrapper keeps the Open3D RandLA-Net model path but adds thesis-critical artifacts:

- run-local config snapshot, git commit, seed, CLI args, start/end time, and stdout log;
- explicit non-resume behavior;
- per-epoch validation metrics;
- active-class confusion matrices;
- lane recall by distance bucket;
- run-local checkpoints.

**Framework:** Open3D-ML (torch backend)

- Pipeline base class: `open3d._ml3d.torch.pipelines.SemanticSegmentation`
- Config file: `configs/randlanet_pandaset_ff_lane3.yml`

### RandLA-Net model location

**Status:** Imported from Open3D-ML, not vendored or fork

**Source:** `from open3d.ml.torch.models import RandLANet` ([src/thesis_pipeline/checks/day1.py:1](src/thesis_pipeline/checks/day1.py#L1))

**Variant:** Open3D-ML implementation of the Hu et al. 2020 RandLA-Net architecture (no custom modifications applied in this thesis pipeline; the model is used as-is from the Open3D library version 0.19.0)

---

## 2. Data Pipeline

### PandaSet loading and frame structure

**Loading code:** [src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py:202–249](src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py#L202-L249)

**Fields read per frame:**

- `xyz_world`: point coordinates in world frame (from `pc_df[["x", "y", "z"]]`, dtype float32)
- `intensity`: raw LiDAR reflectance (from `pc_df["i"]`, dtype float32)
- `semseg_labels`: raw PandaSet semantic class IDs (from `semseg_df.loc[pc_df.index, "class"]`)
- `pose`: ego vehicle pose (from `seq.lidar.poses[frame_idx]`)

**Preprocessing chain (exact order with file:line references)**

1. **Raw label read and remap:** [pandaset_ff_lane3_dataset.py:219–220](src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py#L219-L220)

   ```python
   raw_labels = semseg_df.loc[pc_df.index, "class"].to_numpy(dtype=np.int32)
   labels = remap_raw_pandaset_ids(raw_labels)
   ```

2. **XYZ world-to-ego transform:** [pandaset_ff_lane3_dataset.py:222–226](src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py#L222-L226)

   ```python
   xyz_world = pc_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
   pose = seq.lidar.poses[frame_idx]
   xyz_ego = pds_geometry.lidar_points_to_ego(xyz_world, pose).astype(
       np.float32, copy=False
   )
   ```

3. **Intensity clipping:** [pandaset_ff_lane3_dataset.py:228–229](src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py#L228-L229)

   ```python
   intensity = pc_df["i"].to_numpy(dtype=np.float32)
   intensity = np.clip(intensity, self.intensity_clip_low, self.intensity_clip_high)
   ```

   Values: clip_low=0.0, clip_high=114.0 (p0.5/p99.5 from training data; [logs/milestone_b_training_statistics.json:15–18](logs/milestone_b_training_statistics.json#L15-L18))

4. **Intensity standardization:** [pandaset_ff_lane3_dataset.py:230](src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py#L230)

   ```python
   intensity = (intensity - self.intensity_mean) / self.intensity_std
   ```

   Mean: 22.481164932250977, Std: 14.912428855895996 (from training split; [logs/milestone_b_training_statistics.json:17–18](logs/milestone_b_training_statistics.json#L17-L18))

5. **Feature tensor construction:** [pandaset_ff_lane3_dataset.py:231](src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py#L231)

   ```python
   feat = intensity[:, None].astype(np.float32, copy=False)
   ```

6. **Memory cleanup:** [pandaset_ff_lane3_dataset.py:239–247](src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py#L239-L247)
   Releases cached frame data after extraction to avoid memory pressure from repeated dataset access.

### Semseg label remapping

**Location:** [src/thesis_pipeline/adapters/pandaset_ff_lane3.py:70–77](src/thesis_pipeline/adapters/pandaset_ff_lane3.py#L70-L77)

**Mapping code:**

```python
def remap_raw_pandaset_ids(raw_labels: np.ndarray) -> np.ndarray:
    remapped = np.full(raw_labels.shape, OTHER_LABEL, dtype=np.int32)  # default=3
    remapped[np.isin(raw_labels, list(RAW_IGNORE_IDS))] = IGNORE_LABEL  # {1,2,3,4}→0
    remapped[raw_labels == RAW_ROAD_ID] = ROAD_LABEL                    # 7→1
    remapped[raw_labels == RAW_LANE_ID] = LANE_LABEL                    # 8→2
    return remapped
```

**Raw IDs:** [src/thesis_pipeline/adapters/pandaset_ff_lane3.py:20–23](src/thesis_pipeline/adapters/pandaset_ff_lane3.py#L20-L23)

- Raw ignore set: `{1, 2, 3, 4}` → remapped to `0`
- Raw road: `7` → remapped to `1`
- Raw lane: `8` → remapped to `2`
- Raw other (everything else): → remapped to `3`

### Intensity handling summary

**Current approach:**

- Clipped to p0.5/p99.5 range of training split (0.0–114.0)
- Standardized (z-score normalization) using training-split mean/std
- Passed to model as single-channel feature via `feat[:, None]` (shape: [N, 1])

**Analysis findings:** Raw intensity helps separate lane from road (median lane=33 vs road=28), but does NOT separate lane from other road paint (stop line, other markings all≈median 33). Local contrast within 0.5m radius is strong (lane points ≈12.46 units brighter than local neighborhood). Intensity signal decays with distance (useful <20m, very weak >30m). Conclusion: keep intensity as feature but model needs geometry and lane-aware sampling. See [logs/raw_intensity_analysis/reports/final/intensity_notes.md](logs/raw_intensity_analysis/reports/final/intensity_notes.md).

### Patch/subsampling strategy

**Grid subsampling:** `grid_size: 0.04` (voxel grid at 4cm resolution; [configs/randlanet_pandaset_ff_lane3.yml:37](configs/randlanet_pandaset_ff_lane3.yml#L37))

**Sampling type for C0:** Open3D-ML `SemSegRandomSampler` (declared in config [configs/randlanet_pandaset_ff_lane3.yml](configs/randlanet_pandaset_ff_lane3.yml)).

`SemSegSpatiallyRegularSampler` was benchmarked on the server and deferred. It eagerly initializes sampler state by loading/preprocessing every frame before epoch 1, with a measured estimate of about `1.30h` for train initialization plus about `12min` for validation initialization on the full split. It is also class-blind, so it does not directly address rare lane-class sampling.

**Lane-aware sampling:** Not yet implemented. Current split evidence:

- Lane fraction in training data: 0.70% (2098182 lane points / ~298M total; [logs/milestone_b_training_statistics.json:20–25](logs/milestone_b_training_statistics.json#L20-L25))
- Expected deep anchor count (heuristic E = num_points × p_lane / 64): 1.8 anchors at baseline (16384 points) – below 2.0 threshold
- **Carry-forward:** Lane-aware patch oversampling indicated but not yet implemented. See [logs/milestone_b_stop_conditions.txt:58–66](logs/milestone_b_stop_conditions.txt#L58-L66).

### Train/val/test split

**Split type:** Sequence-level (entire sequences assigned to splits, not individual frames mixed)

**Sequences and frame counts:**

- Training: 58 sequences, 4640 frames
- Validation: 9 sequences, 720 frames
- Test: 9 sequences, 720 frames

**Split definition location:** `configs/splits/train.txt`, `configs/splits/val.txt`, `configs/splits/test.txt` (frozen during Milestone B)

**Split reading:** [src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py:87–96](src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py#L87-L96)

---

## 3. Model and Training Config

### RandLA-Net hyperparameters

**Location:** [configs/randlanet_pandaset_ff_lane3.yml:22–40](configs/randlanet_pandaset_ff_lane3.yml#L22-L40)

- `num_neighbors: 16` (KNN neighborhood size)
- `num_layers: 3` (encoder/decoder layers)
- `num_points: 16384` (target point cloud size per sample; **note:** sanity run used 4096 as OOM concession)
- `num_classes: 3` (road, lane, other; ignore class 0 excluded during loss)
- `in_channels: 4` (xyz + intensity = 3 + 1)
- `dim_features: 8` (initial feature dimension)
- `dim_output: [16, 64, 128]` (output feature dims per encoder layer)
- `sub_sampling_ratio: [4, 4, 4]` (point reduction ratios across layers)
- `grid_size: 0.04` (grid subsampling size in meters)
- `augment.recenter.dim: [0, 1]` (augmentation: recenter in xy plane)

### Loss function

**Type:** Weighted cross-entropy

**Loss class:** `SemSegLoss` (from `open3d._ml3d.torch.modules.losses.semseg_loss`)

**Loss construction:** [logs/milestone_b_loss_interface.json:1–11](logs/milestone_b_loss_interface.json#L1-L11)

- SemSegLoss reads `dataset.cfg.class_weights` (a list)
- Internally transforms it via `DataProcessing.get_class_weights` (Open3D native transform)
- Constructs `torch.nn.CrossEntropyLoss(weight=...)` for active classes only (ignore label 0 omitted)
- Active-class order after filtering: [road, lane, other]

**Class weights (as passed to dataset.cfg):** [configs/randlanet_pandaset_ff_lane3.yml:6](configs/randlanet_pandaset_ff_lane3.yml#L6)

```yaml
class_weights: [119562394.0, 2098182.0, 176504630.0]
```

These are measured class counts from training data, not direct CE weights. Open3D's `get_class_weights` transforms them internally. See [logs/milestone_b_training_statistics.json:61–89](logs/milestone_b_training_statistics.json#L61-L89) for detailed weight semantics and alternatives.

### Optimizer and schedule

**Optimizer:** Adam (configured in Open3D pipeline, not explicitly named in YAML)

**Learning rate:** `0.001` [configs/randlanet_pandaset_ff_lane3.yml:45](configs/randlanet_pandaset_ff_lane3.yml#L45)

**LR scheduler:** Exponential decay with `scheduler_gamma: 0.99` [configs/randlanet_pandaset_ff_lane3.yml:52](configs/randlanet_pandaset_ff_lane3.yml#L52)

**Batch size:** `1` (official C0 recommendation after medium-run comparison) [configs/randlanet_pandaset_ff_lane3.yml](configs/randlanet_pandaset_ff_lane3.yml)

**Validation batch size:** `1`

**Num workers:** `0` (required on current server; worker subprocesses segfault)

**Pin memory:** `false` (server benchmark showed `pin_memory=true` is slower with zero workers)

**Num epochs:** official full C0 completed for 30 epochs.

**Steps per epoch:**

- Live local config contains bounded values for short local checks.
- Official server C0 full run used:
  ```text
  steps_per_epoch_train: 4640
  steps_per_epoch_valid: 720
  ```
  This is approximately one sampled patch per train/validation frame per epoch with `SemSegRandomSampler`.

**Official full C0 config snapshot:** [logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/config_snapshot.yml](logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/config_snapshot.yml)

**Why batch size 1 remains the recommendation:**

- `C0_baseline_medium_10ep_random` with batch size 1 learned successfully and gave the best lane balance.
- `C0_baseline_medium_10ep_random_bs2` with batch size 2 also learned, but was slower in the realistic 10-epoch run and over-predicted lane by epoch 10.
- Batch size 2 final lane behavior was high recall but low precision/F1:
  ```text
  lane_precision 0.104447
  lane_recall    0.925006
  lane_f1        0.187700
  pred_lane_pct  about 5.88%
  true_lane_pct  about 0.66%
  ```

**DataLoader worker and pin-memory status:**

- `num_workers=1`, `2`, and `4` all failed with worker segmentation faults on the current server.
- `pin_memory=true` was slower than `pin_memory=false` when `num_workers=0`.
- Treat multiprocessing DataLoader support as future engineering work, not a C0 blocker.

### Current best sanity-run metrics

**Day 6 sanity run outcome:** [logs/milestone_b_sanity_train_report.txt](logs/milestone_b_sanity_train_report.txt)

- Status: `PASS`
- Weight variant used: `open3d_native_from_measured_counts` (measured counts list above)
- Steps captured: 15 (across 2 epochs × ~5–7 steps per epoch)
- Loss values: [1.137, 1.227, 0.989, 1.207, 1.168, 1.650, 1.090, 1.076, 1.074, 1.097, 0.872, 1.081, 1.000, 0.949, 1.080]
- Loss first: 1.1369
- Loss last: 1.0802
- Loss ratio (final/initial): 0.9501 (pass: <10.0)
- All losses finite: yes
- Loss changed across iterations: yes

**Important caveat:** This sanity run used `num_points: 4096` (OOM concession). Baseline intent is `16384`. See [logs/milestone_b_stop_conditions.txt:34–51](logs/milestone_b_stop_conditions.txt#L34-L51).

### Augmentations during training

**Defined augmentations:** [configs/randlanet_pandaset_ff_lane3.yml:38–39](configs/randlanet_pandaset_ff_lane3.yml#L38-L39)

```yaml
augment:
  recenter:
    dim: [0, 1]
```

- Recenter: shift point cloud to origin in xy dimensions only

**No other augmentations configured:** rotation, flip, intensity jitter, etc. are not explicitly declared in the YAML. Open3D-ML may apply default augmentations; see [logs/milestone_b_pipeline_interface.json](logs/milestone_b_pipeline_interface.json) for discovered runtime semantics.

---

## 4. Features Fed to the Model

### Per-point input channels

**Exact list with shapes:**

- `xyz_ego`: 3 channels (x, y, z in ego frame), dtype float32, shape [N, 3]
- `intensity`: 1 channel (standardized reflectance), dtype float32, shape [N, 1]

**Total input:** 4 channels = `in_channels: 4` [configs/randlanet_pandaset_ff_lane3.yml:34](configs/randlanet_pandaset_ff_lane3.yml#L34)

### Feature engineering

**Module location:** None yet implemented for engineered features.

**Current feature path:** Raw intensity (single channel after clipping & standardization) is the only engineered feature. No height-above-ground, local-contrast, PCA, or other per-point features computed online or cached offline.

**Candidate features (under consideration, not yet implemented):**

- Local contrast (intensity relative to neighborhood) – analyzed in [logs/raw_intensity_analysis](logs/raw_intensity_analysis) but not yet implemented as model input
- Height above ground (z-offset from ego vehicle frame)
- Distance-to-ego (range normalization)

---

## 5. Evaluation

### Metrics reported

The official full C0 baseline is complete. The canonical detailed report is [logs/milestone_c/reports/c0_full_baseline_results.md](logs/milestone_c/reports/c0_full_baseline_results.md).

**Official full C0 run:**

```text
run: C0_baseline_full_30ep_random_bs1
run_dir: logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/
epochs: 30
train/valid steps: 4640 / 720
batch_size: 1
val_batch_size: 1
num_workers: 0
pin_memory: false
seed: 42
wall_clock: about 41h 00m 31s
```

**Selected C0 checkpoint:** epoch 18, because it has best validation lane F1, lane IoU, and mIoU.

```text
checkpoint: logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints/ckpt_epoch_00018.pth
mIoU           0.703805
road_iou       0.881990
lane_iou       0.310355
other_iou      0.919072
lane_precision 0.437552
lane_recall    0.516349
lane_f1        0.473696
```

**Final epoch checkpoint:** epoch 30.

```text
checkpoint: logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints/ckpt_epoch_00030.pth
mIoU           0.683506
road_iou       0.866600
lane_iou       0.264658
other_iou      0.919261
lane_precision 0.321885
lane_recall    0.598172
lane_f1        0.418545
```

The final epoch has higher lane recall but lower lane precision and lower lane F1 than epoch 18. It predicts lane about `1.86x` as often as lane appears in validation (`1.487%` predicted lane vs `0.800%` true lane). Use epoch 18 as the official selected C0 checkpoint.

**Sanity-run loss trajectory:** Captured in [logs/milestone_b_sanity_train_report.txt](logs/milestone_b_sanity_train_report.txt)

**Milestone C metric plumbing:** `tools/train_milestone_c.py` writes per-epoch validation artifacts for mIoU, per-class IoU, precision, recall, F1, lane recall by distance bucket, confusion matrix, validation wall-clock time, and validation peak GPU memory.

**Server smoke evidence:** random-sampler tiny and full-model one-step runs completed training, validation, and checkpoint writing. These are readiness checks, not final performance baselines.

**Medium C0 calibration evidence:** see [logs/milestone_c/reports/c0_medium_runs_and_speed_benchmarks.md](logs/milestone_c/reports/c0_medium_runs_and_speed_benchmarks.md). These runs justified the full C0 runtime settings but are not the primary C0 performance baseline.

Batch-size-1 medium run:

```text
run: C0_baseline_medium_10ep_random
epochs: 10
train/valid steps: 500 / 200
batch_size: 1
num_workers: 0
pin_memory: false

train_loss 0.708701 -> 0.339711
val_loss   0.489297 -> 0.408176
mIoU       0.502071 -> 0.558003
lane_iou   0.077419 -> 0.160835
lane_f1    0.143712 -> 0.277103
```

Best lane epoch for the batch-size-1 medium run:

```text
epoch 7
lane_iou       0.192446
lane_precision 0.250676
lane_recall    0.453093
lane_f1        0.322776
```

Batch-size-2 medium run:

```text
run: C0_baseline_medium_10ep_random_bs2
epochs: 10
train/valid steps: 500 / 200
batch_size: 2
num_workers: 0
pin_memory: false
```

It completed and learned, but by epoch 10 it over-predicted lane:

```text
lane_precision 0.104447
lane_recall    0.925006
lane_f1        0.187700
true_lane_pct  about 0.66%
pred_lane_pct  about 5.88%
```

### Confusion matrix and per-class breakdown

The Milestone C validation path writes active-class confusion matrices for `road/lane/other` each epoch. Active class indexing after ignore-label filtering is:

```text
0 = road
1 = lane
2 = other
```

Epoch 18 selected checkpoint confusion matrix:

```text
              predicted road   predicted lane   predicted other
true road        4,696,484          44,326            42,531
true lane           44,563          49,775             2,060
true other         496,970          19,657         6,373,555
```

Epoch 30 final checkpoint confusion matrix:

```text
              predicted road   predicted lane   predicted other
true road        4,380,056          88,801            45,572
true lane           35,604          56,346             2,247
true other         504,265          29,903         6,626,263
```

The selected checkpoint's main lane error is missed lane predicted as road. The final checkpoint finds more true lane points but also produces more false-positive lane predictions from road and other.

### Validation result logging

**Sanity-run logs location:**

- Report: [logs/milestone_b_sanity_train_report.txt](logs/milestone_b_sanity_train_report.txt)
- TensorBoard: `logs/train_log/` (if enabled; minimal logging for sanity run)
- Checkpoint dir: `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/` (ckpt_00000.pth, ckpt_00002.pth from sanity run; see [logs/milestone_b_stop_conditions.txt:130–134](logs/milestone_b_stop_conditions.txt#L130-L134))

**Logging mechanism:** Open3D-ML's `SemanticSegmentation.save_logs()` (called once per epoch). Custom subclass `_SanityCapturer` in [tools/sanity_train_check.py:57–77](tools/sanity_train_check.py#L57-L77) accumulates per-step losses across epochs by overriding `save_logs()`.

---

## 6. State of Analysis Artifacts

### Intensity analysis directory

**Location:** `logs/raw_intensity_analysis/`

**Exists:** Yes, with full multi-stage analysis

**Contents:**

- `reports/` – markdown and PDF reports for stages 1–5
  - `stage1_global_raw_id_report.md` – global intensity by raw class ID
  - `stage2_sequence_frame_report.md` – per-sequence and per-frame breakdowns
  - `stage3_local_contrast_report.md` – local neighborhood contrast analysis
  - `stage4_distance_report.md` – intensity signal decay with distance
  - `stage5_benchmark_report.md` – ROC/PR AUC for lane classification tasks
  - `final/intensity_notes.md` – consolidated findings and takeaways
  - `final/raw_intensity_report.pdf` – full PDF report
- `tables/` – CSV data tables for all stages
- `plots/` – PNG plots per stage

**Key finding:** Raw intensity is useful for lane vs road (AUC 0.705), but weak for lane vs all road paint (AUC 0.585). Keep intensity as a feature but do not rely on it alone. See [logs/raw_intensity_analysis/reports/final/intensity_notes.md](logs/raw_intensity_analysis/reports/final/intensity_notes.md).

### Existing baselines/checkpoints

**Sanity-run checkpoints:**

- `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/ckpt_00000.pth` (epoch 0)
- `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/ckpt_00002.pth` (epoch 2)
- Status: May be auto-resumed by Open3D pipeline if not explicitly disabled. See [logs/milestone_b_stop_conditions.txt:130–134](logs/milestone_b_stop_conditions.txt#L130-L134).

Milestone C explicitly disables resume (`ckpt_path: None`, `is_resume: false`) inside [tools/train_milestone_c.py](tools/train_milestone_c.py), so these sanity checkpoints are not used by C0 unless that behavior is deliberately changed.

**Current best headline metrics:**

- Loss (sanity run): 1.080 at final iteration (ratio to initial: 0.95, stable)
- Batch-size-1 medium C0: best lane F1 `0.322776` at epoch 7; final lane F1 `0.277103` at epoch 10
- Batch-size-2 medium C0: best lane F1 `0.311518` at epoch 4; final lane F1 `0.187700` at epoch 10, with lane over-prediction
- Official full C0 baseline: best lane F1 `0.473696` and best lane IoU `0.310355` at epoch 18; final epoch lane F1 `0.418545` and lane IoU `0.264658`

---

## 7. Constraints and Conventions

### Project-specific constraints

**Milestone status:** Currently in Milestone C after completed C0 full baseline

- Milestone A: complete (single-sample verification)
- Milestone B: complete (multi-sequence validation, split frozen, sanity run passed)
- Milestone C: training entrypoint, validation metrics, server setup, dataset transfer, class-weight verification, sampler diagnosis, random-sampler smoke tests, two 10-epoch medium C0 runs, speed benchmarks, and the official 30-epoch C0 baseline are complete. C1-C4 improvement runs are still pending.

**Memory/compute:**

- Server GPU currently verified as `NVIDIA A100 80GB PCIe MIG 3g.40gb`.
- OOM was observed locally at `num_points: 16384` during Day 6, but the server full-model smoke, both 10-epoch medium runs, and the official full C0 run completed successfully at `num_points: 16384`.
- Official C0 runtime settings are now empirically used and validated: `batch_size=1`, `val_batch_size=1`, `num_workers=0`, `pin_memory=false`, `SemSegRandomSampler`, `steps_per_epoch_train=4640`, `steps_per_epoch_valid=720`.
- Official full C0 runtime was about `41h 00m 31s` for 30 epochs.
- Checkpoint cadence was patched so periodic saves use human epoch numbers from `cfg.save_ckpt_freq`; do not use `steps_per_epoch_train: 1` / `steps_per_epoch_valid: 1` outside smoke tests.

**Reproducibility:**

- Frozen sequence-level train/val/test split: [configs/splits/train.txt, val.txt, test.txt](configs/splits/)
- Measured training statistics: [logs/milestone_b_training_statistics.json](logs/milestone_b_training_statistics.json)
- Sanity config snapshot: [logs/milestone_b_sanity_config_snapshot.yml](logs/milestone_b_sanity_config_snapshot.yml)
- Official C0 run directory: [logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1](logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1)
- Official C0 result report: [logs/milestone_c/reports/c0_full_baseline_results.md](logs/milestone_c/reports/c0_full_baseline_results.md)
- Official C0 selected checkpoint: [logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints/ckpt_epoch_00018.pth](logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints/ckpt_epoch_00018.pth)
- Official C0 final checkpoint: [logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints/ckpt_epoch_00030.pth](logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/checkpoints/ckpt_epoch_00030.pth)

**Real training guard:** Local config includes `real_training_allowed: false` ([configs/randlanet_pandaset_ff_lane3.yml:63](configs/randlanet_pandaset_ff_lane3.yml#L63)) – a sentinel, not enforced by Open3D. Server run config snapshots are the authority for completed long runs.

### Code style/patterns

**Language:** Python 3.12 (in local venv `panda/`)

**Type annotations:** Used throughout (e.g., `dict[str, int]`, `list[float] | None`) – modern Python 3.10+ union syntax [src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py:1–2](src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py#L1-L2)

**Dataclasses:** Used for metadata structures (e.g., `SampleMetadata` in [src/thesis_pipeline/adapters/pandaset_ff_lane3.py:27–32](src/thesis_pipeline/adapters/pandaset_ff_lane3.py#L27-L32))

**Logging:** Explicit status strings (e.g., `script_status PASS/FAIL`, `sanity_result INSTABILITY`) printed to report files for machine-readable verification

**Path handling:** `pathlib.Path` consistently used; canonical paths defined in [src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py:16–21](src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py#L16-L21)

**JSON/YAML:** Standard library `json` and `yaml` for config/artifact I/O; measured stats stored in JSON for later validation

---

## 8. Open TODOs and Known Issues

### TODOs in code (grep-found)

**No grep results for TODO/FIXME/XXX in src/ directly**, but the following carry-forward items are documented in stop conditions:

**[logs/milestone_b_stop_conditions.txt:115–137](logs/milestone_b_stop_conditions.txt#L115-L137) – Explicit carry-forward items for Milestone C:**

1. **Restore and revalidate `num_points: 16384`**
   - Status: done for server C0 execution, including official full C0.
   - Current config uses baseline intent of 16384.
   - Server full-model smoke, 10-epoch medium runs, and the official 30-epoch C0 run completed successfully at this value.
   - File: [configs/randlanet_pandaset_ff_lane3.yml:28](configs/randlanet_pandaset_ff_lane3.yml#L28)

2. **Implement lane-aware patch oversampling**
   - Global heuristic E = num_points × p_lane / 64 = 1.8 at baseline (below 2.0 threshold)
   - Either implement oversampling or measure per-patch lane-point survival post grid-subsampling with direct evidence
   - Status: deferred to C2; C0 intentionally uses `SemSegRandomSampler` so the baseline is clean.
   - Reference: [logs/milestone_b_stop_conditions.txt:58–66](logs/milestone_b_stop_conditions.txt#L58-L66)

3. **Fix dual-writer race in `tools/sanity_train_check.py`**
   - Script writes to REPORT_FILE via `write_text()` while shell redirect '>' also targets same file
   - Results in duplicated/padded content (cosmetic, does not affect PASS/FAIL)
   - Solution: drop stdout prints or redirect to separate channel
   - File: [tools/sanity_train_check.py:84–85](tools/sanity_train_check.py#L84-L85)

4. **Clean duplicate `num_workers: 0` keys**
   - Status: done in the live config; only one `num_workers: 0` key remains.
   - File: [configs/randlanet_pandaset_ff_lane3.yml](configs/randlanet_pandaset_ff_lane3.yml)

5. **Decide on sanity-run checkpoints**
   - Status: effectively resolved for Milestone C.
   - `tools/train_milestone_c.py` sets `ckpt_path=None` and `is_resume=False`, so C0 starts fresh.
   - The old Day 6 checkpoint files may be retained as historical sanity artifacts.

6. **Finalize class-weight policy**
   - Three variants measured and documented in [logs/milestone_b_training_statistics.json:26–89](logs/milestone_b_training_statistics.json#L26-L89):
     - `raw_inverse_frequency` (direct CE weights)
     - `sqrt_inverse_frequency` (downweighted strong imbalance)
     - `open3d_native_from_measured_counts` (measured counts, Open3D transforms them; used in sanity run)
   - Server check confirmed the third variant is actually active in Open3D `CrossEntropyLoss`, with effective weights `road=2.3753`, `lane=36.9864`, `other=1.6341`.
   - Status: use `open3d_native_from_measured_counts` for official C0 unless C0 collapses and a later ablation intentionally changes weights.

7. **Official full C0 baseline**
   - Status: complete.
   - Run: `logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/`.
   - Completed config: `SemSegRandomSampler`, `batch_size=1`, `val_batch_size=1`, `num_workers=0`, `pin_memory=false`, `steps_per_epoch_train=4640`, `steps_per_epoch_valid=720`, 30 epochs.
   - Runtime: about `41h 00m 31s`.
   - Selected checkpoint: epoch 18, lane F1 `0.473696`, lane IoU `0.310355`, mIoU `0.703805`.

### Known limitations / README notes

**[README.md](README.md) – Milestone C state and notes:**

- This repo has completed a **mechanical sanity pass**, not a final experiment campaign
- No performance claims should be inferred from Milestone B sanity run
- Successful Day 6 run does NOT prove final model performance
- Milestone C server smoke runs prove training/validation/checkpoint mechanics on the server, not final C0 quality
- Milestone C medium runs prove the C0 baseline learns and help choose stable runtime settings, but they are not the official full C0 result
- Official full C0 is now complete and should be used as the baseline reference for C1-C4 comparisons
- Live YAML is NOT the historical Day 6 run record; snapshot is at [logs/milestone_b_sanity_config_snapshot.yml](logs/milestone_b_sanity_config_snapshot.yml)
- Measured values are canonical in JSON artifacts, not YAML

**[MILESTONE_B_DAY6_PREP_AND_STATUS.md:373–388](MILESTONE_B_DAY6_PREP_AND_STATUS.md#L373-L388) – Day 6 readiness verdict:**

- Day 6 is prepared for a fresh detached sanity run (and has been executed successfully)
- Script forces non-resume behavior, writes incremental reports, handles crashes robustly
- Local Open3D runtime transforms `dataset.cfg.class_weights` internally (non-standard semantics)

---

## Summary: C0 Complete, Ready For C1/C2

**Status:** The official full C0 baseline is complete. The repo is ready for the next Milestone C improvement experiment.

**Completed official C0 runtime settings:**

```text
sampler: SemSegRandomSampler
num_points: 16384
steps_per_epoch_train: 4640
steps_per_epoch_valid: 720
batch_size: 1
val_batch_size: 1
num_workers: 0
pin_memory: false
device: cuda
seed: 42
```

**What has been de-risked by C0:**

1. `num_points: 16384` works on the server for full-model smoke and medium runs.
2. Open3D class weights are confirmed active in the actual CE loss.
3. `SemSegRandomSampler` works and avoids spatial-sampler eager initialization.
4. `num_workers=0` is stable; `num_workers>0` segfaults and should not be used for C0.
5. `pin_memory=false` is faster than pinned memory in the zero-worker setup.
6. Batch size 1 gives better lane balance than batch size 2 in the realistic medium comparison.
7. Checkpoints, validation JSON/CSV, confusion matrices, and stdout/training logs are written correctly.
8. The full 30-epoch C0 baseline learns lane and provides a non-collapsed reference for ablations.

**Key artifacts for downstream Claude:**

- [configs/randlanet_pandaset_ff_lane3.yml](configs/randlanet_pandaset_ff_lane3.yml) – live config
- [logs/milestone_b_training_statistics.json](logs/milestone_b_training_statistics.json) – canonical measured stats
- [logs/milestone_b_sanity_train_report.txt](logs/milestone_b_sanity_train_report.txt) – sanity run outcome
- [logs/raw_intensity_analysis/reports/final/intensity_notes.md](logs/raw_intensity_analysis/reports/final/intensity_notes.md) – intensity signal findings
- [logs/milestone_b_stop_conditions.txt](logs/milestone_b_stop_conditions.txt) – explicit carry-forward items
- [logs/milestone_c/reports/c0_sampler_and_server_readiness.md](logs/milestone_c/reports/c0_sampler_and_server_readiness.md) – server readiness, sampler decision, class-weight verification
- [logs/milestone_c/reports/c0_medium_runs_and_speed_benchmarks.md](logs/milestone_c/reports/c0_medium_runs_and_speed_benchmarks.md) – medium-run metrics and runtime-setting decision
- [logs/milestone_c/reports/c0_full_baseline_results.md](logs/milestone_c/reports/c0_full_baseline_results.md) – official full C0 result and interpretation
- [logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/plots/run_summary.md](logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/plots/run_summary.md) – generated C0 summary table
- [docs/milestone_c/README.md](docs/milestone_c/README.md) – concise Milestone C orientation

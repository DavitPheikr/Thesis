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

logs/                             # Artifacts, statistics, sanity-run reports, analysis
logs/raw_intensity_analysis/      # Multi-stage intensity signal analysis
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

### Training entrypoint

**Location:** `tools/sanity_train_check.py` (Day 6 sanity run; Milestone C full training path not yet defined)

**Exact command (for sanity run):**

```bash
./panda/bin/python -u tools/sanity_train_check.py > logs/milestone_b_sanity_train_report.txt 2>&1
```

**Framework:** Open3D-ML (torch backend)

- Pipeline class: `open3d._ml3d.torch.pipelines.SemanticSegmentation`
- Training method: `pipeline.run_train()`
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

**Batch size:** `1` (single-sample batches; conservative for memory) [configs/randlanet_pandaset_ff_lane3.yml:46](configs/randlanet_pandaset_ff_lane3.yml#L46)

**Num workers:** `0` (all in main process) [configs/randlanet_pandaset_ff_lane3.yml:47](configs/randlanet_pandaset_ff_lane3.yml#L47)

**Num epochs:** `2` (for sanity run only; Milestone C config not yet finalized) [configs/randlanet_pandaset_ff_lane3.yml:50](configs/randlanet_pandaset_ff_lane3.yml#L50)

**Steps per epoch:**

- Train: `5` (for sanity run)
- Valid: `2` (for sanity run)
  [configs/randlanet_pandaset_ff_lane3.yml:19–20](configs/randlanet_pandaset_ff_lane3.yml#L19-L20)

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

No final C0 performance metrics have been reported yet.

**Sanity-run loss trajectory:** Captured in [logs/milestone_b_sanity_train_report.txt](logs/milestone_b_sanity_train_report.txt)

**Milestone C metric plumbing:** `tools/train_milestone_c.py` now writes per-epoch validation artifacts for mIoU, per-class IoU, precision, recall, F1, lane recall by distance bucket, confusion matrix, validation wall-clock time, and validation peak GPU memory.

**Server smoke evidence:** random-sampler tiny and full-model one-step runs completed training, validation, and checkpoint writing. These are readiness checks, not final performance baselines.

### Confusion matrix and per-class breakdown

**Not yet computed for a final C0 run.** The Milestone C validation path can now write active-class confusion matrices, but the completed random-sampler smoke runs are too small to interpret as model performance.

**Expected metrics (to be computed in Milestone C):**

- mIoU (mean Intersection-over-Union)
- Per-class IoU (separately for road, lane, other)
- Confusion matrix (road vs lane vs other)
- Per-class precision/recall

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

**Current best headline metric:**

- Loss (sanity run): 1.080 at final iteration (ratio to initial: 0.95, stable)
- IoU/per-class metrics: Not yet measured from full training

---

## 7. Constraints and Conventions

### Project-specific constraints

**Milestone status:** Currently in Milestone C server-ready pre-C0 state

- Milestone A: complete (single-sample verification)
- Milestone B: complete (multi-sequence validation, split frozen, sanity run passed)
- Milestone C: training entrypoint, validation metrics, server setup, dataset transfer, class-weight verification, sampler diagnosis, and random-sampler smoke tests are complete; final C0 baseline still pending.

**Memory/compute:**

- Server GPU currently verified as `NVIDIA A100 80GB PCIe MIG 3g.40gb`.
- OOM was observed locally at `num_points: 16384` during Day 6; the server full-model random-sampler one-step smoke at `num_points: 16384` completed successfully.
- **Carry-forward:** before long C0, choose non-smoke train/validation step counts and inspect checkpoint cadence. Do not use `steps_per_epoch_train: 1` / `steps_per_epoch_valid: 1` outside smoke tests.

**Reproducibility:**

- Frozen sequence-level train/val/test split: [configs/splits/train.txt, val.txt, test.txt](configs/splits/)
- Measured training statistics: [logs/milestone_b_training_statistics.json](logs/milestone_b_training_statistics.json)
- Sanity config snapshot: [logs/milestone_b_sanity_config_snapshot.yml](logs/milestone_b_sanity_config_snapshot.yml)

**Real training guard:** Config includes `real_training_allowed: false` ([configs/randlanet_pandaset_ff_lane3.yml:63](configs/randlanet_pandaset_ff_lane3.yml#L63)) – a sentinel, not enforced by Open3D (can be set to true once Milestone C full training is approved)

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
   - Current config uses baseline intent of 16384, but sanity run (and 4 failed Day 6 attempts) used/failed at this value
   - Must investigate root cause (batch size, num_workers, num_neighbors, sub_sampling_ratio, model dims) rather than lowering permanently
   - File: [configs/randlanet_pandaset_ff_lane3.yml:28](configs/randlanet_pandaset_ff_lane3.yml#L28)

2. **Implement lane-aware patch oversampling**
   - Global heuristic E = num_points × p_lane / 64 = 1.8 at baseline (below 2.0 threshold)
   - Either implement oversampling or measure per-patch lane-point survival post grid-subsampling with direct evidence
   - Reference: [logs/milestone_b_stop_conditions.txt:58–66](logs/milestone_b_stop_conditions.txt#L58-L66)

3. **Fix dual-writer race in `tools/sanity_train_check.py`**
   - Script writes to REPORT_FILE via `write_text()` while shell redirect '>' also targets same file
   - Results in duplicated/padded content (cosmetic, does not affect PASS/FAIL)
   - Solution: drop stdout prints or redirect to separate channel
   - File: [tools/sanity_train_check.py:84–85](tools/sanity_train_check.py#L84-L85)

4. **Clean duplicate `num_workers: 0` keys**
   - Config currently declares the key three times; YAML resolves to last value
   - File: [configs/randlanet_pandaset_ff_lane3.yml](configs/randlanet_pandaset_ff_lane3.yml)

5. **Decide on sanity-run checkpoints**
   - `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/ckpt_00000.pth` and `ckpt_00002.pth` from Day 6
   - Open3D defaults `is_resume=True`, so Milestone C training will auto-resume unless explicitly disabled
   - Decide: retain for warm-start, or archive

6. **Finalize class-weight policy**
   - Three variants measured and documented in [logs/milestone_b_training_statistics.json:26–89](logs/milestone_b_training_statistics.json#L26-L89):
     - `raw_inverse_frequency` (direct CE weights)
     - `sqrt_inverse_frequency` (downweighted strong imbalance)
     - `open3d_native_from_measured_counts` (measured counts, Open3D transforms them; used in sanity run)
   - Server check confirmed the third variant is actually active in Open3D `CrossEntropyLoss`, with effective weights `road=2.3753`, `lane=36.9864`, `other=1.6341`.

### Known limitations / README notes

**[README.md](README.md) – Milestone C state and notes:**

- This repo has completed a **mechanical sanity pass**, not a final experiment campaign
- No performance claims should be inferred from Milestone B sanity run
- Successful Day 6 run does NOT prove final model performance
- Milestone C server smoke runs prove training/validation/checkpoint mechanics on the server, not final C0 quality
- Live YAML is NOT the historical Day 6 run record; snapshot is at [logs/milestone_b_sanity_config_snapshot.yml](logs/milestone_b_sanity_config_snapshot.yml)
- Measured values are canonical in JSON artifacts, not YAML

**[MILESTONE_B_DAY6_PREP_AND_STATUS.md:373–388](MILESTONE_B_DAY6_PREP_AND_STATUS.md#L373-L388) – Day 6 readiness verdict:**

- Day 6 is prepared for a fresh detached sanity run (and has been executed successfully)
- Script forces non-resume behavior, writes incremental reports, handles crashes robustly
- Local Open3D runtime transforms `dataset.cfg.class_weights` internally (non-standard semantics)

---

## Summary: Ready for Milestone C?

**Status:** Yes, ready for a non-smoke C0 pilot with `SemSegRandomSampler`.

**Prerequisites before launching Milestone C:**

1. Restore `num_points: 16384` in YAML and revalidate the pipeline (target or investigate OOM)
2. Decide on lane oversampling strategy with evidence
3. Finalize class-weight policy (all variants measured, only one tested)
4. Clean up YAML duplicate keys and cosmetic issues
5. Handle sanity checkpoints explicitly (resume or archive)

**Key artifacts for downstream Claude:**

- [configs/randlanet_pandaset_ff_lane3.yml](configs/randlanet_pandaset_ff_lane3.yml) – live config
- [logs/milestone_b_training_statistics.json](logs/milestone_b_training_statistics.json) – canonical measured stats
- [logs/milestone_b_sanity_train_report.txt](logs/milestone_b_sanity_train_report.txt) – sanity run outcome
- [logs/raw_intensity_analysis/reports/final/intensity_notes.md](logs/raw_intensity_analysis/reports/final/intensity_notes.md) – intensity signal findings
- [logs/milestone_b_stop_conditions.txt](logs/milestone_b_stop_conditions.txt) – explicit carry-forward items

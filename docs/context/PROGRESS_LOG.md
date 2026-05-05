# PROGRESS_LOG

## Milestone C Server And Sampler Update - 2026-05-05

**What this update did**: recorded the current Milestone C server state and resolved the C0 sampler decision after GPU smoke tests and a bounded spatial-sampler benchmark.

**Facts confirmed**:
- Server workspace is `/home/coder/project`.
- Server environment is Conda env `panda312`, activated with `source envStart.sh`.
- Server GPU is visible to PyTorch as `NVIDIA A100 80GB PCIe MIG 3g.40gb`.
- Correct dataset source is Kaggle `pz19930809/pandaset`.
- Server dataset root is `/home/coder/project/pandaset/PandaSet`.
- PandaSet structure and access checks passed for sequence `001`, frame `0`, forward sensor `1`.
- Dataset class split lengths are `training=4640`, `validation=720`, `test=720`.
- Server loss check confirmed class weights are active in Open3D CE loss:
  ```text
  road  = 2.3753318786621094
  lane  = 36.98638153076172
  other = 1.6340690851211548
  ```
- `SemSegSpatiallyRegularSampler` benchmark on one 80-frame sequence took `80.982s`, estimating about `1.30h` for train sampler initialization and roughly `12min` for validation initialization.
- `SemSegRandomSampler` completed both tiny and full-model one-step GPU smoke tests in about `6s`, including validation and checkpoint writing.

**Decision**:
- Use `SemSegRandomSampler` for C0.
- Defer `SemSegSpatiallyRegularSampler` because it is class-blind and imposes a large eager CPU preprocessing cost before epoch 1.
- Keep the planned C2 lane-aware sampler as the actual sampling contribution for the rare lane class.

**New/updated reports**:
- `logs/milestone_c/reports/c0_sampler_and_server_readiness.md`
- `logs/milestone_c/reports/class_weight_decision.md`
- `docs/milestone_c/README.md`
- `docs/milestone_c_option_a_execution_plan.md`
- `logs/milestone_c/reports/c0_prep_summary.md`

## Day 1

**What this day did**: Day 1 verifies that the existing local project environment is the one the thesis pipeline should use, and that the core runtime foundation is still intact before any dataset or adapter work proceeds.

**Steps completed**:
- Step 1: Activated the existing `panda` virtual environment and verified the Python executable and version.
  Actual observed output:
  ```text
  /home/pheikara/University/Y3S2/Thesis/Pipeline/panda/bin/python
  Python 3.12.3
  ```
- Step 2: Created `logs/day1_environment_report.txt` and recorded the current local runtime and platform facts.
  Actual observed output:
  ```text
  project_root /home/pheikara/University/Y3S2/Thesis/Pipeline
  python_executable /home/pheikara/University/Y3S2/Thesis/Pipeline/panda/bin/python
  python_version 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]
  platform Linux-6.17.0-20-generic-x86_64-with-glibc2.39
  machine x86_64
  system Linux
  ```
- Step 3: Created `logs/dataset_root.txt` and verified that the canonical dataset root exists.
  Actual observed output:
  ```text
  dataset_root /home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset/PandaSet
  exists True
  ```
- Step 4: Created `tools/check_versions.py` and verified the core runtime versions plus Open3D-ML torch import.
  Actual observed output:
  ```text
  python 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]
  torch 2.2.2+cu121
  open3d 0.19.0
  numpy 1.26.4
  pandas 3.0.2
  cuda_available True
  ml_torch_ok True
  ```
- Step 5: Created `tools/check_imports.py` and verified the focused Open3D, RandLA-Net, PandaSet, and ego-helper imports.
  Actual observed output:
  ```text
  python 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]
  torch 2.2.2+cu121
  cuda_available True
  open3d 0.19.0
  ml_torch_ok True
  randlanet_ok True
  pandaset_ok True
  ego_helper_exists True
  pandaset_module_path /home/pheikara/University/Y3S2/Thesis/Pipeline/panda/lib/python3.12/site-packages/pandaset/__init__.py
  ```
- Step 6: Created `tools/check_pandaset_devkit_origin.py` and verified the local patched devkit source plus reinstall provenance.
  Actual observed output:
  ```text
  pandaset_module_path /home/pheikara/University/Y3S2/Thesis/Pipeline/panda/lib/python3.12/site-packages/pandaset/__init__.py
  contains_project_root True
  local_patched_source_exists True
  preserve_patch_reinstall_command pip install --no-deps ./pandaset-devkit/python
  ```

**Local facts confirmed**:
- Active Python executable confirmed as `/home/pheikara/University/Y3S2/Thesis/Pipeline/panda/bin/python`.
- Python version confirmed as `3.12.3`.
- Project root confirmed as `/home/pheikara/University/Y3S2/Thesis/Pipeline`.
- Platform confirmed as `Linux-6.17.0-20-generic-x86_64-with-glibc2.39`.
- Machine architecture confirmed as `x86_64`.
- System confirmed as `Linux`.
- Canonical dataset root confirmed as `/home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset/PandaSet`.
- Dataset root existence confirmed as `True`.
- PyTorch version confirmed as `2.2.2+cu121`.
- Open3D version confirmed as `0.19.0`.
- NumPy version confirmed as `1.26.4`.
- pandas version confirmed as `3.0.2`.
- `torch.cuda.is_available()` confirmed as `True`.
- `open3d.ml.torch` import confirmed by `ml_torch_ok True`.
- `RandLANet` import confirmed by `randlanet_ok True`.
- PandaSet import confirmed by `pandaset_ok True`.
- `geometry.lidar_points_to_ego` confirmed present by `ego_helper_exists True`.
- Installed `pandaset` module path confirmed as `/home/pheikara/University/Y3S2/Thesis/Pipeline/panda/lib/python3.12/site-packages/pandaset/__init__.py`.
- Patched PandaSet source path `./pandaset-devkit/python` confirmed present.
- Safe devkit reinstall command confirmed as `pip install --no-deps ./pandaset-devkit/python`.

**Provisional items still open**:
- Raw semantic ID mapping — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- Intensity preprocessing statistics — **[SET AFTER LOCAL VERIFICATION: compute clipping percentiles and global mean/std on the training split after front-only filtering]**
- Measured class counts — **[SET AFTER LOCAL VERIFICATION: count active class frequencies on the training split after remap]**
- Loss weights — **[SET AFTER LOCAL VERIFICATION: define and document the transform from measured class counts to loss weights]**
- Sparse-class-safe configuration values — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- Exact sequence split membership — **[SET AFTER LOCAL VERIFICATION: freeze exact 58/9/9 IDs after auditing the semseg-enabled sequence pool for lane presence and split diversity]**
- Exact environment choices — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- Exact API behavior — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**

**Deviations from the guide**: None.

**Day stop condition**: Satisfied. The active environment is confirmed as `panda`, the core stack imports successfully, the canonical dataset root file exists and points to the real dataset root, and the patched devkit setup is confirmed well enough to proceed to Day 2.

## Day 2

**What this day did**: Day 2 verifies that the local PandaSet copy can be read in the way the thesis needs, that the forward-facing LiDAR selection is locally usable, and that semantic labels remain aligned after filtering.

**Steps completed**:
- Step 1: Attempted `tools/precheck_sequence_and_sensor.py`.
  Actual observed output:
  ```text
  Traceback (most recent call last):
  sequence_count 103
  sequence_head ['001', '002', '003', '004', '005', '006', '008', '011', '012', '013']
    File "/home/pheikara/University/Y3S2/Thesis/Pipeline/tools/precheck_sequence_and_sensor.py", line 22, in <module>
      raw_frame_keys = sorted(seq.lidar.data.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
                              ^^^^^^^^^^^^^^^^^^^
  AttributeError: 'list' object has no attribute 'keys'
  ```
- Step 1 rerun after local compatibility fix: reorganized the script logic into `src/thesis_pipeline/` and updated the precheck to support list-backed `seq.lidar.data`.
  Actual observed output:
  ```text
  sequence_count 103
  sequence_head ['001', '002', '003', '004', '005', '006', '008', '011', '012', '013']
  chosen_seq 001
  chosen_frame 0
  all_points 169171
  all_columns ['x', 'y', 'z', 'i', 't', 'd']
  all_unique_sensor_ids [0, 1]
  front_points 62285
  front_unique_sensor_ids [1]
  forward_sensor_candidate 1
  forward_sensor_usable True
  ```
- Step 2: Created `tools/check_pandaset_frame_access.py` and verified direct access to the chosen LiDAR frame.
  Actual observed output:
  ```text
  chosen_seq 001
  chosen_frame 0
  frame_type <class 'pandas.DataFrame'>
  frame_rows 169171
  frame_columns ['x', 'y', 'z', 'i', 't', 'd']
  ```
- Step 3: Created `tools/check_pandaset_semseg_access.py` and verified semantic-segmentation access on the chosen sequence.
  Actual observed output:
  ```text
  chosen_seq 001
  semseg_loaded True
  classes_type <class 'dict'>
  classes_sample [('1', 'Smoke'), ('2', 'Exhaust'), ('3', 'Spray or rain'), ('4', 'Reflection'), ('5', 'Vegetation'), ('6', 'Ground'), ('7', 'Road'), ('8', 'Lane Line Marking'), ('9', 'Stop Line Marking'), ('10', 'Other Road Marking'), ('11', 'Sidewalk'), ('12', 'Driveway')]
  ```
- Step 4: Created `tools/check_raw_label_ids.py` and verified the locally relevant raw semantic IDs.
  Actual observed output:
  ```text
  classes_type <class 'dict'>
  id_1 Smoke
  id_2 Exhaust
  id_3 Spray or rain
  id_4 Reflection
  id_7 Road
  id_8 Lane Line Marking
  id_9 Stop Line Marking
  id_10 Other Road Marking
  ```
- Step 5: Created `tools/check_forward_only_filter.py` and verified forward-only filtering on the chosen frame.
  Actual observed output:
  ```text
  chosen_seq 001
  chosen_frame 0
  all_points 169171
  all_columns ['x', 'y', 'z', 'i', 't', 'd']
  all_unique_sensor_ids [0, 1]
  front_sensor 1
  front_points 62285
  front_unique_sensor_ids [1]
  ```
- Step 6: Created `tools/check_semseg_alignment.py` and verified semantic-label alignment after forward-only filtering.
  Actual observed output:
  ```text
  point_count 62285
  aligned_label_count 62285
  same_count True
  label_head [5, 5, 5, 5, 5]
  ```
- Step 7: Created `tools/check_intensity_field.py` and verified the local intensity column on the chosen forward-only frame.
  Actual observed output:
  ```text
  columns ['x', 'y', 'z', 'i', 't', 'd']
  intensity_column i
  intensity_head [14.0, 11.0, 12.0, 16.0, 14.0]
  intensity_min 0.0
  intensity_max 255.0
  ```

**Local facts confirmed**:
- PandaSet sequence count confirmed as `103`.
- Leading local sequence IDs confirmed as `['001', '002', '003', '004', '005', '006', '008', '011', '012', '013']`.
- Local `seq.lidar.data` container shape deviates from the guide assumption: it is `list`-backed at this step, not dict-backed.
- Chosen sequence confirmed as `001`.
- Chosen frame confirmed as `0`.
- Full-frame point count confirmed as `169171` on the chosen frame.
- Local LiDAR columns confirmed as `['x', 'y', 'z', 'i', 't', 'd']`.
- Local sensor IDs in the unfiltered frame confirmed as `[0, 1]`.
- Forward-only point count confirmed as `62285`.
- Forward-only sensor IDs confirmed as `[1]`.
- Forward sensor candidate `1` confirmed locally usable.
- Chosen LiDAR frame type confirmed as `pandas.DataFrame`.
- Chosen LiDAR frame row count confirmed as `169171`.
- Semantic segmentation access confirmed by `semseg_loaded True`.
- Local semseg class mapping type confirmed as `dict`.
- Local semseg class keys are string-typed in the sampled mapping.
- Raw semantic IDs locally verified as `id_4 Reflection`, `id_7 Road`, `id_8 Lane Line Marking`, `id_9 Stop Line Marking`, and `id_10 Other Road Marking`.
- Forward-only filtering reconfirmed in standalone script form with `front_sensor 1`.
- Forward-only point count reconfirmed as `62285`, strictly below the full-frame count `169171`.
- Semseg alignment after filtering confirmed exactly with `point_count 62285`, `aligned_label_count 62285`, and `same_count True`.
- Intensity column confirmed locally as `'i'`.
- Intensity range confirmed on sequence `001` frame `0` forward-only cloud as `0.0-255.0`.

**Provisional items still open**:
- Raw semantic ID mapping — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- Intensity preprocessing statistics — **[SET AFTER LOCAL VERIFICATION: compute clipping percentiles and global mean/std on the training split after front-only filtering]**
- Measured class counts — **[SET AFTER LOCAL VERIFICATION: count active class frequencies on the training split after remap]**
- Loss weights — **[SET AFTER LOCAL VERIFICATION: define and document the transform from measured class counts to loss weights]**
- Sparse-class-safe configuration values — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- Exact sequence split membership — **[SET AFTER LOCAL VERIFICATION: freeze exact 58/9/9 IDs after auditing the semseg-enabled sequence pool for lane presence and split diversity]**
- Exact environment choices — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- Exact API behavior — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**

**Deviations from the guide**: The original Day 2 Step 1 script assumed `seq.lidar.data.keys()` existed. Locally `seq.lidar.data` is list-backed, so the precheck was updated to support both dict-like and list-like devkit variants. The repo code layout was also reorganized so reusable logic now lives under `src/thesis_pipeline/`, while `tools/` remains as guide-compatible wrappers.

**Day stop condition**: Satisfied. The dataset is readable, semantic segmentation access is real, raw IDs are verified, the chosen sequence/frame/sensor are usable, forward-only filtering works, semseg alignment is preserved, and the intensity column is confirmed.

## Day 3

**What this day did**: Day 3 builds the first real adapter-style sample path so a chosen PandaSet frame becomes the intended `point / feat / label` contract for later model checks.

**Steps completed**:
- Steps 1-10 combined in one clean implementation pass: created `tools/inspect_pandaset_ff_lane3_sample.py` as the guide-facing entry point and moved the reusable adapter logic into `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`.
  Actual observed output:
  ```text
  chosen_seq 001
  chosen_frame 0
  forward_sensor 1
  intensity_column i
  point_shape (62285, 3)
  feat_shape (62285, 1)
  label_shape (62285,)
  label_unique [0, 1, 2, 3]
  lane_fraction 0.0025688367985871397
  ```

**Local facts confirmed**:
- The chosen real forward-only sample can be converted end to end into `point / feat / label`.
- Ego-local point tensor shape confirmed as `(62285, 3)`.
- Processed feature tensor shape confirmed as `(62285, 1)`.
- Label tensor shape confirmed as `(62285,)`.
- Remapped label values confirmed as a subset of `[0, 1, 2, 3]`.
- Lane fraction on sequence `001` frame `0` forward-only cloud measured as `0.0025688367985871397`.
- Local pose access path confirmed as `seq.lidar.poses[frame_idx]`, not `seq.poses[frame_idx]`.

**Provisional items still open**:
- Intensity preprocessing statistics — **[SET AFTER LOCAL VERIFICATION: compute clipping percentiles and global mean/std on the training split after front-only filtering]**
- Measured class counts — **[SET AFTER LOCAL VERIFICATION: count active class frequencies on the training split after remap]**
- Loss weights — **[SET AFTER LOCAL VERIFICATION: define and document the transform from measured class counts to loss weights]**
- Sparse-class-safe configuration values — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- Exact sequence split membership — **[SET AFTER LOCAL VERIFICATION: freeze exact 58/9/9 IDs after auditing the semseg-enabled sequence pool for lane presence and split diversity]**
- Exact environment choices — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- Exact API behavior — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**

**Deviations from the guide**: The guide described the Day 3 script as a standalone implementation, but the actual logic was placed in `src/thesis_pipeline/adapters/pandaset_ff_lane3.py` with `tools/inspect_pandaset_ff_lane3_sample.py` kept as a thin wrapper for a cleaner architecture. Also, the local ego-pose path had to use `seq.lidar.poses[frame_idx]` because `Sequence` does not expose `seq.poses` in this runtime. For the first structural remap, raw IDs `1-4` were treated as `ignore`, `7` as `road`, `8` as `lane`, and everything else as `other`.

**Day stop condition**: Satisfied. The adapter-inspection path runs end to end on a real frame and prints a valid `point / feat / label` contract.

## Day 4

**What this day did**: Day 4 turned the verified sample path into a guide-facing adapter module, added a minimal RandLA-Net config scaffold, and proved that the local adapter contract and Open3D model constructor are structurally compatible without starting any real training.

**Steps completed**:
- Step 1: Created the guide-facing `datasets/` package, adapter module, config file, and Day 4 build-check script paths.
  Actual observed output:
  ```text
  /home/pheikara/University/Y3S2/Thesis/Pipeline/datasets/__init__.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/datasets/pandaset_ff_lane3.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/configs/randlanet_pandaset_ff_lane3.yml
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/check_forward_only_filter.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/check_imports.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/check_intensity_field.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/check_pandaset_devkit_origin.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/check_pandaset_frame_access.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/check_pandaset_semseg_access.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/check_randlanet_build.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/check_raw_label_ids.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/check_semseg_alignment.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/check_versions.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/inspect_pandaset_ff_lane3_sample.py
  /home/pheikara/University/Y3S2/Thesis/Pipeline/tools/precheck_sequence_and_sensor.py
  ```
- Step 2: Created and validated `configs/randlanet_pandaset_ff_lane3.yml`.
  Actual observed output:
  ```text
  config_exists True
  top_keys ['dataset', 'model', 'pipeline']
  model_num_classes 3
  model_in_channels 4
  pipeline_device cuda
  ```
- Step 3 / Step 5: Verified that `datasets/pandaset_ff_lane3.py` exposes a callable sample-building entry point and returns a real sample dict.
  Actual observed output:
  ```text
  adapter_module_ok True
  point_shape (62285, 3)
  feat_shape (62285, 1)
  label_shape (62285,)
  label_unique [0, 1, 2, 3]
  ```
- Step 4 / Step 6: Ran the Day 4 build-check script and confirmed config parse, sample contract, and RandLA-Net construction.
  Actual observed output:
  ```text
  config_parse_ok True
  point_shape (62285, 3)
  feat_shape (62285, 1)
  label_shape (62285,)
  label_unique [0, 1, 2, 3]
  sample_contract_ok True
  randlanet_build_ok True
  ```
- Step 7 / Step 8: Verified the final structural checks and wrote explicit Day 4 no-training stop conditions.
  Actual observed output:
  ```text
  config_num_classes 3
  report_has_sample_contract_ok True
  report_has_randlanet_build_ok True
  stop_conditions_exists True
  Do not start real training during Day 4.
  
  Work does not move into real training unless all of the following are true:
  - the environment is stable enough
  - PandaSet access is verified
  - raw IDs are verified
  - forward-only filtering is verified
  - semseg alignment is verified
  - the adapter is working
  - the config placeholder is working
  - the model-build or contract check passes
  ```

**Local facts confirmed**:
- Guide-facing adapter module confirmed at `datasets/pandaset_ff_lane3.py`.
- Adapter entry point `build_one_sample()` confirmed callable and returns a real sample dict.
- Day 4 config parses successfully with top-level keys `dataset`, `model`, and `pipeline`.
- `num_classes` confirmed as `3` in the Day 4 config scaffold.
- Local RandLA-Net constructor expects `in_channels` to match concatenated `xyz + feat`, so the correct local value for the current contract is `4`.
- RandLA-Net can be instantiated locally against the current scaffold and sample contract.
- `logs/day4_stop_conditions.txt` exists and explicitly preserves the no-training boundary.

**Provisional items still open**:
- Intensity preprocessing statistics — **[SET AFTER LOCAL VERIFICATION: compute clipping percentiles and global mean/std on the training split after front-only filtering]**
- Measured class counts — **[SET AFTER LOCAL VERIFICATION: count active class frequencies on the training split after remap]**
- Loss weights — **[SET AFTER LOCAL VERIFICATION: define and document the transform from measured class counts to loss weights]**
- Sparse-class-safe configuration values — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- Exact sequence split membership — **[SET AFTER LOCAL VERIFICATION: freeze exact 58/9/9 IDs after auditing the semseg-enabled sequence pool for lane presence and split diversity]**
- Exact environment choices — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**
- Exact API behavior — **[PROVISIONAL UNTIL LOCALLY VERIFIED]**

**Deviations from the guide**: The required `datasets/pandaset_ff_lane3.py` and `tools/check_randlanet_build.py` files were created as requested, but the underlying reusable logic remains in `src/thesis_pipeline/` for cleaner maintenance. The Day 4 build-check script includes the required project-root `sys.path` insertion and an additional `src` path insertion because the clean package layout places reusable code under `src/`. Also, the local Open3D constructor required `in_channels: 4` for the current `xyz + one processed intensity feature` contract, not `1`.

**Day stop condition**: Satisfied. The adapter works in module form, the config placeholder parses, RandLA-Net constructs locally, and explicit stop conditions before real training are written down.

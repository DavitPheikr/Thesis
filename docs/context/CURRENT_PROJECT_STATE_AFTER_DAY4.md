# Current Project State After Day 4

## 1. Purpose of this document

This document records the actual state of the repository and the locally verified project state after the completed Day 1–4 work. It is a state snapshot, not a plan, not an execution guide, and not a new project specification.

Its purpose is to tell a future reader or coding agent:

- what the project currently means
- what has been implemented in this repository
- what was verified locally on this machine
- what local realities differed from earlier assumptions
- what artifacts now exist
- what is still provisional
- what exact boundary was reached at the end of Day 4

When this document conflicts with older planning assumptions, the code and logs in this repository take priority.

## 2. Project meaning that remains fixed

The project remains a point-wise semantic segmentation task on PandaSet LiDAR. The active semantic classes remain `road`, `lane`, and `other`, and the positive class `lane` still means **Lane Line Marking only**, not all painted road markings.

The baseline scope remains LiDAR-only, with forward-facing LiDAR only as the intended baseline sensor choice and ego-local coordinates as the intended baseline coordinate frame. The project still requires a custom adapter that produces `point / feat / label` samples from real PandaSet data.

Days 1–4 were governed by a strict no-real-training boundary. That boundary was preserved. The work stopped at environment verification, dataset verification, adapter/data-contract verification, config scaffold creation, model construction, and explicit stop conditions before training.

## 3. Confirmed local environment and system state

The following are confirmed local facts for this machine and this repository:

- Project root: `/home/pheikara/University/Y3S2/Thesis/Pipeline`
- Active virtual environment Python executable: `/home/pheikara/University/Y3S2/Thesis/Pipeline/panda/bin/python`
- Active virtual environment path: `/home/pheikara/University/Y3S2/Thesis/Pipeline/panda`
- Dataset root recorded in `logs/dataset_root.txt`: `/home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset/PandaSet`
- Python: `3.12.3`
- PyTorch: `2.2.2+cu121`
- Open3D: `0.19.0`
- NumPy: `1.26.4`
- pandas: `3.0.2`
- Platform string recorded in `logs/day1_environment_report.txt`: `Linux-6.17.0-20-generic-x86_64-with-glibc2.39`
- Machine architecture: `x86_64`
- System: `Linux`
- `open3d.ml.torch` import: verified successful
- `open3d.ml.torch.models.RandLANet` import: verified successful
- `torch.cuda.is_available()`: `True`
- `pandaset` import: verified successful
- `geometry.lidar_points_to_ego` existence: verified successful
- Installed `pandaset` module path: `/home/pheikara/University/Y3S2/Thesis/Pipeline/panda/lib/python3.12/site-packages/pandaset/__init__.py`

The current PandaSet devkit install is a local patched install from `./pandaset-devkit/python`, recorded in `requirements_working_panda.txt` as:

- `pandaset @ file:///home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset-devkit/python`

The critical local setup constraint is unchanged: this environment depends on the patched local PandaSet devkit source. Reinstalling `pandaset` from another source risks losing `.pkl` fallback support and breaking dataset loading on this local dataset copy. The preserved reinstall command recorded in the repo is:

```bash
pip install --no-deps ./pandaset-devkit/python
```

These are confirmed local truths for this machine. They must not be assumed to generalize automatically to another system, another Python environment, or another PandaSet installation.

## 4. Dataset and devkit reality discovered locally

The local PandaSet dataset root is `/home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset/PandaSet`, and the local sequence count discovered during Day 2 was `103`.

The Day 1–4 work used:

- chosen sequence: `001`
- chosen frame: `0`
- chosen forward sensor: `1`

The local dataset stores per-frame files as `.pkl`, not `.pkl.gz`. That mattered immediately because the unpatched devkit expected `.pkl.gz` only. `docs/context/PROJECT_SETUP_CONTEXT.md` records that before patching, `seq.load_lidar()` loaded `0` frames on this dataset copy. The local patch added fallback support so the devkit can read `.pkl` frames.

Confirmed local devkit/data behavior:

- `seq.load_lidar().load_semseg()` works in the current environment
- the top-level sanity check also confirmed `seq.load_lidar().load_cuboids().load_semseg()` on sequence `001`
- sequence `001` currently loads `80` LiDAR frames, `80` cuboid frames, and `80` semseg frames
- frame `0` is accessible by indexing
- the chosen LiDAR frame object is a `pandas.DataFrame`
- the chosen LiDAR frame row count before forward filtering is `169171`
- the chosen forward-only LiDAR frame row count is `62285`
- the LiDAR dataframe columns on the chosen frame are `['x', 'y', 'z', 'i', 't', 'd']`
- the local sensor ID column is `d`
- unfiltered sensor IDs on the chosen frame are `[0, 1]`
- forward-only sensor IDs on the chosen frame are `[1]`

Important corrections from local reality:

- Earlier generated scaffold logic assumed `seq.lidar.data` was dict-like and supported `.keys()`. On this local runtime it was observed as list-backed for Day 2 precheck purposes. This required a compatibility helper in `src/thesis_pipeline/core/pandaset_compat.py`.
- The current sequence object does not expose `seq.poses`. The working pose access path for LiDAR is `seq.lidar.poses[frame_idx]`.
- The local semseg class mapping was observed as a `dict` with string keys in at least one direct check.

`seq.load_lidar().load_semseg()` and index-based frame access are therefore part of the current verified local truth, but the exact underlying container shapes and helper paths are local implementation realities, not universal PandaSet truths.

## 5. Raw semantic label truth confirmed locally

The current local PandaSet devkit exposes semantic class meaning through `seq.semseg.classes`. On the chosen sequence, this mapping was successfully read and inspected during Day 2.

The following raw semantic IDs were verified locally:

- `4 -> Reflection`
- `7 -> Road`
- `8 -> Lane Line Marking`
- `9 -> Stop Line Marking`
- `10 -> Other Road Marking`

The Day 2 semseg access check also showed that the class mapping keys were string-typed in the sampled mapping, for example:

- `('7', 'Road')`
- `('8', 'Lane Line Marking')`
- `('9', 'Stop Line Marking')`
- `('10', 'Other Road Marking')`

The current practical remap used by the Day 3 / Day 4 adapter path is:

- raw IDs `1, 2, 3, 4` -> `0` (`ignore`)
- raw ID `7` -> `1` (`road`)
- raw ID `8` -> `2` (`lane`)
- all remaining raw IDs -> `3` (`other`)

This practical remap is the current implemented local adapter behavior. It is grounded in the verified local ID meanings for `7` and `8` and in a conservative ignore policy for `1–4`. It does not settle every broader label-policy question for later stages. In particular, this document does not treat the broader treatment of all non-lane painted markings, all artifact-like IDs, or future training-time loss-policy choices as globally finalized beyond the current implemented adapter path.

## 6. What Days 1–4 actually built

The following artifacts were created and are present in the repository.

Core records and reports:

- `docs/context/PROGRESS_LOG.md`
  Running record of the Day 1–4 work, including actual observed outputs and deviations.
- `docs/context/CODE_STRUCTURE.md`
  Short note describing the current organization pattern.
- `logs/dataset_root.txt`
  Canonical dataset root used by scripts.
- `logs/day1_environment_report.txt`
  Verified environment and import report.
- `logs/day2_dataset_report.txt`
  Verified dataset, semseg, filtering, alignment, and intensity report.
- `logs/day2_chosen_sequence.txt`
  Recorded chosen sequence ID (`001`).
- `logs/day2_chosen_frame.txt`
  Recorded chosen frame index (`0`).
- `logs/day2_forward_sensor.txt`
  Recorded chosen forward sensor (`1`).
- `logs/day3_adapter_report.txt`
  Output of the adapter inspection sample contract.
- `logs/day4_model_build_report.txt`
  Output of the Day 4 model/build check.
- `logs/day4_stop_conditions.txt`
  Explicit no-training boundary note.

Guide-facing scripts under `tools/`:

- `tools/check_versions.py`
  Thin wrapper for Day 1 version reporting.
- `tools/check_imports.py`
  Thin wrapper for Day 1 focused import checks.
- `tools/check_pandaset_devkit_origin.py`
  Thin wrapper for patched devkit provenance checks.
- `tools/precheck_sequence_and_sensor.py`
  Thin wrapper for Day 2 sequence/frame/sensor precheck.
- `tools/check_pandaset_frame_access.py`
  Thin wrapper for direct LiDAR frame access verification.
- `tools/check_pandaset_semseg_access.py`
  Thin wrapper for semseg access verification.
- `tools/check_raw_label_ids.py`
  Thin wrapper for raw semantic ID verification.
- `tools/check_forward_only_filter.py`
  Thin wrapper for forward-only filtering verification.
- `tools/check_semseg_alignment.py`
  Thin wrapper for index-preserving semseg alignment verification.
- `tools/check_intensity_field.py`
  Thin wrapper for intensity field verification.
- `tools/inspect_pandaset_ff_lane3_sample.py`
  Guide-facing Day 3 inspection entry point.
- `tools/check_randlanet_build.py`
  Day 4 config/sample/model structural compatibility check.

Reusable code under `src/thesis_pipeline/`:

- `src/thesis_pipeline/core/paths.py`
  Canonical path readers for logs and known project directories.
- `src/thesis_pipeline/core/pandaset_compat.py`
  Compatibility helper for frame-key discovery across dict-like and list-like PandaSet runtime behavior.
- `src/thesis_pipeline/checks/day1.py`
  Day 1 verification implementations used by wrappers.
- `src/thesis_pipeline/checks/day2.py`
  Day 2 verification implementations used by wrappers.
- `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`
  Reusable sample-building logic, remap logic, intensity scaling, and sample metadata for the current adapter path.

Guide-facing adapter and config artifacts:

- `datasets/__init__.py`
  Makes `datasets/` importable.
- `datasets/pandaset_ff_lane3.py`
  Canonical Day 4 guide-facing adapter module exposing `build_one_sample()`.
- `configs/randlanet_pandaset_ff_lane3.yml`
  Minimal RandLA-Net config scaffold used by the Day 4 build check.

## 7. Current code structure and architectural pattern

The current repository uses a two-layer pattern.

Reusable logic lives under `src/thesis_pipeline/...`. This includes:

- path readers
- PandaSet compatibility helpers
- Day 1 and Day 2 check implementations
- the reusable PandaSet forward-only lane-3 sample-building logic

Thin wrapper scripts live under `tools/`. These wrappers exist because the first-4-days execution guide refers to exact script paths under `tools/`, and those paths need to remain runnable from the project root. The wrappers mostly import one function from `src/thesis_pipeline/...` and execute it.

`datasets/pandaset_ff_lane3.py` still exists because the Day 4 guide requires a guide-facing adapter module importable as `datasets.pandaset_ff_lane3`. The Day 4 build-check script imports that exact module path and calls its `build_one_sample()` entry point.

`configs/randlanet_pandaset_ff_lane3.yml` still exists because the Day 4 guide requires a concrete config placeholder file at that exact path. The build-check script loads that exact YAML file.

`tools/check_randlanet_build.py` uses two `sys.path` insertions:

- project root insertion so `datasets.pandaset_ff_lane3` is importable
- `src` insertion so reusable `thesis_pipeline` modules are importable

This structure is therefore a deliberate compromise between guide compatibility and cleaner code organization:

- `src/thesis_pipeline/...` is the maintainable implementation layer
- `tools/...` is the stable CLI layer
- `datasets/...` and `configs/...` are retained because the Day 4 contract depends on those exact project-root paths

## 8. What was verified on each day

### Day 1

Day 1 was trying to prove that the local project environment was real, usable, and still matched the intended stack closely enough to proceed.

What actually passed:

- the active Python came from `panda/bin/python`
- Python version was `3.12.3`
- the project root and dataset root were recorded
- `torch`, `open3d`, `open3d.ml.torch`, `pandaset`, and `RandLANet` imported successfully
- CUDA was visible to Torch
- the local patched PandaSet source path existed
- the preserved reinstall command was recorded

Most important outputs:

- `ml_torch_ok True`
- `randlanet_ok True`
- `pandaset_ok True`
- `ego_helper_exists True`
- `cuda_available True`

Important local facts established:

- this exact environment is usable for the current baseline scaffold
- the devkit provenance matters and is local-patch-dependent

Important deviations from earlier assumptions:

- the locally verified Python version is `3.12.3`
- the locally verified pandas version is `3.0.2`
- those values now override older generic planning assumptions for this machine

### Day 2

Day 2 was trying to prove that PandaSet access, label meaning, forward-only filtering, and semseg alignment all worked on real local data.

What actually passed:

- local sequence count check
- chosen sequence/frame/sensor selection
- direct frame access
- semseg access
- raw ID verification
- forward-only filtering
- semseg alignment
- intensity-field verification

Most important outputs:

- `sequence_count 103`
- `chosen_seq 001`
- `chosen_frame 0`
- `front_points 62285`
- `front_unique_sensor_ids [1]`
- `same_count True`
- `intensity_column i`
- `intensity_min 0.0`
- `intensity_max 255.0`

Important local facts established:

- the current local forward-only baseline uses sensor ID `1`
- LiDAR frame columns are `x, y, z, i, t, d`
- semantic labels remain exactly aligned after forward-only filtering
- semseg class keys are string-typed in the sampled mapping

Important deviations from earlier assumptions:

- the original precheck assumption that `seq.lidar.data` had `.keys()` was false locally
- a list-safe compatibility path was required

### Day 3

Day 3 was trying to prove that one real PandaSet frame could be converted into the intended `point / feat / label` contract without training.

What actually passed:

- a full real-frame sample path executed end to end
- ego-local `point` was produced
- one processed intensity channel was produced
- remapped labels were produced
- shapes and unique labels were correct for the tested sample

Most important outputs:

- `point_shape (62285, 3)`
- `feat_shape (62285, 1)`
- `label_shape (62285,)`
- `label_unique [0, 1, 2, 3]`
- `lane_fraction 0.0025688367985871397`

Important local facts established:

- the current sample contract works on real data
- the local pose access path is `seq.lidar.poses[frame_idx]`

Important deviations from earlier assumptions:

- the runtime does not expose `seq.poses`
- the working pose path is LiDAR-sensor-specific

### Day 4

Day 4 was trying to prove that the verified adapter contract could be promoted into a guide-facing module, paired with a config scaffold, and used to instantiate RandLA-Net locally.

What actually passed:

- guide-facing `datasets/` adapter module creation
- minimal config scaffold creation
- adapter module entry-point verification
- config parse verification
- local RandLA-Net construction
- explicit stop-condition writing

Most important outputs:

- `config_parse_ok True`
- `point_shape (62285, 3)`
- `feat_shape (62285, 1)`
- `label_shape (62285,)`
- `label_unique [0, 1, 2, 3]`
- `sample_contract_ok True`
- `randlanet_build_ok True`

Important local facts established:

- the current adapter module and current config are structurally compatible with the local Open3D RandLA-Net constructor
- the correct local `in_channels` value for the current contract is `4`

Important deviations from earlier assumptions:

- `in_channels` is not `1` for the current contract even though there is only one processed intensity feature channel; Open3D concatenates `xyz` and `feat`, so the constructor must see `4`

## 9. The current adapter/data contract

The current actual sample contract is implemented in `datasets/pandaset_ff_lane3.py` and `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`.

Current meaning of the returned fields:

- `point`
  Ego-local `xyz` coordinates as a `float32` array produced by applying `geometry.lidar_points_to_ego(...)` to world-frame LiDAR positions from the chosen forward-only frame.
- `feat`
  One processed intensity channel as a `float32` array with shape `[N, 1]`. The current implementation uses frame-local min-max scaling.
- `label`
  Integer remapped semantic labels with the current encoding:
  - `0 = ignore`
  - `1 = road`
  - `2 = lane`
  - `3 = other`

Verified sample used for this contract:

- sequence: `001`
- frame: `0`
- sensor: forward-only `1`

Observed shapes on the verified sample:

- `point_shape (62285, 3)`
- `feat_shape (62285, 1)`
- `label_shape (62285,)`
- `label_unique [0, 1, 2, 3]`

Current remap used:

- `1, 2, 3, 4 -> 0`
- `7 -> 1`
- `8 -> 2`
- everything else -> `3`

Observed lane fraction on the verified sample:

- `0.0025688367985871397`

What is structurally working now:

- real local forward-only PandaSet frame loading
- index-preserving semseg alignment
- ego-local point conversion
- one-channel feature creation
- current remap into `{0,1,2,3}`

What is only a temporary Day 3 / Day 4 preprocessing choice:

- the current frame-local min-max intensity scaling

The code explicitly marks that preprocessing as provisional:

```python
# [PROVISIONAL: replace with training-split statistics before training]
```

## 10. The current model/build state

The current Day 4 config scaffold is `configs/randlanet_pandaset_ff_lane3.yml`. It currently contains:

- `model` section
- `dataset` section
- `pipeline` section
- `num_classes: 3`
- `ignored_label_inds: [0]`
- `in_channels: 4`
- provisional scaffold values:
  - `num_layers: 3`
  - `num_points: 16384`
  - `sub_sampling_ratio: [4, 4, 4]`
  - `grid_size: 0.04`
- `pipeline.device: cuda`
- `pipeline.real_training_allowed: false`

The important local discovery was `in_channels: 4`. This is not a contradiction of the “xyz plus one processed intensity channel” feature concept. It is a constructor-interface reality of the local Open3D RandLA-Net implementation, which concatenates `pc` and `feat`, so `xyz (3)` plus one feature channel yields an input width of `4`.

The Day 4 build check in `tools/check_randlanet_build.py` proves the following:

- the YAML config parses
- `datasets.pandaset_ff_lane3` imports
- the adapter entry point `build_one_sample()` exists
- one real sample can be built
- the sample has the expected structural shapes
- `label_unique` stays within `{0,1,2,3}`
- RandLA-Net can be instantiated locally with the current scaffold

The Day 4 build check does **not** prove:

- that the model trains successfully
- that dataloading for a full dataset split is complete
- that training hyperparameters are safe
- that performance is meaningful
- that GPU training throughput is acceptable
- that metrics, loss weighting, or class imbalance handling are correct

Model construction is proven. Training is not proven. Performance is completely unproven. No real training has begun.

## 11. Important local deviations from earlier assumptions

- Earlier assumption: the dataset would be readable through the stock PandaSet devkit.
  Actual local truth: this dataset uses `.pkl` frame files, and the local devkit needed a patch to support fallback from `.pkl.gz` to `.pkl`.
  Why it matters: without the patch, `seq.load_lidar()` previously loaded `0` frames on this dataset copy.

- Earlier assumption: `seq.lidar.data` would be dict-like and expose `.keys()`.
  Actual local truth: in the runtime behavior encountered during Day 2, it needed to be treated as list-backed.
  Why it matters: the original Day 2 precheck crashed until a compatibility helper was added.

- Earlier assumption: sequence pose access could use `seq.poses`.
  Actual local truth: the working pose path is `seq.lidar.poses[frame_idx]`.
  Why it matters: ego-local conversion depends on the actual pose access path.

- Earlier assumption: `in_channels` would be just the feature-channel width.
  Actual local truth: the local Open3D RandLA-Net path requires `in_channels: 4` for the current `xyz + one processed intensity feature` contract.
  Why it matters: the build check would fail if `in_channels` did not match the concatenated feature width expected by Open3D.

- Earlier assumption: generic planning documents could be trusted for exact environment details.
  Actual local truth: the verified runtime is Python `3.12.3`, pandas `3.0.2`, Torch `2.2.2+cu121`, Open3D `0.19.0`, NumPy `1.26.4`.
  Why it matters: future work should plan from these local facts, not from older generic environment guesses.

## 12. What is now solid and can be treated as current local truth

The following can now be treated as solid current local truth for this repository and machine unless later work disproves them:

- the project root is `/home/pheikara/University/Y3S2/Thesis/Pipeline`
- the active working environment is `panda`
- the canonical dataset root is `/home/pheikara/University/Y3S2/Thesis/Pipeline/pandaset/PandaSet`
- this dataset copy stores frames as `.pkl`
- the current environment depends on the patched local PandaSet devkit source
- the preserved reinstall command is `pip install --no-deps ./pandaset-devkit/python`
- the current environment imports `open3d.ml.torch`, `RandLANet`, and `pandaset`
- CUDA is visible to Torch on this machine
- the local PandaSet sequence count is `103`
- the Day 1–4 chosen sample is sequence `001`, frame `0`, forward sensor `1`
- forward-only filtering with sensor ID `1` is locally usable
- semseg alignment after forward-only filtering is preserved exactly on the verified sample
- the local LiDAR frame columns on the verified sample are `x, y, z, i, t, d`
- the local intensity column is `i`
- the current sample contract `point / feat / label` works on a real frame
- the current canonical adapter entry point is `datasets.pandaset_ff_lane3.build_one_sample`
- the current build-check script is `tools/check_randlanet_build.py`
- the current config scaffold parses and RandLA-Net constructs locally
- Day 4 ended before any real training

## 13. What is still provisional or unresolved

The following remain unresolved after Day 4 and must not be treated as frozen:

- intensity preprocessing statistics over the training split
  `[SET AFTER LOCAL VERIFICATION: compute clipping percentiles and global mean/std on the training split after front-only filtering]`
- measured class counts over the training split after remap
  `[SET AFTER LOCAL VERIFICATION: count active class frequencies on the training split after remap]`
- loss-weight transform
  `[SET AFTER LOCAL VERIFICATION: define and document the transform from measured class counts to loss weights]`
- final sparse-class-safe config values
  `[PROVISIONAL UNTIL LOCALLY VERIFIED]`
  This includes the currently scaffolded `num_layers`, `sub_sampling_ratio`, `grid_size`, and `num_points`.
- exact final split IDs
  `[SET AFTER LOCAL VERIFICATION: freeze exact 58/9/9 IDs after auditing the semseg-enabled sequence pool for lane presence and split diversity]`
- any claim that the current temporary intensity preprocessing should remain the training-time preprocessing
- any claim that the current remap is the final thesis-wide label-policy treatment for all later stages
- any claim that model training, optimization, convergence, or performance are already validated

These are not yet frozen local truths. They remain provisional inputs for later milestone planning and later verification work.

## 14. Exact boundary reached at the end of Day 4

Days 1–4 are complete.

The stop-before-training boundary was preserved. The project reached a real verified baseline path with:

- a working local environment
- a readable local PandaSet installation
- locally verified raw semantic IDs relevant to the baseline
- verified forward-only filtering
- verified post-filter semseg alignment
- a working real-frame adapter path producing `point / feat / label`
- a parsing config scaffold
- a successful local RandLA-Net construction check
- explicit written stop conditions before training

What has **not** been reached:

- any real training run
- any validated training loop
- any performance claim
- any metric result
- any final hyperparameter decision

The project is therefore at the end of the first-4-days verification stage and at the beginning of the next milestone build-up, with a verified baseline path instead of a speculative one.

## 15. What future planning documents should assume from this state

Future planning documents, branches, or agent sessions should assume the following as established:

- the repository already contains a working first-pass adapter/data-contract path
- the canonical adapter entry point is `datasets/pandaset_ff_lane3.build_one_sample()`
- reusable implementation logic lives under `src/thesis_pipeline/...`
- guide-facing wrapper scripts remain under `tools/...`
- the Day 4 build-check path and config path already exist and are functional
- `in_channels: 4` is the correct current local constructor value for the existing contract
- the patched local PandaSet devkit requirement must not be overwritten or forgotten

They should still treat the following as provisional:

- training-time intensity normalization statistics
- training-split class counts
- loss weights
- final sparse-class-safe configuration values
- exact final split membership
- any training or performance behavior

They must not accidentally regress or overwrite:

- the local `.pkl` compatibility path
- the `seq.lidar.poses[frame_idx]` ego-pose path
- the list-safe frame-key compatibility logic
- the guide-facing import path `datasets.pandaset_ff_lane3`
- the explicit no-training boundary recorded in `logs/day4_stop_conditions.txt`

If a future agent is continuing work, the highest-value files to read first are:

- `docs/context/CURRENT_PROJECT_STATE_AFTER_DAY4.md`
- `docs/context/PROJECT_SETUP_CONTEXT.md`
- `docs/context/PROGRESS_LOG.md`
- `docs/context/CODE_STRUCTURE.md`
- `logs/day1_environment_report.txt`
- `logs/day2_dataset_report.txt`
- `logs/day3_adapter_report.txt`
- `logs/day4_model_build_report.txt`
- `datasets/pandaset_ff_lane3.py`
- `configs/randlanet_pandaset_ff_lane3.yml`
- `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`
- `src/thesis_pipeline/core/pandaset_compat.py`

## 16. Evidence sources used for this document

The following repo-relative files were used directly:

- `docs/context/first_4_days_project_context (4).md`
- `docs/context/first_4_days_execution_guide (4).md`
- `docs/context/PROJECT_SETUP_CONTEXT.md`
- `docs/context/CODE_STRUCTURE.md`
- `docs/context/PROGRESS_LOG.md`
- `requirements_working_panda.txt`
- `logs/day1_environment_report.txt`
- `logs/day2_dataset_report.txt`
- `logs/day3_adapter_report.txt`
- `logs/day4_model_build_report.txt`
- `logs/day4_stop_conditions.txt`
- `logs/dataset_root.txt`
- `logs/day2_chosen_sequence.txt`
- `logs/day2_chosen_frame.txt`
- `logs/day2_forward_sensor.txt`
- `datasets/__init__.py`
- `datasets/pandaset_ff_lane3.py`
- `configs/randlanet_pandaset_ff_lane3.yml`
- `tools/check_versions.py`
- `tools/check_imports.py`
- `tools/check_pandaset_devkit_origin.py`
- `tools/precheck_sequence_and_sensor.py`
- `tools/check_pandaset_frame_access.py`
- `tools/check_pandaset_semseg_access.py`
- `tools/check_raw_label_ids.py`
- `tools/check_forward_only_filter.py`
- `tools/check_semseg_alignment.py`
- `tools/check_intensity_field.py`
- `tools/inspect_pandaset_ff_lane3_sample.py`
- `tools/check_randlanet_build.py`
- `src/thesis_pipeline/__init__.py`
- `src/thesis_pipeline/core/paths.py`
- `src/thesis_pipeline/core/pandaset_compat.py`
- `src/thesis_pipeline/checks/day1.py`
- `src/thesis_pipeline/checks/day2.py`
- `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`

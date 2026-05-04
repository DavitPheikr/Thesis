# Milestone B: Implementation and Closeout

## 1. What this document is

This document explains the completed Milestone B work as it actually exists in this repository. It is not a future plan, and it is not a chat summary. It is a technical and narrative closeout document that reconstructs what Milestone B was supposed to do, what the repository looked like when it started, what was implemented day by day, what local runtime realities were discovered, what failed operationally, what was corrected, what was finally accepted, and what Milestone C inherits.

It should be read together with:

- `docs/context/milestone_b_project_context_FINAL.md`
- `docs/context/milestone_b_implementation_guide_FINAL.md`
- `docs/milestone_a/MILESTONE_A_DAY1_TO_DAY4_IMPLEMENTATION.md`
- `MILESTONE_B_DAY6_PREP_AND_STATUS.md`

The two final Milestone B context documents define the intended meaning and required execution order. This document explains how that logic was carried out in practice and what exact repository state now represents the Milestone B close.

## 2. What Milestone B meant in this repository

For this project, Milestone B was the **mechanical trainability milestone**. It was not a model-quality milestone, not a thesis-results milestone, and not a performance-validation milestone.

In practical terms, Milestone B had one central purpose:

> turn the Milestone A single-sample build proof into a mechanically complete, split-driven, statistically grounded, Open3D-ML training path that can execute a small real sanity run without crashing and without obviously unstable loss.

That meant proving all of the following:

- forward-only multi-frame access is safe beyond one verified sample
- the semseg-enabled sequence pool is known and large enough
- a deterministic sequence-level split exists and is frozen
- measured intensity statistics replace provisional per-frame scaling
- measured class counts replace guesswork
- the local Open3D-ML interfaces are discovered from source rather than assumed
- a real dataset class exists under `src/thesis_pipeline/datasets/`
- the YAML config mirrors the measured artifact authority
- the end-to-end training path can execute a real sanity run
- explicit stop conditions exist before any later full-training milestone

Milestone B did **not** mean:

- a trained baseline
- a validated lane detector
- a tuned class-weight policy
- restored baseline-intent `num_points: 16384`
- a decision on oversampling
- a thesis result claim

## 3. Starting point before Milestone B

Milestone B began from the **actual** Milestone A foundation, not from older aspirational text that assumed a prior sanity run. This boundary correction is explicit in `docs/context/milestone_b_project_context_FINAL.md`.

What Milestone A had already established, as confirmed by surviving Milestone A artifacts:

- `logs/day2_dataset_report.txt`
- `logs/day3_adapter_report.txt`
- `logs/day4_model_build_report.txt`
- `logs/day4_stop_conditions.txt`

Those artifacts show that, before Milestone B started, the repository already had:

- a working environment and import chain
- a readable local PandaSet copy
- one verified sample at sequence `001`, frame `0`, sensor `1`
- a working single-frame adapter returning `point / feat / label`
- a parsing RandLA-Net config scaffold
- a locally constructing RandLA-Net instance
- an explicit stop boundary forbidding real training at Milestone A

They also show what Milestone A did **not** have:

- no multi-sequence audit
- no frozen split
- no measured training statistics
- no full dataset class
- no verified pipeline interface discovery artifacts
- no real sanity training run
- no Milestone B stop conditions

That is why Milestone B started from a trustworthy but still narrow foundation: one verified sample and one model-construction proof, not a training-capable pipeline.

## 4. Milestone B strategy and operating model

The Milestone B final documents imposed a strict structure:

- Day 1: preflight and full sequence audit
- Day 2: split freeze
- Day 3: Open3D-ML interface discovery
- Day 4: training statistics and loss weights
- Day 5: dataset class and config update
- Day 6: sanity check and stop conditions

The strategy behind that structure mattered:

- **artifact-first truth flow**: measured values had canonical homes and were not allowed to drift
- **verification-first development**: each major step wrote an artifact and had explicit pass criteria
- **local discovery before coding**: runtime interfaces were read from the installed Open3D code, not guessed
- **minimal safe deviation**: when local reality differed from the guide’s default assumptions, the repo adjusted only enough to preserve milestone intent
- **architecture separation**:
  - reusable logic in `src/thesis_pipeline/...`
  - thin runnable scripts in `tools/...`
  - thin compatibility wrapper in `datasets/pandaset_ff_lane3.py`

That strategy is now visible in the final repository structure and in the resulting Milestone B artifacts.

## 5. The implementation story, phase by phase

### 5.1 Day 1: preflight and full sequence audit

Day 1 had two jobs:

1. verify that forward-only sensor filtering and semseg alignment remained correct across multiple consecutive frames
2. audit all local PandaSet sequence directories to discover the real semseg-enabled pool and the real lane-bearing pool

The work created:

- `tools/preflight_multi_frame_check.py`
- `tools/audit_semseg_sequences.py`
- `logs/milestone_b_preflight.txt`
- `logs/milestone_b_audit_progress.jsonl`
- `logs/milestone_b_sequence_manifest.json`
- `logs/milestone_b_sequence_audit.txt`

What Day 1 proved:

- `set_sensor(1)` persisted across 5 consecutive frames on sequence `001`
- semseg alignment remained intact on those frames
- the local pool contains `103` sequence directories
- `76` sequences are semseg-enabled
- `66` semseg-enabled sequences contain lane points
- `10` semseg-enabled sequences are lane-free
- the semseg pool is large enough for the planned split
- no frame failures were recorded in the final completed audit

The final Day 1 pass evidence is explicit:

- `logs/milestone_b_preflight.txt` ends with `script_status PASS`
- `logs/milestone_b_sequence_audit.txt` ends with:
  - `audit_complete True`
  - `semseg_enabled_count 76`
  - `lane_bearing_count 66`
  - `pool_size_adequate True`
  - `global_frames_failed 0`
  - `script_status PASS`

Important Day 1 correction history:

- The audit was not correct on the first try.
- The repository still contains:
  - `logs/milestone_b_audit_progress.invalid_before_semseg_fix.jsonl`
  - `logs/milestone_b_sequence_audit.invalid_before_semseg_fix.txt`
  - `logs/milestone_b_audit_progress.pre_cleanup_backup.jsonl`
  - `logs/milestone_b_sequence_audit.pre_cleanup_backup.txt`

Those artifacts show that Day 1 had a real correction cycle. The final script now:

- treats `semseg_unavailable` and `semseg_empty` as terminal outcomes
- canonicalizes progress by `seq_id`
- rejects conflicting duplicate records

This was not scope creep. It was a necessary repair to make the resume checkpoint logically correct and trustworthy under interrupted runs.

### 5.2 Day 2: split freeze

Once the semseg-enabled pool was known, the milestone froze a deterministic sequence-level split.

The work created:

- `tools/freeze_split.py`
- `configs/splits/train.txt`
- `configs/splits/val.txt`
- `configs/splits/test.txt`
- `logs/milestone_b_split_report.txt`

What Day 2 proved:

- the split is deterministic from the manifest and seed
- the split covers the full semseg-enabled pool
- val and test contain lane-bearing sequences
- the training split is the remainder

The final split, from `logs/milestone_b_split_report.txt`, is:

- train: `58` sequences
- val: `9` sequences
- test: `9` sequences
- lane-bearing sequences in val: `7`
- lane-bearing sequences in test: `9`

This freeze mattered because all later measured statistics, dataset indexing, and sanity-run behavior depend on these exact split files.

### 5.3 Day 3: Open3D-ML interface discovery

Day 3 was deliberately a **source-discovery phase** rather than a coding phase. The goal was to inspect the local Open3D install and capture what the repo was actually allowed to assume.

The work produced:

- `logs/milestone_b_loss_interface.json`
- `logs/milestone_b_dataset_interface.json`
- `logs/milestone_b_pipeline_interface.json`
- `logs/milestone_b_open3d_interface_notes.txt`
- `logs/milestone_b_source_locations.txt`

The most important discoveries were:

- the dataset base class is available from `open3d._ml3d.datasets.base_dataset.BaseDataset`
- the training pipeline class is `open3d.ml.torch.pipelines.SemanticSegmentation`
- the dataset split object must provide `__len__`, `get_data`, and `get_attr`
- the dataset should return `point`, `feat`, and `label`
- xyz and feat are concatenated later by the model transform path, not by the dataset
- `dataset.cfg.class_weights` is **not** consumed directly as the final CE tensor

That last point became the major Day 3-to-Day 4 deviation:

- the guide’s generic template implied “put the final recommended weights into config”
- the local runtime actually transforms `dataset.cfg.class_weights` internally via `DataProcessing.get_class_weights`

This discovery changed how class-weight handling was implemented for the rest of Milestone B.

### 5.4 Day 4: training statistics and class-weight handling

Day 4 measured the training split rather than reusing any provisional or sample-local numbers.

The work created:

- `tools/compute_training_statistics.py`
- `tools/compute_loss_weights.py`
- `logs/milestone_b_statistics_progress.jsonl`
- `logs/stats_per_seq/{seq_id}.npy`
- `logs/milestone_b_statistics_log.txt`
- `logs/milestone_b_weights_log.txt`
- `logs/milestone_b_training_statistics.json`

What Day 4 measured:

- training sequences completed: `58 / 58`
- frames processed: `4640`
- frames semseg-missing: `0`
- frames failed: `0`
- skip rate: `0.000000`

Measured class counts from `logs/milestone_b_training_statistics.json`:

- ignore: `398730`
- road: `119562394`
- lane: `2098182`
- other: `176504630`

Measured intensity statistics:

- clip low (P0.5): `0.0`
- clip high (P99.5): `114.0`
- clip mean: `22.481164932250977`
- clip std: `14.912428855895996`

Measured weight artifacts:

- raw inverse-frequency lane weight: `84.12264998937177`
- sqrt inverse-frequency lane weight: `9.171840054720304`
- candidate recommended variant: `sqrt_inverse_frequency`

But the final runtime recommendation for the actual sanity run was different:

- `sanity_run_recommended_variant: open3d_native_from_measured_counts`
- `sanity_run_recommended_list: [119562394.0, 2098182.0, 176504630.0]`

Why that happened:

- candidate direct CE vectors were still computed and recorded
- the local Open3D loss implementation transforms `dataset.cfg.class_weights` internally
- therefore the config field had to carry **count-like runtime input semantics**, not the already-final direct CE weights

This is one of the most important Milestone B decisions. The repository chose the minimum safe adjustment:

- preserve the thesis-analysis candidate vectors
- preserve them in the canonical statistics JSON
- record the local runtime semantics explicitly
- wire the runtime config to the local Open3D expectation instead of forcing the guide’s generic template onto a different runtime behavior

Day 4 also measured one representative post-grid-subsampling count:

- raw point count on the verified sample: `62285`
- post-grid count at `grid_size: 0.04`: `53480`
- duplication at baseline-intent `num_points: 16384`: `no`

This did **not** solve the oversampling question, but it established that simple point duplication from grid subsampling was not the immediate baseline problem.

### 5.5 Day 5: dataset class and config assembly

Day 5 replaced the Milestone A single-frame-only path with a proper split-driven dataset implementation.

The work created or updated:

- `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py`
- `src/thesis_pipeline/datasets/__init__.py`
- `datasets/pandaset_ff_lane3.py`
- `configs/randlanet_pandaset_ff_lane3.yml`
- `tools/check_dataset_class.py`
- `logs/milestone_b_dataset_class_report.txt`

What Day 5 had to achieve:

- move the core dataset logic under `src/thesis_pipeline/datasets/`
- keep `datasets/pandaset_ff_lane3.py` as a thin compatibility wrapper
- preserve `build_one_sample()` backward compatibility
- use split files and statistics JSON instead of hardcoded values
- return `point`, `feat`, `label`
- verify train/val/test split objects

The final dataset class does several important things:

- reads split files from `configs/splits/`
- reads canonical measured statistics from `logs/milestone_b_training_statistics.json`
- uses the Day 1 manifest’s per-sequence `lidar_frame_count` values to build deterministic frame indexes
- avoids eager `load_lidar()` loops during index construction
- normalizes intensity with the measured clip-and-standardize path
- remaps labels through the canonical adapter function

The manifest-based frame indexing choice mattered. Earlier interactive Day 5 verification attempts were too heavy because eager sequence loads across the whole split repeatedly stressed the session. The accepted solution was:

- use audited `lidar_frame_count` values from `logs/milestone_b_sequence_manifest.json`
- reserve actual `load_lidar()` / `load_semseg()` calls for real sample access only

The final Day 5 report confirms:

- YAML and JSON class weights are consistent
- YAML and JSON intensity values are consistent
- backward compatibility lane fraction on the verified sample remains `0.002568837`
- train split length is `4640`
- val split length is `720`
- test split length is `720`
- all three split objects return correct sample shapes and labels

Day 5 therefore closed with a true dataset-class path, not just a script shim over the Milestone A adapter.

### 5.6 Day 6: preparation, crashes, memory hardening, sanity run, and closeout

Day 6 was the milestone’s operationally messiest phase and the one most shaped by laptop/runtime reality.

The final Day 6 code and artifacts are:

- `tools/sanity_train_check.py`
- `logs/milestone_b_sanity_train_report.txt`
- `logs/milestone_b_sanity_config_snapshot.yml`
- `logs/milestone_b_stop_conditions.txt`
- `MILESTONE_B_DAY6_PREP_AND_STATUS.md`

There are also multiple failed-attempt artifacts, including:

- `logs/milestone_b_sanity_train_report.fast_fail_attempt.txt`
- `logs/milestone_b_sanity_train_report.overlapped_invalid.txt`
- `logs/milestone_b_sanity_train_report.killed_before_clean_rerun.txt`
- `logs/milestone_b_sanity_train_report.stale_empty_before_fresh_run.txt`
- `logs/milestone_b_sanity_train_report.prior_successful_run_2026-04-15_07-21.txt`
- archived zero-byte Open3D train logs under `logs/RandLANet_PandaSetFFLane3_torch/archived_day6_prererun_2026-04-15/`

The Day 6 implementation story had several stages.

#### Day 6 preparation

The repo first hardened the Day 6 script and execution plan:

- `tools/sanity_train_check.py` was aligned to the actual local runtime
- `MILESTONE_B_DAY6_PREP_AND_STATUS.md` documented the Day 6 assumptions, fixes, and run procedure
- stale empty Day 6 artifacts were archived instead of being treated as valid outputs

The most important Day 6 script adjustments were:

- force `ckpt_path = None` and `is_resume = False`
- use the correct local primary weight path:
  - `stats["class_weights"]["sanity_run_recommended_list"]`
- treat completed instability as a valid reported outcome rather than a raw execution failure
- flush report content incrementally so interrupted runs are diagnosable

#### Day 6 failed attempts and the `num_points` concession

The final closeout state shows that baseline-intent `num_points: 16384` did **not** survive Day 6 on this machine.

This is recorded directly in `logs/milestone_b_stop_conditions.txt`, which states:

- `sanity_num_points_used: 4096`
- `num_points_baseline_intent: 16384`
- a concession note documenting multiple earlier failed attempts

The archived Day 6 artifacts support that story:

- several early report files stop at `attempt_1_status STARTED`
- multiple Open3D train logs are zero bytes
- a later successful report at `07:21` exists
- the canonical final report and snapshot are from the `15:35` run

The accepted Milestone B interpretation is therefore:

- `16384` remained the baseline intent
- the local machine could not establish a clean Day 6 sanity pass at that value
- reducing to `4096` was allowed by the guide as a **sanity-check-only concession**
- the concession had to be documented and carried forward explicitly

#### Day 6 dataset memory hardening

By the time Milestone B closed, `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py` contained an explicit cleanup block inside `_load_sample()`:

- clear `seq.lidar._data`
- clear `seq.lidar._poses`
- clear `seq.lidar._timestamps`
- clear `seq.semseg._data`
- call `gc.collect()`

This is important because the local PandaSet `DataSet` caches sequence objects, and `load_lidar()` / `load_semseg()` load full-sequence structures even when the dataset only needs one frame. The final code explicitly releases those sequence-level caches after extracting the single-frame numpy arrays.

That behavior is part of the final accepted Milestone B implementation and is one of the key Day 6 stabilization decisions.

#### Day 6 final accepted sanity run

The canonical final run is represented by:

- `logs/milestone_b_sanity_train_report.txt`
- `logs/milestone_b_sanity_config_snapshot.yml`
- `logs/milestone_b_stop_conditions.txt`

The accepted result is:

- `num_points: 4096`
- `batch_size: 1`
- `steps_per_epoch_train: 5`
- `steps_per_epoch_valid: 2`
- `max_epoch: 2`
- primary runtime weight variant:
  - `open3d_native_from_measured_counts`
- fallback variant available:
  - `sqrt_inverse_frequency`

Final reported sanity outcome:

- `attempt_1_steps 15`
- `attempt_1_ok True`
- `loss_first 1.136897`
- `loss_last 1.080209`
- `loss_ratio_final_to_initial 0.9501`
- `sanity_train_check_ok True`
- `script_status PASS`

This satisfied the Day 6 pass criteria.

The closeout also wrote explicit stop conditions and carried-forward constraints, which is what turns a sanity-run success into a proper milestone close rather than just a lucky detached process.

## 6. Major technical decisions and why they were made

Several Milestone B decisions shaped the final repository state.

### 6.1 `get_frame_count(seq)` and compatibility helpers remained canonical

The repo did not scatter ad hoc frame-count assumptions across scripts. Instead it kept `src/thesis_pipeline/core/pandaset_compat.py::get_frame_count()` as the one compatible counting entrypoint and reused it in scripts that had to reason about PandaSet frame containers.

This preserved compatibility across dict-like and list-like devkit behaviors.

### 6.2 The canonical remap stayed centralized

The repo did not reimplement remapping per script. `src/thesis_pipeline/adapters/pandaset_ff_lane3.py` exposes:

- `remap_raw_pandaset_ids()` as the canonical Milestone B entrypoint
- `remap_raw_labels()` as the backward-compatible Milestone A alias

That kept label semantics centralized and preserved old usage paths.

### 6.3 Measured statistics, not hardcoded values, control the dataset path

The dataset class does not carry hardcoded clip bounds or normalization constants. It reads them from the Day 4 canonical statistics JSON.

This matters because Milestone B was explicitly about replacing provisional values with measured truths.

### 6.4 Runtime config uses measured counts, not direct candidate CE weights

This was the most important semantic deviation from the guide’s default template.

The repo discovered that local Open3D transforms `dataset.cfg.class_weights` internally. Therefore:

- direct CE weight candidates are recorded for thesis analysis
- runtime config uses measured active-class counts as the input list

This was the minimum safe adjustment that preserved both the milestone’s intent and the actual runtime truth.

### 6.5 Frame indexing became manifest-based

The dataset class does not eagerly call `load_lidar()` across all sequences when constructing split indexes. It uses `lidar_frame_count` values already measured by the Day 1 audit manifest.

This avoided repeated interactive-session crashes and aligned with the guide’s requirement to maintain a deterministic frame index table **without loading all frames upfront**.

### 6.6 `_load_sample()` releases PandaSet sequence caches

The final dataset class explicitly frees the heavy full-sequence PandaSet structures after extracting the one requested sample. That change was a concrete response to real Day 6 stability pressure.

### 6.7 Day 6 detached execution and non-resume behavior

The Day 6 path had to be detached from the VS Code process tree, and the script had to force non-resume semantics even though the local Open3D pipeline defaults toward resuming from checkpoints.

This was necessary because:

- prior runs crashed or overlapped
- checkpoint files exist after the successful sanity run
- silent resume would corrupt the meaning of later reruns

### 6.8 `num_points: 4096` was accepted for the sanity run only

This is an important Milestone B boundary decision.

The repo accepted `4096` because:

- the guide explicitly permits temporary reduction under OOM pressure during Day 6
- the sanity run is a mechanical proof, not the final baseline experiment
- the stop conditions document the concession and require restoration of `16384` in Milestone C

This did **not** redefine the baseline intent. It only allowed the milestone to close with a truthful, documented sanity-only configuration.

## 7. Major deviations, mismatches, and corrections

Milestone B did not unfold as a perfect straight-line implementation. Several meaningful mismatches appeared between the final guide’s generic expectations and the local repo/runtime reality.

### 7.0 Compact accepted-deviation map

| Original intended behavior                                                              | Observed local reality                                                                                | Accepted adjustment                                                                                                      | Why safe                                                                                                             | Milestone C revisit? |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | -------------------- |
| Milestone A had already included a tiny sanity run                                      | Milestone A stopped before any real training                                                          | Milestone B documents explicitly corrected the milestone boundary and treated the sanity run as a Day 6 deliverable      | Restored operational truth rather than changing project scope                                                        | No                   |
| Recommended class-weight vector could be placed into config as final runtime CE weights | Local Open3D transforms `dataset.cfg.class_weights` internally before constructing `CrossEntropyLoss` | Preserve direct CE candidate vectors for analysis, but use `sanity_run_recommended_list` as the runtime-safe config list | Preserved artifact truth while matching the actual installed runtime                                                 | Yes                  |
| Dataset indexing could eagerly load sequences while building split tables               | Repeated eager `load_lidar()` loops across many sequences stressed the local interactive environment  | Build deterministic frame indexes from Day 1 manifest `lidar_frame_count` values instead                                 | Used already-measured authoritative counts without changing sample semantics                                         | No                   |
| Reading one sample would only retain sample-sized memory                                | PandaSet cached sequence objects retain full-sequence lidar and semseg state                          | `_load_sample()` now clears the loaded sequence caches after extracting numpy arrays                                     | Changed memory lifetime only; did not change data semantics                                                          | No                   |
| Day 6 should succeed at baseline-intent `num_points: 16384`                             | The accepted sanity run only completed cleanly at `num_points: 4096` on this machine                  | Preserve the exact Day 6 run in the sanity snapshot, but restore the live YAML to `16384` for Milestone C entry          | The guide explicitly permits a sanity-only reduction under OOM pressure, and the snapshot preserves historical truth | Yes                  |

### 7.1 The original Day 6 assumption of a prior sanity run was wrong

Original assumption in stale high-level text:

- Milestone A already included a tiny sanity run

Observed local truth:

- Milestone A stopped before any real training

Adjustment:

- Milestone B documents explicitly corrected the milestone boundary

Why it was safe:

- it restored operational truth rather than changing the project

Final status:

- accepted final Milestone B behavior

### 7.2 Open3D class-weight semantics differed from the guide’s generic template

Original assumption:

- a recommended weight vector could be put into config as the final runtime loss weights

Observed local reality:

- `dataset.cfg.class_weights` is transformed internally before CrossEntropyLoss is built

Adjustment:

- record direct CE candidate vectors for analysis
- record runtime count-like input semantics separately
- use `sanity_run_recommended_list` as the runtime-safe config list

Why it was the minimum safe change:

- it respected the actual installed runtime
- it preserved artifact truth flow

Final status:

- accepted final Milestone B behavior
- policy still carried into Milestone C for possible refinement

### 7.3 Dataset frame indexing could not rely on eager sequence loading

Original expectation:

- dataset-class verification would just instantiate and index cleanly

Observed local reality:

- eager `load_lidar()` loops across many sequences repeatedly stressed the interactive environment

Adjustment:

- build deterministic frame indexes from the Day 1 manifest’s frame counts

Why it was the minimum safe change:

- the manifest already contained authoritative frame counts
- it removed unnecessary heavy loading without changing sample semantics

Final status:

- accepted final Milestone B behavior

### 7.4 PandaSet sequence caches had to be explicitly released

Original assumption:

- reading one sample would only retain the sample-sized data

Observed local reality:

- PandaSet cached sequence objects retain full-sequence lidar and semseg structures

Adjustment:

- `_load_sample()` now clears those cached loaded structures after extracting the frame

Why it was the minimum safe change:

- it changed only memory lifetime, not data semantics

Final status:

- accepted final Milestone B behavior

### 7.5 Day 6 accepted run config and post-closeout live config now differ

Original baseline intent:

- `num_points: 16384`

Observed local Day 6 reality:

- the successful sanity run used `num_points: 4096`

Adjustment:

- preserve the exact Day 6 run config in `logs/milestone_b_sanity_config_snapshot.yml`
- document the concession in stop conditions
- after Milestone B closeout, restore the live YAML to `model.num_points: 16384`
- clean the duplicated `pipeline.num_workers: 0` keys in the live YAML

Why it was safe:

- the guide explicitly allows this as a temporary Day 6 concession
- the snapshot preserves the successful Day 6 evidence even after the live YAML is cleaned for Milestone C entry

Final status:

- accepted Day 6 run state remains the `4096` sanity snapshot
- the live YAML is now a cleaned Milestone C entry config with `model.num_points: 16384`
- the live YAML is still explicitly not a fully revalidated baseline-training config

## 8. Crash and debugging history

Milestone B’s operational history matters because the final closeout is only trustworthy if the failures are acknowledged honestly.

### 8.1 Day 1 audit had a real resume-correctness repair

Evidence:

- invalid-before-fix progress and audit files remain on disk

What was learned:

- interrupted JSONL progress needs explicit canonicalization rules
- terminal unavailable sequences must count as completed

Why the final Day 1 result is still trustworthy:

- the final script rejects conflicting duplicates
- the final manifest and final audit logs are internally consistent

### 8.2 Day 5 verification repeatedly stressed the interactive session

Evidence:

- the final dataset class avoids eager split-wide sequence loading
- empty staging probe logs remain as traces of abandoned approaches

What was learned:

- interactive verification strategy mattered as much as code correctness

Why the final Day 5 result is still trustworthy:

- the final dataset report is a completed `script_status PASS` artifact

### 8.3 Day 6 produced overlapping, partial, empty, and archived attempts

Evidence:

- multiple partial report files
- archived zero-byte Open3D train logs
- archived prior PID files
- a prep-only Day 6 document
- a prior successful Day 6 report
- the final canonical Day 6 report and stop conditions

What failed operationally:

- attached and overlapping runs
- zero-byte or partial report artifacts
- repeated baseline-intent `16384` attempts that did not produce a clean canonical success artifact

What was learned:

- detached execution was necessary
- the report needed to be made crash-robust
- the pipeline had to be forced non-resuming
- PandaSet sequence memory had to be released explicitly

Why the final Day 6 result is still trustworthy despite the mess:

- there is a complete final report with `sanity_train_check_ok True`
- there is a matching config snapshot
- there is a fully populated stop-conditions file
- there is also a prior successful run artifact with the same `4096` sanity-only configuration and the same primary weight variant
- the final successful report’s metrics are mechanically plausible and consistent with the milestone’s intended scope

### 8.4 Day 6 still closed with a cosmetic artifact race

The final `logs/milestone_b_sanity_train_report.txt` includes duplicated content and padded bytes. This is also acknowledged directly in `logs/milestone_b_stop_conditions.txt`.

Cause:

- `tools/sanity_train_check.py` writes to `REPORT_FILE`
- the shell invocation simultaneously redirects stdout to the same file

Effect:

- cosmetic duplication and padding in the report file

Why this did not invalidate Milestone B:

- the PASS semantics are still visible
- the first valid report block is complete
- the stop conditions extracted the outcome cleanly
- the issue is explicitly carried forward to Milestone C rather than hidden

## 9. Repository structure and important Milestone B files

Milestone B left the repository in a clearer structure than it inherited.

### `src/thesis_pipeline/`

This is where reusable pipeline logic now lives.

- `core/pandaset_compat.py`
  - compatibility helpers for PandaSet frame counting
- `adapters/pandaset_ff_lane3.py`
  - canonical remap logic
  - Milestone A backward-compatible sample builder
- `datasets/pandaset_ff_lane3_dataset.py`
  - the full split-driven dataset implementation used by the training path

### `datasets/`

This is now a thin wrapper layer, not the core implementation.

- `datasets/pandaset_ff_lane3.py`
  - exports `PandaSetFFLane3Dataset`
  - preserves `build_one_sample()` and `inspect_sample_main()`

### `tools/`

This is where runnable milestone scripts live.

- `preflight_multi_frame_check.py`
- `audit_semseg_sequences.py`
- `freeze_split.py`
- `compute_training_statistics.py`
- `compute_loss_weights.py`
- `check_dataset_class.py`
- `sanity_train_check.py`

### `configs/`

- `configs/splits/`
  - frozen train/val/test split files
- `configs/randlanet_pandaset_ff_lane3.yml`
  - the cleaned live config carried into Milestone C entry
  - restored to `model.num_points: 16384` after Milestone B closeout
  - no longer identical to the Day 6 sanity snapshot

### `logs/`

This directory now contains the operational truth of the milestone:

- preflight, audit, split, interface, statistics, weights, dataset-class, sanity-run, stop-condition, progress, and archived-failure artifacts

### `docs/milestone_b/`

This folder now contains:

- `README.md`
- this implementation and closeout document

## 10. The implemented end-to-end pipeline

At the end of Milestone B, the actual implemented training-capable path is:

1. Read local PandaSet from `logs/dataset_root.txt`
2. Use Day 1’s verified forward-only behavior and audited semseg pool
3. Freeze train/val/test split by sequence
4. Measure training-split intensity and class statistics
5. Record class-weight candidates and runtime semantics in the statistics JSON
6. Build a split-driven dataset class from the split files and statistics JSON
7. Return `point`, `feat`, and `label` per sample
8. Let RandLA-Net concatenate xyz and feat in its transform path
9. Feed the dataset instance into the Open3D `SemanticSegmentation` pipeline
10. Run a bounded sanity training loop
11. Write stop conditions before any later full-training milestone

That is the real Milestone B pipeline completion claim.

## 11. What was actually verified

Milestone B closed with several different kinds of truth.

### 11.1 Fixed project truths

These remained unchanged:

- task: point-wise semantic segmentation on PandaSet LiDAR
- classes: road / lane / other with ignore excluded from loss
- positive class: Lane Line Marking only
- sensor: forward-facing lidar only, sensor ID `1`
- coordinate frame: ego-local
- feature concept: xyz plus one intensity-derived channel
- model family: RandLA-Net via Open3D-ML

### 11.2 Measured truths

These were measured across the real training split:

- semseg-enabled sequence count: `76`
- lane-bearing sequence count: `66`
- training split size: `58`
- training frames processed: `4640`
- class counts
- clip bounds
- clip mean and std
- representative post-grid point count

### 11.3 Local-runtime-discovered truths

These came from inspecting local code rather than from the documents:

- dataset base class import path
- pipeline class and training entrypoint
- concat location at the model transform step
- active-class-only loss weighting scope
- internal Open3D class-weight transform semantics
- no need for dataset registry for this path

### 11.4 Sanity-only truths

These are true for the Day 6 sanity configuration, not for the final full-training baseline:

- `num_points: 4096`
- `batch_size: 1`
- `num_workers: 0`
- `max_epoch: 2`
- `steps_per_epoch_train: 5`
- `steps_per_epoch_valid: 2`
- `15` captured step losses
- non-diverging sanity loss on the runtime-recommended variant

### 11.5 Deferred truths

These remain for Milestone C:

- revalidation of the restored `16384` live config
- lane-aware oversampling decision
- whether direct candidate CE weights should ever become runtime policy
- cleanup of the Day 6 dual-writer report race

## 12. Final Milestone B artifact inventory

### 12.1 Canonical artifact / authority map

| Authority type                    | Canonical artifact                                                                                                            | Role                                                                                                                        |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Milestone meaning/spec authority  | `docs/context/milestone_b_project_context_FINAL.md`                                                                           | Defines what Milestone B means, its certainty model, and what it is trying to prove                                         |
| Milestone execution authority     | `docs/context/milestone_b_implementation_guide_FINAL.md`                                                                      | Defines the required day-by-day execution order, pass criteria, and stop logic                                              |
| Split authority                   | `configs/splits/train.txt`, `configs/splits/val.txt`, `configs/splits/test.txt`                                               | Frozen sequence-level split used by later measurement and runtime code                                                      |
| Statistics authority              | `logs/milestone_b_training_statistics.json`                                                                                   | Single authoritative source for measured intensity values, class counts, and recorded weight semantics                      |
| Interface authority               | `logs/milestone_b_loss_interface.json`, `logs/milestone_b_dataset_interface.json`, `logs/milestone_b_pipeline_interface.json` | Captured local Open3D runtime discoveries that later code is allowed to rely on                                             |
| YAML mirror / live entry config   | `configs/randlanet_pandaset_ff_lane3.yml`                                                                                     | Live config mirror and current Milestone C entry config; if numeric values disagree with the statistics JSON, the JSON wins |
| Historical Day 6 sanity-run truth | `logs/milestone_b_sanity_config_snapshot.yml`, `logs/milestone_b_sanity_train_report.txt`                                     | Exact accepted Day 6 run shape and reported sanity outcome                                                                  |
| Milestone closeout gate           | `logs/milestone_b_stop_conditions.txt`                                                                                        | Final Milestone B closeout statement and carry-forward authority                                                            |
| Historical/debug-only evidence    | archived invalid, stale, prior-success, and partial Day 1 / Day 6 logs under `logs/`                                          | Preserves failure history and debugging provenance; not canonical for final measured truth                                  |

The most important final artifacts are:

- `logs/milestone_b_preflight.txt`
  - multi-frame forward-only and alignment proof
- `logs/milestone_b_sequence_manifest.json`
  - canonical semseg/lane pool inventory and per-sequence details
- `logs/milestone_b_sequence_audit.txt`
  - final Day 1 audit report
- `logs/milestone_b_split_report.txt`
  - frozen split provenance
- `configs/splits/train.txt`
- `configs/splits/val.txt`
- `configs/splits/test.txt`
  - canonical split authority
- `logs/milestone_b_loss_interface.json`
- `logs/milestone_b_dataset_interface.json`
- `logs/milestone_b_pipeline_interface.json`
  - local Open3D interface discoveries
- `logs/milestone_b_training_statistics.json`
  - canonical measured statistics and weight artifacts
- `logs/milestone_b_statistics_log.txt`
- `logs/milestone_b_weights_log.txt`
  - Day 4 reports
- `logs/milestone_b_dataset_class_report.txt`
  - Day 5 verification artifact
- `logs/milestone_b_sanity_config_snapshot.yml`
  - exact final sanity-run config
- `logs/milestone_b_sanity_train_report.txt`
  - final sanity-run outcome, albeit cosmetically duplicated/padded
- `logs/milestone_b_stop_conditions.txt`
  - final milestone closeout gate and Milestone C carry-forward authority
- `MILESTONE_B_DAY6_PREP_AND_STATUS.md`
  - Day 6 preparation and runtime-safety documentation

## 13. The accepted Day 6 sanity run

The accepted final Day 6 run proved:

- the assembled pipeline instantiates
- the dataset class can feed the pipeline end to end
- at least `15` step-level losses were captured
- losses remained finite
- loss changed between first and last captured steps
- final loss was not explosively larger than initial loss
- the runtime-recommended class-weight input was usable
- Milestone B stop conditions could be written from a real sanity outcome

The accepted final Day 6 run did **not** prove:

- that `16384` is currently safe on this machine
- that the chosen weights are final for thesis experiments
- that oversampling is unnecessary
- that training quality is good
- that full training can proceed unchanged without Milestone C decisions

The exact accepted sanity configuration was:

- `num_points: 4096`
- `batch_size: 1`
- `num_workers: 0`
- `real_training_allowed: false`

That configuration was accepted because it satisfied the guide’s Day 6 mechanical purpose while staying within the guide’s explicit allowance for a sanity-only `num_points` reduction under OOM pressure.

Milestone B therefore treats Day 6 as accepted when a real bounded run produces a non-empty report with finite, changing, non-exploding loss, ends with `sanity_train_check_ok True` and `script_status PASS`, and is accompanied by a matching sanity snapshot and written stop conditions.

## 14. What Milestone B did not claim

Milestone B did not claim any of the following:

- final baseline training hyperparameters are settled
- `16384` is already revalidated
- full training is now safe to run unattended
- lane oversampling is solved
- class-weight policy is final
- any validation or test metric has been earned
- any qualitative output has been produced
- any thesis conclusion can be drawn from the sanity run

That distinction matters. Milestone B closed because it achieved mechanical completeness plus explicit stop conditions, not because it solved all later experimental questions.

## 15. Important local realities Milestone C must respect

Milestone C must not forget these concrete realities:

- `logs/milestone_b_sanity_config_snapshot.yml` is the historical Day 6 run config at `4096`
- `configs/randlanet_pandaset_ff_lane3.yml` is now the cleaned Milestone C entry config with `model.num_points: 16384`
- the baseline intent remains `16384`
- the local Open3D runtime transforms `dataset.cfg.class_weights` internally
- the dataset class must keep reading canonical statistics from JSON rather than hardcoding values
- PandaSet sequence caches can accumulate heavy memory unless explicitly released
- Day 6 produced checkpoint files under `logs/RandLANet_PandaSetFFLane3_torch/checkpoint/`
- the pipeline can silently resume unless `is_resume=False` is forced
- the current sanity report file has a dual-writer cosmetic issue
- the live YAML no longer repeats `num_workers: 0`, but the Day 6 snapshot still preserves the original run-time file shape
- oversampling is indicated by the recorded heuristic and has not been resolved

## 16. Carry-forward items for Milestone C

The final stop conditions already list several carry-forward items, and they should be treated as authoritative.

The most important ones are:

1. Revalidate the restored `model.num_points: 16384` live config before any full training run.
2. Investigate the true root cause if `16384` still fails instead of silently making `4096` permanent.
3. Implement lane-aware patch oversampling, or reject it with direct post-subsampling evidence.
4. Fix the dual-writer race in `tools/sanity_train_check.py`.
5. Decide whether to retain or archive the Day 6 checkpoint files.
6. Finalize class-weight policy for training beyond the sanity run.

Milestone C therefore does **not** start from “train normally now.” It starts from a mechanically working but intentionally constrained and explicitly documented Milestone B end state.

## 17. Recommended reading order for a new engineer or coding agent

The most practical reading order from the current repo state is:

1. `docs/milestone_a/MILESTONE_A_DAY1_TO_DAY4_IMPLEMENTATION.md`
2. `docs/context/milestone_b_project_context_FINAL.md`
3. `docs/context/milestone_b_implementation_guide_FINAL.md`
4. `logs/milestone_b_sequence_manifest.json`
5. `logs/milestone_b_split_report.txt`
6. `logs/milestone_b_training_statistics.json`
7. `logs/milestone_b_loss_interface.json`
8. `logs/milestone_b_dataset_interface.json`
9. `logs/milestone_b_pipeline_interface.json`
10. `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py`
11. `configs/randlanet_pandaset_ff_lane3.yml`
12. `logs/milestone_b_dataset_class_report.txt`
13. `MILESTONE_B_DAY6_PREP_AND_STATUS.md`
14. `logs/milestone_b_sanity_train_report.txt`
15. `logs/milestone_b_stop_conditions.txt`

That order reconstructs the milestone in the same dependency chain it was implemented.

## 18. Final Milestone B summary

Milestone B achieved what it was supposed to achieve for this repository.

It took the project from:

- a single verified PandaSet sample
- a working adapter
- a parsing config scaffold
- a model-construction proof

to:

- a verified semseg-enabled sequence pool
- a frozen split
- measured training statistics
- a documented local Open3D interface contract
- a split-driven dataset class
- a config populated from measured artifacts
- a real sanity training run with finite, non-diverging loss
- explicit stop conditions and Milestone C carry-forward items

It did **not** solve everything, and it does not pretend to. It closed with:

- a sanctioned Day 6 `4096` sanity-only concession
- unresolved oversampling work
- an unfinished `16384` revalidation
- a few operational cleanup items

But those are precisely the kinds of things Milestone B was supposed to surface and hand off clearly, not hide.

The milestone is therefore considered **closed** because its own definition was achieved:

- the pipeline is mechanically complete
- the sanity run succeeded
- no full training has occurred
- no performance claim has been made
- stop conditions now exist

Milestone C inherits a repo that is no longer guessing about its training path. It inherits one that has already discovered its local constraints, measured its real data, proven a real sanity pass, and written down exactly what still must be resolved before serious training begins.

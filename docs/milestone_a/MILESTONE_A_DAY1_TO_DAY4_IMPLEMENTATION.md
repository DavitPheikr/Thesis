# Milestone A: Day 1-4 Implementation Documentation

## 1. What this document is

This document explains the completed Day 1-4 work as a coherent Milestone A foundation pass for the thesis project. It is not a plan for future work. It is a narrative and technical explanation of what the four-day task was trying to achieve, how that work was implemented in this repository, what decisions were made, what local realities were discovered, and what exact baseline state was reached at the end.

It should be read together with:
  
- `docs/context/first_4_days_project_context (4).md`
- `docs/context/first_4_days_execution_guide (4).md`
- `docs/context/CURRENT_PROJECT_STATE_AFTER_DAY4.md`
- `docs/context/PROGRESS_LOG.md`

The context and execution-guide documents define the intended first-four-days logic. This document explains how that logic was carried out in practice and what the repository now contains as a result.

## 2. Milestone A purpose in practical terms

For this repository, Milestone A at the end of Day 4 does **not** mean a trained model, a validated experiment, or any performance result. It means the project moved from planning assumptions to a locally verified technical baseline path.

The four-day task had one central purpose:

> prove that the thesis pipeline is buildable on the local machine with correct semantics before any real training begins.

In practical terms, that meant proving all of the following:

- the local environment can import the required stack
- the local PandaSet copy is readable through the installed devkit
- the forward-facing LiDAR-only baseline can be extracted correctly
- semantic labels remain aligned after filtering
- one real PandaSet frame can be converted into `point / feat / label`
- a RandLA-Net config scaffold can be written against that contract
- RandLA-Net can be instantiated locally without immediate structural failure

This is why Days 1-4 were designed as a trust-building stage rather than a model-development stage. The output of the work is a credible starting point, not a result claim.

## 3. Fixed project meaning that guided the work

Several semantic truths were treated as fixed throughout the implementation:

- The task is point-wise semantic segmentation on PandaSet LiDAR.
- The active semantic classes are `road`, `lane`, and `other`.
- `lane` means **Lane Line Marking only**.
- The baseline is LiDAR-only.
- The baseline uses the forward-facing LiDAR only.
- The intended coordinate frame is ego-local.
- The intended minimal feature concept is `xyz` plus one processed intensity channel.
- The model family is RandLA-Net inside Open3D-ML.
- Days 1-4 stop before real training.

These were treated as project truth. The implementation work did not change them. What the implementation work did change was the level of certainty about how those ideas map onto the local environment and the local PandaSet/devkit behavior.

## 4. Starting point before implementation

Before the four-day task was implemented, the repository already had several important local facts established:

- the project root was known
- the `panda` virtual environment already existed
- the local PandaSet dataset was present
- the local PandaSet devkit had already been patched to support `.pkl` frames
- `open3d.ml.torch` imported
- CUDA was available

However, those facts were not yet captured as a full, reproducible early-stage verification pipeline inside the repository. The goal of the Day 1-4 work was to turn those facts into:

- stable verification scripts
- stable log outputs
- a canonical sample-building path
- a canonical config scaffold
- a documented stop boundary before training

## 5. Overall implementation strategy

The implementation followed the four-day logic from the execution guide, but it also introduced a cleaner repository structure than the original step-by-step script outline.

The strategy was:

1. keep the guide-facing script entry points under `tools/`
2. move reusable logic into `src/thesis_pipeline/`
3. keep the guide-required adapter module under `datasets/`
4. keep the guide-required config file under `configs/`
5. record actual results under `logs/`
6. record evolving narrative state under `docs/context/`

This structure let the project satisfy the execution guide without becoming a collection of duplicated one-off scripts.

## 6. Day-by-day implementation story

### Day 1: Environment grounding

Day 1 focused on turning environment assumptions into recorded local facts.

The work created:

- `tools/check_versions.py`
- `tools/check_imports.py`
- `tools/check_pandaset_devkit_origin.py`
- `logs/day1_environment_report.txt`
- `logs/dataset_root.txt`

The key decisions on Day 1 were:

- reuse the existing `panda` environment instead of creating a new one
- record the dataset root in one canonical file
- verify imports through scripts rather than relying on memory
- explicitly record the patched-devkit reinstall command because it is operationally critical

What Day 1 proved:

- the repository is running on Python `3.12.3`
- the local stack is Torch `2.2.2+cu121`, Open3D `0.19.0`, NumPy `1.26.4`, pandas `3.0.2`
- `open3d.ml.torch` imports successfully
- `RandLANet` imports successfully
- `pandaset` imports successfully
- `geometry.lidar_points_to_ego` exists
- CUDA is visible to Torch
- the patched PandaSet source is present locally

Why this mattered:

Without Day 1, the rest of the work would still depend on implicit environment trust. After Day 1, the environment state became explicit and reproducible.

### Day 2: Dataset truth and alignment truth

Day 2 focused on whether the local PandaSet/devkit behavior actually supports the intended baseline semantics.

The work created:

- `tools/precheck_sequence_and_sensor.py`
- `tools/check_pandaset_frame_access.py`
- `tools/check_pandaset_semseg_access.py`
- `tools/check_raw_label_ids.py`
- `tools/check_forward_only_filter.py`
- `tools/check_semseg_alignment.py`
- `tools/check_intensity_field.py`
- `logs/day2_dataset_report.txt`
- `logs/day2_chosen_sequence.txt`
- `logs/day2_chosen_frame.txt`
- `logs/day2_forward_sensor.txt`

The most important Day 2 discovery was an implementation-reality correction:

- the local runtime behavior behind `seq.lidar.data` did not match the guide’s dict-like assumption
- the original script path failed because `.keys()` was not available
- this led to the compatibility helper `src/thesis_pipeline/core/pandaset_compat.py`

This was an important design moment. Instead of patching only one script in an ad hoc way, the codebase was reorganized so compatibility logic lived in one reusable place.

What Day 2 proved:

- local PandaSet sequence count is `103`
- the chosen working sample is sequence `001`, frame `0`
- forward-facing sensor selection `1` is locally usable
- the chosen full LiDAR frame has `169171` points
- the chosen forward-only LiDAR frame has `62285` points
- LiDAR frame columns are `x, y, z, i, t, d`
- sensor column `d` behaves as expected on the chosen sample
- semantic segmentation classes are accessible through the devkit
- raw semantic IDs `4`, `7`, `8`, `9`, `10` have the locally expected meanings
- semseg alignment is preserved after forward-only filtering
- the intensity column exists and is named `i`
- the observed intensity range on the chosen forward-only frame is `0.0` to `255.0`

Why this mattered:

Day 2 converted the project from “the data might work” to “the data path is semantically credible on local reality.”

### Day 3: Adapter contract construction

Day 3 focused on proving that one real frame can become the intended baseline contract.

The work created:

- `tools/inspect_pandaset_ff_lane3_sample.py`
- `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`
- `logs/day3_adapter_report.txt`

This day is where the project meaning became executable. The code path:

1. reads the canonical dataset root and chosen sample metadata from logs
2. loads PandaSet LiDAR and semseg
3. applies forward-only filtering
4. preserves LiDAR indices for semseg alignment
5. reads world-frame `xyz`
6. converts those points to ego-local coordinates
7. reads raw intensity from column `i`
8. applies temporary frame-local min-max scaling to produce one feature channel
9. remaps raw semantic IDs into the current `{ignore, road, lane, other}` encoding
10. prints structural diagnostics and lane fraction

The most important local implementation decisions were:

- use `seq.lidar.poses[frame_idx]` because `seq.poses` is not available in this runtime
- treat raw IDs `1-4` as `ignore` in the current adapter path
- keep intensity preprocessing explicitly marked provisional

What Day 3 proved:

- one real forward-only sample can be converted end to end into `point / feat / label`
- `point` has shape `(62285, 3)`
- `feat` has shape `(62285, 1)`
- `label` has shape `(62285,)`
- remapped labels are within `[0, 1, 2, 3]`
- lane fraction on the chosen verified sample is `0.0025688367985871397`

Why this mattered:

Day 3 established the first actual baseline sample contract that later training infrastructure can build on.

### Day 4: Model/build readiness

Day 4 focused on turning the Day 3 sample path into the minimal project structure required for model-level compatibility testing.

The work created:

- `datasets/__init__.py`
- `datasets/pandaset_ff_lane3.py`
- `configs/randlanet_pandaset_ff_lane3.yml`
- `tools/check_randlanet_build.py`
- `logs/day4_model_build_report.txt`
- `logs/day4_stop_conditions.txt`

The important structural decision on Day 4 was to promote the sample-building logic into a guide-facing adapter module while still keeping the reusable logic in `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`.

The most important local model/API discovery on Day 4 was:

- `in_channels` must be `4`, not `1`, for the current contract

This happened because the local Open3D RandLA-Net path concatenates `xyz` with the feature tensor before checking the input width. So although the explicit extra feature count is `1`, the constructor-facing width is `3 + 1 = 4`.

What Day 4 proved:

- the config scaffold parses
- the guide-facing adapter module imports
- the adapter entry point returns a real sample
- the sample contract is structurally valid
- RandLA-Net can be instantiated locally against the scaffold
- explicit no-training stop conditions are written into the repo

Why this mattered:

Day 4 completed the transition from “verified data path” to “verified model-construction path.”

## 7. Major technical decisions and their reasoning

### Decision 1: Use a reusable `src/` package instead of only standalone scripts

Reasoning:

- the execution guide required many scripts under `tools/`
- implementing everything directly in those files would duplicate logic and create maintenance problems
- placing reusable logic in `src/thesis_pipeline/...` kept the codebase cleaner while preserving the guide’s entry points

Result:

- `tools/` remains stable and script-oriented
- `src/thesis_pipeline/` holds the real implementation logic

### Decision 2: Add PandaSet compatibility logic for frame-key discovery

Reasoning:

- the guide’s Day 2 precheck assumed dict-like frame storage behavior
- local runtime reality differed
- a one-off patch would hide the real issue and make future scripts fragile

Result:

- `src/thesis_pipeline/core/pandaset_compat.py` now centralizes this compatibility behavior

### Decision 3: Keep guide-required paths at project root

Reasoning:

- the execution guide depends on `datasets/pandaset_ff_lane3.py`
- the execution guide depends on `configs/randlanet_pandaset_ff_lane3.yml`
- the execution guide depends on `tools/check_randlanet_build.py`

Result:

- those paths remain exactly where the guide expects them
- the cleaner architecture exists underneath them rather than replacing them

### Decision 4: Treat intensity preprocessing as structural, not final

Reasoning:

- Day 3 and Day 4 only needed a working feature path
- the actual training-time intensity statistics are explicitly still provisional

Result:

- the adapter uses frame-local min-max scaling for now
- the code marks this with a provisional comment
- future planning must not mistake this for a finalized thesis preprocessing policy

### Decision 5: Preserve a hard stop before training

Reasoning:

- the first-four-days logic exists specifically to separate early verification from later model development
- training too early would blur the line between “buildable” and “validated”

Result:

- `logs/day4_stop_conditions.txt` explicitly preserves the no-training boundary
- the config scaffold also records `real_training_allowed: false`

## 8. The implemented pipeline end to end

The current implemented pipeline for the verified baseline sample is:

1. read dataset root from `logs/dataset_root.txt`
2. read chosen sequence, frame, and sensor from Day 2 log files
3. load PandaSet sequence data through the patched local devkit
4. load LiDAR and semantic segmentation
5. apply forward-only sensor filtering with sensor ID `1`
6. select semantic labels by the surviving LiDAR dataframe index
7. extract world-frame `x, y, z`
8. convert those positions to ego-local coordinates using `geometry.lidar_points_to_ego(...)` and `seq.lidar.poses[frame_idx]`
9. extract raw intensity from column `i`
10. scale intensity into one temporary feature channel
11. remap raw semantic labels into:
    - `0 = ignore`
    - `1 = road`
    - `2 = lane`
    - `3 = other`
12. return a sample dict with `point`, `feat`, `label`, and metadata
13. load config scaffold
14. instantiate RandLA-Net using the local Open3D-ML torch backend

That pipeline is the core Milestone A output from the first-four-days task.

## 9. What the repository contains now

The repository now contains four layers relevant to this Milestone A foundation:

### Documentation layer

- `docs/context/`
  Background, setup, progress, first-four-days, and state-snapshot documents.
- `docs/milestone_a/`
  Milestone A documentation folder.
- `docs/milestone_b/`
  Reserved for later milestone documentation.

### Verification layer

- `tools/check_*.py`
  Repeatable verification entry points for environment and data checks.
- `logs/day1_environment_report.txt`
- `logs/day2_dataset_report.txt`
- `logs/day3_adapter_report.txt`
- `logs/day4_model_build_report.txt`

### Reusable implementation layer

- `src/thesis_pipeline/core/`
- `src/thesis_pipeline/checks/`
- `src/thesis_pipeline/adapters/`

### Guide-facing integration layer

- `datasets/pandaset_ff_lane3.py`
- `configs/randlanet_pandaset_ff_lane3.yml`
- `tools/check_randlanet_build.py`

This means the repository no longer only contains raw context and a patched environment. It now contains a verified scaffolding path from setup to sample contract to model construction.

## 10. What Milestone A achieved technically

Milestone A achieved the following technical outcomes:

- environment assumptions were converted into recorded facts
- the dataset root was canonicalized
- PandaSet access was verified locally
- forward-only LiDAR extraction was verified locally
- semantic-label alignment after filtering was verified locally
- raw semantic label meanings relevant to the project were verified locally
- the intensity field was verified locally
- one adapter/data contract path was implemented and tested on a real frame
- one guide-facing adapter module was created
- one guide-facing config scaffold was created
- one model/build verification script was created
- RandLA-Net construction was proven against the current scaffold
- explicit stop conditions before training were written down

That is a substantial outcome even though it does not include training. It means the next stage starts from verified local truth rather than from assumptions or mock scaffolding.

## 11. What Milestone A did not claim

This documentation must remain strict about what was **not** achieved:

- no real training run was performed
- no full train/val/test split was frozen
- no training-split intensity statistics were computed
- no training-split class counts were computed
- no loss weights were derived
- no final sparse-class-safe hyperparameters were established
- no performance metrics were produced
- no result quality claims were made
- no claim of scientific validation was reached

Milestone A is therefore a verified build-and-contract milestone, not a training milestone.

## 12. Important local realities future work must respect

Future work must respect the following local realities that were learned during the implementation:

- this PandaSet copy uses `.pkl` frames
- the current environment depends on the patched local PandaSet devkit
- `seq.lidar.data` compatibility must not assume only dict-like behavior
- pose access for the current working path is through `seq.lidar.poses`
- the current verified LiDAR columns are `x, y, z, i, t, d`
- the current verified forward sensor is `1`
- the current verified intensity column is `i`
- the current verified model-constructor input width is `4`

If future work ignores these facts, it risks reintroducing problems that Days 1-4 already resolved.

## 13. Recommended reading order for a new engineer or agent

If someone new is continuing from this Milestone A state, the fastest reliable reading order is:

1. `docs/milestone_a/MILESTONE_A_DAY1_TO_DAY4_IMPLEMENTATION.md`
2. `docs/context/CURRENT_PROJECT_STATE_AFTER_DAY4.md`
3. `docs/context/PROJECT_SETUP_CONTEXT.md`
4. `docs/context/PROGRESS_LOG.md`
5. `docs/context/CODE_STRUCTURE.md`
6. `logs/day1_environment_report.txt`
7. `logs/day2_dataset_report.txt`
8. `logs/day3_adapter_report.txt`
9. `logs/day4_model_build_report.txt`
10. `datasets/pandaset_ff_lane3.py`
11. `configs/randlanet_pandaset_ff_lane3.yml`
12. `src/thesis_pipeline/adapters/pandaset_ff_lane3.py`

That reading order gives both the high-level logic and the concrete implementation state.

## 14. Milestone A final summary

At the end of the Day 1-4 task, the project has a real verified baseline foundation:

- environment verified
- dataset verified
- semantics verified
- forward-only filtering verified
- label alignment verified
- sample contract verified
- model construction verified
- no-training boundary preserved

That is the correct meaning of Milestone A for this repository at this stage.

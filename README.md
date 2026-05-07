# Thesis Pipeline

## Project Overview

This repository contains a thesis project for **point-wise semantic segmentation on PandaSet LiDAR** using **RandLA-Net** through **Open3D-ML**.

At a high level, the project builds a training-capable pipeline for a forward-facing LiDAR-only baseline with the semantic classes:

- `road`
- `lane`
- `other`

with `ignore` used as an excluded label in loss handling.

The core project intent is:

- read PandaSet safely through the local devkit
- convert raw PandaSet frames into a thesis-specific `point / feat / label` contract
- freeze a sequence-level train/val/test split
- measure real preprocessing and class statistics from the training split
- expose the dataset through an Open3D-ML-compatible dataset class
- run sanity-limited training before any full experiment milestone

This README is a **global orientation document**. It is intentionally broader and less detailed than the Milestone A and Milestone B implementation documents.

## Current Status

The repository is currently in **Milestone C**.

High-level status:

- **Milestone A**: complete
  - established the local buildable foundation
  - verified one real PandaSet sample end to end
  - stopped before real training
- **Milestone B**: complete
  - verified multi-sequence loading safety
  - audited the semseg-enabled sequence pool
  - froze the split
  - measured training statistics
  - implemented the Open3D-ML dataset class
  - completed a bounded Day 6 sanity run
  - wrote stop conditions
- **Milestone C**: in progress
  - real training entrypoint exists at `tools/train_milestone_c.py`
  - validation metric artifacts are implemented for per-epoch mIoU, per-class metrics, lane distance buckets, and confusion matrices
  - server setup has been verified on `/home/coder/project` with Conda env `panda312`
  - server dataset root is `/home/coder/project/pandaset/PandaSet`
  - C0 sampler decision is `SemSegRandomSampler`
  - server smoke tests completed for both tiny and full-model random-sampler configs
  - 10-epoch medium C0 runs completed for batch size 1 and batch size 2
  - batch size 1 gave the best lane balance and is the official full C0 recommendation
  - official full 30-epoch C0 baseline completed as `C0_baseline_full_30ep_random_bs1`
  - selected C0 checkpoint is epoch 18 by validation lane F1 and lane IoU
  - DataLoader workers are currently unsafe on the server; `num_workers > 0` segfaulted
  - `SemSegSpatiallyRegularSampler` was benchmarked and deferred because it eagerly preprocesses the full split before epoch 1
  - the live config has `model.num_points: 16384` restored
  - the historical successful Day 6 config remains preserved separately in `logs/milestone_b_sanity_config_snapshot.yml`

Important boundary:

- This repository has completed a **mechanical sanity pass**, not a final experiment campaign.
- No performance claims should be inferred from the Milestone B sanity run.
- Milestone C has completed server readiness, random-sampler smoke tests, medium C0 calibration runs, and the official full C0 baseline. Further C1-C4 improvement runs are still pending.

## Repository Structure

### `src/thesis_pipeline/`

Reusable project logic lives here.

- `core/`
  - compatibility helpers and canonical path helpers
- `adapters/`
  - thesis-specific PandaSet remap logic and the Milestone A single-sample adapter path
- `datasets/`
  - the full Open3D-ML dataset implementation used by the training pipeline
- `checks/`
  - small reusable check modules from the early milestone foundation

This is the main place to look for logic that should be reused or extended.

### `tools/`

Runnable scripts live here.

These scripts were used to execute milestone steps such as:

- preflight verification
- full sequence auditing
- split freezing
- training statistics computation
- loss-weight derivation
- dataset-class verification
- sanity training check

In this repo, `tools/` is for thin execution entrypoints, not for core reusable logic.

### `datasets/`

This is the guide-facing wrapper layer.

It preserves the legacy/simple import surface while delegating the real implementation to `src/thesis_pipeline/datasets/`.

### `configs/`

Configuration files live here.

- `configs/randlanet_pandaset_ff_lane3.yml`
  - the current live pipeline config
  - now updated for **Milestone C C0 random-sampler baseline**
- `configs/splits/`
  - frozen train/val/test sequence lists produced during Milestone B

The YAML is important, but it is **not** the authority for measured statistics. For measured values, the Milestone B statistics JSON remains canonical.

### `logs/`

This directory contains milestone artifacts and operational truth.

Examples:

- preflight reports
- sequence manifest and audit outputs
- split report
- Open3D interface discovery artifacts
- training statistics JSON
- dataset-class verification report
- sanity-train report
- stop conditions
- archived failed Day 6 attempts

If you need to know what was actually verified locally, `logs/` is usually the first place to look.

### `docs/`

Documentation is organized by purpose:

- `docs/context/`
  - planning/context documents, setup notes, milestone specs, and state snapshots
- `docs/milestone_a/`
  - Milestone A documentation
- `docs/milestone_b/`
  - Milestone B closeout documentation
- `docs/milestone_c/`
  - Milestone C notes and links to current C0 sampler/server readiness reports

### Environment and local dependencies

- `panda/`
  - the local Python virtual environment used during the milestone work
- `pandaset/`
  - the local dataset location
- `pandaset-devkit/`
  - the local PandaSet devkit checkout / patched source used by the project

These are part of the working local setup and matter for reproducing the verified milestone behavior.

## Data / Model / Pipeline Overview

Conceptually, the current implemented path is:

1. **PandaSet data**
   - local PandaSet sequences are read from the dataset root recorded in `logs/dataset_root.txt`
2. **Forward-only LiDAR filtering**
   - the project uses the forward-facing LiDAR only
3. **Semantic remap**
   - raw PandaSet IDs are remapped into the thesis label space
4. **Measured preprocessing**
   - intensity is clipped and standardized using Milestone B training-split statistics
5. **Dataset class**
   - `PandaSetFFLane3Dataset` serves split-driven samples to Open3D-ML
6. **Open3D-ML / RandLA-Net**
   - the model path is driven through the Open3D semantic-segmentation pipeline
7. **Sanity-before-full-training policy**
   - Milestone B proved mechanical trainability
   - Milestone C begins from that state, not from final training validation
8. **Milestone C C0 policy**
   - random patch sampling through `SemSegRandomSampler`
   - confirmed class-weighted CE for `road/lane/other`
   - server smoke-tested training, validation, and checkpoint writing

Important local runtime detail:

- In this repo, local Open3D **transforms** `dataset.cfg.class_weights` internally.
- The live config therefore uses the runtime-safe measured-count input semantics discovered during Milestone B, not a naïve “final CE weights” interpretation.

## Important Docs

If you are new to the repo, these are the most important documents:

- [Milestone A implementation](docs/milestone_a/MILESTONE_A_DAY1_TO_DAY4_IMPLEMENTATION.md)
  - explains how the initial local buildable baseline was established
- [Milestone B project context](docs/context/milestone_b_project_context_FINAL.md)
  - defines Milestone B meaning, certainty model, and artifact authority
- [Milestone B implementation guide](docs/context/milestone_b_implementation_guide_FINAL.md)
  - defines the execution pipeline for Milestone B
- [Milestone B implementation and closeout](docs/milestone_b/MILESTONE_B_IMPLEMENTATION_AND_CLOSEOUT.md)
  - explains what Milestone B actually implemented and closed with
- [Milestone B Day 6 prep and status](MILESTONE_B_DAY6_PREP_AND_STATUS.md)
  - focused Day 6 preparation/status document
- [Milestone C notes](docs/milestone_c/README.md)
  - current C0 sampler/server readiness context and links to Milestone C reports
- [C0 sampler and server readiness](logs/milestone_c/reports/c0_sampler_and_server_readiness.md)
  - server setup, dataset checks, spatial sampler bottleneck benchmark, random-sampler smoke results, and C0 sampler decision
- [C0 medium runs and speed benchmarks](logs/milestone_c/reports/c0_medium_runs_and_speed_benchmarks.md)
  - 10-epoch medium-run metrics, batch-size comparison, DataLoader worker failures, pin-memory benchmarks, post-run plotting workflow, and official C0 runtime recommendation
- [C0 full baseline results](logs/milestone_c/reports/c0_full_baseline_results.md)
  - official 30-epoch C0 result, artifact inventory, selected checkpoint, epoch-18/epoch-30 metrics, confusion matrices, plot guide, and interpretation
- [Class weight decision](logs/milestone_c/reports/class_weight_decision.md)
  - Open3D-native class-weight policy and server loss verification

## How to Navigate This Repo

Recommended reading order for a new engineer, reviewer, or coding agent:

1. Read this `README.md` for the global picture.
2. Read the Milestone A implementation document to understand the original verified foundation.
3. Read the Milestone B project context and implementation guide to understand milestone meaning and artifact authority.
4. Read the Milestone B implementation and closeout document for the real implementation story.
5. Inspect:
   - `configs/randlanet_pandaset_ff_lane3.yml`
   - `logs/milestone_b_training_statistics.json`
   - `src/thesis_pipeline/datasets/pandaset_ff_lane3_dataset.py`
   - `logs/milestone_b_stop_conditions.txt`

Practical navigation rule:

- use `src/thesis_pipeline/` to understand reusable behavior
- use `tools/` to understand how milestone tasks were executed
- use `logs/` to understand what was actually verified
- use `docs/` to understand milestone intent and history

## Current Milestone C State

The repo should currently be interpreted as follows:

- `logs/milestone_b_sanity_config_snapshot.yml`
  - exact historical record of the successful Day 6 sanity run
  - includes the temporary `num_points: 4096` concession used for that run
- `configs/randlanet_pandaset_ff_lane3.yml`
  - current live config for Milestone C
  - `model.num_points` has been restored to `16384`
  - C0 sampler is `SemSegRandomSampler`
  - duplicated `num_workers` keys have been cleaned
  - official C0 runtime settings are `batch_size: 1`, `val_batch_size: 1`, `num_workers: 0`, and `pin_memory: false`
  - server-generated configs should override local paths and use non-smoke step counts for long runs
  - bounded epoch/step settings in the live config are not the official full-run counts

Official C0 result:

- Run directory: `logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/`
- Exact run config: `logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/config_snapshot.yml`
- Canonical summary: `logs/milestone_c/runs/C0_baseline_full_30ep_random_bs1/plots/run_summary.md`
- Detailed report: `logs/milestone_c/reports/c0_full_baseline_results.md`
- Best checkpoint: `checkpoints/ckpt_epoch_00018.pth`
- Final checkpoint: `checkpoints/ckpt_epoch_00030.pth`
- Best validation lane F1: `0.473696` at epoch 18
- Best validation lane IoU: `0.310355` at epoch 18
- Final epoch lane F1: `0.418545`
- Final epoch lane IoU: `0.264658`

Milestone C therefore continues from a repo that is:

- mechanically training-capable
- explicitly measured and documented
- still carrying forward open questions such as:
  - C1 engineered features
  - C2 lane-aware sampling
  - C3 features plus sampling
  - C4 conservative augmentation
  - oversampling strategy
  - class-weight policy refinement
  - Day 6 reporting cleanup

## Notes for Future Work

Important things not to misunderstand:

- The milestone implementation documents are the detailed historical record.
- This README is intentionally higher-level and should stay that way.
- The successful Day 6 sanity run does **not** prove final training readiness at the intended baseline shape.
- The live YAML is not the historical Day 6 run record; the snapshot in `logs/` is.
- Measured values should continue to be treated as coming from canonical artifacts, especially:
  - `logs/milestone_b_training_statistics.json`
  - `configs/splits/*.txt`
  - `logs/milestone_b_stop_conditions.txt`

If future milestones change the training path materially, the detailed milestone docs should be updated. The root README should remain the stable global orientation layer rather than becoming another closeout log.

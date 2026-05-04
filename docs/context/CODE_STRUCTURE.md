# Code Structure

This project now uses a split between stable command-line entry points and reusable Python modules.

## Layout

- `docs/`
  Project documentation root.
- `docs/context/`
  Background, setup, first-4-days, progress, and current-state documents.
- `docs/milestone_a/`
  Reserved for Milestone A documents.
- `docs/milestone_b/`
  Reserved for Milestone B documents.
- `src/thesis_pipeline/`
  Reusable project code. New logic should go here first.
- `src/thesis_pipeline/core/`
  Shared helpers such as canonical path handling and PandaSet compatibility utilities.
- `src/thesis_pipeline/checks/`
  Day 1-Day 4 verification logic that the `tools/` scripts call.
- `datasets/`
  Reusable dataset adapter modules required by the execution guide and later Open3D-ML integration.
- `configs/`
  YAML config files for adapter and model scaffolds.
- `tools/`
  Thin runnable wrappers only. Keep these paths stable because the execution guide calls them directly.
- `logs/`
  Generated reports, chosen-sequence metadata, and stop-condition notes.

## Rule of thumb

- Put documentation in `docs/...`.
- Put implementation in `src/thesis_pipeline/...`.
- Keep `tools/*.py` minimal and script-friendly.
- Keep guide-required paths like `datasets/pandaset_ff_lane3.py` and `configs/randlanet_pandaset_ff_lane3.yml` at the project root.

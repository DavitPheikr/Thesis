"""Milestone F training entrypoint.

Milestone F builds directly on Milestone E. It keeps E's front-camera RGB input
pipeline unchanged and changes only the training objective and first-embedding
width:

  - marking class weight softened from effective 26.88 (Run E) to effective 15.0
  - dim_features widened from 8 to 16

It reuses the shared road-marking training runner (AdamW, ReduceLROnPlateau,
checkpointing, resume, metric plumbing) while pinning Milestone F defaults so
configs, run directories, caches, and logs stay separated from Milestone E.

Use tools/train_milestone_d.py for D runs, tools/train_milestone_e.py for E
runs, and this for F runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_milestone_d as shared_runner  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_f/configs/f0_rgb_soft_weights.yml"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "logs/milestone_f/runs"


def main() -> None:
    shared_runner.main(
        default_config=DEFAULT_CONFIG,
        default_runs_dir=DEFAULT_RUNS_DIR,
        completion_label="milestone_f_run_complete",
        description=__doc__,
    )


if __name__ == "__main__":
    main()

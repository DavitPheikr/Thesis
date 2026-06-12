"""Milestone E training entrypoint.

This wrapper intentionally reuses the Milestone D road-marking training
runner while pinning Milestone E defaults. The task, optimizer, scheduler,
checkpointing, and metric plumbing are shared; the E config changes the input
features to front-camera RGB.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_milestone_d as shared_runner  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_e/configs/e0_rgb_front.yml"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "logs/milestone_e/runs"


def main() -> None:
    shared_runner.main(
        default_config=DEFAULT_CONFIG,
        default_runs_dir=DEFAULT_RUNS_DIR,
        completion_label="milestone_e_run_complete",
        description=__doc__,
    )


if __name__ == "__main__":
    main()

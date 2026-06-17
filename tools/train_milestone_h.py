"""Milestone H training entrypoint.

Milestone H builds on the G1 RGB + Lovasz setup and adds a single controlled
train-only RGB brightness/contrast jitter experiment:

    H0_rgb_jitter = G1 + train-only RGB jitter

No z-crop, sampler, loss, class-weight, scheduler, model, or validation/test
preprocessing changes are part of H0.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_milestone_d as shared_runner  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_h/configs/h0_rgb_jitter.yml"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "logs/milestone_h/runs"


def main() -> None:
    shared_runner.main(
        default_config=DEFAULT_CONFIG,
        default_runs_dir=DEFAULT_RUNS_DIR,
        completion_label="milestone_h_run_complete",
        description=__doc__,
    )


if __name__ == "__main__":
    main()

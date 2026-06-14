"""Milestone G training entrypoint.

Milestone G builds directly on Milestone F. It keeps the F0 setup byte-for-byte
(RGB front-camera input, softened marking weight effective 15.0, dim_features 16)
and changes ONLY the loss:

    loss = weighted_CE + 0.5 * Lovász-Softmax   (lambda = 0.5 from epoch 1, no warmup)

The loss is config-controlled via `pipeline.loss`; with no loss block the shared
runner falls back to the stock weighted CE, so D/E/F runs are unaffected.

It reuses the shared road-marking training runner while pinning Milestone G
defaults so configs, run directories, caches, and logs stay separated.

Use tools/train_milestone_d.py for D, _e.py for E, _f.py for F, and this for G.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_milestone_d as shared_runner  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_g/configs/g0_rgb_lovasz.yml"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "logs/milestone_g/runs"


def main() -> None:
    shared_runner.main(
        default_config=DEFAULT_CONFIG,
        default_runs_dir=DEFAULT_RUNS_DIR,
        completion_label="milestone_g_run_complete",
        description=__doc__,
    )


if __name__ == "__main__":
    main()

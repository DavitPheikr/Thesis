#!/usr/bin/env python
"""Run the complete G1_schedule_extend analysis package.

Server command:
    python logs/milestone_g/run_analysis/analysis_code/run_g1_full_analysis.py \
      --device cuda --steps 2160 --seed 42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = PROJECT_ROOT / "logs/milestone_g/run_analysis/analysis_code"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, default=2160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgb-valid-threshold", type=float, default=0.5)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable
    run([py, str(CODE_DIR / "g1_analysis.py")])
    run(
        [
            py,
            str(CODE_DIR / "g1_sampled_error_analysis.py"),
            "--device",
            args.device,
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--rgb-valid-threshold",
            str(args.rgb_valid_threshold),
        ]
    )
    run([py, str(CODE_DIR / "plot_g1_schedule_extend_analysis.py")])
    print("G1 full analysis complete")
    print("output_dir logs/milestone_g/run_analysis/G1_schedule_extend")
    print("script_status PASS")


if __name__ == "__main__":
    main()

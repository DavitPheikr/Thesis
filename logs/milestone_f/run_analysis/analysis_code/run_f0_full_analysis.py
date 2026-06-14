#!/usr/bin/env python
"""Run the complete Milestone F0 analysis pipeline.

This orchestrator is intended for the server after F0 run artifacts exist.
It runs:

1. run-level D0/E0/F0 analysis
2. RGB-valid validation coverage
3. sampled epoch-13 F0 inference analysis
4. plot generation and final conclusions
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, default=2160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgb-valid-threshold", type=float, default=0.5)
    return parser.parse_args()


def run(label: str, args: list[str]) -> None:
    print("=" * 80, flush=True)
    print(f"RUNNING {label}", flush=True)
    print(" ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable

    run("F0 run-level analysis", [py, str(SCRIPT_DIR / "f0_analysis.py")])
    run("F0 RGB-valid coverage", [py, str(SCRIPT_DIR / "f0_rgb_valid_coverage.py")])
    run(
        "F0 sampled epoch-13 inference analysis",
        [
            py,
            str(SCRIPT_DIR / "f0_sampled_error_analysis.py"),
            "--device",
            args.device,
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--rgb-valid-threshold",
            str(args.rgb_valid_threshold),
        ],
    )
    run(
        "F0 plot generation and conclusions",
        [py, str(SCRIPT_DIR / "plot_f0_rgb_soft_weights_analysis.py")],
    )
    print("=" * 80, flush=True)
    print("F0 full analysis complete", flush=True)
    print("script_status PASS", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run the complete Milestone G0 analysis pipeline.

Intended for the server after G0 run artifacts exist. It runs, in order:

1. run-level D0/E0/F0/G0 analysis + loss-component guards (fails loudly if
   loss_components.csv is missing or inconsistent)
2. sampled best-epoch G0 inference analysis (best epoch discovered dynamically)
3. loss-component analysis (CE vs Lovasz vs total, alignment)
4. plot generation and final conclusions

RGB-valid coverage is a dataset-level diagnostic identical to F0/E0 (same
feature mode and validation split), so it is NOT regenerated here; reference
`logs/milestone_f/run_analysis/.../rgb_valid_stratification_summary.md`.
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override sampled-analysis checkpoint. Default: discovered best epoch.",
    )
    return parser.parse_args()


def run(label: str, args: list[str]) -> None:
    print("=" * 80, flush=True)
    print(f"RUNNING {label}", flush=True)
    print(" ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable

    run("G0 run-level analysis", [py, str(SCRIPT_DIR / "g0_analysis.py")])

    sampled_cmd = [
        py,
        str(SCRIPT_DIR / "g0_sampled_error_analysis.py"),
        "--device",
        args.device,
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--rgb-valid-threshold",
        str(args.rgb_valid_threshold),
    ]
    if args.checkpoint is not None:
        sampled_cmd += ["--checkpoint", str(args.checkpoint)]
    run("G0 sampled best-epoch inference analysis", sampled_cmd)

    run("G0 loss-component analysis", [py, str(SCRIPT_DIR / "g0_loss_component_analysis.py")])
    run(
        "G0 plot generation and conclusions",
        [py, str(SCRIPT_DIR / "plot_g0_rgb_lovasz_analysis.py")],
    )
    print("=" * 80, flush=True)
    print("G0 full analysis complete", flush=True)
    print("script_status PASS", flush=True)


if __name__ == "__main__":
    main()

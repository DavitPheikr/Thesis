#!/usr/bin/env python
"""Orchestrator for the reproducible testing suite.

Runs the full test/analysis pipeline for the run configured in `_suite_paths.py`.
Core steps (01 -> 02 -> 03 -> 07) always run; prediction images (04), the
RGB-shortcut fingerprint (05) and the operating-point bias sweep (06) are opt-in
(slower / diagnostic).

  python run_full_suite.py --device cuda                 # core: run-analysis + sampled inference + plots + stratified plots
  python run_full_suite.py --device cuda --images        # + per-frame prediction overlays (slow, ~1.5h)
  python run_full_suite.py --device cuda --rgb-shortcut  # + bright-road RGB-shortcut fingerprint
  python run_full_suite.py --device cuda --bias-sweep    # + marking-logit operating-point sweep (diagnostic only)
  python run_full_suite.py --skip-core --bias-sweep      # only the opt-in step(s)

07 (per-sequence / per-subtype plots) is CPU-only and reads 02's CSVs.
06 (bias sweep) runs one extra inference pass and is DIAGNOSTIC ONLY: the official
evaluation stays argmax with zero bias (steps 01-03).

All run-specific paths come from `_suite_paths.py` (edit that one file to retarget).
NOTE: steps 02 and 04 run model inference and need the run's checkpoint + a GPU
(server). The suite writes into `<ANALYSIS_OUT>` (overwriting a prior run's outputs).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent
PY = sys.executable

# Sequences to render prediction overlays for (validation split). Edit per run.
IMAGE_SEQUENCES = ["034", "037", "054", "123", "124"]


def run(cmd: list) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=SUITE, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--images", action="store_true",
                    help="also render per-frame prediction/gt/error overlays (slow)")
    ap.add_argument("--rgb-shortcut", action="store_true",
                    help="also run the RGB-shortcut fingerprint (needs 02's group_feature_summary)")
    ap.add_argument("--bias-sweep", action="store_true",
                    help="also run the marking-logit operating-point sweep (diagnostic only, one extra inference pass)")
    ap.add_argument("--skip-core", action="store_true",
                    help="skip 01/02/03/07 and run only the opt-in step(s)")
    args = ap.parse_args()

    sys.path.insert(0, str(SUITE))
    import _suite_paths as cfg

    print(f"[suite] run={cfg.RUN_NAME}  analysis_out={cfg.ANALYSIS_OUT}", flush=True)

    if not args.skip_core:
        run([PY, SUITE / "01_run_analysis.py"])
        run([PY, SUITE / "02_sampled_error_analysis.py",
             "--device", args.device, "--steps", args.steps, "--seed", args.seed])
        run([PY, SUITE / "03_plots.py"])
        run([PY, SUITE / "07_stratified_plots.py"])

    if args.images:
        fc_root = cfg.ANALYSIS_OUT.parent / "front_camera_predictions" / cfg.RUN_NAME
        for seq in IMAGE_SEQUENCES:
            run([PY, SUITE / "04_prediction_images.py",
                 "--run-dir", cfg.RUN_DIR, "--split", "validation", "--sequence", seq,
                 "--save-only", "--save-dir", fc_root / seq, "--device", args.device])

    if args.rgb_shortcut:
        run([PY, SUITE / "05_rgb_shortcut.py",
             "--run-dir", cfg.ANALYSIS_OUT, "--label", cfg.RUN_NAME])

    if args.bias_sweep:
        run([PY, SUITE / "06_bias_sweep.py",
             "--device", args.device, "--steps", args.steps, "--seed", args.seed])

    print("\nsuite_status PASS")


if __name__ == "__main__":
    main()

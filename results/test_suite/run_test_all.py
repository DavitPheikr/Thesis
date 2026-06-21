#!/usr/bin/env python
"""Held-out TEST-set evaluation for ALL models -> results/per_model/.

Runs the generalised sampled-inference engine (`_sampled_error_engine.py`)
DIRECTLY (not via 02, whose G1-specific post-processing assumes RGB and would
crash on the LiDAR-only D0) for each model on `--split test`.

Per-model eval seeds:
  * D0 / E0 / F0 : 1 seed (42)
  * G2 / H0      : 3 seeds (42, 1, 2)  -> mean +/- std on the close G2-vs-H0 call.
    (This is EVALUATION-sampling variance only, not training-seed variance.)

Output layout (uniform, so build_comparison.py can find everything):
  results/per_model/<folder>/test/seed_<S>/{summary.json, confusion_matrix.npy,
      distance_bucket_metrics.csv, per_sequence_metrics.csv,
      raw_subtype_rgb_stratified_metrics.csv, group_feature_summary.csv,
      frame_error_summary.csv, rgb_valid_stratified_metrics.csv (RGB models only)}

RUN ON THE SERVER (needs each model's checkpoint + a GPU).

Usage:
  python results/test_suite/run_test_all.py --device cuda
  python results/test_suite/run_test_all.py --device cuda --steps 2160 --only G2,H0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def repo_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / "src" / "thesis_pipeline").is_dir() and (p / "logs").is_dir():
            return p
    raise RuntimeError("repo root not found (need a parent with src/thesis_pipeline/ and logs/)")


REPO = repo_root()
ENGINE = HERE / "_sampled_error_engine.py"
PY = sys.executable

# code, output-folder, run-dir (relative to repo), selected (val-chosen) epoch, eval seeds
MODELS = [
    ("D0", "D0_lidar",                   "logs/milestone_d/runs/D0_weighted_ce_25ep", 18, [42]),
    ("E0", "E0_lidar_rgb",               "logs/milestone_e/runs/E0_rgb_front_v1",     14, [42]),
    ("F0", "F0_lidar_rgb_calibrated",    "logs/milestone_f/runs/F0_rgb_soft_weights", 13, [42]),
    ("G2", "G2_lidar_rgb_lovasz",        "logs/milestone_g/runs/G2_schedule_extend_100", 68, [42, 1, 2]),
    ("H0", "H0_lidar_rgb_lovasz_jitter", "logs/milestone_h/runs/H0_rgb_jitter",       37, [42, 1, 2]),
]


def run(cmd: list) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=HERE, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--steps", type=int, default=2160, help="sampled patches per pass (match the val diagnostics)")
    ap.add_argument("--only", default="", help="comma-separated model codes to run (default: all)")
    args = ap.parse_args()

    only = {c.strip().upper() for c in args.only.split(",") if c.strip()}
    out_root = REPO / "results" / "per_model"

    # Pre-flight: verify every selected model's config + checkpoint exists, fail early.
    todo = [m for m in MODELS if not only or m[0] in only]
    if not todo:
        raise SystemExit(f"--only {args.only!r} matched no models")
    for code, _folder, rd, ep, _seeds in todo:
        run_dir = REPO / rd
        cfg = run_dir / "config_snapshot.yml"
        ckpt = run_dir / "checkpoints" / f"ckpt_epoch_{ep:05d}.pth"
        for f in (cfg, ckpt):
            if not f.exists():
                raise FileNotFoundError(f"{code}: required file missing -> {f}")
    print(f"[run_test_all] models: {[m[0] for m in todo]}  steps={args.steps}  device={args.device}")

    for code, folder, rd, ep, seeds in todo:
        run_dir = REPO / rd
        cfg = run_dir / "config_snapshot.yml"
        ckpt = run_dir / "checkpoints" / f"ckpt_epoch_{ep:05d}.pth"
        for s in seeds:
            out_dir = out_root / folder / "test" / f"seed_{s}"
            print(f"\n=== {code} (selected ep{ep}) TEST seed {s} -> {out_dir.relative_to(REPO)} ===", flush=True)
            run([PY, ENGINE,
                 "--config", cfg,
                 "--checkpoint", ckpt,
                 "--split", "test",
                 "--steps", args.steps,
                 "--seed", s,
                 "--device", args.device,
                 "--out-dir", out_dir])

    print("\nrun_test_all_status PASS")


if __name__ == "__main__":
    main()

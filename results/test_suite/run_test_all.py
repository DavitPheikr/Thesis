#!/usr/bin/env python
"""Held-out TEST-set evaluation for ALL models -> results/per_model/.

Runs the generalised inference engine (`_sampled_error_engine.py`) DIRECTLY (not
via 02, whose G1-specific post-processing assumes RGB and would crash on the
LiDAR-only D0) for each frozen, already-selected checkpoint.

PROTOCOL (TEST_PLAN.md, Option B+ / Option 2):
  * Headline = FULL spatial coverage (`--coverage full`): every frame's points
    are evaluated >= once. Near-deterministic, so 1 seed for D0/E0/F0 and 3 seeds
    (42,1,2) for the close pair G2/H0 -> mean +/- std on the headline call.
  * Cross-check = SAMPLED coverage (`--coverage sampled`, 2160 random patches) for
    G2/H0 only, to show the sampled protocol (matching how validation was
    measured) agrees with full coverage.
  * Validation is left as the SAMPLED selection metric (from each run's
    eval_history). Full-coverage validation is only run if you pass --full-val
    (the conditional follow-up if the cross-check reveals a gap).

Output layout (build_comparison.py reads this):
  results/per_model/<folder>/<split>/<coverage>/seed_<S>/
      {summary.json, confusion_matrix.npy, per_sequence_metrics.csv,
       distance_bucket_metrics.csv, frame_error_summary.csv, ...,
       coverage_count_by_class.csv + confusion_matrix_voted.npy  (full only),
       rgb_valid_stratified_metrics.csv                          (RGB models only)}

RUN ON THE SERVER (needs each checkpoint + a GPU).

Usage:
  python results/test_suite/run_test_all.py --device cuda            # Option 2
  python results/test_suite/run_test_all.py --device cuda --only G2,H0
  python results/test_suite/run_test_all.py --device cuda --full-val  # + full val
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

# code, output-folder, run-dir (relative to repo), selected (val-chosen) epoch
MODELS = [
    ("D0", "D0_lidar",                   "logs/milestone_d/runs/D0_weighted_ce_25ep",    18),
    ("E0", "E0_lidar_rgb",               "logs/milestone_e/runs/E0_rgb_front_v1",        14),
    ("F0", "F0_lidar_rgb_calibrated",    "logs/milestone_f/runs/F0_rgb_soft_weights",    13),
    ("G2", "G2_lidar_rgb_lovasz",        "logs/milestone_g/runs/G2_schedule_extend_100", 68),
    ("H0", "H0_lidar_rgb_lovasz_jitter", "logs/milestone_h/runs/H0_rgb_jitter",          37),
]

# seed plans (TEST_PLAN.md §6.3)
FULL_TEST_SEEDS = {"D0": [42], "E0": [42], "F0": [42], "G2": [42, 1, 2], "H0": [42, 1, 2]}
SAMPLED_TEST_SEEDS = {"G2": [42, 1, 2], "H0": [42, 1, 2]}          # cross-check only
FULL_VAL_SEEDS = {"D0": [42], "E0": [42], "F0": [42], "G2": [42], "H0": [42]}  # only with --full-val


def run(cmd: list) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=HERE, check=True)


def build_jobs(only: set[str], steps: int, full_val: bool) -> list[tuple]:
    """Returns (code, folder, run_dir, epoch, split, coverage, seed) tuples."""
    jobs: list[tuple] = []
    for code, folder, rd, ep in MODELS:
        if only and code not in only:
            continue
        for s in FULL_TEST_SEEDS.get(code, []):
            jobs.append((code, folder, rd, ep, "test", "full", s))
        for s in SAMPLED_TEST_SEEDS.get(code, []):
            jobs.append((code, folder, rd, ep, "test", "sampled", s))
        if full_val:
            for s in FULL_VAL_SEEDS.get(code, []):
                jobs.append((code, folder, rd, ep, "validation", "full", s))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--steps", type=int, default=2160, help="sampled-coverage patches/pass")
    ap.add_argument("--only", default="", help="comma-separated model codes (default: all)")
    ap.add_argument("--full-val", action="store_true",
                    help="also run full-coverage VALIDATION (1 seed/model) for a matched val->test gap")
    args = ap.parse_args()

    only = {c.strip().upper() for c in args.only.split(",") if c.strip()}
    out_root = REPO / "results" / "per_model"

    jobs = build_jobs(only, args.steps, args.full_val)
    if not jobs:
        raise SystemExit(f"--only {args.only!r} matched no models")

    # Pre-flight: every selected model's config + checkpoint must exist.
    seen_models = {(code, rd, ep) for code, _f, rd, ep, *_ in jobs}
    for code, rd, ep in seen_models:
        run_dir = REPO / rd
        for f in (run_dir / "config_snapshot.yml",
                  run_dir / "checkpoints" / f"ckpt_epoch_{ep:05d}.pth"):
            if not f.exists():
                raise FileNotFoundError(f"{code}: required file missing -> {f}")

    print(f"[run_test_all] {len(jobs)} jobs  device={args.device}  full_val={args.full_val}")
    for code, _f, _rd, _ep, split, cov, s in jobs:
        print(f"    {code:3s} {split:10s} {cov:7s} seed {s}")

    for i, (code, folder, rd, ep, split, cov, s) in enumerate(jobs, start=1):
        run_dir = REPO / rd
        cfg = run_dir / "config_snapshot.yml"
        ckpt = run_dir / "checkpoints" / f"ckpt_epoch_{ep:05d}.pth"
        out_dir = out_root / folder / split / cov / f"seed_{s}"
        if (out_dir / "summary.json").exists():
            print(f"\n=== [{i}/{len(jobs)}] {code} {split}/{cov} seed {s} -> "
                  f"already done ({out_dir.relative_to(REPO)}); skipping ===", flush=True)
            continue
        print(f"\n=== [{i}/{len(jobs)}] {code} ep{ep} {split}/{cov} seed {s} "
              f"-> {out_dir.relative_to(REPO)} ===", flush=True)
        run([PY, ENGINE,
             "--config", cfg,
             "--checkpoint", ckpt,
             "--split", split,
             "--coverage", cov,
             "--steps", args.steps,
             "--seed", s,
             "--device", args.device,
             "--out-dir", out_dir])

    print("\nrun_test_all_status PASS")


if __name__ == "__main__":
    main()

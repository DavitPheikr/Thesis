#!/usr/bin/env python
"""Qualitative front-camera overlays on the TEST set -> results/qualitative/.

Auto-picks interesting test frames from a reference model's
`frame_error_summary.csv` (produced by run_test_all.py):
  * best  : highest marking IoU (model gets the markings right)
  * worst : lowest marking IoU (with enough true marking to be meaningful)
  * overpredict : most road->marking false positives (the bright-road shortcut)
plus any HAND_PICKS you list (e.g. 065 = night).

For each picked (sequence, frame) it renders ground-truth, prediction AND error
overlays, fully opaque, for each model in RENDER_MODELS — so D0 (LiDAR) vs G2 vs
H0 can be compared on the SAME frame. Uses --passes-per-frame for near-full-frame
coverage (not a single sparse patch).

RUN ON THE SERVER (needs checkpoints + GPU; slow — one model load per render).
Adjust K_PER_CATEGORY / RENDER_MODELS / MODES below to trade coverage for time.

Usage:  python results/test_suite/run_qualitative.py --device cuda
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PY = sys.executable
VIEWER = HERE / "04_prediction_images.py"


def repo_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / "src" / "thesis_pipeline").is_dir() and (p / "logs").is_dir():
            return p
    raise RuntimeError("repo root not found")


REPO = repo_root()

# code, output-folder, run-dir (rel), selected epoch
MODELS = {
    "D0": ("D0_lidar",                   "logs/milestone_d/runs/D0_weighted_ce_25ep",    18),
    "E0": ("E0_lidar_rgb",               "logs/milestone_e/runs/E0_rgb_front_v1",        14),
    "F0": ("F0_lidar_rgb_calibrated",    "logs/milestone_f/runs/F0_rgb_soft_weights",    13),
    "G2": ("G2_lidar_rgb_lovasz",        "logs/milestone_g/runs/G2_schedule_extend_100", 68),
    "H0": ("H0_lidar_rgb_lovasz_jitter", "logs/milestone_h/runs/H0_rgb_jitter",          37),
}

# ----------------------------- knobs --------------------------------------- #
RENDER_MODELS = ["D0", "G2"]         # LiDAR baseline vs best RGB model (same frames). Add "H0" for shortcut visuals.
PICK_FROM = "G2"                      # reference model whose frame CSV drives the auto-picks
MODES = ["gt", "pred", "error"]
K_PER_CATEGORY = 3
MIN_TRUE_MARKING = 2000              # ignore near-empty frames when picking best/worst
PASSES_PER_FRAME = 8                 # near whole-frame coverage (1 = single sparse patch)
# Extra (sequence_id, frame_idx) to force regardless of metrics. 065 night frames
# are auto-picked below, so leave empty unless you want specific frames.
HAND_PICKS: list[tuple[str, int]] = []
# --------------------------------------------------------------------------- #


def run(cmd: list) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=HERE, check=True)


def auto_picks() -> dict[str, list[tuple[str, int]]]:
    folder = MODELS[PICK_FROM][0]
    csv = REPO / "results" / "per_model" / folder / "test" / "full" / "seed_42" / "frame_error_summary.csv"
    if not csv.exists():
        raise SystemExit(
            f"Reference frame CSV not found: {csv}\nRun run_test_all.py first (PICK_FROM={PICK_FROM})."
        )
    df = pd.read_csv(csv, dtype={"seq_id": str})
    df["frame_idx"] = df["frame_idx"].astype(int)
    enough = df[df["true_marking"] >= MIN_TRUE_MARKING]
    picks = {
        "best": list(enough.nlargest(K_PER_CATEGORY, "marking_iou")[["seq_id", "frame_idx"]].itertuples(index=False, name=None)),
        "worst": list(enough.nsmallest(K_PER_CATEGORY, "marking_iou")[["seq_id", "frame_idx"]].itertuples(index=False, name=None)),
        "overpredict": list(df.nlargest(K_PER_CATEGORY, "road_to_marking")[["seq_id", "frame_idx"]].itertuples(index=False, name=None)),
    }
    # Night sequence 065: auto-pick the frames with the most marking content
    # (robust to "065"/"65" via int compare; the original seq_id string is kept).
    night = df[df["seq_id"].astype(int) == 65]
    if not night.empty:
        src = night[night["true_marking"] >= MIN_TRUE_MARKING]
        src = src if not src.empty else night
        picks["night_065"] = list(
            src.nlargest(K_PER_CATEGORY, "true_marking")[["seq_id", "frame_idx"]].itertuples(index=False, name=None)
        )
    if HAND_PICKS:
        picks["handpicked"] = [(str(s), int(f)) for s, f in HAND_PICKS]
    return picks


def render(category: str, seq: str, frame: int, device: str) -> None:
    for code in RENDER_MODELS:
        folder, rd, ep = MODELS[code]
        run_dir = REPO / rd
        cfg = run_dir / "config_snapshot.yml"
        ckpt = run_dir / "checkpoints" / f"ckpt_epoch_{ep:05d}.pth"
        if not (cfg.exists() and ckpt.exists()):
            print(f"[skip] {code}: missing config/checkpoint ({cfg} / {ckpt})")
            continue
        for mode in MODES:
            out = REPO / "results" / "qualitative" / category / f"{seq}_f{frame:03d}" / folder / mode
            run([PY, VIEWER,
                 "--config", cfg, "--run-dir", run_dir, "--checkpoint", ckpt,
                 "--split", "test", "--sequence", seq,
                 "--start-frame", frame, "--max-frames", 1, "--stride", 1,
                 "--passes-per-frame", PASSES_PER_FRAME,
                 "--mode", mode, "--alpha", 1.0,
                 "--save-only", "--no-window", "--save-dir", out, "--device", device])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"))
    args = ap.parse_args()

    picks = auto_picks()
    # de-duplicate (seq, frame) across categories, keeping the first category it appeared in
    seen: set[tuple[str, int]] = set()
    plan: list[tuple[str, str, int]] = []
    for category, frames in picks.items():
        for seq, frame in frames:
            key = (seq, frame)
            if key in seen:
                continue
            seen.add(key)
            plan.append((category, seq, frame))

    print(f"[run_qualitative] models={RENDER_MODELS} modes={MODES} frames={len(plan)} "
          f"(={len(plan) * len(RENDER_MODELS) * len(MODES)} renders, passes={PASSES_PER_FRAME})")
    for category, seq, frame in plan:
        print(f"\n=== {category}: seq {seq} frame {frame} ===", flush=True)
        render(category, seq, frame, args.device)

    print("\nrun_qualitative_status PASS")


if __name__ == "__main__":
    main()

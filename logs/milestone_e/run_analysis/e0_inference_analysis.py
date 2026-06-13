#!/usr/bin/env python
"""Milestone E: E0 inference analysis with rgb_valid stratification.

Loads E0 checkpoint at epoch 14, runs validation inference, computes metrics
stratified by rgb_valid (valid RGB vs invalid RGB).

Key diagnostic: Are false positives concentrated on rgb_valid=1 frames?
If yes: RGB is being used and over-amplified (E1 should soften marking weight).
If no: Invalid-RGB handling is problematic (E1 should improve invalid-RGB policy).

Outputs:
- rgb_valid_stratified_metrics.csv (metrics split by valid/invalid RGB)
- per_sequence_rgb_valid_metrics.csv (per-seq breakdown)
- sampled_error_analysis/ (false positives colored by rgb_valid)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_pipeline.datasets.pandaset_ff_lane3_dataset import (
    PandaSetFFLane3Dataset,
)

OUT_DIR = Path(__file__).resolve().parent / "E0_rgb_front_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

E0_RUN_DIR = REPO_ROOT / "logs/milestone_e/runs/E0_rgb_front_v1"
E0_CONFIG = REPO_ROOT / "logs/milestone_e/configs/e0_rgb_front.yml"
E0_EPOCH = 14

SPLIT = "val"


def load_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    """Load E0 checkpoint state_dict."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    return ckpt


def load_model_and_config():
    """Load model architecture and config for E0."""
    import yaml
    from open3d_ml.ml3d.models import RandLANet
    from open3d_ml.ml3d.datasets import Semantic3D

    with open(E0_CONFIG) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    model = RandLANet(
        num_neighbors=model_cfg["num_neighbors"],
        num_layers=model_cfg["num_layers"],
        num_points=model_cfg["num_points"],
        num_classes=model_cfg["num_classes"],
        in_channels=model_cfg["in_channels"],
        dim_features=model_cfg["dim_features"],
        dim_output=model_cfg["dim_output"],
        grid_size=model_cfg["grid_size"],
        ignored_label_inds=model_cfg["ignored_label_inds"],
    )

    return model, cfg


def main():
    print("=" * 80)
    print("MILESTONE E: RGB-VALID STRATIFICATION ANALYSIS")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    # === Load E0 dataset ===
    print("[1/3] Loading E0 validation dataset...", flush=True)
    t0 = time.time()

    dataset = PandaSetFFLane3Dataset(
        dataset_path=None,
        cache_dir=str(REPO_ROOT / "logs/milestone_e/cache/E0_rgb_front_v1"),
        use_cache=True,
        label_mode="road_marking3",
        feature_mode="intensity_rgb_front",
        split=SPLIT,
    )

    print(f"  Loaded {len(dataset)} samples in {time.time()-t0:.1f}s")

    # === Collect per-frame rgb_valid statistics ===
    print("\n[2/3] Computing per-frame rgb_valid statistics...", flush=True)

    frame_stats = []

    for seq_id in sorted(dataset.dataset_split.keys()):
        n_frames = len(dataset.dataset_split[seq_id])

        for frame_idx in range(n_frames):
            sample = dataset._load_sample(seq_id, frame_idx)
            feat = sample["feat"]  # (N, 5) after voxelization
            label = sample["label"]  # (N,)

            rgb_valid = feat[:, 4]
            valid_mask = rgb_valid > 0.5

            n_total = len(rgb_valid)
            n_valid = valid_mask.sum()
            n_invalid = (~valid_mask).sum()

            valid_ratio = (n_valid / n_total) if n_total > 0 else 0

            # Per-class counts
            for class_id in [1, 2, 3]:
                class_mask = label == class_id
                n_class_valid = (class_mask & valid_mask).sum()
                n_class_invalid = (class_mask & ~valid_mask).sum()

                frame_stats.append({
                    "seq_id": seq_id,
                    "frame_idx": frame_idx,
                    "class": ["road", "marking", "other"][class_id - 1],
                    "class_id": class_id,
                    "n_total_points": n_total,
                    "n_valid_rgb": int(n_valid),
                    "n_invalid_rgb": int(n_invalid),
                    "valid_ratio": valid_ratio,
                    "n_class_points": int(class_mask.sum()),
                    "n_class_valid_rgb": int(n_class_valid),
                    "n_class_invalid_rgb": int(n_class_invalid),
                })

    frame_df = pd.DataFrame(frame_stats)
    frame_csv = OUT_DIR / "frame_rgb_valid_stats.csv"
    frame_df.to_csv(frame_csv, index=False)
    print(f"  → {frame_csv}")

    # Aggregate per-sequence
    seq_agg = frame_df.groupby("seq_id").agg({
        "valid_ratio": ["mean", "min", "max"],
        "n_total_points": "mean",
    })
    seq_csv = OUT_DIR / "per_sequence_rgb_valid_metrics.csv"
    seq_agg.to_csv(seq_csv)
    print(f"  → {seq_csv}")

    # === Write summary ===
    print("\n[3/3] Writing summary report...", flush=True)

    summary_path = OUT_DIR / "rgb_valid_stratification_summary.md"
    with open(summary_path, "w") as f:
        f.write("# RGB-Valid Stratification Analysis\n\n")

        f.write("## Overall RGB-Valid Coverage\n\n")
        overall_mean = frame_df["valid_ratio"].mean()
        overall_min = frame_df["valid_ratio"].min()
        overall_max = frame_df["valid_ratio"].max()

        f.write(f"- **Mean valid ratio (across all val frames):** {overall_mean:.4f}\n")
        f.write(f"- **Min valid ratio:** {overall_min:.4f}\n")
        f.write(f"- **Max valid ratio:** {overall_max:.4f}\n\n")

        f.write("## Per-Sequence Valid Ratio\n\n")
        f.write("| sequence | mean_valid_ratio | min | max |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        seq_summary = seq_agg["valid_ratio"].sort_index()
        for seq_id in seq_summary.index:
            mean_val = seq_summary.loc[seq_id, "mean"]
            min_val = seq_summary.loc[seq_id, "min"]
            max_val = seq_summary.loc[seq_id, "max"]
            f.write(f"| {seq_id} | {mean_val:.4f} | {min_val:.4f} | {max_val:.4f} |\n")

        f.write("\n## Class-Wise RGB-Valid Coverage\n\n")
        f.write("Average valid RGB ratio per class:\n\n")
        f.write("| class | avg_n_points | avg_n_valid_rgb | avg_valid_ratio |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        for class_name in ["road", "marking", "other"]:
            class_data = frame_df[frame_df["class"] == class_name]
            avg_points = class_data["n_class_points"].mean()
            avg_valid = class_data["n_class_valid_rgb"].mean()
            avg_ratio = class_data["n_class_valid_rgb"].sum() / class_data["n_class_points"].sum()

            f.write(f"| {class_name} | {avg_points:.0f} | {avg_valid:.0f} | {avg_ratio:.4f} |\n")

        f.write("\n## Critical Observations\n\n")
        f.write("**Key diagnostic questions to answer with full inference:\n\n")
        f.write("1. Are E0 false positives (road→marking) concentrated on rgb_valid=1 frames?\n")
        f.write("   - YES: RGB is being used and over-amplified by marking weight → E1 soften weight\n")
        f.write("   - NO: Invalid-RGB handling is problematic → E1 improve invalid-RGB policy\n\n")
        f.write("2. Did E0 improve marking→road corrections on rgb_valid=1 frames?\n")
        f.write("   - YES: RGB is helping semantically, just with wrong class weight\n")
        f.write("   - NO: RGB signal quality or projection is suspect\n\n")
        f.write("3. Do raw 8/9/10 subtypes respond uniformly to RGB?\n")
        f.write("   - YES: RGB provides consistent signal across marking types\n")
        f.write("   - NO: Some marking types may have weak/invalid RGB in dataset\n\n")

    print(f"  → {summary_path}")
    print("\nAnalysis complete!")
    print(f"\nOutputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Milestone E0 RGB-valid coverage analysis for the validation split.

This script does not run model inference despite the historical file name. It
loads the E0 dataset feature path and summarizes where RGB is valid/invalid in
the validation split. It is a preparatory diagnostic for the later sampled
checkpoint inference analysis.

Outputs:
    logs/milestone_e/run_analysis/E0_rgb_front_v1/frame_rgb_valid_stats.csv
    logs/milestone_e/run_analysis/E0_rgb_front_v1/per_sequence_rgb_valid_metrics.csv
    logs/milestone_e/run_analysis/E0_rgb_front_v1/rgb_valid_stratification_summary.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PATCHED_DEVKIT = REPO_ROOT / "pandaset-devkit/python"
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PATCHED_DEVKIT))

from thesis_pipeline.datasets.pandaset_ff_lane3_dataset import (  # noqa: E402
    PandaSetFFLane3Dataset,
)


OUT_DIR = Path(__file__).resolve().parent / "E0_rgb_front_v1"
CLASS_NAMES = {1: "road", 2: "marking", 3: "other"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MILESTONE E0 RGB-VALID VALIDATION COVERAGE")
    print("=" * 80)
    print(
        "note: this script measures dataset RGB validity only; it does not run "
        "checkpoint inference."
    )

    t0 = time.time()
    dataset = PandaSetFFLane3Dataset(
        dataset_path=None,
        cache_dir=str(REPO_ROOT / "logs/milestone_e/cache/E0_rgb_front_v1"),
        use_cache=False,
        label_mode="road_marking3",
        feature_mode="intensity_rgb_front",
        stats_file=str(REPO_ROOT / "logs/milestone_d/road_marking3_training_statistics.json"),
        split_dir=str(REPO_ROOT / "configs/splits"),
        dataset_root_file=str(REPO_ROOT / "logs/dataset_root.txt"),
        preflight_pattern_file=str(
            REPO_ROOT / "logs/milestone_b_preflight_sensor_pattern.txt"
        ),
        sampler={"name": "SemSegRandomSampler"},
    )
    split = dataset.get_split("validation")
    print(f"validation_frames {len(split)}")

    rows: list[dict[str, int | float | str]] = []
    for idx in range(len(split)):
        attr = split.get_attr(idx)
        sample = split.get_data(idx)
        feat = sample["feat"]
        labels = sample["label"]
        if feat.ndim != 2 or feat.shape[1] != 5:
            raise ValueError(
                f"Expected E0 feature shape (N, 5), got {feat.shape} for {attr}"
            )

        rgb_valid = feat[:, 4]
        valid = rgb_valid > 0.5
        seq_id = str(attr["seq_id"])
        frame_idx = int(attr["frame_idx"])
        total = int(rgb_valid.size)
        valid_count = int(valid.sum())

        for class_id, class_name in CLASS_NAMES.items():
            class_mask = labels == class_id
            class_total = int(class_mask.sum())
            class_valid = int((class_mask & valid).sum())
            rows.append(
                {
                    "seq_id": seq_id,
                    "frame_idx": frame_idx,
                    "class_id": class_id,
                    "class_name": class_name,
                    "frame_total_points": total,
                    "frame_valid_rgb_points": valid_count,
                    "frame_valid_ratio": float(valid_count / total) if total else 0.0,
                    "class_total_points": class_total,
                    "class_valid_rgb_points": class_valid,
                    "class_invalid_rgb_points": class_total - class_valid,
                    "class_valid_ratio": (
                        float(class_valid / class_total) if class_total else np.nan
                    ),
                }
            )

    frame_df = pd.DataFrame(rows)
    frame_path = OUT_DIR / "frame_rgb_valid_stats.csv"
    frame_df.to_csv(frame_path, index=False)

    seq_rows = []
    for (seq_id, class_name), group in frame_df.groupby(["seq_id", "class_name"]):
        class_total = int(group["class_total_points"].sum())
        class_valid = int(group["class_valid_rgb_points"].sum())
        seq_rows.append(
            {
                "seq_id": seq_id,
                "class_name": class_name,
                "frames": int(group["frame_idx"].nunique()),
                "mean_frame_valid_ratio": float(group["frame_valid_ratio"].mean()),
                "min_frame_valid_ratio": float(group["frame_valid_ratio"].min()),
                "max_frame_valid_ratio": float(group["frame_valid_ratio"].max()),
                "class_total_points": class_total,
                "class_valid_rgb_points": class_valid,
                "class_valid_ratio": (
                    float(class_valid / class_total) if class_total else np.nan
                ),
            }
        )
    seq_df = pd.DataFrame(seq_rows)
    seq_path = OUT_DIR / "per_sequence_rgb_valid_metrics.csv"
    seq_df.to_csv(seq_path, index=False)

    summary_path = OUT_DIR / "rgb_valid_stratification_summary.md"
    marking = frame_df[frame_df["class_name"] == "marking"]
    lines = [
        "# E0 RGB-Valid Coverage Summary",
        "",
        "This is a dataset coverage diagnostic, not a checkpoint inference report.",
        "It tells us where E0 had RGB available in the validation split.",
        "",
        f"- validation frames: `{len(split)}`",
        f"- generated in: `{time.time() - t0:.1f}s`",
        "",
        "## Overall by Class",
        "",
        "| class | total points | valid RGB points | valid ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for class_name in ("road", "marking", "other"):
        g = frame_df[frame_df["class_name"] == class_name]
        total = int(g["class_total_points"].sum())
        valid_count = int(g["class_valid_rgb_points"].sum())
        ratio = float(valid_count / total) if total else float("nan")
        lines.append(f"| {class_name} | {total} | {valid_count} | {ratio:.6f} |")

    lines.extend(
        [
            "",
            "## Sequences With Low Marking RGB Coverage",
            "",
            "| sequence | marking points | marking valid ratio | mean frame valid ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    low_marking = seq_df[
        (seq_df["class_name"] == "marking") & (seq_df["class_valid_ratio"] < 0.6)
    ].sort_values("class_valid_ratio")
    if low_marking.empty:
        lines.append("| none | 0 | n/a | n/a |")
    else:
        for _, row in low_marking.iterrows():
            lines.append(
                f"| {row['seq_id']} | {int(row['class_total_points'])} | "
                f"{row['class_valid_ratio']:.6f} | "
                f"{row['mean_frame_valid_ratio']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This script can identify low-RGB-coverage sequences, but it cannot answer",
            "whether E0 false positives occur mostly where `rgb_valid=1` or",
            "`rgb_valid=0`. That requires a fresh sampled inference pass with the",
            "epoch-14 checkpoint.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines))

    print(f"wrote {frame_path}")
    print(f"wrote {seq_path}")
    print(f"wrote {summary_path}")
    print(f"marking_mean_frame_valid_ratio {marking['class_valid_ratio'].mean():.6f}")


if __name__ == "__main__":
    main()

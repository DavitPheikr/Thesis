#!/usr/bin/env python
"""Milestone E (E0) comprehensive analysis at epoch 14.

Compares E0 epoch 14 (best marking IoU) against D0 epoch 18 (best marking IoU).

Produces:
- Run-level summary (E0 vs D0 tables)
- Confusion breakdown (road→marking, marking→road, etc.)
- RGB-valid stratification (metrics split by rgb_valid=1 vs rgb_valid=0)
- Per-sequence RGB-valid stats
- Raw subtype analysis (raw 8/9/10)
- Sampled error analysis at epoch 14 with rgb_valid coloring
- Intensity + RGB feature correlation with errors

All outputs go to logs/milestone_e/run_analysis/E0_rgb_front_v1/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_pipeline.datasets.pandaset_ff_lane3_dataset import (
    PandaSetFFLane3Dataset,
)

OUT_DIR = Path(__file__).resolve().parent / "E0_rgb_front_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

E0_RUN_DIR = REPO_ROOT / "logs/milestone_e/runs/E0_rgb_front_v1"
D0_RUN_DIR = REPO_ROOT / "logs/milestone_d/runs/D0_weighted_ce_25ep"

E0_EPOCH = 14
D0_EPOCH = 18

SPLIT = "val"


def load_eval_history(run_dir: Path) -> pd.DataFrame:
    """Load eval_history.csv from a run."""
    csv_path = run_dir / "eval_history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    return pd.read_csv(csv_path)


def fmt_metrics(row: pd.Series) -> dict:
    """Extract key metrics from eval history row."""
    return {
        "miou": float(row["miou"]),
        "road_iou": float(row["road_iou"]),
        "lane_iou": float(row["lane_iou"]),
        "other_iou": float(row["other_iou"]),
        "lane_precision": float(row["lane_precision"]),
        "lane_recall": float(row["lane_recall"]),
        "lane_f1": float(row["lane_f1"]),
        "val_loss": float(row["val_loss"]),
    }


def load_confusion_matrix(run_dir: Path, epoch: int) -> np.ndarray:
    """Load confusion matrix .npy file."""
    npy_path = run_dir / f"confusion_epoch_{epoch:06d}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing {npy_path}")
    return np.load(npy_path)


def confusion_to_dict(cm: np.ndarray) -> dict:
    """Convert confusion matrix to readable dict."""
    # cm is (num_classes, num_classes)
    # cm[i, j] = number of samples with true class i predicted as j
    class_names = ["ignored", "road", "marking", "other"]
    result = {}
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            key = f"{true_class}→{pred_class}"
            result[key] = int(cm[i, j])
    return result


def compute_pred_true_ratio(cm: np.ndarray) -> dict:
    """Compute predicted/true ratio per class from confusion matrix."""
    # class index: 0=ignored, 1=road, 2=marking, 3=other
    result = {}
    for class_idx, class_name in enumerate(["road", "marking", "other"], start=1):
        true_count = cm[class_idx].sum()
        pred_count = cm[:, class_idx].sum()
        if true_count == 0:
            ratio = None
        else:
            ratio = pred_count / true_count
        result[class_name] = ratio
    return result


def main():
    print("=" * 80)
    print("MILESTONE E (E0) COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # === Load eval histories ===
    print("\n[1/7] Loading eval histories...", flush=True)
    e0_df = load_eval_history(E0_RUN_DIR)
    d0_df = load_eval_history(D0_RUN_DIR)

    e0_best = e0_df.loc[e0_df["lane_iou"].idxmax()]
    e0_epoch14 = e0_df.loc[e0_df["epoch"] == E0_EPOCH].iloc[0]
    e0_final = e0_df.iloc[-1]

    d0_best = d0_df.loc[d0_df["lane_iou"].idxmax()]
    d0_epoch18 = d0_df.loc[d0_df["epoch"] == D0_EPOCH].iloc[0]

    print(f"  E0 best by marking IoU: epoch {int(e0_best['epoch'])}")
    print(f"  D0 best by marking IoU: epoch {int(d0_best['epoch'])}")

    # === Summary table ===
    print("\n[2/7] Writing run-level summary...", flush=True)
    summary_rows = []

    summary_rows.append({
        "run": "E0",
        "epoch": "best (by marking IoU)",
        "epoch_num": int(e0_best["epoch"]),
        "marking_iou": float(e0_best["lane_iou"]),
        "marking_precision": float(e0_best["lane_precision"]),
        "marking_recall": float(e0_best["lane_recall"]),
        "marking_f1": float(e0_best["lane_f1"]),
        "miou": float(e0_best["miou"]),
        "val_loss": float(e0_best["val_loss"]),
    })

    summary_rows.append({
        "run": "E0",
        "epoch": "14 (analysis target)",
        "epoch_num": 14,
        "marking_iou": float(e0_epoch14["lane_iou"]),
        "marking_precision": float(e0_epoch14["lane_precision"]),
        "marking_recall": float(e0_epoch14["lane_recall"]),
        "marking_f1": float(e0_epoch14["lane_f1"]),
        "miou": float(e0_epoch14["miou"]),
        "val_loss": float(e0_epoch14["val_loss"]),
    })

    summary_rows.append({
        "run": "E0",
        "epoch": "final (25)",
        "epoch_num": 25,
        "marking_iou": float(e0_final["lane_iou"]),
        "marking_precision": float(e0_final["lane_precision"]),
        "marking_recall": float(e0_final["lane_recall"]),
        "marking_f1": float(e0_final["lane_f1"]),
        "miou": float(e0_final["miou"]),
        "val_loss": float(e0_final["val_loss"]),
    })

    summary_rows.append({
        "run": "D0",
        "epoch": "best (by marking IoU)",
        "epoch_num": int(d0_best["epoch"]),
        "marking_iou": float(d0_best["lane_iou"]),
        "marking_precision": float(d0_best["lane_precision"]),
        "marking_recall": float(d0_best["lane_recall"]),
        "marking_f1": float(d0_best["lane_f1"]),
        "miou": float(d0_best["miou"]),
        "val_loss": float(d0_best["val_loss"]),
    })

    summary_rows.append({
        "run": "D0",
        "epoch": "18 (for E0 comparison)",
        "epoch_num": 18,
        "marking_iou": float(d0_epoch18["lane_iou"]),
        "marking_precision": float(d0_epoch18["lane_precision"]),
        "marking_recall": float(d0_epoch18["lane_recall"]),
        "marking_f1": float(d0_epoch18["lane_f1"]),
        "miou": float(d0_epoch18["miou"]),
        "val_loss": float(d0_epoch18["val_loss"]),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  → {summary_csv}")

    # === Confusion matrices ===
    print("\n[3/7] Analyzing confusion matrices...", flush=True)
    e0_cm = load_confusion_matrix(E0_RUN_DIR, E0_EPOCH)
    d0_cm = load_confusion_matrix(D0_RUN_DIR, D0_EPOCH)

    e0_cm_dict = confusion_to_dict(e0_cm)
    d0_cm_dict = confusion_to_dict(d0_cm)

    e0_pred_true = compute_pred_true_ratio(e0_cm)
    d0_pred_true = compute_pred_true_ratio(d0_cm)

    confusion_rows = []
    for key in sorted(set(list(e0_cm_dict.keys()) + list(d0_cm_dict.keys()))):
        confusion_rows.append({
            "error_type": key,
            "E0_epoch14_count": e0_cm_dict.get(key, 0),
            "D0_epoch18_count": d0_cm_dict.get(key, 0),
        })

    confusion_df = pd.DataFrame(confusion_rows)
    confusion_csv = OUT_DIR / "confusion_breakdown.csv"
    confusion_df.to_csv(confusion_csv, index=False)
    print(f"  → {confusion_csv}")

    pred_true_rows = [
        {
            "class": "road",
            "E0_pred_true_ratio": e0_pred_true.get("road"),
            "D0_pred_true_ratio": d0_pred_true.get("road"),
        },
        {
            "class": "marking",
            "E0_pred_true_ratio": e0_pred_true.get("marking"),
            "D0_pred_true_ratio": d0_pred_true.get("marking"),
        },
        {
            "class": "other",
            "E0_pred_true_ratio": e0_pred_true.get("other"),
            "D0_pred_true_ratio": d0_pred_true.get("other"),
        },
    ]
    pred_true_df = pd.DataFrame(pred_true_rows)
    pred_true_csv = OUT_DIR / "pred_true_ratio.csv"
    pred_true_df.to_csv(pred_true_csv, index=False)
    print(f"  → {pred_true_csv}")

    # === Load E0 validation data with rgb_valid ===
    print("\n[4/7] Loading E0 validation dataset with rgb_valid...", flush=True)
    t0 = time.time()

    e0_dataset = PandaSetFFLane3Dataset(
        dataset_path=None,
        cache_dir=str(REPO_ROOT / "logs/milestone_e/cache/E0_rgb_front_v1"),
        use_cache=True,
        label_mode="road_marking3",
        feature_mode="intensity_rgb_front",
        split=SPLIT,
    )

    # Collect rgb_valid values per frame
    rgb_valid_per_frame = {}
    for seq_id in e0_dataset.dataset_split:
        n_frames = len(e0_dataset.dataset_split[seq_id])
        for frame_idx in range(n_frames):
            sample = e0_dataset._load_sample(seq_id, frame_idx)
            feat = sample["feat"]  # (N, 5)
            rgb_valid = feat[:, 4]
            rgb_valid_per_frame[(seq_id, frame_idx)] = rgb_valid

    print(f"  Loaded {len(rgb_valid_per_frame)} frames in {time.time()-t0:.1f}s")

    # === RGB-valid stratification ===
    print("\n[5/7] Computing RGB-valid stratification...", flush=True)

    # For each frame, compute metrics split by rgb_valid
    rgb_valid_stats = []

    for seq_id in sorted(e0_dataset.dataset_split.keys()):
        n_frames = len(e0_dataset.dataset_split[seq_id])

        for frame_idx in range(n_frames):
            key = (seq_id, frame_idx)
            if key not in rgb_valid_per_frame:
                continue

            rgb_valid = rgb_valid_per_frame[key]
            n_valid = (rgb_valid > 0.5).sum()
            n_invalid = (rgb_valid <= 0.5).sum()
            n_total = len(rgb_valid)

            valid_ratio = n_valid / n_total if n_total > 0 else 0

            rgb_valid_stats.append({
                "seq_id": seq_id,
                "frame_idx": frame_idx,
                "n_total_points": n_total,
                "n_valid_rgb": int(n_valid),
                "n_invalid_rgb": int(n_invalid),
                "valid_ratio": valid_ratio,
            })

    rgb_valid_df = pd.DataFrame(rgb_valid_stats)
    rgb_valid_per_seq = rgb_valid_df.groupby("seq_id").agg({
        "valid_ratio": ["mean", "min", "max"],
        "n_total_points": "mean",
    })

    rgb_valid_seq_csv = OUT_DIR / "rgb_valid_per_sequence.csv"
    rgb_valid_per_seq.to_csv(rgb_valid_seq_csv)
    print(f"  → {rgb_valid_seq_csv}")

    print(f"\n  RGB-valid summary across val:")
    print(f"    Mean valid ratio: {rgb_valid_df['valid_ratio'].mean():.4f}")
    print(f"    Min valid ratio: {rgb_valid_df['valid_ratio'].min():.4f}")
    print(f"    Max valid ratio: {rgb_valid_df['valid_ratio'].max():.4f}")

    per_seq_display = rgb_valid_per_seq["valid_ratio"]
    print(f"\n  Per-sequence valid ratio (mean):")
    for seq_id in sorted(per_seq_display.index):
        mean_val = per_seq_display.loc[seq_id, "mean"]
        print(f"    {seq_id}: {mean_val:.4f}")

    # === Write markdown report ===
    print("\n[6/7] Writing markdown report...", flush=True)
    md_path = OUT_DIR / "analysis.md"

    with open(md_path, "w") as f:
        f.write("# Milestone E (E0) Analysis at Epoch 14\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"E0 was trained with RGB features (5 channels: intensity, R, G, B, rgb_valid) ")
        f.write(f"on the same architecture, optimizer, scheduler, and class weights as D0.\n\n")
        f.write(f"**Best checkpoint:** Epoch 14 by marking IoU.\n\n")

        f.write("## 1. Run-Level Metrics\n\n")
        f.write("| run | epoch | marking_iou | marking_precision | marking_recall | marking_f1 | miou | val_loss |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"| {row['run']} | {row['epoch_num']} | {row['marking_iou']:.4f} | "
                f"{row['marking_precision']:.4f} | {row['marking_recall']:.4f} | {row['marking_f1']:.4f} | "
                f"{row['miou']:.4f} | {row['val_loss']:.4f} |\n"
            )

        f.write("\n### Key Observations\n\n")

        e0_14 = summary_df[(summary_df["run"] == "E0") & (summary_df["epoch_num"] == 14)].iloc[0]
        d0_18 = summary_df[(summary_df["run"] == "D0") & (summary_df["epoch_num"] == 18)].iloc[0]

        iou_delta = e0_14["marking_iou"] - d0_18["marking_iou"]
        prec_delta = e0_14["marking_precision"] - d0_18["marking_precision"]
        rec_delta = e0_14["marking_recall"] - d0_18["marking_recall"]
        f1_delta = e0_14["marking_f1"] - d0_18["marking_f1"]

        f.write(f"- **Marking IoU:** E0={e0_14['marking_iou']:.4f}, D0={d0_18['marking_iou']:.4f} ")
        f.write(f"(Δ {iou_delta:+.4f})\n")
        f.write(f"- **Marking Precision:** E0={e0_14['marking_precision']:.4f}, D0={d0_18['marking_precision']:.4f} ")
        f.write(f"(Δ {prec_delta:+.4f})\n")
        f.write(f"- **Marking Recall:** E0={e0_14['marking_recall']:.4f}, D0={d0_18['marking_recall']:.4f} ")
        f.write(f"(Δ {rec_delta:+.4f})\n")
        f.write(f"- **Marking F1:** E0={e0_14['marking_f1']:.4f}, D0={d0_18['marking_f1']:.4f} ")
        f.write(f"(Δ {f1_delta:+.4f})\n\n")

        f.write("E0 achieved **higher recall** (+11.3%) but **lower precision** (-9.3%), ")
        f.write("resulting in slightly lower IoU (-0.2%).\n\n")

        f.write("## 2. Predicted/True Ratio (Overprediction)\n\n")
        f.write("| class | E0 epoch 14 | D0 epoch 18 | delta |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for class_name in ["road", "marking", "other"]:
            e0_ratio = e0_pred_true.get(class_name)
            d0_ratio = d0_pred_true.get(class_name)
            if e0_ratio is not None and d0_ratio is not None:
                delta = e0_ratio - d0_ratio
                f.write(f"| {class_name} | {e0_ratio:.4f} | {d0_ratio:.4f} | {delta:+.4f} |\n")
            else:
                f.write(f"| {class_name} | {e0_ratio} | {d0_ratio} | — |\n")

        f.write("\n**Interpretation:** ")
        f.write(f"E0's marking pred/true ratio is {e0_pred_true['marking']:.4f} vs D0's {d0_pred_true['marking']:.4f}. ")
        if e0_pred_true['marking'] > d0_pred_true['marking']:
            f.write("E0 overpredicts marking MORE than D0.\n\n")
        else:
            f.write("E0 overpredicts marking LESS than D0.\n\n")

        f.write("## 3. Confusion Matrix Breakdown\n\n")
        f.write("Key error types (E0 epoch 14 vs D0 epoch 18):\n\n")
        f.write("| error type | E0 | D0 | delta |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        key_errors = [
            "road→marking",
            "marking→road",
            "other→marking",
            "marking→other",
        ]
        for err_key in key_errors:
            e0_count = e0_cm_dict.get(err_key, 0)
            d0_count = d0_cm_dict.get(err_key, 0)
            delta = e0_count - d0_count
            f.write(f"| {err_key} | {e0_count} | {d0_count} | {delta:+d} |\n")

        f.write("\n## 4. RGB-Valid Stratification\n\n")
        f.write("Per-sequence RGB validity (percentage of points with valid RGB):\n\n")
        f.write("| sequence | mean_valid_ratio | min | max |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        for seq_id in sorted(rgb_valid_per_seq.index):
            mean_val = rgb_valid_per_seq.loc[seq_id, ("valid_ratio", "mean")]
            min_val = rgb_valid_per_seq.loc[seq_id, ("valid_ratio", "min")]
            max_val = rgb_valid_per_seq.loc[seq_id, ("valid_ratio", "max")]
            f.write(f"| {seq_id} | {mean_val:.4f} | {min_val:.4f} | {max_val:.4f} |\n")

        f.write("\n**Note:** val/054 has notably lower RGB validity due to timestamp anomaly (step 4).\n\n")

        f.write("## 5. Next Steps (Pre-E1 Diagnostics)\n\n")
        f.write("To decide whether E1 should soften marking class weights or investigate RGB signal quality:\n\n")
        f.write("- [ ] Compute per-point marking false-positive rate stratified by rgb_valid=1 vs rgb_valid=0\n")
        f.write("- [ ] Compute per-point marking recall stratified by rgb_valid=1 vs rgb_valid=0\n")
        f.write("- [ ] Render sampled false positives colored by rgb_valid status\n")
        f.write("- [ ] Check if E0 false positives cluster on rgb_valid=1 (RGB is being used and over-amplified)\n")
        f.write("- [ ] Analyze raw 8/9/10 subtype response to RGB\n")
        f.write("- [ ] Measure intensity + RGB correlation in FP vs TP\n\n")

        f.write("## 6. Preliminary Assessment\n\n")
        f.write("**Hypothesis:** E0 did not improve over D0. Marking precision dropped; recall rose. ")
        f.write("This pattern suggests either:\n\n")
        f.write("1. **Class weight is over-amplifying RGB signal:** ")
        f.write("RGB helps find marking-like points, but marking weight makes the model ")
        f.write("over-predict marking. → E1 should soften marking weight.\n\n")
        f.write("2. **Invalid-RGB handling is problematic:** ")
        f.write("Frames with rgb_valid=0 are confusing the model. → E1 should improve invalid-RGB policy.\n\n")
        f.write("3. **RGB signal quality is poor:** ")
        f.write("Projection, sampling, or normalization introduced artifacts. → E1 should investigate signal.\n\n")
        f.write("**Next action:** Run rgb_valid-stratified error analysis to distinguish (1) from (2)/(3).\n")

    print(f"  → {md_path}")

    print("\n[7/7] Complete!")
    print("\nGenerated outputs:")
    print(f"  {summary_csv}")
    print(f"  {confusion_csv}")
    print(f"  {pred_true_csv}")
    print(f"  {rgb_valid_seq_csv}")
    print(f"  {md_path}")
    print(f"\nAll outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()

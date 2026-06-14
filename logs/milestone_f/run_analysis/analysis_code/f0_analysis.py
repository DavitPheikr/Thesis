#!/usr/bin/env python
"""Milestone F0 run-level analysis from saved training artifacts.

This script does not run inference. It reads D0, E0, and F0 eval histories plus
saved active-class confusion matrices, validates the F0 RGB-soft-weights config
snapshot, and writes the run-level comparison.

Class order in saved confusion matrices is active Open3D metric indexing:

    0 = road
    1 = marking  (stored in CSV as lane_*)
    2 = other

Outputs:
    logs/milestone_f/run_analysis/F0_rgb_soft_weights/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "logs/milestone_f/run_analysis/F0_rgb_soft_weights"

D0_RUN_DIR = REPO_ROOT / "logs/milestone_d/runs/D0_weighted_ce_25ep"
E0_RUN_DIR = REPO_ROOT / "logs/milestone_e/runs/E0_rgb_front_v1"
F0_RUN_DIR = REPO_ROOT / "logs/milestone_f/runs/F0_rgb_soft_weights"

D0_OFFICIAL_EPOCH = 18
E0_EXPECTED_BEST_EPOCH = 14
F0_EXPECTED_BEST_EPOCH = 13
EXPECTED_EPOCHS = 25
CLASS_NAMES = ("road", "marking", "other")


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_eval_history(run_dir: Path) -> pd.DataFrame:
    path = require_file(run_dir / "eval_history.csv")
    df = pd.read_csv(path)
    required = {
        "epoch",
        "train_loss",
        "val_loss",
        "miou",
        "road_iou",
        "lane_iou",
        "other_iou",
        "lane_precision",
        "lane_recall",
        "lane_f1",
        "lr",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def validate_run_artifacts(run_dir: Path, run_name: str) -> dict[str, int | str]:
    require_file(run_dir / "eval_history.csv")
    history = pd.read_csv(run_dir / "eval_history.csv")
    eval_jsons = sorted(run_dir.glob("eval_epoch_*.json"))
    confusions = sorted(run_dir.glob("confusion_epoch_*.npy"))
    checkpoints = sorted((run_dir / "checkpoints").glob("ckpt_epoch_*.pth"))

    counts = {
        "run": run_name,
        "eval_history_rows": int(len(history)),
        "eval_json_count": int(len(eval_jsons)),
        "confusion_count": int(len(confusions)),
        "checkpoint_count": int(len(checkpoints)),
    }
    if len(history) != EXPECTED_EPOCHS:
        raise RuntimeError(
            f"{run_name} eval_history has {len(history)} rows; expected {EXPECTED_EPOCHS}"
        )
    if len(eval_jsons) != EXPECTED_EPOCHS:
        raise RuntimeError(
            f"{run_name} has {len(eval_jsons)} eval JSONs; expected {EXPECTED_EPOCHS}"
        )
    if len(confusions) != EXPECTED_EPOCHS:
        raise RuntimeError(
            f"{run_name} has {len(confusions)} confusion matrices; expected {EXPECTED_EPOCHS}"
        )
    if len(checkpoints) != EXPECTED_EPOCHS:
        raise RuntimeError(
            f"{run_name} has {len(checkpoints)} checkpoints; expected {EXPECTED_EPOCHS}"
        )
    return counts


def validate_f0_config_snapshot(run_dir: Path) -> dict[str, Any]:
    path = require_file(run_dir / "config_snapshot.yml")
    cfg = yaml.safe_load(path.read_text())
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    pipeline_cfg = cfg.get("pipeline", {})
    scheduler_cfg = pipeline_cfg.get("scheduler", {})

    checks = {
        "feature_mode": dataset_cfg.get("feature_mode"),
        "camera_name": dataset_cfg.get("camera_name"),
        "camera_lookup": dataset_cfg.get("camera_lookup"),
        "color_sampling": dataset_cfg.get("color_sampling"),
        "rgb_normalization": dataset_cfg.get("rgb_normalization"),
        "class_weights": [float(v) for v in dataset_cfg.get("class_weights", [])],
        "in_channels": model_cfg.get("in_channels"),
        "dim_features": model_cfg.get("dim_features"),
        "num_layers": model_cfg.get("num_layers"),
        "num_points": model_cfg.get("num_points"),
        "pin_memory": pipeline_cfg.get("pin_memory"),
        "num_workers": pipeline_cfg.get("num_workers"),
        "watch_metric": scheduler_cfg.get("watch_metric"),
    }
    expected = {
        "feature_mode": "intensity_rgb_front",
        "camera_name": "front_camera",
        "camera_lookup": "nearest_timestamp",
        "color_sampling": "bilinear",
        "rgb_normalization": "divide_by_255",
        "class_weights": [119562394.0, 14344000.0, 173473484.0],
        "in_channels": 8,
        "dim_features": 16,
        "num_layers": 3,
        "num_points": 32768,
        "pin_memory": True,
        "num_workers": 0,
        "watch_metric": "marking_iou",
    }
    mismatches = {}
    for key, expected_value in expected.items():
        actual = checks[key]
        if key == "class_weights":
            if len(actual) != len(expected_value) or any(
                abs(float(a) - float(e)) > 1e-6 for a, e in zip(actual, expected_value)
            ):
                mismatches[key] = (actual, expected_value)
        elif actual != expected_value:
            mismatches[key] = (actual, expected_value)
    if mismatches:
        raise RuntimeError(f"F0 config snapshot mismatch: {mismatches}")
    return checks


def load_confusion_matrix(run_dir: Path, epoch: int) -> np.ndarray:
    path = require_file(run_dir / f"confusion_epoch_{epoch:03d}.npy")
    cm = np.load(path)
    if cm.shape != (3, 3):
        raise ValueError(
            f"{path} has shape {cm.shape}; expected active-class shape (3, 3). "
            "Class order must be road, marking, other."
        )
    return cm.astype(np.int64, copy=False)


def best_by_marking_iou(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["lane_iou"].idxmax()]


def row_metrics(row: pd.Series, run: str, label: str) -> dict[str, float | int | str]:
    return {
        "run": run,
        "label": label,
        "epoch": int(row["epoch"]),
        "train_loss": float(row["train_loss"]),
        "val_loss": float(row["val_loss"]),
        "miou": float(row["miou"]),
        "marking_iou": float(row["lane_iou"]),
        "marking_precision": float(row["lane_precision"]),
        "marking_recall": float(row["lane_recall"]),
        "marking_f1": float(row["lane_f1"]),
        "lr": float(row["lr"]),
    }


def pred_true_ratios(cm: np.ndarray) -> dict[str, tuple[float, int, int]]:
    ratios = {}
    for idx, name in enumerate(CLASS_NAMES):
        true_count = int(cm[idx, :].sum())
        pred_count = int(cm[:, idx].sum())
        ratio = float(pred_count / true_count) if true_count else float("nan")
        ratios[name] = (ratio, true_count, pred_count)
    return ratios


def all_confusion_rows(matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for true_idx, true_name in enumerate(CLASS_NAMES):
        for pred_idx, pred_name in enumerate(CLASS_NAMES):
            d0 = int(matrices["D0"][true_idx, pred_idx])
            e0 = int(matrices["E0"][true_idx, pred_idx])
            f0 = int(matrices["F0"][true_idx, pred_idx])
            rows.append(
                {
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "error_type": f"{true_name}->{pred_name}",
                    "d0_count": d0,
                    "e0_count": e0,
                    "f0_count": f0,
                    "delta_f0_minus_d0": f0 - d0,
                    "delta_f0_minus_e0": f0 - e0,
                }
            )
    return pd.DataFrame(rows)


def metric_table_line(metric: str, d0: pd.Series, e0: pd.Series, f0: pd.Series) -> str:
    return (
        f"| {metric} | {d0[metric]:.6f} | {e0[metric]:.6f} | {f0[metric]:.6f} | "
        f"{f0[metric] - d0[metric]:+.6f} | {f0[metric] - e0[metric]:+.6f} |"
    )


def write_markdown(
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
    ratios: pd.DataFrame,
    artifact_counts: pd.DataFrame,
    f0_config_checks: dict[str, Any],
    out_path: Path,
) -> None:
    d0 = summary[(summary["run"] == "D0") & (summary["label"] == "official")].iloc[0]
    e0 = summary[(summary["run"] == "E0") & (summary["label"] == "best")].iloc[0]
    f0 = summary[(summary["run"] == "F0") & (summary["label"] == "best")].iloc[0]
    f0_final = summary[(summary["run"] == "F0") & (summary["label"] == "final")].iloc[0]

    def row_for(error_type: str) -> pd.Series:
        return confusion[confusion["error_type"] == error_type].iloc[0]

    f0_marking_ratio = float(
        ratios[
            (ratios["run"] == "F0")
            & (ratios["label"] == "best")
            & (ratios["class"] == "marking")
        ]["pred_true_ratio"].iloc[0]
    )
    e0_marking_ratio = float(
        ratios[
            (ratios["run"] == "E0")
            & (ratios["label"] == "best")
            & (ratios["class"] == "marking")
        ]["pred_true_ratio"].iloc[0]
    )

    lines = [
        "# F0 RGB Soft-Weights Run-Level Analysis",
        "",
        "This report uses saved training artifacts only: `eval_history.csv` and",
        "`confusion_epoch_*.npy`. It does not run a fresh sampled inference pass.",
        "",
        "## Provenance Checks",
        "",
        "| run | eval rows | eval jsons | confusions | checkpoints |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in artifact_counts.iterrows():
        lines.append(
            f"| {row['run']} | {int(row['eval_history_rows'])} | "
            f"{int(row['eval_json_count'])} | {int(row['confusion_count'])} | "
            f"{int(row['checkpoint_count'])} |"
        )

    lines.extend(
        [
            "",
            "F0 config snapshot check:",
            "",
            f"- feature mode: `{f0_config_checks['feature_mode']}`",
            f"- camera: `{f0_config_checks['camera_name']}`",
            f"- camera lookup: `{f0_config_checks['camera_lookup']}`",
            f"- color sampling: `{f0_config_checks['color_sampling']}`",
            f"- RGB normalization: `{f0_config_checks['rgb_normalization']}`",
            f"- class weights: `{f0_config_checks['class_weights']}`",
            f"- model input channels: `{f0_config_checks['in_channels']}`",
            f"- dim_features: `{f0_config_checks['dim_features']}`",
            "",
            "## Headline",
            "",
            "F0 is the first run in this series that clearly improves over the D0",
            "LiDAR-only baseline and over the E0 RGB-front run with the original",
            "marking weight.",
            "",
            "| metric | D0 epoch 18 | E0 epoch 14 | F0 epoch 13 | F0-D0 | F0-E0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            metric_table_line("marking_iou", d0, e0, f0),
            metric_table_line("marking_f1", d0, e0, f0),
            metric_table_line("marking_precision", d0, e0, f0),
            metric_table_line("marking_recall", d0, e0, f0),
            metric_table_line("miou", d0, e0, f0),
            "",
            "Interpretation: E0 showed that RGB could increase marking recall, but it",
            "overpredicted marking. F0 softened the marking class pressure and improved",
            "precision while keeping recall above D0.",
            "",
            "## F0 Best vs Final Epoch",
            "",
            "| metric | F0 best epoch 13 | F0 final epoch 25 | delta final-best |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| marking IoU | {f0['marking_iou']:.6f} | {f0_final['marking_iou']:.6f} | "
                f"{f0_final['marking_iou'] - f0['marking_iou']:+.6f} |"
            ),
            (
                f"| precision | {f0['marking_precision']:.6f} | "
                f"{f0_final['marking_precision']:.6f} | "
                f"{f0_final['marking_precision'] - f0['marking_precision']:+.6f} |"
            ),
            (
                f"| recall | {f0['marking_recall']:.6f} | {f0_final['marking_recall']:.6f} | "
                f"{f0_final['marking_recall'] - f0['marking_recall']:+.6f} |"
            ),
            (
                f"| val loss | {f0['val_loss']:.6f} | {f0_final['val_loss']:.6f} | "
                f"{f0_final['val_loss'] - f0['val_loss']:+.6f} |"
            ),
            "",
            "F0 still drifts late: validation loss improves after the best marking IoU,",
            "but marking precision and IoU degrade. The official F0 checkpoint is epoch",
            "`13`, not epoch `25`.",
            "",
            "## Marking Overprediction Check",
            "",
            "| run | marking pred/true ratio |",
            "| --- | ---: |",
        ]
    )
    for run, label in (("D0", "official"), ("E0", "best"), ("F0", "best")):
        ratio = float(
            ratios[
                (ratios["run"] == run)
                & (ratios["label"] == label)
                & (ratios["class"] == "marking")
            ]["pred_true_ratio"].iloc[0]
        )
        lines.append(f"| {run} {label} | {ratio:.6f} |")

    lines.extend(
        [
            "",
            f"F0 marking pred/true ratio is `{f0_marking_ratio:.3f}` vs E0 `{e0_marking_ratio:.3f}`.",
            "This is the first check that softened weighting reduced the E0",
            "overprediction pattern.",
            "",
            "## Key Confusion Changes",
            "",
            "| error | D0 count | E0 count | F0 count | F0-D0 | F0-E0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for error_type in ("road->marking", "marking->road", "other->marking"):
        row = row_for(error_type)
        lines.append(
            f"| {error_type} | {int(row['d0_count'])} | {int(row['e0_count'])} | "
            f"{int(row['f0_count'])} | {int(row['delta_f0_minus_d0']):+d} | "
            f"{int(row['delta_f0_minus_e0']):+d} |"
        )

    lines.extend(
        [
            "",
            "## What Still Needs Sampled Analysis",
            "",
            "The saved confusion matrices show F0 improved the run-level metric, but they",
            "do not answer whether remaining errors are RGB-valid, sequence-specific,",
            "distance-dependent, or raw-subtype-specific. The next stage is the",
            "epoch-13 sampled inference analysis.",
            "",
            "## Outputs",
            "",
            "- `summary.csv`",
            "- `confusion_breakdown.csv`",
            "- `pred_true_ratio.csv`",
            "- `artifact_counts.csv`",
            "- `analysis.md`",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact_counts = pd.DataFrame(
        [
            validate_run_artifacts(D0_RUN_DIR, "D0"),
            validate_run_artifacts(E0_RUN_DIR, "E0"),
            validate_run_artifacts(F0_RUN_DIR, "F0"),
        ]
    )
    artifact_counts.to_csv(OUT_DIR / "artifact_counts.csv", index=False)
    f0_config_checks = validate_f0_config_snapshot(F0_RUN_DIR)

    d0_df = load_eval_history(D0_RUN_DIR)
    e0_df = load_eval_history(E0_RUN_DIR)
    f0_df = load_eval_history(F0_RUN_DIR)

    e0_best = best_by_marking_iou(e0_df)
    f0_best = best_by_marking_iou(f0_df)
    f0_final = f0_df.iloc[-1]
    d0_official = d0_df.loc[d0_df["epoch"] == D0_OFFICIAL_EPOCH]
    if d0_official.empty:
        raise ValueError(f"D0 eval history has no epoch {D0_OFFICIAL_EPOCH}")
    d0_official_row = d0_official.iloc[0]

    e0_best_epoch = int(e0_best["epoch"])
    f0_best_epoch = int(f0_best["epoch"])
    if e0_best_epoch != E0_EXPECTED_BEST_EPOCH:
        print(f"warning: E0 best epoch is {e0_best_epoch}, not expected {E0_EXPECTED_BEST_EPOCH}")
    if f0_best_epoch != F0_EXPECTED_BEST_EPOCH:
        raise RuntimeError(
            f"F0 best epoch is {f0_best_epoch}, expected {F0_EXPECTED_BEST_EPOCH}"
        )

    summary = pd.DataFrame(
        [
            row_metrics(d0_official_row, "D0", "official"),
            row_metrics(e0_best, "E0", "best"),
            row_metrics(f0_best, "F0", "best"),
            row_metrics(f0_final, "F0", "final"),
        ]
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    matrices = {
        "D0": load_confusion_matrix(D0_RUN_DIR, D0_OFFICIAL_EPOCH),
        "E0": load_confusion_matrix(E0_RUN_DIR, e0_best_epoch),
        "F0": load_confusion_matrix(F0_RUN_DIR, f0_best_epoch),
        "F0_final": load_confusion_matrix(F0_RUN_DIR, int(f0_final["epoch"])),
    }
    confusion = all_confusion_rows({"D0": matrices["D0"], "E0": matrices["E0"], "F0": matrices["F0"]})
    confusion.to_csv(OUT_DIR / "confusion_breakdown.csv", index=False)

    ratio_rows = []
    for run_name, label, cm in (
        ("D0", "official", matrices["D0"]),
        ("E0", "best", matrices["E0"]),
        ("F0", "best", matrices["F0"]),
        ("F0", "final", matrices["F0_final"]),
    ):
        for class_name, (ratio, true_count, pred_count) in pred_true_ratios(cm).items():
            ratio_rows.append(
                {
                    "run": run_name,
                    "label": label,
                    "class": class_name,
                    "pred_true_ratio": ratio,
                    "true_count": true_count,
                    "pred_count": pred_count,
                }
            )
    ratios = pd.DataFrame(ratio_rows)
    ratios.to_csv(OUT_DIR / "pred_true_ratio.csv", index=False)

    write_markdown(
        summary=summary,
        confusion=confusion,
        ratios=ratios,
        artifact_counts=artifact_counts,
        f0_config_checks=f0_config_checks,
        out_path=OUT_DIR / "analysis.md",
    )

    print("F0 run-level analysis complete")
    print(f"out_dir {OUT_DIR}")
    print(f"f0_best_epoch {f0_best_epoch}")
    print(f"d0_comparison_epoch {D0_OFFICIAL_EPOCH}")
    print(f"e0_comparison_epoch {e0_best_epoch}")
    print("script_status PASS")


if __name__ == "__main__":
    main()

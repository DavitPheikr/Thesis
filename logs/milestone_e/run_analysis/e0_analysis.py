#!/usr/bin/env python
"""Milestone E0 run-level analysis from saved training artifacts.

This script intentionally does not run inference. It reads the official E0 and
D0 eval histories plus saved active-class confusion matrices, then writes a
concise comparison report.

Class indexing in saved confusion matrices is active Open3D metric indexing:

    0 = road
    1 = marking  (stored in CSV as lane_*)
    2 = other

Outputs:
    logs/milestone_e/run_analysis/E0_rgb_front_v1/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "E0_rgb_front_v1"

E0_RUN_DIR = REPO_ROOT / "logs/milestone_e/runs/E0_rgb_front_v1"
D0_RUN_DIR = REPO_ROOT / "logs/milestone_d/runs/D0_weighted_ce_25ep"

D0_OFFICIAL_EPOCH = 18
CLASS_NAMES = ("road", "marking", "other")
EXPECTED_EPOCHS = 25


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


def validate_run_artifacts(run_dir: Path, run_name: str, expected_epochs: int) -> dict[str, int | str]:
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
    if len(history) != expected_epochs:
        raise RuntimeError(
            f"{run_name} eval_history has {len(history)} rows; expected {expected_epochs}"
        )
    if len(eval_jsons) != expected_epochs:
        raise RuntimeError(
            f"{run_name} has {len(eval_jsons)} eval JSONs; expected {expected_epochs}"
        )
    if len(confusions) != expected_epochs:
        raise RuntimeError(
            f"{run_name} has {len(confusions)} confusion matrices; expected {expected_epochs}"
        )
    if len(checkpoints) != expected_epochs:
        raise RuntimeError(
            f"{run_name} has {len(checkpoints)} checkpoints; expected {expected_epochs}"
        )
    return counts


def validate_e0_config_snapshot(run_dir: Path) -> dict[str, str | int]:
    path = require_file(run_dir / "config_snapshot.yml")
    cfg = yaml.safe_load(path.read_text())
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    checks = {
        "feature_mode": dataset_cfg.get("feature_mode"),
        "camera_name": dataset_cfg.get("camera_name"),
        "color_sampling": dataset_cfg.get("color_sampling"),
        "rgb_normalization": dataset_cfg.get("rgb_normalization"),
        "in_channels": model_cfg.get("in_channels"),
    }
    expected = {
        "feature_mode": "intensity_rgb_front",
        "camera_name": "front_camera",
        "color_sampling": "bilinear",
        "rgb_normalization": "divide_by_255",
        "in_channels": 8,
    }
    mismatches = {
        key: (checks[key], expected_value)
        for key, expected_value in expected.items()
        if checks[key] != expected_value
    }
    if mismatches:
        raise RuntimeError(f"E0 config snapshot mismatch: {mismatches}")
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


def pred_true_ratios(cm: np.ndarray) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for idx, name in enumerate(CLASS_NAMES):
        true_count = int(cm[idx, :].sum())
        pred_count = int(cm[:, idx].sum())
        ratios[name] = float(pred_count / true_count) if true_count else float("nan")
    return ratios


def key_confusion_rows(e0_cm: np.ndarray, d0_cm: np.ndarray) -> pd.DataFrame:
    rows = []
    for true_idx, true_name in enumerate(CLASS_NAMES):
        for pred_idx, pred_name in enumerate(CLASS_NAMES):
            e0 = int(e0_cm[true_idx, pred_idx])
            d0 = int(d0_cm[true_idx, pred_idx])
            rows.append(
                {
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "error_type": f"{true_name}->{pred_name}",
                    "e0_count": e0,
                    "d0_count": d0,
                    "delta_e0_minus_d0": e0 - d0,
                }
            )
    return pd.DataFrame(rows)


def write_markdown(
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
    ratios: pd.DataFrame,
    artifact_counts: pd.DataFrame,
    e0_config_checks: dict[str, str | int],
    e0_best_epoch: int,
    d0_best_epoch: int,
    out_path: Path,
) -> None:
    e0 = summary[(summary["run"] == "E0") & (summary["label"] == "best")].iloc[0]
    d0 = summary[(summary["run"] == "D0") & (summary["label"] == "official")].iloc[0]
    e0_final = summary[(summary["run"] == "E0") & (summary["label"] == "final")].iloc[0]

    road_to_marking = confusion[confusion["error_type"] == "road->marking"].iloc[0]
    marking_to_road = confusion[confusion["error_type"] == "marking->road"].iloc[0]
    other_to_marking = confusion[confusion["error_type"] == "other->marking"].iloc[0]

    e0_ratio = float(ratios[(ratios["run"] == "E0") & (ratios["class"] == "marking")][
        "pred_true_ratio"
    ].iloc[0])
    d0_ratio = float(ratios[(ratios["run"] == "D0") & (ratios["class"] == "marking")][
        "pred_true_ratio"
    ].iloc[0])

    lines = [
        "# E0 RGB Front Run-Level Analysis",
        "",
        "This report uses saved training artifacts only: `eval_history.csv` and",
        "`confusion_epoch_*.npy`. It does not run a fresh inference pass, so it",
        "cannot yet answer per-point `rgb_valid` prediction questions.",
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
            "E0 config snapshot check:",
            "",
            f"- feature mode: `{e0_config_checks['feature_mode']}`",
            f"- camera: `{e0_config_checks['camera_name']}`",
            f"- color sampling: `{e0_config_checks['color_sampling']}`",
            f"- RGB normalization: `{e0_config_checks['rgb_normalization']}`",
            f"- model input channels: `{e0_config_checks['in_channels']}`",
            "",
            "## Headline",
            "",
        ]
    )

    lines.extend(
        [
        f"E0's best marking-IoU checkpoint is epoch `{e0_best_epoch}`. D0's",
        f"official LiDAR-only checkpoint is epoch `{D0_OFFICIAL_EPOCH}`.",
        "",
        "| metric | D0 epoch 18 | E0 best | delta |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| marking IoU | {d0['marking_iou']:.6f} | {e0['marking_iou']:.6f} | "
            f"{e0['marking_iou'] - d0['marking_iou']:+.6f} |"
        ),
        (
            f"| marking F1 | {d0['marking_f1']:.6f} | {e0['marking_f1']:.6f} | "
            f"{e0['marking_f1'] - d0['marking_f1']:+.6f} |"
        ),
        (
            f"| marking precision | {d0['marking_precision']:.6f} | "
            f"{e0['marking_precision']:.6f} | "
            f"{e0['marking_precision'] - d0['marking_precision']:+.6f} |"
        ),
        (
            f"| marking recall | {d0['marking_recall']:.6f} | "
            f"{e0['marking_recall']:.6f} | "
            f"{e0['marking_recall'] - d0['marking_recall']:+.6f} |"
        ),
        (
            f"| mIoU | {d0['miou']:.6f} | {e0['miou']:.6f} | "
            f"{e0['miou'] - d0['miou']:+.6f} |"
        ),
        "",
        "E0 did not beat D0 on the main balanced marking metrics. It strongly",
        "increased recall, but precision fell enough that IoU and F1 stayed slightly",
        "below D0.",
        "",
        "## E0 Best vs Final Epoch",
        "",
        "| metric | E0 best | E0 final | delta final-best |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| marking IoU | {e0['marking_iou']:.6f} | {e0_final['marking_iou']:.6f} | "
            f"{e0_final['marking_iou'] - e0['marking_iou']:+.6f} |"
        ),
        (
            f"| precision | {e0['marking_precision']:.6f} | "
            f"{e0_final['marking_precision']:.6f} | "
            f"{e0_final['marking_precision'] - e0['marking_precision']:+.6f} |"
        ),
        (
            f"| recall | {e0['marking_recall']:.6f} | "
            f"{e0_final['marking_recall']:.6f} | "
            f"{e0_final['marking_recall'] - e0['marking_recall']:+.6f} |"
        ),
        (
            f"| val loss | {e0['val_loss']:.6f} | {e0_final['val_loss']:.6f} | "
            f"{e0_final['val_loss'] - e0['val_loss']:+.6f} |"
        ),
        "",
        "The final epoch has lower validation loss and higher recall but worse marking",
        "IoU. This repeats D0's pattern: loss keeps improving while positive-class",
        "calibration drifts toward overprediction.",
        "",
        "## Overprediction Check",
        "",
        "| class | D0 pred/true | E0 pred/true | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    )

    for class_name in CLASS_NAMES:
        d0_class_ratio = float(
            ratios[(ratios["run"] == "D0") & (ratios["class"] == class_name)][
                "pred_true_ratio"
            ].iloc[0]
        )
        e0_class_ratio = float(
            ratios[(ratios["run"] == "E0") & (ratios["class"] == class_name)][
                "pred_true_ratio"
            ].iloc[0]
        )
        lines.append(
            f"| {class_name} | {d0_class_ratio:.6f} | {e0_class_ratio:.6f} | "
            f"{e0_class_ratio - d0_class_ratio:+.6f} |"
        )

    lines.extend(
        [
            "",
            f"E0 marking pred/true ratio is `{e0_ratio:.3f}` vs D0 `{d0_ratio:.3f}`.",
            "This confirms that E0 increased marking overprediction.",
            "",
            "## Key Confusion Changes",
            "",
            "| error | D0 count | E0 count | delta |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| road->marking | {int(road_to_marking['d0_count'])} | "
                f"{int(road_to_marking['e0_count'])} | "
                f"{int(road_to_marking['delta_e0_minus_d0']):+d} |"
            ),
            (
                f"| marking->road | {int(marking_to_road['d0_count'])} | "
                f"{int(marking_to_road['e0_count'])} | "
                f"{int(marking_to_road['delta_e0_minus_d0']):+d} |"
            ),
            (
                f"| other->marking | {int(other_to_marking['d0_count'])} | "
                f"{int(other_to_marking['e0_count'])} | "
                f"{int(other_to_marking['delta_e0_minus_d0']):+d} |"
            ),
            "",
            "Interpretation: E0 likely reduced missed markings, but it also produced many",
            "more false-positive markings from road/other. The next diagnostic must be a",
            "fresh sampled inference pass that stratifies errors by `rgb_valid`.",
            "",
            "## What This Does Not Yet Prove",
            "",
            "This report does not prove that RGB itself is harmful. It only proves that",
            "the E0 training setup with RGB plus the D0 weighted-CE objective did not",
            "improve the balanced marking metric. The likely causes remain:",
            "",
            "1. RGB signal is useful but amplified by the aggressive marking class weight.",
            "2. Invalid-RGB fallback behavior hurts some frames/sequences.",
            "3. RGB projection/color quality introduces false-positive cues.",
            "",
            "The next required analysis is sampled checkpoint inference at E0 epoch 14",
            "with per-point `rgb_valid`, sequence, distance, raw subtype, intensity, and",
            "RGB summaries.",
            "",
            "## Outputs",
            "",
            "- `summary.csv`",
            "- `confusion_breakdown.csv`",
            "- `pred_true_ratio.csv`",
            "- `analysis.md`",
            "",
        ]
    )

    out_path.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact_counts = pd.DataFrame(
        [
            validate_run_artifacts(D0_RUN_DIR, "D0", EXPECTED_EPOCHS),
            validate_run_artifacts(E0_RUN_DIR, "E0", EXPECTED_EPOCHS),
        ]
    )
    artifact_counts.to_csv(OUT_DIR / "artifact_counts.csv", index=False)
    e0_config_checks = validate_e0_config_snapshot(E0_RUN_DIR)

    e0_df = load_eval_history(E0_RUN_DIR)
    d0_df = load_eval_history(D0_RUN_DIR)

    e0_best = best_by_marking_iou(e0_df)
    d0_best = best_by_marking_iou(d0_df)
    e0_final = e0_df.iloc[-1]

    e0_best_epoch = int(e0_best["epoch"])
    d0_best_epoch = int(d0_best["epoch"])
    if e0_best_epoch != 14:
        print(f"warning: E0 best epoch is {e0_best_epoch}, not expected 14")
    if d0_best_epoch != D0_OFFICIAL_EPOCH:
        print(
            f"warning: D0 best epoch is {d0_best_epoch}, "
            f"official comparison uses {D0_OFFICIAL_EPOCH}"
        )

    d0_official = d0_df.loc[d0_df["epoch"] == D0_OFFICIAL_EPOCH]
    if d0_official.empty:
        raise ValueError(f"D0 eval history has no epoch {D0_OFFICIAL_EPOCH}")
    d0_official_row = d0_official.iloc[0]

    summary = pd.DataFrame(
        [
            row_metrics(d0_official_row, "D0", "official"),
            row_metrics(e0_best, "E0", "best"),
            row_metrics(e0_final, "E0", "final"),
        ]
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    e0_cm = load_confusion_matrix(E0_RUN_DIR, e0_best_epoch)
    d0_cm = load_confusion_matrix(D0_RUN_DIR, D0_OFFICIAL_EPOCH)

    confusion = key_confusion_rows(e0_cm, d0_cm)
    confusion.to_csv(OUT_DIR / "confusion_breakdown.csv", index=False)

    ratio_rows = []
    for run_name, cm in (("D0", d0_cm), ("E0", e0_cm)):
        for class_name, ratio in pred_true_ratios(cm).items():
            ratio_rows.append(
                {
                    "run": run_name,
                    "class": class_name,
                    "pred_true_ratio": ratio,
                    "true_count": int(cm[CLASS_NAMES.index(class_name), :].sum()),
                    "pred_count": int(cm[:, CLASS_NAMES.index(class_name)].sum()),
                }
            )
    ratios = pd.DataFrame(ratio_rows)
    ratios.to_csv(OUT_DIR / "pred_true_ratio.csv", index=False)

    write_markdown(
        summary=summary,
        confusion=confusion,
        ratios=ratios,
        artifact_counts=artifact_counts,
        e0_config_checks=e0_config_checks,
        e0_best_epoch=e0_best_epoch,
        d0_best_epoch=d0_best_epoch,
        out_path=OUT_DIR / "analysis.md",
    )

    print("E0 run-level analysis complete")
    print(f"out_dir {OUT_DIR}")
    print(f"e0_best_epoch {e0_best_epoch}")
    print(f"d0_comparison_epoch {D0_OFFICIAL_EPOCH}")


if __name__ == "__main__":
    main()

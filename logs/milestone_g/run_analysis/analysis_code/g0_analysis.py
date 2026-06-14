#!/usr/bin/env python
"""Milestone G0 run-level analysis from saved training artifacts.

Mirrors the Milestone F0 run-level analysis (logs/milestone_f/.../f0_analysis.py)
but extends it to a four-way D0/E0/F0/G0 comparison and adds the G-specific
loss-component guards.

Milestone G keeps the F0 setup byte-for-byte and changes ONLY the loss:

    loss = weighted_CE + lovasz_lambda * Lovasz-Softmax      (lovasz_lambda = 0.5)

Consequences this script encodes:

- For D0/E0/F0 (weighted-CE only), ``val_loss`` IS the pure weighted CE, so for
  those runs CE == total. We expose ``val_ce``/``train_ce`` columns equal to the
  loss columns and Lovasz = 0.0, giving a single comparable CE column.
- For G0, ``eval_history.csv`` ``train_loss``/``val_loss`` are the TOTAL loss
  (CE + 0.5*Lovasz). The CE component lives in ``loss_components.csv``.
- The ONLY fair loss-vs-loss comparison is F0 ``val_loss`` (pure CE) against G0
  ``val_ce`` (CE component) -- never F0 ``val_loss`` vs G0 ``val_loss``.

This script does not run inference. It validates artifacts, validates that the
G0 config is F0 + loss only, fails loudly if ``loss_components.csv`` is missing
or inconsistent, and writes the run-level comparison.

Class order in saved confusion matrices is active Open3D metric indexing:

    0 = road
    1 = marking  (stored in eval_history as lane_*)
    2 = other

Outputs:
    logs/milestone_g/run_analysis/G0_rgb_lovasz/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "logs/milestone_g/run_analysis/G0_rgb_lovasz"

D0_RUN_DIR = REPO_ROOT / "logs/milestone_d/runs/D0_weighted_ce_25ep"
E0_RUN_DIR = REPO_ROOT / "logs/milestone_e/runs/E0_rgb_front_v1"
F0_RUN_DIR = REPO_ROOT / "logs/milestone_f/runs/F0_rgb_soft_weights"
G0_RUN_DIR = REPO_ROOT / "logs/milestone_g/runs/G0_rgb_lovasz"

D0_OFFICIAL_EPOCH = 18
E0_EXPECTED_BEST_EPOCH = 14
F0_EXPECTED_BEST_EPOCH = 13
LOVASZ_LAMBDA_EXPECTED = 0.5
CLASS_NAMES = ("road", "marking", "other")

# Tolerance for float identities recovered from CSV round-trips.
COMPOSE_ATOL = 1e-4
COMPOSE_RTOL = 1e-3

SUMMARY_COLUMNS = [
    "run",
    "label",
    "epoch",
    "train_loss",
    "val_loss",
    "train_ce",
    "val_ce",
    "train_lovasz",
    "val_lovasz",
    "miou",
    "marking_iou",
    "marking_precision",
    "marking_recall",
    "marking_f1",
    "lr",
]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= COMPOSE_ATOL + COMPOSE_RTOL * abs(float(b))


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


def load_loss_components(run_dir: Path) -> pd.DataFrame:
    """Load and validate G0 loss_components.csv. Fails loudly if missing/empty."""
    path = run_dir / "loss_components.csv"
    if not path.exists():
        raise RuntimeError(
            f"loss_components.csv not found in {run_dir}. For Milestone G the "
            "combined loss MUST log per-epoch CE/Lovasz/total components. Its "
            "absence means the combined-loss path did not engage -- treat this "
            "as a hard failure, not a skip."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} exists but is empty; combined loss logged no components.")
    required = {
        "epoch",
        "train_ce_loss",
        "train_lovasz_loss",
        "train_total_loss",
        "val_ce_loss",
        "val_lovasz_loss",
        "val_total_loss",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    return df


def validate_loss_components(
    components: pd.DataFrame, eval_df: pd.DataFrame, lovasz_lambda: float
) -> dict[str, Any]:
    """Cross-check loss_components.csv against eval_history.csv and composition.

    Raises RuntimeError on any inconsistency:
      - row count / epoch alignment
      - val_total_loss == eval_history.val_loss   (and train side)
      - total == ce + lambda * lovasz             (tolerance check)
    """
    if len(components) != len(eval_df):
        raise RuntimeError(
            f"loss_components rows ({len(components)}) != eval_history rows "
            f"({len(eval_df)}); logging is inconsistent."
        )
    comp = components.sort_values("epoch").reset_index(drop=True)
    ev = eval_df.sort_values("epoch").reset_index(drop=True)
    if not np.array_equal(comp["epoch"].to_numpy(), ev["epoch"].to_numpy()):
        raise RuntimeError("loss_components epochs do not match eval_history epochs.")

    max_total_train_resid = 0.0
    max_total_val_resid = 0.0
    max_compose_train_resid = 0.0
    max_compose_val_resid = 0.0
    for (_, crow), (_, erow) in zip(comp.iterrows(), ev.iterrows()):
        # eval_history loss columns are the TOTAL loss for G; must equal stored totals.
        if not close(crow["train_total_loss"], erow["train_loss"]):
            raise RuntimeError(
                f"epoch {int(crow['epoch'])}: train_total_loss "
                f"{crow['train_total_loss']} != eval_history train_loss {erow['train_loss']}"
            )
        if not close(crow["val_total_loss"], erow["val_loss"]):
            raise RuntimeError(
                f"epoch {int(crow['epoch'])}: val_total_loss "
                f"{crow['val_total_loss']} != eval_history val_loss {erow['val_loss']}"
            )
        # Composition: total == ce + lambda * lovasz (raw, unscaled Lovasz).
        train_compose = crow["train_ce_loss"] + lovasz_lambda * crow["train_lovasz_loss"]
        val_compose = crow["val_ce_loss"] + lovasz_lambda * crow["val_lovasz_loss"]
        if not close(crow["train_total_loss"], train_compose):
            raise RuntimeError(
                f"epoch {int(crow['epoch'])}: train_total_loss "
                f"{crow['train_total_loss']} != train_ce + {lovasz_lambda}*train_lovasz "
                f"({train_compose})"
            )
        if not close(crow["val_total_loss"], val_compose):
            raise RuntimeError(
                f"epoch {int(crow['epoch'])}: val_total_loss "
                f"{crow['val_total_loss']} != val_ce + {lovasz_lambda}*val_lovasz "
                f"({val_compose})"
            )
        max_total_train_resid = max(
            max_total_train_resid, abs(crow["train_total_loss"] - erow["train_loss"])
        )
        max_total_val_resid = max(
            max_total_val_resid, abs(crow["val_total_loss"] - erow["val_loss"])
        )
        max_compose_train_resid = max(
            max_compose_train_resid, abs(crow["train_total_loss"] - train_compose)
        )
        max_compose_val_resid = max(
            max_compose_val_resid, abs(crow["val_total_loss"] - val_compose)
        )
    return {
        "rows": int(len(comp)),
        "lovasz_lambda": float(lovasz_lambda),
        "max_total_train_residual": float(max_total_train_resid),
        "max_total_val_residual": float(max_total_val_resid),
        "max_compose_train_residual": float(max_compose_train_resid),
        "max_compose_val_residual": float(max_compose_val_resid),
    }


def validate_run_artifacts(
    run_dir: Path, run_name: str, expect_loss_components: bool = False
) -> dict[str, int | str]:
    """Validate artifact counts are internally consistent (count == eval rows)."""
    require_file(run_dir / "eval_history.csv")
    history = pd.read_csv(run_dir / "eval_history.csv")
    n = int(len(history))
    eval_jsons = sorted(run_dir.glob("eval_epoch_*.json"))
    confusions = sorted(run_dir.glob("confusion_epoch_*.npy"))
    checkpoints = sorted((run_dir / "checkpoints").glob("ckpt_epoch_*.pth"))
    loss_components_rows = None
    if (run_dir / "loss_components.csv").exists():
        loss_components_rows = int(len(pd.read_csv(run_dir / "loss_components.csv")))

    counts = {
        "run": run_name,
        "eval_history_rows": n,
        "eval_json_count": int(len(eval_jsons)),
        "confusion_count": int(len(confusions)),
        "checkpoint_count": int(len(checkpoints)),
        "loss_components_rows": (
            loss_components_rows if loss_components_rows is not None else -1
        ),
    }
    for key, value in (
        ("eval_json_count", len(eval_jsons)),
        ("confusion_count", len(confusions)),
        ("checkpoint_count", len(checkpoints)),
    ):
        if value != n:
            raise RuntimeError(
                f"{run_name} {key}={value} but eval_history has {n} rows; "
                "artifact counts are inconsistent."
            )
    if expect_loss_components:
        if loss_components_rows is None:
            raise RuntimeError(
                f"{run_name} has no loss_components.csv; required for Milestone G."
            )
        if loss_components_rows != n:
            raise RuntimeError(
                f"{run_name} loss_components rows={loss_components_rows} but "
                f"eval_history has {n} rows; inconsistent."
            )
    return counts


def validate_g0_config_snapshot(run_dir: Path) -> dict[str, Any]:
    """Validate the G0 config is F0 + loss only (the formal fairness proof)."""
    path = require_file(run_dir / "config_snapshot.yml")
    cfg = yaml.safe_load(path.read_text())
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    pipeline_cfg = cfg.get("pipeline", {})
    scheduler_cfg = pipeline_cfg.get("scheduler", {})
    loss_cfg = pipeline_cfg.get("loss", {}) or {}

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
        "loss_name": loss_cfg.get("name"),
        "lovasz_lambda": loss_cfg.get("lovasz_lambda"),
        "lovasz_classes": loss_cfg.get("lovasz_classes"),
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
        "loss_name": "weighted_ce_lovasz",
        "lovasz_lambda": 0.5,
        "lovasz_classes": "present",
    }
    mismatches = {}
    for key, expected_value in expected.items():
        actual = checks[key]
        if key == "class_weights":
            if len(actual) != len(expected_value) or any(
                abs(float(a) - float(e)) > 1e-6 for a, e in zip(actual, expected_value)
            ):
                mismatches[key] = (actual, expected_value)
        elif key == "lovasz_lambda":
            if actual is None or abs(float(actual) - float(expected_value)) > 1e-9:
                mismatches[key] = (actual, expected_value)
        elif actual != expected_value:
            mismatches[key] = (actual, expected_value)
    if mismatches:
        raise RuntimeError(f"G0 config snapshot mismatch (must be F0 + loss only): {mismatches}")
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


def row_metrics(
    row: pd.Series,
    run: str,
    label: str,
    components: pd.DataFrame | None = None,
) -> dict[str, float | int | str]:
    """Base run metrics; CE/Lovasz columns populated for all runs.

    For weighted-CE-only runs (no components), CE == total == loss and Lovasz=0.
    For G0, CE/Lovasz are looked up from loss_components.csv by epoch.
    """
    epoch = int(row["epoch"])
    train_total = float(row["train_loss"])
    val_total = float(row["val_loss"])
    if components is None:
        train_ce, val_ce, train_lov, val_lov = train_total, val_total, 0.0, 0.0
    else:
        comp = components.loc[components["epoch"] == epoch]
        if comp.empty:
            raise RuntimeError(f"{run} loss_components has no epoch {epoch}")
        crow = comp.iloc[0]
        train_ce = float(crow["train_ce_loss"])
        val_ce = float(crow["val_ce_loss"])
        train_lov = float(crow["train_lovasz_loss"])
        val_lov = float(crow["val_lovasz_loss"])
    return {
        "run": run,
        "label": label,
        "epoch": epoch,
        "train_loss": train_total,
        "val_loss": val_total,
        "train_ce": train_ce,
        "val_ce": val_ce,
        "train_lovasz": train_lov,
        "val_lovasz": val_lov,
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
            g0 = int(matrices["G0"][true_idx, pred_idx])
            rows.append(
                {
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "error_type": f"{true_name}->{pred_name}",
                    "d0_count": d0,
                    "e0_count": e0,
                    "f0_count": f0,
                    "g0_count": g0,
                    "delta_g0_minus_d0": g0 - d0,
                    "delta_g0_minus_f0": g0 - f0,
                }
            )
    return pd.DataFrame(rows)


def metric_table_line(metric: str, rows: dict[str, pd.Series]) -> str:
    d0, e0, f0, g0 = rows["D0"], rows["E0"], rows["F0"], rows["G0"]
    return (
        f"| {metric} | {d0[metric]:.6f} | {e0[metric]:.6f} | {f0[metric]:.6f} | "
        f"{g0[metric]:.6f} | {g0[metric] - f0[metric]:+.6f} | {g0[metric] - d0[metric]:+.6f} |"
    )


def write_markdown(
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
    ratios: pd.DataFrame,
    artifact_counts: pd.DataFrame,
    g0_config_checks: dict[str, Any],
    component_check: dict[str, Any],
    out_path: Path,
) -> None:
    def srow(run: str, label: str) -> pd.Series:
        return summary[(summary["run"] == run) & (summary["label"] == label)].iloc[0]

    rows = {
        "D0": srow("D0", "official"),
        "E0": srow("E0", "best"),
        "F0": srow("F0", "best"),
        "G0": srow("G0", "best"),
    }
    f0 = rows["F0"]
    g0 = rows["G0"]
    f0_final = srow("F0", "final")
    g0_final = srow("G0", "final")

    def ratio_for(run: str, label: str, cls: str = "marking") -> float:
        return float(
            ratios[
                (ratios["run"] == run)
                & (ratios["label"] == label)
                & (ratios["class"] == cls)
            ]["pred_true_ratio"].iloc[0]
        )

    def conf_row(error_type: str) -> pd.Series:
        return confusion[confusion["error_type"] == error_type].iloc[0]

    lines = [
        "# G0 RGB + Lovasz Run-Level Analysis",
        "",
        "This report uses saved training artifacts only (`eval_history.csv`,",
        "`loss_components.csv`, `confusion_epoch_*.npy`). It does not run a fresh",
        "sampled inference pass.",
        "",
        "Milestone G = Milestone F0 with the loss changed to",
        f"`weighted_CE + {component_check['lovasz_lambda']} * Lovasz-Softmax`. Everything",
        "else (RGB front features, softened marking weight effective 15.0,",
        "`dim_features=16`) is unchanged. The config-snapshot check below is the",
        "formal proof of that.",
        "",
        "## Provenance Checks",
        "",
        "| run | eval rows | eval jsons | confusions | checkpoints | loss_components |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in artifact_counts.iterrows():
        lc = int(row["loss_components_rows"])
        lc_str = "n/a" if lc < 0 else str(lc)
        lines.append(
            f"| {row['run']} | {int(row['eval_history_rows'])} | "
            f"{int(row['eval_json_count'])} | {int(row['confusion_count'])} | "
            f"{int(row['checkpoint_count'])} | {lc_str} |"
        )

    lines.extend(
        [
            "",
            "G0 loss-component consistency (all within tolerance, else this report",
            "would have failed):",
            "",
            f"- lovasz_lambda: `{component_check['lovasz_lambda']}`",
            f"- max |eval val_loss - val_total_loss|: `{component_check['max_total_val_residual']:.2e}`",
            f"- max |val_total - (val_ce + lambda*val_lovasz)|: `{component_check['max_compose_val_residual']:.2e}`",
            f"- max |train_total - (train_ce + lambda*train_lovasz)|: `{component_check['max_compose_train_residual']:.2e}`",
            "",
            "G0 config snapshot check:",
            "",
            f"- feature mode: `{g0_config_checks['feature_mode']}`",
            f"- class weights: `{g0_config_checks['class_weights']}`",
            f"- dim_features: `{g0_config_checks['dim_features']}`",
            f"- model input channels: `{g0_config_checks['in_channels']}`",
            f"- loss: `{g0_config_checks['loss_name']}`, "
            f"lambda `{g0_config_checks['lovasz_lambda']}`, "
            f"classes `{g0_config_checks['lovasz_classes']}`",
            "",
            "## Headline",
            "",
            "Best checkpoint for every run is selected by maximum raw marking IoU",
            "(`lane_iou`), the same rule across D0/E0/F0/G0.",
            "",
            "| metric | D0 ep18 | E0 ep14 | F0 ep13 | G0 ep"
            f"{int(g0['epoch'])} | G0-F0 | G0-D0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            metric_table_line("marking_iou", rows),
            metric_table_line("marking_f1", rows),
            metric_table_line("marking_precision", rows),
            metric_table_line("marking_recall", rows),
            metric_table_line("miou", rows),
            "",
            "## CE Loss Comparison (fair: pure CE only)",
            "",
            "F0 `val_loss` is pure weighted CE. G0 `val_ce` is the CE component of",
            "the combined loss, computed identically. These are directly comparable.",
            "G0 `val_loss` is the TOTAL (CE + Lovasz) and must NOT be compared to F0.",
            "",
            "| run | epoch | val CE | val total | val Lovasz (raw) |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| F0 best | {int(f0['epoch'])} | {f0['val_ce']:.6f} | {f0['val_loss']:.6f} | {f0['val_lovasz']:.6f} |",
            f"| G0 best | {int(g0['epoch'])} | {g0['val_ce']:.6f} | {g0['val_loss']:.6f} | {g0['val_lovasz']:.6f} |",
            "",
            "## Best vs Final Epoch (drift)",
            "",
            "The F0 hypothesis G tests: does adding the IoU-surrogate Lovasz term",
            "reduce the best-to-final drift seen in F0 (val loss kept improving while",
            "marking IoU degraded after the best epoch)?",
            "",
            "| run | metric | best | final | delta final-best |",
            "| --- | --- | ---: | ---: | ---: |",
            f"| F0 | marking IoU | {f0['marking_iou']:.6f} | {f0_final['marking_iou']:.6f} | {f0_final['marking_iou'] - f0['marking_iou']:+.6f} |",
            f"| G0 | marking IoU | {g0['marking_iou']:.6f} | {g0_final['marking_iou']:.6f} | {g0_final['marking_iou'] - g0['marking_iou']:+.6f} |",
            f"| F0 | precision | {f0['marking_precision']:.6f} | {f0_final['marking_precision']:.6f} | {f0_final['marking_precision'] - f0['marking_precision']:+.6f} |",
            f"| G0 | precision | {g0['marking_precision']:.6f} | {g0_final['marking_precision']:.6f} | {g0_final['marking_precision'] - g0['marking_precision']:+.6f} |",
            f"| F0 | recall | {f0['marking_recall']:.6f} | {f0_final['marking_recall']:.6f} | {f0_final['marking_recall'] - f0['marking_recall']:+.6f} |",
            f"| G0 | recall | {g0['marking_recall']:.6f} | {g0_final['marking_recall']:.6f} | {g0_final['marking_recall'] - g0['marking_recall']:+.6f} |",
            "",
            "## Marking Overprediction Check",
            "",
            "| run | label | marking pred/true ratio |",
            "| --- | --- | ---: |",
        ]
    )
    for run, label in (
        ("D0", "official"),
        ("E0", "best"),
        ("F0", "best"),
        ("G0", "best"),
        ("F0", "final"),
        ("G0", "final"),
    ):
        lines.append(f"| {run} | {label} | {ratio_for(run, label):.6f} |")

    lines.extend(
        [
            "",
            f"G0 marking pred/true ratio at best is `{ratio_for('G0', 'best'):.3f}` vs "
            f"F0 `{ratio_for('F0', 'best'):.3f}`. Closer to 1.0 means better calibrated.",
            "",
            "## Key Confusion Changes (best checkpoints)",
            "",
            "| error | D0 | E0 | F0 | G0 | G0-F0 | G0-D0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for error_type in ("road->marking", "marking->road", "other->marking"):
        row = conf_row(error_type)
        lines.append(
            f"| {error_type} | {int(row['d0_count'])} | {int(row['e0_count'])} | "
            f"{int(row['f0_count'])} | {int(row['g0_count'])} | "
            f"{int(row['delta_g0_minus_f0']):+d} | {int(row['delta_g0_minus_d0']):+d} |"
        )

    lines.extend(
        [
            "",
            "## What Still Needs Sampled / Component Analysis",
            "",
            "The run-level metrics show whether G beat F0, but not whether the Lovasz",
            "term is healthy (scale, stability, IoU-alignment) or whether residual",
            "errors moved. Those are answered by:",
            "",
            "- `g0_loss_component_analysis.py` (CE vs Lovasz vs total, alignment)",
            "- `g0_sampled_error_analysis.py` (RGB-valid, sequence, distance, subtype)",
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
            validate_run_artifacts(G0_RUN_DIR, "G0", expect_loss_components=True),
        ]
    )
    artifact_counts.to_csv(OUT_DIR / "artifact_counts.csv", index=False)

    g0_config_checks = validate_g0_config_snapshot(G0_RUN_DIR)
    lovasz_lambda = float(g0_config_checks["lovasz_lambda"])

    d0_df = load_eval_history(D0_RUN_DIR)
    e0_df = load_eval_history(E0_RUN_DIR)
    f0_df = load_eval_history(F0_RUN_DIR)
    g0_df = load_eval_history(G0_RUN_DIR)
    g0_components = load_loss_components(G0_RUN_DIR)
    component_check = validate_loss_components(g0_components, g0_df, lovasz_lambda)

    e0_best = best_by_marking_iou(e0_df)
    f0_best = best_by_marking_iou(f0_df)
    g0_best = best_by_marking_iou(g0_df)
    f0_final = f0_df.iloc[-1]
    g0_final = g0_df.iloc[-1]
    d0_official = d0_df.loc[d0_df["epoch"] == D0_OFFICIAL_EPOCH]
    if d0_official.empty:
        raise ValueError(f"D0 eval history has no epoch {D0_OFFICIAL_EPOCH}")
    d0_official_row = d0_official.iloc[0]

    e0_best_epoch = int(e0_best["epoch"])
    f0_best_epoch = int(f0_best["epoch"])
    g0_best_epoch = int(g0_best["epoch"])
    if e0_best_epoch != E0_EXPECTED_BEST_EPOCH:
        print(f"warning: E0 best epoch is {e0_best_epoch}, not expected {E0_EXPECTED_BEST_EPOCH}")
    if f0_best_epoch != F0_EXPECTED_BEST_EPOCH:
        print(f"warning: F0 best epoch is {f0_best_epoch}, not expected {F0_EXPECTED_BEST_EPOCH}")

    summary = pd.DataFrame(
        [
            row_metrics(d0_official_row, "D0", "official"),
            row_metrics(e0_best, "E0", "best"),
            row_metrics(f0_best, "F0", "best"),
            row_metrics(f0_final, "F0", "final"),
            row_metrics(g0_best, "G0", "best", components=g0_components),
            row_metrics(g0_final, "G0", "final", components=g0_components),
        ],
        columns=SUMMARY_COLUMNS,
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    matrices = {
        "D0": load_confusion_matrix(D0_RUN_DIR, D0_OFFICIAL_EPOCH),
        "E0": load_confusion_matrix(E0_RUN_DIR, e0_best_epoch),
        "F0": load_confusion_matrix(F0_RUN_DIR, f0_best_epoch),
        "G0": load_confusion_matrix(G0_RUN_DIR, g0_best_epoch),
        "F0_final": load_confusion_matrix(F0_RUN_DIR, int(f0_final["epoch"])),
        "G0_final": load_confusion_matrix(G0_RUN_DIR, int(g0_final["epoch"])),
    }
    confusion = all_confusion_rows(
        {k: matrices[k] for k in ("D0", "E0", "F0", "G0")}
    )
    confusion.to_csv(OUT_DIR / "confusion_breakdown.csv", index=False)

    ratio_rows = []
    for run_name, label, cm in (
        ("D0", "official", matrices["D0"]),
        ("E0", "best", matrices["E0"]),
        ("F0", "best", matrices["F0"]),
        ("F0", "final", matrices["F0_final"]),
        ("G0", "best", matrices["G0"]),
        ("G0", "final", matrices["G0_final"]),
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
        g0_config_checks=g0_config_checks,
        component_check=component_check,
        out_path=OUT_DIR / "analysis.md",
    )

    print("G0 run-level analysis complete")
    print(f"out_dir {OUT_DIR}")
    print(f"g0_best_epoch {g0_best_epoch}")
    print(f"f0_best_epoch {f0_best_epoch}")
    print(f"e0_best_epoch {e0_best_epoch}")
    print(f"d0_comparison_epoch {D0_OFFICIAL_EPOCH}")
    print(f"loss_component_max_compose_residual {component_check['max_compose_val_residual']:.3e}")
    print("script_status PASS")


if __name__ == "__main__":
    main()

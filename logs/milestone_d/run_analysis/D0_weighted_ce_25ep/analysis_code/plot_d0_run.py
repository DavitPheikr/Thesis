"""Generate official run-level plots for Milestone D D0_weighted_ce_25ep.

This script consumes only the completed D0 run artifacts:

  logs/milestone_d/runs/D0_weighted_ce_25ep/

It does not run inference. It validates the artifact set, recomputes the best
epoch from eval_history.csv, verifies the Milestone D lane->marking metric
alias, and writes thesis-quality run plots under:

  logs/milestone_d/run_analysis/D0_weighted_ce_25ep/run_plots/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "logs/milestone_d/configs/d0_weighted_ce.yml").exists():
            return parent
    raise RuntimeError("Could not find project root from analysis script path")


PROJECT_ROOT = find_project_root()
RUN_NAME = "D0_weighted_ce_25ep"
DEFAULT_RUN_DIR = PROJECT_ROOT / f"logs/milestone_d/runs/{RUN_NAME}"
DEFAULT_OUT_DIR = PROJECT_ROOT / f"logs/milestone_d/run_analysis/{RUN_NAME}/run_plots"
DEFAULT_REPORT_DIR = PROJECT_ROOT / f"logs/milestone_d/run_analysis/{RUN_NAME}/reports"

CLASS_NAMES = ("road", "marking", "other")
CSV_CLASS_PREFIXES = ("road", "lane", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--expected-epochs", type=int, default=25)
    parser.add_argument("--expected-best-marking-epoch", type=int, default=18)
    return parser.parse_args()


def as_float(value: str | int | float | None) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def load_eval_history(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "eval_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_history.csv: {path}")
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {key: as_float(value) for key, value in row.items()}
            parsed["epoch"] = int(parsed["epoch"])
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def load_training_log(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "training_log.txt"
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        fields: dict[str, str] = {}
        for part in line.split():
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        if "epoch" not in fields:
            continue
        epoch = int(fields["epoch"])
        out[epoch] = {
            "wall_clock_seconds": as_float(fields.get("wall_clock_seconds")),
            "peak_gpu_memory_bytes": as_float(fields.get("peak_gpu_memory_bytes")),
            "lr": as_float(fields.get("lr")),
        }
    return out


def load_epoch_jsons(run_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(run_dir.glob("eval_epoch_*.json")):
        data = json.loads(path.read_text())
        epoch = int(data["epoch"])
        out[epoch] = data
    return out


def validate_artifacts(
    run_dir: Path,
    rows: list[dict[str, float]],
    eval_jsons: dict[int, dict],
    expected_epochs: int,
) -> None:
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    history_lines = sum(1 for _ in (run_dir / "eval_history.csv").open())
    checkpoints = sorted((run_dir / "checkpoints").glob("ckpt_epoch_*.pth"))
    confusions = sorted(run_dir.glob("confusion_epoch_*.npy"))
    if history_lines != expected_epochs + 1:
        raise RuntimeError(f"eval_history.csv line count {history_lines}, expected {expected_epochs + 1}")
    if len(rows) != expected_epochs:
        raise RuntimeError(f"eval_history rows {len(rows)}, expected {expected_epochs}")
    if len(checkpoints) != expected_epochs:
        raise RuntimeError(f"checkpoint count {len(checkpoints)}, expected {expected_epochs}")
    if len(eval_jsons) != expected_epochs:
        raise RuntimeError(f"eval JSON count {len(eval_jsons)}, expected {expected_epochs}")
    if len(confusions) != expected_epochs:
        raise RuntimeError(f"confusion npy count {len(confusions)}, expected {expected_epochs}")

    for epoch, data in eval_jsons.items():
        aliases = data.get("metric_aliases", {})
        if aliases.get("positive_class_name") != "marking":
            raise RuntimeError(f"Epoch {epoch} does not declare positive_class_name=marking")
        class_indexing = data.get("class_indexing", {})
        if class_indexing.get("0") != "road" or class_indexing.get("1") != "marking" or class_indexing.get("2") != "other":
            raise RuntimeError(f"Epoch {epoch} has unexpected class_indexing: {class_indexing}")
        cm_json = np.asarray(data["metrics"]["confusion_matrix"], dtype=np.int64)
        cm_npy = np.load(run_dir / f"confusion_epoch_{epoch:03d}.npy")
        if cm_json.shape != (3, 3) or cm_npy.shape != (3, 3):
            raise RuntimeError(f"Epoch {epoch} confusion matrix shape is not 3x3")
        if not np.array_equal(cm_json, cm_npy):
            raise RuntimeError(f"Epoch {epoch} JSON and NPY confusion matrices differ")
        support = data["metrics"]["support"]
        if int(cm_json[0, :].sum()) != int(support["road"]):
            raise RuntimeError(f"Epoch {epoch} road support does not match confusion row")
        if int(cm_json[1, :].sum()) != int(support["lane"]):
            raise RuntimeError(f"Epoch {epoch} marking/lane support does not match confusion row")
        if int(cm_json[2, :].sum()) != int(support["other"]):
            raise RuntimeError(f"Epoch {epoch} other support does not match confusion row")


def array(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([row.get(key, float("nan")) for row in rows], dtype=np.float64)


def epochs(rows: list[dict[str, float]]) -> np.ndarray:
    return array(rows, "epoch").astype(np.int64)


def best_row(rows: list[dict[str, float]], key: str, mode: str = "max") -> dict[str, float]:
    valid = [row for row in rows if np.isfinite(row.get(key, float("nan")))]
    if not valid:
        raise RuntimeError(f"No finite values for metric {key}")
    return (max if mode == "max" else min)(valid, key=lambda row: row[key])


def save_line_plot(
    out_path: Path,
    x: np.ndarray,
    series: Iterable[tuple[str, np.ndarray, str | None]],
    title: str,
    ylabel: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for label, y, color in series:
        ax.plot(x, y, marker="o", linewidth=2.2, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_overview_plot(rows: list[dict[str, float]], out_path: Path) -> None:
    x = epochs(rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(x, array(rows, "train_loss"), marker="o", label="train_loss")
    axes[0, 0].plot(x, array(rows, "val_loss"), marker="o", label="val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()

    axes[0, 1].plot(x, array(rows, "miou"), marker="o", label="mIoU")
    axes[0, 1].plot(x, array(rows, "lane_iou"), marker="o", label="marking IoU")
    axes[0, 1].set_title("Core Validation IoU")
    axes[0, 1].set_ylabel("IoU")
    axes[0, 1].legend()

    axes[1, 0].plot(x, array(rows, "lane_precision"), marker="o", label="precision")
    axes[1, 0].plot(x, array(rows, "lane_recall"), marker="o", label="recall")
    axes[1, 0].plot(x, array(rows, "lane_f1"), marker="o", label="F1")
    axes[1, 0].set_title("Marking Precision / Recall / F1")
    axes[1, 0].set_ylabel("score")
    axes[1, 0].legend()

    axes[1, 1].step(x, array(rows, "lr"), where="post", linewidth=2.4, label="LR")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_ylabel("learning rate")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def confusion_from_json(data: dict) -> np.ndarray:
    return np.asarray(data["metrics"]["confusion_matrix"], dtype=np.int64)


def save_confusion_heatmap(cm: np.ndarray, out_path: Path, title: str) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_xticks(np.arange(3), CLASS_NAMES)
    ax.set_yticks(np.arange(3), CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            text = f"{100.0 * row_pct[i, j]:.1f}%\n{cm[i, j]:,}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if row_pct[i, j] > 0.55 else "black",
                fontsize=9,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def class_shares(eval_jsons: dict[int, dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sorted_epochs = np.asarray(sorted(eval_jsons), dtype=np.int64)
    values: dict[str, list[float]] = {}
    for name in CLASS_NAMES:
        values[f"true_{name}"] = []
        values[f"pred_{name}"] = []
    for epoch in sorted_epochs:
        cm = confusion_from_json(eval_jsons[int(epoch)])
        total = max(int(cm.sum()), 1)
        for idx, name in enumerate(CLASS_NAMES):
            values[f"true_{name}"].append(float(cm[idx, :].sum() / total))
            values[f"pred_{name}"].append(float(cm[:, idx].sum() / total))
    return sorted_epochs, {key: np.asarray(val, dtype=np.float64) for key, val in values.items()}


def save_class_share_plot(eval_jsons: dict[int, dict], out_path: Path) -> None:
    x, series = class_shares(eval_jsons)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, name in zip(axes, CLASS_NAMES, strict=True):
        ax.plot(x, 100.0 * series[f"true_{name}"], marker="o", label="true")
        ax.plot(x, 100.0 * series[f"pred_{name}"], marker="o", label="predicted")
        ax.set_title(f"{name} share")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validated points (%)")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_runtime_plot(
    rows: list[dict[str, float]],
    training_log: dict[int, dict[str, float]],
    out_path: Path,
) -> None:
    x = epochs(rows)
    val_wall = array(rows, "val_wall_clock_seconds")
    epoch_wall = np.asarray(
        [
            training_log.get(int(epoch), {}).get("wall_clock_seconds", float("nan"))
            for epoch in x
        ],
        dtype=np.float64,
    )
    train_wall = epoch_wall - val_wall
    peak_mem_mib = np.asarray(
        [
            training_log.get(int(epoch), {}).get(
                "peak_gpu_memory_bytes",
                row["peak_gpu_memory_bytes_val"],
            )
            / (1024.0 * 1024.0)
            for epoch, row in zip(x, rows, strict=True)
        ],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(x, train_wall / 60.0, marker="o", label="train")
    axes[0].plot(x, val_wall / 60.0, marker="o", label="validation")
    axes[0].plot(x, epoch_wall / 60.0, marker="o", label="total")
    axes[0].set_title("Wall Clock")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("minutes")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, peak_mem_mib, marker="o", color="#6a51a3")
    axes[1].set_title("Peak PyTorch GPU Memory")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MiB")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def marking_share_summary(eval_jsons: dict[int, dict], epoch: int) -> dict[str, float]:
    cm = confusion_from_json(eval_jsons[epoch])
    total = max(int(cm.sum()), 1)
    true_marking = int(cm[1, :].sum())
    pred_marking = int(cm[:, 1].sum())
    return {
        "true_marking": true_marking,
        "pred_marking": pred_marking,
        "true_share": true_marking / total,
        "pred_share": pred_marking / total,
        "pred_true_ratio": float("nan") if true_marking == 0 else pred_marking / true_marking,
        "marking_to_road": int(cm[1, 0]),
        "marking_to_other": int(cm[1, 2]),
        "road_to_marking": int(cm[0, 1]),
        "other_to_marking": int(cm[2, 1]),
    }


def write_run_summary(
    run_dir: Path,
    out_dir: Path,
    report_dir: Path,
    rows: list[dict[str, float]],
    eval_jsons: dict[int, dict],
    generated: list[Path],
    expected_best: int,
) -> None:
    first = rows[0]
    final = rows[-1]
    best_marking_iou = best_row(rows, "lane_iou")
    best_marking_f1 = best_row(rows, "lane_f1")
    best_val_loss = best_row(rows, "val_loss", mode="min")
    best_miou = best_row(rows, "miou")
    best_epoch = int(best_marking_iou["epoch"])
    final_epoch = int(final["epoch"])
    if expected_best and best_epoch != expected_best:
        raise RuntimeError(
            f"Best marking IoU epoch is {best_epoch}, expected {expected_best}. "
            "Do not write D0 report until this mismatch is understood."
        )

    best_share = marking_share_summary(eval_jsons, best_epoch)
    final_share = marking_share_summary(eval_jsons, final_epoch)

    table = [
        "| metric | epoch 1 | epoch 18 best marking IoU | epoch 25 final | best epoch | best value |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| train_loss | {fmt(first['train_loss'])} | {fmt(best_marking_iou['train_loss'])} | {fmt(final['train_loss'])} | {int(best_row(rows, 'train_loss', 'min')['epoch'])} | {fmt(best_row(rows, 'train_loss', 'min')['train_loss'])} |",
        f"| val_loss | {fmt(first['val_loss'])} | {fmt(best_marking_iou['val_loss'])} | {fmt(final['val_loss'])} | {int(best_val_loss['epoch'])} | {fmt(best_val_loss['val_loss'])} |",
        f"| mIoU | {fmt(first['miou'])} | {fmt(best_marking_iou['miou'])} | {fmt(final['miou'])} | {int(best_miou['epoch'])} | {fmt(best_miou['miou'])} |",
        f"| marking IoU (`lane_iou`) | {fmt(first['lane_iou'])} | {fmt(best_marking_iou['lane_iou'])} | {fmt(final['lane_iou'])} | {int(best_marking_iou['epoch'])} | {fmt(best_marking_iou['lane_iou'])} |",
        f"| marking F1 (`lane_f1`) | {fmt(first['lane_f1'])} | {fmt(best_marking_iou['lane_f1'])} | {fmt(final['lane_f1'])} | {int(best_marking_f1['epoch'])} | {fmt(best_marking_f1['lane_f1'])} |",
        f"| marking precision | {fmt(first['lane_precision'])} | {fmt(best_marking_iou['lane_precision'])} | {fmt(final['lane_precision'])} | {int(best_row(rows, 'lane_precision')['epoch'])} | {fmt(best_row(rows, 'lane_precision')['lane_precision'])} |",
        f"| marking recall | {fmt(first['lane_recall'])} | {fmt(best_marking_iou['lane_recall'])} | {fmt(final['lane_recall'])} | {int(best_row(rows, 'lane_recall')['epoch'])} | {fmt(best_row(rows, 'lane_recall')['lane_recall'])} |",
    ]

    lines = [
        f"# D0 Weighted CE Run Summary",
        "",
        "This report is generated from the official D0 run artifacts. It does not run new inference.",
        "",
        "## Provenance",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- run_dir: `{run_dir.relative_to(PROJECT_ROOT)}`",
        f"- script: `logs/milestone_d/run_analysis/{RUN_NAME}/analysis_code/plot_d0_run.py`",
        "- label_mode: `road_marking3`",
        "- class index order after Open3D ignore filtering: `0 road`, `1 marking`, `2 other`",
        "- metric alias: CSV columns named `lane_*` mean `marking_*` for Milestone D",
        "",
        "## Headline Metrics",
        "",
        *table,
        "",
        "## Epoch Selection",
        "",
        f"- Best checkpoint by raw marking IoU is epoch `{best_epoch}`.",
        f"- Epoch `{final_epoch}` has higher recall but lower precision and lower marking IoU than epoch `{best_epoch}`.",
        "- Best checkpoint selection is post-run by raw `lane_iou`/marking IoU, not validation loss and not the smoothed scheduler metric.",
        "",
        "## Marking Prediction Share",
        "",
        "| epoch | true marking share | predicted marking share | predicted/true ratio | marking->road | road->marking | other->marking |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {best_epoch} | {100.0 * best_share['true_share']:.3f}% | {100.0 * best_share['pred_share']:.3f}% | {best_share['pred_true_ratio']:.2f} | {int(best_share['marking_to_road']):,} | {int(best_share['road_to_marking']):,} | {int(best_share['other_to_marking']):,} |",
        f"| {final_epoch} | {100.0 * final_share['true_share']:.3f}% | {100.0 * final_share['pred_share']:.3f}% | {final_share['pred_true_ratio']:.2f} | {int(final_share['marking_to_road']):,} | {int(final_share['road_to_marking']):,} | {int(final_share['other_to_marking']):,} |",
        "",
        "## Initial Reading",
        "",
        "- D0 learned the marking class substantially in early epochs: marking IoU rises from epoch 1 to epoch 18.",
        "- After epoch 18, recall stays high but precision drops enough that marking IoU and F1 decline.",
        "- The final checkpoint is therefore not the best checkpoint for the positive class.",
        "- The LR scheduler reduced LR after the plateau, but the best raw marking IoU had already occurred before the reduction produced a clear improvement.",
        "- The main run-level symptom to investigate in sampled error analysis is marking overprediction: predicted marking share grows relative to true marking share.",
        "",
        "## Generated Figures",
        "",
    ]
    for path in generated:
        lines.append(f"- `{path.relative_to(out_dir.parent)}`")
    lines.append("")

    summary_path = out_dir / "run_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "d0_weighted_ce_results.md"
    report_lines = list(lines)
    report_lines += [
        "## Pending Linked Analysis",
        "",
        "- Dataset-level remapped-class intensity analysis: `logs/milestone_d/dataset_analysis/road_marking3_intensity/`",
        "- Sampled epoch-18 error analysis: `logs/milestone_d/run_analysis/D0_weighted_ce_25ep/sampled_error_analysis_epoch18/`",
        "",
        "This report should be updated after sampled error analysis is generated, because run-level metrics alone do not identify whether remaining errors are intensity ambiguity, geometry, labels, or model behavior.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    report_dir = args.report_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_history(run_dir)
    eval_jsons = load_epoch_jsons(run_dir)
    validate_artifacts(run_dir, rows, eval_jsons, args.expected_epochs)
    training_log = load_training_log(run_dir)
    x = epochs(rows)

    generated: list[Path] = []
    best_epoch = int(best_row(rows, "lane_iou")["epoch"])
    final_epoch = int(rows[-1]["epoch"])

    plot_path = out_dir / "metrics_overview.png"
    save_overview_plot(rows, plot_path)
    generated.append(plot_path)

    plot_path = out_dir / "loss_curves.png"
    save_line_plot(
        plot_path,
        x,
        (
            ("train loss", array(rows, "train_loss"), "#1f77b4"),
            ("validation loss", array(rows, "val_loss"), "#ff7f0e"),
        ),
        "D0 loss curves",
        "loss",
    )
    generated.append(plot_path)

    plot_path = out_dir / "per_class_iou.png"
    save_line_plot(
        plot_path,
        x,
        (
            ("road IoU", array(rows, "road_iou"), "#5f6368"),
            ("marking IoU", array(rows, "lane_iou"), "#1b9e77"),
            ("other IoU", array(rows, "other_iou"), "#7570b3"),
            ("mIoU", array(rows, "miou"), "#d95f02"),
        ),
        "D0 per-class IoU",
        "IoU",
        y_min=0.0,
        y_max=1.0,
    )
    generated.append(plot_path)

    plot_path = out_dir / "marking_precision_recall_f1.png"
    save_line_plot(
        plot_path,
        x,
        (
            ("precision", array(rows, "lane_precision"), "#1f78b4"),
            ("recall", array(rows, "lane_recall"), "#33a02c"),
            ("F1", array(rows, "lane_f1"), "#e31a1c"),
            ("IoU", array(rows, "lane_iou"), "#6a3d9a"),
        ),
        "D0 marking metrics",
        "score",
        y_min=0.0,
        y_max=1.0,
    )
    generated.append(plot_path)

    plot_path = out_dir / "marking_recall_by_distance.png"
    save_line_plot(
        plot_path,
        x,
        (
            ("0-10 m", array(rows, "lane_recall_0to10m"), "#1b9e77"),
            ("10-20 m", array(rows, "lane_recall_10to20m"), "#d95f02"),
            ("20-30 m", array(rows, "lane_recall_20to30m"), "#7570b3"),
            ("30+ m", array(rows, "lane_recall_30plus"), "#e7298a"),
        ),
        "D0 marking recall by distance",
        "recall",
        y_min=0.0,
        y_max=1.0,
    )
    generated.append(plot_path)

    plot_path = out_dir / "class_true_vs_predicted_share.png"
    save_class_share_plot(eval_jsons, plot_path)
    generated.append(plot_path)

    plot_path = out_dir / "runtime_and_memory.png"
    save_runtime_plot(rows, training_log, plot_path)
    generated.append(plot_path)

    plot_path = out_dir / f"confusion_best_marking_epoch_{best_epoch:03d}.png"
    save_confusion_heatmap(
        confusion_from_json(eval_jsons[best_epoch]),
        plot_path,
        f"D0 confusion matrix, best marking IoU epoch {best_epoch}",
    )
    generated.append(plot_path)

    plot_path = out_dir / f"confusion_epoch_{final_epoch:03d}.png"
    save_confusion_heatmap(
        confusion_from_json(eval_jsons[final_epoch]),
        plot_path,
        f"D0 confusion matrix, final epoch {final_epoch}",
    )
    generated.append(plot_path)

    write_run_summary(run_dir, out_dir, report_dir, rows, eval_jsons, generated, args.expected_best_marking_epoch)

    print(f"plots_dir {out_dir}")
    print(f"report_dir {report_dir}")
    print(f"best_marking_epoch {best_epoch}")
    print("script_status PASS")


if __name__ == "__main__":
    main()

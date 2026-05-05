"""Generate plots and a compact report for a Milestone C training run.

The training driver writes machine-readable artifacts during the run. This
script turns those artifacts into the human-readable curves/figures we want for
run inspection and thesis writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "logs/milestone_c/runs"
CLASS_NAMES = ("road", "lane", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path, help="Path to a Milestone C run directory.")
    group.add_argument("--run-name", help="Run name under logs/milestone_c/runs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory. Defaults to <run_dir>/plots.",
    )
    return parser.parse_args()


def as_float(value: str | int | float | None) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_eval_history(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "eval_history.csv"
    if not path.exists():
        raise SystemExit(f"Missing eval_history.csv: {path}")

    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = {key: as_float(value) for key, value in row.items()}
            parsed["epoch"] = int(parsed["epoch"])
            rows.append(parsed)

    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def load_training_log(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "training_log.txt"
    if not path.exists():
        return {}

    out: dict[int, dict[str, float]] = {}
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
        }
    return out


def load_epoch_jsons(run_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(run_dir.glob("eval_epoch_*.json")):
        data = json.loads(path.read_text())
        epoch = int(data["epoch"])
        out[epoch] = data
    return out


def array(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([row.get(key, float("nan")) for row in rows], dtype=np.float64)


def epochs(rows: list[dict[str, float]]) -> np.ndarray:
    return array(rows, "epoch").astype(np.int64)


def save_line_plot(
    out_path: Path,
    x: np.ndarray,
    series: Iterable[tuple[str, np.ndarray]],
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, y in series:
        ax.plot(x, y, marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_overview_plot(rows: list[dict[str, float]], out_path: Path) -> None:
    x = epochs(rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(x, array(rows, "train_loss"), marker="o", label="train_loss")
    axes[0, 0].plot(x, array(rows, "val_loss"), marker="o", label="val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()

    axes[0, 1].plot(x, array(rows, "miou"), marker="o", label="mIoU")
    axes[0, 1].plot(x, array(rows, "lane_iou"), marker="o", label="lane IoU")
    axes[0, 1].set_title("Core IoU")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("IoU")
    axes[0, 1].legend()

    axes[1, 0].plot(x, array(rows, "lane_precision"), marker="o", label="lane precision")
    axes[1, 0].plot(x, array(rows, "lane_recall"), marker="o", label="lane recall")
    axes[1, 0].plot(x, array(rows, "lane_f1"), marker="o", label="lane F1")
    axes[1, 0].set_title("Lane Precision/Recall/F1")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("score")
    axes[1, 0].legend()

    axes[1, 1].plot(x, array(rows, "road_iou"), marker="o", label="road IoU")
    axes[1, 1].plot(x, array(rows, "lane_iou"), marker="o", label="lane IoU")
    axes[1, 1].plot(x, array(rows, "other_iou"), marker="o", label="other IoU")
    axes[1, 1].set_title("Per-Class IoU")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("IoU")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def confusion_from_json(data: dict) -> np.ndarray:
    return np.asarray(data["metrics"]["confusion_matrix"], dtype=np.int64)


def save_confusion_heatmap(cm: np.ndarray, out_path: Path, title: str) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_norm = np.divide(cm, row_sum, where=row_sum != 0)
    row_norm = np.nan_to_num(row_norm)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(row_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:,}\n{row_norm[i, j]:.1%}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def class_percentages(eval_jsons: dict[int, dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    epochs_sorted = np.asarray(sorted(eval_jsons), dtype=np.int64)
    true_pct = {name: [] for name in CLASS_NAMES}
    pred_pct = {name: [] for name in CLASS_NAMES}

    for epoch in epochs_sorted:
        cm = confusion_from_json(eval_jsons[int(epoch)])
        total = max(int(cm.sum()), 1)
        for idx, name in enumerate(CLASS_NAMES):
            true_pct[name].append(float(cm[idx, :].sum() / total))
            pred_pct[name].append(float(cm[:, idx].sum() / total))

    series = {}
    for name in CLASS_NAMES:
        series[f"true_{name}"] = np.asarray(true_pct[name], dtype=np.float64)
        series[f"pred_{name}"] = np.asarray(pred_pct[name], dtype=np.float64)
    return epochs_sorted, series


def save_class_distribution_plot(eval_jsons: dict[int, dict], out_path: Path) -> None:
    if not eval_jsons:
        return
    x, series = class_percentages(eval_jsons)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, class_name in zip(axes, CLASS_NAMES, strict=True):
        ax.plot(x, series[f"true_{class_name}"] * 100.0, marker="o", label="true")
        ax.plot(x, series[f"pred_{class_name}"] * 100.0, marker="o", label="predicted")
        ax.set_title(f"{class_name} share")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("percent of validated points")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_runtime_plot(
    rows: list[dict[str, float]],
    training_log: dict[int, dict[str, float]],
    out_path: Path,
) -> None:
    x = epochs(rows)
    epoch_wall = np.asarray(
        [training_log.get(int(epoch), {}).get("wall_clock_seconds", float("nan")) for epoch in x],
        dtype=np.float64,
    )
    val_wall = array(rows, "val_wall_clock_seconds")
    train_wall = epoch_wall - val_wall
    peak_memory_mb = np.asarray(
        [
            training_log.get(int(epoch), {}).get("peak_gpu_memory_bytes", row["peak_gpu_memory_bytes_val"])
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
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, peak_memory_mb, marker="o")
    axes[1].set_title("Peak PyTorch GPU Memory")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MiB")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def best_row(rows: list[dict[str, float]], key: str, mode: str = "max") -> dict[str, float]:
    valid = [row for row in rows if not math.isnan(row.get(key, float("nan")))]
    if not valid:
        return rows[-1]
    return (max if mode == "max" else min)(valid, key=lambda row: row[key])


def fmt(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_summary(
    run_dir: Path,
    out_dir: Path,
    rows: list[dict[str, float]],
    eval_jsons: dict[int, dict],
    generated: list[Path],
) -> Path:
    first = rows[0]
    final = rows[-1]
    best_miou = best_row(rows, "miou")
    best_lane_iou = best_row(rows, "lane_iou")
    best_lane_f1 = best_row(rows, "lane_f1")
    best_val_loss = best_row(rows, "val_loss", mode="min")

    lines = [
        f"# Milestone C Run Summary: {run_dir.name}",
        "",
        "## Headline Curves",
        "",
        "| metric | first epoch | final epoch | best epoch | best value |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| train_loss | {fmt(first['train_loss'])} | {fmt(final['train_loss'])} | {int(best_row(rows, 'train_loss', 'min')['epoch'])} | {fmt(best_row(rows, 'train_loss', 'min')['train_loss'])} |",
        f"| val_loss | {fmt(first['val_loss'])} | {fmt(final['val_loss'])} | {int(best_val_loss['epoch'])} | {fmt(best_val_loss['val_loss'])} |",
        f"| mIoU | {fmt(first['miou'])} | {fmt(final['miou'])} | {int(best_miou['epoch'])} | {fmt(best_miou['miou'])} |",
        f"| lane_iou | {fmt(first['lane_iou'])} | {fmt(final['lane_iou'])} | {int(best_lane_iou['epoch'])} | {fmt(best_lane_iou['lane_iou'])} |",
        f"| lane_f1 | {fmt(first['lane_f1'])} | {fmt(final['lane_f1'])} | {int(best_lane_f1['epoch'])} | {fmt(best_lane_f1['lane_f1'])} |",
        "",
        "## Final Epoch Lane Metrics",
        "",
        "```text",
        f"epoch          {int(final['epoch'])}",
        f"lane_iou       {fmt(final['lane_iou'])}",
        f"lane_precision {fmt(final['lane_precision'])}",
        f"lane_recall    {fmt(final['lane_recall'])}",
        f"lane_f1        {fmt(final['lane_f1'])}",
        "```",
        "",
    ]

    if eval_jsons:
        final_json = eval_jsons[int(final["epoch"])]
        cm = confusion_from_json(final_json)
        total = max(int(cm.sum()), 1)
        lines += [
            "## Final Epoch Class Shares",
            "",
            "| class | true percent | predicted percent | support | predicted count |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for idx, class_name in enumerate(CLASS_NAMES):
            true_count = int(cm[idx, :].sum())
            pred_count = int(cm[:, idx].sum())
            lines.append(
                f"| {class_name} | {100.0 * true_count / total:.3f}% | "
                f"{100.0 * pred_count / total:.3f}% | {true_count:,} | {pred_count:,} |"
            )
        lines.append("")

    lines += [
        "## Generated Figures",
        "",
    ]
    for path in generated:
        lines.append(f"- `{path.relative_to(run_dir)}`")

    lines += [
        "",
        "## Reading Guide",
        "",
        "- Prefer the best validation epoch for analysis if lane F1/IoU peaks before the final epoch.",
        "- Watch lane precision and predicted-lane percent together; high recall with very low precision means lane over-prediction.",
        "- Use distance-bucket lane recall to see whether the model loses lanes at range.",
        "- Use row-normalized confusion matrices to diagnose whether lane is confused with road or other.",
    ]

    summary_path = out_dir / "run_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir else RUNS_ROOT / args.run_name
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    out_dir = (args.out_dir or run_dir / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_history(run_dir)
    training_log = load_training_log(run_dir)
    eval_jsons = load_epoch_jsons(run_dir)
    x = epochs(rows)

    generated: list[Path] = []

    plots = [
        out_dir / "metrics_overview.png",
        out_dir / "loss_curves.png",
        out_dir / "per_class_iou.png",
        out_dir / "lane_precision_recall_f1.png",
        out_dir / "lane_recall_by_distance.png",
        out_dir / "runtime_and_memory.png",
        out_dir / "class_true_vs_predicted_share.png",
    ]

    save_overview_plot(rows, plots[0])
    generated.append(plots[0])
    save_line_plot(
        plots[1],
        x,
        (("train_loss", array(rows, "train_loss")), ("val_loss", array(rows, "val_loss"))),
        "Loss Curves",
        "loss",
    )
    generated.append(plots[1])
    save_line_plot(
        plots[2],
        x,
        (
            ("road IoU", array(rows, "road_iou")),
            ("lane IoU", array(rows, "lane_iou")),
            ("other IoU", array(rows, "other_iou")),
            ("mIoU", array(rows, "miou")),
        ),
        "Per-Class IoU",
        "IoU",
    )
    generated.append(plots[2])
    save_line_plot(
        plots[3],
        x,
        (
            ("precision", array(rows, "lane_precision")),
            ("recall", array(rows, "lane_recall")),
            ("F1", array(rows, "lane_f1")),
            ("IoU", array(rows, "lane_iou")),
        ),
        "Lane Metrics",
        "score",
    )
    generated.append(plots[3])
    save_line_plot(
        plots[4],
        x,
        (
            ("0-10m", array(rows, "lane_recall_0to10m")),
            ("10-20m", array(rows, "lane_recall_10to20m")),
            ("20-30m", array(rows, "lane_recall_20to30m")),
            ("30m+", array(rows, "lane_recall_30plus")),
        ),
        "Lane Recall By Distance",
        "recall",
    )
    generated.append(plots[4])
    save_runtime_plot(rows, training_log, plots[5])
    generated.append(plots[5])
    save_class_distribution_plot(eval_jsons, plots[6])
    generated.append(plots[6])

    if eval_jsons:
        final_epoch = int(rows[-1]["epoch"])
        best_lane_epoch = int(best_row(rows, "lane_f1")["epoch"])
        final_confusion = out_dir / f"confusion_epoch_{final_epoch:03d}.png"
        save_confusion_heatmap(
            confusion_from_json(eval_jsons[final_epoch]),
            final_confusion,
            f"Confusion Matrix, Epoch {final_epoch}",
        )
        generated.append(final_confusion)

        if best_lane_epoch != final_epoch and best_lane_epoch in eval_jsons:
            best_confusion = out_dir / f"confusion_best_lane_f1_epoch_{best_lane_epoch:03d}.png"
            save_confusion_heatmap(
                confusion_from_json(eval_jsons[best_lane_epoch]),
                best_confusion,
                f"Confusion Matrix, Best Lane F1 Epoch {best_lane_epoch}",
            )
            generated.append(best_confusion)

    summary_path = write_summary(run_dir, out_dir, rows, eval_jsons, generated)
    print(f"plots_dir {out_dir}")
    print(f"summary {summary_path}")
    print("generated")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()

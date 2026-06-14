#!/usr/bin/env python
"""Generate thesis-oriented F0 RGB-soft-weights analysis plots and conclusions.

This script plots already-computed D0/E0/F0 run artifacts and sampled F0
analysis CSVs. It does not run model inference.

Outputs:
    logs/milestone_f/run_analysis/F0_rgb_soft_weights/plots/
    logs/milestone_f/run_analysis/F0_rgb_soft_weights/f0_conclusions.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUN_NAME = "F0_rgb_soft_weights"
F0_RUN_DIR = PROJECT_ROOT / f"logs/milestone_f/runs/{RUN_NAME}"
E0_RUN_DIR = PROJECT_ROOT / "logs/milestone_e/runs/E0_rgb_front_v1"
D0_RUN_DIR = PROJECT_ROOT / "logs/milestone_d/runs/D0_weighted_ce_25ep"
ANALYSIS_DIR = PROJECT_ROOT / f"logs/milestone_f/run_analysis/{RUN_NAME}"
SAMPLED_DIR = ANALYSIS_DIR / "sampled_error_analysis_epoch13"
PLOTS_DIR = ANALYSIS_DIR / "plots"

CLASS_NAMES = ("road", "marking", "other")
D0_COLOR = "#4c78a8"
E0_COLOR = "#f58518"
F0_COLOR = "#54a24b"
MARKING_COLOR = "#d95f02"
VALID_COLOR = "#1b9e77"
INVALID_COLOR = "#7570b3"
ROAD_COLOR = "#666666"
OTHER_COLOR = "#8c6bb1"


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.4)
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_inputs() -> dict[str, pd.DataFrame]:
    required = {
        "f0_history": F0_RUN_DIR / "eval_history.csv",
        "e0_history": E0_RUN_DIR / "eval_history.csv",
        "d0_history": D0_RUN_DIR / "eval_history.csv",
        "summary": ANALYSIS_DIR / "summary.csv",
        "confusion_breakdown": ANALYSIS_DIR / "confusion_breakdown.csv",
        "pred_true_ratio": ANALYSIS_DIR / "pred_true_ratio.csv",
        "rgb_metrics": SAMPLED_DIR / "rgb_valid_stratified_metrics.csv",
        "raw_subtype": SAMPLED_DIR / "raw_subtype_rgb_stratified_metrics.csv",
        "distance": SAMPLED_DIR / "distance_bucket_metrics.csv",
        "group_features": SAMPLED_DIR / "group_feature_summary.csv",
        "sequence": SAMPLED_DIR / "per_sequence_metrics.csv",
    }
    return {name: pd.read_csv(require_file(path)) for name, path in required.items()}


def load_confusion(run_dir: Path, epoch: int) -> np.ndarray:
    path = require_file(run_dir / f"confusion_epoch_{epoch:03d}.npy")
    cm = np.load(path)
    if cm.shape != (3, 3):
        raise ValueError(f"{path} shape is {cm.shape}; expected (3, 3)")
    return cm.astype(np.int64, copy=False)


def load_training_log() -> pd.DataFrame:
    path = require_file(F0_RUN_DIR / "training_log.txt")
    rows = []
    for line in path.read_text().splitlines():
        fields = {}
        for part in line.split():
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        if "epoch" not in fields:
            continue
        rows.append(
            {
                "epoch": int(fields["epoch"]),
                "wall_clock_seconds": float(fields["wall_clock_seconds"]),
                "peak_gpu_memory_bytes": float(fields["peak_gpu_memory_bytes"]),
                "lr": float(fields["lr"]),
            }
        )
    if not rows:
        raise RuntimeError(f"No epoch rows found in {path}")
    return pd.DataFrame(rows).sort_values("epoch")


def best_summary_rows(summary: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    d0 = summary[(summary["run"] == "D0") & (summary["label"] == "official")].iloc[0]
    e0 = summary[(summary["run"] == "E0") & (summary["label"] == "best")].iloc[0]
    f0 = summary[(summary["run"] == "F0") & (summary["label"] == "best")].iloc[0]
    f0_final = summary[(summary["run"] == "F0") & (summary["label"] == "final")].iloc[0]
    return d0, e0, f0, f0_final


def plot_d0_e0_f0_best_metrics(data: dict[str, pd.DataFrame]) -> None:
    d0, e0, f0, _ = best_summary_rows(data["summary"])
    metrics = [
        ("marking_iou", "IoU"),
        ("marking_f1", "F1"),
        ("marking_precision", "precision"),
        ("marking_recall", "recall"),
        ("miou", "mIoU"),
    ]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    vals = {
        "D0 LiDAR": [float(d0[key]) for key, _ in metrics],
        "E0 RGB": [float(e0[key]) for key, _ in metrics],
        "F0 RGB soft weights": [float(f0[key]) for key, _ in metrics],
    }
    ax.bar(x - width, vals["D0 LiDAR"], width, label="D0 LiDAR", color=D0_COLOR)
    ax.bar(x, vals["E0 RGB"], width, label="E0 RGB", color=E0_COLOR)
    ax.bar(x + width, vals["F0 RGB soft weights"], width, label="F0 RGB soft weights", color=F0_COLOR)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("D0 vs E0 vs F0 best-checkpoint marking metrics")
    style_axes(ax)
    ax.legend()
    for i, key in enumerate([key for key, _ in metrics]):
        ax.text(i, float(f0[key]) + 0.025, f"F0-D0 {float(f0[key] - d0[key]):+.3f}", ha="center", fontsize=8.5)
    savefig(fig, PLOTS_DIR / "d0_e0_f0_best_metrics.png")


def plot_metrics_overview(data: dict[str, pd.DataFrame]) -> None:
    f0 = data["f0_history"]
    best_epoch = int(f0.loc[f0["lane_iou"].idxmax(), "epoch"])
    x = f0["epoch"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        (axes[0, 0], [("marking IoU", "lane_iou"), ("mIoU", "miou")], "IoU"),
        (axes[0, 1], [("precision", "lane_precision"), ("recall", "lane_recall"), ("F1", "lane_f1")], "score"),
        (axes[1, 0], [("train loss", "train_loss"), ("val loss", "val_loss")], "loss"),
        (axes[1, 1], [("road IoU", "road_iou"), ("marking IoU", "lane_iou"), ("other IoU", "other_iou")], "IoU"),
    ]
    for ax, series, ylabel in panels:
        for label, column in series:
            ax.plot(x, f0[column], marker="o", linewidth=2.0, label=label)
        ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ax.legend()
    axes[0, 0].set_title("F0 core IoU over epochs")
    axes[0, 1].set_title("F0 marking precision, recall, F1")
    axes[1, 0].set_title("F0 losses")
    axes[1, 1].set_title("F0 per-class IoU")
    fig.suptitle(f"F0 training behavior, best marking IoU at epoch {best_epoch}", y=0.995, fontsize=15)
    savefig(fig, PLOTS_DIR / "metrics_overview.png")


def plot_loss_curves(data: dict[str, pd.DataFrame]) -> None:
    f0 = data["f0_history"]
    best_epoch = int(f0.loc[f0["lane_iou"].idxmax(), "epoch"])
    best_loss_epoch = int(f0.loc[f0["val_loss"].idxmin(), "epoch"])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(f0["epoch"], f0["train_loss"], marker="o", linewidth=2.2, label="train loss", color=D0_COLOR)
    ax.plot(f0["epoch"], f0["val_loss"], marker="o", linewidth=2.2, label="val loss", color=F0_COLOR)
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="best marking IoU")
    ax.axvline(best_loss_epoch, color="#b2182b", linestyle=":", linewidth=1.8, alpha=0.9, label="best val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted CE loss")
    ax.set_title("F0 loss curves: loss keeps improving after marking IoU peaks")
    style_axes(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "loss_curves.png")


def plot_lr_schedule(data: dict[str, pd.DataFrame]) -> None:
    f0 = data["f0_history"]
    best_epoch = int(f0.loc[f0["lane_iou"].idxmax(), "epoch"])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.step(f0["epoch"], f0["lr"], where="post", linewidth=2.6, color="#1f78b4", label="learning rate")
    ax.scatter(f0["epoch"], f0["lr"], color="#1f78b4", s=30)
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="best marking IoU")
    lr_changes = f0[f0["lr"].diff().fillna(0.0) != 0.0]
    for _, row in lr_changes.iterrows():
        ax.annotate(
            f"drop to {row['lr']:.5f}",
            (row["epoch"], row["lr"]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("F0 ReduceLROnPlateau schedule")
    ax.set_ylim(0.0, max(f0["lr"]) * 1.15)
    style_axes(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "lr_schedule.png")


def plot_predicted_true_ratio_over_epochs(data: dict[str, pd.DataFrame]) -> None:
    rows = []
    for path in sorted(F0_RUN_DIR.glob("confusion_epoch_*.npy")):
        epoch = int(path.stem.split("_")[-1])
        cm = np.load(path)
        true_marking = cm[1, :].sum()
        pred_marking = cm[:, 1].sum()
        rows.append((epoch, pred_marking / true_marking))
    ratios = pd.DataFrame(rows, columns=["epoch", "pred_true_ratio"]).sort_values("epoch")
    best_epoch = int(data["f0_history"].loc[data["f0_history"]["lane_iou"].idxmax(), "epoch"])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(ratios["epoch"], ratios["pred_true_ratio"], marker="o", color=MARKING_COLOR, linewidth=2.2)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.4, label="perfect calibration")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label=f"best epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("predicted marking / true marking")
    ax.set_title("F0 marking overprediction over training")
    style_axes(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "predicted_true_marking_ratio_over_epochs.png")


def plot_confusion_matrix(epoch: int, out_name: str, title: str) -> None:
    cm = load_confusion(F0_RUN_DIR, epoch)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    image = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(3), CLASS_NAMES)
    ax.set_yticks(np.arange(3), CLASS_NAMES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            color = "white" if row_pct[i, j] >= 50 else "black"
            ax.text(j, i, f"{row_pct[i, j]:.1f}%\n{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row share (%)")
    savefig(fig, PLOTS_DIR / out_name)


def plot_class_true_vs_predicted_share(data: dict[str, pd.DataFrame]) -> None:
    best_epoch = int(data["f0_history"].loc[data["f0_history"]["lane_iou"].idxmax(), "epoch"])
    epochs = [best_epoch, 25]
    labels = [f"epoch {best_epoch} best IoU", "epoch 25 final"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    x = np.arange(len(CLASS_NAMES))
    width = 0.36
    colors = [ROAD_COLOR, MARKING_COLOR, OTHER_COLOR]
    for ax, epoch, label in zip(axes, epochs, labels):
        cm = load_confusion(F0_RUN_DIR, epoch)
        total = cm.sum()
        true_share = cm.sum(axis=1) / total * 100.0
        pred_share = cm.sum(axis=0) / total * 100.0
        ax.bar(x - width / 2, true_share, width, color=colors, alpha=0.45, label="true share")
        ax.bar(x + width / 2, pred_share, width, color=colors, alpha=0.95, label="predicted share")
        ax.set_xticks(x, CLASS_NAMES)
        ax.set_title(label)
        ax.set_ylabel("share of active validation points (%)")
        style_axes(ax)
        ax.legend()
        for idx in range(len(CLASS_NAMES)):
            ax.text(idx, max(true_share[idx], pred_share[idx]) + 0.25, f"{pred_share[idx] / true_share[idx]:.2f}x", ha="center", fontsize=9)
    fig.suptitle("F0 true vs predicted class share", y=0.995, fontsize=15)
    savefig(fig, PLOTS_DIR / "class_true_vs_predicted_share.png")


def plot_runtime_and_memory(data: dict[str, pd.DataFrame]) -> None:
    history = data["f0_history"]
    log = load_training_log()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.4), sharex=True)
    axes[0].plot(log["epoch"], log["wall_clock_seconds"] / 60.0, marker="o", linewidth=2.2, color=F0_COLOR)
    axes[0].set_ylabel("train+val wall clock (min)")
    axes[0].set_title("F0 runtime by epoch")
    style_axes(axes[0])
    axes[1].plot(history["epoch"], history["val_wall_clock_seconds"], marker="o", linewidth=2.2, color=D0_COLOR, label="validation seconds")
    mem_mib = history["peak_gpu_memory_bytes_val"] / (1024.0 * 1024.0)
    ax_mem = axes[1].twinx()
    ax_mem.plot(history["epoch"], mem_mib, marker="s", linewidth=2.0, color=MARKING_COLOR, label="peak GPU MiB")
    axes[1].set_ylabel("validation time (s)")
    ax_mem.set_ylabel("validation peak GPU memory (MiB)")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("F0 validation time and GPU memory")
    style_axes(axes[1])
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax_mem.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout(pad=1.4)
    fig.savefig(PLOTS_DIR / "runtime_and_memory.png", dpi=240, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def plot_rgb_valid_vs_invalid_metrics(data: dict[str, pd.DataFrame]) -> None:
    df = data["rgb_metrics"]
    order = ["all", "rgb_valid", "rgb_invalid"]
    metrics = [
        ("marking_iou", "IoU"),
        ("marking_precision", "precision"),
        ("marking_recall", "recall"),
        ("marking_f1", "F1"),
    ]
    x = np.arange(len(metrics))
    width = 0.25
    colors = {"all": "#999999", "rgb_valid": VALID_COLOR, "rgb_invalid": INVALID_COLOR}
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for offset, stratum in zip((-width, 0.0, width), order):
        row = df[df["stratum"] == stratum].iloc[0]
        values = [float(row[key]) for key, _ in metrics]
        ax.bar(x + offset, values, width, color=colors[stratum], label=stratum)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("F0 sampled metrics split by RGB validity")
    style_axes(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "rgb_valid_vs_invalid_metrics.png")


def plot_rgb_valid_vs_invalid_confusion(data: dict[str, pd.DataFrame]) -> None:
    df = data["rgb_metrics"]
    order = ["all", "rgb_valid", "rgb_invalid"]
    metrics = [
        ("road_to_marking", "road -> marking"),
        ("marking_to_road", "marking -> road"),
        ("other_to_marking", "other -> marking"),
    ]
    x = np.arange(len(metrics))
    width = 0.25
    colors = {"all": "#999999", "rgb_valid": VALID_COLOR, "rgb_invalid": INVALID_COLOR}
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for offset, stratum in zip((-width, 0.0, width), order):
        row = df[df["stratum"] == stratum].iloc[0]
        values = [int(row[key]) for key, _ in metrics]
        ax.bar(x + offset, values, width, color=colors[stratum], label=stratum)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("sampled point count")
    ax.set_title("F0 key marking errors split by RGB validity")
    style_axes(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "rgb_valid_vs_invalid_confusion.png")


def plot_raw_subtype_recall_d0_e0_f0(data: dict[str, pd.DataFrame]) -> None:
    d0_path = require_file(
        PROJECT_ROOT / "logs/milestone_d/run_analysis/D0_weighted_ce_25ep/sampled_error_analysis_epoch18/raw_marking_subtype_summary.csv"
    )
    e0_path = require_file(
        PROJECT_ROOT / "logs/milestone_e/run_analysis/E0_rgb_front_v1/sampled_error_analysis_epoch14/raw_subtype_rgb_stratified_metrics.csv"
    )
    d0 = pd.read_csv(d0_path)
    e0 = pd.read_csv(e0_path)
    f0 = data["raw_subtype"]
    e0_all = e0[e0["stratum"] == "all"].copy()
    f0_all = f0[f0["stratum"] == "all"].copy()
    labels = [f"raw {int(row.raw_id)}\n{row.raw_name.replace('_', ' ')}" for row in f0_all.itertuples()]
    d0_recall, e0_recall, f0_recall = [], [], []
    for raw_id in f0_all["raw_id"]:
        d0_recall.append(float(d0[d0["raw_id"] == raw_id]["recall"].iloc[0]))
        e0_recall.append(float(e0_all[e0_all["raw_id"] == raw_id]["marking_recall"].iloc[0]))
        f0_recall.append(float(f0_all[f0_all["raw_id"] == raw_id]["marking_recall"].iloc[0]))
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.bar(x - width, d0_recall, width, label="D0", color=D0_COLOR)
    ax.bar(x, e0_recall, width, label="E0", color=E0_COLOR)
    ax.bar(x + width, f0_recall, width, label="F0", color=F0_COLOR)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("recall on true raw marking subtype")
    ax.set_title("Raw marking subtype recall, D0 vs E0 vs F0")
    style_axes(ax)
    ax.legend()
    for i, (d, f) in enumerate(zip(d0_recall, f0_recall)):
        ax.text(i, max(d, f) + 0.025, f"F0-D0 {f - d:+.3f}", ha="center", fontsize=8.5)
    savefig(fig, PLOTS_DIR / "raw_subtype_recall_d0_e0_f0.png")


def plot_distance_marking_metrics(data: dict[str, pd.DataFrame]) -> None:
    df = data["distance"]
    labels = df["bucket"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.4), sharex=True)
    for column, label, color in [
        ("marking_iou", "IoU", MARKING_COLOR),
        ("marking_precision", "precision", VALID_COLOR),
        ("marking_recall", "recall", INVALID_COLOR),
    ]:
        axes[0].plot(x, df[column], marker="o", linewidth=2.2, label=label, color=color)
    axes[0].set_ylabel("score")
    axes[0].set_title("F0 marking quality by distance bucket")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()
    style_axes(axes[0])

    axes[1].bar(x, df["predicted_true_marking_ratio"], color=MARKING_COLOR, alpha=0.85)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.4)
    axes[1].set_ylabel("predicted / true")
    axes[1].set_xlabel("distance bucket")
    axes[1].set_title("F0 marking overprediction by distance")
    axes[1].set_xticks(x, labels)
    style_axes(axes[1])
    savefig(fig, PLOTS_DIR / "distance_marking_metrics.png")


def plot_rgb_by_outcome_boxplots(data: dict[str, pd.DataFrame]) -> None:
    df = data["group_features"]
    groups = ["road_tp", "road_to_marking", "marking_tp", "marking_to_road", "other_to_marking"]
    labels = ["road TP", "road -> marking", "marking TP", "marking -> road", "other -> marking"]
    channels = [("red", "#d73027"), ("green", "#1a9850"), ("blue", "#4575b4")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
    for ax, (channel, color) in zip(axes, channels):
        medians, lower, upper = [], [], []
        for group in groups:
            row = df[df["group"] == group].iloc[0]
            p25 = float(row[f"{channel}_p25"])
            median = float(row[f"{channel}_median"])
            p75 = float(row[f"{channel}_p75"])
            medians.append(median)
            lower.append(median - p25)
            upper.append(p75 - median)
        x = np.arange(len(groups))
        ax.errorbar(
            x,
            medians,
            yerr=[lower, upper],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=3,
            capsize=6,
            markersize=7,
        )
        ax.set_title(f"{channel.upper()} channel")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylim(0.0, 0.75)
        style_axes(ax)
    axes[0].set_ylabel("RGB value, normalized 0-1\nmedian with p25-p75 interval")
    fig.suptitle("F0 RGB feature summaries by prediction outcome", y=0.995, fontsize=15)
    savefig(fig, PLOTS_DIR / "rgb_by_outcome_boxplots.png")


def plot_top_sequences_road_to_marking(data: dict[str, pd.DataFrame]) -> None:
    df = data["sequence"].sort_values("road_to_marking", ascending=False).head(9)
    labels = [str(seq) for seq in df["seq_id"]]
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(11, 5.8))
    ax1.bar(x, df["road_to_marking"], color=MARKING_COLOR, alpha=0.85, label="road -> marking count")
    ax1.set_ylabel("road -> marking count")
    ax1.set_xlabel("validation sequence")
    ax1.set_xticks(x, labels)
    style_axes(ax1)
    ax2 = ax1.twinx()
    ax2.plot(x, df["predicted_true_marking_ratio"], marker="o", color="black", linewidth=2.0, label="pred/true ratio")
    ax2.set_ylabel("predicted / true marking")
    ax1.set_title("F0 worst sequences by road-to-marking false positives")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout(pad=1.4)
    fig.savefig(PLOTS_DIR / "top_sequences_road_to_marking.png", dpi=240, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def write_readme(data: dict[str, pd.DataFrame]) -> None:
    rgb = data["rgb_metrics"]
    all_row = rgb[rgb["stratum"] == "all"].iloc[0]
    valid = rgb[rgb["stratum"] == "rgb_valid"].iloc[0]
    invalid = rgb[rgb["stratum"] == "rgb_invalid"].iloc[0]
    readme = f"""# F0 RGB-Soft-Weights Plots

These plots summarize the completed F0 run and the epoch-13 sampled error
analysis. They are generated by:

`logs/milestone_f/run_analysis/analysis_code/plot_f0_rgb_soft_weights_analysis.py`

## How To Read The Plots

1. `d0_e0_f0_best_metrics.png`
   - Compares D0 epoch 18, E0 epoch 14, and F0 epoch 13.
   - Main reading: F0 should show whether softened weights turned RGB into a real IoU/F1 gain.

2. `metrics_overview.png`
   - Shows F0 training loss, validation loss, IoU, precision, recall, and F1 by epoch.
   - Main reading: confirms where F0 peaks and whether late epochs drift.

3. `loss_curves.png`
   - Standalone train/validation loss curves.
   - Main reading: checks whether lowest loss still disagrees with best marking IoU.

4. `lr_schedule.png`
   - Shows the ReduceLROnPlateau learning-rate schedule.
   - Main reading: documents whether LR drops occurred before or after the best F0 epoch.

5. `predicted_true_marking_ratio_over_epochs.png`
   - Shows F0 predicted marking count divided by true marking count for each epoch.
   - Main reading: values above 1 mean overprediction.

6. `confusion_best_marking_epoch_013.png`
   - Row-normalized confusion matrix for the official F0 best marking-IoU checkpoint.

7. `confusion_epoch_025.png`
   - Row-normalized confusion matrix for the final F0 checkpoint.

8. `class_true_vs_predicted_share.png`
   - True vs predicted class shares for best and final F0 checkpoints.

9. `runtime_and_memory.png`
   - Documents F0 compute cost.

10. `rgb_valid_vs_invalid_metrics.png`
   - Compares sampled marking metrics for all, RGB-valid, and RGB-invalid points.

11. `rgb_valid_vs_invalid_confusion.png`
   - Splits key errors by RGB validity.

12. `raw_subtype_recall_d0_e0_f0.png`
   - Compares true raw marking subtype recall for D0, E0, and F0.

13. `distance_marking_metrics.png`
   - Shows F0 marking IoU/precision/recall and predicted/true ratio by distance bucket.

14. `rgb_by_outcome_boxplots.png`
   - Shows RGB channel p25/median/p75 summaries by outcome group.
   - This is an IQR-summary plot from saved CSV summaries, not a raw per-point boxplot.

15. `top_sequences_road_to_marking.png`
   - Shows sequences with the most road-to-marking false positives.

## Key Numeric Anchor

Sampled F0 epoch 13:

- all marking IoU: `{all_row['marking_iou']:.6f}`
- all precision: `{all_row['marking_precision']:.6f}`
- all recall: `{all_row['marking_recall']:.6f}`
- all predicted/true marking ratio: `{all_row['predicted_true_marking_ratio']:.3f}`
- RGB-valid predicted/true ratio: `{valid['predicted_true_marking_ratio']:.3f}`
- RGB-invalid predicted/true ratio: `{invalid['predicted_true_marking_ratio']:.3f}`
"""
    (PLOTS_DIR / "README.md").write_text(readme)


def write_conclusions(data: dict[str, pd.DataFrame]) -> None:
    summary = data["summary"]
    d0, e0, f0, f0_final = best_summary_rows(summary)
    sampled = json.loads(require_file(SAMPLED_DIR / "summary.json").read_text())
    rgb = data["rgb_metrics"]
    all_row = rgb[rgb["stratum"] == "all"].iloc[0]
    valid = rgb[rgb["stratum"] == "rgb_valid"].iloc[0]
    invalid = rgb[rgb["stratum"] == "rgb_invalid"].iloc[0]
    ratio = data["pred_true_ratio"]
    f0_ratio = float(
        ratio[(ratio["run"] == "F0") & (ratio["label"] == "best") & (ratio["class"] == "marking")]["pred_true_ratio"].iloc[0]
    )
    e0_ratio = float(
        ratio[(ratio["run"] == "E0") & (ratio["label"] == "best") & (ratio["class"] == "marking")]["pred_true_ratio"].iloc[0]
    )

    text = f"""# F0 Conclusions

## Main Conclusion

F0 is the first clear improvement over the D0 LiDAR-only baseline and the E0
RGB-front run. The official F0 checkpoint is epoch `{int(f0['epoch'])}`, selected
by maximum raw marking IoU.

| metric | D0 epoch 18 | E0 epoch 14 | F0 epoch 13 |
| --- | ---: | ---: | ---: |
| marking IoU | {d0['marking_iou']:.6f} | {e0['marking_iou']:.6f} | {f0['marking_iou']:.6f} |
| F1 | {d0['marking_f1']:.6f} | {e0['marking_f1']:.6f} | {f0['marking_f1']:.6f} |
| precision | {d0['marking_precision']:.6f} | {e0['marking_precision']:.6f} | {f0['marking_precision']:.6f} |
| recall | {d0['marking_recall']:.6f} | {e0['marking_recall']:.6f} | {f0['marking_recall']:.6f} |
| mIoU | {d0['miou']:.6f} | {e0['miou']:.6f} | {f0['miou']:.6f} |

Interpretation: E0 showed that RGB increased recall but overpredicted marking.
F0 softened the marking weight and widened the first feature embedding, improving
precision while keeping recall above D0.

## Best Epoch vs Final Epoch

F0 still should not use the final epoch as the official checkpoint.

| metric | epoch 13 best | epoch 25 final |
| --- | ---: | ---: |
| marking IoU | {f0['marking_iou']:.6f} | {f0_final['marking_iou']:.6f} |
| precision | {f0['marking_precision']:.6f} | {f0_final['marking_precision']:.6f} |
| recall | {f0['marking_recall']:.6f} | {f0_final['marking_recall']:.6f} |
| val loss | {f0['val_loss']:.6f} | {f0_final['val_loss']:.6f} |

Validation loss continues improving after the best marking IoU, so loss alone
would again select a later, more recall-heavy but less balanced checkpoint.

## Overprediction

F0 reduced E0's overprediction pattern. Training-time marking predicted/true
ratio at the best checkpoint:

```text
E0: {e0_ratio:.3f}
F0: {f0_ratio:.3f}
```

The sampled epoch-13 pass gives:

```text
all:         pred/true {all_row['predicted_true_marking_ratio']:.3f}, IoU {all_row['marking_iou']:.6f}
rgb_valid:   pred/true {valid['predicted_true_marking_ratio']:.3f}, IoU {valid['marking_iou']:.6f}
rgb_invalid: pred/true {invalid['predicted_true_marking_ratio']:.3f}, IoU {invalid['marking_iou']:.6f}
```

## Remaining Questions

Use the sampled CSVs and plots to inspect whether remaining errors are still
concentrated in RGB-valid points, long range, specific validation sequences,
or raw marking subtypes. The key files are:

- `sampled_error_analysis_epoch13/rgb_valid_stratified_metrics.csv`
- `sampled_error_analysis_epoch13/per_sequence_metrics.csv`
- `sampled_error_analysis_epoch13/distance_bucket_metrics.csv`
- `sampled_error_analysis_epoch13/raw_subtype_rgb_stratified_metrics.csv`
- `plots/README.md`

## Provenance

- sampled analysis checkpoint: `{sampled['checkpoint']}`
- sampled steps: `{sampled['steps']}`
- sampled seed: `{sampled['seed']}`
- sampled split: `{sampled['split']}`
"""
    (ANALYSIS_DIR / "f0_conclusions.md").write_text(text)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    best_epoch = int(data["f0_history"].loc[data["f0_history"]["lane_iou"].idxmax(), "epoch"])
    plot_d0_e0_f0_best_metrics(data)
    plot_metrics_overview(data)
    plot_loss_curves(data)
    plot_lr_schedule(data)
    plot_predicted_true_ratio_over_epochs(data)
    plot_confusion_matrix(best_epoch, "confusion_best_marking_epoch_013.png", "F0 epoch 13 confusion matrix, row-normalized")
    plot_confusion_matrix(25, "confusion_epoch_025.png", "F0 epoch 25 confusion matrix, row-normalized")
    plot_class_true_vs_predicted_share(data)
    plot_runtime_and_memory(data)
    plot_rgb_valid_vs_invalid_metrics(data)
    plot_rgb_valid_vs_invalid_confusion(data)
    plot_raw_subtype_recall_d0_e0_f0(data)
    plot_distance_marking_metrics(data)
    plot_rgb_by_outcome_boxplots(data)
    plot_top_sequences_road_to_marking(data)
    write_readme(data)
    write_conclusions(data)
    print(f"wrote plots to {PLOTS_DIR}")
    print(f"wrote conclusions to {ANALYSIS_DIR / 'f0_conclusions.md'}")
    print("script_status PASS")


if __name__ == "__main__":
    main()

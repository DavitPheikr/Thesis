#!/usr/bin/env python
"""Generate thesis-oriented G0 RGB+Lovasz analysis plots and conclusions.

Adapted from the Milestone F0 plot script. It plots already-computed D0/E0/F0/G0
run artifacts and the G0 sampled analysis CSVs. It does not run model inference.

Loss handling: for G0, ``eval_history`` ``train_loss``/``val_loss`` are the TOTAL
loss (CE + lambda*Lovasz). The detailed CE/Lovasz breakdown and the fair
F0-vs-G0 CE overlay live in ``g0_loss_component_analysis.py``; this script keeps
the run-level overview and the four-way comparisons.

Outputs:
    logs/milestone_g/run_analysis/G0_rgb_lovasz/plots/
    logs/milestone_g/run_analysis/G0_rgb_lovasz/g0_conclusions.md
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
RUN_NAME = "G0_rgb_lovasz"
G0_RUN_DIR = PROJECT_ROOT / f"logs/milestone_g/runs/{RUN_NAME}"
F0_RUN_DIR = PROJECT_ROOT / "logs/milestone_f/runs/F0_rgb_soft_weights"
E0_RUN_DIR = PROJECT_ROOT / "logs/milestone_e/runs/E0_rgb_front_v1"
D0_RUN_DIR = PROJECT_ROOT / "logs/milestone_d/runs/D0_weighted_ce_25ep"
ANALYSIS_DIR = PROJECT_ROOT / f"logs/milestone_g/run_analysis/{RUN_NAME}"
PLOTS_DIR = ANALYSIS_DIR / "plots"

CLASS_NAMES = ("road", "marking", "other")
D0_COLOR = "#4c78a8"
E0_COLOR = "#f58518"
F0_COLOR = "#54a24b"
G0_COLOR = "#b279a2"
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


def g0_best_epoch() -> int:
    df = pd.read_csv(require_file(G0_RUN_DIR / "eval_history.csv"))
    return int(df.loc[df["lane_iou"].idxmax(), "epoch"])


def find_sampled_dir() -> Path:
    best = g0_best_epoch()
    candidate = ANALYSIS_DIR / f"sampled_error_analysis_epoch{best}"
    if candidate.exists():
        return candidate
    matches = sorted(ANALYSIS_DIR.glob("sampled_error_analysis_epoch*"))
    if not matches:
        raise FileNotFoundError(
            f"No sampled_error_analysis_epoch* dir under {ANALYSIS_DIR}; "
            "run g0_sampled_error_analysis.py first."
        )
    return matches[-1]


SAMPLED_DIR = None  # resolved in main()


def load_inputs() -> dict[str, pd.DataFrame]:
    required = {
        "g0_history": G0_RUN_DIR / "eval_history.csv",
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
    path = require_file(G0_RUN_DIR / "training_log.txt")
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


def summary_rows(summary: pd.DataFrame) -> dict[str, pd.Series]:
    def srow(run: str, label: str) -> pd.Series:
        return summary[(summary["run"] == run) & (summary["label"] == label)].iloc[0]

    return {
        "D0": srow("D0", "official"),
        "E0": srow("E0", "best"),
        "F0": srow("F0", "best"),
        "G0": srow("G0", "best"),
        "F0_final": srow("F0", "final"),
        "G0_final": srow("G0", "final"),
    }


def plot_best_metrics(data: dict[str, pd.DataFrame]) -> None:
    rows = summary_rows(data["summary"])
    d0, e0, f0, g0 = rows["D0"], rows["E0"], rows["F0"], rows["G0"]
    metrics = [
        ("marking_iou", "IoU"),
        ("marking_f1", "F1"),
        ("marking_precision", "precision"),
        ("marking_recall", "recall"),
        ("miou", "mIoU"),
    ]
    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.bar(x - 1.5 * width, [float(d0[k]) for k, _ in metrics], width, label="D0 LiDAR", color=D0_COLOR)
    ax.bar(x - 0.5 * width, [float(e0[k]) for k, _ in metrics], width, label="E0 RGB", color=E0_COLOR)
    ax.bar(x + 0.5 * width, [float(f0[k]) for k, _ in metrics], width, label="F0 RGB soft weights", color=F0_COLOR)
    ax.bar(x + 1.5 * width, [float(g0[k]) for k, _ in metrics], width, label="G0 RGB + Lovasz", color=G0_COLOR)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("D0 vs E0 vs F0 vs G0 best-checkpoint marking metrics")
    style_axes(ax)
    ax.legend()
    for i, key in enumerate([k for k, _ in metrics]):
        ax.text(i + 1.5 * width, float(g0[key]) + 0.02, f"{float(g0[key] - f0[key]):+.3f}", ha="center", fontsize=8)
    savefig(fig, PLOTS_DIR / "d0_e0_f0_g0_best_metrics.png")


def plot_metrics_overview(data: dict[str, pd.DataFrame]) -> None:
    g0 = data["g0_history"]
    best_epoch = int(g0.loc[g0["lane_iou"].idxmax(), "epoch"])
    x = g0["epoch"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        (axes[0, 0], [("marking IoU", "lane_iou"), ("mIoU", "miou")], "IoU"),
        (axes[0, 1], [("precision", "lane_precision"), ("recall", "lane_recall"), ("F1", "lane_f1")], "score"),
        (axes[1, 0], [("train total loss", "train_loss"), ("val total loss", "val_loss")], "loss (total)"),
        (axes[1, 1], [("road IoU", "road_iou"), ("marking IoU", "lane_iou"), ("other IoU", "other_iou")], "IoU"),
    ]
    for ax, series, ylabel in panels:
        for label, column in series:
            ax.plot(x, g0[column], marker="o", linewidth=2.0, label=label)
        ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        ax.legend()
    axes[0, 0].set_title("G0 core IoU over epochs")
    axes[0, 1].set_title("G0 marking precision, recall, F1")
    axes[1, 0].set_title("G0 TOTAL loss (CE + lambda*Lovasz); see component plots for split")
    axes[1, 1].set_title("G0 per-class IoU")
    fig.suptitle(f"G0 training behavior, best marking IoU at epoch {best_epoch}", y=0.995, fontsize=15)
    savefig(fig, PLOTS_DIR / "metrics_overview.png")


def plot_lr_schedule(data: dict[str, pd.DataFrame]) -> None:
    g0 = data["g0_history"]
    best_epoch = int(g0.loc[g0["lane_iou"].idxmax(), "epoch"])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.step(g0["epoch"], g0["lr"], where="post", linewidth=2.6, color="#1f78b4", label="learning rate")
    ax.scatter(g0["epoch"], g0["lr"], color="#1f78b4", s=30)
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="best marking IoU")
    lr_changes = g0[g0["lr"].diff().fillna(0.0) != 0.0]
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
    ax.set_title("G0 ReduceLROnPlateau schedule")
    ax.set_ylim(0.0, max(g0["lr"]) * 1.15)
    style_axes(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "lr_schedule.png")


def plot_predicted_true_ratio_over_epochs(data: dict[str, pd.DataFrame]) -> None:
    rows = []
    for path in sorted(G0_RUN_DIR.glob("confusion_epoch_*.npy")):
        epoch = int(path.stem.split("_")[-1])
        cm = np.load(path)
        true_marking = cm[1, :].sum()
        pred_marking = cm[:, 1].sum()
        rows.append((epoch, pred_marking / true_marking))
    ratios = pd.DataFrame(rows, columns=["epoch", "pred_true_ratio"]).sort_values("epoch")
    best_epoch = int(data["g0_history"].loc[data["g0_history"]["lane_iou"].idxmax(), "epoch"])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(ratios["epoch"], ratios["pred_true_ratio"], marker="o", color=MARKING_COLOR, linewidth=2.2, label="G0")
    # F0 reference curve for overprediction comparison.
    f0_rows = []
    for path in sorted(F0_RUN_DIR.glob("confusion_epoch_*.npy")):
        epoch = int(path.stem.split("_")[-1])
        cm = np.load(path)
        f0_rows.append((epoch, cm[:, 1].sum() / cm[1, :].sum()))
    if f0_rows:
        f0_ratios = pd.DataFrame(f0_rows, columns=["epoch", "pred_true_ratio"]).sort_values("epoch")
        ax.plot(f0_ratios["epoch"], f0_ratios["pred_true_ratio"], marker="s", color=F0_COLOR, linewidth=1.8, alpha=0.7, label="F0")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.4, label="perfect calibration")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label=f"G0 best epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("predicted marking / true marking")
    ax.set_title("G0 vs F0 marking overprediction over training")
    style_axes(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "predicted_true_marking_ratio_over_epochs.png")


def plot_confusion_matrix(epoch: int, out_name: str, title: str) -> None:
    cm = load_confusion(G0_RUN_DIR, epoch)
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
    g0 = data["g0_history"]
    best_epoch = int(g0.loc[g0["lane_iou"].idxmax(), "epoch"])
    final_epoch = int(g0["epoch"].max())
    epochs = [best_epoch, final_epoch]
    labels = [f"epoch {best_epoch} best IoU", f"epoch {final_epoch} final"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    x = np.arange(len(CLASS_NAMES))
    width = 0.36
    colors = [ROAD_COLOR, MARKING_COLOR, OTHER_COLOR]
    for ax, epoch, label in zip(axes, epochs, labels):
        cm = load_confusion(G0_RUN_DIR, epoch)
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
    fig.suptitle("G0 true vs predicted class share", y=0.995, fontsize=15)
    savefig(fig, PLOTS_DIR / "class_true_vs_predicted_share.png")


def plot_runtime_and_memory(data: dict[str, pd.DataFrame]) -> None:
    history = data["g0_history"]
    log = load_training_log()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.4), sharex=True)
    axes[0].plot(log["epoch"], log["wall_clock_seconds"] / 60.0, marker="o", linewidth=2.2, color=G0_COLOR)
    axes[0].set_ylabel("train+val wall clock (min)")
    axes[0].set_title("G0 runtime by epoch")
    style_axes(axes[0])
    axes[1].plot(history["epoch"], history["val_wall_clock_seconds"], marker="o", linewidth=2.2, color=D0_COLOR, label="validation seconds")
    mem_mib = history["peak_gpu_memory_bytes_val"] / (1024.0 * 1024.0)
    ax_mem = axes[1].twinx()
    ax_mem.plot(history["epoch"], mem_mib, marker="s", linewidth=2.0, color=MARKING_COLOR, label="peak GPU MiB")
    axes[1].set_ylabel("validation time (s)")
    ax_mem.set_ylabel("validation peak GPU memory (MiB)")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("G0 validation time and GPU memory")
    style_axes(axes[1])
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax_mem.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout(pad=1.4)
    fig.savefig(PLOTS_DIR / "runtime_and_memory.png", dpi=240, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def plot_f0_vs_g0_curves(data: dict[str, pd.DataFrame]) -> None:
    f0 = data["f0_history"]
    g0 = data["g0_history"]
    panels = [
        ("lane_iou", "marking IoU"),
        ("lane_f1", "marking F1"),
        ("lane_precision", "marking precision"),
        ("lane_recall", "marking recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (column, label) in zip(axes.ravel(), panels):
        ax.plot(f0["epoch"], f0[column], marker="s", linewidth=2.0, color=F0_COLOR, label="F0")
        ax.plot(g0["epoch"], g0[column], marker="o", linewidth=2.2, color=G0_COLOR, label="G0")
        ax.axvline(int(f0.loc[f0["lane_iou"].idxmax(), "epoch"]), color=F0_COLOR, linestyle=":", alpha=0.6)
        ax.axvline(int(g0.loc[g0["lane_iou"].idxmax(), "epoch"]), color=G0_COLOR, linestyle="--", alpha=0.6)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.set_title(f"F0 vs G0 {label}")
        style_axes(ax)
        ax.legend()
    fig.suptitle("F0 vs G0 marking-metric curves", y=0.995, fontsize=15)
    savefig(fig, PLOTS_DIR / "f0_vs_g0_marking_curves.png")


def plot_best_to_final_drift(data: dict[str, pd.DataFrame]) -> None:
    rows = summary_rows(data["summary"])
    ratio = data["pred_true_ratio"]

    def pt(run: str, label: str) -> float:
        return float(
            ratio[(ratio["run"] == run) & (ratio["label"] == label) & (ratio["class"] == "marking")]["pred_true_ratio"].iloc[0]
        )

    metrics = ["marking_iou", "marking_precision", "marking_recall"]
    metric_labels = ["IoU", "precision", "recall"]
    f0_drift = [float(rows["F0_final"][m] - rows["F0"][m]) for m in metrics]
    g0_drift = [float(rows["G0_final"][m] - rows["G0"][m]) for m in metrics]
    f0_drift.append(pt("F0", "final") - pt("F0", "best"))
    g0_drift.append(pt("G0", "final") - pt("G0", "best"))
    labels = metric_labels + ["pred/true"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - width / 2, f0_drift, width, color=F0_COLOR, label="F0 final-best")
    ax.bar(x + width / 2, g0_drift, width, color=G0_COLOR, label="G0 final-best")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("final minus best (drift)")
    ax.set_title("Best-to-final drift: G0 vs F0 (closer to 0 = steadier)")
    style_axes(ax)
    ax.legend()
    for i, (a, b) in enumerate(zip(f0_drift, g0_drift)):
        ax.text(i - width / 2, a, f"{a:+.3f}", ha="center", va="bottom" if a >= 0 else "top", fontsize=8)
        ax.text(i + width / 2, b, f"{b:+.3f}", ha="center", va="bottom" if b >= 0 else "top", fontsize=8)
    savefig(fig, PLOTS_DIR / "best_to_final_drift.png")


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
    ax.set_title("G0 sampled metrics split by RGB validity")
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
    ax.set_title("G0 key marking errors split by RGB validity")
    style_axes(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "rgb_valid_vs_invalid_confusion.png")


def plot_raw_subtype_recall(data: dict[str, pd.DataFrame]) -> None:
    d0_path = require_file(
        PROJECT_ROOT / "logs/milestone_d/run_analysis/D0_weighted_ce_25ep/sampled_error_analysis_epoch18/raw_marking_subtype_summary.csv"
    )
    e0_path = require_file(
        PROJECT_ROOT / "logs/milestone_e/run_analysis/E0_rgb_front_v1/sampled_error_analysis_epoch14/raw_subtype_rgb_stratified_metrics.csv"
    )
    f0_path = require_file(
        PROJECT_ROOT / "logs/milestone_f/run_analysis/F0_rgb_soft_weights/sampled_error_analysis_epoch13/raw_subtype_rgb_stratified_metrics.csv"
    )
    d0 = pd.read_csv(d0_path)
    e0 = pd.read_csv(e0_path)
    f0 = pd.read_csv(f0_path)
    g0 = data["raw_subtype"]
    e0_all = e0[e0["stratum"] == "all"].copy()
    f0_all = f0[f0["stratum"] == "all"].copy()
    g0_all = g0[g0["stratum"] == "all"].copy()
    labels = [f"raw {int(row.raw_id)}\n{row.raw_name.replace('_', ' ')}" for row in g0_all.itertuples()]
    d0_recall, e0_recall, f0_recall, g0_recall = [], [], [], []
    for raw_id in g0_all["raw_id"]:
        d0_recall.append(float(d0[d0["raw_id"] == raw_id]["recall"].iloc[0]))
        e0_recall.append(float(e0_all[e0_all["raw_id"] == raw_id]["marking_recall"].iloc[0]))
        f0_recall.append(float(f0_all[f0_all["raw_id"] == raw_id]["marking_recall"].iloc[0]))
        g0_recall.append(float(g0_all[g0_all["raw_id"] == raw_id]["marking_recall"].iloc[0]))
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(x - 1.5 * width, d0_recall, width, label="D0", color=D0_COLOR)
    ax.bar(x - 0.5 * width, e0_recall, width, label="E0", color=E0_COLOR)
    ax.bar(x + 0.5 * width, f0_recall, width, label="F0", color=F0_COLOR)
    ax.bar(x + 1.5 * width, g0_recall, width, label="G0", color=G0_COLOR)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("recall on true raw marking subtype")
    ax.set_title("Raw marking subtype recall, D0 vs E0 vs F0 vs G0")
    style_axes(ax)
    ax.legend()
    for i, (f, g) in enumerate(zip(f0_recall, g0_recall)):
        ax.text(i + 1.5 * width, max(f, g) + 0.02, f"{g - f:+.3f}", ha="center", fontsize=8)
    savefig(fig, PLOTS_DIR / "raw_subtype_recall_d0_e0_f0_g0.png")


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
    axes[0].set_title("G0 marking quality by distance bucket")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()
    style_axes(axes[0])

    axes[1].bar(x, df["predicted_true_marking_ratio"], color=MARKING_COLOR, alpha=0.85)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.4)
    axes[1].set_ylabel("predicted / true")
    axes[1].set_xlabel("distance bucket")
    axes[1].set_title("G0 marking overprediction by distance")
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
        ax.errorbar(x, medians, yerr=[lower, upper], fmt="o", color=color, ecolor=color, elinewidth=3, capsize=6, markersize=7)
        ax.set_title(f"{channel.upper()} channel")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylim(0.0, 0.75)
        style_axes(ax)
    axes[0].set_ylabel("RGB value, normalized 0-1\nmedian with p25-p75 interval")
    fig.suptitle("G0 RGB feature summaries by prediction outcome", y=0.995, fontsize=15)
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
    ax1.set_title("G0 worst sequences by road-to-marking false positives")
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
    readme = f"""# G0 RGB + Lovasz Plots

These plots summarize the completed G0 run, the G0 best-epoch sampled error
analysis, and the loss-component analysis. They are generated by:

- `plot_g0_rgb_lovasz_analysis.py` (run-level + four-way + sampled plots)
- `g0_loss_component_analysis.py` (CE/Lovasz/total component plots)

## Fairness Caveat (read first)

For G0, `eval_history` `train_loss`/`val_loss` are the TOTAL loss
(CE + lambda*Lovasz). The ONLY fair loss-vs-loss comparison against F0 is F0
`val_loss` (pure CE) vs G0 `val_ce` (CE component), shown in
`f0_vs_g0_val_ce.png`. Do not compare F0 `val_loss` to G0 total loss.

## Plot Index

1. `d0_e0_f0_g0_best_metrics.png` - four-way best-checkpoint marking metrics.
2. `metrics_overview.png` - G0 IoU/precision/recall/F1 and TOTAL loss by epoch.
3. `f0_vs_g0_marking_curves.png` - F0 vs G0 IoU/F1/precision/recall curves.
4. `best_to_final_drift.png` - final-minus-best drift, G0 vs F0 (the core test).
5. `predicted_true_marking_ratio_over_epochs.png` - G0 vs F0 overprediction.
6. `lr_schedule.png` - ReduceLROnPlateau schedule.
7. `confusion_best_marking_epoch_*.png`, `confusion_final_epoch_*.png`.
8. `class_true_vs_predicted_share.png` - best and final share calibration.
9. `runtime_and_memory.png` - compute cost.
10. `rgb_valid_vs_invalid_metrics.png`, `rgb_valid_vs_invalid_confusion.png`.
11. `raw_subtype_recall_d0_e0_f0_g0.png` - four-way subtype recall.
12. `distance_marking_metrics.png` - metrics by distance bucket.
13. `rgb_by_outcome_boxplots.png` - RGB p25/median/p75 by outcome.
14. `top_sequences_road_to_marking.png` - worst sequences.
15. Component plots (from `g0_loss_component_analysis.py`):
    `train_loss_components.png`, `val_loss_components.png`,
    `val_ce_vs_marking_iou.png`, `val_lovasz_vs_marking_iou.png`,
    `val_total_vs_marking_iou.png`, `f0_vs_g0_val_ce.png`.

## Key Numeric Anchor

Sampled G0 best epoch:

- all marking IoU: `{all_row['marking_iou']:.6f}`
- all precision: `{all_row['marking_precision']:.6f}`
- all recall: `{all_row['marking_recall']:.6f}`
- all predicted/true marking ratio: `{all_row['predicted_true_marking_ratio']:.3f}`
- RGB-valid predicted/true ratio: `{valid['predicted_true_marking_ratio']:.3f}` (F0 was 1.550)
- RGB-invalid predicted/true ratio: `{invalid['predicted_true_marking_ratio']:.3f}` (F0 was 1.148)
"""
    (PLOTS_DIR / "README.md").write_text(readme)


def write_conclusions(data: dict[str, pd.DataFrame]) -> None:
    rows = summary_rows(data["summary"])
    d0, e0, f0, g0 = rows["D0"], rows["E0"], rows["F0"], rows["G0"]
    f0_final, g0_final = rows["F0_final"], rows["G0_final"]
    sampled = json.loads(require_file(SAMPLED_DIR / "summary.json").read_text())
    rgb = data["rgb_metrics"]
    all_row = rgb[rgb["stratum"] == "all"].iloc[0]
    valid = rgb[rgb["stratum"] == "rgb_valid"].iloc[0]
    invalid = rgb[rgb["stratum"] == "rgb_invalid"].iloc[0]
    ratio = data["pred_true_ratio"]

    def pt(run: str, label: str) -> float:
        return float(
            ratio[(ratio["run"] == run) & (ratio["label"] == label) & (ratio["class"] == "marking")]["pred_true_ratio"].iloc[0]
        )

    align_path = ANALYSIS_DIR / "loss_alignment.json"
    align = json.loads(align_path.read_text()) if align_path.exists() else {}

    iou_win = "YES" if float(g0["marking_iou"]) > float(f0["marking_iou"]) else "NO"
    g0_iou_drift = float(g0_final["marking_iou"] - g0["marking_iou"])
    f0_iou_drift = float(f0_final["marking_iou"] - f0["marking_iou"])
    drift_win = "YES" if abs(g0_iou_drift) < abs(f0_iou_drift) else "NO"
    calib_win = "YES" if pt("G0", "best") <= pt("F0", "best") else "NO"

    lines = [
        "# G0 Conclusions",
        "",
        "## Main Conclusion",
        "",
        "G0 = F0 with the loss changed to `weighted_CE + lambda*Lovasz-Softmax`.",
        f"The official G0 checkpoint is epoch `{int(g0['epoch'])}`, selected by maximum",
        "raw marking IoU (same rule as D0/E0/F0).",
        "",
        "| metric | D0 ep18 | E0 ep14 | F0 ep13 | G0 ep" + f"{int(g0['epoch'])} |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| marking IoU | {d0['marking_iou']:.6f} | {e0['marking_iou']:.6f} | {f0['marking_iou']:.6f} | {g0['marking_iou']:.6f} |",
        f"| F1 | {d0['marking_f1']:.6f} | {e0['marking_f1']:.6f} | {f0['marking_f1']:.6f} | {g0['marking_f1']:.6f} |",
        f"| precision | {d0['marking_precision']:.6f} | {e0['marking_precision']:.6f} | {f0['marking_precision']:.6f} | {g0['marking_precision']:.6f} |",
        f"| recall | {d0['marking_recall']:.6f} | {e0['marking_recall']:.6f} | {f0['marking_recall']:.6f} | {g0['marking_recall']:.6f} |",
        f"| mIoU | {d0['miou']:.6f} | {e0['miou']:.6f} | {f0['miou']:.6f} | {g0['miou']:.6f} |",
        "",
        "## Success Criteria (vs F0)",
        "",
        "| criterion | F0 | G0 | G beats F0? |",
        "| --- | ---: | ---: | :---: |",
        f"| best marking IoU | {f0['marking_iou']:.6f} | {g0['marking_iou']:.6f} | {iou_win} |",
        f"| best-to-final IoU drift | {f0_iou_drift:+.6f} | {g0_iou_drift:+.6f} | {drift_win} |",
        f"| best pred/true ratio | {pt('F0','best'):.3f} | {pt('G0','best'):.3f} | {calib_win} |",
        "",
        "## Fair CE Comparison",
        "",
        "F0 `val_loss` is pure weighted CE; G0 `val_ce` is the CE component "
        "(comparable). G0 `val_loss` is the TOTAL and is not compared.",
        "",
        f"- F0 best val CE: `{f0['val_ce']:.6f}`",
        f"- G0 best val CE: `{g0['val_ce']:.6f}`",
        f"- G0 best val total: `{g0['val_loss']:.6f}`",
        f"- G0 best val Lovasz (raw): `{g0['val_lovasz']:.6f}`",
        "",
        "## Lovasz Term Health",
        "",
    ]
    if align:
        lines.extend(
            [
                f"- scale: **{align.get('scale_verdict')}** (Lovasz share of total ~"
                f"{align.get('best_val_lovasz_share', float('nan')):.3f})",
                f"- stability: **{align.get('stability_verdict')}** (val Lovasz CV "
                f"{align.get('val_lovasz_cv', float('nan')):.3f})",
                f"- alignment: **{align.get('alignment_verdict')}**",
                f"  - corr(val CE, IoU) = {align.get('corr_ce_iou', float('nan')):+.3f}; "
                f"corr(val Lovasz, IoU) = {align.get('corr_lovasz_iou', float('nan')):+.3f}",
                f"  - argmin val Lovasz epoch {align.get('argmin_val_lovasz_epoch')}, "
                f"argmin val CE epoch {align.get('argmin_val_ce_epoch')}, "
                f"argmax IoU epoch {align.get('argmax_iou_epoch')}",
            ]
        )
    else:
        lines.append("- (loss_alignment.json not found; run g0_loss_component_analysis.py)")

    lines.extend(
        [
            "",
            "## Best vs Final Epoch (drift)",
            "",
            "| run | metric | best | final | delta |",
            "| --- | --- | ---: | ---: | ---: |",
            f"| F0 | marking IoU | {f0['marking_iou']:.6f} | {f0_final['marking_iou']:.6f} | {f0_iou_drift:+.6f} |",
            f"| G0 | marking IoU | {g0['marking_iou']:.6f} | {g0_final['marking_iou']:.6f} | {g0_iou_drift:+.6f} |",
            f"| F0 | pred/true | {pt('F0','best'):.3f} | {pt('F0','final'):.3f} | {pt('F0','final') - pt('F0','best'):+.3f} |",
            f"| G0 | pred/true | {pt('G0','best'):.3f} | {pt('G0','final'):.3f} | {pt('G0','final') - pt('G0','best'):+.3f} |",
            "",
            "## Sampled Best-Epoch Residuals",
            "",
            "```text",
            f"all:         pred/true {all_row['predicted_true_marking_ratio']:.3f}, IoU {all_row['marking_iou']:.6f}",
            f"rgb_valid:   pred/true {valid['predicted_true_marking_ratio']:.3f}, IoU {valid['marking_iou']:.6f}  (F0 was 1.550)",
            f"rgb_invalid: pred/true {invalid['predicted_true_marking_ratio']:.3f}, IoU {invalid['marking_iou']:.6f}  (F0 was 1.148)",
            "```",
            "",
            "## Provenance",
            "",
            f"- sampled analysis checkpoint: `{sampled['checkpoint']}`",
            f"- sampled steps: `{sampled['steps']}`",
            f"- sampled seed: `{sampled['seed']}`",
            f"- sampled split: `{sampled['split']}`",
            "",
        ]
    )
    (ANALYSIS_DIR / "g0_conclusions.md").write_text("\n".join(lines))


def main() -> None:
    global SAMPLED_DIR
    SAMPLED_DIR = find_sampled_dir()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    best_epoch = int(data["g0_history"].loc[data["g0_history"]["lane_iou"].idxmax(), "epoch"])
    final_epoch = int(data["g0_history"]["epoch"].max())

    plot_best_metrics(data)
    plot_metrics_overview(data)
    plot_lr_schedule(data)
    plot_predicted_true_ratio_over_epochs(data)
    plot_confusion_matrix(best_epoch, f"confusion_best_marking_epoch_{best_epoch:03d}.png", f"G0 epoch {best_epoch} confusion matrix, row-normalized")
    plot_confusion_matrix(final_epoch, f"confusion_final_epoch_{final_epoch:03d}.png", f"G0 epoch {final_epoch} confusion matrix, row-normalized")
    plot_class_true_vs_predicted_share(data)
    plot_runtime_and_memory(data)
    plot_f0_vs_g0_curves(data)
    plot_best_to_final_drift(data)
    plot_rgb_valid_vs_invalid_metrics(data)
    plot_rgb_valid_vs_invalid_confusion(data)
    plot_raw_subtype_recall(data)
    plot_distance_marking_metrics(data)
    plot_rgb_by_outcome_boxplots(data)
    plot_top_sequences_road_to_marking(data)
    write_readme(data)
    write_conclusions(data)
    print(f"sampled_dir {SAMPLED_DIR}")
    print(f"wrote plots to {PLOTS_DIR}")
    print(f"wrote conclusions to {ANALYSIS_DIR / 'g0_conclusions.md'}")
    print("script_status PASS")


if __name__ == "__main__":
    main()

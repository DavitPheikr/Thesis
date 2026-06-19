#!/usr/bin/env python
"""Generate the candidate-run plot package and conclusions (labels from _suite_paths)."""

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


import _suite_style as S  # noqa: E402
from _suite_paths import (  # noqa: E402
    REPO as PROJECT_ROOT, RUN_NAME, RUN_DIR as G1_RUN_DIR, ANALYSIS_OUT as OUT_DIR,
    CANDIDATE_LABEL, BASELINE_LABEL, BASELINE_LABELS, BASELINE_DIR,
)
S.apply()
CAND = CANDIDATE_LABEL    # current run's label in the CSVs/plots (e.g. "G2")
BASE = BASELINE_LABEL     # primary baseline (headline delta in conclusions)
BASELINES = list(BASELINE_LABELS)         # all comparison baselines, in order
RUN_COLORS = S.run_colors(CAND, BASELINES)  # stable colour per run label
G0_RUN_DIR = BASELINE_DIR  # primary baseline run dir
PLOTS_DIR = OUT_DIR / "plots"

# Semantic colours come from the shared style module so every figure in the
# suite (and across G1/G2/H0) is mutually consistent. Local aliases below keep
# the call sites readable; CAND_COLOR/BASE_COLOR follow the candidate/baseline.
CLASS_NAMES = ("road", "marking", "other")
ROAD_COLOR = S.ROAD
MARKING_COLOR = S.MARKING
OTHER_COLOR = S.OTHER
G0_COLOR = S.BASELINE     # baseline run colour (purple)
G1_COLOR = S.CANDIDATE    # candidate run colour (teal)
GRID_COLOR = S.GRID
CORE_PLOTS = {
    "loss_curves.png",
    "train_loss_components.png",
    "validation_loss_components.png",
    "lr_schedule.png",
    "marking_metrics_over_epochs.png",
    "iou_per_class_and_miou_over_epochs.png",
    "comparison_marking_curves.png",
    "predicted_true_marking_ratio_over_epochs.png",
    "confusion_best_checkpoint.png",
    "confusion_final_epoch.png",
    "runtime_and_memory.png",
}
SAMPLED_PLOTS = {
    "distance_metrics_best_checkpoint.png",
    "rgb_valid_vs_invalid_best_checkpoint.png",
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.8)
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def remove_unrequested_pngs(directory: Path, allowed: set[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        if path.name not in allowed:
            path.unlink()


def style(ax: plt.Axes) -> None:
    S.style_axes(ax)


def load_inputs() -> dict[str, pd.DataFrame]:
    files = {
        "summary": OUT_DIR / "summary.csv",
        "best_vs_final": OUT_DIR / "best_vs_final.csv",
        "epoch": OUT_DIR / "comparison_epoch_metrics.csv",
        "pred_true": OUT_DIR / "pred_true_ratio_by_epoch.csv",
        "events": OUT_DIR / "lr_events.csv",
        "loss": OUT_DIR / "loss_component_summary.csv",
        "alignment": OUT_DIR / "loss_alignment.csv",
        "confusion": OUT_DIR / "confusion_breakdown.csv",
    }
    return {name: pd.read_csv(require_file(path)) for name, path in files.items()}


def g1_best_epoch(summary: pd.DataFrame) -> int:
    return int(summary[(summary["run"] == CAND) & (summary["label"] == "best")]["epoch"].iloc[0])


def g1_final_epoch(summary: pd.DataFrame) -> int:
    return int(summary[(summary["run"] == CAND) & (summary["label"] == "final")]["epoch"].iloc[0])


def sampled_dir(summary: pd.DataFrame) -> Path:
    return OUT_DIR / f"sampled_error_analysis_epoch{g1_best_epoch(summary)}"


def plot_loss_curves(data: dict[str, pd.DataFrame]) -> None:
    g1 = data["epoch"][data["epoch"]["run"] == CAND]
    best = g1_best_epoch(data["summary"])
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.plot(g1["epoch"], g1["train_loss"], marker="o", label="train total loss", color="#666666")
    ax.plot(g1["epoch"], g1["val_loss"], marker="o", label="validation total loss", color=G1_COLOR)
    ax.axvline(best, color="black", linestyle="--", linewidth=1.2, label=f"best epoch {best}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.set_title(f"{CAND} train and validation total loss")
    style(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "loss_curves.png")


def plot_loss_components(data: dict[str, pd.DataFrame], split: str, out_name: str) -> None:
    loss = data["loss"]
    best = g1_best_epoch(data["summary"])
    prefix = "train" if split == "train" else "val"
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.plot(loss["epoch"], loss[f"{prefix}_ce_loss"], marker="o", label="weighted CE", color=ROAD_COLOR)
    ax.plot(loss["epoch"], loss[f"{prefix}_lovasz_scaled"], marker="o", label="0.5 * Lovasz", color=MARKING_COLOR)
    ax.plot(loss["epoch"], loss[f"{prefix}_total_loss"], marker="o", label="total", color=G1_COLOR)
    ax.axvline(best, color="black", linestyle="--", linewidth=1.2, label=f"best epoch {best}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{CAND} {split} loss components")
    style(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / out_name)


def plot_lr_schedule(data: dict[str, pd.DataFrame]) -> None:
    g1 = data["epoch"][data["epoch"]["run"] == CAND]
    events = data["events"][data["events"]["run"] == CAND]
    best = g1_best_epoch(data["summary"])
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.step(g1["epoch"], g1["lr"], where="post", linewidth=2.4, color=G1_COLOR)
    ax.scatter(g1["epoch"], g1["lr"], color=G1_COLOR, s=28)
    ax.axvline(best, color="black", linestyle="--", linewidth=1.2, label=f"best epoch {best}")
    for _, row in events.iterrows():
        ax.annotate(
            f"{row['lr_before']:.4g}->{row['lr_after']:.4g}",
            (row["epoch"], row["lr_after"]),
            xytext=(8, 14),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title(f"{CAND} ReduceLROnPlateau schedule")
    ax.set_ylim(0, max(g1["lr"]) * 1.18)
    style(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "lr_schedule.png")


def plot_marking_metrics(data: dict[str, pd.DataFrame]) -> None:
    g1 = data["epoch"][data["epoch"]["run"] == CAND]
    best = g1_best_epoch(data["summary"])
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    for col, label, color in [
        ("lane_iou", "IoU", S.METRIC["iou"]),
        ("lane_precision", "precision", S.METRIC["precision"]),
        ("lane_recall", "recall", S.METRIC["recall"]),
        ("lane_f1", "F1", S.METRIC["f1"]),
    ]:
        ax.plot(g1["epoch"], g1[col], marker="o", linewidth=2.1, label=label, color=color)
    ax.axvline(best, color="black", linestyle="--", linewidth=1.2, label=f"best epoch {best}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score (0–1)")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{CAND} marking metrics over epochs")
    style(ax)
    ax.legend(ncol=2)
    savefig(fig, PLOTS_DIR / "marking_metrics_over_epochs.png")


def plot_iou_per_class(data: dict[str, pd.DataFrame]) -> None:
    g1 = data["epoch"][data["epoch"]["run"] == CAND]
    best = g1_best_epoch(data["summary"])
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    for col, label, color in [
        ("road_iou", "road IoU", ROAD_COLOR),
        ("lane_iou", "marking IoU", MARKING_COLOR),
        ("other_iou", "other IoU", OTHER_COLOR),
        ("miou", "mIoU", "black"),
    ]:
        ax.plot(g1["epoch"], g1[col], marker="o", linewidth=2.1, label=label, color=color)
    ax.axvline(best, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{CAND} per-class IoU and mIoU over epochs")
    style(ax)
    ax.legend(ncol=2)
    savefig(fig, PLOTS_DIR / "iou_per_class_and_miou_over_epochs.png")


def _runs_in_order() -> list[str]:
    """Baselines first (drawn thin), candidate last (drawn thick, on top)."""
    return [*BASELINES, CAND]


def plot_comparison(data: dict[str, pd.DataFrame]) -> None:
    epoch = data["epoch"]
    panels = [
        ("lane_iou", "marking IoU"),
        ("lane_f1", "marking F1"),
        ("lane_precision", "marking precision"),
        ("lane_recall", "marking recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))
    for ax, (col, label) in zip(axes.ravel(), panels):
        for run in _runs_in_order():
            sub = epoch[epoch["run"] == run]
            if sub.empty:
                continue
            is_cand = run == CAND
            ax.plot(sub["epoch"], sub[col], marker="o",
                    linewidth=2.4 if is_cand else 1.8,
                    zorder=4 if is_cand else 3,
                    label=run, color=RUN_COLORS[run])
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        style(ax)
        ax.legend()
    others = " vs ".join(BASELINES)
    fig.suptitle(f"{CAND} vs {others} — marking curves", y=0.995, fontsize=15, fontweight="bold")
    savefig(fig, PLOTS_DIR / "comparison_marking_curves.png")


def plot_pred_true(data: dict[str, pd.DataFrame]) -> None:
    df = data["pred_true"]
    best = g1_best_epoch(data["summary"])
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    for run in _runs_in_order():
        sub = df[(df["run"] == run) & (df["class"] == "marking")].sort_values("epoch")
        if sub.empty:
            continue
        is_cand = run == CAND
        ax.plot(sub["epoch"], sub["pred_true_ratio"], marker="o",
                linewidth=2.4 if is_cand else 1.8,
                zorder=4 if is_cand else 3,
                label=run, color=RUN_COLORS[run])
    ax.axhline(1.0, color=S.CALIBRATED, linestyle=":", linewidth=1.3, label="calibrated (1.0)")
    ax.axvline(best, color=S.REFERENCE, linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"{CAND} best epoch {best}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("predicted marking / true marking")
    ax.set_title("Predicted / true marking ratio over epochs")
    style(ax)
    ax.legend()
    savefig(fig, PLOTS_DIR / "predicted_true_marking_ratio_over_epochs.png")


def confusion_matrix(epoch: int) -> np.ndarray:
    cm = np.load(require_file(G1_RUN_DIR / f"confusion_epoch_{epoch:03d}.npy"))
    if cm.shape != (3, 3):
        raise RuntimeError(f"{CAND} confusion epoch {epoch} shape {cm.shape}; expected (3, 3)")
    return cm.astype(np.int64, copy=False)


def plot_confusion(epoch: int, out_name: str, title: str) -> None:
    cm = confusion_matrix(epoch)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    image = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(3), CLASS_NAMES)
    ax.set_yticks(np.arange(3), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            color = "white" if pct[i, j] >= 50 else "black"
            ax.text(j, i, f"{pct[i, j]:.1f}%\n{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row share (%)")
    savefig(fig, PLOTS_DIR / out_name)


def load_training_log() -> pd.DataFrame:
    rows = []
    path = require_file(G1_RUN_DIR / "training_log.txt")
    for line in path.read_text().splitlines():
        fields = {}
        for part in line.split():
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k] = v
        if "epoch" in fields:
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


def plot_runtime(data: dict[str, pd.DataFrame]) -> None:
    g1 = data["epoch"][data["epoch"]["run"] == CAND]
    log = load_training_log()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.2), sharex=True)
    axes[0].plot(log["epoch"], log["wall_clock_seconds"] / 60.0, marker="o", color=G1_COLOR, linewidth=2.1)
    axes[0].set_ylabel("train+val minutes")
    axes[0].set_title(f"{CAND} runtime per epoch")
    style(axes[0])
    axes[1].plot(g1["epoch"], g1["val_wall_clock_seconds"], marker="o", color=ROAD_COLOR, linewidth=2.1, label="validation seconds")
    ax2 = axes[1].twinx()
    ax2.plot(g1["epoch"], g1["peak_gpu_memory_bytes_val"] / (1024.0 * 1024.0), marker="s", color=MARKING_COLOR, linewidth=2.0, label="peak GPU MiB")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation seconds")
    ax2.set_ylabel("peak GPU memory (MiB)")
    axes[1].set_title(f"{CAND} validation time and memory")
    style(axes[1])
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="upper right")
    savefig(fig, PLOTS_DIR / "runtime_and_memory.png")


def plot_sampled_distance(data: dict[str, pd.DataFrame]) -> None:
    sdir = sampled_dir(data["summary"])
    df = pd.read_csv(require_file(sdir / "distance_bucket_metrics.csv"))
    labels = df["bucket"].astype(str).tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.2), sharex=True)
    for col, label, color in [
        ("marking_iou", "IoU", S.METRIC["iou"]),
        ("marking_precision", "precision", S.METRIC["precision"]),
        ("marking_recall", "recall", S.METRIC["recall"]),
        ("marking_f1", "F1", S.METRIC["f1"]),
    ]:
        axes[0].plot(x, df[col], marker="o", linewidth=2.0, label=label, color=color)
    axes[0].set_ylabel("score (0–1)")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"{CAND} best-checkpoint marking metrics by distance")
    axes[0].legend(ncol=2)
    style(axes[0])
    axes[1].bar(x, df["predicted_true_marking_ratio"], color=S.PRED_TRUE, alpha=0.85)
    axes[1].axhline(1.0, color=S.CALIBRATED, linestyle=":", linewidth=1.3, label="calibrated (1.0)")
    axes[1].set_ylabel("predicted / true marking")
    axes[1].set_xlabel("distance bucket")
    axes[1].set_xticks(x, labels)
    axes[1].set_title(f"{CAND} best-checkpoint marking calibration by distance")
    style(axes[1])
    savefig(fig, sdir / "plots/distance_metrics_best_checkpoint.png")


def plot_sampled_rgb_valid(data: dict[str, pd.DataFrame]) -> None:
    sdir = sampled_dir(data["summary"])
    df = pd.read_csv(require_file(sdir / "rgb_valid_stratified_metrics.csv"))
    required = {"all", "rgb_valid", "rgb_invalid"}
    missing = sorted(required - set(df["stratum"].astype(str)))
    if missing:
        raise RuntimeError(f"RGB-valid sampled metrics missing strata: {missing}")
    order = ["all", "rgb_valid", "rgb_invalid"]
    sub = df.set_index("stratum").loc[order].reset_index()
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), sharex=True)
    width = 0.25
    axes[0].bar(x - width, sub["active_point_share"], width, label="active point share", color=S.NEUTRAL)
    axes[0].bar(x, sub["true_marking_share"], width, label="true marking share", color=S.MARKING, alpha=0.85)
    axes[0].bar(x + width, sub["predicted_marking_share"], width, label="predicted marking share", color=S.PRED_TRUE, alpha=0.85)
    axes[0].set_ylabel("share")
    axes[0].set_title(f"{CAND} sampled point contribution by RGB validity")
    axes[0].legend()
    style(axes[0])
    for col, label, color in [
        ("marking_iou", "IoU", S.METRIC["iou"]),
        ("marking_precision", "precision", S.METRIC["precision"]),
        ("marking_recall", "recall", S.METRIC["recall"]),
        ("predicted_true_marking_ratio", "pred/true", S.PRED_TRUE),
    ]:
        axes[1].plot(x, sub[col], marker="o", linewidth=2.0, label=label, color=color)
    axes[1].axhline(1.0, color=S.CALIBRATED, linestyle=":", linewidth=1.0, alpha=0.6)
    axes[1].set_xticks(x, order)
    axes[1].set_ylabel("score / ratio")
    axes[1].set_title(f"{CAND} best-checkpoint metrics by RGB validity")
    axes[1].legend(ncol=2)
    style(axes[1])
    savefig(fig, sdir / "plots/rgb_valid_vs_invalid_best_checkpoint.png")


def write_plots_readme(data: dict[str, pd.DataFrame]) -> None:
    best = g1_best_epoch(data["summary"])
    final = g1_final_epoch(data["summary"])
    text = f"""# {CAND} Plot Package

Generated by `03_plots.py` (reproducible testing suite).

This folder intentionally contains only the requested 11 core plots.

1. `loss_curves.png` - {CAND} train/validation total loss.
2. `train_loss_components.png` - training CE, scaled Lovasz, total.
3. `validation_loss_components.png` - validation CE, scaled Lovasz, total.
4. `lr_schedule.png` - LR changes; best epoch is {best}.
5. `marking_metrics_over_epochs.png` - marking IoU, precision, recall, F1.
6. `iou_per_class_and_miou_over_epochs.png` - road/marking/other IoU plus mIoU.
7. `comparison_marking_curves.png` - {CAND} vs {', '.join(BASELINES)} marking curves.
8. `predicted_true_marking_ratio_over_epochs.png` - marking over/underprediction.
9. `confusion_best_checkpoint.png` - row-normalized confusion at epoch {best}.
10. `confusion_final_epoch.png` - row-normalized confusion at epoch {final}.
11. `runtime_and_memory.png` - compute cost and memory.

Sampled best-checkpoint plots live under `sampled_error_analysis_epoch{best}/plots/`.
"""
    (PLOTS_DIR / "README.md").write_text(text)


def write_conclusions(data: dict[str, pd.DataFrame]) -> None:
    summary = data["summary"]
    pred = data["pred_true"]

    def best_of(run):
        return summary[(summary["run"] == run) & (summary["label"] == "best")].iloc[0]

    def ptrue(run, epoch):
        sub = pred[(pred["run"] == run) & (pred["epoch"] == int(epoch)) & (pred["class"] == "marking")]
        return float(sub["pred_true_ratio"].iloc[0]) if not sub.empty else float("nan")

    def row_md(label, r, pt):
        pt_s = f"{pt:.3f}" if pt == pt else ""  # NaN -> blank
        return (f"| {label} | {int(r['epoch'])} | {r['marking_iou']:.6f} | {r['marking_f1']:.6f} | "
                f"{r['marking_precision']:.6f} | {r['marking_recall']:.6f} | {r['miou']:.6f} | {pt_s} |")

    g1 = best_of(CAND)
    g1_final = summary[(summary["run"] == CAND) & (summary["label"] == "final")].iloc[0]
    g1_pt = ptrue(CAND, g1["epoch"])
    bases = {b: best_of(b) for b in BASELINES}

    sampled = json.loads(require_file(sampled_dir(summary) / "summary.json").read_text())
    rgb = pd.read_csv(require_file(sampled_dir(summary) / "rgb_valid_stratified_metrics.csv"))
    all_row = rgb[rgb["stratum"] == "all"].iloc[0]
    valid = rgb[rgb["stratum"] == "rgb_valid"].iloc[0]
    invalid = rgb[rgb["stratum"] == "rgb_invalid"].iloc[0]

    base_table = [row_md(f"{b} best", bases[b], ptrue(b, bases[b]["epoch"])) for b in BASELINES]
    base_interp = [
        f"- vs **{b}** best: marking IoU `{g1['marking_iou'] - bases[b]['marking_iou']:+.6f}`, "
        f"precision `{g1['marking_precision'] - bases[b]['marking_precision']:+.6f}`, "
        f"recall `{g1['marking_recall'] - bases[b]['marking_recall']:+.6f}`."
        for b in BASELINES
    ]
    lines = [
        f"# {CAND} Conclusions",
        "",
        f"This compares {CAND} (candidate) against baseline(s) {', '.join(BASELINES)} on the held-out test set.",
        "Single-seed (seed 42) controlled comparison; see "
        "`logs/milestone_g/notes/single_seed_limitation.md` before reading IoU deltas "
        "near the ~0.008 noise floor.",
        "",
        "## Headline",
        "",
        "| run | epoch | marking IoU | F1 | precision | recall | mIoU | pred/true |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *base_table,
        row_md(f"{CAND} best", g1, g1_pt),
        row_md(f"{CAND} final", g1_final, float("nan")),
        "",
        "## Interpretation",
        "",
        *base_interp,
        f"- {CAND} final-minus-best IoU drift is `{g1_final['marking_iou'] - g1['marking_iou']:+.6f}`.",
        "",
        "## Sampled Best-Checkpoint Check",
        "",
        f"- sampled checkpoint: `{sampled['checkpoint']}`",
        f"- sampled all marking IoU: `{all_row['marking_iou']:.6f}`",
        f"- sampled all pred/true: `{all_row['predicted_true_marking_ratio']:.3f}`",
        f"- RGB-valid active point share: `{valid['active_point_share']:.3f}`",
        f"- RGB-valid marking IoU / pred-true: `{valid['marking_iou']:.6f}` / `{valid['predicted_true_marking_ratio']:.3f}`",
        f"- RGB-invalid active point share: `{invalid['active_point_share']:.3f}`",
        f"- RGB-invalid marking IoU / pred-true: `{invalid['marking_iou']:.6f}` / `{invalid['predicted_true_marking_ratio']:.3f}`",
        "",
        "## Source Files",
        "",
        "- `summary.csv`",
        "- `best_vs_final.csv`",
        "- `comparison_epoch_metrics.csv`",
        "- `pred_true_ratio_by_epoch.csv`",
        "- `loss_component_summary.csv`",
        "- `sampled_error_analysis_epoch*/`",
        "",
    ]
    (OUT_DIR / f"{CAND.lower()}_conclusions.md").write_text("\n".join(lines))


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    best = g1_best_epoch(data["summary"])
    final = g1_final_epoch(data["summary"])
    if not sampled_dir(data["summary"]).exists():
        raise FileNotFoundError(f"Sampled analysis dir missing: {sampled_dir(data['summary'])}")
    remove_unrequested_pngs(PLOTS_DIR, CORE_PLOTS)
    remove_unrequested_pngs(sampled_dir(data["summary"]) / "plots", SAMPLED_PLOTS)

    plot_loss_curves(data)
    plot_loss_components(data, "train", "train_loss_components.png")
    plot_loss_components(data, "validation", "validation_loss_components.png")
    plot_lr_schedule(data)
    plot_marking_metrics(data)
    plot_iou_per_class(data)
    plot_comparison(data)
    plot_pred_true(data)
    plot_confusion(best, "confusion_best_checkpoint.png", f"{CAND} epoch {best} confusion matrix")
    plot_confusion(final, "confusion_final_epoch.png", f"{CAND} epoch {final} confusion matrix")
    plot_runtime(data)
    plot_sampled_distance(data)
    plot_sampled_rgb_valid(data)
    write_plots_readme(data)
    write_conclusions(data)
    print(f"wrote plots to {PLOTS_DIR}")
    print(f"wrote sampled plots to {sampled_dir(data['summary']) / 'plots'}")
    print("script_status PASS")


if __name__ == "__main__":
    main()

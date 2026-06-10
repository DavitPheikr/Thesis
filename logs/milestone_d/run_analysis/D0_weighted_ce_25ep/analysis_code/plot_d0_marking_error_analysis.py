"""Plot D0 sampled marking error diagnostics.

Input:
  logs/milestone_d/run_analysis/D0_weighted_ce_25ep/sampled_error_analysis_epoch18/

This script only plots already-computed sampled analysis CSV/NPY files. It does
not run model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

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
DEFAULT_ANALYSIS_DIR = (
    PROJECT_ROOT
    / f"logs/milestone_d/run_analysis/{RUN_NAME}/sampled_error_analysis_epoch18"
)
CLASS_NAMES = ("road", "marking", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float:
    if value in ("", "None", None):
        return float("nan")
    return float(value)


def as_int(value: str | None) -> int:
    if value in ("", "None", None):
        return 0
    return int(float(value))


def bucket_label(row: dict[str, str]) -> str:
    low = as_float(row["range_min_m"])
    high = as_float(row["range_max_m"])
    if np.isnan(high):
        return f"{low:.0f}+"
    return f"{low:.0f}-{high:.0f}"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{100.0 * value:.1f}%"


def plot_distance_marking_outcomes(rows: list[dict[str, str]], plots_dir: Path) -> None:
    labels = [bucket_label(row) for row in rows]
    true_marking = np.asarray([as_float(row["true_marking"]) for row in rows])
    tp = np.asarray([as_float(row["marking_tp"]) for row in rows])
    to_road = np.asarray([as_float(row["marking_to_road"]) for row in rows])
    to_other = np.asarray([as_float(row["marking_to_other"]) for row in rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        tp_rate = tp / true_marking * 100.0
        road_rate = to_road / true_marking * 100.0
        other_rate = to_other / true_marking * 100.0

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(x, tp_rate, color="#1b9e77", label="correct marking")
    ax.bar(x, road_rate, bottom=tp_rate, color="#d73027", label="marking predicted road")
    ax.bar(x, other_rate, bottom=tp_rate + road_rate, color="#7570b3", label="marking predicted other")
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of true marking points (%)")
    ax.set_xlabel("range bucket (m)")
    ax.set_xticks(x, labels)
    ax.set_title("D0 true-marking outcomes by distance bucket")
    ax.grid(axis="y", alpha=0.25)

    ax_count = ax.twinx()
    ax_count.plot(x, true_marking, color="#08519c", marker="o", linewidth=2, label="true marking count")
    ax_count.set_ylabel("true marking point count")
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = ax_count.get_legend_handles_labels()
    ax.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "distance_marking_outcomes.png", dpi=220)
    plt.close(fig)


def plot_distance_intensity(rows: list[dict[str, str]], plots_dir: Path) -> None:
    labels = [bucket_label(row) for row in rows]
    x = np.arange(len(labels))
    series = [
        ("marking TP", "marking_tp_intensity_median", "#1b9e77", "-"),
        ("marking -> road", "marking_to_road_intensity_median", "#d73027", "-"),
        ("marking -> other", "marking_to_other_intensity_median", "#7570b3", "-"),
        ("road TP", "road_tp_intensity_median", "#5f6368", "--"),
        ("road -> marking", "road_to_marking_intensity_median", "#fdae61", "--"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for label, key, color, linestyle in series:
        values = np.asarray([as_float(row.get(key)) for row in rows], dtype=np.float64)
        if np.isfinite(values).any():
            ax.plot(x, values, marker="o", linewidth=2.2, linestyle=linestyle, color=color, label=label)
    ax.set_xticks(x, labels)
    ax.set_xlabel("range bucket (m)")
    ax.set_ylabel("median recovered raw intensity")
    ax.set_title("D0 outcome intensity medians by distance")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "distance_marking_intensity_medians.png", dpi=220)
    plt.close(fig)


def plot_frame_scatter(
    rows: list[dict[str, str]],
    plots_dir: Path,
    x_key: str,
    y_key: str,
    count_key: str,
    out_name: str,
    title: str,
    y_label: str,
) -> None:
    filtered = [row for row in rows if as_float(row.get(x_key)) > 0]
    if not filtered:
        return
    x = np.asarray([as_float(row[x_key]) for row in filtered])
    y = np.asarray([as_float(row[y_key]) * 100.0 for row in filtered])
    counts = np.asarray([as_float(row[count_key]) for row in filtered])
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.scatter(x, y, s=18, color="#737373", alpha=0.45, label="sampled frame")
    top_idx = np.argsort(counts)[-10:][::-1]
    ax.scatter(
        x[top_idx],
        y[top_idx],
        s=72,
        color="#d73027",
        edgecolor="black",
        linewidth=0.6,
        label=f"top 10 by {count_key}",
    )
    for idx in top_idx[:5]:
        row = filtered[int(idx)]
        ax.annotate(
            f"{row['seq_id']}/{row['frame_idx']}",
            (x[idx], y[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / out_name, dpi=220)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, plots_dir: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(3), CLASS_NAMES)
    ax.set_yticks(np.arange(3), CLASS_NAMES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("D0 sampled confusion matrix, row-normalized")
    for i in range(3):
        for j in range(3):
            color = "white" if row_pct[i, j] >= 50 else "black"
            ax.text(j, i, f"{row_pct[i, j]:.1f}%\n{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row share (%)")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix_row_normalized.png", dpi=220)
    plt.close(fig)


def plot_raw_subtype_recall(rows: list[dict[str, str]], plots_dir: Path) -> None:
    labels = [f"{row['raw_id']} {row['raw_name']}" for row in rows]
    recall = np.asarray([as_float(row["recall"]) for row in rows]) * 100.0
    to_road = np.asarray([as_float(row["to_road_rate"]) for row in rows]) * 100.0
    to_other = np.asarray([as_float(row["to_other_rate"]) for row in rows]) * 100.0
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(x - width, recall, width=width, color="#1b9e77", label="recall")
    ax.bar(x, to_road, width=width, color="#d73027", label="to road")
    ax.bar(x + width, to_other, width=width, color="#7570b3", label="to other")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("share of raw subtype points (%)")
    ax.set_title("D0 recall and miss rates by raw marking subtype")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "raw_marking_subtype_recall.png", dpi=220)
    plt.close(fig)


def write_plot_index(
    analysis_dir: Path,
    distance_rows: list[dict[str, str]],
    frame_rows: list[dict[str, str]],
    raw_rows: list[dict[str, str]],
    cm: np.ndarray,
    summary: dict,
) -> None:
    marking_total = cm[1, :].sum()
    road_total = cm[0, :].sum()
    other_total = cm[2, :].sum()
    marking_recall = cm[1, 1] / marking_total if marking_total else float("nan")
    marking_to_road = cm[1, 0] / marking_total if marking_total else float("nan")
    road_to_marking = cm[0, 1] / road_total if road_total else float("nan")
    other_to_marking = cm[2, 1] / other_total if other_total else float("nan")
    worst_bucket = max(distance_rows, key=lambda row: as_float(row["marking_to_road_rate"]))
    support_bucket = max(distance_rows, key=lambda row: as_float(row["true_marking"]))
    top_miss = max(frame_rows, key=lambda row: as_float(row["marking_to_road"]))
    hardest_subtype = min(raw_rows, key=lambda row: math.inf if not np.isfinite(as_float(row["recall"])) else as_float(row["recall"]))

    lines = [
        "# D0 Sampled Diagnostic Plots",
        "",
        "These plots summarize the D0 epoch-18 sampled validation analysis. They are designed to explain marking misses and false-positive markings.",
        "",
        "## `distance_marking_outcomes.png`",
        "",
        "Purpose: shows what happens to true marking points in each distance bucket.",
        "",
        "How to read it: green is correctly predicted marking, red is marking predicted as road, purple is marking predicted as other. The blue line is true marking support.",
        "",
        f"Finding: largest marking support is `{support_bucket['bucket']}` with `{as_int(support_bucket['true_marking']):,}` true marking points. Highest marking-to-road rate is `{worst_bucket['bucket']}` at `{pct(as_float(worst_bucket['marking_to_road_rate']))}`.",
        "",
        "## `distance_marking_intensity_medians.png`",
        "",
        "Purpose: compares intensity medians for correct markings, missed markings, correct road, and road false positives by distance.",
        "",
        "How to read it: if red marking-to-road tracks gray road-TP more closely than green marking-TP, the miss is intensity-consistent with road at that range.",
        "",
        "## `frame_marking_to_road_scatter.png`",
        "",
        "Purpose: identifies whether missed markings are concentrated in a small set of frames.",
        "",
        f"Finding: top sampled frame by marking-to-road count is `{top_miss['seq_id']}/{top_miss['frame_idx']}` with `{as_int(top_miss['marking_to_road']):,}` marking points predicted as road.",
        "",
        "## `frame_road_to_marking_scatter.png` and `frame_other_to_marking_scatter.png`",
        "",
        "Purpose: identifies false-positive marking concentration from road and other.",
        "",
        "How to read it: high points indicate frames where D0 is over-predicting marking. These frames are good viewer targets.",
        "",
        "## `confusion_matrix_row_normalized.png`",
        "",
        "Purpose: shows the sampled confusion matrix as row percentages and counts.",
        "",
        f"Finding: sampled marking recall is `{pct(marking_recall)}`, marking-to-road is `{pct(marking_to_road)}`, road-to-marking is `{pct(road_to_marking)}`, and other-to-marking is `{pct(other_to_marking)}`.",
        "",
        "## `raw_marking_subtype_recall.png`",
        "",
        "Purpose: checks whether D0 learned all merged raw marking classes or mostly one subtype.",
        "",
        f"Finding: hardest subtype in this sampled pass is raw `{hardest_subtype['raw_id']}` `{hardest_subtype['raw_name']}` with recall `{pct(as_float(hardest_subtype['recall']))}`.",
        "",
        "## Provenance",
        "",
        f"- checkpoint: `{summary.get('checkpoint')}`",
        f"- steps: `{summary.get('steps')}`",
        f"- seed: `{summary.get('seed')}`",
        f"- device: `{summary.get('device')}`",
        "- note: this is a fresh sampled inference pass, not the exact training validation sample.",
    ]
    (analysis_dir / "diagnostic_plots.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    distance_rows = read_csv_rows(analysis_dir / "distance_bucket_summary.csv")
    frame_rows = read_csv_rows(analysis_dir / "frame_error_summary.csv")
    raw_rows = read_csv_rows(analysis_dir / "raw_marking_subtype_summary.csv")
    cm_path = analysis_dir / "confusion_matrix.npy"
    if not cm_path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {cm_path}")
    cm = np.load(cm_path)
    if cm.shape != (3, 3):
        raise RuntimeError(f"Expected 3x3 confusion matrix, got {cm.shape}")
    summary_path = analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    plot_distance_marking_outcomes(distance_rows, plots_dir)
    plot_distance_intensity(distance_rows, plots_dir)
    plot_frame_scatter(
        frame_rows,
        plots_dir,
        x_key="true_marking",
        y_key="marking_to_road_rate",
        count_key="marking_to_road",
        out_name="frame_marking_to_road_scatter.png",
        title="Frame-level marking-to-road misses",
        y_label="true marking predicted as road (%)",
    )
    plot_frame_scatter(
        frame_rows,
        plots_dir,
        x_key="active_points",
        y_key="road_to_marking_rate",
        count_key="road_to_marking",
        out_name="frame_road_to_marking_scatter.png",
        title="Frame-level road-to-marking false positives",
        y_label="true road predicted as marking (%)",
    )
    plot_frame_scatter(
        frame_rows,
        plots_dir,
        x_key="active_points",
        y_key="other_to_marking_rate",
        count_key="other_to_marking",
        out_name="frame_other_to_marking_scatter.png",
        title="Frame-level other-to-marking false positives",
        y_label="true other predicted as marking (%)",
    )
    plot_confusion(cm, plots_dir)
    plot_raw_subtype_recall(raw_rows, plots_dir)
    write_plot_index(analysis_dir, distance_rows, frame_rows, raw_rows, cm, summary)
    print(f"plots_dir {plots_dir}")
    print(f"diagnostic_index {analysis_dir / 'diagnostic_plots.md'}")
    print("script_status PASS")


if __name__ == "__main__":
    main()

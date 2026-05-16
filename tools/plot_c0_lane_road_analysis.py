"""Create focused plots for the C0 lane-road error analysis folder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = (
    PROJECT_ROOT / "logs/milestone_c/analysis/c0_epoch18_lane_road_errors_2160"
)
CLASS_NAMES = ("road", "lane", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float:
    return float(value) if value not in ("", "None", None) else float("nan")


def bucket_label(row: dict[str, str]) -> str:
    lo = as_float(row["range_min_m"])
    hi = as_float(row["range_max_m"])
    if np.isnan(hi):
        return f"{lo:.0f}+"
    return f"{lo:.0f}-{hi:.0f}"


def plot_distance_lane_outcomes(rows: list[dict[str, str]], out_dir: Path) -> None:
    labels = [bucket_label(row) for row in rows]
    true_lane = np.array([as_float(row["true_lane"]) for row in rows])
    lane_tp = np.array([as_float(row["lane_tp"]) for row in rows])
    lane_to_road = np.array([as_float(row["lane_to_road"]) for row in rows])
    lane_to_other = np.array([as_float(row["lane_to_other"]) for row in rows])

    with np.errstate(divide="ignore", invalid="ignore"):
        tp_rate = lane_tp / true_lane * 100.0
        road_rate = lane_to_road / true_lane * 100.0
        other_rate = lane_to_other / true_lane * 100.0

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x, tp_rate, color="#2ca25f", label="correct lane")
    ax.bar(x, road_rate, bottom=tp_rate, color="#de2d26", label="lane predicted road")
    ax.bar(
        x,
        other_rate,
        bottom=tp_rate + road_rate,
        color="#8c8c8c",
        label="lane predicted other",
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of true lane points (%)")
    ax.set_xlabel("range bucket (m)")
    ax.set_xticks(x, labels)
    ax.set_title("C0 true-lane outcomes by distance bucket")
    ax.grid(axis="y", alpha=0.25)

    ax_count = ax.twinx()
    ax_count.plot(x, true_lane, color="#08519c", marker="o", linewidth=2, label="true lane count")
    ax_count.set_ylabel("true lane point count")

    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = ax_count.get_legend_handles_labels()
    ax.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "distance_lane_outcomes.png", dpi=180)
    plt.close(fig)


def plot_distance_intensity(rows: list[dict[str, str]], out_dir: Path) -> None:
    labels = [bucket_label(row) for row in rows]
    lane_tp = np.array([as_float(row["lane_tp_intensity_median"]) for row in rows])
    lane_to_road = np.array([as_float(row["lane_to_road_intensity_median"]) for row in rows])
    road_tp = np.array([as_float(row.get("road_tp_intensity_median")) for row in rows])
    gap = lane_tp - lane_to_road
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(x, lane_tp, marker="o", linewidth=2.5, color="#2ca25f", label="correct lane")
    ax.plot(
        x,
        lane_to_road,
        marker="o",
        linewidth=2.5,
        color="#de2d26",
        label="lane predicted road",
    )
    if np.isfinite(road_tp).any():
        ax.plot(
            x,
            road_tp,
            marker="o",
            linewidth=2.3,
            linestyle="--",
            color="#525252",
            label="correct road",
        )
    ax.fill_between(x, lane_to_road, lane_tp, color="#fcbba1", alpha=0.35, label="median gap")
    for xi, value in zip(x, gap):
        if np.isfinite(value):
            ax.text(xi, max(lane_tp[xi], lane_to_road[xi]) + 2, f"+{value:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_xlabel("range bucket (m)")
    ax.set_ylabel("median recovered raw intensity")
    ax.set_title("Lane and road intensity medians by distance")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "distance_lane_intensity_medians.png", dpi=180)
    plt.close(fig)


def plot_frame_scatter(rows: list[dict[str, str]], out_dir: Path) -> None:
    filtered = [row for row in rows if as_float(row["true_lane"]) > 0]
    true_lane = np.array([as_float(row["true_lane"]) for row in filtered])
    rate = np.array([as_float(row["lane_to_road_rate"]) * 100.0 for row in filtered])
    lane_to_road = np.array([as_float(row["lane_to_road"]) for row in filtered])
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.scatter(true_lane, rate, s=18, color="#737373", alpha=0.45, label="sampled frame")

    top_idx = np.argsort(lane_to_road)[-10:][::-1]
    ax.scatter(
        true_lane[top_idx],
        rate[top_idx],
        s=72,
        color="#de2d26",
        edgecolor="black",
        linewidth=0.6,
        label="top 10 by lane-to-road count",
    )
    for idx in top_idx[:5]:
        row = filtered[int(idx)]
        label = f"{row['seq_id']}/{row['frame_idx']}"
        ax.annotate(label, (true_lane[idx], rate[idx]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("true lane points in sampled frame")
    ax.set_ylabel("lane predicted as road (%)")
    ax.set_title("Frame-level lane-to-road error concentration")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")

    ax_recall = ax.twinx()
    ax_recall.set_ylim(ax.get_ylim())
    ax_recall.set_ylabel("lane recall reference (%)")
    ax_recall.set_yticks([0, 25, 50, 75, 100])
    ax_recall.set_yticklabels(["100", "75", "50", "25", "0"])
    ax_recall.tick_params(axis="y", colors="#2ca25f")
    ax_recall.spines["right"].set_color("#2ca25f")

    fig.tight_layout()
    fig.savefig(out_dir / "frame_lane_to_road_scatter.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_dir: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("C0 sampled confusion matrix, row-normalized")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if row_pct[i, j] >= 50 else "black"
            ax.text(
                j,
                i,
                f"{row_pct[i, j]:.1f}%\n{cm[i, j]:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row share (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_row_normalized.png", dpi=180)
    plt.close(fig)


def percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def write_plot_index(
    out_dir: Path,
    distance_rows: list[dict[str, str]],
    frame_rows: list[dict[str, str]],
    cm: np.ndarray,
) -> None:
    lane_row = cm[1]
    road_row = cm[0]
    other_row = cm[2]
    lane_total = lane_row.sum()
    road_total = road_row.sum()
    other_total = other_row.sum()
    lane_recall = lane_row[1] / lane_total
    lane_to_road = lane_row[0] / lane_total
    road_recall = road_row[0] / road_total
    other_recall = other_row[2] / other_total

    strongest_bucket = max(distance_rows, key=lambda row: as_float(row["lane_to_road_rate"]))
    support_bucket = max(distance_rows, key=lambda row: as_float(row["true_lane"]))
    top_frame = max(frame_rows, key=lambda row: as_float(row["lane_to_road"]))

    lines = [
        "# Diagnostic Plots",
        "",
        "These plots summarize the C0 epoch-18 sampled inference analysis. They are meant to explain why true lane points are often predicted as road.",
        "",
        "## `distance_lane_outcomes.png`",
        "",
        "Purpose: shows what happens to true lane points at each distance range.",
        "",
        "How to read it: each bar is one distance bucket. The bar is split into true lane points predicted as lane, road, or other. Higher red means more lane points were missed as road. The blue line shows true-lane support, so buckets with more lane points carry more evidence.",
        "",
        f"Finding: the largest lane support is `{support_bucket['bucket']}` with {int(as_float(support_bucket['true_lane'])):,} true lane points and {percent(as_float(support_bucket['lane_to_road_rate']))} lane-to-road error. The highest lane-to-road rate is `{strongest_bucket['bucket']}` at {percent(as_float(strongest_bucket['lane_to_road_rate']))}.",
        "",
        "## `distance_lane_intensity_medians.png`",
        "",
        "Purpose: compares median intensity for correctly detected lane, lane predicted as road, and correctly detected road within each distance bucket.",
        "",
        "How to read it: green is true lane predicted as lane, red is true lane predicted as road, and gray dashed is true road predicted as road. If red tracks gray more closely than green, the missed lane points are road-like in intensity at that distance.",
        "",
        "Finding: this is the distance-controlled test of the main hypothesis. Use it to check whether missed lane intensity follows road intensity, not just whether missed lane is lower than detected lane.",
        "",
        "## `frame_lane_to_road_scatter.png`",
        "",
        "Purpose: identifies whether the lane-to-road problem is spread evenly or concentrated in specific bad frames.",
        "",
        "How to read it: each dot is one sampled frame. X-axis is true lane support in that sampled frame. Y-axis is the percent of those true lane points predicted as road. Red outlined dots are the top 10 frames by lane-to-road count.",
        "",
        f"Finding: the worst sampled frame is `{top_frame['seq_id']}/{top_frame['frame_idx']}` with {int(as_float(top_frame['true_lane'])):,} true lane points, {int(as_float(top_frame['lane_to_road'])):,} lane points predicted as road, and {percent(as_float(top_frame['lane_to_road_rate']))} lane-to-road error.",
        "",
        "## `confusion_matrix_row_normalized.png`",
        "",
        "Purpose: summarizes sampled model behavior across road, lane, and other.",
        "",
        "How to read it: rows are true classes and columns are predicted classes. The diagonal is correct prediction. Off-diagonal cells are errors. Each cell shows row percentage and raw count.",
        "",
        f"Finding: road recall is {percent(road_recall)}, other recall is {percent(other_recall)}, and lane recall is {percent(lane_recall)}. For true lane points, {percent(lane_to_road)} are predicted as road, so the main lane failure mode is lane-to-road confusion.",
        "",
        "## `intensity_hist_by_outcome.png`",
        "",
        "Purpose: shows full intensity distributions for prediction outcomes.",
        "",
        "How to read it: compare the lane-to-road curve with correct-lane and correct-road curves. If lane-to-road overlaps road, missed lane points are intensity-wise similar to road.",
        "",
        "Finding: the histogram gives the distribution-level version of the median-intensity result. Correct lane is generally brighter, while missed lane overlaps strongly with road.",
    ]
    (out_dir / "diagnostic_plots.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.analysis_dir
    if not out_dir.exists():
        raise SystemExit(f"analysis directory does not exist: {out_dir}")

    distance_rows = read_csv_rows(out_dir / "distance_bucket_summary.csv")
    frame_rows = read_csv_rows(out_dir / "frame_error_summary.csv")
    cm = np.load(out_dir / "confusion_matrix.npy")

    plot_distance_lane_outcomes(distance_rows, out_dir)
    plot_distance_intensity(distance_rows, out_dir)
    plot_frame_scatter(frame_rows, out_dir)
    plot_confusion_matrix(cm, out_dir)
    write_plot_index(out_dir, distance_rows, frame_rows, cm)

    print(f"wrote diagnostic plots to {out_dir}")


if __name__ == "__main__":
    main()

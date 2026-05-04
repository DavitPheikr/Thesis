"""
Stage-oriented raw LiDAR intensity analysis for lane visibility and separability.

Outputs:
  - CSV summaries
  - saved plots
  - stage markdown reports

Emits script_status PASS/FAIL as final line.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_pipeline.analysis.raw_intensity_lane_visibility import (
    DEFAULT_FORWARD_DISTANCE_BUCKETS,
    DEFAULT_LOCAL_CONTRAST_RADII,
    RAW_IDS_OF_INTEREST,
    RAW_LANE_ID,
    RAW_OTHER_ROAD_MARKING_ID,
    RAW_ROAD_ID,
    RAW_STOP_LINE_ID,
    ROAD_SURFACE_RAW_IDS,
    FrameSample,
    IntensityAccumulator,
    auc_from_histograms,
    cohens_d,
    compute_local_contrast,
    forward_distance_bucket_label,
    iter_frame_samples,
    label_name,
    lane_vs_group_auc,
    make_threshold_benchmark,
    raw_id_name,
    read_lane_bearing_sequence_ids,
    sensor_reapply_required,
)


DEFAULT_OUTPUT_DIR = Path("logs/raw_intensity_analysis")
MIN_LANE_POINTS_FOR_FRAME_METRICS = 10
PLOT_SAMPLE_CAP = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sequence-ids", nargs="*", default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument(
        "--skip-kde",
        action="store_true",
        help="Disable KDE plots if runtime is a concern.",
    )
    parser.add_argument(
        "--skip-local-contrast",
        action="store_true",
        help="Skip the expensive nearest-neighbor local contrast stage.",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "base": base,
        "tables": base / "tables",
        "plots": base / "plots",
        "reports": base / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    for stage in ("stage1", "stage2", "stage3", "stage4", "stage5"):
        (dirs["plots"] / stage).mkdir(parents=True, exist_ok=True)
    return dirs


def read_dataset_root() -> str:
    return Path("logs/dataset_root.txt").read_text().strip()


def plot_histogram(samples_by_name: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in samples_by_name.items():
        if values.size == 0:
            continue
        clipped = values[:PLOT_SAMPLE_CAP]
        ax.hist(clipped, bins=np.arange(257) - 0.5, density=True, alpha=0.35, label=name)
    ax.set_xlabel("raw intensity")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_kde(samples_by_name: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    grid = np.linspace(0, 255, 256)
    for name, values in samples_by_name.items():
        values = values[: min(PLOT_SAMPLE_CAP, values.size)]
        if values.size < 2:
            continue
        kde = gaussian_kde(values)
        ax.plot(grid, kde(grid), label=name)
    ax.set_xlabel("raw intensity")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df[x], df[y], alpha=0.65, s=18)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_distribution(df: pd.DataFrame, col: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    clean = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if not clean.empty:
        ax.hist(clean, bins=30, alpha=0.8)
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_report(path: Path, title: str, sections: list[str]) -> None:
    body = [f"# {title}", ""]
    body.extend(sections)
    path.write_text("\n".join(body).rstrip() + "\n")


def table_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def summarize_stage1(
    global_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> list[str]:
    lane = raw_df[raw_df["label"] == "lane_raw_8"].iloc[0]
    road = raw_df[raw_df["label"] == "road_raw_7"].iloc[0]
    stop = raw_df[raw_df["label"] == "stop_line_raw_9"].iloc[0]
    orm = raw_df[raw_df["label"] == "other_road_marking_raw_10"].iloc[0]
    sections = [
        "## Key observations",
        "",
        f"- Lane median raw intensity: `{lane['median']:.3f}`",
        f"- Road median raw intensity: `{road['median']:.3f}`",
        f"- Stop-line median raw intensity: `{stop['median']:.3f}`",
        f"- Other-road-marking median raw intensity: `{orm['median']:.3f}`",
        "",
        "## Interpretation prompts",
        "",
        f"- Lane minus road median shift: `{lane['median'] - road['median']:.3f}`",
        f"- Lane minus stop-line median shift: `{lane['median'] - stop['median']:.3f}`",
        f"- Lane minus other-road-marking median shift: `{lane['median'] - orm['median']:.3f}`",
        "- If lane and road overlap strongly here, intensity alone is likely weak globally.",
        "- If lane is separated from road but close to other markings, the harder confusion is paint-vs-paint, not paint-vs-asphalt.",
    ]
    return sections


def frame_metrics(frame: FrameSample) -> dict:
    raw = frame.raw_labels
    intensity = frame.raw_intensity
    xyz = frame.xyz_ego

    lane_mask = raw == RAW_LANE_ID
    road_mask = raw == RAW_ROAD_ID
    other_marking_mask = np.isin(raw, [RAW_STOP_LINE_ID, RAW_OTHER_ROAD_MARKING_ID])
    road_surface_non_lane_mask = np.isin(raw, ROAD_SURFACE_RAW_IDS) & (~lane_mask)

    lane = intensity[lane_mask]
    road = intensity[road_mask]
    road_surface_non_lane = intensity[road_surface_non_lane_mask]

    lane_count = int(lane.size)
    road_count = int(road.size)
    raw9_count = int((raw == RAW_STOP_LINE_ID).sum())
    raw10_count = int((raw == RAW_OTHER_ROAD_MARKING_ID).sum())

    lane_median = float(np.median(lane)) if lane.size else float("nan")
    road_median = float(np.median(road)) if road.size else float("nan")
    lane_mean = float(lane.mean()) if lane.size else float("nan")
    road_mean = float(road.mean()) if road.size else float("nan")
    road_p90 = float(np.percentile(road, 90)) if road.size else float("nan")
    lane_above_road_p90 = (
        float((lane > road_p90).mean()) if lane.size and road.size else float("nan")
    )
    lane_forward_median = (
        float(np.median(xyz[lane_mask, 0])) if lane_count else float("nan")
    )

    return {
        "seq_id": frame.seq_id,
        "frame_idx": frame.frame_idx,
        "lane_count": lane_count,
        "road_count": road_count,
        "raw9_count": raw9_count,
        "raw10_count": raw10_count,
        "lane_mean": lane_mean,
        "road_mean": road_mean,
        "lane_median": lane_median,
        "road_median": road_median,
        "lane_minus_road_mean": lane_mean - road_mean
        if lane.size and road.size
        else float("nan"),
        "lane_minus_road_median": lane_median - road_median
        if lane.size and road.size
        else float("nan"),
        "cohens_d_lane_vs_road": cohens_d(lane, road),
        "frac_lane_above_road_p90": lane_above_road_p90,
        "auc_lane_vs_road": lane_vs_group_auc(lane, road),
        "auc_lane_vs_road_plus_other_marking": lane_vs_group_auc(
            lane, road_surface_non_lane
        ),
        "lane_forward_median_x": lane_forward_median,
        "road_p90": road_p90,
        "insufficient_support": lane_count < MIN_LANE_POINTS_FOR_FRAME_METRICS
        or road_count < MIN_LANE_POINTS_FOR_FRAME_METRICS,
    }


def aggregate_sequence_metrics(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seq_id, group in frame_df.groupby("seq_id"):
        rows.append(
            {
                "seq_id": seq_id,
                "frame_count": int(group.shape[0]),
                "lane_count_total": int(group["lane_count"].sum()),
                "road_count_total": int(group["road_count"].sum()),
                "raw9_count_total": int(group["raw9_count"].sum()),
                "raw10_count_total": int(group["raw10_count"].sum()),
                "lane_mean_avg": float(group["lane_mean"].mean()),
                "road_mean_avg": float(group["road_mean"].mean()),
                "lane_median_avg": float(group["lane_median"].mean()),
                "road_median_avg": float(group["road_median"].mean()),
                "lane_minus_road_mean_avg": float(group["lane_minus_road_mean"].mean()),
                "lane_minus_road_median_avg": float(
                    group["lane_minus_road_median"].mean()
                ),
                "cohens_d_lane_vs_road_avg": float(
                    group["cohens_d_lane_vs_road"].mean()
                ),
                "frac_lane_above_road_p90_avg": float(
                    group["frac_lane_above_road_p90"].mean()
                ),
                "auc_lane_vs_road_avg": float(group["auc_lane_vs_road"].mean()),
                "auc_lane_vs_road_plus_other_marking_avg": float(
                    group["auc_lane_vs_road_plus_other_marking"].mean()
                ),
                "lane_forward_median_x_avg": float(group["lane_forward_median_x"].mean()),
                "metric_std_lane_minus_road_median": float(
                    group["lane_minus_road_median"].std()
                ),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["auc_lane_vs_road_plus_other_marking_avg", "lane_minus_road_median_avg"],
        ascending=[False, False],
    )
    return out


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root or read_dataset_root()
    output_dir = Path(args.output_dir)
    dirs = ensure_dirs(output_dir)

    sequence_ids = args.sequence_ids or read_lane_bearing_sequence_ids()
    if args.max_sequences is not None:
        sequence_ids = sequence_ids[: args.max_sequences]

    sensor_reapply = sensor_reapply_required()
    print(f"sequence_count {len(sequence_ids)}")
    print(f"sensor_reapply_per_frame {sensor_reapply}")

    samples = iter_frame_samples(
        dataset_root=dataset_root,
        sequence_ids=sequence_ids,
        sensor_reapply=sensor_reapply,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    if not samples:
        print("FAIL: no lane-bearing forward-facing semseg frames were loaded")
        print("script_status FAIL")
        sys.exit(1)

    print(f"frames_loaded {len(samples)}")

    raw_accumulators = {
        raw_id_name(raw_id): IntensityAccumulator(seed=raw_id)
        for raw_id in RAW_IDS_OF_INTEREST
    }
    remapped_accumulators = {
        label_name(label): IntensityAccumulator(seed=100 + label) for label in (0, 1, 2, 3)
    }
    global_accumulators = {
        "lane_vs_road_surface_non_lane_lane": IntensityAccumulator(seed=201),
        "lane_vs_road_surface_non_lane_non_lane": IntensityAccumulator(seed=202),
        "lane_vs_road_lane": IntensityAccumulator(seed=203),
        "lane_vs_road_road": IntensityAccumulator(seed=204),
        "lane_vs_other_markings_lane": IntensityAccumulator(seed=205),
        "lane_vs_other_markings_other_markings": IntensityAccumulator(seed=206),
    }
    distance_accumulators: dict[str, dict[str, IntensityAccumulator]] = {}
    sequence_accumulators: dict[str, dict[str, IntensityAccumulator]] = {}
    local_contrast_rows = []
    frame_rows = []

    for frame in samples:
        raw = frame.raw_labels
        remapped = frame.remapped_labels
        intensity = frame.raw_intensity
        x_forward = frame.xyz_ego[:, 0]

        for raw_id in RAW_IDS_OF_INTEREST:
            mask = raw == raw_id
            raw_accumulators[raw_id_name(raw_id)].add(intensity[mask])

        for label in (0, 1, 2, 3):
            mask = remapped == label
            remapped_accumulators[label_name(label)].add(intensity[mask])

        lane_mask = raw == RAW_LANE_ID
        road_mask = raw == RAW_ROAD_ID
        other_marking_mask = np.isin(raw, [RAW_STOP_LINE_ID, RAW_OTHER_ROAD_MARKING_ID])
        road_surface_non_lane_mask = np.isin(raw, ROAD_SURFACE_RAW_IDS) & (~lane_mask)

        if frame.seq_id not in sequence_accumulators:
            sequence_accumulators[frame.seq_id] = {
                "lane": IntensityAccumulator(seed=401),
                "road": IntensityAccumulator(seed=402),
                "road_surface_non_lane": IntensityAccumulator(seed=403),
                "stop_line": IntensityAccumulator(seed=404),
                "other_marking": IntensityAccumulator(seed=405),
            }
        sequence_accumulators[frame.seq_id]["lane"].add(intensity[lane_mask])
        sequence_accumulators[frame.seq_id]["road"].add(intensity[road_mask])
        sequence_accumulators[frame.seq_id]["road_surface_non_lane"].add(
            intensity[road_surface_non_lane_mask]
        )
        sequence_accumulators[frame.seq_id]["stop_line"].add(intensity[raw == RAW_STOP_LINE_ID])
        sequence_accumulators[frame.seq_id]["other_marking"].add(
            intensity[raw == RAW_OTHER_ROAD_MARKING_ID]
        )

        global_accumulators["lane_vs_road_surface_non_lane_lane"].add(intensity[lane_mask])
        global_accumulators["lane_vs_road_surface_non_lane_non_lane"].add(
            intensity[road_surface_non_lane_mask]
        )
        global_accumulators["lane_vs_road_lane"].add(intensity[lane_mask])
        global_accumulators["lane_vs_road_road"].add(intensity[road_mask])
        global_accumulators["lane_vs_other_markings_lane"].add(intensity[lane_mask])
        global_accumulators["lane_vs_other_markings_other_markings"].add(
            intensity[other_marking_mask]
        )

        for low, high in zip(
            DEFAULT_FORWARD_DISTANCE_BUCKETS[:-1], DEFAULT_FORWARD_DISTANCE_BUCKETS[1:]
        ):
            bucket_label = forward_distance_bucket_label(low, DEFAULT_FORWARD_DISTANCE_BUCKETS)
            bucket_mask = (x_forward >= low) & (x_forward < high)
            if bucket_label not in distance_accumulators:
                distance_accumulators[bucket_label] = {
                    "lane": IntensityAccumulator(seed=301),
                    "road": IntensityAccumulator(seed=302),
                    "road_surface_non_lane": IntensityAccumulator(seed=303),
                }
            distance_accumulators[bucket_label]["lane"].add(
                intensity[bucket_mask & lane_mask]
            )
            distance_accumulators[bucket_label]["road"].add(
                intensity[bucket_mask & road_mask]
            )
            distance_accumulators[bucket_label]["road_surface_non_lane"].add(
                intensity[bucket_mask & road_surface_non_lane_mask]
            )

        frame_rows.append(frame_metrics(frame))

        if not args.skip_local_contrast:
            for radius in DEFAULT_LOCAL_CONTRAST_RADII:
                contrast = compute_local_contrast(frame, radius=radius)
                delta_mean = contrast["delta_mean"]
                delta_median = contrast["delta_median"]
                zscore = contrast["zscore"]
                above_p75 = contrast["above_p75"]
                above_p90 = contrast["above_p90"]
                local_contrast_rows.append(
                    {
                        "seq_id": frame.seq_id,
                        "frame_idx": frame.frame_idx,
                        "radius_m": radius,
                        "lane_points_evaluated": int(delta_mean.size),
                        "delta_mean_avg": float(np.nanmean(delta_mean)) if delta_mean.size else float("nan"),
                        "delta_median_avg": float(np.nanmean(delta_median)) if delta_median.size else float("nan"),
                        "zscore_avg": float(np.nanmean(zscore)) if zscore.size else float("nan"),
                        "frac_positive_delta_mean": float((delta_mean > 0).mean()) if delta_mean.size else float("nan"),
                        "frac_above_local_p75": float(np.nanmean(above_p75)) if above_p75.size else float("nan"),
                        "frac_above_local_p90": float(np.nanmean(above_p90)) if above_p90.size else float("nan"),
                    }
                )

    raw_id_df = pd.DataFrame(
        [acc.summary_dict(name) for name, acc in raw_accumulators.items()]
    ).sort_values("label")
    remapped_df = pd.DataFrame(
        [acc.summary_dict(name) for name, acc in remapped_accumulators.items()]
    ).sort_values("label")
    global_df = pd.DataFrame(
        [acc.summary_dict(name) for name, acc in global_accumulators.items()]
    ).sort_values("label")
    frame_df = pd.DataFrame(frame_rows).sort_values(
        by=["auc_lane_vs_road_plus_other_marking", "lane_minus_road_median"],
        ascending=[False, False],
    )
    sequence_df = aggregate_sequence_metrics(frame_df)
    sequence_rows = []
    for seq_id, accs in sequence_accumulators.items():
        lane_summary = accs["lane"].summary_dict("lane")
        road_summary = accs["road"].summary_dict("road")
        stop_summary = accs["stop_line"].summary_dict("stop_line")
        other_summary = accs["other_marking"].summary_dict("other_road_marking")
        sequence_rows.append(
            {
                "seq_id": seq_id,
                "lane_point_count_exact": lane_summary["count"],
                "road_point_count_exact": road_summary["count"],
                "stop_line_point_count_exact": stop_summary["count"],
                "other_marking_point_count_exact": other_summary["count"],
                "lane_median_exact": lane_summary["median"],
                "road_median_exact": road_summary["median"],
                "lane_minus_road_median_exact": lane_summary["median"] - road_summary["median"]
                if lane_summary["count"] and road_summary["count"]
                else float("nan"),
                "auc_lane_vs_road_exact": auc_from_histograms(
                    accs["lane"].hist, accs["road"].hist
                ),
                "auc_lane_vs_road_surface_non_lane_exact": auc_from_histograms(
                    accs["lane"].hist, accs["road_surface_non_lane"].hist
                ),
            }
        )
    sequence_exact_df = pd.DataFrame(sequence_rows).sort_values(
        by=["auc_lane_vs_road_surface_non_lane_exact", "lane_minus_road_median_exact"],
        ascending=[False, False],
    )
    sequence_df = sequence_df.merge(sequence_exact_df, on="seq_id", how="left")
    local_contrast_columns = [
        "seq_id",
        "frame_idx",
        "radius_m",
        "lane_points_evaluated",
        "delta_mean_avg",
        "delta_median_avg",
        "zscore_avg",
        "frac_positive_delta_mean",
        "frac_above_local_p75",
        "frac_above_local_p90",
    ]
    local_contrast_df = pd.DataFrame(local_contrast_rows, columns=local_contrast_columns)
    if local_contrast_df.empty:
        local_contrast_sequence_df = pd.DataFrame(
            columns=[
                "seq_id",
                "radius_m",
                "lane_points_evaluated",
                "delta_mean_avg",
                "delta_median_avg",
                "zscore_avg",
                "frac_positive_delta_mean",
                "frac_above_local_p75",
                "frac_above_local_p90",
            ]
        )
    else:
        local_contrast_sequence_df = (
            local_contrast_df.groupby(["seq_id", "radius_m"])[
                [
                    "lane_points_evaluated",
                    "delta_mean_avg",
                    "delta_median_avg",
                    "zscore_avg",
                    "frac_positive_delta_mean",
                    "frac_above_local_p75",
                    "frac_above_local_p90",
                ]
            ]
            .mean()
            .reset_index()
            .sort_values(["radius_m", "frac_positive_delta_mean"], ascending=[True, False])
        )

    distance_rows = []
    for bucket_label, group in sorted(distance_accumulators.items()):
        lane_summary = group["lane"].summary_dict("lane")
        road_summary = group["road"].summary_dict("road")
        distance_rows.append(
            {
                "bucket": bucket_label,
                "lane_count": lane_summary["count"],
                "road_count": road_summary["count"],
                "lane_median": lane_summary["median"],
                "road_median": road_summary["median"],
                "lane_minus_road_median": lane_summary["median"] - road_summary["median"]
                if lane_summary["count"] and road_summary["count"]
                else float("nan"),
                "auc_lane_vs_road": auc_from_histograms(
                    group["lane"].hist, group["road"].hist
                ),
                "auc_lane_vs_road_surface_non_lane": auc_from_histograms(
                    group["lane"].hist, group["road_surface_non_lane"].hist
                ),
            }
        )
    distance_df = pd.DataFrame(distance_rows).sort_values("bucket")

    lane_values = raw_accumulators[raw_id_name(RAW_LANE_ID)].samples.values()
    road_values = raw_accumulators[raw_id_name(RAW_ROAD_ID)].samples.values()
    other_marking_values = np.concatenate(
        [
            raw_accumulators[raw_id_name(RAW_STOP_LINE_ID)].samples.values(),
            raw_accumulators[raw_id_name(RAW_OTHER_ROAD_MARKING_ID)].samples.values(),
        ]
    ).astype(np.float32, copy=False)
    road_surface_non_lane_values = np.concatenate([road_values, other_marking_values]).astype(
        np.float32, copy=False
    )
    benchmark_df = pd.DataFrame(
        [
            make_threshold_benchmark(lane_values, road_values, "lane_vs_road"),
            make_threshold_benchmark(
                lane_values,
                road_surface_non_lane_values,
                "lane_vs_road_plus_other_markings",
            ),
        ]
    )

    raw_id_df.to_csv(dirs["tables"] / "raw_id_summary.csv", index=False)
    remapped_df.to_csv(dirs["tables"] / "remapped_label_summary.csv", index=False)
    global_df.to_csv(dirs["tables"] / "global_intensity_summary.csv", index=False)
    frame_df.to_csv(dirs["tables"] / "frame_summary.csv", index=False)
    sequence_df.to_csv(dirs["tables"] / "sequence_summary.csv", index=False)
    local_contrast_df.to_csv(dirs["tables"] / "local_contrast_summary.csv", index=False)
    local_contrast_sequence_df.to_csv(
        dirs["tables"] / "local_contrast_sequence_summary.csv", index=False
    )
    distance_df.to_csv(dirs["tables"] / "distance_bucket_summary.csv", index=False)
    benchmark_df.to_csv(dirs["tables"] / "benchmark_summary.csv", index=False)

    plot_histogram(
        {row["label"]: raw_accumulators[row["label"]].samples.values() for _, row in raw_id_df.iterrows()},
        dirs["plots"] / "stage1" / "raw_id_histograms.png",
        "Raw intensity by PandaSet road-surface IDs",
    )
    if not args.skip_kde:
        plot_kde(
            {
                row["label"]: raw_accumulators[row["label"]].samples.values()
                for _, row in raw_id_df.iterrows()
            },
            dirs["plots"] / "stage1" / "raw_id_kde.png",
            "Raw intensity KDE by PandaSet road-surface IDs",
        )

    plot_metric_distribution(
        frame_df,
        "auc_lane_vs_road_plus_other_marking",
        dirs["plots"] / "stage2" / "frame_auc_distribution.png",
        "Frame-level intensity-only AUC distribution",
    )
    plot_scatter(
        frame_df,
        "lane_count",
        "auc_lane_vs_road_plus_other_marking",
        dirs["plots"] / "stage2" / "lane_count_vs_auc.png",
        "Frame lane count vs lane-vs-road+markings AUC",
    )
    plot_scatter(
        frame_df,
        "lane_forward_median_x",
        "lane_minus_road_median",
        dirs["plots"] / "stage2" / "distance_proxy_vs_median_shift.png",
        "Frame distance proxy vs lane-road median shift",
    )
    plot_scatter(
        frame_df,
        "road_p90",
        "lane_median",
        dirs["plots"] / "stage2" / "road_p90_vs_lane_median.png",
        "Road P90 vs lane median intensity",
    )
    if not local_contrast_df.empty:
        plot_metric_distribution(
            local_contrast_df[local_contrast_df["radius_m"] == 0.5],
            "delta_mean_avg",
            dirs["plots"] / "stage3" / "local_contrast_delta_mean_r0p5.png",
            "Local contrast mean delta at radius 0.5m",
        )
    if not distance_df.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(distance_df["bucket"], distance_df["auc_lane_vs_road"], marker="o", label="lane vs road")
        ax.plot(
            distance_df["bucket"],
            distance_df["auc_lane_vs_road_surface_non_lane"],
            marker="o",
            label="lane vs road-surface non-lane",
        )
        ax.set_ylabel("AUC")
        ax.set_title("Distance-conditioned intensity-only AUC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dirs["plots"] / "stage4" / "distance_auc.png")
        plt.close(fig)

    write_report(
        dirs["reports"] / "stage1_global_raw_id_report.md",
        "Stage 1 Global Raw Intensity Report",
        summarize_stage1(global_df, raw_id_df),
    )
    write_report(
        dirs["reports"] / "stage2_sequence_frame_report.md",
        "Stage 2 Sequence and Frame Report",
        [
            "## Highest-ranked sequences by intensity-only separability",
            "",
            table_to_markdown(sequence_df.head(10)),
            "",
            "## Highest-ranked frames by intensity-only separability",
            "",
            table_to_markdown(frame_df.head(20)),
        ],
    )
    write_report(
        dirs["reports"] / "stage3_local_contrast_report.md",
        "Stage 3 Local Contrast Report",
        [
            "## Radius sweep summary",
            "",
            "Local contrast was skipped for this run."
            if local_contrast_df.empty
            else table_to_markdown(
                local_contrast_df.groupby("radius_m")[
                    ["delta_mean_avg", "delta_median_avg", "frac_positive_delta_mean", "frac_above_local_p90"]
                ]
                .mean()
                .reset_index()
            ),
            "",
            "- Positive local contrast means lane points tend to be brighter than nearby non-lane road-surface points.",
            "- Compare this report against Stage 1 to see whether contextual signal is stronger than global absolute intensity.",
        ],
    )
    write_report(
        dirs["reports"] / "stage4_distance_report.md",
        "Stage 4 Distance-Conditioned Report",
        [
            "## Distance bucket summary",
            "",
            table_to_markdown(distance_df),
            "",
            "- Falling AUC with distance suggests the raw intensity signal weakens as lanes get farther from the sensor.",
        ],
    )
    write_report(
        dirs["reports"] / "stage5_benchmark_report.md",
        "Stage 5 Intensity-Only Benchmark Report",
        [
            "## Threshold benchmark",
            "",
            table_to_markdown(benchmark_df),
            "",
            "- This is a diagnostic baseline only. Strong scores indicate raw intensity contains signal; weak scores indicate intensity alone is not enough.",
        ],
    )
    write_report(
        dirs["reports"] / "analysis_overview.md",
        "Raw Intensity Analysis Overview",
        [
            "## Run summary",
            "",
            f"- Sequences analyzed: `{len(sequence_ids)}`",
            f"- Lane-bearing frames analyzed: `{len(samples)}`",
            "",
            "## Most intensity-friendly sequences (exact aggregated AUC)",
            "",
            table_to_markdown(
                sequence_df[
                    [
                        "seq_id",
                        "lane_point_count_exact",
                        "lane_minus_road_median_exact",
                        "auc_lane_vs_road_exact",
                        "auc_lane_vs_road_surface_non_lane_exact",
                    ]
                ].head(10)
            ),
            "",
            "## Stage artifacts",
            "",
            "- Stage 1 report: `reports/stage1_global_raw_id_report.md`",
            "- Stage 2 report: `reports/stage2_sequence_frame_report.md`",
            "- Stage 3 report: `reports/stage3_local_contrast_report.md`",
            "- Stage 4 report: `reports/stage4_distance_report.md`",
            "- Stage 5 report: `reports/stage5_benchmark_report.md`",
        ],
    )

    cumulative = {
        "sequence_count": len(sequence_ids),
        "frame_count": len(samples),
        "tables": sorted(str(p.relative_to(dirs["base"])) for p in dirs["tables"].glob("*.csv")),
        "reports": sorted(str(p.relative_to(dirs["base"])) for p in dirs["reports"].glob("*.md")),
    }
    (dirs["base"] / "run_manifest.json").write_text(json.dumps(cumulative, indent=2) + "\n")

    print(f"tables_dir {dirs['tables']}")
    print(f"plots_dir {dirs['plots']}")
    print(f"reports_dir {dirs['reports']}")
    print("analysis_complete True")
    print("script_status PASS")


if __name__ == "__main__":
    main()

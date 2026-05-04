from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pandaset import DataSet, geometry as pds_geometry
from scipy.spatial import cKDTree

from thesis_pipeline.adapters.pandaset_ff_lane3 import remap_raw_pandaset_ids
from thesis_pipeline.core.pandaset_compat import get_frame_count

RAW_ROAD_ID = 7
RAW_LANE_ID = 8
RAW_STOP_LINE_ID = 9
RAW_OTHER_ROAD_MARKING_ID = 10
ROAD_SURFACE_RAW_IDS = (
    RAW_ROAD_ID,
    RAW_STOP_LINE_ID,
    RAW_OTHER_ROAD_MARKING_ID,
)
RAW_IDS_OF_INTEREST = (
    RAW_ROAD_ID,
    RAW_LANE_ID,
    RAW_STOP_LINE_ID,
    RAW_OTHER_ROAD_MARKING_ID,
)

DEFAULT_FORWARD_DISTANCE_BUCKETS = (0.0, 10.0, 20.0, 30.0, 40.0, math.inf)
DEFAULT_LOCAL_CONTRAST_RADII = (0.25, 0.5, 0.75, 1.0)
GROUND_BAND_BELOW = 0.30
GROUND_BAND_ABOVE = 0.80


@dataclass(frozen=True)
class FrameSample:
    seq_id: str
    frame_idx: int
    xyz_ego: np.ndarray
    raw_intensity: np.ndarray
    raw_labels: np.ndarray
    remapped_labels: np.ndarray


class ReservoirSampler:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._values = np.empty(self.capacity, dtype=np.float32)
        self._seen = 0
        self._size = 0

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        for value in values:
            self._seen += 1
            if self._size < self.capacity:
                self._values[self._size] = value
                self._size += 1
            else:
                idx = self._rng.integers(0, self._seen)
                if idx < self.capacity:
                    self._values[idx] = value

    def values(self) -> np.ndarray:
        return self._values[: self._size].copy()


class IntensityAccumulator:
    def __init__(self, sample_capacity: int = 200_000, seed: int = 0):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.hist = np.zeros(256, dtype=np.int64)
        self.non_integer_count = 0
        self.out_of_range_count = 0
        self.samples = ReservoirSampler(sample_capacity, seed=seed)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return
        self.count += int(values.size)
        self.sum += float(values.sum(dtype=np.float64))
        self.sum_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        self.min = min(self.min, float(values.min()))
        self.max = max(self.max, float(values.max()))
        self.samples.add(values)

        rounded = np.rint(values)
        self.non_integer_count += int((np.abs(values - rounded) > 1e-6).sum())
        in_range_mask = (rounded >= 0) & (rounded <= 255)
        self.out_of_range_count += int((~in_range_mask).sum())
        if in_range_mask.any():
            bincount = np.bincount(
                rounded[in_range_mask].astype(np.int16), minlength=256
            ).astype(np.int64)
            self.hist += bincount

    def percentile(self, q: float) -> float:
        if self.count == 0:
            return float("nan")
        if self.hist.sum() <= 0:
            return float("nan")
        rank = q / 100.0 * (self.hist.sum() - 1)
        cumulative = np.cumsum(self.hist)
        idx = int(np.searchsorted(cumulative, rank + 1, side="left"))
        return float(idx)

    def summary_dict(self, label: str) -> dict:
        if self.count == 0:
            return {
                "label": label,
                "count": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "p5": float("nan"),
                "p25": float("nan"),
                "p50": float("nan"),
                "p75": float("nan"),
                "p95": float("nan"),
                "iqr": float("nan"),
                "non_integer_count": self.non_integer_count,
                "out_of_range_count": self.out_of_range_count,
            }

        mean = self.sum / self.count
        variance = max(self.sum_sq / self.count - mean * mean, 0.0)
        p25 = self.percentile(25.0)
        p75 = self.percentile(75.0)
        return {
            "label": label,
            "count": int(self.count),
            "mean": float(mean),
            "median": float(self.percentile(50.0)),
            "std": float(math.sqrt(variance)),
            "min": float(self.min),
            "max": float(self.max),
            "p5": float(self.percentile(5.0)),
            "p25": float(p25),
            "p50": float(self.percentile(50.0)),
            "p75": float(p75),
            "p95": float(self.percentile(95.0)),
            "iqr": float(p75 - p25),
            "non_integer_count": int(self.non_integer_count),
            "out_of_range_count": int(self.out_of_range_count),
        }


def read_lane_bearing_sequence_ids(
    manifest_path: Path = Path("logs/milestone_b_sequence_manifest.json"),
) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    return list(manifest["lane_bearing_ids"])


def sensor_reapply_required(
    pattern_path: Path = Path("logs/milestone_b_preflight_sensor_pattern.txt"),
) -> bool:
    if not pattern_path.exists():
        return False
    return "reapply_per_frame: true" in pattern_path.read_text()


def release_sequence_cache(seq) -> None:
    seq.lidar._data = None
    seq.lidar._poses = None
    seq.lidar._timestamps = None
    seq.semseg._data = None
    gc.collect()


def iter_frame_samples(
    dataset_root: str,
    sequence_ids: list[str],
    sensor_reapply: bool = False,
    max_frames_per_sequence: int | None = None,
) -> list[FrameSample]:
    ds = DataSet(dataset_root)
    samples: list[FrameSample] = []
    for seq_id in sequence_ids:
        seq = ds[seq_id]
        try:
            seq.load_lidar().load_semseg()
        except Exception:
            seq.load_lidar()
            seq.load_semseg()

        if seq.semseg is None:
            release_sequence_cache(seq)
            continue

        seq.lidar.set_sensor(1)
        frame_count = get_frame_count(seq)
        limit = (
            frame_count
            if max_frames_per_sequence is None
            else min(frame_count, max_frames_per_sequence)
        )
        for frame_idx in range(limit):
            if sensor_reapply:
                seq.lidar.set_sensor(1)
            pc_df = seq.lidar[frame_idx]
            semseg_df = seq.semseg[frame_idx]
            raw_labels = semseg_df.loc[pc_df.index, "class"].to_numpy(dtype=np.int32)
            if not np.any(raw_labels == RAW_LANE_ID):
                continue

            xyz_world = pc_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            pose = seq.lidar.poses[frame_idx]
            xyz_ego = pds_geometry.lidar_points_to_ego(xyz_world, pose).astype(
                np.float32, copy=False
            )
            raw_intensity = pc_df["i"].to_numpy(dtype=np.float32)
            remapped = remap_raw_pandaset_ids(raw_labels)
            samples.append(
                FrameSample(
                    seq_id=seq_id,
                    frame_idx=frame_idx,
                    xyz_ego=xyz_ego,
                    raw_intensity=raw_intensity,
                    raw_labels=raw_labels,
                    remapped_labels=remapped,
                )
            )
        release_sequence_cache(seq)
    return samples


def raw_id_name(raw_id: int) -> str:
    return {
        RAW_ROAD_ID: "road_raw_7",
        RAW_LANE_ID: "lane_raw_8",
        RAW_STOP_LINE_ID: "stop_line_raw_9",
        RAW_OTHER_ROAD_MARKING_ID: "other_road_marking_raw_10",
    }.get(raw_id, f"raw_{raw_id}")


def label_name(remapped_label: int) -> str:
    return {
        0: "ignore",
        1: "road",
        2: "lane",
        3: "other",
    }[remapped_label]


def frame_ground_band_mask(z_values: np.ndarray) -> np.ndarray:
    ground_z = float(np.percentile(z_values, 5.0))
    low = ground_z - GROUND_BAND_BELOW
    high = ground_z + GROUND_BAND_ABOVE
    return (z_values >= low) & (z_values <= high)


def lane_vs_group_auc(lane_values: np.ndarray, other_values: np.ndarray) -> float:
    lane_values = np.asarray(lane_values, dtype=np.float32)
    other_values = np.asarray(other_values, dtype=np.float32)
    if lane_values.size == 0 or other_values.size == 0:
        return float("nan")

    lane_hist = np.bincount(np.rint(lane_values).astype(np.int16), minlength=256)
    other_hist = np.bincount(np.rint(other_values).astype(np.int16), minlength=256)
    other_cum_below = np.cumsum(other_hist) - other_hist
    concordant = float((lane_hist * other_cum_below).sum())
    ties = float((lane_hist * other_hist).sum())
    total = float(lane_values.size * other_values.size)
    if total <= 0:
        return float("nan")
    return (concordant + 0.5 * ties) / total


def auc_from_histograms(lane_hist: np.ndarray, other_hist: np.ndarray) -> float:
    lane_hist = np.asarray(lane_hist, dtype=np.int64)
    other_hist = np.asarray(other_hist, dtype=np.int64)
    if lane_hist.sum() <= 0 or other_hist.sum() <= 0:
        return float("nan")
    other_cum_below = np.cumsum(other_hist) - other_hist
    concordant = float((lane_hist * other_cum_below).sum())
    ties = float((lane_hist * other_hist).sum())
    total = float(lane_hist.sum() * other_hist.sum())
    return (concordant + 0.5 * ties) / total if total > 0 else float("nan")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size < 2 or b.size < 2:
        return float("nan")
    mean_diff = float(a.mean() - b.mean())
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    pooled = ((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2)
    if pooled <= 0:
        return float("nan")
    return mean_diff / math.sqrt(pooled)


def intensity_summary(values: np.ndarray, label: str) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            "label": label,
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p5": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "iqr": float("nan"),
        }
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "label": label,
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(p50),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "iqr": float(p75 - p25),
    }


def make_threshold_benchmark(
    lane_values: np.ndarray,
    other_values: np.ndarray,
    task_name: str,
) -> dict:
    lane_values = np.asarray(lane_values, dtype=np.float32)
    other_values = np.asarray(other_values, dtype=np.float32)
    if lane_values.size == 0 or other_values.size == 0:
        return {
            "task": task_name,
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "best_threshold": float("nan"),
            "best_f1": float("nan"),
            "recall_at_precision_0p90": float("nan"),
            "recall_at_precision_0p95": float("nan"),
            "false_positive_rate_at_best_f1": float("nan"),
        }

    lane_hist = np.bincount(np.rint(lane_values).astype(np.int16), minlength=256)
    other_hist = np.bincount(np.rint(other_values).astype(np.int16), minlength=256)
    lane_total = lane_hist.sum()
    other_total = other_hist.sum()
    tp = np.cumsum(lane_hist[::-1])[::-1]
    fp = np.cumsum(other_hist[::-1])[::-1]
    fn = lane_total - tp
    tn = other_total - fp

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(lane_total, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    thresholds = np.arange(256, dtype=np.float32)
    best_idx = int(np.nanargmax(f1))

    recalls_for_precision = {}
    for min_precision in (0.90, 0.95):
        valid = np.where(precision >= min_precision)[0]
        recalls_for_precision[min_precision] = (
            float(recall[valid].max()) if valid.size else float("nan")
        )

    fpr = fp / np.maximum(other_total, 1)
    tpr = recall
    sort_idx = np.argsort(fpr)
    roc_auc = float(np.trapezoid(tpr[sort_idx], fpr[sort_idx]))

    order = np.argsort(recall)
    pr_auc = float(np.trapezoid(precision[order], recall[order]))

    return {
        "task": task_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1[best_idx]),
        "recall_at_precision_0p90": recalls_for_precision[0.90],
        "recall_at_precision_0p95": recalls_for_precision[0.95],
        "false_positive_rate_at_best_f1": float(fpr[best_idx]),
    }


def compute_local_contrast(
    frame: FrameSample,
    radius: float,
) -> dict[str, np.ndarray]:
    lane_mask = frame.raw_labels == RAW_LANE_ID
    ground_mask = np.isin(frame.raw_labels, ROAD_SURFACE_RAW_IDS)
    neighbor_mask = ground_mask & (~lane_mask)

    lane_xy = frame.xyz_ego[lane_mask, :2]
    lane_intensity = frame.raw_intensity[lane_mask]
    neighbor_xy = frame.xyz_ego[neighbor_mask, :2]
    neighbor_intensity = frame.raw_intensity[neighbor_mask]

    if lane_xy.shape[0] == 0 or neighbor_xy.shape[0] == 0:
        empty = np.array([], dtype=np.float32)
        return {
            "delta_mean": empty,
            "delta_median": empty,
            "zscore": empty,
            "above_p75": empty,
            "above_p90": empty,
            "neighbor_count": empty,
        }

    tree = cKDTree(neighbor_xy)
    neighborhoods = tree.query_ball_point(lane_xy, r=radius)

    delta_mean = []
    delta_median = []
    zscore = []
    above_p75 = []
    above_p90 = []
    neighbor_count = []
    for lane_value, idxs in zip(lane_intensity, neighborhoods):
        if not idxs:
            continue
        neighbors = neighbor_intensity[np.asarray(idxs, dtype=np.int32)]
        n_mean = float(neighbors.mean())
        n_median = float(np.median(neighbors))
        n_std = float(neighbors.std())
        p75 = float(np.percentile(neighbors, 75))
        p90 = float(np.percentile(neighbors, 90))
        delta_mean.append(float(lane_value - n_mean))
        delta_median.append(float(lane_value - n_median))
        if n_std > 0:
            zscore.append(float((lane_value - n_mean) / n_std))
        else:
            zscore.append(float("nan"))
        above_p75.append(float(lane_value > p75))
        above_p90.append(float(lane_value > p90))
        neighbor_count.append(float(neighbors.size))

    return {
        "delta_mean": np.asarray(delta_mean, dtype=np.float32),
        "delta_median": np.asarray(delta_median, dtype=np.float32),
        "zscore": np.asarray(zscore, dtype=np.float32),
        "above_p75": np.asarray(above_p75, dtype=np.float32),
        "above_p90": np.asarray(above_p90, dtype=np.float32),
        "neighbor_count": np.asarray(neighbor_count, dtype=np.float32),
    }


def forward_distance_bucket_label(value: float, bucket_edges: tuple[float, ...]) -> str:
    for low, high in zip(bucket_edges[:-1], bucket_edges[1:]):
        if low <= value < high:
            upper = "inf" if math.isinf(high) else f"{high:.0f}"
            return f"{low:.0f}_{upper}m"
    return "out_of_range"

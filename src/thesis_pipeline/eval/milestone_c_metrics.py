"""Milestone C semantic-segmentation metrics.

Open3D-ML RandLA-Net is configured with ``num_classes=3`` and
``ignored_label_inds=[0]``. During loss/metric computation, Open3D filters raw
ignore label 0 and compresses the active thesis labels like this:

- raw label 1, road -> model/metric index 0
- raw label 2, lane -> model/metric index 1
- raw label 3, other -> model/metric index 2

Accordingly, ``compute_metrics`` expects ``y_true`` and ``y_pred`` as active
contiguous indices ``{0, 1, 2}``, not raw dataset labels ``{1, 2, 3}``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


CLASS_NAMES = ("road", "lane", "other")
ACTIVE_CLASS_INDICES = (0, 1, 2)
DISTANCE_BUCKETS = (
    ("0to10m", 0.0, 10.0),
    ("10to20m", 10.0, 20.0),
    ("20to30m", 20.0, 30.0),
    ("30plus", 30.0, math.inf),
)
EVAL_CSV_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "miou",
    "road_iou",
    "lane_iou",
    "other_iou",
    "road_precision",
    "road_recall",
    "road_f1",
    "lane_precision",
    "lane_recall",
    "lane_f1",
    "other_precision",
    "other_recall",
    "other_f1",
    "lane_recall_0to10m",
    "lane_recall_10to20m",
    "lane_recall_20to30m",
    "lane_recall_30plus",
    "peak_gpu_memory_bytes_val",
    "val_wall_clock_seconds",
)


def _safe_divide(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def _none_if_nan(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _none_if_nan(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_none_if_nan(val) for val in value]
    return value


def json_ready(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert NaN float values to ``None`` for strict JSON snapshots."""

    return _none_if_nan(metrics)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ranges: np.ndarray,
) -> dict[str, Any]:
    """Compute active-class segmentation metrics.

    Args:
        y_true: Ground-truth active contiguous labels, shape ``[N]``.
        y_pred: Predicted active contiguous labels, shape ``[N]``.
        ranges: Ego-frame Euclidean point ranges in meters, shape ``[N]``.

    Returns:
        A dictionary containing mIoU, per-class IoU/precision/recall/F1,
        active-class confusion matrix, and lane recall by distance bucket.
    """

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)

    if not (y_true.shape == y_pred.shape == ranges.shape):
        raise ValueError(
            "y_true, y_pred, and ranges must have the same flattened shape: "
            f"{y_true.shape}, {y_pred.shape}, {ranges.shape}"
        )

    if y_true.size == 0:
        cm = np.zeros((len(ACTIVE_CLASS_INDICES), len(ACTIVE_CLASS_INDICES)), dtype=np.int64)
    else:
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(ACTIVE_CLASS_INDICES),
        ).astype(np.int64, copy=False)

    per_class: dict[str, dict[str, float]] = {}
    ious: list[float] = []

    for idx, class_name in enumerate(CLASS_NAMES):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        iou = _safe_divide(tp, tp + fp + fn)
        f1 = (
            float("nan")
            if math.isnan(precision) or math.isnan(recall) or precision + recall == 0
            else float(2 * precision * recall / (precision + recall))
        )

        per_class[class_name] = {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        ious.append(iou)

    lane_recall_by_distance: dict[str, float] = {}
    for bucket_name, low, high in DISTANCE_BUCKETS:
        if math.isinf(high):
            mask = ranges >= low
        else:
            mask = (ranges >= low) & (ranges < high)

        lane_mask = mask & (y_true == 1)
        lane_total = int(lane_mask.sum())
        if lane_total == 0:
            lane_recall_by_distance[bucket_name] = float("nan")
        else:
            lane_correct = int((lane_mask & (y_pred == 1)).sum())
            lane_recall_by_distance[bucket_name] = float(lane_correct / lane_total)

    return {
        "miou": _nanmean(ious),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "lane_recall_by_distance": lane_recall_by_distance,
        "support": {
            class_name: int(cm[idx, :].sum())
            for idx, class_name in enumerate(CLASS_NAMES)
        },
        "prediction_count": int(y_pred.size),
    }


def flatten_metrics_for_csv(
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, Any],
    peak_gpu_memory_bytes_val: int,
    val_wall_clock_seconds: float,
) -> dict[str, Any]:
    """Return one flat row matching Milestone C ``eval_history.csv`` columns."""

    per_class = metrics["per_class"]
    distance = metrics["lane_recall_by_distance"]
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "miou": metrics["miou"],
        "road_iou": per_class["road"]["iou"],
        "lane_iou": per_class["lane"]["iou"],
        "other_iou": per_class["other"]["iou"],
        "road_precision": per_class["road"]["precision"],
        "road_recall": per_class["road"]["recall"],
        "road_f1": per_class["road"]["f1"],
        "lane_precision": per_class["lane"]["precision"],
        "lane_recall": per_class["lane"]["recall"],
        "lane_f1": per_class["lane"]["f1"],
        "other_precision": per_class["other"]["precision"],
        "other_recall": per_class["other"]["recall"],
        "other_f1": per_class["other"]["f1"],
        "lane_recall_0to10m": distance["0to10m"],
        "lane_recall_10to20m": distance["10to20m"],
        "lane_recall_20to30m": distance["20to30m"],
        "lane_recall_30plus": distance["30plus"],
        "peak_gpu_memory_bytes_val": peak_gpu_memory_bytes_val,
        "val_wall_clock_seconds": val_wall_clock_seconds,
    }


def _assert_close(actual: float, expected: float, name: str) -> None:
    if math.isnan(expected):
        assert math.isnan(actual), f"{name}: expected NaN, got {actual}"
    else:
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (
            f"{name}: expected {expected}, got {actual}"
        )


def _assert_metric_set(metrics: dict[str, Any], expected_cm: np.ndarray) -> None:
    actual_cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    assert np.array_equal(actual_cm, expected_cm), (actual_cm, expected_cm)

    expected_ious = []
    for idx, class_name in enumerate(CLASS_NAMES):
        tp = float(expected_cm[idx, idx])
        fp = float(expected_cm[:, idx].sum() - tp)
        fn = float(expected_cm[idx, :].sum() - tp)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        iou = _safe_divide(tp, tp + fp + fn)
        f1 = (
            float("nan")
            if math.isnan(precision) or math.isnan(recall) or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        expected_ious.append(iou)
        _assert_close(metrics["per_class"][class_name]["precision"], precision, f"{class_name}_precision")
        _assert_close(metrics["per_class"][class_name]["recall"], recall, f"{class_name}_recall")
        _assert_close(metrics["per_class"][class_name]["iou"], iou, f"{class_name}_iou")
        _assert_close(metrics["per_class"][class_name]["f1"], f1, f"{class_name}_f1")

    _assert_close(metrics["miou"], _nanmean(expected_ious), "miou")


def _self_test() -> None:
    """Run deterministic metric checks without importing Open3D-ML."""

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    range_parts: list[np.ndarray] = []

    def add(true_label: int, pred_label: int, count: int, range_value: float) -> None:
        y_true_parts.append(np.full(count, true_label, dtype=np.int64))
        y_pred_parts.append(np.full(count, pred_label, dtype=np.int64))
        range_parts.append(np.full(count, range_value, dtype=np.float64))

    # Road: 700 total, 600 correct, 50 predicted lane, 50 predicted other.
    add(0, 0, 350, 5.0)
    add(0, 0, 240, 15.0)
    add(0, 0, 10, 25.0)
    add(0, 1, 50, 25.0)
    add(0, 2, 50, 25.0)

    # Lane: 200 total, 100 correct, 60 predicted road, 40 predicted other.
    add(1, 1, 25, 5.0)
    add(1, 0, 15, 5.0)
    add(1, 2, 10, 5.0)
    add(1, 1, 30, 15.0)
    add(1, 0, 20, 15.0)
    add(1, 2, 10, 15.0)
    add(1, 1, 20, 25.0)
    add(1, 0, 10, 25.0)
    add(1, 2, 10, 25.0)
    add(1, 1, 25, 35.0)
    add(1, 0, 15, 35.0)
    add(1, 2, 10, 35.0)

    # Other: 100 total, 80 correct, 10 predicted road, 10 predicted lane.
    add(2, 2, 50, 25.0)
    add(2, 2, 30, 35.0)
    add(2, 0, 10, 35.0)
    add(2, 1, 10, 35.0)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    ranges = np.concatenate(range_parts)
    assert y_true.size == 1000
    assert int(((ranges >= 0) & (ranges < 10)).sum()) == 400
    assert int(((ranges >= 10) & (ranges < 20)).sum()) == 300
    assert int(((ranges >= 20) & (ranges < 30)).sum()) == 200
    assert int((ranges >= 30).sum()) == 100

    expected_cm = np.asarray(
        [
            [600, 50, 50],
            [60, 100, 40],
            [10, 10, 80],
        ],
        dtype=np.int64,
    )
    metrics = compute_metrics(y_true, y_pred, ranges)
    _assert_metric_set(metrics, expected_cm)
    _assert_close(metrics["lane_recall_by_distance"]["0to10m"], 0.5, "lane_recall_0to10m")
    _assert_close(metrics["lane_recall_by_distance"]["10to20m"], 0.5, "lane_recall_10to20m")
    _assert_close(metrics["lane_recall_by_distance"]["20to30m"], 0.5, "lane_recall_20to30m")
    _assert_close(metrics["lane_recall_by_distance"]["30plus"], 0.5, "lane_recall_30plus")

    perfect = compute_metrics(y_true, y_true, ranges)
    for class_name in CLASS_NAMES:
        for metric_name in ("iou", "precision", "recall", "f1"):
            _assert_close(
                perfect["per_class"][class_name][metric_name],
                1.0,
                f"perfect_{class_name}_{metric_name}",
            )
    _assert_close(perfect["miou"], 1.0, "perfect_miou")

    no_lane_true = np.asarray([0, 0, 2, 2], dtype=np.int64)
    no_lane_pred = np.asarray([0, 0, 2, 2], dtype=np.int64)
    no_lane_ranges = np.asarray([5.0, 15.0, 25.0, 35.0], dtype=np.float64)
    no_lane = compute_metrics(no_lane_true, no_lane_pred, no_lane_ranges)
    for metric_name in ("iou", "precision", "recall", "f1"):
        _assert_close(no_lane["per_class"]["lane"][metric_name], float("nan"), f"no_lane_{metric_name}")

    empty = compute_metrics(
        np.asarray([], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        np.asarray([], dtype=np.float64),
    )
    _assert_close(empty["miou"], float("nan"), "empty_miou")
    for class_name in CLASS_NAMES:
        for metric_name in ("iou", "precision", "recall", "f1"):
            _assert_close(empty["per_class"][class_name][metric_name], float("nan"), f"empty_{class_name}_{metric_name}")

    print("milestone_c_metrics_self_test: PASS")

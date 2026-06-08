from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pandaset import DataSet, geometry

from thesis_pipeline.core.paths import (
    read_dataset_root,
    read_day2_chosen_frame,
    read_day2_chosen_sequence,
    read_day2_forward_sensor,
)

IGNORE_LABEL = 0
ROAD_LABEL = 1
LANE_LABEL = 2
OTHER_LABEL = 3

LABEL_MODE_LANE3 = "lane3"
LABEL_MODE_ROAD_MARKING3 = "road_marking3"
VALID_LABEL_MODES = {LABEL_MODE_LANE3, LABEL_MODE_ROAD_MARKING3}

RAW_IGNORE_IDS = {1, 2, 3, 4}
RAW_ROAD_ID = 7
RAW_LANE_ID = 8
RAW_STOP_LINE_ID = 9
RAW_OTHER_ROAD_MARKING_ID = 10
RAW_ROAD_MARKING_IDS = {
    RAW_LANE_ID,
    RAW_STOP_LINE_ID,
    RAW_OTHER_ROAD_MARKING_ID,
}

INTENSITY_COLUMN = "i"


@dataclass(frozen=True)
class SampleMetadata:
    sequence_id: str
    frame_idx: int | str
    forward_sensor: int
    intensity_column: str


def load_chosen_frame_components():
    root = read_dataset_root()
    seq_id = read_day2_chosen_sequence()
    frame_idx = read_day2_chosen_frame()
    forward_sensor = read_day2_forward_sensor()

    ds = DataSet(root)
    seq = ds[seq_id]

    try:
        seq.load_lidar().load_semseg()
    except Exception:
        seq.load_lidar()
        seq.load_semseg()

    seq.lidar.set_sensor(forward_sensor)
    pc_df = seq.lidar[frame_idx]
    semseg_df = seq.semseg[frame_idx]
    lidar_pose = seq.lidar.poses[frame_idx]
    return seq_id, frame_idx, forward_sensor, pc_df, semseg_df, lidar_pose


def scale_intensity_minmax(intensity: np.ndarray) -> np.ndarray:
    intensity = intensity.astype(np.float32, copy=False)
    value_min = float(intensity.min())
    value_max = float(intensity.max())
    scale = value_max - value_min
    if scale <= 0.0:
        return np.zeros((intensity.shape[0], 1), dtype=np.float32)

    # [PROVISIONAL: replace with training-split statistics before training]
    scaled = (intensity - value_min) / scale
    return scaled[:, None].astype(np.float32, copy=False)


def validate_label_mode(label_mode: str) -> str:
    if label_mode not in VALID_LABEL_MODES:
        valid = ", ".join(sorted(VALID_LABEL_MODES))
        raise ValueError(f"Unsupported label_mode {label_mode!r}; expected one of: {valid}")
    return label_mode


def remap_raw_pandaset_ids(
    raw_labels: np.ndarray,
    label_mode: str = LABEL_MODE_LANE3,
) -> np.ndarray:
    label_mode = validate_label_mode(label_mode)
    remapped = np.full(raw_labels.shape, OTHER_LABEL, dtype=np.int32)

    # Local fragility: this remap depends on the Day 2 raw-ID verification.
    remapped[np.isin(raw_labels, list(RAW_IGNORE_IDS))] = IGNORE_LABEL
    remapped[raw_labels == RAW_ROAD_ID] = ROAD_LABEL
    if label_mode == LABEL_MODE_LANE3:
        remapped[raw_labels == RAW_LANE_ID] = LANE_LABEL
    elif label_mode == LABEL_MODE_ROAD_MARKING3:
        remapped[np.isin(raw_labels, list(RAW_ROAD_MARKING_IDS))] = LANE_LABEL
    return remapped


def remap_raw_labels(
    raw_labels: np.ndarray,
    label_mode: str = LABEL_MODE_LANE3,
) -> np.ndarray:
    """Backward-compatible alias for the Milestone A adapter path."""
    return remap_raw_pandaset_ids(raw_labels, label_mode=label_mode)


def build_one_sample():
    seq_id, frame_idx, forward_sensor, pc_df, semseg_df, lidar_pose = (
        load_chosen_frame_components()
    )

    # Local fragility: semantic alignment must preserve the filtered LiDAR index exactly.
    raw_labels = semseg_df.loc[pc_df.index, "class"].to_numpy(np.int32)

    xyz_world = pc_df[["x", "y", "z"]].to_numpy(np.float32)

    # Local fragility: this helper depends on the local devkit exposing seq.lidar.poses.
    point = geometry.lidar_points_to_ego(xyz_world, lidar_pose).astype(
        np.float32, copy=False
    )

    # Local fragility: this assumes the locally verified intensity column remains `i`.
    feat = scale_intensity_minmax(pc_df[INTENSITY_COLUMN].to_numpy(np.float32))

    label = remap_raw_pandaset_ids(raw_labels)

    metadata = SampleMetadata(
        sequence_id=seq_id,
        frame_idx=frame_idx,
        forward_sensor=forward_sensor,
        intensity_column=INTENSITY_COLUMN,
    )
    return {"point": point, "feat": feat, "label": label, "meta": metadata}


def inspect_sample_main() -> None:
    sample = build_one_sample()
    point = sample["point"]
    feat = sample["feat"]
    label = sample["label"]
    meta = sample["meta"]

    print("chosen_seq", meta.sequence_id)
    print("chosen_frame", meta.frame_idx)
    print("forward_sensor", meta.forward_sensor)
    print("intensity_column", meta.intensity_column)
    print("point_shape", point.shape)
    print("feat_shape", feat.shape)
    print("label_shape", label.shape)
    print("label_unique", np.unique(label).tolist())
    print("lane_fraction", float((label == LANE_LABEL).mean()))

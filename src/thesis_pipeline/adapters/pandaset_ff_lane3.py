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

FEATURE_MODE_INTENSITY = "intensity"
FEATURE_MODE_INTENSITY_RGB_FRONT = "intensity_rgb_front"
VALID_FEATURE_MODES = {
    FEATURE_MODE_INTENSITY,
    FEATURE_MODE_INTENSITY_RGB_FRONT,
}

CAMERA_LOOKUP_NEAREST_TIMESTAMP = "nearest_timestamp"
VALID_CAMERA_LOOKUPS = {CAMERA_LOOKUP_NEAREST_TIMESTAMP}

COLOR_SAMPLING_BILINEAR = "bilinear"
COLOR_SAMPLING_NEAREST = "nearest"
VALID_COLOR_SAMPLINGS = {COLOR_SAMPLING_BILINEAR, COLOR_SAMPLING_NEAREST}

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


def validate_feature_mode(feature_mode: str) -> str:
    if feature_mode not in VALID_FEATURE_MODES:
        valid = ", ".join(sorted(VALID_FEATURE_MODES))
        raise ValueError(
            f"Unsupported feature_mode {feature_mode!r}; expected one of: {valid}"
        )
    return feature_mode


def validate_camera_lookup(camera_lookup: str) -> str:
    if camera_lookup not in VALID_CAMERA_LOOKUPS:
        valid = ", ".join(sorted(VALID_CAMERA_LOOKUPS))
        raise ValueError(
            f"Unsupported camera_lookup {camera_lookup!r}; expected one of: {valid}"
        )
    return camera_lookup


def validate_color_sampling(color_sampling: str) -> str:
    if color_sampling not in VALID_COLOR_SAMPLINGS:
        valid = ", ".join(sorted(VALID_COLOR_SAMPLINGS))
        raise ValueError(
            f"Unsupported color_sampling {color_sampling!r}; expected one of: {valid}"
        )
    return color_sampling


# --- camera/projection helpers for FEATURE_MODE_INTENSITY_RGB_FRONT ---


def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_to_mat(pose: dict) -> np.ndarray:
    heading = pose["heading"]
    position = pose["position"]
    rotation = _quat_to_rot(heading["w"], heading["x"], heading["y"], heading["z"])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [position["x"], position["y"], position["z"]]
    return transform


def project_points_to_camera(
    points_world: np.ndarray,
    camera_pose: dict,
    intrinsics: dict,
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-frame points to a pinhole camera.

    Returns (uv, depth, in_image_mask) all of length N. uv and depth are
    only meaningful where in_image_mask is True.
    """
    camera_mat = _pose_to_mat(camera_pose)
    transform = np.linalg.inv(camera_mat)
    pts_cam = transform[:3, :3] @ points_world.T + transform[:3, 3:4]
    depth = pts_cam[2, :]
    safe_z = np.where(depth > 1e-6, depth, 1.0)
    K = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    uvw = K @ pts_cam
    u = uvw[0, :] / safe_z
    v = uvw[1, :] / safe_z
    in_img = (depth > 0) & (u > 0) & (u < image_w) & (v > 0) & (v < image_h)
    uv = np.stack([u, v], axis=1)
    return uv, depth, in_img


def bilinear_sample_rgb(
    image_array: np.ndarray, uv: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Bilinear-sample image_array at floating uv coordinates.

    image_array: uint8 (H, W, 3)
    uv:          float (N, 2)
    valid_mask:  bool  (N,) where True indicates we should sample

    Returns float32 (N, 3) in [0, 1]. Rows where valid_mask is False are
    zero-filled.
    """
    n = uv.shape[0]
    out = np.zeros((n, 3), dtype=np.float32)
    if not valid_mask.any():
        return out

    h, w = image_array.shape[0], image_array.shape[1]
    idx = np.where(valid_mask)[0]
    u = uv[idx, 0]
    v = uv[idx, 1]
    # Clamp so the +1 neighbor never escapes the image. valid_mask already
    # enforces 0 < u < w and 0 < v < h, so u/v are strictly inside.
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.minimum(u0 + 1, w - 1)
    v1 = np.minimum(v0 + 1, h - 1)
    u0 = np.clip(u0, 0, w - 1)
    v0 = np.clip(v0, 0, h - 1)
    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)

    img_f = image_array.astype(np.float32) / 255.0
    c00 = img_f[v0, u0]
    c10 = img_f[v0, u1]
    c01 = img_f[v1, u0]
    c11 = img_f[v1, u1]

    one_minus_du = 1.0 - du
    one_minus_dv = 1.0 - dv
    sampled = (
        c00 * (one_minus_du * one_minus_dv)[:, None]
        + c10 * (du * one_minus_dv)[:, None]
        + c01 * (one_minus_du * dv)[:, None]
        + c11 * (du * dv)[:, None]
    )
    out[idx] = sampled.astype(np.float32, copy=False)
    return out


def nearest_sample_rgb(
    image_array: np.ndarray, uv: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    n = uv.shape[0]
    out = np.zeros((n, 3), dtype=np.float32)
    if not valid_mask.any():
        return out
    h, w = image_array.shape[0], image_array.shape[1]
    idx = np.where(valid_mask)[0]
    u = np.clip(np.round(uv[idx, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.round(uv[idx, 1]).astype(np.int64), 0, h - 1)
    out[idx] = (image_array[v, u].astype(np.float32) / 255.0).astype(
        np.float32, copy=False
    )
    return out


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

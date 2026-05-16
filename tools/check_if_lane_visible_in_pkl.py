"""
Tool for visually checking whether lane lines are present in PandaSet LiDAR
point clouds, both by geometric shape (top-down view of the road plane) and
by intensity (paint reflectivity).

Workflow:
  1. Press T  -> "lanes mode": top-down BEV + ground-plane filter +
                 rank-normalized intensity + high-intensity highlight overlay.
                 This is the configuration where lane paint, retroreflective
                 markers, and the painted geometry of the road are easiest
                 to spot.
  2. Cross-check against the camera image (left window) to confirm what you
     see in the LiDAR matches the painted lines on the road.
  3. Step through frames with N/P (or the on-image buttons) to see whether
     lanes are visible across the sequence, not just one cherry-picked frame.
  4. Press Y to reset filters back to the standard perspective view.

This file is a self-contained extension of  inspect_pkl.py.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from pandaset import DataSet

# ============================================================
# CHANGE THESE
# ============================================================

DATASET_ROOT = "pandaset/PandaSet"

SEQUENCE_ID = "015"
FRAME_ID = 1

CAMERA_NAME = "front_camera"

# PandaSet:
# d == 0 -> 360 mechanical LiDAR
# d == 1 -> front-facing LiDAR
FORWARD_SENSOR_ID = 1

LANE_LABEL_IDS = None
ONLY_LOAD_FRAMES_WITH_LANE_LINES = True
SHOW_ONLY_LANE_POINTS = False


# ============================================================
# VIEW SETTINGS
# ============================================================

POINT_SIZE = 3.0
AXIS_SIZE = 1.5

MOVE_STEP = 1.5
VERTICAL_STEP = 1.0

INITIAL_EYE = np.array([-2.5, 0.0, 1.8], dtype=np.float64)
INITIAL_TARGET = np.array([20.0, 0.0, 0.0], dtype=np.float64)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# BEV altitudes (meters above the local ground plane). Lower = more zoomed in.
BEV_ALTITUDE_DEFAULT = 55.0
BEV_ALTITUDE_LANES = 35.0

CAMERA_WINDOW_X = 50
CAMERA_WINDOW_Y = 80
CAMERA_WINDOW_W = 850
CAMERA_WINDOW_H = 520

POINTCLOUD_WINDOW_X = 930
POINTCLOUD_WINDOW_Y = 50
POINTCLOUD_WINDOW_W = 1400
POINTCLOUD_WINDOW_H = 900


# ============================================================
# LANE-CHECK SETTINGS
# ============================================================
#
# Ground-plane filter band, expressed relative to the per-frame estimated
# ground height (5th percentile of z). Defaults: keep ~30 cm below and 80 cm
# above the road, which holds onto curbs and short lane reflectors but cuts
# everything tall (cars, walls, trees, foliage).
GROUND_BAND_BELOW = 0.30
GROUND_BAND_ABOVE = 0.80

# Initial threshold for the high-intensity overlay (live-tunable with [ / ]).
HIGHLIGHT_PERCENTILE_INITIAL = 92.0
HIGHLIGHT_PERCENTILE_STEP = 2.0
HIGHLIGHT_PERCENTILE_MIN = 50.0
HIGHLIGHT_PERCENTILE_MAX = 99.0

HIGHLIGHT_COLOR = np.array([0.05, 1.0, 1.0])  # cyan
DARKEN_NON_HIGHLIGHT = 0.28

# Cyclable colormaps for the intensity color mode. inferno is the default
# because the point-cloud window background is near black: inferno's low end
# stays in dark warm tones (visible) while turbo's blue end vanishes.
INTENSITY_COLORMAPS = ["inferno", "magma", "turbo", "viridis", "gray"]


# ============================================================
# BASIC HELPERS
# ============================================================


def frame_name(frame_id=None) -> str:
    if frame_id is None:
        frame_id = FRAME_ID
    return f"{frame_id:02d}"


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n < 1e-8 else v / n


# ============================================================
# CAMERA EXTRINSIC HELPERS
# ============================================================


def look_at_extrinsic(eye, target, world_up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    z_cam = normalize(target - eye)
    x_cam = normalize(np.cross(z_cam, world_up))
    y_cam = np.cross(z_cam, x_cam)

    R = np.stack([x_cam, y_cam, z_cam], axis=0)
    t = -R @ eye

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic


def camera_basis(extrinsic):
    R = extrinsic[:3, :3]
    cam_pos = -R.T @ extrinsic[:3, 3]
    cam_right = R[0, :]
    cam_forward = R[2, :]
    return cam_pos, cam_right, cam_forward


def translate_extrinsic(extrinsic, delta_world):
    R = extrinsic[:3, :3]
    cam_pos = -R.T @ extrinsic[:3, 3]
    new_pos = cam_pos + delta_world

    new_extr = extrinsic.copy()
    new_extr[:3, 3] = -R @ new_pos
    return new_extr


def get_extrinsic(vis):
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    return np.asarray(params.extrinsic, dtype=np.float64).copy()


def apply_extrinsic(vis, extrinsic):
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()

    new_params = o3d.camera.PinholeCameraParameters()
    new_params.intrinsic = params.intrinsic
    new_params.extrinsic = np.ascontiguousarray(extrinsic, dtype=np.float64)

    try:
        ctr.convert_from_pinhole_camera_parameters(new_params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(new_params)


# ============================================================
# FILE HELPERS
# ============================================================


def get_seq_dir():
    return Path(DATASET_ROOT) / SEQUENCE_ID


def get_available_frame_ids():
    lidar_dir = get_seq_dir() / "lidar"

    files = list(lidar_dir.glob("*.pkl")) + list(lidar_dir.glob("*.pkl.gz"))

    frame_ids = []
    for p in files:
        stem = p.name.replace(".pkl.gz", "").replace(".pkl", "")
        try:
            frame_ids.append(int(stem))
        except ValueError:
            pass

    frame_ids = sorted(set(frame_ids))

    if not frame_ids:
        raise FileNotFoundError(f"No lidar frames found in {lidar_dir}")

    return frame_ids


def read_frame_pickle(seq_dir: Path, relative_dir: str, frame_id: int):
    name = frame_name(frame_id)

    candidates = [
        seq_dir / relative_dir / f"{name}.pkl",
        seq_dir / relative_dir / f"{name}.pkl.gz",
    ]

    for p in candidates:
        if p.exists():
            return pd.read_pickle(p), p

    raise FileNotFoundError(f"Could not find frame {name} in {seq_dir / relative_dir}")


# ============================================================
# COLOR / NORMALIZATION HELPERS
# ============================================================


def percentile_normalize_intensity(intensity, lo_pct=1.0, hi_pct=99.5, gamma=0.6):
    intensity = intensity.astype(np.float32)

    if intensity.size == 0:
        return intensity

    lo = np.percentile(intensity, lo_pct)
    hi = np.percentile(intensity, hi_pct)

    intensity = np.clip(intensity, lo, hi)
    intensity = (intensity - lo) / (hi - lo + 1e-8)

    return intensity**gamma


def rank_normalize_intensity(intensity):
    intensity = intensity.astype(np.float32)
    n = intensity.size

    if n == 0:
        return intensity
    if n == 1:
        return np.array([0.5], dtype=np.float32)

    ranks = intensity.argsort().argsort().astype(np.float32)
    return ranks / (n - 1)


def color_for_label(label):
    label = int(label)

    if label == 0:
        return np.array([0.12, 0.12, 0.12])

    rng = np.random.default_rng(1000 + label * 17)
    return rng.uniform(0.2, 1.0, size=3)


def make_semseg_colors(labels):
    labels = labels.astype(np.int32)
    colors = np.zeros((len(labels), 3), dtype=np.float64)

    for label in np.unique(labels):
        colors[labels == label] = color_for_label(label)

    return colors


def make_lane_overlay(intensity_colors, labels, lane_ids):
    """Darken the base intensity colors and paint labelled lane points yellow."""
    colors = intensity_colors.copy() * 0.18

    if not lane_ids:
        return colors

    lane_mask = np.isin(labels, lane_ids)
    colors[lane_mask] = np.array([1.0, 1.0, 0.0])
    return colors


# ============================================================
# GROUND-PLANE FILTER
# ============================================================


def estimate_ground_z(points):
    if len(points) == 0:
        return 0.0
    return float(np.percentile(points[:, 2], 5))


def ground_band_mask(points, ground_z):
    z = points[:, 2]
    return (z >= ground_z - GROUND_BAND_BELOW) & (z <= ground_z + GROUND_BAND_ABOVE)


# ============================================================
# SEMANTIC ANNOTATION HELPERS
# ============================================================


def get_semseg_label_column(semseg_df):
    preferred_cols = ["class", "label", "semseg", "id"]

    for col in preferred_cols:
        if col in semseg_df.columns:
            return col

    if len(semseg_df.columns) == 1:
        return semseg_df.columns[0]

    raise ValueError(
        f"Could not identify semantic label column. Columns: {list(semseg_df.columns)}"
    )


def get_semseg_class_names_from_devkit():
    try:
        dataset = DataSet(DATASET_ROOT)
        seq = dataset[SEQUENCE_ID]
        seq.load_semseg()

        classes = seq.semseg.classes

        if isinstance(classes, dict):
            return {int(k): str(v) for k, v in classes.items()}

        if isinstance(classes, (list, tuple)):
            return {i: str(v) for i, v in enumerate(classes)}

        if hasattr(classes, "iterrows"):
            class_names = {}
            cols = list(classes.columns)

            id_candidates = ["id", "class", "class_id", "index"]
            name_candidates = ["name", "class_name", "label", "label_name"]

            id_col = next((c for c in id_candidates if c in cols), None)
            name_col = next((c for c in name_candidates if c in cols), None)

            if id_col is not None and name_col is not None:
                for _, row in classes.iterrows():
                    class_names[int(row[id_col])] = str(row[name_col])
                return class_names

            first_col = cols[0]
            for idx, row in classes.iterrows():
                class_names[int(idx)] = str(row[first_col])

            return class_names

    except Exception as e:
        print("\nCould not load semantic class names from devkit:")
        print(e)

    return {}


def find_lane_label_ids(class_names):
    if LANE_LABEL_IDS is not None:
        return list(LANE_LABEL_IDS)

    return [cid for cid, name in class_names.items() if "lane" in name.lower()]


# ============================================================
# CAMERA IMAGE HELPERS
# ============================================================


def find_camera_image(frame_id):
    seq_dir = get_seq_dir()
    camera_dir = seq_dir / "camera"
    name = frame_name(frame_id)

    candidates = [
        camera_dir / CAMERA_NAME / f"{name}.jpg",
        camera_dir / CAMERA_NAME / f"{name}.png",
        camera_dir / CAMERA_NAME / f"{name}.jpeg",
    ]
    for p in candidates:
        if p.exists():
            return p

    all_candidates = (
        list(camera_dir.glob(f"**/{name}.jpg"))
        + list(camera_dir.glob(f"**/{name}.png"))
        + list(camera_dir.glob(f"**/{name}.jpeg"))
    )
    if all_candidates:
        return all_candidates[0]
    return None


def make_placeholder_camera_image(frame_id, message):
    img = np.full((CAMERA_WINDOW_H, CAMERA_WINDOW_W, 3), 25, dtype=np.uint8)
    cv2.putText(
        img,
        f"frame {frame_name(frame_id)}",
        (24, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (220, 220, 220),
        2,
    )
    cv2.putText(
        img,
        message,
        (24, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        2,
    )
    return img


def draw_axis_legend_on_image(img):
    img = img.copy()
    h, w = img.shape[:2]
    x0 = w - 180
    y0 = 35

    cv2.rectangle(img, (x0 - 15, y0 - 25), (w - 15, y0 + 95), (0, 0, 0), -1)
    cv2.putText(
        img, "Axis", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    cv2.line(img, (x0, y0 + 25), (x0 + 45, y0 + 25), (0, 0, 255), 4)
    cv2.putText(
        img, "X red", (x0 + 55, y0 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
    )

    cv2.line(img, (x0, y0 + 50), (x0 + 45, y0 + 50), (0, 255, 0), 4)
    cv2.putText(
        img,
        "Y green",
        (x0 + 55, y0 + 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    cv2.line(img, (x0, y0 + 75), (x0 + 45, y0 + 75), (255, 0, 0), 4)
    cv2.putText(
        img, "Z blue", (x0 + 55, y0 + 81), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
    )

    return img


def load_camera_base_image(frame_id):
    """
    Disk read + resize + axis legend. Done once per frame and cached;
    filter / colormap toggles only redraw the on-image overlay on top.
    """
    camera_file = find_camera_image(frame_id)
    if camera_file is None:
        msg = f"camera '{CAMERA_NAME}' image not found for this frame"
        print(msg)
        img = make_placeholder_camera_image(frame_id, msg)
        img = draw_axis_legend_on_image(img)
        return img, img.shape[1], img.shape[0], None

    img = cv2.imread(str(camera_file))
    if img is None:
        msg = f"could not decode {camera_file.name}"
        print(msg)
        img = make_placeholder_camera_image(frame_id, msg)
        img = draw_axis_legend_on_image(img)
        return img, img.shape[1], img.shape[0], camera_file

    h, w = img.shape[:2]
    scale = min(CAMERA_WINDOW_W / w, CAMERA_WINDOW_H / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img = cv2.resize(img, (new_w, new_h))
    img = draw_axis_legend_on_image(img)
    return img, new_w, new_h, camera_file


def draw_navigation_overlay(base_img, state):
    img = base_img.copy()
    h, w = img.shape[:2]

    button_h = 44
    margin = 14
    btn_w = max(140, min(180, (w - 4 * margin) // 4))
    y1 = h - button_h - margin
    y2 = h - margin

    prev_rect = (margin, y1, margin + btn_w, y2)
    next_rect = (w - margin - btn_w, y1, w - margin, y2)
    info_rect = (prev_rect[2] + 10, y1, next_rect[0] - 10, y2)

    def draw_button(rect, label):
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (40, 40, 40), -1)
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 2)
        cv2.putText(
            img,
            label,
            (rect[0] + 14, rect[1] + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    draw_button(prev_rect, "< Prev")
    draw_button(next_rect, "Next >")

    cv2.rectangle(
        img, (info_rect[0], info_rect[1]), (info_rect[2], info_rect[3]), (0, 0, 0), -1
    )

    line1 = (
        f"seq {SEQUENCE_ID}  f {state['frame_id']:02d}  "
        f"mode {state['color_mode']}  lanes {state['lane_count']}"
    )
    line2 = (
        f"G:{'on' if state['ground_filter_on'] else 'off'}  "
        f"H:{'on' if state['highlight_top_on'] else 'off'} "
        f"({state['highlight_percentile']:.0f}%)  "
        f"norm:{state['intensity_norm']}  cmap:{state['intensity_cmap']}"
    )

    cv2.putText(
        img,
        line1,
        (info_rect[0] + 8, info_rect[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        img,
        line2,
        (info_rect[0] + 8, info_rect[1] + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 255),
        1,
    )

    return img, {"prev": prev_rect, "next": next_rect}


# ============================================================
# FRAME LOADING
# ============================================================


def load_frame_data(frame_id, class_names, require_lane=False, verbose=True):
    """
    Load one frame's raw arrays. Color computation is deferred to render time
    so filter / colormap toggles can re-color without reloading from disk.
    Semseg colors (which are deterministic per label) are precomputed once.
    """
    seq_dir = get_seq_dir()

    try:
        lidar_df, lidar_path = read_frame_pickle(seq_dir, "lidar", frame_id)
    except FileNotFoundError:
        return None

    required = {"x", "y", "z", "i", "d"}
    missing = required - set(lidar_df.columns)
    if missing:
        raise ValueError(f"Missing required LiDAR columns: {missing}")

    try:
        semseg_df, semseg_path = read_frame_pickle(
            seq_dir, "annotations/semseg", frame_id
        )
        label_col = get_semseg_label_column(semseg_df)

        if len(semseg_df) != len(lidar_df):
            raise ValueError(
                f"Semseg length mismatch: lidar={len(lidar_df)}, semseg={len(semseg_df)}"
            )

        merged = lidar_df.join(semseg_df[[label_col]])
        has_semseg = True

    except Exception as e:
        if require_lane:
            return None
        if verbose:
            print("\nNo usable semantic segmentation for this frame:")
            print(e)
        merged = lidar_df.copy()
        label_col = None
        semseg_path = None
        has_semseg = False

    merged = merged[merged["d"] == FORWARD_SENSOR_ID].copy()

    points = merged[["x", "y", "z"]].to_numpy(dtype=np.float64)
    intensity = merged["i"].to_numpy(dtype=np.float32)

    labels = None
    semseg_colors_full = None
    lane_ids = find_lane_label_ids(class_names)
    lane_mask = np.zeros(len(points), dtype=bool)
    lane_count = 0

    if has_semseg and label_col is not None:
        labels = merged[label_col].astype(np.int32).to_numpy()
        if lane_ids:
            lane_mask = np.isin(labels, lane_ids)
        lane_count = int(lane_mask.sum())

        if require_lane and lane_count == 0:
            return None

        if SHOW_ONLY_LANE_POINTS:
            points = points[lane_mask]
            intensity = intensity[lane_mask]
            labels = labels[lane_mask]
            lane_mask = np.ones(len(points), dtype=bool)
            lane_count = len(points)

        semseg_colors_full = make_semseg_colors(labels)

    elif require_lane:
        return None

    ground_z = estimate_ground_z(points)

    if verbose:
        print("\nLoaded frame:")
        print("  frame:", frame_name(frame_id))
        print("  lidar:", lidar_path)
        print("  semseg:", semseg_path)
        print("  points after d filter:", len(points))
        print("  lane ids:", lane_ids)
        print("  lane count:", lane_count)
        print("  estimated ground z:", f"{ground_z:.2f}")
        if labels is not None:
            print("  unique labels:", np.unique(labels))

    return {
        "frame_id": frame_id,
        "points": points,
        "intensity": intensity,
        "labels": labels,
        "semseg_colors_full": semseg_colors_full,
        "has_semseg": has_semseg,
        "lane_ids": lane_ids,
        "lane_mask": lane_mask,
        "lane_count": lane_count,
        "ground_z": ground_z,
    }


def find_next_valid_frame(
    current_frame_id, direction, available_frame_ids, class_names
):
    if current_frame_id not in available_frame_ids:
        idx = min(
            range(len(available_frame_ids)),
            key=lambda i: abs(available_frame_ids[i] - current_frame_id),
        )
    else:
        idx = available_frame_ids.index(current_frame_id)

    i = idx + direction
    while 0 <= i < len(available_frame_ids):
        candidate = available_frame_ids[i]
        data = load_frame_data(
            candidate,
            class_names=class_names,
            require_lane=ONLY_LOAD_FRAMES_WITH_LANE_LINES,
            verbose=False,
        )
        if data is not None:
            return data
        i += direction

    return None


def find_initial_valid_frame(start_frame_id, available_frame_ids, class_names):
    data = load_frame_data(
        start_frame_id,
        class_names=class_names,
        require_lane=ONLY_LOAD_FRAMES_WITH_LANE_LINES,
        verbose=True,
    )
    if data is not None:
        return data

    print(
        f"\nInitial frame {start_frame_id:02d} does not have lane-line semseg. Searching forward..."
    )
    data = find_next_valid_frame(start_frame_id, 1, available_frame_ids, class_names)
    if data is not None:
        return data

    print("No forward frame found. Searching backward...")
    data = find_next_valid_frame(start_frame_id, -1, available_frame_ids, class_names)
    if data is not None:
        return data

    raise RuntimeError(
        "Could not find any frame with semantic lane-line labels. "
        "Try setting ONLY_LOAD_FRAMES_WITH_LANE_LINES=False or manually set LANE_LABEL_IDS."
    )


# ============================================================
# VIEW COMPUTATION (filters + colors)
# ============================================================


def _intensity_norm(intensity, mode):
    if mode == "rank":
        return rank_normalize_intensity(intensity)
    return percentile_normalize_intensity(intensity)


def compute_view(frame_data, state):
    """Apply ground filter + color mode to raw frame data → (points, colors)."""
    points = frame_data["points"]
    intensity = frame_data["intensity"]
    labels = frame_data["labels"]
    lane_mask = frame_data["lane_mask"]
    lane_ids = frame_data["lane_ids"]
    semseg_colors_full = frame_data["semseg_colors_full"]

    mask = np.ones(len(points), dtype=bool)

    if state["ground_filter_on"]:
        mask &= ground_band_mask(points, frame_data["ground_z"])

    if state["lane_only"] and labels is not None and len(lane_ids) > 0:
        mask &= lane_mask

    pts = points[mask]
    ints = intensity[mask]
    sub_labels = labels[mask] if labels is not None else None

    cmap = plt.get_cmap(state["intensity_cmap"])
    color_mode = state["color_mode"]

    if color_mode == "semseg" and semseg_colors_full is not None:
        colors = semseg_colors_full[mask].copy()
    elif color_mode == "lane" and sub_labels is not None:
        ints_norm = _intensity_norm(ints, state["intensity_norm"])
        intensity_colors = cmap(ints_norm)[:, :3]
        colors = make_lane_overlay(intensity_colors, sub_labels, lane_ids)
    else:
        ints_norm = _intensity_norm(ints, state["intensity_norm"])
        colors = cmap(ints_norm)[:, :3]

    if state["highlight_top_on"] and len(ints) > 0:
        threshold = np.percentile(ints, state["highlight_percentile"])
        bright = ints >= threshold
        colors = colors.copy()
        colors[~bright] *= DARKEN_NON_HIGHLIGHT
        colors[bright] = HIGHLIGHT_COLOR

    if len(pts) == 0:
        # Open3D dislikes empty geometry — push a single dummy point at origin.
        pts = np.zeros((1, 3), dtype=np.float64)
        colors = np.zeros((1, 3), dtype=np.float64)

    return pts, colors


# ============================================================
# CONTROLS PRINTOUT
# ============================================================


def print_controls():
    print("\nControls (click point-cloud window first to give it focus):")
    print("")
    print("  Lane-visibility presets:")
    print(
        "    T                 LANES MODE: BEV + ground filter + rank-norm + highlight"
    )
    print("    Y                 reset filters off (back to plain perspective view)")
    print("")
    print("  Filter toggles:")
    print("    G                 toggle ground-plane filter (z-band around road)")
    print("    H                 toggle high-intensity highlight overlay")
    print("    [ / ]             decrease / increase highlight percentile")
    print("    J                 cycle intensity colormap")
    print("    K                 toggle intensity normalization (rank / percentile)")
    print("")
    print("  Color modes:")
    print("    I                 intensity")
    print("    L                 semantic labels")
    print("    M                 lane-marking highlight (yellow on labelled paint)")
    print("")
    print("  Navigation:")
    print("    N / P             next / previous frame with lane-line semseg")
    print("    < Prev / Next >   click on the camera-image window")
    print("")
    print("  View:")
    print("    Mouse drag        rotate")
    print("    Mouse wheel       zoom")
    print("    W / S             move forward / backward (relative to view)")
    print("    A / D             strafe left / right")
    print("    E / C             move up / down (world Z)")
    print("    R                 reset view")
    print("    B                 toggle BEV / top-down")
    print("    + / =             larger points")
    print("    -                 smaller points")
    print("    Q or Esc          close")
    print("\nAxis colors: X=red, Y=green, Z=blue\n")


# ============================================================
# MAIN
# ============================================================


def main():
    print("\nSelected:")
    print("  DATASET_ROOT:", DATASET_ROOT)
    print("  SEQUENCE_ID:", SEQUENCE_ID)
    print("  START FRAME_ID:", FRAME_ID)
    print("  CAMERA_NAME:", CAMERA_NAME)
    print("  LiDAR sensor d:", FORWARD_SENSOR_ID)
    print("  ONLY_LOAD_FRAMES_WITH_LANE_LINES:", ONLY_LOAD_FRAMES_WITH_LANE_LINES)

    available_frame_ids = get_available_frame_ids()
    print("\nAvailable lidar frames:")
    print(available_frame_ids[:10], "...", available_frame_ids[-10:])

    class_names = get_semseg_class_names_from_devkit()
    lane_ids = find_lane_label_ids(class_names)

    print("\nSemantic lane label IDs:")
    print(lane_ids)

    if class_names:
        print("\nSemantic classes:")
        for class_id, class_name in sorted(class_names.items()):
            print(f"  {class_id}: {class_name}")

    frame_data = find_initial_valid_frame(
        start_frame_id=FRAME_ID,
        available_frame_ids=available_frame_ids,
        class_names=class_names,
    )

    state = {
        "frame_id": frame_data["frame_id"],
        "point_size": POINT_SIZE,
        "bev": False,
        "saved_extrinsic": None,
        # Color / filter state.
        "color_mode": "intensity",  # intensity | semseg | lane
        "intensity_norm": "rank",  # rank | percentile
        "intensity_cmap": INTENSITY_COLORMAPS[0],
        "ground_filter_on": False,
        "highlight_top_on": False,
        "highlight_percentile": HIGHLIGHT_PERCENTILE_INITIAL,
        "lane_only": SHOW_ONLY_LANE_POINTS,
        "lane_count": frame_data["lane_count"],
        # UI plumbing.
        "pending_action": None,
        "button_rects": {},
        "camera_base_img": None,  # disk-loaded resized image with axis legend
        "camera_img": None,  # base + button overlay (what gets displayed)
        "camera_size": None,
    }

    # Initial cloud.
    points0, colors0 = compute_view(frame_data, state)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points0)
    cloud.colors = o3d.utility.Vector3dVector(colors0)

    car_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=AXIS_SIZE, origin=[0, 0, 0]
    )

    initial_extrinsic = look_at_extrinsic(INITIAL_EYE, INITIAL_TARGET, WORLD_UP)

    # Camera image: load base from disk, render display image with overlay.
    base_img, cam_w, cam_h, _ = load_camera_base_image(state["frame_id"])
    state["camera_base_img"] = base_img
    state["camera_size"] = (cam_w, cam_h)
    rendered, button_rects = draw_navigation_overlay(base_img, state)
    state["camera_img"] = rendered
    state["button_rects"] = button_rects

    camera_window_name = f"Camera | {CAMERA_NAME} | seq {SEQUENCE_ID}"
    cv2.namedWindow(camera_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(camera_window_name, cam_w, cam_h)
    cv2.moveWindow(camera_window_name, CAMERA_WINDOW_X, CAMERA_WINDOW_Y)

    def on_camera_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for action, rect in state["button_rects"].items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["pending_action"] = action
                return

    cv2.setMouseCallback(camera_window_name, on_camera_mouse)

    # Open3D point-cloud window.
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"PandaSet front-LiDAR lane-check | seq {SEQUENCE_ID}",
        width=POINTCLOUD_WINDOW_W,
        height=POINTCLOUD_WINDOW_H,
        left=POINTCLOUD_WINDOW_X,
        top=POINTCLOUD_WINDOW_Y,
    )

    vis.add_geometry(cloud)
    vis.add_geometry(car_axes)

    render = vis.get_render_option()
    render.background_color = np.array([0.02, 0.02, 0.02])
    render.point_size = POINT_SIZE
    try:
        render.show_coordinate_frame = True
    except Exception:
        pass

    # ---------------- View / movement helpers ----------------

    def redraw_camera_overlay():
        """Cheap: just redraw the on-image overlay over the cached base image."""
        rendered, btn_rects = draw_navigation_overlay(state["camera_base_img"], state)
        state["camera_img"] = rendered
        state["button_rects"] = btn_rects

    def reload_camera_base():
        """Disk read; only call on frame change."""
        base, w, h, _ = load_camera_base_image(state["frame_id"])
        state["camera_base_img"] = base
        state["camera_size"] = (w, h)

    def push_geometry(vis, points, colors):
        """
        Update the point cloud in a way that's robust to large point-count
        changes (which is exactly what the ground filter triggers).
        """
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(cloud)

    def refresh_cloud(vis):
        pts, cols = compute_view(frame_data, state)
        push_geometry(vis, pts, cols)
        redraw_camera_overlay()

    def reset_view(vis):
        apply_extrinsic(vis, initial_extrinsic)
        state["bev"] = False
        print("Reset view: at car, looking forward.")
        return False

    def make_move_callback(forward_units, right_units, up_units, step):
        def cb(vis):
            extr = get_extrinsic(vis)
            cam_pos, cam_right, cam_forward = camera_basis(extr)

            fwd_g = cam_forward.copy()
            fwd_g[2] = 0.0
            n = np.linalg.norm(fwd_g)
            fwd_g = fwd_g / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

            right_g = cam_right.copy()
            right_g[2] = 0.0
            n = np.linalg.norm(right_g)
            right_g = right_g / n if n > 1e-6 else np.array([0.0, -1.0, 0.0])

            delta = (
                forward_units * fwd_g + right_units * right_g + up_units * WORLD_UP
            ) * step

            new_extr = translate_extrinsic(extr, delta)
            apply_extrinsic(vis, new_extr)

            new_pos = cam_pos + delta
            print(f"Camera @ [{new_pos[0]:6.2f}, {new_pos[1]:6.2f}, {new_pos[2]:6.2f}]")
            return False

        return cb

    def go_bev(vis, altitude=BEV_ALTITUDE_DEFAULT):
        cam_pos, _, _ = camera_basis(get_extrinsic(vis))
        ground_target = np.array([cam_pos[0], cam_pos[1], 0.0])
        eye = ground_target + np.array([0.0, 0.0, altitude])
        extr = look_at_extrinsic(eye, ground_target, world_up=np.array([1.0, 0.0, 0.0]))
        apply_extrinsic(vis, extr)
        state["bev"] = True

    def toggle_bev(vis):
        if not state["bev"]:
            state["saved_extrinsic"] = get_extrinsic(vis)
            go_bev(vis, altitude=BEV_ALTITUDE_DEFAULT)
            print("BEV / top-down ON")
        else:
            if state["saved_extrinsic"] is not None:
                apply_extrinsic(vis, state["saved_extrinsic"])
            state["bev"] = False
            print("BEV / top-down OFF")
        return False

    def increase_point_size(vis):
        state["point_size"] += 1.0
        vis.get_render_option().point_size = state["point_size"]
        print("Point size:", state["point_size"])
        return False

    def decrease_point_size(vis):
        state["point_size"] = max(1.0, state["point_size"] - 1.0)
        vis.get_render_option().point_size = state["point_size"]
        print("Point size:", state["point_size"])
        return False

    # ---------------- Filter / color toggles ----------------

    def toggle_ground_filter(vis):
        state["ground_filter_on"] = not state["ground_filter_on"]
        print(f"Ground-plane filter: {'ON' if state['ground_filter_on'] else 'off'}")
        refresh_cloud(vis)
        return False

    def toggle_highlight_top(vis):
        state["highlight_top_on"] = not state["highlight_top_on"]
        print(
            f"High-intensity overlay: {'ON' if state['highlight_top_on'] else 'off'}"
            f" @ {state['highlight_percentile']:.0f}%"
        )
        refresh_cloud(vis)
        return False

    def adjust_highlight_percentile(delta):
        def cb(vis):
            new_v = state["highlight_percentile"] + delta
            new_v = max(HIGHLIGHT_PERCENTILE_MIN, min(HIGHLIGHT_PERCENTILE_MAX, new_v))
            state["highlight_percentile"] = new_v
            print(f"Highlight percentile: {new_v:.0f}%")
            refresh_cloud(vis)
            return False

        return cb

    def cycle_intensity_cmap(vis):
        idx = INTENSITY_COLORMAPS.index(state["intensity_cmap"])
        state["intensity_cmap"] = INTENSITY_COLORMAPS[
            (idx + 1) % len(INTENSITY_COLORMAPS)
        ]
        print(f"Intensity colormap: {state['intensity_cmap']}")
        refresh_cloud(vis)
        return False

    def toggle_intensity_norm(vis):
        state["intensity_norm"] = (
            "percentile" if state["intensity_norm"] == "rank" else "rank"
        )
        print(f"Intensity normalization: {state['intensity_norm']}")
        refresh_cloud(vis)
        return False

    def set_color_mode(mode_name):
        def cb(vis):
            if mode_name in ("semseg", "lane") and frame_data["labels"] is None:
                print(
                    f"Color mode '{mode_name}' needs semantic labels — not available for this frame."
                )
                return False
            state["color_mode"] = mode_name
            print(f"Color mode: {mode_name}")
            refresh_cloud(vis)
            return False

        return cb

    def lanes_mode(vis):
        """One-key preset: optimal config for judging lane visibility."""
        if not state["bev"]:
            state["saved_extrinsic"] = get_extrinsic(vis)

        state["ground_filter_on"] = True
        state["highlight_top_on"] = True
        state["intensity_norm"] = "rank"
        state["intensity_cmap"] = "inferno"
        state["color_mode"] = "intensity"
        refresh_cloud(vis)
        go_bev(vis, altitude=BEV_ALTITUDE_LANES)
        print(
            "LANES MODE: BEV + ground filter + rank intensity + high-intensity highlight."
        )
        print("  Look for thin parallel cyan strips along the road.")
        return False

    def lanes_mode_off(vis):
        state["ground_filter_on"] = False
        state["highlight_top_on"] = False
        state["intensity_norm"] = "rank"
        state["intensity_cmap"] = INTENSITY_COLORMAPS[0]
        state["color_mode"] = "intensity"
        refresh_cloud(vis)
        apply_extrinsic(vis, initial_extrinsic)
        state["bev"] = False
        print("Lanes mode OFF — back to plain perspective view.")
        return False

    # ---------------- Navigation ----------------

    def request_next(vis):
        state["pending_action"] = "next"
        return False

    def request_prev(vis):
        state["pending_action"] = "prev"
        return False

    def process_navigation_action(action, vis):
        nonlocal frame_data
        direction = 1 if action == "next" else -1
        print(f"\nSearching for {action} frame with lane-line semseg...")

        next_data = find_next_valid_frame(
            current_frame_id=state["frame_id"],
            direction=direction,
            available_frame_ids=available_frame_ids,
            class_names=class_names,
        )
        if next_data is None:
            print(f"No {action} valid lane frame found.")
            return

        frame_data = next_data
        state["frame_id"] = next_data["frame_id"]
        state["lane_count"] = next_data["lane_count"]

        if state["color_mode"] in ("semseg", "lane") and next_data["labels"] is None:
            state["color_mode"] = "intensity"

        reload_camera_base()
        refresh_cloud(vis)

        cam_w2, cam_h2 = state["camera_size"]
        cv2.resizeWindow(camera_window_name, cam_w2, cam_h2)

        # Re-center the camera on the new frame's car (origin). Each frame's
        # points are in their own LiDAR-sensor frame with the car at (0,0,0),
        # so without this snap the user would be left wherever they had flown
        # to in the previous frame and would have to W/S back to the car.
        if state["bev"]:
            altitude = (
                BEV_ALTITUDE_LANES
                if state["ground_filter_on"]
                else BEV_ALTITUDE_DEFAULT
            )
            eye = np.array([0.0, 0.0, altitude])
            target = np.array([0.0, 0.0, 0.0])
            extr = look_at_extrinsic(eye, target, world_up=np.array([1.0, 0.0, 0.0]))
            apply_extrinsic(vis, extr)
            state["saved_extrinsic"] = initial_extrinsic
        else:
            apply_extrinsic(vis, initial_extrinsic)
            state["saved_extrinsic"] = None

        print("\nLoaded new frame:")
        print("  frame:", frame_name(state["frame_id"]))
        print("  ground z:", f"{next_data['ground_z']:.2f}")
        print("  lane points:", state["lane_count"])

    # ---------------- Key bindings ----------------

    vis.register_key_callback(ord("W"), make_move_callback(1, 0, 0, MOVE_STEP))
    vis.register_key_callback(ord("S"), make_move_callback(-1, 0, 0, MOVE_STEP))
    vis.register_key_callback(ord("D"), make_move_callback(0, 1, 0, MOVE_STEP))
    vis.register_key_callback(ord("A"), make_move_callback(0, -1, 0, MOVE_STEP))
    vis.register_key_callback(ord("E"), make_move_callback(0, 0, 1, VERTICAL_STEP))
    vis.register_key_callback(ord("C"), make_move_callback(0, 0, -1, VERTICAL_STEP))

    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("B"), toggle_bev)

    vis.register_key_callback(ord("="), increase_point_size)
    vis.register_key_callback(ord("+"), increase_point_size)
    vis.register_key_callback(ord("-"), decrease_point_size)

    vis.register_key_callback(ord("I"), set_color_mode("intensity"))
    vis.register_key_callback(ord("L"), set_color_mode("semseg"))
    vis.register_key_callback(ord("M"), set_color_mode("lane"))

    vis.register_key_callback(ord("G"), toggle_ground_filter)
    vis.register_key_callback(ord("H"), toggle_highlight_top)
    vis.register_key_callback(
        ord("["), adjust_highlight_percentile(-HIGHLIGHT_PERCENTILE_STEP)
    )
    vis.register_key_callback(
        ord("]"), adjust_highlight_percentile(+HIGHLIGHT_PERCENTILE_STEP)
    )
    vis.register_key_callback(ord("J"), cycle_intensity_cmap)
    vis.register_key_callback(ord("K"), toggle_intensity_norm)
    vis.register_key_callback(ord("T"), lanes_mode)
    vis.register_key_callback(ord("Y"), lanes_mode_off)

    vis.register_key_callback(ord("N"), request_next)
    vis.register_key_callback(ord("P"), request_prev)

    # Initial view: poll once so view-control is initialised, then set extrinsic.
    vis.poll_events()
    vis.update_renderer()
    apply_extrinsic(vis, initial_extrinsic)

    print_controls()

    # ---------------- Main loop ----------------
    while True:
        cv2.imshow(camera_window_name, state["camera_img"])

        key = cv2.waitKey(10) & 0xFF
        if key in (27, ord("q")):
            break

        # Camera-window keyboard mirrors the on-image buttons.
        if key == ord("n"):
            state["pending_action"] = "next"
        elif key == ord("p"):
            state["pending_action"] = "prev"

        if state["pending_action"] is not None:
            action = state["pending_action"]
            state["pending_action"] = None
            process_navigation_action(action, vis)

        alive = vis.poll_events()
        vis.update_renderer()
        if not alive:
            break

    vis.destroy_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

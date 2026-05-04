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

SEQUENCE_ID = "042"
FRAME_ID = 1

CAMERA_NAME = "front_camera"

# PandaSet:
# d == 0 -> 360 mechanical LiDAR
# d == 1 -> front-facing LiDAR
FORWARD_SENSOR_ID = 1

# If auto-detection fails, manually set this after checking printed classes.
# Example:
# LANE_LABEL_IDS = [24]
LANE_LABEL_IDS = None

# Navigation behavior:
# True = next/previous skips frames without semantic lane-line labels.
ONLY_LOAD_FRAMES_WITH_LANE_LINES = True

# Usually keep this False because you want road context too.
# True = display only annotated lane-line points.
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

CAMERA_WINDOW_X = 50
CAMERA_WINDOW_Y = 80
CAMERA_WINDOW_W = 850
CAMERA_WINDOW_H = 520

POINTCLOUD_WINDOW_X = 930
POINTCLOUD_WINDOW_Y = 50
POINTCLOUD_WINDOW_W = 1400
POINTCLOUD_WINDOW_H = 900


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
        stem = p.name
        stem = stem.replace(".pkl.gz", "")
        stem = stem.replace(".pkl", "")

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
# COLOR HELPERS
# ============================================================


def normalize_intensity(intensity):
    intensity = intensity.astype(np.float32)

    lo = np.percentile(intensity, 1)
    hi = np.percentile(intensity, 99.5)

    intensity = np.clip(intensity, lo, hi)
    intensity = (intensity - lo) / (hi - lo + 1e-8)

    return intensity**0.5


def make_intensity_colors(intensity):
    intensity_norm = normalize_intensity(intensity)

    # low intensity = blue/purple, high intensity = yellow/red
    colors = plt.get_cmap("turbo")(intensity_norm)[:, :3]
    return colors


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


def make_cloud_from_colors(points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def update_cloud_geometry(cloud, points, colors, vis):
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    vis.update_geometry(cloud)


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

            id_col = None
            name_col = None

            for c in id_candidates:
                if c in cols:
                    id_col = c
                    break

            for c in name_candidates:
                if c in cols:
                    name_col = c
                    break

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

    lane_ids = []

    for class_id, class_name in class_names.items():
        name = class_name.lower()

        if "lane" in name:
            lane_ids.append(class_id)

    return lane_ids


def make_lane_highlight_colors(intensity_colors, labels, lane_ids):
    colors = intensity_colors.copy() * 0.18

    if not lane_ids:
        return colors

    lane_mask = np.isin(labels, lane_ids)
    colors[lane_mask] = np.array([1.0, 1.0, 0.0])

    return colors


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

    if not all_candidates:
        raise FileNotFoundError(
            f"No camera image found for sequence {SEQUENCE_ID}, frame {name}"
        )

    print(f"Camera '{CAMERA_NAME}' not found. Using:", all_candidates[0])
    return all_candidates[0]


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


def draw_navigation_buttons(img, state):
    """
    Draw clickable prev/next buttons on the camera image.

    Returns:
      image_with_overlay
      button_rects dict
    """
    img = img.copy()

    h, w = img.shape[:2]

    button_h = 48
    margin = 18
    y1 = h - button_h - margin
    y2 = h - margin

    prev_rect = (margin, y1, margin + 190, y2)
    next_rect = (w - margin - 190, y1, w - margin, y2)

    info_rect = (prev_rect[2] + 15, y1, next_rect[0] - 15, y2)

    # Prev button
    cv2.rectangle(
        img,
        (prev_rect[0], prev_rect[1]),
        (prev_rect[2], prev_rect[3]),
        (40, 40, 40),
        -1,
    )
    cv2.rectangle(
        img,
        (prev_rect[0], prev_rect[1]),
        (prev_rect[2], prev_rect[3]),
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        "< Prev lane",
        (prev_rect[0] + 22, prev_rect[1] + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Next button
    cv2.rectangle(
        img,
        (next_rect[0], next_rect[1]),
        (next_rect[2], next_rect[3]),
        (40, 40, 40),
        -1,
    )
    cv2.rectangle(
        img,
        (next_rect[0], next_rect[1]),
        (next_rect[2], next_rect[3]),
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        "Next lane >",
        (next_rect[0] + 18, next_rect[1] + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Info area
    cv2.rectangle(
        img, (info_rect[0], info_rect[1]), (info_rect[2], info_rect[3]), (0, 0, 0), -1
    )

    text = (
        f"seq {SEQUENCE_ID} | frame {state['frame_id']:02d} | "
        f"mode {state['color_mode']} | lanes {state['lane_count']}"
    )

    cv2.putText(
        img,
        text,
        (info_rect[0] + 10, info_rect[1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    button_rects = {
        "prev": prev_rect,
        "next": next_rect,
    }

    return img, button_rects


def load_camera_image(frame_id, state):
    camera_file = find_camera_image(frame_id)

    img = cv2.imread(str(camera_file))
    if img is None:
        raise RuntimeError(f"Could not read camera image: {camera_file}")

    h, w = img.shape[:2]
    scale = min(CAMERA_WINDOW_W / w, CAMERA_WINDOW_H / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img = cv2.resize(img, (new_w, new_h))
    img = draw_axis_legend_on_image(img)
    img, button_rects = draw_navigation_buttons(img, state)

    return img, new_w, new_h, camera_file, button_rects


# ============================================================
# FRAME LOADING
# ============================================================


def load_frame_data(frame_id, class_names, require_lane=False, verbose=True):
    """
    Loads one frame.

    If require_lane=True:
      returns None when semseg is missing or lane labels are absent.
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

    intensity_colors = make_intensity_colors(intensity)

    color_modes = {
        "intensity": intensity_colors,
    }

    labels = None
    lane_ids = find_lane_label_ids(class_names)
    lane_count = 0

    if has_semseg and label_col is not None:
        labels = merged[label_col].astype(np.int32).to_numpy()

        semseg_colors = make_semseg_colors(labels)

        lane_mask = (
            np.isin(labels, lane_ids) if lane_ids else np.zeros(len(labels), dtype=bool)
        )
        lane_count = int(lane_mask.sum())

        if require_lane and lane_count == 0:
            return None

        lane_colors = make_lane_highlight_colors(
            intensity_colors=intensity_colors,
            labels=labels,
            lane_ids=lane_ids,
        )

        color_modes["semseg"] = semseg_colors
        color_modes["lane"] = lane_colors

        if SHOW_ONLY_LANE_POINTS:
            points = points[lane_mask]
            intensity_colors = intensity_colors[lane_mask]
            semseg_colors = semseg_colors[lane_mask]
            lane_colors = lane_colors[lane_mask]

            color_modes = {
                "intensity": intensity_colors,
                "semseg": semseg_colors,
                "lane": lane_colors,
            }

            lane_count = len(points)

    elif require_lane:
        return None

    if verbose:
        print("\nLoaded frame:")
        print("  frame:", frame_name(frame_id))
        print("  lidar:", lidar_path)
        print("  semseg:", semseg_path)
        print("  points after d filter:", len(points))
        print("  lane ids:", lane_ids)
        print("  lane count:", lane_count)

        if labels is not None:
            print("  unique labels:", np.unique(labels))

    return {
        "frame_id": frame_id,
        "points": points,
        "color_modes": color_modes,
        "has_semseg": has_semseg,
        "lane_ids": lane_ids,
        "lane_count": lane_count,
    }


def find_next_valid_frame(
    current_frame_id, direction, available_frame_ids, class_names
):
    """
    Finds next/previous frame.

    If ONLY_LOAD_FRAMES_WITH_LANE_LINES=True:
      skips frames without semantic lane-line labels.
    """
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
        f"\nInitial frame {start_frame_id:02d} does not have lane-line semseg."
        " Searching forward..."
    )

    data = find_next_valid_frame(
        current_frame_id=start_frame_id,
        direction=1,
        available_frame_ids=available_frame_ids,
        class_names=class_names,
    )

    if data is not None:
        return data

    print("No forward frame found. Searching backward...")

    data = find_next_valid_frame(
        current_frame_id=start_frame_id,
        direction=-1,
        available_frame_ids=available_frame_ids,
        class_names=class_names,
    )

    if data is not None:
        return data

    raise RuntimeError(
        "Could not find any frame with semantic lane-line labels. "
        "Try setting ONLY_LOAD_FRAMES_WITH_LANE_LINES=False or manually set LANE_LABEL_IDS."
    )


# ============================================================
# CONTROLS
# ============================================================


def print_controls():
    print("\nControls:")
    print("  Camera-window buttons:")
    print("    < Prev lane       previous frame with lane-line semseg")
    print("    Next lane >       next frame with lane-line semseg")
    print("")
    print("  Keyboard:")
    print("    N                 next frame with lane-line semseg")
    print("    P                 previous frame with lane-line semseg")
    print("    I                 intensity colors")
    print("    L                 semantic label colors")
    print("    M                 lane marking highlight")
    print("    Mouse drag        rotate")
    print("    Mouse wheel       zoom")
    print("    W / S             move forward / backward")
    print("    A / D             strafe left / right")
    print("    E / C             move up / down")
    print("    R                 reset view")
    print("    B                 toggle top-down / BEV")
    print("    + / =             increase point size")
    print("    -                 decrease point size")
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

    cloud = make_cloud_from_colors(
        frame_data["points"],
        frame_data["color_modes"]["intensity"],
    )

    car_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=AXIS_SIZE,
        origin=[0, 0, 0],
    )

    initial_extrinsic = look_at_extrinsic(INITIAL_EYE, INITIAL_TARGET, WORLD_UP)

    state = {
        "frame_id": frame_data["frame_id"],
        "point_size": POINT_SIZE,
        "bev": False,
        "saved_extrinsic": None,
        "color_mode": "intensity",
        "color_modes": frame_data["color_modes"],
        "lane_count": frame_data["lane_count"],
        "pending_action": None,
        "button_rects": {},
        "camera_img": None,
        "camera_size": None,
    }

    # ---------------- Camera image window ----------------
    camera_img, cam_w, cam_h, camera_file, button_rects = load_camera_image(
        state["frame_id"],
        state,
    )

    state["camera_img"] = camera_img
    state["camera_size"] = (cam_w, cam_h)
    state["button_rects"] = button_rects

    camera_window_name = f"Camera | {CAMERA_NAME} | seq {SEQUENCE_ID}"

    cv2.namedWindow(camera_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(camera_window_name, cam_w, cam_h)
    cv2.moveWindow(camera_window_name, CAMERA_WINDOW_X, CAMERA_WINDOW_Y)

    def on_camera_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        rects = state["button_rects"]

        for action, rect in rects.items():
            x1, y1, x2, y2 = rect

            if x1 <= x <= x2 and y1 <= y <= y2:
                state["pending_action"] = action
                return

    cv2.setMouseCallback(camera_window_name, on_camera_mouse)

    # ---------------- Open3D point-cloud window ----------------
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"PandaSet front-facing LiDAR | seq {SEQUENCE_ID}",
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

    def toggle_bev(vis):
        if not state["bev"]:
            state["saved_extrinsic"] = get_extrinsic(vis)
            cam_pos, _, _ = camera_basis(state["saved_extrinsic"])

            ground_target = np.array([cam_pos[0], cam_pos[1], 0.0])
            eye = ground_target + np.array([0.0, 0.0, 60.0])

            extr = look_at_extrinsic(
                eye,
                ground_target,
                world_up=np.array([1.0, 0.0, 0.0]),
            )

            apply_extrinsic(vis, extr)
            state["bev"] = True
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

    def set_color_mode(mode_name):
        def cb(vis):
            if mode_name not in state["color_modes"]:
                print(f"\nColor mode '{mode_name}' not available.")
                print("Available modes:", list(state["color_modes"].keys()))
                return False

            state["color_mode"] = mode_name
            update_cloud_geometry(
                cloud,
                np.asarray(cloud.points),
                state["color_modes"][mode_name],
                vis,
            )

            print(f"\nColor mode: {mode_name}")
            return False

        return cb

    def request_next(vis):
        state["pending_action"] = "next"
        return False

    def request_prev(vis):
        state["pending_action"] = "prev"
        return False

    def process_navigation_action(action):
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

        state["frame_id"] = next_data["frame_id"]
        state["color_modes"] = next_data["color_modes"]
        state["lane_count"] = next_data["lane_count"]

        if state["color_mode"] not in state["color_modes"]:
            state["color_mode"] = "intensity"

        update_cloud_geometry(
            cloud,
            next_data["points"],
            state["color_modes"][state["color_mode"]],
            vis,
        )

        camera_img, cam_w, cam_h, camera_file, button_rects = load_camera_image(
            state["frame_id"],
            state,
        )

        state["camera_img"] = camera_img
        state["camera_size"] = (cam_w, cam_h)
        state["button_rects"] = button_rects

        cv2.resizeWindow(camera_window_name, cam_w, cam_h)

        print("\nLoaded new frame:")
        print("  frame:", frame_name(state["frame_id"]))
        print("  camera:", camera_file)
        print("  mode:", state["color_mode"])
        print("  lane points:", state["lane_count"])

    # Movement.
    vis.register_key_callback(ord("W"), make_move_callback(1, 0, 0, MOVE_STEP))
    vis.register_key_callback(ord("S"), make_move_callback(-1, 0, 0, MOVE_STEP))
    vis.register_key_callback(ord("D"), make_move_callback(0, 1, 0, MOVE_STEP))
    vis.register_key_callback(ord("A"), make_move_callback(0, -1, 0, MOVE_STEP))
    vis.register_key_callback(ord("E"), make_move_callback(0, 0, 1, VERTICAL_STEP))
    vis.register_key_callback(ord("C"), make_move_callback(0, 0, -1, VERTICAL_STEP))

    # View controls.
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("B"), toggle_bev)

    # Point size.
    vis.register_key_callback(ord("="), increase_point_size)
    vis.register_key_callback(ord("+"), increase_point_size)
    vis.register_key_callback(ord("-"), decrease_point_size)

    # Color modes.
    vis.register_key_callback(ord("I"), set_color_mode("intensity"))
    vis.register_key_callback(ord("L"), set_color_mode("semseg"))
    vis.register_key_callback(ord("M"), set_color_mode("lane"))

    # Navigation.
    vis.register_key_callback(ord("N"), request_next)
    vis.register_key_callback(ord("P"), request_prev)

    # Initial view.
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

        # Camera window keyboard shortcuts.
        if key == ord("n"):
            state["pending_action"] = "next"
        elif key == ord("p"):
            state["pending_action"] = "prev"

        if state["pending_action"] is not None:
            action = state["pending_action"]
            state["pending_action"] = None
            process_navigation_action(action)

        alive = vis.poll_events()
        vis.update_renderer()

        if not alive:
            break

    vis.destroy_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

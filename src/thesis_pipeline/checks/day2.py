from pathlib import Path

from pandaset import DataSet

from thesis_pipeline.core.pandaset_compat import sorted_frame_keys
from thesis_pipeline.core.paths import (
    LOGS_DIR,
    read_dataset_root,
    read_day2_chosen_frame,
    read_day2_chosen_sequence,
    read_day2_forward_sensor,
)


def precheck_sequence_and_sensor_main() -> None:
    root = read_dataset_root()
    ds = DataSet(root)

    sequence_ids = sorted([p.name for p in Path(root).iterdir() if p.is_dir() and p.name.isdigit()])
    print("sequence_count", len(sequence_ids))
    print("sequence_head", sequence_ids[:10])

    chosen_seq = "001" if "001" in sequence_ids else sequence_ids[0]
    seq = ds[chosen_seq]

    try:
        seq.load_lidar().load_semseg()
    except Exception:
        seq.load_lidar()
        seq.load_semseg()

    raw_frame_keys = sorted_frame_keys(seq.lidar)
    first_key = raw_frame_keys[0]
    chosen_frame = int(first_key) if str(first_key).isdigit() else first_key

    pc_all = seq.lidar[chosen_frame]
    all_cols = list(pc_all.columns)
    print("chosen_seq", chosen_seq)
    print("chosen_frame", chosen_frame)
    print("all_points", len(pc_all))
    print("all_columns", all_cols)

    sensor_ok = False
    sensor_column_present = "d" in all_cols
    if sensor_column_present:
        print("all_unique_sensor_ids", sorted(pc_all["d"].unique().tolist()))

    seq.lidar.set_sensor(1)
    pc_front = seq.lidar[chosen_frame]
    print("front_points", len(pc_front))
    if sensor_column_present and "d" in pc_front.columns:
        front_ids = sorted(pc_front["d"].unique().tolist())
        print("front_unique_sensor_ids", front_ids)
        sensor_ok = len(pc_front) > 0 and front_ids == [1]
    else:
        print("front_unique_sensor_ids", "COLUMN_NOT_AVAILABLE")
        sensor_ok = len(pc_front) > 0

    (LOGS_DIR / "day2_chosen_sequence.txt").write_text(f"{chosen_seq}\n")
    (LOGS_DIR / "day2_chosen_frame.txt").write_text(f"{chosen_frame}\n")
    (LOGS_DIR / "day2_forward_sensor.txt").write_text("1\n")
    print("forward_sensor_candidate", 1)
    print("forward_sensor_usable", sensor_ok)


def check_pandaset_frame_access_main() -> None:
    root = read_dataset_root()
    seq_id = read_day2_chosen_sequence()
    frame_idx = read_day2_chosen_frame()

    ds = DataSet(root)
    seq = ds[seq_id]
    seq.load_lidar()
    frame = seq.lidar[frame_idx]
    print("chosen_seq", seq_id)
    print("chosen_frame", frame_idx)
    print("frame_type", type(frame))
    print("frame_rows", len(frame))
    print("frame_columns", list(frame.columns))


def check_pandaset_semseg_access_main() -> None:
    root = read_dataset_root()
    seq_id = read_day2_chosen_sequence()

    ds = DataSet(root)
    seq = ds[seq_id]
    seq.load_semseg()
    print("chosen_seq", seq_id)
    print("semseg_loaded", seq.semseg is not None)
    print("classes_type", type(seq.semseg.classes))
    print("classes_sample", list(seq.semseg.classes.items())[:12])


def check_raw_label_ids_main() -> None:
    root = read_dataset_root()
    seq_id = read_day2_chosen_sequence()

    ds = DataSet(root)
    seq = ds[seq_id]
    seq.load_semseg()

    classes = seq.semseg.classes
    print("classes_type", type(classes))

    for k in [1, 2, 3, 4, 7, 8, 9, 10]:
        val = classes.get(k, classes.get(str(k), "NOT_FOUND"))
        print(f"id_{k}", val)


def check_forward_only_filter_main() -> None:
    root = read_dataset_root()
    seq_id = read_day2_chosen_sequence()
    frame_idx = read_day2_chosen_frame()
    forward_sensor = read_day2_forward_sensor()

    ds = DataSet(root)
    seq = ds[seq_id]
    seq.load_lidar()
    pc_all = seq.lidar[frame_idx]
    print("chosen_seq", seq_id)
    print("chosen_frame", frame_idx)
    print("all_points", len(pc_all))
    print("all_columns", list(pc_all.columns))

    if "d" in pc_all.columns:
        print("all_unique_sensor_ids", sorted(pc_all["d"].unique().tolist()))

    seq.lidar.set_sensor(forward_sensor)
    pc_front = seq.lidar[frame_idx]
    print("front_sensor", forward_sensor)
    print("front_points", len(pc_front))
    if "d" in pc_front.columns:
        print("front_unique_sensor_ids", sorted(pc_front["d"].unique().tolist()))
    else:
        print("front_unique_sensor_ids", "COLUMN_NOT_AVAILABLE")


def check_semseg_alignment_main() -> None:
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

    raw_labels = semseg_df.loc[pc_df.index, "class"]
    print("point_count", len(pc_df))
    print("aligned_label_count", len(raw_labels))
    print("same_count", len(pc_df) == len(raw_labels))
    print("label_head", raw_labels.head().tolist())


def check_intensity_field_main() -> None:
    root = read_dataset_root()
    seq_id = read_day2_chosen_sequence()
    frame_idx = read_day2_chosen_frame()
    forward_sensor = read_day2_forward_sensor()

    ds = DataSet(root)
    seq = ds[seq_id]
    seq.load_lidar()
    seq.lidar.set_sensor(forward_sensor)
    pc_df = seq.lidar[frame_idx]

    print("columns", list(pc_df.columns))
    intensity_column = "i" if "i" in pc_df.columns else None
    print("intensity_column", intensity_column)
    if intensity_column is not None:
        print("intensity_head", pc_df[intensity_column].head().tolist())
        print("intensity_min", float(pc_df[intensity_column].min()))
        print("intensity_max", float(pc_df[intensity_column].max()))
    else:
        print("intensity_head", "COLUMN_NOT_FOUND")

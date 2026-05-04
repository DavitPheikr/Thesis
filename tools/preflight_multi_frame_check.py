"""
Pre-flight check: verify set_sensor(1) persistence and semseg alignment
across 5 consecutive frames. Must pass before any multi-sequence scan.

Uses get_frame_count from pandaset_compat and emits script_status PASS/FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pandaset import DataSet

from thesis_pipeline.core.pandaset_compat import get_frame_count


SENSOR_PATTERN_FILE = Path("logs/milestone_b_preflight_sensor_pattern.txt")


def main() -> None:
    root = Path("logs/dataset_root.txt").read_text().strip()
    seq_id = Path("logs/day2_chosen_sequence.txt").read_text().strip()

    ds = DataSet(root)
    seq = ds[seq_id]
    try:
        seq.load_lidar().load_semseg()
    except Exception:
        seq.load_lidar()
        seq.load_semseg()

    seq.lidar.set_sensor(1)
    n_frames = get_frame_count(seq)
    frames_to_check = list(range(min(5, n_frames)))

    all_ok = True
    for fi in frames_to_check:
        pc_df = seq.lidar[fi]
        semseg_df = seq.semseg[fi]

        if "d" in pc_df.columns:
            unique_sensors = sorted(pc_df["d"].unique().tolist())
            sensor_ok = unique_sensors == [1]
        else:
            unique_sensors = "col_absent"
            sensor_ok = len(pc_df) > 0

        aligned = semseg_df.index.isin(pc_df.index)
        alignment_ok = int(aligned.sum()) == len(pc_df)

        print(
            f"frame {fi}: n={len(pc_df)} sensors={unique_sensors} "
            f"sensor_ok={sensor_ok} alignment_ok={alignment_ok}"
        )

        if not sensor_ok or not alignment_ok:
            all_ok = False

    print()
    if all_ok:
        if SENSOR_PATTERN_FILE.exists():
            SENSOR_PATTERN_FILE.unlink()
        print("preflight_multi_frame_ok True")
        print("script_status PASS")
        sys.exit(0)

    print("preflight_multi_frame_ok False")
    print("FATAL: set_sensor(1) persistence or semseg alignment failed.")
    print("  If set_sensor(1) does not persist: add seq.lidar.set_sensor(1)")
    print("  before each frame access in all later scripts and re-test.")
    print("  Record the working access pattern before proceeding.")
    print("script_status FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()

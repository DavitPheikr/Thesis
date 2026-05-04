"""
Audit all local PandaSet sequences for semseg availability and lane presence.
Checks all forward-only frames per sequence, writes incremental JSONL progress,
and emits script_status PASS/FAIL as the final line.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pandaset import DataSet

from thesis_pipeline.core.pandaset_compat import get_frame_count


LANE_RAW_ID = 8
PROGRESS_FILE = Path("logs/milestone_b_audit_progress.jsonl")
MANIFEST_FILE = Path("logs/milestone_b_sequence_manifest.json")
PREFLIGHT_PATTERN_FILE = Path("logs/milestone_b_preflight_sensor_pattern.txt")
TERMINAL_ERRORS = {"semseg_unavailable", "semseg_empty"}


def is_completed_record(record: dict) -> bool:
    error = record.get("error")
    return error is None or error in TERMINAL_ERRORS


def load_canonical_progress(progress_path: Path) -> dict[str, dict]:
    """Load one canonical terminal record per seq_id from the progress file.

    Identical duplicates are tolerated and collapsed to one record. Any
    non-identical duplicate for the same seq_id is treated as a conflict.
    """
    canonical: dict[str, dict] = {}
    if not progress_path.exists():
        return canonical

    for line_no, line in enumerate(progress_path.read_text().splitlines(), start=1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed JSON in {progress_path} at line {line_no}: {exc}"
            ) from exc

        seq_id = rec.get("seq_id")
        if not seq_id:
            raise RuntimeError(
                f"Missing seq_id in {progress_path} at line {line_no}: {rec}"
            )

        existing = canonical.get(seq_id)
        if existing is None:
            canonical[seq_id] = rec
            continue

        if rec != existing:
            raise RuntimeError(
                "Conflicting records found for seq_id "
                f"{seq_id} in {progress_path}. "
                "Resolve manually before resuming the audit."
            )

    return canonical


def load_completed(progress_path: Path) -> set[str]:
    """Return seq_ids already resolved to terminal outcomes."""
    return {
        seq_id
        for seq_id, record in load_canonical_progress(progress_path).items()
        if is_completed_record(record)
    }


def sensor_reapply_required() -> bool:
    if not PREFLIGHT_PATTERN_FILE.exists():
        return False
    return "reapply_per_frame: true" in PREFLIGHT_PATTERN_FILE.read_text()


def audit_one_sequence(ds: DataSet, seq_id: str, sensor_reapply_per_frame: bool = False) -> dict:
    result = {
        "seq_id": seq_id,
        "semseg_available": False,
        "lane_present": False,
        "lidar_frame_count": 0,
        "semseg_frame_count": 0,
        "frames_scanned": 0,
        "frames_with_lane": 0,
        "total_lane_points": 0,
        "frames_failed": 0,
        "frames_semseg_missing": 0,
        "sensor_first_frame_ok": False,
        "error": None,
    }
    try:
        seq = ds[seq_id]
        try:
            seq.load_lidar().load_semseg()
        except Exception:
            seq.load_lidar()
            seq.load_semseg()

        if seq.semseg is None:
            result["error"] = "semseg_unavailable"
            return result

        lidar_n = get_frame_count(seq)
        result["lidar_frame_count"] = lidar_n

        try:
            semseg_n = len(seq.semseg.data)
        except Exception:
            semseg_n = lidar_n
        result["semseg_frame_count"] = semseg_n

        if semseg_n == 0:
            result["error"] = "semseg_empty"
            return result

        result["semseg_available"] = True

        if not sensor_reapply_per_frame:
            seq.lidar.set_sensor(1)

        for fi in range(lidar_n):
            try:
                if sensor_reapply_per_frame:
                    seq.lidar.set_sensor(1)
                pc_df = seq.lidar[fi]

                if fi == 0 and "d" in pc_df.columns:
                    result["sensor_first_frame_ok"] = (
                        sorted(pc_df["d"].unique().tolist()) == [1]
                    )
                elif fi == 0:
                    result["sensor_first_frame_ok"] = len(pc_df) > 0

                if fi >= semseg_n:
                    result["frames_semseg_missing"] += 1
                    continue

                semseg_df = seq.semseg[fi]
                raw_labels = semseg_df.loc[pc_df.index, "class"].to_numpy(dtype="int32")
                lane_count = int((raw_labels == LANE_RAW_ID).sum())

                result["frames_scanned"] += 1
                result["total_lane_points"] += lane_count
                if lane_count > 0:
                    result["frames_with_lane"] += 1
            except Exception:
                result["frames_failed"] += 1

        result["lane_present"] = result["total_lane_points"] > 0
    except Exception as exc:
        result["error"] = str(exc)

    return result


def main() -> None:
    root = Path("logs/dataset_root.txt").read_text().strip()
    ds = DataSet(root)
    sensor_reapply = sensor_reapply_required()

    print(f"sensor_reapply_per_frame {sensor_reapply}")

    all_seq_ids = sorted(
        [p.name for p in Path(root).iterdir() if p.is_dir() and p.name.isdigit()]
    )
    print(f"total_sequence_dirs {len(all_seq_ids)}")

    completed = load_completed(PROGRESS_FILE)
    print(f"already_completed {len(completed)}")

    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

    canonical_progress = load_canonical_progress(PROGRESS_FILE)
    results: list[dict] = list(canonical_progress.values())

    with PROGRESS_FILE.open("a") as progress_f:
        for seq_id in all_seq_ids:
            if seq_id in completed:
                continue
            print(f"  auditing {seq_id}...", end=" ", flush=True)
            record = audit_one_sequence(ds, seq_id, sensor_reapply)
            print(
                f"semseg={record['semseg_available']} lane={record['lane_present']} "
                f"failed={record['frames_failed']} missing={record['frames_semseg_missing']} "
                f"err={record['error']}"
            )
            progress_f.write(json.dumps(record) + "\n")
            progress_f.flush()
            results.append(record)
            del record
            gc.collect()

    semseg_pool = [r for r in results if r["semseg_available"]]
    lane_pool = [r for r in semseg_pool if r["lane_present"]]

    manifest = {
        "total_dirs": len(all_seq_ids),
        "semseg_enabled_count": len(semseg_pool),
        "lane_bearing_count": len(lane_pool),
        "semseg_enabled_ids": sorted(r["seq_id"] for r in semseg_pool),
        "lane_bearing_ids": sorted(r["seq_id"] for r in lane_pool),
        "non_lane_semseg_ids": sorted(
            r["seq_id"] for r in semseg_pool if not r["lane_present"]
        ),
        "per_sequence_details": {r["seq_id"]: r for r in semseg_pool},
        "global_frames_failed": sum(r["frames_failed"] for r in semseg_pool),
        "global_frames_semseg_missing": sum(r["frames_semseg_missing"] for r in semseg_pool),
    }

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))

    print()
    print("audit_complete True")
    print(f"semseg_enabled_count {manifest['semseg_enabled_count']}")
    print(f"lane_bearing_count {manifest['lane_bearing_count']}")
    print(f"pool_size_adequate {manifest['semseg_enabled_count'] >= 70}")
    print(f"global_frames_failed {manifest['global_frames_failed']}")

    ok = manifest["semseg_enabled_count"] >= 70 and manifest["lane_bearing_count"] >= 10
    print(f"script_status {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

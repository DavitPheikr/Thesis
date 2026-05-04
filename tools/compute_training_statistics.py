"""
Compute intensity statistics and class counts on the training split only.
Resume-safe: persists per-sequence artifacts and aggregates from all completed
per-sequence records.

Output:
  - logs/milestone_b_training_statistics.json (statistics section)
  - logs/stats_per_seq/{seq_id}.npy (per-sequence intensity samples)
  - logs/milestone_b_statistics_progress.jsonl (resume checkpoint)

Emits script_status PASS/FAIL as final line.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pandaset import DataSet

from thesis_pipeline.adapters.pandaset_ff_lane3 import remap_raw_pandaset_ids
from thesis_pipeline.core.pandaset_compat import get_frame_count


TRAIN_SPLIT_FILE = Path("configs/splits/train.txt")
PROGRESS_FILE = Path("logs/milestone_b_statistics_progress.jsonl")
STATS_FILE = Path("logs/milestone_b_training_statistics.json")
PER_SEQ_DIR = Path("logs/stats_per_seq")
PREFLIGHT_PATTERN = Path("logs/milestone_b_preflight_sensor_pattern.txt")

PER_SEQUENCE_SAMPLE_CAP = 100_000
SKIP_RATE_THRESHOLD = 0.05


def get_sensor_reapply() -> bool:
    if PREFLIGHT_PATTERN.exists():
        return "reapply_per_frame: true" in PREFLIGHT_PATTERN.read_text()
    return False


def is_completed_record(record: dict) -> bool:
    return record.get("error") is None


def load_canonical_progress(progress_path: Path) -> dict[str, dict]:
    canonical: dict[str, dict] = {}
    if not progress_path.exists():
        return canonical

    for line_no, line in enumerate(progress_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed JSON in {progress_path} at line {line_no}: {exc}"
            ) from exc

        seq_id = record.get("seq_id")
        if not seq_id:
            raise RuntimeError(
                f"Missing seq_id in {progress_path} at line {line_no}: {record}"
            )

        existing = canonical.get(seq_id)
        if existing is None:
            canonical[seq_id] = record
            continue
        if record != existing:
            raise RuntimeError(
                "Conflicting statistics records found for seq_id "
                f"{seq_id} in {progress_path}. Resolve manually before resuming."
            )

    return canonical


def load_completed() -> dict[str, dict]:
    canonical = load_canonical_progress(PROGRESS_FILE)
    return {
        seq_id: record
        for seq_id, record in canonical.items()
        if is_completed_record(record)
    }


def sample_intensity_values(intensity_chunks: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    if not intensity_chunks:
        return np.array([], dtype=np.float32)

    values = np.concatenate(intensity_chunks).astype(np.float32, copy=False)
    if values.shape[0] <= PER_SEQUENCE_SAMPLE_CAP:
        return values

    idx = rng.choice(values.shape[0], size=PER_SEQUENCE_SAMPLE_CAP, replace=False)
    return values[idx]


def process_sequence(ds: DataSet, seq_id: str, sensor_reapply: bool, rng: np.random.Generator) -> dict:
    result = {
        "seq_id": seq_id,
        "lidar_frames": 0,
        "semseg_frames": 0,
        "frames_processed": 0,
        "frames_semseg_missing": 0,
        "frames_failed": 0,
        "class_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        "intensity_sample_path": None,
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
        result["lidar_frames"] = lidar_n

        try:
            semseg_n = len(seq.semseg.data)
        except Exception:
            semseg_n = lidar_n
        result["semseg_frames"] = semseg_n

        if not sensor_reapply:
            seq.lidar.set_sensor(1)

        intensity_chunks: list[np.ndarray] = []
        for fi in range(lidar_n):
            try:
                if sensor_reapply:
                    seq.lidar.set_sensor(1)
                pc_df = seq.lidar[fi]

                if fi >= semseg_n:
                    result["frames_semseg_missing"] += 1
                    continue

                semseg_df = seq.semseg[fi]
                raw_labels = semseg_df.loc[pc_df.index, "class"].to_numpy(dtype="int32")
                remapped = remap_raw_pandaset_ids(raw_labels)

                for cls in (0, 1, 2, 3):
                    result["class_counts"][cls] += int((remapped == cls).sum())

                intensity_chunks.append(pc_df["i"].to_numpy(dtype="float32"))
                result["frames_processed"] += 1
            except Exception:
                result["frames_failed"] += 1

        samples = sample_intensity_values(intensity_chunks, rng)
        if samples.size > 0:
            out_path = PER_SEQ_DIR / f"{seq_id}.npy"
            np.save(out_path, samples)
            result["intensity_sample_path"] = str(out_path)
    except Exception as exc:
        result["error"] = str(exc)

    return result


def aggregate_from_artifacts(completed_records: dict[str, dict]):
    all_intensity = []
    total_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_frames_processed = 0
    total_frames_semseg_missing = 0
    total_frames_failed = 0

    for record in completed_records.values():
        class_counts = record["class_counts"]
        for cls in (0, 1, 2, 3):
            total_class_counts[cls] += class_counts.get(str(cls), class_counts.get(cls, 0))

        total_frames_processed += record["frames_processed"]
        total_frames_semseg_missing += record["frames_semseg_missing"]
        total_frames_failed += record["frames_failed"]

        npy_path = record.get("intensity_sample_path")
        if npy_path and Path(npy_path).exists():
            all_intensity.append(np.load(npy_path))

    if all_intensity:
        intensity_all = np.concatenate(all_intensity).astype(np.float32, copy=False)
    else:
        intensity_all = np.array([], dtype=np.float32)

    denominator = total_frames_processed + total_frames_semseg_missing
    skip_rate = total_frames_semseg_missing / denominator if denominator else 0.0

    return (
        total_class_counts,
        intensity_all,
        total_frames_processed,
        total_frames_semseg_missing,
        total_frames_failed,
        skip_rate,
    )


def main() -> None:
    sensor_reapply = get_sensor_reapply()
    print(f"sensor_reapply_per_frame {sensor_reapply}")

    train_ids = [line.strip() for line in TRAIN_SPLIT_FILE.read_text().splitlines() if line.strip()]
    print(f"training_sequences {len(train_ids)}")

    root = Path("logs/dataset_root.txt").read_text().strip()
    ds = DataSet(root)
    rng = np.random.default_rng(seed=42)

    PER_SEQ_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_completed()
    print(f"already_completed {len(completed)}")

    with PROGRESS_FILE.open("a") as progress_f:
        for seq_id in train_ids:
            if seq_id in completed:
                continue
            print(f"  processing {seq_id}...", end=" ", flush=True)
            record = process_sequence(ds, seq_id, sensor_reapply, rng)
            print(
                f"processed={record['frames_processed']} failed={record['frames_failed']} "
                f"missing={record['frames_semseg_missing']} err={record['error']}"
            )
            progress_f.write(json.dumps(record) + "\n")
            progress_f.flush()
            if is_completed_record(record):
                completed[seq_id] = record
            del record
            gc.collect()

    (
        class_counts,
        intensity_all,
        frames_processed,
        frames_semseg_missing,
        frames_failed,
        skip_rate,
    ) = aggregate_from_artifacts(completed)

    print()
    print(f"sequences_completed {len(completed)}")
    print(f"sequences_expected {len(train_ids)}")
    print(f"frames_processed {frames_processed}")
    print(f"frames_semseg_missing {frames_semseg_missing}")
    print(f"frames_failed {frames_failed}")
    print(f"skip_rate {skip_rate:.6f}")
    print(f"class_counts {class_counts}")

    if skip_rate > SKIP_RATE_THRESHOLD:
        print(f"FAIL: skip_rate {skip_rate:.6f} exceeds threshold {SKIP_RATE_THRESHOLD}")
        print("  Investigate semseg frame count mismatches before proceeding.")
        print("script_status FAIL")
        sys.exit(1)

    lane_count = int(class_counts.get(2, class_counts.get("2", 0)))
    if lane_count <= 0:
        print("FAIL: zero lane-class points in training split")
        print("script_status FAIL")
        sys.exit(1)

    if intensity_all.size == 0:
        print("FAIL: no intensity samples collected")
        print("script_status FAIL")
        sys.exit(1)

    p_low = float(np.percentile(intensity_all, 0.5))
    p_high = float(np.percentile(intensity_all, 99.5))
    clipped = np.clip(intensity_all, p_low, p_high)
    clip_mean = float(clipped.mean())
    clip_std = float(clipped.std())

    if clip_std <= 0:
        print("FAIL: clip_std is zero after clipping")
        print("script_status FAIL")
        sys.exit(1)

    stats = {
        "provenance": {
            "split_file": str(TRAIN_SPLIT_FILE),
            "sequences_in_split": len(train_ids),
            "sequences_completed": len(completed),
            "frames_processed": frames_processed,
            "frames_semseg_missing": frames_semseg_missing,
            "frames_failed": frames_failed,
            "skip_rate": skip_rate,
            "intensity_samples_total": int(intensity_all.shape[0]),
            "per_sequence_sample_cap": PER_SEQUENCE_SAMPLE_CAP,
            "class_weight_runtime_note": "Local Open3D SemSegLoss transforms dataset.cfg.class_weights internally; candidate direct-weight vectors are recorded separately in the class_weights section."
        },
        "intensity": {
            "clip_low_p0p5": p_low,
            "clip_high_p99p5": p_high,
            "clip_mean": clip_mean,
            "clip_std": clip_std,
        },
        "class_counts": {
            "ignore": int(class_counts.get(0, class_counts.get("0", 0))),
            "road": int(class_counts.get(1, class_counts.get("1", 0))),
            "lane": lane_count,
            "other": int(class_counts.get(3, class_counts.get("3", 0))),
        },
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"intensity_clip_low {p_low:.6f}")
    print(f"intensity_clip_high {p_high:.6f}")
    print(f"intensity_clip_mean {clip_mean:.6f}")
    print(f"intensity_clip_std {clip_std:.6f}")
    print(f"lane_points {lane_count}")
    print("statistics_complete True")
    print("script_status PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()

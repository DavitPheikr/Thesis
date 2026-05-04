"""
Freeze the train/val/test split from the audited semseg-enabled sequence pool.

Rules:
  - val = 9 sequences, test = 9 sequences, train = remainder
  - both val and test must contain >= 1 lane-bearing sequence
  - split is deterministic given the seed and manifest
  - provenance is recorded in the split report

Emits script_status PASS/FAIL as final line.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path


SEED = 42
VAL_SIZE = 9
TEST_SIZE = 9
MIN_LANE_PER_HOLDOUT = 1

MANIFEST_FILE = Path("logs/milestone_b_sequence_manifest.json")
SPLIT_DIR = Path("configs/splits")
REPORT_FILE = Path("logs/milestone_b_split_report.txt")


def main() -> None:
    manifest = json.loads(MANIFEST_FILE.read_text())
    manifest_hash = hashlib.md5(MANIFEST_FILE.read_bytes()).hexdigest()

    pool = sorted(manifest["semseg_enabled_ids"])
    lane_ids = set(manifest["lane_bearing_ids"])
    pool_size = len(pool)
    lane_seqs = [seq_id for seq_id in pool if seq_id in lane_ids]

    assert pool_size >= VAL_SIZE + TEST_SIZE + 1, (
        f"Pool size {pool_size} too small for {VAL_SIZE}+{TEST_SIZE}+1 split"
    )
    assert len(lane_seqs) >= 2, (
        f"Need >= 2 lane-bearing sequences for holdout guarantee; got {len(lane_seqs)}"
    )

    rng = random.Random(SEED)

    lane_shuffled = lane_seqs.copy()
    rng.shuffle(lane_shuffled)
    lane_for_val = [lane_shuffled[0]]
    lane_for_test = [lane_shuffled[1]]
    reserved = set(lane_for_val + lane_for_test)

    remaining = [seq_id for seq_id in pool if seq_id not in reserved]
    rng.shuffle(remaining)

    val_fill = remaining[: VAL_SIZE - 1]
    test_fill = remaining[VAL_SIZE - 1 : VAL_SIZE - 1 + TEST_SIZE - 1]
    train = remaining[VAL_SIZE - 1 + TEST_SIZE - 1 :]

    val = sorted(lane_for_val + val_fill)
    test = sorted(lane_for_test + test_fill)
    train = sorted(train)

    assert len(val) == VAL_SIZE, f"val has {len(val)} != {VAL_SIZE}"
    assert len(test) == TEST_SIZE, f"test has {len(test)} != {TEST_SIZE}"
    assert not (set(val) & set(test)), "val/test overlap"
    assert not (set(val) & set(train)), "val/train overlap"
    assert not (set(test) & set(train)), "test/train overlap"
    assert set(val) | set(test) | set(train) == set(pool), "split does not cover pool"

    val_lane = [seq_id for seq_id in val if seq_id in lane_ids]
    test_lane = [seq_id for seq_id in test if seq_id in lane_ids]
    assert len(val_lane) >= MIN_LANE_PER_HOLDOUT, (
        f"val has {len(val_lane)} lane-bearing sequences, need >= {MIN_LANE_PER_HOLDOUT}"
    )
    assert len(test_lane) >= MIN_LANE_PER_HOLDOUT, (
        f"test has {len(test_lane)} lane-bearing sequences, need >= {MIN_LANE_PER_HOLDOUT}"
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "train.txt").write_text("\n".join(train) + "\n")
    (SPLIT_DIR / "val.txt").write_text("\n".join(val) + "\n")
    (SPLIT_DIR / "test.txt").write_text("\n".join(test) + "\n")

    report_lines = [
        f"seed {SEED}",
        f"manifest_file {MANIFEST_FILE}",
        f"manifest_hash_md5 {manifest_hash}",
        f"pool_size {pool_size}",
        f"train_count {len(train)}",
        f"val_count {len(val)}",
        f"test_count {len(test)}",
        f"train_ids {train}",
        f"val_ids {val}",
        f"test_ids {test}",
        f"val_lane_ids {val_lane}",
        f"test_lane_ids {test_lane}",
        f"lane_per_holdout_val {len(val_lane)}",
        f"lane_per_holdout_test {len(test_lane)}",
        "lane_guaranteed_in_val True",
        "lane_guaranteed_in_test True",
        "split_status FROZEN",
    ]
    REPORT_FILE.write_text("\n".join(report_lines) + "\n")

    print("\n".join(report_lines))
    print("script_status PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()

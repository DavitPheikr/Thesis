"""
Derive class-weight artifacts from measured class counts.

This script records two Milestone B candidate direct weight vectors for thesis
analysis, while also recording the local Open3D runtime semantics discovered in
Day 3: dataset.cfg.class_weights is transformed internally by
DataProcessing.get_class_weights before CrossEntropyLoss is built.

Emits script_status PASS/FAIL as final line.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from thesis_pipeline.adapters.pandaset_ff_lane3 import build_one_sample

from open3d._ml3d.datasets.utils import DataProcessing


STATS_FILE = Path("logs/milestone_b_training_statistics.json")
INTERFACE_FILE = Path("logs/milestone_b_loss_interface.json")
LARGE_WEIGHT_THRESHOLD = 50.0
GRID_SIZE = 0.04
NUM_POINTS_INTENT = 16384


def normalize_by_min(weights_by_class: dict[str, float]) -> dict[str, float]:
    weight_min = min(weights_by_class.values())
    return {key: value / weight_min for key, value in weights_by_class.items()}


def main() -> None:
    stats = json.loads(STATS_FILE.read_text())
    interface = json.loads(INTERFACE_FILE.read_text())

    counts = stats["class_counts"]
    c_road = counts["road"]
    c_lane = counts["lane"]
    c_other = counts["other"]

    if c_lane <= 0:
        print("FAIL: lane class has zero points; cannot derive weights")
        print("script_status FAIL")
        sys.exit(1)

    active_total = c_road + c_lane + c_other
    k_classes = 3

    raw_direct = normalize_by_min({
        "road": active_total / (k_classes * c_road),
        "lane": active_total / (k_classes * c_lane),
        "other": active_total / (k_classes * c_other),
    })
    sqrt_direct = normalize_by_min({
        "road": math.sqrt(active_total / (k_classes * c_road)),
        "lane": math.sqrt(active_total / (k_classes * c_lane)),
        "other": math.sqrt(active_total / (k_classes * c_other)),
    })

    raw_direct_list = [raw_direct["road"], raw_direct["lane"], raw_direct["other"]]
    sqrt_direct_list = [sqrt_direct["road"], sqrt_direct["lane"], sqrt_direct["other"]]

    raw_lane_weight = raw_direct["lane"]
    sqrt_lane_weight = sqrt_direct["lane"]
    candidate_recommended_variant = (
        "sqrt_inverse_frequency" if raw_lane_weight > LARGE_WEIGHT_THRESHOLD else "raw_inverse_frequency"
    )
    candidate_recommended_list = (
        sqrt_direct_list if candidate_recommended_variant == "sqrt_inverse_frequency" else raw_direct_list
    )

    runtime_input_by_class = {
        "road": float(c_road),
        "lane": float(c_lane),
        "other": float(c_other),
    }
    runtime_input_as_list = [
        runtime_input_by_class["road"],
        runtime_input_by_class["lane"],
        runtime_input_by_class["other"],
    ]
    runtime_effective_ce_weights = DataProcessing.get_class_weights(runtime_input_as_list).astype(
        np.float32
    ).tolist()

    sample = build_one_sample()
    sub_points, sub_feat, sub_label = DataProcessing.grid_subsampling(
        sample["point"],
        features=sample["feat"],
        labels=sample["label"],
        grid_size=GRID_SIZE,
    )
    post_grid_point_count = int(sub_points.shape[0])
    raw_point_count = int(sample["point"].shape[0])
    duplication_will_occur = post_grid_point_count < NUM_POINTS_INTENT

    stats["class_weights"] = {
        "deviation_note": "Approved Milestone B local-runtime deviation: candidate direct CE weight vectors are recorded for thesis analysis, but local Open3D SemSegLoss transforms dataset.cfg.class_weights internally via DataProcessing.get_class_weights.",
        "candidate_weight_semantics": "direct final CrossEntropyLoss weights for analysis only; not consumed directly by the local Open3D runtime",
        "raw_inverse_frequency": {
            "by_class": raw_direct,
            "as_list": raw_direct_list,
            "lane_weight": raw_lane_weight,
        },
        "sqrt_inverse_frequency": {
            "by_class": sqrt_direct,
            "as_list": sqrt_direct_list,
            "lane_weight": sqrt_lane_weight,
        },
        "candidate_recommended_variant": candidate_recommended_variant,
        "candidate_recommended_list": candidate_recommended_list,
        "runtime_open3d_native_transform": {
            "input_semantics": "per-class counts or count-like values passed into DataProcessing.get_class_weights",
            "runtime_input_source": "measured active-class counts from class_counts",
            "runtime_input_by_class": runtime_input_by_class,
            "runtime_input_as_list": runtime_input_as_list,
            "runtime_effective_ce_weights": runtime_effective_ce_weights,
        },
        "weight_format_source": str(INTERFACE_FILE),
        "weight_scope_discovered": interface.get("weight_scope", "UNKNOWN"),
        "weight_list_length_discovered": interface.get("weight_list_length", "UNKNOWN"),
        "sanity_run_recommended_variant": "open3d_native_from_measured_counts",
        "sanity_run_recommended_list": runtime_input_as_list,
        "large_weight_warning": raw_lane_weight > LARGE_WEIGHT_THRESHOLD,
    }

    stats["grid_subsampling"] = {
        "grid_size": GRID_SIZE,
        "source_sequence_id": sample["meta"].sequence_id,
        "source_frame_idx": sample["meta"].frame_idx,
        "raw_point_count": raw_point_count,
        "post_grid_point_count": post_grid_point_count,
        "below_num_points_intent": duplication_will_occur,
        "duplication_will_occur": duplication_will_occur,
        "num_points_intent": NUM_POINTS_INTENT,
    }

    STATS_FILE.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"raw_lane_weight {raw_lane_weight:.6f}")
    print(f"sqrt_lane_weight {sqrt_lane_weight:.6f}")
    print(f"candidate_recommended_variant {candidate_recommended_variant}")
    print(f"runtime_sanity_variant open3d_native_from_measured_counts")
    print(f"runtime_sanity_list {runtime_input_as_list}")
    print(f"runtime_effective_ce_weights {runtime_effective_ce_weights}")
    print(f"post_grid_point_count {post_grid_point_count}")
    print(f"duplication_will_occur {duplication_will_occur}")
    print("script_status PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()

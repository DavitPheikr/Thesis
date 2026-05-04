"""
Verify the PandaSetFFLane3 dataset class and config wiring.

Checks:
  - YAML class_weights matches statistics JSON sanity_run_recommended_list
  - build_one_sample() backward compatibility lane fraction remains near 0.0026
  - train/validation/test split objects instantiate and return valid samples

Emits script_status PASS/FAIL as final line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.pandaset_ff_lane3 import PandaSetFFLane3Dataset, build_one_sample


STATS_FILE = Path("logs/milestone_b_training_statistics.json")
CFG_FILE = Path("configs/randlanet_pandaset_ff_lane3.yml")
REPORT_FILE = Path("logs/milestone_b_dataset_class_report.txt")

EXPECTED_LANE_FRACTION = 0.0026
LANE_FRACTION_TOLERANCE = 0.10


def main() -> None:
    lines: list[str] = []
    all_ok = True

    stats = json.loads(STATS_FILE.read_text())
    cfg = yaml.safe_load(CFG_FILE.read_text())
    dataset_cfg = cfg.get("dataset", {})

    json_weights = stats["class_weights"]["sanity_run_recommended_list"]
    yaml_weights = dataset_cfg.get("class_weights", [])
    weights_match = list(yaml_weights) == list(json_weights)
    lines.append(f"yaml_json_weights_consistent {weights_match}")
    if not weights_match:
        lines.append(f"FAIL yaml_weights {yaml_weights}")
        lines.append(f"FAIL json_weights {json_weights}")
        all_ok = False

    intensity_checks = {
        "intensity_clip_low": (
            dataset_cfg.get("intensity_clip_low"),
            stats["intensity"]["clip_low_p0p5"],
        ),
        "intensity_clip_high": (
            dataset_cfg.get("intensity_clip_high"),
            stats["intensity"]["clip_high_p99p5"],
        ),
        "intensity_mean": (
            dataset_cfg.get("intensity_mean"),
            stats["intensity"]["clip_mean"],
        ),
        "intensity_std": (
            dataset_cfg.get("intensity_std"),
            stats["intensity"]["clip_std"],
        ),
    }
    intensity_ok = True
    for field_name, (yaml_value, json_value) in intensity_checks.items():
        field_ok = float(yaml_value) == float(json_value)
        lines.append(f"{field_name}_consistent {field_ok}")
        if not field_ok:
            lines.append(f"FAIL {field_name}_yaml {yaml_value}")
            lines.append(f"FAIL {field_name}_json {json_value}")
            intensity_ok = False
            all_ok = False
    lines.append(f"yaml_json_intensity_consistent {intensity_ok}")

    sample = build_one_sample()
    label = sample["label"]
    lane_fraction = float((label == 2).sum()) / len(label)
    rel_diff = abs(lane_fraction - EXPECTED_LANE_FRACTION) / EXPECTED_LANE_FRACTION
    compat_ok = rel_diff <= LANE_FRACTION_TOLERANCE
    lines.append(f"backward_compat_lane_fraction {lane_fraction:.9f}")
    lines.append(f"backward_compat_ok {compat_ok}")
    if not compat_ok:
        lines.append(
            f"FAIL lane_fraction_relative_diff {rel_diff:.6f} "
            f"threshold {LANE_FRACTION_TOLERANCE:.6f}"
        )
        all_ok = False

    dataset = PandaSetFFLane3Dataset(**cfg["dataset"])
    split_name_map = {
        "train": "training",
        "val": "validation",
        "test": "test",
    }

    for short_name, split_name in split_name_map.items():
        split_obj = dataset.get_split(split_name)
        split_ok = True
        try:
            for idx in range(min(3, len(split_obj))):
                data = split_obj.get_data(idx)
                point = data["point"]
                feat = data["feat"]
                label = data["label"]

                if point.ndim != 2 or point.shape[1] != 3:
                    raise RuntimeError(f"unexpected point shape {point.shape}")
                if feat.ndim != 2 or feat.shape[1] != 1:
                    raise RuntimeError(f"unexpected feat shape {feat.shape}")
                if label.ndim != 1 or label.shape[0] != point.shape[0]:
                    raise RuntimeError(
                        f"unexpected label shape {label.shape} for point shape {point.shape}"
                    )
                if label.dtype not in (np.int32, np.int64):
                    raise RuntimeError(f"unexpected label dtype {label.dtype}")
                unique = np.unique(label).tolist()
                if not set(unique).issubset({0, 1, 2, 3}):
                    raise RuntimeError(f"unexpected labels {unique}")
            lines.append(f"split_{short_name}_ok True")
            lines.append(f"split_{short_name}_length {len(split_obj)}")
        except Exception as exc:
            split_ok = False
            lines.append(f"split_{short_name}_ok False")
            lines.append(f"FAIL split_{short_name}_error {exc}")
            all_ok = False

    REPORT_FILE.write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    status = "PASS" if all_ok else "FAIL"
    print(f"script_status {status}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
from pandaset import DataSet, geometry as pds_geometry

from open3d._ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit

from thesis_pipeline.adapters.pandaset_ff_lane3 import (
    LABEL_MODE_LANE3,
    LABEL_MODE_ROAD_MARKING3,
    remap_raw_pandaset_ids,
    validate_label_mode,
)
from thesis_pipeline.core.pandaset_compat import get_frame_count


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATS_FILE = PROJECT_ROOT / "logs/milestone_b_training_statistics.json"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "configs/splits"
DEFAULT_DATASET_ROOT_FILE = PROJECT_ROOT / "logs/dataset_root.txt"
DEFAULT_PREFLIGHT_PATTERN_FILE = PROJECT_ROOT / "logs/milestone_b_preflight_sensor_pattern.txt"
DEFAULT_MANIFEST_FILE = PROJECT_ROOT / "logs/milestone_b_sequence_manifest.json"


def _read_text_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_stats(stats_file: Path) -> dict:
    return json.loads(stats_file.read_text())


def _sensor_reapply_required(preflight_pattern_file: Path) -> bool:
    if not preflight_pattern_file.exists():
        return False
    return "reapply_per_frame: true" in preflight_pattern_file.read_text()


class PandaSetFFLane3Dataset(BaseDataset):
    def __init__(
        self,
        dataset_path: str | None = None,
        name: str = "PandaSetFFLane3",
        cache_dir: str = "./logs/cache",
        use_cache: bool = False,
        class_weights: list[float] | None = None,
        ignored_label_inds: list[int] | None = None,
        test_result_folder: str = "./logs/test_results",
        sampler: dict | None = None,
        stats_file: str | None = None,
        split_dir: str | None = None,
        dataset_root_file: str | None = None,
        preflight_pattern_file: str | None = None,
        manifest_file: str | None = None,
        label_mode: str = LABEL_MODE_LANE3,
        steps_per_epoch_train: int | None = None,
        steps_per_epoch_valid: int | None = None,
        **kwargs,
    ):
        self.label_mode = validate_label_mode(label_mode)
        self.stats_file = Path(stats_file) if stats_file else DEFAULT_STATS_FILE
        self.split_dir = Path(split_dir) if split_dir else DEFAULT_SPLIT_DIR
        dataset_root_path = (
            Path(dataset_root_file) if dataset_root_file else DEFAULT_DATASET_ROOT_FILE
        )
        self.preflight_pattern_file = (
            Path(preflight_pattern_file)
            if preflight_pattern_file
            else DEFAULT_PREFLIGHT_PATTERN_FILE
        )
        manifest_path = Path(manifest_file) if manifest_file else DEFAULT_MANIFEST_FILE

        if dataset_path is None:
            dataset_path = dataset_root_path.read_text().strip()

        self._stats = _load_stats(self.stats_file)
        self._sensor_reapply = _sensor_reapply_required(self.preflight_pattern_file)
        self._dataset_path = dataset_path
        self._data = DataSet(dataset_path)

        # Load manifest-based frame counts to avoid calling load_lidar() during index
        # construction.  The audit script already measured lidar_frame_count for all
        # semseg-enabled sequences using the authoritative get_frame_count() helper.
        self._frame_counts: dict[str, int] = {}
        if manifest_path.exists():
            _manifest = json.loads(manifest_path.read_text())
            for seq_id, details in _manifest.get("per_sequence_details", {}).items():
                self._frame_counts[seq_id] = int(details["lidar_frame_count"])

        self._split_ids = {
            "training": _read_text_lines(self.split_dir / "train.txt"),
            "validation": _read_text_lines(self.split_dir / "val.txt"),
            "test": _read_text_lines(self.split_dir / "test.txt"),
        }
        self._split_ids["all"] = (
            self._split_ids["training"]
            + self._split_ids["validation"]
            + self._split_ids["test"]
        )

        if class_weights is None:
            class_weights = self._stats["class_weights"]["sanity_run_recommended_list"]
        if ignored_label_inds is None:
            ignored_label_inds = [0]
        if sampler is None:
            sampler = {"name": "SemSegSpatiallyRegularSampler"}

        super().__init__(
            dataset_path=dataset_path,
            name=name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            class_weights=class_weights,
            ignored_label_inds=ignored_label_inds,
            test_result_folder=test_result_folder,
            training_split=self._split_ids["training"],
            validation_split=self._split_ids["validation"],
            test_split=self._split_ids["test"],
            all_split=self._split_ids["all"],
            sampler=sampler,
            label_mode=self.label_mode,
            steps_per_epoch_train=steps_per_epoch_train,
            steps_per_epoch_valid=steps_per_epoch_valid,
            **kwargs,
        )

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k in self.label_to_names.keys()])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_values)}
        self.ignored_labels = np.array(self.cfg.ignored_label_inds)

        intensity_stats = self._stats["intensity"]
        self.intensity_clip_low = float(intensity_stats["clip_low_p0p5"])
        self.intensity_clip_high = float(intensity_stats["clip_high_p99p5"])
        self.intensity_mean = float(intensity_stats["clip_mean"])
        self.intensity_std = float(intensity_stats["clip_std"])

        self._frame_index_by_split: dict[str, list[tuple[str, int]]] = {}

    @staticmethod
    def get_label_to_names():
        return {
            0: "ignore",
            1: "road",
            2: "lane",
            3: "other",
        }

    def get_positive_class_name(self) -> str:
        if self.label_mode == LABEL_MODE_ROAD_MARKING3:
            return "marking"
        return "lane"

    def _build_frame_index_for_split(self, split_name: str) -> list[tuple[str, int]]:
        if split_name == "all":
            all_entries: list[tuple[str, int]] = []
            for member_split in ("training", "validation", "test"):
                all_entries.extend(self._get_frame_index_for_split(member_split))
            return all_entries

        entries: list[tuple[str, int]] = []
        for seq_id in self._split_ids[split_name]:
            if seq_id in self._frame_counts:
                # Fast path: use prebuilt frame count from the sequence manifest.
                # Avoids load_lidar() for index construction; manifest values were
                # computed by the audit script with the authoritative get_frame_count().
                frame_count = self._frame_counts[seq_id]
            else:
                # Fallback: load lidar to count frames (defensive; should not trigger
                # for any sequence that passed the Day 1 audit).
                seq = self._data[seq_id]
                seq.load_lidar()
                frame_count = get_frame_count(seq)
            entries.extend((seq_id, frame_idx) for frame_idx in range(frame_count))
        return entries

    def _get_frame_index_for_split(self, split_name: str) -> list[tuple[str, int]]:
        if split_name not in self._frame_index_by_split:
            self._frame_index_by_split[split_name] = self._build_frame_index_for_split(
                split_name
            )
        return self._frame_index_by_split[split_name]

    def _normalize_split_name(self, split: str) -> str:
        if split in ("train", "training"):
            return "training"
        if split in ("val", "validation", "valid"):
            return "validation"
        if split in ("test", "testing"):
            return "test"
        if split == "all":
            return "all"
        raise ValueError(f"Invalid split {split}")

    def get_split(self, split):
        return PandaSetFFLane3Split(self, split=self._normalize_split_name(split))

    def get_split_list(self, split):
        return list(self._get_frame_index_for_split(self._normalize_split_name(split)))

    def is_tested(self, attr):
        result_path = Path(self.cfg.test_result_folder) / f"{attr['name']}.npy"
        return result_path.exists()

    def save_test_result(self, results, attr):
        result_dir = Path(self.cfg.test_result_folder)
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / f"{attr['name']}.npy", results["predict_labels"])

    def _load_sample(self, seq_id: str, frame_idx: int) -> dict:
        seq = self._data[seq_id]
        try:
            seq.load_lidar().load_semseg()
        except Exception:
            seq.load_lidar()
            seq.load_semseg()

        if seq.semseg is None:
            raise RuntimeError(f"Sequence {seq_id} has no semseg available")

        seq.lidar.set_sensor(1)
        if self._sensor_reapply:
            seq.lidar.set_sensor(1)
        pc_df = seq.lidar[frame_idx]
        semseg_df = seq.semseg[frame_idx]

        raw_labels = semseg_df.loc[pc_df.index, "class"].to_numpy(dtype=np.int32)
        labels = remap_raw_pandaset_ids(raw_labels, label_mode=self.label_mode)

        xyz_world = pc_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        pose = seq.lidar.poses[frame_idx]
        xyz_ego = pds_geometry.lidar_points_to_ego(xyz_world, pose).astype(
            np.float32, copy=False
        )

        intensity = pc_df["i"].to_numpy(dtype=np.float32)
        intensity = np.clip(intensity, self.intensity_clip_low, self.intensity_clip_high)
        intensity = (intensity - self.intensity_mean) / self.intensity_std
        feat = intensity[:, None].astype(np.float32, copy=False)

        sample = {
            "point": xyz_ego,
            "feat": feat,
            "label": labels.astype(np.int32, copy=False),
        }

        # Release full-sequence data loaded by the PandaSet devkit. DataSet
        # returns the same cached Sequence object on subsequent access, so we
        # clear the loaded frame lists after extracting the single-frame numpy
        # arrays needed for this sample.
        seq.lidar._data = None
        seq.lidar._poses = None
        seq.lidar._timestamps = None
        seq.semseg._data = None
        gc.collect()

        return sample


class PandaSetFFLane3Split(BaseDatasetSplit):
    def __init__(self, dataset: PandaSetFFLane3Dataset, split: str = "training"):
        super().__init__(dataset, split=split)

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        seq_id, frame_idx = self.path_list[idx]
        return self.dataset._load_sample(seq_id, frame_idx)

    def get_attr(self, idx):
        seq_id, frame_idx = self.path_list[idx]
        return {
            "name": f"{seq_id}_{frame_idx:03d}",
            "seq_id": seq_id,
            "frame_idx": frame_idx,
            "path": f"{seq_id}/lidar/{frame_idx}",
            "split": self.split,
        }

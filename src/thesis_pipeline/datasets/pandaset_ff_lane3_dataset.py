from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pandaset import DataSet, geometry as pds_geometry
from PIL import Image

from open3d._ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit

from thesis_pipeline.adapters.pandaset_ff_lane3 import (
    CAMERA_LOOKUP_NEAREST_TIMESTAMP,
    COLOR_SAMPLING_BILINEAR,
    COLOR_SAMPLING_NEAREST,
    FEATURE_MODE_INTENSITY,
    FEATURE_MODE_INTENSITY_RGB_FRONT,
    LABEL_MODE_LANE3,
    LABEL_MODE_ROAD_MARKING3,
    bilinear_sample_rgb,
    nearest_sample_rgb,
    project_points_to_camera,
    remap_raw_pandaset_ids,
    validate_camera_lookup,
    validate_color_sampling,
    validate_feature_mode,
    validate_label_mode,
    validate_motion_compensation,
    validate_rgb_normalization,
)
from thesis_pipeline.core.pandaset_compat import get_frame_count


CACHE_MANIFEST_FILENAME = "cache_manifest.json"
CACHE_MANIFEST_SCHEMA_VERSION = 1


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
        feature_mode: str = FEATURE_MODE_INTENSITY,
        camera_name: str = "front_camera",
        camera_lookup: str = CAMERA_LOOKUP_NEAREST_TIMESTAMP,
        rgb_max_dt_s: float = 0.060,
        color_sampling: str = COLOR_SAMPLING_BILINEAR,
        rgb_normalization: str = "divide_by_255",
        motion_compensation: str = "none",
        cache_grid_size: float | None = None,
        steps_per_epoch_train: int | None = None,
        steps_per_epoch_valid: int | None = None,
        **kwargs,
    ):
        self.label_mode = validate_label_mode(label_mode)
        self.feature_mode = validate_feature_mode(feature_mode)
        self.camera_name = str(camera_name)
        self.camera_lookup = validate_camera_lookup(camera_lookup)
        self.rgb_max_dt_s = float(rgb_max_dt_s)
        self.color_sampling = validate_color_sampling(color_sampling)
        self.rgb_normalization = validate_rgb_normalization(str(rgb_normalization))
        self.motion_compensation = validate_motion_compensation(str(motion_compensation))
        self.cache_grid_size = None if cache_grid_size is None else float(cache_grid_size)
        # Per-sequence camera metadata cache (JSON sidecars only; never images).
        self._camera_meta_by_seq: dict[str, dict] = {}
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
            feature_mode=self.feature_mode,
            camera_name=self.camera_name,
            camera_lookup=self.camera_lookup,
            rgb_max_dt_s=self.rgb_max_dt_s,
            color_sampling=self.color_sampling,
            rgb_normalization=self.rgb_normalization,
            motion_compensation=self.motion_compensation,
            cache_grid_size=self.cache_grid_size,
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

        # Cache manifest: only meaningful when use_cache is on AND the new
        # feature mode is in use. D0 (intensity-only) configs keep their
        # existing cache semantics unchanged.
        if use_cache and self.feature_mode != FEATURE_MODE_INTENSITY:
            self._validate_or_write_cache_manifest(Path(cache_dir))

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

    def _build_cache_manifest(self) -> dict:
        return {
            "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
            "label_mode": self.label_mode,
            "feature_mode": self.feature_mode,
            "camera_name": self.camera_name,
            "camera_lookup": self.camera_lookup,
            "rgb_max_dt_s": float(self.rgb_max_dt_s),
            "color_sampling": self.color_sampling,
            "rgb_normalization": self.rgb_normalization,
            "motion_compensation": self.motion_compensation,
            "rgb_invalid_fill": [0.0, 0.0, 0.0],
            "rgb_valid_policy": "zero_with_valid_flag",
            "rgb_valid_aggregation": "passthrough_fractional",
            "intensity_clip_low": float(self.intensity_clip_low),
            "intensity_clip_high": float(self.intensity_clip_high),
            "intensity_mean": float(self.intensity_mean),
            "intensity_std": float(self.intensity_std),
            "cache_grid_size": self.cache_grid_size,
            "forward_sensor_id": 1,
        }

    def _validate_or_write_cache_manifest(self, cache_dir: Path) -> None:
        manifest_path = cache_dir / CACHE_MANIFEST_FILENAME
        current = self._build_cache_manifest()

        if cache_dir.exists() and any(cache_dir.iterdir()):
            # Cache dir is non-empty: there must be a manifest, and it
            # must agree with the active config field-for-field.
            if not manifest_path.exists():
                raise RuntimeError(
                    "Refusing to use cache_dir without a cache_manifest.json: "
                    f"{cache_dir}. This looks like a foreign cache (e.g. a D0 "
                    "intensity-only cache or an earlier E0 build). Point "
                    "cache_dir at a new versioned path."
                )
            existing = json.loads(manifest_path.read_text())
            diff = {
                k: (existing.get(k), current[k])
                for k in current
                if existing.get(k) != current[k]
            }
            if diff:
                lines = "\n".join(
                    f"  {k}: cache={v[0]!r} active={v[1]!r}" for k, v in diff.items()
                )
                raise RuntimeError(
                    "Cache manifest mismatch -- refusing to reuse stale cache.\n"
                    f"cache_dir: {cache_dir}\n"
                    f"manifest:  {manifest_path}\n"
                    f"diff:\n{lines}\n"
                    "Bump the cache version (e.g. _v1 -> _v2) and rebuild."
                )
            return

        # Fresh cache: create dir and write the manifest now so any later
        # process attaching to this cache sees the contract.
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(current)
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _get_camera_metadata(self, seq_id: str) -> dict:
        """Lazy per-sequence load of camera intrinsics / poses / timestamps.

        Reads JSON sidecars directly. Never calls Camera.load() (that would
        decode all 80 images into memory).
        """
        if seq_id in self._camera_meta_by_seq:
            return self._camera_meta_by_seq[seq_id]
        cam_dir = Path(self._dataset_path) / seq_id / "camera" / self.camera_name
        intrinsics = json.loads((cam_dir / "intrinsics.json").read_text())
        cam_poses = json.loads((cam_dir / "poses.json").read_text())
        cam_ts = json.loads((cam_dir / "timestamps.json").read_text())
        lid_ts = json.loads(
            (Path(self._dataset_path) / seq_id / "lidar" / "timestamps.json").read_text()
        )
        meta = {
            "cam_dir": cam_dir,
            "intrinsics": intrinsics,
            "cam_poses": cam_poses,
            "cam_ts": np.asarray(cam_ts, dtype=np.float64),
            "lid_ts": np.asarray(lid_ts, dtype=np.float64),
        }
        self._camera_meta_by_seq[seq_id] = meta
        return meta

    def _compute_rgb_features(
        self,
        seq_id: str,
        frame_idx: int,
        xyz_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (rgb (N, 3) float32 in [0, 1], rgb_valid (N,) float32 in {0, 1}).

        Applies the step-4 timestamp policy. If the chosen camera frame is
        outside the configured dt window, all points get rgb=0, rgb_valid=0
        and no image is loaded.
        """
        meta = self._get_camera_metadata(seq_id)
        n = xyz_world.shape[0]

        target_t = float(meta["lid_ts"][frame_idx])
        diffs = np.abs(meta["cam_ts"] - target_t)
        cam_idx = int(np.argmin(diffs))
        dt = float(meta["cam_ts"][cam_idx] - target_t)

        rgb = np.zeros((n, 3), dtype=np.float32)
        rgb_valid = np.zeros(n, dtype=np.float32)
        if abs(dt) > self.rgb_max_dt_s:
            return rgb, rgb_valid

        # Load just the one image we need.
        img_path = meta["cam_dir"] / f"{cam_idx:02d}.jpg"
        with Image.open(img_path) as img_lazy:
            img = img_lazy.convert("RGB")
            image_array = np.asarray(img, dtype=np.uint8)
            image_w, image_h = img.size

        uv, _depth, in_img = project_points_to_camera(
            xyz_world.astype(np.float64, copy=False),
            meta["cam_poses"][cam_idx],
            meta["intrinsics"],
            image_w,
            image_h,
        )
        if self.color_sampling == COLOR_SAMPLING_BILINEAR:
            rgb = bilinear_sample_rgb(image_array, uv, in_img)
        else:
            rgb = nearest_sample_rgb(image_array, uv, in_img)
        rgb_valid[in_img] = 1.0
        # Drop image refs explicitly so subsequent frames don't pile up.
        del image_array
        return rgb, rgb_valid

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
        intensity_col = intensity[:, None].astype(np.float32, copy=False)

        if self.feature_mode == FEATURE_MODE_INTENSITY_RGB_FRONT:
            rgb, rgb_valid = self._compute_rgb_features(seq_id, frame_idx, xyz_world)
            # Layout: [intensity, r, g, b, rgb_valid], all float32, shape (N, 5).
            feat = np.concatenate(
                [intensity_col, rgb, rgb_valid[:, None]], axis=1
            ).astype(np.float32, copy=False)
        else:
            feat = intensity_col

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

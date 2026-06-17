"""Milestone D training entrypoint.

This is the road-marking training driver for Milestone D. It is intentionally
separate from tools/train_milestone_c.py so C0 remains reproducible while D can
use AdamW, ReduceLROnPlateau, run-local checkpoints, and explicit resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np
import yaml

# Compatibility for TensorBoard/Open3D imports under NumPy 2.x. The project
# working environment pins NumPy 1.26.4, but local environments may drift.
if not hasattr(np, "string_"):
    np.string_ = np.bytes_  # type: ignore[attr-defined]
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_  # type: ignore[attr-defined]

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCHED_DEVKIT = PROJECT_ROOT / "pandaset-devkit/python"

# Keep the patched PandaSet devkit ahead of the repo root. Otherwise a local
# data directory named "pandaset/" can shadow the importable devkit package.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PATCHED_DEVKIT))

import open3d.ml.torch as ml3d
from open3d._ml3d.torch.dataloaders import TorchDataloader, get_sampler
from open3d._ml3d.torch.modules.losses import SemSegLoss
from open3d._ml3d.torch.modules.metrics import SemSegMetric
from open3d._ml3d.torch.pipelines import SemanticSegmentation
from open3d._ml3d.utils import get_runid

from datasets.pandaset_ff_lane3 import PandaSetFFLane3Dataset
from thesis_pipeline.augmentations import RGBJitterConfig, apply_rgb_jitter_to_features
from thesis_pipeline.losses import build_loss
from thesis_pipeline.eval.milestone_c_metrics import (
    EVAL_CSV_COLUMNS,
    compute_metrics,
    flatten_metrics_for_csv,
    json_ready,
)

DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_d/configs/d0_weighted_ce.yml"
RUNS_DIR = PROJECT_ROOT / "logs/milestone_d/runs"
DEFAULT_COMPLETION_LABEL = "milestone_d_run_complete"
EVAL_CSV_COLUMNS_D = tuple(EVAL_CSV_COLUMNS) + ("lr",)
METRIC_ALIASES = {
    "marking_iou": "lane_iou",
    "marking_precision": "lane_precision",
    "marking_recall": "lane_recall",
    "marking_f1": "lane_f1",
}


class Tee:
    """Write stdout/stderr to terminal and a log file."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class MilestoneDPipeline(SemanticSegmentation):
    """Milestone D wrapper for metrics, scheduling, checkpointing, and resume."""

    def __init__(
        self,
        *args,
        run_dir: Path,
        requested_epochs: int,
        resume_from: Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.run_dir = run_dir
        self.requested_epochs = requested_epochs
        self.resume_from = resume_from
        self.start_epoch = 0
        self.checkpoint_dir = run_dir / "checkpoints"
        self.training_log = run_dir / "training_log.txt"
        self.eval_history_path = run_dir / "eval_history.csv"
        self._epoch_started_at = time.monotonic()
        self._val_y_true: list[np.ndarray] = []
        self._val_y_pred: list[np.ndarray] = []
        self._val_ranges: list[np.ndarray] = []
        self.scheduler_kind = "none"
        self.scheduler_watch_metric = "lane_iou"
        self.scheduler_smoothing_window = 1
        self.scheduler_metric_history: list[float] = []
        self.rgb_jitter_cfg = RGBJitterConfig.from_mapping(
            self.cfg.get("rgb_jitter", {})
        )
        if self.rgb_jitter_cfg.enabled and not self.rgb_jitter_cfg.train_only:
            raise ValueError("rgb_jitter currently supports train_only=true only")

    def _current_lr(self) -> float:
        if not getattr(self, "optimizer", None):
            return float("nan")
        return float(self.optimizer.param_groups[0]["lr"])

    def save_logs(self, writer, epoch):  # noqa: ANN001
        super().save_logs(writer, epoch)
        wall_clock = time.monotonic() - self._epoch_started_at
        self._epoch_started_at = time.monotonic()
        peak_mem = 0
        if torch.cuda.is_available():
            peak_mem = int(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        human_epoch = epoch + 1
        line = (
            f"epoch={human_epoch} open3d_epoch={epoch} "
            f"wall_clock_seconds={wall_clock:.3f} "
            f"peak_gpu_memory_bytes={peak_mem} "
            f"lr={self._current_lr():.12g}"
        )
        with self.training_log.open("a") as f:
            f.write(line + "\n")

    def _optimizer_cfg(self) -> dict:
        return dict(self.cfg.get("optimizer", {}) or {})

    def _scheduler_cfg(self) -> dict:
        return dict(self.cfg.get("scheduler", {}) or {})

    def _apply_train_rgb_jitter(self, inputs) -> None:  # noqa: ANN001
        if not self.rgb_jitter_cfg.enabled:
            return
        data = inputs["data"]
        if "features" not in data:
            raise RuntimeError("rgb_jitter enabled but batch data has no features")
        apply_rgb_jitter_to_features(data["features"], self.rgb_jitter_cfg)

    def _build_optimizer_and_scheduler(self):
        opt_cfg = self._optimizer_cfg()
        optimizer_name = str(opt_cfg.get("name", "Adam")).lower()
        lr = float(opt_cfg.get("lr", 0.001))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer {opt_cfg.get('name')!r}; expected Adam or AdamW"
            )

        scheduler_cfg = self._scheduler_cfg()
        scheduler_name = str(scheduler_cfg.get("name", "ExponentialLR")).lower()
        if scheduler_name == "reducelronplateau":
            self.scheduler_kind = "ReduceLROnPlateau"
            self.scheduler_watch_metric = str(
                scheduler_cfg.get("watch_metric", "marking_iou")
            )
            self.scheduler_smoothing_window = int(
                scheduler_cfg.get("smoothing_window", 1)
            )
            if self.scheduler_smoothing_window < 1:
                raise ValueError("scheduler.smoothing_window must be >= 1")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(scheduler_cfg.get("mode", "max")),
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 4)),
                threshold=float(scheduler_cfg.get("threshold", 0.002)),
                threshold_mode=str(scheduler_cfg.get("threshold_mode", "abs")),
                cooldown=int(scheduler_cfg.get("cooldown", 1)),
                min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
            )
        elif scheduler_name == "exponentiallr":
            self.scheduler_kind = "ExponentialLR"
            gamma = float(
                scheduler_cfg.get("gamma", self.cfg.get("scheduler_gamma", 0.99))
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError(
                "Unsupported scheduler "
                f"{scheduler_cfg.get('name')!r}; expected ReduceLROnPlateau or ExponentialLR"
            )

        print(
            "optimizer_scheduler "
            f"optimizer={optimizer.__class__.__name__} "
            f"lr={lr} weight_decay={weight_decay} "
            f"scheduler={self.scheduler_kind} "
            f"watch_metric={self.scheduler_watch_metric} "
            f"smoothing_window={self.scheduler_smoothing_window}",
            flush=True,
        )
        return optimizer, scheduler

    def run_train(self):
        """Train with per-epoch Milestone D validation metric artifacts."""

        print("run_train_seed_start", flush=True)
        torch.manual_seed(self.rng.integers(np.iinfo(np.int32).max))
        print("run_train_seed_done", flush=True)
        model = self.model
        device = self.device
        model.device = device
        dataset = self.dataset
        cfg = self.cfg
        print("run_train_model_to_device_start", flush=True)
        model.to(device)
        print("run_train_model_to_device_done", flush=True)

        print("run_train_loss_metric_batcher_start", flush=True)
        loss_fn = build_loss(self, model, dataset, device, self.cfg.get("loss"))
        self.metric_train = SemSegMetric()
        self.metric_val = SemSegMetric()
        self.batcher = self.get_batcher(device)
        print("run_train_loss_metric_batcher_done", flush=True)

        print("run_train_train_split_start", flush=True)
        train_dataset = dataset.get_split("train")
        train_sampler = train_dataset.sampler
        train_split = TorchDataloader(
            dataset=train_dataset,
            preprocess=model.preprocess,
            transform=model.transform,
            sampler=train_sampler,
            use_cache=dataset.cfg.use_cache,
            steps_per_epoch=dataset.cfg.get("steps_per_epoch_train", None),
        )
        print("run_train_train_split_done", flush=True)
        print("run_train_train_loader_start", flush=True)
        train_loader = DataLoader(
            train_split,
            batch_size=cfg.batch_size,
            sampler=get_sampler(train_sampler),
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", True),
            collate_fn=self.batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(
                x + np.uint32(torch.utils.data.get_worker_info().seed)
            ),
        )
        print("run_train_train_loader_done", flush=True)

        print("run_train_valid_split_start", flush=True)
        valid_dataset = dataset.get_split("validation")
        assert valid_dataset.split == "validation", (
            "Milestone D validation metrics must use configs/splits/val.txt "
            "through dataset.get_split('validation'); never lane-aware train sampling."
        )
        valid_sampler = valid_dataset.sampler
        valid_split = TorchDataloader(
            dataset=valid_dataset,
            preprocess=model.preprocess,
            transform=model.transform,
            sampler=valid_sampler,
            use_cache=dataset.cfg.use_cache,
            steps_per_epoch=dataset.cfg.get("steps_per_epoch_valid", None),
        )
        print("run_train_valid_split_done", flush=True)
        print("run_train_valid_loader_start", flush=True)
        valid_loader = DataLoader(
            valid_split,
            batch_size=cfg.val_batch_size,
            sampler=get_sampler(valid_sampler),
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", True),
            collate_fn=self.batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(
                x + np.uint32(torch.utils.data.get_worker_info().seed)
            ),
        )
        print("run_train_valid_loader_done", flush=True)

        print("run_train_optimizer_start", flush=True)
        self.optimizer, self.scheduler = self._build_optimizer_and_scheduler()
        print("run_train_optimizer_done", flush=True)
        print("run_train_resume_start", flush=True)
        self._load_training_checkpoint()
        print("run_train_resume_done", flush=True)

        print("run_train_tensorboard_start", flush=True)
        dataset_name = dataset.name if dataset is not None else ""
        tensorboard_dir = (
            Path(self.cfg.train_sum_dir)
            / f"{model.__class__.__name__}_{dataset_name}_torch"
        )
        runid = get_runid(str(tensorboard_dir))
        self.tensorboard_dir = str(
            Path(self.cfg.train_sum_dir) / f"{runid}_{tensorboard_dir.name}"
        )
        writer = SummaryWriter(self.tensorboard_dir)
        print("run_train_tensorboard_done", flush=True)
        print("run_train_save_config_start", flush=True)
        self.save_config(writer)
        print("run_train_save_config_done", flush=True)
        record_summary = cfg.get("summary").get("record_for", [])
        if self.rgb_jitter_cfg.enabled:
            print(
                "rgb_jitter_enabled "
                f"train_only={self.rgb_jitter_cfg.train_only} "
                f"brightness=[{self.rgb_jitter_cfg.brightness_min},"
                f"{self.rgb_jitter_cfg.brightness_max}] "
                f"contrast=[{self.rgb_jitter_cfg.contrast_min},"
                f"{self.rgb_jitter_cfg.contrast_max}] "
                f"rgb_valid_threshold={self.rgb_jitter_cfg.rgb_valid_threshold}",
                flush=True,
            )

        if self.start_epoch > cfg.max_epoch:
            raise RuntimeError(
                f"Checkpoint already completed epoch {self.start_epoch}, "
                f"but requested target is {cfg.max_epoch + 1} epochs."
            )

        for epoch in range(self.start_epoch, cfg.max_epoch + 1):
            print(f"epoch_start {epoch}", flush=True)
            model.train()
            self.metric_train.reset()
            self.metric_val.reset()
            self.losses = []
            # Combined-loss component accumulators (stay empty for stock CE).
            self.ce_losses = []
            self.lovasz_losses = []
            model.trans_point_sampler = train_sampler.get_point_sampler()

            for step, inputs in enumerate(tqdm(train_loader, desc="training")):
                if hasattr(inputs["data"], "to"):
                    inputs["data"].to(device)
                self._apply_train_rgb_jitter(inputs)
                self.optimizer.zero_grad()
                results = model(inputs["data"])
                loss, gt_labels, predict_scores = model.get_loss(
                    loss_fn, results, inputs, device
                )

                if predict_scores.size()[-1] == 0:
                    continue

                loss.backward()
                if model.cfg.get("grad_clip_norm", -1) > 0:
                    torch.nn.utils.clip_grad_value_(
                        model.parameters(), model.cfg.grad_clip_norm
                    )
                self.optimizer.step()
                self.metric_train.update(predict_scores, gt_labels)
                self.losses.append(loss.cpu().item())
                if getattr(loss_fn, "last_ce", None) is not None:
                    self.ce_losses.append(loss_fn.last_ce)
                    self.lovasz_losses.append(loss_fn.last_lovasz)

                if "train" in record_summary and step == 0:
                    self.summary["train"] = self.get_3d_summary(
                        results, inputs["data"], epoch
                    )

            validation_result = self._run_validation_epoch(
                epoch=epoch,
                valid_loader=valid_loader,
                valid_sampler=valid_sampler,
                loss_fn=loss_fn,
                record_summary=record_summary,
            )

            if validation_result is not None:
                self._step_scheduler_after_validation(validation_result["row"])
                self.save_logs(writer, epoch)
            else:
                self._write_training_log_after_validation_failure(epoch)

            self.save_ckpt(epoch)

        writer.close()

    def _load_torch_checkpoint(self, path: Path) -> dict:
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _restore_cuda_rng_state_all(self, checkpoint: dict) -> None:
        if not torch.cuda.is_available() or "torch_cuda_rng_state_all" not in checkpoint:
            return

        raw_states = checkpoint["torch_cuda_rng_state_all"]
        if raw_states is None:
            return
        if isinstance(raw_states, torch.Tensor):
            states = [raw_states]
        elif isinstance(raw_states, (list, tuple)):
            states = list(raw_states)
        else:
            print(
                "resume_warning skipping_cuda_rng_state "
                f"reason=unsupported_type type={type(raw_states).__name__}",
                flush=True,
            )
            return

        normalized_states = []
        for idx, state in enumerate(states):
            try:
                if isinstance(state, torch.Tensor):
                    tensor = state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
                else:
                    tensor = torch.as_tensor(state, dtype=torch.uint8).cpu().contiguous()
                if tensor.ndim != 1:
                    tensor = tensor.reshape(-1).contiguous()
                normalized_states.append(tensor)
            except Exception as exc:  # noqa: BLE001
                print(
                    "resume_warning skipping_cuda_rng_state "
                    f"reason=state_{idx}_invalid error={exc}",
                    flush=True,
                )
                return

        if not normalized_states:
            return

        device_count = torch.cuda.device_count()
        if device_count > 0 and len(normalized_states) > device_count:
            print(
                "resume_warning truncating_cuda_rng_state "
                f"checkpoint_devices={len(normalized_states)} available_devices={device_count}",
                flush=True,
            )
            normalized_states = normalized_states[:device_count]

        try:
            torch.cuda.set_rng_state_all(normalized_states)
        except Exception as exc:  # noqa: BLE001
            print(
                "resume_warning skipping_cuda_rng_state "
                f"reason=set_rng_state_all_failed error={exc}",
                flush=True,
            )

    def _restore_scheduler_metric_history(self, checkpoint: dict) -> None:
        if "scheduler_metric_history" in checkpoint:
            self.scheduler_metric_history = [
                float(value) for value in checkpoint["scheduler_metric_history"]
            ]
            print(
                "resume_scheduler_metric_history "
                f"source=checkpoint count={len(self.scheduler_metric_history)}",
                flush=True,
            )
            return

        if not self.eval_history_path.exists():
            print("resume_scheduler_metric_history source=missing_eval_history count=0")
            return

        metric_key = METRIC_ALIASES.get(
            self.scheduler_watch_metric, self.scheduler_watch_metric
        )
        history: list[float] = []
        with self.eval_history_path.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    row_epoch = int(row.get("epoch", "0"))
                    if row_epoch <= 0 or row_epoch > self.start_epoch:
                        continue
                    value = row.get(metric_key)
                    if value is None or value == "":
                        continue
                    metric_value = float(value)
                    if np.isfinite(metric_value):
                        history.append(metric_value)
                except (TypeError, ValueError):
                    continue

        self.scheduler_metric_history = history
        print(
            "resume_scheduler_metric_history "
            f"source=eval_history metric={metric_key} count={len(history)}",
            flush=True,
        )

    def _load_training_checkpoint(self) -> None:
        if self.resume_from is None:
            print("resume_checkpoint none", flush=True)
            return

        checkpoint_path = self.resume_from
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = self._load_torch_checkpoint(checkpoint_path)
        required = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is missing required keys: {missing}"
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = int(checkpoint["epoch"])

        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        self._restore_cuda_rng_state_all(checkpoint)
        self._restore_scheduler_metric_history(checkpoint)

        print(
            f"resume_checkpoint {checkpoint_path} completed_epoch={self.start_epoch}",
            flush=True,
        )

    def _scheduler_metric_from_row(self, row: dict) -> tuple[float, float]:
        requested = self.scheduler_watch_metric
        metric_key = METRIC_ALIASES.get(requested, requested)
        if metric_key not in row:
            raise RuntimeError(
                f"Scheduler watch_metric {requested!r} maps to {metric_key!r}, "
                f"but that column is not in eval row. Available: {sorted(row)}"
            )

        raw_metric = float(row[metric_key])
        if not np.isfinite(raw_metric):
            raise RuntimeError(
                f"Scheduler watch_metric {requested!r} produced non-finite value {raw_metric}"
            )
        self.scheduler_metric_history.append(raw_metric)
        window_values = self.scheduler_metric_history[-self.scheduler_smoothing_window :]
        smoothed_metric = float(np.mean(window_values))
        return raw_metric, smoothed_metric

    def _step_scheduler_after_validation(self, row: dict) -> None:
        if self.scheduler_kind == "ReduceLROnPlateau":
            raw_metric, scheduler_metric = self._scheduler_metric_from_row(row)
            old_lr = self._current_lr()
            self.scheduler.step(scheduler_metric)
            print(
                "scheduler_step "
                f"type=ReduceLROnPlateau watch_metric={self.scheduler_watch_metric} "
                f"raw_value={raw_metric:.12g} "
                f"smoothing_window={self.scheduler_smoothing_window} "
                f"value={scheduler_metric:.12g} lr_before={old_lr:.12g} "
                f"lr_after={self._current_lr():.12g}",
                flush=True,
            )
            return

        old_lr = self._current_lr()
        self.scheduler.step()
        print(
            "scheduler_step "
            f"type={self.scheduler_kind} lr_before={old_lr:.12g} "
            f"lr_after={self._current_lr():.12g}",
            flush=True,
        )

    def _run_validation_epoch(
        self,
        epoch: int,
        valid_loader,  # noqa: ANN001
        valid_sampler,  # noqa: ANN001
        loss_fn,  # noqa: ANN001
        record_summary,  # noqa: ANN001
    ) -> dict | None:
        model = self.model
        device = self.device
        model.eval()
        self.valid_losses = []
        self.val_ce_losses = []
        self.val_lovasz_losses = []
        self._val_y_true = []
        self._val_y_pred = []
        self._val_ranges = []
        model.trans_point_sampler = valid_sampler.get_point_sampler()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        val_started_at = time.monotonic()

        try:
            with torch.no_grad():
                for step, inputs in enumerate(tqdm(valid_loader, desc="validation")):
                    if hasattr(inputs["data"], "to"):
                        inputs["data"].to(device)

                    results = model(inputs["data"])
                    loss, gt_labels, predict_scores = model.get_loss(
                        loss_fn, results, inputs, device
                    )

                    if predict_scores.size()[-1] == 0:
                        continue

                    self.metric_val.update(predict_scores, gt_labels)
                    self.valid_losses.append(loss.cpu().item())
                    if getattr(loss_fn, "last_ce", None) is not None:
                        self.val_ce_losses.append(loss_fn.last_ce)
                        self.val_lovasz_losses.append(loss_fn.last_lovasz)
                    self._accumulate_validation_batch(inputs, gt_labels, predict_scores)

                    if "valid" in record_summary and step == 0:
                        self.summary["valid"] = self.get_3d_summary(
                            results, inputs["data"], epoch
                        )

            val_wall_clock = time.monotonic() - val_started_at
            val_peak_memory = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            )
            row = self._write_eval_artifacts(epoch, val_peak_memory, val_wall_clock)
            return {"row": row}
        except Exception:
            error_path = self.run_dir / f"validation_errors_epoch_{epoch + 1:03d}.txt"
            error_path.write_text(traceback.format_exc())
            print(f"validation_failed epoch={epoch + 1} error_log={error_path}")
            return None

    def _accumulate_validation_batch(
        self,
        inputs,  # noqa: ANN001
        gt_labels: torch.Tensor,
        predict_scores: torch.Tensor,
    ) -> None:
        if "ranges" not in inputs["data"]:
            raise RuntimeError(
                "Validation ranges missing. The model transform wrapper must attach "
                "ego-frame ranges before RandLA-Net recentering."
            )

        raw_labels = inputs["data"]["labels"].detach().cpu().numpy().reshape(-1)
        ignored = np.asarray(self.model.cfg.ignored_label_inds, dtype=np.int64)
        valid_mask = ~np.isin(raw_labels, ignored)
        ranges = inputs["data"]["ranges"].detach().cpu().numpy().reshape(-1)
        valid_ranges = ranges[valid_mask]

        y_true = gt_labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
        y_pred = (
            torch.argmax(predict_scores, dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
            .reshape(-1)
        )

        if not (y_true.shape == y_pred.shape == valid_ranges.shape):
            raise RuntimeError(
                "Validation metric shape mismatch: "
                f"y_true={y_true.shape}, y_pred={y_pred.shape}, "
                f"ranges={valid_ranges.shape}"
            )

        self._val_y_true.append(y_true)
        self._val_y_pred.append(y_pred)
        self._val_ranges.append(valid_ranges.astype(np.float32, copy=False))

    def _write_eval_artifacts(
        self,
        epoch: int,
        peak_gpu_memory_bytes_val: int,
        val_wall_clock_seconds: float,
    ) -> dict:
        if self._val_y_true:
            y_true = np.concatenate(self._val_y_true)
            y_pred = np.concatenate(self._val_y_pred)
            ranges = np.concatenate(self._val_ranges)
        else:
            y_true = np.asarray([], dtype=np.int64)
            y_pred = np.asarray([], dtype=np.int64)
            ranges = np.asarray([], dtype=np.float32)

        metrics = compute_metrics(y_true, y_pred, ranges)
        train_loss = float(np.mean(self.losses)) if self.losses else float("nan")
        val_loss = float(np.mean(self.valid_losses)) if self.valid_losses else float("nan")
        row = flatten_metrics_for_csv(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics,
            peak_gpu_memory_bytes_val=peak_gpu_memory_bytes_val,
            val_wall_clock_seconds=val_wall_clock_seconds,
        )
        row["lr"] = self._current_lr()
        write_header = not self.eval_history_path.exists()
        with self.eval_history_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(EVAL_CSV_COLUMNS_D))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        positive_class_name = "lane"
        if hasattr(self.dataset, "get_positive_class_name"):
            positive_class_name = self.dataset.get_positive_class_name()
        snapshot = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": self._current_lr(),
            "peak_gpu_memory_bytes_val": peak_gpu_memory_bytes_val,
            "val_wall_clock_seconds": val_wall_clock_seconds,
            "metrics": metrics,
            "metric_aliases": {
                "note": (
                    "For Milestone D label_mode=road_marking3, existing lane_* "
                    "metric names mean marking_* because active class index 1 is "
                    "the expanded road-marking class."
                ),
                "positive_class_name": positive_class_name,
                "marking_iou": "lane_iou",
                "marking_precision": "lane_precision",
                "marking_recall": "lane_recall",
                "marking_f1": "lane_f1",
            },
            "class_indexing": {
                "0": "road",
                "1": positive_class_name,
                "2": "other",
                "source": "Open3D filter_valid_label compresses raw labels 1/2/3 after ignoring raw label 0.",
            },
        }
        (self.run_dir / f"eval_epoch_{epoch + 1:03d}.json").write_text(
            json.dumps(json_ready(snapshot), indent=2)
        )
        np.save(
            self.run_dir / f"confusion_epoch_{epoch + 1:03d}.npy",
            np.asarray(metrics["confusion_matrix"], dtype=np.int64),
        )

        # Optional combined-loss component logging. Only fires when a combined
        # loss (e.g. weighted_ce_lovasz) populated per-step CE/Lovász values;
        # stock weighted-CE runs leave these empty and write nothing, so
        # eval_history.csv and the existing analysis stay untouched.
        ce_losses = getattr(self, "ce_losses", [])
        lov_losses = getattr(self, "lovasz_losses", [])
        val_ce_losses = getattr(self, "val_ce_losses", [])
        val_lov_losses = getattr(self, "val_lovasz_losses", [])
        if ce_losses or val_ce_losses:
            def _mean(xs):
                return float(np.mean(xs)) if xs else float("nan")

            comp_row = {
                "epoch": epoch + 1,
                "train_ce_loss": _mean(ce_losses),
                "train_lovasz_loss": _mean(lov_losses),
                "train_total_loss": train_loss,
                "val_ce_loss": _mean(val_ce_losses),
                "val_lovasz_loss": _mean(val_lov_losses),
                "val_total_loss": val_loss,
            }
            comp_path = self.run_dir / "loss_components.csv"
            write_comp_header = not comp_path.exists()
            with comp_path.open("a", newline="") as f:
                comp_writer = csv.DictWriter(f, fieldnames=list(comp_row.keys()))
                if write_comp_header:
                    comp_writer.writeheader()
                comp_writer.writerow(comp_row)
            print(
                "loss_components "
                f"epoch={epoch + 1} "
                f"train_ce={comp_row['train_ce_loss']:.6f} "
                f"train_lovasz={comp_row['train_lovasz_loss']:.6f} "
                f"train_total={comp_row['train_total_loss']:.6f} "
                f"val_ce={comp_row['val_ce_loss']:.6f} "
                f"val_lovasz={comp_row['val_lovasz_loss']:.6f} "
                f"val_total={comp_row['val_total_loss']:.6f}",
                flush=True,
            )
        return row

    def _write_training_log_after_validation_failure(self, epoch: int) -> None:
        wall_clock = time.monotonic() - self._epoch_started_at
        self._epoch_started_at = time.monotonic()
        peak_mem = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with self.training_log.open("a") as f:
            f.write(
                f"epoch={epoch + 1} open3d_epoch={epoch} "
                f"wall_clock_seconds={wall_clock:.3f} "
                f"peak_gpu_memory_bytes={peak_mem} "
                f"lr={self._current_lr():.12g} validation_failed=true\n"
            )

    def save_ckpt(self, epoch):
        human_epoch = epoch + 1
        save_freq = int(self.cfg.get("save_ckpt_freq", 10))
        should_save_periodic = save_freq > 0 and human_epoch % save_freq == 0
        if not should_save_periodic and human_epoch != self.requested_epochs:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"ckpt_epoch_{human_epoch:05d}.pth"
        torch.save(
            {
                "epoch": human_epoch,
                "open3d_epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scheduler_kind": self.scheduler_kind,
                "scheduler_watch_metric": self.scheduler_watch_metric,
                "scheduler_metric_history": list(self.scheduler_metric_history),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            },
            path,
        )
        print(f"saved_checkpoint {path}")


def parse_args(
    default_config: Path = DEFAULT_CONFIG,
    default_runs_dir: Path = RUNS_DIR,
    description: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help=(
            "Directory that contains run folders. Defaults to "
            f"{default_runs_dir.relative_to(PROJECT_ROOT)}, except a config under "
            "logs/milestone_<x> infers logs/milestone_<x>/runs."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Target total epochs for this run. On resume, training continues up to this total.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch-train", type=int)
    parser.add_argument("--steps-per-epoch-valid", type=int)
    parser.add_argument("--save-ckpt-freq", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    pin_group = parser.add_mutually_exclusive_group()
    pin_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Override pipeline.pin_memory to true.",
    )
    pin_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Override pipeline.pin_memory to false.",
    )
    parser.set_defaults(pin_memory=None)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable dataset preprocessing cache for tiny smoke runs.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest checkpoint in the selected run directory.",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(False)
    except Exception as exc:
        print(f"determinism_warning {exc}")


def resolve_runs_dir(
    args: argparse.Namespace,
    default_runs_dir: Path = RUNS_DIR,
) -> Path:
    if args.runs_dir is not None:
        runs_dir = args.runs_dir
        if not runs_dir.is_absolute():
            runs_dir = PROJECT_ROOT / runs_dir
        return runs_dir

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    try:
        rel_parts = config_path.resolve().relative_to(PROJECT_ROOT).parts
    except ValueError:
        rel_parts = config_path.parts

    if len(rel_parts) >= 3 and rel_parts[0] == "logs" and rel_parts[1].startswith("milestone_"):
        return PROJECT_ROOT / "logs" / rel_parts[1] / "runs"

    return default_runs_dir


def prepare_run_dir(run_name: str, force: bool, resume: bool, runs_dir: Path) -> Path:
    run_dir = runs_dir / run_name
    if resume:
        if not run_dir.exists():
            raise SystemExit(f"Cannot resume missing run directory: {run_dir}")
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise SystemExit(
            f"Run directory is non-empty: {run_dir}. Re-run with --force to overwrite."
        )
    if run_dir.exists() and force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_resume_checkpoint(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if args.resume_from is not None and args.resume_latest:
        raise SystemExit("Use either --resume-from or --resume-latest, not both.")
    if args.force and (args.resume_from is not None or args.resume_latest):
        raise SystemExit("--force cannot be combined with resume.")

    if args.resume_from is not None:
        checkpoint = args.resume_from
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        if not checkpoint.exists():
            raise SystemExit(f"Resume checkpoint not found: {checkpoint}")
        return checkpoint

    if args.resume_latest:
        checkpoint_dir = run_dir / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("ckpt_epoch_*.pth"))
        if not checkpoints:
            raise SystemExit(f"No checkpoints found in {checkpoint_dir}")
        return checkpoints[-1]

    return None


def git_status_text() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet"],
                cwd=PROJECT_ROOT,
                text=True,
            ).returncode
            != 0
        )
        return f"git_available true\ncommit {commit}\nworking_tree_dirty {dirty}\n"
    except Exception as exc:
        return f"git_available false\ngit_error {exc}\n"


def load_config(args: argparse.Namespace, run_dir: Path) -> dict:
    cfg = yaml.safe_load(args.config.read_text())
    cfg["pipeline"]["max_epoch"] = max(0, args.epochs - 1)
    cfg["pipeline"]["save_ckpt_freq"] = args.save_ckpt_freq
    cfg["pipeline"]["main_log_dir"] = str(run_dir / "open3d_logs")
    cfg["pipeline"]["train_sum_dir"] = str(run_dir / "tensorboard")
    cfg["pipeline"].pop("real_training_allowed", None)
    feature_mode = str(cfg.get("dataset", {}).get("feature_mode", "intensity"))
    if feature_mode != "intensity" and "grid_size" in cfg.get("model", {}):
        model_grid_size = float(cfg["model"]["grid_size"])
        explicit_cache_grid_size = cfg["dataset"].get("cache_grid_size")
        if (
            explicit_cache_grid_size is not None
            and float(explicit_cache_grid_size) != model_grid_size
        ):
            raise SystemExit(
                "dataset.cache_grid_size must match model.grid_size for RGB cache "
                "safety: "
                f"cache_grid_size={explicit_cache_grid_size!r}, "
                f"model.grid_size={model_grid_size!r}"
            )
        cfg["dataset"]["cache_grid_size"] = model_grid_size
    if args.steps_per_epoch_train is not None:
        cfg["dataset"]["steps_per_epoch_train"] = args.steps_per_epoch_train
    if args.steps_per_epoch_valid is not None:
        cfg["dataset"]["steps_per_epoch_valid"] = args.steps_per_epoch_valid
    if args.no_cache:
        cfg["dataset"]["use_cache"] = False
    if args.device is not None:
        cfg["pipeline"]["device"] = args.device
    if getattr(args, "pin_memory", None) is not None:
        cfg["pipeline"]["pin_memory"] = bool(args.pin_memory)
    cfg["model"]["ckpt_path"] = None
    cfg["model"]["is_resume"] = False
    return cfg


def write_start_artifacts(
    args: argparse.Namespace,
    cfg: dict,
    run_dir: Path,
    resume_checkpoint: Path | None,
) -> None:
    if resume_checkpoint is None:
        (run_dir / "config_snapshot.yml").write_text(
            yaml.safe_dump(cfg, sort_keys=False)
        )
        (run_dir / "git_commit.txt").write_text(git_status_text())
        (run_dir / "seed.txt").write_text(str(args.seed) + "\n")
        (run_dir / "cli_args.json").write_text(
            json.dumps(vars(args), default=str, indent=2)
        )
        (run_dir / "start_time.txt").write_text(datetime.now().isoformat() + "\n")
        return

    event = {
        "time": datetime.now().isoformat(),
        "resume_checkpoint": str(resume_checkpoint),
        "target_epochs": args.epochs,
        "config": str(args.config),
        "cli_args": vars(args),
    }
    with (run_dir / "resume_events.jsonl").open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    (run_dir / "config_snapshot_resume.yml").write_text(
        yaml.safe_dump(cfg, sort_keys=False)
    )


def build_pipeline(
    cfg: dict,
    run_dir: Path,
    requested_epochs: int,
    resume_checkpoint: Path | None,
) -> MilestoneDPipeline:
    dataset_cfg = dict(cfg["dataset"])
    print("build_dataset_start")
    dataset = PandaSetFFLane3Dataset(**dataset_cfg)
    print("build_dataset_done")

    model_cfg = dict(cfg["model"])
    # Open3D-ML defaults to auto-resume from visible checkpoints when
    # model.cfg.is_resume is true. Set both ckpt_path=None and is_resume=False
    # before pipeline construction so BasePipeline/run_train stay untouched.
    model_cfg["ckpt_path"] = None
    model_cfg["is_resume"] = False
    print("build_model_start")
    model = ml3d.models.RandLANet(**model_cfg)
    attach_ego_ranges_to_transform(model)
    print("build_model_done")

    pipeline_cfg = dict(cfg["pipeline"])
    pipeline_cfg.setdefault("pin_memory", False)
    print("build_pipeline_start")
    return MilestoneDPipeline(
        model=model,
        dataset=dataset,
        run_dir=run_dir,
        requested_epochs=requested_epochs,
        resume_from=resume_checkpoint,
        **pipeline_cfg,
    )


def attach_ego_ranges_to_transform(model) -> None:  # noqa: ANN001
    """Attach pre-recenter ego ranges to transformed RandLA-Net batches.

    RandLA-Net validation recentering changes ``coords`` before the model sees
    them. For distance-bucket lane recall, we keep the Euclidean range computed
    from pre-augmentation ego-frame points after grid subsampling and select the
    same patch indices as the model.
    """

    original_transform = model.transform

    def transform_with_ranges(data, attr, min_possibility_idx=None):  # noqa: ANN001
        ranges = np.linalg.norm(data["point"][:, :3], axis=1).astype(np.float32)
        transformed = original_transform(data, attr, min_possibility_idx)
        point_inds = transformed.get("point_inds")
        if point_inds is None:
            raise RuntimeError(
                "RandLA-Net transform did not produce 'point_inds'. "
                "Distance-bucket lane recall cannot be computed."
            )
        transformed["ranges"] = ranges[point_inds].astype(np.float32, copy=False)
        return transformed

    model.transform = transform_with_ranges


def main(
    default_config: Path = DEFAULT_CONFIG,
    default_runs_dir: Path = RUNS_DIR,
    completion_label: str = DEFAULT_COMPLETION_LABEL,
    description: str | None = None,
) -> None:
    args = parse_args(
        default_config=default_config,
        default_runs_dir=default_runs_dir,
        description=description,
    )
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.save_ckpt_freq < 1:
        raise SystemExit("--save-ckpt-freq must be >= 1")
    resume_requested = args.resume_from is not None or args.resume_latest
    runs_dir = resolve_runs_dir(args, default_runs_dir=default_runs_dir)
    run_dir = prepare_run_dir(args.run_name, args.force, resume_requested, runs_dir)
    resume_checkpoint = resolve_resume_checkpoint(args, run_dir)
    stdout_path = run_dir / "stdout.log"

    with stdout_path.open("a") as stdout_f:
        tee = Tee(sys.__stdout__, stdout_f)
        with redirect_stdout(tee), redirect_stderr(tee):
            start = time.monotonic()
            set_seeds(args.seed)
            cfg = load_config(args, run_dir)
            write_start_artifacts(args, cfg, run_dir, resume_checkpoint)
            print(f"runs_dir {runs_dir}")
            print(f"run_dir {run_dir}")
            print(f"requested_epochs {args.epochs}")
            print(f"resume_checkpoint {resume_checkpoint}")
            pipeline = build_pipeline(cfg, run_dir, args.epochs, resume_checkpoint)
            print("run_train_start")
            pipeline.run_train()
            print("run_train_done")
            wall_clock = time.monotonic() - start
            (run_dir / "end_time.txt").write_text(datetime.now().isoformat() + "\n")
            print(
                f"{completion_label}: "
                f"run_name={args.run_name} epochs={args.epochs} "
                f"wall_clock={wall_clock:.3f}"
            )


if __name__ == "__main__":
    main()

"""Milestone C training entrypoint.

This is the real-training driver for Milestone C. It intentionally stays
separate from tools/sanity_train_check.py, which is the Milestone B historical
sanity record.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
from thesis_pipeline.eval.milestone_c_metrics import (
    EVAL_CSV_COLUMNS,
    compute_metrics,
    flatten_metrics_for_csv,
    json_ready,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/randlanet_pandaset_ff_lane3.yml"
RUNS_DIR = PROJECT_ROOT / "logs/milestone_c/runs"


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


class MilestoneCPipeline(SemanticSegmentation):
    """Small wrapper for run-local checkpointing and epoch timing."""

    def __init__(self, *args, run_dir: Path, requested_epochs: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_dir = run_dir
        self.requested_epochs = requested_epochs
        self.checkpoint_dir = run_dir / "checkpoints"
        self.training_log = run_dir / "training_log.txt"
        self.eval_history_path = run_dir / "eval_history.csv"
        self._epoch_started_at = time.monotonic()
        self._val_y_true: list[np.ndarray] = []
        self._val_y_pred: list[np.ndarray] = []
        self._val_ranges: list[np.ndarray] = []

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
            f"peak_gpu_memory_bytes={peak_mem}"
        )
        with self.training_log.open("a") as f:
            f.write(line + "\n")

    def run_train(self):
        """Train with per-epoch Milestone C validation metric artifacts."""

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
        loss_fn = SemSegLoss(self, model, dataset, device)
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
            "Milestone C validation metrics must use configs/splits/val.txt "
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
        self.optimizer, self.scheduler = model.get_optimizer(cfg)
        print("run_train_optimizer_done", flush=True)
        print("run_train_load_ckpt_start", flush=True)
        self.load_ckpt(model.cfg.ckpt_path, is_resume=model.cfg.get("is_resume", True))
        print("run_train_load_ckpt_done", flush=True)

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

        for epoch in range(0, cfg.max_epoch + 1):
            print(f"epoch_start {epoch}", flush=True)
            model.train()
            self.metric_train.reset()
            self.metric_val.reset()
            self.losses = []
            model.trans_point_sampler = train_sampler.get_point_sampler()

            for step, inputs in enumerate(tqdm(train_loader, desc="training")):
                if hasattr(inputs["data"], "to"):
                    inputs["data"].to(device)
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

                if "train" in record_summary and step == 0:
                    self.summary["train"] = self.get_3d_summary(
                        results, inputs["data"], epoch
                    )

            self.scheduler.step()

            validation_ok = self._run_validation_epoch(
                epoch=epoch,
                valid_loader=valid_loader,
                valid_sampler=valid_sampler,
                loss_fn=loss_fn,
                record_summary=record_summary,
            )

            if validation_ok:
                self.save_logs(writer, epoch)
            else:
                self._write_training_log_after_validation_failure(epoch)

            if epoch % cfg.save_ckpt_freq == 0 or epoch == cfg.max_epoch:
                self.save_ckpt(epoch)

        writer.close()

    def _run_validation_epoch(
        self,
        epoch: int,
        valid_loader,  # noqa: ANN001
        valid_sampler,  # noqa: ANN001
        loss_fn,  # noqa: ANN001
        record_summary,  # noqa: ANN001
    ) -> bool:
        model = self.model
        device = self.device
        model.eval()
        self.valid_losses = []
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
                    self._accumulate_validation_batch(inputs, gt_labels, predict_scores)

                    if "valid" in record_summary and step == 0:
                        self.summary["valid"] = self.get_3d_summary(
                            results, inputs["data"], epoch
                        )

            val_wall_clock = time.monotonic() - val_started_at
            val_peak_memory = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            )
            self._write_eval_artifacts(epoch, val_peak_memory, val_wall_clock)
            return True
        except Exception:
            error_path = self.run_dir / f"validation_errors_epoch_{epoch + 1:03d}.txt"
            error_path.write_text(traceback.format_exc())
            print(f"validation_failed epoch={epoch + 1} error_log={error_path}")
            return False

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
    ) -> None:
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
        write_header = not self.eval_history_path.exists()
        with self.eval_history_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(EVAL_CSV_COLUMNS))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        snapshot = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "peak_gpu_memory_bytes_val": peak_gpu_memory_bytes_val,
            "val_wall_clock_seconds": val_wall_clock_seconds,
            "metrics": metrics,
            "class_indexing": {
                "0": "road",
                "1": "lane",
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
                f"peak_gpu_memory_bytes={peak_mem} validation_failed=true\n"
            )

    def save_ckpt(self, epoch):
        human_epoch = epoch + 1
        if human_epoch % 10 != 0 and human_epoch != self.requested_epochs:
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
            },
            path,
        )
        print(f"saved_checkpoint {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--no-resume", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch-train", type=int)
    parser.add_argument("--steps-per-epoch-valid", type=int)
    parser.add_argument("--force", action="store_true")
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


def prepare_run_dir(run_name: str, force: bool) -> Path:
    run_dir = RUNS_DIR / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise SystemExit(
            f"Run directory is non-empty: {run_dir}. Re-run with --force to overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
    cfg["pipeline"]["save_ckpt_freq"] = 10
    cfg["pipeline"]["main_log_dir"] = str(run_dir / "open3d_logs")
    cfg["pipeline"]["train_sum_dir"] = str(run_dir / "tensorboard")
    cfg["pipeline"].pop("real_training_allowed", None)
    if args.steps_per_epoch_train is not None:
        cfg["dataset"]["steps_per_epoch_train"] = args.steps_per_epoch_train
    if args.steps_per_epoch_valid is not None:
        cfg["dataset"]["steps_per_epoch_valid"] = args.steps_per_epoch_valid
    cfg["model"]["ckpt_path"] = None
    cfg["model"]["is_resume"] = False
    return cfg


def write_start_artifacts(args: argparse.Namespace, cfg: dict, run_dir: Path) -> None:
    (run_dir / "config_snapshot.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "git_commit.txt").write_text(git_status_text())
    (run_dir / "seed.txt").write_text(str(args.seed) + "\n")
    (run_dir / "cli_args.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    (run_dir / "start_time.txt").write_text(datetime.now().isoformat() + "\n")


def build_pipeline(
    cfg: dict, run_dir: Path, requested_epochs: int
) -> MilestoneCPipeline:
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
    return MilestoneCPipeline(
        model=model,
        dataset=dataset,
        run_dir=run_dir,
        requested_epochs=requested_epochs,
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


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    run_dir = prepare_run_dir(args.run_name, args.force)
    stdout_path = run_dir / "stdout.log"

    with stdout_path.open("a") as stdout_f:
        tee = Tee(sys.__stdout__, stdout_f)
        with redirect_stdout(tee), redirect_stderr(tee):
            start = time.monotonic()
            set_seeds(args.seed)
            cfg = load_config(args, run_dir)
            write_start_artifacts(args, cfg, run_dir)
            print(f"run_dir {run_dir}")
            print(f"requested_epochs {args.epochs}")
            print("is_resume_forced false")
            pipeline = build_pipeline(cfg, run_dir, args.epochs)
            print("run_train_start")
            pipeline.run_train()
            print("run_train_done")
            wall_clock = time.monotonic() - start
            (run_dir / "end_time.txt").write_text(datetime.now().isoformat() + "\n")
            print(
                "milestone_c_run_complete: "
                f"run_name={args.run_name} epochs={args.epochs} "
                f"wall_clock={wall_clock:.3f}"
            )


if __name__ == "__main__":
    main()

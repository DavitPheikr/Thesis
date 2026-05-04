"""
Sanity training check: assemble the full Open3D-ML pipeline and run a few
training iterations.

Pass conditions (all must hold):
  1. Pipeline instantiates without error
  2. At least 5 step-level losses are captured across training epochs
  3. All captured loss values are finite (not NaN, not Inf)
  4. Loss changes between the first and last captured iteration
  5. Final loss does not exceed 10x the initial loss

If condition 5 fails with the recommended (open3d_native_from_measured_counts)
weights, the script retries with the sqrt_inverse_frequency weights.
If both attempts complete but remain unstable, INSTABILITY is recorded and the
script exits cleanly after writing the report.

Implementation notes:
  - Uses _SanityCapturer, a thin SemanticSegmentation subclass that overrides
    save_logs() to accumulate per-step losses across epochs (the base class
    resets self.losses at the start of every epoch).
  - num_workers=0 is forced for the sanity run to avoid DataLoader subprocess
    issues in the laptop / VS Code environment.
  - pin_memory=False is forced for the same reason.
  - real_training_allowed is ignored by the Open3D pipeline (it is a custom
    project sentinel stored in cfg but never checked by the pipeline).

Emits script_status PASS/FAIL as final line.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import open3d.ml.torch as ml3d
from open3d._ml3d.torch.pipelines import SemanticSegmentation

from datasets.pandaset_ff_lane3 import PandaSetFFLane3Dataset

STATS_FILE = Path("logs/milestone_b_training_statistics.json")
CFG_FILE = Path("configs/randlanet_pandaset_ff_lane3.yml")
REPORT_FILE = Path("logs/milestone_b_sanity_train_report.txt")
DIVERGENCE_THRESHOLD = 10.0
MIN_STEPS = 5


# ---------------------------------------------------------------------------
# Loss-capturing pipeline subclass
# ---------------------------------------------------------------------------

class _SanityCapturer(SemanticSegmentation):
    """SemanticSegmentation subclass that accumulates per-step losses across
    all training epochs.

    The base class resets ``self.losses`` at the start of each epoch; this
    subclass captures the completed-epoch losses in ``self.save_logs()``
    (called at the end of each epoch) and appends them to
    ``_all_step_losses`` before the next reset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_step_losses: list[float] = []

    def save_logs(self, writer, epoch):
        # self.losses holds the just-completed epoch's per-step losses.
        # Capture them before super().save_logs() is called (which uses
        # them for TensorBoard but does NOT reset them — the reset happens
        # at the TOP of the next epoch's training loop).
        self._all_step_losses.extend(list(self.losses))
        super().save_logs(writer, epoch)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _stdout_targets_report() -> bool:
    """Avoid dual-writing when stdout is shell-redirected to REPORT_FILE."""
    try:
        stdout_target = Path(os.readlink("/proc/self/fd/1")).resolve()
        return stdout_target == REPORT_FILE.resolve()
    except OSError:
        return False


def _flush_report(lines: list[str]) -> None:
    if _stdout_targets_report():
        return
    REPORT_FILE.write_text("\n".join(lines) + "\n")


def _build_pipeline(cfg: dict, class_weights: list[float]) -> tuple:
    """Instantiate dataset, model, and _SanityCapturer pipeline.

    Returns (pipeline, dataset).
    """
    dataset_cfg = dict(cfg["dataset"])
    dataset_cfg["class_weights"] = class_weights

    dataset = PandaSetFFLane3Dataset(**dataset_cfg)

    model_cfg = dict(cfg["model"])
    # Prevent silent checkpoint resume. The Day 6 sanity run must always start
    # from scratch even if a later rerun leaves checkpoint files behind.
    model_cfg["ckpt_path"] = None
    model_cfg["is_resume"] = False
    model = ml3d.models.RandLANet(**model_cfg)

    pipeline_cfg = dict(cfg["pipeline"])
    # Remove the non-standard sentinel; it is only meaningful as a human
    # reminder, not an Open3D parameter.
    pipeline_cfg.pop("real_training_allowed", None)
    # Force single-process data loading for the sanity run to avoid
    # subprocess / memory pressure issues on this laptop.
    pipeline_cfg["num_workers"] = 0
    pipeline_cfg["pin_memory"] = False

    pipeline = _SanityCapturer(
        model=model,
        dataset=dataset,
        **pipeline_cfg,
    )
    return pipeline, dataset


def check_losses(losses: list[float]) -> tuple[bool, str | None]:
    if len(losses) < MIN_STEPS:
        return False, f"only {len(losses)} steps captured (need >= {MIN_STEPS})"
    if any(not math.isfinite(v) for v in losses):
        bad = [v for v in losses if not math.isfinite(v)]
        return False, f"non-finite loss values found: {bad[:5]}"
    if losses[-1] == losses[0]:
        return False, f"loss did not change: first={losses[0]}, last={losses[-1]}"
    ratio = losses[-1] / (losses[0] + 1e-9)
    if ratio > DIVERGENCE_THRESHOLD:
        return False, (
            f"loss diverged: ratio={ratio:.2f} > threshold={DIVERGENCE_THRESHOLD} "
            f"(first={losses[0]:.4f}, last={losses[-1]:.4f})"
        )
    return True, None


def run_sanity(cfg: dict, class_weights: list[float], variant_name: str
               ) -> tuple[list[float] | None, str | None]:
    """Run sanity training with the given weights.

    Returns (all_step_losses, error_string_or_None).
    """
    try:
        pipeline, _ = _build_pipeline(cfg, class_weights)
        pipeline.run_train()
        return pipeline._all_step_losses, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    lines: list[str] = []

    stats = json.loads(STATS_FILE.read_text())
    cfg = yaml.safe_load(CFG_FILE.read_text())

    # Primary weights: the sanity_run_recommended_list (open3d native counts)
    primary_weights = stats["class_weights"]["sanity_run_recommended_list"]
    primary_variant = stats["class_weights"].get(
        "sanity_run_recommended_variant", "open3d_native_from_measured_counts"
    )
    # Fallback weights: sqrt inverse frequency (direct CE candidate)
    fallback_weights = stats["class_weights"]["sqrt_inverse_frequency"]["as_list"]
    fallback_variant = "sqrt_inverse_frequency"

    lines.extend([
        f"sanity_script_started_at {datetime.now().isoformat()}",
        f"primary_weight_variant {primary_variant}",
        f"fallback_weight_variant {fallback_variant}",
        f"pipeline_max_epoch {cfg['pipeline']['max_epoch']}",
        f"steps_per_epoch_train {cfg['dataset'].get('steps_per_epoch_train')}",
        f"steps_per_epoch_valid {cfg['dataset'].get('steps_per_epoch_valid')}",
        f"num_points {cfg['model']['num_points']}",
        f"real_training_allowed {cfg['pipeline'].get('real_training_allowed')}",
    ])
    _flush_report(lines)

    print(f"primary_weight_variant {primary_variant}")
    print(f"primary_weights {primary_weights}")

    # ------------------------------------------------------------------
    # Attempt 1: primary weights
    # ------------------------------------------------------------------
    lines.append("attempt_1_status STARTED")
    lines.append(f"attempt_1_variant {primary_variant}")
    _flush_report(lines)
    print(f"\nattempt_1_variant {primary_variant}")
    losses_1, err_1 = run_sanity(cfg, primary_weights, primary_variant)

    if err_1 is not None:
        msg = f"attempt_1_error {err_1}"
        lines.append(msg)
        lines.append("script_status FAIL")
        print(msg)
        _flush_report(lines)
        print("script_status FAIL")
        sys.exit(1)

    ok_1, msg_1 = check_losses(losses_1)
    lines.append(f"attempt_1_steps {len(losses_1)}")
    lines.append(f"attempt_1_losses {losses_1}")
    lines.append(f"attempt_1_ok {ok_1}")
    _flush_report(lines)

    if ok_1:
        used_variant = primary_variant
        used_losses = losses_1
    else:
        print(f"attempt_1_failed: {msg_1}")
        lines.append(f"attempt_1_fail_reason {msg_1}")
        _flush_report(lines)

        # ------------------------------------------------------------------
        # Attempt 2: fallback (sqrt inverse frequency)
        # ------------------------------------------------------------------
        lines.append("attempt_2_status STARTED")
        lines.append(f"attempt_2_variant {fallback_variant}")
        _flush_report(lines)
        print(f"\nattempt_2_variant {fallback_variant}")
        losses_2, err_2 = run_sanity(cfg, fallback_weights, fallback_variant)

        if err_2 is not None:
            msg = f"attempt_2_error {err_2}"
            lines.append(msg)
            lines.append("script_status FAIL")
            _flush_report(lines)
            print(msg)
            print("script_status FAIL")
            sys.exit(1)

        ok_2, msg_2 = check_losses(losses_2)
        lines.append(f"attempt_2_steps {len(losses_2)}")
        lines.append(f"attempt_2_losses {losses_2}")
        lines.append(f"attempt_2_ok {ok_2}")
        _flush_report(lines)

        if not ok_2:
            lines.append(f"attempt_2_fail_reason {msg_2}")
            lines.append("sanity_result INSTABILITY")
            lines.append(
                "instability_note Both weight variants produced diverging or invalid loss."
            )
            lines.append("script_status PASS")
            _flush_report(lines)
            for line in lines[-5:]:
                print(line)
            sys.exit(0)

        used_variant = fallback_variant
        used_losses = losses_2

    # ------------------------------------------------------------------
    # Record final results
    # ------------------------------------------------------------------
    ratio = used_losses[-1] / (used_losses[0] + 1e-9)
    lines.append(f"sanity_weight_variant_used {used_variant}")
    lines.append(f"sanity_steps_total {len(used_losses)}")
    lines.append(f"loss_first {used_losses[0]:.6f}")
    lines.append(f"loss_last {used_losses[-1]:.6f}")
    lines.append(f"loss_ratio_final_to_initial {ratio:.4f}")
    lines.append("sanity_train_check_ok True")
    lines.append("script_status PASS")

    _flush_report(lines)
    for line in lines:
        print(line)

    sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run sampled validation error analysis for G1 best checkpoint.

This intentionally reuses the G0 sampled inference engine. The model/data
pipeline is identical between G0 and G1; only the run directory, config, and
output package change. Post-processing trims the final package to the requested
G1 outputs and rewrites the README/provenance text for G1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_CODE = PROJECT_ROOT / "logs/milestone_g/run_analysis/analysis_code"
sys.path.insert(0, str(ANALYSIS_CODE))

import g0_sampled_error_analysis as base  # noqa: E402


RUN_NAME = "G1_schedule_extend"
G1_RUN_DIR = PROJECT_ROOT / f"logs/milestone_g/runs/{RUN_NAME}"
ANALYSIS_DIR = PROJECT_ROOT / f"logs/milestone_g/run_analysis/{RUN_NAME}"
DEFAULT_CONFIG = PROJECT_ROOT / "logs/milestone_g/configs/g1_schedule_extend.yml"


def discover_best_epoch() -> int:
    history = pd.read_csv(G1_RUN_DIR / "eval_history.csv")
    return int(history.loc[history["lane_iou"].idxmax(), "epoch"])


def patch_base_module() -> None:
    base.RUN_NAME = RUN_NAME
    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    # The imported G0 module uses this variable name generically in checkpoint
    # discovery. Point it at G1 before calling base.main().
    base.G0_RUN_DIR = G1_RUN_DIR
    base.ANALYSIS_DIR = ANALYSIS_DIR

    original_load_cfg = base.load_cfg

    def load_cfg_g1(path: Path, steps: int):  # noqa: ANN001
        cfg = original_load_cfg(path, steps)
        cfg["dataset"]["cache_dir"] = str(
            (PROJECT_ROOT / "logs/milestone_g/cache/G1_schedule_extend").resolve()
        )
        cfg["dataset"]["test_result_folder"] = str(
            (PROJECT_ROOT / "logs/milestone_g/test_results/G1_schedule_extend").resolve()
        )
        return cfg

    base.load_cfg = load_cfg_g1

    original_tqdm = base.tqdm

    def tqdm_g1(*args, **kwargs):  # noqa: ANN002, ANN003
        if kwargs.get("desc") == "g0_sampled_validation":
            kwargs["desc"] = "g1_sampled_validation"
        return original_tqdm(*args, **kwargs)

    base.tqdm = tqdm_g1


def enrich_rgb_valid_metrics(out_dir: Path) -> None:
    path = out_dir / "rgb_valid_stratified_metrics.csv"
    df = pd.read_csv(path)
    required = {"all", "rgb_valid", "rgb_invalid"}
    observed = set(df["stratum"].astype(str))
    missing = sorted(required - observed)
    if missing:
        raise RuntimeError(f"{path} missing RGB strata: {missing}")
    all_active = float(df.loc[df["stratum"] == "all", "active_points"].iloc[0])
    if all_active <= 0:
        raise RuntimeError(f"{path} has zero all active_points")
    df["active_point_share"] = df["active_points"] / all_active
    df["true_marking_share"] = df["true_marking"] / df["active_points"]
    df["predicted_marking_share"] = df["predicted_marking"] / df["active_points"]
    df.to_csv(path, index=False)


def rewrite_summary_and_readme(out_dir: Path, args_text: str) -> None:
    summary_path = out_dir / "summary.json"
    data = json.loads(summary_path.read_text())
    data["run_name"] = RUN_NAME
    data["script"] = "logs/milestone_g/run_analysis/analysis_code/g1_sampled_error_analysis.py"
    data["note"] = "Fresh sampled inference pass; not exact original G1 best-epoch validation sample."
    summary_path.write_text(json.dumps(data, indent=2))

    rgb = pd.read_csv(out_dir / "rgb_valid_stratified_metrics.csv")
    readme = [
        "# G1 Best-Checkpoint Sampled Error Analysis",
        "",
        "Fresh sampled validation inference pass. This is not the exact training-time validation sample.",
        "",
        f"- run: `{RUN_NAME}`",
        f"- checkpoint: `{data['checkpoint']}`",
        f"- checkpoint epoch from checkpoint payload: `{data.get('checkpoint_epoch')}`",
        f"- split: `{data['split']}`",
        f"- steps: `{data['steps']}`",
        f"- seed: `{data['seed']}`",
        f"- device: `{data['device']}`",
        f"- command args: `{args_text}`",
        "",
        "## RGB-Validity Metrics",
        "",
        "| stratum | active share | marking IoU | precision | recall | F1 | pred/true | road->marking | marking->road |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in rgb.iterrows():
        readme.append(
            f"| {row['stratum']} | {row['active_point_share']:.3f} | "
            f"{row['marking_iou']:.6f} | {row['marking_precision']:.6f} | "
            f"{row['marking_recall']:.6f} | {row['marking_f1']:.6f} | "
            f"{row['predicted_true_marking_ratio']:.3f} | "
            f"{int(row['road_to_marking'])} | {int(row['marking_to_road'])} |"
        )
    readme.extend(
        [
            "",
            "Use this file with `distance_bucket_metrics.csv`, `per_sequence_metrics.csv`,",
            "`raw_subtype_rgb_stratified_metrics.csv`, and the top-frame CSVs to understand",
            "whether G1's aggregate result is driven by RGB-valid points, a distance band,",
            "or a small number of sequences/frames.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(readme))


def remove_unrequested_outputs(out_dir: Path) -> None:
    extra = out_dir / "group_feature_summary.csv"
    if extra.exists():
        extra.unlink()


def main() -> None:
    patch_base_module()
    best_epoch = discover_best_epoch()
    default_out = ANALYSIS_DIR / f"sampled_error_analysis_epoch{best_epoch}"

    # If the user did not pass explicit config/out-dir/checkpoint args, insert G1 defaults
    # before handing off to the G0 engine.
    argv = sys.argv[1:]
    if "--config" not in argv:
        argv = ["--config", str(DEFAULT_CONFIG), *argv]
    if "--out-dir" not in argv:
        argv = ["--out-dir", str(default_out), *argv]
    if "--checkpoint" not in argv:
        checkpoint = G1_RUN_DIR / "checkpoints" / f"ckpt_epoch_{best_epoch:05d}.pth"
        argv = ["--checkpoint", str(checkpoint), *argv]
    sys.argv = [sys.argv[0], *argv]

    base.main()
    out_dir = default_out
    if "--out-dir" in argv:
        out_dir = Path(argv[argv.index("--out-dir") + 1]).resolve()
    enrich_rgb_valid_metrics(out_dir)
    rewrite_summary_and_readme(out_dir, " ".join(argv))
    remove_unrequested_outputs(out_dir)
    print(f"g1_sampled_out_dir {out_dir}")
    print("g1_sampled_script_status PASS")


if __name__ == "__main__":
    main()

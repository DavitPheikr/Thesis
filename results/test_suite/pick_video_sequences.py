#!/usr/bin/env python
"""Rank test SEQUENCES by suitability for a prediction fly-through video.

Goal: pick whole sequences (car driving through, predictions overlaid) that are
GOOD candidates (model consistently does well) and BAD candidates (model
consistently struggles) -- not just lucky/unlucky individual frames.

How it works (CPU / file-only):
  * Reads a model's per-frame test results `frame_error_summary.csv`. For full
    coverage that file has ONE ROW PER PATCH, so we aggregate patches -> per FRAME
    (sum marking tp/fp/fn -> frame marking IoU; sum true_marking -> frame marking
    content).
  * A frame counts as "marking-bearing" only if it has enough true marking
    (`--min-marking`); IoU on near-empty frames is meaningless and is excluded
    from IoU stats. The FRACTION of marking-bearing frames is itself a key signal
    (a good video needs markings present throughout).
  * Per sequence we then report: how many frames bear marking, the median/spread
    of frame IoU over those frames, and how consistent it is (fraction of frames
    above `--good` / below `--bad`).
  * BEST candidates  = markings present in most frames AND consistently high IoU.
    WORST candidates = markings present in most frames (so errors are visible)
    AND consistently low IoU.

Usage:
  python results/test_suite/pick_video_sequences.py                 # G2, full coverage
  python results/test_suite/pick_video_sequences.py --model D0
  python results/test_suite/pick_video_sequences.py --min-marking 1500 --good 0.55 --bad 0.30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def repo_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / "src" / "thesis_pipeline").is_dir() and (p / "logs").is_dir():
            return p
    raise RuntimeError("repo root not found")


REPO = repo_root()

FOLDERS = {
    "D0": "D0_lidar",
    "E0": "E0_lidar_rgb",
    "F0": "F0_lidar_rgb_calibrated",
    "G2": "G2_lidar_rgb_lovasz",
    "H0": "H0_lidar_rgb_lovasz_jitter",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="G2", choices=list(FOLDERS))
    ap.add_argument("--coverage", default="full", choices=("full", "sampled"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-marking", type=int, default=2000,
                    help="per-frame (patch-summed) true-marking points to count a frame as marking-bearing")
    ap.add_argument("--good", type=float, default=0.55, help="frame IoU >= this counts as a 'good' frame")
    ap.add_argument("--bad", type=float, default=0.35, help="frame IoU <= this counts as a 'bad' frame")
    ap.add_argument("--min-frac-marking", type=float, default=0.40,
                    help="a sequence needs at least this fraction of marking-bearing frames to be a video candidate")
    ap.add_argument("--top", type=int, default=3, help="how many best/worst sequences to detail")
    return ap.parse_args()


def per_frame(df: pd.DataFrame, min_marking: int) -> pd.DataFrame:
    """Aggregate per-patch rows -> per-frame marking metrics."""
    g = (df.groupby(["seq_id", "frame_idx"], as_index=False)
           .agg(tp=("marking_tp", "sum"), fp=("marking_fp", "sum"),
                fn=("marking_fn", "sum"), true_marking=("true_marking", "sum"),
                n_patches=("marking_tp", "size")))
    denom = (g["tp"] + g["fp"] + g["fn"]).replace(0, np.nan)
    g["frame_iou"] = g["tp"] / denom
    g["has_marking"] = g["true_marking"] >= min_marking
    g["seq"] = g["seq_id"].astype(int).map(lambda s: f"{s:03d}")
    return g


def per_sequence(frames: pd.DataFrame, good: float, bad: float) -> pd.DataFrame:
    rows = []
    for seq, sub in frames.groupby("seq"):
        mk = sub[sub["has_marking"] & sub["frame_iou"].notna()]
        n_frames = len(sub)
        n_mark = len(mk)
        frac_mark = n_mark / n_frames if n_frames else 0.0
        if n_mark:
            iou = mk["frame_iou"]
            rows.append({
                "seq": seq,
                "n_frames": n_frames,
                "marking_frames": n_mark,
                "frac_marking": round(frac_mark, 3),
                "median_iou": round(float(iou.median()), 4),
                "mean_iou": round(float(iou.mean()), 4),
                "iou_p25": round(float(iou.quantile(0.25)), 4),
                "iou_p75": round(float(iou.quantile(0.75)), 4),
                "frac_good": round(float((iou >= good).mean()), 3),
                "frac_bad": round(float((iou <= bad).mean()), 3),
                "median_true_marking": int(mk["true_marking"].median()),
            })
        else:
            rows.append({"seq": seq, "n_frames": n_frames, "marking_frames": 0,
                         "frac_marking": round(frac_mark, 3), "median_iou": np.nan,
                         "mean_iou": np.nan, "iou_p25": np.nan, "iou_p75": np.nan,
                         "frac_good": np.nan, "frac_bad": np.nan, "median_true_marking": 0})
    return pd.DataFrame(rows)


def frame_trend(frames: pd.DataFrame, seq: str) -> str:
    sub = frames[frames["seq"] == seq].sort_values("frame_idx")
    parts = []
    for _, r in sub.iterrows():
        if not r["has_marking"] or pd.isna(r["frame_iou"]):
            parts.append(f"{int(r['frame_idx']):>2}:  --")  # no/low marking
        else:
            parts.append(f"{int(r['frame_idx']):>2}:{r['frame_iou']:.2f}")
    # wrap 8 per line
    lines = ["   " + "  ".join(parts[i:i + 8]) for i in range(0, len(parts), 8)]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    folder = FOLDERS[args.model]
    csv = REPO / "results" / "per_model" / folder / "test" / args.coverage / f"seed_{args.seed}" / "frame_error_summary.csv"
    if not csv.exists():
        raise SystemExit(f"frame CSV not found: {csv}")

    df = pd.read_csv(csv, dtype={"seq_id": str})
    frames = per_frame(df, args.min_marking)
    seqs = per_sequence(frames, args.good, args.bad)

    out_dir = REPO / "results" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs.sort_values("median_iou", ascending=False).to_csv(
        out_dir / f"video_sequence_suitability_{args.model}.csv", index=False)

    print(f"\n=== Per-sequence video suitability  (model {args.model}, {args.coverage} coverage) ===")
    print(f"   marking-bearing frame = >= {args.min_marking} true-marking pts (patch-summed); "
          f"good >= {args.good}, bad <= {args.bad}\n")
    show = seqs.sort_values("median_iou", ascending=False)
    print(show.to_string(index=False))

    # candidates must have markings present in enough frames
    cand = seqs[seqs["frac_marking"] >= args.min_frac_marking].dropna(subset=["median_iou"])

    best = cand.sort_values(["median_iou", "frac_good"], ascending=False).head(args.top)
    worst = cand.sort_values(["median_iou", "frac_good"], ascending=True).head(args.top)

    print(f"\n=== BEST video candidates (consistently high IoU, markings present) ===")
    print(best[["seq", "marking_frames", "frac_marking", "median_iou", "frac_good", "median_true_marking"]].to_string(index=False))
    print(f"\n=== WORST video candidates (consistently low IoU, markings present) ===")
    print(worst[["seq", "marking_frames", "frac_marking", "median_iou", "frac_bad", "median_true_marking"]].to_string(index=False))

    if len(best):
        s = best.iloc[0]["seq"]
        print(f"\n--- BEST #1 = seq {s}: per-frame IoU trend (-- = no/low marking) ---")
        print(frame_trend(frames, s))
    if len(worst):
        s = worst.iloc[0]["seq"]
        print(f"\n--- WORST #1 = seq {s}: per-frame IoU trend (-- = no/low marking) ---")
        print(frame_trend(frames, s))

    print(f"\nwrote {(out_dir / f'video_sequence_suitability_{args.model}.csv').relative_to(REPO)}")
    print("pick_video_sequences_status PASS")


if __name__ == "__main__":
    main()

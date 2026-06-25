#!/usr/bin/env python
"""Render full TEST sequences -> per-frame PNGs + MP4, for one or more modes.

For each (sequence, model, mode) it calls the overlay viewer ONCE for the whole
frame range (one model load, not one per frame), saves the per-frame PNGs, and
stitches them into an MP4 with ffmpeg (frames ordered by their real frame index).

Modes (`--modes`, default `plain,gt,pred` = all three):
  * plain  -- raw front camera, NO overlay and NO HUD ("normal" video)
  * gt     -- ground-truth marking/road/other painted on the camera
  * pred   -- the model's predictions painted on the camera
  * error  -- TP/FP/FN error map (optional)
Each mode -> its own MP4 in results/videos/<seq>/<model>/, e.g.
  101_G2_lidar_rgb_lovasz_plain.mp4 / _gt.mp4 / _pred.mp4
so one sequence folder holds the normal, ground-truth and prediction videos.

Default model is G2 (best); pass --models D0,G2 to also do the LiDAR baseline.
CPU for stitching; GPU for the render. RUN ON THE SERVER (needs checkpoints + GPU).

Usage:
  # all three videos (normal + gt + pred) for each sequence:
  python results/test_suite/make_sequence_video.py --sequences 101,002,065 --device cuda
  # just the normal (no-overlay) video:
  python results/test_suite/make_sequence_video.py --sequences 117 --modes plain
  python results/test_suite/make_sequence_video.py --sequences 101 --models D0,G2 --fps 12
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
VIEWER = HERE / "04_prediction_images.py"


def repo_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / "src" / "thesis_pipeline").is_dir() and (p / "logs").is_dir():
            return p
    raise RuntimeError("repo root not found")


REPO = repo_root()
VIDEOS = REPO / "results" / "videos"

# code -> (output-folder, run-dir (rel), selected epoch)
MODELS = {
    "D0": ("D0_lidar",                   "logs/milestone_d/runs/D0_weighted_ce_25ep",    18),
    "E0": ("E0_lidar_rgb",               "logs/milestone_e/runs/E0_rgb_front_v1",        14),
    "F0": ("F0_lidar_rgb_calibrated",    "logs/milestone_f/runs/F0_rgb_soft_weights",    13),
    "G2": ("G2_lidar_rgb_lovasz",        "logs/milestone_g/runs/G2_schedule_extend_100", 68),
    "H0": ("H0_lidar_rgb_lovasz_jitter", "logs/milestone_h/runs/H0_rgb_jitter",          37),
}
FRAME_RE = re.compile(r"_lidar(\d+)_")   # raw 04 output filenames
IMG_RE = re.compile(r"_f(\d+)_")         # our flattened images/ filenames


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", required=True, help="comma-separated, e.g. 101,002,065 (3-digit ok)")
    ap.add_argument("--models", default="G2", help="comma-separated model codes (default: G2)")
    ap.add_argument("--modes", default="plain,gt,pred",
                    help="comma-separated, any of plain,gt,pred,error. "
                         "Default renders all three: plain (raw camera), gt, pred.")
    ap.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--passes-per-frame", type=int, default=8, help="overlay coverage per frame")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0, help="1.0 = fully opaque")
    ap.add_argument("--force", action="store_true", help="re-render even if images already exist")
    return ap.parse_args()


def render_sequence(code: str, seq: str, mode: str, args, raw_dir: Path) -> None:
    folder, rd, ep = MODELS[code]
    run_dir = REPO / rd
    cfg = run_dir / "config_snapshot.yml"
    ckpt = run_dir / "checkpoints" / f"ckpt_epoch_{ep:05d}.pth"
    for f in (cfg, ckpt):
        if not f.exists():
            raise FileNotFoundError(f"{code}: missing {f}")
    cmd = [PY, str(VIEWER),
           "--config", str(cfg), "--run-dir", str(run_dir), "--checkpoint", str(ckpt),
           "--split", "test", "--sequence", seq,
           "--start-frame", str(args.start_frame), "--max-frames", str(args.max_frames),
           "--stride", str(args.stride), "--passes-per-frame", str(args.passes_per_frame),
           "--mode", mode, "--alpha", str(args.alpha),
           "--save-only", "--no-window", "--save-dir", str(raw_dir), "--device", args.device]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)


def collect_sorted(raw_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in raw_dir.rglob("*.png"):
        m = FRAME_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda t: t[0])


def stitch(sorted_frames: list[tuple[int, Path]], out_mp4: Path, fps: int) -> bool:
    if not shutil.which("ffmpeg"):
        print("[warn] ffmpeg not found -> images saved, video skipped. "
              "Install ffmpeg (e.g. `conda install -c conda-forge ffmpeg`) and re-run.")
        return False
    tmp = Path(tempfile.mkdtemp(prefix="vid_"))
    try:
        for i, (_idx, p) in enumerate(sorted_frames):
            shutil.copy(p, tmp / f"{i:05d}.png")
        cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp / "%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out_mp4)]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    args = parse_args()
    seqs = [f"{int(s.strip()):03d}" for s in args.sequences.split(",") if s.strip()]
    codes = [c.strip().upper() for c in args.models.split(",") if c.strip()]
    valid_modes = ("plain", "gt", "pred", "error")
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    for c in codes:
        if c not in MODELS:
            raise SystemExit(f"unknown model {c}; choose from {list(MODELS)}")
    for m in modes:
        if m not in valid_modes:
            raise SystemExit(f"unknown mode {m}; choose from {list(valid_modes)}")

    for code in codes:
        folder = MODELS[code][0]
        for seq in seqs:
            base = VIDEOS / seq / folder
            for mode in modes:
                # per-mode image dir so plain/gt/pred never clobber each other
                images = base / "images" / mode
                raw = base / "_raw" / mode
                out_mp4 = base / f"{seq}_{folder}_{mode}.mp4"

                if images.is_dir() and any(images.glob("*.png")) and not args.force:
                    print(f"\n=== {code} seq {seq} [{mode}]: images exist; reusing (use --force) ===")
                else:
                    print(f"\n=== {code} seq {seq} [{mode}]: rendering -> {raw.relative_to(REPO)} ===")
                    if raw.exists():
                        shutil.rmtree(raw)
                    render_sequence(code, seq, mode, args, raw)
                    frames = collect_sorted(raw)
                    if not frames:
                        print(f"[warn] no PNGs produced for {code} seq {seq} [{mode}]; skipping")
                        continue
                    images.mkdir(parents=True, exist_ok=True)
                    for idx, p in frames:
                        shutil.copy(p, images / f"{seq}_f{idx:03d}_{mode}.png")
                    shutil.rmtree(raw, ignore_errors=True)

                sorted_frames = sorted(
                    ((int(IMG_RE.search(p.name).group(1)), p)
                     for p in images.glob("*.png") if IMG_RE.search(p.name)),
                    key=lambda t: t[0])
                print(f"    {len(sorted_frames)} frames -> {out_mp4.relative_to(REPO)}")
                if stitch(sorted_frames, out_mp4, args.fps):
                    print(f"    video: {out_mp4.relative_to(REPO)}")

    print("\nmake_sequence_video_status PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stack a sequence's per-mode videos (plain / gt / pred) into ONE side-by-side MP4.

Takes the videos produced by make_sequence_video.py and lays them out in a single
clip so you can watch all three in sync (they are the same camera frames, same
length and fps, so they line up exactly). Each panel gets a label.

Default layout is horizontal (side by side); pass --layout vstack to stack them
vertically instead. CPU-only (ffmpeg) — no GPU, no model needed.

Usage:
  python results/test_suite/make_comparison_video.py --sequences 117,001,065
  python results/test_suite/make_comparison_video.py --sequences 117 --modes plain,pred
  python results/test_suite/make_comparison_video.py --sequences 117 --layout vstack
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def repo_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / "src" / "thesis_pipeline").is_dir() and (p / "logs").is_dir():
            return p
    raise RuntimeError("repo root not found")


REPO = repo_root()
VIDEOS = REPO / "results" / "videos"

# model code -> output-folder (matches make_sequence_video.MODELS)
CODE_TO_FOLDER = {
    "D0": "D0_lidar",
    "E0": "E0_lidar_rgb",
    "F0": "F0_lidar_rgb_calibrated",
    "G2": "G2_lidar_rgb_lovasz",
    "H0": "H0_lidar_rgb_lovasz_jitter",
}
MODE_LABELS = {"plain": "Camera", "gt": "Ground truth", "pred": "Prediction", "error": "Errors"}


def find_font() -> str | None:
    """Locate a TTF for ffmpeg drawtext. matplotlib ships DejaVuSans, so use it."""
    try:
        import matplotlib
        f = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if f.exists():
            return str(f)
    except Exception:
        pass
    for c in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(c).exists():
            return c
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", required=True, help="comma-separated, e.g. 117,001,065")
    ap.add_argument("--models", default="G2", help="comma-separated model codes (default: G2)")
    ap.add_argument("--modes", default="plain,gt,pred",
                    help="comma-separated order, left->right / top->bottom (default plain,gt,pred)")
    ap.add_argument("--layout", default="hstack", choices=("hstack", "vstack"))
    ap.add_argument("--panel-height", type=int, default=540,
                    help="scale each panel to this height (keeps aspect); 0 = keep original")
    ap.add_argument("--no-labels", action="store_true", help="do not draw the per-panel labels")
    ap.add_argument("--force", action="store_true", help="overwrite an existing compare video")
    return ap.parse_args()


def build_filter(n: int, modes: list[str], layout: str, panel_h: int, font: str | None) -> str:
    chains = []
    for i, mode in enumerate(modes):
        steps = []
        if panel_h > 0:
            steps.append(f"scale=-2:{panel_h}")
        if font:
            label = MODE_LABELS.get(mode, mode).replace(":", r"\:").replace("'", "")
            fs = max(16, (panel_h or 1080) // 16)
            steps.append(
                f"drawtext=fontfile='{font}':text='{label}':"
                f"x=(w-text_w)/2:y=8:fontsize={fs}:fontcolor=white:"
                f"box=1:boxcolor=black@0.5:boxborderw=8"
            )
        chain = f"[{i}:v]" + (",".join(steps) if steps else "null") + f"[v{i}]"
        chains.append(chain)
    joined = "".join(f"[v{i}]" for i in range(n))
    stack = "hstack" if layout == "hstack" else "vstack"
    chains.append(f"{joined}{stack}=inputs={n}[out]")
    return ";".join(chains)


def make_one(seq: str, code: str, modes: list[str], args, font: str | None) -> bool:
    folder = CODE_TO_FOLDER[code]
    base = VIDEOS / seq / folder
    present = [(m, base / f"{seq}_{folder}_{m}.mp4") for m in modes]
    present = [(m, p) for m, p in present if p.exists()]
    if len(present) < 2:
        have = [p.name for _m, p in present]
        print(f"[skip] {code} seq {seq}: need >=2 mode videos, found {have or 'none'} "
              f"(run make_sequence_video.py first)")
        return False

    use_modes = [m for m, _p in present]
    inputs = [p for _m, p in present]
    out = base / f"{seq}_{folder}_compare.mp4"
    if out.exists() and not args.force:
        print(f"[skip] {out.relative_to(REPO)} exists (use --force)")
        return True

    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", build_filter(len(inputs), use_modes, args.layout, args.panel_height, font),
            "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    print(f"\n=== {code} seq {seq}: {' + '.join(use_modes)} -> {out.relative_to(REPO)} ===")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"    wrote {out.relative_to(REPO)}")
    return True


def main() -> None:
    args = parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found (conda install -c conda-forge ffmpeg)")
    seqs = [f"{int(s.strip()):03d}" for s in args.sequences.split(",") if s.strip()]
    codes = [c.strip().upper() for c in args.models.split(",") if c.strip()]
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    for c in codes:
        if c not in CODE_TO_FOLDER:
            raise SystemExit(f"unknown model {c}; choose from {list(CODE_TO_FOLDER)}")

    font = None if args.no_labels else find_font()
    if not args.no_labels and font is None:
        print("[warn] no TTF font found -> rendering without labels")

    ok = True
    for code in codes:
        for seq in seqs:
            ok = make_one(seq, code, modes, args, font) and ok
    print("\nmake_comparison_video_status", "PASS" if ok else "PARTIAL")


if __name__ == "__main__":
    main()

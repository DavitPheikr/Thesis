#!/usr/bin/env python3
"""Run pwa_aco_hybrid_fast.py sequentially over every .tsp file in this directory.

Default mode: one run per .tsp file with --velocity --save-results (output goes to
experiments/ as fast.py's default).

Sweep mode (--seed): each .tsp file is run 30 times with seeds 42..71. Per-seed
output goes under results/<tspfile_stem>/seed_<N>/ ; an aggregate results.csv and
four plots are written under results/<tspfile_stem>/ once all 30 seeds finish.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "pwa_aco_hybrid_fast.py"
BASE_FLAGS = ["--velocity", "--save-results"]
RESULTS_ROOT = HERE / "results"
SEEDS = list(range(42, 72))  # 42..71 inclusive, 30 seeds

try:
    import psutil  # type: ignore

    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


class ProcSampler(threading.Thread):
    """Samples CPU% and RSS of a subprocess and tracks peaks.

    Uses psutil when available; otherwise it sleeps and reports None.
    cpu_percent() is process-local (sum across threads), 100% = one full core.
    """

    def __init__(self, pid: int, interval: float = 0.5):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.peak_cpu_pct: float | None = None
        self.peak_rss_mb: float | None = None
        self._stop = threading.Event()

    def run(self) -> None:
        if not HAVE_PSUTIL:
            return
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        # First call primes the cpu_percent counter; discard it.
        try:
            proc.cpu_percent(interval=None)
        except psutil.Error:
            return

        peak_cpu = 0.0
        peak_rss = 0.0
        while not self._stop.is_set():
            try:
                # Walk children too — fast.py spawns a multiprocessing Pool.
                procs = [proc] + proc.children(recursive=True)
                cpu = 0.0
                rss = 0
                for p in procs:
                    try:
                        cpu += p.cpu_percent(interval=None)
                        rss += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                if cpu > peak_cpu:
                    peak_cpu = cpu
                rss_mb = rss / (1024.0 * 1024.0)
                if rss_mb > peak_rss:
                    peak_rss = rss_mb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop.wait(self.interval)

        self.peak_cpu_pct = peak_cpu if peak_cpu > 0 else None
        self.peak_rss_mb = peak_rss if peak_rss > 0 else None

    def stop(self) -> None:
        self._stop.set()


def run_single(tsp_name: str, seed: int, out_dir: Path) -> tuple[int, ProcSampler]:
    """Run fast.py with the given seed; output goes to out_dir. Returns (returncode, sampler)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT),
        tsp_name,
        *BASE_FLAGS,
        "--seed",
        str(seed),
        "--output-dir",
        str(out_dir),
    ]
    p = subprocess.Popen(cmd, cwd=HERE)
    sampler = ProcSampler(p.pid, interval=0.5)
    sampler.start()
    rc = p.wait()
    sampler.stop()
    sampler.join(timeout=2.0)
    return rc, sampler


def load_metrics(out_dir: Path) -> dict | None:
    mp = out_dir / "metrics.json"
    if not mp.exists():
        return None
    try:
        with mp.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "seed",
        "hybrid_cost",
        "pwa_cost",
        "improvement_pct",
        "elapsed_seconds",
        "peak_memory_mb",
        "peak_cpu_pct",
        "n_ants",
        "n_iters",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def make_plots(
    tsp_stem: str, rows: list[dict], per_seed_metrics: dict[int, dict], out_dir: Path
) -> None:
    """Render 4 plots for the per-tsp aggregate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter to successful runs for the cost-based plots.
    successes = [
        (r, per_seed_metrics[r["seed"]])
        for r in rows
        if r.get("error") in ("", None) and r["seed"] in per_seed_metrics
    ]

    if not successes:
        print(f"  [plots] No successful runs for {tsp_stem}; skipping plots.")
        return

    costs = [r["hybrid_cost"] for r, _ in successes]
    best_idx = min(range(len(successes)), key=lambda i: costs[i])
    best_row, best_metrics = successes[best_idx]
    best_seed = best_row["seed"]
    best_tour = best_metrics.get("best_tour") or []
    best_history = best_metrics.get("history") or []

    # ── 1. Best-run route ─────────────────────────────────────────────────
    # Reload city coordinates from the .tsp file for the route plot.
    city_coords = read_tsp_coords(HERE / f"{tsp_stem}.tsp")
    if city_coords and best_tour:
        fig, ax = plt.subplots(figsize=(8, 8))
        xs = [c[0] for c in city_coords]
        ys = [c[1] for c in city_coords]
        ax.scatter(xs, ys, zorder=3)
        for idx, (x, y) in enumerate(city_coords):
            ax.text(x, y, f" {idx}", fontsize=7)
        n_t = len(best_tour)
        for k in range(n_t):
            a, b = best_tour[k], best_tour[(k + 1) % n_t]
            if a >= len(city_coords) or b >= len(city_coords):
                continue
            x1, y1 = city_coords[a]
            x2, y2 = city_coords[b]
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="steelblue"),
            )
        ax.set_title(
            f"{tsp_stem} — best route (seed {best_seed}, cost {best_row['hybrid_cost']:.4f})"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        plt.tight_layout()
        plt.savefig(out_dir / "best_run_route.png", dpi=180, bbox_inches="tight")
        plt.close()

    # ── 2. Best-run convergence ──────────────────────────────────────────
    if best_history:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(best_history, color="steelblue", linewidth=1.6)
        pwa = best_metrics.get("pwa_cost")
        if pwa is not None:
            ax.axhline(
                pwa, color="coral", linestyle="--", label=f"PWA best ({pwa:.4f})"
            )
            ax.legend()
        ax.set_title(f"{tsp_stem} — best run convergence (seed {best_seed})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Global best cost")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "best_run_convergence.png", dpi=180, bbox_inches="tight")
        plt.close()

    # ── 3. Distribution of best-of-run cost across seeds ────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        costs,
        vert=True,
        widths=0.4,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightsteelblue", alpha=0.6),
        meanprops=dict(color="firebrick", linestyle="--", linewidth=1.5),
        medianprops=dict(color="navy", linewidth=1.5),
    )
    jitter = np.random.RandomState(0).uniform(-0.08, 0.08, size=len(costs))
    ax.scatter(1 + jitter, costs, color="steelblue", alpha=0.7, s=28, zorder=3)
    mean_c = statistics.mean(costs)
    med_c = statistics.median(costs)
    std_c = statistics.stdev(costs) if len(costs) > 1 else 0.0
    txt = (
        f"n = {len(costs)}\n"
        f"best = {min(costs):.4f}\n"
        f"worst = {max(costs):.4f}\n"
        f"mean = {mean_c:.4f}\n"
        f"median = {med_c:.4f}\n"
        f"std = {std_c:.4f}"
    )
    ax.text(
        0.98,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.set_xticks([1])
    ax.set_xticklabels([tsp_stem])
    ax.set_ylabel("Hybrid best cost")
    ax.set_title(
        f"{tsp_stem} — distribution of best-of-run cost across {len(costs)} seeds"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ── 4. Convergence overlay of all seeds + mean/median ───────────────
    histories = [m.get("history") or [] for _, m in successes]
    histories = [h for h in histories if h]
    if histories:
        max_len = max(len(h) for h in histories)
        arr = np.full((len(histories), max_len), np.nan)
        for i, h in enumerate(histories):
            arr[i, : len(h)] = h
        mean_curve = np.nanmean(arr, axis=0)
        med_curve = np.nanmedian(arr, axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        for h in histories:
            ax.plot(h, color="steelblue", alpha=0.25, linewidth=0.9)
        ax.plot(mean_curve, color="firebrick", linewidth=2.0, label="mean")
        ax.plot(med_curve, color="navy", linewidth=1.5, linestyle="--", label="median")
        ax.set_title(f"{tsp_stem} — convergence overlay ({len(histories)} seeds)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Global best cost")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "convergence_overlay.png", dpi=180, bbox_inches="tight")
        plt.close()


def read_tsp_coords(path: Path) -> list[tuple[float, float]]:
    """Lightweight coord reader supporting both TSPLIB and the legacy plaintext format.

    Mirrors load_tsp() in fast.py — only the coordinates are needed here.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text()
    except OSError:
        return []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    is_tsplib = any(
        l.upper().startswith("NAME")
        or l.upper().startswith("TYPE")
        or l.upper() == "NODE_COORD_SECTION"
        for l in lines[:10]
    )
    coords: list[tuple[float, float]] = []
    if is_tsplib:
        reading = False
        for line in lines:
            up = line.upper()
            if up == "NODE_COORD_SECTION":
                reading = True
            elif up in ("EOF", "TOUR_SECTION"):
                break
            elif reading and line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    else:
        tokens = text.split()
        try:
            it = iter(tokens)
            next(it)  # name
            n = int(next(it))
            for _ in range(n):
                next(it)  # id
                x = float(next(it))
                y = float(next(it))
                coords.append((x, y))
        except (StopIteration, ValueError):
            return coords
    return coords


def sweep(tsp_files: list[Path]) -> int:
    if not HAVE_PSUTIL:
        print("  [warn] psutil not installed; peak CPU/memory will be blank in CSV.")
        print("         pip install psutil  to enable.")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    per_file_summary: list[
        tuple[str, float | None, int | None, float | None, float | None]
    ] = []
    total_failures: list[tuple[str, int, int]] = []
    total_runs = 0

    for fi, tsp in enumerate(tsp_files, start=1):
        stem = tsp.stem
        file_dir = RESULTS_ROOT / stem
        file_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"[file {fi}/{len(tsp_files)}] {tsp.name}  →  {file_dir}")
        print("=" * 60)

        rows: list[dict] = []
        per_seed_metrics: dict[int, dict] = {}

        for si, seed in enumerate(SEEDS, start=1):
            seed_dir = file_dir / f"seed_{seed}"
            print(
                f"  [{si:2d}/{len(SEEDS)}] seed={seed} → {seed_dir.relative_to(HERE)}"
            )
            t0 = time.time()
            rc, sampler = run_single(tsp.name, seed, seed_dir)
            wall = time.time() - t0
            total_runs += 1

            metrics = load_metrics(seed_dir)
            row: dict = {"seed": seed}

            if rc != 0 or metrics is None:
                row["error"] = f"exit={rc}" + (
                    "" if metrics is not None else "; no metrics.json"
                )
                row["elapsed_seconds"] = round(wall, 3)
                row["peak_cpu_pct"] = (
                    round(sampler.peak_cpu_pct, 2) if sampler.peak_cpu_pct else ""
                )
                row["peak_memory_mb"] = (
                    round(sampler.peak_rss_mb, 2) if sampler.peak_rss_mb else ""
                )
                total_failures.append((tsp.name, seed, rc))
                print(f"      FAILED (exit {rc})")
            else:
                per_seed_metrics[seed] = metrics
                # fast.py's internal peak (rusage) is for the parent only; psutil's
                # sum across children is more accurate. Take the larger value.
                fast_peak_mb = float(metrics.get("peak_memory_mb") or 0.0)
                psutil_peak_mb = sampler.peak_rss_mb or 0.0
                peak_mb = max(fast_peak_mb, psutil_peak_mb)
                row.update(
                    {
                        "hybrid_cost": metrics.get("hybrid_cost"),
                        "pwa_cost": (
                            metrics.get("pwa_cost")
                            if metrics.get("pwa_cost") is not None
                            else ""
                        ),
                        "improvement_pct": metrics.get("improvement_pct"),
                        "elapsed_seconds": round(
                            float(metrics.get("elapsed_seconds") or wall), 3
                        ),
                        "peak_memory_mb": round(peak_mb, 2) if peak_mb else "",
                        "peak_cpu_pct": (
                            round(sampler.peak_cpu_pct, 2)
                            if sampler.peak_cpu_pct
                            else ""
                        ),
                        "n_ants": metrics.get("n_ants"),
                        "n_iters": metrics.get("n_iters"),
                        "error": "",
                    }
                )
                print(
                    f"      cost={metrics.get('hybrid_cost'):.4f}  "
                    f"elapsed={row['elapsed_seconds']:.1f}s  "
                    f"peak_mem={row['peak_memory_mb']}MB  "
                    f"peak_cpu={row['peak_cpu_pct']}%"
                )

            rows.append(row)

        rows.sort(key=lambda r: r["seed"])
        csv_path = file_dir / "results.csv"
        write_csv(csv_path, rows)
        print(f"  wrote {csv_path.relative_to(HERE)}")

        try:
            make_plots(stem, rows, per_seed_metrics, file_dir)
            print(f"  wrote plots under {file_dir.relative_to(HERE)}/")
        except Exception as e:
            print(f"  [warn] plot generation failed for {stem}: {e}")

        successes = [r for r in rows if r.get("error") in ("", None)]
        if successes:
            costs = [r["hybrid_cost"] for r in successes]
            best = min(costs)
            best_seed = next(r["seed"] for r in successes if r["hybrid_cost"] == best)
            mean_c = statistics.mean(costs)
            std_c = statistics.stdev(costs) if len(costs) > 1 else 0.0
            per_file_summary.append((tsp.name, best, best_seed, mean_c, std_c))
        else:
            per_file_summary.append((tsp.name, None, None, None, None))

    print()
    print("=" * 60)
    print("SEED-SWEEP COMPLETE")
    print("=" * 60)
    print(f"Files processed: {len(tsp_files)}/{len(tsp_files)}")
    print(f"Total runs:      {total_runs}  (failures: {len(total_failures)})")
    print(f"Output root:     {RESULTS_ROOT}")
    print()
    print("Per-file best costs:")
    for name, best, best_seed, mean_c, std_c in per_file_summary:
        if best is None:
            print(f"  {name:30s}  (all 30 seeds failed)")
        else:
            print(
                f"  {name:30s}  best={best:.4f} (seed {best_seed}) | "
                f"mean={mean_c:.4f} ± {std_c:.4f}"
            )

    if total_failures:
        print()
        print(f"Failures ({len(total_failures)}):")
        for name, seed, rc in total_failures:
            print(f"  - {name} seed={seed} (exit {rc})")
        return 2
    return 0


def single_runs(tsp_files: list[Path]) -> int:
    """Legacy behavior: one run per .tsp with --velocity --save-results to experiments/."""
    print(f"Found {len(tsp_files)} .tsp file(s):")
    for f in tsp_files:
        print(f"  - {f.name}")
    print()

    done: set[str] = set()
    failures: list[tuple[str, int]] = []

    for idx, tsp in enumerate(tsp_files, start=1):
        if tsp.name in done:
            print(f"[{idx}/{len(tsp_files)}] SKIP (already ran): {tsp.name}")
            continue
        print(f"[{idx}/{len(tsp_files)}] RUN: {tsp.name}")
        print("-" * 60)
        cmd = [sys.executable, str(SCRIPT), tsp.name, *BASE_FLAGS]
        result = subprocess.run(cmd, cwd=HERE)
        done.add(tsp.name)
        if result.returncode != 0:
            print(f"[{idx}/{len(tsp_files)}] FAILED ({result.returncode}): {tsp.name}")
            failures.append((tsp.name, result.returncode))
        else:
            print(f"[{idx}/{len(tsp_files)}] OK: {tsp.name}")
        print()

    print("=" * 60)
    print(f"Completed {len(done)}/{len(tsp_files)}.")
    if failures:
        print(f"Failures ({len(failures)}):")
        for name, rc in failures:
            print(f"  - {name} (exit {rc})")
        return 2
    print("All runs completed successfully.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pwa_aco_hybrid_fast.py over all .tsp files in this dir."
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help=f"Sweep mode: run each .tsp 30 times with seeds {SEEDS[0]}..{SEEDS[-1]} "
        f"and aggregate per-tsp output under results/.",
    )
    args = parser.parse_args()

    if not SCRIPT.exists():
        print(f"ERROR: {SCRIPT.name} not found in {HERE}", file=sys.stderr)
        return 1

    tsp_files = sorted(HERE.glob("*.tsp"))
    if not tsp_files:
        print(f"No .tsp files found in {HERE}", file=sys.stderr)
        return 1

    if args.seed:
        return sweep(tsp_files)
    return single_runs(tsp_files)


if __name__ == "__main__":
    raise SystemExit(main())

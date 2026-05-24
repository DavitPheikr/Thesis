#!/usr/bin/env python3
"""Run pwa_aco_hybrid_fast.py sequentially over every .tsp file in this directory."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "pwa_aco_hybrid_fast.py"
FLAGS = ["--velocity", "--save-results"]


def main() -> int:
    if not SCRIPT.exists():
        print(f"ERROR: {SCRIPT.name} not found in {HERE}", file=sys.stderr)
        return 1

    tsp_files = sorted(HERE.glob("*.tsp"))
    if not tsp_files:
        print(f"No .tsp files found in {HERE}", file=sys.stderr)
        return 1

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
        cmd = [sys.executable, str(SCRIPT), tsp.name, *FLAGS]
        # cwd=HERE so the fast script sees relative paths the same way you would
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


if __name__ == "__main__":
    raise SystemExit(main())

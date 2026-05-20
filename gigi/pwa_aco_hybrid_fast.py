"""
PWA-ACO Hybrid Solver — Optimized build
========================================
Drop-in faster version of pwa_aco_hybrid_fixed.py.  Same algorithm,
same default arguments, same random.seed(42) — given the same input
file and CLI flags this should produce the same final cost (and the
same best tour, up to symmetry / numerical-noise-free reordering) as
the original implementation.

What changed (execution only, not logic):
    1. cost / pheromone / velocity / eta are NumPy float64 arrays
       instead of nested Python lists.
    2. All hot inner kernels are JIT-compiled with Numba @njit:
         - PWA walk + Snell router
         - 2-opt local search
         - tour cost
         - roulette-wheel weight sum + pick
         - pheromone evaporation + deposit
    3. PWA phase parallelised across CPU cores using multiprocessing
       (one job per chunk of (j, i) start pairs).  Results are
       sorted back into the original (j, i) order before being
       handed to ACO's warm-start, so the pheromone deposit
       sequence matches the single-process baseline bit-for-bit.
    4. ACO ants remain SEQUENTIAL inside an iteration so that the
       order of random.randrange / random.random / random.choice
       calls is identical to the original — same RNG stream, same
       tours, same convergence.
    5. matplotlib is lazy-imported (only when --save-results is on).

Requirements:
    pip install numpy numba
    (matplotlib only needed for --save-results)

Usage is unchanged:
    python pwa_aco_hybrid_fast.py input.tsp
    python pwa_aco_hybrid_fast.py input.tsp --velocity --ants 30 --iters 100
    python pwa_aco_hybrid_fast.py input.tsp --workers 32 --save-results

New flag:
    --workers   Number of worker processes for the PWA phase
                (default: os.cpu_count(), e.g. 32 on your server)
"""

import argparse
import math
import multiprocessing
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numba import njit


# ─────────────────────────────────────────────────────────────────────────────
#  JIT kernels
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _router_jit(normal_x, normal_y, v1, v2, inc_x, inc_y):
    """Snell-law refraction / reflection in 2D. Returns refracted (or
    reflected) direction vector components."""
    inc_norm = math.sqrt(inc_x * inc_x + inc_y * inc_y)
    nrm_norm = math.sqrt(normal_x * normal_x + normal_y * normal_y)

    inc_ux = inc_x / inc_norm
    inc_uy = inc_y / inc_norm
    nrm_ux = normal_x / nrm_norm
    nrm_uy = normal_y / nrm_norm

    cos_t1 = -(inc_ux * nrm_ux + inc_uy * nrm_uy)
    ratio = v2 / v1
    sin2_t2 = ratio * ratio * (1.0 - cos_t1 * cos_t1)

    if sin2_t2 > 1.0:
        return (inc_ux + 2.0 * cos_t1 * nrm_ux,
                inc_uy + 2.0 * cos_t1 * nrm_uy)

    par_x = ratio * (inc_ux + cos_t1 * nrm_ux)
    par_y = ratio * (inc_uy + cos_t1 * nrm_uy)

    dot = inc_ux * nrm_ux + inc_uy * nrm_uy
    if dot < 0.0:
        perp_dx = -nrm_ux
        perp_dy = -nrm_uy
    else:
        perp_dx = nrm_ux
        perp_dy = nrm_uy

    perp_scale = math.sqrt(1.0 - sin2_t2)
    return (par_x + perp_scale * perp_dx,
            par_y + perp_scale * perp_dy)


@njit(cache=True)
def _walk_jit(start_j, start_i, n, city, cost, velocity):
    """PWA greedy walk from edge (start_j -> start_i).  Returns
    (total_tour_cost, path_array) where path_array[k] = next city
    after k in the walk."""
    fix = np.zeros(n, dtype=np.int8)
    path = np.zeros(n, dtype=np.int64)

    fix[start_j] = 1
    fix[start_i] = 1
    path[start_j] = start_i

    prev = start_j
    curr = start_i
    walk_cost = 0.0

    remaining = n - 2
    while remaining > 0:
        min_time = 1e308
        best_pos = -1

        t_curr_x = city[curr, 0] - city[prev, 0]
        t_curr_y = city[curr, 1] - city[prev, 1]

        for i in range(n):
            if fix[i] != 0:
                continue

            if (city[curr, 1] - city[prev, 1]) * (city[curr, 1] - city[i, 1]) > 0.0:
                line_x, line_y = _router_jit(
                    1.0, 0.0,
                    velocity[prev, curr], velocity[curr, i],
                    t_curr_x, t_curr_y,
                )
            else:
                line_x, line_y = _router_jit(
                    0.0, 1.0,
                    velocity[prev, curr], velocity[curr, i],
                    t_curr_x, t_curr_y,
                )

            tx = city[i, 0] - city[curr, 0]
            ty = city[i, 1] - city[curr, 1]

            ln = math.sqrt(line_x * line_x + line_y * line_y)
            dx = abs((line_x * ty - line_y * tx) / ln)

            tx2 = -tx
            ty2 = -ty
            dy = abs((tx2 * line_x + ty2 * line_y) / ln)

            time_cand = (dx + dy) / velocity[curr, i]
            if time_cand < min_time:
                min_time = time_cand
                best_pos = i

        walk_cost += cost[curr, best_pos]
        fix[best_pos] = 1
        path[curr] = best_pos
        prev = curr
        curr = best_pos
        remaining -= 1

    path[curr] = start_j
    walk_cost += cost[curr, start_j]
    return walk_cost + cost[start_j, start_i], path


@njit(cache=True)
def _reconstruct_jit(start, path, n):
    """Trace path[] starting from `start` until we revisit a city.
    Returns the route array (length up to n+1, closing edge included)."""
    route = np.full(n + 1, -1, dtype=np.int64)
    route[0] = start
    visited = np.zeros(n, dtype=np.bool_)
    visited[start] = True
    cur = start
    length = 1

    for _ in range(n):
        nxt = path[cur]
        route[length] = nxt
        length += 1
        if nxt == start or visited[nxt]:
            break
        visited[nxt] = True
        cur = nxt

    return route[:length]


@njit(cache=True)
def two_opt_jit(tour, cost, passes):
    """First-improvement 2-opt — identical swap criterion and scan
    order to the original Python version."""
    n = len(tour)
    t = tour.copy()
    improved = True
    p = 0
    while improved and p < passes:
        improved = False
        p += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a = t[i]
                b = t[i + 1]
                c = t[j]
                if j + 1 < n:
                    d = t[j + 1]
                else:
                    d = t[0]
                if cost[a, c] + cost[b, d] < cost[a, b] + cost[c, d] - 1e-10:
                    lo = i + 1
                    hi = j
                    while lo < hi:
                        tmp = t[lo]
                        t[lo] = t[hi]
                        t[hi] = tmp
                        lo += 1
                        hi -= 1
                    improved = True
    return t


@njit(cache=True)
def tour_cost_jit(tour, cost):
    n = len(tour)
    total = 0.0
    for k in range(n):
        if k + 1 < n:
            total += cost[tour[k], tour[k + 1]]
        else:
            total += cost[tour[k], tour[0]]
    return total


@njit(cache=True)
def _roulette_total(visited, tau_row, eta_row, alpha, beta):
    """Sum of τ^α · η^β over unvisited cities — same accumulation
    order as the original (ascending city index)."""
    n = len(visited)
    total = 0.0
    for j in range(n):
        if not visited[j]:
            total += (tau_row[j] ** alpha) * (eta_row[j] ** beta)
    return total


@njit(cache=True)
def _roulette_pick(visited, tau_row, eta_row, alpha, beta, r_threshold):
    """Walks unvisited cities in ascending index order, accumulating
    weights; returns the first city where acc >= r_threshold.
    Falls back to the last unvisited city (matches original
    `chosen = cities[-1]` default)."""
    n = len(visited)
    acc = 0.0
    last_unvisited = -1
    for j in range(n):
        if visited[j]:
            continue
        last_unvisited = j
        acc += (tau_row[j] ** alpha) * (eta_row[j] ** beta)
        if acc >= r_threshold:
            return j
    return last_unvisited


@njit(cache=True)
def _update_pheromones_jit(tau, iter_tour, iter_cost,
                           best_tour, best_cost,
                           rho, Q, has_best):
    """Elitist Ant System pheromone update (same order as original):
        1) Global evaporation + 1e-10 floor.
        2) Deposit Q / iter_cost on every edge of iter_tour.
        3) If has_best, deposit Q / best_cost on every edge of best_tour."""
    n = tau.shape[0]
    one_minus_rho = 1.0 - rho
    for i in range(n):
        for j in range(n):
            tau[i, j] *= one_minus_rho
            if tau[i, j] < 1e-10:
                tau[i, j] = 1e-10

    dep_iter = Q / iter_cost
    n_it = len(iter_tour)
    for k in range(n_it):
        a = iter_tour[k]
        if k + 1 < n_it:
            b = iter_tour[k + 1]
        else:
            b = iter_tour[0]
        tau[a, b] += dep_iter
        tau[b, a] += dep_iter

    if has_best:
        dep_gb = Q / best_cost
        n_gb = len(best_tour)
        for k in range(n_gb):
            a = best_tour[k]
            if k + 1 < n_gb:
                b = best_tour[k + 1]
            else:
                b = best_tour[0]
            tau[a, b] += dep_gb
            tau[b, a] += dep_gb


# ─────────────────────────────────────────────────────────────────────────────
#  PWA — multiprocessing
# ─────────────────────────────────────────────────────────────────────────────

_PWA_CTX = {}


def _pwa_worker_init(n, city, cost, velocity):
    """Pool initializer — populates per-worker globals so we don't
    re-pickle the cost matrix for every chunk."""
    _PWA_CTX['n'] = n
    _PWA_CTX['city'] = city
    _PWA_CTX['cost'] = cost
    _PWA_CTX['velocity'] = velocity


def _pwa_worker(chunk):
    n = _PWA_CTX['n']
    city = _PWA_CTX['city']
    cost = _PWA_CTX['cost']
    velocity = _PWA_CTX['velocity']
    out = []
    for (j, i) in chunk:
        c, path = _walk_jit(j, i, n, city, cost, velocity)
        route = _reconstruct_jit(j, path, n)
        out.append((j, i, c, route.tolist()))
    return out


def solve_pwa_parallel(n, city, cost, velocity, n_workers, verbose):
    """Parallel PWA: every (j, i) start pair is independent.  We sort
    results back into the original (j, i) order before returning so
    that the downstream warm-start sees the same deposit sequence as
    the single-process baseline."""

    # Warm up the JIT cache in the parent first, so children load the
    # compiled artifact from disk instead of recompiling.
    _walk_jit(0, 1, n, city, cost, velocity)
    _reconstruct_jit(0, np.zeros(n, dtype=np.int64), n)

    pairs = [(j, i) for j in range(n) for i in range(n) if i != j]
    total = len(pairs)

    chunk_size = max(1, (total + n_workers * 4 - 1) // (n_workers * 4))
    chunks = [pairs[k:k + chunk_size] for k in range(0, total, chunk_size)]

    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(
        processes=n_workers,
        initializer=_pwa_worker_init,
        initargs=(n, city, cost, velocity),
    ) as pool:
        results = []
        for batch in pool.imap_unordered(_pwa_worker, chunks):
            results.extend(batch)

    results.sort(key=lambda x: (x[0], x[1]))

    all_tours: List[Tuple[float, List[int]]] = []
    best_cost = float("inf")
    best_tour: List[int] = []
    for (_j, _i, c, tour) in results:
        all_tours.append((c, tour))
        if c < best_cost:
            best_cost = c
            best_tour = list(tour)

    if verbose:
        print(f"  [PWA] best cost: {best_cost:.6f}  ({len(all_tours)} tours)")

    return best_cost, best_tour, all_tours


# ─────────────────────────────────────────────────────────────────────────────
#  ACO — sequential ants so the RNG stream matches the original
# ─────────────────────────────────────────────────────────────────────────────

class ACO:
    def __init__(self, n, cost, n_ants=20, n_iters=80,
                 alpha=1.0, beta=2.5, rho=0.1, Q=1.0,
                 tau_init=1.0, seed=42):
        self.n = n
        self.cost = cost
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        random.seed(seed)

        self.tau = np.full((n, n), tau_init, dtype=np.float64)

        self.eta = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j and cost[i, j] > 1e-12:
                    self.eta[i, j] = 1.0 / cost[i, j]

        self.best_cost = float("inf")
        self.best_tour = np.empty(0, dtype=np.int64)
        self.history: List[float] = []

    def warm_start_pheromone(self, pwa_tours, elite_pct=20.0, boost=5.0):
        if not pwa_tours:
            return
        pwa_best = min(c for c, _ in pwa_tours)
        threshold = pwa_best * (1.0 + elite_pct / 100.0)
        elite = [(c, t) for c, t in pwa_tours if c <= threshold]
        if not elite:
            return
        for cost_val, tour in elite:
            deposit = boost * self.Q / cost_val
            n_nodes = len(tour) - 1
            for k in range(n_nodes):
                i = tour[k]
                j = tour[(k + 1) % n_nodes]
                if 0 <= i < self.n and 0 <= j < self.n:
                    self.tau[i, j] += deposit
                    self.tau[j, i] += deposit

    def _build_tour(self):
        n = self.n
        visited = np.zeros(n, dtype=np.bool_)

        # ─── identical RNG call sequence to original ──────────────────
        start = random.randrange(n)
        tour = [start]
        visited[start] = True
        current = start

        for _ in range(n - 1):
            tau_row = self.tau[current]
            eta_row = self.eta[current]
            total = _roulette_total(visited, tau_row, eta_row,
                                    self.alpha, self.beta)

            if total < 1e-15:
                # Match original fallback path EXACTLY: no random.random()
                # consumed here, only a random.choice() over unvisited
                # cities in ascending-index order.
                unvisited = [j for j in range(n) if not visited[j]]
                chosen = random.choice(unvisited)
            else:
                r = random.random()
                chosen = _roulette_pick(visited, tau_row, eta_row,
                                        self.alpha, self.beta, r * total)
                if chosen < 0:
                    # Numerical safety — should be unreachable
                    for j in range(n - 1, -1, -1):
                        if not visited[j]:
                            chosen = j
                            break

            tour.append(int(chosen))
            visited[chosen] = True
            current = chosen

        return tour

    def run(self, verbose=True):
        for it in range(1, self.n_iters + 1):
            iter_best_cost = float("inf")
            iter_best_tour = None

            for _ in range(self.n_ants):
                tour_list = self._build_tour()
                tour_arr = np.asarray(tour_list, dtype=np.int64)
                tour_arr = two_opt_jit(tour_arr, self.cost, 2)
                c = float(tour_cost_jit(tour_arr, self.cost))

                if c < iter_best_cost:
                    iter_best_cost = c
                    iter_best_tour = tour_arr.copy()

            if iter_best_cost < self.best_cost:
                self.best_cost = iter_best_cost
                self.best_tour = iter_best_tour.copy()

            has_best = self.best_tour.size > 0
            best_tour_arr = self.best_tour if has_best else np.zeros(1, dtype=np.int64)
            _update_pheromones_jit(
                self.tau,
                iter_best_tour, iter_best_cost,
                best_tour_arr, self.best_cost,
                self.rho, self.Q, has_best,
            )

            self.history.append(self.best_cost)

            if verbose:
                print(f"  [ACO] iter {it:4d}/{self.n_iters}  "
                      f"iter_best={iter_best_cost:.6f}  "
                      f"global_best={self.best_cost:.6f}")

        return self.best_tour, self.best_cost


# ─────────────────────────────────────────────────────────────────────────────
#  File I/O — same format support, returns numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def load_tsp(file_path, use_file_velocity=False):
    text = Path(file_path).read_text()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    is_tsplib = any(
        l.upper().startswith("NAME") or
        l.upper().startswith("TYPE") or
        l.upper() == "NODE_COORD_SECTION"
        for l in lines[:10]
    )

    if is_tsplib:
        filename = Path(file_path).stem
        n = 0
        city: List[Tuple[float, float]] = []
        reading_coords = False
        for line in lines:
            up = line.upper()
            if up.startswith("NAME"):
                filename = line.split(":")[-1].strip()
            elif up.startswith("DIMENSION"):
                n = int(line.split(":")[-1].strip())
            elif up == "NODE_COORD_SECTION":
                reading_coords = True
            elif up in ("EOF", "TOUR_SECTION"):
                break
            elif reading_coords and line and line[0].isdigit():
                parts = line.split()
                city.append((float(parts[1]), float(parts[2])))

        if len(city) != n:
            raise ValueError(f"TSPLIB parse error: expected {n} cities, got {len(city)}")

        velocity = np.ones((n, n), dtype=np.float64)
        cost = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt((city[i][0] - city[j][0]) ** 2 +
                              (city[i][1] - city[j][1]) ** 2)
                cost[i, j] = d
                cost[j, i] = d
    else:
        tokens = text.split()
        it = iter(tokens)
        filename = next(it)
        n = int(next(it))
        city = []
        for _ in range(n):
            _id = int(next(it))
            x = float(next(it))
            y = float(next(it))
            city.append((x, y))

        velocity = np.zeros((n, n), dtype=np.float64)
        cost = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                v = float(next(it)) if use_file_velocity else 1.0
                if v <= 0:
                    raise ValueError(f"Velocity[{i}][{j}] must be positive.")
                velocity[i, j] = v
                velocity[j, i] = v
                d = math.sqrt((city[i][0] - city[j][0]) ** 2 +
                              (city[i][1] - city[j][1]) ** 2)
                cost[i, j] = d / v
                cost[j, i] = d / v

    city_arr = np.array(city, dtype=np.float64)
    return filename, n, city_arr, cost, velocity


# ─────────────────────────────────────────────────────────────────────────────
#  Results saving (matplotlib imported lazily here)
# ─────────────────────────────────────────────────────────────────────────────

def save_results(filename, n, city, pwa_cost, final_cost, best_tour,
                 aco_history, elapsed, out_dir="experiments"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = out / f"pwa_aco_summary_{ts}.txt"
    improvement = (pwa_cost - final_cost) / pwa_cost * 100 if pwa_cost > 0 else 0.0

    with summary.open("w") as f:
        f.write("PWA-ACO Hybrid  —  run summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset   : {filename}\n")
        f.write(f"Cities    : {n}\n")
        f.write(f"Elapsed   : {elapsed:.1f}s\n\n")
        f.write(f"PWA best  : {pwa_cost:.6f}\n")
        f.write(f"Hybrid    : {final_cost:.6f}\n")
        f.write(f"Improvement: {improvement:+.3f}%\n\n")
        f.write("ACO convergence history (best per iteration):\n")
        for k, v in enumerate(aco_history, 1):
            f.write(f"  iter {k:4d}: {v:.6f}\n")
        f.write("\nBest tour:\n")
        f.write(" -> ".join(map(str, best_tour)) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    xs = [p[0] for p in city]
    ys = [p[1] for p in city]
    ax.scatter(xs, ys, zorder=3)
    for idx, (x, y) in enumerate(city):
        ax.text(x, y, f" {idx}", fontsize=8)
    tour = best_tour
    for k in range(len(tour)):
        a, b = tour[k], tour[(k + 1) % len(tour)]
        x1, y1 = city[a]
        x2, y2 = city[b]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="steelblue"))
    ax.set_title(f"Best route\ncost = {final_cost:.6f}  (PWA: {pwa_cost:.6f})")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.axis("equal")

    ax2 = axes[1]
    ax2.plot(aco_history, color="steelblue", linewidth=1.5)
    ax2.axhline(pwa_cost, color="coral", linestyle="--",
                label=f"PWA best ({pwa_cost:.4f})")
    ax2.set_title("ACO convergence")
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Best cost")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = out / f"pwa_aco_plot_{ts}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  Saved summary: {summary}")
    print(f"  Saved plot   : {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WORKERS = os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(
        description="PWA-ACO Hybrid Solver (optimized — same logic, much faster)"
    )
    parser.add_argument("input_file", nargs="?", default="TestDataVelocity2.tsp")
    parser.add_argument("--velocity", "--veloctity",
                        dest="use_file_velocity", action="store_true")
    parser.add_argument("--ants",      type=int,   default=20)
    parser.add_argument("--iters",     type=int,   default=80)
    parser.add_argument("--alpha",     type=float, default=1.0)
    parser.add_argument("--beta",      type=float, default=2.5)
    parser.add_argument("--rho",       type=float, default=0.1)
    parser.add_argument("--q",         type=float, default=1.0)
    parser.add_argument("--elite-pct", type=float, default=20.0, dest="elite_pct")
    parser.add_argument("--tau-init",  type=float, default=1.0, dest="tau_init")
    parser.add_argument("--pwa-boost", type=float, default=5.0, dest="pwa_boost")
    parser.add_argument("--skip-pwa",  action="store_true", dest="skip_pwa")
    parser.add_argument("--quiet",     action="store_true")
    parser.add_argument("--save-results", action="store_true", dest="save_results")
    parser.add_argument("--workers",   type=int, default=DEFAULT_WORKERS,
                        help=f"PWA worker processes (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    verbose = not args.quiet
    t0 = time.time()

    print(f"Loading: {args.input_file}")
    filename, n, city, cost, velocity = load_tsp(args.input_file, args.use_file_velocity)
    print(f"  {filename}  |  {n} cities")

    pwa_cost = float("inf")
    pwa_tours: List[Tuple[float, List[int]]] = []

    if not args.skip_pwa:
        print(f"\n{'─'*50}")
        print(f"Phase 1: Photon Walk Algorithm  ({args.workers} workers)")
        print(f"  Trying all {n*(n-1)} ordered start pairs …")
        pwa_cost, _pwa_best_tour, pwa_tours = solve_pwa_parallel(
            n, city, cost, velocity,
            n_workers=args.workers,
            verbose=verbose,
        )
        print(f"  PWA best cost: {pwa_cost:.6f}  ({len(pwa_tours)} tours generated)")
    else:
        print("\n[--skip-pwa]  Skipping PWA phase.")

    print(f"\n{'─'*50}")
    print(f"Phase 2+3: ACO  ({args.ants} ants × {args.iters} iterations)")
    if pwa_tours:
        print(f"  Warm-starting pheromone from {args.elite_pct:.0f}% elite PWA tours "
              f"(boost ×{args.pwa_boost})")

    aco = ACO(
        n=n, cost=cost,
        n_ants=args.ants, n_iters=args.iters,
        alpha=args.alpha, beta=args.beta,
        rho=args.rho, Q=args.q,
        tau_init=args.tau_init,
    )
    if pwa_tours:
        aco.warm_start_pheromone(pwa_tours, args.elite_pct, args.pwa_boost)

    best_tour, best_cost = aco.run(verbose=verbose)

    print(f"\n{'─'*50}")
    print("Phase 4: final 2-opt polish …")
    best_tour = two_opt_jit(best_tour, cost, 5)
    best_cost = float(tour_cost_jit(best_tour, cost))

    elapsed = time.time() - t0
    improvement = (pwa_cost - best_cost) / pwa_cost * 100 if pwa_cost < float("inf") else 0.0

    print(f"\n{'='*50}")
    print(f"  Dataset   : {filename}")
    print(f"  PWA best  : {pwa_cost:.6f}")
    print(f"  Hybrid    : {best_cost:.6f}  ({improvement:+.3f}% vs PWA)")
    print(f"  Elapsed   : {elapsed:.1f}s")
    print(f"{'='*50}")

    if args.save_results:
        save_results(filename, n, city, pwa_cost, best_cost,
                     best_tour.tolist(), aco.history, elapsed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

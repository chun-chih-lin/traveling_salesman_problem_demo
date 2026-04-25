"""Exact TSP (Held–Karp) from a JSON `positions` list; save optimal tour figure + JSON.

Reads e.g. simulation/two_regular_20/two_regular_20nodes_simulation.json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

COORD_LIM = (0.0, 10.0)


def positions_to_dist_matrix(positions: list[dict]) -> tuple[list[tuple[float, float]], int]:
    """Sort by id; return coords list (index 0..n-1 == id 1..n) and n."""
    ordered = sorted(positions, key=lambda p: p["id"])
    n = len(ordered)
    coords = [(float(p["x"]), float(p["y"])) for p in ordered]
    return coords, n


def dist_matrix(coords: list[tuple[float, float]]) -> list[list[float]]:
    n = len(coords)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i][j] = math.hypot(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1])
    return d


def held_karp(coords: list[tuple[float, float]]) -> tuple[float, list[int]]:
    """Minimum Hamilton cycle length and vertex visit order (0-based indices), fixed start 0."""
    n = len(coords)
    if n < 3:
        raise ValueError("Need at least 3 points for TSP.")
    d = dist_matrix(coords)
    size = 1 << n
    inf = float("inf")
    dp = [inf] * (size * n)
    par = [-1] * (size * n)

    def idx(mask: int, j: int) -> int:
        return mask * n + j

    dp[idx(1, 0)] = 0.0

    for mask in range(1, size):
        if not (mask & 1):
            continue
        for j in range(n):
            if not (mask >> j) & 1:
                continue
            cur = dp[idx(mask, j)]
            if cur >= inf:
                continue
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nmask = mask | (1 << k)
                nd = cur + d[j][k]
                ii = idx(nmask, k)
                if nd < dp[ii]:
                    dp[ii] = nd
                    par[ii] = j

    full = size - 1
    best = inf
    best_j = -1
    for j in range(1, n):
        val = dp[idx(full, j)] + d[j][0]
        if val < best:
            best = val
            best_j = j

    # Reconstruct order ending at best_j before return to 0
    tour_rev: list[int] = []
    mask = full
    j = best_j
    while True:
        tour_rev.append(j)
        pj = par[idx(mask, j)]
        if j == 0 and mask == 1:
            break
        mask ^= 1 << j
        j = pj

    tour_0based = list(reversed(tour_rev))
    return best, tour_0based


def tour_length_from_order(coords: list[tuple[float, float]], order: list[int]) -> float:
    n = len(order)
    s = 0.0
    for i in range(n):
        a, b = order[i], order[(i + 1) % n]
        s += math.hypot(coords[b][0] - coords[a][0], coords[b][1] - coords[a][1])
    return s


def draw_optimal_tour(
    positions: list[dict],
    tour_0based: list[int],
    optimal_length: float,
    output_path: Path,
) -> None:
    ordered = sorted(positions, key=lambda p: p["id"])
    pmap = {p["id"]: p for p in positions}
    tour_ids = [ordered[i]["id"] for i in tour_0based]
    n = len(tour_ids)

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    fig.patch.set_edgecolor("#00FF00")
    fig.patch.set_linewidth(12)
    ax = plt.gca()
    ax.set_facecolor("black")

    for i in range(n):
        u, v = tour_ids[i], tour_ids[(i + 1) % n]
        pu, pv = pmap[u], pmap[v]
        plt.plot([pu["x"], pv["x"]], [pu["y"], pv["y"]], color="#00FF88", linewidth=2.0)

    xs = [p["x"] for p in positions]
    ys = [p["y"] for p in positions]
    plt.scatter(xs, ys, color="#00FFFF", s=80, zorder=5)
    for p in positions:
        plt.annotate(
            f"P{p['id']}",
            (p["x"], p["y"]),
            xytext=(3, 3),
            textcoords="offset points",
            color="#FFFF00",
            fontsize=8,
            zorder=6,
        )

    lo, hi = COORD_LIM
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    plt.title(f"{optimal_length:.6f}", color="white", fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def solve_and_save_optimal_tsp(
    positions: list[dict],
    out_dir: Path,
    output_png: Path = None,
    output_result_json: Path = None,
) -> dict:
    """Compute Held–Karp optimal TSP, save figure + JSON next to out_dir."""
    coords, _n = positions_to_dist_matrix(positions)
    optimal_len, tour_0 = held_karp(coords)
    chk = tour_length_from_order(coords, tour_0)
    if abs(chk - optimal_len) > 1e-3:
        optimal_len = chk

    ordered = sorted(positions, key=lambda p: p["id"])
    tour_one_based = [ordered[i]["id"] for i in tour_0]

    out_dir.mkdir(parents=True, exist_ok=True)
    if output_png is None:
        output_png = out_dir / "two_opt_steps" / "cycle_01" / "tsp_optimal.png"
    if output_result_json is None:
        output_result_json = out_dir / "two_opt_steps" / "cycle_01" / "tsp_optimal.json"

    result = {
        "optimal_length": round(optimal_len, 6),
        "tour_vertex_order_one_based": tour_one_based,
        "tour_vertex_order_zero_based": list(tour_0),
    }

    draw_optimal_tour(positions, tour_0, optimal_len, output_png)
    output_result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["output_png"] = str(output_png)
    result["output_json"] = str(output_result_json)
    return result


def run_from_json(
    input_json: Path,
    output_png: Path = None,
    output_result_json: Path = None,
) -> dict:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    positions = data["positions"]
    result = solve_and_save_optimal_tsp(
        positions,
        input_json.parent,
        output_png=output_png,
        output_result_json=output_result_json,
    )
    result["source_json"] = str(input_json)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact TSP from JSON positions (Held–Karp).")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("simulation") / "two_regular_20" / "two_regular_20nodes_simulation.json",
        help="JSON file with a 'positions' array.",
    )
    p.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Output figure path (default: next to input, tsp_optimal.png).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output result JSON (default: next to input, tsp_optimal.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    r = run_from_json(args.input, args.output_png, args.output_json)
    print(f"Optimal TSP length: {r['optimal_length']}")
    print(f"Saved: {r['output_png']}")
    print(f"Saved: {r['output_json']}")


if __name__ == "__main__":
    main()

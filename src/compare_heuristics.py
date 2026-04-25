import argparse
import json
import math
import random
from itertools import combinations, permutations
from pathlib import Path


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def generate_points(num_points: int, low: float = 0.0, high: float = 10.0) -> list[tuple[float, float]]:
    return [(random.uniform(low, high), random.uniform(low, high)) for _ in range(num_points)]


def tour_distance(points: list[tuple[float, float]], tour: tuple[int, ...]) -> float:
    return sum(euclidean_distance(points[a], points[b]) for a, b in zip(tour, tour[1:]))


def optimal_distance(points: list[tuple[float, float]]) -> float:
    n = len(points)
    best = float("inf")
    for perm in permutations(range(1, n)):
        # Fix start node at 0 and remove reverse duplicates.
        if perm[0] > perm[-1]:
            continue
        tour = (0, *perm, 0)
        dist = tour_distance(points, tour)
        if dist < best:
            best = dist
    return best


def nearest_neighbor_distance(points: list[tuple[float, float]], start_idx: int) -> tuple[float, list[int]]:
    n = len(points)
    unvisited = set(range(n))
    unvisited.remove(start_idx)
    path = [start_idx]
    current = start_idx
    total = 0.0

    while unvisited:
        nxt = min(unvisited, key=lambda idx: euclidean_distance(points[current], points[idx]))
        total += euclidean_distance(points[current], points[nxt])
        path.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    total += euclidean_distance(points[current], points[start_idx])
    path.append(start_idx)
    return total, path


def _find(parent: dict[int, int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union(parent: dict[int, int], a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def greedy_distance(points: list[tuple[float, float]]) -> tuple[float, list[tuple[int, int]]]:
    n = len(points)
    edges = []
    for a, b in combinations(range(n), 2):
        edges.append((a, b, euclidean_distance(points[a], points[b])))
    edges.sort(key=lambda e: e[2])

    parent = {i: i for i in range(n)}
    degree = {i: 0 for i in range(n)}
    chosen: list[tuple[int, int]] = []
    total = 0.0

    for a, b, dist in edges:
        if degree[a] >= 2 or degree[b] >= 2:
            continue

        is_last_edge = len(chosen) == n - 1
        creates_cycle = _find(parent, a) == _find(parent, b)
        if creates_cycle and not is_last_edge:
            continue

        chosen.append((a, b))
        degree[a] += 1
        degree[b] += 1
        total += dist

        if not creates_cycle:
            _union(parent, a, b)

        if len(chosen) == n:
            break

    return total, chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Nearest Neighbor and Greedy TSP distances "
            "against brute-force optimal across multiple trials."
        )
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of random trials to run (default: 10).",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=6,
        help="Number of points per trial (default: 6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = args.runs
    num_points = args.points
    if runs <= 0:
        raise ValueError("--runs must be a positive integer.")
    if num_points < 3:
        raise ValueError("--points must be at least 3.")

    all_results: list[dict] = []

    print(f"Running {runs} trials with {num_points} random points each...")

    for run_idx in range(1, runs + 1):
        points = generate_points(num_points)
        nn_start = random.randrange(num_points)
        nn_dist, nn_path = nearest_neighbor_distance(points, nn_start)
        gr_dist, gr_edges = greedy_distance(points)
        op_dist = optimal_distance(points)

        result = {
            "run": run_idx,
            "num_points": num_points,
            "nearest_neighbor": {
                "start_node_zero_based": nn_start,
                "distance": round(nn_dist, 6),
                "path_zero_based": nn_path,
            },
            "greedy": {
                "distance": round(gr_dist, 6),
                "edges_zero_based": gr_edges,
            },
            "optimal": {"distance": round(op_dist, 6)},
            "ratios_vs_optimal": {
                "nearest_neighbor": round(nn_dist / op_dist, 6),
                "greedy": round(gr_dist / op_dist, 6),
            },
            "points": [{"x": round(x, 6), "y": round(y, 6)} for x, y in points],
        }
        all_results.append(result)

        print(
            f"Run {run_idx:02d} | "
            f"NN={nn_dist:.3f} | Greedy={gr_dist:.3f} | Optimal={op_dist:.3f} | "
            f"NN/Opt={nn_dist / op_dist:.3f} | Greedy/Opt={gr_dist / op_dist:.3f}"
        )

    avg_nn = sum(r["nearest_neighbor"]["distance"] for r in all_results) / runs
    avg_gr = sum(r["greedy"]["distance"] for r in all_results) / runs
    avg_op = sum(r["optimal"]["distance"] for r in all_results) / runs
    avg_nn_ratio = sum(r["ratios_vs_optimal"]["nearest_neighbor"] for r in all_results) / runs
    avg_gr_ratio = sum(r["ratios_vs_optimal"]["greedy"] for r in all_results) / runs

    print("-" * 80)
    print(f"Average NN distance     : {avg_nn:.3f}")
    print(f"Average Greedy distance : {avg_gr:.3f}")
    print(f"Average Optimal distance: {avg_op:.3f}")
    print(f"Average NN/Optimal      : {avg_nn_ratio:.3f}")
    print(f"Average Greedy/Optimal  : {avg_gr_ratio:.3f}")

    output_dir = Path("images")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"heuristic_comparison_{num_points}_points_{runs}_runs.json"
    payload = {
        "runs": runs,
        "num_points": num_points,
        "averages": {
            "nearest_neighbor_distance": round(avg_nn, 6),
            "greedy_distance": round(avg_gr, 6),
            "optimal_distance": round(avg_op, 6),
            "nearest_neighbor_over_optimal": round(avg_nn_ratio, 6),
            "greedy_over_optimal": round(avg_gr_ratio, 6),
        },
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved comparison file: {output_path.resolve()}")


if __name__ == "__main__":
    main()

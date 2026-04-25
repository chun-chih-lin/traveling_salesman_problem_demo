"""Build a Minimum Spanning Tree from points in tsp_routes.json (Kruskal).

Reads the JSON produced by generate_random_points.py, runs Kruskal's algorithm,
and saves one figure per MST edge addition (plus an initial empty graph).
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MST (Kruskal) from tsp_routes.json with per-step figures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("images") / "tsp_routes.json",
        help="Path to JSON from generate_random_points.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("images") / "mst_steps",
        help="Directory for step images and mst_result.json.",
    )
    return parser.parse_args()


def load_points(json_path: Path) -> list[dict]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    points = data.get("points")
    if not points:
        raise ValueError("JSON must contain a non-empty 'points' list.")
    return points


def dist(p1: dict, p2: dict) -> float:
    return math.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"])


def find(parent: dict[int, int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent: dict[int, int], a: int, b: int) -> None:
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[rb] = ra


def kruskal_steps(points: list[dict]) -> list[dict]:
    """Return list of steps; each step is {edge: (u,v), weight, mst_edges so far}."""
    ids = [p["id"] for p in points]
    point_map = {p["id"]: p for p in points}
    edges: list[tuple[int, int, float]] = []
    for u, v in combinations(ids, 2):
        w = dist(point_map[u], point_map[v])
        edges.append((u, v, w))
    edges.sort(key=lambda e: e[2])

    parent = {i: i for i in ids}
    mst_edges: list[tuple[int, int, float]] = []
    steps: list[dict] = []

    for u, v, w in edges:
        if find(parent, u) == find(parent, v):
            continue
        union(parent, u, v)
        mst_edges.append((u, v, w))
        steps.append(
            {
                "step": len(mst_edges),
                "edge": {"u": u, "v": v, "weight": round(w, 6)},
                "mst_edges": [
                    {"u": a, "v": b, "weight": round(c, 6)} for a, b, c in mst_edges
                ],
                "cumulative_weight": round(sum(c for _, _, c in mst_edges), 6),
            }
        )
        if len(mst_edges) == len(ids) - 1:
            break

    return steps


def draw_step(
    points: list[dict],
    mst_edges: list[tuple[int, int, float]],
    step_index: int,
    title_suffix: str,
    output_path: Path,
    highlight_last: bool = False,
) -> None:
    point_map = {p["id"]: p for p in points}
    x_vals = [p["x"] for p in points]
    y_vals = [p["y"] for p in points]

    plt.figure(figsize=(6, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    # Optional faint complete graph for context
    for p1, p2 in combinations(points, 2):
        plt.plot(
            [p1["x"], p2["x"]],
            [p1["y"], p2["y"]],
            color="white",
            linewidth=0.5,
            alpha=0.25,
        )

    for idx, (u, v, w) in enumerate(mst_edges):
        pu, pv = point_map[u], point_map[v]
        is_last = highlight_last and idx == len(mst_edges) - 1
        lw = 3.2 if is_last else 2.2
        col = "#FFFF00" if is_last else "#00FF88"
        plt.plot([pu["x"], pv["x"]], [pu["y"], pv["y"]], color=col, linewidth=lw)

    plt.scatter(x_vals, y_vals, color="#00FFFF", s=120, zorder=5)
    for p in points:
        plt.annotate(
            f"P{p['id']}",
            (p["x"], p["y"]),
            xytext=(5, 5),
            textcoords="offset points",
            color="#FFFF00",
            zorder=6,
        )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    total_w = sum(e[2] for e in mst_edges) if mst_edges else 0.0
    # plt.title(
    #     f"MST Kruskal {title_suffix}\nStep {step_index} | Edges: {len(mst_edges)} | "
    #     f"Weight: {total_w:.3f}",
    #     color="white",
    #     fontsize=16,
    # )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def draw_initial(points: list[dict], output_path: Path) -> None:
    x_vals = [p["x"] for p in points]
    y_vals = [p["y"] for p in points]
    plt.figure(figsize=(6, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")
    plt.scatter(x_vals, y_vals, color="#00FFFF", s=120)
    for p in points:
        plt.annotate(
            f"P{p['id']}",
            (p["x"], p["y"]),
            xytext=(5, 5),
            textcoords="offset points",
            color="#FFFF00",
        )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    plt.title("MST Kruskal — Initial (no tree edges)", color="white", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    points = load_points(args.input)
    n = len(points)
    if n < 2:
        raise ValueError("Need at least 2 points for an MST.")

    steps = kruskal_steps(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    draw_initial(points, args.output_dir / "mst_step_000_initial.png")

    for step in steps:
        mst_tuples = [(e["u"], e["v"], e["weight"]) for e in step["mst_edges"]]
        idx = step["step"]
        out = args.output_dir / f"mst_step_{idx:03d}.png"
        draw_step(
            points,
            mst_tuples,
            step_index=idx,
            title_suffix=f"(+ edge P{step['edge']['u']}-P{step['edge']['v']})",
            output_path=out,
            highlight_last=True,
        )

    result = {
        "source": str(args.input),
        "algorithm": "Kruskal",
        "node_count": n,
        "mst_edge_count": len(steps),
        "total_mst_weight": steps[-1]["cumulative_weight"] if steps else 0.0,
        "steps": steps,
    }
    (args.output_dir / "mst_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(f"Loaded: {args.input.resolve()}")
    print(f"Saved {1 + len(steps)} figures to: {args.output_dir.resolve()}")
    print(f"Log: {(args.output_dir / 'mst_result.json').resolve()}")


if __name__ == "__main__":
    main()

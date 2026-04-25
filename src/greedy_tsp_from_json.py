import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def load_data(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def find(parent: dict[int, int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def union(parent: dict[int, int], a: int, b: int) -> None:
    root_a = find(parent, a)
    root_b = find(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a


def greedy_tsp_edges(points: list[dict]) -> tuple[list[dict], list[dict]]:
    node_ids = [p["id"] for p in points]
    point_map = {p["id"]: (p["x"], p["y"]) for p in points}

    all_edges: list[dict] = []
    for a, b in combinations(node_ids, 2):
        distance = euclidean_distance(point_map[a], point_map[b])
        all_edges.append({"u": a, "v": b, "distance": distance})
    all_edges.sort(key=lambda e: e["distance"])

    parent = {node: node for node in node_ids}
    degree = {node: 0 for node in node_ids}
    selected_edges: list[dict] = []
    steps: list[dict] = []
    total_distance = 0.0
    target_edge_count = len(node_ids)

    for edge in all_edges:
        u = edge["u"]
        v = edge["v"]

        if degree[u] >= 2 or degree[v] >= 2:
            continue

        is_last_edge = len(selected_edges) == target_edge_count - 1
        creates_cycle = find(parent, u) == find(parent, v)
        if creates_cycle and not is_last_edge:
            continue

        selected_edges.append(edge)
        degree[u] += 1
        degree[v] += 1
        total_distance += edge["distance"]

        if not creates_cycle:
            union(parent, u, v)

        steps.append(
            {
                "step": len(selected_edges),
                "added_edge": {
                    "u": u,
                    "v": v,
                    "distance": round(edge["distance"], 6),
                },
                "selected_edges": [
                    {
                        "u": e["u"],
                        "v": e["v"],
                        "distance": round(e["distance"], 6),
                    }
                    for e in selected_edges
                ],
                "cumulative_distance": round(total_distance, 6),
            }
        )

        if len(selected_edges) == target_edge_count:
            break

    return selected_edges, steps


def draw_step_figure(
    points: list[dict],
    selected_edges: list[dict],
    step_index: int,
    cumulative_distance: float,
    optimal_distance: float,
    output_path: Path,
) -> None:
    x_values = [p["x"] for p in points]
    y_values = [p["y"] for p in points]
    point_map = {p["id"]: (p["x"], p["y"]) for p in points}

    plt.figure(figsize=(6, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    for p1, p2 in combinations(points, 2):
        plt.plot(
            [p1["x"], p2["x"]],
            [p1["y"], p2["y"]],
            color="white",
            linewidth=0.6,
            alpha=0.35,
        )

    for edge in selected_edges:
        x1, y1 = point_map[edge["u"]]
        x2, y2 = point_map[edge["v"]]
        plt.plot([x1, x2], [y1, y2], color="#FFFF00", linewidth=2.8)

    plt.scatter(x_values, y_values, color="#00FFFF", s=120)
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
    ax.set_axis_off()
    plt.title(
        f"Distance: {cumulative_distance:.3f} | "
        f"Optimal: {optimal_distance:.3f}",
        color="white",
        fontsize=18,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def draw_initial_figure(points: list[dict], optimal_distance: float, output_path: Path) -> None:
    x_values = [p["x"] for p in points]
    y_values = [p["y"] for p in points]

    plt.figure(figsize=(6, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")
    plt.scatter(x_values, y_values, color="#00FFFF", s=120)

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
    ax.set_axis_off()
    plt.title(
        f"Greedy Initial | Optimal: {optimal_distance:.3f}",
        color="white",
        fontsize=18,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    json_path = Path("images") / "tsp_routes.json"
    data = load_data(json_path)
    points = data["points"]
    routes = data.get("routes", [])
    optimal_distance = min((route["total_distance"] for route in routes), default=0.0)

    output_dir = Path("images") / "greedy_steps"
    output_dir.mkdir(parents=True, exist_ok=True)

    draw_initial_figure(
        points=points,
        optimal_distance=optimal_distance,
        output_path=output_dir / "greedy_step_000_initial.png",
    )

    final_edges, steps = greedy_tsp_edges(points)
    for step in steps:
        output_path = output_dir / f"greedy_step_{step['step']:03d}.png"
        draw_step_figure(
            points=points,
            selected_edges=step["selected_edges"],
            step_index=step["step"],
            cumulative_distance=step["cumulative_distance"],
            optimal_distance=optimal_distance,
            output_path=output_path,
        )

    result_path = output_dir / "greedy_result.json"
    result_payload = {
        "source_file": str(json_path),
        "optimal_distance": optimal_distance,
        "final_edge_count": len(final_edges),
        "final_edges": [
            {"u": e["u"], "v": e["v"], "distance": round(e["distance"], 6)}
            for e in final_edges
        ],
        "total_distance": round(sum(e["distance"] for e in final_edges), 6),
        "steps": steps,
    }
    result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    print(f"Loaded points from: {json_path.resolve()}")
    print(f"Saved greedy step images to: {output_dir.resolve()}")
    print(f"Saved greedy result to: {result_path.resolve()}")


if __name__ == "__main__":
    main()

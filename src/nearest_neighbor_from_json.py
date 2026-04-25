import json
import math
import random
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt


def load_points(json_path: Path) -> list[dict]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data["points"]


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def nearest_neighbor_path(points: list[dict], start_id: int) -> tuple[list[int], list[dict]]:
    point_map = {p["id"]: (p["x"], p["y"]) for p in points}
    unvisited = set(point_map.keys())
    unvisited.remove(start_id)

    path = [start_id]
    current = start_id
    total_distance = 0.0
    steps: list[dict] = []

    while unvisited:
        current_xy = point_map[current]
        next_id = min(
            unvisited,
            key=lambda nid: euclidean_distance(current_xy, point_map[nid]),
        )
        step_distance = euclidean_distance(current_xy, point_map[next_id])
        total_distance += step_distance
        path.append(next_id)
        steps.append(
            {
                "from": current,
                "to": next_id,
                "step_distance": round(step_distance, 6),
                "cumulative_distance": round(total_distance, 6),
                "path_so_far": list(path),
            }
        )
        unvisited.remove(next_id)
        current = next_id

    return_distance = euclidean_distance(point_map[current], point_map[start_id])
    total_distance += return_distance
    path.append(start_id)
    steps.append(
        {
            "from": current,
            "to": start_id,
            "step_distance": round(return_distance, 6),
            "cumulative_distance": round(total_distance, 6),
            "path_so_far": list(path),
            "is_return_to_start": True,
        }
    )
    return path, steps


def draw_step_figure(
    points: list[dict],
    step_path: list[int],
    cumulative_distance: float,
    optimal_distance: float,
    output_path: Path,
    step_index: int,
) -> None:
    point_map = {p["id"]: (p["x"], p["y"]) for p in points}
    x_values = [p["x"] for p in points]
    y_values = [p["y"] for p in points]
    route_xy = [point_map[node_id] for node_id in step_path]
    route_x = [xy[0] for xy in route_xy]
    route_y = [xy[1] for xy in route_xy]

    plt.figure(figsize=(6, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    for p1, p2 in combinations(points, 2):
        plt.plot(
            [p1["x"], p2["x"]],
            [p1["y"], p2["y"]],
            color="white",
            linewidth=0.6,
            alpha=0.45,
        )

    plt.plot(route_x, route_y, color="#FFFF00", linewidth=2.8)
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
        f"Distance: {cumulative_distance:.3f}  |  "
        f"Optimal: {optimal_distance:.3f}",
        color="white",
        fontsize=20,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def draw_initial_figure(
    points: list[dict],
    optimal_distance: float,
    output_path: Path,
) -> None:
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
        f"Initial Graph |  Optimal: {optimal_distance:.3f}",
        color="white",
        fontsize=20,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    json_path = Path("images") / "tsp_routes.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    points = data["points"]
    routes = data.get("routes", [])
    optimal_distance = min((route["total_distance"] for route in routes), default=0.0)

    output_dir = Path("images") / "nearest_neighbor_steps"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_id = random.choice([p["id"] for p in points])
    full_path, steps = nearest_neighbor_path(points, start_id=start_id)
    draw_initial_figure(
        points=points,
        optimal_distance=optimal_distance,
        output_path=output_dir / "nn_step_000_initial.png",
    )

    for idx, step in enumerate(steps, start=1):
        output_path = output_dir / f"nn_step_{idx:03d}.png"
        draw_step_figure(
            points=points,
            step_path=step["path_so_far"],
            cumulative_distance=step["cumulative_distance"],
            optimal_distance=optimal_distance,
            output_path=output_path,
            step_index=idx,
        )

    result_path = output_dir / "nearest_neighbor_result.json"
    result_payload = {
        "source_file": str(json_path),
        "start_node_id": start_id,
        "final_path_one_based": full_path,
        "total_distance": steps[-1]["cumulative_distance"] if steps else 0.0,
        "optimal_distance": optimal_distance,
        "steps": steps,
    }
    result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    print(f"Loaded points from: {json_path.resolve()}")
    print(f"Saved step images to: {output_dir.resolve()}")
    print(f"Saved nearest-neighbor log to: {result_path.resolve()}")


if __name__ == "__main__":
    main()

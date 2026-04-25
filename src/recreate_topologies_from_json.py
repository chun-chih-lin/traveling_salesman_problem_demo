import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt


def load_routes(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def draw_route(
    points: list[dict],
    route: dict,
    output_path: Path,
) -> None:
    # Use one-based topology from JSON, then map back to point coordinates.
    topology_one_based = route["topology_one_based"]
    point_map = {p["id"]: (p["x"], p["y"]) for p in points}

    route_xy = [point_map[node_id] for node_id in topology_one_based]
    route_x = [xy[0] for xy in route_xy]
    route_y = [xy[1] for xy in route_xy]
    x_values = [p["x"] for p in points]
    y_values = [p["y"] for p in points]

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

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax.set_axis_off()
    # plt.xlabel("X", color="white")
    # plt.ylabel("Y", color="white")
    plt.title(f"Distance: {route['total_distance']:.3f}", color="white", fontsize=24)
    plt.grid(False)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    for p in points:
        plt.annotate(
            f"P{p['id']}",
            (p["x"], p["y"]),
            xytext=(5, 5),
            textcoords="offset points",
            color="#FFFF00",
        )

    # ax.text(
    #     0.98,
    #     0.02,
    #     f"Distance: {route['total_distance']:.3f}",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     color="white",
    # )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def find_best_route(routes: list[dict]) -> dict:
    if not routes:
        raise ValueError("No routes found in JSON data.")
    return min(routes, key=lambda route: route["total_distance"])

def find_worst_route(routes: list[dict]) -> dict:
    if not routes:
        raise ValueError("No routes found in JSON data.")
    return max(routes, key=lambda route: route["total_distance"])


def draw_best_route(points: list[dict], best_route: dict, output_path: Path) -> None:
    topology_one_based = best_route["topology_one_based"]
    point_map = {p["id"]: (p["x"], p["y"]) for p in points}
    route_xy = [point_map[node_id] for node_id in topology_one_based]
    route_x = [xy[0] for xy in route_xy]
    route_y = [xy[1] for xy in route_xy]
    x_values = [p["x"] for p in points]
    y_values = [p["y"] for p in points]

    fig = plt.figure(figsize=(6, 6), facecolor="black")
    fig.patch.set_edgecolor("#00FF00")
    fig.patch.set_linewidth(20)
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

    plt.title(
        f"Distance: {best_route['total_distance']:.3f}", color="white", fontsize=24
    )
    # ax.text(
    #     0.98,
    #     0.02,
    #     f"Best Distance: {best_route['total_distance']:.3f}",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     color="white",
    # )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()

def draw_worst_route(points: list[dict], worst_route: dict, output_path: Path) -> None:
    topology_one_based = worst_route["topology_one_based"]
    point_map = {p["id"]: (p["x"], p["y"]) for p in points}
    route_xy = [point_map[node_id] for node_id in topology_one_based]
    route_x = [xy[0] for xy in route_xy]
    route_y = [xy[1] for xy in route_xy]
    x_values = [p["x"] for p in points]
    y_values = [p["y"] for p in points]

    fig = plt.figure(figsize=(6, 6), facecolor="black")
    fig.patch.set_edgecolor("#FF0000")
    fig.patch.set_linewidth(20)
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

    plt.title(
        f"Distance: {worst_route['total_distance']:.3f}", color="white", fontsize=24
    )
    # ax.text(
    #     0.98,
    #     0.02,
    #     f"Worst Distance: {best_route['total_distance']:.3f}",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     color="white",
    # )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def main() -> None:
    json_path = Path("images") / "tsp_routes.json"
    data = load_routes(json_path)

    points = data["points"]
    routes = data["routes"]

    output_dir = Path("images") / "recreated_topologies"
    output_dir.mkdir(parents=True, exist_ok=True)

    for route in routes:
        output_path = output_dir / f"recreated_tsp_tour_{route['index']:03d}.png"
        draw_route(points, route, output_path)

    best_route = find_best_route(routes)
    best_output_path = output_dir / "best_tsp_route.png"
    draw_best_route(points, best_route, best_output_path)

    worst_route = find_worst_route(routes)
    worst_output_path = output_dir / "worst_tsp_route.png"
    draw_worst_route(points, worst_route, worst_output_path)

    print(f"Loaded routes from: {json_path.resolve()}")
    print(f"Recreated {len(routes)} topologies in: {output_dir.resolve()}")
    print(
        "Best route: "
        f"#{best_route['index']:03d}, distance={best_route['total_distance']:.6f}"
    )
    print(f"Best route image: {best_output_path.resolve()}")


if __name__ == "__main__":
    main()

import json
import math
import random
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt


def generate_unique_tours(num_points: int) -> list[tuple[int, ...]]:
    # Fix the start node (0) and remove reverse-duplicate cycles.
    tours: list[tuple[int, ...]] = []
    for perm in permutations(range(1, num_points)):
        if perm[0] < perm[-1]:
            tours.append((0, *perm, 0))
    return tours


def calculate_tour_distance(
    points: list[tuple[float, float]], tour: tuple[int, ...]
) -> float:
    total = 0.0
    for start_idx, end_idx in zip(tour, tour[1:]):
        x1, y1 = points[start_idx]
        x2, y2 = points[end_idx]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def save_tour_image(
    points: list[tuple[float, float]],
    tour: tuple[int, ...],
    total_distance: float,
    output_path: Path,
    image_index: int,
) -> None:
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    tour_x = [points[i][0] for i in tour]
    tour_y = [points[i][1] for i in tour]

    plt.figure(figsize=(6, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    plt.plot(tour_x, tour_y, color="white", linewidth=1.0)
    plt.scatter(x_values, y_values, color="#00FFFF", s=120)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("X", color="white")
    plt.ylabel("Y", color="white")
    plt.title(f"TSP Tour #{image_index:03d}", color="white")
    plt.grid(False)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.text(
        0.98,
        0.02,
        f"Distance: {total_distance:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
    )

    for idx, (x, y) in enumerate(points, start=1):
        plt.annotate(
            f"P{idx}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            color="#FFFF00",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    # Generate 5 random points where x and y are both in [0, 10].
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(6)]

    output_dir = Path("images")
    output_dir.mkdir(parents=True, exist_ok=True)
    tours = generate_unique_tours(len(points))
    tour_records: list[dict[str, object]] = []

    for idx, tour in enumerate(tours, start=1):
        output_path = output_dir / f"tsp_tour_{idx:03d}.png"
        total_distance = calculate_tour_distance(points, tour)
        save_tour_image(points, tour, total_distance, output_path, idx)
        tour_records.append(
            {
                "index": idx,
                "topology_zero_based": list(tour),
                "topology_one_based": [node + 1 for node in tour],
                "total_distance": round(total_distance, 6),
                "image_file": output_path.name,
            }
        )

    json_path = output_dir / "tsp_routes.json"
    json_payload = {
        "points": [
            {"id": idx + 1, "x": point[0], "y": point[1]}
            for idx, point in enumerate(points)
        ],
        "route_count": len(tour_records),
        "routes": tour_records,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(f"Saved {len(tours)} images to: {output_dir.resolve()}")
    print(f"Saved route data to: {json_path.resolve()}")
    print("Generated points:")
    for idx, (x, y) in enumerate(points, start=1):
        print(f"P{idx}: ({x:.3f}, {y:.3f})")
    print("Tour files:")
    for idx in range(1, len(tours) + 1):
        print(f"tsp_tour_{idx:03d}.png")


if __name__ == "__main__":
    main()

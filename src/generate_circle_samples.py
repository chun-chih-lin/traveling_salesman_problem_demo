import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt


def generate_circle_points(num_points: int, radius: float = 4.0) -> list[tuple[float, float]]:
    center_x, center_y = 5.0, 5.0
    points: list[tuple[float, float]] = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append((x, y))
    return points


def draw_complete_graph(points: list[tuple[float, float]], output_path: Path) -> None:
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]

    plt.figure(figsize=(6, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    ttl_possible_route = math.factorial(len(points)-1)/2

    for (x1, y1), (x2, y2) in combinations(points, 2):
        plt.plot([x1, x2], [y1, y2], color="white", linewidth=0.6, alpha=0.45)

    plt.scatter(x_values, y_values, color="#00FFFF", s=120)

    for idx, (x, y) in enumerate(points, start=1):
        plt.annotate(
            f"P{idx}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            color="#FFFF00",
        )

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    plt.title(f"Total possible route: {ttl_possible_route:.0f}", color="white", fontsize=24)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    output_dir = Path("images") / "circle_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in range(5, 15):
        points = generate_circle_points(n)
        output_path = output_dir / f"circle_points_{n}.png"
        draw_complete_graph(points, output_path)

    print(f"Saved samples for 5 to 10 points to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average distances vs node count from multiple comparison JSON files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("images"),
        help="Directory to save output charts.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("images"),
        help="Directory containing heuristic comparison JSON files.",
    )
    return parser.parse_args()


def load_results(input_path: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def plot_average_distances(
    node_counts: list[int],
    nn_averages: list[float],
    greedy_averages: list[float],
    optimal_averages: list[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    plt.plot(
        node_counts,
        nn_averages,
        marker="o",
        color="#00FFFF",
        linewidth=2,
        label="Nearest Neighbor",
    )
    plt.plot(
        node_counts,
        greedy_averages,
        marker="s",
        color="#00FF00",
        linewidth=2,
        label="Greedy Heuristic Approach",
    )
    plt.plot(
        node_counts,
        optimal_averages,
        marker="^",
        color="#DDDDDD",
        linewidth=2,
        label="Optimal",
    )

    ax.set_xlabel("Cities Number", color="white")
    ax.set_ylabel("Average Distance", color="white")
    # ax.set_title("Average TSP Distance by Node Count", color="white", fontsize=18)
    ax.set_xticks(node_counts)
    ax.set_xticklabels([str(n) for n in node_counts], color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(True, linestyle="--", alpha=0.3, color="white")
    legend = ax.legend(
        facecolor="black",
        edgecolor="white",
        fontsize=20,
        markerscale=2.0,
    )
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    filenames = [
        "heuristic_comparison_6_points_20_runs.json",
        "heuristic_comparison_7_points_20_runs.json",
        "heuristic_comparison_8_points_20_runs.json",
        "heuristic_comparison_9_points_20_runs.json",
        "heuristic_comparison_10_points_20_runs.json",
    ]

    node_counts: list[int] = []
    nn_averages: list[float] = []
    greedy_averages: list[float] = []
    optimal_averages: list[float] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_chart = args.output_dir / "heuristic_average_distance_by_nodes.png"

    for filename in filenames:
        file_path = args.input_dir / filename
        data = load_results(file_path)
        averages = data.get("averages", {})
        num_points = data.get("num_points")
        if num_points is None:
            raise ValueError(f"'num_points' missing in {file_path}")
        if not averages:
            raise ValueError(f"'averages' missing in {file_path}")

        node_counts.append(int(num_points))
        nn_averages.append(float(averages["nearest_neighbor_distance"]))
        greedy_averages.append(float(averages["greedy_distance"]))
        optimal_averages.append(float(averages["optimal_distance"]))
        print(f"Loaded input: {file_path.resolve()}")

    plot_average_distances(
        node_counts=node_counts,
        nn_averages=nn_averages,
        greedy_averages=greedy_averages,
        optimal_averages=optimal_averages,
        output_path=output_chart,
    )
    print(f"Saved chart: {output_chart.resolve()}")


if __name__ == "__main__":
    main()

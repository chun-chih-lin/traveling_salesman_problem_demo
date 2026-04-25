"""20 nodes on one random Hamiltonian cycle (a single closed loop).

Each node has degree 2 on that one circle. Saves under simulation/two_regular_20/:
final figure, simulation JSON, 2-opt step figures, and TSP optimal outputs.
"""

import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt

from tsp_optimal_from_json import solve_and_save_optimal_tsp


NODE_COUNT = 6
COORD_MIN = 0.0
COORD_MAX = 10.0
# Dedicated folder for this simulation (not images/)
SIM_ROOT = Path("simulation") / f"two_regular_{NODE_COUNT}"


def random_positions(n: int) -> list[dict]:
    return [
        {
            "id": i + 1,
            "x": round(random.uniform(COORD_MIN, COORD_MAX), 6),
            "y": round(random.uniform(COORD_MIN, COORD_MAX), 6),
        }
        for i in range(n)
    ]


def _norm_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def build_single_random_cycle(n: int) -> tuple[list[tuple[int, int]], list[dict]]:
    """One Hamiltonian cycle: random visit order, n edges, construction log in build order."""
    perm = list(range(1, n + 1))
    random.shuffle(perm)
    edges: list[tuple[int, int]] = []
    log: list[dict] = []
    step = 0
    for i in range(n - 1):
        u, v = perm[i], perm[i + 1]
        a, b = _norm_edge(u, v)
        edges.append((a, b))
        step += 1
        log.append({"step": step, "u": a, "v": b})
    u, v = perm[-1], perm[0]
    a, b = _norm_edge(u, v)
    edges.append((a, b))
    step += 1
    log.append({"step": step, "u": a, "v": b})
    return edges, log


def extract_cycles_from_edges(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Return list of cycles (ordered node ids). With a single Hamilton cycle, len == 1."""
    adj: dict[int, list[int]] = {i: [] for i in range(1, n + 1)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited: set[int] = set()
    cycles: list[list[int]] = []
    for start in range(1, n + 1):
        if start in visited:
            continue
        # Walk one cycle starting at smallest unvisited in this component
        cycle: list[int] = []
        prev = -1
        cur = start
        while cur not in visited:
            visited.add(cur)
            cycle.append(cur)
            nbrs = adj[cur]
            nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
            prev, cur = cur, nxt
        cycles.append(cycle)
    return cycles


def tour_length(
    tour: list[int], pos_map: dict[int, tuple[float, float]]
) -> float:
    n = len(tour)
    total = 0.0
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        x1, y1 = pos_map[a]
        x2, y2 = pos_map[b]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def two_opt_swap(tour: list[int], i: int, k: int) -> list[int]:
    """Reverse tour[i+1 : k+1] (inclusive); 0 <= i < k < len(tour)."""
    return tour[: i + 1] + tour[i + 1 : k + 1][::-1] + tour[k + 1 :]


def _edge_len(p: int, q: int, pos_map: dict[int, tuple[float, float]]) -> float:
    x1, y1 = pos_map[p]
    x2, y2 = pos_map[q]
    return math.hypot(x2 - x1, y2 - y1)


def run_two_opt(
    tour: list[int],
    pos_map: dict[int, tuple[float, float]],
) -> tuple[list[int], list[dict]]:
    """Best-improvement 2-opt until local optimum. Returns (best_tour, steps_log)."""
    n = len(tour)
    if n < 4:
        return list(tour), []

    current = list(tour)
    log: list[dict] = []
    step_idx = 0

    while True:
        best_new: list[int] = None
        best_i, best_k = -1, -1
        best_delta = 0.0

        for i in range(n):
            for k in range(i + 2, n):
                if i == 0 and k == n - 1:
                    continue
                a, b = current[i], current[(i + 1) % n]
                c, d = current[k], current[(k + 1) % n]
                removed = _edge_len(a, b, pos_map) + _edge_len(c, d, pos_map)
                added = _edge_len(a, c, pos_map) + _edge_len(b, d, pos_map)
                delta = added - removed
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_i, best_k = i, k
                    best_new = two_opt_swap(current, i, k)

        if best_new is None or best_i < 0:
            break
        step_idx += 1
        log.append(
            {
                "step": step_idx,
                "i": best_i,
                "k": best_k,
                "delta": round(best_delta, 6),
                "tour_after": list(best_new),
                "length_after": round(tour_length(best_new, pos_map), 6),
            }
        )
        current = best_new

    return current, log


def graph_total_edge_length(
    positions: list[dict], edges: list[tuple[int, int]]
) -> float:
    pmap = {p["id"]: (float(p["x"]), float(p["y"])) for p in positions}
    total = 0.0
    for u, v in edges:
        x1, y1 = pmap[u]
        x2, y2 = pmap[v]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def tour_to_edges(tour: list[int]) -> list[tuple[int, int]]:
    n = len(tour)
    out: list[tuple[int, int]] = []
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        out.append((u, v) if u < v else (v, u))
    return out


def draw_tour(
    positions: list[dict],
    tour: list[int],
    current_distance: float,
    output_path: Path,
    highlight_edges: set[tuple[int, int]] = None,
    background_edges: list[tuple[int, int]] = None,
) -> None:
    pmap = {p["id"]: p for p in positions}
    n = len(tour)
    plt.figure(figsize=(8, 8), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    if background_edges:
        for u, v in background_edges:
            pu, pv = pmap[u], pmap[v]
            plt.plot(
                [pu["x"], pv["x"]],
                [pu["y"], pv["y"]],
                color="white",
                linewidth=0.5,
                alpha=0.22,
                zorder=1,
            )

    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        pu, pv = pmap[u], pmap[v]
        edge_key = (u, v) if u < v else (v, u)
        lw = 2.4 if highlight_edges and edge_key in highlight_edges else 1.2
        col = "#FFFF00" if highlight_edges and edge_key in highlight_edges else "white"
        plt.plot([pu["x"], pv["x"]], [pu["y"], pv["y"]], color=col, linewidth=lw)

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

    ax.set_xlim(COORD_MIN, COORD_MAX)
    ax.set_ylim(COORD_MIN, COORD_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    plt.title(f"{current_distance:.6f}", color="white", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def draw_final_graph(
    positions: list[dict],
    edges: list[tuple[int, int]],
    output_path: Path,
    total_length: float,
) -> None:
    pmap = {p["id"]: p for p in positions}
    plt.figure(figsize=(8, 8), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    for u, v in edges:
        pu, pv = pmap[u], pmap[v]
        plt.plot([pu["x"], pv["x"]], [pu["y"], pv["y"]], color="white", linewidth=1.2)

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

    ax.set_xlim(COORD_MIN, COORD_MAX)
    ax.set_ylim(COORD_MIN, COORD_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    plt.title(f"{total_length:.6f}", color="white", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_two_opt_simulation_for_cycles(
    positions: list[dict],
    cycles: list[list[int]],
    two_opt_dir: Path,
    all_graph_edges: list[tuple[int, int]],
) -> list[dict]:
    """Run 2-opt per cycle; save figures per cycle. Returns summary list for JSON."""
    pos_map = {p["id"]: (float(p["x"]), float(p["y"])) for p in positions}
    two_opt_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for c_idx, cycle in enumerate(cycles, start=1):
        if len(cycle) < 4:
            # No meaningful 2-opt; still save initial tour figure
            sub = two_opt_dir / f"cycle_{c_idx:02d}"
            sub.mkdir(parents=True, exist_ok=True)
            c_len = tour_length(cycle, pos_map)
            draw_tour(
                positions,
                cycle,
                c_len,
                sub / "two_opt_step_000_initial.png",
                background_edges=all_graph_edges,
            )
            summary.append(
                {
                    "cycle_index": c_idx,
                    "node_ids": cycle,
                    "skipped_2opt": True,
                    "reason": "cycle length < 4",
                    "steps": [],
                }
            )
            continue

        tour = list(cycle)
        initial_len = tour_length(tour, pos_map)
        sub = two_opt_dir / f"cycle_{c_idx:02d}"
        sub.mkdir(parents=True, exist_ok=True)

        draw_tour(
            positions,
            tour,
            initial_len,
            sub / "two_opt_step_000_initial.png",
            background_edges=all_graph_edges,
        )

        final_tour, steps = run_two_opt(tour, pos_map)
        prev_edges = set(tour_to_edges(tour))
        for s in steps:
            new_tour = s["tour_after"]
            cur_len = tour_length(new_tour, pos_map)
            new_edges = set(tour_to_edges(new_tour))
            highlight = new_edges.symmetric_difference(prev_edges)
            prev_edges = new_edges
            draw_tour(
                positions,
                new_tour,
                cur_len,
                sub / f"two_opt_step_{s['step']:03d}.png",
                highlight_edges=highlight,
                background_edges=all_graph_edges,
            )

        summary.append(
            {
                "cycle_index": c_idx,
                "node_ids": cycle,
                "initial_length": round(initial_len, 6),
                "final_length": round(tour_length(final_tour, pos_map), 6),
                "step_count": len(steps),
                "final_tour": final_tour,
                "steps": steps,
            }
        )

    return summary


def main() -> None:
    random.seed()
    SIM_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = SIM_ROOT / f"two_regular_{NODE_COUNT}nodes_simulation.json"
    png_path = SIM_ROOT / "two_opt_steps" / "cycle_01" / f"two_regular_{NODE_COUNT}nodes_final.png"
    two_opt_dir = SIM_ROOT / "two_opt_steps"

    if Path(SIM_ROOT):
        if Path(json_path).is_file():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            positions = data["positions"]
            print("Exist positions. Use old one.")
        else:
            positions = random_positions(NODE_COUNT)
            print("Positions do not exist. Create new simulation.")
    else:
        positions = random_positions(NODE_COUNT)
        print("Positions do not exist. Create new simulation.")

    edges, construction_log = build_single_random_cycle(NODE_COUNT)
    cycles = extract_cycles_from_edges(NODE_COUNT, edges)

    payload = {
        "node_count": NODE_COUNT,
        "output_folder": str(SIM_ROOT.as_posix()),
        "description": (
            "Single Hamiltonian cycle on all nodes: a random permutation defines "
            "the circle; each node has exactly two incident edges on that one loop."
        ),
        "positions": positions,
        "edges": [{"u": u, "v": v} for u, v in edges],
        "edge_count": len(edges),
        "cycles": cycles,
        "construction_log": construction_log,
    }
    two_opt_summary = run_two_opt_simulation_for_cycles(
        positions, cycles, two_opt_dir, edges
    )
    payload["two_opt"] = two_opt_summary

    graph_len = graph_total_edge_length(positions, edges)
    payload["two_regular_total_edge_length"] = round(graph_len, 6)

    print(f"Computing optimal TSP (Held–Karp); n={NODE_COUNT} may take ~1–3 minutes …")
    tsp_opt = solve_and_save_optimal_tsp(positions, SIM_ROOT)
    payload["tsp_optimal"] = {
        "optimal_length": tsp_opt["optimal_length"],
        "tour_vertex_order_one_based": tsp_opt["tour_vertex_order_one_based"],
        "tour_vertex_order_zero_based": tsp_opt["tour_vertex_order_zero_based"],
        "output_png": tsp_opt["output_png"],
        "output_json": tsp_opt["output_json"],
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draw_final_graph(positions, edges, png_path, graph_len)

    (SIM_ROOT / "two_opt_summary.json").write_text(
        json.dumps({"cycles": two_opt_summary}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved figure: {png_path.resolve()}")
    print(f"2-opt figures: {two_opt_dir.resolve()}")
    print(f"2-opt summary: {(SIM_ROOT / 'two_opt_summary.json').resolve()}")
    print(f"Optimal TSP: {tsp_opt['output_png']}")
    print(f"Optimal TSP JSON: {tsp_opt['output_json']}")


if __name__ == "__main__":
    main()

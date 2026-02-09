import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install: pip install matplotlib networkx numpy"
    ) from exc

BASE_DIR = "/gpfs0/bgu-benshimo/users/shanibud/fattree_lossless_networks"
DEFAULT_PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DEFAULT_REPORTS_DIR = os.path.join(BASE_DIR, "reports")


@dataclass
class Flow:
    name: str
    path: List[str]
    color: str
    width: float = 2.4


@dataclass
class Scenario:
    title: str
    flows: List[Flow]
    cells_per_flow: Dict[str, int]
    buffer_capacity: int
    pfc_threshold: int
    injection_rate: Dict[str, int]
    total_packets: Dict[str, int]
    cycle_links: List[Tuple[str, str]]
    extra_links: List[Tuple[str, str]]


@dataclass
class SimulationResult:
    occupancy_history: Dict[str, List[int]]
    packets_passed: Dict[str, int]
    deadlock_step: Optional[int]
    initial_occupancy: Dict[str, int]
    final_occupancy: Dict[str, int]
    cycle_buffers: List[str]


def build_leaf_spine_topology() -> Tuple[nx.Graph, List[str], List[str]]:
    spines = ["S0", "S1"]
    leaves = ["L0", "L1", "L2", "L3"]

    G = nx.Graph()
    for s in spines:
        G.add_node(s, layer="spine")
    for l in leaves:
        G.add_node(l, layer="leaf")

    for s in spines:
        for l in leaves:
            G.add_edge(s, l)

    return G, spines, leaves


def fixed_positions(spines: List[str], leaves: List[str]) -> Dict[str, Tuple[float, float]]:
    pos = {}
    spine_xs = np.linspace(0.3, 0.7, len(spines))
    leaf_xs = np.linspace(0.1, 0.9, len(leaves))

    for x, s in zip(spine_xs, spines):
        pos[s] = (x, 0.8)
    for x, l in zip(leaf_xs, leaves):
        pos[l] = (x, 0.2)

    return pos


def buffer_id(u: str, v: str) -> str:
    return f"{u}->{v}"


def switch_buffer_id(node: str) -> str:
    return f"{node}"


def compute_buffer_occupancy(flows: List[Flow], cells_per_flow: Dict[str, int]) -> Dict[str, int]:
    occupancy: Dict[str, int] = {}
    for flow in flows:
        cells = cells_per_flow.get(flow.name, 1)
        for u in flow.path[:-1]:
            bid = switch_buffer_id(u)
            occupancy[bid] = occupancy.get(bid, 0) + cells
    return occupancy


def ordered_unique_links(flows: List[Flow]) -> List[str]:
    seen = set()
    links: List[str] = []
    for flow in flows:
        for u, v in zip(flow.path[:-1], flow.path[1:]):
            bid = buffer_id(u, v)
            if bid not in seen:
                seen.add(bid)
                links.append(bid)
    return links


def ordered_unique_switches(flows: List[Flow]) -> List[str]:
    seen = set()
    switches: List[str] = []
    for flow in flows:
        for u in flow.path[:-1]:
            if u not in seen:
                seen.add(u)
                switches.append(u)
    return switches


def build_dependency_graph(cycle_links: List[Tuple[str, str]]) -> nx.DiGraph:
    dg = nx.DiGraph()
    cycle_ids = [switch_buffer_id(u) for u, _ in cycle_links]
    for i in range(len(cycle_ids)):
        dg.add_edge(cycle_ids[i], cycle_ids[(i + 1) % len(cycle_ids)])
    return dg


def detect_deadlock(dep_graph: nx.DiGraph, full_buffers: Dict[str, bool]) -> Tuple[bool, List[str]]:
    for cycle in nx.simple_cycles(dep_graph):
        if all(full_buffers.get(buf, False) for buf in cycle):
            return True, cycle
    return False, []


def draw_flow(ax, pos, flow: Flow) -> None:
    for u, v in zip(flow.path[:-1], flow.path[1:]):
        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops={
                "arrowstyle": "->",
                "color": flow.color,
                "lw": flow.width,
                "shrinkA": 10,
                "shrinkB": 10,
            },
            zorder=3,
        )


def draw_cycle(ax, pos, cycle_links: List[Tuple[str, str]]) -> None:
    for u, v in cycle_links:
        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops={
                "arrowstyle": "-|>",
                "color": "darkred",
                "lw": 2.0,
                "linestyle": "--",
                "shrinkA": 12,
                "shrinkB": 12,
            },
            zorder=2,
        )


def draw_cycle_labels(
    ax,
    pos,
    cycle_links: List[Tuple[str, str]],
    extra_links: Optional[List[Tuple[str, str]]] = None,
) -> None:
    edge_labels = {}
    for idx, (u, v) in enumerate(cycle_links, start=1):
        edge_labels[(u, v)] = f"Link{idx}"
    if extra_links:
        start_idx = len(cycle_links) + 1
        for offset, (u, v) in enumerate(extra_links):
            edge_labels[(u, v)] = f"Link{start_idx + offset}"
    nx.draw_networkx_edge_labels(
        nx.DiGraph(cycle_links + (extra_links or [])),
        pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="darkred",
        ax=ax,
    )


def scenario_text(flows: List[Flow]) -> str:
    lines = ["Flows:"]
    for flow in flows:
        lines.append(f"{flow.name}: " + " -> ".join(flow.path))
    return "\n".join(lines)


def cycle_text(cycle_links: List[Tuple[str, str]]) -> str:
    nodes = [cycle_links[0][0]] + [v for _, v in cycle_links]
    return "Buffer dependency cycle:\n" + " -> ".join(nodes)


def render_scenario(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    scenario: Scenario,
    out_path: str,
    deadlock: bool,
    deadlock_step: Optional[int],
) -> None:
    occupancy = compute_buffer_occupancy(scenario.flows, scenario.cells_per_flow)
    full_buffers = {
        switch_buffer_id(u): occupancy.get(switch_buffer_id(u), 0) >= scenario.pfc_threshold
        for u, _ in scenario.cycle_links
    }

    dep_graph = build_dependency_graph(scenario.cycle_links)
    deadlock_static, _ = detect_deadlock(dep_graph, full_buffers)
    deadlock = deadlock or deadlock_static

    fig, (ax, ax_text) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]}
    )

    ax.set_title(scenario.title)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgray", width=1.0, alpha=0.6)

    node_colors = []
    for n in G.nodes():
        layer = G.nodes[n]["layer"]
        if layer == "spine":
            node_colors.append("#ff7f0e")
        else:
            node_colors.append("#8ccf72")

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=650,
        edgecolors="black",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

    for flow in scenario.flows:
        draw_flow(ax, pos, flow)
    draw_cycle(ax, pos, scenario.cycle_links)
    draw_cycle_labels(ax, pos, scenario.cycle_links, scenario.extra_links)

    ax.set_axis_off()

    ax_text.set_axis_off()
    ax_text.text(0.0, 0.95, scenario_text(scenario.flows), va="top", fontsize=10)
    ax_text.text(0.0, 0.58, cycle_text(scenario.cycle_links), va="top", fontsize=10)

    state = "DEADLOCK" if deadlock else "NO DEADLOCK"
    state_color = "red" if deadlock else "green"
    if deadlock and deadlock_step is not None:
        state = f"DEADLOCK (t={deadlock_step})"
    ax_text.text(
        0.0,
        0.28,
        f"System state: {state}",
        va="top",
        fontsize=11,
        color=state_color,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=" * 72)
    print(scenario.title)
    print(scenario_text(scenario.flows))
    print(cycle_text(scenario.cycle_links))
    print("Buffer occupancy (cycle switches):")
    for u, _ in scenario.cycle_links:
        bid = switch_buffer_id(u)
        occ = occupancy.get(bid, 0)
        status = "FULL" if occ >= scenario.buffer_capacity else "OK"
        print(f"  {bid}: {occ}/{scenario.buffer_capacity} ({status})")
    print(f"System state: {state}")
    print(f"Saved plot to: {out_path}")


def simulate_buffer_dynamics(scenario: Scenario, steps: int) -> SimulationResult:
    switches = ordered_unique_switches(scenario.flows)
    occupancy = compute_buffer_occupancy(scenario.flows, scenario.cells_per_flow)
    for sw in switches:
        occupancy.setdefault(sw, 0)

    initial_occupancy = dict(occupancy)
    occupancy_history = {sw: [] for sw in switches}
    packets_passed = {sw: 0 for sw in switches}
    remaining_packets = {
        flow.name: scenario.total_packets.get(flow.name, 0) for flow in scenario.flows
    }
    first_switch = {flow.name: flow.path[0] for flow in scenario.flows}

    cycle_buffers = [switch_buffer_id(u) for u, _ in scenario.cycle_links]
    cycle_next = {
        cycle_buffers[i]: cycle_buffers[(i + 1) % len(cycle_buffers)]
        for i in range(len(cycle_buffers))
    }
    switch_next: Dict[str, str] = {}
    for flow in scenario.flows:
        for u, v in zip(flow.path[:-1], flow.path[1:]):
            switch_next[u] = v

    deadlock_step: Optional[int] = None

    for step in range(steps):
        for flow in scenario.flows:
            rate = scenario.injection_rate.get(flow.name, 0)
            sw = first_switch[flow.name]
            for _ in range(rate):
                if remaining_packets[flow.name] <= 0:
                    break
                if occupancy[sw] >= scenario.buffer_capacity:
                    break
                occupancy[sw] += 1
                remaining_packets[flow.name] -= 1

        for sw in switches:
            occupancy_history[sw].append(occupancy[sw])

        moves: Dict[str, int] = {sw: 0 for sw in switches}
        for sw in switches:
            if occupancy[sw] <= 0:
                continue
            downstream = cycle_next.get(sw) or switch_next.get(sw)
            if downstream is None:
                moves[sw] = 1
            else:
                if (
                    occupancy[downstream] < scenario.pfc_threshold
                    and occupancy[downstream] < scenario.buffer_capacity
                ):
                    moves[sw] = 1

        if deadlock_step is None:
            cycle_full = all(
                occupancy[sw] >= scenario.pfc_threshold for sw in cycle_buffers
            )
            cycle_blocked = not any(moves[sw] for sw in cycle_buffers)
            if cycle_full and cycle_blocked:
                deadlock_step = step

        for sw, count in moves.items():
            if count <= 0:
                continue
            occupancy[sw] -= count
            packets_passed[sw] += count
            downstream = cycle_next.get(sw) or switch_next.get(sw)
            if downstream is not None:
                occupancy[downstream] += count

    final_occupancy = dict(occupancy)

    return SimulationResult(
        occupancy_history=occupancy_history,
        packets_passed=packets_passed,
        deadlock_step=deadlock_step,
        initial_occupancy=initial_occupancy,
        final_occupancy=final_occupancy,
        cycle_buffers=cycle_buffers,
    )


def plot_buffer_occupancy(
    scenario: Scenario,
    sim_result: SimulationResult,
    out_path: str,
) -> None:
    times = list(range(len(next(iter(sim_result.occupancy_history.values())))))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]

    plt.figure(figsize=(8, 5))
    cycle_series = {
        sw: sim_result.occupancy_history[sw] for sw in sim_result.cycle_buffers
    }
    for idx, (sw, series) in enumerate(cycle_series.items()):
        lw = 2.2
        alpha = 0.9
        marker = markers[idx % len(markers)]
        plt.plot(
            times,
            series,
            label=sw,
            linewidth=lw,
            alpha=alpha,
            marker=marker,
            markevery=max(1, len(times) // 12),
            markersize=4,
        )

    plt.axhline(
        scenario.pfc_threshold,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="PFC threshold",
    )
    plt.axhline(
        scenario.buffer_capacity,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Buffer capacity",
    )

    plt.title(f"Buffer occupancy over time (cycle switches)\n{scenario.title}")
    plt.xlabel("Time step")
    plt.ylabel("Buffered packets")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pfc_pauses(
    scenario: Scenario,
    sim_result: SimulationResult,
    out_path: str,
) -> None:
    times = list(range(len(next(iter(sim_result.occupancy_history.values())))))
    switches = list(sim_result.cycle_buffers)

    fig, ax = plt.subplots(figsize=(8, 4))
    y_positions = range(len(switches))

    for y, sw in zip(y_positions, switches):
        series = sim_result.occupancy_history[sw]
        paused = [t for t, val in zip(times, series) if val >= scenario.pfc_threshold]
        ax.scatter(paused, [y] * len(paused), s=16, color="red")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(switches)
    ax.set_xlabel("Time step")
    ax.set_title("PFC pause messages (buffer >= threshold)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_link_pauses(
    scenario: Scenario,
    sim_result: SimulationResult,
    out_path: str,
) -> None:
    times = list(range(len(next(iter(sim_result.occupancy_history.values())))))
    links = [f"Link{idx}" for idx in range(1, len(scenario.cycle_links) + 1)]
    downstream_map = {
        f"Link{idx}": v for idx, (_, v) in enumerate(scenario.cycle_links, start=1)
    }
    if scenario.extra_links:
        start_idx = len(scenario.cycle_links) + 1
        for offset, (_, v) in enumerate(scenario.extra_links):
            link_name = f"Link{start_idx + offset}"
            links.append(link_name)
            downstream_map[link_name] = v

    fig, ax = plt.subplots(figsize=(8, 4))
    y_positions = range(len(links))

    for y, link in zip(y_positions, links):
        downstream = downstream_map[link]
        series = sim_result.occupancy_history[downstream]
        paused = [t for t, val in zip(times, series) if val >= scenario.pfc_threshold]
        ax.scatter(paused, [y] * len(paused), s=16, color="purple")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(links)
    ax.set_xlabel("Time step")
    ax.set_title("Link pause timeline (downstream buffer >= threshold)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    scenario: Scenario,
    sim_result: SimulationResult,
    out_path: str,
) -> None:
    switches = ordered_unique_switches(scenario.flows)
    cycle_set = set(sim_result.cycle_buffers)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"Scenario: {scenario.title}\n")
        handle.write(f"Buffer capacity: {scenario.buffer_capacity}\n")
        handle.write(f"PFC threshold: {scenario.pfc_threshold}\n")
        handle.write("Injection rate (pkts/step): " + str(scenario.injection_rate) + "\n")
        handle.write("Total packets per flow: " + str(scenario.total_packets) + "\n")
        if sim_result.deadlock_step is None:
            handle.write("Deadlock: NO\n")
        else:
            handle.write(f"Deadlock: YES (t={sim_result.deadlock_step})\n")
        handle.write("Cycle buffers: " + " -> ".join(sim_result.cycle_buffers) + "\n")
        non_cycle = [sw for sw in switches if sw not in cycle_set]
        if non_cycle:
            handle.write(
                "Note: these switches are not part of the cycle buffer dependency: "
                + ", ".join(non_cycle)
                + "\n"
            )
        handle.write("Initial = initial buffer occupancy; Final = final buffer occupancy.\n")
        handle.write("\n")
        handle.write("Switch | Initial | Final | PacketsPassed\n")
        handle.write("------ | ------- | ----- | -------------\n")
        for sw in switches:
            handle.write(
                f"{sw} | {sim_result.initial_occupancy.get(sw, 0)} | "
                f"{sim_result.final_occupancy.get(sw, 0)} | "
                f"{sim_result.packets_passed.get(sw, 0)}\n"
            )


def build_scenarios() -> List[Scenario]:
    cycle_links = [("L0", "S0"), ("S0", "L2"), ("L2", "S1"), ("S1", "L0")]

    scenario1 = Scenario(
        title="Scenario 1: Cyclic buffer dependency without deadlock",
        flows=[
            Flow("F1", ["L0", "S0", "L2"], "red"),
            Flow("F2", ["L2", "S1", "L0"], "blue"),
        ],
        cells_per_flow={"F1": 2, "F2": 2},
        buffer_capacity=10,
        pfc_threshold=7,
        injection_rate={"F1": 1, "F2": 1},
        total_packets={"F1": 8, "F2": 8},
        cycle_links=cycle_links,
        extra_links=[],
    )

    scenario2 = Scenario(
        title="Scenario 2: Same dependency, extra flow -> deadlock",
        flows=[
            Flow("F1", ["L0", "S0", "L2"], "red"),
            Flow("F2", ["L2", "S1", "L0"], "blue"),
            Flow("F3", ["L1", "S0", "L2"], "purple"),
        ],
        cells_per_flow={"F1": 2, "F2": 2, "F3": 2},
        buffer_capacity=10,
        pfc_threshold=7,
        injection_rate={"F1": 1, "F2": 1, "F3": 3},
        total_packets={"F1": 8, "F2": 8, "F3": 30},
        cycle_links=cycle_links,
        extra_links=[("L1", "S0")],
    )

    return [scenario1, scenario2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate cyclic buffer dependency with PFC in a leaf-spine fabric."
    )
    parser.add_argument(
        "--scenario",
        choices=["1", "2", "both"],
        default="both",
        help="Which scenario to run",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_PLOTS_DIR,
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--reports_dir",
        default=DEFAULT_REPORTS_DIR,
        help="Output directory for generated reports",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of time steps for buffer occupancy simulation",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    G, spines, leaves = build_leaf_spine_topology()
    pos = fixed_positions(spines, leaves)

    scenarios = build_scenarios()
    if args.scenario == "1":
        scenarios = [scenarios[0]]
    elif args.scenario == "2":
        scenarios = [scenarios[1]]

    for idx, scenario in enumerate(scenarios, start=1):
        sim_result = simulate_buffer_dynamics(scenario, steps=args.steps)

        out_path = f"{args.out_dir}/pfc_scenario_{idx}.png"
        render_scenario(
            G,
            pos,
            scenario,
            out_path,
            deadlock=sim_result.deadlock_step is not None,
            deadlock_step=sim_result.deadlock_step,
        )

        occupancy_path = f"{args.out_dir}/pfc_scenario_{idx}_occupancy.png"
        plot_buffer_occupancy(scenario, sim_result, occupancy_path)

        pause_path = f"{args.out_dir}/pfc_scenario_{idx}_pfc_pause.png"
        plot_pfc_pauses(scenario, sim_result, pause_path)

        link_pause_path = f"{args.out_dir}/pfc_scenario_{idx}_link_pause.png"
        plot_link_pauses(scenario, sim_result, link_pause_path)

        report_path = f"{args.reports_dir}/pfc_scenario_{idx}_summary.txt"
        write_report(scenario, sim_result, report_path)


if __name__ == "__main__":
    main()

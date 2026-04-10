"""
visualizer.py
-------------
Optional matplotlib-based plotting for simulation results.

Plots produced:
  1. Total congestion over time (main metric)
  2. Per-road queue length over time (sub-plots)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

if TYPE_CHECKING:
    from traffic_sim.simulator import TrafficSimulator


def plot_results(sim: "TrafficSimulator", save_path: str | None = None) -> None:
    """
    Generate a dashboard plot of simulation results.

    Parameters
    ----------
    sim       : TrafficSimulator  – Completed simulation instance
    save_path : str | None        – If given, save figure to this path;
                                    otherwise display interactively.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("[Visualizer] matplotlib not installed – skipping plots.")
        print("             Run:  pip install matplotlib")
        return

    steps = list(range(1, len(sim.congestion_history) + 1))
    roads = list(sim.roads.values())
    n_roads = len(roads)

    # ── Layout: 1 big congestion chart + n_roads smaller per-road charts ──
    fig = plt.figure(figsize=(14, 3 + 2 * n_roads), facecolor="#1a1a2e")
    fig.suptitle(
        "Urban Traffic Simulation – Rule-Based Adaptive Control",
        fontsize=14,
        color="white",
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        n_roads + 1,
        1,
        hspace=0.55,
        top=0.93,
        bottom=0.05,
    )

    # ── Panel 1: Total congestion ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    _apply_dark_style(ax0)
    ax0.plot(
        steps,
        sim.congestion_history,
        color="#e94560",
        linewidth=1.8,
        label="Total Congestion",
    )
    ax0.fill_between(steps, sim.congestion_history, alpha=0.25, color="#e94560")
    ax0.axhline(
        y=sum(sim.congestion_history) / len(sim.congestion_history),
        color="#ffd700",
        linewidth=1,
        linestyle="--",
        label="Mean",
    )
    ax0.set_title("Total Network Congestion", color="white", fontsize=10)
    ax0.set_ylabel("Vehicles", color="#aaaaaa", fontsize=8)
    ax0.legend(fontsize=8, facecolor="#0f3460", labelcolor="white")

    # ── Panels 2–N+1: Per-road queue lengths ──────────────────────────
    palette = [
        "#00b4d8",
        "#48cae4",
        "#90e0ef",
        "#ade8f4",  # blues
        "#f4a261",
        "#e76f51",
        "#2a9d8f",
        "#e9c46a",  # warm / teal
    ]

    for i, road in enumerate(roads):
        ax = fig.add_subplot(gs[i + 1])
        _apply_dark_style(ax)
        color = palette[i % len(palette)]
        ax.plot(
            range(len(road.history)),
            road.history,
            color=color,
            linewidth=1.4,
            label=f"{road.id}: {road.start_junction}→{road.end_junction}",
        )
        ax.fill_between(range(len(road.history)), road.history, alpha=0.15, color=color)
        ax.set_title(
            f"Road {road.id}  ({road.start_junction}→{road.end_junction})",
            color="white",
            fontsize=9,
        )
        ax.set_ylabel("Queue", color="#aaaaaa", fontsize=8)
        if road.max_capacity > 0:
            ax.axhline(
                y=road.max_capacity,
                color="red",
                linewidth=0.8,
                linestyle=":",
                alpha=0.6,
                label="Capacity",
            )
        ax.legend(fontsize=7, facecolor="#0f3460", labelcolor="white")

    # X-axis label only on the last subplot
    ax.set_xlabel("Time Step", color="#aaaaaa", fontsize=9)

    if save_path:
        plt.savefig(
            save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        print(f"[Visualizer] Figure saved to: {save_path}")
    else:
        plt.show()


def _apply_dark_style(ax) -> None:
    """Apply a dark-theme style to a matplotlib Axes object."""
    ax.set_facecolor("#0f3460")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.grid(True, color="#333355", linewidth=0.5, linestyle="--", alpha=0.7)

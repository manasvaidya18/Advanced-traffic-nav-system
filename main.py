"""
main.py  (v6)
-------------
Entry point for the Urban Traffic Simulation.

Usage
-----
  python main.py [options]

Options
-------
  --steps N                  Number of simulation steps            (default: 100)
  --seed S                   Random seed for reproducibility       (default: None)
  --k FLOAT                  Green-time scaling per queued vehicle  (default: 0.8)
  --gamma FLOAT              Downstream congestion priority penalty(default: 1.0)
  --smoothing-alpha FLOAT    Alpha for green time EMA smoothing    (default: 0.3)
  --switch-threshold FLOAT   Hysteresis penalty for signal switch  (default: 2.0)
  --eq-target FLOAT          Equilibrium pressure target ratio     (default: 0.25)
  --no-plot                  Disable matplotlib plots
  --save-plot PATH           Save plot to file instead of showing
  --quiet                    Suppress per-step console output
"""

import argparse
from traffic_sim import TrafficSimulator, plot_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Urban Traffic Simulation v6 - Strict Equilibrium Control"
    )
    p.add_argument("--steps", type=int, default=100, help="Simulation steps")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--k", type=float, default=0.8, help="Green-time scale factor")
    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Downstream congestion priority penalty",
    )
    p.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.3,
        help="EMA smoothing alpha for signal times",
    )
    p.add_argument(
        "--switch-threshold",
        type=float,
        default=2.0,
        help="Hysteresis to avoid signal flickering",
    )
    p.add_argument(
        "--eq-target",
        type=float,
        default=0.25,
        help="System pressure equilibrium target",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip plots")
    p.add_argument("--save-plot", type=str, default=None, metavar="PATH")
    p.add_argument("--quiet", action="store_true", help="No per-step output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sim = TrafficSimulator(
        seed=args.seed,
        k=args.k,
        gamma=args.gamma,
        smoothing_alpha=args.smoothing_alpha,
        switch_threshold=args.switch_threshold,
        equilibrium_target=args.eq_target,
    )
    sim.initialize_network()
    sim.run_simulation(steps=args.steps, verbose=not args.quiet)

    if not args.no_plot:
        print("\n[Main] Generating plots...")
        plot_results(sim, save_path=args.save_plot)


if __name__ == "__main__":
    main()

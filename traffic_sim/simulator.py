"""
simulator.py  (v6 - Strict Flow & Target Equilibrium)
----------------------------------------------------------
Constructs a physically accurate traffic network using Sources, Sinks,
Internal roads, robust anti-oscillation controllers, and mathematical float constraints.
"""

from __future__ import annotations
import random
import math
from typing import Optional
from collections import deque

from traffic_sim.road import Road, RoadType
from traffic_sim.junction import Junction


class TrafficSimulator:
    """
    Orchestrates the v6 traffic simulation built for rigid mathematical equilibrium.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        k: float = 0.8,
        gamma: float = 1.0,
        smoothing_alpha: float = 0.3,
        switch_threshold: float = 2.0,
        equilibrium_target: float = 0.25,  # Target roughly 100 queue globally (100/400)
    ):
        self.k = k
        self.gamma = gamma
        self.smoothing_alpha = smoothing_alpha
        self.switch_threshold = switch_threshold
        self.equilibrium_target = equilibrium_target

        self.junctions: dict[str, Junction] = {}
        self.roads: dict[str, Road] = {}
        self.current_time_step: int = 0

        self.global_pressure: float = 0.0
        self.total_network_capacity: int = 0

        self.step_arrivals: int = 0
        self.step_departures: int = 0
        self.step_exits: int = 0

        self.total_congestion: int = 0
        self.total_exited: int = 0
        self.total_generated: int = 0

        self.congestion_history: list[int] = []
        self.arrivals_history: list[int] = []
        self.departures_history: list[int] = []
        self.exits_history: list[int] = []

        # Strict Flow control variables
        self.strict_correction_mode: bool = False
        self.net_flow_window = deque(maxlen=10)  # Track moving avg drift

        if seed is not None:
            random.seed(seed)

    def initialize_network(self) -> None:
        """
        Builds the physical graph topology.
        """
        for jid in ("J1", "J2", "J3", "J4"):
            self.junctions[jid] = Junction(
                jid,
                k=self.k,
                gamma=self.gamma,
                smoothing_alpha=self.smoothing_alpha,
                switch_threshold=self.switch_threshold,
            )

        road_specs = [
            ("SRC_N", "EXT_N", "J1", RoadType.SOURCE, 2.5),
            ("SRC_E", "EXT_E", "J2", RoadType.SOURCE, 2.5),
            ("SRC_S", "EXT_S", "J3", RoadType.SOURCE, 2.5),
            ("SRC_W", "EXT_W", "J4", RoadType.SOURCE, 2.5),
            ("SNK_N", "J1", "EXT_N_Out", RoadType.SINK, 0.0),
            ("SNK_E", "J2", "EXT_E_Out", RoadType.SINK, 0.0),
            ("SNK_S", "J3", "EXT_S_Out", RoadType.SINK, 0.0),
            ("SNK_W", "J4", "EXT_W_Out", RoadType.SINK, 0.0),
            ("R1", "J1", "J2", RoadType.INTERNAL, 0.0),
            ("R2", "J2", "J3", RoadType.INTERNAL, 0.0),
            ("R3", "J3", "J4", RoadType.INTERNAL, 0.0),
            ("R4", "J4", "J1", RoadType.INTERNAL, 0.0),
            ("R5", "J1", "J3", RoadType.INTERNAL, 0.0),
            ("R6", "J2", "J4", RoadType.INTERNAL, 0.0),
        ]

        for rid, src, dst, rtype, lam in road_specs:
            cap = 0 if rtype == RoadType.SINK else 50
            mu = 10.0 if rtype == RoadType.SINK else 4.0

            road = Road(
                rid,
                src,
                dst,
                rtype,
                arrival_rate=lam,
                service_rate=mu,
                max_capacity=cap,
            )
            self.roads[rid] = road

            if rtype != RoadType.SINK:
                self.junctions[dst].add_incoming_road(road)

        self.junctions["J1"].add_outgoing_road(self.roads["SNK_N"], weight=3.0)
        self.junctions["J1"].add_outgoing_road(self.roads["R1"], weight=4.0)
        self.junctions["J1"].add_outgoing_road(self.roads["R5"], weight=3.0)

        self.junctions["J2"].add_outgoing_road(self.roads["SNK_E"], weight=3.0)
        self.junctions["J2"].add_outgoing_road(self.roads["R2"], weight=4.0)
        self.junctions["J2"].add_outgoing_road(self.roads["R6"], weight=3.0)

        self.junctions["J3"].add_outgoing_road(self.roads["SNK_S"], weight=4.0)
        self.junctions["J3"].add_outgoing_road(self.roads["R3"], weight=6.0)

        self.junctions["J4"].add_outgoing_road(self.roads["SNK_W"], weight=4.0)
        self.junctions["J4"].add_outgoing_road(self.roads["R4"], weight=6.0)

        self.total_network_capacity = sum(r.max_capacity for r in self.roads.values())

        for junction in self.junctions.values():
            junction.update_signal_logic(global_pressure=0.0)

    def generate_traffic(self) -> None:
        self.step_arrivals = 0
        for road in self.roads.values():
            if road.road_type == RoadType.SOURCE:
                n = _poisson_sample(road.arrival_rate)
                road.add_vehicles(n)
                self.step_arrivals += n
                self.total_generated += n

        # Register the step net flow into moving average (arrivals vs exits computed post-step)

    def update_signals(self) -> None:
        self.total_congestion = sum(r.queue_length for r in self.roads.values())
        if self.total_network_capacity > 0:
            self.global_pressure = self.total_congestion / self.total_network_capacity

        # Check Flow Conservation moving average
        avg_flow = (
            sum(self.net_flow_window) / len(self.net_flow_window)
            if self.net_flow_window
            else 0.0
        )

        # Determine correction modes
        if avg_flow > 1.0 and self.global_pressure > self.equilibrium_target + 0.10:
            self.strict_correction_mode = True
        elif avg_flow <= 0.0 and self.global_pressure <= self.equilibrium_target:
            self.strict_correction_mode = False

        # Throttle sources based on target
        for road in self.roads.values():
            # In strict mode, arrivals are severely restricted
            if self.strict_correction_mode and road.road_type == RoadType.SOURCE:
                road.arrival_rate = road.base_arrival_rate * 0.1
            elif road.road_type == RoadType.SOURCE:
                # Normal throttling above target
                if self.global_pressure > self.equilibrium_target:
                    road.throttle_source(self.global_pressure)
                else:
                    road.arrival_rate = road.base_arrival_rate

        # Update junctions
        for junction in self.junctions.values():
            junction.update_signal_logic(
                self.global_pressure, self.strict_correction_mode
            )

    def _perfect_split(self, count: int, probs: list[float]) -> list[int]:
        """
        Splits an integer 'count' perfectly proportionally to 'probs'.
        Floor rounds cleanly, then random drops the exact correct remaining items
        according to the probabilities to guarantee integer conservation.
        """
        assert len(probs) > 0
        # If there's only one path, dump it all there.
        if len(probs) == 1:
            return [count]

        allocations = [int(math.floor(count * p)) for p in probs]
        remainder = count - sum(allocations)

        # While there's remainder, randomly distribute it proportionally to probabilities
        while remainder > 0:
            idx = _weighted_choice(list(range(len(probs))), probs)
            allocations[idx] += 1
            remainder -= 1

        assert sum(allocations) == count
        return allocations

    def move_vehicles(self) -> None:
        self.step_departures = 0

        sinks = [r for r in self.roads.values() if r.road_type == RoadType.SINK]
        start_sink_exited = sum(r.total_exited for r in sinks)

        for junction in self.junctions.values():
            green_road = junction.current_green_road
            if green_road is None or green_road.queue_length == 0:
                continue

            # int casting drops safely if the service rate scales oddly, but base queue is int.
            departed = green_road.remove_vehicles(int(round(junction.green_time)))
            if departed == 0:
                continue

            self.step_departures += departed

            if junction.outgoing_roads:
                probs = junction.get_routing_probabilities()
                allocations = self._perfect_split(departed, probs)

                for dest_road, moved_count in zip(junction.outgoing_roads, allocations):
                    if moved_count > 0:
                        dest_road.add_vehicles(moved_count)

        end_sink_exited = sum(r.total_exited for r in sinks)
        self.step_exits = end_sink_exited - start_sink_exited
        self.total_exited = end_sink_exited

        self.net_flow_window.append(self.step_arrivals - self.step_exits)

        # The ultimate test
        self._assert_conservation()

    def _assert_conservation(self) -> None:
        """
        Ensures mathematically perfect flow conservation.
        Total Generated MUST EXACTLY equal Currently Queued + Total Exited.
        """
        current_queued = sum(r.queue_length for r in self.roads.values())
        if self.total_generated != (current_queued + self.total_exited):
            raise AssertionError(
                f"Vehicle Conservation Lost! "
                f"Gen={self.total_generated} != Q({current_queued}) + Ex({self.total_exited})"
            )

    def update_queues(self) -> None:
        for road in self.roads.values():
            road.history.append(road.queue_length)

    def _print_step_report(self) -> None:
        sep = "-" * 66
        print(f"\n{sep}")
        tag = (
            "[STRICT CORRECTION]"
            if self.strict_correction_mode
            else "[EQUILIBRIUM ASSURED]"
        )
        avg_drift = (
            sum(self.net_flow_window) / len(self.net_flow_window)
            if self.net_flow_window
            else 0
        )
        print(f"  TIME STEP: {self.current_time_step}  {tag}")
        print(
            f"  Global Pressure: {self.global_pressure:.1%} | Target: {self.equilibrium_target:.1%} | Avg Net Drift: {avg_drift:+.1f}"
        )
        print(sep)

        print("  SIGNALS:")
        for junction in self.junctions.values():
            gid = (
                junction.current_green_road.id
                if junction.current_green_road
                else "none"
            )
            gq = (
                junction.current_green_road.queue_length
                if junction.current_green_road
                else 0
            )

            probs = (
                junction.get_routing_probabilities()
                if junction.current_green_road
                else []
            )
            route_str = (
                " + ".join(
                    f"{r.id}:{p:.0%}" for r, p in zip(junction.outgoing_roads, probs)
                )
                if probs
                else "none"
            )

            print(
                f"    {junction.id}: green={gid:5s}(Q={gq:>2})  gt={junction.green_time:>2.1f}s  route=[{route_str}]"
            )

    def run_simulation(self, steps: int = 100, verbose: bool = True) -> list[int]:
        print("=" * 66)
        print("  URBAN TRAFFIC SIMULATION v6 - Absolute Strict Equilibrium")
        print("=" * 66)

        for _ in range(steps):
            self.current_time_step += 1
            self.update_signals()
            self.generate_traffic()
            self.move_vehicles()
            self.update_queues()

            self.congestion_history.append(self.total_congestion)
            self.arrivals_history.append(self.step_arrivals)
            self.departures_history.append(self.step_departures)
            self.exits_history.append(self.step_exits)

            if verbose:
                self._print_step_report()

        print("\n" + "=" * 66)
        print("  SIMULATION COMPLETE")
        print(f"  Final congestion   : {self.total_congestion}")
        print(f"  Total vehicles in  : {self.total_generated}")
        print(f"  Total vehicles out : {self.total_exited}")
        print(
            f"  Drift / Remainder  : {(self.total_generated) - (self.total_congestion + self.total_exited)}"
        )
        print("=" * 66)

        return self.congestion_history


def _poisson_sample(lam: float) -> int:
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


def _weighted_choice(options: list, weights: list[float]):
    r = random.random()
    cumulative = 0.0
    for option, weight in zip(options, weights):
        cumulative += weight
        if r <= cumulative:
            return option
    return options[-1]


def _queue_bar(length: int, capacity: int, width: int = 16) -> str:
    if capacity <= 0:
        return ""
    filled = min(int((length / capacity) * width), width)
    pct = int((length / capacity) * 100)
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct:>3}%"

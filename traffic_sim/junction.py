"""
junction.py  (v6 - Damped Control & Flow Conservation)
------------------------------------------------------------
Represents a node (intersection) in the traffic-network graph.

v6 CHANGES:
-----------
1. DAMPED CONTROL: Green times are smoothed using an EMA (alpha).
2. SOFT PRIORITY: Switching the green road requires beating the current road by `switch_threshold`.
3. DYNAMIC SINK ROUTING: Outgoing weights to Sink roads smoothly scale up or down
   depending on the global equilibrium target pressure to safely vent congestion.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traffic_sim.road import Road

MIN_GREEN: int = 5
MAX_GREEN: int = 30
FAIRNESS_THRESHOLD: int = 8


class Junction:
    """
    Traffic junction controlling flow with robust anti-oscillation controllers.

    Parameters
    ----------
    k                 : green-time scaling factor per queued vehicle  [default 0.8]
    beta              : congestion routing penalty coefficient        [default 0.2]
    gamma             : downstream congestion priority penalty        [default 1.0]
    smoothing_alpha   : exponential moving average for green time     [default 0.3]
    switch_threshold  : priority hysteresis for stable signals        [default 2.0]
    """

    def __init__(
        self,
        junction_id: str,
        k: float = 0.8,
        beta: float = 0.2,
        gamma: float = 1.0,
        smoothing_alpha: float = 0.3,
        switch_threshold: float = 2.0,
    ):
        self.id = junction_id

        self.k = k
        self.beta = beta
        self.gamma = gamma
        self.alpha = smoothing_alpha
        self.switch_threshold = switch_threshold

        self.incoming_roads: list[Road] = []
        self.outgoing_roads: list[Road] = []

        # Track base network weights vs current dynamic values
        self._base_weights: list[float] = []
        self.outgoing_weights: list[float] = []

        self.current_green_road: Road | None = None
        self.green_time: float = float(MIN_GREEN)
        self.cycle_time: int = MIN_GREEN

        self.force_max_green: bool = (
            False  # Activated strictly by simulator correction modes
        )

        self._steps_since_green: dict[str, int] = {}

    def add_incoming_road(self, road: Road) -> None:
        from traffic_sim.road import RoadType

        if road.road_type == RoadType.SINK:
            raise ValueError("Sink roads cannot be incoming to a junction.")

        self.incoming_roads.append(road)
        self._steps_since_green[road.id] = 0

    def add_outgoing_road(self, road: Road, weight: float = 1.0) -> None:
        from traffic_sim.road import RoadType

        if road.road_type == RoadType.SOURCE:
            raise ValueError("Source roads cannot be outgoing from a junction.")

        self.outgoing_roads.append(road)
        self._base_weights.append(weight)
        self.outgoing_weights.append(weight)

    # ------------------------------------------------------------------
    # Priority & Routing
    # ------------------------------------------------------------------

    def _downstream_congestion(self) -> float:
        if not self.outgoing_roads:
            return 0.0
        return sum(r.congestion_ratio for r in self.outgoing_roads) / len(
            self.outgoing_roads
        )

    def _priority_score(self, road: Road) -> float:
        avg_down_q = self._downstream_congestion() * 50.0
        return road.queue_length - (self.gamma * avg_down_q)

    def select_next_road(self) -> Road | None:
        if not self.incoming_roads:
            return None

        # Fairness override skips hysteresis to ensure no road starves
        starved = [
            r
            for r in self.incoming_roads
            if self._steps_since_green.get(r.id, 0) >= FAIRNESS_THRESHOLD
        ]
        if starved:
            return max(starved, key=lambda r: self._steps_since_green[r.id])

        # Hysteresis (Soft Priority selection)
        best_road = max(self.incoming_roads, key=self._priority_score)

        if self.current_green_road:
            current_score = self._priority_score(self.current_green_road)
            best_score = self._priority_score(best_road)

            # If the current road isn't beat by the threshold, stick with it
            if best_score <= current_score + self.switch_threshold:
                return self.current_green_road

        return best_road

    def _compute_green_time(self, chosen: Road) -> float:
        """
        Calculates raw target, then applies Exponential Moving Average smoothing.
        """
        if self.force_max_green:
            return float(MAX_GREEN)

        target_gt = float(MIN_GREEN) + (self.k * chosen.queue_length)
        target_gt = max(float(MIN_GREEN), min(float(MAX_GREEN), target_gt))

        # Smooth adoption of new green time (Damped Control)
        smoothed = (self.green_time * (1.0 - self.alpha)) + (target_gt * self.alpha)
        return smoothed

    def adjust_sink_routing(
        self, pressure: float, strict_correction: bool = False
    ) -> None:
        """
        Dynamically adjusts the baseline routing probability directed at Sink Roads
        in order to flush excess system pressure, mimicking dynamic exit control.
        """
        from traffic_sim.road import RoadType

        for i, (road, base_w) in enumerate(
            zip(self.outgoing_roads, self._base_weights)
        ):
            if road.road_type == RoadType.SINK:
                if strict_correction:
                    # In correction mode, forcefully point almost all volume to exits
                    self.outgoing_weights[i] = base_w * 5.0
                elif pressure > 0.25:  # Target is 25% utilization (~100/400)
                    # For every 10% pressure over target, increase exit weight smoothly
                    scale = 1.0 + ((pressure - 0.25) * 2.0)
                    # EMA smoothing on the exit probability modifier
                    target = base_w * min(3.0, scale)
                    self.outgoing_weights[i] = (
                        self.outgoing_weights[i] * (1.0 - self.alpha)
                    ) + (target * self.alpha)
                else:
                    # Relax to base
                    self.outgoing_weights[i] = (
                        self.outgoing_weights[i] * (1.0 - self.alpha)
                    ) + (base_w * self.alpha)

    def get_routing_probabilities(self) -> list[float]:
        if not self.outgoing_roads:
            return []

        adjusted = []
        for road, param_weight in zip(self.outgoing_roads, self.outgoing_weights):
            penalty = 1.0 / (1.0 + self.beta * road.queue_length)
            adjusted.append(param_weight * penalty)

        total = sum(adjusted)
        if total == 0:
            n = len(adjusted)
            return [1.0 / n] * n
        return [a / total for a in adjusted]

    # ------------------------------------------------------------------
    # Update Tick
    # ------------------------------------------------------------------

    def update_signal_logic(
        self, global_pressure: float, strict_correction: bool = False
    ) -> None:
        if not self.incoming_roads:
            return

        self.force_max_green = strict_correction
        self.adjust_sink_routing(global_pressure, strict_correction)

        chosen = self.select_next_road()
        if chosen is None:
            return

        for road in self.incoming_roads:
            if road.id == chosen.id:
                self._steps_since_green[road.id] = 0
            else:
                self._steps_since_green[road.id] += 1

        self.current_green_road = chosen
        self.green_time = self._compute_green_time(chosen)

        avg_green = (MIN_GREEN + MAX_GREEN) / 2
        self.cycle_time = max(1, int(len(self.incoming_roads) * avg_green))

    @property
    def avg_incoming_queue(self) -> float:
        if not self.incoming_roads:
            return 0.0
        return sum(r.queue_length for r in self.incoming_roads) / len(
            self.incoming_roads
        )

    def __repr__(self) -> str:
        gid = self.current_green_road.id if self.current_green_road else "None"
        mode = " [STRICT DRAIN]" if self.force_max_green else ""
        return (
            f"Junction({self.id}{mode}, green={gid}, "
            f"green_time={self.green_time:.1f}, cycle={self.cycle_time})"
        )

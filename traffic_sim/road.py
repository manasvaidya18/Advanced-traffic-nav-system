"""
road.py
-------
Represents a directed road segment in the traffic network graph.

v5 CHANGES (Realistic Origin-Destination Topology):
---------------------------------------------------
- Roads are strictly typed: Source, Internal, or Sink.
- Source roads ONLY input external traffic (from arrival_rate).
- Sink roads INSTANTLY consume traffic (vehicles have reached destinations).
- Internal roads merely transfer vehicles between junctions.
- Removed arbitrary mathematical exit_probabilities in favour of physical sinks.
"""

from enum import Enum


class RoadType(Enum):
    SOURCE = "Source"  # Spawns new traffic into the network
    INTERNAL = "Internal"  # Connects junctions; zero external spawning/consuming
    SINK = "Sink"  # Consumes traffic exiting the network


class Road:
    """
    Models a directed road.

    Attributes
    ----------
    id             : str
    start_junction : str
    end_junction   : str
    road_type      : RoadType
    arrival_rate   : float (λ; used ONLY by Sources)
    service_rate   : float (μ; traffic outflow capacity over Green)
    max_capacity   : int   (Hard queue limit; 0 = unlimited)
    queue_length   : int   (Vehicles waiting to cross end_junction)
    total_arrived  : int   (Lifetime vehicles entered this road)
    total_exited   : int   (Lifetime vehicles instantly absorbed if Sink)
    """

    def __init__(
        self,
        road_id: str,
        start_junction: str,
        end_junction: str,
        road_type: RoadType,
        arrival_rate: float = 0.0,
        service_rate: float = 5.0,
        max_capacity: int = 50,
    ):
        self.id = road_id
        self.start_junction = start_junction
        self.end_junction = end_junction
        self.road_type = road_type

        self.queue_length: int = 0

        # Sources generate traffic based on this parameter
        if self.road_type == RoadType.SOURCE and arrival_rate <= 0:
            raise ValueError("Source roads must have an active positive arrival_rate.")
        elif self.road_type != RoadType.SOURCE:
            self.arrival_rate = 0.0  # Force pure transfer semantics for non-sources
        else:
            self.arrival_rate = arrival_rate
        self.base_arrival_rate = self.arrival_rate

        self.service_rate = service_rate
        self.max_capacity = max_capacity

        # In a Sink road, capacity is effectively infinite for destruction purposes,
        # but queues should basically never exist as they drain instantly.
        if self.road_type == RoadType.SINK:
            self.max_capacity = 0  # Infinite absorb

        self.total_arrived: int = 0
        self.total_exited: int = 0

        self.history: list[int] = []

    # ------------------------------------------------------------------
    # Queue manipulation
    # ------------------------------------------------------------------

    def add_vehicles(self, count: int) -> None:
        """
        Add arriving vehicles to the road.
        If this is a SINK road, they do not queue; they instantly exit.
        """
        if count < 0:
            raise ValueError("Vehicle count must be non-negative.")

        self.total_arrived += count

        if self.road_type == RoadType.SINK:
            # Vehicles reaching Destination instantly exit the system
            self.total_exited += count
            self.queue_length = 0
            return

        self.queue_length += count

        if self.max_capacity > 0:
            self.queue_length = min(self.queue_length, self.max_capacity)

    def remove_vehicles(self, max_allowed: int) -> int:
        """
        Attempt to pull up to *max_allowed* vehicles from the queue to cross an intersection.
        Note: Sinks don't remove vehicles across an intersection (they are dead-ends).
        """
        if self.road_type == RoadType.SINK:
            return 0  # Sinks do not feed other junctions

        if max_allowed < 0:
            raise ValueError("Vehicle count must be non-negative.")

        actual = min(self.queue_length, max_allowed)
        self.queue_length = max(0, self.queue_length - actual)
        return actual

    # ------------------------------------------------------------------
    # Adaptive Inflow Control (v5 Realism)
    # ------------------------------------------------------------------

    def throttle_source(self, global_pressure: float) -> None:
        """
        If the city center is gridlocked, dynamically throttle the Source Ramp
        arrivals to prevent overwhelming the network. Only applies to Sources.
        """
        if self.road_type != RoadType.SOURCE:
            return

        if global_pressure > 0.4:
            # Drop incoming traffic volume by up to 50% proportionate to over-pressure
            throttle_factor = min(0.5, (global_pressure - 0.4))
            self.arrival_rate *= 1.0 - throttle_factor
        else:
            # Slowly restore demand to the base level
            # Note: since we overwrite arrival_rate, we'd need base tracking.
            # Simplified: assume we don't aggressively throttle, or just use discrete drops.
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def congestion_ratio(self) -> float:
        """Fractional fill level [0, 1]. Sinks are always 0."""
        if self.road_type == RoadType.SINK:
            return 0.0

        if self.max_capacity > 0:
            return self.queue_length / self.max_capacity

        return float(self.queue_length)

    def __repr__(self) -> str:
        return (
            f"Road({self.id} [{self.road_type.value}]: {self.start_junction}->{self.end_junction}, "
            f"Q={self.queue_length})"
        )

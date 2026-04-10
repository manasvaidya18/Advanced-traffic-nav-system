"""
tests/test_simulator.py  (v6)
------------------------------
Integration tests for TrafficSimulator v6.
Validates perfect exact splitting and absolute flow conservation.
"""

import pytest
from traffic_sim.simulator import TrafficSimulator


@pytest.fixture()
def sim() -> TrafficSimulator:
    s = TrafficSimulator(seed=0)
    s.initialize_network()
    return s


class TestStrictFlowConservation:
    def test_perfect_split_conserves_integer(self, sim):
        # Even split
        allocs = sim._perfect_split(10, [0.5, 0.5])
        assert sum(allocs) == 10
        assert allocs == [5, 5]

        # Awkward fractional split
        allocs = sim._perfect_split(100, [0.33, 0.33, 0.34])
        assert sum(allocs) == 100

        # Micro count
        allocs = sim._perfect_split(1, [0.1, 0.8, 0.1])
        assert sum(allocs) == 1

        # Zero edge case
        allocs = sim._perfect_split(0, [0.5, 0.5])
        assert sum(allocs) == 0

    def test_absolute_flow_assertion_during_run(self, sim):
        """
        The v6 system natively runs self._assert_conservation() on every framework step.
        If any car is lost, it instantly crashes the simulation.
        So if run_simulation finishes, conservation is perfect.
        """
        try:
            sim.run_simulation(steps=50, verbose=False)
        except AssertionError as e:
            pytest.fail(f"Conservation broken during run: {e}")

        assert sim.total_generated == (
            sum(r.queue_length for r in sim.roads.values()) + sim.total_exited
        )


class TestDampedControllers:
    def test_green_time_smoothing_damps_jumps(self, sim):
        # With default smooth=0.3, a sudden change shouldn't fully apply immediately.
        j = sim.junctions["J1"]
        # Force a massive queue
        sim.roads["SRC_N"].queue_length = 50
        j.select_next_road()  # should select SRC_N
        j.current_green_road = sim.roads["SRC_N"]

        # Old green time is MIN_GREEN 5.0
        # Target green time is 5 + 0.8*50 = 45 -> cap 30
        # Smoothed = 5*0.7 + 30*0.3 = 3.5 + 9.0 = 12.5
        new_gt = j._compute_green_time(sim.roads["SRC_N"])
        assert new_gt < 30.0
        assert new_gt == pytest.approx(12.5)

    def test_priority_hysteresis(self, sim):
        j = sim.junctions["J1"]
        # r1 is slightly worse than r2, but because of hysteresis, r1 currently green keeps it
        r1 = sim.roads["SRC_N"]
        r1.queue_length = 10
        r2 = sim.roads["R4"]
        r2.queue_length = 11

        # 11 - 10 = 1.0 (Less than 2.0 switch threshold)
        j.current_green_road = r1

        chosen = j.select_next_road()
        assert chosen.id == r1.id  # Kept the current road

        # Now r2 dominates
        r2.queue_length = 15
        chosen = j.select_next_road()
        assert chosen.id == r2.id  # Swept threshold

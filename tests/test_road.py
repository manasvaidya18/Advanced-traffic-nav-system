"""
tests/test_road.py
------------------
Unit tests for the Road class v5.
"""

import pytest
from traffic_sim.road import Road, RoadType


def test_source_requires_arrival_rate():
    with pytest.raises(ValueError):
        Road("SRC", "EXT", "J1", RoadType.SOURCE, arrival_rate=0)


def test_internal_forces_arrival_zero():
    r = Road("INT", "J1", "J2", RoadType.INTERNAL, arrival_rate=100)
    assert r.arrival_rate == 0.0


def test_sink_drains_instantly():
    r = Road("SNK", "J1", "EXT", RoadType.SINK)
    r.add_vehicles(50)
    assert r.queue_length == 0
    assert r.total_exited == 50
    assert r.congestion_ratio == 0.0


def test_internal_queues_normally():
    r = Road("INT", "J1", "J2", RoadType.INTERNAL, max_capacity=10)
    r.add_vehicles(5)
    assert r.queue_length == 5
    assert r.total_arrived == 5
    assert r.total_exited == 0

    r.add_vehicles(10)
    assert r.queue_length == 10  # Capped


def test_sink_cannot_feed():
    r = Road("SNK", "J1", "EXT", RoadType.SINK)
    r.add_vehicles(50)
    assert r.remove_vehicles(10) == 0


def test_source_throttling():
    r = Road("SRC", "EXT", "J1", RoadType.SOURCE, arrival_rate=2.0)
    r.throttle_source(0.6)  # Pressure is 0.6 (>0.4), throttle by 0.2
    assert r.arrival_rate == pytest.approx(1.6)

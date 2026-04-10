"""
tests/test_junction.py  (v5)
"""

import pytest
from traffic_sim.road import Road, RoadType
from traffic_sim.junction import Junction


def make_junction(**kwargs) -> Junction:
    defaults = dict(junction_id="J_test", k=0.8, beta=0.2, gamma=1.0)
    defaults.update(kwargs)
    return Junction(**defaults)


def make_road(rid: str, rtype: RoadType, q: int = 0, lam: float = 0.0) -> Road:
    r = Road(rid, "JA", "JB", road_type=rtype, arrival_rate=lam)
    r.queue_length = q
    return r


class TestJunctionWiring:
    def test_sink_cannot_be_incoming(self):
        j = make_junction()
        r = make_road("SINK", RoadType.SINK)
        with pytest.raises(ValueError):
            j.add_incoming_road(r)

    def test_source_cannot_be_outgoing(self):
        j = make_junction()
        r = make_road("SRC", RoadType.SOURCE, lam=1.0)
        with pytest.raises(ValueError):
            j.add_outgoing_road(r)


class TestPriorityScore:
    def test_downstream_congestion_lowers_priority(self):
        j = make_junction(gamma=5.0)
        r_in = make_road("R1", RoadType.INTERNAL, q=5)
        r_out_clear = make_road("R_out", RoadType.INTERNAL)

        j.add_incoming_road(r_in)
        j.add_outgoing_road(r_out_clear)
        pri_clear = j._priority_score(r_in)

        r_out_clear.queue_length = 50
        pri_jammed = j._priority_score(r_in)

        assert pri_jammed < pri_clear

    def test_sink_always_clear(self):
        j = make_junction(gamma=5.0)
        r_in = make_road("R1", RoadType.INTERNAL, q=5)
        r_snk = make_road("R_out", RoadType.SINK)

        j.add_incoming_road(r_in)
        j.add_outgoing_road(r_snk)

        # It shouldn't matter if we push vehicles to sink, its q is logically 0
        r_snk.add_vehicles(500)

        assert j._downstream_congestion() == 0.0

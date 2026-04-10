"""
traffic_sim/__init__.py
-----------------------
Public API surface for the traffic_sim package.
"""

from traffic_sim.road import Road
from traffic_sim.junction import Junction
from traffic_sim.simulator import TrafficSimulator
from traffic_sim.visualizer import plot_results

__all__ = ["Road", "Junction", "TrafficSimulator", "plot_results"]

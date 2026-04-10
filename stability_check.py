from traffic_sim import TrafficSimulator

sim = TrafficSimulator(seed=42)
sim.initialize_network()
sim.run_simulation(steps=100, verbose=False)

h = sim.congestion_history
print("=== TOPOLOGICAL EQUILIBRIUM ===")
for i in range(0, 100, 10):
    block = h[i : i + 10]
    avg = sum(block) / len(block)
    bar = "#" * min(50, int(avg))
    print(
        "  Steps %3d-%3d: avg active vehicles = %5.1f vehicles %s"
        % (i + 1, i + 10, avg, bar)
    )

print("")
print("=== METRICS ===")
print("  Total entered    :", sum(sim.arrivals_history))
print("  Total exited     :", sim.total_exited)
print("  Final congestion :", h[-1])
print("  Peak congestion  :", max(h))

drift = sum(sim.arrivals_history) - sim.total_exited - h[-1]
print("  Unexplained Drift:", drift, "(must be exactly 0)")

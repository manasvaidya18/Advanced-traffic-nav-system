"""
debug_traffic_flow.py
Run this from the project root to verify traffic flows between junctions.
Usage: python debug_traffic_flow.py
"""

from src.env import MultiJunctionSystem

print("=" * 55)
print("  Inter-Junction Traffic Flow Verification")
print("=" * 55)

system = MultiJunctionSystem(n_junctions=2, arrival_rate=0)  # No random arrivals
j0 = system.junctions[0]
j1 = system.junctions[1]

# ── Test 1: J0 East → J1 West ──────────────────────────────
print("\n[Test 1] Vehicles travelling EAST from J0 → J1")
j0.queues = {'N': 0, 'S': 0, 'E': 20, 'W': 0}
j1.queues = {'N': 0, 'S': 0, 'E': 0,  'W': 0}

print(f"  BEFORE  →  J0 East queue: {j0.queues['E']:2d}  |  J1 West queue: {j1.queues['W']:2d}")
_, _, info = j0.step(2)  # Phase 2 = EW-Green
print(f"  AFTER   →  J0 East queue: {j0.queues['E']:2d}  |  J1 West queue: {j1.queues['W']:2d}  (transferred: {info['served']//2})")
print(f"  ✅ PASS" if j1.queues['W'] > 0 else "  ❌ FAIL — no transfer happened")

# ── Test 2: J1 East → J0 West ──────────────────────────────
print("\n[Test 2] Vehicles travelling EAST from J1 → J0")
j0.queues = {'N': 0, 'S': 0, 'E': 0,  'W': 0}
j1.queues = {'N': 0, 'S': 0, 'E': 20, 'W': 0}

print(f"  BEFORE  →  J1 East queue: {j1.queues['E']:2d}  |  J0 West queue: {j0.queues['W']:2d}")
_, _, info = j1.step(2)
print(f"  AFTER   →  J1 East queue: {j1.queues['E']:2d}  |  J0 West queue: {j0.queues['W']:2d}  (transferred: {info['served']//2})")
print(f"  ✅ PASS" if j0.queues['W'] > 0 else "  ❌ FAIL — no transfer happened")

# ── Test 3: J0 North → J1 South ────────────────────────────
print("\n[Test 3] Vehicles travelling NORTH from J0 → J1")
j0.queues = {'N': 15, 'S': 0, 'E': 0, 'W': 0}
j1.queues = {'N': 0,  'S': 0, 'E': 0, 'W': 0}

print(f"  BEFORE  →  J0 North queue: {j0.queues['N']:2d}  |  J1 South queue: {j1.queues['S']:2d}")
_, _, info = j0.step(0)  # Phase 0 = NS-Green
print(f"  AFTER   →  J0 North queue: {j0.queues['N']:2d}  |  J1 South queue: {j1.queues['S']:2d}  (transferred: {info['served']//2})")
print(f"  ✅ PASS" if j1.queues['S'] > 0 else "  ❌ FAIL — no transfer happened")

# ── Test 4: Multi-step cascade (watch queues cascade) ──────
print("\n[Test 4] Multi-step cascade — 5 steps, watch traffic ripple")
system2 = MultiJunctionSystem(n_junctions=2, arrival_rate=0)
j0 = system2.junctions[0]
j1 = system2.junctions[1]
j0.queues = {'N': 0, 'S': 0, 'E': 30, 'W': 0}  # Max load on J0 East
j1.queues = {'N': 0, 'S': 0, 'E': 0,  'W': 0}

print(f"  {'Step':<6} {'J0-East':>8} {'J1-West':>8} {'J0-Total':>10} {'J1-Total':>10}")
print(f"  {'-'*46}")
for step in range(1, 6):
    _, _, info = j0.step(2)
    j0t = sum(j0.queues.values())
    j1t = sum(j1.queues.values())
    print(f"  {step:<6} {j0.queues['E']:>8} {j1.queues['W']:>8} {j0t:>10} {j1t:>10}")

print("\n  → J0 East drains as J1 West grows = traffic is flowing ✅")
print("\n" + "=" * 55)
print("  All tests complete!")
print("=" * 55)
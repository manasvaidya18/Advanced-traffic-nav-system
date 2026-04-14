import numpy as np
import random
from collections import deque

class TrafficJunction:
    """
    Simulates a single 4-way traffic junction with coordinated signal control.
    """
    PHASES = ['NS-Green', 'NS-Left', 'EW-Green', 'EW-Left']
    PHASE_ARMS = {
        0: ['N', 'S'],
        1: ['N', 'S'],
        2: ['E', 'W'],
        3: ['E', 'W']
    }

    T_MIN = 10
    T_MAX = 60
    Q_MAX = 30
    SERVE_RATE = 3
    SWITCH_PENALTY = 3

    def __init__(self, junction_id, arrival_rate=2.0):
        self.junction_id = junction_id
        self.arrival_rate = arrival_rate
        self.neighbor = None
        self.reset()

    def connect_neighbor(self, neighbor):
        self.neighbor = neighbor

    def reset(self):
        self.queues = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self.phase = 0
        self.step_count = 0
        self.total_wait = 0
        self.total_served = 0
        self._last_phase = 0
        return self.get_state()

    def get_state(self):
        """Returns the raw state of the junction."""
        return (
            self.queues['N'],
            self.queues['S'],
            self.queues['E'],
            self.queues['W'],
            self.phase
        )

    def get_compact_state(self):
        """
        Returns a compact state for RL discretization.
        Aggregates N+S and E+W to reduce state space.
        """
        q_ns = self.queues['N'] + self.queues['S']
        q_ew = self.queues['E'] + self.queues['W']
        
        # Discretization into 7 buckets (0-60 range)
        def bucket(q):
            return min(int(q // 10), 6)
        
        base_state = (bucket(q_ns), bucket(q_ew), self.phase)
        
        if self.neighbor:
            n_state = self.neighbor.get_state()
            n_total = sum(n_state[:4])
            # Neighbor total queue bucketed
            n_bucket = min(int(n_total // 20), 6)
            return base_state + (n_bucket, self.neighbor.phase)
        
        return base_state + (0, 0)

    def step(self, action):
        self._last_phase = self.phase
        self._arrive()
        
        # Calculate green time based on demand
        arms = self.PHASE_ARMS[action]
        q_max_arm = max(self.queues[a] for a in arms)
        green_time = self.T_MIN + (q_max_arm / self.Q_MAX) * (self.T_MAX - self.T_MIN)
        
        # Add coordination adjustment
        if self.neighbor:
            n_total = sum(self.neighbor.get_state()[:4])
            if n_total > 20: # Neighbor is congested
                green_time *= 0.9 # Slightly reduce our time to help neighbor
        
        green_time = int(np.clip(green_time, self.T_MIN, self.T_MAX))
        
        # Serve vehicles
        served = 0
        for arm in arms:
            cleared = min(self.queues[arm], int(self.SERVE_RATE * green_time * 0.1))
            self.queues[arm] -= cleared
            served += cleared
            
        self.total_served += served
        self.phase = action
        self.step_count += 1
        
        reward = self._compute_reward()
        return self.get_compact_state(), reward, {"served": served, "green_time": green_time}

    def _arrive(self):
        for arm in ['N', 'S', 'E', 'W']:
            arrivals = np.random.poisson(self.arrival_rate * 0.4)
            self.queues[arm] = min(self.Q_MAX, self.queues[arm] + arrivals)

    def _compute_reward(self):
        total_q = sum(self.queues.values())
        switch_pen = self.SWITCH_PENALTY if self.phase != self._last_phase else 0
        
        # Reward is negative of total queue (minimizing wait time)
        reward = -total_q - switch_pen
        
        # Coordination bonus: Penalize if neighbor is very congested
        if self.neighbor:
            n_q = sum(self.neighbor.get_state()[:4])
            reward -= 0.1 * n_q
            
        return reward

class MultiJunctionSystem:
    def __init__(self, n_junctions=2, arrival_rate=2.0):
        self.junctions = [TrafficJunction(i, arrival_rate) for i in range(n_junctions)]
        if n_junctions >= 2:
            self.junctions[0].connect_neighbor(self.junctions[1])
            self.junctions[1].connect_neighbor(self.junctions[0])

    def reset(self):
        return [j.reset() and j.get_compact_state() for j in self.junctions]

    def step(self, actions):
        results = [j.step(a) for j, a in zip(self.junctions, actions)]
        states, rewards, infos = zip(*results)
        return list(states), list(rewards), list(infos)

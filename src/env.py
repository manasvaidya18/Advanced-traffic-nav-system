import numpy as np
import random
from collections import deque

class TrafficJunction:
    """
    Simulates a single 4-way traffic junction with smart signal control.
    Each junction can also coordinate with a neighboring junction.
    """

    # Different signal phases (actions)
    PHASES = ['NS-Green', 'NS-Left', 'EW-Green', 'EW-Left']

    # Which directions (arms) are active in each phase
    PHASE_ARMS = {
        0: ['N', 'S'],  # North-South traffic moves
        1: ['N', 'S'],   
        2: ['E', 'W'],  # East-West traffic moves
        3: ['E', 'W']    
    }

    # Signal timing constraints
    T_MIN = 10   # Minimum green signal time
    T_MAX = 60   # Maximum green signal time

    # Traffic constraints
    Q_MAX = 30   # Maximum cars allowed in a queue
    SERVE_RATE = 3  # Rate at which cars pass per unit time

    # Penalty for switching signals frequently
    SWITCH_PENALTY = 3

    # Mapping of outgoing traffic to neighbor junction
    # Example: cars leaving East will enter neighbor from West
    OUTFLOW_TO_NEIGHBOR = {
        'E': 'W',
        'W': 'E',
        'N': 'S',
        'S': 'N',
    }

    def _init_(self, junction_id, arrival_rate=2.0):
        # Unique ID of junction
        self.junction_id = junction_id

        # Average rate of incoming vehicles
        self.arrival_rate = arrival_rate

        # Neighbor junction (for coordination)
        self.neighbor = None

        # Initialize all variables
        self.reset()

    def connect_neighbor(self, neighbor):
        """
        Connect this junction with another junction.
        Enables traffic flow between them.
        """
        self.neighbor = neighbor

    def reset(self):
        """
        Reset the junction to initial state.
        """

        # Initialize queues for all directions
        self.queues = {'N': 0, 'S': 0, 'E': 0, 'W': 0}

        # Current signal phase
        self.phase = 0

        # Count number of simulation steps
        self.step_count = 0

        # Total waiting time (not actively used but useful for extension)
        self.total_wait = 0

        # Total number of vehicles served
        self.total_served = 0

        # Store previous phase (used for penalty calculation)
        self._last_phase = 0

        # Return initial compact state (important for RL)
        return self.get_compact_state()

    def get_state(self):
        """
        Returns full detailed state:
        (N queue, S queue, E queue, W queue, current phase)
        """
        return (
            self.queues['N'],
            self.queues['S'],
            self.queues['E'],
            self.queues['W'],
            self.phase
        )

    def get_compact_state(self):
        """
        Returns simplified (compact) state for RL.
        Reduces state size using bucketing.
        """

        # Combine opposite directions
        q_ns = self.queues['N'] + self.queues['S']
        q_ew = self.queues['E'] + self.queues['W']

        # Convert queue values into discrete buckets (0–6)
        def bucket(q):
            return min(int(q // 10), 6)

        # Base state: (NS traffic level, EW traffic level, current phase)
        base_state = (bucket(q_ns), bucket(q_ew), self.phase)

        # If neighbor exists, include its congestion info
        if self.neighbor:
            n_state = self.neighbor.get_state()

            # Total cars at neighbor
            n_total = sum(n_state[:4])

            # Bucket neighbor congestion
            n_bucket = min(int(n_total // 20), 6)

            # Return combined state
            return base_state + (n_bucket, self.neighbor.phase)

        # If no neighbor, add default values
        return base_state + (0, 0)

    def step(self, action):
        """
        Simulates one time step:
        - cars arrive
        - signal is applied
        - cars move
        - reward is calculated
        """

        # Store previous phase (for penalty calculation)
        self._last_phase = self.phase

        # Add new arriving cars
        self._arrive()

        # Determine which directions get green signal
        arms = self.PHASE_ARMS[action]

        # Find busiest lane among active arms
        q_max_arm = max(self.queues[a] for a in arms)

        # Dynamically calculate green signal time based on traffic
        green_time = self.T_MIN + (q_max_arm / self.Q_MAX) * (self.T_MAX - self.T_MIN)

        # Adjust green time if neighbor is congested
        if self.neighbor:
            n_total = sum(self.neighbor.get_state()[:4])
            if n_total > 20:
                green_time *= 0.9  # reduce time slightly

        # Ensure green time stays within limits
        green_time = int(np.clip(green_time, self.T_MIN, self.T_MAX))

        # Serve vehicles (cars pass through junction)
        served = 0
        for arm in arms:
            # Calculate number of cars that can pass
            cleared = min(self.queues[arm], int(self.SERVE_RATE * green_time * 0.1))

            # Remove cars from queue
            self.queues[arm] -= cleared
            served += cleared

            # Transfer some cars to neighbor junction
            if self.neighbor and cleared > 0:
                neighbor_arm = self.OUTFLOW_TO_NEIGHBOR[arm]

                # Send ~50% cars to neighbor
                transfer = max(1, int(cleared * 0.5))

                self.neighbor.queues[neighbor_arm] = min(
                    self.Q_MAX,
                    self.neighbor.queues[neighbor_arm] + transfer
                )

        # Update statistics
        self.total_served += served
        self.phase = action
        self.step_count += 1

        # Compute reward for RL
        reward = self._compute_reward()

        # Return new state, reward, and extra info
        return self.get_compact_state(), reward, {
            "served": served,
            "green_time": green_time
        }

    def _arrive(self):
        """
        Simulates arrival of new vehicles using Poisson distribution.
        """
        for arm in ['N', 'S', 'E', 'W']:
            arrivals = np.random.poisson(self.arrival_rate * 0.4)

            # Add cars but limit to max queue size
            self.queues[arm] = min(self.Q_MAX, self.queues[arm] + arrivals)

    def _compute_reward(self):
        """
        Computes reward:
        - penalizes high traffic
        - penalizes frequent signal switching
        - penalizes neighbor congestion
        """

        # Total vehicles waiting
        total_q = sum(self.queues.values())

        # Penalty if signal changed
        switch_pen = self.SWITCH_PENALTY if self.phase != self._last_phase else 0

        # Base reward (negative of traffic)
        reward = -total_q - switch_pen

        # Extra penalty if neighbor is congested
        if self.neighbor:
            n_q = sum(self.neighbor.get_state()[:4])
            reward -= 0.1 * n_q

        return reward


class MultiJunctionSystem:
    """
    Manages multiple traffic junctions together.
    """

    def _init_(self, n_junctions=2, arrival_rate=2.0):
        # Create multiple junctions
        self.junctions = [TrafficJunction(i, arrival_rate) for i in range(n_junctions)]

        # Connect first two junctions as neighbors
        if n_junctions >= 2:
            self.junctions[0].connect_neighbor(self.junctions[1])
            self.junctions[1].connect_neighbor(self.junctions[0])

    def reset(self):
        """
        Reset all junctions and return their initial states.
        """
        return [j.reset() for j in self.junctions]

    def step(self, actions):
        """
        Perform one step for all junctions.
        Each junction receives its own action.
        """

        # Run step for each junction
        results = [j.step(a) for j, a in zip(self.junctions, actions)]

        # Separate outputs
        states, rewards, infos = zip(*results)

        return list(states), list(rewards), list(infos)

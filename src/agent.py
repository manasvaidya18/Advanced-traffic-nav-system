import numpy as np
import random
import pickle

class MultiAgentQLearning:
    def __init__(
        self,
        n_actions=4,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        eps_decay_rate=0.99
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay_rate = eps_decay_rate
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.random.uniform(-1, 0, self.n_actions)
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state):
        q_current = self.get_q_values(state)
        q_next = self.get_q_values(next_state)
        
        td_target = reward + self.gamma * np.max(q_next)
        td_error = td_target - q_current[action]
        q_current[action] += self.alpha * td_error
        self.q_table[state] = q_current

    def decay_epsilon(self):
        """Called once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.eps_decay_rate)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

"""Epsilon Greedy Exploration Strategy."""
import numpy as np


class EpsilonGreedy:
    """Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        """Initialize Epsilon Greedy Exploration Strategy."""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space, done=False):
        """Choose action based on epsilon greedy strategy."""
        np.random.seed(123456789)
        if np.random.rand() < self.epsilon and not done:
            action = int(action_space.sample())
        else:
            np.random.seed()
            max_value = max(q_table[state])
            max_indices = [i for i, value in enumerate(q_table[state]) if value == max_value]
            action = np.random.choice(max_indices)

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        # print(self.epsilon)
        return action

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon

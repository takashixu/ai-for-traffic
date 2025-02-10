"""
Epsilon Greedy Exploration Strategy using Modified Q-Table
short, short: green   | short, medium: green   | short, long: green
medium, short: green  | medium, medium: green  | medium, long: green
long, short: green    | long, medium: green    | long, long: green
short, short: red     | short, medium: red     | short, long: red
medium, short: red    | medium, medium: red    | medium, long: red
long, short: red      | long, medium: red      | long, long: red
"""
import numpy as np


class DiscreteEpsilonGreedy:
    """Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        """Initialize Epsilon Greedy Exploration Strategy."""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space):
        """Choose action based on epsilon greedy strategy."""
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())
        else:
            action = np.argmax(q_table[state])

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        # print(self.epsilon)
        return action

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon

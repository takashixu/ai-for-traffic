"""Q-learning Agent class."""
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import csv
import pickle


class QLAgent:
    """Q-learning Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def export_q_table(self, filename, timestep):
        """Export Q-table to a CSV file with a column for the timestep."""
        print(self.q_table)
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            for state, actions in self.q_table.items():
                writer.writerow([timestep, state, actions])

    def act(self, done=False):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space, done=done)
        return self.action

    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        if not done:
            self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
                reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
            )
        self.state = s1
        self.acc_reward += reward

    def save_q_table(self, filename):
        """Save the Q-table to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load_q_table(self, filename):
        """Load the Q-table from a file."""
        with open(filename, 'rb') as file:
            self.q_table = pickle.load(file)

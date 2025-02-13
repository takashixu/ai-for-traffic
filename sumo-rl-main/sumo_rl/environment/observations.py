"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
    
class DiscreteObservationFunction(ObservationFunction):
    """Discrete observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal, queue_categories=3):
        """Initialize discrete observation function."""
        super().__init__(ts)
        self.queue_categories = queue_categories

    def __call__(self) -> int:
        """Return the discrete observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        # TODO: Implement per lane queue categorization
        queue_sizes = self.ts.categorize_queue_density()
        observation = np.array(phase_id + queue_sizes, dtype=np.int32)
        return observation

    def observation_space(self) -> spaces.Discrete:
        """Return the observation space."""
        low = np.zeros(len(self.ts.lanes) + 1, dtype=np.int32)
        high = np.array([2] * len(self.ts.lanes) + [self.ts.num_green_phases - 1], dtype=np.int32)
        return spaces.Box(low=low, high=high, dtype=np.int32)

    # def _categorize_queue_size(self, total_queued):
    #     """Categorize the total queued vehicles into discrete categories."""
    #     if total_queued < 10:
    #         return 0  # Small
    #     elif total_queued < 20:
    #         return 1  # Medium
    #     else:
    #         return 2  # Large

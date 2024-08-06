from dataclasses import dataclass
from collections import deque

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module


@dataclass
class DelayParameters:
    delay: int  # Delay time in ms
    feedback: float  # Amount of the delayed signal injected again in the process

    def __post_init__(self):
        self.delay = max(1, self.delay)
        # Note that a feedback of 0.0 won't produce any delay
        # TODO: is it what we want?
        # A feedback of 1.0 will produce an infinite loop
        # The Delay module can be used as a looper if feedback is set to 1.0
        self.feedback = max(0.0, min(self.feedback, 1.0))


class Delay(Module):
    def __init__(self, params: DelayParameters, sample_rate: int):
        self.params = params
        self.sample_rate = sample_rate
        self.memory = deque(maxlen=self.ms_to_samples(params.delay))

    def ms_to_samples(self, delay: int) -> int:
        """
        Convert the given delay in ms to a number of samples.
        """
        return max(1, int(self.sample_rate * (delay / 1000)))

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        output = input

        if len(self.memory) > 0:
            delayed_sample = self.memory[0]  # Oldest sample in the deque
            output += (
                delayed_sample * self.params.feedback
            )  # Mix the input with the delayed one

        # Add the sample to the end of the deque (right)
        # Once the memory is full, the oldest sample (left) will be removed
        self.memory.append(output)

        return output

from typing import ClassVar
from dataclasses import dataclass
from collections import deque

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module


@dataclass
class DelayParameters:
    duration: int  # Lazy implementation where the duration is the number of samples
    decay: float  # Value between 0 and 1, dictating the loss of sound over repetitions

    MAX_DURATION: ClassVar[int] = 44_800  # TODO: change this

    def __post_init__(self):
        self.duration = max(1, min(self.duration, self.MAX_DURATION))
        self.decay = max(0, min(self.decay, 1))


class Delay(Module):
    def __init__(self, params: DelayParameters):
        self.params = params
        self.memory = deque(maxlen=DelayParameters.MAX_DURATION)

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        output = input
        if self.params.duration < len(self.memory):
            output += (self.memory[self.params.duration] * self.params.decay).astype(
                "int32"
            )

        self.memory.appendleft(output)
        return output

from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module, ModuleParams


@dataclass
class DelayParameters(ModuleParams):
    delay: int
    "Delay time in ms"

    feedback: float
    "Amount of the delayed signal injected again in the process"

    def __post_init__(self):
        self.delay = max(1, self.delay)
        # Note that a feedback of 0.0 won't produce any delay
        # TODO: is it what we want?
        # A feedback of 1.0 will produce an infinite loop
        # The Delay module can be used as a looper if feedback is set to 1.0
        self.feedback = max(0.0, min(self.feedback, 1.0))


class _Memory:
    def __init__(self) -> None:
        self.mem = []
        self.length = 0

    @property
    def count(self) -> int:
        return len(self.mem)

    def update_length(self, length: int) -> None:
        self.length = length
        del self.mem[length:]

    def add(self, new_elements: np.ndarray) -> None:
        for sample in new_elements:
            self.mem.append(sample)
        deletable = self.count - self.length
        if deletable > 0:
            del self.mem[:deletable]

    def retrieve(self, count: int) -> np.ndarray:
        start = self.count - self.length  # might be < 0
        missing_samples = max(0, -start)

        if missing_samples >= count:
            # Too early to provide feedback, return zeroes
            raise ValueError
        delayed_samples = np.array(self.mem[: count - missing_samples])
        if missing_samples > 0:
            delayed_samples = np.pad(
                delayed_samples,
                ((0, missing_samples), (0, 0)),
                "constant",
                constant_values=0,
            )
        return delayed_samples


class Delay(Module):
    def __init__(self, signal_info: SignalInfo, params: DelayParameters):
        super().__init__(signal_info)
        self.memory = _Memory()
        self.params = params

    @property
    def params(self) -> DelayParameters:
        return self._params

    @params.setter
    def params(self, value) -> None:
        self._params = value
        n_samples = self.signal_info.ms_to_samples(self.params.delay)
        n_samples = max(self.signal_info.buffer_size, n_samples)
        self.memory.update_length(n_samples)

    def process(self, input: np.ndarray) -> np.ndarray:
        output = input

        try:
            delayed_sample = self.memory.retrieve(self.signal_info.buffer_size)
        except ValueError:
            delayed_sample = np.zeros(
                (self.signal_info.buffer_size, self.signal_info.channels),
                dtype=np.float32,
            )
        output += delayed_sample * self.params.feedback

        # Add the sample to the end of the deque (right)
        # Once the memory is full, the oldest sample (left) will be removed
        self.memory.add(output)

        return output

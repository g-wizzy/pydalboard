from pydalboard.signal.base import SignalInfo, SignalSource

import math
import numpy as np
from time import perf_counter_ns

from enum import Enum


class Waveform(Enum):
    SINE = 1
    TRIANGLE = 2
    SQUARE = 3
    SAWTOOTH = 4


class Oscillator(SignalSource):
    def __init__(self, waveform: Waveform, frequency: float, signal_info: SignalInfo):
        self.waveform = waveform
        self.frequency = frequency
        self.info = signal_info

        self.t = 0

    @property
    def signal_info(self) -> SignalInfo:
        return self.info

    def get_signal(self) -> tuple[np.ndarray, SignalInfo]:
        self.t += 1 / 44_800

        theta = 2 * math.pi * self.t * self.frequency
        match self.waveform:
            case Waveform.SINE:
                value = math.sin(theta)
            case Waveform.TRIANGLE:
                value = 2 / math.pi * math.asin(math.sin(theta))
            case Waveform.SQUARE:
                value = (
                    1
                    if self.t % (1 / self.frequency) < 1 / (2 * self.frequency)
                    else -1
                )
            case Waveform.SAWTOOTH:
                value = 2 / math.pi * math.atan(math.tan(theta))

        value *= 2147483647
        frame = np.array([value, value], dtype="int32")
        return (frame, self.signal_info)

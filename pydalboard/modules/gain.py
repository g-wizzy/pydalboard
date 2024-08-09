from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module


@dataclass
class GainParameters:
    gain: float
    "Amount of gain to add/remove to/from the signal"


    min: float = -66.0 # From Ableton Live Utility plugin
    "Minimal value the gain can reduce the incoming signal to"


    max: float = 36.0 # From Ableton Live Utility plugin
    "Maximum value the gain can increase the incoming signal to"


    def __post_init__(self):
        self.gain = min(max(self.min, self.gain), self.max)


class Gain(Module):
    def __init__(self, params: GainParameters):
        self.params = params

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        # Convert dB gain to linear scale
        linear_gain = 10 ** (self.params.gain / 20.0)
        output = input * linear_gain

        return output

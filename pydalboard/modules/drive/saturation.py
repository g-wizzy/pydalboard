from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module
from pydalboard.modules.gain import Gain, GainParameters

@dataclass
class SaturationParameters():
    drive: float
    "Amount of drive to add to the signal (can be negative to reduce incoming signal)"


    def __post_init__(self):
        # From Ableton Live Saturator plugin
        self.gain = Gain(GainParameters(gain=self.drive, min=-36.0, max=36.0))


class Saturation(Module):
    def __init__(self, params: SaturationParameters):
        self.params = params

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        # Apply gain before saturation
        output = self.params.gain.process(input, signal_info)

        # Apply soft saturation using tanh function
        # tanh function returns values between -1.0 and 1.0
        output = np.tanh(output)

        return output

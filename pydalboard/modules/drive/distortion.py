from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module
from pydalboard.modules.gain import Gain, GainParameters

@dataclass
class DistortionParameters():
    drive: float
    "Amount of drive to add to the signal (can be negative to reduce incoming signal)"


    def __post_init__(self):
        self.gain = Gain(GainParameters(gain=self.drive, min=-36.0, max=36.0))


class Distortion(Module):
    def __init__(self, params: DistortionParameters):
        self.params = params

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        # Apply gain before saturation
        output = self.params.gain.process(input, signal_info)
        
        # Apply distortion (severe clipping)
        output = np.clip(output, -1.0, 1.0)

        return output

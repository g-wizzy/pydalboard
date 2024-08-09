from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module
from pydalboard.modules.gain import Gain, GainParameters

@dataclass
class OverdriveParameters():
    drive: float
    "Amount of drive to add to the signal (can be negative to reduce incoming signal)"


    threshold: float = 1.0
    "Clipping threshold"


    asymmetry: float = 0.5
    "0.0 for symmetrical, 1.0 for extreme asymmetry"
    

    def __post_init__(self):
        self.gain = Gain(GainParameters(gain=self.drive, min=-36.0, max=36.0))
        self.threshold = min(1.0, self.threshold)
        self.asymmetry = np.clip(self.asymmetry, 0.0, 1.0)


class Overdrive(Module):
    def __init__(self, params: OverdriveParameters):
        self.params = params

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        # Apply gain before saturation
        output = self.params.gain.process(input, signal_info)
        
        # Apply asymmetrical clipping
        positive_clip = self.params.threshold
        negative_clip = -self.params.threshold * (1 - self.params.asymmetry)
        
        output = np.where(output > positive_clip, positive_clip, output)
        output = np.where(output < negative_clip, negative_clip, output)

        return output

from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module
from .drive import Drive, DriveParameters

@dataclass
class OverdriveParameters(DriveParameters):
    threshold: float = 1.0
    "Clipping threshold"


    asymmetry: float = 0.5
    "0.0 for symmetrical, 1.0 for extreme asymmetry"


    def __post_init__(self):
        super().__post_init__()
        self.threshold = max(0.01, self.threshold)
        self.asymmetry = np.clip(self.asymmetry, 0.0, 1.0)


class Overdrive(Drive):
    def __init__(self, params: OverdriveParameters):
        self.params = params

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        output = input * self.params.drive
        
        # Apply overdrive (asymmetrical clipping based on the threshold)
        # Positive side clipping
        output[output > self.params.threshold] = self.params.threshold
        # Negative side clipping with asymmetry
        output[output < -self.params.threshold] = -self.params.threshold * (1 - self.params.asymmetry)
        
        self._clip(output)

        return output

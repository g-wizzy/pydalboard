from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module
from .drive import Drive, DriveParameters

@dataclass
class DistortionParameters(DriveParameters):

    def __post_init__(self):
        self.drive = max(0.01, self.drive) # Negative drive is possible to add distortion while reducing gain


class Distortion(Drive):
    def __init__(self, params: DistortionParameters):
        self.params = params

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        output = input * self.params.drive
        
        # Apply distortion (severe clipping)
        output = np.clip(output, -1.0, 1.0)
        
        # Clip the output to avoid distortion
        if self.params.clipping:
            output = self._clip(output)

        return output

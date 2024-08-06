from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module
from .drive import Drive, DriveParameters

@dataclass
class SaturationParameters(DriveParameters):

    def __post_init__(self):
        self.drive = max(0.01, self.drive) # Negative drive is possible to add saturation while reducing gain


class Saturation(Drive):
    def __init__(self, params: SaturationParameters):
        self.params = params

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        output = input * self.params.drive

        # Apply soft saturation using tanh function
        output = np.tanh(output)

        # Clip the output to avoid distortion
        if self.params.clipping:
            output = self._clip(output)

        return output

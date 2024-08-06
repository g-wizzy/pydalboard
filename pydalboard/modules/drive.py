from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module

@dataclass
class DriveParameters:
    drive: float # Amount of drive to add to the signal
    clipping: bool = False # Whether to clip the signal or not

    def __post_init__(self):
        self.drive = max(0.01, self.drive) # Negative drive is possible to reduce the signal


class Drive(Module):
    def __init__(self, params: DriveParameters):
        self.params = params

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        output = input * self.params.drive

        if self.params.clipping:
            output = self._clip(output)

        return output

    def _clip(self, input: np.ndarray) -> np.ndarray:
        # Clip the output to reduce distortion
        return np.clip(input, -1.0, 1.0)

from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module
from .drive import Drive, DriveParameters

@dataclass
class SaturationParameters(DriveParameters):

    def __post_init__(self):
        super().__post_init__()


class Saturation(Drive):
    def __init__(self, params: SaturationParameters):
        self.params = params

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        output = input * self.params.drive

        # Apply soft saturation using tanh function
        output = np.tanh(output)

        self._clip(output)

        return output

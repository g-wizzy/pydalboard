from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module


@dataclass
class DriveParameters:
    gain: float
    "Amount of drive to add to the signal"

    clipping: bool
    "Whether to clip the signal or not"

    def __post_init__(self):
        self.gain = max(1.0, self.gain)


class Drive(Module):
    def __init__(self, params: DriveParameters):
        self.params = params

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        output = input * self.params.gain

        # Clip the output to avoid distortion
        if self.params.clipping:
            output = np.clip(output, -1.0, 1.0)

        return output

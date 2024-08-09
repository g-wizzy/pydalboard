from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module


@dataclass
class DriveParameters():
    drive: float
    "Amount of drive to add to the signal"

    clipping: bool = False
    "Whether to clip the signal or not"

    def __post_init__(self):
        self.drive = max(1.0, self.drive)


class Drive(Module):
    def __init__(self, params: DriveParameters):
        self.params = params

    @abstractmethod
    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray: ...

    def _clip(self, input: np.ndarray) -> np.ndarray:
        # Clip the output to reduce distortion
        if self.params.clipping:
            return np.clip(input, -1.0, 1.0)
        return input

from dataclasses import dataclass

import numpy as np

from pydalboard.modules.base import Module, ModuleParams
from pydalboard.modules.gain import Gain, GainParameters
from pydalboard.signal.base import SignalInfo


@dataclass
class OverdriveParameters(ModuleParams):
    drive: float
    "Amount of drive to add to the signal (can be negative to reduce incoming signal)"

    threshold: float = 1.0
    "Clipping threshold"

    asymmetry: float = 0.5
    "0.0 for symmetrical, 1.0 for extreme asymmetry"

    def __post_init__(self):
        self.drive = min(max(-36.0, self.drive), 36.0)
        self.threshold = min(1.0, self.threshold)
        self.asymmetry = np.clip(self.asymmetry, 0.0, 1.0)


class Overdrive(Module):
    def __init__(self, signal_info: SignalInfo, params: OverdriveParameters):
        super().__init__(signal_info)
        self.params = params
        self.gain = Gain(
            signal_info, GainParameters(gain=self.params.drive, min=-36.0, max=36.0)
        )

    def process(self, input: np.ndarray) -> np.ndarray:
        # Apply gain before saturation
        output = self.gain.process(input)

        # Apply asymmetrical clipping
        positive_clip = self.params.threshold
        negative_clip = -self.params.threshold * (1 - self.params.asymmetry)

        output = np.where(output > positive_clip, positive_clip, output)
        output = np.where(output < negative_clip, negative_clip, output)

        return output

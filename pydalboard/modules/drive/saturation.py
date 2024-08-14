from dataclasses import dataclass

import numpy as np

from pydalboard.modules.base import Module, ModuleParams
from pydalboard.modules.gain import Gain, GainParameters
from pydalboard.signal.base import SignalInfo


@dataclass
class SaturationParameters(ModuleParams):
    drive: float
    "Amount of drive to add to the signal (can be negative to reduce incoming signal)"

    def __post_init__(self):
        # From Ableton Live Saturator plugin
        self.drive = min(max(-36.0, self.drive), 36.0)


class Saturation(Module):
    def __init__(self, signal_info: SignalInfo, params: SaturationParameters):
        super().__init__(signal_info)
        self.params = params
        self.gain = Gain(
            signal_info, GainParameters(gain=self.params.drive, min=-36.0, max=36.0)
        )

    def process(self, input: np.ndarray) -> np.ndarray:
        # Apply gain before saturation
        output = self.gain.process(input)

        # Apply soft saturation using tanh function
        # tanh function returns values between -1.0 and 1.0
        output = np.tanh(output)

        return output

from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module

from enum import Enum


class FilterType(Enum):
    LOW_PASS = 1
    HIGH_PASS = 2
    BAND_PASS = 3


@dataclass
class FilterParameters:
    cutoff: float
    "Cutoff frequency in Hz"

    resonance: float
    "Resonance (Q)"  # TODO: define possible values (+6db is ~ √2)

    filter_type: FilterType
    "Type of filterType of filter"

    slope: int
    "Slope in dB/octave: 12 or 24Slope in dB/octave: 12 or 24"

    def __post_init__(self):
        self.b0, self.b1, self.b2, self.a1, self.a2 = (
            self.calculate_biquad_coefficients()
        )

    def calculate_biquad_coefficients(self) -> tuple:
        """
        Digital Biquad filter (2nd order filter)
        """
        Q = max(1.0, self.resonance)
        omega = 2 * np.pi * self.cutoff / 44100  # TODO: get sample rate for audio
        alpha = np.sin(omega) / (2 * Q)
        cos_omega = np.cos(omega)

        match self.filter_type:
            case FilterType.LOW_PASS:
                b0 = (1 - cos_omega) / 2
                b1 = 1 - cos_omega
                b2 = (1 - cos_omega) / 2
                a0 = 1 + alpha
                a1 = -2 * cos_omega
                a2 = 1 - alpha
            case FilterType.HIGH_PASS:
                b0 = (1 + cos_omega) / 2
                b1 = -(1 + cos_omega)
                b2 = (1 + cos_omega) / 2
                a0 = 1 + alpha
                a1 = -2 * cos_omega
                a2 = 1 - alpha
            case FilterType.BAND_PASS:
                b0 = alpha
                b1 = 0
                b2 = -alpha
                a0 = 1 + alpha
                a1 = -2 * cos_omega
                a2 = 1 - alpha

        # Normalize coefficients
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0

        return b0, b1, b2, a1, a2


class Filter(Module):
    def __init__(self, params: FilterParameters):
        self.params = params

        self.prev_inputs = [
            np.zeros(2, "float32"),
            np.zeros(2, "float32"),
        ]  # [x1[n-1], x1[n-2]], [x2[n-1], x2[n-2]]
        self.prev_outputs = [
            np.zeros(2, "float32"),
            np.zeros(2, "float32"),
        ]  # [y1[n-1], y1[n-2]], [y2[n-1], y2[n-2]]

    def apply_biquad_filter(self, sample: np.ndarray) -> np.ndarray:
        """
        Apply Biquad filter to the given sample.
        """
        filtered_sample = (
            self.params.b0 * sample
            + self.params.b1 * self.prev_inputs[-1]
            + self.params.b2 * self.prev_inputs[-2]
            - self.params.a1 * self.prev_outputs[-1]
            - self.params.a2 * self.prev_outputs[-2]
        )
        return filtered_sample

    def update_memory(self, input: np.ndarray, output: np.ndarray) -> None:
        self.prev_inputs.pop(0)
        self.prev_inputs.append(input)
        self.prev_outputs.pop(0)
        self.prev_outputs.append(output)

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        # Apply filters based on the slope
        filtered = self.apply_biquad_filter(input)
        self.update_memory(input, filtered)

        if self.params.slope == 24:
            # Apply filters twice if the slope is 24db/octave
            filtered = self.apply_biquad_filter(filtered)
            self.update_memory(input, filtered)

        return filtered

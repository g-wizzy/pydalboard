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
    cutoff: float  # Cutoff frequency in Hz
    resonance: float  # Resonance (Q) # TODO: define possible values (+6db is ~ √2)
    filter_type: FilterType  # Type of filter
    slope: int  # Slope in dB/octave: 12 or 24

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

        self.prev_input = [
            (0.0, 0.0),
            (0.0, 0.0),
        ]  # [x1[n-1], x1[n-2]], [x2[n-1], x2[n-2]]
        self.prev_output = [
            (0.0, 0.0),
            (0.0, 0.0),
        ]  # [y1[n-1], y1[n-2]], [y2[n-1], y2[n-2]]

    def apply_biquad_filter(self, sample: float, look_back: int) -> float:
        """
        Apply Biquad filter to the given sample.
        """
        filtered_sample = (
            self.params.b0 * sample
            + self.params.b1 * self.prev_input[look_back][0]
            + self.params.b2 * self.prev_input[look_back][1]
            - self.params.a1 * self.prev_output[look_back][0]
            - self.params.a2 * self.prev_output[look_back][1]
        )
        return filtered_sample

    def update_memories(self, sample: tuple[float, float]):
        """
        Update the previous inputs and outputs memories.
        """
        self.prev_input.pop()
        self.prev_input.insert(0, sample)
        self.prev_input.pop()
        self.prev_input.insert(0, sample)

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        # Apply filters based on the slope
        filtered_left = self.apply_biquad_filter(input[0], 0)
        filtered_right = self.apply_biquad_filter(input[1], 0)

        if self.params.slope == 24:
            # Apply filters twice if the slope is 24db/octave
            filtered_left = self.apply_biquad_filter(filtered_left, 1)
            filtered_right = self.apply_biquad_filter(filtered_right, 1)

        self.update_memories((filtered_left, filtered_right))

        return np.array([filtered_left, filtered_right])

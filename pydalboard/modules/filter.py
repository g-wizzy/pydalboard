from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module

@dataclass
class FilterParameters:
    cutoff: float  # Cutoff frequency in Hz
    resonance: float  # Resonance (Q) # TODO: define possible values (+6db is ~ √2)
    filter_type: str # Type of filter: 'low', 'high' or 'band'
    slope: int  # Slope in dB/octave: 12 or 24

    def __post_init__(self):
        self.b0, self.b1, self.b2, self.a1, self.a2 = self.calculate_biquad_coefficients()
        # Initialize filters memories
        self.prev_input = [[0.0, 0.0], [0.0, 0.0]]  # [x1[n-1], x1[n-2]], [x2[n-1], x2[n-2]]
        self.prev_output = [[0.0, 0.0], [0.0, 0.0]]  # [y1[n-1], y1[n-2]], [y2[n-1], y2[n-2]]

    def calculate_biquad_coefficients(self) -> tuple:
        """
        Digital Biquad filter (2nd order filter)
        """
        Q = max(1.0, self.resonance)
        omega = 2 * np.pi * self.cutoff / 44100  # TODO: get sample rate for audio
        alpha = np.sin(omega) / (2 * Q)
        cos_omega = np.cos(omega)

        if self.filter_type == 'low':
            b0 = (1 - cos_omega) / 2
            b1 = 1 - cos_omega
            b2 = (1 - cos_omega) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_omega
            a2 = 1 - alpha
        elif self.filter_type == 'high':
            b0 = (1 + cos_omega) / 2
            b1 = -(1 + cos_omega)
            b2 = (1 + cos_omega) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_omega
            a2 = 1 - alpha
        elif self.filter_type == 'band':
            b0 = alpha
            b1 = 0
            b2 = -alpha
            a0 = 1 + alpha
            a1 = -2 * cos_omega
            a2 = 1 - alpha
        else:
            raise ValueError(f"Unsupported filter type: {self.filter_type}")

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

    def apply_biquad_filter(self, sample: float, prev_input: list, prev_output: list) -> float:
        """
        Apply Biquad filter to the given sample.
        """
        filtered_sample = (self.params.b0 * sample +
                           self.params.b1 * prev_input[0] +
                           self.params.b2 * prev_input[1] -
                           self.params.a1 * prev_output[0] -
                           self.params.a2 * prev_output[1])
        return filtered_sample

    def update_memories(self, sample: float, prev_input: list, prev_output: list):
        """
        Update the previous inputs and outputs memories.
        """
        prev_input[1] = prev_input[0]
        prev_input[0] = sample
        prev_output[1] = prev_output[0]
        prev_output[0] = sample

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        output = np.zeros_like(input)
        
        left_sample = input[0]
        right_sample = input[1]

        # Apply filters based on the slope
        filtered_left_1 = self.apply_biquad_filter(left_sample, self.params.prev_input[0], self.params.prev_output[0])
        filtered_right_1 = self.apply_biquad_filter(right_sample, self.params.prev_input[0], self.params.prev_output[0])

        if self.params.slope == 24:
            # Apply filters twice if the slope is 24db/octave
            filtered_left_2 = self.apply_biquad_filter(filtered_left_1, self.params.prev_input[1], self.params.prev_output[1])
            filtered_right_2 = self.apply_biquad_filter(filtered_right_1, self.params.prev_input[1], self.params.prev_output[1])
            
            self.update_memories(filtered_left_1, self.params.prev_input[0], self.params.prev_output[0])
            self.update_memories(filtered_right_1, self.params.prev_input[0], self.params.prev_output[0])
            self.update_memories(filtered_left_2, self.params.prev_input[1], self.params.prev_output[1])
            self.update_memories(filtered_right_2, self.params.prev_input[1], self.params.prev_output[1])
            
            output[0] = filtered_left_2
            output[1] = filtered_right_2
        else:
            # Filters were applied once (12db/octave slope)
            self.update_memories(filtered_left_1, self.params.prev_input[0], self.params.prev_output[0])
            self.update_memories(filtered_right_1, self.params.prev_input[0], self.params.prev_output[0])
            
            output[0] = filtered_left_1
            output[1] = filtered_right_1

        return output

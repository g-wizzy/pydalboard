from dataclasses import dataclass

import numpy as np
from numba import jit

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module, ModuleParams

from enum import Enum


class FilterType(Enum):
    LOW_PASS = 1
    LOW_SHELF_PASS = 2
    HIGH_PASS = 3
    HIGH_SHELF_PASS = 4
    BAND_PASS = 5


@dataclass
class FilterParameters(ModuleParams):
    cutoff: float
    "Cutoff frequency in Hz"

    gain_db: float
    "Gain in dB. Only applied for shelving filters"

    resonance: float
    "Resonance (Q)"  # TODO: define possible values (+6db is ~ √2)

    filter_type: FilterType
    "Type of filter"

    slope: int
    "Slope in dB/octave: choice among 6dB, 12dB, 18dB and 24dB"

    def compute_coefficients(
        self, signal_info: SignalInfo
    ) -> tuple[np.ndarray, np.ndarray]:
        Q = max(1.0, self.resonance)
        A = 10 ** (self.gain_db / 40)
        S = self.slope / 24

        omega = 2 * np.pi * self.cutoff / signal_info.sample_rate
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)

        alpha = sin_omega / (2 * Q)
        beta = np.sqrt(A) * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)

        a = np.zeros(3, np.float32)
        b = np.zeros(3, np.float32)

        match self.filter_type:
            case FilterType.LOW_PASS:
                b[0] = (1 - cos_omega) / 2
                b[1] = 1 - cos_omega
                b[2] = (1 - cos_omega) / 2
                a[0] = 1 + alpha
                a[1] = -2 * cos_omega
                a[2] = 1 - alpha
            case FilterType.LOW_SHELF_PASS:
                b[0] = A * ((A + 1) + (A - 1) * cos_omega + beta * sin_omega)
                b[1] = -2 * A * ((A - 1) + (A + 1) * cos_omega)
                b[2] = A * ((A + 1) + (A - 1) * cos_omega - beta * sin_omega)
                a[0] = (A + 1) - (A - 1) * cos_omega + beta * sin_omega
                a[1] = 2 * ((A - 1) - (A + 1) * cos_omega)
                a[2] = (A + 1) - (A - 1) * cos_omega - beta * sin_omega
            case FilterType.HIGH_PASS:
                b[0] = (1 + cos_omega) / 2
                b[1] = -(1 + cos_omega)
                b[2] = (1 + cos_omega) / 2
                a[0] = 1 + alpha
                a[1] = -2 * cos_omega
                a[2] = 1 - alpha
            case FilterType.HIGH_SHELF_PASS:
                b[0] = A * ((A + 1) - (A - 1) * cos_omega + beta * sin_omega)
                b[1] = 2 * A * ((A - 1) - (A + 1) * cos_omega)
                b[2] = A * ((A + 1) - (A - 1) * cos_omega - beta * sin_omega)
                a[0] = (A + 1) + (A - 1) * cos_omega + beta * sin_omega
                a[1] = -2 * ((A - 1) + (A + 1) * cos_omega)
                a[2] = (A + 1) + (A - 1) * cos_omega - beta * sin_omega
            case FilterType.BAND_PASS:
                b[0] = alpha
                b[1] = 0
                b[2] = -alpha
                a[0] = 1 + alpha
                a[1] = -2 * cos_omega
                a[2] = 1 - alpha

        # Normalize coefficients
        b /= a[0]
        a /= a[0]

        return a, b


@jit
def apply_biquad(
    inputs: np.ndarray,
    outputs: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    count: int,
) -> None:
    for i in range(2, 2 + count):
        outputs[i] = (
            b[0] * inputs[i]
            + b[1] * inputs[i - 1]
            + b[2] * inputs[i - 2]
            - a[1] * outputs[i - 1]
            - a[2] * outputs[i - 2]
        )


class Filter(Module):
    def __init__(self, signal_info: SignalInfo, params: FilterParameters):
        super().__init__(signal_info)
        self.params = params
        self.a, self.b = self.params.compute_coefficients(self.signal_info)

        if self.signal_info.channels == 2:
            self.prev_inputs = np.array(
                [
                    np.zeros(2, "float32"),
                    np.zeros(2, "float32"),
                ]
            )
            self.prev_outputs = np.array(
                [
                    np.zeros(2, "float32"),
                    np.zeros(2, "float32"),
                ]
            )
        else:
            self.prev_inputs = np.array([0, 0])
            self.prev_outputs = np.array([0, 0])

    def process(self, input: np.ndarray) -> np.ndarray:
        outputs = np.concatenate([self.prev_outputs, np.zeros(input.shape, np.float32)])
        inputs = np.concatenate([self.prev_inputs, input])

        apply_biquad(
            inputs,
            outputs,
            self.a,
            self.b,
            self.signal_info.buffer_size,
        )

        self.prev_inputs = inputs[-2:]
        self.prev_outputs = outputs[-2:]

        outputs = outputs.clip(-1, 1)

        return np.array(outputs[2:])

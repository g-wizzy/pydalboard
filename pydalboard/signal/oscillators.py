from pydalboard.signal.base import SignalInfo, SignalSource

import math
import matplotlib.pyplot as plt
import numpy as np
import os
from time import perf_counter_ns

from enum import Enum


class Waveform(Enum):
    SINE = 1
    TRIANGLE = 2
    SQUARE = 3
    SAWTOOTH = 4


class Oscillator(SignalSource):
    __PLOT_WAVEFORM__ = True

    def __init__(
        self,
        waveform: Waveform,
        frequency: float,
        phase: float,
        signal_info: SignalInfo,
        table_size: int = 1024,
        cycles: int | None = None,
    ):
        self.waveform = waveform
        self.frequency = frequency
        self.phase = phase

        self.info = signal_info
        self.table_size = table_size  # Audio resolution (number of samples)
        self.cycles = cycles

        # Current position in the waveform
        self.t = phase % 1.0
        self.cycles_played = 0
        self.total_samples = 0

        # Precompute waveform table
        self._table = self._compute_waveform_table()
        self._table_size = len(self._table)

        # Plot the waveform and save it to a file if __PLOT_WAVEFORM__ is True
        if self.__PLOT_WAVEFORM__:
            self._plot_waveform(self._table, self.waveform)

    @property
    def signal_info(self) -> SignalInfo:
        return self.info

    def get_signal(self) -> tuple[np.ndarray, SignalInfo]:
        # Stop the sound if the number of cycles is reached
        if self.cycles is not None and self.cycles_played >= self.cycles:
            return (np.zeros(2, dtype="float32"), self.signal_info)

        index = int(self.t * self.frequency * self._table_size) % self.table_size
        value = self._table[index]

        frame = np.array([value, value], dtype="float32")
        self.t += 1 / self.signal_info.sample_rate
        self.total_samples += 1

        # Check if a cycle has completed
        if self.total_samples >= self.signal_info.sample_rate / self.frequency:
            self.cycles_played += 1
            self.total_samples = 0

        return (frame, self.signal_info)

    def _compute_waveform_table(self) -> np.ndarray:
        table = np.zeros(self.table_size, dtype="float32")
        theta_step = 2 * math.pi / self.table_size

        for i in range(self.table_size):
            theta = i * theta_step + self.phase * 2 * math.pi

            match self.waveform:
                case Waveform.SINE:
                    table[i] = math.sin(theta)
                case Waveform.TRIANGLE:
                    table[i] = 2 / math.pi * math.asin(math.sin(theta))
                case Waveform.SQUARE:
                    table[i] = 1 if (i / self.table_size) % 1 < 0.5 else -1
                case Waveform.SAWTOOTH:
                    table[i] = 2 / math.pi * math.atan(math.tan(theta / 2))

        # Normalize the table
        table /= max(abs(table.min()), abs(table.max()))

        return table

    def _plot_waveform(self, data: np.ndarray, waveform: Waveform):
        # Plot the waveform
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title(f"{waveform.name} Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Save the plot to a file
        if not os.path.exists("_waveforms"):
            os.mkdir("_waveforms")
        filename = os.path.join("_waveforms", f"{waveform.name.lower()}_waveform.png")
        plt.savefig(filename)
        plt.close()

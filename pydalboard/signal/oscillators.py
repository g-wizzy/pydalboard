from pydalboard.signal.base import SignalInfo, SignalSource

import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
        self.t = 0
        self.cycles_played = 0

        # Precompute waveform table
        self._table = self._compute_waveform_table()

        # Plot the waveform and save it to a file if __PLOT_WAVEFORM__ is True
        if self.__PLOT_WAVEFORM__:
            self._plot_waveform(self._table)

    @property
    def signal_info(self) -> SignalInfo:
        return self.info

    def get_signal(self) -> np.ndarray:
        # Stop the sound if the number of cycles is reached
        if self.cycles is not None and self.cycles_played >= self.cycles:
            return np.zeros((self.info.buffer_size, 2), dtype=np.float32)

        delta_t = self.info.buffer_size / self.info.sample_rate * self.frequency
        buffer = self._compute_buffer(delta_t)
        self.t += delta_t

        if self.t >= 1:
            cycles, self.t = divmod(self.t, 1)
            self.cycles_played += cycles

        return buffer

    def _compute_buffer(self, delta_t: float) -> np.ndarray:
        indices = [
            int(t * self.table_size) % self.table_size
            for t in np.linspace(
                self.t + self.phase,
                self.t + self.phase + delta_t,
                num=self.info.buffer_size,
            )
        ]
        values = self._table.take(indices)
        buffer = np.dstack((values, values))[0] if self.info.channels == 2 else values

        return buffer

    def _compute_waveform_table(self) -> np.ndarray:
        table = np.linspace(0, 2 * math.pi, num=self.table_size)

        match self.waveform:
            case Waveform.SINE:
                table = np.sin(table)
            case Waveform.TRIANGLE:
                table = 2 / math.pi * np.asin(np.sin(table))
            case Waveform.SQUARE:
                table = -np.sign(table - math.pi)
            case Waveform.SAWTOOTH:
                table = 2 / math.pi * np.atan(np.tan(table / 2))

        return table

    def _plot_waveform(self, data):
        # Plot the waveform
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title(f"{self.waveform.name} Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # Save the plot to a file
        output_directory = Path("_waveforms")
        output_directory.mkdir(exist_ok=True)

        output_file = output_directory / f"{self.waveform.name.lower()}_waveform.png"
        plt.savefig(output_file)
        plt.close()

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pathlib import Path

import numpy as np
from scipy.io import wavfile


@dataclass
class SignalInfo:
    """
    Wrapper for the information defining a signal:
        - Sample rate, expressed in kHz
        - Sample format, in number of bits
    """

    sample_rate: int
    sample_format: int
    stereo: bool


class SignalSource(ABC):
    @property
    @abstractmethod
    def signal_info(self) -> SignalInfo: ...

    @abstractmethod
    def get_signal(self) -> tuple[np.ndarray, SignalInfo]: ...


class Wav(SignalSource):
    def __init__(self, file: Path, loop: bool) -> None:
        sample_rate, data = wavfile.read(file.absolute())
        self.info = SignalInfo(
            sample_rate=sample_rate, sample_format=32, stereo=data.shape[1] == 2
        )
        self.loop = loop
        self.ended = False

        self.data = data
        self.read_index = 0

    @property
    def signal_info(self) -> SignalInfo:
        return self.info

    def get_signal(self) -> tuple[np.ndarray, SignalInfo]:
        if self.ended:
            return (np.array([0, 0]), self.signal_info)

        state = self.data[self.read_index]
        self.read_index += 1
        if self.read_index >= len(self.data) and not self.loop:
            self.ended = True
        else:
            self.read_index %= len(self.data)

        return (
            state,
            self.signal_info,
        )

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


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

    def convert_to_format(self, frame: np.ndarray) -> np.ndarray:
        match self.sample_format:
            case 16:
                return (frame * 32767).astype(np.int16)
            case 32:
                return (frame * 2147483647).astype(np.int32)
            case _:
                return np.zeros(2 if self.stereo else 1, np.float32)


class SignalSource(ABC):
    @property
    @abstractmethod
    def signal_info(self) -> SignalInfo: ...

    @abstractmethod
    def get_signal(self) -> tuple[np.ndarray, SignalInfo]: ...

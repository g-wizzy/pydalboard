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


class SignalSource(ABC):
    @property
    @abstractmethod
    def signal_info(self) -> SignalInfo: ...

    @abstractmethod
    def get_signal(self) -> tuple[np.ndarray, SignalInfo]: ...

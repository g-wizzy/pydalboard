from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SignalInfo:
    """
    Wrapper for the information defining a signal."""

    sample_rate: int
    "Sample rate, expressed in Hz"

    sample_format: int
    "Sample format, in number of bits"

    channels: int
    "Audio channels, aka 1 for mono and 2 for stereo"


class SignalSource(ABC):
    @property
    @abstractmethod
    def signal_info(self) -> SignalInfo: ...

    @abstractmethod
    def get_signal(self) -> tuple[np.ndarray, SignalInfo]: ...

from abc import ABC, abstractmethod

import numpy as np

from pydalboard.signal import SignalInfo


class Module(ABC):
    """
    Abstract class, intended to be specialized into many modules, whose role is
    to process Signal objects into many interesting effects.
    """

    @abstractmethod
    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray: ...

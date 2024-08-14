from abc import ABC, abstractmethod

import numpy as np

from pydalboard.signal import SignalInfo


class ModuleParams:
    pass


class Module(ABC):
    """
    Abstract class, intended to be specialized into many modules, whose role is
    to process Signal objects into many interesting effects.
    """

    def __init__(self, signal_info: SignalInfo) -> None:
        self.signal_info = signal_info

    @abstractmethod
    def process(self, input: np.ndarray) -> np.ndarray: ...

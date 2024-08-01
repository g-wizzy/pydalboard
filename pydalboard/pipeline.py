import numpy as np

from pydalboard.modules.base import Module
from pydalboard.signal import SignalSource


class Pipeline:
    def __init__(self, source: SignalSource) -> None:
        self.source = source
        self._modules = []

    @property
    def modules(self) -> list[Module]:
        return self._modules

    def run(self) -> np.ndarray:
        frame, signal_info = self.source.get_signal()
        for module in self.modules:
            frame = module.process(frame, self.source.signal_info)
        return frame

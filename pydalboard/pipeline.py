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
        frame = self.source.get_signal()
        signal_info = self.source.signal_info
        for module in self.modules:
            frame = module.process(frame, signal_info)

        # Signal was converted to float for processing.
        # We need to convert it back to its original format for ouptut
        frame = signal_info.convert_to_format(frame)
        
        return frame

import numpy as np

from pydalboard.modules.base import Module, ModuleParams
from pydalboard.modules.delay import Delay, DelayParameters
from pydalboard.modules.drive.distortion import Distortion, DistortionParameters
from pydalboard.modules.drive.overdrive import Overdrive, OverdriveParameters
from pydalboard.modules.drive.saturation import Saturation, SaturationParameters
from pydalboard.modules.filter import Filter, FilterParameters
from pydalboard.signal import SignalSource


class Pipeline:
    def __init__(self, source: SignalSource) -> None:
        self.source = source
        self.modules: list[Module] = []

    def add_module(self, module_parameters: ModuleParams) -> None:
        match module_parameters:
            case DelayParameters():
                self.modules.append(Delay(self.source.signal_info, module_parameters))
            case FilterParameters():
                self.modules.append(Filter(self.source.signal_info, module_parameters))
            case DistortionParameters():
                self.modules.append(
                    Distortion(self.source.signal_info, module_parameters)
                )
            case OverdriveParameters():
                self.modules.append(
                    Overdrive(self.source.signal_info, module_parameters)
                )
            case SaturationParameters():
                self.modules.append(
                    Saturation(self.source.signal_info, module_parameters)
                )

    def run(self) -> np.ndarray:
        buffer = self.source.get_signal()
        signal_info = self.source.signal_info
        for module in self.modules:
            buffer = module.process(buffer)

        # Signal was converted to float for processing.
        # We need to convert it back to its original format for ouptut
        buffer = signal_info.convert_to_format(buffer)

        return buffer

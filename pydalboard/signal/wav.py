from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pydalboard.signal.base import SignalSource, SignalInfo


class Wav(SignalSource):
    def __init__(self, file: Path, loop: bool) -> None:
        sample_rate, data = wavfile.read(file.absolute())

        # Determine bit depth and max value for normalization
        match data.dtype:
            case np.int16:
                sample_format = 16
                max_value = np.iinfo(np.int16).max
            case np.int32:
                sample_format = 32
                max_value = np.iinfo(np.int32).max
            case np.float32:
                sample_format = 32
                max_value = 1.0
            case np.float64:
                sample_format = 64
                max_value = 1.0
            case _:
                raise ValueError("Unsupported audio format")

        self.info = SignalInfo(
            sample_rate=sample_rate,
            sample_format=sample_format,
            channels=data.shape[1],
        )
        self.loop = loop
        self.ended = False
        self.read_index = 0

        # Normalize audio data to float32
        # Float32 is better suited to process audio, especially when adding gain to avoid clipping
        if sample_format in [16, 32]:
            self.data = data.astype(np.float32) / max_value
        else:
            self.data = data

    @property
    def signal_info(self) -> SignalInfo:
        return self.info

    def get_signal(self) -> tuple[np.ndarray, SignalInfo]:
        if self.ended:
            return (np.zeros(2, dtype=np.float32), self.signal_info)

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

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pydalboard.signal.base import SignalSource, SignalInfo


class Wav(SignalSource):
    def __init__(self, file: Path, buffer_size: int, loop: bool) -> None:
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
            buffer_size=buffer_size,
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

    def get_signal(self) -> np.ndarray:
        if self.ended:
            return np.zeros((self.info.buffer_size, 2), dtype=np.float32)

        buffer = self.data[
            self.read_index : self.read_index + self.info.buffer_size
        ]  # Numpy accepts out of range values, but does not pad the output with zeros of nans
        self.read_index += self.info.buffer_size

        if self.read_index >= len(self.data):
            self.read_index = self.read_index - len(self.data)
            if self.loop:
                buffer = np.concatenate([buffer, self.data[0 : self.read_index]])
            else:
                buffer = np.pad(
                    buffer,
                    ((0, self.read_index), (0, 0)),
                    "constant",
                    constant_values=0,
                )
                self.ended = True

        return buffer

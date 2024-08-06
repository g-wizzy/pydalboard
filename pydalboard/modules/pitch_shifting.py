from typing import ClassVar
from dataclasses import dataclass
from collections import deque

import numpy as np
from scipy.signal import resample

from pydalboard.signal import SignalInfo
from pydalboard.modules.base import Module


@dataclass
class PitchShiftingParameters:
    pitch_factor: float # Factor by which to shift the pitch
    warp: bool # Whether to preserve the sample length or not

    def __post_init__(self):
        # 0.5: lower pitch by one octave
        # 2.0: raises pitch by one octave
        # Note that this pitch shifting algorithm will introduce artifacts
        self.pitch_factor = max(0.5, min(self.pitch_factor, 2.0))


# Basic resampling (not using Phase Vocoder)
class PitchShifting(Module):
    def __init__(self, params: PitchShiftingParameters, sample_rate: int, frame_size: int = 2048, hop_size: int = 512):
        self.params = params
        self.sample_rate = sample_rate
        self.frame_size = frame_size # Define the required number of samples before resampling (chunk of data)
        self.hop_size = hop_size # Define the required number of processed samples before outputting (rate of output)

        self.input_buffers = [deque(maxlen=self.frame_size) for _ in range(2)]
        self.output_buffers = [deque() for _ in range(2)]

    def process(self, input: np.ndarray, signal_info: SignalInfo) -> np.ndarray:
        output = np.zeros_like(input)

        for channel in range(2):
            # Fill the input buffer, because pitch shifting operates on
            # frames (chunks) of audio data, not individual samples
            self.input_buffers[channel].append(input[channel])

            if len(self.input_buffers[channel]) == self.frame_size:
                # The input buffer has accumulated enough samples
                # Apply the pitch shifting algorithm to the frame (chunk)
                frame = np.array(self.input_buffers[channel])
                pitched_frame = self.pitch_shift(frame, self.params.pitch_factor)

                # Add the resampled frame to the output buffer, which
                # accumulates the processed samples
                self.output_buffers[channel].extend(pitched_frame)
                # Clear the input buffer for a new frame (chunk)
                self.input_buffers[channel].clear()

            if len(self.output_buffers[channel]) >= self.hop_size:
                # The output buffer has enough processed samples to
                # output one sample
                output[channel] = self.output_buffers[channel].popleft()
            else:
                output[channel] = 0.0

        return output

    def pitch_shift(self, frame: np.ndarray, pitch_factor: float) -> np.ndarray:
        # Resample the frame (chunk of data) to change the pitch
        # Higher pitch implies lower sample rate
        # Lower pitch implies higher sample rate
        resampled_frame = resample(frame, int(len(frame) / pitch_factor))
        
        if not self.params.warp:
            # Sample original length is not preserved
            return resampled_frame

        # If the resampled frame is too short, pad with zeros
        if len(resampled_frame) < len(frame):
            padded_frame = np.zeros(len(frame))
            padded_frame[:len(resampled_frame)] = resampled_frame
            return padded_frame

        # If the resampled frame is too long, truncate it
        return resampled_frame[:len(frame)]

from pathlib import Path
import sys

import numpy as np
import pyaudio

from pydalboard.pipeline import Pipeline
from pydalboard.signal import Wav
from pydalboard.modules import Delay, DelayParameters, Drive, DriveParameters, Filter, FilterParameters
from pydalboard.signal.base import SignalInfo
from pydalboard.signal.oscillators import Oscillator, Waveform

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Argument needed: wav file")
        sys.exit()

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    wav_source = Wav(Path(sys.argv[1]), loop=False)
    sample_rate = wav_source.signal_info.sample_rate
    sample_format = wav_source.signal_info.sample_format
    channels = 2 if wav_source.signal_info.stereo else 1

    if sample_format == 16:
        pa_format = pyaudio.paInt16
    elif sample_format == 32:
        pa_format = pyaudio.paInt32
    else:
        raise ValueError("Unsupported sample format")

    player = p.open(
        rate=sample_rate,
        channels=channels,
        output=True,
        frames_per_buffer=1,
        format=pa_format,
    )

    pipeline = Pipeline(
        wav_source
        # Oscillator(
        #     waveform=Waveform.SINE,
        #     frequency=440e3,
        #     signal_info=SignalInfo(44_800, 32, True),
        # )
    )
    pipeline.modules.append(Drive(DriveParameters(gain=3.0, clipping=True)))
    # pipeline.modules.append(Filter(FilterParameters(cutoff=3000, resonance=1.414, filter_type='low', slope=12)))
    # pipeline.modules.append(Delay(DelayParameters(duration=10000, decay=0.5)))
    
    while True:
        try:
            frame = pipeline.run()
            if wav_source.signal_info.sample_format in [16, 32]:
                # Convert back from float32 to int16/32
                frame = (frame * (2**31 - 1)).astype(np.int32)
            player.write(frame.tobytes(), 1)
        except KeyboardInterrupt:
            break

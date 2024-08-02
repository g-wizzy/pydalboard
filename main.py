from pathlib import Path
import sys

import pyaudio

from pydalboard.pipeline import Pipeline
from pydalboard.signal import Wav
from pydalboard.modules import Delay, DelayParameters, Filter, FilterParameters
from pydalboard.signal.base import SignalInfo
from pydalboard.signal.oscillators import Oscillator, Waveform

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Argument needed: wav file")
        sys.exit()

    p = pyaudio.PyAudio()
    player = p.open(
        rate=44_100,
        channels=2,
        output=True,
        frames_per_buffer=1,
        format=pyaudio.paInt32,
    )

    pipeline = Pipeline(
        Wav(Path(sys.argv[1]), loop=False)
        # Oscillator(
        #     waveform=Waveform.SINE,
        #     frequency=440e3,
        #     signal_info=SignalInfo(44_800, 32, True),
        # )
    )
    pipeline.modules.append(Delay(DelayParameters(duration=10000, decay=0.5)))
    pipeline.modules.append(Filter(FilterParameters(cutoff=2000, resonance=1.414, filter_type='band', slope=12)))
    
    while True:
        try:
            frame = pipeline.run()
            player.write(frame.tobytes(), 1)
        except KeyboardInterrupt:
            break

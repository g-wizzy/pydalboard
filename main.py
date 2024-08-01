from pathlib import Path
import sys

import pyaudio

from pydalboard.signal import Wav
from pydalboard.modules import Delay, DelayParameters

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Argument needed: wav file")
        sys.exit()

    source = Wav(Path(sys.argv[1]), loop=False)
    delay = Delay(DelayParameters(duration=10000, decay=0.2))

    p = pyaudio.PyAudio()
    player = p.open(
        rate=source.signal_info.sample_rate,
        channels=(2 if source.signal_info.stereo else 1),
        output=True,
        frames_per_buffer=1,
        format=pyaudio.paInt32,
    )

    while True:
        frame, signal_info = source.get_signal()
        processeed = delay.process(frame, signal_info)
        player.write(frame.tobytes(), 1)

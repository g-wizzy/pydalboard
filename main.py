from pathlib import Path
import sys

import numpy as np
import pyaudio

from pydalboard.pipeline import Pipeline
from pydalboard.signal import Wav
from pydalboard.modules import Delay, DelayParameters, Drive, DriveParameters, Filter, FilterParameters, PitchShifting, PitchShiftingParameters
from pydalboard.signal.base import SignalInfo
from pydalboard.signal.oscillators import Oscillator, Waveform

def main():
    try:
        if len(sys.argv) != 3:
            raise ValueError("Usage: main.py [-f <file_path> | -w <waveform_name>]")

        # Retrieve arguments
        flag = sys.argv[1]
        value = sys.argv[2]

        if flag == '-f':
            # Use WAV file
            print(f"File path provided: {value}")
            play_file(value)
        elif flag == '-w':
            # Use Waveform
            try:
                waveform = Waveform[value.upper()]
                print(f"Waveform selected: {waveform}")
                play_waveform(waveform)
            except KeyError:
                raise ValueError(f"Invalid enum value. Allowed values are: {[e.name for e in Waveform]}")
        else:
            raise ValueError("Invalid flag. Use -f for file path or -w for waveform.")
    except Exception as e:
        print(e)
        sys.exit(1)


def play_file(file_path):
    """
    Play the audio file.
    """
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Retrieve WAV data and information
        wav_source = Wav(Path(file_path), loop=False)
        sample_rate = wav_source.signal_info.sample_rate
        sample_format = wav_source.signal_info.sample_format
        channels = 2 if wav_source.signal_info.stereo else 1

        # Define sample format
        if sample_format == 16:
            pa_format = pyaudio.paInt16
        elif sample_format == 32:
            pa_format = pyaudio.paInt32
        else:
            raise ValueError("Unsupported sample format")

        # Initialize audio player
        player = p.open(
            rate=sample_rate,
            channels=channels,
            output=True,
            frames_per_buffer=1,
            format=pa_format,
        )

        # Create the pipeline
        pipeline = Pipeline(wav_source)
        # pipeline.modules.append(Drive(DriveParameters(gain=2.0, clipping=True)))
        # pipeline.modules.append(PitchShifting(PitchShiftingParameters(pitch_factor=0.8, warp=False), sample_rate=sample_rate))
        # pipeline.modules.append(Filter(FilterParameters(cutoff=3000, resonance=1.41, filter_type='low', slope=12)))
        # pipeline.modules.append(Delay(DelayParameters(delay=300, feedback=0.3), sample_rate=sample_rate))

        # Play the audio
        while True:
            try:
                frame = pipeline.run()
                if wav_source.signal_info.sample_format in [16, 32]:
                    # Convert back from float32 to int16/32
                    frame = (frame * (2**31 - 1)).astype(np.int32)
                player.write(frame.tobytes(), 1)
            except KeyboardInterrupt:
                break
    except Exception as e:
        print(e)
        sys.exit(2)


def play_waveform(waveform):
    """
    Play the waveform.
    """
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Initialize audio player
        player = p.open(
            rate=44100,
            channels=2,
            output=True,
            frames_per_buffer=1,
            format=pyaudio.paInt32,
        )
        
        # Create the pipeline
        pipeline = Pipeline(
            Oscillator(
                waveform=waveform,
                frequency=440,
                signal_info=SignalInfo(44100, 32, True),
                cycles=100
            )
        )
        # pipeline.modules.append(Drive(DriveParameters(gain=2.0, clipping=True)))
        # pipeline.modules.append(PitchShifting(PitchShiftingParameters(pitch_factor=0.8, warp=False), sample_rate=44100))
        # pipeline.modules.append(Filter(FilterParameters(cutoff=3000, resonance=1.41, filter_type='low', slope=12)))
        # pipeline.modules.append(Delay(DelayParameters(delay=300, feedback=0.3), sample_rate=44100))

        # Play the audio
        while True:
            try:
                frame = pipeline.run()
                player.write(frame.tobytes(), 1)
            except KeyboardInterrupt:
                break
    except Exception as e:
        print(e)
        sys.exit(2)


if __name__ == "__main__":
    main()

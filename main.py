from pathlib import Path
import sys

import pyaudio

from pydalboard.pipeline import Pipeline
from pydalboard.signal import Wav
from pydalboard.modules import (
    DelayParameters,
    DistortionParameters,
    FilterParameters,
    Gain,
    GainParameters,
    Overdrive,
    FilterType,
    OverdriveParameters,
    PitchShiftingParameters,
    SaturationParameters,
)
from pydalboard.signal.base import SignalInfo
from pydalboard.signal.oscillators import Oscillator, Waveform


# For testing purposes, the modules are instanciated
distortion = DistortionParameters(drive=12.0)
delay = DelayParameters(delay=300, feedback=0.4)
filter = FilterParameters(
    cutoff=700,
    gain_db=-6,
    resonance=1.14,
    filter_type=FilterType.LOW_SHELF_PASS,
    slope=24,
)
overdrive = OverdriveParameters(drive=12.0, threshold=1.0, asymmetry=0.5)
pitch = (PitchShiftingParameters(pitch_factor=0.8, warp=False),)
saturation = SaturationParameters(drive=12.0)


def main():
    usage = "Usage: main.py [-b <buffer_size>] [-f <file_path> | -w <waveform_name>]"
    function = None
    kwargs = {}

    args = sys.argv[1:]
    while args:
        match args:
            case ["-b", buffer_size_str, *other_args]:
                try:
                    buffer_size = int(buffer_size_str)
                except ValueError:
                    print(
                        f"Invalid buffer_size. Expected integer, got {buffer_size_str}"
                    )
                    exit(1)
                if buffer_size not in {32, 64, 128, 256, 512, 1024, 2048}:
                    print(
                        f"Invalid buffer size. Expected value in {{32, 64, 128, 256, 512, 1024, 2048}}, got {buffer_size}"
                    )
                    exit(1)
                kwargs["buffer_size"] = buffer_size

                args = other_args

            case ["-f", file_path, *other_args]:
                function = play_file
                kwargs["file_path"] = file_path

                args = other_args

            case ["-w", waveform_str, *other_args]:
                try:
                    waveform = Waveform[waveform_str.upper()]
                except KeyError:
                    print(
                        f"Invalid enum value. Allowed values are: {[e.name for e in Waveform]}"
                    )
                    exit(1)

                function = play_waveform
                kwargs["waveform"] = waveform

                args = other_args

            case _:
                print(f"Unexpected argument: {args[0]}. {usage}")
                sys.exit(1)

    if function is None:
        print(f"Missing file or waveform. {usage}")
        exit(1)

    function(**kwargs)


def play_file(*, file_path: str, buffer_size: int = 64):
    """
    Play the audio file.
    """
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Retrieve WAV data and information
    wav_source = Wav(Path(file_path), buffer_size, loop=False)

    # Define sample format
    match wav_source.signal_info.sample_format:
        case 16:
            pa_format = pyaudio.paInt16
        case 32:
            pa_format = pyaudio.paInt32
        case _:
            raise ValueError("Unsupported sample format")

    # Initialize audio player
    player = p.open(
        rate=wav_source.signal_info.sample_rate,
        channels=wav_source.signal_info.channels,
        output=True,
        format=pa_format,
    )

    # Create the pipeline
    pipeline = Pipeline(wav_source)
    # pipeline.modules.append(pitch)
    pipeline.add_module(filter)
    # pipeline.add_module(saturation)
    # pipeline.add_module(overdrive)
    # pipeline.add_module(distortion)
    # pipeline.add_module(delay)

    # Play the audio
    while True:
        try:
            buffer = pipeline.run()
            player.write(buffer.tobytes())
        except KeyboardInterrupt:
            break


def play_waveform(*, waveform: Waveform, buffer_size: int = 64):
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
            format=pyaudio.paInt32,
        )

        # Create the pipeline
        pipeline = Pipeline(
            Oscillator(
                waveform=waveform,
                frequency=440,
                phase=0.0,
                signal_info=SignalInfo(44100, 32, True, buffer_size),
                cycles=100,
            )
        )

        # Play the audio
        while True:
            try:
                buffer = pipeline.run()
                player.write(buffer.tobytes())
            except KeyboardInterrupt:
                break
    except Exception as e:
        print(e)
        sys.exit(2)


if __name__ == "__main__":
    main()

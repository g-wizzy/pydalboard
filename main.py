from pathlib import Path
import sys

import numpy as np
import pyaudio

from pydalboard.pipeline import Pipeline
from pydalboard.signal import Wav
from pydalboard.modules import (
    Delay,
    DelayParameters,
    Distortion,
    DistortionParameters,
    Drive,
    DriveParameters,
    Filter,
    FilterParameters,
    Overdrive,
    OverdriveParameters,
    PitchShifting,
    PitchShiftingParameters,
    Saturation,
    SaturationParameters,
)
from pydalboard.signal.base import SignalInfo
from pydalboard.signal.oscillators import Oscillator, Waveform


def main():
    match sys.argv[1:]:
        case ["-f", file_path]:
            play_file(file_path)
        case ["-w", waveform_str]:
            try:
                waveform = Waveform[waveform_str.upper()]
                print(f"Waveform selected: {waveform}")
                play_waveform(waveform)
            except KeyError:
                print(
                    f"Invalid enum value. Allowed values are: {[e.name for e in Waveform]}"
                )
                exit(1)
        case _:
            print("Usage: main.py [-f <file_path> | -w <waveform_name>]")
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
        infos = wav_source.signal_info

        # Define sample format
        match infos.sample_format:
            case 16:
                pa_format = pyaudio.paInt16
            case 32:
                pa_format = pyaudio.paInt32
            case _:
                raise ValueError("Unsupported sample format")

        # Initialize audio player
        player = p.open(
            rate=infos.sample_rate,
            channels=2 if infos.stereo else 1,
            output=True,
            frames_per_buffer=1,
            format=pa_format,
        )

        # Create the pipeline
        pipeline = Pipeline(wav_source)
        # pipeline.modules.append(Saturation(SaturationParameters(drive=3.0)))
        # pipeline.modules.append(Overdrive(OverdriveParameters(drive=3.0)))
        # pipeline.modules.append(Distortion(DistortionParameters(drive=3.0)))
        # pipeline.modules.append(PitchShifting(PitchShiftingParameters(pitch_factor=0.8, warp=False), sample_rate=sample_rate))
        # pipeline.modules.append(Filter(FilterParameters(cutoff=3000, resonance=1.41, filter_type='low', slope=12)))
        # pipeline.modules.append(Delay(DelayParameters(delay=300, feedback=0.3), sample_rate=sample_rate))

        # Play the audio
        while True:
            try:
                frame = pipeline.run()
                if wav_source.signal_info.sample_format in [16, 32]:
                    # Convert back from float32 to int16/32
                    # TODO: I think this belongs in the Pipeline class
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
                phase=0.0,
                signal_info=SignalInfo(44100, 32, True),
                cycles=100,
            )
        )
        # pipeline.modules.append(Saturation(SaturationParameters(drive=3.0)))
        # pipeline.modules.append(Overdrive(OverdriveParameters(drive=3.0)))
        # pipeline.modules.append(Distortion(DistortionParameters(drive=3.0)))
        # pipeline.modules.append(PitchShifting(PitchShiftingParameters(pitch_factor=0.8, warp=False), sample_rate=sample_rate))
        # pipeline.modules.append(Filter(FilterParameters(cutoff=3000, resonance=1.41, filter_type='low', slope=12)))
        # pipeline.modules.append(Delay(DelayParameters(delay=300, feedback=0.3), sample_rate=sample_rate))

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

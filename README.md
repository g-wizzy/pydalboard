# Pydalboard

This is a prototype that is being developed for fun, aiming to create a GUI application that allows for composition of multiple sound effect modules, and hearing the results in real-time.

## Setup

Install `portaudio` on your computer:

```bash
# MacOS
brew install portaudio
```

Sync dependencies:

```bash
uv sync
```

Run application:

```bash
# Play waveform
uv run main.py -w <waveform>

# Play sample
uv run main.py -f /path/to/file.wav

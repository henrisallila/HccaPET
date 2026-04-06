# HccaPET

Converts MP4 video to C64 PETSCII art video, preserving the original audio.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and available in PATH (required for audio)

## Install

```
pip install -r requirements.txt
```

## Usage

```
python hccapet.py input.mp4 output.mp4
python hccapet.py input.mp4 output.mp4 --cols 80 --rows 50 --fps 10 --palette c64
```

## Options

| Option | Default | Description |
|---|---|---|
| `--cols` | `80` | Character columns |
| `--rows` | `50` | Character rows |
| `--fps` | source fps (max 25) | Output frames per second |
| `--palette` | `c64` | Color mode: `c64`, `mono`, `grayscale`, or `rgb` |

## Output

- Video is rendered as PETSCII art using the original C64 character ROM and 16-color palette
- `--palette rgb` renders with full RGB color, not restricted to the C64 palette
- `--palette grayscale` renders using C64's 5 grey shades (black to white)
- `--palette mono` renders in black and white only
- Audio is copied from the source video using ffmpeg and muxed into the output file
- If ffmpeg is not found, the output is saved without audio

## Notes

- Output resolution is `cols × 8` by `rows × 8` pixels (e.g. 80×50 → 640×400)
- Higher `--cols`/`--rows` values increase detail but slow down conversion

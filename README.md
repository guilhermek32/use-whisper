# use-whisper

Transcribe audio files (mp3, wav, m4a, mp4, ...) to text locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Uses the GPU (CUDA 12) when
available and falls back to the CPU.

## Install

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Install the dependencies (Python is downloaded too if needed):

   ```bash
   uv sync
   ```

The first transcription downloads the model (~3 GB for `large-v3`).

## Use

- Put audio files in `audios/` and double-click `transcribe.bat`, or drag files onto it.
- Or from a terminal: `uv run transcribe.py a.mp3 b.m4a [--language en] [--model large-v3-turbo]`

Each audio produces a folder `transcriptions/<audio>/`:

- `transcript.txt`: the text in paragraphs, each with its start time
- `timestamps.txt`: one line per Whisper segment

Defaults: `large-v3` (most accurate; use `--model large-v3-turbo` for speed) and Portuguese
(`--language auto` to detect).

## Better accuracy: context

Copy `context.example.txt` to `context.txt` (applies to every audio) or to
`audios/<audio name>.txt` (applies to one audio), and edit it:

```
hotwords: Acme, Kubernetes, OKR, roadmap
prompt: Reunião semanal do time de produto. Falamos sobre o roadmap, os prazos e as prioridades.
```

- `hotwords`: names, acronyms and jargon. They are sent with every 30 s window. Keep the
  list short, because the model can start writing these words where they weren't said.
- `prompt`: one or two well-punctuated sentences in the style of the audio. It only affects
  the start of the transcription.

`--hotwords` and `--prompt` on the command line take precedence over the files.

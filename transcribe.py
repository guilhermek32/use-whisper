"""Transcribe audio files (mp3, wav, m4a, ...) to text with Whisper, locally.

Folders (created automatically next to this script):
    audios/                     put your audio files here
    transcriptions/<audio>/     results: transcript.txt and timestamps.txt

Usage:
    transcribe.bat              -> transcribes every new file in audios/, then lets you
                                   drag/paste more files (they are copied into audios/)
    transcribe.bat a.mp3 b.mp3 [--language pt]
    (or drag files onto transcribe.bat, or right-click -> Send to -> Transcribe)
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# Make the pip-installed CUDA DLLs (cuBLAS / cuDNN) visible on Windows.
_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
for bin_dir in _nvidia.glob("*/bin"):
    os.add_dll_directory(str(bin_dir))
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parent
AUDIOS = ROOT / "audios"
RESULTS = ROOT / "transcriptions"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac", ".aac", ".wma",
              ".webm", ".mp4", ".mkv", ".mov"}

_model: WhisperModel | None = None


def get_model(name: str) -> WhisperModel:
    """Load the model once and reuse it for every file."""
    global _model
    if _model is None:
        print(f"[loading model {name}...]", file=sys.stderr)
        try:
            _model = WhisperModel(name, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"[GPU unavailable ({e}); using CPU]", file=sys.stderr)
            _model = WhisperModel(name, device="cpu", compute_type="int8")
    return _model


def split_paths(text: str) -> list[str]:
    """Split pasted/dragged input into paths: "quoted paths" or unquoted tokens."""
    text = text.replace("﻿", "").strip()
    if Path(text.strip('"')).exists():  # a single path, even if it has unquoted spaces
        return [text.strip('"')]
    return [q or u for q, u in re.findall(r'"([^"]+)"|(\S+)', text)]


def result_dir(audio: Path) -> Path:
    return RESULTS / audio.stem


def is_done(audio: Path) -> bool:
    return (result_dir(audio) / "transcript.txt").is_file()


def import_audio(src: Path) -> Path | None:
    """Copy a file from anywhere into audios/ (no-op if it's already there)."""
    if not src.is_file():
        print(f"File not found: {src}", file=sys.stderr)
        return None
    if src.resolve().parent == AUDIOS:
        return src
    dst = AUDIOS / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def transcribe(audio: Path, args: argparse.Namespace) -> None:
    print(f"\n=== {audio.name} ===", file=sys.stderr)
    segments, info = get_model(args.model).transcribe(
        str(audio), language=args.language, vad_filter=True,
        initial_prompt=args.prompt,
        # avoids repetition loops / hallucinated text on long audio
        condition_on_previous_text=False)
    print(f"[language: {info.language}, duration: {info.duration:.0f}s]", file=sys.stderr)

    lines, stamped = [], []
    total = round(info.duration, 1)
    with tqdm(total=total, unit="s", desc="progress", file=sys.stderr,
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s audio "
                         "[{elapsed}<{remaining}]") as bar:
        for seg in segments:
            text = seg.text.strip()
            m, s = divmod(int(seg.start), 60)
            stamp = f"[{m // 60:02d}:{m % 60:02d}:{s:02d}]"
            bar.write(f"{stamp} {text}", file=sys.stdout)
            lines.append(text)
            stamped.append(f"{stamp} {text}")
            bar.update(min(round(seg.end, 1), total) - bar.n)
        bar.update(total - bar.n)

    out = result_dir(audio)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "timestamps.txt").write_text("\n".join(stamped), encoding="utf-8")
    print(f"[saved to {out}]", file=sys.stderr)


def run(files: list[Path], args: argparse.Namespace) -> None:
    for i, audio in enumerate(files, 1):
        if len(files) > 1:
            print(f"\n[file {i}/{len(files)}]", file=sys.stderr)
        try:
            transcribe(audio, args)
        except Exception as e:
            print(f"[failed: {audio.name}: {e}]", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="*")
    p.add_argument("--model", default="large-v3-turbo",
                   help="tiny, base, small, medium, large-v3, large-v3-turbo")
    p.add_argument("--language", default=None, help="e.g. pt, en (auto-detect if omitted)")
    p.add_argument("--prompt", default=None,
                   help="vocabulary hint, e.g. names and acronyms spoken in the audio")
    p.add_argument("--redo", action="store_true", help="re-transcribe files already done")
    args = p.parse_args()

    AUDIOS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    if args.audio:
        files = [f for a in args.audio if (f := import_audio(Path(a.strip('"'))))]
        run(files, args)
        return

    pending = sorted(f for f in AUDIOS.iterdir()
                     if f.suffix.lower() in AUDIO_EXTS and (args.redo or not is_done(f)))
    if pending:
        print(f"[{len(pending)} new file(s) in {AUDIOS}]", file=sys.stderr)
        run(pending, args)
    else:
        print(f"[no new audio in {AUDIOS}]", file=sys.stderr)

    while True:
        try:
            text = input("\nDrag or paste audio file(s) here, then press Enter (empty to quit): ")
        except (EOFError, KeyboardInterrupt):
            break
        if not text.strip():
            break
        run([f for a in split_paths(text) if (f := import_audio(Path(a)))], args)


if __name__ == "__main__":
    main()

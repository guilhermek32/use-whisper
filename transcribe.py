"""Transcribe audio files (mp3, wav, m4a, ...) to text with Whisper, locally.

Folders (created automatically next to this script):
    audios/                     put your audio files here
    transcriptions/<audio>/     results: transcript.txt and timestamps.txt

Usage:
    transcribe.bat              -> transcribes every new file in audios/, then lets you
                                   drag/paste more files (they are copied into audios/)
    transcribe.bat a.mp3 b.mp3 [--language en] [--hotwords "Acme, OKR"] [--prompt "..."]
    (or drag files onto transcribe.bat, or right-click -> Send to -> Transcribe)

Context (vocabulary + style hints), from lowest to highest priority:
    context.txt                 next to this script: defaults for every audio
    audios/<audio name>.txt     for one audio, e.g. audios/meeting.m4a.txt or meeting.txt
    --hotwords / --prompt       on the command line
Context file format (either line is optional):
    hotwords: Acme, Kubernetes, OKR, roadmap
    prompt: Reunião semanal do time de produto. Falamos sobre o roadmap, os prazos e as prioridades.
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# Make the pip-installed CUDA DLLs (cuBLAS / cuDNN) visible on Windows.
if sys.platform == "win32":
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


def read_context(path: Path) -> dict[str, str]:
    """Parse 'hotwords: ...' / 'prompt: ...' lines from a context file."""
    ctx = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip().lower() in ("hotwords", "prompt") and value.strip():
                ctx[key.strip().lower()] = value.strip()
    return ctx


def context_for(audio: Path, args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Merge context.txt, audios/<audio>.txt and command-line flags (later wins)."""
    ctx = read_context(ROOT / "context.txt")
    for f in (AUDIOS / f"{audio.stem}.txt", AUDIOS / f"{audio.name}.txt"):
        ctx.update(read_context(f))
    if args.hotwords:
        ctx["hotwords"] = args.hotwords
    if args.prompt:
        ctx["prompt"] = args.prompt
    return ctx.get("hotwords"), ctx.get("prompt")


def stamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"[{m // 60:02d}:{m % 60:02d}:{s:02d}]"


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


# A new paragraph starts after a sentence end followed by a pause this long,
# or after any sentence end once the paragraph is this many characters.
# Whisper sometimes drops the final period, so a longer pause or a much longer
# paragraph also breaks, punctuation or not.
PARAGRAPH_PAUSE = 1.5
PARAGRAPH_MAX_CHARS = 600
PARAGRAPH_HARD_PAUSE = 4.0
PARAGRAPH_HARD_MAX_CHARS = 1200


def transcribe(audio: Path, args: argparse.Namespace) -> None:
    print(f"\n=== {audio.name} ===", file=sys.stderr)
    hotwords, prompt = context_for(audio, args)
    if hotwords or prompt:
        print(f"[hotwords: {hotwords or '-'} | prompt: {prompt or '-'}]", file=sys.stderr)
    segments, info = get_model(args.model).transcribe(
        str(audio), language=None if args.language == "auto" else args.language,
        initial_prompt=prompt,  # style: only reaches the first 30s window
        hotwords=hotwords,      # vocabulary: sent with every window
        # avoids repetition loops / hallucinated text on long audio
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=700, speech_pad_ms=300),
        beam_size=5,
        word_timestamps=True,
        # drops made-up text during long silences (needs word_timestamps)
        hallucination_silence_threshold=2.0)
    print(f"[language: {info.language}, duration: {info.duration:.0f}s]", file=sys.stderr)

    paragraphs, stamped = [], []
    cur, cur_start, last_end, last_norm = [], 0.0, 0.0, ""
    total = round(info.duration, 1)
    with tqdm(total=total, unit="s", desc="progress", file=sys.stderr,
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s audio "
                         "[{elapsed}<{remaining}]") as bar:
        for seg in segments:
            bar.update(min(round(seg.end, 1), total) - bar.n)
            text = seg.text.strip()
            norm = normalize(text)
            if not norm or norm == last_norm:  # empty or repeated line
                continue
            last_norm = norm
            bar.write(f"{stamp(seg.start)} {text}", file=sys.stdout)
            stamped.append(f"{stamp(seg.start)} {text}")

            sentence_end = bool(cur) and cur[-1].endswith((".", "?", "!", "…"))
            pause = seg.start - last_end
            size = sum(len(t) + 1 for t in cur)
            soft = sentence_end and (pause > PARAGRAPH_PAUSE or size > PARAGRAPH_MAX_CHARS)
            hard = bool(cur) and (pause > PARAGRAPH_HARD_PAUSE or size > PARAGRAPH_HARD_MAX_CHARS)
            if soft or hard:
                paragraphs.append(f"{stamp(cur_start)} {' '.join(cur)}")
                cur = []
            if not cur:
                cur_start = seg.start
            cur.append(text)
            last_end = seg.end
        bar.update(total - bar.n)
    if cur:
        paragraphs.append(f"{stamp(cur_start)} {' '.join(cur)}")

    out = result_dir(audio)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.txt").write_text("\n\n".join(paragraphs), encoding="utf-8")
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
    p.add_argument("--model", default="large-v3",
                   help="large-v3 (most accurate), large-v3-turbo (much faster), "
                        "medium, small, base, tiny")
    p.add_argument("--language", default="pt", help="e.g. pt, en, or auto (default: pt)")
    p.add_argument("--hotwords", default=None,
                   help="names, acronyms and jargon spoken in the audio, e.g. \"Acme, OKR\"")
    p.add_argument("--prompt", default=None,
                   help="1-2 well-punctuated sentences in the style of the audio")
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

#!/usr/bin/env python3
"""Local speech-to-text via whisper.cpp. No network, no API key."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from transcribe import parse_vtt  # noqa: E402

DEFAULT_MODEL = "large-v3-turbo"
MODEL_DIR = Path.home() / ".cache" / "whisper-cpp"
MODEL_URL_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
BINARY = "whisper-cli"
MODEL_MIN_SIZE_BYTES = 1_000_000


def model_path(model: str = DEFAULT_MODEL) -> Path:
    return MODEL_DIR / f"ggml-{model}.bin"


def model_url(model: str = DEFAULT_MODEL) -> str:
    return f"{MODEL_URL_BASE}/ggml-{model}.bin"


def model_present(path: Path) -> bool:
    """True when ``path`` exists and is at least ``MODEL_MIN_SIZE_BYTES``."""
    return path.exists() and path.stat().st_size >= MODEL_MIN_SIZE_BYTES


def find_binary() -> str | None:
    return shutil.which(BINARY)


def build_command(
    binary: str,
    model_file: Path,
    audio_path: Path,
    out_prefix: Path,
    language: str = "en",
    threads: int | None = None,
) -> list[str]:
    cmd = [
        binary,
        "-m", str(model_file),
        "-f", str(audio_path),
        "-l", language,
        "-ovtt",
        "-of", str(out_prefix),
    ]
    if threads is not None:
        cmd += ["-t", str(threads)]
    return cmd


def extract_audio(video_path: str, out_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(Path(video_path).resolve()),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path


def transcribe_video(
    video_path: str,
    audio_out: Path,
    model: str = DEFAULT_MODEL,
    language: str = "en",
    threads: int | None = None,
) -> tuple[list[dict], str]:
    binary = find_binary()
    if binary is None:
        raise SystemExit(
            "whisper-cli is not installed. Install with: brew install whisper-cpp"
        )

    model_file = model_path(model)
    if not model_present(model_file):
        raise SystemExit(
            f"Whisper model not found at {model_file}. "
            f"Download it with: curl -fL --create-dirs -o {model_file} {model_url(model)}"
        )

    print("[watch] extracting audio for local whisper.cpp…", file=sys.stderr)
    audio_path = extract_audio(video_path, audio_out)

    out_prefix = audio_out.parent / "transcript"
    print(f"[watch] transcribing locally with {model}…", file=sys.stderr)
    result = subprocess.run(
        build_command(binary, model_file, audio_path, out_prefix, language, threads),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"whisper-cli failed: {result.stderr.strip()}")

    vtt_path = out_prefix.with_suffix(".vtt")
    if not vtt_path.exists():
        raise SystemExit(f"whisper-cli produced no VTT at {vtt_path}")

    segments = parse_vtt(str(vtt_path))
    if not segments:
        raise SystemExit("whisper.cpp returned no transcript segments")

    engine = f"whisper.cpp ({model})"
    print(f"[watch] transcribed {len(segments)} segments via {engine}", file=sys.stderr)
    return segments, engine

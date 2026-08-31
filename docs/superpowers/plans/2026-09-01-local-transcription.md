# Local Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hosted Groq/OpenAI Whisper fallback with local whisper.cpp, so `/watch` transcribes with no API key, no network, and no third-party billing.

**Architecture:** `transcribe.py` parses WebVTT and is backend-agnostic. whisper.cpp emits VTT via `-ovtt`, so a new `localstt.py` replaces `whisper.py` and feeds the existing parser unchanged. All upload-cap machinery (chunking, multipart, retry/backoff) is deleted rather than ported. Two unrelated defects found on the same codebase are fixed in passing, and frame selection gains a uniform top-up.

**Tech Stack:** Python 3.9+ (stdlib only, no new deps), whisper.cpp 1.9.2 via Homebrew (`whisper-cli`), ffmpeg, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-local-transcription-design.md`

## Global Constraints

- **No new Python dependencies.** The scripts are stdlib-only by design; keep them so.
- **No code comments explaining rationale.** Repo convention: intent goes in identifiers, quirks become named tests, rationale goes in commit messages. Existing comments in untouched code stay; do not add new explanatory ones.
- **`transcribe.py` must not be modified.** It is the proven seam. Changing it invalidates the design.
- **Frame dict shape is fixed:** `{"index": int, "path": str, "timestamp_seconds": float, "reason": str}`.
- **Transcript segment shape is fixed:** `{"start": float, "end": float, "text": str}`.
- **Binary is `whisper-cli`**, model flag `-m`, VTT output `-ovtt` with `-of <prefix>` (no extension — whisper.cpp appends `.vtt`).
- **Model default `large-v3-turbo`**, cached at `~/.cache/whisper-cpp/ggml-large-v3-turbo.bin` (1,624,555,275 bytes).
- **Never download a model without explicit consent.** It is 1.5 GB.
- Run tests with `python3 -m pytest tests/ -v` from the repo root.

---

### Task 1: Fix the ffmpeg `-vsync` breakage

`-vsync` was removed in ffmpeg 7.x. The host runs 9.0.1, so **every** scene and keyframe extraction currently aborts. This lands first because no other frame work is testable until it does.

**Files:**
- Modify: `skills/watch/scripts/frames.py:256`, `skills/watch/scripts/frames.py:615`
- Test: `tests/test_frames.py`

**Interfaces:**
- Consumes: nothing.
- Produces: working frame extraction. Every later task that touches frames depends on this.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frames.py`:

```python
import subprocess
from pathlib import Path

import frames


def _ffmpeg_major() -> int:
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
    return int(out.split("ffmpeg version ")[1].split(".")[0].lstrip("n"))


def test_frames_uses_fps_mode_not_removed_vsync():
    source = Path(frames.__file__).read_text(encoding="utf-8")
    assert '"-vsync"' not in source
    assert source.count('"-fps_mode"') == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_frames.py::test_frames_uses_fps_mode_not_removed_vsync -v`
Expected: FAIL — `assert '"-vsync"' not in source` fails, because both call sites still use it.

- [ ] **Step 3: Write minimal implementation**

At `frames.py:256` and `frames.py:615`, change:

```python
        "-vsync", "vfr",
```

to:

```python
        "-fps_mode", "vfr",
```

- [ ] **Step 4: Run the full frame suite to verify it passes**

Run: `python3 -m pytest tests/test_frames.py -v`
Expected: PASS — all 7 tests (6 existing + 1 new). The existing tests synthesize real clips via ffmpeg, so they exercise the fixed flag end to end.

- [ ] **Step 5: Commit**

```bash
git add skills/watch/scripts/frames.py tests/test_frames.py
git commit -m "fix: replace ffmpeg -vsync with -fps_mode

ffmpeg removed -vsync in 7.x. On ffmpeg 9 every scene-aware and keyframe
extraction aborted with Unrecognized option 'vsync', making the skill
non-functional. -fps_mode has been supported since ffmpeg 5.1."
```

---

### Task 2: Fix the config parser disagreeing with itself

`check-setup.sh` reads keys with `awk ... exit` (first match wins) while `config.py` builds a dict (last match wins). A file with a blank key above a populated one makes the hook and the script disagree.

**Files:**
- Modify: `hooks/scripts/check-setup.sh`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: shell and Python agreeing on duplicate-key resolution.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
import subprocess
from pathlib import Path

import config

HOOK = Path(__file__).resolve().parent.parent / "hooks" / "scripts" / "check-setup.sh"


def test_env_parser_takes_last_duplicate_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("WATCH_DETAIL=\nWATCH_DETAIL=efficient\n", encoding="utf-8")

    assert config.read_env_file(env)["WATCH_DETAIL"] == "efficient"

    script = (
        f'read_key() {{ awk -F= -v k="$1" \'/^[[:space:]]*#/ {{next}} $1 == k '
        f"{{sub(/^[[:space:]]*/, \"\", $2); print $2}}' \"{env}\" | tail -1; }}\n"
        "read_key WATCH_DETAIL\n"
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.stdout.strip() == "efficient"


def test_hook_reads_last_duplicate_key():
    source = HOOK.read_text(encoding="utf-8")
    assert "exit }" not in source and "; exit" not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_config.py -v -k duplicate`
Expected: `test_hook_reads_last_duplicate_key` FAILS — the hook's awk still contains `; exit`.

- [ ] **Step 3: Write minimal implementation**

In `hooks/scripts/check-setup.sh`, replace the `read_key` awk body. Change the action block so it does not `exit`, and pipe through `tail -1`:

```bash
  if [[ -f "$CONFIG_FILE" ]]; then
    awk -F= -v k="$name" '
      /^[[:space:]]*#/ { next }
      $1 == k {
        sub(/^[[:space:]]*/, "", $2); sub(/[[:space:]]*$/, "", $2);
        gsub(/^["'\'']|["'\'']$/, "", $2);
        print $2
      }
    ' "$CONFIG_FILE" | tail -1
  fi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: PASS — 7 tests (5 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add hooks/scripts/check-setup.sh tests/test_config.py
git commit -m "fix: make the hook's env parser last-wins like config.py

check-setup.sh took the first occurrence of a duplicate key while
config.py takes the last, so a blank key above a populated one made the
hook report missing configuration the script could see."
```

---

### Task 3: Build the whisper.cpp adapter

New module producing the same `(segments, engine_label)` contract the old `transcribe_video()` produced, so Task 4's call-site change is small.

**Files:**
- Create: `skills/watch/scripts/localstt.py`
- Create: `tests/test_localstt.py`

**Interfaces:**
- Consumes: `transcribe.parse_vtt(path) -> list[dict]` (unchanged).
- Produces, for Task 4 and Task 5:
  - `DEFAULT_MODEL: str = "large-v3-turbo"`
  - `MODEL_DIR: Path`
  - `model_path(model: str = DEFAULT_MODEL) -> Path`
  - `model_url(model: str = DEFAULT_MODEL) -> str`
  - `find_binary() -> str | None`
  - `build_command(binary: str, model_file: Path, audio_path: Path, out_prefix: Path, language: str = "en", threads: int | None = None) -> list[str]`
  - `extract_audio(video_path: str, out_path: Path) -> Path`
  - `audio_duration(audio_path: Path) -> float`
  - `transcribe_video(video_path: str, audio_out: Path, model: str = DEFAULT_MODEL, language: str = "en", threads: int | None = None) -> tuple[list[dict], str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_localstt.py`:

```python
from pathlib import Path

import pytest

import localstt


def test_build_command_emits_vtt_with_prefix(tmp_path):
    cmd = localstt.build_command(
        "/opt/homebrew/bin/whisper-cli",
        tmp_path / "ggml-large-v3-turbo.bin",
        tmp_path / "audio.wav",
        tmp_path / "transcript",
    )
    assert cmd[0] == "/opt/homebrew/bin/whisper-cli"
    assert "-ovtt" in cmd
    assert cmd[cmd.index("-of") + 1] == str(tmp_path / "transcript")
    assert cmd[cmd.index("-m") + 1] == str(tmp_path / "ggml-large-v3-turbo.bin")
    assert cmd[cmd.index("-l") + 1] == "en"
    assert "-t" not in cmd


def test_build_command_includes_threads_when_given(tmp_path):
    cmd = localstt.build_command(
        "whisper-cli",
        tmp_path / "m.bin",
        tmp_path / "a.wav",
        tmp_path / "out",
        threads=8,
    )
    assert cmd[cmd.index("-t") + 1] == "8"


def test_model_path_and_url_agree_on_name():
    assert localstt.model_path("base").name == "ggml-base.bin"
    assert localstt.model_url("base").endswith("ggml-base.bin")


def test_transcribe_raises_when_binary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(localstt, "find_binary", lambda: None)
    with pytest.raises(SystemExit) as exc:
        localstt.transcribe_video("video.mp4", tmp_path / "audio.wav")
    assert "brew install whisper-cpp" in str(exc.value)


def test_transcribe_raises_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(localstt, "find_binary", lambda: "whisper-cli")
    monkeypatch.setattr(localstt, "model_path", lambda model=None: tmp_path / "absent.bin")
    with pytest.raises(SystemExit) as exc:
        localstt.transcribe_video("video.mp4", tmp_path / "audio.wav")
    assert "absent.bin" in str(exc.value)


def test_extract_audio_produces_16k_mono_wav(tmp_path):
    from conftest import build_cut_clip

    clip = tmp_path / "clip.mp4"
    build_cut_clip(clip, n=4, seg=0.4, audio=True)

    out = localstt.extract_audio(str(clip), tmp_path / "audio.wav")
    assert out.exists() and out.stat().st_size > 0

    import json
    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(out)],
        capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1
    assert stream["codec_name"] == "pcm_s16le"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_localstt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'localstt'`.

- [ ] **Step 3: Check whether the audio fixture supports an audio track**

Run: `grep -n "def build_cut_clip" -A 20 tests/conftest.py`

If `build_cut_clip` has no `audio` parameter, add one before continuing — the fixture must be able to emit a clip with sound, or `test_extract_audio_produces_16k_mono_wav` cannot run:

```python
def build_cut_clip(path: Path, n: int = 14, seg: float = 0.4, audio: bool = False) -> None:
    ...
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={n * seg}",
                "-c:a", "aac", "-shortest"]
```

Place the new args before the output path in the existing ffmpeg invocation.

- [ ] **Step 4: Write the implementation**

Create `skills/watch/scripts/localstt.py`:

```python
#!/usr/bin/env python3
"""Local speech-to-text via whisper.cpp. No network, no API key."""
from __future__ import annotations

import json
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


def model_path(model: str = DEFAULT_MODEL) -> Path:
    return MODEL_DIR / f"ggml-{model}.bin"


def model_url(model: str = DEFAULT_MODEL) -> str:
    return f"{MODEL_URL_BASE}/ggml-{model}.bin"


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


def audio_duration(audio_path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is not installed. Install with: brew install ffmpeg")

    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
         str(audio_path.resolve())],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")
    fmt = json.loads(result.stdout or "{}").get("format", {})
    return float(fmt.get("duration") or 0.0)


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
    if not model_file.exists():
        raise SystemExit(
            f"Whisper model not found at {model_file}. "
            f"Download it with: curl -L --create-dirs -o {model_file} {model_url(model)}"
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_localstt.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 6: Verify against the real binary end to end**

Run:

```bash
python3 -c "
import sys; sys.path.insert(0, 'skills/watch/scripts')
import localstt, tempfile, pathlib
work = pathlib.Path(tempfile.mkdtemp())
segs, engine = localstt.transcribe_video(
    '/Users/karl.scally/github/tradingview-liquidity-hunter-strategy/videos/Liquidity Grabs Part 1.mp4',
    work / 'audio.wav')
print(engine, len(segs), segs[0])
"
```

Expected: `whisper.cpp (large-v3-turbo) 147 {'start': 0.0, ...}` in roughly 45 seconds.

- [ ] **Step 7: Commit**

```bash
git add skills/watch/scripts/localstt.py tests/test_localstt.py tests/conftest.py
git commit -m "feat: add local whisper.cpp transcription adapter

Emits VTT via -ovtt straight into the existing parse_vtt(), so the
transcript contract is unchanged. No key, no upload, no network."
```

---

### Task 4: Wire the adapter in and delete the hosted client

**Files:**
- Modify: `skills/watch/scripts/watch.py:22` (import), `skills/watch/scripts/watch.py:52-62` (flags), `skills/watch/scripts/watch.py:239-262` (call site), `skills/watch/scripts/watch.py:381` (hint text)
- Delete: `skills/watch/scripts/whisper.py`, `tests/test_whisper.py`
- Test: `tests/test_watch.py`

**Interfaces:**
- Consumes: everything in Task 3's Produces block.
- Produces: `--model` and `--language` CLI flags; `--whisper` and `--no-whisper` semantics replaced.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_watch.py`:

```python
from pathlib import Path

import watch

SCRIPTS = Path(watch.__file__).resolve().parent


def test_no_hosted_whisper_client_remains():
    assert not (SCRIPTS / "whisper.py").exists()
    source = (SCRIPTS / "watch.py").read_text(encoding="utf-8")
    for banned in ("GROQ_API_KEY", "OPENAI_API_KEY", "load_api_key", "api.groq.com"):
        assert banned not in source


def test_cli_exposes_model_and_language_flags():
    parser = watch.build_parser()
    args = parser.parse_args(["video.mp4", "--model", "base", "--language", "fr"])
    assert args.model == "base"
    assert args.language == "fr"


def test_cli_defaults_to_turbo_and_english():
    args = watch.build_parser().parse_args(["video.mp4"])
    assert args.model == "large-v3-turbo"
    assert args.language == "en"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_watch.py -v -k "hosted or flags or turbo"`
Expected: FAIL — `whisper.py` still exists; `build_parser` may not exist as a separate function.

- [ ] **Step 3: Extract the parser if it is inline**

Run: `grep -n "argparse.ArgumentParser\|^def main\|^def build_parser" skills/watch/scripts/watch.py`

If the parser is built inside `main()`, extract it so it is testable without running a transcription:

```python
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(...)
    # ... all existing add_argument calls ...
    return ap
```

and have `main()` call `args = build_parser().parse_args()`.

- [ ] **Step 4: Replace the flags**

Delete the `--whisper` and `--no-whisper` argument definitions at `watch.py:52-62` and add:

```python
    ap.add_argument(
        "--model",
        type=str,
        default="large-v3-turbo",
        help="whisper.cpp model name (default: large-v3-turbo)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default="en",
        help="Spoken language, or 'auto' to detect (default: en)",
    )
    ap.add_argument(
        "--no-transcript",
        action="store_true",
        help="Skip transcription entirely; return frames only",
    )
```

- [ ] **Step 5: Replace the import and the call site**

At `watch.py:22`, replace:

```python
from whisper import load_api_key, transcribe_video  # noqa: E402
```

with:

```python
from localstt import transcribe_video  # noqa: E402
```

At `watch.py:239-262`, replace the whole block with:

```python
    if not transcript_segments and not args.no_transcript and video_path and meta.get("has_audio"):
        try:
            all_segments, used_engine = transcribe_video(
                video_path,
                work / "audio.wav",
                model=args.model,
                language=args.language,
            )
            transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
            transcript_text = format_transcript(transcript_segments)
            transcript_source = used_engine
        except SystemExit as exc:
            print(f"[watch] local transcription failed: {exc}", file=sys.stderr)
```

- [ ] **Step 6: Update the no-transcript hint**

At `watch.py:381`, replace the parenthetical about API keys with:

```python
            "(whisper-cli or its model is missing, or `--no-transcript` was used). "
```

- [ ] **Step 7: Delete the hosted client and its tests**

```bash
git rm skills/watch/scripts/whisper.py tests/test_whisper.py
```

- [ ] **Step 8: Run the full suite**

Run: `python3 -m pytest tests/ -v`
Expected: PASS. Count should be 55 previously-passing + 1 (Task 1) + 2 (Task 2) + 6 (Task 3) + 3 (Task 4) = 67, with the 16 hosted-API tests gone.

- [ ] **Step 9: Commit**

```bash
git add -A skills/watch/scripts/ tests/
git commit -m "feat!: transcribe locally, delete the hosted Whisper client

--whisper/--no-whisper become --model/--language/--no-transcript.

Removes whisper.py entirely: key loading, the 24MB upload cap, chunk
planning, multipart encoding and 429 backoff all existed to serve an
upload that no longer happens."
```

---

### Task 5: Rewrite setup for whisper.cpp

Setup no longer collects secrets. It checks two binaries and one model file, and offers the download.

**Files:**
- Modify: `skills/watch/scripts/setup.py`
- Modify: `hooks/scripts/check-setup.sh`
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: `localstt.model_path`, `localstt.model_url`, `localstt.find_binary`, `localstt.DEFAULT_MODEL`.
- Produces: `--json` status object with keys `status`, `can_proceed`, `first_run`, `missing_binaries`, `model_present`, `model_path`, `config_file`, `watch_detail`, `platform`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_setup.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"


def test_setup_json_reports_model_state():
    result = subprocess.run(
        [sys.executable, str(SETUP), "--json"], capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    assert "model_present" in data
    assert "model_path" in data
    assert data["status"] in {"ready", "needs_install", "needs_model", "needs_install_and_model"}


def test_setup_no_longer_mentions_api_keys():
    source = SETUP.read_text(encoding="utf-8")
    for banned in ("GROQ_API_KEY", "OPENAI_API_KEY", "console.groq.com", "platform.openai.com"):
        assert banned not in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_setup.py -v -k "model_state or api_keys"`
Expected: FAIL — `model_present` missing from the JSON; `GROQ_API_KEY` still present in the source.

- [ ] **Step 3: Rewrite the status logic**

Replace the key-detection block in `setup.py` with:

```python
def status() -> dict:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if localstt.find_binary() is None:
        missing.append("whisper-cli")

    model_file = localstt.model_path(localstt.DEFAULT_MODEL)
    model_present = model_file.exists()
    cfg = get_config()

    if missing and not model_present:
        state = "needs_install_and_model"
    elif missing:
        state = "needs_install"
    elif not model_present:
        state = "needs_model"
    else:
        state = "ready"

    return {
        "status": state,
        "can_proceed": not missing and model_present,
        "first_run": not CONFIG_FILE.exists(),
        "missing_binaries": missing,
        "model_present": model_present,
        "model_path": str(model_file),
        "config_file": str(CONFIG_FILE),
        "watch_detail": cfg["detail"],
        "platform": platform.system(),
    }
```

- [ ] **Step 4: Replace the key wizard with a model download offer**

Replace the API-key prompt section with:

```python
def offer_model_download(model: str = localstt.DEFAULT_MODEL) -> bool:
    dest = localstt.model_path(model)
    url = localstt.model_url(model)
    print(f"[setup] model {dest.name} is not present (~1.5 GB).")
    print(f"[setup] download it with:\n  curl -L --create-dirs -o {dest} {url}")
    return dest.exists()
```

Print the command rather than running it — the model is 1.5 GB and must not be fetched without the user choosing to.

- [ ] **Step 5: Strip the key checks from the hook**

In `hooks/scripts/check-setup.sh`, delete the `HAS_GROQ` / `HAS_OPENAI` lookups and the permissions warning about the secrets file, and replace the readiness message chain with:

```bash
HAS_WHISPER=""
command -v whisper-cli >/dev/null 2>&1 && HAS_WHISPER="yes"
MODEL="$HOME/.cache/whisper-cpp/ggml-large-v3-turbo.bin"

if [[ -z "$HAS_FFMPEG" || -z "$HAS_YTDLP" || -z "$HAS_WHISPER" ]]; then
  echo "/watch: needs ffmpeg + yt-dlp + whisper-cpp. Run \`brew install ffmpeg yt-dlp whisper-cpp\`."
elif [[ ! -f "$MODEL" ]]; then
  echo "/watch: model missing. Run \`python3 \$CLAUDE_PLUGIN_ROOT/skills/watch/scripts/setup.py\` for the download command."
fi
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_setup.py tests/test_config.py -v`
Expected: PASS — 6 setup tests + 7 config tests.

- [ ] **Step 7: Commit**

```bash
git add skills/watch/scripts/setup.py hooks/scripts/check-setup.sh tests/test_setup.py
git commit -m "feat: replace the API-key wizard with a whisper.cpp preflight

Setup now checks binaries and the model file instead of collecting
secrets. The model download is printed, never run unprompted: it is
1.5 GB and that is the user's call."
```

---

### Task 6: Top up scene selection with uniform frames

Scene detection alone gave 12 frames against a budget of 100 on a chart screencast, because `SCENE_MIN_FRAMES` is a floor of 8, not a budget — anything above 8 cuts takes the scene path however far short of budget it lands.

**Files:**
- Modify: `skills/watch/scripts/frames.py:510-572` (`extract_scene_or_uniform`)
- Test: `tests/test_frames.py`

**Interfaces:**
- Consumes: `extract()`, `merge_frames()`, `dedupe_perceptual()`, `_even_sample()` — all existing in `frames.py`.
- Produces: a `"scene+uniform"` value for the `engine` key in the returned metadata dict, plus a `topup_count` key.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_frames.py`:

```python
def test_scene_selection_tops_up_toward_budget(tmp_path):
    from conftest import build_cut_clip

    clip = tmp_path / "sparse.mp4"
    build_cut_clip(clip, n=10, seg=1.0)

    selected, meta = frames.extract_scene_or_uniform(
        str(clip),
        tmp_path / "out",
        fps=2.0,
        target_frames=40,
        resolution=256,
        max_frames=40,
    )

    assert meta["engine"] == "scene+uniform"
    assert meta["topup_count"] > 0
    assert len(selected) > 10
    stamps = [f["timestamp_seconds"] for f in selected]
    assert stamps == sorted(stamps)
    assert [f["index"] for f in selected] == list(range(len(selected)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_frames.py::test_scene_selection_tops_up_toward_budget -v`
Expected: FAIL — `meta["engine"]` is `"scene"` and there is no `topup_count` key.

- [ ] **Step 3: Add the top-up constant**

Near `SCENE_MIN_FRAMES` at the top of `frames.py`:

```python
SCENE_TOPUP_FRACTION = 0.5
```

- [ ] **Step 4: Implement the top-up**

In `extract_scene_or_uniform`, replace the `if scene_count >= SCENE_MIN_FRAMES:` return block with:

```python
    if scene_count >= SCENE_MIN_FRAMES:
        deduped, n_dropped = dedupe_perceptual(scene_frames) if dedup else (scene_frames, 0)
        cap = len(deduped) if max_frames is None else max_frames
        selected = _even_sample(deduped, cap)

        budget = cap if max_frames is not None else target_frames
        topup: list[dict] = []
        if len(selected) < int(budget * SCENE_TOPUP_FRACTION):
            topup = extract(
                video_path,
                out_dir / "topup",
                fps=fps,
                resolution=resolution,
                max_frames=budget - len(selected),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            for frame in topup:
                frame["reason"] = "uniform-topup"
            selected = merge_frames(selected, topup)

        return selected, {
            "engine": "scene+uniform" if topup else "scene",
            "candidate_count": scene_count,
            "deduped_count": n_dropped,
            "topup_count": len(topup),
            "selected_count": len(selected),
            "fallback": False,
        }
```

The top-up **must** write to `out_dir / "topup"`. `extract()` deletes `frame_*.jpg` in its output directory on entry, so reusing `out_dir` would destroy the scene frames it is supplementing.

- [ ] **Step 5: Add `topup_count` to the uniform branch**

So the metadata shape is consistent whichever branch returns. In the fallback return dict, add:

```python
        "topup_count": 0,
```

- [ ] **Step 6: Run the frame suite**

Run: `python3 -m pytest tests/test_frames.py -v`
Expected: PASS — 8 tests.

- [ ] **Step 7: Verify the reporting line renders the new engine**

Run: `grep -n "engine\|candidate_count" skills/watch/scripts/watch.py`

If the frames report line interpolates `meta["engine"]`, no change is needed. If it hardcodes `"scene"`, update it to read the key.

- [ ] **Step 8: Commit**

```bash
git add skills/watch/scripts/frames.py tests/test_frames.py
git commit -m "feat: top up sparse scene selection with uniform frames

SCENE_MIN_FRAMES is a floor of 8, not a budget, so a clip with any cuts
took the scene path however far short of budget it landed. A 10-minute
chart screencast yielded 12 frames against a budget of 100, leaving long
static stretches unsampled."
```

---

### Task 7: Update the user-facing documentation

The SKILL.md is the operating manual an agent reads at runtime. Leaving it describing API keys would make every future run act on instructions that no longer match the code.

**Files:**
- Modify: `skills/watch/SKILL.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `skills/watch/.claude-plugin/plugin.json` (version)

**Interfaces:**
- Consumes: the final flag set from Task 4 and the setup states from Task 5.
- Produces: documentation matching the shipped behaviour.

- [ ] **Step 1: Rewrite the transcription section of SKILL.md**

Replace the "## Transcription" section with:

```markdown
## Transcription

The script gets a timestamped transcript in one of two ways:

1. **Native captions (free, preferred).** yt-dlp pulls manual or auto-generated
   subtitles from the source platform if available.
2. **Local whisper.cpp.** If no captions came back (or the source is a local
   file), the script extracts 16 kHz mono PCM audio and transcribes it on this
   machine with `whisper-cli`. Nothing is uploaded and no API key is involved.

Requires `brew install whisper-cpp` and one GGML model at
`~/.cache/whisper-cpp/`. Default `large-v3-turbo` (1.5 GB), which runs on the
GPU via Metal on Apple Silicon — roughly 22x realtime.

The turbo family is not trained for translation. A non-English source
transcribes in its own language; pass `--model large-v3` if you need English
output from other languages.
```

- [ ] **Step 2: Replace the flag documentation**

In SKILL.md's "Optional flags" list, delete the `--whisper` and `--no-whisper` entries and add:

```markdown
- `--model NAME` — whisper.cpp model (default `large-v3-turbo`; also `large-v3`, `medium`, `base`)
- `--language LANG` — spoken language, or `auto` to detect (default `en`)
- `--no-transcript` — skip transcription entirely, frames only
```

- [ ] **Step 3: Rewrite the Security & Permissions section**

Replace the two bullets about sending audio to Groq and OpenAI with:

```markdown
- Runs `whisper-cli` locally to transcribe extracted audio — no network request
  is made and no API key exists
```

and in the "does NOT do" list replace the API-key bullet with:

```markdown
- Does not upload the video or its audio anywhere — transcription is entirely local
```

- [ ] **Step 4: Update Step 0 of SKILL.md**

Replace the exit-code table with:

| Exit | Meaning | Action |
|------|---------|--------|
| `2` | Missing binaries (`ffmpeg` / `ffprobe` / `yt-dlp` / `whisper-cli`) | Run installer |
| `3` | Model file absent | Run setup for the download command; the user must approve 1.5 GB |
| `4` | Both missing | Run installer, then offer the model download |

- [ ] **Step 5: Add the CHANGELOG entry**

At the top of `CHANGELOG.md`:

```markdown
## 0.3.0

### Changed
- **Breaking:** transcription is now local via whisper.cpp. `--whisper` and
  `--no-whisper` are replaced by `--model`, `--language` and `--no-transcript`.
  No API key is required and no audio leaves the machine.
- Sparse scene selection is topped up with uniform frames, so screencasts no
  longer return a handful of frames against a large budget.

### Fixed
- Frame extraction failed on ffmpeg 7 and later, which removed `-vsync`.
- The session hook and `config.py` disagreed on duplicate keys in `.env`.

### Removed
- The hosted Groq/OpenAI Whisper client, its upload chunking and its retry logic.
```

- [ ] **Step 6: Bump the version**

In `skills/watch/.claude-plugin/plugin.json`, change `"version": "0.2.0"` to `"version": "0.3.0"`.

- [ ] **Step 7: Verify no stale references survive**

Run:

```bash
grep -rn "GROQ_API_KEY\|OPENAI_API_KEY\|--no-whisper\|api.groq.com\|platform.openai.com" \
  --include="*.md" --include="*.py" --include="*.sh" --include="*.json" . | grep -v docs/superpowers
```

Expected: no output. Any hit is a doc that would mislead a future run.

- [ ] **Step 8: Run the full suite one final time**

Run: `python3 -m pytest tests/ -v`
Expected: PASS, 68 tests.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "docs: document local transcription and release 0.3.0"
```

---

## Self-Review

**Spec coverage.** Engine choice and CLI surface → Task 3. Model acquisition → Tasks 3 and 5. Audio preprocessing change → Task 3. Seam preserved (`transcribe.py` untouched) → enforced by Global Constraints and Task 4's deletion test. Component table → Tasks 3, 4, 5, 6. `-vsync` defect → Task 1. Config parser defect → Task 2. Sampling strategy → Task 6. Error handling → Task 3 (binary, model, ffmpeg messages) and Task 5. Testing → each task's own steps; the 55 surviving tests act as the regression net at Task 4 Step 8. Out-of-scope items are absent from the plan, as intended.

**Type consistency.** `transcribe_video()` returns `tuple[list[dict], str]` in Task 3 and is unpacked as `all_segments, used_engine` in Task 4. `model_path()` and `model_url()` take the same `model: str` and are used with that signature in Task 5. `merge_frames(primary, pinned)` is called positionally in Task 6, matching its definition. Frame dicts carry `timestamp_seconds`, `path`, `index`, `reason` throughout.

**Known soft spots.** Task 3 Step 3 and Task 4 Step 3 are conditional — they inspect the existing code and adapt only if the fixture lacks audio support or the parser is inline. Both name the exact grep to run and the exact change to make. Task 6 Step 7 is likewise conditional on how the report line is written.

# Local transcription for /watch

Date: 2026-09-01
Status: Approved, pending implementation plan
Fork: `Scalldog/claude-video`, forked from `bradautomates/claude-video` at v0.2.0

## Problem

`/watch` cannot transcribe a video without sending its audio to a third-party
API. The Whisper fallback posts extracted audio to Groq or OpenAI and requires
the user to hold an API key for one of them. Four consequences, all unwanted:

1. Audio leaves the machine.
2. The user carries a billing relationship with a second vendor.
3. First-run setup includes an API-key wizard.
4. A key with no credit fails the run outright. Observed 2026-09-01: an
   OpenAI key returned `HTTP 429 credit_balance_exhausted`, and a 15-minute
   local video came back frames-only.

A local file has no native captions, so for local files the Whisper path is the
only transcript path. That makes the API dependency total, not incidental.

## Constraint that shapes the design

The Claude API has no audio input modality and no speech-to-text endpoint.
Documented input types are text, images, and documents/PDFs. Routing
transcription "through Claude instead" is not available at any price.

Local inference therefore beats the Anthropic-hosted alternative on every axis
that motivated the change, because the Anthropic-hosted alternative does not
exist.

## Decision

Replace the hosted Whisper client with a local Whisper process. Keep native
captions as the preferred transcript source.

### Engine

`whisper.cpp` (`ggml-org/whisper.cpp`, MIT) via Homebrew. Chosen over
`openai-whisper` for one reason: on Apple Silicon whisper.cpp runs inference
fully on the GPU via Metal, while `openai-whisper`'s `--device` defaults to
`"cuda" if available else "cpu"` and so lands on CPU.

Both are local. `openai-whisper` makes no API calls and needs no key — the
"openai" in its name identifies who published the model weights, which are the
same weights whisper.cpp runs. Privacy is not the deciding factor here; speed
is.

Verified 2026-09-01 against `brew info` and `examples/cli/cli.cpp`:

- Formula `whisper-cpp`, stable 1.9.2, binary `whisper-cli`
- `-m, --model` takes a GGML `.bin`
- `-ovtt, --output-vtt` and `-of, --output-file` produce `<name>.vtt`
- `-l, --language`, `-t, --threads`, `-pp, --print-progress`

Invoked as:

```
whisper-cli -m <model.bin> -f <audio.wav> -ovtt -of <workdir>/transcript
```

### Measured, not assumed

Benchmarked end to end on 2026-09-01 against a 15:12 (912 s) 1080p screencast,
`large-v3-turbo`, Apple Silicon:

| Stage | Wall clock |
|---|---|
| ffmpeg audio extract to 16 kHz mono WAV (28 MB) | 0.6 s |
| `whisper-cli` transcription | 41.7 s |

That is ~22x realtime. CPU sat at 31% and the run reported `ggml_metal_free`,
confirming inference ran on the GPU. No chunking, no retries, no network.

The seam was verified in the same run: the emitted VTT fed the unmodified
`transcribe.parse_vtt()` and produced 147 segments, 0 malformed, final segment
ending at 911.94 s against a 912.0 s source. Domain vocabulary transcribed
correctly. `transcribe.py` requires no changes.

### Model acquisition

Homebrew does not ship model files. They are a separate download, verified
2026-09-01 (HTTP 200):

| Model | Size |
|---|---|
| `ggml-large-v3-turbo.bin` | 1.51 GB |
| `ggml-medium.bin` | 1.43 GB |
| `ggml-base.bin` | 0.14 GB |

From `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/<model>`.
Default `large-v3-turbo`, cached under `~/.cache/whisper-cpp/`, downloaded once
on first run with explicit user consent.

### Audio preprocessing changes

`whisper-cli` accepts only 16-bit PCM WAV at 16 kHz mono. The existing
`extract_audio()` produces 64 kbps compressed audio sized to fit an upload cap:

```
ffmpeg -i <video> -vn -ar 16000 -ac 1 -c:a pcm_s16le <audio.wav>
```

Nothing is uploaded any more, so the size constraint that motivated compression
is gone along with `plan_chunks()`, `split_audio()`, and `MAX_UPLOAD_BYTES`.

## Seam

`transcribe.py` is pure WebVTT parsing and is backend-agnostic. It defines the
contract every transcript source must satisfy:

```python
{"start": float, "end": float, "text": str}
```

Both native captions (via yt-dlp) and local Whisper emit VTT. Both therefore
enter through the existing `parse_vtt()` with no adapter. `transcribe.py` is
not modified by this work.

All hosted-API coupling is quarantined inside `whisper.py`: key loading, the
24 MB upload cap, chunk planning, multipart encoding, HTTP retry and 429
backoff. None of it survives a local binary.

## Data flow

```
source ─┬─ URL ──→ download.py (yt-dlp) ──→ native captions? ──→ VTT ──┐
        └─ local file ────────────────────────────────────────────────┤
                                                                      │
             no captions ──→ ffmpeg audio ──→ whisper CLI ──→ VTT ────┤
                                                                      │
                                                  transcribe.parse_vtt()
                                                                      │
        ffmpeg frames ──→ JPEGs ────────────────────────────→ Claude reads
```

## Components

| Module | Change | Rationale |
|---|---|---|
| `whisper.py` | Delete; replace with `localstt.py` | Entirely API-client plumbing. Port `audio_duration()`, and `extract_audio()` rewritten to emit 16 kHz mono PCM WAV. |
| `transcribe.py` | None | Already parses VTT. The seam is free. |
| `setup.py` | Rewrite wizard | Key prompts become: is the `whisper` CLI present. No secrets, no `.env` key storage. |
| `frames.py` | Two fixes | See below. |
| `download.py`, `watch.py`, `config.py` | Flag plumbing | `--whisper groq\|openai` becomes `--model`. `--no-whisper` loses its key semantics. |
| `hooks/check-setup.sh` | Rewrite | Drop the key check; fix the parser bug below. |

## Defects fixed in passing

Both were found while running v0.2.0 on macOS on 2026-09-01. Each becomes a
named test rather than a comment, so it fails loudly if reintroduced.

### `-vsync` was removed from ffmpeg

`frames.py` lines 256 and 615 pass `-vsync vfr`. ffmpeg removed the option in
7.x; the host runs 9.0.1. Every scene-aware and keyframe extraction aborts with
`Unrecognized option 'vsync'`, making the skill non-functional on any current
ffmpeg. Fix: `-fps_mode vfr`, supported since ffmpeg 5.1.

### The config parser stops at the first match

`hooks/scripts/check-setup.sh` reads keys with `awk '... {print; exit}'`, taking
the first occurrence. `config.py` builds a dict, so it takes the last. A file
with a blank `OPENAI_API_KEY=` above a populated one makes the two disagree —
the hook reports no key while the script finds one. Fix: make the shell read
last-wins, matching `config.py`.

## Sampling strategy

Scene-change detection is the sole frame selector at `balanced` and
`token-burner`. It is near-blind to screencasts, where the picture changes
little across a cut.

Measured 2026-09-01: a 10-minute range of a 15-minute chart tutorial yielded
**12 frames against a budget of 100**. Long static stretches went entirely
unsampled.

Change: treat scene hits as a supplement rather than the whole strategy. Take
the scene-selected frames, and where they fall short of the frame budget, top
up with uniform samples across the uncovered spans. Mixed talking-head and
screencast content then gets both cut boundaries and even coverage.

Existing frame-budget tables, the 2 fps cap, the dedup pass, and focus mode are
retained unchanged.

## Error handling

Failures are now local, so they are reported specifically and without retry:

- `whisper-cli` absent: name `brew install whisper-cpp`.
- Model file absent: name the model, its size, and offer the download. Never
  download 1.5 GB without consent.
- Model download fails: surface the URL, the HTTP status, and the cache path.
- ffmpeg fails: surface stderr verbatim rather than a generic `SystemExit`.

The entire network error class — 429 backoff, upload chunking, partial-chunk
transcripts — is deleted rather than reimplemented.

## Testing

71 tests exist. The 55 outside `test_whisper.py` are the regression net for the
swap and must pass untouched.

`test_whisper.py`'s 16 hosted-API tests are replaced by local-engine tests
covering: CLI argument construction, VTT discovery in the output directory,
missing-binary handling, and range filtering.

Two regression tests are added for the defects above, named so the quirk is
legible without commentary:

- `test_frames_uses_fps_mode_not_removed_vsync`
- `test_env_parser_takes_last_duplicate_key`

## Out of scope

- Any second transcription backend. One local engine is enough; a fallback
  would be speculative work.
- Core ML / Apple Neural Engine encoder builds. whisper.cpp supports them via a
  separate build process; the Homebrew bottle with Metal is expected to be fast
  enough. Revisit only if measurement says otherwise.
- Renaming the skill or the `/watch` command. Muscle memory and the published
  SKILL.md are preserved.
- Upstreaming the two defect fixes to `bradautomates/claude-video`. Worth
  doing, tracked separately, not a dependency of this work.

## Open questions

1. Whether `large-v3-turbo` is the right default. It is the fastest of the
   large models, but the turbo family is not trained for translation — a
   non-English source transcribes in its own language rather than translating
   to English. If translation matters, the default becomes `large-v3` and the
   run gets slower. Currently assumed English-language sources.

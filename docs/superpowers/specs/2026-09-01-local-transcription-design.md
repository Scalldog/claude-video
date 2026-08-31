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

`openai-whisper` — OpenAI's open-source local implementation (MIT). Despite the
name it involves no API, no key, and no network at inference time.

Installed isolated, because the host Python is 3.14.7 and whisper documents
3.8-3.11 support:

```
uv tool install --python 3.11 openai-whisper
```

Invoked as:

```
whisper <audio> --model turbo --output_format vtt --output_dir <workdir>
```

`--output_format` accepts `vtt`, which is the reason this swap is cheap: see
Seam below.

### Known engine limitation

`--device` defaults to `"cuda" if available else "cpu"`. There is no MPS
default, so inference runs on CPU on Apple Silicon. The README's ~8x-realtime
figure for `turbo` is a GPU measurement and does not transfer.

This is accepted unmeasured. Implementation must benchmark a real 15-minute
video before the design is considered settled. If wall-clock is unacceptable, a
`whisper.cpp` backend (Metal-accelerated, `brew install whisper-cpp`, `-ovtt`)
is the fallback — it emits VTT into the same seam, so it is an additive change,
not a redesign. It is deliberately not built up front.

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
| `whisper.py` | Delete; replace with `localstt.py` | Entirely API-client plumbing. Port only `extract_audio()` and `audio_duration()`. |
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

- `whisper` CLI absent: name the `uv tool install` command.
- Model download fails: surface the path and the underlying error.
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

- A `whisper.cpp` backend. Additive later if benchmarking demands it.
- Renaming the skill or the `/watch` command. Muscle memory and the published
  SKILL.md are preserved.
- Upstreaming the two defect fixes to `bradautomates/claude-video`. Worth
  doing, tracked separately, not a dependency of this work.

## Open questions

1. Which Whisper model is the default — `turbo` is the fastest multilingual
   option but is not trained for translation. Confirm before implementation if
   non-English sources matter.
2. Whether CPU wall-clock is acceptable. Resolved by benchmark, not discussion.

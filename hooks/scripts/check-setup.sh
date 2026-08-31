#!/usr/bin/env bash
# SessionStart hook for /watch — one-line status so users know what's wired up.
# Silent on ready state to avoid spam. Points at the installer when something
# is missing.
set -euo pipefail

CONFIG_FILE="$HOME/.config/watch/.env"

read_key() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    echo "${!name}"
    return
  fi
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
}

HAS_FFMPEG=""
HAS_YTDLP=""
command -v ffmpeg >/dev/null 2>&1 && HAS_FFMPEG="yes"
command -v yt-dlp >/dev/null 2>&1 && HAS_YTDLP="yes"

HAS_WHISPER=""
command -v whisper-cli >/dev/null 2>&1 && HAS_WHISPER="yes"
MODEL="$HOME/.cache/whisper-cpp/ggml-large-v3-turbo.bin"

SETUP_COMPLETE="$(read_key SETUP_COMPLETE)"

# Fully configured, or setup already completed once → silent.
if [[ "$SETUP_COMPLETE" == "true" && -n "$HAS_FFMPEG" && -n "$HAS_YTDLP" && -n "$HAS_WHISPER" ]]; then
  exit 0
fi

if [[ -z "$HAS_FFMPEG" || -z "$HAS_YTDLP" || -z "$HAS_WHISPER" ]]; then
  echo "/watch: needs ffmpeg + yt-dlp + whisper-cpp. Run \`python3 \$CLAUDE_PLUGIN_ROOT/skills/watch/scripts/setup.py\` once to install and scaffold config."
elif [[ ! -f "$MODEL" ]]; then
  echo "/watch: model missing. Run \`python3 \$CLAUDE_PLUGIN_ROOT/skills/watch/scripts/setup.py\` for the download command."
else
  echo "/watch: ready."
fi

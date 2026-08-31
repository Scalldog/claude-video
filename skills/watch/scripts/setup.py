#!/usr/bin/env python3
"""Setup / preflight for /watch's whisper.cpp transcription.

Modes:
  setup.py --check      Silent preflight. Exit 0 if ready, 2/3/4 on failure.
  setup.py --json       Machine-readable status for Claude to parse.
  setup.py              Installer. Auto-installs deps, offers the model download.

Design:
- Silent on success: --check exits 0 with no output when everything's ready so
  that /watch doesn't spam "setup is complete" on every turn.
- Idempotent: re-running the installer is safe — it never clobbers an
  existing config file.
- Never sudo. On macOS, auto-install via brew. Elsewhere, print exact commands.
- Never download the whisper.cpp model automatically — it is ~1.5 GB, so
  setup.py only prints the command and leaves running it to the user.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from config import get_config, read_env_file  # noqa: E402
import localstt  # noqa: E402


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"
ENV_TEMPLATE = """# /watch configuration

# Default watch behavior (the /watch first-run wizard sets this for you).
# Allowed values: transcript | efficient | balanced | token-burner
# Keep the value on its own line with no trailing comment.
# WATCH_DETAIL=balanced

# Set by setup.py once ffmpeg, whisper-cli, and the whisper.cpp model are
# all in place.
# SETUP_COMPLETE=true
"""


def _which(name: str) -> str | None:
    return shutil.which(name)


def _check_binaries() -> list[str]:
    missing = [b for b in ("ffmpeg", "ffprobe", "yt-dlp") if _which(b) is None]
    if localstt.find_binary() is None:
        missing.append("whisper-cli")
    return missing


def _setup_complete() -> bool:
    value = os.environ.get("SETUP_COMPLETE")
    if not value:
        value = read_env_file(CONFIG_FILE).get("SETUP_COMPLETE")
    return value == "true"


def _scaffold_env() -> bool:
    """Create ~/.config/watch/.env with placeholders if missing."""
    if CONFIG_FILE.exists():
        return False
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(ENV_TEMPLATE, encoding="utf-8")
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass
    return True


def _write_setup_complete() -> None:
    """Idempotently append SETUP_COMPLETE=true to .env.

    Used only after a fully successful install (deps + model). Future
    sessions detect this marker to skip wizard-style UI and stay silent.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = ""
    if CONFIG_FILE.exists():
        existing = CONFIG_FILE.read_text(encoding="utf-8")
        for line in existing.splitlines():
            if line.strip().startswith("SETUP_COMPLETE="):
                return
        if existing and not existing.endswith("\n"):
            existing += "\n"
        CONFIG_FILE.write_text(existing + "SETUP_COMPLETE=true\n", encoding="utf-8")
    else:
        CONFIG_FILE.write_text(ENV_TEMPLATE + "\nSETUP_COMPLETE=true\n", encoding="utf-8")
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass


def _brew_pkg(missing: list[str]) -> list[str]:
    pkgs: list[str] = []
    for bin_name in missing:
        if bin_name in ("ffmpeg", "ffprobe"):
            if "ffmpeg" not in pkgs:
                pkgs.append("ffmpeg")
        elif bin_name == "yt-dlp":
            if "yt-dlp" not in pkgs:
                pkgs.append("yt-dlp")
        elif bin_name == "whisper-cli":
            if "whisper-cpp" not in pkgs:
                pkgs.append("whisper-cpp")
        else:
            pkgs.append(bin_name)
    return pkgs


def _install_macos(missing: list[str]) -> tuple[bool, str]:
    if _which("brew") is None:
        return False, (
            "Homebrew is not installed. Install it from https://brew.sh, then re-run setup. "
            "Or install manually: `brew install " + " ".join(_brew_pkg(missing)) + "`"
        )
    pkgs = _brew_pkg(missing)
    if not pkgs:
        return True, "nothing to install"
    cmd = ["brew", "install", *pkgs]
    print(f"[setup] running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return False, f"brew install failed with exit code {result.returncode}"
    return True, f"installed via brew: {', '.join(pkgs)}"


def _install_hint_linux(missing: list[str]) -> str:
    pkgs = _brew_pkg(missing)
    hints = []
    if "ffmpeg" in pkgs:
        hints.append("apt: `sudo apt install ffmpeg` or dnf: `sudo dnf install ffmpeg`")
    if "yt-dlp" in pkgs:
        hints.append("`pipx install yt-dlp` (recommended) or `pip install --user yt-dlp`")
    if "whisper-cpp" in pkgs:
        hints.append(
            "`brew install whisper-cpp` (Homebrew on Linux) or build from source: "
            "https://github.com/ggerganov/whisper.cpp"
        )
    return "\n  ".join(hints) if hints else "nothing to install"


def _install_hint_windows(missing: list[str]) -> str:
    pkgs = _brew_pkg(missing)
    hints = []
    if "ffmpeg" in pkgs:
        hints.append("winget: `winget install Gyan.FFmpeg`")
    if "yt-dlp" in pkgs:
        hints.append("winget: `winget install yt-dlp.yt-dlp` or pip: `pip install --user yt-dlp`")
    if "whisper-cpp" in pkgs:
        hints.append("build from source: https://github.com/ggerganov/whisper.cpp")
    return "\n  ".join(hints) if hints else "nothing to install"


def offer_model_download(model: str = localstt.DEFAULT_MODEL) -> bool:
    dest = localstt.model_path(model)
    url = localstt.model_url(model)
    print(f"[setup] model {dest.name} is not present (~1.5 GB).")
    print(f"[setup] download it with:\n  curl -L --create-dirs -o {dest} {url}")
    return dest.exists()


def _status() -> dict:
    """Structured preflight snapshot for whisper.cpp readiness.

    `status` describes the *ideal* state, so a completed-but-modelless
    install still reports `needs_model` — that's the agent's cue to
    encourage downloading it.

    `can_proceed` is the operational gate: /watch can run as long as the
    binaries are present AND either the model is there or setup was already
    completed once (a captions-only workflow, declining the 1.5 GB download,
    is a legitimate deliberate choice). A modelless user who completed setup
    is NOT nagged on every call; a genuine first run is.
    """
    missing = _check_binaries()
    model_file = localstt.model_path(localstt.DEFAULT_MODEL)
    model_present = model_file.exists()
    setup_complete = _setup_complete()
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
        "can_proceed": not missing and (model_present or setup_complete),
        "first_run": not CONFIG_FILE.exists(),
        "missing_binaries": missing,
        "model_present": model_present,
        "model_path": str(model_file),
        "setup_complete": setup_complete,
        "config_file": str(CONFIG_FILE),
        "watch_detail": cfg["detail"],
        "platform": platform.system(),
    }


def cmd_check() -> int:
    """Silent-on-success preflight.

    Exit 0 with no output when /watch can run: ffmpeg, ffprobe, yt-dlp, and
    whisper-cli are all on PATH, and either the whisper.cpp model is present
    or setup has already been completed once. A modelless user who finished
    setup is never nagged again on follow-up calls.

    On a state that blocks /watch, print one actionable line to stderr:
      2 → binaries missing
      3 → genuine first run with no model (encourage downloading one)
      4 → both missing
    """
    s = _status()
    if s["can_proceed"]:
        return 0

    parts = []
    if s["missing_binaries"]:
        parts.append(f"missing binaries: {', '.join(s['missing_binaries'])}")
    if not s["model_present"]:
        parts.append(f"whisper.cpp model missing: {s['model_path']}")
    installer = Path(__file__).resolve()
    sys.stderr.write(
        f"[watch] setup incomplete ({'; '.join(parts)}). "
        f"Run: python3 {installer}\n"
    )
    sys.stderr.flush()

    if s["missing_binaries"] and not s["model_present"]:
        return 4
    if s["missing_binaries"]:
        return 2
    return 3


def cmd_json() -> int:
    json.dump(_status(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_install() -> int:
    missing = _check_binaries()
    installed_deps = False
    if missing:
        system = platform.system()
        if system == "Darwin":
            ok, msg = _install_macos(missing)
            print(f"[setup] {msg}", file=sys.stderr)
            if not ok:
                return 2
            still_missing = _check_binaries()
            if still_missing:
                print(f"[setup] still missing after install: {', '.join(still_missing)}", file=sys.stderr)
                return 2
            installed_deps = True
        elif system == "Linux":
            print("[setup] dependencies missing on Linux — please install:", file=sys.stderr)
            print("  " + _install_hint_linux(missing), file=sys.stderr)
            return 2
        elif system == "Windows":
            print("[setup] dependencies missing on Windows — please install:", file=sys.stderr)
            print("  " + _install_hint_windows(missing), file=sys.stderr)
            return 2
        else:
            print(f"[setup] unsupported platform ({system}) for auto-install. Install manually:", file=sys.stderr)
            print(f"  missing: {', '.join(missing)}", file=sys.stderr)
            return 2

    created = _scaffold_env()
    if created:
        print(f"[setup] created config: {CONFIG_FILE}")
    else:
        print(f"[setup] config exists: {CONFIG_FILE}")

    if localstt.model_path(localstt.DEFAULT_MODEL).exists():
        _write_setup_complete()
        print("[setup] ready. whisper.cpp model is present.")
        if installed_deps:
            print("[setup] installed dependencies; /watch is fully set up.")
        return 0

    offer_model_download()
    return 3


def main() -> int:
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--check":
            return cmd_check()
        if arg == "--json":
            return cmd_json()
    return cmd_install()


if __name__ == "__main__":
    raise SystemExit(main())

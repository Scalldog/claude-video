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


def test_model_present_requires_minimum_size(tmp_path):
    small = tmp_path / "ggml-base.bin"
    small.write_bytes(b"x" * (localstt.MODEL_MIN_SIZE_BYTES - 1))
    assert localstt.model_present(small) is False

    big = tmp_path / "ggml-large.bin"
    big.write_bytes(b"x" * localstt.MODEL_MIN_SIZE_BYTES)
    assert localstt.model_present(big) is True


def test_model_present_is_false_when_missing(tmp_path):
    assert localstt.model_present(tmp_path / "absent.bin") is False


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


# --- VTT discovery: out_prefix.with_suffix(".vtt") lookup + its error path ---

def _stub_ready_model(monkeypatch, tmp_path):
    monkeypatch.setattr(localstt, "find_binary", lambda: "whisper-cli")
    model_file = tmp_path / "ggml-large-v3-turbo.bin"
    model_file.write_bytes(b"x" * localstt.MODEL_MIN_SIZE_BYTES)
    monkeypatch.setattr(localstt, "model_path", lambda model=None: model_file)
    monkeypatch.setattr(localstt, "extract_audio", lambda video_path, out_path: out_path)


class _FakeResult:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


def test_transcribe_discovers_vtt_at_out_prefix(tmp_path, monkeypatch):
    _stub_ready_model(monkeypatch, tmp_path)

    def fake_run(cmd, capture_output=True, text=True):
        out_prefix = Path(cmd[cmd.index("-of") + 1])
        out_prefix.with_suffix(".vtt").write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8"
        )
        return _FakeResult()

    monkeypatch.setattr(localstt.subprocess, "run", fake_run)

    segments, engine = localstt.transcribe_video("video.mp4", tmp_path / "audio.wav")
    assert segments == [{"start": 0.0, "end": 1.0, "text": "hello"}]
    assert engine == "whisper.cpp (large-v3-turbo)"


def test_transcribe_raises_when_no_vtt_produced(tmp_path, monkeypatch):
    """whisper-cli can exit 0 and still leave no VTT (e.g. an unwritable
    output dir) — that must surface as an actionable error, not a silent
    empty transcript."""
    _stub_ready_model(monkeypatch, tmp_path)
    monkeypatch.setattr(localstt.subprocess, "run", lambda *a, **k: _FakeResult())

    with pytest.raises(SystemExit) as exc:
        localstt.transcribe_video("video.mp4", tmp_path / "audio.wav")
    assert "produced no VTT" in str(exc.value)


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

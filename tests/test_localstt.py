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

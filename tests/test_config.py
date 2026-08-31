"""WATCH_DETAIL resolution and frame_cap mapping."""
from __future__ import annotations

import subprocess
from pathlib import Path

import config


def test_default_detail_is_balanced(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "balanced"


def test_env_overrides_detail(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_DETAIL", "efficient")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "efficient"


def test_invalid_detail_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_DETAIL", "bogus")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "balanced"


def test_get_config_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    cfg = config.get_config()
    assert set(cfg) == {"detail", "config_file"}


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100


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
    HOOK = Path(__file__).resolve().parent.parent / "hooks" / "scripts" / "check-setup.sh"
    source = HOOK.read_text(encoding="utf-8")
    assert "exit }" not in source and "; exit" not in source

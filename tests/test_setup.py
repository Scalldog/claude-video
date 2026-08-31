"""setup.py --json surfaces the resolved watch detail and whisper.cpp readiness."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"


def _run(args, *, home=None, extra_env=None):
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    if home is not None:
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)  # Windows
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SETUP), *args],
        capture_output=True, text=True, env=env,
    )


def _write_env(home: Path, body: str) -> None:
    cfg = home / ".config" / "watch"
    cfg.mkdir(parents=True, exist_ok=True)
    f = cfg / ".env"
    f.write_text(body, encoding="utf-8")
    f.chmod(0o600)


def _write_model(home: Path, model: str = "large-v3-turbo") -> Path:
    cache = home / ".cache" / "whisper-cpp"
    cache.mkdir(parents=True, exist_ok=True)
    model_file = cache / f"ggml-{model}.bin"
    model_file.write_bytes(b"")
    return model_file


def test_json_reports_watch_detail():
    proc = _run(["--json"])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["watch_detail"] == "balanced"


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


def test_completed_setup_without_model_is_not_treated_as_first_run(tmp_path):
    """A user who has already run setup is not treated like a stranger just
    because the model download hasn't happened yet — but the model is a hard
    requirement now, so --check still nags until it's there."""
    _write_env(tmp_path, "SETUP_COMPLETE=true\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["first_run"] is False
    assert js["model_present"] is False
    assert js["can_proceed"] is False
    assert js["status"] == "needs_model"


def test_modelless_first_run_is_encouraged(tmp_path):
    """Genuine first run with no config file and no model: --check reports
    exit 3 and first_run is True."""
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is False
    assert js["first_run"] is True
    assert js["model_present"] is False


def test_model_present_is_ready(tmp_path):
    _write_model(tmp_path)
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, chk.stderr
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["model_present"] is True

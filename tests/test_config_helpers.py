import os
from pathlib import Path

from main import load_local_env, resolve_path, resolve_project_path


def test_resolve_path_keeps_absolute_path(tmp_path):
    absolute_path = tmp_path / "model.xml"
    assert resolve_path(Path("/project"), str(absolute_path)) == str(absolute_path)


def test_resolve_path_anchors_relative_path():
    assert resolve_path("/project", "models/policy.pt") == "/project/models/policy.pt"


def test_resolve_project_path_uses_env_override(monkeypatch):
    monkeypatch.setenv("ROBOT_MODEL_XML", "custom/model.xml")
    assert resolve_project_path("/project", "ROBOT_MODEL_XML", "default.xml") == "/project/custom/model.xml"


def test_load_local_env_does_not_override_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("EXAMPLE_KEY=from_file\nKEEP_ME=from_file\n", encoding="utf-8")
    monkeypatch.setenv("KEEP_ME", "already_set")

    load_local_env(str(tmp_path), ".env")

    assert os.environ["EXAMPLE_KEY"] == "from_file"
    assert os.environ["KEEP_ME"] == "already_set"

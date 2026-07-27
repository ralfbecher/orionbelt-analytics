"""OUTPUT_DIR must never resolve onto the installation itself.

Startup cleanup deletes loose files directly under OUTPUT_DIR, and with
AUTO_CLEANUP_ON_STARTUP=true removes every subdirectory without a
metadata.json. That is correct for a dedicated output directory and
catastrophic anywhere else -- pathlib collapses "" and "." onto PROJECT_ROOT,
so a blanked-out OUTPUT_DIR= in .env would target the checkout: unlinking
.env and pyproject.toml, then removing src/, tests/ and .git/.

The value cases call the resolver directly. Deliberately no sys.modules
purging: reloading src.* mid-session gives other tests different module
objects than their mocks were bound to, which silently broke 14 unrelated
tests when this file first did that. Import-time behaviour is checked in a
subprocess instead, which is both isolated and closer to what happens at
startup.
"""

import subprocess
import sys

import pytest

from src.paths import PROJECT_ROOT, _resolve_output_dir


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_blank_output_dir_is_rejected(monkeypatch, value):
    """An empty value resolves to the project root -- refuse it."""
    monkeypatch.setenv("OUTPUT_DIR", value)
    with pytest.raises(ValueError, match="empty"):
        _resolve_output_dir()


@pytest.mark.parametrize("value", [".", "..", "./", "src/.."])
def test_output_dir_containing_the_installation_is_rejected(monkeypatch, value):
    """Anything resolving to PROJECT_ROOT or above it is refused."""
    monkeypatch.setenv("OUTPUT_DIR", value)
    with pytest.raises(ValueError, match="contains the installation"):
        _resolve_output_dir()


def test_absolute_project_root_is_rejected(monkeypatch):
    """The check is on the resolved path, not on the spelling."""
    monkeypatch.setenv("OUTPUT_DIR", str(PROJECT_ROOT))
    with pytest.raises(ValueError, match="contains the installation"):
        _resolve_output_dir()


def test_default_is_used_when_unset(monkeypatch):
    """No OUTPUT_DIR means the packaged default."""
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    assert _resolve_output_dir().name == "tmp"


@pytest.mark.parametrize("value", ["tmp", "output/data"])
def test_relative_paths_below_the_root_are_accepted(monkeypatch, value):
    """Normal configurations still work."""
    monkeypatch.setenv("OUTPUT_DIR", value)
    resolved = _resolve_output_dir()
    assert PROJECT_ROOT.resolve() in resolved.parents


def test_absolute_path_outside_the_project_is_accepted(monkeypatch, tmp_path):
    """A production deployment pointing at /var/lib/... must work."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    assert _resolve_output_dir() == tmp_path.resolve()


def test_guard_fires_at_import_time():
    """Startup must fail fast, not at the first destructive operation.

    Run in a subprocess so importing src.paths with a hostile OUTPUT_DIR
    cannot disturb this test session.
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import src.paths"],
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "OUTPUT_DIR": ".",
            "PYTHONPATH": str(PROJECT_ROOT),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode != 0, "importing with OUTPUT_DIR=. should have failed"
    assert "contains the installation" in proc.stderr

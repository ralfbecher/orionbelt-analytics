"""Tests for AUTO_CLEANUP_ON_STARTUP workspace cleanup.

The output directory holds two different kinds of child directory:

    OUTPUT_DIR/{connection_id}/          <- a workspace (has metadata.json)
    OUTPUT_DIR/chromadb/{connection_id}/ <- satellite store, keyed one level down
    OUTPUT_DIR/oxigraph/{connection_id}/ <- satellite store, keyed one level down

Conflating them caused two bugs these tests pin:

  * ``chromadb`` and ``oxigraph`` have no metadata.json, so scanning
    OUTPUT_DIR for workspaces classified them as orphaned and deleted every
    connection's vectors and triples on the first retention run.
  * deleting a workspace left its satellite stores behind forever, which is
    exactly the disk growth the retention mode exists to prevent.
"""

import json

import pytest

from src.paths import NON_WORKSPACE_DIRS
from src.utils import utc_now

FRESH = "connfresh"
STALE = "connstale"
ORPHAN = "connorphan"


def _build_output_tree(root):
    """Lay out a realistic OUTPUT_DIR with workspaces and satellite stores."""
    stale_ts = utc_now().replace(year=utc_now().year - 5).isoformat()

    for cid, updated_at in ((FRESH, utc_now().isoformat()), (STALE, stale_ts)):
        ws = root / cid
        ws.mkdir(parents=True)
        (ws / "metadata.json").write_text(
            json.dumps({"workspace": {"updated_at": updated_at, "schemas": {}}}),
            encoding="utf-8",
        )

    (root / ORPHAN).mkdir(parents=True)  # no metadata.json

    for cid in (FRESH, STALE, ORPHAN):
        (root / "chromadb" / cid).mkdir(parents=True)
        (root / "oxigraph" / cid / "store").mkdir(parents=True)

    (root / "stale_top_level.png").write_text("x", encoding="utf-8")
    (root / FRESH / "charts").mkdir()


def _run_cleanup(monkeypatch, root, mode):
    """Invoke the real cleanup_tmp_folder against a temp OUTPUT_DIR."""
    import server

    monkeypatch.setattr("src.paths.OUTPUT_DIR", root)
    monkeypatch.setenv("AUTO_CLEANUP_ON_STARTUP", mode)
    monkeypatch.setenv("WORKSPACE_MAX_AGE_DAYS", "30")
    server.cleanup_tmp_folder()


def test_non_workspace_dirs_covers_the_real_store_locations():
    """The skip-list must match where the stores are actually written."""
    assert {"chromadb", "oxigraph"} <= set(NON_WORKSPACE_DIRS)


def test_disabled_mode_keeps_workspaces_but_clears_ephemera(tmp_path, monkeypatch):
    """With cleanup off, only loose files and chart dirs go."""
    _build_output_tree(tmp_path)
    _run_cleanup(monkeypatch, tmp_path, "false")

    assert (tmp_path / FRESH).exists()
    assert (tmp_path / STALE).exists(), "disabled mode must not delete workspaces"
    assert (tmp_path / ORPHAN).exists()
    assert not (tmp_path / "stale_top_level.png").exists()
    assert not (tmp_path / FRESH / "charts").exists()


def test_retention_mode_preserves_satellite_store_roots(tmp_path, monkeypatch):
    """chromadb/ and oxigraph/ must never be treated as orphaned workspaces.

    This is the data-loss regression: they carry no metadata.json, so a naive
    scan deleted both wholesale -- every connection's vectors and triples.
    """
    _build_output_tree(tmp_path)
    _run_cleanup(monkeypatch, tmp_path, "true")

    assert (tmp_path / "chromadb").exists(), "chromadb root was deleted"
    assert (tmp_path / "oxigraph").exists(), "oxigraph root was deleted"
    assert (tmp_path / "chromadb" / FRESH).exists()
    assert (tmp_path / "oxigraph" / FRESH / "store").exists()


def test_retention_mode_removes_stale_workspace_and_its_stores(tmp_path, monkeypatch):
    """A deleted workspace must not strand its satellite stores."""
    _build_output_tree(tmp_path)
    _run_cleanup(monkeypatch, tmp_path, "true")

    assert not (tmp_path / STALE).exists()
    assert not (tmp_path / "chromadb" / STALE).exists(), "stranded ChromaDB store"
    assert not (tmp_path / "oxigraph" / STALE).exists(), "stranded Oxigraph store"

    assert (tmp_path / FRESH).exists(), "workspace within retention was deleted"


def test_retention_mode_removes_orphaned_workspace_and_its_stores(
    tmp_path, monkeypatch
):
    """A directory with no metadata.json is orphaned and goes, stores included."""
    _build_output_tree(tmp_path)
    _run_cleanup(monkeypatch, tmp_path, "true")

    assert not (tmp_path / ORPHAN).exists()
    assert not (tmp_path / "chromadb" / ORPHAN).exists()
    assert not (tmp_path / "oxigraph" / ORPHAN).exists()


@pytest.mark.parametrize("cid", [FRESH, STALE, ORPHAN])
def test_all_mode_removes_every_workspace_and_store(tmp_path, monkeypatch, cid):
    """`all` is a full reset -- no workspace or satellite store survives."""
    _build_output_tree(tmp_path)
    _run_cleanup(monkeypatch, tmp_path, "all")

    assert not (tmp_path / cid).exists()
    assert not (tmp_path / "chromadb" / cid).exists()
    assert not (tmp_path / "oxigraph" / cid).exists()


def test_workspace_without_updated_at_is_kept(tmp_path, monkeypatch):
    """Metadata lacking updated_at is not evidence of staleness."""
    ws = tmp_path / "connnodate"
    ws.mkdir(parents=True)
    (ws / "metadata.json").write_text(
        json.dumps({"workspace": {"schemas": {}}}), encoding="utf-8"
    )
    _run_cleanup(monkeypatch, tmp_path, "true")

    assert ws.exists(), "workspace with no updated_at must not be deleted"

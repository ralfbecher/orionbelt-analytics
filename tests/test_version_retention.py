"""Per-version retention: recording, policy, and cleanup.

The cleanup half of this design shipped long before the recording half, which
meant ``get_versions_to_cleanup`` returned ``[]`` unconditionally and the whole
mechanism provably did nothing (issue #68). These tests pin both halves and the
seams between them:

  * ``discover_schema`` opens a version; ontology and GraphRAG fill it in
  * opening a version archives its predecessor, so only the newest is active
  * retention deletes archived versions only, honouring env-driven policy
  * shared resources -- named graphs, ChromaDB collections -- survive cleanup of
    a version while another version still references them
"""

import json
from datetime import timedelta
from pathlib import Path
from unittest import mock

import pytest

from src.lifecycle.cleanup import DataCleanupManager
from src.lifecycle.metadata import (
    RetentionPolicy,
    VersionInfo,
    VersionMetadataManager,
    get_active_version_number,
    open_schema_version,
    schema_fingerprint,
    update_schema_version,
)
from src.utils import utc_now

CONN = "retention-test-conn"
SCHEMA = "public"


class _Column:
    def __init__(self, name: str) -> None:
        self.name = name


class _Table:
    """Minimal stand-in for the driver's TableInfo (duck-typed by fingerprint)."""

    def __init__(self, name: str, columns: list[str], schema: str = SCHEMA) -> None:
        self.name = name
        self.schema = schema
        self.columns = [_Column(c) for c in columns]


def _tables(n: int = 2, columns: int = 3) -> list[_Table]:
    return [_Table(f"t{i}", [f"c{j}" for j in range(columns)]) for i in range(n)]


def _mgr(tmp_path: Path) -> VersionMetadataManager:
    return VersionMetadataManager(CONN, tmp_path)


def _age(mgr: VersionMetadataManager, schema: str, version: int, days: int) -> None:
    """Backdate a recorded version so age-based retention can select it."""
    stamp = (utc_now() - timedelta(days=days)).isoformat()
    for entry in mgr.metadata["schemas"][schema]["versions"]:
        if entry["version"] == version:
            entry["created_at"] = stamp
    mgr._save_metadata()


# --- fingerprinting ---------------------------------------------------------


def test_fingerprint_counts_tables_and_columns():
    digest, tables, columns = schema_fingerprint(_tables(3, 4))
    assert tables == 3
    assert columns == 12
    assert digest


def test_fingerprint_is_order_independent():
    tables = _tables(3)
    forward, _, _ = schema_fingerprint(tables)
    backward, _, _ = schema_fingerprint(list(reversed(tables)))
    assert forward == backward, "hash must not depend on driver ordering"


def test_fingerprint_changes_when_structure_changes():
    before, _, _ = schema_fingerprint([_Table("t", ["a", "b"])])
    after, _, _ = schema_fingerprint([_Table("t", ["a", "b", "c"])])
    assert before != after


def test_fingerprint_of_nothing_is_stable():
    digest, tables, columns = schema_fingerprint([])
    assert (tables, columns) == (0, 0)
    assert digest == schema_fingerprint([])[0]


# --- recording --------------------------------------------------------------


def test_open_version_numbers_from_one(tmp_path):
    mgr = _mgr(tmp_path)
    first = mgr.open_version(SCHEMA, "hash-a", 2, 6)
    second = mgr.open_version(SCHEMA, "hash-b", 3, 9)
    assert (first.version, second.version) == (1, 2)


def test_opening_a_version_archives_the_previous(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.open_version(SCHEMA, "hash-a", 2, 6)
    mgr.open_version(SCHEMA, "hash-b", 3, 9)

    versions = mgr.get_versions(SCHEMA)
    assert [v.status for v in versions] == ["archived", "active"]
    assert [v.graphrag_status for v in versions] == ["archived", "active"]
    assert [v.ontology_status for v in versions] == ["archived", "active"]


def test_only_one_version_is_ever_active(tmp_path):
    mgr = _mgr(tmp_path)
    for i in range(5):
        mgr.open_version(SCHEMA, f"hash-{i}", i, i)
    active = [v for v in mgr.get_versions(SCHEMA) if v.status == "active"]
    assert len(active) == 1
    assert active[0].version == 5


def test_second_version_records_the_delta(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.open_version(SCHEMA, "hash-a", 2, 6)
    second = mgr.open_version(SCHEMA, "hash-b", 5, 20)

    assert second.changes == {
        "previous_version": 1,
        "table_count_delta": 3,
        "schema_changed": True,
    }


def test_unchanged_structure_is_reported_as_unchanged(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.open_version(SCHEMA, "same", 2, 6)
    second = mgr.open_version(SCHEMA, "same", 2, 6)
    assert second.changes["schema_changed"] is False
    assert second.changes["table_count_delta"] == 0


def test_first_version_has_no_delta(tmp_path):
    assert _mgr(tmp_path).open_version(SCHEMA, "h", 1, 1).changes is None


def test_update_version_fills_the_ontology_half(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.open_version(SCHEMA, "h", 2, 6)

    updated = mgr.update_version(
        SCHEMA,
        {"ontology_ttl_file": "ontology_x.ttl", "ontology_triple_count": 412},
    )

    assert updated.ontology_ttl_file == "ontology_x.ttl"
    assert updated.ontology_triple_count == 412
    assert mgr.get_current_version(SCHEMA).ontology_triple_count == 412


def test_update_version_ignores_unknown_fields(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.open_version(SCHEMA, "h", 2, 6)
    assert mgr.update_version(SCHEMA, {"not_a_field": 1}) is None


def test_update_version_cannot_rewrite_identity(tmp_path):
    """version and created_at define the record; a producer must not move them."""
    mgr = _mgr(tmp_path)
    opened = mgr.open_version(SCHEMA, "h", 2, 6)
    mgr.update_version(SCHEMA, {"version": 99, "created_at": "1999-01-01"})

    current = mgr.get_current_version(SCHEMA)
    assert current.version == opened.version
    assert current.created_at == opened.created_at


def test_update_without_an_open_version_is_a_no_op(tmp_path):
    assert _mgr(tmp_path).update_version(SCHEMA, {"ontology_triple_count": 1}) is None


def test_update_targets_a_specific_version(tmp_path):
    """A producer must be able to write to the generation it was started for."""
    mgr = _mgr(tmp_path)
    mgr.open_version(SCHEMA, "h1", 1, 1)
    mgr.open_version(SCHEMA, "h2", 2, 2)  # v1 is now archived

    updated = mgr.update_version(SCHEMA, {"ontology_triple_count": 99}, version=1)

    by_number = {v.version: v for v in mgr.get_versions(SCHEMA)}
    assert updated.version == 1
    assert by_number[1].ontology_triple_count == 99
    assert by_number[2].ontology_triple_count == 0, "the newer version is untouched"


def test_update_of_an_unknown_version_is_a_no_op(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.open_version(SCHEMA, "h", 1, 1)
    assert mgr.update_version(SCHEMA, {"ontology_triple_count": 5}, version=42) is None


async def test_a_slow_producer_cannot_capture_a_newer_version(tmp_path):
    """The seam the race lived in: resolve the version early, write to it late.

    GraphRAG init and AUTO_ONTOLOGY generation run in the background and can
    still be going when a second discover_schema for the same schema opens a
    newer version. Resolving "the active version" at completion time stamped
    the first run's snapshots, TTL file and triple counts onto a generation
    they had nothing to do with.
    """
    started_for = await open_schema_version(CONN, tmp_path, SCHEMA, _tables(2))

    # A rediscovery lands while the slow producer is still working.
    await open_schema_version(CONN, tmp_path, SCHEMA, _tables(5))

    await update_schema_version(
        CONN,
        tmp_path,
        SCHEMA,
        {"graphrag_vector_count": 500, "ontology_triple_count": 900},
        version=started_for,
    )

    by_number = {v.version: v for v in _mgr(tmp_path).get_versions(SCHEMA)}
    assert by_number[1].graphrag_vector_count == 500
    assert (
        by_number[2].graphrag_vector_count == 0
    ), "the newer generation must not inherit the older run's output"
    assert by_number[2].ontology_triple_count == 0


def test_versions_survive_a_reload(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.open_version(SCHEMA, "h", 2, 6)
    mgr.update_version(SCHEMA, {"graphrag_vector_count": 77})

    assert _mgr(tmp_path).get_current_version(SCHEMA).graphrag_vector_count == 77


def test_schemas_have_independent_histories(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.open_version("sales", "h", 1, 1)
    mgr.open_version("sales", "h", 1, 1)
    mgr.open_version("hr", "h", 1, 1)

    assert [v.version for v in mgr.get_versions("sales")] == [1, 2]
    assert [v.version for v in mgr.get_versions("hr")] == [1]


# --- migration of pre-feature workspaces ------------------------------------


def test_legacy_workspace_is_seeded_as_version_one(tmp_path):
    """A workspace written before recording existed keeps its current artifacts."""
    mgr = _mgr(tmp_path)
    mgr.update_workspace(
        SCHEMA, "schema", {"table_count": 9, "analyzed_at": "2026-01-01"}
    )
    mgr.update_workspace(
        SCHEMA, "ontology", {"ontology_file": "old.ttl", "graph_uri": "urn:g"}
    )

    opened = mgr.open_version(SCHEMA, "new-hash", 10, 40)

    versions = mgr.get_versions(SCHEMA)
    assert opened.version == 2, "the pre-existing generation must occupy version 1"
    assert versions[0].ontology_ttl_file == "old.ttl"
    assert versions[0].table_count == 9
    assert versions[0].status == "archived"


def test_empty_workspace_is_not_migrated(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.update_workspace_connection("postgresql", "db")
    assert mgr.open_version(SCHEMA, "h", 1, 1).version == 1


def test_migration_runs_only_once(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.update_workspace(SCHEMA, "ontology", {"ontology_file": "old.ttl"})
    mgr.open_version(SCHEMA, "h", 1, 1)
    mgr.open_version(SCHEMA, "h", 1, 1)

    migrated = [
        v
        for v in mgr.get_versions(SCHEMA)
        if (v.changes or {}).get("migrated_from_workspace")
    ]
    assert len(migrated) == 1


# --- tolerating metadata written by other releases --------------------------


def test_version_from_a_newer_release_loads(tmp_path):
    """An unknown field must not make the workspace unreadable."""
    mgr = _mgr(tmp_path)
    mgr.open_version(SCHEMA, "h", 1, 1)
    mgr.metadata["schemas"][SCHEMA]["versions"][0]["field_from_the_future"] = True
    mgr._save_metadata()

    assert _mgr(tmp_path).get_current_version(SCHEMA).version == 1


def test_version_from_an_older_release_loads(tmp_path):
    """Fields added after the workspace was written fall back to defaults."""
    mgr = _mgr(tmp_path)
    mgr.open_version(SCHEMA, "h", 1, 1)
    entry = mgr.metadata["schemas"][SCHEMA]["versions"][0]
    del entry["graphrag_collection"]
    del entry["graphrag_files"]
    mgr._save_metadata()

    restored = _mgr(tmp_path).get_current_version(SCHEMA)
    assert restored.graphrag_collection == ""
    assert restored.graphrag_files == []


# --- retention policy -------------------------------------------------------


def test_policy_defaults_when_environment_is_clean(monkeypatch):
    for name in (
        "GRAPHRAG_KEEP_VERSIONS",
        "GRAPHRAG_MAX_AGE_DAYS",
        "ONTOLOGY_KEEP_VERSIONS",
        "ONTOLOGY_MAX_AGE_DAYS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert RetentionPolicy.from_metadata(None) == RetentionPolicy()


def test_environment_overrides_the_stored_policy(monkeypatch):
    """An operator changing retention must affect workspaces already on disk."""
    monkeypatch.setenv("GRAPHRAG_KEEP_VERSIONS", "7")
    stored = {"graphrag_keep_versions": 3, "ontology_keep_versions": 5}

    policy = RetentionPolicy.from_metadata(stored)

    assert policy.graphrag_keep_versions == 7
    assert policy.ontology_keep_versions == 5, "untouched fields keep the stored value"


@pytest.mark.parametrize("value", ["0", "-1", "abc", ""])
def test_invalid_policy_values_fall_back(monkeypatch, value):
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", value)
    assert RetentionPolicy.from_metadata(None).ontology_max_age_days == 60


def test_stored_policy_tolerates_removed_fields():
    policy = RetentionPolicy.from_metadata({"a_field_we_dropped": 1})
    assert policy == RetentionPolicy()


# --- selecting versions for cleanup -----------------------------------------


def _history(mgr: VersionMetadataManager, count: int, age_days: int) -> None:
    """Record *count* versions, all archived and aged, plus a live one on top."""
    for i in range(count):
        mgr.open_version(SCHEMA, f"h{i}", 1, 1)
        mgr.update_version(
            SCHEMA,
            {
                "ontology_ttl_file": f"ontology_v{i + 1}.ttl",
                "graphrag_files": [f"vector_store_{SCHEMA}_v{i + 1}.json"],
            },
        )
    mgr.open_version(SCHEMA, "live", 1, 1)  # archives all of the above
    for version in range(1, count + 1):
        _age(mgr, SCHEMA, version, age_days)


def test_the_active_version_is_never_a_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=6, age_days=999)

    doomed = {v.version for v in mgr.get_versions_to_cleanup(SCHEMA, "ontology")}
    active = mgr.get_current_version(SCHEMA).version
    assert active not in doomed


def test_recent_versions_are_kept_however_many_there_are(tmp_path):
    mgr = _mgr(tmp_path)
    _history(mgr, count=8, age_days=0)
    assert mgr.get_versions_to_cleanup(SCHEMA, "ontology") == []


def test_old_versions_beyond_the_keep_count_are_selected(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "2")
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", "30")
    mgr = _mgr(tmp_path)
    _history(mgr, count=6, age_days=100)

    doomed = sorted(v.version for v in mgr.get_versions_to_cleanup(SCHEMA, "ontology"))
    # 6 archived; the newest 2 are kept by count, leaving 1-4 old enough to go.
    assert doomed == [1, 2, 3, 4]


def test_a_short_history_is_left_alone(tmp_path):
    """min_versions is a floor even when everything else says delete."""
    mgr = _mgr(tmp_path)
    _history(mgr, count=1, age_days=9999)
    assert mgr.get_versions_to_cleanup(SCHEMA, "ontology") == []


def test_min_versions_spares_the_newest_candidates(tmp_path, monkeypatch):
    """When the floor bites, the versions worth rolling back to must survive.

    Reachable only because keep_count is operator-settable: with keep_count >=
    min_versions the floor never binds. Here 3 archived versions, keep 1, floor
    2 -- so exactly one of the two candidates may go, and it must be the older.
    Trimming the candidate list from the wrong end would delete v2 and keep the
    staler v1.
    """
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "1")
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=3, age_days=500)

    doomed = {v.version for v in mgr.get_versions_to_cleanup(SCHEMA, "ontology")}
    assert doomed == {1}, "the oldest candidate must be the one discarded"


def test_all_requires_both_halves_to_agree(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "2")
    monkeypatch.setenv("GRAPHRAG_KEEP_VERSIONS", "5")
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", "1")
    monkeypatch.setenv("GRAPHRAG_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=6, age_days=100)

    ontology = {v.version for v in mgr.get_versions_to_cleanup(SCHEMA, "ontology")}
    both = {v.version for v in mgr.get_versions_to_cleanup(SCHEMA, "all")}

    assert both < ontology, "the stricter GraphRAG policy must constrain 'all'"


def test_a_schema_with_no_history_yields_nothing(tmp_path):
    assert _mgr(tmp_path).get_versions_to_cleanup("never-seen", "all") == []


# --- cleanup ----------------------------------------------------------------


async def test_dry_run_reports_without_deleting(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "1")
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=4, age_days=100)

    ttl = tmp_path / CONN / "ontology_v1.ttl"
    ttl.write_text("@prefix x: <urn:x> .", encoding="utf-8")

    report = await DataCleanupManager(CONN, tmp_path).cleanup_ontology(
        SCHEMA, dry_run=True
    )

    assert report["deleted"], "dry run must still report what it would remove"
    assert ttl.exists(), "dry run must not delete anything"
    assert _mgr(tmp_path).get_versions(SCHEMA)[0].ontology_status != "deleted"


async def test_cleanup_deletes_ttl_files_and_marks_the_version(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "1")
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=4, age_days=100)

    ttl = tmp_path / CONN / "ontology_v1.ttl"
    ttl.write_text("@prefix x: <urn:x> .", encoding="utf-8")

    await DataCleanupManager(CONN, tmp_path).cleanup_ontology(SCHEMA, dry_run=False)

    assert not ttl.exists()
    reloaded = {v.version: v for v in _mgr(tmp_path).get_versions(SCHEMA)}
    assert reloaded[1].ontology_status == "deleted"
    assert reloaded[4].ontology_status == "archived", "survivors keep their status"


async def test_cleanup_deletes_versioned_graphrag_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_KEEP_VERSIONS", "1")
    monkeypatch.setenv("GRAPHRAG_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=4, age_days=100)

    snapshot = tmp_path / CONN / f"vector_store_{SCHEMA}_v1.json"
    snapshot.write_text("{}", encoding="utf-8")
    current = tmp_path / CONN / f"vector_store_{SCHEMA}.json"
    current.write_text("{}", encoding="utf-8")

    await DataCleanupManager(CONN, tmp_path).cleanup_graphrag(SCHEMA, dry_run=False)

    assert not snapshot.exists()
    assert current.exists(), "the unversioned current-state file must survive"


async def test_a_shared_named_graph_survives(tmp_path, monkeypatch):
    """Generations reuse one graph URI; deleting an old version must not drop it."""
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "1")
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=4, age_days=100)
    for entry in mgr.metadata["schemas"][SCHEMA]["versions"]:
        entry["ontology_graph_uri"] = "urn:graph:public"
    mgr._save_metadata()

    store = mock.Mock()
    await DataCleanupManager(CONN, tmp_path).cleanup_ontology(
        SCHEMA, dry_run=False, oxigraph_store=store
    )

    store.delete_graph.assert_not_called()


async def test_an_orphaned_named_graph_is_deleted(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "1")
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=4, age_days=100)
    versions = mgr.metadata["schemas"][SCHEMA]["versions"]
    versions[0]["ontology_graph_uri"] = "urn:graph:retired"
    for entry in versions[1:]:
        entry["ontology_graph_uri"] = "urn:graph:current"
    mgr._save_metadata()

    store = mock.Mock()
    await DataCleanupManager(CONN, tmp_path).cleanup_ontology(
        SCHEMA, dry_run=False, oxigraph_store=store
    )

    store.delete_graph.assert_called_once_with("urn:graph:retired")


async def test_a_shared_chromadb_collection_survives(tmp_path, monkeypatch):
    """One collection backs every schema and generation -- see the guard's docstring."""
    monkeypatch.setenv("GRAPHRAG_KEEP_VERSIONS", "1")
    monkeypatch.setenv("GRAPHRAG_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=4, age_days=100)
    for entry in mgr.metadata["schemas"][SCHEMA]["versions"]:
        entry["graphrag_collection"] = "schema_public"
    mgr._save_metadata()

    manager = DataCleanupManager(CONN, tmp_path)
    with mock.patch("chromadb.PersistentClient") as client:
        await manager.cleanup_graphrag(SCHEMA, dry_run=False)

    client.assert_not_called()


async def test_cleanup_reports_the_versions_it_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "1")
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=4, age_days=100)

    report = await DataCleanupManager(CONN, tmp_path).cleanup_ontology(
        SCHEMA, dry_run=False
    )

    assert report["dry_run"] is False
    assert not report["errors"]
    assert all(entry["age_days"] >= 100 for entry in report["deleted"])


async def test_nothing_to_do_is_reported_as_such(tmp_path):
    mgr = _mgr(tmp_path)
    _history(mgr, count=2, age_days=0)

    report = await DataCleanupManager(CONN, tmp_path).cleanup_graphrag(SCHEMA)

    assert report["kept_all"] is True
    assert report["deleted"] == []


async def test_metadata_stays_parseable_after_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "1")
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=5, age_days=100)

    await DataCleanupManager(CONN, tmp_path).cleanup_ontology(SCHEMA, dry_run=False)

    raw = (tmp_path / CONN / "metadata.json").read_text(encoding="utf-8")
    assert json.loads(raw)["schemas"][SCHEMA]["versions"]


async def test_cleanup_preserves_other_workspace_sections(tmp_path, monkeypatch):
    """Cleanup writes metadata.json; it must not drop what other writers put there."""
    monkeypatch.setenv("ONTOLOGY_KEEP_VERSIONS", "1")
    monkeypatch.setenv("ONTOLOGY_MAX_AGE_DAYS", "1")
    mgr = _mgr(tmp_path)
    _history(mgr, count=4, age_days=100)
    mgr.update_workspace_connection("postgresql", "analytics")

    await DataCleanupManager(CONN, tmp_path).cleanup_ontology(SCHEMA, dry_run=False)

    workspace = _mgr(tmp_path).get_workspace()
    assert workspace["db_name"] == "analytics"


# --- the locked async helpers -----------------------------------------------


async def test_open_and_update_round_trip_through_the_lock(tmp_path):
    version = await open_schema_version(CONN, tmp_path, SCHEMA, _tables(4, 2))
    assert version == 1

    assert await get_active_version_number(CONN, tmp_path, SCHEMA) == 1

    await update_schema_version(CONN, tmp_path, SCHEMA, {"graphrag_vector_count": 120})

    current = _mgr(tmp_path).get_current_version(SCHEMA)
    assert current.table_count == 4
    assert current.column_count == 8
    assert current.graphrag_vector_count == 120


async def test_active_version_of_an_unknown_schema_is_none(tmp_path):
    assert await get_active_version_number(CONN, tmp_path, "nope") is None


# --- versioned GraphRAG snapshots -------------------------------------------


def _fake_manager(schemas: list[str], payload: str = "{}"):
    """A GraphRAGManager with just enough wired up to exercise save_state."""
    from src.graphrag.manager import GraphRAGManager

    manager = GraphRAGManager.__new__(GraphRAGManager)
    manager._connection_id = CONN
    manager._schema_names = schemas
    manager.graph_retriever = mock.Mock()
    manager.graph_retriever._tables_info = {}
    manager.graph_retriever.export_graph_for_visualization.return_value = {}
    manager.community_detector = None
    manager.vector_store = mock.Mock()
    manager.vector_store.save.side_effect = lambda p: Path(p).write_text(payload)
    return manager


def test_save_state_without_a_version_writes_no_snapshots(tmp_path):
    assert _fake_manager([]).save_state(tmp_path) == []


def test_save_state_snapshots_each_per_schema_file(tmp_path):
    manager = _fake_manager([SCHEMA])

    written = manager.save_state(tmp_path, version=3, snapshot_schema=SCHEMA)

    assert f"vector_store_{SCHEMA}_v3.json" in written
    assert f"graph_{SCHEMA}_v3.json" in written
    assert (
        tmp_path / CONN / f"vector_store_{SCHEMA}.json"
    ).exists(), "the unversioned current-state file must still be written"


def test_snapshot_is_confined_to_the_schema_being_recorded(tmp_path):
    """One schema's version number must not snapshot every accumulated schema.

    GraphRAG is connection-scoped, so this manager holds every schema on the
    connection, but version numbers are per schema. Snapshotting all of them
    under the version being recorded overwrote public's v1 when analytics v1
    was saved, and handed public's snapshot to analytics's version record --
    so cleaning up analytics v1 would have deleted public's history.
    """
    manager = _fake_manager(["public", "analytics"], payload='{"state": "first"}')
    manager.save_state(tmp_path, version=1, snapshot_schema="public")
    public_v1 = tmp_path / CONN / "vector_store_public_v1.json"
    original = public_v1.read_text(encoding="utf-8")

    # analytics is discovered next and accumulates; its own v1 is recorded.
    manager.vector_store.save.side_effect = lambda p: Path(p).write_text(
        '{"state": "after-analytics"}'
    )
    written = manager.save_state(tmp_path, version=1, snapshot_schema="analytics")

    assert (
        public_v1.read_text(encoding="utf-8") == original
    ), "another schema's save must not rewrite public's snapshot"
    assert not any(
        "public" in name for name in written
    ), "analytics's version must not claim ownership of public's files"
    assert (tmp_path / CONN / "vector_store_analytics_v1.json").exists()


def test_no_snapshot_without_a_named_schema(tmp_path):
    """A version number alone is ambiguous on a multi-schema connection."""
    assert _fake_manager(["public", "analytics"]).save_state(tmp_path, version=1) == []


# --- end to end through the handler -----------------------------------------


async def test_discover_schema_records_a_version(tmp_path, monkeypatch):
    """The seam the issue was really about: a real discovery must write history.

    Every other test here drives the metadata API directly, which is exactly the
    state the bug was in -- the recording API can be perfect and still never be
    called. This exercises the handler.
    """
    from unittest.mock import AsyncMock, MagicMock, Mock, patch

    import src.main as main_module
    from src.database_manager import ColumnInfo, DatabaseManager, TableInfo

    monkeypatch.setenv("AUTO_GRAPHRAG", "false")

    db = MagicMock(spec=DatabaseManager)
    db.get_tables.return_value = ["customers", "orders"]
    db.has_engine.return_value = True
    db.prefetch_schema_constraints.return_value = None
    db.analyze_table.side_effect = lambda name, schema=None: TableInfo(
        name=name,
        schema=SCHEMA,
        columns=[
            ColumnInfo(
                name=f"{name}_id",
                data_type="INTEGER",
                is_nullable=False,
                is_primary_key=True,
                is_foreign_key=False,
            )
        ],
        primary_keys=[f"{name}_id"],
        foreign_keys=[],
        row_count=1,
    )

    ctx = Mock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()

    session = Mock()
    session.connection_id = CONN
    session.get_cached_schema.return_value = None
    session.schema_file = None
    session.ontology_file = None
    session.graphrag_initialized = False

    with (
        patch("src.main.get_session_db_manager", return_value=db),
        patch("src.main.get_session_data", return_value=session),
        patch("src.handlers.schema.OUTPUT_DIR", tmp_path),
    ):
        discover = getattr(
            main_module.discover_schema, "fn", main_module.discover_schema
        )
        await discover(ctx, schema_name=SCHEMA, lightweight=True)

    recorded = _mgr(tmp_path).get_current_version(SCHEMA)
    assert recorded is not None, "discover_schema must open a version"
    assert recorded.version == 1
    assert recorded.table_count == 2
    assert recorded.column_count == 2
    assert recorded.schema_hash


async def test_auto_generated_ontology_reaches_the_version(tmp_path, monkeypatch):
    """AUTO_ONTOLOGY=true must fill in the version discovery opened.

    This background path wrote no persisted metadata at all, so an
    auto-generated ontology left the version with no TTL file, no graph URI and
    a zero triple count -- and retention could never clean up the graph it had
    loaded into Oxigraph.
    """
    import types
    from unittest.mock import Mock, patch

    from src.database_manager import ColumnInfo, TableInfo
    from src.handlers import graphrag as graphrag_handler

    await open_schema_version(CONN, tmp_path, SCHEMA, _tables(1))

    tables = [
        TableInfo(
            name="customers",
            schema=SCHEMA,
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False,
                )
            ],
            primary_keys=["id"],
            foreign_keys=[],
            row_count=1,
        )
    ]

    store = Mock()
    store.load_ontology.return_value = 314

    session = Mock()
    session.connection_id = CONN
    session.oxigraph_store = store
    session.get_or_create_schema_state.return_value = types.SimpleNamespace(
        ontology=types.SimpleNamespace(ontology_file=None)
    )

    conn_dir = tmp_path / CONN
    conn_dir.mkdir(parents=True, exist_ok=True)
    with (
        patch("src.handlers.graphrag.OUTPUT_DIR", tmp_path),
        patch("src.handlers.graphrag.get_connection_dir", return_value=conn_dir),
        patch("src.handlers.graphrag.OXIGRAPH_AVAILABLE", True),
    ):
        await graphrag_handler._auto_generate_ontology_background(
            schema_name=SCHEMA, tables_info=tables, session=session, ctx=None
        )

    recorded = _mgr(tmp_path).get_current_version(SCHEMA)
    assert recorded.ontology_ttl_file.endswith(".ttl")
    assert recorded.ontology_triple_count == 314
    assert recorded.ontology_graph_uri


async def test_manual_rdf_store_records_the_graph(tmp_path, monkeypatch):
    """store_ontology_in_rdf is the documented auto_persist=False follow-up.

    Without recording here the version never learns which graph was loaded, so
    retention cannot delete that graph when the version ages out.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from src.handlers import rdf as rdf_handler

    await open_schema_version(CONN, tmp_path, SCHEMA, _tables(1))

    conn_dir = tmp_path / CONN
    conn_dir.mkdir(parents=True, exist_ok=True)
    (conn_dir / "ontology_public.ttl").write_text("@prefix x: <urn:x> .")

    store = Mock()
    store.load_ontology.return_value = 271

    ctx = Mock()
    ctx.info = AsyncMock()

    session = Mock()
    session.connection_id = CONN
    session.ontology_file = "ontology_public.ttl"
    session.get_last_analyzed_schema.return_value = SCHEMA

    services = Mock()
    services.get_session_data.return_value = session
    services.get_oxigraph_store.return_value = store

    with (
        patch("src.handlers.rdf.OUTPUT_DIR", tmp_path),
        patch("src.handlers.rdf.get_connection_dir", return_value=conn_dir),
        patch("src.handlers.rdf.OXIGRAPH_AVAILABLE", True),
    ):
        await rdf_handler.store_ontology_in_rdf(
            ctx, schema_name=SCHEMA, graph_uri=None, services=services
        )

    recorded = _mgr(tmp_path).get_current_version(SCHEMA)
    assert recorded.ontology_triple_count == 271
    assert recorded.ontology_graph_uri, "the loaded graph URI must be recorded"
    assert recorded.ontology_ttl_file == "ontology_public.ttl"


async def test_a_second_discovery_archives_the_first(tmp_path, monkeypatch):
    """Two discoveries must leave a history, not overwrite one record."""
    await open_schema_version(CONN, tmp_path, SCHEMA, _tables(2))
    await open_schema_version(CONN, tmp_path, SCHEMA, _tables(3))

    versions = _mgr(tmp_path).get_versions(SCHEMA)
    assert [(v.version, v.status) for v in versions] == [
        (1, "archived"),
        (2, "active"),
    ]


def test_version_info_round_trips_through_json(tmp_path):
    original = VersionInfo(
        version=2,
        created_at=utc_now().isoformat(),
        schema_hash="abc",
        table_count=3,
        column_count=9,
        graphrag_vector_count=42,
        graphrag_status="active",
        ontology_graph_uri="urn:g",
        ontology_triple_count=17,
        ontology_ttl_file="o.ttl",
        ontology_status="active",
        graphrag_files=["vector_store_public_v2.json"],
    )
    from dataclasses import asdict

    assert VersionInfo.from_dict(json.loads(json.dumps(asdict(original)))) == original

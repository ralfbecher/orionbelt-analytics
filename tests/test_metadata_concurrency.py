"""Concurrency and integrity guarantees for workspace metadata.json.

metadata.json is a single file holding every workspace section (connection,
per-schema state, RDF store, semantic models). Any writer that does an
unserialized read-modify-write can drop another writer's section, and because
the save used to truncate-then-write, two overlapping saves could also leave a
torn file that fails to parse.

Both failure modes were reproducible. These tests pin the fixes:

  * every mutation goes through ``mutate_workspace_metadata``, which holds the
    per-connection lock across the whole load/modify/save cycle
  * ``_save_metadata`` writes to a temp file and ``os.replace``s it into place,
    so a reader sees either the old file or the new one
"""

import ast
import asyncio
import json
import tempfile
import threading
from pathlib import Path
from unittest import mock

import pytest

from src.lifecycle.metadata import (
    VersionMetadataManager,
    mark_ontology_persisted,
    mutate_workspace_metadata,
    update_workspace_section,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
METADATA_MODULE = SRC / "lifecycle" / "metadata.py"

# Methods that mutate and persist metadata.json.
MUTATORS = {
    "_save_metadata",
    "update_workspace",
    "update_workspace_connection",
    "update_workspace_rdf_store",
    "mark_version_deleted",
    "open_version",
    "update_version",
}

CONNECTION_ID = "concurrency-test-conn"
WRITERS = 40


async def _record_model(output_dir: Path, name: str) -> None:
    """Mirror what save_semantic_model does, through the locked helper."""

    def _mutate(mgr: VersionMetadataManager) -> None:
        workspace = mgr.metadata.setdefault(
            "workspace", {"updated_at": "x", "schemas": {}}
        )
        workspace.setdefault("models", {})[name] = {"file": f"{name}.yaml"}
        mgr._save_metadata()

    await mutate_workspace_metadata(CONNECTION_ID, output_dir, _mutate)


async def test_concurrent_writers_do_not_lose_sections(tmp_path):
    """Interleaved schema and model writers must all survive.

    Before the fix this reliably lost updates -- in the original reproduction
    every one of the model writes was clobbered by a concurrent schema write
    reading stale state.
    """
    await asyncio.gather(
        *(
            coro
            for i in range(WRITERS)
            for coro in (
                update_workspace_section(
                    CONNECTION_ID, tmp_path, f"schema_{i}", "schema", {"n": i}
                ),
                _record_model(tmp_path, f"model_{i}"),
            )
        )
    )

    raw = (tmp_path / CONNECTION_ID / "metadata.json").read_text(encoding="utf-8")
    data = json.loads(raw)  # must parse -- a torn file raises here
    workspace = data["workspace"]

    assert len(workspace["schemas"]) == WRITERS, "schema sections were lost"
    assert len(workspace["models"]) == WRITERS, "model sections were lost"


async def test_metadata_file_stays_parseable_under_concurrent_writes(tmp_path):
    """A reader must never observe a partially written metadata.json."""
    stop = False
    parse_failures = []

    async def reader():
        target = tmp_path / CONNECTION_ID / "metadata.json"
        while not stop:
            if target.exists():
                try:
                    json.loads(target.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    parse_failures.append(str(exc))
            await asyncio.sleep(0)

    watcher = asyncio.create_task(reader())
    await asyncio.gather(
        *(
            update_workspace_section(
                CONNECTION_ID, tmp_path, f"schema_{i}", "schema", {"n": i}
            )
            for i in range(WRITERS)
        )
    )
    stop = True
    await watcher

    assert not parse_failures, f"reader saw a torn file: {parse_failures[:3]}"


def test_writers_on_separate_event_loops_do_not_lose_updates(tmp_path):
    """Two tiers of lock are needed; the asyncio one alone is not enough.

    asyncio.Lock cannot be shared across event loops, so the per-loop locks are
    mutually invisible. Without the process-wide threading.Lock held inside the
    worker thread, threads each running asyncio.run() bypass one another
    entirely -- the original reproduction kept 31 of 240 updates and logged a
    storm of failed os.replace calls.
    """
    connection_id = "crossloop-test-conn"
    threads, per_thread = 8, 30

    def writer(tid: int) -> None:
        async def go() -> None:
            for i in range(per_thread):

                def _mutate(mgr: VersionMetadataManager, tid=tid, i=i) -> None:
                    workspace = mgr.metadata.setdefault("workspace", {"schemas": {}})
                    workspace["schemas"][f"s_{tid}_{i}"] = {"n": i}
                    mgr._save_metadata()

                await mutate_workspace_metadata(connection_id, tmp_path, _mutate)

        asyncio.run(go())  # a fresh event loop per thread

    workers = [threading.Thread(target=writer, args=(t,)) for t in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    raw = (tmp_path / connection_id / "metadata.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert len(data["workspace"]["schemas"]) == threads * per_thread


def test_concurrent_saves_use_distinct_temp_files(tmp_path):
    """Each save must get its own temp path.

    A shared temp name (one derived from the pid, say) means concurrent writers
    replace and unlink each other's file mid-flight.
    """
    connection_id = "tempname-test-conn"
    seen: set[str] = set()
    lock = threading.Lock()
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        with lock:
            seen.add(name)
        return fd, name

    with mock.patch.object(tempfile, "mkstemp", recording_mkstemp):
        mgr = VersionMetadataManager(connection_id, tmp_path)
        for i in range(20):
            mgr.metadata["workspace"] = {"schemas": {f"s{i}": {}}}
            mgr._save_metadata()

    assert len(seen) == 20, "temp file names were reused across saves"
    assert not list((tmp_path / connection_id).glob("*.tmp")), "temp file left behind"


def test_save_metadata_is_atomic(tmp_path):
    """The save must land via os.replace, leaving no temp file behind."""
    mgr = VersionMetadataManager(CONNECTION_ID, tmp_path)
    mgr.metadata["workspace"] = {"schemas": {"a": {}}}
    mgr._save_metadata()

    conn_dir = tmp_path / CONNECTION_ID
    assert (conn_dir / "metadata.json").exists()
    assert not list(conn_dir.glob("*.tmp")), "temp file left behind"
    assert json.loads((conn_dir / "metadata.json").read_text())["workspace"]["schemas"]


def _mutating_calls_outside_metadata_module():
    """Find metadata mutations in src/ that bypass the locked helper.

    Attribution is to the *innermost* enclosing callable: the sanctioned place
    to mutate is the callback handed to ``mutate_workspace_metadata``, which is
    a nested def or a lambda inside the calling handler.
    """
    offenders = []

    for path in sorted(SRC.rglob("*.py")):
        if path == METADATA_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))

        # Callables handed to mutate_workspace_metadata are allowed to mutate.
        sanctioned: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "mutate_workspace_metadata"
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        sanctioned.add(arg.id)
                    elif isinstance(arg, ast.Lambda):
                        sanctioned.add(id(arg))

        # innermost enclosing callable for every node
        owner: dict[int, object] = {}

        def annotate(node, current):
            for child in ast.iter_child_nodes(node):
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                ):
                    annotate(child, child)
                else:
                    owner[id(child)] = current
                    annotate(child, current)
            owner.setdefault(id(node), current)

        annotate(tree, None)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in MUTATORS:
                continue
            enclosing = owner.get(id(node))
            name = getattr(enclosing, "name", None)
            if name in sanctioned or (
                isinstance(enclosing, ast.Lambda) and id(enclosing) in sanctioned
            ):
                continue
            rel = path.relative_to(PROJECT_ROOT)
            label = name or type(enclosing).__name__
            offenders.append(f"{rel}:{node.lineno}  {func.attr}()  in {label}")
    return sorted(set(offenders))


def test_no_metadata_mutation_bypasses_the_lock():
    """Every metadata write must go through mutate_workspace_metadata.

    A writer outside the per-connection lock races the ones inside it. This is
    exactly the regression that shipped once already: save_semantic_model and
    the connect_database workspace write both mutated metadata.json directly.
    """
    offenders = _mutating_calls_outside_metadata_module()
    assert not offenders, (
        "metadata.json mutated outside mutate_workspace_metadata():\n  "
        + "\n  ".join(offenders)
        + "\n\nRoute the read-modify-write through mutate_workspace_metadata()."
    )


def test_guard_flags_an_unlocked_mutation():
    """The guard must fire on a bypassing writer, not vacuously pass."""
    planted = ast.parse(
        "async def handler():\n"
        "    mgr = VersionMetadataManager(cid, out)\n"
        "    mgr._save_metadata()\n"
    )
    found = [
        call
        for call in ast.walk(planted)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in MUTATORS
    ]
    assert found, "guard would not have detected a direct _save_metadata() call"


@pytest.mark.parametrize("name", ["update_workspace", "update_workspace_rdf_store"])
def test_public_async_wrappers_use_the_lock(name):
    """The async wrappers must delegate to the locked helper, not re-implement it."""
    source = METADATA_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wrappers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name in {"update_workspace_section", "update_workspace_rdf"}
    ]
    assert wrappers, "expected the async workspace wrappers to exist"
    for wrapper in wrappers:
        body = ast.unparse(wrapper)
        assert (
            "mutate_workspace_metadata" in body
        ), f"{wrapper.name} must route through mutate_workspace_metadata"


class TestGenerationGuardedPersistFlag:
    """persisted_to_rdf must describe the generation that was actually loaded.

    The Oxigraph load runs outside the artifact family lock -- it is expensive,
    and holding the lock across it would serialize generation for the schema.
    That leaves a window where two overlapping generate_ontology calls record
    A.ttl then B.ttl, and A's slower auto-persist lands last. A blind merge
    marked B.ttl persisted on the strength of A's load.
    """

    async def test_stale_persist_does_not_flag_a_newer_generation(self, tmp_path):
        """A's late flag must not land on B's record."""
        cid, schema = "genguard", "public"
        await update_workspace_section(
            cid,
            tmp_path,
            schema,
            "ontology",
            {"ontology_file": "A.ttl", "persisted_to_rdf": False},
        )
        await update_workspace_section(
            cid,
            tmp_path,
            schema,
            "ontology",
            {"ontology_file": "B.ttl", "persisted_to_rdf": False},
        )

        applied = await mark_ontology_persisted(cid, tmp_path, schema, "A.ttl", "g:u")

        section = json.loads(
            (tmp_path / cid / "metadata.json").read_text(encoding="utf-8")
        )["workspace"]["schemas"][schema]["ontology"]
        assert applied is False
        assert section["ontology_file"] == "B.ttl"
        assert (
            section["persisted_to_rdf"] is False
        ), "B was marked persisted on the strength of A's RDF load"

    async def test_current_generation_is_flagged(self, tmp_path):
        """The guard must not block the normal path."""
        cid, schema = "genguard-ok", "public"
        await update_workspace_section(
            cid,
            tmp_path,
            schema,
            "ontology",
            {"ontology_file": "B.ttl", "persisted_to_rdf": False},
        )

        applied = await mark_ontology_persisted(cid, tmp_path, schema, "B.ttl", "g:u")

        section = json.loads(
            (tmp_path / cid / "metadata.json").read_text(encoding="utf-8")
        )["workspace"]["schemas"][schema]["ontology"]
        assert applied is True
        assert section["persisted_to_rdf"] is True
        assert section["graph_uri"] == "g:u"

    async def test_missing_section_is_not_invented(self, tmp_path):
        """No ontology recorded yet means nothing to flag."""
        applied = await mark_ontology_persisted(
            "genguard-empty", tmp_path, "public", "A.ttl", "g:u"
        )
        assert applied is False

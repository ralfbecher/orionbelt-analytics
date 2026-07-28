"""
Version Metadata Management

Tracks versions of GraphRAG and RDF ontology data for each database connection.
Enables version history, comparison, rollback, and automatic cleanup.
Also manages workspace state for session restore across reconnections.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import tempfile
import threading
import weakref
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..utils import parse_timestamp, utc_now

logger = logging.getLogger(__name__)

# Metadata writes are serialized in two tiers, because neither lock type alone
# is sufficient.
#
# Tier 1 -- asyncio.Lock, per (event loop, connection). An asyncio.Lock binds to
# whichever loop first awaits it and raises RuntimeError if reused from another,
# and this process does run more than one loop (get_registered_tool_names in
# main.py calls asyncio.run), so these cannot be shared across loops. Their job
# is to keep a single loop from queueing many worker threads on the same file.
# The WeakKeyDictionary lets a finished loop's locks be collected.
#
# Tier 2 -- threading.Lock, per connection, process-wide, taken inside the
# worker thread. This is what actually guarantees mutual exclusion: tier 1 locks
# in different loops know nothing about each other, so without this two loops
# would read-modify-write the same metadata.json concurrently and lose updates.
_metadata_locks: "weakref.WeakKeyDictionary[Any, dict[str, asyncio.Lock]]" = (
    weakref.WeakKeyDictionary()
)
#
# Neither registry is evicted. Keys are connection fingerprints (a truncated
# sha256 of type/host/port/database/schema), so the count is bounded by the
# distinct databases this server connects to, at roughly 100 bytes each. Adding
# eviction would risk handing two writers different locks for one connection --
# reintroducing precisely the race these exist to prevent -- for a saving of
# well under a megabyte. Left deliberately.
_metadata_thread_locks: dict[str, threading.Lock] = {}
_thread_lock_registry_guard = threading.Lock()


def _get_metadata_lock(connection_id: str) -> asyncio.Lock:
    """Return the per-loop write lock for *connection_id*.

    Args:
        connection_id: Database connection fingerprint.

    Returns:
        An asyncio.Lock scoped to the current loop and connection.
    """
    loop = asyncio.get_running_loop()
    per_loop = _metadata_locks.setdefault(loop, {})
    lock = per_loop.get(connection_id)
    if lock is None:
        lock = asyncio.Lock()
        per_loop[connection_id] = lock
    return lock


def _get_metadata_thread_lock(connection_id: str) -> threading.Lock:
    """Return the process-wide write lock for *connection_id*.

    Held inside the worker thread so writers on different event loops -- whose
    asyncio locks are independent -- still serialize against each other.

    Args:
        connection_id: Database connection fingerprint.

    Returns:
        A threading.Lock shared by every writer in this process.
    """
    with _thread_lock_registry_guard:
        lock = _metadata_thread_locks.get(connection_id)
        if lock is None:
            lock = threading.Lock()
            _metadata_thread_locks[connection_id] = lock
        return lock


@dataclass
class VersionInfo:
    """Information about a specific version.

    A version is one *generation cycle* for a schema. ``discover_schema`` opens
    it, and ``generate_ontology`` and GraphRAG initialization fill in their
    halves as they complete -- so a version is partially populated for as long
    as the workflow that produces it is still running.
    """

    version: int
    created_at: str  # ISO format
    schema_hash: str
    table_count: int
    column_count: int

    # GraphRAG info
    graphrag_vector_count: int
    graphrag_status: str  # "active", "archived" or "deleted"

    # Ontology info
    ontology_graph_uri: str
    ontology_triple_count: int
    ontology_ttl_file: str
    ontology_status: str  # "active", "archived" or "deleted"

    # Changes from previous version (if any)
    changes: dict[str, Any] | None = None

    # Overall status
    status: str = "active"  # "active", "archived" or "deleted"

    # ChromaDB collection backing this version's vectors. Recorded so cleanup
    # can tell whether the collection is still referenced by a live version
    # before deleting it -- see DataCleanupManager._delete_graphrag_files.
    graphrag_collection: str = ""

    # Per-version GraphRAG snapshot files, relative to the connection dir.
    # These are what cleanup deletes; the unversioned files that load_state
    # reads are the current-generation pointer and are never candidates.
    graphrag_files: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionInfo":
        """Build a VersionInfo from a stored dict, tolerating schema drift.

        metadata.json outlives any single release: a file written by a newer
        build can carry fields this one does not know, and a workspace written
        by an older build lacks fields added since. Unknown keys are dropped and
        missing optional ones fall back to their defaults, so neither direction
        raises TypeError while loading a workspace.

        Args:
            data: One entry of a schema's ``versions`` array.

        Returns:
            The parsed version record.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _env_int(name: str) -> int | None:
    """Read a positive integer environment variable.

    Args:
        name: Environment variable name.

    Returns:
        The parsed value, or None if unset, unparseable, or below 1. Invalid
        values are logged and ignored rather than raising, so a typo in .env
        degrades to the default instead of failing server startup.
    """
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}; ignoring")
        return None
    if value < 1:
        logger.warning(f"Invalid {name}={raw!r} (must be >= 1); ignoring")
        return None
    return value


@dataclass
class RetentionPolicy:
    """Retention policy for cleanup."""

    graphrag_keep_versions: int = 3
    graphrag_max_age_days: int = 30
    ontology_keep_versions: int = 5
    ontology_max_age_days: int = 60
    min_versions: int = 2  # Always keep at least this many

    @classmethod
    def from_metadata(cls, stored: dict[str, Any] | None) -> "RetentionPolicy":
        """Build a policy from stored metadata, letting the environment win.

        Precedence is environment > the workspace's recorded policy > the
        dataclass default. The environment wins because it is how an operator
        changes retention for a server that already has workspaces on disk --
        the recorded copy is a snapshot of whatever was in force when the
        workspace was created, and would otherwise pin it forever.

        Unknown keys in *stored* are ignored: metadata.json is long-lived and a
        policy field removed in a later release must not break loading.

        Args:
            stored: The ``retention_policy`` section of metadata.json, if any.

        Returns:
            The effective retention policy.
        """
        known = {f.name for f in fields(cls)}
        values: dict[str, Any] = {k: v for k, v in (stored or {}).items() if k in known}

        for field_name, env_name in (
            ("graphrag_keep_versions", "GRAPHRAG_KEEP_VERSIONS"),
            ("graphrag_max_age_days", "GRAPHRAG_MAX_AGE_DAYS"),
            ("ontology_keep_versions", "ONTOLOGY_KEEP_VERSIONS"),
            ("ontology_max_age_days", "ONTOLOGY_MAX_AGE_DAYS"),
        ):
            from_env = _env_int(env_name)
            if from_env is not None:
                values[field_name] = from_env

        return cls(**values)


class VersionMetadataManager:
    """
    Manages version metadata for a database connection.

    Metadata is stored in: tmp/{connection_id}/metadata.json
    """

    def __init__(self, connection_id: str, output_dir: Path):
        """
        Initialize metadata manager.

        Args:
            connection_id: Database connection fingerprint
            output_dir: Base output directory (usually tmp/)
        """
        self.connection_id = connection_id
        self.connection_dir = output_dir / connection_id
        self.metadata_file = self.connection_dir / "metadata.json"

        # Ensure directory exists
        self.connection_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from disk or create new."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    metadata: dict[str, Any] = json.load(f)
                logger.debug(f"Loaded metadata for connection {self.connection_id}")
                return metadata
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                # Return fresh metadata on error
                return self._create_fresh_metadata()
        else:
            return self._create_fresh_metadata()

    def _create_fresh_metadata(self) -> dict[str, Any]:
        """Create fresh metadata structure."""
        return {
            "connection_id": self.connection_id,
            "connection": {},  # Will be filled with connection details
            "schemas": {},
            "retention_policy": asdict(RetentionPolicy()),
        }

    def _save_metadata(self) -> None:
        """Save metadata to disk atomically.

        Written to a sibling temp file and moved into place with os.replace,
        which is atomic on POSIX and Windows. A plain open(..., "w") truncates
        first, so a concurrent or interrupted write leaves a torn file -- in
        practice a short write over a longer one, whose leftover tail makes the
        JSON unparseable ("Extra data"). Readers now see either the old file or
        the new one, never a mixture.
        """
        # mkstemp gives every writer its own temp path. A shared name (e.g. one
        # derived from the pid) means concurrent writers replace and unlink each
        # other's file, which produced a storm of "No such file or directory"
        # failures and lost nearly every update.
        fd, tmp_name = tempfile.mkstemp(
            dir=self.connection_dir, prefix="metadata.json.", suffix=".tmp"
        )
        tmp_file = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.metadata, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.metadata_file)
            logger.debug(f"Saved metadata for connection {self.connection_id}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            with contextlib.suppress(OSError):
                tmp_file.unlink()

    def get_schema_metadata(self, schema_name: str) -> dict[str, Any] | None:
        """Get metadata for a specific schema."""
        schema_meta: dict[str, Any] | None = self.metadata.get("schemas", {}).get(
            schema_name
        )
        return schema_meta

    def get_current_version(self, schema_name: str) -> VersionInfo | None:
        """Get current (active) version for a schema."""
        schema_meta = self.get_schema_metadata(schema_name)
        if not schema_meta:
            return None

        versions = schema_meta.get("versions", [])
        if not versions:
            return None

        # Find active version
        for v_dict in reversed(versions):
            if v_dict.get("status") == "active":
                return VersionInfo.from_dict(v_dict)

        # Fallback to latest
        return VersionInfo.from_dict(versions[-1])

    def get_versions(self, schema_name: str) -> list[VersionInfo]:
        """All recorded versions for a schema, oldest first.

        Args:
            schema_name: Schema name.

        Returns:
            The schema's version history, empty if it has none.
        """
        schema_meta = self.get_schema_metadata(schema_name)
        if not schema_meta:
            return []
        return [VersionInfo.from_dict(v) for v in schema_meta.get("versions", [])]

    def _version_entries(self, schema_name: str) -> list[dict[str, Any]]:
        """The raw ``versions`` list for a schema, creating the path if absent.

        Returns the live list so callers can mutate it in place; the caller is
        responsible for persisting.

        Args:
            schema_name: Schema name.

        Returns:
            The mutable list of version dicts.
        """
        schemas = self.metadata.setdefault("schemas", {})
        schema_meta = schemas.setdefault(schema_name, {})
        entries: list[dict[str, Any]] = schema_meta.setdefault("versions", [])
        return entries

    def _migrate_legacy_versions(self, schema_name: str) -> bool:
        """Seed a version 1 for a workspace that predates version recording.

        Workspaces created before this feature have a populated ``workspace``
        section but no ``versions`` array, so their current generation would be
        invisible to history and -- worse -- the next ``open_version`` would
        record version 1 as though nothing had come before. Seeding from what
        the workspace already knows keeps the existing artifacts in the history
        instead of orphaning them.

        Only the fields the workspace actually recorded are carried over; the
        rest keep their zero values, which is honest about what is unknown
        rather than inventing counts.

        Args:
            schema_name: Schema name.

        Returns:
            True if a version was seeded, False if there was nothing to migrate.
        """
        if self._version_entries(schema_name):
            return False

        workspace_schema = self.get_workspace_schema(schema_name)
        if not workspace_schema:
            return False

        schema_section = workspace_schema.get("schema") or {}
        ontology_section = workspace_schema.get("ontology") or {}
        graphrag_section = workspace_schema.get("graphrag") or {}

        # Nothing worth recording -- an empty workspace shell is not a version.
        if not (schema_section or ontology_section or graphrag_section):
            return False

        created_at = (
            schema_section.get("analyzed_at")
            or ontology_section.get("generated_at")
            or (self.get_workspace() or {}).get("updated_at")
            or utc_now().isoformat()
        )

        seeded = VersionInfo(
            version=1,
            created_at=created_at,
            schema_hash=schema_section.get("schema_hash", ""),
            table_count=int(schema_section.get("table_count", 0) or 0),
            column_count=int(schema_section.get("column_count", 0) or 0),
            graphrag_vector_count=int(graphrag_section.get("vector_count", 0) or 0),
            graphrag_status="active" if graphrag_section else "archived",
            ontology_graph_uri=ontology_section.get("graph_uri", "") or "",
            ontology_triple_count=int(ontology_section.get("triple_count", 0) or 0),
            ontology_ttl_file=ontology_section.get("ontology_file", "") or "",
            ontology_status="active" if ontology_section else "archived",
            changes={"migrated_from_workspace": True},
            status="active",
            graphrag_collection=graphrag_section.get("collection", "") or "",
        )
        self._version_entries(schema_name).append(asdict(seeded))
        logger.info(
            f"Seeded version 1 for schema '{schema_name}' from existing "
            f"workspace state (connection {self.connection_id})"
        )
        return True

    def open_version(
        self,
        schema_name: str,
        schema_hash: str,
        table_count: int,
        column_count: int,
    ) -> VersionInfo:
        """Start a new version for a schema, archiving the one it supersedes.

        Called from ``discover_schema``: discovery is what defines a generation,
        and ontology generation plus GraphRAG initialization fill in their
        halves afterwards via :meth:`update_version`.

        Args:
            schema_name: Schema name.
            schema_hash: Fingerprint of the discovered structure.
            table_count: Number of tables discovered.
            column_count: Total number of columns across those tables.

        Returns:
            The newly opened version.
        """
        self._migrate_legacy_versions(schema_name)
        entries = self._version_entries(schema_name)

        previous = None
        for entry in entries:
            if entry.get("status") == "active":
                previous = entry
            # Archive every still-live record; only the new version is active.
            for key in ("status", "graphrag_status", "ontology_status"):
                if entry.get(key) == "active":
                    entry[key] = "archived"

        next_number = max((int(e.get("version", 0)) for e in entries), default=0) + 1

        changes: dict[str, Any] | None = None
        if previous is not None:
            changes = {
                "previous_version": int(previous.get("version", 0)),
                "table_count_delta": table_count
                - int(previous.get("table_count", 0) or 0),
                "schema_changed": previous.get("schema_hash", "") != schema_hash,
            }

        opened = VersionInfo(
            version=next_number,
            created_at=utc_now().isoformat(),
            schema_hash=schema_hash,
            table_count=table_count,
            column_count=column_count,
            graphrag_vector_count=0,
            graphrag_status="active",
            ontology_graph_uri="",
            ontology_triple_count=0,
            ontology_ttl_file="",
            ontology_status="active",
            changes=changes,
            status="active",
        )
        entries.append(asdict(opened))
        self._save_metadata()
        logger.debug(
            f"Opened version {next_number} for schema '{schema_name}' "
            f"(connection {self.connection_id})"
        )
        return opened

    def update_version(
        self,
        schema_name: str,
        updates: dict[str, Any],
        version: int | None = None,
    ) -> VersionInfo | None:
        """Merge *updates* into one of a schema's version records.

        Used by the producers that complete a generation after discovery opened
        it -- ontology generation and GraphRAG initialization. If the target
        version does not exist the update is dropped rather than inventing one:
        those producers can legitimately run against a schema discovered before
        this feature existed, or with recording disabled.

        Pass *version* to address a specific generation. Producers must, because
        several of them finish long after they started -- background GraphRAG
        init, AUTO_ONTOLOGY generation, an Oxigraph load -- and a second
        ``discover_schema`` for the same schema during that window opens a newer
        version. Resolving "the active version" at completion time would then
        stamp the first run's snapshots, TTL file and triple counts onto a
        generation they have nothing to do with.

        Args:
            schema_name: Schema name.
            updates: VersionInfo field names to new values. Unknown keys are
                ignored so a caller cannot silently corrupt the record shape.
            version: Version number to update; the active one if omitted.

        Returns:
            The updated version, or None if no matching version exists.
        """
        known = {f.name for f in fields(VersionInfo)} - {"version", "created_at"}
        applicable = {k: v for k, v in updates.items() if k in known}
        if not applicable:
            return None

        for entry in reversed(self._version_entries(schema_name)):
            matches = (
                entry.get("version") == version
                if version is not None
                else entry.get("status") == "active"
            )
            if not matches:
                continue
            entry.update(applicable)
            self._save_metadata()
            return VersionInfo.from_dict(entry)

        return None

    def get_versions_to_cleanup(
        self,
        schema_name: str,
        data_type: str = "all",  # "graphrag", "ontology", or "all"
    ) -> list[VersionInfo]:
        """
        Get versions that should be cleaned up based on retention policy.

        Args:
            schema_name: Schema name
            data_type: Which data type to check

        Returns:
            List of versions to delete
        """
        schema_meta = self.get_schema_metadata(schema_name)
        if not schema_meta:
            return []

        versions = [VersionInfo.from_dict(v) for v in schema_meta.get("versions", [])]
        if not versions:
            return []

        policy = self.get_retention_policy()

        # Separate logic for GraphRAG vs Ontology
        if data_type == "graphrag":
            return self._get_cleanup_list(
                versions,
                policy.graphrag_keep_versions,
                policy.graphrag_max_age_days,
                policy.min_versions,
                "graphrag_status",
            )
        elif data_type == "ontology":
            return self._get_cleanup_list(
                versions,
                policy.ontology_keep_versions,
                policy.ontology_max_age_days,
                policy.min_versions,
                "ontology_status",
            )
        else:  # "all"
            # For "all", only delete if BOTH are eligible
            graphrag_cleanup = self._get_cleanup_list(
                versions,
                policy.graphrag_keep_versions,
                policy.graphrag_max_age_days,
                policy.min_versions,
                "graphrag_status",
            )
            ontology_cleanup = self._get_cleanup_list(
                versions,
                policy.ontology_keep_versions,
                policy.ontology_max_age_days,
                policy.min_versions,
                "ontology_status",
            )

            # Intersection - only delete if both agree
            graphrag_ids = {v.version for v in graphrag_cleanup}
            ontology_ids = {v.version for v in ontology_cleanup}
            both_ids = graphrag_ids & ontology_ids

            return [v for v in versions if v.version in both_ids]

    def _get_cleanup_list(
        self,
        versions: list[VersionInfo],
        keep_count: int,
        max_age_days: int,
        min_versions: int,
        status_field: str,
    ) -> list[VersionInfo]:
        """
        Get versions to cleanup based on policy.

        Args:
            versions: All versions
            keep_count: Number of recent versions to keep
            max_age_days: Maximum age in days
            min_versions: Minimum versions to always keep
            status_field: Which status field to check

        Returns:
            List of versions to delete
        """
        # Filter to only archived versions
        archived = [v for v in versions if getattr(v, status_field) == "archived"]

        if len(archived) < min_versions:
            # Not enough versions - don't delete any
            return []

        # Sort by version number (oldest first)
        sorted_versions = sorted(archived, key=lambda v: v.version)

        # Keep latest N versions
        if len(sorted_versions) <= keep_count:
            return []

        to_check = sorted_versions[:-keep_count]  # Exclude latest N

        # Check age
        now = utc_now()
        to_delete = []

        for version in to_check:
            created = parse_timestamp(version.created_at)
            age_days = (now - created).days

            if age_days > max_age_days:
                to_delete.append(version)

        # Safety check: ensure we keep minimum versions.
        #
        # Unreachable while keep_count >= min_versions, since to_delete is drawn
        # from sorted_versions[:-keep_count] and so always leaves keep_count
        # behind. It becomes reachable now that keep_count is operator-settable
        # (GRAPHRAG_KEEP_VERSIONS=1 with min_versions=2, say).
        #
        # Spare the *newest* candidates, not the oldest: to_delete is ordered
        # oldest-first, so trimming from the front would keep the stalest
        # versions and delete the ones most likely to be worth rolling back to.
        remaining = len(sorted_versions) - len(to_delete)
        if remaining < min_versions:
            excess = min_versions - remaining
            to_delete = to_delete[:-excess] if excess < len(to_delete) else []

        return to_delete

    def mark_version_deleted(
        self, schema_name: str, version: int, data_type: str = "all"
    ) -> None:
        """
        Mark a version as deleted in metadata.

        Args:
            schema_name: Schema name
            version: Version number
            data_type: "graphrag", "ontology", or "all"
        """
        schema_meta = self.metadata["schemas"].get(schema_name)
        if not schema_meta:
            return

        for v_dict in schema_meta.get("versions", []):
            if v_dict["version"] == version:
                if data_type in ["graphrag", "all"]:
                    v_dict["graphrag_status"] = "deleted"
                if data_type in ["ontology", "all"]:
                    v_dict["ontology_status"] = "deleted"
                if data_type == "all":
                    v_dict["status"] = "deleted"
                break

        self._save_metadata()

    def get_retention_policy(self) -> RetentionPolicy:
        """Get the effective retention policy (environment overrides stored)."""
        return RetentionPolicy.from_metadata(self.metadata.get("retention_policy"))

    # --- Workspace State Management ---

    def get_workspace(self) -> dict[str, Any] | None:
        """Get the full workspace section from metadata."""
        return self.metadata.get("workspace")

    def get_workspace_schema(self, schema_name: str) -> dict[str, Any] | None:
        """Get workspace data for a specific schema."""
        workspace = self.get_workspace()
        if not workspace:
            return None
        schema_ws: dict[str, Any] | None = workspace.get("schemas", {}).get(schema_name)
        return schema_ws

    def update_workspace(
        self,
        schema_name: str,
        section: str,
        data: dict[str, Any],
    ) -> None:
        """Update a workspace section for a schema.

        Args:
            schema_name: Database schema name (e.g. "public")
            section: Section key ("schema", "ontology", "graphrag")
            data: Section data dict
        """
        if "workspace" not in self.metadata:
            self.metadata["workspace"] = {
                "updated_at": utc_now().isoformat(),
                "schemas": {},
            }

        workspace = self.metadata["workspace"]

        if schema_name not in workspace.get("schemas", {}):
            workspace.setdefault("schemas", {})[schema_name] = {}

        workspace["schemas"][schema_name][section] = data
        workspace["updated_at"] = utc_now().isoformat()

        self._save_metadata()
        logger.debug(
            f"Updated workspace.{section} for schema '{schema_name}' "
            f"(connection {self.connection_id})"
        )

    def update_workspace_connection(
        self,
        db_type: str,
        db_name: str,
    ) -> None:
        """Update connection-level workspace info.

        Args:
            db_type: Database type (e.g. "postgresql", "snowflake")
            db_name: Database name
        """
        if "workspace" not in self.metadata:
            self.metadata["workspace"] = {
                "updated_at": utc_now().isoformat(),
                "schemas": {},
            }

        workspace = self.metadata["workspace"]
        workspace["db_type"] = db_type
        workspace["db_name"] = db_name
        workspace["updated_at"] = utc_now().isoformat()

        self._save_metadata()

    def update_workspace_rdf_store(self, data: dict[str, Any]) -> None:
        """Update connection-level RDF store info.

        Args:
            data: RDF store state dict (initialized, graph_uris, etc.)
        """
        if "workspace" not in self.metadata:
            self.metadata["workspace"] = {
                "updated_at": utc_now().isoformat(),
                "schemas": {},
            }

        self.metadata["workspace"]["rdf_store"] = data
        self.metadata["workspace"]["updated_at"] = utc_now().isoformat()

        self._save_metadata()
        logger.debug(f"Updated workspace.rdf_store (connection {self.connection_id})")


async def mutate_workspace_metadata(
    connection_id: str,
    output_dir: Path,
    mutate: Callable[["VersionMetadataManager"], None],
) -> None:
    """Run a metadata.json read-modify-write serialized per connection, off-loop.

    Every writer must go through here. metadata.json is a single file holding
    all workspace sections, so two unserialized read-modify-write cycles either
    drop one another's section (last writer wins on a stale read) or interleave
    into an unparseable file. Both were reproducible before this existed.

    The mutation runs in a worker thread -- the file can be large and the loop
    must not stall -- under two locks held across the whole load/modify/save
    cycle, not just the save: the per-loop asyncio lock, and the process-wide
    threading lock that actually provides mutual exclusion between event loops.

    Scope: this serializes writers **within one process**, which is sufficient
    because two processes cannot share an OUTPUT_DIR in the first place --
    Oxigraph's store is RocksDB-backed and takes an exclusive OS lock on its
    directory, so a second server fails to open it outright ("IO error: While
    lock file"). A cross-process file lock here would guard metadata.json while
    the RDF store remained unusable, so it is deliberately not implemented.

    The save is atomic regardless (unique temp file plus os.replace), so even an
    unexpected concurrent writer cannot corrupt or tear metadata.json -- the
    worst case is a lost update from a stale read.

    Args:
        connection_id: Database connection fingerprint.
        output_dir: Base output directory (usually OUTPUT_DIR).
        mutate: Callback applied to a freshly loaded manager. It is responsible
            for persisting, which every ``update_*`` method already does.
    """

    def _apply() -> None:
        # Process-wide lock, held for the whole load/modify/save cycle. The
        # asyncio lock below only covers writers on this event loop.
        with _get_metadata_thread_lock(connection_id):
            mutate(VersionMetadataManager(connection_id, output_dir))

    async with _get_metadata_lock(connection_id):
        await asyncio.to_thread(_apply)


async def update_workspace_section(
    connection_id: str,
    output_dir: Path,
    schema_name: str,
    section: str,
    data: dict[str, Any],
) -> None:
    """Thread-safe workspace section update with per-connection locking.

    Use this from async handlers to prevent concurrent writes from
    racing on the same metadata.json file.

    Args:
        connection_id: Database connection fingerprint
        output_dir: Base output directory (usually OUTPUT_DIR)
        schema_name: Database schema name
        section: Section key ("schema", "ontology", "graphrag")
        data: Section data dict
    """

    await mutate_workspace_metadata(
        connection_id,
        output_dir,
        lambda mgr: mgr.update_workspace(schema_name, section, data),
    )


async def ontology_is_current(
    connection_id: str,
    output_dir: Path,
    schema_name: str,
    ontology_file: str,
) -> bool:
    """True if *ontology_file* is still the generation metadata records.

    Checked before loading into the RDF store, not only before flipping the
    persisted flag. load_ontology() replaces the schema's named graph, so a
    stale request that got as far as loading would overwrite a newer
    generation's triples -- and guarding only the flag left exactly that hole:
    the flag stayed honest while the graph held the wrong ontology.

    Args:
        connection_id: Database connection fingerprint.
        output_dir: Base output directory.
        schema_name: Schema to check.
        ontology_file: The generation the caller holds.

    Returns:
        True if the caller's generation is still current.
    """

    def _read() -> bool:
        mgr = VersionMetadataManager(connection_id, output_dir)
        section = (
            mgr.metadata.get("workspace", {})
            .get("schemas", {})
            .get(schema_name, {})
            .get("ontology")
        )
        # Nothing recorded yet means this caller is the first -- proceed.
        if not section or "ontology_file" not in section:
            return True
        return bool(section["ontology_file"] == ontology_file)

    async with _get_metadata_lock(connection_id):
        return await asyncio.to_thread(_read)


async def mark_ontology_persisted(
    connection_id: str,
    output_dir: Path,
    schema_name: str,
    ontology_file: str,
    graph_uri: str,
) -> bool:
    """Flag an ontology as persisted to RDF, only while it is still current.

    The RDF load happens outside the artifact family lock -- it is expensive,
    and holding the lock across it would serialize generation for the schema.
    That leaves a window: two overlapping generate_ontology calls can record
    A.ttl then B.ttl, and A's slower auto-persist can land last. A blind merge
    would then mark B.ttl as persisted on the strength of A's RDF load.

    So the check and the write happen together under the metadata lock: the
    flag is set only if the recorded ontology_file is still the generation that
    was actually loaded.

    Args:
        connection_id: Database connection fingerprint.
        output_dir: Base output directory.
        schema_name: Schema whose ontology section to update.
        ontology_file: The generation this caller persisted.
        graph_uri: Named graph it was loaded into.

    Returns:
        True if the flag was set, False if a newer generation had superseded
        this one.
    """
    applied = False

    def _mutate(mgr: VersionMetadataManager) -> None:
        nonlocal applied
        section = (
            mgr.metadata.get("workspace", {})
            .get("schemas", {})
            .get(schema_name, {})
            .get("ontology")
        )
        if not section or section.get("ontology_file") != ontology_file:
            return
        section["graph_uri"] = graph_uri
        section["persisted_to_rdf"] = True
        mgr.metadata["workspace"]["updated_at"] = utc_now().isoformat()
        mgr._save_metadata()
        applied = True

    await mutate_workspace_metadata(connection_id, output_dir, _mutate)
    return applied


def schema_fingerprint(tables: Iterable[Any]) -> tuple[str, int, int]:
    """Summarize a discovered schema as (hash, table count, column count).

    The hash covers table names and their column names, sorted, so it is stable
    across the order the driver happens to return objects in and changes only
    when the structure does. It is a change detector for version history, not a
    security primitive.

    Accepts anything table-shaped -- ``.name``, optional ``.schema``, and
    ``.columns`` whose entries have ``.name`` -- rather than importing the
    driver's TableInfo, which would point this module's dependencies the wrong
    way.

    Args:
        tables: Discovered table objects.

    Returns:
        Tuple of (schema hash, table count, total column count).
    """
    entries: list[str] = []
    column_count = 0
    table_count = 0

    for table in tables:
        table_count += 1
        columns = getattr(table, "columns", None) or []
        names = sorted(str(getattr(c, "name", c)) for c in columns)
        column_count += len(names)
        qualified = f"{getattr(table, 'schema', '') or ''}.{getattr(table, 'name', '')}"
        entries.append(f"{qualified}({','.join(names)})")

    digest = hashlib.sha256("|".join(sorted(entries)).encode()).hexdigest()[:16]
    return digest, table_count, column_count


async def open_schema_version(
    connection_id: str,
    output_dir: Path,
    schema_name: str,
    tables: Iterable[Any],
) -> int | None:
    """Open a new version for a freshly discovered schema.

    Args:
        connection_id: Database connection fingerprint.
        output_dir: Base output directory.
        schema_name: Schema that was discovered.
        tables: The discovered table objects (see :func:`schema_fingerprint`).

    Returns:
        The new version number, or None if it could not be recorded.
    """
    schema_hash, table_count, column_count = schema_fingerprint(tables)
    opened: int | None = None

    def _mutate(mgr: VersionMetadataManager) -> None:
        nonlocal opened
        opened = mgr.open_version(
            schema_name,
            schema_hash=schema_hash,
            table_count=table_count,
            column_count=column_count,
        ).version

    await mutate_workspace_metadata(connection_id, output_dir, _mutate)
    return opened


async def update_schema_version(
    connection_id: str,
    output_dir: Path,
    schema_name: str,
    updates: dict[str, Any],
    version: int | None = None,
) -> int | None:
    """Fill in part of one of a schema's versions.

    Callers that started work against a known generation should pass *version*
    -- see :meth:`VersionMetadataManager.update_version` for why resolving the
    active version at completion time is unsafe.

    Args:
        connection_id: Database connection fingerprint.
        output_dir: Base output directory.
        schema_name: Schema whose version to update.
        updates: VersionInfo field names to new values.
        version: Version number to update; the active one if omitted.

    Returns:
        The version number updated, or None if no matching version exists.
    """
    updated: int | None = None

    def _mutate(mgr: VersionMetadataManager) -> None:
        nonlocal updated
        target = mgr.update_version(schema_name, updates, version)
        updated = target.version if target else None

    await mutate_workspace_metadata(connection_id, output_dir, _mutate)
    return updated


async def get_active_version_number(
    connection_id: str,
    output_dir: Path,
    schema_name: str,
) -> int | None:
    """The schema's open version number, or None if it has no history.

    Args:
        connection_id: Database connection fingerprint.
        output_dir: Base output directory.
        schema_name: Schema to look up.

    Returns:
        The active version number, if any.
    """

    def _read() -> int | None:
        mgr = VersionMetadataManager(connection_id, output_dir)
        current = mgr.get_current_version(schema_name)
        return current.version if current else None

    async with _get_metadata_lock(connection_id):
        return await asyncio.to_thread(_read)


async def update_workspace_rdf(
    connection_id: str,
    output_dir: Path,
    data: dict[str, Any],
) -> None:
    """Thread-safe RDF store workspace update.

    Args:
        connection_id: Database connection fingerprint
        output_dir: Base output directory
        data: RDF store state dict
    """

    await mutate_workspace_metadata(
        connection_id,
        output_dir,
        lambda mgr: mgr.update_workspace_rdf_store(data),
    )

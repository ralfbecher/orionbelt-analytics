"""
Data Cleanup Functions

Implements automatic cleanup of old GraphRAG and RDF ontology versions
based on retention policies.
"""

import asyncio
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..paths import OUTPUT_DIR
from ..utils import parse_timestamp, utc_now
from .metadata import VersionInfo, VersionMetadataManager, mutate_workspace_metadata

logger = logging.getLogger(__name__)


class DataCleanupManager:
    """Manages cleanup of old GraphRAG and RDF data based on retention policies.

    Retention is per *version*: each ``discover_schema`` opens a generation that
    ontology generation and GraphRAG initialization then fill in, and this class
    deletes the artifacts of generations that have aged out of the policy. It is
    the counterpart to ``lifecycle/artifacts.py``, which prunes superseded files
    eagerly by count -- this one prunes whole recorded versions by age, and
    leaves a record of what it removed.

    Every metadata write goes through ``mutate_workspace_metadata()``, so the
    read-modify-write of metadata.json is serialized against the workspace
    writers in the handlers. Deleting files and RDF graphs is offloaded with
    ``asyncio.to_thread`` -- both touch disk and neither belongs on the loop.
    """

    def __init__(self, connection_id: str, output_dir: Path):
        """
        Initialize cleanup manager.

        Args:
            connection_id: Database connection fingerprint
            output_dir: Base output directory (usually tmp/)
        """
        self.connection_id = connection_id
        self.output_dir = output_dir
        self.connection_dir = output_dir / connection_id
        self.metadata_mgr = VersionMetadataManager(connection_id, output_dir)

    async def cleanup_graphrag(
        self, schema_name: str, dry_run: bool = True
    ) -> dict[str, Any]:
        """
        Clean up old GraphRAG data based on retention policy.

        Args:
            schema_name: Schema name
            dry_run: If True, only report what would be deleted

        Returns:
            Cleanup report
        """
        versions_to_delete = self.metadata_mgr.get_versions_to_cleanup(
            schema_name, data_type="graphrag"
        )

        if not versions_to_delete:
            return {
                "deleted": [],
                "kept_all": True,
                "reason": "All versions within retention policy",
            }

        max_age = self.metadata_mgr.get_retention_policy().graphrag_max_age_days
        live_collections = self._collections_still_in_use(schema_name)
        deleted = []
        errors = []

        for version in versions_to_delete:
            try:
                if not dry_run:
                    await asyncio.to_thread(
                        self._delete_graphrag_files,
                        schema_name,
                        version,
                        live_collections,
                    )
                    await self._mark_version_deleted(
                        schema_name, version.version, "graphrag"
                    )

                age_days = (utc_now() - parse_timestamp(version.created_at)).days

                deleted.append(
                    {
                        "version": version.version,
                        "age_days": age_days,
                        "created_at": version.created_at,
                        "files": version.graphrag_files,
                        "reason": (f"Age {age_days} days exceeds max {max_age} days"),
                    }
                )

            except Exception as e:
                logger.exception(
                    f"Failed to delete GraphRAG version {version.version}: {e}"
                )
                errors.append({"version": version.version, "error": str(e)})

        return {"deleted": deleted, "errors": errors, "dry_run": dry_run}

    async def cleanup_ontology(
        self,
        schema_name: str,
        dry_run: bool = True,
        oxigraph_store: Any | None = None,
    ) -> dict[str, Any]:
        """
        Clean up old RDF ontology data based on retention policy.

        Args:
            schema_name: Schema name
            dry_run: If True, only report what would be deleted
            oxigraph_store: OxigraphStoreManager instance for deleting graphs

        Returns:
            Cleanup report
        """
        versions_to_delete = self.metadata_mgr.get_versions_to_cleanup(
            schema_name, data_type="ontology"
        )

        if not versions_to_delete:
            return {
                "deleted": [],
                "kept_all": True,
                "reason": "All versions within retention policy",
            }

        max_age = self.metadata_mgr.get_retention_policy().ontology_max_age_days
        live_graphs = self._graph_uris_still_in_use(schema_name)
        deleted = []
        errors = []

        for version in versions_to_delete:
            try:
                if not dry_run:
                    await asyncio.to_thread(
                        self._delete_ontology_artifacts,
                        version,
                        oxigraph_store,
                        live_graphs,
                    )
                    await self._mark_version_deleted(
                        schema_name, version.version, "ontology"
                    )

                age_days = (utc_now() - parse_timestamp(version.created_at)).days

                deleted.append(
                    {
                        "version": version.version,
                        "age_days": age_days,
                        "created_at": version.created_at,
                        "graph_uri": version.ontology_graph_uri,
                        "ttl_file": version.ontology_ttl_file,
                        "reason": (f"Age {age_days} days exceeds max {max_age} days"),
                    }
                )

            except Exception as e:
                logger.exception(
                    f"Failed to delete Ontology version {version.version}: {e}"
                )
                errors.append({"version": version.version, "error": str(e)})

        return {"deleted": deleted, "errors": errors, "dry_run": dry_run}

    async def _mark_version_deleted(
        self, schema_name: str, version: int, data_type: str
    ) -> None:
        """Record a version as deleted, serialized against other metadata writers.

        Args:
            schema_name: Schema name.
            version: Version number that was deleted.
            data_type: "graphrag", "ontology" or "all".
        """

        def _mutate(mgr: VersionMetadataManager) -> None:
            mgr.mark_version_deleted(schema_name, version, data_type)
            # Adopt the state that was just persisted, inside the lock, so this
            # instance's view does not go stale mid-run. The cleanup_old_versions
            # handler reads it back through get_versions() to report the
            # remaining history, which would otherwise still show the versions
            # this run just deleted. Re-running mark_version_deleted on our own
            # manager instead would be a second, unserialized write of
            # metadata.json: the exact race this lock exists to stop.
            self.metadata_mgr.metadata = deepcopy(mgr.metadata)

        await mutate_workspace_metadata(self.connection_id, self.output_dir, _mutate)

    def _all_versions(self) -> list[tuple[str, VersionInfo]]:
        """Every recorded version on this connection, paired with its schema.

        Both ownership guards need connection-wide scope: a ChromaDB collection
        is shared across schemas by design, and a named graph URI can be shared
        whenever a caller passes an explicit ``graph_uri``. Scoping either check
        to the schema being cleaned would let it delete a resource another
        schema is still using.

        Walks the whole workspace, so callers build their survivor set once per
        cleanup run rather than per deletion candidate.

        Returns:
            (schema name, version) for every schema in the workspace.
        """
        return [
            (schema_name, version)
            for schema_name in self.metadata_mgr.metadata.get("schemas", {})
            for version in self.metadata_mgr.get_versions(schema_name)
        ]

    def _surviving_references(
        self, schema_name: str, data_type: str, attribute: str
    ) -> set[str]:
        """Resource ids held by versions that will outlive this cleanup run.

        A doomed version's resource is only safe to delete when nothing that
        survives the run points at it. Computed once up front, which makes the
        result independent of the order candidates happen to be processed in --
        the earlier per-candidate check saw already-processed versions as
        deleted and not-yet-processed ones as live, so whether a shared resource
        went depended on where in the loop it was reached.

        Args:
            schema_name: Schema being cleaned up.
            data_type: "graphrag" or "ontology" -- which retention list defines
                the doomed set.
            attribute: VersionInfo field naming the resource.

        Returns:
            Resource ids that must be preserved.
        """
        doomed = {
            v.version
            for v in self.metadata_mgr.get_versions_to_cleanup(
                schema_name, data_type=data_type
            )
        }
        status_field = f"{data_type}_status"
        return {
            getattr(version, attribute)
            for other_schema, version in self._all_versions()
            if getattr(version, attribute)
            and getattr(version, status_field) != "deleted"
            and not (other_schema == schema_name and version.version in doomed)
        }

    def _collections_still_in_use(self, schema_name: str) -> set[str]:
        """ChromaDB collections referenced by versions surviving this run.

        Args:
            schema_name: Schema being cleaned up.

        Returns:
            Collection names that must be preserved.
        """
        return self._surviving_references(
            schema_name, "graphrag", "graphrag_collection"
        )

    def _graph_uris_still_in_use(self, schema_name: str) -> set[str]:
        """Named graphs referenced by versions that are not being deleted.

        Successive generations of one schema reuse the same named graph URI
        (``schema_graph_uri`` is derived from the schema name), so deleting the
        graph because an *old* version referenced it would take the current
        ontology's triples with it. Other schemas are included too, since an
        explicit ``graph_uri`` argument can point two of them at one graph.

        Args:
            schema_name: Schema being cleaned up.

        Returns:
            Graph URIs that must be preserved.
        """
        return self._surviving_references(schema_name, "ontology", "ontology_graph_uri")

    def _delete_ontology_artifacts(
        self,
        version: VersionInfo,
        oxigraph_store: Any | None,
        live_graphs: set[str],
    ) -> None:
        """Delete one version's TTL file and, if unreferenced, its named graph.

        Args:
            version: Version being cleaned up.
            oxigraph_store: Store to delete the named graph from, if available.
            live_graphs: Graph URIs still referenced by surviving versions.
        """
        if version.ontology_ttl_file:
            ttl_path = self.connection_dir / version.ontology_ttl_file
            if ttl_path.exists():
                ttl_path.unlink()
                logger.info(f"Deleted {ttl_path}")

        graph_uri = version.ontology_graph_uri
        if not (oxigraph_store and graph_uri):
            return
        if graph_uri in live_graphs:
            logger.info(
                f"Kept named graph <{graph_uri}>: still referenced by a "
                "surviving version"
            )
            return
        # Deliberately unguarded. Swallowing a failure here left the triples in
        # Oxigraph while the caller went on to mark the version deleted --
        # destroying the only record of which graph to retry, so a transient
        # store error orphaned the data permanently. Letting it propagate makes
        # the caller record the failure and leave the version intact to retry.
        oxigraph_store.delete_graph(graph_uri)

    def _delete_graphrag_files(
        self,
        schema_name: str,
        version: VersionInfo,
        live_collections: set[str],
    ) -> None:
        """
        Delete GraphRAG files for a specific version.

        Args:
            schema_name: Schema name
            version: Version being cleaned up
            live_collections: Collections held by versions surviving this run
        """
        self._delete_chromadb_collection(version, live_collections)

        # Snapshot files recorded when the version was saved. Older workspaces
        # predate the recording, so fall back to the conventional names.
        names = version.graphrag_files or [
            f"vector_store_{schema_name}_v{version.version}.json",
            f"graph_{schema_name}_v{version.version}.json",
            f"communities_{schema_name}_v{version.version}.json",
        ]

        for name in names:
            file_path = self.connection_dir / name
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted {file_path}")

    def _delete_chromadb_collection(
        self, version: VersionInfo, live_collections: set[str]
    ) -> None:
        """Delete this version's ChromaDB collection, if nothing else holds it.

        GraphRAG is connection-scoped and accumulative by design: one collection
        holds every schema's vectors so cross-schema search and join discovery
        work. Successive versions therefore share a collection rather than each
        getting their own, and dropping it because an archived version referenced
        it would wipe the live vector store for every schema on the connection.

        So the delete is guarded by ownership -- it runs only when no surviving
        version references the same collection. Today that makes it a no-op
        whenever any schema on the connection still has a live version, which is
        the correct outcome and the reason this is a guard rather than an
        unconditional delete.

        Args:
            version: Version being cleaned up.
            live_collections: Collections held by versions surviving this run,
                computed once by the caller (see _collections_still_in_use).
        """
        collection = version.graphrag_collection
        if not collection:
            return

        if collection in live_collections:
            logger.info(
                f"Kept ChromaDB collection '{collection}': still referenced by "
                "a surviving version"
            )
            return

        try:
            import chromadb
        except ImportError:
            logger.info(f"ChromaDB not installed; skipping collection '{collection}'")
            return

        db_path = OUTPUT_DIR / "chromadb" / self.connection_id
        if not db_path.exists():
            return

        # Absence is success, not failure: a rerun after a partial cleanup finds
        # the collection already gone. Distinguishing that from a real error
        # matters because errors propagate -- swallowing them would let the
        # caller mark the version deleted and lose the record needed to retry,
        # while propagating "already absent" would make every rerun fail.
        client = chromadb.PersistentClient(path=str(db_path))
        existing = {c.name for c in client.list_collections()}
        if collection not in existing:
            logger.info(f"ChromaDB collection '{collection}' already absent")
            return

        client.delete_collection(collection)
        logger.info(f"Deleted ChromaDB collection '{collection}'")

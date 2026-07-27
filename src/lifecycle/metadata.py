"""
Version Metadata Management

Tracks versions of GraphRAG and RDF ontology data for each database connection.
Enables version history, comparison, rollback, and automatic cleanup.
Also manages workspace state for session restore across reconnections.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level lock dict for serializing concurrent metadata writes per connection
_metadata_locks: dict[str, asyncio.Lock] = {}


@dataclass
class VersionInfo:
    """Information about a specific version."""

    version: int
    created_at: str  # ISO format
    schema_hash: str
    table_count: int
    column_count: int

    # GraphRAG info
    graphrag_vector_count: int
    graphrag_status: str  # "active" or "archived"

    # Ontology info
    ontology_graph_uri: str
    ontology_triple_count: int
    ontology_ttl_file: str
    ontology_status: str  # "active" or "archived"

    # Changes from previous version (if any)
    changes: dict[str, Any] | None = None

    # Overall status
    status: str = "active"  # "active" or "archived"


@dataclass
class RetentionPolicy:
    """Retention policy for cleanup."""

    graphrag_keep_versions: int = 3
    graphrag_max_age_days: int = 30
    ontology_keep_versions: int = 5
    ontology_max_age_days: int = 60
    min_versions: int = 2  # Always keep at least this many


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
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug(f"Saved metadata for connection {self.connection_id}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

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
                return VersionInfo(**v_dict)

        # Fallback to latest
        return VersionInfo(**versions[-1])

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

        versions = [VersionInfo(**v) for v in schema_meta.get("versions", [])]
        if not versions:
            return []

        policy = RetentionPolicy(**self.metadata.get("retention_policy", {}))

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
        now = datetime.now()
        to_delete = []

        for version in to_check:
            created = datetime.fromisoformat(version.created_at)
            age_days = (now - created).days

            if age_days > max_age_days:
                to_delete.append(version)

        # Safety check: ensure we keep minimum versions
        remaining = len(sorted_versions) - len(to_delete)
        if remaining < min_versions:
            # Delete fewer to maintain minimum
            excess = min_versions - remaining
            to_delete = to_delete[excess:]

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
        """Get current retention policy."""
        return RetentionPolicy(**self.metadata.get("retention_policy", {}))

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
                "updated_at": datetime.now().isoformat(),
                "schemas": {},
            }

        workspace = self.metadata["workspace"]

        if schema_name not in workspace.get("schemas", {}):
            workspace.setdefault("schemas", {})[schema_name] = {}

        workspace["schemas"][schema_name][section] = data
        workspace["updated_at"] = datetime.now().isoformat()

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
                "updated_at": datetime.now().isoformat(),
                "schemas": {},
            }

        workspace = self.metadata["workspace"]
        workspace["db_type"] = db_type
        workspace["db_name"] = db_name
        workspace["updated_at"] = datetime.now().isoformat()

        self._save_metadata()

    def update_workspace_rdf_store(self, data: dict[str, Any]) -> None:
        """Update connection-level RDF store info.

        Args:
            data: RDF store state dict (initialized, graph_uris, etc.)
        """
        if "workspace" not in self.metadata:
            self.metadata["workspace"] = {
                "updated_at": datetime.now().isoformat(),
                "schemas": {},
            }

        self.metadata["workspace"]["rdf_store"] = data
        self.metadata["workspace"]["updated_at"] = datetime.now().isoformat()

        self._save_metadata()
        logger.debug(f"Updated workspace.rdf_store (connection {self.connection_id})")


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
    lock = _metadata_locks.setdefault(connection_id, asyncio.Lock())
    async with lock:
        mgr = VersionMetadataManager(connection_id, output_dir)
        mgr.update_workspace(schema_name, section, data)


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
    lock = _metadata_locks.setdefault(connection_id, asyncio.Lock())
    async with lock:
        mgr = VersionMetadataManager(connection_id, output_dir)
        mgr.update_workspace_rdf_store(data)

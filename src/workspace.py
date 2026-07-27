"""Workspace detection and summary helpers.

Provides utilities to detect existing workspace artifacts for a connection
and format summaries for user-facing messages.
"""

import logging
from typing import Any

from .lifecycle.metadata import VersionMetadataManager
from .paths import OUTPUT_DIR, get_connection_dir

logger = logging.getLogger(__name__)


def detect_workspace(connection_id: str) -> dict[str, Any] | None:
    """Detect an existing workspace for a connection.

    Reads metadata.json, checks the workspace section exists, and validates
    that referenced artifact files still exist on disk.

    Args:
        connection_id: Database connection fingerprint (16-char SHA256)

    Returns:
        Workspace summary dict if a valid workspace exists, None otherwise.
        The dict contains:
        - schemas: dict of schema_name -> {schema, ontology, graphrag} status
        - rdf_store: RDF store status or None
        - updated_at: last update timestamp
    """
    try:
        mgr = VersionMetadataManager(connection_id, OUTPUT_DIR)
        workspace = mgr.get_workspace()

        if not workspace:
            return None

        schemas = workspace.get("schemas", {})
        if not schemas:
            return None

        conn_dir = get_connection_dir(connection_id)
        validated: dict[str, Any] = {}

        for schema_name, schema_data in schemas.items():
            schema_status: dict[str, Any] = {}

            # Validate schema artifact
            schema_section = schema_data.get("schema", {})
            if schema_section:
                schema_file = schema_section.get("schema_file")
                if schema_file and (conn_dir / schema_file).exists():
                    schema_status["schema"] = {
                        "available": True,
                        "table_count": schema_section.get("table_count", 0),
                        "analyzed_at": schema_section.get("analyzed_at"),
                    }
                else:
                    schema_status["schema"] = {"available": False}

            # Validate ontology artifact
            ontology_section = schema_data.get("ontology", {})
            if ontology_section:
                ontology_file = ontology_section.get("ontology_file")
                if ontology_file and (conn_dir / ontology_file).exists():
                    schema_status["ontology"] = {
                        "available": True,
                        "enriched": ontology_section.get("enriched", False),
                        "persisted_to_rdf": ontology_section.get(
                            "persisted_to_rdf", False
                        ),
                        "generated_at": ontology_section.get("generated_at"),
                    }
                else:
                    schema_status["ontology"] = {"available": False}

            # Validate GraphRAG (ChromaDB reconnects implicitly, just check metadata)
            graphrag_section = schema_data.get("graphrag", {})
            if graphrag_section:
                schema_status["graphrag"] = {
                    "available": graphrag_section.get("initialized", False),
                    "table_count": graphrag_section.get("table_count", 0),
                    "embedding_count": graphrag_section.get("embedding_count", 0),
                    "initialized_at": graphrag_section.get("initialized_at"),
                }

            # Only include schema if it has at least one available artifact
            if any(
                v.get("available", False)
                for v in schema_status.values()
                if isinstance(v, dict)
            ):
                validated[schema_name] = schema_status

        if not validated:
            return None

        result: dict[str, Any] = {
            "schemas": validated,
            "updated_at": workspace.get("updated_at"),
            "db_type": workspace.get("db_type"),
            "db_name": workspace.get("db_name"),
        }

        # RDF store status
        rdf_store = workspace.get("rdf_store")
        if rdf_store and rdf_store.get("initialized"):
            result["rdf_store"] = rdf_store

        return result

    except Exception as e:
        logger.debug(f"Failed to detect workspace for {connection_id}: {e}")
        return None


def format_workspace_summary(workspace: dict[str, Any]) -> str:
    """Format workspace detection result for user display.

    Args:
        workspace: Result from detect_workspace()

    Returns:
        Formatted multi-line string for inclusion in connect_database response.
    """
    lines = []
    db_info = ""
    if workspace.get("db_type"):
        db_info = f" ({workspace['db_type']}"
        if workspace.get("db_name"):
            db_info += f": {workspace['db_name']}"
        db_info += ")"

    lines.append(f"Previous workspace found{db_info}:")

    for schema_name, schema_data in workspace.get("schemas", {}).items():
        lines.append(f"  Schema '{schema_name}':")

        schema_info = schema_data.get("schema", {})
        if schema_info.get("available"):
            lines.append(
                f"    - Schema analysis: {schema_info.get('table_count', '?')} tables"
            )
        else:
            lines.append("    - Schema analysis: not available")

        ontology_info = schema_data.get("ontology", {})
        if ontology_info.get("available"):
            enriched = " (enriched)" if ontology_info.get("enriched") else ""
            rdf = " + RDF store" if ontology_info.get("persisted_to_rdf") else ""
            lines.append(f"    - Ontology: available{enriched}{rdf}")
        else:
            lines.append("    - Ontology: not available")

        graphrag_info = schema_data.get("graphrag", {})
        if graphrag_info.get("available"):
            lines.append(
                f"    - GraphRAG: {graphrag_info.get('embedding_count', '?')} embeddings"
            )
        else:
            lines.append("    - GraphRAG: not available")

    if workspace.get("rdf_store", {}).get("initialized"):
        graph_count = len(workspace["rdf_store"].get("graph_uris", []))
        lines.append(f"  RDF store: {graph_count} graph(s)")

    lines.append("")
    lines.append(
        "NOTE: Auto-restore was not available. Workspace artifacts exist on disk."
    )
    lines.append(
        "Call discover_schema() to re-analyze, or reconnect to trigger auto-restore."
    )

    return "\n".join(lines)

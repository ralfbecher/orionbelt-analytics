"""Artifact download handlers: ontology and R2RML as Turtle files."""

import logging
import re
from typing import Any

from fastmcp import Context

from ..handler_context import HandlerContext
from ..oxigraph_store import OXIGRAPH_AVAILABLE, schema_graph_uri
from ..paths import ensure_output_dir, get_connection_dir
from ..utils import read_text_file, write_text_file

logger = logging.getLogger(__name__)


async def download_ontology(
    ctx: Context,
    schema_name: str | None,
    source: str,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Download ontology as TTL file from RDF store or tmp folder."""
    try:
        session = services.get_session_data(ctx)

        if not schema_name:
            schema_name = session.get_last_analyzed_schema()
            if not schema_name:
                return {
                    "success": False,
                    "error": "No schema_name provided and no schema in session",
                    "error_type": "parameter_error",
                    "hint": "Provide schema_name parameter or generate/load an ontology first",
                }

        schema_safe = schema_name.replace(" ", "_").replace(".", "_")
        conn_dir = (
            get_connection_dir(session.connection_id)
            if session.connection_id
            else ensure_output_dir()
        )

        if source == "rdf" and OXIGRAPH_AVAILABLE:
            store = services.get_oxigraph_store(ctx)
            if not store:
                return {
                    "success": False,
                    "error": "Oxigraph RDF store not initialized",
                    "error_type": "rdf_error",
                    "hint": "Call store_ontology_in_rdf first or use source='file'",
                }

            graph_uri = schema_graph_uri(schema_name)

            try:
                ontology_ttl = store.export_graph(graph_uri, format="turtle")

                if not ontology_ttl or len(ontology_ttl) < 100:
                    return {
                        "success": False,
                        "error": f"Graph <{graph_uri}> is empty or not found in RDF store",
                        "error_type": "rdf_error",
                        "hint": f"Call store_ontology_in_rdf(schema_name='{schema_name}') first",
                    }

                file_name = f"ontology_{schema_safe}_export.ttl"
                file_path = conn_dir / file_name

                await write_text_file(file_path, ontology_ttl)

                triple_count = len(
                    [
                        line
                        for line in ontology_ttl.split("\n")
                        if line.strip()
                        and not line.strip().startswith("#")
                        and not line.strip().startswith("@")
                    ]
                )

                logger.info(
                    f"Exported ontology from RDF store <{graph_uri}> to {file_path}"
                )

                return {
                    "success": True,
                    "content": ontology_ttl,
                    "file_path": str(file_path),
                    "file_name": file_name,
                    "file_size": len(ontology_ttl),
                    "triple_count": triple_count,
                    "graph_uri": graph_uri,
                    "source": "rdf",
                    "note": f"Ontology exported from Oxigraph RDF store. File saved to: {file_path}",
                }
            except Exception as e:
                logger.error(f"Failed to export from RDF store: {e}")
                return {
                    "success": False,
                    "error": f"Failed to export from RDF store: {e!s}",
                    "error_type": "rdf_error",
                    "hint": "Try source='file' to read from tmp folder instead",
                }

        elif source == "file":
            # Read the *requested* schema's ontology file, not whatever schema is
            # currently active in the session — schema_name may differ from the
            # current schema (per-schema ontology state).
            schema_state = session.get_schema_state(schema_name)
            ontology_filename = (
                schema_state.ontology.ontology_file if schema_state else None
            )
            if not ontology_filename:
                pattern = f"ontology_{schema_safe}*.ttl"
                matching_files = list(conn_dir.glob(pattern))
                if matching_files:
                    matching_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                    ontology_file_path = matching_files[0]
                else:
                    return {
                        "success": False,
                        "error": f"No ontology file found for schema '{schema_name}' in connection folder",
                        "error_type": "file_not_found",
                        "hint": "Generate ontology first with generate_ontology()",
                    }
            else:
                ontology_file_path = conn_dir / ontology_filename

            if not ontology_file_path.exists():
                return {
                    "success": False,
                    "error": f"Ontology file not found: {ontology_file_path}",
                    "error_type": "file_not_found",
                }

            ontology_ttl = await read_text_file(ontology_file_path)

            file_stat = ontology_file_path.stat()
            logger.info(f"Read ontology from file: {ontology_file_path}")

            return {
                "success": True,
                "content": ontology_ttl,
                "file_path": str(ontology_file_path),
                "file_name": ontology_file_path.name,
                "file_size": file_stat.st_size,
                "source": "file",
                "note": f"Ontology read from tmp folder: {ontology_file_path}",
            }

        else:
            return {
                "success": False,
                "error": f"Invalid source: {source}. Must be 'rdf' or 'file'",
                "error_type": "parameter_error",
            }

    except Exception as e:
        logger.error(f"Error downloading ontology: {e}")
        return {
            "success": False,
            "error": f"Failed to download ontology: {e!s}",
            "error_type": "internal_error",
        }


async def download_r2rml(
    ctx: Context,
    schema_name: str | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Download R2RML mapping file from tmp folder."""
    try:
        session = services.get_session_data(ctx)

        if not schema_name:
            schema_name = session.get_last_analyzed_schema()
            if not schema_name:
                return {
                    "success": False,
                    "error": "No schema_name provided and no schema in session",
                    "error_type": "parameter_error",
                    "hint": "Provide schema_name parameter or run discover_schema() first",
                }

        schema_safe = schema_name.replace(" ", "_").replace(".", "_")
        conn_dir = (
            get_connection_dir(session.connection_id)
            if session.connection_id
            else ensure_output_dir()
        )

        # Read the *requested* schema's R2RML file, not the current schema's
        # (schema_name may differ from the active schema).
        schema_state = session.get_schema_state(schema_name)
        r2rml_filename = schema_state.ontology.r2rml_file if schema_state else None
        if not r2rml_filename:
            pattern = f"r2rml_{schema_safe}*.ttl"
            matching_files = list(conn_dir.glob(pattern))
            if matching_files:
                matching_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                r2rml_file_path = matching_files[0]
            else:
                return {
                    "success": False,
                    "error": f"No R2RML file found for schema '{schema_name}' in connection folder",
                    "error_type": "file_not_found",
                    "hint": "Run discover_schema() first to generate R2RML mapping",
                }
        else:
            r2rml_file_path = conn_dir / r2rml_filename

        if not r2rml_file_path.exists():
            return {
                "success": False,
                "error": f"R2RML file not found: {r2rml_file_path}",
                "error_type": "file_not_found",
                "hint": "Run discover_schema() to generate R2RML mapping",
            }

        r2rml_content = await read_text_file(r2rml_file_path)

        file_stat = r2rml_file_path.stat()

        base_iri = "http://example.com/r2rml/"
        if "rr:baseIRI" in r2rml_content:
            match = re.search(r'rr:baseIRI\s+"([^"]+)"', r2rml_content)
            if match:
                base_iri = match.group(1)

        logger.info(f"Read R2RML mapping from file: {r2rml_file_path}")

        return {
            "success": True,
            "content": r2rml_content,
            "file_path": str(r2rml_file_path),
            "file_name": r2rml_file_path.name,
            "file_size": file_stat.st_size,
            "base_iri": base_iri,
            "schema_name": schema_name,
            "note": f"R2RML mapping read from tmp folder: {r2rml_file_path}",
            "usage_examples": [
                "Use with D2RQ Server: d2r-server r2rml_mapping.ttl",
                "Use with Ontop: ontop materialize -m r2rml_mapping.ttl",
                "Convert to RDF: r2rml r2rml_mapping.ttl > data.ttl",
            ],
        }

    except Exception as e:
        logger.error(f"Error downloading R2RML: {e}")
        return {
            "success": False,
            "error": f"Failed to download R2RML: {e!s}",
            "error_type": "internal_error",
        }

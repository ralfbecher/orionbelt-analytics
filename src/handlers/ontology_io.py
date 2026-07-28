"""Ontology import/load handlers: load custom .ttl ontologies."""

import asyncio
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastmcp import Context

from ..constants import OBA_NAMESPACE
from ..handler_context import HandlerContext
from ..oxigraph_store import OXIGRAPH_AVAILABLE, schema_graph_uri
from ..paths import PROJECT_ROOT
from ..utils import read_text_file, write_text_file

logger = logging.getLogger(__name__)


def _check_ontology_db_compatibility(
    graph: Any,
    ctx: Context,
    get_session_db_manager: Callable[..., Any],
    session: Any,
) -> dict[str, Any] | None:
    """Compare ontology tables/columns against the connected database.

    Returns a compatibility dict or None if no database is connected.
    """
    try:
        from rdflib import Namespace
        from rdflib.namespace import OWL, RDF

        db_manager = get_session_db_manager(ctx)
        if not db_manager.has_engine():
            return None

        oba = Namespace(OBA_NAMESPACE)
        schema_name = session.current_schema

        # Extract table names from ontology via oba:tableName annotations
        onto_tables = {}
        for subject in graph.subjects(RDF.type, OWL.Class):
            if subject == OWL.Class:
                continue
            table_name = None
            for obj in graph.objects(subject, oba.tableName):
                table_name = str(obj)
                break
            if table_name:
                onto_tables[table_name.lower()] = table_name

        if not onto_tables:
            return {
                "status": "no_oba_annotations",
                "message": "Ontology has no oba:tableName annotations — cannot verify database fit",
            }

        # Get actual database tables
        try:
            db_tables_list = db_manager.get_tables(schema_name)
            db_tables = {t.lower(): t for t in db_tables_list}
        except Exception as e:
            logger.warning(
                f"Could not fetch database tables for compatibility check: {e}"
            )
            return None

        matched = sorted(onto_tables[k] for k in onto_tables if k in db_tables)
        onto_only = sorted(onto_tables[k] for k in onto_tables if k not in db_tables)
        db_only = sorted(db_tables[k] for k in db_tables if k not in onto_tables)

        total_onto = len(onto_tables)
        match_count = len(matched)

        pct = 0 if total_onto == 0 else round(100 * match_count / total_onto)

        if pct == 100:
            status = "full_match"
            message = f"All {total_onto} ontology tables found in database"
        elif pct >= 50:
            status = "partial_match"
            message = (
                f"{match_count}/{total_onto} ontology tables found in database "
                f"({pct}%); {len(onto_only)} not in database"
            )
        elif match_count > 0:
            status = "low_match"
            message = (
                f"Only {match_count}/{total_onto} ontology tables found in database "
                f"({pct}%) — this ontology may not fit the connected database"
            )
        else:
            status = "no_match"
            message = (
                "No ontology tables found in the connected database — "
                "this ontology does not match the current connection"
            )

        result = {
            "status": status,
            "match_percentage": pct,
            "message": message,
            "matched_tables": match_count,
            "total_ontology_tables": total_onto,
            "total_database_tables": len(db_tables),
        }

        if onto_only:
            result["tables_not_in_database"] = onto_only[:20]
        if db_only:
            result["tables_not_in_ontology"] = db_only[:20]

        logger.info(f"Ontology-DB compatibility: {status} ({pct}%)")
        return result

    except Exception as e:
        logger.warning(f"Ontology-DB compatibility check failed: {e}")
        return None


async def load_my_ontology(
    ctx: Context,
    import_folder: str,
    auto_persist: bool,
    graph_uri: str | None,
    services: "HandlerContext",
    ontology_content: str | None = None,
    file_name: str | None = None,
) -> dict[str, Any]:
    """Load an ontology from inline content or the newest .ttl file from the import folder."""
    try:
        from rdflib import Graph
        from rdflib.namespace import OWL, RDF

        newest_file = None
        ttl_files = []
        file_stat = None
        from_chat = ontology_content is not None

        if ontology_content:
            # Inline content provided (e.g. file dropped into chat)
            effective_name = file_name or "ontology_upload.ttl"
            if not effective_name.endswith(".ttl"):
                effective_name += ".ttl"

            # Save to import folder for persistence
            if import_folder.startswith("./"):
                folder_path = PROJECT_ROOT / import_folder[2:]
            elif not os.path.isabs(import_folder):
                folder_path = PROJECT_ROOT / import_folder
            else:
                folder_path = Path(import_folder)

            folder_path.mkdir(parents=True, exist_ok=True)
            saved_path = folder_path / effective_name
            await write_text_file(saved_path, ontology_content)

            newest_file = saved_path
            file_stat = saved_path.stat()
            ttl_files = [saved_path]
            logger.info(f"Saved inline ontology to: {saved_path}")
        else:
            # Read from import folder (existing behavior)
            if import_folder.startswith("./"):
                folder_path = PROJECT_ROOT / import_folder[2:]
            elif not os.path.isabs(import_folder):
                folder_path = PROJECT_ROOT / import_folder
            else:
                folder_path = Path(import_folder)

            if not folder_path.exists():
                return {
                    "success": False,
                    "error": f"Import folder not found: {folder_path}",
                    "error_type": "folder_not_found",
                    "suggestion": "Create the folder and add .ttl files, or specify a different path",
                }

            ttl_files = list(folder_path.glob("*.ttl"))
            if not ttl_files:
                return {
                    "success": False,
                    "error": f"No .ttl files found in: {folder_path}",
                    "error_type": "no_files_found",
                    "suggestion": "Add .ttl ontology files to the import folder",
                }

            ttl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            newest_file = ttl_files[0]

            ontology_content = await read_text_file(newest_file)

            file_stat = newest_file.stat()

        graph = Graph()
        try:
            await asyncio.to_thread(graph.parse, data=ontology_content, format="turtle")
        except Exception as parse_error:
            return {
                "success": False,
                "error": f"Failed to parse ontology file: {parse_error!s}",
                "error_type": "parse_error",
                "file_path": str(newest_file),
                "suggestion": "Ensure the file is valid Turtle format",
            }

        classes_count = len(list(graph.subjects(RDF.type, OWL.Class)))
        datatype_props = len(list(graph.subjects(RDF.type, OWL.DatatypeProperty)))
        object_props = len(list(graph.subjects(RDF.type, OWL.ObjectProperty)))

        modified_time = datetime.fromtimestamp(file_stat.st_mtime, tz=UTC).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        session = services.get_session_data(ctx)
        session.loaded_ontology = ontology_content
        session.loaded_ontology_path = str(newest_file)
        session.obqc_validator = None

        logger.info(f"Loaded ontology from: {newest_file}")
        logger.info(
            f"Ontology contains: {classes_count} classes, {datatype_props} data properties, {object_props} object properties"
        )

        # Check compatibility with connected database
        compatibility = None
        if services.provides("get_session_db_manager"):
            compatibility = _check_ontology_db_compatibility(
                graph, ctx, services.get_session_db_manager, session
            )

        # Auto-persist to RDF store
        stored_in_rdf = False
        triple_count = 0
        used_graph_uri = None

        if auto_persist and OXIGRAPH_AVAILABLE:
            try:
                store = services.get_oxigraph_store(ctx)
                if store:
                    schema_name = newest_file.stem.replace("ontology_", "")
                    if not graph_uri:
                        graph_uri = schema_graph_uri(schema_name)
                    used_graph_uri = graph_uri

                    triple_count = store.load_ontology(
                        ontology_content, graph_uri, schema_name
                    )
                    stored_in_rdf = True

                    logger.info(
                        f"Auto-persisted ontology to Oxigraph: {triple_count} triples in graph <{graph_uri}>"
                    )
                    await ctx.info(
                        f"Ontology loaded and stored in RDF database with {triple_count:,} triples; ready for SPARQL queries"
                    )
                else:
                    logger.warning("Oxigraph store not available for auto-persist")
                    await ctx.info(
                        f"Ontology loaded with {classes_count} classes; ready for SQL generation"
                    )
            except Exception as e:
                logger.warning(
                    f"Auto-persist to Oxigraph failed: {e}, ontology still available in session state"
                )
                await ctx.info(
                    f"Ontology loaded with {classes_count} classes; ready for SQL generation"
                )
        else:
            await ctx.info(
                f"Ontology loaded with {classes_count} classes; ready for SQL generation"
            )

        response: dict[str, Any] = {
            "success": True,
            "source": "chat_upload" if from_chat else "import_folder",
            "file_path": str(newest_file),
            "file_name": newest_file.name,
            "file_size": file_stat.st_size,
            "modified_time": modified_time,
            "classes_count": classes_count,
            "properties_count": datatype_props,
            "relationships_count": object_props,
            "total_files_found": len(ttl_files),
            "other_files": (
                [f.name for f in ttl_files[1:5]] if len(ttl_files) > 1 else []
            ),
            "stored_in_rdf": stored_in_rdf,
        }

        if stored_in_rdf:
            response["graph_uri"] = used_graph_uri
            response["triples_stored"] = triple_count
            response["next_steps"] = {
                "recommended": "query_sparql",
                "reason": "The loaded ontology is now in Oxigraph RDF database",
                "workflow": [
                    "1. load_my_ontology (completed)",
                    "2. query_sparql (explore schema with SPARQL)",
                    "3. execute_sql_query (use ontology context for SQL generation)",
                ],
            }
            response["note"] = (
                f"Ontology stored in RDF database with {triple_count:,} triples. "
                f"Token savings: ~{len(ontology_content)//4} tokens!"
            )
        else:
            preview = ontology_content[:2000]
            if len(ontology_content) > 2000:
                preview += "\n\n... [truncated, full content available in file]"
            response["ontology_preview"] = preview
            response["next_steps"] = {
                "recommended": "execute_sql_query",
                "reason": "The loaded ontology provides semantic context for SQL generation",
                "workflow": [
                    "1. load_my_ontology (completed)",
                    "2. connect_database (if not already connected)",
                    "3. execute_sql_query (use ontology context for accurate SQL)",
                ],
            }
            response["note"] = (
                "This ontology is now active and will be used instead of auto-generated ontologies"
            )

        if compatibility:
            response["compatibility"] = compatibility

        return response

    except Exception as e:
        logger.error(f"Error loading ontology: {e}")
        return {
            "success": False,
            "error": f"Failed to load ontology: {e!s}",
            "error_type": "internal_error",
        }

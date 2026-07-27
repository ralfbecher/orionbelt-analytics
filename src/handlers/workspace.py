"""Workspace restore, cleanup, and semantic model storage handler implementation."""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from fastmcp import Context

from ..database_manager import TableInfo
from ..graphrag import GraphRAGManager
from ..handler_context import HandlerContext
from ..lifecycle.metadata import VersionMetadataManager, mutate_workspace_metadata
from ..oxigraph_store import OXIGRAPH_AVAILABLE
from ..paths import (
    OUTPUT_DIR,
    ensure_output_dir,
    get_connection_dir,
    get_connection_store_dirs,
    get_models_dir,
)
from ..utils import read_json_file, read_text_file, utc_now, write_text_file

logger = logging.getLogger(__name__)


async def _restore_workspace_core(
    ctx: Context,
    session: Any,
    connection_id: str,
    schema_name: str | None,
    services: "HandlerContext",
) -> dict[str, Any] | None:
    """Core workspace restore logic shared by connect_database and cleanup recovery.

    Loads schema cache, ontology, GraphRAG, and RDF store from disk into
    the session. When schema_name is None, restores ALL schemas in the
    workspace. Per-schema state (schema cache, ontology) is restored for
    each schema; connection-scoped state (GraphRAG, RDF store) is restored
    once.

    Args:
        ctx: FastMCP context
        session: SessionData instance (already resolved)
        connection_id: Database connection fingerprint
        schema_name: Schema to restore. If None, restores all schemas.
        get_oxigraph_store: Function to get/init Oxigraph store

    Returns:
        Dict with restore results, or None if workspace is empty/missing.
    """
    conn_dir = get_connection_dir(connection_id)

    # Load workspace metadata
    mgr = VersionMetadataManager(connection_id, OUTPUT_DIR)
    workspace = mgr.get_workspace()

    if not workspace:
        return None

    schemas = workspace.get("schemas", {})
    if not schemas:
        return None

    all_schemas = list(schemas.keys())

    # Determine which schemas to restore
    if schema_name:
        schemas_to_restore = [schema_name] if schema_name in schemas else []
    else:
        schemas_to_restore = all_schemas

    if not schemas_to_restore:
        return None

    restored: list[str] = []
    failed: list[str] = []
    any_ontology_enriched = False

    # --- Per-schema restore: schema cache + ontology ---
    for sname in schemas_to_restore:
        schema_data = schemas[sname]
        session.set_current_schema(sname)

        # 1. Restore schema cache
        schema_section = schema_data.get("schema", {})
        schema_file = schema_section.get("schema_file")
        if schema_file:
            schema_path = conn_dir / schema_file
            if schema_path.exists():
                try:
                    schema_json = await read_json_file(schema_path)

                    tables_raw = schema_json.get("tables", [])
                    tables_info = [TableInfo.from_dict(t) for t in tables_raw]

                    session.cache_schema_analysis(sname, tables_info)
                    session.schema_file = schema_file
                    restored.append(f"Schema '{sname}': {len(tables_info)} tables")
                except Exception as e:
                    logger.error(f"Failed to restore schema cache for '{sname}': {e}")
                    failed.append(f"Schema cache '{sname}': {e}")
            else:
                failed.append(f"Schema file missing for '{sname}': {schema_file}")

        # 2. Restore ontology (into this schema's state)
        ontology_section = schema_data.get("ontology", {})
        ontology_file = ontology_section.get("ontology_file")
        if ontology_file:
            ontology_path = conn_dir / ontology_file
            if ontology_path.exists():
                try:
                    session.ontology_file = ontology_file
                    is_enriched = ontology_section.get("enriched", False)
                    session.ontology_enriched = is_enriched
                    if is_enriched:
                        any_ontology_enriched = True
                    enriched_tag = " (enriched)" if is_enriched else ""
                    restored.append(f"Ontology '{sname}'{enriched_tag}")

                    ontology_content = await read_text_file(ontology_path)
                    session.loaded_ontology = ontology_content
                    session.loaded_ontology_path = str(ontology_path)
                except Exception as e:
                    logger.error(f"Failed to restore ontology for '{sname}': {e}")
                    failed.append(f"Ontology '{sname}': {e}")
            else:
                failed.append(f"Ontology file missing for '{sname}': {ontology_file}")

        # Restore R2RML file reference
        r2rml_file = schema_section.get("r2rml_file")
        if r2rml_file and (conn_dir / r2rml_file).exists():
            session.r2rml_file = r2rml_file

    # --- Connection-scoped restore (once, not per-schema) ---

    # 3. Restore Oxigraph RDF store
    rdf_store = workspace.get("rdf_store", {})
    if rdf_store.get("initialized") and OXIGRAPH_AVAILABLE:
        try:
            store = services.get_oxigraph_store(ctx)
            if store:
                restored.append("RDF store (initialized)")
        except Exception as e:
            logger.error(f"Failed to restore RDF store: {e}")
            failed.append(f"RDF store: {e}")

    # 4. Restore GraphRAG (connection-scoped, accumulative)
    if not session.graphrag_initialized:
        # Find first schema with graphrag initialized to trigger load
        for sname in schemas_to_restore:
            graphrag_section = schemas[sname].get("graphrag", {})
            if graphrag_section.get("initialized"):
                try:
                    manager = GraphRAGManager(
                        embedding_model="tfidf",
                        embedding_dimension=384,
                        connection_id=connection_id,
                        schema_name=sname,
                    )
                    if await asyncio.to_thread(manager.load_state, ensure_output_dir()):
                        session.graphrag_manager = manager
                        session.graphrag_initialized = True
                        stats = manager.vector_store.get_statistics()
                        restored.append(
                            f"GraphRAG: {stats.get('total_elements', 0)} embeddings, "
                            f"{manager.graph_retriever.graph.number_of_nodes()} tables"
                        )
                    else:
                        failed.append("GraphRAG: load_state returned False")
                except Exception as e:
                    logger.error(f"Failed to restore GraphRAG: {e}")
                    failed.append(f"GraphRAG: {e}")
                break  # Connection-scoped — load once from combined state

    # Collect semantic models
    models = workspace.get("models", {})

    # Set current schema to the first restored schema
    session.set_current_schema(schemas_to_restore[0])

    return {
        "schema_name": schemas_to_restore[0],
        "all_schemas": all_schemas,
        "restored_schemas": schemas_to_restore,
        "restored": restored,
        "failed": failed,
        "ontology_enriched": any_ontology_enriched,
        "models": models,
    }


def _format_restore_summary(result: dict[str, Any]) -> str:
    """Format a restore result dict into a user-facing markdown summary.

    Args:
        result: Dict from _restore_workspace_core()

    Returns:
        Formatted markdown string
    """
    restored = result["restored"]
    failed = result["failed"]
    ontology_enriched = result.get("ontology_enriched", False)
    models = result.get("models", {})
    restored_schemas = result.get("restored_schemas", [result["schema_name"]])

    restored_str = str(restored)

    lines = ["# Workspace Auto-Restored", ""]
    if len(restored_schemas) == 1:
        lines.append(f"Schema: {restored_schemas[0]}")
    else:
        lines.append(f"Schemas: {', '.join(restored_schemas)}")
    lines.append("")

    if restored:
        lines.append("## Restored")
        lines.extend(f"- {item}" for item in restored)

    if failed:
        lines.append("")
        lines.append("## Not Restored")
        lines.extend(f"- {item}" for item in failed)
        lines.append("")
        lines.append("Use the relevant tools to regenerate missing components.")

    if not restored and not failed:
        lines.append("No artifacts found to restore.")

    # Build "do not call" list
    skip_tools = []
    if "Schema '" in restored_str or "Ontology '" in restored_str:
        skip_tools.append("discover_schema()")
    if "Ontology '" in restored_str:
        skip_tools.append("generate_ontology()")
        if ontology_enriched:
            skip_tools.append("suggest_semantic_names()")
            skip_tools.append("apply_semantic_names()")
    if skip_tools:
        lines.append("")
        lines.append("## DO NOT CALL (already restored)")
        lines.extend(f"- {tool}" for tool in skip_tools)

    lines.append("")
    lines.append("## Ready to Use")
    if "Schema '" in restored_str:
        if not ontology_enriched and "Ontology '" not in restored_str:
            lines.append("- generate_ontology() to create ontology from cached schema")
        if not ontology_enriched and "Ontology '" in restored_str:
            lines.append("- suggest_semantic_names() to enrich the ontology")
    if "Ontology '" in restored_str:
        lines.append("- query_sparql() for semantic queries")
        lines.append(
            "- execute_sql_query() for data queries (includes OBQC validation)"
        )
    if "GraphRAG" in restored_str:
        lines.append("- graphrag_search() for semantic schema search")

    if models:
        lines.append("")
        lines.append("## Semantic Models Available")
        for model_name, model_info in models.items():
            saved_at = model_info.get("saved_at", "unknown")
            model_schema = model_info.get("schema_name", "")
            lines.append(
                f"- **{model_name}** (schema: {model_schema}, saved: {saved_at})"
            )
        lines.append("")
        lines.append("Use get_semantic_model(model_name) to retrieve model YAML.")

    return "\n".join(lines)


async def cleanup_workspace(
    ctx: Context,
    services: "HandlerContext",
) -> str | dict[str, Any]:
    """Delete all workspace files for the current connection and clear session state.

    Removes schema JSON, ontology TTL, R2RML mappings, GraphRAG data,
    ChromaDB vectors, Oxigraph RDF store, semantic models, and metadata.
    The database connection itself remains active.

    Args:
        ctx: FastMCP context
        get_session_data: Function to get session data
        create_error_response: Function to create error response

    Returns:
        Summary of what was removed
    """
    session = services.get_session_data(ctx)

    if not session.connection_id:
        err: dict[str, Any] = services.create_error_response(
            "No database connection. Call connect_database first.",
            "connection_error",
        )
        return err

    connection_id = session.connection_id
    removed = []

    # 1. Close live resources before deleting their files
    if session.oxigraph_store is not None:
        try:
            session.oxigraph_store.close()
        except Exception as e:
            logger.debug(f"Oxigraph close during cleanup: {e}")

    # Drop GraphRAG reference (connection-scoped, releases ChromaDB handle)
    session.graphrag_manager = None

    # 2. Delete the workspace and the satellite stores keyed to this connection.
    # The store paths come from get_connection_store_dirs() so this list cannot
    # drift from the one the startup cleanup uses.
    labels = {"chromadb": "ChromaDB vector store", "oxigraph": "Oxigraph RDF store"}
    dirs_to_remove: list[tuple[Path, str]] = [(OUTPUT_DIR / connection_id, "workspace")]
    dirs_to_remove += [
        (store_dir, labels.get(store_dir.parent.name, store_dir.parent.name))
        for store_dir in get_connection_store_dirs(connection_id)
    ]

    for dir_path, label in dirs_to_remove:
        if not dir_path.exists():
            continue
        await asyncio.to_thread(shutil.rmtree, dir_path, ignore_errors=True)
        # rmtree(ignore_errors=True) never raises, so success cannot be inferred
        # from "it didn't throw" -- a locked or read-only tree silently survives.
        # Check, so the response does not claim a deletion that did not happen.
        if dir_path.exists():
            logger.warning(f"Failed to remove {label}: {dir_path} still present")
        else:
            removed.append(label)
            logger.info(f"Cleaned up {label}: {dir_path}")

    # 3. Clear all in-memory session state (keep connection alive)
    session.clear_schema_cache()
    session.clear_all_schema_states()
    session.graphrag_manager = None
    session.graphrag_initialized = False
    session.oxigraph_store = None
    session.oxigraph_initialized = False

    await ctx.info(f"Workspace cleaned for connection {connection_id[:8]}...")

    # 4. Build response
    result = "# Workspace Cleaned\n\n"
    result += f"Connection: {connection_id[:8]}... (still active)\n\n"

    if removed:
        result += "## Removed\n"
        for item in removed:
            result += f"- {item}\n"
    else:
        result += "No workspace files found to remove.\n"

    result += "\n## Session State Cleared\n"
    result += "- Schema cache, ontology, GraphRAG, RDF store\n\n"
    result += "Call discover_schema() to start building a new workspace.\n"

    return result


async def save_semantic_model(
    ctx: Context,
    model_yaml: str,
    model_name: str,
    schema_name: str | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Save a semantic model YAML to the workspace for reuse across sessions.

    Args:
        ctx: FastMCP context
        model_yaml: The model definition in YAML format (e.g., OBML)
        model_name: Name to identify this model
        schema_name: Database schema this model is based on
        get_session_data: Function to get session data
        create_error_response: Function to create error response

    Returns:
        Save status with file path
    """
    session = services.get_session_data(ctx)

    if not session.connection_id:
        err: dict[str, Any] = services.create_error_response(
            "No database connection. Call connect_database first.",
            "connection_error",
        )
        return err

    connection_id = session.connection_id
    effective_schema = schema_name or session.get_last_analyzed_schema() or "default"

    # Save model file. Reduce model_name to a bare filename component so it can
    # never escape models_dir (defense-in-depth; the tool boundary also rejects
    # names containing path separators).
    models_dir = get_models_dir(connection_id)
    safe_name = Path(model_name.replace(" ", "_")).name
    if not safe_name or safe_name in {".", ".."}:
        err = services.create_error_response(
            f"Invalid model_name: {model_name!r}", "validation_error"
        )
        return err
    model_filename = f"{safe_name}.yaml"
    model_path = models_dir / model_filename

    try:
        await write_text_file(model_path, model_yaml)
        logger.info(f"Saved semantic model '{model_name}' to: {model_path}")
    except Exception as e:
        logger.error(f"Failed to save semantic model: {e}")
        err = services.create_error_response(
            f"Failed to save model: {e}",
            "file_error",
        )
        return err

    # Update workspace metadata through the shared locked helper. Doing this
    # read-modify-write outside the per-connection lock races every other
    # workspace writer -- reproducibly dropping sections and tearing the file.
    def _record_model(mgr: VersionMetadataManager) -> None:
        workspace = mgr.metadata.setdefault(
            "workspace",
            {
                "updated_at": utc_now().isoformat(),
                "schemas": {},
            },
        )
        models = workspace.setdefault("models", {})
        models[model_name] = {
            "file": model_filename,
            "schema_name": effective_schema,
            "saved_at": utc_now().isoformat(),
        }
        workspace["updated_at"] = utc_now().isoformat()
        mgr._save_metadata()

    try:
        await mutate_workspace_metadata(connection_id, OUTPUT_DIR, _record_model)
    except Exception as e:
        logger.warning(f"Failed to update workspace metadata for model: {e}")

    await ctx.info(
        f"Saved semantic model '{model_name}' for schema '{effective_schema}'"
    )

    return {
        "success": True,
        "model_name": model_name,
        "schema_name": effective_schema,
        "file": model_filename,
        "message": f"Model '{model_name}' saved. Use get_semantic_model('{model_name}') to retrieve it in future sessions.",
    }


async def get_semantic_model(
    ctx: Context,
    model_name: str,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Retrieve a stored semantic model YAML by name.

    Args:
        ctx: FastMCP context
        model_name: Name of the model to retrieve
        get_session_data: Function to get session data
        create_error_response: Function to create error response

    Returns:
        Model YAML content and metadata
    """
    session = services.get_session_data(ctx)

    if not session.connection_id:
        err: dict[str, Any] = services.create_error_response(
            "No database connection. Call connect_database first.",
            "connection_error",
        )
        return err

    connection_id = session.connection_id

    # Look up model in workspace metadata
    mgr = VersionMetadataManager(connection_id, OUTPUT_DIR)
    workspace = mgr.get_workspace()

    if not workspace:
        err = services.create_error_response(
            "No workspace found for this connection.",
            "workspace_not_found",
        )
        return err

    models = workspace.get("models", {})
    model_info = models.get(model_name)

    if not model_info:
        available = list(models.keys())
        err = services.create_error_response(
            f"Model '{model_name}' not found. Available models: {available or 'none'}",
            "model_not_found",
        )
        return err

    # Read model file
    model_filename = model_info["file"]
    models_dir = get_models_dir(connection_id)
    model_path = models_dir / model_filename

    if not model_path.exists():
        err = services.create_error_response(
            f"Model file missing: {model_filename}",
            "file_not_found",
        )
        return err

    try:
        model_yaml = await read_text_file(model_path)
    except Exception as e:
        err = services.create_error_response(
            f"Failed to read model file: {e}",
            "file_error",
        )
        return err

    await ctx.info(f"Retrieved semantic model '{model_name}'")

    return {
        "success": True,
        "model_name": model_name,
        "schema_name": model_info.get("schema_name", ""),
        "saved_at": model_info.get("saved_at", ""),
        "model_yaml": model_yaml,
    }


async def list_semantic_models(
    ctx: Context,
    services: "HandlerContext",
) -> dict[str, Any]:
    """List all stored semantic models for the current connection.

    Args:
        ctx: FastMCP context
        get_session_data: Function to get session data
        create_error_response: Function to create error response

    Returns:
        List of available models with metadata
    """
    session = services.get_session_data(ctx)

    if not session.connection_id:
        err: dict[str, Any] = services.create_error_response(
            "No database connection. Call connect_database first.",
            "connection_error",
        )
        return err

    mgr = VersionMetadataManager(session.connection_id, OUTPUT_DIR)
    workspace = mgr.get_workspace()

    if not workspace:
        return {"models": [], "count": 0}

    models = workspace.get("models", {})
    model_list = [
        {
            "model_name": name,
            "schema_name": info.get("schema_name", ""),
            "saved_at": info.get("saved_at", ""),
        }
        for name, info in models.items()
    ]

    return {"models": model_list, "count": len(model_list)}

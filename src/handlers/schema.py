"""Schema analysis, table details, and cache management handler implementations."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from fastmcp import Context

from ..handler_context import HandlerContext
from ..lifecycle.artifacts import artifact_family_lock, prune_superseded_artifacts
from ..lifecycle.metadata import update_workspace_section
from ..paths import OUTPUT_DIR, ensure_output_dir, get_connection_dir
from ..r2rml_generator import R2RMLGenerator
from ..utils import utc_now, write_json_file, write_text_file

logger = logging.getLogger(__name__)


async def reset_cache(
    ctx: Context,
    cache_type: str | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Reset cached schema and/or ontology data to force re-analysis.

    Args:
        ctx: FastMCP context
        cache_type: Type of cache to reset ("schema", "ontology", "all", or None)
        get_session_data: Function to get session data
    """
    session = services.get_session_data(ctx)
    cleared = []

    cache_type_lower = (cache_type or "all").lower()

    if cache_type_lower in ("schema", "all"):
        # Clear current schema's cache only
        current = session.current_schema
        session.clear_schema_cache(current)
        session.schema_file = None
        session.r2rml_file = None
        cleared.append("schema")

    if cache_type_lower in ("ontology", "all"):
        session.ontology_file = None
        session.loaded_ontology = None
        session.obqc_validator = None
        cleared.append("ontology")

    await ctx.info(f"Cache cleared: {', '.join(cleared)}")

    return {
        "status": "success",
        "cleared_caches": cleared,
        "message": f"Cleared {', '.join(cleared)} cache(s). You can now re-run discover_schema and/or generate_ontology.",
        "next_steps": {
            "schema": "Call discover_schema() to re-analyze database schema",
            "ontology": "Call generate_ontology() to regenerate ontology",
        },
    }


async def discover_schema(
    ctx: Context,
    schema_name: str | None,
    lightweight: bool,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Analyze database schema and return table metadata with relationships.

    Args:
        ctx: FastMCP context
        schema_name: Schema to analyze
        lightweight: If True, return minimal data
        get_session_data: Function to get session data
        get_session_db_manager: Function to get session db manager
        get_session_safe_filename: Function to generate safe filenames
        _auto_initialize_graphrag_background: Background init function
    """
    # Log function entry to verify code is being called
    logger.debug(
        f"discover_schema() called - schema: '{schema_name}', lightweight: {lightweight}"
    )

    session = services.get_session_data(ctx)
    effective_schema = schema_name or ""

    # Set current schema so per-schema state (ontology, GraphRAG) is isolated
    session.set_current_schema(effective_schema or "default")

    # Early exit: workspace already fully restored — nothing to do
    if session.ontology_enriched and session.get_cached_schema(effective_schema):
        await ctx.info("Schema already discovered and ontology enriched — skipping.")
        return {
            "schema": effective_schema or "default",
            "status": "already_complete",
            "message": (
                "Schema is already discovered and the ontology is enriched. "
                "Nothing to do. Use execute_sql_query() to query data, "
                "query_sparql() for semantic queries, or graphrag_search() "
                "for schema navigation."
            ),
        }

    cached_tables = session.get_cached_schema(effective_schema)

    logger.debug(
        f"Cache check result - cached_tables: {bool(cached_tables)} (count: {len(cached_tables) if cached_tables else 0})"
    )

    if cached_tables:
        ontology_also_cached = session.ontology_file is not None

        # Auto-initialize GraphRAG even for cached results (if not already initialized)
        auto_graphrag = os.getenv("AUTO_GRAPHRAG", "true").lower()

        # Debug logging to diagnose auto-init issues
        logger.debug("GraphRAG auto-init check (CACHED path):")
        logger.debug(f"  AUTO_GRAPHRAG env: '{auto_graphrag}'")
        logger.debug(
            f"  cached_tables exists: {bool(cached_tables)} (count: {len(cached_tables) if cached_tables else 0})"
        )
        logger.info(f"  session.graphrag_initialized: {session.graphrag_initialized}")
        logger.debug(
            f"  Will trigger: {auto_graphrag == 'true' and cached_tables and not session.graphrag_initialized}"
        )

        if (
            auto_graphrag == "true"
            and cached_tables
            and not session.graphrag_initialized
        ):
            logger.info(
                f"GraphRAG auto-init triggered for cached schema: {effective_schema or 'default'}"
            )
            task = asyncio.create_task(
                services.auto_initialize_graphrag_background(
                    schema_name=effective_schema or "default",
                    tables_info=cached_tables,
                    session=session,
                    ctx=ctx,
                )
            )
            session.graphrag._init_task = task
            logger.info(
                "GraphRAG auto-initialization started in background (from cache)"
            )

        if ontology_also_cached:
            await ctx.info(
                "Schema AND ontology already cached - proceed directly to suggest_semantic_names()"
            )
            result = {
                "schema": effective_schema or "default",
                "table_count": len(cached_tables),
                "cache_hit": True,
                "ontology_cached": True,
                "message": "STOP! Both schema AND ontology are already CACHED. For enrichment, call suggest_semantic_names() directly!",
                "schema_file": session.schema_file,
                "ontology_file": session.ontology_file,
                "next_step": "suggest_semantic_names",
                "instruction": "Call suggest_semantic_names() NOW - do NOT call any other tools first!",
            }
            if auto_graphrag == "true":
                result["graphrag_auto_init"] = "started in background (from cache)"
            return result
        else:
            await ctx.info(
                f"Schema cached with {len(cached_tables)} tables - proceed to generate_ontology()"
            )
            result = {
                "schema": effective_schema or "default",
                "table_count": len(cached_tables),
                "cache_hit": True,
                "message": f"Schema already CACHED ({len(cached_tables)} tables). Call generate_ontology() next.",
                "schema_file": session.schema_file,
                "next_step": "generate_ontology",
                "instruction": "Call generate_ontology() NOW - do NOT call discover_schema again!",
            }
            if auto_graphrag == "true":
                result["graphrag_auto_init"] = "started in background (from cache)"
            return result

    db_manager = services.get_session_db_manager(ctx)
    tables = db_manager.get_tables(schema_name)

    # Prefetch PKs and FKs at schema level (Snowflake optimization)
    if schema_name:
        db_manager.prefetch_schema_constraints(schema_name)

    # LIGHTWEIGHT MODE
    if lightweight:
        logger.info(f"Analyzing schema in LIGHTWEIGHT mode - {len(tables)} tables")

        table_info_objects = []
        relationships = {}
        fan_trap_warnings = []

        for table_name in tables:
            try:
                table_info = db_manager.analyze_table(table_name, schema_name)
                if table_info:
                    table_info_objects.append(table_info)

                    if table_info.foreign_keys:
                        relationships[table_name] = table_info.foreign_keys

                        if len(table_info.foreign_keys) > 1:
                            referenced_tables = [
                                fk["referenced_table"] for fk in table_info.foreign_keys
                            ]
                            fan_trap_warnings.append(
                                {
                                    "table": table_name,
                                    "warning": f"Table {table_name} connects to multiple tables - potential fan-trap risk",
                                    "referenced_tables": referenced_tables,
                                    "recommendation": "Use separate CTEs or UNION approach for multi-fact aggregations",
                                }
                            )
            except Exception as e:
                logger.warning(f"Failed to analyze table {table_name}: {e}")

        session.cache_schema_analysis(schema_name or "", table_info_objects)
        logger.info(
            f"Cached {len(table_info_objects)} tables for generate_ontology() reuse"
        )

        lightweight_result = {
            "schema": schema_name or "default",
            "table_count": len(tables),
            "table_names": tables,
            "relationships": relationships,
            "mode": "lightweight",
            "next_step": "generate_ontology",
            "cache_hint": "Schema is now CACHED. Call generate_ontology() next - it will use cached data automatically. No need to call get_table_details — all column metadata is already cached server-side.",
        }

        if fan_trap_warnings:
            lightweight_result["fan_trap_warnings"] = fan_trap_warnings

        # Auto-initialize GraphRAG in background
        auto_graphrag = os.getenv("AUTO_GRAPHRAG", "true").lower()

        # Debug logging to diagnose auto-init issues
        logger.debug("GraphRAG auto-init check (NON-CACHED path):")
        logger.debug(f"  AUTO_GRAPHRAG env: '{auto_graphrag}'")
        logger.debug(
            f"  table_info_objects exists: {bool(table_info_objects)} (count: {len(table_info_objects) if table_info_objects else 0})"
        )
        logger.debug(
            f"  Will trigger: {auto_graphrag == 'true' and bool(table_info_objects)}"
        )

        if auto_graphrag == "true" and table_info_objects:
            logger.info(
                f"GraphRAG auto-init triggered for schema: {schema_name or 'default'}"
            )
            task = asyncio.create_task(
                services.auto_initialize_graphrag_background(
                    schema_name=schema_name or "default",
                    tables_info=table_info_objects,
                    session=session,
                    ctx=ctx,
                )
            )
            session.graphrag._init_task = task
            logger.info("GraphRAG auto-initialization started in background")
            lightweight_result["graphrag_auto_init"] = "started in background"

        await ctx.info(
            f"Lightweight schema analysis: {len(tables)} tables cached, {len(relationships)} with FKs. Next: generate_ontology()"
        )
        return lightweight_result

    # FULL MODE
    all_table_info = []
    table_info_objects = []
    for table_name in tables:
        table_info = db_manager.analyze_table(table_name, schema_name)
        if table_info:
            table_info_objects.append(table_info)
            table_dict = {
                "name": table_info.name,
                "schema": table_info.schema,
                "columns": [
                    {
                        "name": col.name,
                        "data_type": col.data_type,
                        "is_nullable": col.is_nullable,
                        "is_primary_key": col.is_primary_key,
                        "is_foreign_key": col.is_foreign_key,
                        "foreign_key_table": col.foreign_key_table,
                        "foreign_key_column": col.foreign_key_column,
                        "comment": col.comment,
                    }
                    for col in table_info.columns
                ],
                "primary_keys": table_info.primary_keys,
                "foreign_keys": table_info.foreign_keys,
                "comment": table_info.comment,
                "row_count": table_info.row_count,
            }
            all_table_info.append(table_dict)

    # Full table details saved to file only — keep response compact
    full_schema_data = {
        "schema": schema_name or "default",
        "table_count": len(all_table_info),
        "tables": all_table_info,
    }

    session = services.get_session_data(ctx)
    session.cache_schema_analysis(schema_name or "", table_info_objects)

    # Both artifact families and the metadata naming them are produced under
    # one lock. Two overlapping discover_schema calls for the same schema
    # would otherwise each write files, and the loser's prune would delete the
    # winner's just-written artifact -- or worse, metadata would end up naming
    # a file the other request already pruned.
    schema_safe_family = (schema_name or "default").replace(" ", "_").replace(".", "_")
    family_dir = (
        get_connection_dir(session.connection_id)
        if session.connection_id
        else ensure_output_dir()
    )
    async with artifact_family_lock(
        family_dir, f"schema_{session.connection_id or 'default'}_{schema_safe_family}"
    ):
        # Save schema analysis to connection-scoped output folder.
        # Paths and the previously-referenced names are captured for the prune that
        # runs only after workspace metadata durably names the new files.
        schema_filename = None
        schema_file_path: Path | None = None
        r2rml_file_path: Path | None = None
        previous_schema_file: str | None = None
        previous_r2rml_file: str | None = None
        try:
            conn_dir = (
                get_connection_dir(session.connection_id)
                if session.connection_id
                else ensure_output_dir()
            )
            schema_safe = (schema_name or "default").replace(" ", "_").replace(".", "_")
            schema_filename = (
                services.get_session_safe_filename(ctx, "schema", schema_safe) + ".json"
            )
            schema_file_path = conn_dir / schema_filename

            await write_json_file(schema_file_path, full_schema_data)

            logger.info(f"Saved schema analysis to: {schema_file_path}")
            session_data = services.get_session_data(ctx)
            previous_schema_file = session_data.schema_file
            session_data.schema_file = schema_filename
        except Exception as e:
            logger.warning(f"Failed to save schema analysis to file: {e}")

        # Full mode: return complete column details so the LLM has all metadata
        relationships = {}
        fan_trap_warnings = []
        for t in table_info_objects:
            if t.foreign_keys:
                relationships[t.name] = t.foreign_keys
                if len(t.foreign_keys) > 1:
                    fan_trap_warnings.append(
                        {
                            "table": t.name,
                            "referenced_tables": [
                                fk["referenced_table"] for fk in t.foreign_keys
                            ],
                        }
                    )

        schema_result = {
            "schema": schema_name or "default",
            "table_count": len(all_table_info),
            "tables": all_table_info,
            "relationships": relationships,
        }
        if schema_filename:
            schema_result["schema_file"] = schema_filename
        if fan_trap_warnings:
            schema_result["fan_trap_warnings"] = fan_trap_warnings

        # Generate R2RML mapping
        if table_info_objects:
            try:
                conn_dir = (
                    get_connection_dir(session.connection_id)
                    if session.connection_id
                    else ensure_output_dir()
                )
                from ..constants import DEFAULT_R2RML_BASE_IRI

                effective_schema = schema_name or "default"
                r2rml_base = os.getenv("R2RML_BASE_IRI", DEFAULT_R2RML_BASE_IRI)
                if not r2rml_base.endswith("/"):
                    r2rml_base += "/"
                base_iri = f"{r2rml_base}{effective_schema}/"

                database_name = db_manager.connection_info.get("database", "database")

                r2rml_generator = R2RMLGenerator(
                    base_iri=base_iri, database_name=database_name
                )
                r2rml_content = r2rml_generator.generate_from_schema(
                    table_info_objects, schema_name=effective_schema
                )

                schema_safe = effective_schema.replace(" ", "_").replace(".", "_")
                r2rml_filename = (
                    services.get_session_safe_filename(ctx, "r2rml", schema_safe)
                    + ".ttl"
                )
                r2rml_file_path = conn_dir / r2rml_filename

                await write_text_file(r2rml_file_path, r2rml_content)

                logger.info(f"Generated R2RML mapping: {r2rml_file_path}")
                r2rml_session = services.get_session_data(ctx)
                previous_r2rml_file = r2rml_session.r2rml_file
                r2rml_session.r2rml_file = r2rml_filename
                schema_result["r2rml_file"] = r2rml_filename
                schema_result["r2rml_base_iri"] = base_iri

                await ctx.info(
                    f"R2RML mapping generated with {len(table_info_objects)} tables"
                )
            except Exception as e:
                logger.warning(f"Failed to generate R2RML mapping: {e}")
                schema_result["r2rml_error"] = str(e)

        # Add workflow guidance
        if all_table_info:
            schema_result["next_steps"] = {
                "recommended": "generate_ontology",
                "reason": "Generate ontology with database schema linking for accurate SQL generation and fan-trap prevention",
                "workflow": [
                    "1. discover_schema (completed - schema is now CACHED)",
                    "2. generate_ontology (recommended next - will use cached schema automatically)",
                    "3. execute_sql_query (with ontology context)",
                ],
            }
            schema_result["schema_cached"] = True
            schema_result["cache_hint"] = (
                "IMPORTANT: Schema analysis is now CACHED for this session. "
                "Do NOT call discover_schema again - just call generate_ontology() directly. "
                "It will automatically use the cached schema data."
            )
            schema_result["analytical_guidance"] = (
                "Recommended next step: Run generate_ontology() - NO parameters needed!\n\n"
                "The schema is CACHED - generate_ontology will use it automatically.\n"
                "Do NOT call discover_schema again.\n\n"
                "This will create an ontology with:\n"
                "- Database schema linking (oba: namespace)\n"
                "- SQL column references for queries\n"
                "- JOIN conditions for relationships\n"
                "- Metadata for fan-trap prevention\n\n"
                "The ontology provides context for accurate SQL generation."
            )
            schema_result["next_tool"] = "generate_ontology"
            await ctx.info(
                f"Schema CACHED with {len(all_table_info)} tables. Next: generate_ontology() - no need to pass schema data, it's cached!"
            )
        else:
            await ctx.info("Schema analysis found no tables")

        # Auto-initialize GraphRAG in background (FULL MODE path)
        auto_graphrag = os.getenv("AUTO_GRAPHRAG", "true").lower()

        # Debug logging to diagnose auto-init issues
        logger.debug("GraphRAG auto-init check (FULL MODE path):")
        logger.debug(f"  AUTO_GRAPHRAG env: '{auto_graphrag}'")
        logger.debug(
            f"  table_info_objects exists: {bool(table_info_objects)} (count: {len(table_info_objects) if table_info_objects else 0})"
        )
        logger.debug(
            f"  Will trigger: {auto_graphrag == 'true' and bool(table_info_objects)}"
        )

        if auto_graphrag == "true" and table_info_objects:
            logger.info(
                f"GraphRAG auto-init triggered for schema: {schema_name or 'default'}"
            )
            task = asyncio.create_task(
                services.auto_initialize_graphrag_background(
                    schema_name=schema_name or "default",
                    tables_info=table_info_objects,
                    session=services.get_session_data(ctx),
                    ctx=ctx,
                )
            )
            session = services.get_session_data(ctx)
            session.graphrag._init_task = task
            logger.info(
                "GraphRAG auto-initialization started in background (full mode)"
            )
            schema_result["graphrag_auto_init"] = "started in background"

        # Write workspace metadata for schema section
        if session.connection_id and schema_filename:
            try:
                await update_workspace_section(
                    connection_id=session.connection_id,
                    output_dir=OUTPUT_DIR,
                    schema_name=schema_name or "default",
                    section="schema",
                    data={
                        "schema_file": schema_filename,
                        "r2rml_file": schema_result.get("r2rml_file"),
                        "table_count": len(table_info_objects),
                        "analyzed_at": utc_now().isoformat(),
                    },
                )
                # Prune only after metadata durably names the new files. Doing it
                # earlier means a crash in this gap leaves metadata pointing at an
                # artifact that has already been deleted.
                for written, previous in (
                    (schema_file_path, previous_schema_file),
                    (r2rml_file_path, previous_r2rml_file),
                ):
                    if written is not None:
                        await prune_superseded_artifacts(
                            written, protect=[previous] if previous else []
                        )
            except Exception as e:
                logger.warning(f"Failed to write workspace metadata: {e}")

    return schema_result


async def get_table_details(
    ctx: Context,
    table_name: str,
    schema_name: str | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Get detailed metadata for a single table.

    Args:
        ctx: FastMCP context
        table_name: Name of the table to analyze
        schema_name: Schema containing the table
        get_session_data: Function to get session data
        get_session_db_manager: Function to get session db manager
    """
    session = services.get_session_data(ctx)

    if not schema_name:
        schema_name = session.get_last_analyzed_schema()

    # Return from cache if schema was already discovered
    cached_tables = session.get_cached_schema(schema_name or "")
    if cached_tables:
        for t in cached_tables:
            if t.name.lower() == table_name.lower():
                await ctx.info(
                    f"Table '{table_name}' found in cache — no database call needed"
                )
                return {
                    "success": True,
                    "name": t.name,
                    "schema": t.schema,
                    "columns": [
                        {
                            "name": col.name,
                            "data_type": col.data_type,
                            "is_nullable": col.is_nullable,
                            "is_primary_key": col.is_primary_key,
                            "is_foreign_key": col.is_foreign_key,
                            "foreign_key_table": col.foreign_key_table,
                            "foreign_key_column": col.foreign_key_column,
                            "comment": col.comment,
                        }
                        for col in t.columns
                    ],
                    "primary_keys": t.primary_keys,
                    "foreign_keys": t.foreign_keys,
                    "comment": t.comment,
                    "row_count": t.row_count,
                    "cache_hit": True,
                }

    db_manager = services.get_session_db_manager(ctx)

    try:
        table_info = db_manager.analyze_table(table_name, schema_name)

        if not table_info:
            await ctx.error(
                f"Table '{table_name}' not found in schema '{schema_name or 'default'}'"
            )
            return {
                "success": False,
                "error": f"Table '{table_name}' not found",
                "table_name": table_name,
                "schema_name": schema_name or "default",
            }

        table_dict = {
            "success": True,
            "name": table_info.name,
            "schema": table_info.schema,
            "columns": [
                {
                    "name": col.name,
                    "data_type": col.data_type,
                    "is_nullable": col.is_nullable,
                    "is_primary_key": col.is_primary_key,
                    "is_foreign_key": col.is_foreign_key,
                    "foreign_key_table": col.foreign_key_table,
                    "foreign_key_column": col.foreign_key_column,
                    "comment": col.comment,
                }
                for col in table_info.columns
            ],
            "primary_keys": table_info.primary_keys,
            "foreign_keys": table_info.foreign_keys,
            "comment": table_info.comment,
            "row_count": table_info.row_count,
        }

        await ctx.info(
            f"Retrieved details for table '{table_name}': {len(table_info.columns)} columns"
        )
        return table_dict

    except Exception as e:
        logger.error(f"Failed to get table details for {table_name}: {e}")
        await ctx.error(f"Failed to analyze table '{table_name}': {e!s}")
        return {
            "success": False,
            "error": str(e),
            "table_name": table_name,
            "schema_name": schema_name or "default",
        }


async def sample_table_data(
    ctx: Context,
    table_name: str,
    schema_name: str | None,
    limit: int,
    services: "HandlerContext",
) -> list[dict[str, Any]]:
    """Sample data from a specific table for analysis.

    Args:
        ctx: FastMCP context
        table_name: Name of the table to sample
        schema_name: Schema containing the table
        limit: Maximum number of rows to return
        get_session_data: Function to get session data
        get_session_db_manager: Function to get session db manager
    """
    if not table_name:
        return [{"error": "Table name is required"}]

    if limit <= 0 or limit > 100:
        limit = 10

    if not schema_name:
        session = services.get_session_data(ctx)
        schema_name = session.get_last_analyzed_schema()

    db_manager = services.get_session_db_manager(ctx)
    sample_data: list[dict[str, Any]] = db_manager.sample_table_data(
        table_name, schema_name, limit
    )

    if sample_data and len(sample_data) > 0:
        await ctx.info(
            f"Sample data retrieved with {len(sample_data)} rows; explore data or continue with other analysis"
        )
    else:
        await ctx.info("No sample data found for table")

    return sample_data

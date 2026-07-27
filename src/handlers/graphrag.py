"""GraphRAG initialization and search handler implementations."""

import asyncio
import logging
import os
import time
from typing import Any, cast

from fastmcp import Context

from ..exceptions import ConnectionError
from ..graphrag import GraphRAGManager
from ..handler_context import HandlerContext
from ..lifecycle.artifacts import prune_superseded_artifacts
from ..lifecycle.metadata import update_workspace_section
from ..ontology_generator import OntologyGenerator
from ..oxigraph_store import OXIGRAPH_AVAILABLE
from ..paths import OUTPUT_DIR, ensure_output_dir, get_connection_dir
from ..utils import utc_now, write_text_file

logger = logging.getLogger(__name__)


def _table_info_to_dict(table_info: Any) -> dict[str, Any]:
    """Convert a TableInfo object to a dictionary for GraphRAG/ontology consumption."""
    return {
        "name": table_info.name,
        "schema": table_info.schema,
        "columns": [
            {
                "name": col.name,
                "data_type": col.data_type,
                "is_nullable": col.is_nullable,
                "is_primary_key": getattr(col, "is_primary_key", False),
                "is_foreign_key": getattr(col, "is_foreign_key", False),
                "foreign_key_table": getattr(col, "foreign_key_table", None),
                "foreign_key_column": getattr(col, "foreign_key_column", None),
                "comment": col.comment,
            }
            for col in table_info.columns
        ],
        "primary_keys": table_info.primary_keys,
        "foreign_keys": [
            {
                "column": fk["column"],
                "referenced_table": fk["referenced_table"],
                "referenced_column": fk["referenced_column"],
            }
            for fk in table_info.foreign_keys
        ],
        "comment": table_info.comment,
        "row_count": getattr(table_info, "row_count", None),
    }


async def _auto_generate_ontology_background(
    schema_name: str,
    tables_info: list[Any],
    session: Any,
    ctx: Context,
) -> None:
    """Background task: Auto-generate ontology after GraphRAG completes.

    Uses direct schema state access (not convenience properties) to avoid
    race conditions when the user switches schemas during background work.
    """
    from ..config import config_manager

    try:
        start_time = time.time()
        logger.info(f"Auto-generating ontology for schema '{schema_name}'...")

        config = config_manager.get_server_config()
        base_uri = config.ontology_base_uri

        ontology_generator = OntologyGenerator(base_uri=base_uri)
        ontology_ttl = ontology_generator.generate_from_schema(tables_info)

        conn_dir = (
            get_connection_dir(session.connection_id)
            if session.connection_id
            else ensure_output_dir()
        )

        timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
        ontology_file = conn_dir / f"ontology_{schema_name}_{timestamp}.ttl"
        await write_text_file(ontology_file, ontology_ttl)

        # Write to the specific schema's state (not current schema)
        schema_state = session.get_or_create_schema_state(schema_name)
        previous_ontology_file = schema_state.ontology.ontology_file
        schema_state.ontology.ontology_file = ontology_file.name

        if OXIGRAPH_AVAILABLE:
            try:
                # Direct store access for background task (connection-scoped)
                if session.oxigraph_store:
                    graph_uri = f"{base_uri}{schema_name}"
                    triple_count = session.oxigraph_store.load_ontology(
                        ontology_ttl, graph_uri, schema_name
                    )
                    logger.info(
                        f"Stored {triple_count} triples in RDF store (graph: {graph_uri})"
                    )
            except Exception as e:
                logger.warning(f"Failed to store in RDF: {e}")

        elapsed = time.time() - start_time
        logger.info(f"Ontology auto-generated successfully ({elapsed:.2f}s)")
        logger.info(f"Saved to: {ontology_file.name}")

        # Pruned last, and never touching what the session previously pointed
        # at: this path writes no persisted metadata, so the in-memory
        # reference is the only record that the older file is still in use.
        await prune_superseded_artifacts(
            ontology_file,
            protect=[previous_ontology_file] if previous_ontology_file else [],
        )

    except Exception as e:
        logger.error(f"Ontology auto-generation failed: {type(e).__name__}: {e}")
        logger.debug("Ontology auto-gen traceback:", exc_info=True)


async def _auto_initialize_graphrag_background(
    schema_name: str,
    tables_info: list[Any],
    session: Any,
    ctx: Context,
) -> None:
    """Background task: Auto-initialize or accumulate GraphRAG after schema analysis.

    GraphRAG is connection-scoped and accumulative. If already initialized,
    new schema tables are added to the existing graph and vector store.
    """
    try:
        start_time = time.time()
        tables_dict = [_table_info_to_dict(t) for t in tables_info]

        if session.graphrag_manager is None:
            # First schema — initialize from scratch
            logger.info(f"Initializing GraphRAG for schema '{schema_name}'...")
            session.graphrag_manager = GraphRAGManager(
                embedding_model="tfidf",
                connection_id=session.connection_id,
                schema_name=schema_name,
            )
            session.graphrag_manager.initialize_from_schema(
                tables_info=tables_dict, schema_name=schema_name
            )
        else:
            # Additional schema — accumulate into existing graph
            logger.info(
                f"Accumulating schema '{schema_name}' into existing GraphRAG..."
            )
            session.graphrag_manager.accumulate_schema(
                tables_info=tables_dict, schema_name=schema_name
            )

        output_dir = ensure_output_dir()
        await asyncio.to_thread(session.graphrag_manager.save_state, output_dir)

        elapsed = time.time() - start_time
        session.graphrag_initialized = True

        total_tables = session.graphrag_manager.graph_retriever.graph.number_of_nodes()
        schemas = session.graphrag_manager._schema_names
        logger.info(
            f"GraphRAG auto-initialized successfully ({elapsed:.2f}s) — "
            f"{total_tables} tables across schemas: {schemas}"
        )

        # Write workspace metadata for graphrag section
        if session.connection_id:
            try:
                stats = session.graphrag_manager.vector_store.get_statistics()
                await update_workspace_section(
                    connection_id=session.connection_id,
                    output_dir=OUTPUT_DIR,
                    schema_name=schema_name,
                    section="graphrag",
                    data={
                        "initialized": True,
                        "table_count": len(tables_dict),
                        "embedding_count": stats.get("total_elements", 0),
                        "schemas": schemas,
                        "initialized_at": utc_now().isoformat(),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to write workspace metadata: {e}")

        # Chain to ontology generation if enabled
        auto_ontology = os.getenv("AUTO_ONTOLOGY", "false").lower()
        if auto_ontology == "true":
            logger.info("Chaining to ontology auto-generation...")
            await _auto_generate_ontology_background(
                schema_name=schema_name,
                tables_info=tables_info,
                session=session,
                ctx=ctx,
            )

    except Exception as e:
        logger.exception(
            f"GraphRAG auto-initialization failed: {type(e).__name__}: {e}",
        )
        session.graphrag_initialized = False


async def initialize_graphrag(
    ctx: Context,
    schema_name: str | None,
    embedding_model: str,
    services: "HandlerContext",
) -> str:
    """Initialize GraphRAG for intelligent schema navigation and retrieval."""
    session = services.get_session_data(ctx)
    db_manager = services.get_session_db_manager(ctx)

    if not db_manager.has_engine():
        return cast(
            str,
            ConnectionError(
                "No database connection. Please use connect_database tool first."
            ).to_response(),
        )

    effective_schema = schema_name
    if not effective_schema:
        effective_schema = session.get_last_analyzed_schema()
        if effective_schema:
            logger.info(f"Using last analyzed schema: {effective_schema}")

    # Set current schema for per-schema state isolation
    session.set_current_schema(effective_schema or "default")

    tables_info = session.get_cached_schema(effective_schema or "")

    if not tables_info:
        try:
            tables = db_manager.get_tables(effective_schema)
            logger.info(
                f"Found {len(tables)} tables in schema '{effective_schema or 'default'}'"
            )

            if effective_schema:
                db_manager.prefetch_schema_constraints(effective_schema)

            tables_info = []
            for table_name in tables:
                try:
                    table_info = db_manager.analyze_table(table_name, effective_schema)
                    if table_info:
                        tables_info.append(table_info)
                except Exception as e:
                    logger.error(f"Failed to analyze table {table_name}: {e}")

            session.cache_schema_analysis(effective_schema or "", tables_info)

        except Exception as e:
            return cast(
                str,
                services.create_error_response(
                    f"Failed to fetch schema: {e!s}", "database_error"
                ),
            )

    if not tables_info:
        return cast(
            str,
            services.create_error_response(
                f"No tables found in schema '{effective_schema or 'default'}'",
                "data_error",
            ),
        )

    # Convert TableInfo objects to dictionaries
    tables_dict = [_table_info_to_dict(t) for t in tables_info]

    eff_schema = effective_schema or "default"

    try:
        if session.graphrag_manager is None:
            session.graphrag_manager = GraphRAGManager(
                embedding_model=embedding_model,
                embedding_dimension=384,
                connection_id=session.connection_id,
                schema_name=eff_schema,
            )
            session.graphrag_manager.initialize_from_schema(
                tables_info=tables_dict, schema_name=eff_schema
            )
        else:
            # Accumulate into existing graph
            session.graphrag_manager.accumulate_schema(
                tables_info=tables_dict, schema_name=eff_schema
            )

        session.graphrag_initialized = True

        output_dir = ensure_output_dir()
        await asyncio.to_thread(session.graphrag_manager.save_state, output_dir)

        total_tables = session.graphrag_manager.graph_retriever.graph.number_of_nodes()
        schemas = session.graphrag_manager._schema_names

        # Write workspace metadata for graphrag section
        if session.connection_id:
            try:
                stats = session.graphrag_manager.vector_store.get_statistics()
                await update_workspace_section(
                    connection_id=session.connection_id,
                    output_dir=OUTPUT_DIR,
                    schema_name=eff_schema,
                    section="graphrag",
                    data={
                        "initialized": True,
                        "table_count": len(tables_dict),
                        "embedding_count": stats.get("total_elements", 0),
                        "schemas": schemas,
                        "initialized_at": utc_now().isoformat(),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to write workspace metadata: {e}")

        await ctx.info(
            f"GraphRAG initialized for schema '{eff_schema}' with {len(tables_dict)} tables "
            f"(total: {total_tables} tables across {len(schemas)} schema(s))"
        )

        return (
            f"GraphRAG initialized successfully!\n\n"
            f"Schema: {eff_schema}\n"
            f"Tables added: {len(tables_dict)}\n"
            f"Total tables in graph: {total_tables}\n"
            f"Schemas: {', '.join(schemas)}\n"
            f"Embedding model: {embedding_model}\n\n"
            f"You can now use:\n"
            f"- graphrag_search() for semantic search across all schemas\n"
            f"- graphrag_search(overview=True) for schema statistics\n"
            f"- graphrag_query_context() for optimized query context\n"
            f"- graphrag_find_join_path() for cross-schema relationship discovery"
        )

    except Exception as e:
        logger.exception(f"GraphRAG initialization failed: {e}")
        return cast(
            str,
            services.create_error_response(
                f"GraphRAG initialization failed: {e!s}", "graphrag_error"
            ),
        )


async def graphrag_search(
    ctx: Context,
    query: str,
    top_k: int,
    element_type: str | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Search schema using natural language via GraphRAG semantic search."""
    session = services.get_session_data(ctx)

    if not session.graphrag_initialized or session.graphrag_manager is None:
        err: dict[str, Any] = services.create_error_response(
            "GraphRAG not initialized. Please call discover_schema() first.",
            "graphrag_not_initialized",
        )
        return err

    try:
        results = session.graphrag_manager.search_schema(
            query=query, top_k=top_k, element_type=element_type
        )

        await ctx.info(f"Found {len(results)} results for query: {query}")

        return {
            "success": True,
            "query": query,
            "result_count": len(results),
            "results": results,
        }

    except Exception as e:
        logger.exception(f"GraphRAG search failed: {e}")
        err = services.create_error_response(
            f"GraphRAG search failed: {e!s}", "graphrag_error"
        )
        return err


async def graphrag_query_context(
    ctx: Context,
    query: str,
    max_tables: int,
    max_columns: int,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Get optimized context for SQL query generation using GraphRAG."""
    session = services.get_session_data(ctx)

    if not session.graphrag_initialized or session.graphrag_manager is None:
        err: dict[str, Any] = services.create_error_response(
            "GraphRAG not initialized. Please call discover_schema() first.",
            "graphrag_not_initialized",
        )
        return err

    try:
        context = session.graphrag_manager.get_query_context(
            query=query, max_tables=max_tables, max_columns=max_columns
        )

        await ctx.info(
            f"Generated context: {len(context['relevant_tables'])} tables, "
            f"{len(context['relevant_columns'])} columns, "
            f"~{context['token_estimate']} tokens"
        )

        return {
            "success": True,
            "query": query,
            "context": context,
            "usage_guidance": (
                "Use this context for SQL generation. "
                "It includes only relevant schema elements, reducing token usage by 85-95%."
            ),
        }

    except Exception as e:
        logger.exception(f"GraphRAG query context failed: {e}")
        err = services.create_error_response(
            f"GraphRAG query context failed: {e!s}", "graphrag_error"
        )
        return err


async def graphrag_find_join_path(
    ctx: Context,
    from_table: str,
    to_table: str,
    max_hops: int,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Find join path between two tables using GraphRAG graph traversal."""
    session = services.get_session_data(ctx)

    if not session.graphrag_initialized or session.graphrag_manager is None:
        err: dict[str, Any] = services.create_error_response(
            "GraphRAG not initialized. Please call discover_schema() first.",
            "graphrag_not_initialized",
        )
        return err

    try:
        join_path = session.graphrag_manager.graph_retriever.find_join_path(
            from_table=from_table, to_table=to_table, max_hops=max_hops
        )

        if join_path is None:
            return {
                "success": False,
                "from": from_table,
                "to": to_table,
                "message": f"No path found between {from_table} and {to_table} within {max_hops} hops",
            }

        path = [from_table]
        for join in join_path:
            if join["to_table"] not in path:
                path.append(join["to_table"])

        await ctx.info(
            f"Found {len(join_path)}-hop path from {from_table} to {to_table}"
        )

        return {
            "success": True,
            "from": from_table,
            "to": to_table,
            "hops": len(join_path),
            "path": path,
            "joins": join_path,
        }

    except Exception as e:
        logger.exception(f"GraphRAG find join path failed: {e}")
        err = services.create_error_response(
            f"GraphRAG find join path failed: {e!s}", "graphrag_error"
        )
        return err


async def reachable_from(
    ctx: Context,
    table: str,
    max_hops: int | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Dimension-capable tables for a query anchored on ``table`` (many-to-one closure)."""
    session = services.get_session_data(ctx)

    if not session.graphrag_initialized or session.graphrag_manager is None:
        err: dict[str, Any] = services.create_error_response(
            "GraphRAG not initialized. Please call discover_schema() first.",
            "graphrag_not_initialized",
        )
        return err

    try:
        result = session.graphrag_manager.graph_retriever.reachable_from(
            table, max_hops=max_hops
        )
        if not result["exists"]:
            err = services.create_error_response(
                f"Table '{table}' not found in the schema graph.", "data_error"
            )
            return err

        await ctx.info(
            f"{len(result['tables'])} dimension-capable tables reachable from '{table}'"
        )
        return {
            "success": True,
            "table": table,
            "direction": "many_to_one",
            "capability": "dimension",
            "reachable_tables": result["tables"],
            "by_hop": result["by_hop"],
            "guidance": (
                f"These coarser-grain tables can be joined from '{table}' without "
                "row multiplication (each join is many-to-one / functional), so their "
                "columns are safe to use as dimensions (GROUP BY / filter)."
            ),
        }

    except Exception as e:
        logger.exception(f"reachable_from failed: {e}")
        err = services.create_error_response(
            f"reachable_from failed: {e!s}", "graphrag_error"
        )
        return err


async def measurable_from(
    ctx: Context,
    table: str,
    max_hops: int | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Measure-capable tables for a query anchored on ``table`` (one-to-many closure)."""
    session = services.get_session_data(ctx)

    if not session.graphrag_initialized or session.graphrag_manager is None:
        err: dict[str, Any] = services.create_error_response(
            "GraphRAG not initialized. Please call discover_schema() first.",
            "graphrag_not_initialized",
        )
        return err

    try:
        result = session.graphrag_manager.graph_retriever.measurable_from(
            table, max_hops=max_hops
        )
        if not result["exists"]:
            err = services.create_error_response(
                f"Table '{table}' not found in the schema graph.", "data_error"
            )
            return err

        await ctx.info(
            f"{len(result['tables'])} measure-capable tables for anchor '{table}'"
        )
        return {
            "success": True,
            "table": table,
            "direction": "one_to_many",
            "capability": "measure",
            "measurable_tables": result["tables"],
            "by_hop": result["by_hop"],
            "guidance": (
                f"These finer-grain tables fan out '{table}' (one-to-many), so their "
                "values must be aggregated into measures (SUM/COUNT/...) and must NOT "
                f"be used as dimensions at the grain of '{table}' — doing so is a fan-trap."
            ),
        }

    except Exception as e:
        logger.exception(f"measurable_from failed: {e}")
        err = services.create_error_response(
            f"measurable_from failed: {e!s}", "graphrag_error"
        )
        return err


async def plan_composite_query(
    ctx: Context,
    facts: list[str],
    dimensions: list[str] | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Advise a Composite Fact Layer (CFL) decomposition for a multi-fact query.

    Detects whether the requested facts are independent grains (disjoint
    siblings) that require a UNION ALL composite, and computes the leg
    structure: per-leg dimensions, conformed (shared) GROUP BY keys, and the
    NULL-pad set for each leg. Advisory only — OBA does not compile SQL; OBSL
    owns CFL compilation.
    """
    session = services.get_session_data(ctx)

    if not session.graphrag_initialized or session.graphrag_manager is None:
        err: dict[str, Any] = services.create_error_response(
            "GraphRAG not initialized. Please call discover_schema() first.",
            "graphrag_not_initialized",
        )
        return err

    if not facts:
        err = services.create_error_response(
            "Provide at least one fact (measure-source) table.", "parameter_error"
        )
        return err

    retriever = session.graphrag_manager.graph_retriever

    missing = [f for f in facts if f not in retriever.graph]
    if missing:
        err = services.create_error_response(
            f"Tables not found in schema graph: {', '.join(missing)}", "data_error"
        )
        return err

    # Validate explicit dimensions too — an unknown dimension would otherwise be
    # silently null-padded into every leg and mislead downstream SQL planning.
    if dimensions:
        missing_dims = [d for d in dimensions if d not in retriever.graph]
        if missing_dims:
            err = services.create_error_response(
                f"Dimensions not found in schema graph: {', '.join(missing_dims)}",
                "data_error",
            )
            return err

    facts = list(dict.fromkeys(facts))  # de-dupe, preserve order

    # Dimension-capable set reachable from each fact (many-to-one closure).
    reach = {f: set(retriever.reachable_from(f)["tables"]) for f in facts}

    # Leg-root facts = facts that are NOT reachable from another fact. A fact
    # reachable from another sits on that fact's grain chain (a coarser table),
    # so it is a dimension of it, not an independent leg.
    leg_roots = [f for f in facts if not any(f in reach[g] for g in facts if g != f)]
    leg_roots = list(dict.fromkeys(leg_roots))

    cfl_required = len(leg_roots) >= 2

    # Requested dimensions: explicit list, else the union of all reachable dims.
    if dimensions:
        requested = list(dict.fromkeys(dimensions))
    else:
        requested = sorted(set().union(*reach.values()) if reach else set())

    # Conformed dims = reachable from every leg root → safe GROUP BY keys.
    if leg_roots:
        conformed_set = set.intersection(*[reach[f] for f in leg_roots])
    else:
        conformed_set = set()
    conformed = [d for d in requested if d in conformed_set]

    legs = []
    for root in leg_roots:
        leg_dims = [d for d in requested if d in reach[root]]
        null_pad = [d for d in requested if d not in reach[root]]
        legs.append(
            {
                "root": root,
                "dimensions": leg_dims,
                "null_pad": null_pad,
            }
        )

    if cfl_required:
        guidance = (
            "These facts are independent grains (disjoint siblings). Emit one "
            "UNION ALL leg per leg root, aggregating its own measures; project the "
            "conformed dimensions in every leg as GROUP BY keys and CAST(NULL AS "
            "<type>) for each leg's null_pad dimensions. OBA advises only — when "
            "OrionBelt Semantic Layer is connected, defer the actual CFL compilation "
            "to it."
        )
    elif leg_roots:
        guidance = (
            f"Single grain '{leg_roots[0]}' — a normal star join suffices, no "
            "Composite Fact Layer needed. The other facts sit on this grain's "
            "chain and act as dimensions."
        )
    else:
        guidance = "Could not determine a leg root."

    await ctx.info(
        f"CFL decomposition: cfl_required={cfl_required}, {len(legs)} leg(s)"
    )
    return {
        "success": True,
        "cfl_required": cfl_required,
        "facts": facts,
        "leg_roots": leg_roots,
        "conformed_dimensions": conformed,
        "legs": legs,
        "guidance": guidance,
    }


async def graphrag_overview(
    ctx: Context,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Get GraphRAG schema overview with statistics and communities."""
    session = services.get_session_data(ctx)

    if not session.graphrag_initialized or session.graphrag_manager is None:
        err: dict[str, Any] = services.create_error_response(
            "GraphRAG not initialized. Please call discover_schema() first.",
            "graphrag_not_initialized",
        )
        return err

    try:
        overview = session.graphrag_manager.get_schema_overview()

        await ctx.info(f"Generated schema overview for: {overview['schema_name']}")

        return {"success": True, "overview": overview}

    except Exception as e:
        logger.exception(f"GraphRAG overview failed: {e}")
        err = services.create_error_response(
            f"GraphRAG overview failed: {e!s}", "graphrag_error"
        )
        return err

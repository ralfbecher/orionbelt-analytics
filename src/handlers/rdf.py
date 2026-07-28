"""Oxigraph RDF store and SPARQL handler implementations."""

import asyncio
import logging
from typing import Any

from fastmcp import Context

from ..exceptions import (
    DependencyError,
    RDFError,
    StoreNotInitializedError,
    ValidationError,
)
from ..handler_context import HandlerContext
from ..lifecycle.metadata import (
    update_schema_version,
    update_workspace_rdf,
    update_workspace_section,
)
from ..oxigraph_store import OXIGRAPH_AVAILABLE, schema_graph_uri
from ..paths import OUTPUT_DIR, ensure_output_dir, get_connection_dir
from ..utils import read_text_file, utc_now

logger = logging.getLogger(__name__)


async def store_ontology_in_rdf(
    ctx: Context,
    schema_name: str | None,
    graph_uri: str | None,
    services: "HandlerContext",
) -> str | dict[str, Any]:
    """Store current session ontology in persistent RDF store with SPARQL access."""
    if not OXIGRAPH_AVAILABLE:
        return DependencyError(
            "pyoxigraph not installed. Install with: pip install pyoxigraph"
        ).to_response()

    store = services.get_oxigraph_store(ctx)
    if store is None:
        return StoreNotInitializedError(
            "Failed to initialize Oxigraph store"
        ).to_response()

    session = services.get_session_data(ctx)
    effective_schema = schema_name or session.get_last_analyzed_schema() or "default"

    if not session.ontology_file:
        return ValidationError(
            "No ontology generated. Please call generate_ontology() first."
        ).to_response()

    try:
        conn_dir = (
            get_connection_dir(session.connection_id)
            if session.connection_id
            else ensure_output_dir()
        )
        ontology_path = conn_dir / session.ontology_file

        if not ontology_path.exists():
            return ValidationError(
                f"Ontology file not found: {session.ontology_file}"
            ).to_response()

        ontology_ttl = await read_text_file(ontology_path)

        if not graph_uri:
            graph_uri = schema_graph_uri(effective_schema)

        triple_count = store.load_ontology(
            ontology_ttl=ontology_ttl, graph_uri=graph_uri, schema_name=effective_schema
        )

        await ctx.info(
            f"Stored ontology for schema '{effective_schema}' in RDF store: {triple_count} triples"
        )

        # Update workspace: mark ontology as persisted + write rdf_store
        if session.connection_id:
            try:
                await update_workspace_section(
                    connection_id=session.connection_id,
                    output_dir=OUTPUT_DIR,
                    schema_name=effective_schema,
                    section="ontology",
                    data={
                        "ontology_file": session.ontology_file,
                        "enriched": True,
                        "graph_uri": graph_uri,
                        "persisted_to_rdf": True,
                        "generated_at": utc_now().isoformat(),
                    },
                )
                await update_workspace_rdf(
                    connection_id=session.connection_id,
                    output_dir=OUTPUT_DIR,
                    data={
                        "initialized": True,
                        "graph_uris": [graph_uri],
                        "initialized_at": utc_now().isoformat(),
                    },
                )
                # This is the documented path for generate_ontology(
                # auto_persist=False), which leaves the version with no graph
                # URI and a zero triple count. Without recording them here the
                # version never learns which graph it loaded, and retention
                # could not delete that graph when the version aged out.
                await update_schema_version(
                    connection_id=session.connection_id,
                    output_dir=OUTPUT_DIR,
                    schema_name=effective_schema,
                    updates={
                        "ontology_ttl_file": session.ontology_file,
                        "ontology_graph_uri": graph_uri,
                        "ontology_triple_count": triple_count,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to write workspace metadata: {e}")

        return (
            f"Ontology stored successfully in RDF store!\n\n"
            f"Schema: {effective_schema}\n"
            f"Graph URI: <{graph_uri}>\n"
            f"Triples loaded: {triple_count}\n\n"
            f"You can now query using:\n"
            f"- query_sparql() - Execute SPARQL queries (SELECT, ASK, CONSTRUCT)"
        )

    except Exception as e:
        logger.exception(f"Failed to store ontology in RDF: {e}")
        return RDFError(f"Failed to store ontology: {e!s}").to_response()


def _detect_sparql_type(sparql_query: str) -> str:
    """Detect SPARQL query type from the query string."""
    stripped = sparql_query.strip()
    # Skip PREFIX declarations to find the actual query keyword
    import re

    body = re.sub(r"(?i)^\s*(PREFIX\s+\S+\s+<[^>]+>\s*)+", "", stripped).strip()
    upper = body.upper()
    if upper.startswith("ASK"):
        return "ASK"
    if upper.startswith("CONSTRUCT"):
        return "CONSTRUCT"
    return "SELECT"


async def query_sparql(
    ctx: Context,
    sparql_query: str,
    timeout_seconds: int,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Execute SPARQL query (SELECT, ASK, or CONSTRUCT) against stored ontologies."""
    if not OXIGRAPH_AVAILABLE:
        return DependencyError("pyoxigraph not installed").to_response()

    store = services.get_oxigraph_store(ctx)
    if store is None:
        return StoreNotInitializedError("Oxigraph store not initialized").to_response()

    query_type = _detect_sparql_type(sparql_query)

    try:
        if query_type == "ASK":
            result = store.query_sparql_ask(sparql_query)
            await ctx.info(f"SPARQL ASK query returned: {result}")
            return {
                "success": True,
                "query_type": "ASK",
                "result": result,
                "query": sparql_query,
            }
        elif query_type == "CONSTRUCT":
            result = await asyncio.to_thread(store.query_sparql_construct, sparql_query)
            await ctx.info("SPARQL CONSTRUCT query completed")
            return {
                "success": True,
                "query_type": "CONSTRUCT",
                "result": result,
                "query": sparql_query,
            }
        else:
            results = store.query_sparql(sparql_query, timeout_seconds=timeout_seconds)
            await ctx.info(f"SPARQL query returned {len(results)} results")
            return {
                "success": True,
                "query_type": "SELECT",
                "result_count": len(results),
                "results": results,
                "query": sparql_query,
            }

    except Exception as e:
        logger.exception(f"SPARQL query failed: {e}")
        return RDFError(f"SPARQL query failed: {e!s}").to_response()


async def query_sparql_ask(
    ctx: Context,
    sparql_query: str,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Execute SPARQL ASK query (returns true/false)."""
    if not OXIGRAPH_AVAILABLE:
        return DependencyError("pyoxigraph not installed").to_response()

    store = services.get_oxigraph_store(ctx)
    if store is None:
        return StoreNotInitializedError("Oxigraph store not initialized").to_response()

    try:
        result = store.query_sparql_ask(sparql_query)

        return {"success": True, "result": result, "query": sparql_query}

    except Exception as e:
        logger.exception(f"SPARQL ASK query failed: {e}")
        return RDFError(f"SPARQL ASK query failed: {e!s}").to_response()


async def add_rdf_knowledge(
    ctx: Context,
    subject: str,
    predicate: str,
    object_value: str,
    metadata: dict[str, Any] | None,
    services: "HandlerContext",
) -> str | dict[str, Any]:
    """Add custom knowledge/metadata to the RDF store."""
    if not OXIGRAPH_AVAILABLE:
        return DependencyError("pyoxigraph not installed").to_response()

    store = services.get_oxigraph_store(ctx)
    if store is None:
        return StoreNotInitializedError("Oxigraph store not initialized").to_response()

    try:
        store.add_knowledge(
            subject=subject, predicate=predicate, object=object_value, metadata=metadata
        )

        await ctx.info(
            f"Added knowledge triple: <{subject}> <{predicate}> {object_value}"
        )

        return (
            f"Knowledge added successfully!\n\n"
            f"Subject: <{subject}>\n"
            f"Predicate: <{predicate}>\n"
            f"Object: {object_value}"
        )

    except Exception as e:
        logger.exception(f"Failed to add knowledge: {e}")
        return RDFError(f"Failed to add knowledge: {e!s}").to_response()


async def list_tables_sparql(
    ctx: Context,
    schema_graph: str | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """List all tables from stored ontology using SPARQL."""
    if not OXIGRAPH_AVAILABLE:
        return DependencyError("pyoxigraph not installed").to_response()

    store = services.get_oxigraph_store(ctx)
    if store is None:
        return StoreNotInitializedError("Oxigraph store not initialized").to_response()

    try:
        if not schema_graph:
            session = services.get_session_data(ctx)
            schema_name = session.get_last_analyzed_schema() or "default"
            schema_graph = schema_graph_uri(schema_name)

        tables = store.list_tables_sparql(schema_graph)

        await ctx.info(f"Found {len(tables)} tables via SPARQL")

        return {
            "success": True,
            "table_count": len(tables),
            "tables": tables,
            "graph": schema_graph,
        }

    except Exception as e:
        logger.exception(f"SPARQL table listing failed: {e}")
        return RDFError(f"Failed to list tables: {e!s}").to_response()


async def find_columns_by_type_sparql(
    ctx: Context,
    data_type: str,
    schema_graph: str | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Find columns by data type using SPARQL."""
    if not OXIGRAPH_AVAILABLE:
        return DependencyError("pyoxigraph not installed").to_response()

    store = services.get_oxigraph_store(ctx)
    if store is None:
        return StoreNotInitializedError("Oxigraph store not initialized").to_response()

    try:
        columns = store.find_columns_by_type(data_type, schema_graph)

        await ctx.info(f"Found {len(columns)} {data_type} columns via SPARQL")

        return {
            "success": True,
            "data_type": data_type,
            "column_count": len(columns),
            "columns": columns,
        }

    except Exception as e:
        logger.exception(f"SPARQL column search failed: {e}")
        return RDFError(f"Failed to find columns: {e!s}").to_response()


async def get_rdf_store_stats(
    ctx: Context,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Get statistics about the persistent RDF store."""
    if not OXIGRAPH_AVAILABLE:
        return DependencyError("pyoxigraph not installed").to_response()

    store = services.get_oxigraph_store(ctx)
    if store is None:
        return StoreNotInitializedError("Oxigraph store not initialized").to_response()

    try:
        stats = store.get_ontology_stats()

        return {"success": True, "stats": stats}

    except Exception as e:
        logger.exception(f"Failed to get store stats: {e}")
        return RDFError(f"Failed to get stats: {e!s}").to_response()

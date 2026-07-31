"""Main MCP server application using FastMCP.

This module is a thin registration layer: it builds the FastMCP server, wires up
resources, and exposes one ``@mcp.tool()`` wrapper per tool that delegates to a
handler in ``src/handlers/``. The supporting machinery lives in dedicated
modules so this file stays mostly decorators + delegation:

- session state, request helpers, error responses -> :mod:`src.server_state`
- constrained MCP parameter types -> :mod:`src.tool_types`
- skill resources -> :mod:`src.resources`
"""

import logging
import os
from typing import Annotated, Any, Literal, cast

from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import Field

from . import __name__ as SERVER_NAME
from . import __version__

# --- Centralized path and env loading (Task 1 & 2) ---
from .paths import ensure_output_dir, get_env_file_path

# Load environment variables using centralized path resolution (Task 1: C4 fix)
env_path = get_env_file_path()
env_loaded = False
if env_path:
    load_dotenv(str(env_path))
    env_loaded = True

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Log environment loading info
if env_loaded:
    logger.info(f"Environment loaded from: {env_path}")
else:
    logger.warning("No .env file found - environment variables may not be available")

# Ensure output directory exists (Task 2: S3)
ensure_output_dir()


# --- MCP Server Setup ---

mcp = FastMCP(
    name=SERVER_NAME,
    version=__version__,
    instructions="""
# OrionBelt Analytics - AI-Powered Database Intelligence

Semantic database analysis with ontology-enhanced Text-to-SQL over 8 databases:
PostgreSQL, MySQL, Snowflake, ClickHouse, Dremio, BigQuery, DuckDB, Databricks.

## Capabilities

- **Schema Intelligence** - table/column analysis with relationship mapping
- **Ontology Generation** - RDF/OWL with the `oba:` namespace linking classes to SQL tables; auto JOIN conditions from relationships; load custom ontologies and emit R2RML
- **GraphRAG** - semantic schema discovery and join-path traversal
- **Persistent RDF Store** - Oxigraph-backed SPARQL 1.1 over generated ontologies
- **Safe SQL** - read-only enforcement, injection prevention, fan-trap detection, timeouts, max 5,000 rows
- **Visualization** - interactive Plotly/Matplotlib charts

## Recommended Workflow

`connect_database()` -> `list_schemas()` -> `discover_schema()` ->
`generate_ontology()` -> `execute_sql_query()` -> `generate_chart()`

## Critical Guides (Claude Skills)

- `/fan-trap-prevention` - prevent data multiplication in multi-table queries
- `/sql-best-practices` - identifier qualification and safe patterns
- `/chart-examples` - visualization guide for all chart types

## Important Notes

- Always fully qualify identifiers: `schema.table.column`
- Review `foreign_keys` from `discover_schema()` before complex JOINs
- `execute_sql_query()` runs built-in syntax, security, and OBQC validation
- For multi-fact aggregation, use the UNION ALL pattern (see `/fan-trap-prevention`)
""",
)


# --- MCP Resources: skill files exposed over skill:// URIs ---
from .resources import register_resources  # noqa: E402

register_resources(mcp)


from .handler_context import HandlerContext  # noqa: E402

# --- Handler imports (Task 7: C1/S2) ---
# Imported here to keep the handler layer decoupled from server setup.
from .handlers import chart as _h_chart  # noqa: E402
from .handlers import connection as _h_connection  # noqa: E402
from .handlers import graphrag as _h_graphrag  # noqa: E402
from .handlers import ontology as _h_ontology  # noqa: E402
from .handlers import query as _h_query  # noqa: E402
from .handlers import rdf as _h_rdf  # noqa: E402
from .handlers import schema as _h_schema  # noqa: E402
from .handlers import workspace as _h_workspace  # noqa: E402

# --- Session state and per-request helpers (extracted to server_state) ---
# ServerState and _calculate_schema_hash are re-exported for tests that import
# them from main; F401 is suppressed for this re-export block.
from .server_state import (  # noqa: E402, F401
    ServerState,
    _calculate_schema_hash,
    _clear_session_state,
    _get_connection_fingerprint,
    _server_state,
    create_error_response,
    get_oxigraph_store,
    get_session_data,
    get_session_db_manager,
    get_session_obqc_validator,
    get_session_safe_filename,
    load_ontology_from_session,
)

# --- Constrained MCP parameter types (extracted to tool_types) ---
from .tool_types import (  # noqa: E402
    _DbType,
    _DocBody,
    _FolderPath,
    _Identifier,
    _QueryBody,
    _QueryText,
    _SafeName,
    _ShortText,
    _Uri,
)


def _services() -> HandlerContext:
    """Build the per-request service bundle handed to handler functions.

    Reads the module-level helpers each call so test patches on these names
    (e.g. via ``src.main`` / ``src.server_state``) are honored.
    """
    return HandlerContext(
        get_session_data=get_session_data,
        get_session_db_manager=get_session_db_manager,
        get_session_safe_filename=get_session_safe_filename,
        get_session_obqc_validator=get_session_obqc_validator,
        get_oxigraph_store=get_oxigraph_store,
        load_ontology_from_session=load_ontology_from_session,
        create_error_response=create_error_response,
        server_state=_server_state,
        get_connection_fingerprint=_get_connection_fingerprint,
        clear_session_state=_clear_session_state,
        auto_initialize_graphrag_background=_h_graphrag._auto_initialize_graphrag_background,
        add_resource=mcp.add_resource,
    )


# ============================================================
# MCP Tool Registration
# ============================================================
# Each tool delegates to its handler module. The @mcp.tool()
# decorators MUST stay here for FastMCP registration.
# ============================================================


@mcp.tool()
async def connect_database(ctx: Context, db_type: _DbType) -> str:  # type: ignore[valid-type]
    """Connect to a database using credentials from environment variables.

    If a previous workspace exists for this connection, it is automatically
    restored (schema cache, ontology, GraphRAG, RDF store). The response
    indicates what was restored and which tools are ready to use.

    Args:
        db_type: Database type - 'postgresql', 'snowflake', 'dremio', 'clickhouse', 'bigquery', 'duckdb', 'databricks', or 'mysql'

    Returns:
        Connection status with auto-restored workspace summary if available
    """
    return await _h_connection.connect_database(
        ctx,
        db_type,
        services=_services(),
    )


@mcp.tool()
async def list_schemas(ctx: Context) -> list[str]:
    """Get a list of available schemas from the connected database.

    REQUIRES: connect_database must be called first.
    """
    return await _h_connection.list_schemas(ctx, services=_services())


@mcp.tool()
async def reset_cache(
    ctx: Context,
    cache_type: Literal["schema", "ontology", "all"] | None = None,
) -> dict[str, Any]:
    """Reset cached schema and/or ontology data to force re-analysis.

    Args:
        cache_type: Type of cache to reset ("schema", "ontology", "all", or None)

    Returns:
        Dictionary with status and cleared cache types
    """
    return await _h_schema.reset_cache(ctx, cache_type, services=_services())


@mcp.tool()
async def discover_schema(
    ctx: Context,
    schema_name: _Identifier | None = None,
    lightweight: bool = True,
) -> dict[str, Any]:
    """Analyze database schema and return table metadata with relationships.

    REQUIRES: connect_database must be called first and must complete before calling this tool.

    Args:
        schema_name: Schema to analyze (optional, uses default if not specified)
        lightweight: If True (default), return minimal data (table names, FK relationships, fan-trap warnings).
                     If False, return full schema with all column details.
    """
    return await _h_schema.discover_schema(
        ctx,
        schema_name,
        lightweight,
        services=_services(),
    )


@mcp.tool()
async def get_table_details(
    ctx: Context,
    table_name: _Identifier,
    schema_name: _Identifier | None = None,
) -> dict[str, Any]:
    """Get detailed metadata for a single table. Only use when you need to
    inspect a specific table that the user asked about — do NOT call this for
    every table. discover_schema() and the ontology already contain full
    schema structure including columns, keys, and relationships.

    REQUIRES: connect_database must be called first.

    Args:
        table_name: Name of the table to analyze
        schema_name: Schema containing the table (optional, auto-detected)
    """
    return await _h_schema.get_table_details(
        ctx,
        table_name,
        schema_name,
        services=_services(),
    )


@mcp.tool()
async def generate_ontology(
    ctx: Context,
    schema_info: _DocBody | None = None,
    schema_name: _Identifier | None = None,
    base_uri: _Uri = "http://example.com/ontology/",
    auto_persist: bool = True,
    graph_uri: _Uri | None = None,
) -> str:
    """Generate an RDF ontology from database schema. AUTO-ANALYZES schema if needed!

    Args:
        schema_name: Name of the schema to analyze and generate ontology for
        schema_info: Optional pre-analyzed schema JSON (usually not needed)
        base_uri: Base URI for the ontology (default: http://example.com/ontology/)
        auto_persist: If True (default), automatically store in Oxigraph RDF database.
        graph_uri: Optional custom graph URI for RDF storage

    Returns:
        Ontology TTL or status message
    """
    return cast(
        str,
        await _h_ontology.generate_ontology(
            ctx,
            schema_info,
            schema_name,
            base_uri,
            auto_persist,
            graph_uri,
            services=_services(),
        ),
    )


@mcp.tool()
async def suggest_semantic_names(
    ctx: Context,
    ontology_file: _SafeName | None = None,
) -> dict[str, Any]:
    """Extract and analyze names from a generated ontology to identify abbreviations and cryptic names.

    When the connected MCP client supports sampling (and ENABLE_SAMPLING=true),
    the server pre-fills a ``suggestions`` dict via the host LLM so the next
    call to ``apply_semantic_names`` can pass them through directly. Otherwise
    the response contains only the cryptic-name lists for manual review.

    Args:
        ontology_file: The ontology filename from generate_ontology response

    Returns:
        Dictionary containing extracted names, analysis results, and instructions
    """
    return await _h_ontology.suggest_semantic_names(
        ctx,
        ontology_file,
        services=_services(),
    )


@mcp.tool()
async def apply_semantic_names(
    ctx: Context,
    suggestions: Annotated[str, Field(max_length=2000000)] | dict[str, Any],
    ontology_file: _SafeName | None = None,
    save_to_file: bool = True,
) -> str:
    """Apply semantic name suggestions to an existing ontology.

    The suggestions parameter accepts a JSON object (or JSON string) with 'classes',
    'properties', and/or 'relationships' arrays. Each entry needs 'original_name'
    and 'suggested_name'.

    Example suggestions:
    {
      "classes": [
        {"original_name": "acctbal", "suggested_name": "AccountBalance", "description": "Account balance records"}
      ],
      "properties": [
        {"original_name": "bankid", "table_name": "acctbal", "suggested_name": "Bank Identifier"}
      ],
      "relationships": [
        {"original_name": "acctbal_to_banks", "suggested_name": "Account Bank Relationship"}
      ]
    }

    Args:
        suggestions: JSON object or string with classes/properties/relationships arrays (see example above)
        ontology_file: The ontology filename from generate_ontology response
        save_to_file: Whether to save the updated ontology to a file
    """
    return cast(
        str,
        await _h_ontology.apply_semantic_names(
            ctx,
            suggestions,
            ontology_file,
            save_to_file,
            services=_services(),
        ),
    )


@mcp.tool()
async def load_my_ontology(
    ctx: Context,
    import_folder: _FolderPath = "./import",
    auto_persist: bool = True,
    graph_uri: _Uri | None = None,
    ontology_content: _DocBody | None = None,
    file_name: _SafeName | None = None,
) -> dict[str, Any]:
    """Load an ontology in Turtle (.ttl) format.

    Accepts either inline content (e.g. when the user drops a .ttl file into the
    chat) or reads the newest .ttl file from the import folder.

    Args:
        import_folder: Path to the folder containing .ttl files (used when ontology_content is not provided)
        auto_persist: If True (default), store in Oxigraph RDF database
        graph_uri: Optional custom graph URI for RDF storage
        ontology_content: Optional TTL content passed directly (e.g. from a file dropped into chat)
        file_name: Optional original file name when providing ontology_content

    Returns:
        Dictionary with ontology information and status
    """
    return await _h_ontology.load_my_ontology(
        ctx,
        import_folder,
        auto_persist,
        graph_uri,
        ontology_content=ontology_content,
        file_name=file_name,
        services=_services(),
    )


@mcp.tool()
async def download_artifact(
    ctx: Context,
    artifact_type: Literal["ontology", "r2rml"],
    schema_name: _Identifier | None = None,
    source: Literal["rdf", "file"] = "rdf",
) -> dict[str, Any]:
    """Download a generated artifact as TTL file.

    Args:
        artifact_type: Type of artifact to download ("ontology" or "r2rml")
        schema_name: Name of the schema
        source: Where to get the artifact from ("rdf" or "file"), only applies to ontology

    Returns:
        Dictionary with artifact content and file info
    """
    if artifact_type == "ontology":
        return await _h_ontology.download_ontology(
            ctx,
            schema_name,
            source,
            services=_services(),
        )
    elif artifact_type == "r2rml":
        return await _h_ontology.download_r2rml(ctx, schema_name, services=_services())
    else:
        return create_error_response(  # type: ignore[unreachable]
            f"Invalid artifact_type: {artifact_type}. Must be 'ontology' or 'r2rml'.",
            "parameter_error",
        )


@mcp.tool()
async def sample_table_data(
    ctx: Context,
    table_name: _Identifier,
    schema_name: _Identifier | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Sample data from a specific table for analysis.

    REQUIRES: connect_database must be called first.

    Args:
        table_name: Name of the table to sample
        schema_name: Schema containing the table (optional)
        limit: Maximum number of rows to return (default: 10, max: 100)
    """
    return await _h_schema.sample_table_data(
        ctx,
        table_name,
        schema_name,
        limit,
        services=_services(),
    )


@mcp.tool()
async def execute_sql_query(
    ctx: Context,
    sql_query: _QueryBody,
    limit: int = 1000,
    checklist_completed: bool = False,
    query_intent: _ShortText | None = None,
) -> dict[str, Any]:
    """Execute SQL query with built-in syntax validation, security checks, OBQC
    validation, and fan-trap protection.

    PREFER SEMANTIC LAYER: If the OrionBelt Semantic Layer MCP server is available,
    create an OBML model and use execute_query/compile_query instead of raw SQL.
    Only use this tool when no semantic model is loaded or the user explicitly asks
    for raw SQL.

    Automatically validates SQL syntax and security before execution. If an ontology
    is loaded, OBQC checks (fan-trap detection, semantic validation) run too — errors
    block execution, warnings are included in the result.

    REQUIRES: connect_database must be called first.

    Args:
        sql_query: SQL SELECT statement (fully qualified identifiers required)
        limit: Maximum rows to return (default: 1000, max: 5,000)
        checklist_completed: Confirmation that pre-execution checklist is complete
        query_intent: Optional natural language description of query intent
    """
    return await _h_query.execute_sql_query(
        ctx,
        sql_query,
        limit,
        checklist_completed,
        query_intent,
        services=_services(),
    )


@mcp.tool()
async def generate_chart(
    ctx: Context,
    data_source: list[dict[str, Any]] | Annotated[str, Field(max_length=5000000)],
    chart_type: Literal["bar", "line", "scatter", "heatmap"],
    x_column: _Identifier,
    y_column: _Identifier | list[_Identifier] | None = None,
    color_column: _Identifier | None = None,
    title: _ShortText | None = None,
    chart_style: Literal["grouped", "stacked"] = "grouped",
    sort_by: _Identifier | None = None,
    sort_order: Literal["ascending", "descending"] | None = None,
    output_format: Literal["interactive", "image"] = "interactive",
) -> str:
    """Generate a chart from query results. Returns a ui:// MCP Apps widget for interactive use.

    Args:
        data_source: JSON array of objects, e.g. [{"name": "A", "value": 10}, ...] — pass as array, not string
        chart_type: 'bar', 'line', 'scatter', or 'heatmap'
        x_column: Column name for X-axis
        y_column: Column name(s) for Y-axis — a string, or a list for multi-series bar/line charts
        color_column: Optional column for grouping/coloring (heatmap: numeric value column for color intensity)
        title: Chart title (auto-generated if omitted)
        chart_style: 'grouped' or 'stacked' — bar charts only
        sort_by: Column to sort by (auto-sorted per chart type if omitted)
        sort_order: 'ascending' or 'descending'
        output_format: "interactive" (default, responsive MCP Apps widget) or "image" (saves PNG file)
    """
    # The handler returns str (interactive widget URI) or a list of image
    # artifacts for output_format="image"; the published tool contract is str,
    # so cast to keep the FastMCP output schema unchanged.
    return cast(
        str,
        await _h_chart.generate_chart(
            ctx,
            data_source,
            chart_type,
            x_column,
            y_column,
            color_column,
            title,
            chart_style,
            sort_by,
            sort_order,
            output_format,
            services=_services(),
        ),
    )


@mcp.tool()
async def cleanup_workspace(ctx: Context) -> str:
    """Delete all workspace files for the current database connection and clear session state.

    Removes schema JSON, ontology TTL, R2RML mappings, GraphRAG data, ChromaDB vectors,
    Oxigraph RDF store, semantic models, and metadata for this connection.
    The database connection itself remains active.

    Use this to start fresh or free disk space. Requires an active connection.

    Returns:
        Summary of what was removed
    """
    return cast(
        str,
        await _h_workspace.cleanup_workspace(
            ctx,
            services=_services(),
        ),
    )


@mcp.tool()
async def cleanup_old_versions(
    ctx: Context,
    schema_name: _Identifier | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Prune old ontology and GraphRAG versions for a schema per the retention policy.

    Unlike cleanup_workspace(), which removes everything for the connection, this
    deletes only archived versions that have aged past the retention policy —
    the current generation and recent history are kept. Also returns the schema's
    remaining version history.

    Retention is controlled by GRAPHRAG_KEEP_VERSIONS, GRAPHRAG_MAX_AGE_DAYS,
    ONTOLOGY_KEEP_VERSIONS and ONTOLOGY_MAX_AGE_DAYS.

    Args:
        schema_name: Schema whose history to prune (last analyzed schema if omitted)
        dry_run: Report what would be deleted without deleting it (default True)
    """
    return await _h_workspace.cleanup_old_versions(
        ctx,
        schema_name,
        dry_run,
        services=_services(),
    )


@mcp.tool()
async def save_semantic_model(
    ctx: Context,
    model_yaml: _DocBody,
    model_name: _SafeName,
    schema_name: _Identifier | None = None,
) -> dict[str, Any]:
    """Save a semantic model (e.g., OBML YAML) to the workspace for reuse across sessions.

    Stores the model definition so it can be retrieved in future sessions via
    get_semantic_model() and loaded into a Semantic Layer if available.

    Args:
        model_yaml: The model definition in YAML format
        model_name: Name to identify this model (e.g., "sales_analytics")
        schema_name: Database schema this model is based on (auto-detected if omitted)
    """
    return await _h_workspace.save_semantic_model(
        ctx,
        model_yaml,
        model_name,
        schema_name,
        services=_services(),
    )


@mcp.tool()
async def get_semantic_model(
    ctx: Context,
    model_name: _SafeName,
) -> dict[str, Any]:
    """Retrieve a stored semantic model YAML by name.

    Use this to get a previously saved model definition, e.g., to pass it
    to a Semantic Layer's load_model() tool.

    Args:
        model_name: Name of the model to retrieve
    """
    return await _h_workspace.get_semantic_model(
        ctx,
        model_name,
        services=_services(),
    )


@mcp.tool()
async def list_semantic_models(ctx: Context) -> dict[str, Any]:
    """List all stored semantic models for the current database connection.

    Returns:
        List of available models with names, schemas, and save dates
    """
    return await _h_workspace.list_semantic_models(
        ctx,
        services=_services(),
    )


# --- GraphRAG Tools ---


@mcp.tool()
async def graphrag_search(
    ctx: Context,
    query: _QueryText | None = None,
    top_k: int = 5,
    element_type: (
        Literal["table", "column", "relationship", "semantic_context"] | None
    ) = None,
    overview: bool = False,
) -> dict[str, Any]:
    """Search schema using natural language via GraphRAG, or get a schema overview.

    GraphRAG is auto-initialized by discover_schema. Pass overview=True to get
    schema statistics and community clustering instead of search results.

    Matching is semantic, but it can only match what the schema says about
    itself. If a concept lives in business vocabulary the column names do not
    use ("profit", "churn"), index it first with add_semantic_context.

    Args:
        query: Natural language search query (required unless overview=True)
        top_k: Number of results to return
        element_type: Filter by type ("table", "column", "relationship",
            "semantic_context", or None)
        overview: If True, return schema statistics and communities instead of search

    Returns:
        Dictionary with search results or schema overview
    """
    if overview:
        return await _h_graphrag.graphrag_overview(ctx, services=_services())
    if not query:
        return create_error_response(
            "query parameter is required when overview=False",
            "parameter_error",
        )
    return await _h_graphrag.graphrag_search(
        ctx,
        query,
        top_k,
        element_type,
        services=_services(),
    )


@mcp.tool()
async def add_semantic_context(
    ctx: Context,
    target: _Identifier,
    context: _QueryText,
) -> dict[str, Any]:
    """Teach GraphRAG what a table or column means in business terms.

    Schema search can only match vocabulary the schema itself contains, which
    is usually abbreviations - salesamount, unitcost, returnquantity. Questions
    about "profit", "margin" or "churn" therefore find nothing, because no
    column name carries those words. Use this to supply the missing vocabulary
    so later graphrag_search calls can reach the right elements.

    Write what an analyst would need to know: what the element measures, the
    words users would search for it with, and any formula worth surfacing.

    The context is added to the semantic index as a separate element. It does
    NOT modify the schema, the ontology, or the RDF store, so rerunning
    discover_schema cannot overwrite it - but it is discarded if the index is
    rebuilt (for example after changing GRAPHRAG_EMBEDDING_MODEL). Treat it as
    session enrichment, not durable knowledge; use add_rdf_knowledge for facts
    that must persist.

    Calling this again for the same target REPLACES the previous context, so
    it can be revised freely.

    Check the returned "searchable" flag. Under the offline "tfidf" embedding
    backend the context is stored but cannot be matched, and the response
    carries a "warning" explaining why - do not report the concept as findable
    in that case.

    REQUIRES: discover_schema must have been called first.

    Args:
        target: Element the context describes, as "table" or "table.column".
        context: Business meaning in natural language, including the terms
            users would search with. Example for sales.salesamount:
            "Revenue per line item. Profit margin is salesamount minus
            unitcost. Used for profitability and margin analysis."

    Returns:
        Dictionary with the indexed element id, target, character count,
        whether it replaced existing context, and whether it is searchable
    """
    return await _h_graphrag.graphrag_add_semantic_context(
        ctx,
        target,
        context,
        services=_services(),
    )


@mcp.tool()
async def graphrag_query_context(
    ctx: Context,
    query: _QueryText,
    max_tables: int = 5,
    max_columns: int = 20,
) -> dict[str, Any]:
    """Get optimized context for SQL query generation using GraphRAG.

    Args:
        query: Natural language description of what you want to query
        max_tables: Maximum tables to include in context
        max_columns: Maximum columns to include in context

    Returns:
        Optimized context with relevant tables, columns, relationships
    """
    return await _h_graphrag.graphrag_query_context(
        ctx,
        query,
        max_tables,
        max_columns,
        services=_services(),
    )


@mcp.tool()
async def graphrag_find_join_path(
    ctx: Context,
    from_table: _Identifier,
    to_table: _Identifier,
    max_hops: int = 3,
) -> dict[str, Any]:
    """Find join path between two tables using GraphRAG graph traversal.

    Args:
        from_table: Source table name
        to_table: Target table name
        max_hops: Maximum number of joins allowed

    Returns:
        Dictionary with join path specifications
    """
    return await _h_graphrag.graphrag_find_join_path(
        ctx,
        from_table,
        to_table,
        max_hops,
        services=_services(),
    )


@mcp.tool()
async def reachable_from(
    ctx: Context,
    table: _Identifier,
    max_hops: int | None = None,
) -> dict[str, Any]:
    """List the dimension-capable tables for a query anchored on a table.

    Follows foreign keys in the many-to-one (finer grain -> coarser grain)
    direction: the returned tables can be joined from `table` without row
    multiplication (each join is functional), so their columns are safe to use
    as dimensions (GROUP BY / filter). This is the directed reachability the
    bidirectional `graphrag_find_join_path` cannot express. Pair with
    `measurable_from` for the measure side.

    Args:
        table: Anchor table name (the query grain)
        max_hops: Maximum FK hops to follow (None = full closure)

    Returns:
        Dictionary with reachable (dimension-capable) tables and per-hop breakdown
    """
    return await _h_graphrag.reachable_from(
        ctx,
        table,
        max_hops,
        services=_services(),
    )


@mcp.tool()
async def measurable_from(
    ctx: Context,
    table: _Identifier,
    max_hops: int | None = None,
) -> dict[str, Any]:
    """List the measure-capable tables for a query anchored on a table.

    Follows foreign keys in the one-to-many (toward finer grain) direction: the
    returned tables fan out `table`, so their values can only be aggregated into
    measures (SUM/COUNT/...) and must NOT be used as dimensions at this grain
    (doing so is a fan-trap). The inverse of `reachable_from`.

    Args:
        table: Anchor table name (the query grain)
        max_hops: Maximum FK hops to follow (None = full closure)

    Returns:
        Dictionary with measure-capable tables and per-hop breakdown
    """
    return await _h_graphrag.measurable_from(
        ctx,
        table,
        max_hops,
        services=_services(),
    )


@mcp.tool()
async def plan_composite_query(
    ctx: Context,
    facts: list[_Identifier],
    dimensions: list[_Identifier] | None = None,
) -> dict[str, Any]:
    """Advise a Composite Fact Layer (CFL) decomposition for a multi-fact query.

    Given the fact (measure-source) tables a query needs, determines whether
    they are independent grains (disjoint siblings) requiring a UNION ALL
    composite, and returns the leg structure: per-leg dimensions, the conformed
    (shared) dimensions that become GROUP BY keys in every leg, and each leg's
    NULL-pad set. Use this before writing cross-fact SQL to avoid fan-traps.

    Advisory only: OBA does not compile SQL. When OrionBelt Semantic Layer is
    connected, defer the actual CFL compilation to it.

    Args:
        facts: The fact (measure-source) tables the query aggregates
        dimensions: Optional explicit dimension tables to project (default: all
            tables reachable from the facts)

    Returns:
        Dictionary with cfl_required flag, leg roots, conformed dimensions, and
        per-leg dimension/NULL-pad decomposition
    """
    return await _h_graphrag.plan_composite_query(
        ctx,
        facts,
        dimensions,
        services=_services(),
    )


# --- Oxigraph RDF Store & SPARQL Tools ---


@mcp.tool()
async def store_ontology_in_rdf(
    ctx: Context,
    schema_name: _Identifier | None = None,
    graph_uri: _Uri | None = None,
) -> str:
    """Store current session ontology in persistent RDF store with SPARQL access.

    Args:
        schema_name: Schema name (uses last analyzed if not specified)
        graph_uri: Named graph URI (auto-generated if not specified)

    Returns:
        Status message with triple count
    """
    return cast(
        str,
        await _h_rdf.store_ontology_in_rdf(
            ctx,
            schema_name,
            graph_uri,
            services=_services(),
        ),
    )


@mcp.tool()
async def query_sparql(
    ctx: Context,
    sparql_query: _QueryBody,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Execute a SPARQL query against the RDF ontology store. Use this to explore
    schema relationships, classes, properties, and semantic metadata loaded via
    generate_ontology or load_my_ontology. Requires an ontology to be loaded first.

    Supports SELECT, ASK, and CONSTRUCT query types (auto-detected from query string).
    Common prefixes (rdf, rdfs, owl, xsd) are available by default.

    Named graphs: each loaded schema lives in its own named graph, and the store
    is accumulative across schemas. Unwrapped patterns are matched against the
    union of every named graph, so a plain "?s ?p ?o" sees all loaded schemas.
    To scope results to one schema, either bind the graph:
        SELECT ?g (COUNT(*) AS ?n) WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g
    or select the dataset explicitly, which suppresses the union and restricts
    the query to exactly the graphs it names:
        SELECT ?s FROM <http://example.com/schema/public> WHERE { ?s ?p ?o }

    Args:
        sparql_query: A complete SPARQL query string (SELECT, ASK, or CONSTRUCT).
            Example: "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
        timeout_seconds: Best-effort SELECT timeout in seconds. Unblocks the
            caller after the timeout; the underlying query may keep running in
            the background (pyoxigraph has no native query cancellation).

    Returns:
        Query results (bindings for SELECT, boolean for ASK, Turtle string for CONSTRUCT)
    """
    return await _h_rdf.query_sparql(
        ctx,
        sparql_query,
        timeout_seconds,
        services=_services(),
    )


@mcp.tool()
async def add_rdf_knowledge(
    ctx: Context,
    subject: _Uri,
    predicate: _Uri,
    object: Annotated[str, Field(max_length=8192)],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Add custom knowledge/metadata to the RDF store.

    Args:
        subject: Subject URI
        predicate: Predicate URI
        object: Object value (literal or URI)
        metadata: Optional metadata dict

    Returns:
        Confirmation message
    """
    return cast(
        str,
        await _h_rdf.add_rdf_knowledge(
            ctx,
            subject,
            predicate,
            object,
            metadata,
            services=_services(),
        ),
    )


# --- Cleanup on shutdown ---


def cleanup_server() -> None:
    """Clean up server resources."""
    _server_state.cleanup()


def get_registered_tool_names() -> list[str]:
    """Return the sorted names of every tool registered on the MCP server.

    This is the single source of truth for tool counts in the startup banner
    and docs, so those numbers cannot drift from the actual registrations.
    """
    import asyncio

    tools = asyncio.run(mcp.list_tools())
    return sorted(tool.name for tool in tools)

#!/usr/bin/env python3
"""Startup script for OrionBelt Analytics."""

import asyncio
import json
import shutil
import signal
import sys
from datetime import timedelta
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import atexit  # noqa: E402

from src import __name__ as SERVER_NAME  # noqa: E402
from src import __version__  # noqa: E402
from src.config import config_manager  # noqa: E402
from src.main import cleanup_server, mcp  # noqa: E402
from src.utils import parse_timestamp, setup_logging, utc_now  # noqa: E402

# Setup logging
config = config_manager.get_server_config()
logger = setup_logging(
    config.log_level, structured=False
)  # Use simple format for startup


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # Set shutdown event to allow cleanup
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(shutdown_event.set)
        except RuntimeError:
            pass  # No event loop running
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return shutdown_event


def print_startup_info():
    """Print server startup information."""
    logger.info("=" * 60)
    logger.info(f"{SERVER_NAME} v{__version__}")
    logger.info(
        "MCP server for database ad hoc analysis with ontology support and interactive charting"
    )
    logger.info("=" * 60)

    # Count comes straight from the registry so the banner can never drift
    # from the tools actually registered in src/main.py.
    from src.main import get_registered_tool_names

    registered = get_registered_tool_names()
    logger.info(f"🔧 Available MCP Tools ({len(registered)}):")
    tool_groups = {
        "Connection & Schema": [
            "connect_database - Connect to any supported database",
            "list_schemas - List available database schemas",
            "discover_schema - Analyze schema with auto GraphRAG + ontology",
            "get_table_details - Detailed table metadata and columns",
            "reset_cache - Clear cached schema and ontology data",
            "cleanup_workspace - Delete workspace files and start fresh",
        ],
        "Ontology & Semantic": [
            "generate_ontology - Generate RDF/OWL ontology with SQL annotations",
            "suggest_semantic_names - Detect abbreviations for renaming",
            "apply_semantic_names - Apply semantic names to ontology",
            "load_my_ontology - Load a custom .ttl ontology file",
            "download_artifact - Download ontology or R2RML as Turtle file",
        ],
        "Query & Visualization": [
            "sample_table_data - Preview table data with security controls",
            "execute_sql_query - Execute SQL with built-in validation and fan-trap checks",
            "generate_chart - Generate Plotly charts with MCP-UI rendering",
        ],
        "GraphRAG": [
            "graphrag_search - Semantic search + overview (auto-initialized by discover_schema)",
            "graphrag_query_context - Optimized context for SQL generation",
            "graphrag_find_join_path - Discover join paths via graph traversal",
        ],
        "Graph Reasoning": [
            "reachable_from - Traverse object-property edges from a class",
            "measurable_from - Find numeric measures reachable from a class",
            "plan_composite_query - Plan a multi-hop query over the ontology",
        ],
        "SPARQL & RDF": [
            "store_ontology_in_rdf - Persist ontology in Oxigraph",
            "query_sparql - Execute SPARQL queries (SELECT, ASK, CONSTRUCT)",
            "add_rdf_knowledge - Add custom metadata triples",
        ],
        "Semantic Models": [
            "save_semantic_model - Save a semantic model to workspace",
            "get_semantic_model - Retrieve a stored semantic model",
            "list_semantic_models - List all stored semantic models",
        ],
    }
    for group, tools in tool_groups.items():
        logger.info(f"  [{group}]")
        for tool in tools:
            logger.info(f"    • {tool}")

    # Warn loudly if the curated banner ever drifts from the real registry.
    listed = {
        entry.split(" - ", 1)[0].strip()
        for tools in tool_groups.values()
        for entry in tools
    }
    missing = sorted(set(registered) - listed)
    extra = sorted(listed - set(registered))
    if missing or extra:
        logger.warning(
            "Startup tool banner out of sync with registry "
            f"(missing: {missing or 'none'}; stale: {extra or 'none'})"
        )

    logger.info("")
    logger.info(
        "🗄️ Supported Databases: PostgreSQL, MySQL, Snowflake, ClickHouse, Dremio, BigQuery, DuckDB/MotherDuck, Databricks"
    )
    logger.info("🧠 LLM Enrichment: Available via MCP prompts and tools")
    logger.info("🔒 Security: Credential handling and input validation")
    logger.info("⚡ Performance: Connection pooling and parallel processing")
    logger.info("📊 Observability: Structured logging and comprehensive error handling")
    logger.info("")
    logger.info("📋 Configuration:")
    logger.info(f"  • Log Level: {config.log_level}")
    logger.info(f"  • Base URI: {config.ontology_base_uri}")
    logger.info(f"  • Transport: {config.mcp_transport}")
    logger.info(f"  • Host: {config.mcp_server_host}")
    logger.info(f"  • Port: {config.mcp_server_port}")
    logger.info("")


def cleanup_tmp_folder():
    """Clean the output directory at startup, preserving live workspace data.

    Always runs, whatever AUTO_CLEANUP_ON_STARTUP is set to:
    - deletes stale loose files sitting directly in OUTPUT_DIR
    - deletes every workspace's charts/ directory (chart images are ephemeral)

    AUTO_CLEANUP_ON_STARTUP then decides whether whole workspaces go:
    - "false" (default): none; every workspace is kept so auto-restore works
    - "true": deletes a workspace with no metadata.json (orphaned), or whose
      workspace.updated_at is older than WORKSPACE_MAX_AGE_DAYS. A workspace
      whose metadata has no updated_at is kept.
    - "all": deletes every workspace (fresh start)

    Deleting a workspace also deletes the satellite stores keyed to that
    connection -- chromadb/{id} and oxigraph/{id} live beside the workspace, not
    inside it, so they would otherwise be stranded on disk forever. The
    chromadb/ and oxigraph/ parents are shared across connections and are never
    themselves treated as workspaces (they carry no metadata.json, so doing so
    would delete every connection's vectors and triples).
    """
    import os

    from src.paths import NON_WORKSPACE_DIRS, OUTPUT_DIR, get_connection_store_dirs

    tmp_dir = OUTPUT_DIR

    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory for generated files: {tmp_dir}")
        return

    def _remove_workspace(conn_dir: Path) -> None:
        """Remove a workspace and the satellite stores keyed to it.

        chromadb/{id} and oxigraph/{id} live outside the workspace directory,
        so deleting only conn_dir would strand them on disk forever.
        """
        shutil.rmtree(conn_dir)
        for store_dir in get_connection_store_dirs(conn_dir.name):
            if store_dir.exists():
                try:
                    shutil.rmtree(store_dir)
                    logger.debug(f"Removed satellite store: {store_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove {store_dir}: {e}")

    # Phase 1: Remove stale top-level files (always)
    removed = 0
    workspace_dirs = []
    try:
        for item in tmp_dir.iterdir():
            if item.is_file():
                item.unlink()
                removed += 1
                logger.debug(f"Removed stale file: {item.name}")
            elif item.is_dir() and item.name not in NON_WORKSPACE_DIRS:
                # chromadb/ and oxigraph/ hold per-connection stores one level
                # deeper. They carry no metadata.json, so counting them as
                # workspaces makes them look orphaned -- which deleted every
                # connection's vectors and triples on the first retention run.
                workspace_dirs.append(item)
    except Exception as e:
        logger.warning(f"Failed to clean tmp directory: {e}")
        return

    # Clean chart images from all workspace directories (ephemeral, not reusable)
    charts_removed = 0
    for conn_dir in workspace_dirs:
        charts_dir = conn_dir / "charts"
        if charts_dir.exists():
            try:
                shutil.rmtree(charts_dir)
                charts_removed += 1
            except Exception as e:
                logger.debug(f"Failed to clean charts in {conn_dir.name}: {e}")

    if removed > 0 or charts_removed > 0:
        logger.info(
            f"Startup cleanup: removed {removed} stale file(s), "
            f"{charts_removed} chart directory/directories"
        )

    if workspace_dirs:
        logger.info(f"Found {len(workspace_dirs)} workspace directory/directories")

    # Phase 2: Workspace cleanup (opt-in via AUTO_CLEANUP_ON_STARTUP)
    cleanup_mode = os.getenv("AUTO_CLEANUP_ON_STARTUP", "false").lower()
    if cleanup_mode == "false":
        return

    if cleanup_mode == "all":
        # Remove ALL workspace directories (fresh start)
        logger.info(
            f"Full cleanup enabled, removing {len(workspace_dirs)} workspace(s)..."
        )
        cleaned = 0
        for conn_dir in workspace_dirs:
            try:
                _remove_workspace(conn_dir)
                cleaned += 1
                logger.info(f"Removed workspace: {conn_dir.name}")
            except Exception as e:
                logger.warning(f"Failed to remove workspace {conn_dir.name}: {e}")
        logger.info(f"Full cleanup: removed {cleaned} workspace(s)")
        return

    # cleanup_mode == "true": retention-based cleanup
    max_age_days = int(os.getenv("WORKSPACE_MAX_AGE_DAYS", "30"))
    logger.info(
        f"Retention cleanup enabled (WORKSPACE_MAX_AGE_DAYS={max_age_days}), "
        f"scanning {len(workspace_dirs)} workspace(s)..."
    )
    cutoff = utc_now() - timedelta(days=max_age_days)
    cleaned = 0

    for conn_dir in workspace_dirs:
        metadata_file = conn_dir / "metadata.json"
        if not metadata_file.exists():
            # No metadata — orphaned directory, remove it
            try:
                _remove_workspace(conn_dir)
                cleaned += 1
                logger.info(f"Removed orphaned workspace directory: {conn_dir.name}")
            except Exception as e:
                logger.warning(
                    f"Failed to remove orphaned directory {conn_dir.name}: {e}"
                )
            continue

        # Check workspace age from metadata
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            workspace = metadata.get("workspace", {})
            updated_at = workspace.get("updated_at")
            if updated_at:
                last_update = parse_timestamp(updated_at)
                if last_update < cutoff:
                    _remove_workspace(conn_dir)
                    cleaned += 1
                    age_days = (utc_now() - last_update).days
                    logger.info(
                        f"Removed stale workspace: {conn_dir.name} "
                        f"(last updated {age_days} days ago, max {max_age_days})"
                    )
        except Exception as e:
            logger.debug(f"Skipping workspace {conn_dir.name}: {e}")

    if cleaned > 0:
        logger.info(f"Retention cleanup: removed {cleaned} stale workspace(s)")
    else:
        logger.info("Retention cleanup: all workspaces within retention period")


def main():
    """Start the OrionBelt Analytics MCP server."""
    try:
        # Validate configuration before anything else
        config_manager.validate_config()

        # Setup signal handlers for graceful shutdown
        setup_signal_handlers()

        # Register cleanup for session resources on exit
        atexit.register(cleanup_server)

        # Clean up temporary files from previous runs
        cleanup_tmp_folder()

        # Print startup information
        print_startup_info()

        transport_name = "streamable-http" if config.mcp_transport == "http" else "SSE"
        logger.info(
            f"🚀 Starting OrionBelt Analytics v{__version__} MCP server with {transport_name} transport..."
        )
        logger.info("📡 Server ready for MCP protocol messages")

        mcp.run(
            transport=config.mcp_transport,
            host=config.mcp_server_host,
            port=config.mcp_server_port,
        )

    except KeyboardInterrupt:
        logger.info("⏹️  Server stopped by user (Ctrl+C)")
    except asyncio.CancelledError:
        # Gracefully handle cancelled tasks during shutdown
        logger.debug("Async tasks cancelled during shutdown (expected)")
    except Exception as e:
        # Only log unexpected exceptions
        if "CancelledError" not in str(type(e).__name__):
            logger.error(f"❌ Critical server error: {type(e).__name__}: {e}")
            logger.error("Please check your configuration and try again")
            return 1
    finally:
        logger.info("✅ Server shutdown complete")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

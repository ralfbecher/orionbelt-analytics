"""Server session state and per-request helpers.

Holds the connection/session lifecycle that used to live inline in
``src/main.py``: the :class:`ServerState` registry (per-session isolation +
idle eviction) and the context helpers the tool layer hands to handlers
(session accessors, error responses, Oxigraph store init).

Kept free of any MCP tool/resource registration so ``main.py`` can stay a thin
registration layer that imports from here.
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import timedelta
from typing import Any, cast

from fastmcp import Context
from pydantic import BaseModel

from .database_manager import DatabaseManager
from .obqc_validator import OBQCValidator
from .ontology_generator import OntologyGenerator
from .oxigraph_store import OXIGRAPH_AVAILABLE, OxigraphStoreManager
from .paths import ensure_output_dir, get_connection_dir, get_oxigraph_store_dir
from .session import SessionData
from .utils import utc_now

logger = logging.getLogger(__name__)


def get_session_id(ctx: Context) -> str:
    """Get a unique session identifier from context."""
    if hasattr(ctx, "session_id") and ctx.session_id:
        return str(ctx.session_id)
    if hasattr(ctx, "session") and ctx.session:
        return f"session_{id(ctx.session)}"
    logger.warning("Could not determine session ID from context, using default_session")
    return "default_session"


def _get_connection_fingerprint(db_manager: DatabaseManager) -> str:
    """Generate unique fingerprint for current database connection."""
    conn_info = db_manager.connection_info
    if not conn_info:
        return "no_connection"

    fingerprint_data = (
        f"{conn_info.get('database_type', '')}://"
        f"{conn_info.get('host', '')}:{conn_info.get('port', '')}/"
        f"{conn_info.get('database', '')}"
        f"@{conn_info.get('schema', '')}"
    )
    return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]


def _calculate_schema_hash(tables_info: list[Any]) -> str:
    """Calculate deterministic hash of schema structure."""
    schema_structure: dict[str, list[dict[str, Any]]] = {"tables": []}

    sorted_tables = sorted(tables_info, key=lambda t: t.name)
    for table in sorted_tables:
        table_data = {
            "name": table.name,
            "schema": table.schema,
            "columns": [],
            "primary_keys": sorted(table.primary_keys) if table.primary_keys else [],
            "foreign_keys": [],
        }

        sorted_columns = sorted(table.columns, key=lambda c: c.name)
        for col in sorted_columns:
            table_data["columns"].append(
                {
                    "name": col.name,
                    "data_type": col.data_type,
                    "nullable": col.is_nullable,
                }
            )

        if table.foreign_keys:
            sorted_fks = sorted(table.foreign_keys, key=lambda f: f["column"])
            for fk in sorted_fks:
                table_data["foreign_keys"].append(
                    {
                        "column": fk["column"],
                        "referenced_table": fk["referenced_table"],
                        "referenced_column": fk["referenced_column"],
                    }
                )

        schema_structure["tables"].append(table_data)

    json_str = json.dumps(schema_structure, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _clear_session_state(
    session: SessionData, reason: str = "connection change"
) -> None:
    """Clear all session state caches and indexes."""
    logger.info(f"Clearing session state ({reason})")

    session.clear_schema_cache()

    # Clear all per-schema state (ontology for every schema)
    session.clear_all_schema_states()

    # Clear connection-scoped state
    session.graphrag_manager = None
    session.graphrag_initialized = False
    session.oxigraph_store = None
    session.oxigraph_initialized = False

    logger.info("Session state cleared")


class ServerState:
    """Manages server state with per-session isolation and idle eviction."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._eviction_task: asyncio.Task[None] | None = None

    @property
    def session_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    def get_session(self, session_id: str) -> SessionData:
        """Get or create session data for a given session ID."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionData()
            logger.debug(f"Created new session: {session_id}")
        session = self._sessions[session_id]
        session.touch()
        self._ensure_eviction_task()
        return session

    def get_ontology_generator(
        self, base_uri: str = "http://example.com/ontology/"
    ) -> OntologyGenerator:
        """Create a new ontology generator instance."""
        return OntologyGenerator(base_uri=base_uri)

    def _cancel_init_tasks(self, session_id: str) -> list["asyncio.Task[Any]"]:
        """Request cancellation of a session's background init tasks.

        Cancellation is only *scheduled* here -- a cancelled task keeps running
        until it next reaches a suspension point. Callers that can await should
        use :meth:`aclose_session`, which waits before tearing down the
        resources those tasks are still touching.

        Args:
            session_id: Session whose tasks should be cancelled.

        Returns:
            The tasks that were still running, for the caller to await.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return []

        pending = [t for t in session.graphrag.init_tasks if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            logger.debug(
                f"Cancelled {len(pending)} pending GraphRAG init task(s) "
                f"for session {session_id}"
            )
        return pending

    def _release_session(self, session_id: str) -> None:
        """Close a session's resources and drop it. Assumes tasks are done."""
        session = self._sessions.get(session_id)
        if session is None:
            return

        if session.db_manager:
            try:
                session.db_manager.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting db for session {session_id}: {e}")
        if session.rdf_store.oxigraph_store:
            try:
                session.rdf_store.oxigraph_store.close()
            except Exception as e:
                logger.warning(f"Error closing Oxigraph for session {session_id}: {e}")

        del self._sessions[session_id]
        logger.debug(f"Cleaned up session: {session_id}")

    async def aclose_session(self, session_id: str) -> None:
        """Clean up a session, waiting for its background tasks to stop first.

        Preferred over :meth:`cleanup_session` wherever a loop is running. A
        background init task can be mid ``save_state`` or mid metadata write;
        closing the Oxigraph store and dropping the session out from under it
        is how a cancelled-but-not-yet-stopped task ends up writing to a closed
        store.

        Args:
            session_id: Session to close.
        """
        pending = self._cancel_init_tasks(session_id)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._release_session(session_id)

    def cleanup_session(self, session_id: str) -> None:
        """Clean up a session from synchronous context.

        Best-effort: cancellation cannot be awaited here, so a background task
        may still be running when the store closes. Used from the atexit hook,
        where the loop is already gone. Prefer :meth:`aclose_session`.

        Args:
            session_id: Session to close.
        """
        self._cancel_init_tasks(session_id)
        self._release_session(session_id)

    def cleanup(self) -> None:
        """Clean up all resources from synchronous context (atexit)."""
        if self._eviction_task and not self._eviction_task.done():
            self._eviction_task.cancel()
            logger.debug("Cancelled session eviction task")
        for session_id in list(self._sessions.keys()):
            self.cleanup_session(session_id)

    # --- Idle eviction ---

    def _ensure_eviction_task(self) -> None:
        """Lazily start the eviction background task if not already running."""
        if self._eviction_task is not None and not self._eviction_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._eviction_task = loop.create_task(self._eviction_loop())
            logger.info("Started session eviction background task")
        except RuntimeError:
            pass  # No event loop (e.g. tests or sync context)

    async def _eviction_loop(self) -> None:
        """Periodically scan for and evict idle sessions."""
        from .config import config_manager

        config = config_manager.get_server_config()
        idle_timeout = config.session_idle_timeout
        scan_interval = config.session_scan_interval

        if idle_timeout <= 0:
            logger.info("Session idle eviction disabled (timeout=0)")
            return

        logger.info(
            f"Session eviction active: timeout={idle_timeout}s, "
            f"scan_interval={scan_interval}s"
        )

        while True:
            try:
                await asyncio.sleep(scan_interval)
                await self._evict_idle_sessions(idle_timeout)
            except asyncio.CancelledError:
                logger.info("Session eviction task cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in session eviction loop: {e}")
                await asyncio.sleep(scan_interval)

    async def _evict_idle_sessions(self, idle_timeout: int) -> None:
        """Scan sessions and evict those idle beyond the timeout.

        Awaits each session's background tasks before releasing its resources,
        so an in-flight GraphRAG init cannot outlive the store it writes to.
        """
        now = utc_now()
        cutoff = now - timedelta(seconds=idle_timeout)

        to_evict = []
        for session_id, session in self._sessions.items():
            if session.last_activity < cutoff:
                idle_secs = (now - session.last_activity).total_seconds()
                to_evict.append((session_id, idle_secs))

        total = len(self._sessions)
        evicting = len(to_evict)
        if total > 0:
            logger.debug(
                f"Session scan: {total} total, {evicting} idle "
                f"(timeout={idle_timeout}s)"
            )

        for session_id, idle_secs in to_evict:
            logger.info(
                f"Evicting idle session {session_id} "
                f"(idle {idle_secs:.0f}s, timeout={idle_timeout}s)"
            )
            await self.aclose_session(session_id)

        if evicting > 0:
            logger.info(
                f"Evicted {evicting} idle session(s). "
                f"Remaining: {len(self._sessions)}"
            )


# Global server state
_server_state = ServerState()


def get_session_data(ctx: Context) -> SessionData:
    """Get session data for the current context."""
    session_id = get_session_id(ctx)
    return _server_state.get_session(session_id)


def get_session_db_manager(ctx: Context) -> DatabaseManager:
    """Get or create a DatabaseManager for the current session."""
    session = get_session_data(ctx)
    if session.db_manager is None:
        session.db_manager = DatabaseManager()
        logger.debug(f"Created new DatabaseManager for session: {get_session_id(ctx)}")
    return cast(DatabaseManager, session.db_manager)


def get_session_obqc_validator(ctx: Context) -> OBQCValidator | None:
    """Get or create OBQC validator for the current session."""
    session = get_session_data(ctx)

    has_generated_ontology = session.ontology_file is not None
    has_loaded_ontology = session.loaded_ontology is not None

    if not has_generated_ontology and not has_loaded_ontology:
        return None

    if session.obqc_validator is None:
        session.obqc_validator = OBQCValidator()

        base_uri = os.getenv("ONTOLOGY_BASE_URI", "http://example.com/ontology/")
        ontology_generator = OntologyGenerator(base_uri)

        if session.ontology_file is not None:
            conn_dir = (
                get_connection_dir(session.connection_id)
                if session.connection_id
                else ensure_output_dir()
            )
            ontology_path = conn_dir / session.ontology_file
            if ontology_path.exists():
                ontology_generator.load_from_file(str(ontology_path))
                logger.debug(
                    f"OBQC loaded ontology from session file: {session.ontology_file}"
                )
        elif session.loaded_ontology is not None:
            ontology_generator.load_from_string(session.loaded_ontology)
            logger.debug(
                f"OBQC loaded ontology from session's loaded ontology: {session.loaded_ontology_path}"
            )

        session.obqc_validator.load_ontology(ontology_generator.graph, base_uri)
        logger.debug(f"Initialized OBQC validator for session: {get_session_id(ctx)}")

    # Registered outside the creation branch, and on every call: views may be
    # discovered after the validator was built, and without them every query
    # against a view is rejected for a table the ontology was never meant to
    # describe. load_views_from_definitions() re-parses only on change.
    # Registered unconditionally, including when the list is empty: an empty
    # set has to be able to *clear* a previously registered one. Gating on
    # truthiness let views survive a reconnect or a rediscovery that found
    # none, leaving OBQC accepting objects from a schema no longer loaded.
    views = session.get_all_cached_views()
    db_type = "postgresql"
    if session.db_manager is not None:
        db_type = session.db_manager.connection_info.get("type", "postgresql")
    session.obqc_validator.load_views_from_definitions(
        {view.name: view.definition for view in views}, dialect=db_type
    )

    return cast(OBQCValidator | None, session.obqc_validator)


def get_session_safe_filename(ctx: Context, prefix: str, suffix: str = "") -> str:
    """Generate a connection-safe filename to prevent cross-database file collisions."""
    session = get_session_data(ctx)
    connection_prefix = (
        session.connection_id[:8]
        if session.connection_id and len(session.connection_id) >= 8
        else "default"
    )
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S%f")
    if suffix:
        return f"{prefix}_{connection_prefix}_{suffix}_{timestamp}"
    return f"{prefix}_{connection_prefix}_{timestamp}"


def load_ontology_from_session(ctx: Context) -> tuple[OntologyGenerator, str]:
    """Load ontology from session state."""
    session = get_session_data(ctx)
    filename = session.ontology_file
    if not filename:
        raise ValueError(
            "No ontology file in session state. Run generate_ontology first."
        )

    conn_dir = (
        get_connection_dir(session.connection_id)
        if session.connection_id
        else ensure_output_dir()
    )
    ontology_path = conn_dir / filename

    if not ontology_path.exists():
        raise ValueError(f"Ontology file not found: {filename}")

    generator = _server_state.get_ontology_generator()
    generator.load_from_file(str(ontology_path))

    return generator, filename


class ErrorResponse(BaseModel):
    """Standardized error response format."""

    error: str
    error_type: str = "unknown"
    details: str | None = None


def create_error_response(
    error_msg: str, error_type: str = "unknown", details: str | None = None
) -> dict[str, Any]:
    """Create a standardized error response.

    DEPRECATED: Use exceptions from src.exceptions instead.
    Example: ConnectionError("message").to_response()

    This function is kept for backward compatibility but new code should
    use the exception hierarchy in src/exceptions.py.
    """
    response = ErrorResponse(error=error_msg, error_type=error_type, details=details)
    return response.model_dump()


def get_oxigraph_store(ctx: Context) -> OxigraphStoreManager | None:
    """Get or initialize connection-scoped Oxigraph store for the session."""
    session = get_session_data(ctx)

    if not OXIGRAPH_AVAILABLE:
        logger.warning("pyoxigraph not available - SPARQL features disabled")
        return None

    if session.oxigraph_store is None:
        try:
            store_path = get_oxigraph_store_dir(connection_id=session.connection_id)
            session.oxigraph_store = OxigraphStoreManager(store_path=store_path)
            session.oxigraph_initialized = True

            if session.connection_id:
                logger.info(
                    f"Initialized connection-scoped Oxigraph store at: {store_path}"
                )
            else:
                logger.info(
                    f"Initialized Oxigraph store at: {store_path} (legacy mode)"
                )

        except Exception as e:
            logger.error(f"Failed to initialize Oxigraph store: {e}")
            return None

    return cast(OxigraphStoreManager | None, session.oxigraph_store)

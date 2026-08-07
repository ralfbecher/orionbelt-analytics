"""Session state management for OrionBelt Analytics.

Provides decomposed, focused state objects for per-session isolation.
Each MCP session gets its own SessionData instance containing:
- ConnectionState: database connection tracking
- SchemaCache: cached schema analysis results (multi-schema)
- SchemaState: per-schema ontology + GraphRAG state (multi-schema)
- RDFStoreState: Oxigraph RDF store (connection-scoped, multi-schema via named graphs)
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from .utils import utc_now

logger = logging.getLogger(__name__)


class ConnectionState:
    """Database connection tracking."""

    def __init__(self) -> None:
        self.connection_id: str | None = None
        self.connected_at: datetime | None = None
        self.db_manager: Any | None = None  # DatabaseManager (avoid circular import)


class OntologyState:
    """Ontology generation and loading state."""

    def __init__(self) -> None:
        self.ontology_file: str | None = None
        self.r2rml_file: str | None = None
        self.loaded_ontology: str | None = None  # TTL content
        self.loaded_ontology_path: str | None = None  # File path
        self.obqc_validator: Any | None = None  # OBQCValidator (avoid circular import)
        self.ontology_enriched: bool = False  # True after semantic names applied


class SchemaCache:
    """Cached schema analysis results (multi-schema capable)."""

    def __init__(self) -> None:
        self._cached_schema: dict[str, list[Any]] | None = (
            None  # schema_name -> List[TableInfo]
        )
        # Views are cached separately, not folded into _cached_schema, because
        # every consumer of that dict feeds the ontology -- and views are
        # deliberately kept out of it. They exist here only to reach GraphRAG.
        self._cached_views: dict[str, list[Any]] | None = (
            None  # schema_name -> List[ViewInfo]
        )
        self._last_analyzed_schema: str | None = None

    def cache_schema_analysis(self, schema_name: str, tables_info: list[Any]) -> None:
        """Cache schema analysis results for reuse."""
        if self._cached_schema is None:
            self._cached_schema = {}
        cache_key = schema_name or "_default_"
        self._cached_schema[cache_key] = tables_info
        self._last_analyzed_schema = schema_name
        logger.debug(
            f"Cached schema analysis for '{cache_key}': {len(tables_info)} tables"
        )

    def get_cached_schema(self, schema_name: str) -> list[Any] | None:
        """Get cached schema analysis results if available."""
        if self._cached_schema is None:
            return None
        cache_key = schema_name or "_default_"
        cached = self._cached_schema.get(cache_key)
        if cached:
            logger.debug(f"Using cached schema for '{cache_key}': {len(cached)} tables")
        return cached

    def cache_views(self, schema_name: str, views_info: list[Any]) -> None:
        """Cache discovered views for reuse by GraphRAG indexing."""
        if self._cached_views is None:
            self._cached_views = {}
        cache_key = schema_name or "_default_"
        self._cached_views[cache_key] = views_info
        logger.debug(f"Cached views for '{cache_key}': {len(views_info)} views")

    def get_cached_views(self, schema_name: str) -> list[Any]:
        """Get cached views, or an empty list when none were discovered."""
        if self._cached_views is None:
            return []
        return self._cached_views.get(schema_name or "_default_", [])

    def get_all_cached_views(self) -> list[Any]:
        """Every discovered view across all schemas on this connection.

        OBQC is per session, not per schema, so it needs the union: a query may
        name a view from a schema discovered earlier.
        """
        if self._cached_views is None:
            return []
        return [view for views in self._cached_views.values() for view in views]

    def clear(self, schema_name: str | None = None) -> None:
        """Clear cached schema analysis.

        Args:
            schema_name: If provided, clear only that schema's cache.
                         If None, clear all cached schemas.
        """
        if schema_name is not None and self._cached_schema is not None:
            cache_key = schema_name or "_default_"
            self._cached_schema.pop(cache_key, None)
            if self._last_analyzed_schema == schema_name:
                self._last_analyzed_schema = None
            logger.debug(f"Cleared schema cache for '{cache_key}'")
        else:
            self._cached_schema = None
            self._last_analyzed_schema = None
            logger.debug("Cleared all schema caches")

    def get_last_analyzed_schema(self) -> str | None:
        """Get the name of the last analyzed schema."""
        return self._last_analyzed_schema


class GraphRAGState:
    """GraphRAG integration state with background init tracking."""

    def __init__(self) -> None:
        self.graphrag_manager: Any | None = None  # GraphRAGManager
        self.graphrag_initialized: bool = False
        # Every in-flight background init, not just the newest. A single slot
        # lost earlier tasks whenever discover_schema overlapped -- normal in
        # the accumulative multi-schema flow -- leaving them running against a
        # session that teardown had already finished with.
        self.init_tasks: set[asyncio.Task[Any]] = set()

    def track_init_task(self, task: "asyncio.Task[Any]") -> None:
        """Register a background init task and forget it once it finishes.

        Args:
            task: The task to track until completion.
        """
        self.init_tasks.add(task)
        task.add_done_callback(self.init_tasks.discard)


class RDFStoreState:
    """Oxigraph RDF store state with Future-based init tracking.

    Connection-scoped (not per-schema). Oxigraph supports multiple schemas
    via named graphs within a single store.
    """

    def __init__(self) -> None:
        self.oxigraph_store: Any | None = None  # OxigraphStoreManager
        self.oxigraph_initialized: bool = False
        self._init_task: asyncio.Task[Any] | None = None


class SchemaState:
    """Per-schema state for ontology data.

    Each analyzed schema gets its own SchemaState so that switching
    schemas does not destroy the previous schema's ontology state.
    GraphRAG is connection-scoped (accumulative across schemas).
    """

    def __init__(self, schema_name: str):
        self.schema_name = schema_name
        self.schema_file: str | None = None
        self.ontology = OntologyState()


class SessionData:
    """Per-session data storage with multi-schema support.

    Connection-level state (ConnectionState, RDFStoreState) is shared
    across all schemas. Per-schema state (ontology, GraphRAG) is isolated
    in SchemaState instances keyed by schema name.
    """

    def __init__(self) -> None:
        self.connection = ConnectionState()
        self.schema_cache = SchemaCache()
        self.rdf_store = RDFStoreState()

        # Connection-scoped state (shared across schemas)
        self.graphrag = GraphRAGState()

        # Multi-schema state (ontology is per-schema)
        self._schema_states: dict[str, SchemaState] = {}
        self._current_schema: str | None = None

        # Activity tracking for idle eviction
        self.created_at: datetime = utc_now()
        self.last_activity: datetime = utc_now()

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = utc_now()

    # --- Multi-schema management ---

    @property
    def current_schema(self) -> str | None:
        """Name of the currently active schema."""
        return self._current_schema

    def set_current_schema(self, schema_name: str) -> "SchemaState":
        """Set the active schema, creating a SchemaState if needed.

        Args:
            schema_name: Schema name to activate

        Returns:
            The SchemaState for the activated schema
        """
        key = schema_name or "default"
        if key not in self._schema_states:
            self._schema_states[key] = SchemaState(key)
            logger.debug(f"Created SchemaState for '{key}'")
        self._current_schema = key
        logger.debug(f"Current schema set to '{key}'")
        return self._schema_states[key]

    def get_schema_state(
        self, schema_name: str | None = None
    ) -> Optional["SchemaState"]:
        """Get SchemaState for a specific or the current schema.

        Args:
            schema_name: Schema name to look up. If None, uses current schema.

        Returns:
            SchemaState if found, None otherwise.
        """
        key = schema_name or self._current_schema
        if key is None:
            return None
        key = key or "default"
        return self._schema_states.get(key)

    def get_or_create_schema_state(
        self, schema_name: str | None = None
    ) -> "SchemaState":
        """Get or create SchemaState for a specific or the current schema.

        If schema_name is None and no current schema is set, uses "default".
        """
        key = schema_name or self._current_schema or "default"
        if key not in self._schema_states:
            self._schema_states[key] = SchemaState(key)
        return self._schema_states[key]

    @property
    def schema_names(self) -> list[str]:
        """List of all schema names with active state."""
        return list(self._schema_states.keys())

    def clear_all_schema_states(self) -> None:
        """Clear all per-schema state (ontology, GraphRAG) for all schemas."""
        self._schema_states.clear()
        self._current_schema = None
        logger.debug("Cleared all schema states")

    # --- Convenience accessors ---
    # Delegate to the current schema's state for backward compatibility.
    # Handler code can also access schema state directly via
    # get_schema_state() for explicit schema targeting.

    @property
    def _current_schema_state(self) -> Optional["SchemaState"]:
        """Internal helper: get current SchemaState or None."""
        if self._current_schema is None:
            return None
        return self._schema_states.get(self._current_schema)

    def _ensure_schema_state(self) -> "SchemaState":
        """Internal helper: get or create current SchemaState."""
        return self.get_or_create_schema_state()

    # Connection properties (session-scoped, unchanged)

    @property
    def db_manager(self) -> Any | None:
        return self.connection.db_manager

    @db_manager.setter
    def db_manager(self, value: Any | None) -> None:
        self.connection.db_manager = value

    @property
    def connection_id(self) -> str | None:
        return self.connection.connection_id

    @connection_id.setter
    def connection_id(self, value: str | None) -> None:
        self.connection.connection_id = value

    @property
    def connected_at(self) -> datetime | None:
        return self.connection.connected_at

    @connected_at.setter
    def connected_at(self, value: datetime | None) -> None:
        self.connection.connected_at = value

    # Ontology properties (per-schema via current schema)

    @property
    def ontology_file(self) -> str | None:
        ss = self._current_schema_state
        return ss.ontology.ontology_file if ss else None

    @ontology_file.setter
    def ontology_file(self, value: str | None) -> None:
        self._ensure_schema_state().ontology.ontology_file = value

    @property
    def r2rml_file(self) -> str | None:
        ss = self._current_schema_state
        return ss.ontology.r2rml_file if ss else None

    @r2rml_file.setter
    def r2rml_file(self, value: str | None) -> None:
        self._ensure_schema_state().ontology.r2rml_file = value

    @property
    def loaded_ontology(self) -> str | None:
        ss = self._current_schema_state
        return ss.ontology.loaded_ontology if ss else None

    @loaded_ontology.setter
    def loaded_ontology(self, value: str | None) -> None:
        self._ensure_schema_state().ontology.loaded_ontology = value

    @property
    def loaded_ontology_path(self) -> str | None:
        ss = self._current_schema_state
        return ss.ontology.loaded_ontology_path if ss else None

    @loaded_ontology_path.setter
    def loaded_ontology_path(self, value: str | None) -> None:
        self._ensure_schema_state().ontology.loaded_ontology_path = value

    @property
    def obqc_validator(self) -> Any | None:
        ss = self._current_schema_state
        return ss.ontology.obqc_validator if ss else None

    @obqc_validator.setter
    def obqc_validator(self, value: Any | None) -> None:
        self._ensure_schema_state().ontology.obqc_validator = value

    @property
    def ontology_enriched(self) -> bool:
        ss = self._current_schema_state
        return ss.ontology.ontology_enriched if ss else False

    @ontology_enriched.setter
    def ontology_enriched(self, value: bool) -> None:
        self._ensure_schema_state().ontology.ontology_enriched = value

    # Schema file (per-schema via current schema)

    @property
    def schema_file(self) -> str | None:
        ss = self._current_schema_state
        return ss.schema_file if ss else None

    @schema_file.setter
    def schema_file(self, value: str | None) -> None:
        self._ensure_schema_state().schema_file = value

    # GraphRAG properties (connection-scoped, accumulative across schemas)

    @property
    def graphrag_manager(self) -> Any | None:
        return self.graphrag.graphrag_manager

    @graphrag_manager.setter
    def graphrag_manager(self, value: Any | None) -> None:
        self.graphrag.graphrag_manager = value

    @property
    def graphrag_initialized(self) -> bool:
        return self.graphrag.graphrag_initialized

    @graphrag_initialized.setter
    def graphrag_initialized(self, value: bool) -> None:
        self.graphrag.graphrag_initialized = value

    # RDF Store properties (connection-scoped, unchanged)

    @property
    def oxigraph_store(self) -> Any | None:
        return self.rdf_store.oxigraph_store

    @oxigraph_store.setter
    def oxigraph_store(self, value: Any | None) -> None:
        self.rdf_store.oxigraph_store = value

    @property
    def oxigraph_initialized(self) -> bool:
        return self.rdf_store.oxigraph_initialized

    @oxigraph_initialized.setter
    def oxigraph_initialized(self, value: bool) -> None:
        self.rdf_store.oxigraph_initialized = value

    # --- Delegated methods ---

    def cache_schema_analysis(self, schema_name: str, tables_info: list[Any]) -> None:
        """Cache schema analysis results for reuse."""
        self.schema_cache.cache_schema_analysis(schema_name, tables_info)

    def get_cached_schema(self, schema_name: str) -> list[Any] | None:
        """Get cached schema analysis results if available."""
        return self.schema_cache.get_cached_schema(schema_name)

    def cache_views(self, schema_name: str, views_info: list[Any]) -> None:
        """Cache discovered views for reuse by GraphRAG indexing."""
        self.schema_cache.cache_views(schema_name, views_info)

    def get_cached_views(self, schema_name: str) -> list[Any]:
        """Get cached views, or an empty list when none were discovered."""
        return self.schema_cache.get_cached_views(schema_name)

    def get_all_cached_views(self) -> list[Any]:
        """Every discovered view across all schemas on this connection."""
        return self.schema_cache.get_all_cached_views()

    def clear_schema_cache(self, schema_name: str | None = None) -> None:
        """Clear cached schema analysis.

        Args:
            schema_name: If provided, clear only that schema. If None, clear all.
        """
        self.schema_cache.clear(schema_name)

    def get_last_analyzed_schema(self) -> str | None:
        """Get the name of the last analyzed schema."""
        return self.schema_cache.get_last_analyzed_schema()

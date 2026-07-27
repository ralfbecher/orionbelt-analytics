"""Snowflake database driver."""

import logging
from collections.abc import Callable
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError, OperationalError, SQLAlchemyError
from sqlalchemy.pool import QueuePool

from ..constants import (
    CONNECTION_TIMEOUT,
    DEFAULT_SAMPLE_LIMIT,
    MAX_SAMPLE_LIMIT,
    MIN_SAMPLE_LIMIT,
    SNOWFLAKE_SYSTEM_SCHEMAS,
)
from ..database_manager import ColumnInfo, TableInfo
from ..serialization import serialize_rows
from .base import DatabaseDriver

logger = logging.getLogger(__name__)


class SnowflakeDriver(DatabaseDriver):
    """Snowflake-specific database operations."""

    db_type = "snowflake"

    def __init__(self, pool_size: int = 5, max_overflow: int = 10):
        self.engine: Engine | None = None
        self.metadata: MetaData | None = None
        self._pool_size = pool_size
        self._max_overflow = max_overflow

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, **params: Any) -> bool:
        """Connect to Snowflake.

        Expected params: account, username, password, warehouse, database,
                         schema (default 'PUBLIC'), role (default 'PUBLIC').
        """
        account = params["account"]
        username = params["username"]
        password = params["password"]
        warehouse = params["warehouse"]
        database = params["database"]
        schema = params.get("schema", "PUBLIC")
        role = params.get("role", "PUBLIC")

        try:
            if not all([account, username, warehouse, database]):
                logger.error("Missing required Snowflake connection parameters")
                return False

            encoded_password = quote_plus(password) if password else ""

            connection_string = (
                f"snowflake://{username}:{encoded_password}@{account}/"
                f"{database}/{schema}?warehouse={warehouse}&role={role}"
            )

            logger.info(
                f"Connecting to Snowflake with account: {account}, database: {database}, "
                f"warehouse: {warehouse}, schema: {schema}"
            )
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=CONNECTION_TIMEOUT,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
                connect_args={
                    "application": "orionbelt-analytics",
                    "network_timeout": CONNECTION_TIMEOUT,
                },
            )
            self.metadata = MetaData()

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            logger.info(
                f"Connected to Snowflake database: {database} (warehouse: {warehouse})"
            )
            return True

        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(
                f"Failed to connect to Snowflake {account}/{database}: "
                f"{type(e).__name__}: {e}"
            )
            self.engine = None
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error connecting to Snowflake: {type(e).__name__}: {e}"
            )
            self.engine = None
            return False

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        excluded_schemas = "', '".join(SNOWFLAKE_SYSTEM_SCHEMAS)
        query = text(
            f"""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('{excluded_schemas}')
            ORDER BY schema_name
        """
        )
        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get schemas: {e}")
            return []

    def get_tables(self, schema_name: str | None = None) -> list[str]:
        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                if schema_name:
                    query = text(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = :schema_name
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """
                    )
                    result = conn.execute(query, {"schema_name": schema_name})
                else:
                    query = text(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """
                    )
                    result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get tables: {e}")
            return []

    # ------------------------------------------------------------------
    # Snowflake-specific: prefetch schema constraints
    # ------------------------------------------------------------------

    def prefetch_schema_constraints(
        self,
        schema_name: str,
        connection_info: dict[str, Any],
        cache_get: Callable[..., Any],
        cache_store: Callable[..., Any],
        log_sql: Callable[..., Any],
    ) -> None:
        """Prefetch all PKs, FKs, and columns for a Snowflake schema at once.

        This avoids repeated ``SHOW PRIMARY KEYS`` / ``SHOW IMPORTED KEYS``
        queries for each table.

        Args:
            schema_name: Schema to prefetch.
            connection_info: Current connection info dict (used for database name).
            cache_get: Callable ``(key) -> Optional[data]`` for cache reads.
            cache_store: Callable ``(key, data) -> None`` for cache writes.
            log_sql: Callable ``(query, params?) -> None`` for SQL logging.
        """
        pk_cache_key = f"schema_pks:{schema_name}"
        fk_cache_key = f"schema_fks:{schema_name}"

        # Skip if already cached
        if cache_get(pk_cache_key) is not None:
            logger.debug(f"Schema constraints already cached for {schema_name}")
            return

        database_name = connection_info.get("database", "")
        if database_name:
            full_schema_path = f'"{database_name}"."{schema_name}"'
        else:
            full_schema_path = f'"{schema_name}"'

        logger.info(f"Prefetching PKs and FKs for schema {full_schema_path}")

        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                # --- Primary keys ---
                pk_by_table: dict[str, list[str]] = {}
                pk_success = False
                try:
                    pk_query = text(f"SHOW PRIMARY KEYS IN SCHEMA {full_schema_path}")
                    log_sql(str(pk_query))
                    result = conn.execute(pk_query)
                    for row in result.fetchall():
                        row_dict = (
                            row._asdict()
                            if hasattr(row, "_asdict")
                            else dict(row._mapping)
                        )
                        table = row_dict.get("table_name")
                        column = row_dict.get("column_name")
                        if table and column:
                            pk_by_table.setdefault(table, []).append(column)
                    logger.info(f"Prefetched PKs for {len(pk_by_table)} tables")
                    pk_success = True
                except Exception as e:
                    logger.warning(f"Failed to prefetch PKs: {e}")

                if pk_success:
                    cache_store(pk_cache_key, pk_by_table)

                # --- Foreign keys ---
                fk_by_table: dict[str, list[dict]] = {}
                fk_success = False
                try:
                    fk_query = text(f"SHOW IMPORTED KEYS IN SCHEMA {full_schema_path}")
                    log_sql(str(fk_query))
                    result = conn.execute(fk_query)
                    for row in result.fetchall():
                        row_dict = (
                            row._asdict()
                            if hasattr(row, "_asdict")
                            else dict(row._mapping)
                        )
                        fk_table = row_dict.get("fk_table_name")
                        pk_table = row_dict.get("pk_table_name")
                        pk_column = row_dict.get("pk_column_name")
                        pk_schema = row_dict.get("pk_schema_name")
                        fk_column = row_dict.get("fk_column_name")
                        fk_name = row_dict.get("fk_name")

                        if fk_table and pk_table and fk_column:
                            fk_by_table.setdefault(fk_table, []).append(
                                {
                                    "constrained_columns": [fk_column],
                                    "referred_schema": pk_schema,
                                    "referred_table": pk_table,
                                    "referred_columns": [pk_column],
                                    "name": fk_name,
                                }
                            )
                    logger.info(f"Prefetched FKs for {len(fk_by_table)} tables")
                    fk_success = True
                except Exception as e:
                    logger.warning(f"Failed to prefetch FKs: {e}")

                if fk_success:
                    cache_store(fk_cache_key, fk_by_table)

                # --- Columns ---
                cols_cache_key = f"schema_cols:{schema_name}"
                cols_by_table: dict[str, list[dict]] = {}
                cols_success = False
                try:
                    cols_query = text(
                        """
                        SELECT table_name, column_name, data_type,
                               character_maximum_length, numeric_precision,
                               numeric_scale, is_nullable, column_default,
                               is_identity, comment
                        FROM information_schema.columns
                        WHERE table_schema = :schema_name
                        ORDER BY table_name, ordinal_position
                    """
                    )
                    log_sql(str(cols_query))
                    result = conn.execute(cols_query, {"schema_name": schema_name})
                    for row in result.fetchall():
                        row_dict = (
                            row._asdict()
                            if hasattr(row, "_asdict")
                            else dict(row._mapping)
                        )
                        table = row_dict.get("table_name") or row_dict.get("TABLE_NAME")
                        if table:
                            cols_by_table.setdefault(table, []).append(
                                {
                                    "name": row_dict.get("column_name")
                                    or row_dict.get("COLUMN_NAME"),
                                    "type": row_dict.get("data_type")
                                    or row_dict.get("DATA_TYPE"),
                                    "nullable": (
                                        row_dict.get("is_nullable")
                                        or row_dict.get("IS_NULLABLE", "YES")
                                    ).upper()
                                    == "YES",
                                    "default": row_dict.get("column_default")
                                    or row_dict.get("COLUMN_DEFAULT"),
                                    "comment": row_dict.get("comment")
                                    or row_dict.get("COMMENT"),
                                }
                            )
                    logger.info(f"Prefetched columns for {len(cols_by_table)} tables")
                    cols_success = True
                except Exception as e:
                    logger.warning(f"Failed to prefetch columns: {e}")

                if cols_success:
                    cache_store(cols_cache_key, cols_by_table)

        except SQLAlchemyError as e:
            logger.error(f"Failed to prefetch schema metadata: {e}")

    def analyze_table(
        self,
        table_name: str,
        schema_name: str | None = None,
        cache_get: Callable[..., Any] | None = None,
        log_sql: Callable[..., Any] | None = None,
    ) -> TableInfo | None:
        """Analyze a Snowflake table.

        If ``cache_get`` is provided, uses prefetched constraint data.
        """
        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                inspector = inspect(self.engine)

                table_columns = None
                primary_keys: list[str] = []
                table_fks: list = []

                if schema_name and cache_get is not None:
                    # Try prefetched cache first
                    cols_cache_key = f"schema_cols:{schema_name}"
                    pk_cache_key = f"schema_pks:{schema_name}"
                    fk_cache_key = f"schema_fks:{schema_name}"

                    cached_cols = cache_get(cols_cache_key)
                    cached_pks = cache_get(pk_cache_key)
                    cached_fks = cache_get(fk_cache_key)

                    if cached_cols is not None:
                        table_columns = cached_cols.get(table_name) or cached_cols.get(
                            table_name.upper()
                        )
                        if table_columns:
                            logger.debug(
                                f"Using cached columns for {table_name}: "
                                f"{len(table_columns)} columns"
                            )
                        else:
                            logger.error(
                                f"Table {schema_name}.{table_name} not found in cache"
                            )
                            return None

                    if cached_pks is not None:
                        primary_keys = cached_pks.get(table_name, []) or cached_pks.get(
                            table_name.upper(), []
                        )
                        logger.debug(
                            f"Using cached PKs for {table_name}: {primary_keys}"
                        )
                    else:
                        table_pk = inspector.get_pk_constraint(
                            table_name, schema=schema_name
                        )
                        primary_keys = (
                            table_pk.get("constrained_columns", []) if table_pk else []
                        )

                    if cached_fks is not None:
                        table_fks = cached_fks.get(table_name, []) or cached_fks.get(
                            table_name.upper(), []
                        )
                        logger.debug(
                            f"Using cached FKs for {table_name}: "
                            f"{len(table_fks)} constraints"
                        )
                    else:
                        table_fks = inspector.get_foreign_keys(
                            table_name, schema=schema_name
                        )
                else:
                    # No cache - use inspector
                    if schema_name:
                        if not inspector.has_table(table_name, schema=schema_name):
                            logger.error(f"Table {schema_name}.{table_name} not found")
                            return None
                        table_columns = inspector.get_columns(
                            table_name, schema=schema_name
                        )
                        table_pk = inspector.get_pk_constraint(
                            table_name, schema=schema_name
                        )
                        table_fks = inspector.get_foreign_keys(
                            table_name, schema=schema_name
                        )
                    else:
                        if not inspector.has_table(table_name):
                            logger.error(f"Table {table_name} not found")
                            return None
                        table_columns = inspector.get_columns(table_name)
                        table_pk = inspector.get_pk_constraint(table_name)
                        table_fks = inspector.get_foreign_keys(table_name)
                    primary_keys = (
                        table_pk.get("constrained_columns", []) if table_pk else []
                    )

                # Fallback: if columns not from cache, fetch with query
                if table_columns is None and schema_name:
                    try:
                        cols_query = text(
                            """
                            SELECT column_name, data_type, is_nullable,
                                   column_default, comment
                            FROM information_schema.columns
                            WHERE table_schema = :schema_name
                              AND table_name = :table_name
                            ORDER BY ordinal_position
                        """
                        )
                        if log_sql:
                            log_sql(str(cols_query))
                        result = conn.execute(
                            cols_query,
                            {
                                "schema_name": schema_name,
                                "table_name": table_name,
                            },
                        )
                        rows = result.fetchall()
                        if not rows:
                            logger.error(f"Table {schema_name}.{table_name} not found")
                            return None
                        table_columns = []
                        for row in rows:
                            row_dict = (
                                row._asdict()
                                if hasattr(row, "_asdict")
                                else dict(row._mapping)
                            )
                            table_columns.append(
                                {
                                    "name": row_dict.get("column_name")
                                    or row_dict.get("COLUMN_NAME"),
                                    "type": row_dict.get("data_type")
                                    or row_dict.get("DATA_TYPE"),
                                    "nullable": (
                                        row_dict.get("is_nullable")
                                        or row_dict.get("IS_NULLABLE", "YES")
                                    ).upper()
                                    == "YES",
                                    "default": row_dict.get("column_default")
                                    or row_dict.get("COLUMN_DEFAULT"),
                                    "comment": row_dict.get("comment")
                                    or row_dict.get("COMMENT"),
                                }
                            )
                    except Exception as e:
                        logger.warning(
                            f"Direct column query failed, falling back to "
                            f"inspector: {e}"
                        )
                        table_columns = inspector.get_columns(
                            table_name, schema=schema_name
                        )

                if table_columns is None:
                    if schema_name:
                        if not inspector.has_table(table_name, schema=schema_name):
                            logger.error(f"Table {schema_name}.{table_name} not found")
                            return None
                        table_columns = inspector.get_columns(
                            table_name, schema=schema_name
                        )
                    else:
                        if not inspector.has_table(table_name):
                            logger.error(f"Table {table_name} not found")
                            return None
                        table_columns = inspector.get_columns(table_name)

                # Build columns / foreign_keys
                primary_keys_upper = [pk.upper() for pk in primary_keys]

                logger.info(
                    f"Table {schema_name}.{table_name}: PKs={primary_keys}, "
                    f"FKs={len(table_fks)} constraints"
                )
                if table_fks:
                    for fk in table_fks:
                        logger.info(f"  FK: {fk}")

                columns = []
                foreign_keys: list[dict[str, Any]] = []
                for col_info in table_columns:
                    column_name = col_info["name"]
                    is_pk = column_name.upper() in primary_keys_upper

                    fk_table = None
                    fk_column = None
                    is_fk = False
                    for fk in table_fks:
                        constrained_cols_upper = [
                            c.upper() for c in fk.get("constrained_columns", [])
                        ]
                        if column_name.upper() in constrained_cols_upper:
                            is_fk = True
                            fk_idx = constrained_cols_upper.index(column_name.upper())
                            fk_table = fk.get("referred_table")
                            referred_cols = fk.get("referred_columns", [])
                            fk_column = (
                                referred_cols[fk_idx]
                                if fk_idx < len(referred_cols)
                                else None
                            )
                            fk_schema = fk.get("referred_schema")
                            if fk_table:
                                foreign_keys.append(
                                    {
                                        "column": column_name,
                                        "referenced_table": fk_table,
                                        "referenced_column": fk_column,
                                        "referenced_schema": fk_schema,
                                    }
                                )
                                logger.debug(
                                    f"Added FK: {column_name} -> "
                                    f"{fk_schema}.{fk_table}.{fk_column}"
                                )
                            break

                    columns.append(
                        ColumnInfo(
                            name=column_name,
                            data_type=str(col_info["type"]),
                            is_nullable=col_info["nullable"],
                            is_primary_key=is_pk,
                            is_foreign_key=is_fk,
                            foreign_key_table=fk_table,
                            foreign_key_column=fk_column,
                            comment=col_info.get("comment"),
                        )
                    )

                return TableInfo(
                    name=table_name,
                    schema=schema_name or "public",
                    columns=columns,
                    primary_keys=primary_keys,
                    foreign_keys=foreign_keys,
                    comment=None,
                    row_count=None,
                    sample_data=None,
                )

        except SQLAlchemyError as e:
            logger.error(f"Failed to analyze table {table_name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Query validation & execution
    # ------------------------------------------------------------------

    def validate_sql_syntax(
        self, sql_query: str, validation_result: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                explain_sql = f"EXPLAIN {sql_query}"
                try:
                    result = conn.execute(text(explain_sql))
                    result.fetchall()
                    validation_result["is_valid"] = True
                except Exception as explain_error:
                    error_msg = str(explain_error)
                    validation_result["database_error"] = error_msg
                    validation_result["error"] = f"Snowflake syntax error: {error_msg}"
                    validation_result["error_type"] = "syntax_error"

                    if "does not exist" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "Object not found - check table/schema names and "
                            "ensure proper qualification (DATABASE.SCHEMA.TABLE)"
                        )
                    elif "sql compilation error" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "SQL compilation failed - review syntax and object references"
                        )
                    elif "invalid identifier" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "Invalid identifier - check column names and use "
                            "double quotes for case-sensitive names"
                        )
        except Exception as conn_error:
            validation_result[
                "error"
            ] = f"Database connection error during validation: {conn_error}"
            validation_result["error_type"] = "connection_error"

        return validation_result

    def execute_sql_query(self, sql_query: str, limit: int = 1000) -> dict[str, Any]:
        import time as time_mod

        result_data: dict[str, Any] = {
            "success": False,
            "data": [],
            "columns": [],
            "row_count": 0,
            "execution_time_ms": None,
            "error": None,
            "error_type": None,
            "warnings": [],
            "query_plan": None,
            "limit_applied": False,
        }

        try:
            start_time = time_mod.time()

            assert self.engine is not None
            with self.engine.connect() as conn:
                logger.info(f"\U0001f50d SNOWFLAKE SQL QUERY: {sql_query}")
                result = conn.execute(text(sql_query))

                try:
                    if result.returns_rows:
                        result_data["columns"] = list(result.keys())
                        try:
                            raw_rows = result.fetchall()
                        except Exception as fetch_error:
                            logger.error(f"Error fetching results: {fetch_error}")
                            try:
                                result.close()
                            except Exception as close_error:
                                # Swallowed on purpose: a cursor-close failure must never
                                # mask the real error being raised. Logged so it stays
                                # diagnosable instead of vanishing.
                                logger.debug(
                                    f"Ignoring cursor close failure: {close_error}"
                                )
                            raise

                        result_data["data"] = serialize_rows(
                            raw_rows, result_data["columns"]
                        )
                        result_data["row_count"] = len(result_data["data"])
                    else:
                        result_data["row_count"] = getattr(result, "rowcount", 0)
                finally:
                    try:
                        result.close()
                    except Exception as close_error:
                        # Swallowed on purpose: a cursor-close failure must never
                        # mask the real error being raised. Logged so it stays
                        # diagnosable instead of vanishing.
                        logger.debug(f"Ignoring cursor close failure: {close_error}")

                end_time = time_mod.time()
                result_data["execution_time_ms"] = round(
                    (end_time - start_time) * 1000, 2
                )
                result_data["success"] = True
                logger.info(
                    f"SQL query executed: {result_data['row_count']} rows "
                    f"in {result_data['execution_time_ms']}ms"
                )

        except SQLAlchemyError as e:
            result_data["error"] = str(e)
            result_data["error_type"] = "execution_error"
            logger.error(f"SQL execution failed: {e}")
        except Exception as e:
            result_data["error"] = f"Unexpected execution error: {e!s}"
            result_data["error_type"] = "internal_error"
            logger.error(f"Unexpected SQL execution error: {e}")

        return result_data

    def sample_table_data(
        self,
        table_name: str,
        schema_name: str | None = None,
        limit: int = DEFAULT_SAMPLE_LIMIT,
    ) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or limit < MIN_SAMPLE_LIMIT:
            limit = DEFAULT_SAMPLE_LIMIT
        elif limit > MAX_SAMPLE_LIMIT:
            limit = MAX_SAMPLE_LIMIT
            logger.warning(f"Sample limit capped at {MAX_SAMPLE_LIMIT}")

        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                if schema_name:
                    full_table_name = f'"{schema_name}"."{table_name}"'
                else:
                    full_table_name = f'"{table_name}"'

                query_str = f"SELECT * FROM {full_table_name} LIMIT :limit"
                params = {"limit": limit}
                logger.info(
                    f"\U0001f50d SNOWFLAKE SQL QUERY: {query_str} | PARAMS: {params}"
                )
                result = conn.execute(text(query_str), params)
                columns = list(result.keys())
                return serialize_rows(result.fetchall(), columns)

        except (SQLAlchemyError, ValueError) as e:
            logger.error(
                f"Failed to sample data from {table_name}: {type(e).__name__}: {e}"
            )
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error sampling {table_name}: {type(e).__name__}: {e}"
            )
            return []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        if not self.engine:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False

    def disconnect(self) -> None:
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.metadata = None
            logger.info("Snowflake connection closed")

"""ClickHouse database driver."""

import logging
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError, OperationalError, SQLAlchemyError
from sqlalchemy.pool import QueuePool

from ..constants import (
    CLICKHOUSE_SYSTEM_SCHEMAS,
    CONNECTION_TIMEOUT,
    DEFAULT_SAMPLE_LIMIT,
    MAX_SAMPLE_LIMIT,
    MIN_SAMPLE_LIMIT,
)
from ..database_manager import ColumnInfo, TableInfo
from ..serialization import serialize_rows
from .base import DatabaseDriver

logger = logging.getLogger(__name__)


class ClickHouseDriver(DatabaseDriver):
    """ClickHouse-specific database operations."""

    db_type = "clickhouse"

    def __init__(self, pool_size: int = 5, max_overflow: int = 10):
        self.engine: Engine | None = None
        self.metadata: MetaData | None = None
        self._pool_size = pool_size
        self._max_overflow = max_overflow

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, **params: Any) -> bool:
        """Connect to ClickHouse.

        Expected params: host, port, database, username, password,
                         protocol ('http'|'native'), secure (bool).
        """
        host = params["host"]
        port = params.get("port", 8123)
        database = params.get("database", "default")
        username = params.get("username", "default")
        password = params.get("password", "")
        protocol = params.get("protocol", "http")
        secure = params.get("secure", False)

        try:
            if not all([host, database]):
                logger.error("Missing required ClickHouse connection parameters")
                return False

            encoded_password = quote_plus(password) if password else ""
            encoded_username = quote_plus(username)

            if protocol == "native":
                scheme = "clickhouse+native"
            elif secure:
                scheme = "clickhouse+https"
            else:
                scheme = "clickhouse+http"

            connection_string = (
                f"{scheme}://{encoded_username}:{encoded_password}"
                f"@{host}:{port}/{database}"
            )

            logger.info(
                f"Connecting to ClickHouse at {host}:{port}/{database} via {scheme}"
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
            )
            self.metadata = MetaData()

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            logger.info(
                f"Connected to ClickHouse database: {database} at {host}:{port}"
            )
            return True

        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(
                f"Failed to connect to ClickHouse {host}:{port}/{database}: "
                f"{type(e).__name__}: {e}"
            )
            self.engine = None
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error connecting to ClickHouse: {type(e).__name__}: {e}"
            )
            self.engine = None
            return False

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        excluded_schemas = "', '".join(CLICKHOUSE_SYSTEM_SCHEMAS)
        query = text(f"""
            SELECT name
            FROM system.databases
            WHERE name NOT IN ('{excluded_schemas}')
            ORDER BY name
        """)
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
                    query = text("""
                        SELECT name
                        FROM system.tables
                        WHERE database = :schema_name
                          AND engine NOT IN ('View', 'MaterializedView')
                        ORDER BY name
                    """)
                    result = conn.execute(query, {"schema_name": schema_name})
                else:
                    # Use the connected database name
                    db_name = self._current_database or "default"
                    query = text("""
                        SELECT name
                        FROM system.tables
                        WHERE database = :db_name
                          AND engine NOT IN ('View', 'MaterializedView')
                        ORDER BY name
                    """)
                    result = conn.execute(query, {"db_name": db_name})
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get tables: {e}")
            return []

    @property
    def _current_database(self) -> str:
        """Return the database name from connection info, falling back to 'default'."""
        # This is set by the manager after connect
        return getattr(self, "_database_name", "default")

    def get_views(self, schema_name: str | None = None) -> dict[str, str | None]:
        # The inverse of the engine filter in get_tables. `as_select` holds the
        # view body; ClickHouse leaves it empty rather than NULL when absent.
        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                db_name = schema_name or self._current_database or "default"
                query = text("""
                    SELECT name, as_select
                    FROM system.tables
                    WHERE database = :db_name
                      AND engine IN ('View', 'MaterializedView')
                    ORDER BY name
                """)
                result = conn.execute(query, {"db_name": db_name})
                return {row[0]: (row[1] or None) for row in result.fetchall()}
        except SQLAlchemyError as e:
            logger.error(f"Failed to get views: {e}")
            return {}

    def analyze_table(
        self,
        table_name: str,
        schema_name: str | None = None,
    ) -> TableInfo | None:
        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                inspector = inspect(self.engine)

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

                # ClickHouse enrichment: engine, sorting_key, row count
                row_count = None
                table_comment = None
                if schema_name:
                    try:
                        ch_meta_query = text("""
                            SELECT engine, sorting_key, total_rows
                            FROM system.tables
                            WHERE database = :db AND name = :tbl
                        """)
                        ch_result = conn.execute(
                            ch_meta_query,
                            {"db": schema_name, "tbl": table_name},
                        )
                        ch_row = ch_result.fetchone()
                        if ch_row:
                            ch_dict = (
                                ch_row._asdict()
                                if hasattr(ch_row, "_asdict")
                                else dict(ch_row._mapping)
                            )
                            engine = ch_dict.get("engine", "")
                            sorting_key = ch_dict.get("sorting_key", "")
                            total_rows = ch_dict.get("total_rows")
                            table_comment = f"Engine: {engine}"
                            if sorting_key:
                                table_comment += f", ORDER BY: {sorting_key}"
                            if total_rows is not None:
                                row_count = int(total_rows)
                            # Use sorting_key as PKs if inspector returned empty
                            if not primary_keys and sorting_key:
                                primary_keys = [
                                    k.strip() for k in sorting_key.split(",")
                                ]
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch ClickHouse metadata for "
                            f"{table_name}: {e}"
                        )

                return TableInfo(
                    name=table_name,
                    schema=schema_name or "public",
                    columns=columns,
                    primary_keys=primary_keys,
                    foreign_keys=foreign_keys,
                    comment=table_comment,
                    row_count=row_count,
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
                    validation_result["error"] = f"ClickHouse syntax error: {error_msg}"
                    validation_result["error_type"] = "syntax_error"

                    if (
                        "unknown table" in error_msg.lower()
                        or "doesn't exist" in error_msg.lower()
                    ):
                        validation_result["suggestions"].append(
                            "Table not found - check database.table name qualification"
                        )
                    elif "syntax error" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "SQL syntax error - review query structure"
                        )
                    elif (
                        "unknown column" in error_msg.lower()
                        or "unknown identifier" in error_msg.lower()
                    ):
                        validation_result["suggestions"].append(
                            "Unknown column - check column names "
                            "(ClickHouse is case-sensitive)"
                        )
        except Exception as conn_error:
            validation_result["error"] = (
                f"Database connection error during validation: {conn_error}"
            )
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
                logger.info(f"\U0001f50d CLICKHOUSE SQL QUERY: {sql_query}")
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
                    f"\U0001f50d CLICKHOUSE SQL QUERY: {query_str} | PARAMS: {params}"
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
            logger.info("ClickHouse connection closed")

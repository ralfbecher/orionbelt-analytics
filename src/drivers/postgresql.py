"""PostgreSQL database driver."""

import logging
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
    POSTGRES_SYSTEM_SCHEMAS,
)
from ..database_manager import ColumnInfo, TableInfo
from ..security import (
    SecureCredentialManager,
    SecurityLevel,
    audit_log_security_event,
    identifier_validator,
)
from ..serialization import serialize_rows
from .base import DatabaseDriver

logger = logging.getLogger(__name__)


class PostgreSQLDriver(DatabaseDriver):
    """PostgreSQL-specific database operations."""

    db_type = "postgresql"

    def __init__(self, pool_size: int = 5, max_overflow: int = 10):
        self.engine: Engine | None = None
        self.metadata: MetaData | None = None
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._credential_manager = SecureCredentialManager()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, **params: Any) -> bool:
        """Connect to PostgreSQL.

        Expected params: host, port, database, username, password.
        """
        host = params["host"]
        port = params["port"]
        database = params["database"]
        username = params["username"]
        password = params["password"]

        try:
            if not all([host, port, database, username]):
                logger.error("Missing required PostgreSQL connection parameters")
                return False

            if not identifier_validator.validate_identifier(database):
                audit_log_security_event(
                    "invalid_identifier_attempt",
                    {"identifier": database[:50]},
                    SecurityLevel.MEDIUM,
                )
                logger.error(f"Invalid database name: {database}")
                return False

            safe_username = quote_plus(username)
            safe_password = quote_plus(password)
            connection_string = (
                f"postgresql://{safe_username}:{safe_password}@{host}:{port}/{database}"
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
                    "connect_timeout": CONNECTION_TIMEOUT,
                    "application_name": "orionbelt-analytics",
                },
            )
            self.metadata = MetaData()

            # Initialize encryption
            try:
                if not self._credential_manager._cipher:
                    key_material = f"{host}:{database}:{username}"
                    self._credential_manager._initialize_encryption(key_material)
                logger.info(
                    f"PostgreSQL connection established successfully to {host}:{port}/{database}"
                )
            except Exception as e:
                logger.warning(f"Could not initialize credential encryption: {e}")

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            logger.info(
                f"Connected to PostgreSQL database: {database} at {host}:{port}"
            )
            return True

        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(
                f"Failed to connect to PostgreSQL {host}:{port}/{database}: "
                f"{type(e).__name__}: {e}"
            )
            self.engine = None
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error connecting to PostgreSQL: {type(e).__name__}: {e}"
            )
            self.engine = None
            return False

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        excluded_schemas = "', '".join(POSTGRES_SYSTEM_SCHEMAS)
        query = text(f"""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('{excluded_schemas}')
              AND schema_name NOT LIKE 'pg_temp_%'
              AND schema_name NOT LIKE 'pg_toast_temp_%'
            ORDER BY schema_name
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
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = :schema_name
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """)
                    result = conn.execute(query, {"schema_name": schema_name})
                else:
                    query = text("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """)
                    result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get tables: {e}")
            return []

    def get_views(self, schema_name: str | None = None) -> dict[str, str | None]:
        # pg_catalog.pg_views rather than information_schema.views: the latter
        # returns view_definition as NULL unless the caller owns the view, which
        # is the common case for a read-only analytics role.
        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                if schema_name:
                    query = text("""
                        SELECT viewname, definition
                        FROM pg_catalog.pg_views
                        WHERE schemaname = :schema_name
                        ORDER BY viewname
                    """)
                    result = conn.execute(query, {"schema_name": schema_name})
                else:
                    query = text("""
                        SELECT viewname, definition
                        FROM pg_catalog.pg_views
                        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                        ORDER BY viewname
                    """)
                    result = conn.execute(query)
                return {row[0]: row[1] for row in result.fetchall()}
        except SQLAlchemyError as e:
            logger.error(f"Failed to get views: {e}")
            return {}

    def analyze_table(
        self, table_name: str, schema_name: str | None = None
    ) -> TableInfo | None:
        try:
            assert self.engine is not None
            with self.engine.connect():
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
                prepare_name = "syntax_check_stmt"
                prepare_sql = f"PREPARE {prepare_name} AS {sql_query}"
                deallocate_sql = f"DEALLOCATE {prepare_name}"
                try:
                    conn.execute(text(prepare_sql))
                    conn.execute(text(deallocate_sql))
                    validation_result["is_valid"] = True
                except Exception as prepare_error:
                    error_msg = str(prepare_error)
                    validation_result["database_error"] = error_msg
                    validation_result["error"] = f"PostgreSQL syntax error: {error_msg}"
                    validation_result["error_type"] = "syntax_error"

                    if (
                        "relation" in error_msg.lower()
                        and "does not exist" in error_msg.lower()
                    ):
                        validation_result["suggestions"].append(
                            "Check table/column names - they may not exist or may need proper schema qualification"
                        )
                    elif "syntax error" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "Review SQL syntax - check for missing commas, parentheses, or keywords"
                        )
                    elif "permission denied" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "Insufficient permissions to access the specified tables"
                        )
        except Exception as conn_error:
            validation_result["error"] = (
                f"Database connection error during validation: {conn_error}"
            )
            validation_result["error_type"] = "connection_error"

        return validation_result

    def execute_sql_query(self, sql_query: str, limit: int = 1000) -> dict[str, Any]:
        """Execute query - delegated from DatabaseManager which handles validation."""
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
            db_type = self.db_type.upper()

            assert self.engine is not None
            with self.engine.connect() as conn:
                logger.info(f"\U0001f50d {db_type} SQL QUERY: {sql_query}")
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
                    f"\U0001f50d {self.db_type.upper()} SQL QUERY: {query_str} | PARAMS: {params}"
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
            logger.info("PostgreSQL connection closed")

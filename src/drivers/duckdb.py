"""DuckDB/MotherDuck database driver."""

import logging
from typing import Any

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError, OperationalError, SQLAlchemyError
from sqlalchemy.pool import NullPool, Pool, StaticPool

from ..constants import (
    CONNECTION_TIMEOUT,
    DEFAULT_SAMPLE_LIMIT,
    DUCKDB_SYSTEM_SCHEMAS,
    MAX_SAMPLE_LIMIT,
    MIN_SAMPLE_LIMIT,
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


class DuckDBDriver(DatabaseDriver):
    """DuckDB/MotherDuck database driver.

    Supports both local DuckDB files and MotherDuck cloud databases.

    Connection types:
    - Local file: database_path="/path/to/file.db"
    - In-memory: database_path=":memory:"
    - MotherDuck: database_path="md:database_name", motherduck_token="token"
    """

    db_type = "duckdb"

    def __init__(self, pool_size: int = 5, max_overflow: int = 10):
        self.engine: Engine | None = None
        self.metadata: MetaData | None = None
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._credential_manager = SecureCredentialManager()
        self._database_path: str | None = None
        self._is_motherduck: bool = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, **params: Any) -> bool:
        """Connect to DuckDB or MotherDuck.

        Expected params:
        - database_path: Path to DuckDB file, ":memory:", or "md:database_name"
        - motherduck_token: API token for MotherDuck (optional, for MotherDuck connections)
        - read_only: Whether to open in read-only mode (default: False)
        """
        database_path = params.get("database_path", ":memory:")
        motherduck_token = params.get("motherduck_token")
        read_only = params.get("read_only", False)

        try:
            self._database_path = database_path
            self._is_motherduck = database_path.startswith("md:")

            # Build connection string
            if self._is_motherduck:
                # MotherDuck connection
                if motherduck_token:
                    connection_string = (
                        f"duckdb:///md:?motherduck_token={motherduck_token}"
                    )
                else:
                    # Try to use default MotherDuck token from environment
                    connection_string = "duckdb:///md:"

                database_name = database_path[3:]  # Remove "md:" prefix
                if database_name and not identifier_validator.validate_identifier(
                    database_name
                ):
                    audit_log_security_event(
                        "invalid_identifier_attempt",
                        {"identifier": database_name[:50]},
                        SecurityLevel.MEDIUM,
                    )
                    logger.error(f"Invalid MotherDuck database name: {database_name}")
                    return False
            else:
                # Local DuckDB connection
                if database_path == ":memory:":
                    connection_string = "duckdb:///:memory:"
                else:
                    connection_string = f"duckdb:///{database_path}"
                    if read_only:
                        connection_string += "?read_only=true"

            # DuckDB uses StaticPool for in-memory, NullPool for file-based
            poolclass: type[Pool]
            poolclass = StaticPool if database_path == ":memory:" else NullPool

            self.engine = create_engine(
                connection_string,
                poolclass=poolclass,
                echo=False,
                connect_args={
                    "timeout": CONNECTION_TIMEOUT,
                },
            )
            self.metadata = MetaData()

            # Initialize encryption
            try:
                if not self._credential_manager._cipher:
                    key_material = f"duckdb:{database_path}"
                    self._credential_manager._initialize_encryption(key_material)
                logger.info(
                    f"DuckDB connection encryption initialized for {database_path}"
                )
            except Exception as e:
                logger.warning(f"Could not initialize credential encryption: {e}")

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            db_type_str = "MotherDuck" if self._is_motherduck else "DuckDB"
            logger.info(f"Connected to {db_type_str}: {database_path}")
            return True

        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(
                f"Failed to connect to DuckDB {database_path}: "
                f"{type(e).__name__}: {e}"
            )
            self.engine = None
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error connecting to DuckDB: {type(e).__name__}: {e}"
            )
            self.engine = None
            return False

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        """Get schemas in DuckDB database.

        DuckDB uses 'main' as the default schema. Additional schemas can be created
        or attached from other databases.
        """
        try:
            excluded_schemas = "', '".join(DUCKDB_SYSTEM_SCHEMAS)
            query = text(f"""
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('{excluded_schemas}')
                ORDER BY schema_name
            """)
            assert self.engine is not None
            with self.engine.connect() as conn:
                result = conn.execute(query)
                schemas = [row[0] for row in result.fetchall()]
                # Ensure 'main' is included if it exists
                if not schemas:
                    schemas = ["main"]
                return schemas
        except SQLAlchemyError as e:
            logger.error(f"Failed to get DuckDB schemas: {e}")
            return ["main"]  # Return default schema on error

    def get_tables(self, schema_name: str | None = None) -> list[str]:
        """Get tables in a schema."""
        try:
            schema = schema_name or "main"

            assert self.engine is not None
            with self.engine.connect() as conn:
                query = text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = :schema_name
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """)
                result = conn.execute(query, {"schema_name": schema})
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get DuckDB tables: {e}")
            return []

    def analyze_table(
        self, table_name: str, schema_name: str | None = None
    ) -> TableInfo | None:
        """Analyze a DuckDB table and return its metadata."""
        try:
            schema = schema_name or "main"

            assert self.engine is not None
            with self.engine.connect():
                inspector = inspect(self.engine)

                # Check if table exists
                if not inspector.has_table(table_name, schema=schema):
                    logger.error(f"Table {schema}.{table_name} not found")
                    return None

                table_columns = inspector.get_columns(table_name, schema=schema)
                table_pk = inspector.get_pk_constraint(table_name, schema=schema)
                table_fks = inspector.get_foreign_keys(table_name, schema=schema)

                primary_keys = (
                    table_pk.get("constrained_columns", []) if table_pk else []
                )
                primary_keys_upper = [pk.upper() for pk in primary_keys]

                logger.info(
                    f"DuckDB table {schema}.{table_name}: PKs={primary_keys}, "
                    f"FKs={len(table_fks)} constraints"
                )

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
                            break

                    columns.append(
                        ColumnInfo(
                            name=column_name,
                            data_type=str(col_info["type"]),
                            is_nullable=col_info.get("nullable", True),
                            is_primary_key=is_pk,
                            is_foreign_key=is_fk,
                            foreign_key_table=fk_table,
                            foreign_key_column=fk_column,
                            comment=col_info.get("comment"),
                        )
                    )

                return TableInfo(
                    name=table_name,
                    schema=schema,
                    columns=columns,
                    primary_keys=primary_keys,
                    foreign_keys=foreign_keys,
                    comment=None,
                    row_count=None,
                    sample_data=None,
                )

        except SQLAlchemyError as e:
            logger.error(f"Failed to analyze DuckDB table {table_name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Query validation & execution
    # ------------------------------------------------------------------

    def validate_sql_syntax(
        self, sql_query: str, validation_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate DuckDB SQL syntax using EXPLAIN."""
        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                try:
                    # Use EXPLAIN to validate syntax without executing
                    explain_query = f"EXPLAIN {sql_query}"
                    conn.execute(text(explain_query))
                    validation_result["is_valid"] = True
                except Exception as syntax_error:
                    error_msg = str(syntax_error)
                    validation_result["database_error"] = error_msg
                    validation_result["error"] = f"DuckDB syntax error: {error_msg}"
                    validation_result["error_type"] = "syntax_error"

                    if "does not exist" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "Check table/column names - they may not exist in the schema"
                        )
                    elif "syntax error" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "Review DuckDB SQL syntax - check for missing commas, parentheses, or keywords"
                        )
        except Exception as conn_error:
            validation_result["error"] = (
                f"Database connection error during validation: {conn_error}"
            )
            validation_result["error_type"] = "connection_error"

        return validation_result

    def execute_sql_query(self, sql_query: str, limit: int = 1000) -> dict[str, Any]:
        """Execute DuckDB SQL query."""
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
            db_type_str = "MOTHERDUCK" if self._is_motherduck else "DUCKDB"

            assert self.engine is not None
            with self.engine.connect() as conn:
                logger.info(f"🔍 {db_type_str} SQL QUERY: {sql_query}")
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
                    f"DuckDB query executed: {result_data['row_count']} rows "
                    f"in {result_data['execution_time_ms']}ms"
                )

        except SQLAlchemyError as e:
            result_data["error"] = str(e)
            result_data["error_type"] = "execution_error"
            logger.error(f"DuckDB SQL execution failed: {e}")
        except Exception as e:
            result_data["error"] = f"Unexpected execution error: {e!s}"
            result_data["error_type"] = "internal_error"
            logger.error(f"Unexpected DuckDB SQL execution error: {e}")

        return result_data

    def sample_table_data(
        self,
        table_name: str,
        schema_name: str | None = None,
        limit: int = DEFAULT_SAMPLE_LIMIT,
    ) -> list[dict[str, Any]]:
        """Sample data from a DuckDB table."""
        if not isinstance(limit, int) or limit < MIN_SAMPLE_LIMIT:
            limit = DEFAULT_SAMPLE_LIMIT
        elif limit > MAX_SAMPLE_LIMIT:
            limit = MAX_SAMPLE_LIMIT
            logger.warning(f"Sample limit capped at {MAX_SAMPLE_LIMIT}")

        try:
            schema = schema_name or "main"

            assert self.engine is not None
            with self.engine.connect() as conn:
                full_table_name = f'"{schema}"."{table_name}"'

                query_str = f"SELECT * FROM {full_table_name} LIMIT :limit"
                params = {"limit": limit}
                db_type_str = "MOTHERDUCK" if self._is_motherduck else "DUCKDB"
                logger.info(
                    f"🔍 {db_type_str} SQL QUERY: {query_str} | PARAMS: {params}"
                )
                result = conn.execute(text(query_str), params)
                columns = list(result.keys())
                return serialize_rows(result.fetchall(), columns)

        except (SQLAlchemyError, ValueError) as e:
            logger.error(
                f"Failed to sample data from DuckDB table {table_name}: {type(e).__name__}: {e}"
            )
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error sampling DuckDB table {table_name}: {type(e).__name__}: {e}"
            )
            return []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Test DuckDB connection health."""
        if not self.engine:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.warning(f"DuckDB connection health check failed: {e}")
            return False

    def disconnect(self) -> None:
        """Close DuckDB connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.metadata = None
            db_type_str = "MotherDuck" if self._is_motherduck else "DuckDB"
            logger.info(f"{db_type_str} connection closed")

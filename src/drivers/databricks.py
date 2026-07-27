"""Databricks SQL database driver."""

import logging
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError, OperationalError, SQLAlchemyError
from sqlalchemy.pool import NullPool

from ..constants import (
    CONNECTION_TIMEOUT,
    DATABRICKS_SYSTEM_SCHEMAS,
    DEFAULT_SAMPLE_LIMIT,
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


class DatabricksDriver(DatabaseDriver):
    """Databricks SQL database driver.

    Uses SQLAlchemy with databricks-sql-connector for Databricks SQL Warehouse
    or Databricks cluster connectivity.

    Connection requires:
    - server_hostname: Databricks workspace URL
    - http_path: SQL Warehouse or cluster HTTP path
    - access_token: Personal access token or service principal token
    - catalog: Unity Catalog name (optional, defaults to 'hive_metastore')
    - schema: Default schema/database (optional)
    """

    db_type = "databricks"

    def __init__(self, pool_size: int = 5, max_overflow: int = 10):
        self.engine: Engine | None = None
        self.metadata: MetaData | None = None
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._credential_manager = SecureCredentialManager()
        self._server_hostname: str | None = None
        self._http_path: str | None = None
        self._catalog: str | None = None
        self._schema: str | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, **params: Any) -> bool:
        """Connect to Databricks SQL.

        Expected params:
        - server_hostname: Databricks workspace URL (e.g., "dbc-a1b2c3d4-e5f6.cloud.databricks.com")
        - http_path: SQL Warehouse HTTP path (e.g., "/sql/1.0/warehouses/abc123def456")
        - access_token: Personal access token or service principal token
        - catalog: Unity Catalog name (default: "hive_metastore")
        - schema: Default schema/database (optional)

        Connection string format:
        databricks://token:{access_token}@{server_hostname}?http_path={http_path}&catalog={catalog}
        """
        server_hostname = params.get("server_hostname")
        http_path = params.get("http_path")
        access_token = params.get("access_token")
        catalog = params.get("catalog", "hive_metastore")
        schema = params.get("schema", "default")

        try:
            if not all([server_hostname, http_path, access_token]):
                logger.error("Missing required Databricks connection parameters")
                return False

            if catalog and not identifier_validator.validate_identifier(catalog):
                audit_log_security_event(
                    "invalid_identifier_attempt",
                    {"identifier": catalog[:50]},
                    SecurityLevel.MEDIUM,
                )
                logger.error(f"Invalid catalog name: {catalog}")
                return False

            self._server_hostname = server_hostname
            self._http_path = http_path
            self._catalog = catalog
            self._schema = schema

            # URL-encode the access token and http_path
            # (guaranteed non-None by the all() check above)
            assert access_token is not None
            assert http_path is not None
            safe_token = quote_plus(access_token)
            safe_http_path = quote_plus(http_path)

            # Build connection string
            # Format: databricks://token:{token}@{hostname}?http_path={path}&catalog={catalog}
            connection_string = (
                f"databricks://token:{safe_token}@{server_hostname}?"
                f"http_path={safe_http_path}&catalog={catalog}"
            )

            if schema:
                connection_string += f"&schema={schema}"

            # Databricks recommends NullPool (no connection pooling)
            self.engine = create_engine(
                connection_string,
                poolclass=NullPool,
                echo=False,
                connect_args={
                    "timeout": CONNECTION_TIMEOUT,
                },
            )
            self.metadata = MetaData()

            # Initialize encryption
            try:
                if not self._credential_manager._cipher:
                    key_material = f"{server_hostname}:{catalog}:{schema}"
                    self._credential_manager._initialize_encryption(key_material)
                logger.info(
                    f"Databricks connection encryption initialized for {server_hostname}"
                )
            except Exception as e:
                logger.warning(f"Could not initialize credential encryption: {e}")

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            logger.info(
                f"Connected to Databricks SQL: {server_hostname}, "
                f"catalog: {catalog}, schema: {schema}"
            )
            return True

        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(
                f"Failed to connect to Databricks {server_hostname}: "
                f"{type(e).__name__}: {e}"
            )
            self.engine = None
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error connecting to Databricks: {type(e).__name__}: {e}"
            )
            self.engine = None
            return False

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        """Get schemas (databases) in the Databricks catalog.

        In Databricks Unity Catalog, schemas are also known as databases.
        """
        try:
            # Query to get schemas in the current catalog
            query = text(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE catalog_name = :catalog
                ORDER BY schema_name
            """
            )
            assert self.engine is not None
            with self.engine.connect() as conn:
                result = conn.execute(query, {"catalog": self._catalog})
                schemas = [row[0] for row in result.fetchall()]

                # Filter out system schemas
                filtered_schemas = [
                    s for s in schemas if s not in DATABRICKS_SYSTEM_SCHEMAS
                ]
                return filtered_schemas
        except SQLAlchemyError as e:
            logger.error(f"Failed to get Databricks schemas: {e}")
            # Fallback to SHOW SCHEMAS
            try:
                assert self.engine is not None
                with self.engine.connect() as conn:
                    result = conn.execute(text(f"SHOW SCHEMAS IN {self._catalog}"))
                    schemas = [row[0] for row in result.fetchall()]
                    return [s for s in schemas if s not in DATABRICKS_SYSTEM_SCHEMAS]
            except Exception as fallback_error:
                logger.error(f"Fallback schema query also failed: {fallback_error}")
                return []

    def get_tables(self, schema_name: str | None = None) -> list[str]:
        """Get tables in a schema."""
        try:
            schema = schema_name or self._schema
            if not schema:
                logger.error("No schema specified and no default schema set")
                return []

            assert self.engine is not None
            with self.engine.connect() as conn:
                query = text(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_catalog = :catalog
                    AND table_schema = :schema
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """
                )
                result = conn.execute(
                    query, {"catalog": self._catalog, "schema": schema}
                )
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get Databricks tables: {e}")
            # Fallback to SHOW TABLES
            try:
                schema = schema_name or self._schema
                assert self.engine is not None
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text(f"SHOW TABLES IN {self._catalog}.{schema}")
                    )
                    return [
                        row[1] for row in result.fetchall()
                    ]  # Table name is in second column
            except Exception as fallback_error:
                logger.error(f"Fallback table query also failed: {fallback_error}")
                return []

    def analyze_table(
        self, table_name: str, schema_name: str | None = None
    ) -> TableInfo | None:
        """Analyze a Databricks table and return its metadata.

        Databricks Unity Catalog supports primary keys and foreign keys (constraints),
        but they are not enforced. We extract them for metadata purposes.
        """
        try:
            schema = schema_name or self._schema
            if not schema:
                logger.error("No schema specified for table analysis")
                return None

            assert self.engine is not None
            with self.engine.connect():
                inspector = inspect(self.engine)

                # Check if table exists
                if not inspector.has_table(table_name, schema=schema):
                    logger.error(
                        f"Table {self._catalog}.{schema}.{table_name} not found"
                    )
                    return None

                table_columns = inspector.get_columns(table_name, schema=schema)
                table_pk = inspector.get_pk_constraint(table_name, schema=schema)
                table_fks = inspector.get_foreign_keys(table_name, schema=schema)

                primary_keys = (
                    table_pk.get("constrained_columns", []) if table_pk else []
                )
                primary_keys_upper = [pk.upper() for pk in primary_keys]

                logger.info(
                    f"Databricks table {self._catalog}.{schema}.{table_name}: "
                    f"PKs={primary_keys}, FKs={len(table_fks)} constraints"
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
            logger.error(f"Failed to analyze Databricks table {table_name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Query validation & execution
    # ------------------------------------------------------------------

    def validate_sql_syntax(
        self, sql_query: str, validation_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate Databricks SQL syntax using EXPLAIN."""
        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                try:
                    # Use EXPLAIN to validate syntax
                    explain_query = f"EXPLAIN {sql_query}"
                    conn.execute(text(explain_query))
                    validation_result["is_valid"] = True
                except Exception as syntax_error:
                    error_msg = str(syntax_error)
                    validation_result["database_error"] = error_msg
                    validation_result["error"] = f"Databricks syntax error: {error_msg}"
                    validation_result["error_type"] = "syntax_error"

                    if (
                        "not found" in error_msg.lower()
                        or "does not exist" in error_msg.lower()
                    ):
                        validation_result["suggestions"].append(
                            "Check table names - use three-part names (catalog.schema.table) in Unity Catalog"
                        )
                    elif "syntax error" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "Review Databricks SQL syntax - it's based on Spark SQL"
                        )
                    elif (
                        "permission" in error_msg.lower()
                        or "denied" in error_msg.lower()
                    ):
                        validation_result["suggestions"].append(
                            "Insufficient permissions to access the specified tables/catalogs"
                        )
        except Exception as conn_error:
            validation_result[
                "error"
            ] = f"Database connection error during validation: {conn_error}"
            validation_result["error_type"] = "connection_error"

        return validation_result

    def execute_sql_query(self, sql_query: str, limit: int = 1000) -> dict[str, Any]:
        """Execute Databricks SQL query."""
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
                logger.info(f"🔍 DATABRICKS SQL QUERY: {sql_query}")
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
                            except Exception:
                                pass
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
                    except Exception:
                        pass

                end_time = time_mod.time()
                result_data["execution_time_ms"] = round(
                    (end_time - start_time) * 1000, 2
                )
                result_data["success"] = True
                logger.info(
                    f"Databricks query executed: {result_data['row_count']} rows "
                    f"in {result_data['execution_time_ms']}ms"
                )

        except SQLAlchemyError as e:
            result_data["error"] = str(e)
            result_data["error_type"] = "execution_error"
            logger.error(f"Databricks SQL execution failed: {e}")
        except Exception as e:
            result_data["error"] = f"Unexpected execution error: {e!s}"
            result_data["error_type"] = "internal_error"
            logger.error(f"Unexpected Databricks SQL execution error: {e}")

        return result_data

    def sample_table_data(
        self,
        table_name: str,
        schema_name: str | None = None,
        limit: int = DEFAULT_SAMPLE_LIMIT,
    ) -> list[dict[str, Any]]:
        """Sample data from a Databricks table."""
        if not isinstance(limit, int) or limit < MIN_SAMPLE_LIMIT:
            limit = DEFAULT_SAMPLE_LIMIT
        elif limit > MAX_SAMPLE_LIMIT:
            limit = MAX_SAMPLE_LIMIT
            logger.warning(f"Sample limit capped at {MAX_SAMPLE_LIMIT}")

        try:
            schema = schema_name or self._schema
            if not schema:
                logger.error("No schema specified for sampling")
                return []

            assert self.engine is not None
            with self.engine.connect() as conn:
                # Use three-part name for Unity Catalog
                full_table_name = f"`{self._catalog}`.`{schema}`.`{table_name}`"

                query_str = f"SELECT * FROM {full_table_name} LIMIT {limit}"
                logger.info(f"🔍 DATABRICKS SQL QUERY: {query_str}")
                result = conn.execute(text(query_str))
                columns = list(result.keys())
                return serialize_rows(result.fetchall(), columns)

        except (SQLAlchemyError, ValueError) as e:
            logger.error(
                f"Failed to sample data from Databricks table {table_name}: "
                f"{type(e).__name__}: {e}"
            )
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error sampling Databricks table {table_name}: "
                f"{type(e).__name__}: {e}"
            )
            return []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Test Databricks connection health."""
        if not self.engine:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.warning(f"Databricks connection health check failed: {e}")
            return False

    def disconnect(self) -> None:
        """Close Databricks connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.metadata = None
            logger.info("Databricks connection closed")

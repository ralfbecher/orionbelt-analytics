"""Database connection and schema analysis manager.

Orchestrates database operations by delegating to database-specific drivers
while managing cross-cutting concerns: caching, credentials, reconnection,
connection pooling configuration, and security validation.
"""

import hashlib
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DatabaseError, OperationalError

from .constants import DB_SQLGLOT_DIALECTS, DEFAULT_SAMPLE_LIMIT, IDENTIFIER_PATTERN
from .security import (
    SecureCredentialManager,
    SecurityLevel,
    analyze_sql_statement,
    audit_log_security_event,
    identifier_validator,
    sql_validator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes (imported by many modules - must stay here)
# ---------------------------------------------------------------------------


@dataclass
class ColumnInfo:
    """Information about a database column."""

    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    foreign_key_table: str | None = None
    foreign_key_column: str | None = None
    comment: str | None = None


@dataclass
class ViewInfo:
    """A database view and the SQL that defines it.

    Deliberately not a :class:`TableInfo`. Views are excluded from the
    ontology -- a view pre-joins its sources, so an OWL class for it would
    restate concepts the base tables already model and leave the FK/fan-trap
    reasoning an isolated node. They are indexed into GraphRAG instead, where
    the definition is the point: it is analyst-authored SQL carrying business
    vocabulary the raw column names never do (``v_revenue_by_client`` explains
    what ``amount`` means) alongside join conditions someone already validated.
    """

    name: str
    schema: str
    definition: str | None = None
    comment: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ViewInfo":
        """Deserialize a ViewInfo from a dict (e.g. saved schema JSON).

        Args:
            data: Dict with keys matching ViewInfo fields.

        Returns:
            ViewInfo instance
        """
        return cls(
            name=data["name"],
            schema=data.get("schema", ""),
            definition=data.get("definition"),
            comment=data.get("comment"),
        )


@dataclass
class TableInfo:
    """Information about a database table."""

    name: str
    schema: str
    columns: list[ColumnInfo]
    primary_keys: list[str]
    foreign_keys: list[dict[str, str]]
    comment: str | None = None
    row_count: int | None = None
    sample_data: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableInfo":
        """Deserialize a TableInfo from a dict (e.g. saved schema JSON).

        Args:
            data: Dict with keys matching TableInfo fields.
                  Columns may be dicts or ColumnInfo instances.

        Returns:
            TableInfo instance
        """
        columns = []
        for col in data.get("columns", []):
            if isinstance(col, ColumnInfo):
                columns.append(col)
            elif isinstance(col, dict):
                columns.append(ColumnInfo(**col))

        return cls(
            name=data["name"],
            schema=data.get("schema", ""),
            columns=columns,
            primary_keys=data.get("primary_keys", []),
            foreign_keys=data.get("foreign_keys", []),
            comment=data.get("comment"),
            row_count=data.get("row_count"),
            sample_data=data.get("sample_data"),
        )


# ---------------------------------------------------------------------------
# DatabaseManager - orchestrator
# ---------------------------------------------------------------------------


class DatabaseManager:
    """Manages database connections and schema analysis with enhanced reliability and security.

    Delegates database-specific work to driver instances while handling:
    - Metadata caching with TTL
    - Credential encryption
    - Automatic reconnection
    - Security validation (identifiers, SQL injection)
    - Connection health checks
    """

    def __init__(self) -> None:
        # Driver holds the active database-specific implementation
        self._driver: Any | None = None  # DatabaseDriver from .drivers.base

        # Legacy attributes kept for backward compatibility with get_connection()
        self.engine: Engine | None = None
        self.metadata = None
        self.connection_info: dict[str, Any] = {}
        self._connection_pool_size = 5
        self._max_overflow = 10
        self._dremio_rest_connection: dict[str, Any] | None = None
        self._last_connection_params: dict[str, Any] | None = None

        # Security and performance
        self._credential_manager = SecureCredentialManager()
        self._metadata_cache: dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes
        self._connection_id: str | None = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache_key(self, operation: str, *args: Any) -> str:
        """Generate cache key for metadata operations."""
        return f"{operation}:{':'.join(str(arg) for arg in args)}"

    def _is_cache_valid(self, cache_entry: dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        return bool(time.time() - cache_entry.get("timestamp", 0) < self._cache_ttl)

    def _get_from_cache(self, cache_key: str) -> Any | None:
        """Get value from cache if valid."""
        if cache_key in self._metadata_cache:
            entry = self._metadata_cache[cache_key]
            if self._is_cache_valid(entry):
                logger.debug(f"Cache hit for {cache_key}")
                return entry["data"]
            else:
                del self._metadata_cache[cache_key]
        return None

    def _store_in_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        self._metadata_cache[cache_key] = {"data": data, "timestamp": time.time()}
        logger.debug(f"Cached data for {cache_key}")

    # ------------------------------------------------------------------
    # SQL / identifier helpers
    # ------------------------------------------------------------------

    def _log_sql_query(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Log SQL query with parameters for debugging."""
        db_type = self.connection_info.get("type", "unknown")
        if params:
            safe_params = {
                k: "***" if "password" in k.lower() or "secret" in k.lower() else v
                for k, v in params.items()
            }
            logger.info(
                f"\U0001f50d {db_type.upper()} SQL QUERY: {query} | PARAMS: {safe_params}"
            )
        else:
            logger.info(f"\U0001f50d {db_type.upper()} SQL QUERY: {query}")

    def _validate_identifier_secure(self, identifier: str) -> bool:
        """Securely validate database identifier to prevent injection."""
        if not identifier_validator.validate_identifier(identifier):
            audit_log_security_event(
                "invalid_identifier_attempt",
                {"identifier": identifier[:50]},
                SecurityLevel.MEDIUM,
            )
            return False
        return True

    def _validate_identifier(self, identifier: str) -> bool:
        """Validate database identifier to prevent injection attacks."""
        if not identifier or len(identifier) > 63:
            return False
        return bool(re.match(IDENTIFIER_PATTERN, identifier))

    def _strip_leading_sql_comments(self, sql_query: str) -> str:
        """Strip leading SQL comments to find the actual SQL statement.

        Handles both -- (line comments) and /* */ (block comments) at the
        beginning of queries.
        """
        lines = sql_query.split("\n")
        result_lines = []
        in_block_comment = False

        for idx, original_line in enumerate(lines):
            line = original_line.strip()

            if in_block_comment:
                if "*/" in line:
                    after_comment = line.split("*/", 1)[1].strip()
                    in_block_comment = False
                    if after_comment:
                        result_lines.append(after_comment)
                continue

            if not line:
                continue

            if line.startswith("--"):
                continue

            if line.startswith("/*"):
                if "*/" in line:
                    after_comment = line.split("*/", 1)[1].strip()
                    if after_comment:
                        result_lines.append(after_comment)
                        break
                else:
                    in_block_comment = True
                continue

            result_lines.append(original_line)
            remaining_index = idx + 1
            if remaining_index < len(lines):
                result_lines.extend(lines[remaining_index:])
            break

        return "\n".join(result_lines).strip()

    def _escape_sql_literal(self, value: str) -> str:
        """Escape a value for safe use inside a single-quoted SQL literal."""
        return value.replace("'", "''")

    def _quote_dremio_identifier(self, identifier: str | None) -> str:
        """Quote and escape a Dremio identifier or path safely."""
        if identifier is None:
            return ""
        parts = [p for p in str(identifier).split(".") if p != ""]
        if not parts:
            return ""
        escaped_parts = [part.replace('"', '""') for part in parts]
        return ".".join(f'"{part}"' for part in escaped_parts)

    # ------------------------------------------------------------------
    # Connection lifecycle helpers
    # ------------------------------------------------------------------

    def _sync_engine_from_driver(self) -> None:
        """Keep legacy ``self.engine`` attribute in sync with the active driver."""
        if self._driver and hasattr(self._driver, "engine"):
            self.engine = self._driver.engine
            self.metadata = getattr(self._driver, "metadata", None)
        else:
            self.engine = None
            self.metadata = None

    def _test_connection(self) -> bool:
        """Test if the current connection is healthy."""
        if self._driver:
            return bool(self._driver.test_connection())
        if not self.engine:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False

    def _test_dremio_connection(self) -> bool:
        """Test if the current Dremio REST connection is healthy."""
        if self._driver and self._driver.db_type == "dremio":
            return bool(self._driver.test_connection())
        return False

    def _ensure_connection(self) -> None:
        """Ensure we have a healthy database connection, reconnecting if necessary."""
        if self._dremio_rest_connection:
            logger.debug("_ensure_connection: Dremio REST connection info available")
            return

        logger.debug(f"_ensure_connection: engine exists: {self.engine is not None}")
        logger.debug(
            f"_ensure_connection: last_params available: {self._last_connection_params is not None}"
        )

        if not self.engine:
            if self._last_connection_params:
                logger.info(
                    "No engine found, reconnecting to database using stored parameters"
                )
                self._reconnect()
            else:
                raise RuntimeError(
                    "No database connection established and no connection parameters available"
                )
        elif not self._test_connection():
            if self._last_connection_params:
                logger.info("Connection health check failed, reconnecting to database")
                self._reconnect()
            else:
                logger.error(
                    "Connection unhealthy but no reconnection parameters available"
                )
                raise RuntimeError(
                    "Database connection is unhealthy and cannot be restored"
                )

        logger.debug(
            f"_ensure_connection: final engine state: {self.engine is not None}"
        )

    def _reconnect(self) -> None:
        """Reconnect to the database using stored parameters."""
        if not self._last_connection_params:
            raise RuntimeError("No connection parameters stored for reconnection")

        params = self._last_connection_params
        if params["type"] == "postgresql":
            success = self.connect_postgresql(
                params["host"],
                params["port"],
                params["database"],
                params["username"],
                params["password"],
            )
        elif params["type"] == "snowflake":
            success = self.connect_snowflake(
                params["account"],
                params["username"],
                params["password"],
                params["warehouse"],
                params["database"],
                params.get("schema", "PUBLIC"),
            )
        elif params["type"] == "clickhouse":
            success = self.connect_clickhouse(
                params["host"],
                params["port"],
                params["database"],
                params.get("username", "default"),
                params.get("password", ""),
                params.get("protocol", "http"),
                params.get("secure", False),
            )
        elif params["type"] == "dremio":
            if params.get("uri") and params.get("pat"):
                success = self.connect_dremio(uri=params["uri"], pat=params["pat"])
            else:
                success = self.connect_dremio(
                    params.get("host"),
                    params.get("port"),
                    params.get("username"),
                    params.get("password"),
                    params.get("ssl", True),
                )
        elif params["type"] == "bigquery":
            success = self.connect_bigquery(
                params["project_id"],
                params.get("dataset", ""),
                params.get("credentials_path"),
                params.get("credentials_json"),
            )
        elif params["type"] == "duckdb":
            success = self.connect_duckdb(
                params.get("database_path", ":memory:"),
                params.get("motherduck_token"),
                params.get("read_only", False),
            )
        elif params["type"] == "databricks":
            success = self.connect_databricks(
                params["server_hostname"],
                params["http_path"],
                params["access_token"],
                params.get("catalog", "hive_metastore"),
                params.get("schema", "default"),
            )
        elif params["type"] == "mysql":
            success = self.connect_mysql(
                params["host"],
                params["port"],
                params["database"],
                params["username"],
                params["password"],
                params.get("charset", "utf8mb4"),
            )
        else:
            raise RuntimeError(
                f"Unsupported database type for reconnection: {params['type']}"
            )

        if not success:
            raise RuntimeError(f"Failed to reconnect to {params['type']} database")

        logger.info(f"Successfully reconnected to {params['type']} database")

    @contextmanager
    def get_connection(self) -> Iterator[Connection]:
        """Context manager for database connections with auto-reconnection."""
        if not self.engine:
            raise RuntimeError("No database connection established")

        max_retries = 2
        last_exception = None

        for attempt in range(max_retries):
            try:
                conn = self.engine.connect()
                try:
                    yield conn
                finally:
                    conn.close()
                return
            except (OperationalError, DatabaseError) as e:
                last_exception = e
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    if self._last_connection_params:
                        logger.info("Attempting reconnection...")
                        try:
                            self._reconnect()
                        except Exception as reconnect_error:
                            logger.error(f"Reconnection failed: {reconnect_error}")
                    else:
                        logger.error(
                            "No connection parameters available for reconnection"
                        )
                        break

        logger.error(f"All connection attempts failed. Last error: {last_exception}")
        raise RuntimeError(
            f"Database connection failed after {max_retries} attempts: {last_exception}"
        )

    # ------------------------------------------------------------------
    # connect_* methods - instantiate the appropriate driver
    # ------------------------------------------------------------------

    def connect_postgresql(
        self, host: str, port: int, database: str, username: str, password: str
    ) -> bool:
        """Connect to PostgreSQL database with enhanced security and reliability."""
        from .drivers.postgresql import PostgreSQLDriver

        driver = PostgreSQLDriver(
            pool_size=self._connection_pool_size,
            max_overflow=self._max_overflow,
        )
        success = driver.connect(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
        )
        if success:
            self._driver = driver
            self._dremio_rest_connection = None
            self._sync_engine_from_driver()

            self.connection_info = {
                "type": "postgresql",
                "host": host,
                "port": port,
                "database": database,
                "username": username,
            }
            self._connection_id = hashlib.sha256(
                f"{host}:{port}:{database}:{username}".encode()
            ).hexdigest()[:16]

            self._last_connection_params = {
                "type": "postgresql",
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "password": password,
            }
        return success

    def connect_snowflake(
        self,
        account: str,
        username: str,
        password: str,
        warehouse: str,
        database: str,
        schema: str = "PUBLIC",
        role: str = "PUBLIC",
    ) -> bool:
        """Connect to Snowflake database with enhanced security and reliability."""
        from .drivers.snowflake import SnowflakeDriver

        driver = SnowflakeDriver(
            pool_size=self._connection_pool_size,
            max_overflow=self._max_overflow,
        )
        success = driver.connect(
            account=account,
            username=username,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema,
            role=role,
        )
        if success:
            self._driver = driver
            self._dremio_rest_connection = None
            self._sync_engine_from_driver()

            self.connection_info = {
                "type": "snowflake",
                "account": account,
                "username": username,
                "warehouse": warehouse,
                "database": database,
                "schema": schema,
                "role": role,
            }
            self._last_connection_params = {
                "type": "snowflake",
                "account": account,
                "username": username,
                "password": password,
                "warehouse": warehouse,
                "database": database,
                "schema": schema,
                "role": role,
            }
        return success

    def connect_clickhouse(
        self,
        host: str,
        port: int = 8123,
        database: str = "default",
        username: str = "default",
        password: str = "",
        protocol: str = "http",
        secure: bool = False,
    ) -> bool:
        """Connect to ClickHouse database via SQLAlchemy."""
        from .drivers.clickhouse import ClickHouseDriver

        driver = ClickHouseDriver(
            pool_size=self._connection_pool_size,
            max_overflow=self._max_overflow,
        )
        success = driver.connect(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            protocol=protocol,
            secure=secure,
        )
        if success:
            self._driver = driver
            # Store database name on driver for get_tables fallback.
            # Set dynamically (read via getattr in the driver); not a declared
            # attribute, so silence the attr-defined check here.
            driver._database_name = database  # type: ignore[attr-defined]
            self._dremio_rest_connection = None
            self._sync_engine_from_driver()

            self.connection_info = {
                "type": "clickhouse",
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "protocol": protocol,
                "secure": secure,
            }
            self._last_connection_params = {
                "type": "clickhouse",
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "password": password,
                "protocol": protocol,
                "secure": secure,
            }
        return success

    def connect_dremio(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        ssl: bool = False,
        uri: str | None = None,
        pat: str | None = None,
    ) -> bool:
        """Connect to Dremio using REST API instead of PostgreSQL protocol."""
        from .drivers.dremio import DremioDriver

        # Dispose existing SQLAlchemy engine if any
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.metadata = None

        driver = DremioDriver()
        success = driver.connect(
            host=host,
            port=port,
            username=username,
            password=password,
            ssl=ssl,
            uri=uri,
            pat=pat,
        )
        if success:
            self._driver = driver
            self.engine = None
            self.metadata = None

            api_port = 9047
            if uri and pat:
                self._dremio_rest_connection = {"uri": uri, "pat": pat}
                self.connection_info = {
                    "type": "dremio",
                    "uri": uri,
                    "auth_method": "PAT",
                    "api": "REST",
                }
                self._last_connection_params = {
                    "type": "dremio",
                    "uri": uri,
                    "pat": pat,
                }
            else:
                self._dremio_rest_connection = {
                    "host": host,
                    "port": api_port,
                    "username": username,
                    "password": password,
                    "ssl": ssl,
                }
                self.connection_info = {
                    "type": "dremio",
                    "host": host,
                    "port": api_port,
                    "username": username,
                    "ssl": ssl,
                    "auth_method": "username_password",
                    "api": "REST",
                }
                self._last_connection_params = {
                    "type": "dremio",
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "ssl": ssl,
                }
        return success

    def connect_bigquery(
        self,
        project_id: str,
        dataset: str = "",
        credentials_path: str | None = None,
        credentials_json: str | None = None,
    ) -> bool:
        """Connect to Google BigQuery."""
        from .drivers.bigquery import BigQueryDriver

        driver = BigQueryDriver(
            pool_size=self._connection_pool_size,
            max_overflow=self._max_overflow,
        )
        success = driver.connect(
            project_id=project_id,
            dataset=dataset,
            credentials_path=credentials_path,
            credentials_json=credentials_json,
        )
        if success:
            self._driver = driver
            self._dremio_rest_connection = None
            self._sync_engine_from_driver()

            self.connection_info = {
                "type": "bigquery",
                "project_id": project_id,
                "dataset": dataset,
            }
            self._connection_id = hashlib.sha256(
                f"{project_id}:{dataset}".encode()
            ).hexdigest()[:16]

            self._last_connection_params = {
                "type": "bigquery",
                "project_id": project_id,
                "dataset": dataset,
                "credentials_path": credentials_path,
                "credentials_json": credentials_json,
            }
        return success

    def connect_duckdb(
        self,
        database_path: str = ":memory:",
        motherduck_token: str | None = None,
        read_only: bool = False,
    ) -> bool:
        """Connect to DuckDB or MotherDuck."""
        from .drivers.duckdb import DuckDBDriver

        driver = DuckDBDriver(
            pool_size=self._connection_pool_size,
            max_overflow=self._max_overflow,
        )
        success = driver.connect(
            database_path=database_path,
            motherduck_token=motherduck_token,
            read_only=read_only,
        )
        if success:
            self._driver = driver
            self._dremio_rest_connection = None
            self._sync_engine_from_driver()

            is_motherduck = database_path.startswith("md:")
            self.connection_info = {
                "type": "duckdb",
                "database_path": database_path,
                "is_motherduck": is_motherduck,
                "read_only": read_only,
            }
            self._connection_id = hashlib.sha256(
                f"duckdb:{database_path}".encode()
            ).hexdigest()[:16]

            self._last_connection_params = {
                "type": "duckdb",
                "database_path": database_path,
                "motherduck_token": motherduck_token,
                "read_only": read_only,
            }
        return success

    def connect_databricks(
        self,
        server_hostname: str,
        http_path: str,
        access_token: str,
        catalog: str = "hive_metastore",
        schema: str = "default",
    ) -> bool:
        """Connect to Databricks SQL."""
        from .drivers.databricks import DatabricksDriver

        driver = DatabricksDriver(
            pool_size=self._connection_pool_size,
            max_overflow=self._max_overflow,
        )
        success = driver.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
            catalog=catalog,
            schema=schema,
        )
        if success:
            self._driver = driver
            self._dremio_rest_connection = None
            self._sync_engine_from_driver()

            self.connection_info = {
                "type": "databricks",
                "server_hostname": server_hostname,
                "http_path": http_path,
                "catalog": catalog,
                "schema": schema,
            }
            self._connection_id = hashlib.sha256(
                f"{server_hostname}:{catalog}:{schema}".encode()
            ).hexdigest()[:16]

            self._last_connection_params = {
                "type": "databricks",
                "server_hostname": server_hostname,
                "http_path": http_path,
                "access_token": access_token,
                "catalog": catalog,
                "schema": schema,
            }
        return success

    def connect_mysql(
        self,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        charset: str = "utf8mb4",
    ) -> bool:
        """Connect to MySQL 8.0+ or MariaDB 10.5+ database.

        Args:
            host: MySQL server hostname or IP
            port: MySQL server port (default: 3306)
            database: Database name
            username: MySQL username
            password: MySQL password
            charset: Character set (default: utf8mb4 for full Unicode support)

        Returns:
            True if connection successful, False otherwise

        Note:
            MySQL 5.7 reached EOL in October 2023 and is not supported.
            Requires MySQL 8.0+ or MariaDB 10.5+.
        """
        from .drivers.mysql import MySQLDriver

        driver = MySQLDriver(
            pool_size=self._connection_pool_size,
            max_overflow=self._max_overflow,
        )
        success = driver.connect(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            charset=charset,
        )
        if success:
            self._driver = driver
            self._dremio_rest_connection = None
            self._sync_engine_from_driver()

            self.connection_info = {
                "type": "mysql",
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "charset": charset,
            }
            self._connection_id = hashlib.sha256(
                f"{host}:{port}:{database}:{username}".encode()
            ).hexdigest()[:16]

            self._last_connection_params = {
                "type": "mysql",
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "password": password,
                "charset": charset,
            }
        return success

    # ------------------------------------------------------------------
    # Schema introspection (delegated to driver)
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        """Get list of available schemas."""
        if self._driver:
            return cast(list[str], self._driver.get_schemas())

        # Fallback: should not happen if connected through connect_* methods
        if not self.engine and not self._dremio_rest_connection:
            raise RuntimeError("No database connection established")
        raise RuntimeError("No driver available - use connect_* methods first")

    def get_tables(self, schema_name: str | None = None) -> list[str]:
        """Get list of tables in a schema with caching for performance."""
        logger.debug(
            f"get_tables: Starting, has_engine: {self.has_engine()}, "
            f"dremio_rest: {bool(self._dremio_rest_connection)}"
        )

        cache_key = self._get_cache_key("get_tables", schema_name or "default")
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cast(list[str], cached_result)

        # Ensure connection
        if not self._dremio_rest_connection:
            try:
                self._ensure_connection()
            except RuntimeError as e:
                logger.error(f"get_tables: Connection check failed: {e}")
                raise

        logger.debug(
            f"get_tables: After ensure_connection, engine exists: {self.has_engine()}"
        )

        if self._driver:
            tables: list[str] = self._driver.get_tables(schema_name)
            self._store_in_cache(cache_key, tables)
            return tables

        raise RuntimeError("No driver available")

    def get_views(self, schema_name: str | None = None) -> list[ViewInfo]:
        """Get views in a schema, with their SQL definitions.

        Mirrors :meth:`get_tables` (same caching, same connection handling).
        A driver that cannot enumerate views inherits the empty base default,
        so this returns an empty list rather than raising.

        Args:
            schema_name: Schema to inspect, or None for the default schema.

        Returns:
            List of ViewInfo, sorted by the driver's own ordering.
        """
        cache_key = self._get_cache_key("get_views", schema_name or "default")
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cast(list[ViewInfo], cached_result)

        if not self._dremio_rest_connection:
            try:
                self._ensure_connection()
            except RuntimeError as e:
                logger.error(f"get_views: Connection check failed: {e}")
                raise

        if not self._driver:
            raise RuntimeError("No driver available")

        raw: dict[str, str | None] = self._driver.get_views(schema_name)
        views = [
            ViewInfo(name=name, schema=schema_name or "", definition=definition)
            for name, definition in raw.items()
        ]
        self._store_in_cache(cache_key, views)
        return views

    def prefetch_schema_constraints(self, schema_name: str) -> None:
        """Prefetch all PKs and FKs for a schema at once (Snowflake optimization).

        This avoids repeated SHOW PRIMARY KEYS/IMPORTED KEYS queries for each table.
        Results are cached and used by analyze_table.
        """
        db_type = self.connection_info.get("type", "")
        if db_type != "snowflake":
            return

        from .drivers.snowflake import SnowflakeDriver

        if isinstance(self._driver, SnowflakeDriver):
            self._driver.prefetch_schema_constraints(
                schema_name=schema_name,
                connection_info=self.connection_info,
                cache_get=self._get_from_cache,
                cache_store=self._store_in_cache,
                log_sql=self._log_sql_query,
            )

    def analyze_table(
        self, table_name: str, schema_name: str | None = None
    ) -> TableInfo | None:
        """Analyze a specific table and return detailed information."""
        logger.debug(
            f"analyze_table: Starting analysis of {table_name}, "
            f"has_engine: {self.has_engine()}, "
            f"dremio_rest: {bool(self._dremio_rest_connection)}"
        )

        if not self._dremio_rest_connection:
            try:
                self._ensure_connection()
            except RuntimeError as e:
                logger.error(f"analyze_table: Connection check failed: {e}")
                raise

        logger.debug(
            f"analyze_table: After ensure_connection, engine exists: {self.has_engine()}"
        )

        if not self._driver:
            raise RuntimeError("No driver available")

        # Snowflake driver accepts extra cache_get / log_sql kwargs
        from .drivers.snowflake import SnowflakeDriver

        if isinstance(self._driver, SnowflakeDriver):
            return self._driver.analyze_table(
                table_name,
                schema_name,
                cache_get=self._get_from_cache,
                log_sql=self._log_sql_query,
            )
        return cast(
            TableInfo | None,
            self._driver.analyze_table(table_name, schema_name),
        )

    # ------------------------------------------------------------------
    # Sample & query (delegated to driver with validation layer)
    # ------------------------------------------------------------------

    def sample_table_data(
        self,
        table_name: str,
        schema_name: str | None = None,
        limit: int = DEFAULT_SAMPLE_LIMIT,
    ) -> list[dict[str, Any]]:
        """Sample data from a table for analysis with enhanced validation."""
        # Dremio REST bypasses engine check
        if not self._dremio_rest_connection:
            if not self.engine:
                raise RuntimeError("No database connection established")
            if not self._validate_identifier_secure(table_name):
                logger.error(f"Invalid table name format: {table_name}")
                raise ValueError(f"Invalid table name format: {table_name}")
            if schema_name and not self._validate_identifier_secure(schema_name):
                logger.error(f"Invalid schema name format: {schema_name}")
                raise ValueError(f"Invalid schema name format: {schema_name}")

        if not self._driver:
            raise RuntimeError("No driver available")

        return cast(
            list[dict[str, Any]],
            self._driver.sample_table_data(table_name, schema_name, limit),
        )

    def validate_sql_syntax(self, sql_query: str) -> dict[str, Any]:
        """Validate SQL query syntax with enhanced security checks.

        Uses both security validation and database-level validation to provide
        comprehensive protection against SQL injection and syntax errors.
        """
        if not self.engine and not self._dremio_rest_connection:
            raise RuntimeError("No database connection established")

        # Security validation
        security_validation = sql_validator.validate_query(sql_query)

        validation_result = {
            "is_valid": False,
            "error": None,
            "error_type": None,
            "database_error": None,
            "query_type": None,
            "affected_tables": [],
            "warnings": [],
            "suggestions": [],
            "security_issues": security_validation.get("issues", []),
            "risk_level": security_validation.get("risk_level", "low"),
        }

        if not security_validation.get("is_safe", False):
            validation_result["error"] = (
                f"Security validation failed: {'; '.join(security_validation['issues'])}"
            )
            validation_result["error_type"] = "security_error"
            audit_log_security_event(
                "sql_injection_attempt",
                {
                    "query_preview": sql_query[:100],
                    "issues": security_validation["issues"],
                    "risk_level": security_validation["risk_level"],
                },
                (
                    SecurityLevel.CRITICAL
                    if security_validation["risk_level"] == "critical"
                    else SecurityLevel.HIGH
                ),
            )
            return validation_result

        try:
            query_stripped = sql_query.strip()
            if not query_stripped:
                validation_result["error"] = "Empty query"
                validation_result["error_type"] = "empty_query"
                return validation_result

            # --- Primary safety gate: dialect-aware parsed validation ---
            # sqlglot parsing is authoritative for multi-statement detection and
            # read-only enforcement: it catches DML hidden after a CTE
            # (WITH x AS (...) INSERT ...) and ignores semicolons inside string
            # literals — both of which the legacy string heuristics get wrong.
            # Those heuristics now run only when the parser cannot handle the SQL.
            db_type = (self.connection_info or {}).get("type", "")
            dialect = DB_SQLGLOT_DIALECTS.get(db_type, "postgres")
            parsed = analyze_sql_statement(query_stripped, dialect=dialect)

            if parsed["parsed"]:
                if not parsed["single_statement"]:
                    validation_result["error"] = (
                        "Multiple SQL statements not allowed for security"
                    )
                    validation_result["error_type"] = "security_error"
                    validation_result["suggestions"].append(
                        "Split multiple statements into separate requests"
                    )
                    return validation_result
                if parsed["query_type"] == "WRITE":
                    validation_result["error"] = (
                        f"Destructive operations not allowed: {', '.join(parsed['write_operations'])}"
                    )
                    validation_result["error_type"] = "forbidden_operation"
                    validation_result["suggestions"].append(
                        "Use SELECT queries for data retrieval only"
                    )
                    return validation_result
                if parsed["query_type"] == "UNKNOWN":
                    validation_result["error"] = (
                        "Only SELECT, CTE, and metadata queries are allowed"
                    )
                    validation_result["error_type"] = "query_type_error"
                    validation_result["suggestions"].append(
                        "Start your query with SELECT, WITH, EXPLAIN, or SHOW"
                    )
                    return validation_result
                validation_result["query_type"] = parsed["query_type"]
            else:
                # Fallback: parser could not handle this SQL (dialect quirk) —
                # use the legacy string heuristics so valid queries still pass.
                query_without_comments = self._strip_leading_sql_comments(
                    query_stripped
                )
                query_upper = query_without_comments.upper()

                if ";" in query_stripped[:-1]:
                    validation_result["error"] = (
                        "Multiple SQL statements not allowed for security"
                    )
                    validation_result["error_type"] = "security_error"
                    validation_result["suggestions"].append(
                        "Split multiple statements into separate requests"
                    )
                    return validation_result

                if query_upper.startswith("SELECT"):
                    validation_result["query_type"] = "SELECT"
                elif query_upper.startswith("WITH"):
                    validation_result["query_type"] = "CTE_SELECT"
                    if "SELECT" not in query_upper:
                        validation_result["warnings"].append(
                            "CTE should end with SELECT statement"
                        )
                elif query_upper.startswith(("EXPLAIN", "DESCRIBE", "DESC", "SHOW")):
                    validation_result["query_type"] = "METADATA"
                else:
                    dangerous_ops = [
                        "DROP",
                        "DELETE",
                        "TRUNCATE",
                        "ALTER",
                        "CREATE",
                        "INSERT",
                        "UPDATE",
                        "MERGE",
                    ]
                    detected_ops = [
                        op for op in dangerous_ops if query_upper.startswith(op)
                    ]
                    if detected_ops:
                        validation_result["error"] = (
                            f"Destructive operations not allowed: {', '.join(detected_ops)}"
                        )
                        validation_result["error_type"] = "forbidden_operation"
                        validation_result["suggestions"].append(
                            "Use SELECT queries for data retrieval only"
                        )
                        return validation_result
                    else:
                        validation_result["error"] = (
                            "Only SELECT, CTE, and metadata queries are allowed"
                        )
                        validation_result["error_type"] = "query_type_error"
                        validation_result["suggestions"].append(
                            "Start your query with SELECT, WITH, EXPLAIN, or SHOW"
                        )
                        return validation_result

            # Database-level syntax validation - delegate to driver
            if not self._driver:
                raise RuntimeError("No driver available")

            validation_result = self._driver.validate_sql_syntax(
                query_stripped, validation_result
            )

            # Extract table references if validation succeeded
            if validation_result["is_valid"]:
                table_patterns = [
                    r'\bFROM\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',
                    r'\bJOIN\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',
                    r'\bUPDATE\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',
                    r'\bINTO\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',
                ]
                tables: set[str] = set()
                for pattern in table_patterns:
                    matches = re.findall(pattern, query_stripped, re.IGNORECASE)
                    tables.update(match.strip("\"'`[]") for match in matches)

                validation_result["affected_tables"] = list(tables)

                if len(tables) > 5:
                    validation_result["warnings"].append(
                        f"Query involves {len(tables)} tables - consider query complexity"
                    )

        except Exception as e:
            validation_result["error"] = f"Validation system error: {e!s}"
            validation_result["error_type"] = "internal_error"
            logger.error(f"SQL validation error: {e}")

        return validation_result

    def execute_sql_query(self, sql_query: str, limit: int = 1000) -> dict[str, Any]:
        """Execute a validated SQL query and return results safely."""
        if not self._driver:
            if self._dremio_rest_connection or self.engine:
                raise RuntimeError("No driver available - use connect_* methods")
            raise RuntimeError("No database connection established")

        # Validate and cap the limit
        if limit <= 0 or limit > 5000:
            limit = min(max(limit, 100), 5000)

        # Mandatory validation
        validation = self.validate_sql_syntax(sql_query)
        if not validation["is_valid"]:
            result_data = {
                "success": False,
                "data": [],
                "columns": [],
                "row_count": 0,
                "execution_time_ms": None,
                "error": validation["error"],
                "error_type": validation["error_type"],
                "database_error": validation.get("database_error"),
                "warnings": [],
                "query_plan": None,
                "limit_applied": False,
            }
            return result_data

        # Apply safety limits
        query_to_execute = sql_query.strip().rstrip(";")
        query_upper = query_to_execute.upper()

        needs_limit = (
            validation["query_type"] in ["SELECT", "CTE_SELECT"]
            and "LIMIT" not in query_upper
            and "TOP " not in query_upper
        )

        warnings = list(validation.get("warnings", []))
        limit_applied = False
        if needs_limit:
            query_to_execute = f"{query_to_execute} LIMIT {limit}"
            limit_applied = True
            warnings.append(
                f"Safety LIMIT {limit} applied to prevent large result sets"
            )

        # Delegate execution to driver
        result_data = self._driver.execute_sql_query(query_to_execute, limit)

        # Merge warnings
        result_data.setdefault("warnings", [])
        result_data["warnings"] = warnings + result_data["warnings"]
        if limit_applied:
            result_data["limit_applied"] = True
            if result_data.get("row_count", 0) == limit and result_data.get(
                "limit_applied"
            ):
                result_data["warnings"].append(
                    f"Result set may be truncated at {limit} rows"
                )

        return cast(dict[str, Any], result_data)

    # ------------------------------------------------------------------
    # Connection status & lifecycle
    # ------------------------------------------------------------------

    def has_engine(self) -> bool:
        """Check if database connection exists (engine for SQL databases, REST for Dremio)."""
        if self._dremio_rest_connection:
            return True
        return self.engine is not None

    def is_connected(self) -> bool:
        """Check if database is currently connected and healthy."""
        if self._dremio_rest_connection:
            return self._test_dremio_connection()
        return self.has_engine() and self._test_connection()

    def disconnect(self) -> None:
        """Close the database connection and clear stored parameters."""
        # Clear metadata cache
        if hasattr(self, "_metadata_cache"):
            self._metadata_cache.clear()

        # Disconnect driver
        if self._driver:
            self._driver.disconnect()
            self._driver = None

        # Clear legacy state
        if self.engine:
            self.engine.dispose()
        self.engine = None
        self.metadata = None
        self.connection_info = {}
        self._last_connection_params = None
        self._dremio_rest_connection = None
        logger.info("Database connection closed and parameters cleared")

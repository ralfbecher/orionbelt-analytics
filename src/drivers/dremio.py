"""Dremio database driver (REST API)."""

import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from ..async_utils import run_async
from ..constants import (
    CONNECTION_TIMEOUT,
    DEFAULT_SAMPLE_LIMIT,
    DREMIO_SYSTEM_SCHEMAS,
    MAX_SAMPLE_LIMIT,
    MIN_SAMPLE_LIMIT,
    QUERY_TIMEOUT,
)
from ..database_manager import ColumnInfo, TableInfo
from .base import DatabaseDriver

if TYPE_CHECKING:
    from ..dremio_client import DremioClient

logger = logging.getLogger(__name__)


class DremioDriver(DatabaseDriver):
    """Dremio-specific database operations via REST API."""

    db_type = "dremio"

    def __init__(self) -> None:
        self._rest_connection: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Dremio client factory (replaces 8 duplicated patterns)
    # ------------------------------------------------------------------

    def _create_dremio_client(self) -> Coroutine[Any, Any, "DremioClient"]:
        """Create a Dremio client based on the stored connection parameters.

        Returns an *awaitable* that resolves to a DremioClient context manager.
        Must be used as:
            async with await self._create_dremio_client() as client:
                ...
        """
        from ..dremio_client import create_dremio_client

        conn = self._rest_connection
        if not conn:
            raise RuntimeError("No Dremio REST connection configured")

        if conn.get("uri") and conn.get("pat"):
            return create_dremio_client(uri=conn["uri"], pat=conn["pat"])
        else:
            return create_dremio_client(
                host=conn.get("host"),
                port=conn.get("port", 9047),
                username=conn.get("username"),
                password=conn.get("password"),
                ssl=conn.get("ssl", False),
            )

    # ------------------------------------------------------------------
    # Identifier helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_sql_literal(value: str) -> str:
        """Escape a value for safe use inside a single-quoted SQL literal."""
        return value.replace("'", "''")

    @staticmethod
    def _quote_dremio_identifier(identifier: str | None) -> str:
        """Quote and escape a Dremio identifier or path safely."""
        if identifier is None:
            return ""
        parts = [p for p in str(identifier).split(".") if p != ""]
        if not parts:
            return ""
        escaped_parts = [part.replace('"', '""') for part in parts]
        return ".".join(f'"{part}"' for part in escaped_parts)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, **params: Any) -> bool:
        """Connect to Dremio via REST API.

        Expected params: (uri + pat) or (host + port + username + password + ssl).
        """
        uri = params.get("uri")
        pat = params.get("pat")
        host = params.get("host")
        port = params.get("port")
        username = params.get("username")
        password = params.get("password")
        ssl = params.get("ssl", False)

        logger.info(
            f"\U0001f50d MCP DEBUG - connect_dremio called with "
            f"host={host}, port={port}, username={username}, "
            f"password={'***SET***' if password else 'NOT SET'}, ssl={ssl}, "
            f"uri={uri}, pat={'***SET***' if pat else 'NOT SET'}"
        )

        try:
            if uri and pat:
                logger.info(
                    "\U0001f50d MCP DEBUG - Using PAT-based authentication (preferred)"
                )
            elif host and username:
                logger.info(
                    "\U0001f50d MCP DEBUG - Using legacy username/password authentication"
                )
            else:
                logger.error(
                    "\U0001f50d MCP DEBUG - Missing required Dremio connection "
                    "parameters. Need either (uri + pat) or (host + username)"
                )
                return False

            # Store REST connection params *before* testing
            api_port = 9047
            if uri and pat:
                self._rest_connection = {"uri": uri, "pat": pat}
            else:
                self._rest_connection = {
                    "host": host,
                    "port": api_port,
                    "username": username,
                    "password": password,
                    "ssl": ssl,
                }

            # Test connection via factory
            async def test_dremio_connection() -> dict[str, Any]:
                async with await self._create_dremio_client() as client:
                    return await client.test_connection()

            logger.debug("Starting async connection test")
            connection_result = run_async(
                test_dremio_connection(), timeout=CONNECTION_TIMEOUT
            )
            logger.info(
                f"\U0001f50d MCP DEBUG - Async connection test result: "
                f"{connection_result}"
            )

            if not connection_result.get("success"):
                error_msg = connection_result.get("error", "Unknown error")
                error_type = connection_result.get("error_type", "Unknown error type")
                logger.error(
                    f"\U0001f50d MCP DEBUG - Dremio connection test failed: "
                    f"{error_msg} (Type: {error_type})"
                )
                logger.error(f"Full connection result: {connection_result}")
                self._rest_connection = None
                return False

            if uri and pat:
                logger.info(
                    f"\u2705 Successfully connected to Dremio via REST API at {uri}"
                )
            else:
                logger.info(
                    f"\u2705 Successfully connected to Dremio via REST API "
                    f"at {host}:{api_port}"
                )
            return True

        except Exception as e:
            logger.error(f"\u274c Failed to connect to Dremio: {type(e).__name__}: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            self._rest_connection = None
            return False

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        async def fetch_schemas() -> list[str]:
            try:
                async with await self._create_dremio_client() as client:
                    catalogs = await client.get_catalogs()
                    schemas: set[str] = set()

                    logger.debug(f"Dremio catalogs response: {catalogs}")

                    for catalog in catalogs:
                        path = catalog.get("path", [])
                        catalog_type = catalog.get("type", "")
                        catalog_id = catalog.get("id", "")

                        logger.debug(
                            f"Top-level catalog: path={path}, "
                            f"type={catalog_type}, id={catalog_id}"
                        )

                        if isinstance(path, list) and len(path) > 0:
                            top_level = path[0]
                            full_path = ".".join(path)

                            if top_level and top_level not in DREMIO_SYSTEM_SCHEMAS:
                                schemas.add(top_level)

                            if len(path) > 1 and full_path not in DREMIO_SYSTEM_SCHEMAS:
                                schemas.add(full_path)

                            if (
                                catalog_type in ("CONTAINER", "SPACE", "SOURCE")
                                and catalog_id
                            ):
                                await self._add_dremio_children_recursive(
                                    client, path, schemas, max_depth=3
                                )
                        else:
                            catalog_name = (
                                catalog.get("name") or str(path) if path else ""
                            )
                            if (
                                catalog_name
                                and catalog_name not in DREMIO_SYSTEM_SCHEMAS
                            ):
                                schemas.add(catalog_name)

                    schema_list = sorted(schemas)

                    if not schema_list:
                        logger.info("No catalogs found, using default Dremio spaces")
                        schema_list = ["@dremio", "Samples"]

                    logger.info(
                        f"Found {len(schema_list)} Dremio schemas/spaces: "
                        f"{schema_list[:10]}"
                    )
                    return schema_list

            except Exception as e:
                logger.error(f"Failed to get Dremio schemas via REST: {e}")
                return []

        return run_async(fetch_schemas(), timeout=CONNECTION_TIMEOUT)

    async def _add_dremio_children_recursive(
        self,
        client: "DremioClient",
        path: list[str],
        schemas: set[str],
        max_depth: int = 3,
        current_depth: int = 0,
    ) -> None:
        """Recursively add children of Dremio containers to the schemas set."""
        if current_depth >= max_depth:
            logger.debug(f"Max depth {max_depth} reached for path: {'.'.join(path)}")
            return

        try:
            catalog_info = await client.get_catalog_info(path)
            logger.debug(f"Catalog info for {'.'.join(path)}: {catalog_info}")

            children = catalog_info.get("children", [])
            if not children:
                logger.debug(f"No children found for path: {'.'.join(path)}")
                return

            for child in children:
                child_path = child.get("path", [])
                child_type = child.get("type", "")

                if not child_path or not isinstance(child_path, list):
                    continue

                if child_type in (
                    "CONTAINER",
                    "SPACE",
                    "SOURCE",
                    "FOLDER",
                    "HOME",
                ):
                    full_child_path = ".".join(child_path)
                    if full_child_path and full_child_path not in DREMIO_SYSTEM_SCHEMAS:
                        schemas.add(full_child_path)
                        logger.debug(
                            f"Added child schema (type: {child_type}): "
                            f"{full_child_path}"
                        )

                    if current_depth < max_depth - 1:
                        await self._add_dremio_children_recursive(
                            client,
                            child_path,
                            schemas,
                            max_depth,
                            current_depth + 1,
                        )
                else:
                    logger.debug(
                        f"Skipping non-container (type: {child_type}): "
                        f"{'.'.join(child_path)}"
                    )

        except Exception as e:
            logger.warning(f"Failed to get children for path {'.'.join(path)}: {e}")

    def get_tables(self, schema_name: str | None = None) -> list[str]:
        async def fetch_tables() -> list[str]:
            try:
                async with await self._create_dremio_client() as client:
                    if not schema_name:
                        query = (
                            "SELECT TABLE_SCHEMA, TABLE_NAME "
                            'FROM INFORMATION_SCHEMA."TABLES" '
                            "WHERE TABLE_TYPE = 'TABLE'"
                        )
                    else:
                        safe_schema = self._escape_sql_literal(schema_name)
                        query = (
                            f"SELECT TABLE_NAME "
                            f'FROM INFORMATION_SCHEMA."TABLES" '
                            f"WHERE TABLE_SCHEMA = '{safe_schema}' "
                            f"AND TABLE_TYPE = 'TABLE'"
                        )

                    result = await client.execute_query(query)

                    if result.get("success"):
                        tables = []
                        for row in result.get("data", []):
                            if schema_name:
                                table_name = row.get("TABLE_NAME", "")
                            else:
                                schema = row.get("TABLE_SCHEMA", "")
                                table_name = row.get("TABLE_NAME", "")
                                if schema and schema != "INFORMATION_SCHEMA":
                                    table_name = (
                                        f"{schema}.{table_name}" if table_name else ""
                                    )
                            if table_name:
                                tables.append(table_name)
                        return sorted(set(tables))
                    else:
                        logger.error(
                            f"Failed to get Dremio tables: {result.get('error')}"
                        )
                        return []

            except Exception as e:
                logger.error(f"Failed to get Dremio tables via REST: {e}")
                return []

        return run_async(fetch_tables(), timeout=CONNECTION_TIMEOUT)

    def analyze_table(
        self, table_name: str, schema_name: str | None = None
    ) -> TableInfo | None:
        async def fetch_table_info() -> TableInfo | None:
            try:
                nonlocal table_name, schema_name
                async with await self._create_dremio_client() as client:
                    # If schema_name isn't provided but table_name is qualified
                    if not schema_name and "." in str(table_name):
                        parts = str(table_name).split(".")
                        schema_name = ".".join(parts[:-1]) if len(parts) > 1 else None
                        table_name = parts[-1]

                    if schema_name:
                        safe_schema = self._escape_sql_literal(schema_name)
                        safe_table = self._escape_sql_literal(table_name)
                        column_query = f"""
                        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE,
                               COLUMN_DEFAULT, ORDINAL_POSITION
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{safe_table}'
                          AND TABLE_SCHEMA = '{safe_schema}'
                        ORDER BY ORDINAL_POSITION
                        """
                    else:
                        safe_table = self._escape_sql_literal(table_name)
                        column_query = f"""
                        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE,
                               COLUMN_DEFAULT, ORDINAL_POSITION
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{safe_table}'
                        ORDER BY ORDINAL_POSITION
                        """

                    result = await client.execute_query(column_query)

                    if not result.get("success"):
                        logger.error(
                            f"Failed to get column info for {table_name}: "
                            f"{result.get('error')}"
                        )
                        return None

                    columns = []
                    for row in result.get("data", []):
                        col_info = ColumnInfo(
                            name=row.get("COLUMN_NAME", ""),
                            data_type=row.get("DATA_TYPE", "VARCHAR"),
                            is_nullable=row.get("IS_NULLABLE", "YES") == "YES",
                            is_primary_key=False,
                            is_foreign_key=False,
                            foreign_key_table=None,
                            foreign_key_column=None,
                            comment=None,
                        )
                        columns.append(col_info)

                    return TableInfo(
                        name=table_name,
                        schema=schema_name or "default",
                        columns=columns,
                        primary_keys=[],
                        foreign_keys=[],
                        comment=None,
                    )

            except Exception as e:
                logger.error(f"Failed to analyze Dremio table {table_name}: {e}")
                return None

        return run_async(fetch_table_info(), timeout=CONNECTION_TIMEOUT)

    # ------------------------------------------------------------------
    # Query validation & execution
    # ------------------------------------------------------------------

    def validate_sql_syntax(
        self, sql_query: str, validation_result: dict[str, Any]
    ) -> dict[str, Any]:
        async def validate_query() -> dict[str, Any]:
            try:
                async with await self._create_dremio_client() as client:
                    explain_sql = f"EXPLAIN PLAN FOR {sql_query}"
                    return await client.execute_query(explain_sql)
            except Exception as e:
                return {"success": False, "error": str(e)}

        try:
            explain_result = run_async(validate_query(), timeout=QUERY_TIMEOUT)

            if explain_result.get("success"):
                validation_result["is_valid"] = True
                return validation_result
            else:
                error_msg = explain_result.get(
                    "error", "Unknown Dremio validation error"
                )
                validation_result["database_error"] = error_msg
                validation_result["error"] = f"Dremio syntax error: {error_msg}"
                validation_result["error_type"] = "syntax_error"

                if (
                    "not found" in error_msg.lower()
                    or "does not exist" in error_msg.lower()
                ):
                    validation_result["suggestions"].append(
                        "Object not found - check table/view names and "
                        "ensure proper qualification"
                    )
                elif "syntax error" in error_msg.lower():
                    validation_result["suggestions"].append(
                        "SQL syntax error - review query structure and keywords"
                    )
                elif "validation error" in error_msg.lower():
                    validation_result["suggestions"].append(
                        "Query validation failed - check column references "
                        "and data types"
                    )

                return validation_result

        except Exception as e:
            validation_result["is_valid"] = True
            validation_result["warnings"].append(
                f"Could not validate syntax via Dremio: {e!s}"
            )
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
            "limit_applied": False,
        }

        try:
            start_time = time_mod.time()

            async def run_dremio_query() -> dict[str, Any]:
                async with await self._create_dremio_client() as client:
                    return await client.execute_query(sql_query, limit)

            query_result = run_async(run_dremio_query(), timeout=QUERY_TIMEOUT)

            execution_time = (time_mod.time() - start_time) * 1000
            result_data["execution_time_ms"] = execution_time

            if query_result.get("success"):
                result_data["success"] = True
                result_data["data"] = query_result.get("data", [])
                result_data["columns"] = query_result.get("columns", [])
                result_data["row_count"] = query_result.get("row_count", 0)

                total_rows = query_result.get("total_rows", result_data["row_count"])
                if total_rows > limit:
                    result_data["limit_applied"] = True
                    result_data["warnings"].append(
                        f"Results limited to {limit} rows (total: {total_rows})"
                    )

                logger.info(
                    f"Dremio query executed successfully: "
                    f"{result_data['row_count']} rows in "
                    f"{execution_time:.2f}ms"
                )
            else:
                result_data["error"] = query_result.get("error", "Unknown Dremio error")
                result_data["error_type"] = query_result.get(
                    "error_type", "dremio_error"
                )
                logger.error(f"Dremio query failed: {result_data['error']}")

        except Exception as e:
            result_data["error"] = f"Dremio execution error: {e!s}"
            result_data["error_type"] = "dremio_connection_error"
            logger.error(f"Dremio query execution failed: {e}")

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

        async def fetch_sample() -> list[dict[str, Any]]:
            try:
                async with await self._create_dremio_client() as client:
                    if schema_name:
                        full_table_name = (
                            f"{self._quote_dremio_identifier(schema_name)}"
                            f".{self._quote_dremio_identifier(table_name)}"
                        )
                    else:
                        full_table_name = self._quote_dremio_identifier(table_name)

                    query = f"SELECT * FROM {full_table_name} LIMIT {limit}"
                    result = await client.execute_query(query, limit)

                    if result.get("success"):
                        sample_rows: list[dict[str, Any]] = result.get("data", [])
                        return sample_rows
                    else:
                        error_msg = result.get("error", "Unknown error")
                        logger.error(
                            f"Failed to sample Dremio table {table_name}: "
                            f"{error_msg}"
                        )
                        raise RuntimeError(
                            f"Failed to sample Dremio table: {error_msg}"
                        )

            except Exception as e:
                logger.error(f"Error sampling Dremio table {table_name}: {e}")
                raise RuntimeError(f"Error sampling Dremio table: {e!s}")

        return run_async(fetch_sample(), timeout=CONNECTION_TIMEOUT)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        if not self._rest_connection:
            return False

        async def test_dremio() -> dict[str, Any]:
            async with await self._create_dremio_client() as client:
                return await client.test_connection()

        try:
            result = run_async(test_dremio(), timeout=CONNECTION_TIMEOUT)
            return bool(result.get("success"))
        except Exception as e:
            logger.warning(f"Dremio health check failed: {e}")
            return False

    def disconnect(self) -> None:
        self._rest_connection = None
        logger.info("Dremio REST connection closed")

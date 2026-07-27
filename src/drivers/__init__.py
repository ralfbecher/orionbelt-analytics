"""Database driver implementations for OrionBelt Analytics.

Each driver encapsulates database-specific logic for connection,
schema introspection, query execution, and validation.

The authoritative mapping of ``db_type`` -> driver class + sqlglot dialect lives
in :mod:`.registry`; the concrete classes are re-exported here for convenience.
"""

from .base import DatabaseDriver
from .bigquery import BigQueryDriver
from .clickhouse import ClickHouseDriver
from .databricks import DatabricksDriver
from .dremio import DremioDriver
from .duckdb import DuckDBDriver
from .mysql import MySQLDriver
from .postgresql import PostgreSQLDriver
from .registry import (
    DATABASE_REGISTRY,
    DriverMeta,
    dialect_for,
    get_driver_class,
    supported_db_types,
)
from .snowflake import SnowflakeDriver

__all__ = [
    # Registry (single source of truth for db_type -> driver/dialect)
    "DATABASE_REGISTRY",
    "BigQueryDriver",
    "ClickHouseDriver",
    "DatabaseDriver",
    "DatabricksDriver",
    "DremioDriver",
    "DriverMeta",
    "DuckDBDriver",
    "MySQLDriver",
    # Concrete drivers
    "PostgreSQLDriver",
    "SnowflakeDriver",
    "dialect_for",
    "get_driver_class",
    "supported_db_types",
]

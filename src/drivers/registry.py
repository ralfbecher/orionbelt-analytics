"""Central registry of supported databases and their drivers.

Single source of truth that ties each ``db_type`` to its driver class and
sqlglot dialect. Consumers (``DatabaseManager``, the connection handler, OBQC)
read from here instead of maintaining parallel lists, which is what previously
let the supported-database list, the driver imports, and the OBQC dialect map
drift apart.

The dialect data lives in :mod:`src.constants` (a dependency-free leaf module so
the driver modules can import it); this registry layers the driver *classes* on
top.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import DB_SQLGLOT_DIALECTS
from .base import DatabaseDriver
from .bigquery import BigQueryDriver
from .clickhouse import ClickHouseDriver
from .databricks import DatabricksDriver
from .dremio import DremioDriver
from .duckdb import DuckDBDriver
from .mysql import MySQLDriver
from .postgresql import PostgreSQLDriver
from .snowflake import SnowflakeDriver


@dataclass(frozen=True)
class DriverMeta:
    """Metadata describing one supported database."""

    db_type: str
    driver_cls: type[DatabaseDriver]
    dialect: str  # sqlglot dialect name


# Map of db_type -> driver class. The single place the driver classes are
# enumerated; everything else derives from this plus DB_SQLGLOT_DIALECTS.
_DRIVER_CLASSES: dict[str, type[DatabaseDriver]] = {
    "postgresql": PostgreSQLDriver,
    "snowflake": SnowflakeDriver,
    "dremio": DremioDriver,
    "clickhouse": ClickHouseDriver,
    "bigquery": BigQueryDriver,
    "duckdb": DuckDBDriver,
    "databricks": DatabricksDriver,
    "mysql": MySQLDriver,
}

DATABASE_REGISTRY: dict[str, DriverMeta] = {
    db_type: DriverMeta(
        db_type=db_type,
        driver_cls=driver_cls,
        dialect=DB_SQLGLOT_DIALECTS[db_type],
    )
    for db_type, driver_cls in _DRIVER_CLASSES.items()
}


def supported_db_types() -> list[str]:
    """Return the supported db_type identifiers, in registration order."""
    return list(DATABASE_REGISTRY)


def get_driver_class(db_type: str) -> type[DatabaseDriver]:
    """Return the driver class for ``db_type``.

    Args:
        db_type: One of the supported database type identifiers.

    Raises:
        ValueError: If ``db_type`` is not a supported database.
    """
    meta = DATABASE_REGISTRY.get(db_type)
    if meta is None:
        raise ValueError(
            f"Unsupported database type '{db_type}'. "
            f"Supported: {', '.join(DATABASE_REGISTRY)}."
        )
    return meta.driver_cls


def dialect_for(db_type: str) -> str:
    """Return the sqlglot dialect for ``db_type`` (postgres if unknown)."""
    meta = DATABASE_REGISTRY.get(db_type)
    return meta.dialect if meta is not None else "postgres"

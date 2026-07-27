"""Abstract base class for database drivers."""

from abc import ABC, abstractmethod
from typing import Any

from ..database_manager import TableInfo


class DatabaseDriver(ABC):
    """Protocol for database-specific operations.

    Each concrete driver encapsulates all logic specific to one database
    backend (connection building, schema introspection, query validation,
    query execution, sampling).  DatabaseManager delegates to the active
    driver and handles cross-cutting concerns (caching, credentials,
    reconnection).
    """

    # Subclasses must set this to the database type string
    db_type: str = ""

    @abstractmethod
    def connect(self, **params: Any) -> bool:
        """Establish a connection to the database.

        Args:
            **params: Database-specific connection parameters.

        Returns:
            True if the connection was established successfully.
        """

    @abstractmethod
    def get_schemas(self) -> list[str]:
        """Return a list of user-visible schema names."""

    @abstractmethod
    def get_tables(self, schema_name: str | None = None) -> list[str]:
        """Return a list of table names in the given schema."""

    @abstractmethod
    def analyze_table(
        self, table_name: str, schema_name: str | None = None
    ) -> TableInfo | None:
        """Analyze a table and return its metadata."""

    @abstractmethod
    def validate_sql_syntax(
        self, sql_query: str, validation_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform database-level SQL syntax validation.

        Args:
            sql_query: The raw SQL query string.
            validation_result: A pre-populated dict with query_type, warnings, etc.
                The driver should set ``is_valid``, ``error``, ``error_type``,
                ``database_error``, and ``suggestions`` as appropriate.

        Returns:
            The updated ``validation_result`` dict.
        """

    @abstractmethod
    def execute_sql_query(self, sql_query: str, limit: int = 1000) -> dict[str, Any]:
        """Execute a SQL query and return structured results."""

    @abstractmethod
    def sample_table_data(
        self,
        table_name: str,
        schema_name: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return sample rows from a table."""

    @abstractmethod
    def test_connection(self) -> bool:
        """Return True if the connection is healthy."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection and release resources."""

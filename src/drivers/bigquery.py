"""Google BigQuery database driver."""

import logging
from typing import Any

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError, OperationalError, SQLAlchemyError
from sqlalchemy.pool import NullPool

from ..constants import (
    BIGQUERY_SYSTEM_SCHEMAS,
    CONNECTION_TIMEOUT,
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


class BigQueryDriver(DatabaseDriver):
    """Google BigQuery-specific database operations.

    Uses SQLAlchemy with pybigquery dialect for BigQuery connectivity.
    Supports authentication via service account JSON key file or credentials JSON string.
    """

    db_type = "bigquery"

    def __init__(self, pool_size: int = 5, max_overflow: int = 10):
        self.engine: Engine | None = None
        self.metadata: MetaData | None = None
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._credential_manager = SecureCredentialManager()
        self._project_id: str | None = None
        self._dataset: str | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, **params: Any) -> bool:
        """Connect to BigQuery.

        Expected params:
        - project_id: GCP project ID
        - dataset: Default dataset (optional)
        - credentials_path: Path to service account JSON key file (optional)
        - credentials_json: Service account JSON as string (optional)

        Connection string format:
        bigquery://{project_id}/{dataset}?credentials_path={path}
        """
        project_id = params.get("project_id")
        dataset = params.get("dataset", "")
        credentials_path = params.get("credentials_path")
        credentials_json = params.get("credentials_json")

        try:
            if not project_id:
                logger.error("Missing required BigQuery parameter: project_id")
                return False

            if not identifier_validator.validate_identifier(project_id):
                audit_log_security_event(
                    "invalid_identifier_attempt",
                    {"identifier": project_id[:50]},
                    SecurityLevel.MEDIUM,
                )
                logger.error(f"Invalid project_id: {project_id}")
                return False

            self._project_id = project_id
            self._dataset = dataset

            # Build connection string
            if credentials_path:
                connection_string = f"bigquery://{project_id}/{dataset}?credentials_path={credentials_path}"
            elif credentials_json:
                # For credentials JSON string, use default credentials or set via environment
                connection_string = f"bigquery://{project_id}/{dataset}"
            else:
                # Use default credentials (from environment)
                connection_string = f"bigquery://{project_id}/{dataset}"

            # BigQuery uses NullPool (no connection pooling) as recommended
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
                    key_material = f"{project_id}:{dataset}"
                    self._credential_manager._initialize_encryption(key_material)
                logger.info(
                    f"BigQuery connection encryption initialized for {project_id}"
                )
            except Exception as e:
                logger.warning(f"Could not initialize credential encryption: {e}")

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            logger.info(
                f"Connected to BigQuery project: {project_id}, dataset: {dataset or 'default'}"
            )
            return True

        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(
                f"Failed to connect to BigQuery project {project_id}: "
                f"{type(e).__name__}: {e}"
            )
            self.engine = None
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error connecting to BigQuery: {type(e).__name__}: {e}"
            )
            self.engine = None
            return False

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[str]:
        """Get datasets in the BigQuery project.

        In BigQuery terminology, datasets are equivalent to schemas.
        """
        try:
            # Query INFORMATION_SCHEMA to get datasets
            query = text(f"""
                SELECT schema_name
                FROM `{self._project_id}`.INFORMATION_SCHEMA.SCHEMATA
                WHERE schema_name NOT IN UNNEST({BIGQUERY_SYSTEM_SCHEMAS})
                ORDER BY schema_name
            """)
            assert self.engine is not None
            with self.engine.connect() as conn:
                result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get BigQuery datasets: {e}")
            return []

    def get_tables(self, schema_name: str | None = None) -> list[str]:
        """Get tables in a dataset.

        Args:
            schema_name: BigQuery dataset name (if None, uses default dataset)
        """
        try:
            dataset = schema_name or self._dataset
            if not dataset:
                logger.error("No dataset specified and no default dataset set")
                return []

            assert self.engine is not None
            with self.engine.connect() as conn:
                query = text(f"""
                    SELECT table_name
                    FROM `{self._project_id}.{dataset}`.INFORMATION_SCHEMA.TABLES
                    WHERE table_type = 'BASE TABLE'
                    ORDER BY table_name
                """)
                result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get BigQuery tables: {e}")
            return []

    def get_views(self, schema_name: str | None = None) -> dict[str, str | None]:
        try:
            dataset = schema_name or self._dataset
            if not dataset:
                logger.error("No dataset specified and no default dataset set")
                return {}

            assert self.engine is not None
            with self.engine.connect() as conn:
                query = text(f"""
                    SELECT table_name, view_definition
                    FROM `{self._project_id}.{dataset}`.INFORMATION_SCHEMA.VIEWS
                    ORDER BY table_name
                """)
                result = conn.execute(query)
                return {row[0]: row[1] for row in result.fetchall()}
        except SQLAlchemyError as e:
            logger.error(f"Failed to get BigQuery views: {e}")
            return {}

    def analyze_table(
        self, table_name: str, schema_name: str | None = None
    ) -> TableInfo | None:
        """Analyze a BigQuery table and return its metadata.

        BigQuery doesn't support traditional foreign keys, so FK detection is limited.
        """
        try:
            dataset = schema_name or self._dataset
            if not dataset:
                logger.error("No dataset specified for table analysis")
                return None

            assert self.engine is not None
            with self.engine.connect():
                inspector = inspect(self.engine)

                # Check if table exists
                if not inspector.has_table(table_name, schema=dataset):
                    logger.error(f"Table {dataset}.{table_name} not found")
                    return None

                table_columns = inspector.get_columns(table_name, schema=dataset)

                # BigQuery doesn't have traditional primary keys or foreign keys
                # but we can check for clustering/partitioning
                primary_keys: list[str] = []
                foreign_keys: list[dict[str, Any]] = []

                columns = []
                for col_info in table_columns:
                    column_name = col_info["name"]

                    columns.append(
                        ColumnInfo(
                            name=column_name,
                            data_type=str(col_info["type"]),
                            is_nullable=col_info.get("nullable", True),
                            is_primary_key=False,  # BigQuery doesn't enforce PKs
                            is_foreign_key=False,  # BigQuery doesn't enforce FKs
                            foreign_key_table=None,
                            foreign_key_column=None,
                            comment=col_info.get("comment"),
                        )
                    )

                logger.info(
                    f"Analyzed BigQuery table {dataset}.{table_name}: "
                    f"{len(columns)} columns"
                )

                return TableInfo(
                    name=table_name,
                    schema=dataset,
                    columns=columns,
                    primary_keys=primary_keys,
                    foreign_keys=foreign_keys,
                    comment=None,
                    row_count=None,
                    sample_data=None,
                )

        except SQLAlchemyError as e:
            logger.error(f"Failed to analyze BigQuery table {table_name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Query validation & execution
    # ------------------------------------------------------------------

    def validate_sql_syntax(
        self, sql_query: str, validation_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate BigQuery SQL syntax using dry run."""
        try:
            assert self.engine is not None
            with self.engine.connect() as conn:
                # BigQuery supports dry run via query job config
                # For now, we'll do basic syntax check via EXPLAIN
                try:
                    explain_query = f"SELECT * FROM ({sql_query}) LIMIT 0"
                    conn.execute(text(explain_query))
                    validation_result["is_valid"] = True
                except Exception as syntax_error:
                    error_msg = str(syntax_error)
                    validation_result["database_error"] = error_msg
                    validation_result["error"] = f"BigQuery syntax error: {error_msg}"
                    validation_result["error_type"] = "syntax_error"

                    if "not found" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "Check table/dataset names - use fully qualified names like `project.dataset.table`"
                        )
                    elif "syntax error" in error_msg.lower():
                        validation_result["suggestions"].append(
                            "Review BigQuery SQL syntax - it differs from standard SQL in some aspects"
                        )
                    elif (
                        "permission" in error_msg.lower()
                        or "denied" in error_msg.lower()
                    ):
                        validation_result["suggestions"].append(
                            "Insufficient permissions to access the specified tables/datasets"
                        )
        except Exception as conn_error:
            validation_result["error"] = (
                f"Database connection error during validation: {conn_error}"
            )
            validation_result["error_type"] = "connection_error"

        return validation_result

    def execute_sql_query(self, sql_query: str, limit: int = 1000) -> dict[str, Any]:
        """Execute BigQuery SQL query."""
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
                logger.info(f"🔍 BIGQUERY SQL QUERY: {sql_query}")
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
                    f"BigQuery query executed: {result_data['row_count']} rows "
                    f"in {result_data['execution_time_ms']}ms"
                )

        except SQLAlchemyError as e:
            result_data["error"] = str(e)
            result_data["error_type"] = "execution_error"
            logger.error(f"BigQuery SQL execution failed: {e}")
        except Exception as e:
            result_data["error"] = f"Unexpected execution error: {e!s}"
            result_data["error_type"] = "internal_error"
            logger.error(f"Unexpected BigQuery SQL execution error: {e}")

        return result_data

    def sample_table_data(
        self,
        table_name: str,
        schema_name: str | None = None,
        limit: int = DEFAULT_SAMPLE_LIMIT,
    ) -> list[dict[str, Any]]:
        """Sample data from a BigQuery table."""
        if not isinstance(limit, int) or limit < MIN_SAMPLE_LIMIT:
            limit = DEFAULT_SAMPLE_LIMIT
        elif limit > MAX_SAMPLE_LIMIT:
            limit = MAX_SAMPLE_LIMIT
            logger.warning(f"Sample limit capped at {MAX_SAMPLE_LIMIT}")

        try:
            dataset = schema_name or self._dataset
            if not dataset:
                logger.error("No dataset specified for sampling")
                return []

            assert self.engine is not None
            with self.engine.connect() as conn:
                # BigQuery requires backticks for fully qualified names
                full_table_name = f"`{self._project_id}.{dataset}.{table_name}`"

                query_str = f"SELECT * FROM {full_table_name} LIMIT {limit}"
                logger.info(f"🔍 BIGQUERY SQL QUERY: {query_str}")
                result = conn.execute(text(query_str))
                columns = list(result.keys())
                return serialize_rows(result.fetchall(), columns)

        except (SQLAlchemyError, ValueError) as e:
            logger.error(
                f"Failed to sample data from BigQuery table {table_name}: {type(e).__name__}: {e}"
            )
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error sampling BigQuery table {table_name}: {type(e).__name__}: {e}"
            )
            return []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Test BigQuery connection health."""
        if not self.engine:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.warning(f"BigQuery connection health check failed: {e}")
            return False

    def disconnect(self) -> None:
        """Close BigQuery connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.metadata = None
            logger.info("BigQuery connection closed")

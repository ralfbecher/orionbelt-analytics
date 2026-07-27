"""Database connection and diagnostic handler implementations."""

import logging
import os
from typing import cast

from fastmcp import Context

from ..constants import SUPPORTED_DB_TYPES
from ..exceptions import ConnectionError, ValidationError
from ..handler_context import HandlerContext
from ..lifecycle.metadata import mutate_workspace_metadata
from ..paths import OUTPUT_DIR
from ..utils import utc_now
from ..workspace import detect_workspace, format_workspace_summary
from .workspace import _format_restore_summary, _restore_workspace_core

logger = logging.getLogger(__name__)


async def connect_database(
    ctx: Context,
    db_type: str,
    services: "HandlerContext",
) -> str:
    """Connect to a database using credentials from environment variables.

    If a previous workspace exists for this connection, it is automatically
    restored (schema cache, ontology, GraphRAG, RDF store).

    Args:
        ctx: FastMCP context
        db_type: Database type - 'postgresql', 'snowflake', 'dremio', or 'clickhouse'
        get_session_db_manager: Function to get session db manager
        get_session_data: Function to get session data
        create_error_response: Error response helper
        _get_connection_fingerprint: Connection fingerprint function
        _clear_session_state: State clearing function
        get_oxigraph_store: Function to get/init Oxigraph store (for auto-restore)

    Returns:
        Connection status message or error JSON
    """
    # Validate input parameters
    if not db_type or db_type not in SUPPORTED_DB_TYPES:
        return cast(
            str,
            ValidationError(
                f"Invalid database type '{db_type}'. "
                f"Use one of: {', '.join(SUPPORTED_DB_TYPES)}."
            ).to_response(),
        )

    db_manager = services.get_session_db_manager(ctx)
    success = False
    db_name = ""

    if db_type == "postgresql":
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        database = os.getenv("POSTGRES_DATABASE")
        username = os.getenv("POSTGRES_USERNAME")
        password = os.getenv("POSTGRES_PASSWORD")

        required_params = {
            "POSTGRES_HOST": host,
            "POSTGRES_PORT": port,
            "POSTGRES_DATABASE": database,
            "POSTGRES_USERNAME": username,
            "POSTGRES_PASSWORD": password,
        }
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            return cast(
                str,
                ValidationError(
                    "Missing required environment variables for PostgreSQL: "
                    f"{', '.join(missing_params)}. Please check your .env file."
                ).to_response(),
            )

        success = db_manager.connect_postgresql(
            host=str(host),
            port=int(str(port)),
            database=str(database),
            username=str(username),
            password=str(password),
        )
        db_name = str(database)

    elif db_type == "snowflake":
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        username = os.getenv("SNOWFLAKE_USERNAME")
        password = os.getenv("SNOWFLAKE_PASSWORD")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        database = os.getenv("SNOWFLAKE_DATABASE")
        schema = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")

        required_params = {
            "SNOWFLAKE_ACCOUNT": account,
            "SNOWFLAKE_USERNAME": username,
            "SNOWFLAKE_PASSWORD": password,
            "SNOWFLAKE_WAREHOUSE": warehouse,
            "SNOWFLAKE_DATABASE": database,
        }
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            return cast(
                str,
                ValidationError(
                    "Missing required environment variables for Snowflake: "
                    f"{', '.join(missing_params)}. Please check your .env file."
                ).to_response(),
            )

        success = db_manager.connect_snowflake(
            account=str(account),
            username=str(username),
            password=str(password),
            warehouse=str(warehouse),
            database=str(database),
            schema=schema,
        )
        db_name = str(database)

    elif db_type == "dremio":
        # Prefer PAT-based authentication (DREMIO_URI + DREMIO_PAT)
        dremio_uri = os.getenv("DREMIO_URI")
        dremio_pat = os.getenv("DREMIO_PAT")

        if dremio_uri and dremio_pat:
            success = db_manager.connect_dremio(uri=dremio_uri, pat=dremio_pat)
            db_name = "DREMIO"
        else:
            # Fall back to legacy username/password authentication
            host = os.getenv("DREMIO_HOST")
            port = os.getenv("DREMIO_PORT")
            username = os.getenv("DREMIO_USERNAME")
            password = os.getenv("DREMIO_PASSWORD")

            required_params = {
                "DREMIO_HOST": host,
                "DREMIO_PORT": port,
                "DREMIO_USERNAME": username,
                "DREMIO_PASSWORD": password,
            }
            missing_params = [k for k, v in required_params.items() if not v]
            if missing_params:
                return cast(
                    str,
                    ValidationError(
                        "Missing required environment variables for Dremio: "
                        f"{', '.join(missing_params)}. "
                        "Please check your .env file. "
                        "For PAT-based auth, set DREMIO_URI and DREMIO_PAT instead."
                    ).to_response(),
                )

            success = db_manager.connect_dremio(
                host=str(host),
                port=int(str(port)),
                username=str(username),
                password=str(password),
            )
            db_name = "DREMIO"

    elif db_type == "clickhouse":
        host = os.getenv("CLICKHOUSE_HOST")
        port = os.getenv("CLICKHOUSE_PORT", "8123")
        database = os.getenv("CLICKHOUSE_DATABASE")
        username = os.getenv("CLICKHOUSE_USERNAME", "default")
        password = os.getenv("CLICKHOUSE_PASSWORD", "")
        protocol = os.getenv("CLICKHOUSE_PROTOCOL", "http")
        secure = os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true"

        required_params = {
            "CLICKHOUSE_HOST": host,
            "CLICKHOUSE_DATABASE": database,
        }
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            return cast(
                str,
                ValidationError(
                    "Missing required environment variables for ClickHouse: "
                    f"{', '.join(missing_params)}. Please check your .env file."
                ).to_response(),
            )

        success = db_manager.connect_clickhouse(
            host=str(host),
            port=int(port),
            database=str(database),
            username=str(username),
            password=str(password),
            protocol=protocol,
            secure=secure,
        )
        db_name = str(database)

    elif db_type == "bigquery":
        project_id = os.getenv("BIGQUERY_PROJECT_ID")
        dataset = os.getenv("BIGQUERY_DATASET", "")
        credentials_path = os.getenv("BIGQUERY_CREDENTIALS_PATH")
        credentials_json = os.getenv("BIGQUERY_CREDENTIALS_JSON")

        required_params = {"BIGQUERY_PROJECT_ID": project_id}
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            return cast(
                str,
                ValidationError(
                    "Missing required environment variables for BigQuery: "
                    f"{', '.join(missing_params)}. Please check your .env file."
                ).to_response(),
            )

        success = db_manager.connect_bigquery(
            project_id=str(project_id),
            dataset=dataset or "",
            credentials_path=credentials_path,
            credentials_json=credentials_json,
        )
        db_name = f"{project_id}/{dataset}" if dataset else str(project_id)

    elif db_type == "duckdb":
        database_path = os.getenv("DUCKDB_DATABASE_PATH", ":memory:")
        motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
        read_only = os.getenv("DUCKDB_READ_ONLY", "false").lower() == "true"

        success = db_manager.connect_duckdb(
            database_path=database_path,
            motherduck_token=motherduck_token,
            read_only=read_only,
        )
        db_name = database_path

    elif db_type == "databricks":
        server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
        http_path = os.getenv("DATABRICKS_HTTP_PATH")
        access_token = os.getenv("DATABRICKS_ACCESS_TOKEN")
        catalog = os.getenv("DATABRICKS_CATALOG", "hive_metastore")
        schema = os.getenv("DATABRICKS_SCHEMA", "default")

        required_params = {
            "DATABRICKS_SERVER_HOSTNAME": server_hostname,
            "DATABRICKS_HTTP_PATH": http_path,
            "DATABRICKS_ACCESS_TOKEN": access_token,
        }
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            return cast(
                str,
                ValidationError(
                    "Missing required environment variables for Databricks: "
                    f"{', '.join(missing_params)}. Please check your .env file."
                ).to_response(),
            )

        success = db_manager.connect_databricks(
            server_hostname=str(server_hostname),
            http_path=str(http_path),
            access_token=str(access_token),
            catalog=catalog,
            schema=schema,
        )
        db_name = f"{catalog}.{schema}"

    elif db_type == "mysql":
        host = os.getenv("MYSQL_HOST")
        port = os.getenv("MYSQL_PORT", "3306")
        database = os.getenv("MYSQL_DATABASE")
        username = os.getenv("MYSQL_USERNAME")
        password = os.getenv("MYSQL_PASSWORD")
        charset = os.getenv("MYSQL_CHARSET", "utf8mb4")

        required_params = {
            "MYSQL_HOST": host,
            "MYSQL_DATABASE": database,
            "MYSQL_USERNAME": username,
            "MYSQL_PASSWORD": password,
        }
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            return cast(
                str,
                ValidationError(
                    "Missing required environment variables for MySQL: "
                    f"{', '.join(missing_params)}. Please check your .env file."
                ).to_response(),
            )

        success = db_manager.connect_mysql(
            host=str(host),
            port=int(port),
            database=str(database),
            username=str(username),
            password=str(password),
            charset=charset,
        )
        db_name = str(database)

    if success:
        session = services.get_session_data(ctx)
        new_conn_id = services.get_connection_fingerprint(db_manager)

        if session.connection_id and session.connection_id != new_conn_id:
            logger.info(
                f"Connection changed (old: {session.connection_id[:8]}..., new: {new_conn_id[:8]}...)"
            )
            services.clear_session_state(session, reason="connection change")
        elif not session.connection_id:
            logger.info(f"Initial connection established: {new_conn_id[:8]}...")

        session.connection_id = new_conn_id
        session.connected_at = utc_now()
        session.clear_schema_cache()

        await ctx.info(f"Connected to {db_type}: {db_name}")

        # Write workspace connection info
        try:
            await mutate_workspace_metadata(
                new_conn_id,
                OUTPUT_DIR,
                lambda mgr: mgr.update_workspace_connection(
                    db_type=db_type, db_name=db_name
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to write workspace connection info: {e}")

        # Detect and auto-restore existing workspace
        response = f"Successfully connected to {db_type} database: {db_name}"
        workspace = detect_workspace(new_conn_id)
        if workspace and services.provides("get_oxigraph_store"):
            try:
                restore_result = await _restore_workspace_core(
                    ctx, session, new_conn_id, None, services
                )
                if restore_result:
                    response += "\n\n" + _format_restore_summary(restore_result)
                else:
                    # Workspace detected but restore returned nothing
                    response += "\n\n" + format_workspace_summary(workspace)
            except Exception as e:
                logger.warning(f"Auto-restore failed: {e}")
                response += "\n\n" + format_workspace_summary(workspace)
        elif workspace:
            # No get_oxigraph_store available (e.g., tests) — show summary only
            response += "\n\n" + format_workspace_summary(workspace)

        return response
    else:
        await ctx.info("Database connection failed; check credentials and try again")
        return cast(
            str,
            ConnectionError(
                f"Failed to connect to {db_type} database: {db_name}"
            ).to_response(),
        )


async def list_schemas(ctx: Context, services: "HandlerContext") -> list[str]:
    """Get a list of available schemas from the connected database.

    Args:
        ctx: FastMCP context
        get_session_db_manager: Function to get session db manager

    Returns:
        List of schema names
    """
    db_manager = services.get_session_db_manager(ctx)
    schemas = db_manager.get_schemas()
    if schemas:
        await ctx.info(
            f"Found {len(schemas)} schemas; next call should be discover_schema"
        )
    else:
        await ctx.info("No schemas found")
    return schemas if schemas else []

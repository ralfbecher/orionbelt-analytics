"""Configuration management for OrionBelt Analytics."""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from .constants import DEFAULT_BASE_URI, DEFAULT_POSTGRES_PORT, DEFAULT_SNOWFLAKE_SCHEMA, DEFAULT_DREMIO_PORT, DEFAULT_CLICKHOUSE_PORT

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration settings."""
    log_level: str
    ontology_base_uri: str
    mcp_transport: str = "http"  # Default to http transport
    mcp_server_host: str = "localhost"
    mcp_server_port: int = 9000

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.ontology_base_uri.endswith('/'):
            self.ontology_base_uri += '/'

        # Validate transport type
        if self.mcp_transport not in ["http", "sse"]:
            logger.warning(f"Invalid MCP_TRANSPORT value '{self.mcp_transport}'. Defaulting to 'http'.")
            self.mcp_transport = "http"


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    # PostgreSQL settings
    postgres_host: Optional[str] = None
    postgres_port: int = DEFAULT_POSTGRES_PORT
    postgres_database: Optional[str] = None
    postgres_username: Optional[str] = None
    postgres_password: Optional[str] = None
    
    # Snowflake settings
    snowflake_account: Optional[str] = None
    snowflake_username: Optional[str] = None
    snowflake_password: Optional[str] = None
    snowflake_warehouse: Optional[str] = None
    snowflake_database: Optional[str] = None
    snowflake_schema: str = DEFAULT_SNOWFLAKE_SCHEMA
    snowflake_role: str = "PUBLIC"
    
    # Dremio settings (following official dremio-mcp approach)
    dremio_uri: Optional[str] = None  # Full API endpoint like https://api.dremio.cloud or https://host:port
    dremio_pat: Optional[str] = None  # Personal Access Token
    dremio_project_id: Optional[str] = None  # Optional project ID
    
    # Legacy settings for backward compatibility
    dremio_host: Optional[str] = None
    dremio_port: int = DEFAULT_DREMIO_PORT
    dremio_username: Optional[str] = None
    dremio_password: Optional[str] = None
    dremio_ssl: bool = False

    # ClickHouse settings
    clickhouse_host: Optional[str] = None
    clickhouse_port: int = DEFAULT_CLICKHOUSE_PORT
    clickhouse_database: Optional[str] = None
    clickhouse_username: str = "default"
    clickhouse_password: str = ""
    clickhouse_protocol: str = "http"
    clickhouse_secure: bool = False


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self):
        """Initialize configuration manager."""
        # Load .env from project root (one level up from src)
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(env_path)
        self._server_config: Optional[ServerConfig] = None
        self._db_config: Optional[DatabaseConfig] = None
    
    def get_server_config(self) -> ServerConfig:
        """Get server configuration."""
        if self._server_config is None:
            self._server_config = ServerConfig(
                log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
                ontology_base_uri=os.getenv("ONTOLOGY_BASE_URI", DEFAULT_BASE_URI),
                mcp_transport=os.getenv("MCP_TRANSPORT", "http").lower(),
                mcp_server_host=os.getenv("MCP_SERVER_HOST", "localhost"),
                mcp_server_port=int(os.getenv("MCP_SERVER_PORT", "9000"))
            )
            logger.info("Server configuration loaded")
        return self._server_config
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        if self._db_config is None:
            self._db_config = DatabaseConfig(
                postgres_host=os.getenv("POSTGRES_HOST"),
                postgres_port=int(os.getenv("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)),
                postgres_database=os.getenv("POSTGRES_DATABASE"),
                postgres_username=os.getenv("POSTGRES_USERNAME"),
                postgres_password=os.getenv("POSTGRES_PASSWORD"),
                snowflake_account=os.getenv("SNOWFLAKE_ACCOUNT"),
                snowflake_username=os.getenv("SNOWFLAKE_USERNAME"),
                snowflake_password=os.getenv("SNOWFLAKE_PASSWORD"),
                snowflake_warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
                snowflake_database=os.getenv("SNOWFLAKE_DATABASE"),
                snowflake_schema=os.getenv("SNOWFLAKE_SCHEMA", DEFAULT_SNOWFLAKE_SCHEMA),
                snowflake_role=os.getenv("SNOWFLAKE_ROLE", "PUBLIC"),
                # New PAT-based settings (preferred)
                dremio_uri=os.getenv("DREMIO_URI"),
                dremio_pat=os.getenv("DREMIO_PAT"),
                dremio_project_id=os.getenv("DREMIO_PROJECT_ID"),
                
                # Legacy settings for backward compatibility
                dremio_host=os.getenv("DREMIO_HOST"),
                dremio_port=int(os.getenv("DREMIO_PORT", DEFAULT_DREMIO_PORT)),
                dremio_username=os.getenv("DREMIO_USERNAME"),
                dremio_password=os.getenv("DREMIO_PASSWORD"),
                dremio_ssl=os.getenv("DREMIO_SSL", "false").lower() == "true",
                # ClickHouse settings
                clickhouse_host=os.getenv("CLICKHOUSE_HOST"),
                clickhouse_port=int(os.getenv("CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_PORT)),
                clickhouse_database=os.getenv("CLICKHOUSE_DATABASE"),
                clickhouse_username=os.getenv("CLICKHOUSE_USERNAME", "default"),
                clickhouse_password=os.getenv("CLICKHOUSE_PASSWORD", ""),
                clickhouse_protocol=os.getenv("CLICKHOUSE_PROTOCOL", "http"),
                clickhouse_secure=os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true"
            )
            logger.info("Database configuration loaded")
        return self._db_config
    
    def validate_db_config(self, db_type: str) -> Dict[str, Any]:
        """Validate database configuration for a specific type."""
        config = self.get_database_config()
        missing_params = []
        
        if db_type == "postgresql":
            required_fields = {
                "host": config.postgres_host,
                "port": config.postgres_port,
                "database": config.postgres_database,
                "username": config.postgres_username,
                "password": config.postgres_password
            }
        elif db_type == "snowflake":
            required_fields = {
                "account": config.snowflake_account,
                "username": config.snowflake_username,
                "password": config.snowflake_password,
                "warehouse": config.snowflake_warehouse,
                "database": config.snowflake_database
            }
        elif db_type == "dremio":
            # Prefer new PAT-based authentication
            if config.dremio_uri and config.dremio_pat:
                required_fields = {
                    "uri": config.dremio_uri,
                    "pat": config.dremio_pat
                }
            else:
                # Fall back to legacy username/password authentication
                required_fields = {
                    "host": config.dremio_host,
                    "port": config.dremio_port,
                    "username": config.dremio_username,
                    "password": config.dremio_password,
                    "ssl": config.dremio_ssl
                }
        elif db_type == "clickhouse":
            required_fields = {
                "host": config.clickhouse_host,
                "database": config.clickhouse_database
            }
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        missing_params = [k for k, v in required_fields.items() if not v]
        
        return {
            "valid": len(missing_params) == 0,
            "missing_params": missing_params,
            "config": required_fields
        }


# Global configuration manager instance
config_manager = ConfigManager()
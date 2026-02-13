"""Database connection and diagnostic tools."""

import logging
from typing import Dict, Any, Optional

from ..config import config_manager
from ..shared import get_db_manager, create_error_response

logger = logging.getLogger(__name__)


def connect_database(
    db_type: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    account: Optional[str] = None,
    warehouse: Optional[str] = None,
    schema: Optional[str] = "PUBLIC",
    role: Optional[str] = None,
    ssl: Optional[bool] = False,
    uri: Optional[str] = None,
    pat: Optional[str] = None
) -> Dict[str, Any]:
    """Connect to database implementation. Full documentation in main.py."""
    import os
    from dotenv import load_dotenv
    
    # Ensure .env is loaded - fallback mechanism with multiple search paths
    possible_env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'),  # From src/tools/connection.py
        os.path.join(os.getcwd(), '.env'),  # Current working directory
        '/Users/ralfbecher/Documents/GitHub/mcp-servers/database-ontology-mcp/.env'  # Absolute path as fallback
    ]
    
    for env_path in possible_env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
            break
    try:
        db_manager = get_db_manager()
        db_config = config_manager.get_database_config()
        
        if db_type.lower() == "postgresql":
            # Use provided parameters or fall back to config
            final_host = host or db_config.postgres_host
            final_port = port or db_config.postgres_port
            final_database = database or db_config.postgres_database
            final_username = username or db_config.postgres_username
            final_password = password or db_config.postgres_password
            
            
            if not all([final_host, final_port, final_database, final_username]):
                return create_error_response(
                    "Missing required PostgreSQL parameters: host, port, database, username (provide via parameters or .env file)",
                    "parameter_error"
                )
            success = db_manager.connect_postgresql(final_host, final_port, final_database, final_username, final_password or "")
            
        elif db_type.lower() == "snowflake":
            # Use provided parameters or fall back to config
            final_account = account or db_config.snowflake_account
            final_username = username or db_config.snowflake_username
            final_password = password or db_config.snowflake_password
            final_warehouse = warehouse or db_config.snowflake_warehouse
            final_database = database or db_config.snowflake_database
            final_schema = schema or db_config.snowflake_schema
            final_role = role or db_config.snowflake_role
            
            if not all([final_account, final_username, final_database, final_warehouse]):
                return create_error_response(
                    "Missing required Snowflake parameters: account, username, database, warehouse (provide via parameters or .env file)", 
                    "parameter_error"
                )
            success = db_manager.connect_snowflake(final_account, final_username, final_password or "", final_warehouse, final_database, final_schema, final_role)
            
        elif db_type.lower() == "dremio":
            # Check for new PAT-based configuration first (parameters override config)
            final_uri = uri or getattr(db_config, 'dremio_uri', None)
            final_pat = pat or getattr(db_config, 'dremio_pat', None)
            
            if final_uri and final_pat:
                # Use PAT-based authentication (preferred)
                logger.info(f"🔍 MCP DEBUG - Dremio PAT config: uri={final_uri}, pat={'***SET***'}")
                success = db_manager.connect_dremio(uri=final_uri, pat=final_pat)
            else:
                # Fall back to legacy username/password authentication
                final_host = host or getattr(db_config, 'dremio_host', None)
                final_port = port or getattr(db_config, 'dremio_port', 9047)
                final_username = username or getattr(db_config, 'dremio_username', None)
                final_password = password or getattr(db_config, 'dremio_password', None)
                final_ssl = ssl if ssl is not None else getattr(db_config, 'dremio_ssl', False)
                
                logger.info(f"🔍 MCP DEBUG - Dremio legacy config: host={final_host}, port={final_port}, username={final_username}, password={'***SET***' if final_password else 'NOT SET'}, ssl={final_ssl}")
                
                if not all([final_host, final_port, final_username]):
                    missing = []
                    if not final_host: missing.append("host")
                    if not final_port: missing.append("port") 
                    if not final_username: missing.append("username")
                    logger.error(f"🔍 MCP DEBUG - Missing Dremio parameters: {missing}")
                    return create_error_response(
                        "Missing required Dremio parameters: host, port, username (provide via parameters or .env file)\nFor PAT-based auth, set DREMIO_URI and DREMIO_PAT in .env file",
                        "parameter_error"
                    )
                success = db_manager.connect_dremio(final_host, final_port, final_username, final_password or "", final_ssl)
            
        elif db_type.lower() == "clickhouse":
            # Use provided parameters or fall back to config
            final_host = host or getattr(db_config, 'clickhouse_host', None)
            final_port = port or getattr(db_config, 'clickhouse_port', 8123)
            final_database = database or getattr(db_config, 'clickhouse_database', None)
            final_username = username or getattr(db_config, 'clickhouse_username', 'default')
            final_password = password or getattr(db_config, 'clickhouse_password', '')
            final_protocol = getattr(db_config, 'clickhouse_protocol', 'http')
            final_secure = getattr(db_config, 'clickhouse_secure', False)

            if not all([final_host, final_database]):
                return create_error_response(
                    "Missing required ClickHouse parameters: host, database (provide via parameters or .env file)",
                    "parameter_error"
                )
            success = db_manager.connect_clickhouse(
                final_host, final_port, final_database,
                final_username, final_password,
                final_protocol, final_secure
            )

        else:
            return create_error_response(
                f"Unsupported database type: {db_type}. Use 'postgresql', 'snowflake', 'dremio', or 'clickhouse'",
                "parameter_error"
            )
        
        if success:
            logger.info(f"Successfully connected to {db_type} database")
            return {
                "success": True,
                "message": f"Successfully connected to {db_type} database",
                "connection_info": db_manager.connection_info
            }
        else:
            logger.error(f"Connection to {db_type} database returned False - no specific error")
            return create_error_response(
                f"Failed to connect to {db_type} database",
                "connection_error",
                "Connection attempt returned False without specific error"
            )
            
    except Exception as e:
        logger.error(f"Connection attempt failed: {type(e).__name__}: {e}")
        return create_error_response(
            f"Connection failed: {type(e).__name__}",
            "connection_error",
            str(e)
        )


def diagnose_connection_issue(
    db_type: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    ssl: Optional[bool] = None
) -> Dict[str, Any]:
    """Diagnose connection issues. Full documentation in main.py."""
    try:
        db_config = config_manager.get_database_config()
        
        # Get actual connection parameters based on database type
        if db_type.lower() == "dremio":
            actual_host = host or db_config.dremio_host
            actual_port = port or db_config.dremio_port
            actual_username = username or db_config.dremio_username
            actual_password = db_config.dremio_password
            actual_ssl = ssl if ssl is not None else False
            
            result = {
                "db_type": "dremio",
                "success": True,
                "connection_parameters": {
                    "host": actual_host,
                    "port": actual_port,
                    "username": actual_username,
                    "password_provided": bool(actual_password),
                    "ssl": actual_ssl
                },
                "issues_found": [],
                "recommendations": [],
                "connection_string_preview": None
            }
            
            # Check for missing parameters
            if not actual_host:
                result["issues_found"].append("Missing DREMIO_HOST")
                result["recommendations"].append("Set DREMIO_HOST in .env file or pass host parameter")
            
            if not actual_port:
                result["issues_found"].append("Missing DREMIO_PORT")
                result["recommendations"].append("Set DREMIO_PORT in .env file (default: 9047) or pass port parameter")
            
            if not actual_username:
                result["issues_found"].append("Missing DREMIO_USERNAME")
                result["recommendations"].append("Set DREMIO_USERNAME in .env file or pass username parameter")
            
            if not actual_password:
                result["issues_found"].append("Missing DREMIO_PASSWORD")
                result["recommendations"].append("Set DREMIO_PASSWORD in .env file")
            
            # Show connection string format (without password)
            if actual_host and actual_port and actual_username:
                sslmode = 'require' if actual_ssl else 'disable'
                result["connection_string_preview"] = (
                    f"postgresql://{actual_username}:****@{actual_host}:{actual_port}/dremio"
                    f"?sslmode={sslmode}&application_name=database-ontology-mcp"
                )
            
            # Add Dremio-specific recommendations
            result["recommendations"].extend([
                "Ensure Dremio coordinator is running and accessible",
                "Verify port 9047 (REST API) is not blocked by firewall",
                "For production Dremio, check if SSL/TLS is required (use ssl=True)",
                "Confirm username/password are correct for Dremio",
                "Try connecting with Dremio's web UI first to verify credentials",
                "For cloud Dremio, check if IP whitelisting is required"
            ])
            
            # Connection test recommendations
            result["connection_test_steps"] = [
                f"1. Test basic connectivity: telnet {actual_host or 'HOST'} {actual_port or 9047}",
                "2. Check Dremio web UI is accessible (usually same host, port 9047)",
                "3. Verify credentials work in Dremio web interface",
                "4. Check Dremio coordinator logs for connection attempts",
                "5. For production environments, test with SSL enabled: ssl=True"
            ]
            
            # Common Dremio issues
            result["common_issues"] = {
                "SSL_HANDSHAKE_FAILURE": {
                    "description": "SSL/TLS connection failed",
                    "solutions": ["Try ssl=False", "Check SSL certificate", "Verify Dremio SSL configuration"]
                },
                "CONNECTION_REFUSED": {
                    "description": "Cannot connect to Dremio coordinator",
                    "solutions": ["Check if Dremio is running", "Verify host and port", "Check firewall rules"]
                },
                "AUTHENTICATION_FAILED": {
                    "description": "Username/password incorrect",
                    "solutions": ["Verify credentials in Dremio UI", "Check for special characters in password", "Confirm user has CONNECT privileges"]
                },
                "TIMEOUT": {
                    "description": "Connection attempt timed out",
                    "solutions": ["Check network connectivity", "Verify Dremio coordinator is responsive", "Try increasing connection timeout"]
                }
            }
            
        elif db_type.lower() == "postgresql":
            actual_host = host or db_config.postgres_host
            actual_port = port or db_config.postgres_port
            actual_username = username or db_config.postgres_username
            actual_password = db_config.postgres_password
            
            result = {
                "db_type": "postgresql",
                "success": True,
                "connection_parameters": {
                    "host": actual_host,
                    "port": actual_port,
                    "username": actual_username,
                    "password_provided": bool(actual_password),
                    "database": db_config.postgres_database
                },
                "issues_found": [],
                "recommendations": []
            }
            
            # Add PostgreSQL-specific checks
            if not actual_host:
                result["issues_found"].append("Missing POSTGRES_HOST")
                result["recommendations"].append("Set POSTGRES_HOST in .env file or pass host parameter")
            if not actual_port:
                result["issues_found"].append("Missing POSTGRES_PORT")
                result["recommendations"].append("Set POSTGRES_PORT in .env file (default: 5432) or pass port parameter")
            if not db_config.postgres_database:
                result["issues_found"].append("Missing POSTGRES_DATABASE")
                result["recommendations"].append("Set POSTGRES_DATABASE in .env file")
            if not actual_username:
                result["issues_found"].append("Missing POSTGRES_USERNAME")
                result["recommendations"].append("Set POSTGRES_USERNAME in .env file or pass username parameter")
            if not actual_password:
                result["issues_found"].append("Missing POSTGRES_PASSWORD")
                result["recommendations"].append("Set POSTGRES_PASSWORD in .env file")
            
            # Add PostgreSQL-specific recommendations
            if actual_host and actual_port and actual_username:
                result["connection_string_preview"] = (
                    f"postgresql://{actual_username}:****@{actual_host}:{actual_port}/{db_config.postgres_database or 'DATABASE'}"
                    f"?sslmode=prefer&application_name=database-ontology-mcp"
                )
            
            result["recommendations"].extend([
                "Ensure PostgreSQL server is running and accessible",
                "Verify pg_hba.conf allows connections from your IP",
                "Check if database exists and user has access",
                "Confirm username/password are correct",
                "Test connection with psql command line tool first"
            ])
            
            result["connection_test_steps"] = [
                f"1. Test basic connectivity: telnet {actual_host or 'HOST'} {actual_port or 5432}",
                f"2. Try psql connection: psql -h {actual_host or 'HOST'} -p {actual_port or 5432} -U {actual_username or 'USERNAME'} -d {db_config.postgres_database or 'DATABASE'}",
                "3. Check PostgreSQL server logs for connection attempts",
                "4. Verify pg_hba.conf configuration allows your connection",
                "5. Check if SSL is required or forbidden by server"
            ]
            
            result["common_issues"] = {
                "CONNECTION_REFUSED": {
                    "description": "Cannot connect to PostgreSQL server",
                    "solutions": ["Check if PostgreSQL is running", "Verify host and port", "Check firewall rules", "Ensure postgresql.conf allows connections"]
                },
                "AUTHENTICATION_FAILED": {
                    "description": "Username/password authentication failed",
                    "solutions": ["Verify credentials", "Check pg_hba.conf authentication method", "Ensure user exists in database"]
                },
                "DATABASE_NOT_FOUND": {
                    "description": "Specified database does not exist",
                    "solutions": ["Create database", "Check database name spelling", "Verify user has access to database"]
                },
                "SSL_ERROR": {
                    "description": "SSL connection issues",
                    "solutions": ["Check SSL configuration", "Try with sslmode=disable for testing", "Verify SSL certificates"]
                }
            }
                
        elif db_type.lower() == "snowflake":
            result = {
                "db_type": "snowflake",
                "success": True,
                "connection_parameters": {
                    "account": db_config.snowflake_account,
                    "username": db_config.snowflake_username,
                    "password_provided": bool(db_config.snowflake_password),
                    "warehouse": db_config.snowflake_warehouse,
                    "database": db_config.snowflake_database,
                    "schema": db_config.snowflake_schema,
                    "role": db_config.snowflake_role
                },
                "issues_found": [],
                "recommendations": []
            }
            
            # Add Snowflake-specific checks
            if not db_config.snowflake_account:
                result["issues_found"].append("Missing SNOWFLAKE_ACCOUNT")
                result["recommendations"].append("Set SNOWFLAKE_ACCOUNT in .env file")
            if not db_config.snowflake_username:
                result["issues_found"].append("Missing SNOWFLAKE_USERNAME") 
                result["recommendations"].append("Set SNOWFLAKE_USERNAME in .env file")
            if not db_config.snowflake_password:
                result["issues_found"].append("Missing SNOWFLAKE_PASSWORD")
                result["recommendations"].append("Set SNOWFLAKE_PASSWORD in .env file")
            if not db_config.snowflake_warehouse:
                result["issues_found"].append("Missing SNOWFLAKE_WAREHOUSE")
                result["recommendations"].append("Set SNOWFLAKE_WAREHOUSE in .env file")
            if not db_config.snowflake_database:
                result["issues_found"].append("Missing SNOWFLAKE_DATABASE")
                result["recommendations"].append("Set SNOWFLAKE_DATABASE in .env file")
            
            # Add Snowflake-specific recommendations
            result["recommendations"].extend([
                "Ensure Snowflake account is active and accessible",
                "Verify account identifier format (orgname-account_name)",
                "Check warehouse is running and user has access",
                "Confirm username/password are correct",
                "Test connection with Snowflake web UI first"
            ])
            
            result["connection_test_steps"] = [
                "1. Test Snowflake web UI login with same credentials",
                "2. Verify warehouse is running and accessible",
                "3. Check database and schema permissions", 
                "4. Test with SnowSQL command line tool",
                "5. Verify account URL format and region"
            ]
            
            result["common_issues"] = {
                "ACCOUNT_NOT_FOUND": {
                    "description": "Snowflake account identifier incorrect",
                    "solutions": ["Check account identifier format", "Verify region and organization name", "Use full account URL format"]
                },
                "AUTHENTICATION_FAILED": {
                    "description": "Username/password incorrect",
                    "solutions": ["Verify credentials in Snowflake UI", "Check for MFA requirements", "Ensure account is not locked"]
                },
                "WAREHOUSE_NOT_FOUND": {
                    "description": "Specified warehouse does not exist or not accessible",
                    "solutions": ["Check warehouse name", "Verify user has USAGE privilege", "Ensure warehouse is not suspended"]
                },
                "DATABASE_SCHEMA_ERROR": {
                    "description": "Database or schema access issues",
                    "solutions": ["Verify database/schema names", "Check user privileges", "Ensure objects exist"]
                }
            }
        elif db_type.lower() == "clickhouse":
            actual_host = host or getattr(db_config, 'clickhouse_host', None)
            actual_port = port or getattr(db_config, 'clickhouse_port', 8123)
            actual_username = username or getattr(db_config, 'clickhouse_username', 'default')
            actual_password = getattr(db_config, 'clickhouse_password', '')
            actual_database = getattr(db_config, 'clickhouse_database', None)

            result = {
                "db_type": "clickhouse",
                "success": True,
                "connection_parameters": {
                    "host": actual_host,
                    "port": actual_port,
                    "database": actual_database,
                    "username": actual_username,
                    "password_provided": bool(actual_password),
                },
                "issues_found": [],
                "recommendations": [],
            }

            if not actual_host:
                result["issues_found"].append("Missing CLICKHOUSE_HOST")
                result["recommendations"].append("Set CLICKHOUSE_HOST in .env file or pass host parameter")
            if not actual_database:
                result["issues_found"].append("Missing CLICKHOUSE_DATABASE")
                result["recommendations"].append("Set CLICKHOUSE_DATABASE in .env file (default: 'default')")

            if actual_host and actual_port:
                result["connection_string_preview"] = (
                    f"clickhouse+http://{actual_username}:****@{actual_host}:{actual_port}/{actual_database or 'default'}"
                )

            result["recommendations"].extend([
                "Ensure ClickHouse server is running and HTTP interface is enabled",
                "Verify the HTTP port (default 8123) is accessible",
                "Test with: curl http://<host>:8123/ping",
                "Try clickhouse-client CLI to verify credentials",
            ])

            result["connection_test_steps"] = [
                f"1. Test HTTP ping: curl http://{actual_host or 'HOST'}:{actual_port or 8123}/ping",
                f"2. Test query: echo 'SELECT 1' | curl http://{actual_host or 'HOST'}:{actual_port or 8123}/ --data-binary @-",
                "3. Check ClickHouse server logs for connection attempts",
                "4. Verify users.xml or config.xml allows remote connections",
            ]

            result["common_issues"] = {
                "CONNECTION_REFUSED": {
                    "description": "Cannot connect to ClickHouse HTTP interface",
                    "solutions": ["Check if ClickHouse is running", "Verify host and HTTP port (8123)", "Check firewall rules"],
                },
                "AUTHENTICATION_FAILED": {
                    "description": "Username/password incorrect",
                    "solutions": ["Verify credentials in users.xml", "Default user is 'default' with empty password", "Check access control settings"],
                },
                "DATABASE_NOT_FOUND": {
                    "description": "Specified database does not exist",
                    "solutions": ["Create database with: CREATE DATABASE <name>", "Check database name spelling", "Use 'default' database"],
                },
            }

        else:
            return create_error_response(
                f"Unsupported database type: {db_type}",
                "parameter_error",
                "Use 'postgresql', 'snowflake', 'dremio', or 'clickhouse'"
            )
        
        # Add general recommendations if issues found
        if result["issues_found"]:
            result["recommendations"].insert(0, "Create or update .env file in project root with missing parameters")
        
        return result
        
    except Exception as e:
        logger.error(f"Connection diagnosis failed: {e}")
        return create_error_response(
            f"Diagnosis failed: {str(e)}",
            "internal_error"
        )
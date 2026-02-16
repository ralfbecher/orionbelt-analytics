"""Database connection and schema analysis manager."""

import asyncio
import decimal
import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DatabaseError, ProgrammingError
from sqlalchemy.pool import QueuePool

from .constants import (
    CONNECTION_TIMEOUT,
    QUERY_TIMEOUT,
    IDENTIFIER_PATTERN,
    POSTGRES_SYSTEM_SCHEMAS,
    SNOWFLAKE_SYSTEM_SCHEMAS,
    DREMIO_SYSTEM_SCHEMAS,
    CLICKHOUSE_SYSTEM_SCHEMAS,
    MIN_SAMPLE_LIMIT,
    MAX_SAMPLE_LIMIT,
    DEFAULT_SAMPLE_LIMIT
)
from .security import (
    SecureCredentialManager,
    sql_validator,
    identifier_validator,
    audit_log_security_event,
    SecurityLevel
)

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    foreign_key_table: Optional[str] = None
    foreign_key_column: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    schema: str
    columns: List[ColumnInfo]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    comment: Optional[str] = None
    row_count: Optional[int] = None
    sample_data: Optional[List[Dict[str, Any]]] = None


class DatabaseManager:
    """Manages database connections and schema analysis with enhanced reliability and security."""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.metadata: Optional[MetaData] = None
        self.connection_info: Dict[str, Any] = {}
        self._connection_pool_size = 5
        self._max_overflow = 10
        self._dremio_rest_connection: Optional[Dict[str, Any]] = None
        self._last_connection_params: Optional[Dict[str, Any]] = None  # Initialize this attribute
        
        # Security and performance improvements
        # SecureCredentialManager will automatically get MCP_MASTER_PASSWORD from .env
        self._credential_manager = SecureCredentialManager()
        self._metadata_cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._connection_id: Optional[str] = None
        
        # Thread pool for concurrent operations
        self._thread_pool = ThreadPoolExecutor(max_workers=5)
    
    def _log_sql_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Log SQL query with parameters for debugging."""
        db_type = self.connection_info.get("type", "unknown")

        if params:
            # Only sanitize sensitive parameters (passwords, secrets)
            safe_params = {k: '***' if 'password' in k.lower() or 'secret' in k.lower() else v
                          for k, v in params.items()}
            logger.info(f"🔍 {db_type.upper()} SQL QUERY: {query} | PARAMS: {safe_params}")
        else:
            logger.info(f"🔍 {db_type.upper()} SQL QUERY: {query}")
    
    def _get_cache_key(self, operation: str, *args) -> str:
        """Generate cache key for metadata operations."""
        return f"{operation}:{':'.join(str(arg) for arg in args)}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - cache_entry.get('timestamp', 0) < self._cache_ttl
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get value from cache if valid."""
        if cache_key in self._metadata_cache:
            entry = self._metadata_cache[cache_key]
            if self._is_cache_valid(entry):
                logger.debug(f"Cache hit for {cache_key}")
                return entry['data']
            else:
                # Remove expired entry
                del self._metadata_cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        self._metadata_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        logger.debug(f"Cached data for {cache_key}")
    
    def _validate_identifier_secure(self, identifier: str) -> bool:
        """Securely validate database identifier to prevent injection."""
        if not identifier_validator.validate_identifier(identifier):
            audit_log_security_event(
                "invalid_identifier_attempt",
                {"identifier": identifier[:50]},  # Limit logged data
                SecurityLevel.MEDIUM
            )
            return False
        return True
        
    def _test_connection(self) -> bool:
        """Test if the current connection is healthy."""
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
        if not self._dremio_rest_connection:
            return False

        from .dremio_client import create_dremio_client

        async def test_dremio():
            conn = self._dremio_rest_connection
            if conn.get("uri") and conn.get("pat"):
                async with await create_dremio_client(uri=conn["uri"], pat=conn["pat"]) as client:
                    return await client.test_connection()
            async with await create_dremio_client(
                host=conn.get("host"),
                port=conn.get("port"),
                username=conn.get("username"),
                password=conn.get("password"),
                ssl=conn.get("ssl", False)
            ) as client:
                return await client.test_connection()

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, test_dremio())
                result = future.result(timeout=CONNECTION_TIMEOUT)
        except RuntimeError:
            result = asyncio.run(test_dremio())
        except Exception as e:
            logger.warning(f"Dremio health check failed: {e}")
            return False

        return bool(result.get("success"))
    
    def _ensure_connection(self):
        """Ensure we have a healthy database connection, reconnecting if necessary."""
        # For Dremio REST connections, just check if connection info exists
        if self._dremio_rest_connection:
            logger.debug("_ensure_connection: Dremio REST connection info available")
            return
        
        logger.debug(f"_ensure_connection: engine exists: {self.engine is not None}")
        logger.debug(f"_ensure_connection: last_params available: {self._last_connection_params is not None}")
        
        if not self.engine:
            if self._last_connection_params:
                logger.info("No engine found, reconnecting to database using stored parameters")
                self._reconnect()
            else:
                raise RuntimeError("No database connection established and no connection parameters available")
        elif not self._test_connection():
            if self._last_connection_params:
                logger.info("Connection health check failed, reconnecting to database")
                self._reconnect()
            else:
                logger.error("Connection unhealthy but no reconnection parameters available")
                raise RuntimeError("Database connection is unhealthy and cannot be restored")
                
        logger.debug(f"_ensure_connection: final engine state: {self.engine is not None}")
    
    def _reconnect(self):
        """Reconnect to the database using stored parameters."""
        if not self._last_connection_params:
            raise RuntimeError("No connection parameters stored for reconnection")
        
        params = self._last_connection_params
        if params["type"] == "postgresql":
            success = self.connect_postgresql(
                params["host"], params["port"], params["database"],
                params["username"], params["password"]
            )
        elif params["type"] == "snowflake":
            success = self.connect_snowflake(
                params["account"], params["username"], params["password"],
                params["warehouse"], params["database"], params.get("schema", "PUBLIC")
            )
        elif params["type"] == "clickhouse":
            success = self.connect_clickhouse(
                params["host"], params["port"], params["database"],
                params.get("username", "default"), params.get("password", ""),
                params.get("protocol", "http"), params.get("secure", False)
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
                    params.get("ssl", True)
                )
        else:
            raise RuntimeError(f"Unsupported database type for reconnection: {params['type']}")
        
        if not success:
            raise RuntimeError(f"Failed to reconnect to {params['type']} database")
        
        logger.info(f"Successfully reconnected to {params['type']} database")
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with auto-reconnection."""
        # Basic connection check first
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        # Try connection with retry logic
        max_retries = 2
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                conn = self.engine.connect()
                try:
                    yield conn
                finally:
                    conn.close()
                return  # Success, exit method
            except (OperationalError, DatabaseError) as e:
                last_exception = e
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:  # Not the last attempt
                    if self._last_connection_params:
                        logger.info("Attempting reconnection...")
                        try:
                            self._reconnect()
                        except Exception as reconnect_error:
                            logger.error(f"Reconnection failed: {reconnect_error}")
                            # Continue to next attempt or fail
                    else:
                        logger.error("No connection parameters available for reconnection")
                        break
                        
        # If we get here, all attempts failed
        logger.error(f"All connection attempts failed. Last error: {last_exception}")
        raise RuntimeError(f"Database connection failed after {max_retries} attempts: {last_exception}")
        
    def connect_postgresql(self, host: str, port: int, database: str, 
                          username: str, password: str) -> bool:
        """Connect to PostgreSQL database with enhanced security and reliability."""
        try:
            # Validate inputs
            if not all([host, port, database, username]):
                logger.error("Missing required PostgreSQL connection parameters")
                return False
            
            # Validate identifiers for security
            if not self._validate_identifier_secure(database):
                logger.error(f"Invalid database name: {database}")
                return False
            
            # Create secure connection string using URL encoding
            safe_username = quote_plus(username)
            safe_password = quote_plus(password)
            connection_string = f"postgresql://{safe_username}:{safe_password}@{host}:{port}/{database}"
            
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self._connection_pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=CONNECTION_TIMEOUT,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                echo=False,
                connect_args={
                    "connect_timeout": CONNECTION_TIMEOUT,
                    "application_name": "database-ontology-mcp"
                }
            )
            self.metadata = MetaData()
            
            # Clear any previous Dremio REST connection state
            self._dremio_rest_connection = None
            
            # Store non-sensitive connection info
            self.connection_info = {
                "type": "postgresql",
                "host": host,
                "port": port,
                "database": database,
                "username": username
            }

            # Generate secure connection ID and store encrypted credentials
            self._connection_id = hashlib.sha256(f"{host}:{port}:{database}:{username}".encode()).hexdigest()[:16]
            
            # Store credentials securely (encrypted in memory)
            credentials = {
                "type": "postgresql",
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "password": password
            }
            
            # Only store credentials if we have encryption capability
            try:
                # Initialize encryption with a derived key (not ideal, but better than plaintext)
                if not self._credential_manager._cipher:
                    # Use connection details to derive a key (better than nothing)
                    key_material = f"{host}:{database}:{username}"
                    self._credential_manager._initialize_encryption(key_material)
                
                # We don't store credentials at all - they're only used for the connection
                logger.info(f"PostgreSQL connection established successfully to {host}:{port}/{database}")
                
            except Exception as e:
                logger.warning(f"Could not initialize credential encryption: {e}")
                # Continue without storing credentials
                pass
            
            # Test connection with timeout
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            # Store connection parameters for reconnection
            self._last_connection_params = {
                "type": "postgresql",
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "password": password
            }
            
            logger.info(f"Connected to PostgreSQL database: {database} at {host}:{port}")
            return True
            
        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(f"Failed to connect to PostgreSQL {host}:{port}/{database}: {type(e).__name__}: {e}")
            self.engine = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to PostgreSQL: {type(e).__name__}: {e}")
            self.engine = None
            return False
    
    def connect_snowflake(self, account: str, username: str, password: str,
                         warehouse: str, database: str, schema: str = "PUBLIC", role: str = "PUBLIC") -> bool:
        """Connect to Snowflake database with enhanced security and reliability."""
        try:
            # Validate inputs
            if not all([account, username, warehouse, database]):
                logger.error("Missing required Snowflake connection parameters")
                return False

            # URL-encode password to handle special characters
            encoded_password = quote_plus(password) if password else ""
            
            connection_string = (
                f"snowflake://{username}:{encoded_password}@{account}/"
                f"{database}/{schema}?warehouse={warehouse}&role={role}"
            )
            
            logger.info(f"Connecting to Snowflake with account: {account}, database: {database}, warehouse: {warehouse}, schema: {schema}")
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self._connection_pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=CONNECTION_TIMEOUT,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
                connect_args={
                    "application": "database-ontology-mcp",
                    "network_timeout": CONNECTION_TIMEOUT
                }
            )
            self.metadata = MetaData()
            
            # Clear any previous Dremio REST connection state
            self._dremio_rest_connection = None
            
            self.connection_info = {
                "type": "snowflake",
                "account": account,
                "username": username,
                "warehouse": warehouse,
                "database": database,
                "schema": schema,
                "role": role
            }
            
            # Store connection parameters for reconnection
            self._last_connection_params = {
                "type": "snowflake",
                "account": account,
                "username": username,
                "password": password,
                "warehouse": warehouse,
                "database": database,
                "schema": schema,
                "role": role
            }
            
            # Test connection with timeout
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.info(f"Connected to Snowflake database: {database} (warehouse: {warehouse})")
            return True
            
        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(f"Failed to connect to Snowflake {account}/{database}: {type(e).__name__}: {e}")
            self.engine = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Snowflake: {type(e).__name__}: {e}")
            self.engine = None
            return False

    def connect_clickhouse(self, host: str, port: int = 8123, database: str = "default",
                           username: str = "default", password: str = "",
                           protocol: str = "http", secure: bool = False) -> bool:
        """Connect to ClickHouse database via SQLAlchemy."""
        try:
            if not all([host, database]):
                logger.error("Missing required ClickHouse connection parameters")
                return False

            # Build connection string based on protocol
            encoded_password = quote_plus(password) if password else ""
            encoded_username = quote_plus(username)

            if protocol == "native":
                scheme = "clickhouse+native"
                connection_string = f"{scheme}://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
            elif secure:
                scheme = "clickhouse+https"
                connection_string = f"{scheme}://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
            else:
                scheme = "clickhouse+http"
                connection_string = f"{scheme}://{encoded_username}:{encoded_password}@{host}:{port}/{database}"

            logger.info(f"Connecting to ClickHouse at {host}:{port}/{database} via {scheme}")
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self._connection_pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=CONNECTION_TIMEOUT,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
            )
            self.metadata = MetaData()

            # Clear any previous Dremio REST connection state
            self._dremio_rest_connection = None

            self.connection_info = {
                "type": "clickhouse",
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "protocol": protocol,
                "secure": secure,
            }

            # Store connection parameters for reconnection
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

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            logger.info(f"Connected to ClickHouse database: {database} at {host}:{port}")
            return True

        except (SQLAlchemyError, OperationalError, DatabaseError) as e:
            logger.error(f"Failed to connect to ClickHouse {host}:{port}/{database}: {type(e).__name__}: {e}")
            self.engine = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to ClickHouse: {type(e).__name__}: {e}")
            self.engine = None
            return False

    def connect_dremio(self, host: str = None, port: int = None, username: str = None, password: str = None,
                       ssl: bool = False, uri: str = None, pat: str = None) -> bool:
        """Connect to Dremio using REST API instead of PostgreSQL protocol."""
        logger.info(f"🔍 MCP DEBUG - connect_dremio called with host={host}, port={port}, username={username}, password={'***SET***' if password else 'NOT SET'}, ssl={ssl}, uri={uri}, pat={'***SET***' if pat else 'NOT SET'}")
        try:
            # Clear any previous SQLAlchemy engine connection state
            if self.engine:
                self.engine.dispose()
                self.engine = None
                self.metadata = None
            
            # Validate inputs - prefer PAT-based authentication
            if uri and pat:
                logger.info("🔍 MCP DEBUG - Using PAT-based authentication (preferred)")
            elif host and username:
                logger.info("🔍 MCP DEBUG - Using legacy username/password authentication")
            else:
                logger.error(f"🔍 MCP DEBUG - Missing required Dremio connection parameters. Need either (uri + pat) or (host + username)")
                return False
            
            # Import the Dremio client
            from .dremio_client import create_dremio_client

            # Create and test Dremio client connection
            async def test_dremio_connection():
                if uri and pat:
                    # PAT-based authentication
                    logger.info(f"🔍 MCP DEBUG - Connecting to Dremio API at {uri} with PAT")
                    async with await create_dremio_client(
                        uri=uri,
                        pat=pat
                    ) as client:
                        return await client.test_connection()
                else:
                    # Legacy username/password authentication
                    # For legacy mode, use port 9047 for REST API
                    api_port = 9047
                    logger.info(f"🔍 MCP DEBUG - Connecting to legacy Dremio REST API at {host}:{api_port} (SSL: {ssl})")
                    async with await create_dremio_client(
                        host=host, 
                        port=api_port, 
                        username=username, 
                        password=password, 
                        ssl=ssl
                    ) as client:
                        return await client.test_connection()
            
            # Run the async connection test
            logger.debug("Starting async connection test")
            
            # Check if there's an existing event loop (e.g., from MCP server)
            try:
                loop = asyncio.get_running_loop()
                logger.debug("Using existing event loop from MCP server")
                # If we're already in an async context, use sync_to_async approach
                import concurrent.futures
                import threading
                
                # Create a new thread to run the async code
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, test_dremio_connection())
                    connection_result = future.result(timeout=CONNECTION_TIMEOUT)
                logger.debug(f"Async connection test completed via thread: {connection_result}")
                
            except RuntimeError:
                # No event loop is running, we can create our own
                logger.debug("No existing event loop, creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    connection_result = loop.run_until_complete(test_dremio_connection())
                    logger.debug(f"Async connection test completed: {connection_result}")
                except Exception as async_error:
                    logger.error(f"Async connection test failed with exception: {type(async_error).__name__}: {async_error}")
                    return False
                finally:
                    loop.close()
            
            logger.info(f"🔍 MCP DEBUG - Async connection test result: {connection_result}")
            
            if not connection_result.get("success"):
                error_msg = connection_result.get('error', 'Unknown error')
                error_type = connection_result.get('error_type', 'Unknown error type')
                logger.error(f"🔍 MCP DEBUG - Dremio connection test failed: {error_msg} (Type: {error_type})")
                logger.error(f"Full connection result: {connection_result}")
                return False
            
            # Store connection info for Dremio REST API
            api_port = 9047
            if uri and pat:
                self.connection_info = {
                    "type": "dremio",
                    "uri": uri,
                    "auth_method": "PAT",
                    "api": "REST"
                }
                # Store connection parameters for reconnection
                self._last_connection_params = {
                    "type": "dremio",
                    "uri": uri,
                    "pat": pat
                }
                # Set connection details for REST API
                self._dremio_rest_connection = {
                    "uri": uri,
                    "pat": pat
                }
            else:
                # Legacy connection info
                self.connection_info = {
                    "type": "dremio", 
                    "host": host,
                    "port": api_port,
                    "username": username,
                    "ssl": ssl,
                    "auth_method": "username_password",
                    "api": "REST"
                }
                # Store connection parameters for reconnection
                self._last_connection_params = {
                    "type": "dremio",
                    "host": host,
                    "port": port,  # Original port for compatibility
                    "username": username,
                    "password": password,
                    "ssl": ssl
                }
                # Set connection details for REST API
                self._dremio_rest_connection = {
                    "host": host,
                    "port": api_port,
                    "username": username,
                    "password": password,
                    "ssl": ssl
                }
            
            
            # Don't set self.engine for Dremio REST API connections
            self.engine = None
            self.metadata = None
            
            if uri and pat:
                logger.info(f"✅ Successfully connected to Dremio via REST API at {uri}")
            else:
                logger.info(f"✅ Successfully connected to Dremio via REST API at {host}:{api_port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Dremio: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.engine = None
            return False
    
    def get_schemas(self) -> List[str]:
        """Get list of available schemas."""
        # Check for Dremio REST connection
        if self._dremio_rest_connection:
            return self._get_dremio_schemas()
        
        # Traditional SQL database check
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        try:
            with self.get_connection() as conn:
                # Use proper parameterized query for schema filtering
                db_type = self.connection_info.get("type")
                if db_type == "snowflake":
                    excluded_schemas = "', '".join(SNOWFLAKE_SYSTEM_SCHEMAS)
                    query = text(f"""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name NOT IN ('{excluded_schemas}')
                        ORDER BY schema_name
                    """)
                elif db_type == "clickhouse":
                    excluded_schemas = "', '".join(CLICKHOUSE_SYSTEM_SCHEMAS)
                    query = text(f"""
                        SELECT name
                        FROM system.databases
                        WHERE name NOT IN ('{excluded_schemas}')
                        ORDER BY name
                    """)
                elif db_type == "dremio":
                    excluded_schemas = "', '".join(DREMIO_SYSTEM_SCHEMAS)
                    query = text(f"""
                        SELECT schema_name
                        FROM information_schema.schemata
                        WHERE schema_name NOT IN ('{excluded_schemas}')
                        ORDER BY schema_name
                    """)
                else:  # postgresql
                    excluded_schemas = "', '".join(POSTGRES_SYSTEM_SCHEMAS)
                    # For PostgreSQL, also exclude temporary schemas (pg_temp_*, pg_toast_temp_*)
                    query = text(f"""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name NOT IN ('{excluded_schemas}')
                          AND schema_name NOT LIKE 'pg_temp_%'
                          AND schema_name NOT LIKE 'pg_toast_temp_%'
                        ORDER BY schema_name
                    """)
                result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get schemas: {e}")
            return []
    
    def _get_dremio_schemas(self) -> List[str]:
        """Get Dremio schemas/spaces via REST API."""
        import asyncio
        from .dremio_client import create_dremio_client
        
        async def fetch_schemas():
            """Fetch Dremio catalogs/schemas via REST API."""
            try:
                conn = self._dremio_rest_connection
                if conn.get('uri') and conn.get('pat'):
                    # PAT-based authentication
                    client = await create_dremio_client(uri=conn['uri'], pat=conn['pat'])
                else:
                    # Legacy authentication
                    client = await create_dremio_client(
                        host=conn['host'], 
                        port=conn['port'],
                        username=conn['username'],
                        password=conn['password'],
                        ssl=conn.get('ssl', False)
                    )
                
                async with client:
                    # Get top-level catalogs first
                    catalogs = await client.get_catalogs()
                    schemas = set()  # Use set to avoid duplicates
                    
                    logger.debug(f"Dremio catalogs response: {catalogs}")
                    
                    # Process top-level catalogs
                    for catalog in catalogs:
                        path = catalog.get('path', [])
                        catalog_type = catalog.get('type', '')
                        catalog_id = catalog.get('id', '')
                        
                        logger.debug(f"Top-level catalog: path={path}, type={catalog_type}, id={catalog_id}")
                        
                        if isinstance(path, list) and len(path) > 0:
                            top_level = path[0]
                            full_path = '.'.join(path)
                            
                            # Add top-level space/source
                            if top_level and top_level not in DREMIO_SYSTEM_SCHEMAS:
                                schemas.add(top_level)
                            
                            # Add full path if nested
                            if len(path) > 1 and full_path not in DREMIO_SYSTEM_SCHEMAS:
                                schemas.add(full_path)
                            
                            # For containers (SPACE, SOURCE), recursively get children
                            if catalog_type in ['CONTAINER', 'SPACE', 'SOURCE'] and catalog_id:
                                await self._add_dremio_children_recursive(client, path, schemas, max_depth=3)
                        else:
                            # Handle simple string paths
                            catalog_name = catalog.get('name') or str(path) if path else ''
                            if catalog_name and catalog_name not in DREMIO_SYSTEM_SCHEMAS:
                                schemas.add(catalog_name)
                    
                    # Convert back to sorted list
                    schema_list = sorted(list(schemas))
                    
                    # If no schemas found, provide defaults
                    if not schema_list:
                        logger.info("No catalogs found, using default Dremio spaces")
                        schema_list = ["@dremio", "Samples"]
                    
                    logger.info(f"Found {len(schema_list)} Dremio schemas/spaces: {schema_list[:10]}")  # Log first 10
                    return schema_list
                    
            except Exception as e:
                logger.error(f"Failed to get Dremio schemas via REST: {e}")
                return []
        
        # Handle async execution in sync context
        try:
            loop = asyncio.get_running_loop()
            # Run in thread if already in async context
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, fetch_schemas())
                return future.result(timeout=CONNECTION_TIMEOUT)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(fetch_schemas())
    
    async def _add_dremio_children_recursive(self, client, path: List[str], schemas: set, max_depth: int = 3, current_depth: int = 0):
        """Recursively add children of Dremio containers to the schemas set."""
        if current_depth >= max_depth:
            logger.debug(f"Max depth {max_depth} reached for path: {'.'.join(path)}")
            return
            
        try:
            # Get detailed info about this catalog item
            catalog_info = await client.get_catalog_info(path)
            logger.debug(f"Catalog info for {'.'.join(path)}: {catalog_info}")
            
            # Check if this catalog has children
            children = catalog_info.get('children', [])
            if not children:
                logger.debug(f"No children found for path: {'.'.join(path)}")
                return
                
            for child in children:
                child_path = child.get('path', [])
                child_type = child.get('type', '')
                child_name = child.get('name', '')
                
                if not child_path or not isinstance(child_path, list):
                    continue
                    
                # Only add containers/folders to schemas, not datasets/tables
                if child_type in ['CONTAINER', 'SPACE', 'SOURCE', 'FOLDER', 'HOME']:
                    full_child_path = '.'.join(child_path)
                    if full_child_path and full_child_path not in DREMIO_SYSTEM_SCHEMAS:
                        schemas.add(full_child_path)
                        logger.debug(f"Added child schema (type: {child_type}): {full_child_path}")
                    
                    # Recurse into this container if we haven't reached max depth
                    if current_depth < max_depth - 1:
                        await self._add_dremio_children_recursive(client, child_path, schemas, max_depth, current_depth + 1)
                else:
                    # Skip datasets/tables/files - don't add them to schemas
                    logger.debug(f"Skipping non-container (type: {child_type}): {'.'.join(child_path)}")
                    
        except Exception as e:
            logger.warning(f"Failed to get children for path {'.'.join(path)}: {e}")
    
    def _get_dremio_tables(self, schema_name: Optional[str] = None) -> List[str]:
        """Get Dremio tables via REST API."""
        import asyncio
        from .dremio_client import create_dremio_client
        
        async def fetch_tables():
            """Fetch tables from Dremio schema/space via REST API."""
            try:
                conn = self._dremio_rest_connection
                if conn.get('uri') and conn.get('pat'):
                    client = await create_dremio_client(uri=conn['uri'], pat=conn['pat'])
                else:
                    client = await create_dremio_client(
                        host=conn['host'],
                        port=conn['port'],
                        username=conn['username'],
                        password=conn['password'],
                        ssl=conn.get('ssl', False)
                    )
                
                async with client:
                    # If no schema specified, get tables from all schemas
                    if not schema_name:
                        # Execute query to get all tables
                        query = "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.\"TABLES\" WHERE TABLE_TYPE = 'TABLE'"
                    else:
                        # Get tables from specific schema
                        safe_schema = self._escape_sql_literal(schema_name)
                        query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.\"TABLES\" WHERE TABLE_SCHEMA = '{safe_schema}' AND TABLE_TYPE = 'TABLE'"
                    
                    result = await client.execute_query(query)
                    
                    if result.get('success'):
                        tables = []
                        for row in result.get('data', []):
                            if schema_name:
                                # Just table name if schema specified
                                table_name = row.get('TABLE_NAME', '')
                            else:
                                # Include schema prefix if no specific schema
                                schema = row.get('TABLE_SCHEMA', '')
                                table_name = row.get('TABLE_NAME', '')
                                if schema and schema != 'INFORMATION_SCHEMA':
                                    table_name = f"{schema}.{table_name}" if table_name else ''
                            
                            if table_name:
                                tables.append(table_name)
                        
                        return sorted(list(set(tables)))  # Remove duplicates and sort
                    else:
                        logger.error(f"Failed to get Dremio tables: {result.get('error')}")
                        return []
                    
            except Exception as e:
                logger.error(f"Failed to get Dremio tables via REST: {e}")
                return []
        
        # Handle async execution
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, fetch_tables())
                return future.result(timeout=CONNECTION_TIMEOUT)
        except RuntimeError:
            return asyncio.run(fetch_tables())
    
    def _analyze_dremio_table(self, table_name: str, schema_name: Optional[str] = None) -> Optional[TableInfo]:
        """Analyze a Dremio table via REST API."""
        import asyncio
        from .dremio_client import create_dremio_client
        
        async def fetch_table_info():
            """Fetch table structure from Dremio via REST API."""
            try:
                conn = self._dremio_rest_connection
                if conn.get('uri') and conn.get('pat'):
                    client = await create_dremio_client(uri=conn['uri'], pat=conn['pat'])
                else:
                    client = await create_dremio_client(
                        host=conn['host'],
                        port=conn['port'],
                        username=conn['username'],
                        password=conn['password'],
                        ssl=conn.get('ssl', False)
                    )
                
                async with client:
                    # If schema_name isn't provided but table_name is qualified, split it.
                    if not schema_name and "." in str(table_name):
                        parts = str(table_name).split(".")
                        schema_name = ".".join(parts[:-1]) if len(parts) > 1 else None
                        table_name = parts[-1]

                    # Get column information
                    if schema_name:
                        safe_schema = self._escape_sql_literal(schema_name)
                        safe_table = self._escape_sql_literal(table_name)
                        column_query = f"""
                        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, ORDINAL_POSITION
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{safe_table}' AND TABLE_SCHEMA = '{safe_schema}'
                        ORDER BY ORDINAL_POSITION
                        """
                    else:
                        safe_table = self._escape_sql_literal(table_name)
                        column_query = f"""
                        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, ORDINAL_POSITION
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{safe_table}'
                        ORDER BY ORDINAL_POSITION
                        """
                    
                    result = await client.execute_query(column_query)
                    
                    if not result.get('success'):
                        logger.error(f"Failed to get column info for {table_name}: {result.get('error')}")
                        return None
                    
                    columns = []
                    for row in result.get('data', []):
                        col_info = ColumnInfo(
                            name=row.get('COLUMN_NAME', ''),
                            data_type=row.get('DATA_TYPE', 'VARCHAR'),
                            is_nullable=row.get('IS_NULLABLE', 'YES') == 'YES',
                            is_primary_key=False,  # Dremio doesn't have traditional PKs
                            is_foreign_key=False,  # Dremio doesn't have traditional FKs
                            foreign_key_table=None,
                            foreign_key_column=None,
                            comment=None
                        )
                        columns.append(col_info)
                    
                    # Create table info
                    table_info = TableInfo(
                        name=table_name,
                        schema=schema_name or 'default',
                        columns=columns,
                        primary_keys=[],  # Dremio doesn't have traditional PKs
                        foreign_keys=[],  # Dremio doesn't have traditional FKs
                        comment=None
                    )
                    
                    return table_info
                    
            except Exception as e:
                logger.error(f"Failed to analyze Dremio table {table_name}: {e}")
                return None
        
        # Handle async execution
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, fetch_table_info())
                return future.result(timeout=CONNECTION_TIMEOUT)
        except RuntimeError:
            return asyncio.run(fetch_table_info())
    
    def get_tables(self, schema_name: Optional[str] = None) -> List[str]:
        """Get list of tables in a schema with caching for performance."""
        logger.debug(f"get_tables: Starting, has_engine: {self.has_engine()}, dremio_rest: {bool(self._dremio_rest_connection)}")
        
        # Check cache first for performance
        cache_key = self._get_cache_key("get_tables", schema_name or "default")
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Check for Dremio REST connection
        if self._dremio_rest_connection:
            return self._get_dremio_tables(schema_name)
        
        # Ensure connection before proceeding
        try:
            self._ensure_connection()
        except RuntimeError as e:
            logger.error(f"get_tables: Connection check failed: {e}")
            raise
        
        logger.debug(f"get_tables: After ensure_connection, engine exists: {self.has_engine()}")
        
        try:
            db_type = self.connection_info.get("type")
            with self.get_connection() as conn:
                if db_type == "clickhouse":
                    # ClickHouse uses system.tables; filter out views
                    if schema_name:
                        query = text("""
                            SELECT name
                            FROM system.tables
                            WHERE database = :schema_name
                              AND engine NOT IN ('View', 'MaterializedView')
                            ORDER BY name
                        """)
                        result = conn.execute(query, {"schema_name": schema_name})
                    else:
                        db_name = self.connection_info.get("database", "default")
                        query = text("""
                            SELECT name
                            FROM system.tables
                            WHERE database = :db_name
                              AND engine NOT IN ('View', 'MaterializedView')
                            ORDER BY name
                        """)
                        result = conn.execute(query, {"db_name": db_name})
                elif schema_name:
                    query = text("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = :schema_name
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """)
                    result = conn.execute(query, {"schema_name": schema_name})
                else:
                    query = text("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """)
                    result = conn.execute(query)

                tables = [row[0] for row in result.fetchall()]

                # Cache the result for performance
                self._store_in_cache(cache_key, tables)
                return tables

        except SQLAlchemyError as e:
            logger.error(f"Failed to get tables: {e}")
            return []

    def prefetch_schema_constraints(self, schema_name: str) -> None:
        """Prefetch all PKs and FKs for a schema at once (Snowflake optimization).

        This avoids repeated SHOW PRIMARY KEYS/IMPORTED KEYS queries for each table.
        Results are cached and used by analyze_table.
        """
        db_type = self.connection_info.get("type", "")
        if db_type != "snowflake":
            return  # Only needed for Snowflake

        pk_cache_key = self._get_cache_key("schema_pks", schema_name)
        fk_cache_key = self._get_cache_key("schema_fks", schema_name)

        # Skip if already cached
        if self._get_from_cache(pk_cache_key) is not None:
            logger.debug(f"Schema constraints already cached for {schema_name}")
            return

        # Get database name for full path (required by Snowflake)
        database_name = self.connection_info.get("database", "")
        if database_name:
            full_schema_path = f'"{database_name}"."{schema_name}"'
        else:
            full_schema_path = f'"{schema_name}"'

        logger.info(f"Prefetching PKs and FKs for schema {full_schema_path}")

        try:
            with self.get_connection() as conn:
                # Fetch all PKs for the schema at once
                pk_by_table: Dict[str, List[str]] = {}
                pk_success = False
                try:
                    pk_query = text(f'SHOW PRIMARY KEYS IN SCHEMA {full_schema_path}')
                    self._log_sql_query(str(pk_query))
                    result = conn.execute(pk_query)
                    for row in result.fetchall():
                        row_dict = row._asdict() if hasattr(row, '_asdict') else dict(row._mapping)
                        table = row_dict.get('table_name')
                        column = row_dict.get('column_name')
                        if table and column:
                            if table not in pk_by_table:
                                pk_by_table[table] = []
                            pk_by_table[table].append(column)
                    logger.info(f"Prefetched PKs for {len(pk_by_table)} tables")
                    pk_success = True
                except Exception as e:
                    logger.warning(f"Failed to prefetch PKs: {e}")

                # Only cache if query succeeded
                if pk_success:
                    self._store_in_cache(pk_cache_key, pk_by_table)

                # Fetch all FKs for the schema at once
                fk_by_table: Dict[str, List[Dict]] = {}
                fk_success = False
                try:
                    fk_query = text(f'SHOW IMPORTED KEYS IN SCHEMA {full_schema_path}')
                    self._log_sql_query(str(fk_query))
                    result = conn.execute(fk_query)
                    for row in result.fetchall():
                        row_dict = row._asdict() if hasattr(row, '_asdict') else dict(row._mapping)
                        fk_table = row_dict.get('fk_table_name')
                        pk_table = row_dict.get('pk_table_name')
                        pk_column = row_dict.get('pk_column_name')
                        pk_schema = row_dict.get('pk_schema_name')
                        fk_column = row_dict.get('fk_column_name')
                        fk_name = row_dict.get('fk_name')

                        if fk_table and pk_table and fk_column:
                            if fk_table not in fk_by_table:
                                fk_by_table[fk_table] = []
                            fk_by_table[fk_table].append({
                                'constrained_columns': [fk_column],
                                'referred_schema': pk_schema,
                                'referred_table': pk_table,
                                'referred_columns': [pk_column],
                                'name': fk_name
                            })
                    logger.info(f"Prefetched FKs for {len(fk_by_table)} tables")
                    fk_success = True
                except Exception as e:
                    logger.warning(f"Failed to prefetch FKs: {e}")

                # Only cache if query succeeded
                if fk_success:
                    self._store_in_cache(fk_cache_key, fk_by_table)

                # Fetch all columns for the schema at once
                cols_cache_key = self._get_cache_key("schema_cols", schema_name)
                cols_by_table: Dict[str, List[Dict]] = {}
                cols_success = False
                try:
                    cols_query = text(f'''
                        SELECT table_name, column_name, data_type,
                               character_maximum_length, numeric_precision, numeric_scale,
                               is_nullable, column_default, is_identity, comment
                        FROM information_schema.columns
                        WHERE table_schema = :schema_name
                        ORDER BY table_name, ordinal_position
                    ''')
                    self._log_sql_query(str(cols_query))
                    result = conn.execute(cols_query, {"schema_name": schema_name})
                    for row in result.fetchall():
                        row_dict = row._asdict() if hasattr(row, '_asdict') else dict(row._mapping)
                        table = row_dict.get('table_name') or row_dict.get('TABLE_NAME')
                        if table:
                            if table not in cols_by_table:
                                cols_by_table[table] = []
                            cols_by_table[table].append({
                                'name': row_dict.get('column_name') or row_dict.get('COLUMN_NAME'),
                                'type': row_dict.get('data_type') or row_dict.get('DATA_TYPE'),
                                'nullable': (row_dict.get('is_nullable') or row_dict.get('IS_NULLABLE', 'YES')).upper() == 'YES',
                                'default': row_dict.get('column_default') or row_dict.get('COLUMN_DEFAULT'),
                                'comment': row_dict.get('comment') or row_dict.get('COMMENT'),
                            })
                    logger.info(f"Prefetched columns for {len(cols_by_table)} tables")
                    cols_success = True
                except Exception as e:
                    logger.warning(f"Failed to prefetch columns: {e}")

                # Only cache if query succeeded
                if cols_success:
                    self._store_in_cache(cols_cache_key, cols_by_table)

        except SQLAlchemyError as e:
            logger.error(f"Failed to prefetch schema metadata: {e}")

    def analyze_table(self, table_name: str, schema_name: Optional[str] = None) -> Optional[TableInfo]:
        """Analyze a specific table and return detailed information."""
        logger.debug(f"analyze_table: Starting analysis of {table_name}, has_engine: {self.has_engine()}, dremio_rest: {bool(self._dremio_rest_connection)}")
        
        # Check for Dremio REST connection
        if self._dremio_rest_connection:
            return self._analyze_dremio_table(table_name, schema_name)
        
        # Ensure connection before proceeding
        try:
            self._ensure_connection()
        except RuntimeError as e:
            logger.error(f"analyze_table: Connection check failed: {e}")
            raise
        
        logger.debug(f"analyze_table: After ensure_connection, engine exists: {self.has_engine()}")
        
        try:
            with self.get_connection() as conn:
                # Use SQLAlchemy inspector for safer metadata reflection
                inspector = inspect(self.engine)
                db_type = self.connection_info.get("type", "")

                # Get columns, PKs, FKs - use cache for Snowflake to avoid repeated queries
                table_columns = None
                primary_keys = []
                table_fks = []

                if db_type == "snowflake" and schema_name:
                    # Try to get from prefetched cache first
                    cols_cache_key = self._get_cache_key("schema_cols", schema_name)
                    pk_cache_key = self._get_cache_key("schema_pks", schema_name)
                    fk_cache_key = self._get_cache_key("schema_fks", schema_name)

                    cached_cols = self._get_from_cache(cols_cache_key)
                    cached_pks = self._get_from_cache(pk_cache_key)
                    cached_fks = self._get_from_cache(fk_cache_key)

                    # Use cached columns if available
                    if cached_cols is not None:
                        # Case-insensitive table lookup
                        table_columns = cached_cols.get(table_name) or cached_cols.get(table_name.upper())
                        if table_columns:
                            logger.debug(f"Using cached columns for {table_name}: {len(table_columns)} columns")
                        else:
                            logger.error(f"Table {schema_name}.{table_name} not found in cache")
                            return None

                    if cached_pks is not None:
                        primary_keys = cached_pks.get(table_name, []) or cached_pks.get(table_name.upper(), [])
                        logger.debug(f"Using cached PKs for {table_name}: {primary_keys}")
                    else:
                        # Fallback to inspector (will trigger SHOW query)
                        table_pk = inspector.get_pk_constraint(table_name, schema=schema_name)
                        primary_keys = table_pk.get('constrained_columns', []) if table_pk else []

                    if cached_fks is not None:
                        table_fks = cached_fks.get(table_name, []) or cached_fks.get(table_name.upper(), [])
                        logger.debug(f"Using cached FKs for {table_name}: {len(table_fks)} constraints")
                    else:
                        # Fallback to inspector (will trigger SHOW query)
                        table_fks = inspector.get_foreign_keys(table_name, schema=schema_name)
                else:
                    # Non-Snowflake: use inspector directly
                    if schema_name:
                        if not inspector.has_table(table_name, schema=schema_name):
                            logger.error(f"Table {schema_name}.{table_name} not found")
                            return None
                        table_columns = inspector.get_columns(table_name, schema=schema_name)
                        table_pk = inspector.get_pk_constraint(table_name, schema=schema_name)
                        table_fks = inspector.get_foreign_keys(table_name, schema=schema_name)
                    else:
                        if not inspector.has_table(table_name):
                            logger.error(f"Table {table_name} not found")
                            return None
                        table_columns = inspector.get_columns(table_name)
                        table_pk = inspector.get_pk_constraint(table_name)
                        table_fks = inspector.get_foreign_keys(table_name)
                    primary_keys = table_pk.get('constrained_columns', []) if table_pk else []

                # Fallback: if columns not from cache, fetch with table-specific query
                if table_columns is None:
                    db_type = self.connection_info.get("type", "")
                    if db_type == "snowflake" and schema_name:
                        # Use efficient table-filtered query for Snowflake
                        try:
                            cols_query = text('''
                                SELECT column_name, data_type, is_nullable, column_default, comment
                                FROM information_schema.columns
                                WHERE table_schema = :schema_name AND table_name = :table_name
                                ORDER BY ordinal_position
                            ''')
                            self._log_sql_query(str(cols_query))
                            result = conn.execute(cols_query, {"schema_name": schema_name, "table_name": table_name})
                            rows = result.fetchall()
                            if not rows:
                                logger.error(f"Table {schema_name}.{table_name} not found")
                                return None
                            table_columns = []
                            for row in rows:
                                row_dict = row._asdict() if hasattr(row, '_asdict') else dict(row._mapping)
                                table_columns.append({
                                    'name': row_dict.get('column_name') or row_dict.get('COLUMN_NAME'),
                                    'type': row_dict.get('data_type') or row_dict.get('DATA_TYPE'),
                                    'nullable': (row_dict.get('is_nullable') or row_dict.get('IS_NULLABLE', 'YES')).upper() == 'YES',
                                    'default': row_dict.get('column_default') or row_dict.get('COLUMN_DEFAULT'),
                                    'comment': row_dict.get('comment') or row_dict.get('COMMENT'),
                                })
                        except Exception as e:
                            logger.warning(f"Direct column query failed, falling back to inspector: {e}")
                            table_columns = inspector.get_columns(table_name, schema=schema_name)
                    elif schema_name:
                        if not inspector.has_table(table_name, schema=schema_name):
                            logger.error(f"Table {schema_name}.{table_name} not found")
                            return None
                        table_columns = inspector.get_columns(table_name, schema=schema_name)
                    else:
                        if not inspector.has_table(table_name):
                            logger.error(f"Table {table_name} not found")
                            return None
                        table_columns = inspector.get_columns(table_name)

                # Process column information
                columns = []
                foreign_keys = []

                # Log PK/FK information for debugging
                logger.info(f"Table {schema_name}.{table_name}: PKs={primary_keys}, FKs={len(table_fks)} constraints")
                if table_fks:
                    for fk in table_fks:
                        logger.info(f"  FK: {fk}")

                # Normalize primary_keys to uppercase for case-insensitive matching (Snowflake uses uppercase)
                primary_keys_upper = [pk.upper() for pk in primary_keys]

                for col_info in table_columns:
                    column_name = col_info['name']
                    # Case-insensitive PK matching
                    is_pk = column_name.upper() in primary_keys_upper
                    
                    # Check for foreign keys
                    fk_table = None
                    fk_column = None
                    is_fk = False
                    
                    for fk in table_fks:
                        # Case-insensitive FK matching
                        constrained_cols_upper = [c.upper() for c in fk.get('constrained_columns', [])]
                        if column_name.upper() in constrained_cols_upper:
                            is_fk = True
                            fk_idx = constrained_cols_upper.index(column_name.upper())
                            fk_table = fk.get('referred_table')
                            referred_cols = fk.get('referred_columns', [])
                            fk_column = referred_cols[fk_idx] if fk_idx < len(referred_cols) else None
                            # Handle schema-qualified references (common in Snowflake)
                            fk_schema = fk.get('referred_schema')
                            if fk_table:
                                foreign_keys.append({
                                    "column": column_name,
                                    "referenced_table": fk_table,
                                    "referenced_column": fk_column,
                                    "referenced_schema": fk_schema
                                })
                                logger.debug(f"Added FK: {column_name} -> {fk_schema}.{fk_table}.{fk_column}")
                            break
                    
                    column_info = ColumnInfo(
                        name=column_name,
                        data_type=str(col_info['type']),
                        is_nullable=col_info['nullable'],
                        is_primary_key=is_pk,
                        is_foreign_key=is_fk,
                        foreign_key_table=fk_table,
                        foreign_key_column=fk_column,
                        comment=col_info.get('comment')
                    )
                    columns.append(column_info)
                
                # Row count and sample data are not fetched during schema analysis
                # Use sample_table_data tool to get sample data for specific tables
                row_count = None
                sample_data = None
                table_comment = None

                # ClickHouse-specific: enrich with engine, sorting_key, row count
                if db_type == "clickhouse" and schema_name:
                    try:
                        ch_meta_query = text("""
                            SELECT engine, sorting_key, total_rows
                            FROM system.tables
                            WHERE database = :db AND name = :tbl
                        """)
                        ch_result = conn.execute(ch_meta_query, {"db": schema_name, "tbl": table_name})
                        ch_row = ch_result.fetchone()
                        if ch_row:
                            ch_dict = ch_row._asdict() if hasattr(ch_row, '_asdict') else dict(ch_row._mapping)
                            engine = ch_dict.get("engine", "")
                            sorting_key = ch_dict.get("sorting_key", "")
                            total_rows = ch_dict.get("total_rows")
                            table_comment = f"Engine: {engine}"
                            if sorting_key:
                                table_comment += f", ORDER BY: {sorting_key}"
                            if total_rows is not None:
                                row_count = int(total_rows)
                            # Use sorting_key as primary keys if Inspector returned empty
                            if not primary_keys and sorting_key:
                                primary_keys = [k.strip() for k in sorting_key.split(",")]
                    except Exception as e:
                        logger.warning(f"Failed to fetch ClickHouse metadata for {table_name}: {e}")

                return TableInfo(
                    name=table_name,
                    schema=schema_name or "public",
                    columns=columns,
                    primary_keys=primary_keys,
                    foreign_keys=foreign_keys,
                    comment=table_comment,
                    row_count=row_count,
                    sample_data=sample_data
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to analyze table {table_name}: {e}")
            return None
    
    def _validate_identifier(self, identifier: str) -> bool:
        """Validate database identifier to prevent injection attacks."""
        if not identifier or len(identifier) > 63:  # PostgreSQL limit
            return False
        
        # Use regex pattern for strict validation
        return bool(re.match(IDENTIFIER_PATTERN, identifier))
    
    def _strip_leading_sql_comments(self, sql_query: str) -> str:
        """Strip leading SQL comments to find the actual SQL statement.

        Handles both -- (line comments) and /* */ (block comments) at the beginning of queries.

        Args:
            sql_query: SQL query that may contain leading comments

        Returns:
            SQL query with leading comments removed
        """
        lines = sql_query.split('\n')
        result_lines = []
        in_block_comment = False

        for idx, original_line in enumerate(lines):
            line = original_line.strip()

            # Handle block comment continuation
            if in_block_comment:
                if '*/' in line:
                    # End of block comment found
                    after_comment = line.split('*/', 1)[1].strip()
                    in_block_comment = False
                    if after_comment:  # If there's SQL after the comment
                        result_lines.append(after_comment)
                continue

            # Skip empty lines
            if not line:
                continue

            # Handle line comments
            if line.startswith('--'):
                continue

            # Handle block comments
            if line.startswith('/*'):
                if '*/' in line:
                    # Single-line block comment
                    after_comment = line.split('*/', 1)[1].strip()
                    if after_comment:  # If there's SQL after the comment
                        result_lines.append(after_comment)
                        break
                else:
                    # Multi-line block comment starts
                    in_block_comment = True
                continue

            # This line contains actual SQL - add it and all remaining lines
            result_lines.append(original_line)
            remaining_index = idx + 1
            if remaining_index < len(lines):
                result_lines.extend(lines[remaining_index:])
            break

        return '\n'.join(result_lines).strip()

    def _escape_sql_literal(self, value: str) -> str:
        """Escape a value for safe use inside a single-quoted SQL literal."""
        return value.replace("'", "''")

    def _quote_dremio_identifier(self, identifier: str) -> str:
        """Quote and escape a Dremio identifier or path safely."""
        if identifier is None:
            return ""

        parts = [p for p in str(identifier).split('.') if p != ""]
        if not parts:
            return ""

        escaped_parts = [part.replace('"', '""') for part in parts]
        return ".".join(f'"{part}"' for part in escaped_parts)
    
    def get_table_relationships(self, schema_name: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """Get relationships between tables in a schema."""
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        relationships = {}
        tables = self.get_tables(schema_name)
        
        for table_name in tables:
            table_info = self.analyze_table(table_name, schema_name)
            if table_info and table_info.foreign_keys:
                relationships[table_name] = table_info.foreign_keys
        
        return relationships
    
    def analyze_schema_concurrent(self, schema_name: Optional[str] = None, max_workers: int = 5) -> List[TableInfo]:
        """Analyze schema with concurrent table processing for better performance."""
        tables = self.get_tables(schema_name)
        if not tables:
            return []
        
        # Limit concurrent workers to avoid overwhelming the database
        max_workers = min(max_workers, len(tables), 10)
        
        logger.info(f"Analyzing {len(tables)} tables concurrently with {max_workers} workers")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all table analysis tasks
            future_to_table = {
                executor.submit(self.analyze_table, table_name, schema_name): table_name 
                for table_name in tables
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_table):
                table_name = future_to_table[future]
                try:
                    table_info = future.result(timeout=QUERY_TIMEOUT)  # Use query timeout for table analysis
                    if table_info:
                        results.append(table_info)
                        logger.debug(f"Completed analysis of table: {table_name}")
                    else:
                        logger.warning(f"No information returned for table: {table_name}")
                except Exception as e:
                    logger.error(f"Failed to analyze table {table_name}: {e}")
        
        logger.info(f"Concurrent schema analysis completed: {len(results)}/{len(tables)} tables analyzed")
        return results
    
    def sample_table_data(self, table_name: str, schema_name: Optional[str] = None, 
                         limit: int = DEFAULT_SAMPLE_LIMIT) -> List[Dict[str, Any]]:
        """Sample data from a table for analysis with enhanced validation."""
        # Check for Dremio REST connection first
        if self._dremio_rest_connection:
            return self._sample_dremio_table(table_name, schema_name, limit)
        
        # For traditional SQL databases, check engine
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        # Validate inputs securely
        if not self._validate_identifier_secure(table_name):
            logger.error(f"Invalid table name format: {table_name}")
            raise ValueError(f"Invalid table name format: {table_name}")
        
        if schema_name and not self._validate_identifier_secure(schema_name):
            logger.error(f"Invalid schema name format: {schema_name}")
            raise ValueError(f"Invalid schema name format: {schema_name}")
        
        # Validate and normalize limit
        if not isinstance(limit, int) or limit < MIN_SAMPLE_LIMIT:
            limit = DEFAULT_SAMPLE_LIMIT
        elif limit > MAX_SAMPLE_LIMIT:
            limit = MAX_SAMPLE_LIMIT
            logger.warning(f"Sample limit capped at {MAX_SAMPLE_LIMIT}")
        
        try:
            with self.get_connection() as conn:
                # Use SQLAlchemy's text with bound parameters for safety
                if schema_name:
                    # Double-quote identifiers to handle case sensitivity
                    full_table_name = f'"{schema_name}"."{table_name}"'
                else:
                    full_table_name = f'"{table_name}"'
                
                query_str = f'SELECT * FROM {full_table_name} LIMIT :limit'
                params = {"limit": limit}
                self._log_sql_query(query_str, params)
                query = text(query_str)
                result = conn.execute(query, params)
                columns = list(result.keys())
                
                rows = []
                for row in result.fetchall():
                    row_dict = {}
                    for i, value in enumerate(row):
                        # Convert non-serializable types to strings
                        column_name = columns[i]
                        if value is not None:
                            if isinstance(value, decimal.Decimal):  # decimal/numeric types
                                row_dict[column_name] = float(value)
                            elif hasattr(value, 'isoformat'):  # datetime objects
                                row_dict[column_name] = value.isoformat()
                            elif isinstance(value, (bytes, bytearray)):
                                # Convert binary data to hex string
                                row_dict[column_name] = value.hex()
                            elif hasattr(value, '__dict__'):  # Complex objects
                                row_dict[column_name] = str(value)
                            else:
                                row_dict[column_name] = value
                        else:
                            row_dict[column_name] = None
                    rows.append(row_dict)
                
                return rows
                
        except (SQLAlchemyError, ValueError) as e:
            logger.error(f"Failed to sample data from {table_name}: {type(e).__name__}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error sampling {table_name}: {type(e).__name__}: {e}")
            return []
    
    def validate_sql_syntax(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query syntax with enhanced security checks.
        
        Uses both security validation and database-level validation to provide
        comprehensive protection against SQL injection and syntax errors.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Dictionary with validation results including security and syntax checks
        """
        # Check for database connection (either SQLAlchemy engine or Dremio REST)
        if not self.engine and not self._dremio_rest_connection:
            raise RuntimeError("No database connection established")
        
        # First, perform security validation
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
            "risk_level": security_validation.get("risk_level", "low")
        }
        
        # If security validation fails, return immediately
        if not security_validation.get("is_safe", False):
            validation_result["error"] = f"Security validation failed: {'; '.join(security_validation['issues'])}"
            validation_result["error_type"] = "security_error"
            
            # Log security violation
            audit_log_security_event(
                "sql_injection_attempt",
                {
                    "query_preview": sql_query[:100],
                    "issues": security_validation["issues"],
                    "risk_level": security_validation["risk_level"]
                },
                SecurityLevel.CRITICAL if security_validation["risk_level"] == "critical" else SecurityLevel.HIGH
            )
            
            return validation_result
        
        try:
            # Step 1: Basic security checks
            query_stripped = sql_query.strip()
            if not query_stripped:
                validation_result["error"] = "Empty query"
                validation_result["error_type"] = "empty_query"
                return validation_result
            
            # Step 2: Remove leading SQL comments to find actual SQL statement
            query_without_comments = self._strip_leading_sql_comments(query_stripped)
            query_upper = query_without_comments.upper()
            
            # Check for multiple statements (security risk) - use original query for this check
            if ';' in query_stripped[:-1]:  # Allow trailing semicolon
                validation_result["error"] = "Multiple SQL statements not allowed for security"
                validation_result["error_type"] = "security_error"
                validation_result["suggestions"].append("Split multiple statements into separate requests")
                return validation_result
            
            # Determine and validate query type using comment-stripped query
            if query_upper.startswith('SELECT'):
                validation_result["query_type"] = "SELECT"
            elif query_upper.startswith('WITH'):
                validation_result["query_type"] = "CTE_SELECT"
                if 'SELECT' not in query_upper:
                    validation_result["warnings"].append("CTE should end with SELECT statement")
            elif query_upper.startswith(('EXPLAIN', 'DESCRIBE', 'DESC', 'SHOW')):
                validation_result["query_type"] = "METADATA"
            else:
                # Check for potentially dangerous operations
                dangerous_ops = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE', 'MERGE']
                detected_ops = [op for op in dangerous_ops if query_upper.startswith(op)]
                if detected_ops:
                    validation_result["error"] = f"Destructive operations not allowed: {', '.join(detected_ops)}"
                    validation_result["error_type"] = "forbidden_operation"
                    validation_result["suggestions"].append("Use SELECT queries for data retrieval only")
                    return validation_result
                else:
                    validation_result["error"] = "Only SELECT, CTE, and metadata queries are allowed"
                    validation_result["error_type"] = "query_type_error"
                    validation_result["suggestions"].append("Start your query with SELECT, WITH, EXPLAIN, or SHOW")
                    return validation_result
            
            # Step 2: Database-level syntax validation
            # Handle Dremio REST API validation separately
            if self._dremio_rest_connection:
                return self._validate_dremio_syntax(query_stripped, validation_result)
            
            with self.get_connection() as conn:
                try:
                    if self.connection_info.get("type") == "postgresql":
                        # PostgreSQL: Use PREPARE to validate syntax without execution
                        prepare_name = "syntax_check_stmt"
                        prepare_sql = f"PREPARE {prepare_name} AS {query_stripped}"
                        deallocate_sql = f"DEALLOCATE {prepare_name}"
                        
                        try:
                            # Prepare the statement (validates syntax)
                            conn.execute(text(prepare_sql))
                            # Clean up the prepared statement
                            conn.execute(text(deallocate_sql))
                            validation_result["is_valid"] = True
                        except Exception as prepare_error:
                            # Extract meaningful error from PostgreSQL
                            error_msg = str(prepare_error)
                            validation_result["database_error"] = error_msg
                            validation_result["error"] = f"PostgreSQL syntax error: {error_msg}"
                            validation_result["error_type"] = "syntax_error"
                            
                            # Add suggestions based on common errors
                            if "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
                                validation_result["suggestions"].append("Check table/column names - they may not exist or may need proper schema qualification")
                            elif "syntax error" in error_msg.lower():
                                validation_result["suggestions"].append("Review SQL syntax - check for missing commas, parentheses, or keywords")
                            elif "permission denied" in error_msg.lower():
                                validation_result["suggestions"].append("Insufficient permissions to access the specified tables")
                            
                    elif self.connection_info.get("type") == "snowflake":
                        # Snowflake: Use EXPLAIN to validate syntax
                        explain_sql = f"EXPLAIN {query_stripped}"
                        try:
                            result = conn.execute(text(explain_sql))
                            result.fetchall()  # Consume the explain plan
                            validation_result["is_valid"] = True
                        except Exception as explain_error:
                            # Extract meaningful error from Snowflake
                            error_msg = str(explain_error)
                            validation_result["database_error"] = error_msg
                            validation_result["error"] = f"Snowflake syntax error: {error_msg}"
                            validation_result["error_type"] = "syntax_error"
                            
                            # Add suggestions based on common Snowflake errors
                            if "does not exist" in error_msg.lower():
                                validation_result["suggestions"].append("Object not found - check table/schema names and ensure proper qualification (DATABASE.SCHEMA.TABLE)")
                            elif "sql compilation error" in error_msg.lower():
                                validation_result["suggestions"].append("SQL compilation failed - review syntax and object references")
                            elif "invalid identifier" in error_msg.lower():
                                validation_result["suggestions"].append("Invalid identifier - check column names and use double quotes for case-sensitive names")
                    
                    elif self.connection_info.get("type") == "clickhouse":
                        # ClickHouse: Use EXPLAIN to validate syntax
                        explain_sql = f"EXPLAIN {query_stripped}"
                        try:
                            result = conn.execute(text(explain_sql))
                            result.fetchall()
                            validation_result["is_valid"] = True
                        except Exception as explain_error:
                            error_msg = str(explain_error)
                            validation_result["database_error"] = error_msg
                            validation_result["error"] = f"ClickHouse syntax error: {error_msg}"
                            validation_result["error_type"] = "syntax_error"

                            if "unknown table" in error_msg.lower() or "doesn't exist" in error_msg.lower():
                                validation_result["suggestions"].append("Table not found - check database.table name qualification")
                            elif "syntax error" in error_msg.lower():
                                validation_result["suggestions"].append("SQL syntax error - review query structure")
                            elif "unknown column" in error_msg.lower() or "unknown identifier" in error_msg.lower():
                                validation_result["suggestions"].append("Unknown column - check column names (ClickHouse is case-sensitive)")

                    elif self.connection_info.get("type") == "dremio":
                        # Dremio: Use EXPLAIN to validate syntax
                        explain_sql = f"EXPLAIN PLAN FOR {query_stripped}"
                        try:
                            result = conn.execute(text(explain_sql))
                            result.fetchall()  # Consume the explain plan
                            validation_result["is_valid"] = True
                        except Exception as explain_error:
                            # Extract meaningful error from Dremio
                            error_msg = str(explain_error)
                            validation_result["database_error"] = error_msg
                            validation_result["error"] = f"Dremio syntax error: {error_msg}"
                            validation_result["error_type"] = "syntax_error"
                            
                            # Add suggestions based on common Dremio errors
                            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                                validation_result["suggestions"].append("Object not found - check table/view names and ensure proper qualification")
                            elif "syntax error" in error_msg.lower():
                                validation_result["suggestions"].append("SQL syntax error - review query structure and keywords")
                            elif "validation error" in error_msg.lower():
                                validation_result["suggestions"].append("Query validation failed - check column references and data types")
                    
                    # Step 3: Extract table references if validation succeeded
                    if validation_result["is_valid"]:
                        # Extract table names using regex (basic approach)
                        table_patterns = [
                            r'\bFROM\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',  # FROM table
                            r'\bJOIN\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',  # JOIN table
                            r'\bUPDATE\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',  # UPDATE table
                            r'\bINTO\s+(?:[\w"\'`\[\]]+\.)*(["\w`\[\]]+)',  # INSERT INTO table
                        ]
                        
                        tables = set()
                        for pattern in table_patterns:
                            matches = re.findall(pattern, query_stripped, re.IGNORECASE)
                            tables.update(match.strip('"\'`[]') for match in matches)
                        
                        validation_result["affected_tables"] = list(tables)
                        
                        # Add informational warnings
                        if len(tables) > 5:
                            validation_result["warnings"].append(f"Query involves {len(tables)} tables - consider query complexity")
                        
                except Exception as conn_error:
                    validation_result["error"] = f"Database connection error during validation: {str(conn_error)}"
                    validation_result["error_type"] = "connection_error"
                    
        except Exception as e:
            validation_result["error"] = f"Validation system error: {str(e)}"
            validation_result["error_type"] = "internal_error"
            logger.error(f"SQL validation error: {e}")
            
        return validation_result
    
    def execute_sql_query(self, sql_query: str, limit: int = 1000) -> Dict[str, Any]:
        """Execute a validated SQL query and return results safely.
        
        Args:
            sql_query: SQL query to execute (must pass validation first)
            limit: Maximum number of rows to return (safety limit)
            
        Returns:
            Dictionary with query results and execution metadata
        """
        # Handle Dremio REST API connection
        if self._dremio_rest_connection:
            return self._execute_dremio_query(sql_query, limit)
        
        if not self.engine:
            raise RuntimeError("No database connection established")
        
        # Validate and cap the limit
        if limit <= 0 or limit > 5000:
            limit = min(max(limit, 100), 5000)
            
        result_data = {
            "success": False,
            "data": [],
            "columns": [],
            "row_count": 0,
            "execution_time_ms": None,
            "error": None,
            "error_type": None,
            "warnings": [],
            "query_plan": None,
            "limit_applied": False
        }
        
        try:
            import time
            start_time = time.time()
            
            # Step 1: Mandatory validation
            validation = self.validate_sql_syntax(sql_query)
            if not validation["is_valid"]:
                result_data["error"] = validation["error"]
                result_data["error_type"] = validation["error_type"]
                result_data["database_error"] = validation.get("database_error")
                return result_data
            
            # Transfer warnings from validation
            result_data["warnings"].extend(validation.get("warnings", []))
            
            # Step 2: Apply safety limits
            query_to_execute = sql_query.strip().rstrip(';')
            query_upper = query_to_execute.upper()
            
            # Add LIMIT if not present and it's a data-returning query
            needs_limit = (validation["query_type"] in ["SELECT", "CTE_SELECT"] and 
                         "LIMIT" not in query_upper and 
                         "TOP " not in query_upper)  # Snowflake also supports TOP
            
            if needs_limit:
                query_to_execute = f"{query_to_execute} LIMIT {limit}"
                result_data["limit_applied"] = True
                result_data["warnings"].append(f"Safety LIMIT {limit} applied to prevent large result sets")
            
            # Step 3: Execute the query
            self._log_sql_query(query_to_execute)
            with self.get_connection() as conn:
                result = conn.execute(text(query_to_execute))

                try:
                    if result.returns_rows:
                        # Handle SELECT-type queries
                        result_data["columns"] = list(result.keys())

                        # Fetch all rows at once to avoid generator issues with complex queries
                        try:
                            raw_rows = result.fetchall()
                        except Exception as fetch_error:
                            logger.error(f"Error fetching results: {fetch_error}")
                            # Try to close result cursor explicitly
                            try:
                                result.close()
                            except Exception:
                                pass
                            raise

                        rows = []
                        for row in raw_rows:
                            row_dict = {}
                            for i, value in enumerate(row):
                                column_name = result_data["columns"][i]
                                # Serialize complex data types
                                if value is not None:
                                    if isinstance(value, decimal.Decimal):  # decimal/numeric types
                                        row_dict[column_name] = float(value)
                                    elif hasattr(value, 'isoformat'):  # datetime/date objects
                                        row_dict[column_name] = value.isoformat()
                                    elif isinstance(value, (bytes, bytearray)):  # binary data
                                        row_dict[column_name] = f"<binary:{len(value)} bytes>"
                                    elif isinstance(value, (dict, list)):  # JSON/array types
                                        row_dict[column_name] = value
                                    elif hasattr(value, '__dict__'):  # Complex objects
                                        row_dict[column_name] = str(value)
                                    else:
                                        row_dict[column_name] = value
                                else:
                                    row_dict[column_name] = None
                            rows.append(row_dict)

                        result_data["data"] = rows
                        result_data["row_count"] = len(rows)

                        if result_data["row_count"] == limit and result_data["limit_applied"]:
                            result_data["warnings"].append(f"Result set may be truncated at {limit} rows")

                    else:
                        # Handle non-returning queries (EXPLAIN, etc.)
                        result_data["row_count"] = getattr(result, 'rowcount', 0)
                finally:
                    # Ensure result cursor is closed
                    try:
                        result.close()
                    except Exception:
                        pass

                end_time = time.time()
                result_data["execution_time_ms"] = round((end_time - start_time) * 1000, 2)
                result_data["success"] = True

                logger.info(f"SQL query executed: {result_data['row_count']} rows in {result_data['execution_time_ms']}ms")
                
        except (SQLAlchemyError, ProgrammingError) as e:
            result_data["error"] = str(e)
            result_data["error_type"] = "execution_error"
            logger.error(f"SQL execution failed: {e}")
        except Exception as e:
            result_data["error"] = f"Unexpected execution error: {str(e)}"
            result_data["error_type"] = "internal_error"
            logger.error(f"Unexpected SQL execution error: {e}")
            
        return result_data
    
    def _execute_dremio_query(self, sql_query: str, limit: int = 1000) -> Dict[str, Any]:
        """Execute SQL query using Dremio REST API."""
        import asyncio
        import time
        from .dremio_client import create_dremio_client
        
        result_data = {
            "success": False,
            "data": [],
            "columns": [],
            "row_count": 0,
            "execution_time_ms": None,
            "error": None,
            "error_type": None,
            "warnings": [],
            "limit_applied": False
        }
        
        try:
            start_time = time.time()
            
            # Get Dremio connection parameters
            dremio_params = self._dremio_rest_connection
            
            async def run_dremio_query():
                # Handle both PAT and legacy authentication
                if dremio_params.get('uri') and dremio_params.get('pat'):
                    # PAT-based authentication
                    async with await create_dremio_client(
                        uri=dremio_params['uri'],
                        pat=dremio_params['pat']
                    ) as client:
                        return await client.execute_query(sql_query, limit)
                else:
                    # Legacy authentication
                    async with await create_dremio_client(
                        host=dremio_params["host"],
                        port=dremio_params["port"], 
                        username=dremio_params["username"],
                        password=dremio_params["password"],
                        ssl=dremio_params["ssl"]
                    ) as client:
                        return await client.execute_query(sql_query, limit)
            
            # Handle async execution in sync context (same pattern as other methods)
            try:
                loop = asyncio.get_running_loop()
                # Run in thread if already in async context (MCP server)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, run_dremio_query())
                    query_result = future.result(timeout=QUERY_TIMEOUT)  # Use configured query timeout
            except RuntimeError:
                # No event loop, create one
                query_result = asyncio.run(run_dremio_query())
            
            execution_time = (time.time() - start_time) * 1000
            result_data["execution_time_ms"] = execution_time
            
            if query_result.get("success"):
                result_data["success"] = True
                result_data["data"] = query_result.get("data", [])
                result_data["columns"] = query_result.get("columns", [])
                result_data["row_count"] = query_result.get("row_count", 0)
                
                # Check if limit was applied
                total_rows = query_result.get("total_rows", result_data["row_count"])
                if total_rows > limit:
                    result_data["limit_applied"] = True
                    result_data["warnings"].append(f"Results limited to {limit} rows (total: {total_rows})")
                
                logger.info(f"Dremio query executed successfully: {result_data['row_count']} rows in {execution_time:.2f}ms")
            else:
                result_data["error"] = query_result.get("error", "Unknown Dremio error")
                result_data["error_type"] = query_result.get("error_type", "dremio_error")
                logger.error(f"Dremio query failed: {result_data['error']}")
            
        except Exception as e:
            result_data["error"] = f"Dremio execution error: {str(e)}"
            result_data["error_type"] = "dremio_connection_error"
            logger.error(f"Dremio query execution failed: {e}")
        
        return result_data
    
    def _sample_dremio_table(self, table_name: str, schema_name: Optional[str] = None, 
                            limit: int = DEFAULT_SAMPLE_LIMIT) -> List[Dict[str, Any]]:
        """Sample data from Dremio table via REST API."""
        import asyncio
        from .dremio_client import create_dremio_client
        
        # Validate and normalize limit
        if not isinstance(limit, int) or limit < MIN_SAMPLE_LIMIT:
            limit = DEFAULT_SAMPLE_LIMIT
        elif limit > MAX_SAMPLE_LIMIT:
            limit = MAX_SAMPLE_LIMIT
            logger.warning(f"Sample limit capped at {MAX_SAMPLE_LIMIT}")
        
        async def fetch_sample():
            """Fetch sample data from Dremio table."""
            try:
                conn = self._dremio_rest_connection
                if conn.get('uri') and conn.get('pat'):
                    # PAT-based authentication
                    client = await create_dremio_client(uri=conn['uri'], pat=conn['pat'])
                else:
                    # Legacy authentication
                    client = await create_dremio_client(
                        host=conn['host'],
                        port=conn['port'],
                        username=conn['username'],
                        password=conn['password'],
                        ssl=conn.get('ssl', False)
                    )
                
                async with client:
                    # Build the query
                    if schema_name:
                        # Use dot notation for Dremio paths
                        full_table_name = f"{self._quote_dremio_identifier(schema_name)}.{self._quote_dremio_identifier(table_name)}"
                    else:
                        full_table_name = self._quote_dremio_identifier(table_name)
                    
                    query = f'SELECT * FROM {full_table_name} LIMIT {limit}'
                    
                    # Execute query
                    result = await client.execute_query(query, limit)
                    
                    if result.get('success'):
                        return result.get('data', [])
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        logger.error(f"Failed to sample Dremio table {table_name}: {error_msg}")
                        raise RuntimeError(f"Failed to sample Dremio table: {error_msg}")
                        
            except Exception as e:
                logger.error(f"Error sampling Dremio table {table_name}: {e}")
                raise RuntimeError(f"Error sampling Dremio table: {str(e)}")
        
        # Handle async execution in sync context
        try:
            loop = asyncio.get_running_loop()
            # Run in thread if already in async context (MCP server)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, fetch_sample())
                return future.result(timeout=CONNECTION_TIMEOUT)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(fetch_sample())
    
    def _validate_dremio_syntax(self, query: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SQL syntax using Dremio REST API."""
        import asyncio
        from .dremio_client import create_dremio_client
        
        async def validate_query():
            """Validate query using Dremio EXPLAIN."""
            try:
                dremio_params = self._dremio_rest_connection
                
                # Create client with appropriate authentication
                if dremio_params.get('uri') and dremio_params.get('pat'):
                    # PAT-based authentication
                    async with await create_dremio_client(
                        uri=dremio_params['uri'],
                        pat=dremio_params['pat']
                    ) as client:
                        explain_sql = f"EXPLAIN PLAN FOR {query}"
                        result = await client.execute_query(explain_sql)
                        return result
                else:
                    # Legacy authentication
                    async with await create_dremio_client(
                        host=dremio_params["host"],
                        port=dremio_params["port"], 
                        username=dremio_params["username"],
                        password=dremio_params["password"],
                        ssl=dremio_params["ssl"]
                    ) as client:
                        explain_sql = f"EXPLAIN PLAN FOR {query}"
                        result = await client.execute_query(explain_sql)
                        return result
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        try:
            # Handle async execution in sync context
            try:
                loop = asyncio.get_running_loop()
                # Run in thread if already in async context (MCP server)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, validate_query())
                    explain_result = future.result(timeout=QUERY_TIMEOUT)
            except RuntimeError:
                # No event loop, create one
                explain_result = asyncio.run(validate_query())
            
            if explain_result.get("success"):
                validation_result["is_valid"] = True
                return validation_result
            else:
                # Handle Dremio syntax error
                error_msg = explain_result.get("error", "Unknown Dremio validation error")
                validation_result["database_error"] = error_msg
                validation_result["error"] = f"Dremio syntax error: {error_msg}"
                validation_result["error_type"] = "syntax_error"
                
                # Add suggestions based on common Dremio errors
                if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                    validation_result["suggestions"].append("Object not found - check table/view names and ensure proper qualification")
                elif "syntax error" in error_msg.lower():
                    validation_result["suggestions"].append("SQL syntax error - review query structure and keywords")
                elif "validation error" in error_msg.lower():
                    validation_result["suggestions"].append("Query validation failed - check column references and data types")
                
                return validation_result
                
        except Exception as e:
            # If validation fails, just allow the query through with a warning
            validation_result["is_valid"] = True
            validation_result["warnings"].append(f"Could not validate syntax via Dremio: {str(e)}")
            return validation_result
    
    def has_engine(self) -> bool:
        """Check if database connection exists (engine for SQL databases, REST client for Dremio)."""
        # For Dremio REST API, check if connection info is available
        if self._dremio_rest_connection:
            return True
        # For traditional SQL databases, check engine
        return self.engine is not None
    
    def restore_connection_if_needed(self) -> bool:
        """Attempt to restore connection if engine is missing but params are available."""
        if not self.has_engine() and self._last_connection_params:
            logger.info("restore_connection_if_needed: Attempting to restore connection")
            try:
                self._reconnect()
                return True
            except Exception as e:
                logger.error(f"restore_connection_if_needed: Failed to restore connection: {e}")
                return False
        return self.has_engine()
    
    def force_reconnect(self) -> bool:
        """Force a reconnection even if engine exists (for troubleshooting)."""
        if not self._last_connection_params:
            logger.error("force_reconnect: No connection parameters available")
            return False
        
        logger.info("force_reconnect: Forcing database reconnection")
        try:
            # Clear current connection
            if self.engine:
                self.engine.dispose()
                self.engine = None
                logger.debug("force_reconnect: Disposed existing engine")
            
            # Reconnect
            self._reconnect()
            logger.info("force_reconnect: Reconnection successful")
            return True
        except Exception as e:
            logger.error(f"force_reconnect: Failed to reconnect: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if database is currently connected and healthy."""
        if self._dremio_rest_connection:
            return self._test_dremio_connection()
        return self.has_engine() and self._test_connection()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status information."""
        if self._dremio_rest_connection:
            return {
                "connected": self._test_dremio_connection(),
                "connection_info": self.connection_info.copy(),
                "last_params_available": self._last_connection_params is not None
            }

        if not self.engine:
            return {
                "connected": False,
                "connection_info": None,
                "last_params_available": self._last_connection_params is not None
            }
        
        is_healthy = self._test_connection()
        return {
            "connected": is_healthy,
            "connection_info": self.connection_info.copy(),
            "last_params_available": self._last_connection_params is not None,
            "engine_pool_size": self.engine.pool.size() if hasattr(self.engine, 'pool') else None,
            "engine_checked_out": self.engine.pool.checkedout() if hasattr(self.engine, 'pool') else None
        }
    
    def disconnect(self):
        """Close the database connection and clear stored parameters."""
        # Shutdown thread pool to prevent leaked threads
        if hasattr(self, '_thread_pool') and self._thread_pool:
            try:
                self._thread_pool.shutdown(wait=False)
                logger.debug("Thread pool shut down")
            except Exception as e:
                logger.warning(f"Error shutting down thread pool: {e}")
            self._thread_pool = None

        # Clear metadata cache to prevent stale data in reused instances
        if hasattr(self, '_metadata_cache'):
            self._metadata_cache.clear()

        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.metadata = None
            self.connection_info = {}
            self._last_connection_params = None
            logger.info("Database connection closed and parameters cleared")

"""Comprehensive tests for the enhanced database ontology MCP server."""

import json
import pytest
import unittest
from unittest.mock import patch, MagicMock, Mock, PropertyMock, AsyncMock
from concurrent.futures import Future

import src.main as main_module
from src.database_manager import DatabaseManager, TableInfo, ColumnInfo
from src.ontology_generator import OntologyGenerator
from src.config import ConfigManager
from src.constants import SUPPORTED_DB_TYPES


def create_mock_context(session_id: str = "test-session-123"):
    """Create a mock MCP Context for testing."""
    mock_ctx = Mock()
    mock_ctx.request_context = Mock()
    mock_ctx.request_context.session = Mock()
    mock_ctx.request_context.session.id = session_id
    # Mock async methods
    mock_ctx.info = AsyncMock()
    mock_ctx.warning = AsyncMock()
    mock_ctx.error = AsyncMock()
    return mock_ctx


class TestMCPTools(unittest.TestCase):
    """Comprehensive test suite for MCP tool functions with enhanced coverage."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample table info for users table
        self.sample_users_table = TableInfo(
            name="users",
            schema="public",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False,
                    comment="User unique identifier"
                ),
                ColumnInfo(
                    name="name",
                    data_type="VARCHAR(255)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="User full name"
                ),
                ColumnInfo(
                    name="email",
                    data_type="VARCHAR(255)",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="User email address"
                )
            ],
            primary_keys=["id"],
            foreign_keys=[],
            comment="User accounts table",
            row_count=150
        )

        # Sample table info for orders table with foreign key
        self.sample_orders_table = TableInfo(
            name="orders",
            schema="public",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="user_id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=True,
                    foreign_key_table="users",
                    foreign_key_column="id"
                ),
                ColumnInfo(
                    name="total_amount",
                    data_type="DECIMAL(10,2)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["id"],
            foreign_keys=[
                {
                    "column": "user_id",
                    "referenced_table": "users",
                    "referenced_column": "id"
                }
            ],
            row_count=500
        )

        # Mock context
        self.mock_ctx = create_mock_context()


@pytest.mark.asyncio
class TestMCPToolsAsync:
    """Async test suite for MCP tool functions with session isolation."""

    @pytest.fixture
    def sample_users_table(self):
        """Sample users table info."""
        return TableInfo(
            name="users",
            schema="public",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False,
                    comment="User unique identifier"
                ),
                ColumnInfo(
                    name="name",
                    data_type="VARCHAR(255)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="User full name"
                ),
                ColumnInfo(
                    name="email",
                    data_type="VARCHAR(255)",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="User email address"
                )
            ],
            primary_keys=["id"],
            foreign_keys=[],
            comment="User accounts table",
            row_count=150
        )

    @pytest.fixture
    def sample_orders_table(self):
        """Sample orders table info with foreign key."""
        return TableInfo(
            name="orders",
            schema="public",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="user_id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=True,
                    foreign_key_table="users",
                    foreign_key_column="id"
                ),
                ColumnInfo(
                    name="total_amount",
                    data_type="DECIMAL(10,2)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["id"],
            foreign_keys=[
                {
                    "column": "user_id",
                    "referenced_table": "users",
                    "referenced_column": "id"
                }
            ],
            row_count=500
        )

    @pytest.fixture
    def mock_ctx(self):
        """Create mock MCP Context."""
        return create_mock_context()

    @pytest.fixture
    def mock_session_data(self):
        """Create mock SessionData."""
        session = Mock()
        session.db_manager = None
        session.schema_file = None
        session.ontology_file = None
        session.r2rml_file = None
        session.obqc_validator = None
        session.loaded_ontology = None
        session.loaded_ontology_path = None
        # Mock cache methods - return None to simulate no cache
        session.get_cached_schema = Mock(return_value=None)
        session.cache_schema_analysis = Mock()
        session.clear_schema_cache = Mock()
        session.get_last_analyzed_schema = Mock(return_value=None)
        return session

    async def test_connect_database_postgresql_success(self, mock_ctx, mock_session_data):
        """Test successful PostgreSQL connection with session isolation."""
        mock_db_manager = Mock()
        mock_db_manager.connect_postgresql.return_value = True
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager), \
             patch.dict('os.environ', {
                 'POSTGRES_HOST': 'localhost',
                 'POSTGRES_PORT': '5432',
                 'POSTGRES_DATABASE': 'testdb',
                 'POSTGRES_USERNAME': 'testuser',
                 'POSTGRES_PASSWORD': 'testpass'
             }):

            result = await main_module.connect_database.fn(
                mock_ctx,
                db_type="postgresql"
            )

        assert "Successfully connected" in result
        assert "postgresql" in result
        assert "testdb" in result
        mock_db_manager.connect_postgresql.assert_called_once_with(
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass"
        )

    async def test_connect_database_postgresql_failure(self, mock_ctx, mock_session_data):
        """Test PostgreSQL connection failure with proper error handling."""
        mock_db_manager = Mock()
        mock_db_manager.connect_postgresql.return_value = False
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager), \
             patch.dict('os.environ', {
                 'POSTGRES_HOST': 'localhost',
                 'POSTGRES_PORT': '5432',
                 'POSTGRES_DATABASE': 'testdb',
                 'POSTGRES_USERNAME': 'testuser',
                 'POSTGRES_PASSWORD': 'wrongpass'
             }):

            result = await main_module.connect_database.fn(
                mock_ctx,
                db_type="postgresql"
            )

        # Should return error JSON
        error_data = json.loads(result)
        assert error_data["error_type"] == "connection_error"
        assert "Failed to connect" in error_data["error"]

    async def test_connect_database_snowflake_success(self, mock_ctx, mock_session_data):
        """Test successful Snowflake connection."""
        mock_db_manager = Mock()
        mock_db_manager.connect_snowflake.return_value = True
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager), \
             patch.dict('os.environ', {
                 'SNOWFLAKE_ACCOUNT': 'test-account',
                 'SNOWFLAKE_USERNAME': 'testuser',
                 'SNOWFLAKE_PASSWORD': 'testpass',
                 'SNOWFLAKE_WAREHOUSE': 'COMPUTE_WH',
                 'SNOWFLAKE_DATABASE': 'TESTDB',
                 'SNOWFLAKE_SCHEMA': 'PUBLIC'
             }):

            result = await main_module.connect_database.fn(
                mock_ctx,
                db_type="snowflake"
            )

        assert "Successfully connected" in result
        assert "snowflake" in result

    async def test_connect_database_unsupported_type(self, mock_ctx):
        """Test connection with unsupported database type returns proper validation error."""
        result = await main_module.connect_database.fn(mock_ctx, db_type="mysql")

        error_data = json.loads(result)
        assert error_data["error_type"] == "validation_error"
        assert "Invalid database type" in error_data["error"]
        assert "mysql" in error_data["error"]

    async def test_connect_database_missing_parameters(self, mock_ctx, mock_session_data):
        """Test connection with missing required environment variables."""
        # Use os.environ.get patching to simulate missing env vars
        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('os.getenv') as mock_getenv:
            # Only return value for POSTGRES_HOST, return None for others
            def getenv_side_effect(key, default=None):
                env_map = {
                    'POSTGRES_HOST': 'localhost',
                    'POSTGRES_PORT': None,
                    'POSTGRES_DATABASE': None,
                    'POSTGRES_USERNAME': None,
                    'POSTGRES_PASSWORD': None,
                }
                return env_map.get(key, default)
            mock_getenv.side_effect = getenv_side_effect

            result = await main_module.connect_database.fn(
                mock_ctx,
                db_type="postgresql"
            )

        error_data = json.loads(result)
        assert error_data["error_type"] == "validation_error"
        assert "Missing required environment variables" in error_data["error"]

    async def test_connect_database_exception(self, mock_ctx, mock_session_data):
        """Test connection with exception raises."""
        mock_db_manager = Mock()
        mock_db_manager.connect_postgresql.side_effect = Exception("Connection error")
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager), \
             patch.dict('os.environ', {
                 'POSTGRES_HOST': 'localhost',
                 'POSTGRES_PORT': '5432',
                 'POSTGRES_DATABASE': 'testdb',
                 'POSTGRES_USERNAME': 'testuser',
                 'POSTGRES_PASSWORD': 'testpass'
             }):

            # The function raises exception (no internal error handling)
            with pytest.raises(Exception, match="Connection error"):
                await main_module.connect_database.fn(
                    mock_ctx,
                    db_type="postgresql"
                )

    async def test_list_schemas_success(self, mock_ctx, mock_session_data):
        """Test successful schema listing with session isolation."""
        mock_db_manager = Mock()
        mock_db_manager.get_schemas.return_value = ["public", "private", "analytics"]
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager):

            result = await main_module.list_schemas.fn(mock_ctx)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "public" in result
        assert "private" in result
        assert "analytics" in result

    async def test_list_schemas_no_connection(self, mock_ctx, mock_session_data):
        """Test schema listing without connection raises exception."""
        mock_db_manager = Mock()
        mock_db_manager.get_schemas.side_effect = RuntimeError("No database connection")
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager):

            # The function raises exception (no internal error handling)
            with pytest.raises(RuntimeError, match="No database connection"):
                await main_module.list_schemas.fn(mock_ctx)

    async def test_analyze_schema_success(self, mock_ctx, mock_session_data, sample_users_table, sample_orders_table):
        """Test successful schema analysis with session isolation."""
        mock_db_manager = Mock()
        mock_db_manager.get_tables.return_value = ["users", "orders"]
        mock_db_manager.analyze_table.side_effect = [
            sample_users_table,
            sample_orders_table
        ]
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager), \
             patch('src.main.get_session_id', return_value="test-session"), \
             patch('src.main.get_session_safe_filename', return_value="test_schema.json"), \
             patch('builtins.open', MagicMock()), \
             patch('json.dump'):

            result = await main_module.analyze_schema.fn(mock_ctx, "public")

        assert isinstance(result, dict)
        assert result["schema"] == "public"
        assert result["table_count"] == 2
        assert len(result["tables"]) == 2

        # Check users table structure
        users_table = next(t for t in result["tables"] if t["name"] == "users")
        assert len(users_table["columns"]) == 3
        assert users_table["primary_keys"] == ["id"]
        assert users_table["row_count"] == 150

        # Check orders table with foreign key
        orders_table = next(t for t in result["tables"] if t["name"] == "orders")
        assert len(orders_table["foreign_keys"]) == 1
        assert orders_table["foreign_keys"][0]["referenced_table"] == "users"

    async def test_analyze_schema_no_connection(self, mock_ctx, mock_session_data):
        """Test schema analysis without connection raises exception."""
        mock_db_manager = Mock()
        mock_db_manager.get_tables.side_effect = RuntimeError("No database connection")
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager):

            # The function raises exception (no internal error handling)
            with pytest.raises(RuntimeError, match="No database connection"):
                await main_module.analyze_schema.fn(mock_ctx, "public")

    async def test_generate_ontology_success(self, mock_ctx, mock_session_data, sample_users_table):
        """Test successful ontology generation with session isolation."""
        mock_db_manager = Mock()
        mock_db_manager.get_tables.return_value = ["users"]
        mock_db_manager.analyze_table.return_value = sample_users_table
        mock_session_data.db_manager = mock_db_manager
        mock_session_data.ontology_file = None

        # Mock the ontology generator
        mock_generator = Mock()
        mock_generator.generate_from_schema.return_value = "@prefix ns: <http://example.com/ontology/> ."
        mock_generator.serialize_ontology.return_value = "@prefix ns: <http://example.com/ontology/> ."

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager), \
             patch('src.main.get_session_safe_filename', return_value="test_ontology.ttl"), \
             patch('src.main.OntologyGenerator', return_value=mock_generator), \
             patch('builtins.open', MagicMock()):

            result = await main_module.generate_ontology.fn(
                mock_ctx,
                schema_name="public",
                base_uri="http://example.com/ontology/"
            )

        assert isinstance(result, str)
        assert "@prefix ns:" in result

    async def test_generate_ontology_no_tables(self, mock_ctx, mock_session_data):
        """Test ontology generation with no tables."""
        mock_db_manager = Mock()
        mock_db_manager.get_tables.return_value = []
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager):

            result = await main_module.generate_ontology.fn(
                mock_ctx,
                schema_name="public"
            )

        # Result is JSON error response
        error_data = json.loads(result)
        assert error_data["error_type"] == "data_error"
        assert "No tables found" in error_data["error"]

    async def test_sample_table_data_success(self, mock_ctx, mock_session_data):
        """Test successful table data sampling with session isolation."""
        mock_db_manager = Mock()
        sample_data = [
            {"id": 1, "name": "John Doe", "email": "john@example.com"},
            {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
        ]
        mock_db_manager.sample_table_data.return_value = sample_data
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager):

            result = await main_module.sample_table_data.fn(mock_ctx, "users", "public", 10)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "John Doe"
        assert result[1]["email"] == "jane@example.com"

        # Verify the call was made with correct parameters
        mock_db_manager.sample_table_data.assert_called_once_with("users", "public", 10)

    async def test_sample_table_data_invalid_table_name(self, mock_ctx, mock_session_data):
        """Test table data sampling with invalid table name returns error."""
        with patch('src.main.get_session_data', return_value=mock_session_data):
            # Empty table name returns error list
            result = await main_module.sample_table_data.fn(mock_ctx, "", "public", 10)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]
        assert "Table name is required" in result[0]["error"]

    async def test_sample_table_data_database_error(self, mock_ctx, mock_session_data):
        """Test table data sampling with database error raises exception."""
        mock_db_manager = Mock()
        mock_db_manager.sample_table_data.side_effect = ValueError("Invalid table name format")
        mock_session_data.db_manager = mock_db_manager

        with patch('src.main.get_session_data', return_value=mock_session_data), \
             patch('src.main.get_session_db_manager', return_value=mock_db_manager):

            # The function raises exception (no internal error handling)
            with pytest.raises(ValueError, match="Invalid table name format"):
                await main_module.sample_table_data.fn(mock_ctx, "invalid-table", "public", 10)


class TestOntologyGenerator(unittest.TestCase):
    """Enhanced test suite for ontology generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from src.ontology_generator import OntologyGenerator
        self.generator = OntologyGenerator()

        self.sample_table = TableInfo(
            name="test_table",
            schema="public",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False,
                    comment="Primary key"
                ),
                ColumnInfo(
                    name="name",
                    data_type="VARCHAR(100)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="Entity name"
                ),
                ColumnInfo(
                    name="created_at",
                    data_type="TIMESTAMP",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="Creation timestamp"
                )
            ],
            primary_keys=["id"],
            foreign_keys=[],
            comment="Test table for ontology generation",
            row_count=10
        )

    def test_generate_ontology_structure(self):
        """Test that generated ontology has proper structure."""
        result = self.generator.generate_from_schema([self.sample_table])

        # The ontology uses ns: prefix instead of ex:
        self.assertIn("@prefix ns:", result)
        self.assertIn("@prefix owl:", result)
        self.assertIn("@prefix rdfs:", result)
        self.assertIn("@prefix xsd:", result)
        self.assertIn("owl:Class", result)
        self.assertIn("owl:DatatypeProperty", result)

    def test_xsd_type_mapping(self):
        """Test XSD type mapping for various SQL types."""
        from rdflib.namespace import XSD

        # Test integer mapping (returns tuple of (type, override_info))
        xsd_type, _ = self.generator._map_sql_to_xsd("INTEGER")
        self.assertEqual(xsd_type, XSD.integer)
        xsd_type, _ = self.generator._map_sql_to_xsd("BIGINT")
        self.assertEqual(xsd_type, XSD.integer)

        # Test string mapping
        xsd_type, _ = self.generator._map_sql_to_xsd("VARCHAR(255)")
        self.assertEqual(xsd_type, XSD.string)
        xsd_type, _ = self.generator._map_sql_to_xsd("TEXT")
        self.assertEqual(xsd_type, XSD.string)

        # Test boolean mapping
        xsd_type, _ = self.generator._map_sql_to_xsd("BOOLEAN")
        self.assertEqual(xsd_type, XSD.boolean)

        # Test datetime mapping
        xsd_type, _ = self.generator._map_sql_to_xsd("TIMESTAMP")
        self.assertEqual(xsd_type, XSD.dateTime)
        xsd_type, _ = self.generator._map_sql_to_xsd("DATE")
        self.assertEqual(xsd_type, XSD.date)

        # Test numeric mapping
        xsd_type, _ = self.generator._map_sql_to_xsd("DECIMAL(10,2)")
        self.assertEqual(xsd_type, XSD.decimal)
        xsd_type, _ = self.generator._map_sql_to_xsd("FLOAT")
        self.assertEqual(xsd_type, XSD.float)

    def test_clean_name_function(self):
        """Test name cleaning for URI generation with edge cases."""
        self.assertEqual(self.generator._clean_name("test_table"), "test_table")
        self.assertEqual(self.generator._clean_name("test-table"), "test_table")
        self.assertEqual(self.generator._clean_name("test table"), "test_table")
        self.assertEqual(self.generator._clean_name("123test"), "_123test")
        self.assertEqual(self.generator._clean_name(""), "unnamed")
        self.assertEqual(self.generator._clean_name("test@table#123"), "test_table_123")
        self.assertEqual(self.generator._clean_name("test.table"), "test_table")

    def test_enrichment_data_generation(self):
        """Test generation of enrichment data structure."""
        sample_data = {
            "test_table": [
                {"id": 1, "name": "Test Item 1", "created_at": "2023-01-01T00:00:00"},
                {"id": 2, "name": "Test Item 2", "created_at": "2023-01-02T00:00:00"}
            ]
        }

        enrichment_data = self.generator.get_enrichment_data([self.sample_table], sample_data)

        self.assertIn("schema_data", enrichment_data)
        self.assertIn("instructions", enrichment_data)

        schema_data = enrichment_data["schema_data"]
        self.assertEqual(len(schema_data), 1)

        table_data = schema_data[0]
        self.assertEqual(table_data["table_name"], "test_table")
        self.assertEqual(len(table_data["columns"]), 3)
        self.assertIn("sample_data", table_data)
        self.assertEqual(len(table_data["sample_data"]), 2)

    def test_apply_enrichment(self):
        """Test application of enrichment suggestions."""
        # Generate base ontology
        ontology_ttl = self.generator.generate_from_schema([self.sample_table])

        # Define enrichment suggestions
        enrichment_suggestions = {
            "classes": [
                {
                    "original_name": "test_table",
                    "suggested_name": "TestEntity",
                    "description": "A test entity for demonstration purposes"
                }
            ],
            "properties": [
                {
                    "table_name": "test_table",
                    "original_name": "name",
                    "suggested_name": "entityName",
                    "description": "The name of the test entity"
                }
            ],
            "relationships": []
        }

        # Apply enrichment
        self.generator.apply_enrichment(enrichment_suggestions)

        # Serialize and check result
        enriched_ontology = self.generator.serialize_ontology()
        self.assertIn("TestEntity", enriched_ontology)
        self.assertIn("entityName", enriched_ontology)


class TestConfigManager(unittest.TestCase):
    """Test suite for configuration management."""

    def setUp(self):
        """Set up test fixtures."""
        from src.config import ConfigManager
        self.config_manager = ConfigManager()

    def test_validate_db_config_postgresql(self):
        """Test PostgreSQL configuration validation."""
        with patch.dict('os.environ', {
            'POSTGRES_HOST': 'localhost',
            'POSTGRES_PORT': '5432',
            'POSTGRES_DATABASE': 'testdb',
            'POSTGRES_USERNAME': 'testuser',
            'POSTGRES_PASSWORD': 'testpass'
        }):
            validation = self.config_manager.validate_db_config('postgresql')
            self.assertTrue(validation['valid'])
            self.assertEqual(len(validation['missing_params']), 0)

    def test_validate_db_config_missing_params(self):
        """Test configuration validation with missing parameters."""
        with patch.dict('os.environ', {}, clear=True):
            validation = self.config_manager.validate_db_config('postgresql')
            self.assertFalse(validation['valid'])
            self.assertGreater(len(validation['missing_params']), 0)

    def test_validate_db_config_invalid_type(self):
        """Test configuration validation with invalid database type."""
        with self.assertRaises(ValueError):
            self.config_manager.validate_db_config('invalid_db_type')


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions."""

    def test_sanitize_for_logging(self):
        """Test sanitization of sensitive data for logging."""
        from src.utils import sanitize_for_logging

        sensitive_data = {
            'host': 'localhost',
            'port': 5432,
            'password': 'secret123',
            'api_key': 'sk-1234567890',
            'config': {
                'username': 'user',
                'secret': 'hidden'
            }
        }

        sanitized = sanitize_for_logging(sensitive_data)

        self.assertEqual(sanitized['host'], 'localhost')
        self.assertEqual(sanitized['port'], 5432)
        self.assertEqual(sanitized['password'], '***REDACTED***')
        self.assertEqual(sanitized['api_key'], '***REDACTED***')
        self.assertEqual(sanitized['config']['username'], 'user')
        self.assertEqual(sanitized['config']['secret'], '***REDACTED***')

    def test_validate_uri(self):
        """Test URI validation function."""
        from src.utils import validate_uri

        self.assertTrue(validate_uri('https://example.com/'))
        self.assertTrue(validate_uri('http://localhost:8080/'))
        self.assertTrue(validate_uri('https://api.example.com/v1/ontology/'))

        self.assertFalse(validate_uri(''))
        self.assertFalse(validate_uri('not-a-uri'))
        self.assertFalse(validate_uri('ftp://example.com/'))

    def test_format_bytes(self):
        """Test bytes formatting function."""
        from src.utils import format_bytes

        self.assertEqual(format_bytes(0), "0 B")
        self.assertEqual(format_bytes(1024), "1.0 KB")
        self.assertEqual(format_bytes(1024 * 1024), "1.0 MB")
        self.assertEqual(format_bytes(1536), "1.5 KB")


if __name__ == '__main__':
    # Use pytest for better test discovery and reporting if available
    try:
        import pytest
        pytest.main([__file__, '-v'])
    except ImportError:
        unittest.main(verbosity=2)

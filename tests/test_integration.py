"""Integration tests with real database instances."""

import unittest
import os
import time
from unittest.mock import patch

from src.database_manager import DatabaseManager, TableInfo
from src.ontology_generator import OntologyGenerator

# Only run integration tests if explicitly requested
INTEGRATION_TESTS = (
    os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
)


try:
    from testcontainers.postgres import PostgresContainer  # type: ignore
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    PostgresContainer = None


@unittest.skipIf(
    not INTEGRATION_TESTS or not TESTCONTAINERS_AVAILABLE,
    "Integration tests disabled or testcontainers not available"
)
class TestRealDatabaseIntegration(unittest.TestCase):
    """Integration tests with real PostgreSQL database."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test database container."""
        cls.postgres = PostgresContainer("postgres:15")
        cls.postgres.start()  # type: ignore

        # Wait for database to be ready
        time.sleep(2)

        # Create test data
        cls._create_test_schema()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up test database container."""
        if hasattr(cls, 'postgres'):
            cls.postgres.stop()  # type: ignore

    @classmethod
    def _create_test_schema(cls) -> None:
        """Create test schema with sample data."""
        import psycopg2  # type: ignore

        conn = psycopg2.connect(
            host=cls.postgres.get_container_host_ip(),  # type: ignore
            port=cls.postgres.get_exposed_port(5432),  # type: ignore
            database=cls.postgres.dbname,  # type: ignore
            user=cls.postgres.username,  # type: ignore
            password=cls.postgres.password  # type: ignore
        )

        with conn.cursor() as cur:
            # Create test tables with relationships
            cur.execute("""
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                );
            """)

            cur.execute("""
                CREATE TABLE orders (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    order_date DATE DEFAULT CURRENT_DATE,
                    total_amount DECIMAL(10,2) NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending'
                );
            """)

            cur.execute("""
                CREATE TABLE order_items (
                    id SERIAL PRIMARY KEY,
                    order_id INTEGER REFERENCES orders(id),
                    product_name VARCHAR(100) NOT NULL,
                    quantity INTEGER NOT NULL,
                    price DECIMAL(10,2) NOT NULL
                );
            """)

            # Insert sample data
            cur.execute("""
                INSERT INTO users (username, email) VALUES
                ('john_doe', 'john@example.com'),
                ('jane_smith', 'jane@example.com'),
                ('bob_wilson', 'bob@example.com');
            """)

            cur.execute("""
                INSERT INTO orders (user_id, total_amount, status) VALUES
                (1, 150.00, 'completed'),
                (1, 75.50, 'pending'),
                (2, 200.00, 'completed'),
                (3, 50.25, 'shipped');
            """)

            cur.execute("""
                INSERT INTO order_items (order_id, product_name, quantity, price)
                VALUES
                (1, 'Laptop', 1, 150.00),
                (2, 'Mouse', 2, 25.00),
                (2, 'Keyboard', 1, 50.50),
                (3, 'Monitor', 1, 200.00),
                (4, 'USB Cable', 3, 16.75);
            """)

        conn.commit()
        conn.close()

    def setUp(self) -> None:
        """Set up test database manager."""
        self.db_manager = DatabaseManager()

        # Connect to test database
        success = self.db_manager.connect_postgresql(
            host=self.postgres.get_container_host_ip(),  # type: ignore
            port=self.postgres.get_exposed_port(5432),  # type: ignore
            database=self.postgres.dbname,  # type: ignore
            username=self.postgres.username,  # type: ignore
            password=self.postgres.password  # type: ignore
        )
        self.assertTrue(success, "Failed to connect to test database")

    def test_database_connection(self) -> None:
        """Test basic database connection functionality."""
        # Test connection health
        self.assertTrue(self.db_manager._test_connection())

        # Test connection info
        info = self.db_manager.connection_info
        self.assertEqual(info["type"], "postgresql")
        self.assertEqual(info["database"], self.postgres.dbname)  # type: ignore

    def test_schema_discovery(self) -> None:
        """Test schema and table discovery."""
        # Get schemas
        schemas = self.db_manager.get_schemas()
        self.assertIn("public", schemas)

        # Get tables
        tables = self.db_manager.get_tables("public")
        expected_tables = {"users", "orders", "order_items"}
        actual_tables = set(tables)

        self.assertTrue(
            expected_tables.issubset(actual_tables),
            f"Expected tables {expected_tables} not found in {actual_tables}"
        )

    def test_table_analysis(self) -> None:
        """Test detailed table analysis."""
        # Analyze users table
        users_info = self.db_manager.analyze_table("users", "public")
        self.assertIsInstance(users_info, TableInfo)
        self.assertEqual(users_info.name, "users")
        self.assertEqual(users_info.schema, "public")

        # Check columns
        column_names = [col.name for col in users_info.columns]
        expected_columns = {
            "id", "username", "email", "created_at", "is_active"
        }
        self.assertTrue(expected_columns.issubset(set(column_names)))

        # Check primary key
        self.assertIn("id", users_info.primary_keys)

        # Check column properties
        id_column = next(col for col in users_info.columns if col.name == "id")
        self.assertTrue(id_column.is_primary_key)
        self.assertFalse(id_column.is_nullable)

    def test_foreign_key_relationships(self) -> None:
        """Test foreign key relationship detection."""
        # Analyze orders table (has foreign key to users)
        orders_info = self.db_manager.analyze_table("orders", "public")
        self.assertIsInstance(orders_info, TableInfo)

        # Check foreign keys
        self.assertTrue(len(orders_info.foreign_keys) > 0)

        # Find user_id foreign key
        user_fk = next((fk for fk in orders_info.foreign_keys
                       if fk["column"] == "user_id"), None)
        self.assertIsNotNone(user_fk)
        self.assertEqual(user_fk["referenced_table"], "users")
        self.assertEqual(user_fk["referenced_column"], "id")

    def test_table_relationships(self) -> None:
        """Test comprehensive table relationship mapping."""
        relationships = self.db_manager.get_table_relationships("public")

        # Check that orders table has relationship to users
        self.assertIn("orders", relationships)
        orders_rels = relationships["orders"]

        user_rel = next((rel for rel in orders_rels
                        if rel["referenced_table"] == "users"), None)
        self.assertIsNotNone(user_rel)

    def test_data_sampling(self) -> None:
        """Test table data sampling."""
        # Sample users table
        sample_data = self.db_manager.sample_table_data("users", "public", limit=5)

        self.assertIsInstance(sample_data, list)
        self.assertTrue(len(sample_data) > 0)
        self.assertLessEqual(len(sample_data), 5)

        # Check data structure
        first_row = sample_data[0]
        self.assertIn("id", first_row)
        self.assertIn("username", first_row)
        self.assertIn("email", first_row)

    def test_concurrent_schema_analysis(self) -> None:
        """Test concurrent schema analysis performance."""
        start_time = time.time()

        # Use concurrent analysis
        tables_info = self.db_manager.analyze_schema_concurrent("public", max_workers=3)

        concurrent_time = time.time() - start_time

        # Verify results
        self.assertTrue(len(tables_info) >= 3)  # At least users, orders, order_items

        # Verify all tables were analyzed
        table_names = {table.name for table in tables_info}
        expected_tables = {"users", "orders", "order_items"}
        self.assertTrue(expected_tables.issubset(table_names))

        # Performance should be reasonable (concurrent should be faster than sequential)
        self.assertLess(concurrent_time, 10.0, "Concurrent analysis took too long")

    def test_sql_query_validation_and_execution(self) -> None:
        """Test SQL query validation and safe execution."""
        # Test safe query
        safe_query = "SELECT COUNT(*) FROM users WHERE is_active = true"
        validation = self.db_manager.validate_sql_syntax(safe_query)

        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["risk_level"], "low")

        # Execute safe query
        result = self.db_manager.execute_sql_query(safe_query, limit=10)
        self.assertTrue(result["success"])
        self.assertIn("data", result)

        # Test dangerous query (should be blocked)
        dangerous_query = "SELECT * FROM users; DROP TABLE users; --"
        validation = self.db_manager.validate_sql_syntax(dangerous_query)

        self.assertFalse(validation["is_valid"])
        self.assertEqual(validation["error_type"], "security_error")
        self.assertIn("critical", validation["risk_level"])

    def test_caching_performance(self) -> None:
        """Test that caching improves performance."""
        # Clear any existing cache
        self.db_manager._metadata_cache.clear()

        # First call (should cache)
        start_time = time.time()
        tables1 = self.db_manager.get_tables("public")
        first_call_time = time.time() - start_time

        # Second call (should use cache)
        start_time = time.time()
        tables2 = self.db_manager.get_tables("public")
        second_call_time = time.time() - start_time

        # Results should be identical
        self.assertEqual(tables1, tables2)

        # Second call should be faster (cached)
        self.assertLess(second_call_time, first_call_time * 0.5,
                       "Cached call should be significantly faster")


@unittest.skipIf(not INTEGRATION_TESTS, "Integration tests disabled")
class TestOntologyGenerationIntegration(unittest.TestCase):
    """Integration tests for ontology generation with real data."""

    def setUp(self) -> None:
        """Set up test data."""
        from src.database_manager import ColumnInfo

        # Create sample table info that would come from real database
        self.sample_table_info = [
            TableInfo(
                name="users",
                schema="public",
                columns=[
                    ColumnInfo("id", "INTEGER", False, True, False),
                    ColumnInfo("username", "VARCHAR(50)", False, False, False),
                    ColumnInfo("email", "VARCHAR(100)", True, False, False),
                    ColumnInfo("created_at", "TIMESTAMP", True, False, False),
                ],
                primary_keys=["id"],
                foreign_keys=[],
                comment="User accounts table",
                row_count=100
            ),
            TableInfo(
                name="orders",
                schema="public",
                columns=[
                    ColumnInfo("id", "INTEGER", False, True, False),
                    ColumnInfo("user_id", "INTEGER", False, False, True,
                             foreign_key_table="users", foreign_key_column="id"),
                    ColumnInfo("total_amount", "DECIMAL(10,2)", False, False, False),
                ],
                primary_keys=["id"],
                foreign_keys=[{
                    "column": "user_id",
                    "referenced_table": "users",
                    "referenced_column": "id"
                }],
                row_count=500
            )
        ]

    def test_ontology_generation_structure(self) -> None:
        """Test that generated ontology has proper RDF structure."""
        generator = OntologyGenerator("http://test.example.com/ontology/")
        ontology_ttl = generator.generate_from_schema(self.sample_table_info)

        # Basic structure checks
        self.assertIn("@prefix", ontology_ttl)
        self.assertIn("owl:Ontology", ontology_ttl)
        self.assertIn("owl:Class", ontology_ttl)
        self.assertIn("owl:DatatypeProperty", ontology_ttl)
        self.assertIn("owl:ObjectProperty", ontology_ttl)

        # Check that tables are represented as classes
        self.assertIn("users", ontology_ttl)
        self.assertIn("orders", ontology_ttl)

        # Check that relationships are represented
        self.assertIn("user_id", ontology_ttl)

    def test_ontology_validation(self) -> None:
        """Test that generated ontology is valid RDF."""
        from rdflib import Graph

        generator = OntologyGenerator("http://test.example.com/ontology/")
        ontology_ttl = generator.generate_from_schema(self.sample_table_info)

        # Parse with rdflib to validate syntax
        graph = Graph()
        try:
            graph.parse(data=ontology_ttl, format="turtle")
            triples_count = len(graph)
            self.assertGreater(triples_count, 10, "Ontology should have substantial content")
        except Exception as e:  # pylint: disable=broad-except
            self.fail(f"Generated ontology is not valid RDF: {e}")


@unittest.skipIf(not INTEGRATION_TESTS, "Integration tests disabled")
class TestPerformanceIntegration(unittest.TestCase):
    """Performance tests with realistic data sizes."""

    def test_large_schema_analysis_performance(self) -> None:
        """Test performance with larger schemas (simulated)."""
        # This would test with a larger test database
        # For now, we'll test the concurrent processing logic

        db_manager = DatabaseManager()

        # Mock a large number of tables
        with patch.object(db_manager, 'get_tables') as mock_get_tables, \
             patch.object(db_manager, 'analyze_table') as mock_analyze:

            # Simulate 50 tables
            table_names = [f"table_{i}" for i in range(50)]
            mock_get_tables.return_value = table_names

            # Mock analyze_table to return quickly
            mock_analyze.return_value = TableInfo(
                name="mock_table", schema="public", columns=[],
                primary_keys=[], foreign_keys=[]
            )

            start_time = time.time()
            results = db_manager.analyze_schema_concurrent("public", max_workers=10)
            duration = time.time() - start_time

            # Should complete in reasonable time even with 50 tables
            self.assertLess(duration, 5.0, "Large schema analysis took too long")
            self.assertEqual(len(results), 50)


if __name__ == '__main__':
    if INTEGRATION_TESTS:
        if not TESTCONTAINERS_AVAILABLE:
            print("WARNING: testcontainers not available, skipping integration tests")
            print("Install with: pip install testcontainers")
        else:
            print("Running integration tests with real databases...")
            unittest.main(verbosity=2)
    else:
        print("Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.")
        print("Note: Integration tests require Docker and testcontainers library.")

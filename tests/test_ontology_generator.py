"""Tests for the OntologyGenerator class with comprehensive coverage."""

import pytest
import unittest
from unittest.mock import Mock, patch
from rdflib.namespace import XSD, RDF, RDFS, OWL

from src.ontology_generator import OntologyGenerator
from src.database_manager import TableInfo, ColumnInfo


class TestOntologyGenerator(unittest.TestCase):
    """Comprehensive test suite for OntologyGenerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = OntologyGenerator("http://test.com/ontology/")
        
        # Create sample table with various column types
        self.sample_table = TableInfo(
            name="users",
            schema="public",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False,
                    comment="User ID"
                ),
                ColumnInfo(
                    name="username",
                    data_type="VARCHAR(50)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="Username"
                ),
                ColumnInfo(
                    name="email",
                    data_type="VARCHAR(255)",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="Email address"
                ),
                ColumnInfo(
                    name="created_at",
                    data_type="TIMESTAMP",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="Creation timestamp"
                ),
                ColumnInfo(
                    name="is_active",
                    data_type="BOOLEAN",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="Active status"
                ),
                ColumnInfo(
                    name="balance",
                    data_type="DECIMAL(10,2)",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False,
                    comment="Account balance"
                )
            ],
            primary_keys=["id"],
            foreign_keys=[],
            comment="Users table",
            row_count=1000
        )
        
        # Create table with foreign key
        self.orders_table = TableInfo(
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
                    name="total",
                    data_type="DECIMAL(12,2)",
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
            row_count=5000
        )
    
    def test_initialization(self):
        """Test OntologyGenerator initialization."""
        self.assertEqual(str(self.generator.base_uri), "http://test.com/ontology/")
        self.assertIsNotNone(self.generator.graph)
        
        # Check namespace bindings
        namespaces = dict(self.generator.graph.namespaces())
        self.assertIn('ns', namespaces)
        self.assertIn('rdf', namespaces)
        self.assertIn('rdfs', namespaces)
        self.assertIn('owl', namespaces)
        self.assertIn('xsd', namespaces)
    
    def test_initialization_with_default_uri(self):
        """Test initialization with default URI."""
        default_generator = OntologyGenerator()
        self.assertTrue(str(default_generator.base_uri).startswith("http://example.com/ontology/"))
    
    def test_clean_name_comprehensive(self):
        """Test name cleaning with comprehensive cases."""
        test_cases = [
            ("simple_name", "simple_name"),
            ("name-with-hyphens", "name_with_hyphens"),
            ("name with spaces", "name_with_spaces"),
            ("name.with.dots", "name_with_dots"),
            ("name@with#special$chars", "name_with_special_chars"),
            ("123_starts_with_number", "_123_starts_with_number"),
            ("", "unnamed"),
            ("a", "a"),
            ("_valid_underscore", "_valid_underscore"),
            ("MixedCaseTable", "MixedCaseTable"),
            ("table123", "table123"),
            ("order-items", "order_items"),
            ("user_account_info", "user_account_info")
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = self.generator._clean_name(input_name)
                self.assertEqual(result, expected, 
                    f"Expected {expected}, got {result} for input {input_name}")
    
    def test_map_sql_to_xsd_comprehensive(self):
        """Test SQL to XSD type mapping with comprehensive coverage."""
        test_cases = [
            # Integer types
            ("INTEGER", XSD.integer),
            ("INT", XSD.integer),
            ("BIGINT", XSD.integer),
            ("SMALLINT", XSD.integer),
            ("SERIAL", XSD.integer),
            ("TINYINT", XSD.byte),
            
            # String types
            ("VARCHAR(255)", XSD.string),
            ("CHAR(10)", XSD.string),
            ("TEXT", XSD.string),
            ("CLOB", XSD.string),
            ("BLOB", XSD.string),
            ("STRING", XSD.string),
            
            # Temporal types
            ("TIMESTAMP", XSD.dateTime),
            ("DATETIME", XSD.dateTime),
            ("DATE", XSD.date),
            ("TIME", XSD.time),
            
            # Numeric types
            ("FLOAT", XSD.float),
            ("REAL", XSD.float),
            ("DOUBLE", XSD.double),
            ("DOUBLE PRECISION", XSD.double),
            ("DECIMAL(10,2)", XSD.decimal),
            ("NUMERIC(8,2)", XSD.decimal),
            ("MONEY", XSD.decimal),
            
            # Boolean types
            ("BOOLEAN", XSD.boolean),
            ("BOOL", XSD.boolean),
            ("BIT", XSD.boolean),
            
            # Binary types
            ("BINARY", XSD.base64Binary),
            ("VARBINARY", XSD.base64Binary),
            ("BYTEA", XSD.base64Binary),
            
            # UUID types
            ("UUID", XSD.string),
            
            # JSON types
            ("JSON", XSD.string),
            ("JSONB", XSD.string),
            
            # Unknown type should default to string
            ("UNKNOWN_TYPE", XSD.string)
        ]
        
        for sql_type, expected_xsd in test_cases:
            with self.subTest(sql_type=sql_type):
                result, override = self.generator._map_sql_to_xsd(sql_type)
                self.assertEqual(result, expected_xsd,
                    f"Expected {expected_xsd}, got {result} for SQL type {sql_type}")
    
    def test_generate_basic_ontology_structure(self):
        """Test generation of basic ontology structure."""
        result = self.generator.generate_from_schema([self.sample_table])
        
        # Check that result is a valid Turtle string
        self.assertIsInstance(result, str)
        
        # Check for required prefixes
        self.assertIn("@prefix", result)
        self.assertIn("ns:", result)
        self.assertIn("owl:", result)
        self.assertIn("rdfs:", result)
        self.assertIn("xsd:", result)
        
        # Check for ontology declaration
        self.assertIn("owl:Ontology", result)
        
        # Check for class declaration
        self.assertIn("owl:Class", result)
        
        # Check for property declarations
        self.assertIn("owl:DatatypeProperty", result)
        
        # Check for table name in ontology
        self.assertIn("users", result)
    
    def test_generate_ontology_with_relationships(self):
        """Test ontology generation with foreign key relationships."""
        tables = [self.sample_table, self.orders_table]
        result = self.generator.generate_from_schema(tables)
        
        # Check for object properties (relationships)
        self.assertIn("owl:ObjectProperty", result)
        
        # Check for both tables
        self.assertIn("users", result)
        self.assertIn("orders", result)
        
        # Check for relationship properties
        self.assertIn("has", result)  # Should contain relationship naming
    
    def test_add_table_with_various_constraints(self):
        """Test adding table with various column constraints."""
        # Create table with different constraint types
        constrained_table = TableInfo(
            name="products",
            schema="shop",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="name",
                    data_type="VARCHAR(100)",
                    is_nullable=False,  # Required field
                    is_primary_key=False,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="description",
                    data_type="TEXT",
                    is_nullable=True,  # Optional field
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["id"],
            foreign_keys=[]
        )
        
        result = self.generator.generate_from_schema([constrained_table])

        # Check that tables are top-level classes (not subclasses of restrictions)
        self.assertNotIn("owl:Restriction", result)
        self.assertNotIn("rdfs:subClassOf", result)

        # Check that constraint metadata is properly annotated
        self.assertIn("db:isPrimaryKey true", result)  # Primary key annotation
        self.assertIn("db:isNullable false", result)  # Required field annotation
        self.assertIn("db:isNullable true", result)  # Optional field annotation
    
    def test_get_enrichment_data(self):
        """Test generation of enrichment data structure."""
        sample_data = {
            "users": [
                {"id": 1, "username": "john_doe", "email": "john@example.com"},
                {"id": 2, "username": "jane_smith", "email": "jane@example.com"}
            ]
        }
        
        enrichment_data = self.generator.get_enrichment_data(
            [self.sample_table], 
            sample_data
        )
        
        # Check structure
        self.assertIn("schema_data", enrichment_data)
        self.assertIn("instructions", enrichment_data)
        
        schema_data = enrichment_data["schema_data"]
        self.assertEqual(len(schema_data), 1)
        
        table_data = schema_data[0]
        self.assertEqual(table_data["table_name"], "users")
        self.assertEqual(len(table_data["columns"]), 6)  # All columns from sample_table
        self.assertIn("sample_data", table_data)
        self.assertEqual(len(table_data["sample_data"]), 2)
        
        # Check instructions
        instructions = enrichment_data["instructions"]
        self.assertIn("task", instructions)
        self.assertIn("expected_format", instructions)
        self.assertIn("guidelines", instructions)
    
    def test_get_enrichment_data_no_samples(self):
        """Test enrichment data generation without sample data."""
        enrichment_data = self.generator.get_enrichment_data([self.sample_table], {})
        
        schema_data = enrichment_data["schema_data"]
        table_data = schema_data[0]
        self.assertNotIn("sample_data", table_data)
    
    def test_get_enrichment_data_limited_samples(self):
        """Test enrichment data with limited sample data."""
        large_sample_data = {
            "users": [{"id": i, "username": f"user_{i}"} for i in range(10)]
        }
        
        enrichment_data = self.generator.get_enrichment_data(
            [self.sample_table], 
            large_sample_data
        )
        
        # Should be limited to first 3 samples
        table_data = enrichment_data["schema_data"][0]
        self.assertEqual(len(table_data["sample_data"]), 3)
    
    def test_apply_enrichment_classes(self):
        """Test applying class enrichment suggestions."""
        # Generate base ontology first
        self.generator.generate_from_schema([self.sample_table])
        
        enrichment_suggestions = {
            "classes": [
                {
                    "original_name": "users",
                    "suggested_name": "UserAccount",
                    "description": "User account entity representing registered users"
                }
            ],
            "properties": [],
            "relationships": []
        }
        
        self.generator.apply_enrichment(enrichment_suggestions)
        result = self.generator.serialize_ontology()
        
        # Check that enrichment was applied
        self.assertIn("UserAccount", result)
        self.assertIn("User account entity", result)
    
    def test_apply_enrichment_properties(self):
        """Test applying property enrichment suggestions."""
        # Generate base ontology first
        self.generator.generate_from_schema([self.sample_table])
        
        enrichment_suggestions = {
            "classes": [],
            "properties": [
                {
                    "table_name": "users",
                    "original_name": "username",
                    "suggested_name": "accountName",
                    "description": "Unique account identifier for user login"
                }
            ],
            "relationships": []
        }
        
        self.generator.apply_enrichment(enrichment_suggestions)
        result = self.generator.serialize_ontology()
        
        # Check that property enrichment was applied
        self.assertIn("accountName", result)
        self.assertIn("Unique account identifier", result)
    
    def test_apply_enrichment_relationships(self):
        """Test applying relationship enrichment suggestions."""
        # Generate ontology with relationships first
        self.generator.generate_from_schema([self.sample_table, self.orders_table])
        
        enrichment_suggestions = {
            "classes": [],
            "properties": [],
            "relationships": [
                {
                    "from_table": "orders",
                    "to_table": "users",
                    "suggested_name": "belongsToUser",
                    "description": "Associates an order with the user who placed it"
                }
            ]
        }
        
        self.generator.apply_enrichment(enrichment_suggestions)
        result = self.generator.serialize_ontology()
        
        # Check that relationship enrichment was applied
        self.assertIn("belongsToUser", result)
        self.assertIn("Associates an order", result)
    
    def test_serialize_ontology(self):
        """Test ontology serialization."""
        # Generate a basic ontology
        self.generator.generate_from_schema([self.sample_table])
        
        # Test serialization
        result = self.generator.serialize_ontology()
        
        self.assertIsInstance(result, str)
        self.assertIn("@prefix", result)
        
        # Should be valid Turtle format
        from rdflib import Graph
        test_graph = Graph()
        try:
            test_graph.parse(data=result, format="turtle")
            # If parsing succeeds, the serialization is valid
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Generated ontology is not valid Turtle: {e}")
    
    def test_enrich_with_llm_placeholder(self):
        """Test LLM enrichment placeholder functionality."""
        sample_data = {"users": [{"id": 1, "username": "test"}]}
        
        result = self.generator.enrich_with_llm([self.sample_table], sample_data)
        
        # Should return basic ontology (LLM enrichment is handled by MCP tools)
        self.assertIsInstance(result, str)
        self.assertIn("@prefix", result)
    
    def test_multiple_tables_ontology(self):
        """Test ontology generation with multiple related tables."""
        # Create a third table for more complex relationships
        products_table = TableInfo(
            name="products",
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
                    name="name",
                    data_type="VARCHAR(100)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["id"],
            foreign_keys=[]
        )
        
        tables = [self.sample_table, self.orders_table, products_table]
        result = self.generator.generate_from_schema(tables)
        
        # Check all tables are represented
        self.assertIn("users", result)
        self.assertIn("orders", result)
        self.assertIn("products", result)
        
        # Should have multiple classes
        class_count = result.count("owl:Class")
        self.assertGreaterEqual(class_count, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
"""Tests for OBQC (Ontology-Based Query Check) validator."""

import pytest
import unittest
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

from src.obqc_validator import (
    OBQCValidator,
    OBQCResult,
    OBQCIssue,
    OBQCIssueType,
    OBQCSeverity,
    OntologySchema,
    TableSchema,
    ColumnSchema,
    RelationshipInfo,
)


def create_sample_ontology_graph() -> tuple[Graph, str]:
    """Create a sample ontology graph for testing.

    Returns:
        Tuple of (Graph, base_uri)
    """
    base_uri = "http://test.com/ontology/"
    g = Graph()
    ns = Namespace(base_uri)
    db = Namespace(f"{base_uri}db/")

    g.bind("ns", ns)
    g.bind("db", db)

    # Add users table
    users = ns["users"]
    g.add((users, RDF.type, OWL.Class))
    g.add((users, db.tableName, Literal("users")))
    g.add((users, db.schemaName, Literal("public")))
    g.add((users, db.primaryKey, Literal("id")))
    g.add((users, db.rowCount, Literal(1000)))

    # Add users.id column (PK, integer)
    users_id = ns["users_id"]
    g.add((users_id, RDF.type, OWL.DatatypeProperty))
    g.add((users_id, db.columnName, Literal("id")))
    g.add((users_id, db.tableName, Literal("users")))
    g.add((users_id, db.sqlDataType, Literal("INTEGER")))
    g.add((users_id, db.isPrimaryKey, Literal("true")))
    g.add((users_id, db.isForeignKey, Literal("false")))
    g.add((users_id, db.isNullable, Literal("false")))
    g.add((users_id, RDFS.domain, users))
    g.add((users_id, RDFS.range, XSD.integer))

    # Add users.name column (string)
    users_name = ns["users_name"]
    g.add((users_name, RDF.type, OWL.DatatypeProperty))
    g.add((users_name, db.columnName, Literal("name")))
    g.add((users_name, db.tableName, Literal("users")))
    g.add((users_name, db.sqlDataType, Literal("VARCHAR(100)")))
    g.add((users_name, db.isPrimaryKey, Literal("false")))
    g.add((users_name, db.isForeignKey, Literal("false")))
    g.add((users_name, db.isNullable, Literal("true")))
    g.add((users_name, RDFS.domain, users))
    g.add((users_name, RDFS.range, XSD.string))

    # Add users.email column (string)
    users_email = ns["users_email"]
    g.add((users_email, RDF.type, OWL.DatatypeProperty))
    g.add((users_email, db.columnName, Literal("email")))
    g.add((users_email, db.tableName, Literal("users")))
    g.add((users_email, db.sqlDataType, Literal("VARCHAR(255)")))
    g.add((users_email, db.isPrimaryKey, Literal("false")))
    g.add((users_email, db.isForeignKey, Literal("false")))
    g.add((users_email, db.isNullable, Literal("true")))
    g.add((users_email, RDFS.domain, users))
    g.add((users_email, RDFS.range, XSD.string))

    # Add orders table
    orders = ns["orders"]
    g.add((orders, RDF.type, OWL.Class))
    g.add((orders, db.tableName, Literal("orders")))
    g.add((orders, db.schemaName, Literal("public")))
    g.add((orders, db.primaryKey, Literal("id")))

    # Add orders.id column
    orders_id = ns["orders_id"]
    g.add((orders_id, RDF.type, OWL.DatatypeProperty))
    g.add((orders_id, db.columnName, Literal("id")))
    g.add((orders_id, db.tableName, Literal("orders")))
    g.add((orders_id, db.sqlDataType, Literal("INTEGER")))
    g.add((orders_id, db.isPrimaryKey, Literal("true")))
    g.add((orders_id, RDFS.domain, orders))
    g.add((orders_id, RDFS.range, XSD.integer))

    # Add orders.user_id column (FK)
    orders_user_id = ns["orders_user_id"]
    g.add((orders_user_id, RDF.type, OWL.DatatypeProperty))
    g.add((orders_user_id, db.columnName, Literal("user_id")))
    g.add((orders_user_id, db.tableName, Literal("orders")))
    g.add((orders_user_id, db.sqlDataType, Literal("INTEGER")))
    g.add((orders_user_id, db.isPrimaryKey, Literal("false")))
    g.add((orders_user_id, db.isForeignKey, Literal("true")))
    g.add((orders_user_id, db.isNullable, Literal("false")))
    g.add((orders_user_id, RDFS.domain, orders))
    g.add((orders_user_id, RDFS.range, XSD.integer))

    # Add orders.total column (decimal)
    orders_total = ns["orders_total"]
    g.add((orders_total, RDF.type, OWL.DatatypeProperty))
    g.add((orders_total, db.columnName, Literal("total")))
    g.add((orders_total, db.tableName, Literal("orders")))
    g.add((orders_total, db.sqlDataType, Literal("DECIMAL(10,2)")))
    g.add((orders_total, RDFS.domain, orders))
    g.add((orders_total, RDFS.range, XSD.decimal))

    # Add orders.order_date column (date)
    orders_date = ns["orders_order_date"]
    g.add((orders_date, RDF.type, OWL.DatatypeProperty))
    g.add((orders_date, db.columnName, Literal("order_date")))
    g.add((orders_date, db.tableName, Literal("orders")))
    g.add((orders_date, db.sqlDataType, Literal("DATE")))
    g.add((orders_date, RDFS.domain, orders))
    g.add((orders_date, RDFS.range, XSD.date))

    # Add relationship: orders -> users (many_to_one)
    rel = ns["orders_has_users"]
    g.add((rel, RDF.type, OWL.ObjectProperty))
    g.add((rel, RDFS.domain, orders))
    g.add((rel, RDFS.range, users))
    g.add((rel, db.foreignKeyColumn, Literal("user_id")))
    g.add((rel, db.referencedTable, Literal("users")))
    g.add((rel, db.referencedColumn, Literal("id")))
    g.add((rel, db.relationshipType, Literal("many_to_one")))
    g.add((rel, db.sqlJoinCondition, Literal("orders.user_id = users.id")))

    # Add order_items table for fan-trap testing
    order_items = ns["order_items"]
    g.add((order_items, RDF.type, OWL.Class))
    g.add((order_items, db.tableName, Literal("order_items")))
    g.add((order_items, db.schemaName, Literal("public")))

    # Add order_items.order_id column (FK)
    items_order_id = ns["order_items_order_id"]
    g.add((items_order_id, RDF.type, OWL.DatatypeProperty))
    g.add((items_order_id, db.columnName, Literal("order_id")))
    g.add((items_order_id, db.tableName, Literal("order_items")))
    g.add((items_order_id, db.isForeignKey, Literal("true")))
    g.add((items_order_id, RDFS.domain, order_items))
    g.add((items_order_id, RDFS.range, XSD.integer))

    # Add order_items.quantity column
    items_qty = ns["order_items_quantity"]
    g.add((items_qty, RDF.type, OWL.DatatypeProperty))
    g.add((items_qty, db.columnName, Literal("quantity")))
    g.add((items_qty, db.tableName, Literal("order_items")))
    g.add((items_qty, RDFS.domain, order_items))
    g.add((items_qty, RDFS.range, XSD.integer))

    # Add relationship: order_items -> orders (many_to_one, inverse is one_to_many)
    rel2 = ns["order_items_has_orders"]
    g.add((rel2, RDF.type, OWL.ObjectProperty))
    g.add((rel2, RDFS.domain, order_items))
    g.add((rel2, RDFS.range, orders))
    g.add((rel2, db.foreignKeyColumn, Literal("order_id")))
    g.add((rel2, db.referencedTable, Literal("orders")))
    g.add((rel2, db.referencedColumn, Literal("id")))
    g.add((rel2, db.relationshipType, Literal("many_to_one")))
    g.add((rel2, db.sqlJoinCondition, Literal("order_items.order_id = orders.id")))

    # Add shipments table for fan-trap testing
    shipments = ns["shipments"]
    g.add((shipments, RDF.type, OWL.Class))
    g.add((shipments, db.tableName, Literal("shipments")))
    g.add((shipments, db.schemaName, Literal("public")))

    # Add shipments.order_id column (FK)
    ship_order_id = ns["shipments_order_id"]
    g.add((ship_order_id, RDF.type, OWL.DatatypeProperty))
    g.add((ship_order_id, db.columnName, Literal("order_id")))
    g.add((ship_order_id, db.tableName, Literal("shipments")))
    g.add((ship_order_id, db.isForeignKey, Literal("true")))
    g.add((ship_order_id, RDFS.domain, shipments))
    g.add((ship_order_id, RDFS.range, XSD.integer))

    # Add shipments.cost column
    ship_cost = ns["shipments_cost"]
    g.add((ship_cost, RDF.type, OWL.DatatypeProperty))
    g.add((ship_cost, db.columnName, Literal("cost")))
    g.add((ship_cost, db.tableName, Literal("shipments")))
    g.add((ship_cost, RDFS.domain, shipments))
    g.add((ship_cost, RDFS.range, XSD.decimal))

    # Add relationship: shipments -> orders (many_to_one)
    rel3 = ns["shipments_has_orders"]
    g.add((rel3, RDF.type, OWL.ObjectProperty))
    g.add((rel3, RDFS.domain, shipments))
    g.add((rel3, RDFS.range, orders))
    g.add((rel3, db.foreignKeyColumn, Literal("order_id")))
    g.add((rel3, db.referencedTable, Literal("orders")))
    g.add((rel3, db.referencedColumn, Literal("id")))
    g.add((rel3, db.relationshipType, Literal("many_to_one")))
    g.add((rel3, db.sqlJoinCondition, Literal("shipments.order_id = orders.id")))

    return g, base_uri


class TestOBQCValidator(unittest.TestCase):
    """Test suite for OBQC validator."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def test_valid_simple_select(self):
        """Test validation of a valid simple SELECT."""
        result = self.validator.validate("SELECT id, name FROM users")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.to_dict()["obqc_error_count"], 0)
        self.assertIn("users", result.parsed_tables)

    def test_valid_select_with_where(self):
        """Test validation of SELECT with WHERE clause."""
        result = self.validator.validate(
            "SELECT id, name FROM users WHERE id = 1"
        )
        self.assertTrue(result.is_valid)

    def test_table_not_found(self):
        """Test detection of non-existent table."""
        result = self.validator.validate("SELECT * FROM nonexistent_table")
        self.assertFalse(result.is_valid)
        issue_types = [i.issue_type for i in result.issues]
        self.assertIn(OBQCIssueType.TABLE_NOT_FOUND, issue_types)

    def test_column_not_found(self):
        """Test detection of non-existent column."""
        result = self.validator.validate(
            "SELECT users.nonexistent_column FROM users"
        )
        self.assertFalse(result.is_valid)
        issue_types = [i.issue_type for i in result.issues]
        self.assertIn(OBQCIssueType.COLUMN_NOT_FOUND, issue_types)

    def test_valid_join(self):
        """Test validation of join with correct FK relationship."""
        result = self.validator.validate(
            "SELECT users.name, orders.total "
            "FROM users JOIN orders ON users.id = orders.user_id"
        )
        self.assertTrue(result.is_valid)
        self.assertIn("users", result.parsed_tables)
        self.assertIn("orders", result.parsed_tables)

    def test_missing_join_condition_cartesian(self):
        """Test detection of Cartesian product (multiple tables without JOIN)."""
        result = self.validator.validate("SELECT * FROM users, orders")
        self.assertFalse(result.is_valid)
        issue_types = [i.issue_type for i in result.issues]
        self.assertIn(OBQCIssueType.MISSING_JOIN_CONDITION, issue_types)

    def test_aggregation_without_group_by(self):
        """Test detection of aggregation without GROUP BY."""
        result = self.validator.validate(
            "SELECT users.name, SUM(orders.total) "
            "FROM users JOIN orders ON users.id = orders.user_id"
        )
        self.assertFalse(result.is_valid)
        issue_types = [i.issue_type for i in result.issues]
        self.assertIn(OBQCIssueType.NON_AGGREGATED_COLUMN, issue_types)

    def test_valid_aggregation_with_group_by(self):
        """Test valid aggregation with GROUP BY."""
        result = self.validator.validate(
            "SELECT users.name, SUM(orders.total) "
            "FROM users JOIN orders ON users.id = orders.user_id "
            "GROUP BY users.name"
        )
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_aggregation)
        self.assertTrue(result.has_group_by)

    def test_type_mismatch_warning(self):
        """Test detection of type mismatch in comparison."""
        # Comparing integer id with string literal
        result = self.validator.validate(
            "SELECT * FROM users WHERE users.id = 'abc'"
        )
        # Should warn about type mismatch
        warning_issues = [
            i for i in result.issues if i.issue_type == OBQCIssueType.TYPE_MISMATCH
        ]
        self.assertTrue(len(warning_issues) > 0)

    def test_ambiguous_column_warning(self):
        """Test detection of ambiguous column reference."""
        # 'id' exists in both users and orders
        result = self.validator.validate(
            "SELECT id FROM users JOIN orders ON users.id = orders.user_id"
        )
        # Should warn about ambiguous column
        warning_issues = [
            i for i in result.issues if i.issue_type == OBQCIssueType.AMBIGUOUS_COLUMN
        ]
        self.assertTrue(len(warning_issues) > 0)

    def test_no_ontology_loaded(self):
        """Test validation when no ontology is loaded."""
        validator = OBQCValidator()  # No ontology loaded
        result = validator.validate("SELECT * FROM users")
        # Should return with warning, but not fail hard
        self.assertTrue(result.is_valid)  # No errors, just warning
        self.assertTrue(len(result.issues) > 0)

    def test_sql_parse_error(self):
        """Test handling of SQL syntax errors."""
        result = self.validator.validate("SELECT FROM")  # Invalid SQL
        self.assertFalse(result.is_valid)

    def test_cte_query(self):
        """Test validation of CTE (WITH clause) query."""
        result = self.validator.validate(
            "WITH user_orders AS ("
            "  SELECT users.id, users.name, SUM(orders.total) as total "
            "  FROM users JOIN orders ON users.id = orders.user_id "
            "  GROUP BY users.id, users.name"
            ") "
            "SELECT * FROM user_orders"
        )
        # CTE creates a derived table, so validation should work
        self.assertIn("users", result.parsed_tables)

    def test_result_serialization(self):
        """Test OBQCResult serialization to dict."""
        result = self.validator.validate("SELECT id FROM users")
        result_dict = result.to_dict()

        self.assertIn("obqc_valid", result_dict)
        self.assertIn("obqc_issues", result_dict)
        self.assertIn("parsed_tables", result_dict)
        self.assertIn("parsed_columns", result_dict)
        self.assertIn("has_aggregation", result_dict)
        self.assertIn("obqc_error_count", result_dict)
        self.assertIn("obqc_warning_count", result_dict)

    def test_dialect_support(self):
        """Test different SQL dialects."""
        # PostgreSQL
        result = self.validator.validate(
            "SELECT * FROM users", dialect="postgresql"
        )
        self.assertTrue(result.is_valid)

        # Snowflake
        result = self.validator.validate(
            "SELECT * FROM users", dialect="snowflake"
        )
        self.assertTrue(result.is_valid)

        # Dremio (uses trino dialect)
        result = self.validator.validate(
            "SELECT * FROM users", dialect="dremio"
        )
        self.assertTrue(result.is_valid)


class TestOBQCFanTrapDetection(unittest.TestCase):
    """Test suite specifically for fan-trap detection."""

    def setUp(self):
        """Set up test fixtures with fan-trap prone schema."""
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def test_fan_trap_detected(self):
        """Test detection of fan-trap pattern."""
        # Query that joins orders with both order_items and shipments
        # then aggregates - classic fan-trap pattern
        result = self.validator.validate(
            "SELECT orders.id, SUM(order_items.quantity), SUM(shipments.cost) "
            "FROM orders "
            "JOIN order_items ON orders.id = order_items.order_id "
            "JOIN shipments ON orders.id = shipments.order_id "
            "GROUP BY orders.id"
        )
        # Should detect fan-trap risk
        self.assertTrue(result.fan_trap_risk)
        fan_trap_issues = [
            i for i in result.issues
            if i.issue_type == OBQCIssueType.FAN_TRAP_DETECTED
        ]
        self.assertTrue(len(fan_trap_issues) > 0)

    def test_no_fan_trap_single_one_to_many(self):
        """Test that single 1:many join doesn't trigger fan-trap warning."""
        result = self.validator.validate(
            "SELECT users.name, SUM(orders.total) "
            "FROM users JOIN orders ON users.id = orders.user_id "
            "GROUP BY users.name"
        )
        # Single 1:many relationship should not be a fan-trap
        self.assertFalse(result.fan_trap_risk)

    def test_no_fan_trap_without_aggregation(self):
        """Test that multiple joins without aggregation don't trigger fan-trap."""
        result = self.validator.validate(
            "SELECT orders.id, order_items.quantity, shipments.cost "
            "FROM orders "
            "JOIN order_items ON orders.id = order_items.order_id "
            "JOIN shipments ON orders.id = shipments.order_id"
        )
        # Without aggregation, no fan-trap risk
        self.assertFalse(result.fan_trap_risk)


class TestOBQCIssue(unittest.TestCase):
    """Test suite for OBQCIssue data class."""

    def test_issue_creation(self):
        """Test creating an OBQC issue."""
        issue = OBQCIssue(
            issue_type=OBQCIssueType.TABLE_NOT_FOUND,
            severity=OBQCSeverity.ERROR,
            message="Table 'foo' not found",
            location="FROM clause",
            suggestion="Check table name spelling",
            related_entities=["foo"]
        )

        self.assertEqual(issue.issue_type, OBQCIssueType.TABLE_NOT_FOUND)
        self.assertEqual(issue.severity, OBQCSeverity.ERROR)
        self.assertEqual(issue.message, "Table 'foo' not found")
        self.assertEqual(issue.location, "FROM clause")
        self.assertEqual(issue.suggestion, "Check table name spelling")
        self.assertEqual(issue.related_entities, ["foo"])


class TestOntologySchemaExtraction(unittest.TestCase):
    """Test suite for ontology schema extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def test_tables_extracted(self):
        """Test that tables are correctly extracted from ontology."""
        schema = self.validator._schema_cache
        self.assertIn("users", schema.tables)
        self.assertIn("orders", schema.tables)
        self.assertIn("order_items", schema.tables)
        self.assertIn("shipments", schema.tables)

    def test_columns_extracted(self):
        """Test that columns are correctly extracted."""
        schema = self.validator._schema_cache
        users_table = schema.tables["users"]

        self.assertIn("id", users_table.columns)
        self.assertIn("name", users_table.columns)
        self.assertIn("email", users_table.columns)

        # Check column properties
        id_col = users_table.columns["id"]
        self.assertTrue(id_col.is_primary_key)
        self.assertEqual(id_col.xsd_type, XSD.integer)

    def test_relationships_extracted(self):
        """Test that relationships are correctly extracted."""
        schema = self.validator._schema_cache
        # Should have relationships for orders->users, order_items->orders, shipments->orders
        self.assertTrue(len(schema.relationships) >= 3)

        # Check that join conditions are captured
        found_orders_users = False
        for rel in schema.relationships.values():
            if rel.from_table == "orders" and rel.to_table == "users":
                found_orders_users = True
                self.assertEqual(rel.from_column, "user_id")
                self.assertEqual(rel.to_column, "id")
                self.assertIn("orders.user_id = users.id", rel.join_condition)

        self.assertTrue(found_orders_users)


class TestIncompatibleOntology(unittest.TestCase):
    """Test suite for ontologies without db: namespace annotations."""

    def test_ontology_without_db_annotations(self):
        """Test that ontology without db: annotations is detected as incompatible."""
        # Create a basic OWL ontology without db: namespace annotations
        g = Graph()
        ns = Namespace("http://example.org/")
        g.bind("ex", ns)

        # Add a class without db:tableName
        person = ns["Person"]
        g.add((person, RDF.type, OWL.Class))
        g.add((person, RDFS.label, Literal("Person")))

        # Add a property without db:columnName
        name_prop = ns["name"]
        g.add((name_prop, RDF.type, OWL.DatatypeProperty))
        g.add((name_prop, RDFS.domain, person))
        g.add((name_prop, RDFS.range, XSD.string))

        # Load into validator
        validator = OBQCValidator()
        validator.load_ontology(g, "http://example.org/")

        # Should be marked as incompatible
        self.assertFalse(validator.is_compatible)

        # Validation should skip with INFO message
        result = validator.validate("SELECT * FROM Person")
        self.assertTrue(result.is_valid)  # Not an error, just skipped
        self.assertFalse(result.ontology_compatible)
        self.assertTrue(len(result.issues) > 0)
        self.assertEqual(result.issues[0].severity, OBQCSeverity.INFO)

    def test_compatible_ontology_flag(self):
        """Test that compatible ontology is properly flagged."""
        graph, base_uri = create_sample_ontology_graph()
        validator = OBQCValidator()
        validator.load_ontology(graph, base_uri)

        # Should be marked as compatible
        self.assertTrue(validator.is_compatible)

        # Result should indicate compatibility
        result = validator.validate("SELECT id FROM users")
        self.assertTrue(result.ontology_compatible)

    def test_result_dict_includes_compatibility(self):
        """Test that result dict includes compatibility flag."""
        graph, base_uri = create_sample_ontology_graph()
        validator = OBQCValidator()
        validator.load_ontology(graph, base_uri)

        result = validator.validate("SELECT id FROM users")
        result_dict = result.to_dict()

        self.assertIn("obqc_ontology_compatible", result_dict)
        self.assertTrue(result_dict["obqc_ontology_compatible"])


if __name__ == "__main__":
    unittest.main()

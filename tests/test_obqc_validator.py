"""Tests for OBQC (Ontology-Based Query Check) validator."""

import json
import unittest

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

from src.obqc_validator import (
    OBQCIssue,
    OBQCIssueType,
    OBQCResult,
    OBQCSeverity,
    OBQCValidator,
)


def create_sample_ontology_graph() -> tuple[Graph, str]:
    """Create a sample ontology graph for testing.

    Returns:
        Tuple of (Graph, base_uri)
    """
    base_uri = "http://test.com/ontology/"
    g = Graph()
    ns = Namespace(base_uri)
    oba = Namespace("https://ralforion.com/ns/oba#")

    g.bind("ns", ns)
    g.bind("oba", oba)

    # Add users table
    users = ns["users"]
    g.add((users, RDF.type, OWL.Class))
    g.add((users, oba.tableName, Literal("users")))
    g.add((users, oba.schemaName, Literal("public")))
    g.add((users, oba.primaryKey, Literal("id")))
    g.add((users, oba.rowCount, Literal(1000)))

    # Add users.id column (PK, integer)
    users_id = ns["users_id"]
    g.add((users_id, RDF.type, OWL.DatatypeProperty))
    g.add((users_id, oba.columnName, Literal("id")))
    g.add((users_id, oba.tableName, Literal("users")))
    g.add((users_id, oba.sqlDataType, Literal("INTEGER")))
    g.add((users_id, oba.isPrimaryKey, Literal("true")))
    g.add((users_id, oba.isForeignKey, Literal("false")))
    g.add((users_id, oba.isNullable, Literal("false")))
    g.add((users_id, RDFS.domain, users))
    g.add((users_id, RDFS.range, XSD.integer))

    # Add users.name column (string)
    users_name = ns["users_name"]
    g.add((users_name, RDF.type, OWL.DatatypeProperty))
    g.add((users_name, oba.columnName, Literal("name")))
    g.add((users_name, oba.tableName, Literal("users")))
    g.add((users_name, oba.sqlDataType, Literal("VARCHAR(100)")))
    g.add((users_name, oba.isPrimaryKey, Literal("false")))
    g.add((users_name, oba.isForeignKey, Literal("false")))
    g.add((users_name, oba.isNullable, Literal("true")))
    g.add((users_name, RDFS.domain, users))
    g.add((users_name, RDFS.range, XSD.string))

    # Add users.email column (string)
    users_email = ns["users_email"]
    g.add((users_email, RDF.type, OWL.DatatypeProperty))
    g.add((users_email, oba.columnName, Literal("email")))
    g.add((users_email, oba.tableName, Literal("users")))
    g.add((users_email, oba.sqlDataType, Literal("VARCHAR(255)")))
    g.add((users_email, oba.isPrimaryKey, Literal("false")))
    g.add((users_email, oba.isForeignKey, Literal("false")))
    g.add((users_email, oba.isNullable, Literal("true")))
    g.add((users_email, RDFS.domain, users))
    g.add((users_email, RDFS.range, XSD.string))

    # Add orders table
    orders = ns["orders"]
    g.add((orders, RDF.type, OWL.Class))
    g.add((orders, oba.tableName, Literal("orders")))
    g.add((orders, oba.schemaName, Literal("public")))
    g.add((orders, oba.primaryKey, Literal("id")))

    # Add orders.id column
    orders_id = ns["orders_id"]
    g.add((orders_id, RDF.type, OWL.DatatypeProperty))
    g.add((orders_id, oba.columnName, Literal("id")))
    g.add((orders_id, oba.tableName, Literal("orders")))
    g.add((orders_id, oba.sqlDataType, Literal("INTEGER")))
    g.add((orders_id, oba.isPrimaryKey, Literal("true")))
    g.add((orders_id, RDFS.domain, orders))
    g.add((orders_id, RDFS.range, XSD.integer))

    # Add orders.user_id column (FK)
    orders_user_id = ns["orders_user_id"]
    g.add((orders_user_id, RDF.type, OWL.DatatypeProperty))
    g.add((orders_user_id, oba.columnName, Literal("user_id")))
    g.add((orders_user_id, oba.tableName, Literal("orders")))
    g.add((orders_user_id, oba.sqlDataType, Literal("INTEGER")))
    g.add((orders_user_id, oba.isPrimaryKey, Literal("false")))
    g.add((orders_user_id, oba.isForeignKey, Literal("true")))
    g.add((orders_user_id, oba.isNullable, Literal("false")))
    g.add((orders_user_id, RDFS.domain, orders))
    g.add((orders_user_id, RDFS.range, XSD.integer))

    # Add orders.total column (decimal)
    orders_total = ns["orders_total"]
    g.add((orders_total, RDF.type, OWL.DatatypeProperty))
    g.add((orders_total, oba.columnName, Literal("total")))
    g.add((orders_total, oba.tableName, Literal("orders")))
    g.add((orders_total, oba.sqlDataType, Literal("DECIMAL(10,2)")))
    g.add((orders_total, RDFS.domain, orders))
    g.add((orders_total, RDFS.range, XSD.decimal))

    # Add orders.order_date column (date)
    orders_date = ns["orders_order_date"]
    g.add((orders_date, RDF.type, OWL.DatatypeProperty))
    g.add((orders_date, oba.columnName, Literal("order_date")))
    g.add((orders_date, oba.tableName, Literal("orders")))
    g.add((orders_date, oba.sqlDataType, Literal("DATE")))
    g.add((orders_date, RDFS.domain, orders))
    g.add((orders_date, RDFS.range, XSD.date))

    # Add relationship: orders -> users (many_to_one)
    rel = ns["orders_has_users"]
    g.add((rel, RDF.type, OWL.ObjectProperty))
    g.add((rel, RDFS.domain, orders))
    g.add((rel, RDFS.range, users))
    g.add((rel, oba.foreignKeyColumn, Literal("user_id")))
    g.add((rel, oba.referencedTable, Literal("users")))
    g.add((rel, oba.referencedColumn, Literal("id")))
    g.add((rel, oba.relationshipType, Literal("many_to_one")))
    g.add((rel, oba.sqlJoinCondition, Literal("orders.user_id = users.id")))

    # Add order_items table for fan-trap testing
    order_items = ns["order_items"]
    g.add((order_items, RDF.type, OWL.Class))
    g.add((order_items, oba.tableName, Literal("order_items")))
    g.add((order_items, oba.schemaName, Literal("public")))

    # Add order_items.order_id column (FK)
    items_order_id = ns["order_items_order_id"]
    g.add((items_order_id, RDF.type, OWL.DatatypeProperty))
    g.add((items_order_id, oba.columnName, Literal("order_id")))
    g.add((items_order_id, oba.tableName, Literal("order_items")))
    g.add((items_order_id, oba.isForeignKey, Literal("true")))
    g.add((items_order_id, RDFS.domain, order_items))
    g.add((items_order_id, RDFS.range, XSD.integer))

    # Add order_items.quantity column
    items_qty = ns["order_items_quantity"]
    g.add((items_qty, RDF.type, OWL.DatatypeProperty))
    g.add((items_qty, oba.columnName, Literal("quantity")))
    g.add((items_qty, oba.tableName, Literal("order_items")))
    g.add((items_qty, RDFS.domain, order_items))
    g.add((items_qty, RDFS.range, XSD.integer))

    # Add relationship: order_items -> orders (many_to_one, inverse is one_to_many)
    rel2 = ns["order_items_has_orders"]
    g.add((rel2, RDF.type, OWL.ObjectProperty))
    g.add((rel2, RDFS.domain, order_items))
    g.add((rel2, RDFS.range, orders))
    g.add((rel2, oba.foreignKeyColumn, Literal("order_id")))
    g.add((rel2, oba.referencedTable, Literal("orders")))
    g.add((rel2, oba.referencedColumn, Literal("id")))
    g.add((rel2, oba.relationshipType, Literal("many_to_one")))
    g.add((rel2, oba.sqlJoinCondition, Literal("order_items.order_id = orders.id")))

    # Add shipments table for fan-trap testing
    shipments = ns["shipments"]
    g.add((shipments, RDF.type, OWL.Class))
    g.add((shipments, oba.tableName, Literal("shipments")))
    g.add((shipments, oba.schemaName, Literal("public")))

    # Add shipments.order_id column (FK)
    ship_order_id = ns["shipments_order_id"]
    g.add((ship_order_id, RDF.type, OWL.DatatypeProperty))
    g.add((ship_order_id, oba.columnName, Literal("order_id")))
    g.add((ship_order_id, oba.tableName, Literal("shipments")))
    g.add((ship_order_id, oba.isForeignKey, Literal("true")))
    g.add((ship_order_id, RDFS.domain, shipments))
    g.add((ship_order_id, RDFS.range, XSD.integer))

    # Add shipments.cost column
    ship_cost = ns["shipments_cost"]
    g.add((ship_cost, RDF.type, OWL.DatatypeProperty))
    g.add((ship_cost, oba.columnName, Literal("cost")))
    g.add((ship_cost, oba.tableName, Literal("shipments")))
    g.add((ship_cost, RDFS.domain, shipments))
    g.add((ship_cost, RDFS.range, XSD.decimal))

    # Add relationship: shipments -> orders (many_to_one)
    rel3 = ns["shipments_has_orders"]
    g.add((rel3, RDF.type, OWL.ObjectProperty))
    g.add((rel3, RDFS.domain, shipments))
    g.add((rel3, RDFS.range, orders))
    g.add((rel3, oba.foreignKeyColumn, Literal("order_id")))
    g.add((rel3, oba.referencedTable, Literal("orders")))
    g.add((rel3, oba.referencedColumn, Literal("id")))
    g.add((rel3, oba.relationshipType, Literal("many_to_one")))
    g.add((rel3, oba.sqlJoinCondition, Literal("shipments.order_id = orders.id")))

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
        result = self.validator.validate("SELECT id, name FROM users WHERE id = 1")
        self.assertTrue(result.is_valid)

    def test_table_not_found(self):
        """Test detection of non-existent table."""
        result = self.validator.validate("SELECT * FROM nonexistent_table")
        self.assertFalse(result.is_valid)
        issue_types = [i.issue_type for i in result.issues]
        self.assertIn(OBQCIssueType.TABLE_NOT_FOUND, issue_types)

    def test_column_not_found(self):
        """Test detection of non-existent column."""
        result = self.validator.validate("SELECT users.nonexistent_column FROM users")
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
        result = self.validator.validate("SELECT * FROM users WHERE users.id = 'abc'")
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
        result = self.validator.validate("SELECT * FROM users", dialect="postgresql")
        self.assertTrue(result.is_valid)

        # Snowflake
        result = self.validator.validate("SELECT * FROM users", dialect="snowflake")
        self.assertTrue(result.is_valid)

        # Dremio (uses trino dialect)
        result = self.validator.validate("SELECT * FROM users", dialect="dremio")
        self.assertTrue(result.is_valid)


class TestOBQCCatalogQueries(unittest.TestCase):
    """Database catalog schemas are exempt from the ontology-existence rules.

    The ontology describes user data, so catalog tables are never in it.
    Requiring them to be blocked every metadata query -- listing tables,
    counting columns -- with "Table 'tables' not found in ontology", because
    the schema qualifier was discarded before the check ran.
    """

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def test_information_schema_query_is_allowed(self):
        result = self.validator.validate(
            "SELECT table_name FROM information_schema.tables"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_unqualified_column_of_catalog_table_is_not_flagged(self):
        """The column rule reached the same failure by another route: the
        catalog table resolves to nothing, so every column looked missing."""
        result = self.validator.validate(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'users'"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_catalog_tables_are_recorded(self):
        result = self.validator.validate("SELECT * FROM pg_catalog.pg_class")

        self.assertIn("pg_class", result.catalog_tables)

    def test_dialect_specific_catalogs_are_recognized(self):
        for query in (
            "SELECT * FROM pg_catalog.pg_class",
            "SELECT * FROM system.tables",
            "SELECT * FROM performance_schema.events_statements_summary_by_digest",
            "SELECT * FROM snowflake.account_usage.query_history",
        ):
            with self.subTest(query=query):
                result = self.validator.validate(query)
                self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_catalog_schema_match_is_case_insensitive(self):
        """Snowflake upper-cases identifiers."""
        result = self.validator.validate(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_unknown_table_without_catalog_qualifier_still_errors(self):
        """The exemption is keyed on the qualifier, not on being unknown."""
        result = self.validator.validate("SELECT * FROM nonexistent_table")

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any(i.issue_type == OBQCIssueType.TABLE_NOT_FOUND for i in result.issues)
        )

    def test_bare_table_named_like_a_catalog_table_still_errors(self):
        """'tables' unqualified is not a catalog reference."""
        result = self.validator.validate("SELECT * FROM tables")

        self.assertFalse(result.is_valid)

    def test_mysql_schema_is_not_a_catalog(self):
        """mysql.* is the server's own data, not metadata.

        mysql.user holds account names and password hashes; exempting the
        schema as a "catalog" would have waved those through. src/security.py
        blocks them outright, and this keeps the ontology rule applying too.
        """
        result = self.validator.validate(
            "SELECT User, authentication_string FROM mysql.user", dialect="mysql"
        )

        self.assertFalse(result.is_valid)
        self.assertNotIn("user", result.catalog_tables)

    def test_shadowed_name_is_not_exempted(self):
        """A user table sharing a catalog table's name keeps being checked.

        Membership is tracked by bare name, so without this an unknown table
        called "tables" would be hidden by information_schema.tables appearing
        in the same query.
        """
        result = self.validator.validate(
            "SELECT * FROM tables JOIN information_schema.tables t2 ON 1 = 1"
        )

        self.assertFalse(result.is_valid)
        self.assertNotIn("tables", result.catalog_tables)

    def test_unqualified_column_allowed_when_a_catalog_table_is_present(self):
        """A name that matches no user table may belong to the catalog one."""
        result = self.validator.validate(
            "SELECT table_name FROM information_schema.tables "
            "JOIN users ON users.name = table_name"
        )

        column_errors = [
            i
            for i in result.issues
            if i.issue_type == OBQCIssueType.COLUMN_NOT_FOUND
            and i.severity == OBQCSeverity.ERROR
        ]
        self.assertEqual(column_errors, [])

    def test_ontology_table_errors_are_unaffected_by_a_catalog_join(self):
        """Mixing the two must not silence checks on the real table."""
        result = self.validator.validate(
            "SELECT u.bogus_col FROM users u, information_schema.tables t"
        )

        self.assertFalse(result.is_valid)


class TestOBQCSelectAliases(unittest.TestCase):
    """SELECT aliases referenced by later clauses are not table columns.

    ORDER BY / GROUP BY / HAVING are evaluated after the select list and can
    see its output names. The column rule knew only about table columns, so
    "ORDER BY revenue" over "SUM(total) AS revenue" was reported missing --
    an error, which blocks the query.
    """

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def test_order_by_aggregate_alias(self):
        result = self.validator.validate(
            "SELECT SUM(total) AS revenue FROM orders ORDER BY revenue DESC"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_order_by_plain_column_alias(self):
        result = self.validator.validate(
            "SELECT name AS customer FROM users ORDER BY customer"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_group_by_and_order_by_alias(self):
        result = self.validator.validate(
            "SELECT COUNT(*) AS n FROM orders GROUP BY user_id ORDER BY n DESC"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_having_alias_rejected_on_postgres(self):
        """PostgreSQL permits an output name in GROUP BY and ORDER BY, but not
        in WHERE or HAVING -- so this is invalid and must be reported."""
        result = self.validator.validate(
            "SELECT SUM(total) AS revenue FROM orders "
            "GROUP BY user_id HAVING revenue > 10",
            dialect="postgresql",
        )

        self.assertFalse(result.is_valid)

    def test_having_alias_allowed_on_dialects_that_permit_it(self):
        """Every supported database except PostgreSQL resolves the alias here.

        Checked against vendor documentation, and directly against DuckDB.
        Dremio is included by policy rather than by evidence: Trino's docs do
        not state whether an output alias resolves in HAVING, and OBQC errors
        block execution, so an unverified clause is left open -- a wrongly
        allowed alias is rejected by the database itself, a wrongly forbidden
        one stops a query that would have run.
        """
        for dialect in (
            "mysql",
            "clickhouse",
            "snowflake",
            "databricks",
            "bigquery",
            "duckdb",
            "dremio",
        ):
            with self.subTest(dialect=dialect):
                result = self.validator.validate(
                    "SELECT user_id, SUM(total) AS revenue FROM orders "
                    "GROUP BY user_id HAVING revenue > 10",
                    dialect=dialect,
                )

                self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_where_alias_errors_except_on_duckdb(self):
        """DuckDB resolves aliases laterally, including in WHERE."""
        for dialect in ("postgresql", "mysql", "snowflake", "bigquery"):
            with self.subTest(dialect=dialect):
                result = self.validator.validate(
                    "SELECT total AS t FROM orders WHERE t > 5", dialect=dialect
                )
                self.assertFalse(result.is_valid)

        duckdb_result = self.validator.validate(
            "SELECT total AS t FROM orders WHERE t > 5", dialect="duckdb"
        )

        self.assertTrue(
            duckdb_result.is_valid, [i.message for i in duckdb_result.issues]
        )

    def test_order_by_alias_allowed_on_every_dialect(self):
        from src.constants import SUPPORTED_DB_TYPES

        for dialect in SUPPORTED_DB_TYPES:
            with self.subTest(dialect=dialect):
                result = self.validator.validate(
                    "SELECT SUM(total) AS revenue FROM orders ORDER BY revenue",
                    dialect=dialect,
                )

                self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_group_by_alias_satisfies_the_aggregation_rule(self):
        """Grouping by an alias groups by its source column.

        The GROUP BY key was recorded as the alias while the SELECT expression
        was checked as the source column, so the two never matched and a valid
        query was rejected as not grouped.
        """
        result = self.validator.validate(
            "SELECT user_id AS uid, SUM(total) FROM orders GROUP BY uid"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_group_by_name_that_is_also_a_column_resolves_to_the_column(self):
        """An ambiguous GROUP BY name is the input column, not the alias.

        "SELECT total AS user_id ... GROUP BY user_id" groups by orders.user_id
        and leaves total ungrouped. Treating the name as the alias marked total
        as grouped and passed a query the database rejects. Verified against
        DuckDB, which fails it with "column total must appear in the GROUP BY
        clause", and documented for PostgreSQL.
        """
        for dialect in ("postgresql", "duckdb"):
            with self.subTest(dialect=dialect):
                result = self.validator.validate(
                    "SELECT total AS user_id, SUM(id) FROM orders GROUP BY user_id",
                    dialect=dialect,
                )

                self.assertFalse(result.is_valid)

    def test_unambiguous_group_by_alias_still_resolves(self):
        """uid is not a column of any queried table, so it is the alias."""
        result = self.validator.validate(
            "SELECT user_id AS uid, SUM(total) FROM orders GROUP BY uid"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_alias_inside_an_order_by_expression_is_rejected_on_postgres(self):
        """PostgreSQL takes an output name as a sort key only, not inside an
        expression: ORDER BY t + 1 is evaluated over input columns."""
        result = self.validator.validate(
            "SELECT total AS t FROM orders ORDER BY t + 1", dialect="postgresql"
        )

        self.assertFalse(result.is_valid)

    def test_alias_inside_an_order_by_expression_is_allowed_on_duckdb(self):
        """DuckDB accepts it -- verified against duckdb 1.5.5."""
        result = self.validator.validate(
            "SELECT total AS t FROM orders ORDER BY t + 1", dialect="duckdb"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_standalone_sort_key_still_resolves_on_postgres(self):
        """The restriction is about expressions, not about sort modifiers."""
        for query in (
            "SELECT total AS t FROM orders ORDER BY t",
            "SELECT SUM(total) AS revenue FROM orders ORDER BY revenue DESC",
            "SELECT SUM(total) AS revenue FROM orders ORDER BY revenue DESC NULLS LAST",
        ):
            with self.subTest(query=query):
                result = self.validator.validate(query, dialect="postgresql")

                self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_subquery_alias_does_not_excuse_the_outer_query(self):
        """Alias visibility is per SELECT scope.

        Recording alias names globally let an inner query's alias excuse an
        unrelated bogus column in the outer SELECT.
        """
        result = self.validator.validate(
            "SELECT bogus, (SELECT total AS bogus FROM orders ORDER BY bogus "
            "LIMIT 1) FROM orders"
        )

        self.assertFalse(result.is_valid)

    def test_subquery_alias_still_works_in_its_own_scope(self):
        result = self.validator.validate(
            "SELECT (SELECT total AS t FROM orders ORDER BY t LIMIT 1) FROM orders"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_alias_is_recorded(self):
        result = self.validator.validate(
            "SELECT SUM(total) AS revenue FROM orders ORDER BY revenue"
        )

        self.assertIn("revenue", result.select_aliases)

    def test_alias_matching_is_case_insensitive(self):
        result = self.validator.validate(
            "SELECT SUM(total) AS Revenue FROM orders ORDER BY REVENUE"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_alias_in_where_still_errors(self):
        """WHERE is evaluated before the select list, so this is invalid SQL
        and must keep failing."""
        result = self.validator.validate("SELECT total AS t FROM orders WHERE t > 5")

        self.assertFalse(result.is_valid)

    def test_unknown_order_by_column_still_errors(self):
        """The exemption covers declared aliases, not any ORDER BY name."""
        result = self.validator.validate(
            "SELECT total FROM orders ORDER BY nonexistent_col"
        )

        self.assertFalse(result.is_valid)

    def test_qualified_reference_to_an_alias_still_errors(self):
        """orders.t names a table column, and there is none called t."""
        result = self.validator.validate(
            "SELECT total AS t FROM orders ORDER BY orders.t"
        )

        self.assertFalse(result.is_valid)

    def test_alias_does_not_leak_into_other_column_checks(self):
        """Declaring an alias must not excuse a genuinely bad column."""
        result = self.validator.validate(
            "SELECT total AS t, bogus_col FROM orders ORDER BY t"
        )

        self.assertFalse(result.is_valid)


class TestOBQCSubqueryScoping(unittest.TestCase):
    """Rules apply per SELECT, not across the whole parsed query.

    Tables, columns and aggregation were collected into flat query-wide state,
    so a subquery's contents were judged as if they belonged to the outer
    query. Every IN / EXISTS / scalar subquery was rejected outright.
    """

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def test_in_subquery_is_not_a_cartesian_product(self):
        """Two tables in total, no joins in total -- but one table per SELECT."""
        result = self.validator.validate(
            "SELECT id FROM users WHERE id IN (SELECT user_id FROM orders)"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_exists_subquery_is_not_a_cartesian_product(self):
        result = self.validator.validate(
            "SELECT name FROM users u "
            "WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_scalar_subquery_in_select_list_is_allowed(self):
        result = self.validator.validate(
            "SELECT name, (SELECT COUNT(*) FROM orders) AS n FROM users"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_aggregate_in_subquery_does_not_require_outer_group_by(self):
        """The outer SELECT aggregates nothing, so it needs no GROUP BY."""
        result = self.validator.validate(
            "SELECT name FROM users WHERE id = (SELECT MAX(user_id) FROM orders)"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_subquery_table_cannot_resolve_an_outer_column(self):
        """quantity belongs to order_items, which the outer SELECT cannot see."""
        result = self.validator.validate(
            "SELECT quantity FROM users WHERE id IN (SELECT order_id FROM order_items)"
        )

        self.assertFalse(result.is_valid)

    def test_no_ambiguity_across_scopes(self):
        """users.id and orders.id live in different scopes here."""
        result = self.validator.validate(
            "SELECT id FROM users WHERE id IN (SELECT user_id FROM orders)"
        )

        ambiguous = [
            i for i in result.issues if i.issue_type == OBQCIssueType.AMBIGUOUS_COLUMN
        ]
        self.assertEqual(ambiguous, [])

    def test_correlated_subquery_may_use_outer_tables(self):
        """An enclosing SELECT's tables stay in scope, or this reads as
        missing."""
        result = self.validator.validate(
            "SELECT u.name FROM users u WHERE EXISTS ("
            "SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.total > 5)"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_real_cartesian_product_still_errors(self):
        result = self.validator.validate("SELECT * FROM users, orders")

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any(
                i.issue_type == OBQCIssueType.MISSING_JOIN_CONDITION
                for i in result.issues
            )
        )

    def test_real_missing_group_by_still_errors(self):
        result = self.validator.validate("SELECT name, SUM(id) FROM users")

        self.assertFalse(result.is_valid)

    def test_nested_aggregate_does_not_excuse_an_outer_column(self):
        """An aggregate in a subquery cannot aggregate an outer column.

        The per-column check scanned aggregates anywhere under the SELECT, so
        a nested MAX(id) made the outer bare id look aggregated and a query
        genuinely missing its GROUP BY passed.
        """
        result = self.validator.validate(
            "SELECT id, SUM(total), (SELECT MAX(id) FROM users) FROM orders"
        )

        self.assertFalse(result.is_valid)

    def test_subquery_aggregate_does_not_trigger_a_fan_trap(self):
        """Fan-out only inflates totals the joins are aggregated over."""
        result = self.validator.validate(
            "SELECT users.name FROM users "
            "JOIN orders ON users.id = orders.user_id "
            "JOIN shipments ON shipments.order_id = orders.id "
            "WHERE EXISTS (SELECT COUNT(*) FROM order_items)"
        )

        self.assertFalse(
            result.fan_trap_risk,
            [i.message for i in result.issues],
        )

    def test_real_fan_trap_in_an_aggregating_select_still_warns(self):
        result = self.validator.validate(
            "SELECT orders.id, SUM(order_items.quantity), SUM(shipments.cost) "
            "FROM orders "
            "JOIN order_items ON orders.id = order_items.order_id "
            "JOIN shipments ON orders.id = shipments.order_id "
            "GROUP BY orders.id"
        )

        self.assertTrue(result.fan_trap_risk)

    def test_inner_name_resolves_in_the_innermost_scope(self):
        """SQL stops at the innermost scope that provides a name.

        Flattening the scope levels made the inner id look ambiguous between
        orders.id and users.id, when it is simply orders.id.
        """
        result = self.validator.validate(
            "SELECT name FROM users WHERE EXISTS ("
            "SELECT 1 FROM orders WHERE id = user_id)"
        )

        ambiguous = [
            i for i in result.issues if i.issue_type == OBQCIssueType.AMBIGUOUS_COLUMN
        ]
        self.assertEqual(ambiguous, [], [i.message for i in result.issues])

    def test_ambiguity_within_one_scope_still_warns(self):
        result = self.validator.validate(
            "SELECT id FROM users JOIN orders ON users.id = orders.user_id"
        )

        ambiguous = [
            i for i in result.issues if i.issue_type == OBQCIssueType.AMBIGUOUS_COLUMN
        ]
        self.assertEqual(len(ambiguous), 1)


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
            i for i in result.issues if i.issue_type == OBQCIssueType.FAN_TRAP_DETECTED
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


class TestOBQCFanTrapDirection(unittest.TestCase):
    """Fan-out is a property of a join's direction, not of a table.

    The heuristic asked whether a joined table sat on the "many" side of any
    relationship anywhere in the schema, and counted once per matching
    relationship rather than once per join. A dimension with its own foreign
    keys therefore scored a fan-out for merely existing, so walking a chain of
    many-to-one lookups -- where no row is ever duplicated -- drew a warning.
    """

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def test_many_to_one_chain_is_not_a_fan_trap(self):
        """order_items -> orders -> users: every hop is a lookup.

        The measure is taken at the finest grain in the query, so walking up to
        the dimensions repeats nothing.
        """
        result = self.validator.validate(
            "SELECT users.name, SUM(order_items.quantity) "
            "FROM order_items "
            "JOIN orders ON order_items.order_id = orders.id "
            "JOIN users ON orders.user_id = users.id "
            "GROUP BY users.name"
        )

        self.assertFalse(
            result.fan_trap_risk,
            [i.message for i in result.issues],
        )

    def test_many_to_one_chain_with_aliases_is_not_a_fan_trap(self):
        """ON conditions are qualified by alias, so aliases must resolve."""
        result = self.validator.validate(
            "SELECT u.name, SUM(oi.quantity) AS units "
            "FROM public.order_items oi "
            "JOIN public.orders o ON oi.order_id = o.id "
            "JOIN public.users u ON o.user_id = u.id "
            "GROUP BY u.name ORDER BY SUM(oi.quantity) DESC"
        )

        self.assertFalse(
            result.fan_trap_risk,
            [i.message for i in result.issues],
        )

    def test_parent_measure_at_child_grain_is_a_fan_trap(self):
        """The same chain, but summing the parent's measure, does inflate.

        ``FROM order_items JOIN orders`` yields one row per item, so each
        order's total is added once per item it contains. Checked against
        DuckDB: two orders totalling 150 come back as 250 when order 1 has two
        items. Which table sits in FROM makes no difference to that.
        """
        result = self.validator.validate(
            "SELECT users.name, SUM(orders.total) "
            "FROM order_items "
            "JOIN orders ON order_items.order_id = orders.id "
            "JOIN users ON orders.user_id = users.id "
            "GROUP BY users.name"
        )

        self.assertTrue(result.fan_trap_risk)
        self.assertEqual(
            [
                (f["measure_table"], f["fan_out_table"])
                for f in result.fan_trap_findings
            ],
            [("orders", "order_items")],
        )

    def test_duplication_proof_aggregates_survive_two_fan_outs(self):
        """MAX reads the same answer however often its rows are repeated.

        The count heuristic fired on any aggregate at all, so it blocked
        queries the measure rule deliberately leaves alone -- and once
        fan-traps became errors, that stopped them running.
        """
        for aggregate in (
            "MAX(orders.total)",
            "MIN(orders.total)",
            "COUNT(DISTINCT orders.id)",
        ):
            with self.subTest(aggregate=aggregate):
                result = self.validator.validate(
                    f"SELECT orders.id, {aggregate} "
                    "FROM orders "
                    "JOIN order_items ON orders.id = order_items.order_id "
                    "JOIN shipments ON orders.id = shipments.order_id "
                    "GROUP BY orders.id"
                )

                self.assertFalse(
                    result.fan_trap_risk, [i.message for i in result.issues]
                )

    def test_count_star_across_two_fan_outs_is_still_a_fan_trap(self):
        """COUNT(*) names no table, but it counts rows -- here, their product."""
        result = self.validator.validate(
            "SELECT orders.id, COUNT(*) "
            "FROM orders "
            "JOIN order_items ON orders.id = order_items.order_id "
            "JOIN shipments ON orders.id = shipments.order_id "
            "GROUP BY orders.id"
        )

        self.assertTrue(result.fan_trap_risk)

    def test_two_facts_on_one_dimension_is_still_a_fan_trap(self):
        """The true positive must survive: orders fans out twice."""
        result = self.validator.validate(
            "SELECT orders.id, SUM(order_items.quantity), SUM(shipments.cost) "
            "FROM orders "
            "JOIN order_items ON orders.id = order_items.order_id "
            "JOIN shipments ON orders.id = shipments.order_id "
            "GROUP BY orders.id"
        )

        self.assertTrue(result.fan_trap_risk)

    def test_single_fan_out_join_is_not_flagged(self):
        result = self.validator.validate(
            "SELECT users.name, SUM(orders.total) "
            "FROM users JOIN orders ON users.id = orders.user_id "
            "GROUP BY users.name"
        )

        self.assertFalse(result.fan_trap_risk)

    def test_direction_is_judged_per_join_not_per_relationship(self):
        """Joining one dimension that has its own FK must score zero fan-outs.

        orders references users, so the old rule counted 'orders' as a many
        side even when the query joins *to* it from its own child.
        """
        result = self.validator.validate(
            "SELECT SUM(order_items.quantity) "
            "FROM order_items JOIN orders ON order_items.order_id = orders.id"
        )

        self.assertFalse(
            result.fan_trap_risk,
            [i.message for i in result.issues],
        )

    def test_subquery_alias_does_not_shadow_an_outer_join_anchor(self):
        """Table aliases are scoped to the SELECT that declares them.

        A single alias map over the whole tree let "FROM order_items u" inside
        an EXISTS overwrite the outer "FROM users u", so the outer join was
        judged against order_items instead of users -- and the fan-trap
        warning silently disappeared.
        """
        base = (
            "SELECT u.name, SUM(s.cost) FROM users u "
            "JOIN orders o ON u.id = o.user_id "
            "JOIN shipments s ON s.order_id = o.id GROUP BY u.name"
        )
        with_subquery = base.replace(
            "GROUP BY u.name",
            "WHERE EXISTS (SELECT 1 FROM order_items u WHERE u.order_id = o.id) "
            "GROUP BY u.name",
        )

        baseline = self.validator.validate(base)
        shadowed = self.validator.validate(with_subquery)

        self.assertTrue(baseline.fan_trap_risk)
        self.assertTrue(
            shadowed.fan_trap_risk,
            "subquery alias suppressed the outer query's fan-trap warning",
        )

    def test_outer_join_anchors_are_unaffected_by_a_subquery(self):
        """The anchors themselves must be identical, warning aside."""
        base = "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id"
        with_subquery = (
            "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id "
            "WHERE EXISTS (SELECT 1 FROM order_items u WHERE u.order_id = o.id)"
        )

        baseline = self.validator.validate(base)
        shadowed = self.validator.validate(with_subquery)

        self.assertEqual(baseline.parsed_joins[0]["on_tables"], ["users", "orders"])
        self.assertEqual(shadowed.parsed_joins[0]["on_tables"], ["users", "orders"])

    def test_fan_outs_are_counted_per_select_not_pooled(self):
        """Two subqueries fanning out once each are two safe aggregations.

        Rows are multiplied by joins in the same query. Summing fan-outs
        across unrelated SELECTs reported a fan-trap that exists in neither.
        """
        result = self.validator.validate(
            "SELECT "
            "(SELECT SUM(order_items.quantity) FROM orders "
            " JOIN order_items ON orders.id = order_items.order_id), "
            "(SELECT SUM(shipments.cost) FROM orders "
            " JOIN shipments ON orders.id = shipments.order_id) "
            "FROM users"
        )

        self.assertFalse(
            result.fan_trap_risk,
            [i.message for i in result.issues],
        )

    def test_fan_trap_inside_a_subquery_is_still_detected(self):
        """Scoping the count must not switch detection off in subqueries."""
        result = self.validator.validate(
            "SELECT name, ("
            "SELECT SUM(order_items.quantity) + SUM(shipments.cost) FROM orders "
            "JOIN order_items ON orders.id = order_items.order_id "
            "JOIN shipments ON orders.id = shipments.order_id) "
            "FROM users"
        )

        self.assertTrue(result.fan_trap_risk)

    def test_only_the_fanning_join_is_blamed(self):
        """A lookup joined alongside a fan-out must not be named as the cause.

        shipments multiplies orders; users is a many-to-one lookup and
        multiplies nothing. The finding must say so.
        """
        result = self.validator.validate(
            "SELECT SUM(orders.total) FROM orders "
            "JOIN users ON orders.user_id = users.id "
            "JOIN shipments ON shipments.order_id = orders.id"
        )

        self.assertTrue(result.fan_trap_risk)
        blamed = {f["fan_out_table"] for f in result.fan_trap_findings}
        self.assertEqual(blamed, {"shipments"})


class TestOBQCFanTrapDimensionWithOwnKeys(unittest.TestCase):
    """A dimension carrying several foreign keys must not manufacture fan-outs.

    This is the reported false positive, and it needs a dimension with two
    outgoing FKs to reproduce: the old rule incremented once per matching
    *relationship* rather than once per join, so joining that one dimension
    scored two fan-outs by itself and crossed the warning threshold -- even
    though the query only walks many-to-one lookups.
    """

    def setUp(self):
        from src.database_manager import ColumnInfo, TableInfo
        from src.ontology_generator import OntologyGenerator

        def col(name, fk_to=None, pk=False):
            return ColumnInfo(
                name=name,
                data_type="INTEGER",
                is_nullable=False,
                is_primary_key=pk,
                is_foreign_key=fk_to is not None,
                foreign_key_table=fk_to,
                foreign_key_column="id" if fk_to else None,
            )

        def fk(column, to_table):
            return {
                "column": column,
                "referenced_table": to_table,
                "referenced_column": "id",
            }

        tables = [
            TableInfo(
                name="sales",
                schema="public",
                columns=[col("id", pk=True), col("client_id", "clients")],
                primary_keys=["id"],
                foreign_keys=[fk("client_id", "clients")],
            ),
            # Two outgoing FKs -- the shape that triggered the false positive.
            TableInfo(
                name="clients",
                schema="public",
                columns=[
                    col("id", pk=True),
                    col("country_id", "countries"),
                    col("region_id", "regions"),
                ],
                primary_keys=["id"],
                foreign_keys=[
                    fk("country_id", "countries"),
                    fk("region_id", "regions"),
                ],
            ),
            TableInfo(
                name="countries",
                schema="public",
                columns=[col("id", pk=True)],
                primary_keys=["id"],
                foreign_keys=[],
            ),
            TableInfo(
                name="regions",
                schema="public",
                columns=[col("id", pk=True)],
                primary_keys=["id"],
                foreign_keys=[],
            ),
        ]

        generator = OntologyGenerator()
        generator.generate_from_schema(tables)
        self.validator = OBQCValidator()
        self.validator.load_ontology(generator.graph, str(generator.base_uri))

    def test_dimension_chain_is_not_flagged(self):
        """many sales -> one client -> one country: nothing is duplicated."""
        result = self.validator.validate(
            "SELECT co.id, COUNT(*) AS orders, SUM(s.id) AS total "
            "FROM sales s "
            "JOIN clients cl ON s.client_id = cl.id "
            "JOIN countries co ON cl.country_id = co.id "
            "GROUP BY co.id"
        )

        self.assertFalse(
            result.fan_trap_risk,
            [i.message for i in result.issues],
        )

    def test_real_fan_out_in_the_same_schema_is_still_flagged(self):
        """Joining the fact from two directions still multiplies rows."""
        result = self.validator.validate(
            "SELECT cl.id, SUM(s1.id), SUM(s2.id) "
            "FROM clients cl "
            "JOIN sales s1 ON s1.client_id = cl.id "
            "JOIN sales s2 ON s2.client_id = cl.id "
            "GROUP BY cl.id"
        )

        self.assertTrue(result.fan_trap_risk)


class TestOBQCAxiomDrivenFanTrap(unittest.TestCase):
    """Phase 2: fan-trap detection grounded in owl:disjointWith axioms."""

    def setUp(self):
        from src.database_manager import ColumnInfo, TableInfo
        from src.ontology_generator import OntologyGenerator

        def fact(name, fk_to):
            return TableInfo(
                name=name,
                schema="public",
                columns=[
                    ColumnInfo(
                        name="id",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=True,
                        is_foreign_key=False,
                    ),
                    ColumnInfo(
                        name="customer_id",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=False,
                        is_foreign_key=True,
                        foreign_key_table=fk_to,
                        foreign_key_column="id",
                    ),
                    ColumnInfo(
                        name="amount",
                        data_type="DECIMAL(12,2)",
                        is_nullable=False,
                        is_primary_key=False,
                        is_foreign_key=False,
                    ),
                ],
                primary_keys=["id"],
                foreign_keys=[
                    {
                        "column": "customer_id",
                        "referenced_table": fk_to,
                        "referenced_column": "id",
                    }
                ],
                row_count=1000,
            )

        customers = TableInfo(
            name="customers",
            schema="public",
            columns=[
                ColumnInfo(
                    name="id",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False,
                ),
                ColumnInfo(
                    name="name",
                    data_type="VARCHAR(200)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False,
                ),
            ],
            primary_keys=["id"],
            foreign_keys=[],
            row_count=500,
        )

        base_uri = "http://test.com/ontology/"
        gen = OntologyGenerator(base_uri)
        ttl = gen.generate_from_schema(
            [customers, fact("orders", "customers"), fact("returns", "customers")],
            include_inferred_relationships=False,
        )
        graph = Graph()
        graph.parse(data=ttl, format="turtle")

        self.validator = OBQCValidator()
        self.validator.load_ontology(graph, base_uri)

    def test_disjoint_pairs_extracted(self):
        self.assertIn(frozenset({"orders", "returns"}), self.validator._disjoint_pairs)

    def test_cross_fact_aggregation_flagged_via_axiom(self):
        result = self.validator.validate(
            "SELECT customers.name, SUM(orders.amount), SUM(returns.amount) "
            "FROM customers "
            "JOIN orders ON customers.id = orders.customer_id "
            "JOIN returns ON customers.id = returns.customer_id "
            "GROUP BY customers.name"
        )
        self.assertTrue(result.fan_trap_risk)
        fan = [
            i for i in result.issues if i.issue_type == OBQCIssueType.FAN_TRAP_DETECTED
        ]
        self.assertEqual(len(fan), 1)
        # cites the actual disjoint pair and recommends a composite (UNION ALL)
        self.assertEqual(set(fan[0].related_entities), {"orders", "returns"})
        self.assertIn("UNION ALL", fan[0].suggestion)

    def test_single_fact_not_flagged(self):
        result = self.validator.validate(
            "SELECT customers.name, SUM(orders.amount) "
            "FROM customers JOIN orders ON customers.id = orders.customer_id "
            "GROUP BY customers.name"
        )
        self.assertFalse(result.fan_trap_risk)

    def test_sibling_fact_in_a_semi_join_filter_is_not_flagged(self):
        """A fact reached only through EXISTS cannot multiply anything.

        The axiom check read its disjoint pair off every table named in the
        query, so a returns table used purely as a filter flagged an
        aggregation over orders that it cannot affect.
        """
        result = self.validator.validate(
            "SELECT c.id, SUM(o.amount) FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "WHERE EXISTS (SELECT 1 FROM returns r WHERE r.customer_id = c.id) "
            "GROUP BY c.id"
        )

        self.assertFalse(
            result.fan_trap_risk,
            [i.message for i in result.issues],
        )

    def test_sibling_fact_in_a_subquery_without_outer_join_is_not_flagged(self):
        result = self.validator.validate(
            "SELECT customer_id, SUM(amount) FROM orders "
            "WHERE EXISTS (SELECT 1 FROM returns "
            "WHERE returns.customer_id = orders.customer_id) "
            "GROUP BY customer_id"
        )

        self.assertFalse(
            result.fan_trap_risk,
            [i.message for i in result.issues],
        )

    def test_both_facts_joined_in_one_select_is_still_flagged(self):
        """The true positive the axiom path exists for."""
        result = self.validator.validate(
            "SELECT c.id, SUM(o.amount), SUM(r.amount) FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN returns r ON r.customer_id = c.id "
            "GROUP BY c.id"
        )

        self.assertTrue(result.fan_trap_risk)


class TestOBQCResponseShape(unittest.TestCase):
    """to_dict() is the wire format, so internal bookkeeping must not appear.

    Fan-trap grouping needs to know which SELECT owns a join, but that is a
    detail of one parse. Leaving it on the join dicts published it through
    every execute_sql_query response.
    """

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)
        self.query = "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id"

    def test_joins_expose_only_public_keys(self):
        joins = self.validator.validate(self.query).to_dict()["parsed_joins"]

        self.assertTrue(joins)
        for join in joins:
            self.assertEqual(
                set(join) - set(OBQCResult.PUBLIC_JOIN_KEYS),
                set(),
                f"internal keys leaked into the response: {sorted(join)}",
            )

    def test_response_is_reproducible(self):
        """An identifier tied to object identity varies between runs."""
        first = self.validator.validate(self.query).to_dict()
        second = self.validator.validate(self.query).to_dict()

        self.assertEqual(first["parsed_joins"], second["parsed_joins"])

    def test_response_is_json_serializable(self):
        payload = self.validator.validate(self.query).to_dict()

        json.dumps(payload)

    def test_useful_join_details_are_still_published(self):
        """Stripping internals must not strip what callers rely on."""
        join = self.validator.validate(self.query).to_dict()["parsed_joins"][0]

        self.assertEqual(join["table"], "orders")
        self.assertEqual(join["on_tables"], ["users", "orders"])
        self.assertIn("on_condition", join)


class TestOBQCDialectParity(unittest.TestCase):
    """Guard that OBQC maps every supported database to a real sqlglot dialect."""

    def test_dialect_map_covers_all_supported_databases(self):
        from src.constants import SUPPORTED_DB_TYPES

        missing = [
            db for db in SUPPORTED_DB_TYPES if db not in OBQCValidator.DIALECT_MAP
        ]
        self.assertEqual(
            missing, [], f"databases missing from OBQC DIALECT_MAP: {missing}"
        )

    def test_mapped_dialects_resolve_in_sqlglot(self):
        from sqlglot.dialects.dialect import Dialect

        for db, dialect in OBQCValidator.DIALECT_MAP.items():
            with self.subTest(db=db):
                Dialect.get_or_raise(dialect)  # raises if the dialect is unknown


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
            related_entities=["foo"],
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
    """Test suite for ontologies without oba: namespace annotations."""

    def test_ontology_without_oba_annotations(self):
        """Test that ontology without oba: annotations is detected as incompatible."""
        # Create a basic OWL ontology without oba: namespace annotations
        g = Graph()
        ns = Namespace("http://example.org/")
        g.bind("ex", ns)

        # Add a class without oba:tableName
        person = ns["Person"]
        g.add((person, RDF.type, OWL.Class))
        g.add((person, RDFS.label, Literal("Person")))

        # Add a property without oba:columnName
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


class TestSingleFanOutMeasure(unittest.TestCase):
    """One 1:many join is enough to inflate a measure taken from the one side.

    The count heuristic needed two fan-out joins before it said anything, so
    ``orders JOIN order_items`` summing ``orders.total`` -- which repeats each
    order's total once per item -- passed in silence.
    """

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def test_measure_from_the_one_side_is_blocked(self):
        result = self.validator.validate(
            "SELECT users.name, SUM(orders.total) "
            "FROM orders "
            "JOIN order_items ON order_items.order_id = orders.id "
            "JOIN users ON orders.user_id = users.id "
            "GROUP BY users.name"
        )

        self.assertTrue(result.fan_trap_risk)
        self.assertFalse(result.is_valid)
        self.assertEqual(
            [
                i.severity
                for i in result.issues
                if i.issue_type == OBQCIssueType.FAN_TRAP_DETECTED
            ],
            [OBQCSeverity.ERROR],
        )

    def test_measure_from_the_many_side_is_fine(self):
        """The repeated rows *are* the measure's rows, so nothing is doubled."""
        result = self.validator.validate(
            "SELECT orders.id, SUM(order_items.quantity) "
            "FROM orders JOIN order_items ON order_items.order_id = orders.id "
            "GROUP BY orders.id"
        )

        self.assertFalse(result.fan_trap_risk, [i.message for i in result.issues])

    def test_count_star_is_not_attributed_to_any_table(self):
        """Counting the joined rows is usually the intent, so it is left alone."""
        result = self.validator.validate(
            "SELECT orders.id, COUNT(*) "
            "FROM orders JOIN order_items ON order_items.order_id = orders.id "
            "GROUP BY orders.id"
        )

        self.assertFalse(result.fan_trap_risk, [i.message for i in result.issues])

    def test_min_and_max_survive_duplication(self):
        result = self.validator.validate(
            "SELECT orders.id, MAX(orders.total) "
            "FROM orders JOIN order_items ON order_items.order_id = orders.id "
            "GROUP BY orders.id"
        )

        self.assertFalse(result.fan_trap_risk, [i.message for i in result.issues])

    def test_count_distinct_survives_duplication(self):
        result = self.validator.validate(
            "SELECT users.name, COUNT(DISTINCT orders.id) "
            "FROM orders "
            "JOIN order_items ON order_items.order_id = orders.id "
            "JOIN users ON orders.user_id = users.id "
            "GROUP BY users.name"
        )

        self.assertFalse(result.fan_trap_risk, [i.message for i in result.issues])

    def test_dimension_lookup_never_fans_out(self):
        result = self.validator.validate(
            "SELECT users.name, SUM(orders.total) "
            "FROM orders JOIN users ON orders.user_id = users.id "
            "GROUP BY users.name"
        )

        self.assertFalse(result.fan_trap_risk, [i.message for i in result.issues])

    def test_comma_join_spelling_is_caught_too(self):
        """The older syntax states the same join and inflates identically.

        Anchors were read only from ON, so writing the join the pre-SQL-92 way
        skipped fan-trap detection entirely.
        """
        result = self.validator.validate(
            "SELECT SUM(o.total) FROM orders o, order_items i "
            "WHERE i.order_id = o.id"
        )

        self.assertTrue(result.fan_trap_risk)
        self.assertFalse(result.is_valid)
        self.assertEqual(
            [
                (f["measure_table"], f["fan_out_table"])
                for f in result.fan_trap_findings
            ],
            [("orders", "order_items")],
        )

    def test_comma_join_measure_from_the_many_side_is_fine(self):
        result = self.validator.validate(
            "SELECT SUM(i.quantity) FROM orders o, order_items i "
            "WHERE i.order_id = o.id"
        )

        self.assertFalse(result.fan_trap_risk, [i.message for i in result.issues])

    def test_finding_is_structured_not_prose(self):
        result = self.validator.validate(
            "SELECT SUM(orders.total) FROM orders "
            "JOIN order_items ON order_items.order_id = orders.id"
        )

        report = result.to_dict()["obqc_fan_trap"]
        self.assertTrue(report["detected"])
        self.assertTrue(report["blocking"])
        self.assertEqual(
            report["findings"],
            [
                {
                    "kind": "measure_across_fan_out",
                    "measure_table": "orders",
                    "fan_out_table": "order_items",
                    "tables": ["order_items", "orders"],
                }
            ],
        )

    def test_allow_fan_out_downgrades_to_a_warning(self):
        sql = (
            "SELECT SUM(orders.total) FROM orders "
            "JOIN order_items ON order_items.order_id = orders.id"
        )

        result = self.validator.validate(sql, allow_fan_out=True)

        self.assertTrue(result.is_valid)
        self.assertTrue(result.fan_trap_risk)
        self.assertEqual(
            [
                i.severity
                for i in result.issues
                if i.issue_type == OBQCIssueType.FAN_TRAP_DETECTED
            ],
            [OBQCSeverity.WARNING],
        )
        self.assertFalse(result.to_dict()["obqc_fan_trap"]["blocking"])

    def test_clean_query_reports_no_fan_trap(self):
        """The verdict is a field even when the answer is no."""
        report = self.validator.validate("SELECT users.name FROM users").to_dict()[
            "obqc_fan_trap"
        ]

        self.assertEqual(report, {"detected": False, "blocking": True, "findings": []})


class TestWindowFunctions(unittest.TestCase):
    """A windowed aggregate collapses no rows, so it imposes no GROUP BY."""

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def _errors(self, sql):
        result = self.validator.validate(sql)
        return [i.message for i in result.issues if i.severity == OBQCSeverity.ERROR]

    def test_window_aggregate_needs_no_group_by(self):
        """The reported false positive: a window SUM demanded a GROUP BY."""
        self.assertEqual(
            self._errors(
                "SELECT o.id, o.total, SUM(o.total) OVER (PARTITION BY o.user_id) "
                "FROM orders o"
            ),
            [],
        )

    def test_running_total_over_order_by(self):
        self.assertEqual(
            self._errors(
                "SELECT o.order_date, SUM(o.total) OVER (ORDER BY o.order_date) "
                "FROM orders o"
            ),
            [],
        )

    def test_rank_beside_a_real_aggregate(self):
        self.assertEqual(
            self._errors(
                "SELECT u.name, SUM(o.total), RANK() OVER (ORDER BY SUM(o.total) DESC) "
                "FROM orders o JOIN users u ON o.user_id = u.id "
                "GROUP BY u.name"
            ),
            [],
        )

    def test_grouping_is_still_required_beside_a_window(self):
        """A window does not excuse a bare column from a real GROUP BY."""
        errors = self._errors(
            "SELECT u.name, o.order_date, SUM(o.total), "
            "AVG(o.total) OVER (PARTITION BY u.name) "
            "FROM orders o JOIN users u ON o.user_id = u.id "
            "GROUP BY u.name"
        )

        self.assertTrue(any("order_date" in e for e in errors), errors)


class TestTemporalLiterals(unittest.TestCase):
    """Dates are written as string literals in every dialect."""

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def _messages(self, sql):
        return [i.message for i in self.validator.validate(sql).issues]

    def test_date_literal_is_not_a_type_mismatch(self):
        self.assertEqual(
            self._messages(
                "SELECT o.id FROM orders o WHERE o.order_date >= '2024-01-01'"
            ),
            [],
        )

    def test_timestamp_literal_is_not_a_type_mismatch(self):
        self.assertEqual(
            self._messages(
                "SELECT o.id FROM orders o WHERE o.order_date < '2024-01-01 12:30:00'"
            ),
            [],
        )

    def test_string_column_against_a_date_shaped_literal_is_fine(self):
        """The literal is only temporal opposite a temporal column.

        Typing every ISO-looking string as a date fixed date columns and broke
        string ones: "email = '2024-01-01'" is an ordinary string comparison,
        and was reported as "string vs dateTime".
        """
        self.assertEqual(
            self._messages("SELECT u.id FROM users u WHERE u.email = '2024-01-01'"),
            [],
        )

    def test_a_string_that_is_not_a_date_still_mismatches(self):
        self.assertTrue(
            any(
                "mismatch" in m.lower()
                for m in self._messages(
                    "SELECT o.id FROM orders o WHERE o.order_date = 'hello'"
                )
            )
        )

    def test_alias_qualified_comparison_is_type_checked(self):
        """Unresolved aliases meant most real SQL was never type-checked."""
        self.assertTrue(
            any(
                "mismatch" in m.lower()
                for m in self._messages(
                    "SELECT o.id FROM orders o JOIN users u ON o.user_id = u.name"
                )
            )
        )


class TestCommonTableExpressions(unittest.TestCase):
    """A WITH alias is a table the query defines, not one the ontology owns."""

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def _errors(self, sql):
        result = self.validator.validate(sql)
        return [i.message for i in result.issues if i.severity == OBQCSeverity.ERROR]

    def test_cte_name_is_not_reported_missing(self):
        """The reported bug: a WITH alias was rejected as an unknown table."""
        self.assertEqual(
            self._errors(
                "WITH user_totals AS ("
                "  SELECT user_id, SUM(total) AS revenue FROM orders GROUP BY user_id"
                ") "
                "SELECT u.name, ut.revenue "
                "FROM user_totals ut JOIN users u ON u.id = ut.user_id"
            ),
            [],
        )

    def test_cte_output_column_is_not_reported_missing(self):
        """A CTE's columns come from its select list, not from the ontology."""
        self.assertEqual(
            self._errors(
                "WITH t AS (SELECT SUM(total) AS revenue FROM orders) "
                "SELECT revenue FROM t"
            ),
            [],
        )

    def test_join_to_cte_is_not_warned_about_foreign_keys(self):
        """A CTE has no declared FK, so the FK check can only cry wolf."""
        result = self.validator.validate(
            "WITH t AS (SELECT user_id FROM orders) "
            "SELECT u.name FROM t JOIN users u ON t.user_id = u.id"
        )

        self.assertEqual(
            [
                i.message
                for i in result.issues
                if i.issue_type == OBQCIssueType.INVALID_JOIN
            ],
            [],
        )

    def test_cte_body_is_still_validated(self):
        """Exempting the WITH alias must not exempt the query behind it."""
        errors = self._errors(
            "WITH t AS (SELECT bogus_column FROM orders) SELECT user_id FROM t"
        )

        self.assertTrue(any("bogus_column" in e for e in errors), errors)

    def test_cte_body_cannot_see_the_declaring_query(self):
        """A CTE body is not a nested scope, so outer tables cannot excuse it."""
        errors = self._errors(
            "WITH t AS (SELECT o.nonexistent FROM orders o) SELECT user_id FROM t"
        )

        self.assertTrue(any("nonexistent" in e for e in errors), errors)

    def test_inner_cte_does_not_excuse_an_outer_real_table(self):
        """A CTE is only a table where its WITH clause is in scope.

        Collecting names across the whole query let a CTE declared inside a
        subquery hide a real table of the same name in the outer one.
        """
        errors = self._errors(
            "SELECT users.nonexistent FROM users "
            "WHERE EXISTS (WITH users AS (SELECT 1 AS x FROM orders) "
            "SELECT 1 FROM users)"
        )

        self.assertTrue(any("nonexistent" in e for e in errors), errors)

    def test_both_readings_of_one_name_coexist(self):
        """A name can be a CTE in one scope and a real table in another.

        The exemption is per reference. Tracking it by name broke one half or
        the other: globally, the inner CTE excused the outer real table;
        by-name-minus-conflicts, the outer real table stopped the inner CTE's
        own columns from being exempt, and a valid query was blocked.
        """
        self.assertEqual(
            self._errors(
                "SELECT u.name FROM users u WHERE u.id IN ("
                "WITH users AS (SELECT id AS uid FROM orders) SELECT uid FROM users)"
            ),
            [],
        )

    def test_cte_named_after_a_real_table_still_shadows_it(self):
        """Where the WITH *is* in scope, the CTE wins."""
        self.assertEqual(
            self._errors(
                "WITH users AS (SELECT id AS uid FROM orders) SELECT uid FROM users"
            ),
            [],
        )

    def test_cte_declared_inside_a_subquery_is_exempt_there(self):
        self.assertEqual(
            self._errors(
                "SELECT u.name FROM users u WHERE u.id IN ("
                "WITH t AS (SELECT user_id FROM orders) SELECT user_id FROM t)"
            ),
            [],
        )

    def test_a_cte_body_reads_the_real_table_of_the_same_name(self):
        """A non-recursive CTE cannot see itself, so its body reads the table.

        Exposing every name in the WITH to every reference under it skipped
        validation of the body. Checked against DuckDB, which resolves the
        inner FROM to the real table and rejects the unknown column.
        """
        errors = self._errors(
            "WITH orders AS (SELECT nonexistent FROM orders) SELECT id FROM orders"
        )

        self.assertTrue(any("nonexistent" in e for e in errors), errors)

    def test_forward_reference_to_a_later_sibling_is_not_a_cte(self):
        """CTEs see the siblings declared before them, not after."""
        errors = self._errors(
            "WITH a AS (SELECT id FROM b), b AS (SELECT id FROM orders) "
            "SELECT id FROM a"
        )

        self.assertTrue(any("'b'" in e for e in errors), errors)

    def test_earlier_sibling_is_visible(self):
        self.assertEqual(
            self._errors(
                "WITH a AS (SELECT user_id FROM orders), "
                "b AS (SELECT user_id FROM a) SELECT user_id FROM b"
            ),
            [],
        )

    def test_non_recursive_self_reference_is_not_a_cte(self):
        """Without RECURSIVE this is a circular reference, not a CTE use."""
        errors = self._errors(
            "WITH chain AS (SELECT id FROM chain) SELECT id FROM chain"
        )

        self.assertTrue(any("chain" in e for e in errors), errors)

    def test_recursive_cte_self_reference(self):
        """A recursive CTE names itself; that reference is not a missing table."""
        self.assertEqual(
            self._errors(
                "WITH RECURSIVE chain AS ("
                "  SELECT id, user_id FROM orders WHERE user_id = 1"
                "  UNION ALL"
                "  SELECT o.id, o.user_id FROM orders o JOIN chain c ON o.id = c.id"
                ") SELECT id FROM chain"
            ),
            [],
        )


class TestGroupingSetConstructs(unittest.TestCase):
    """ROLLUP, CUBE and GROUPING SETS declare grouping keys like GROUP BY does."""

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def _errors(self, sql):
        result = self.validator.validate(sql)
        return [i.message for i in result.issues if i.severity == OBQCSeverity.ERROR]

    def test_rollup_columns_count_as_grouped(self):
        """The reported bug: ROLLUP keys read as grouping by nothing at all."""
        self.assertEqual(
            self._errors(
                "SELECT u.name, o.order_date, SUM(o.total) "
                "FROM orders o JOIN users u ON o.user_id = u.id "
                "GROUP BY ROLLUP(u.name, o.order_date)"
            ),
            [],
        )

    def test_cube_columns_count_as_grouped(self):
        self.assertEqual(
            self._errors(
                "SELECT u.name, SUM(o.total) "
                "FROM orders o JOIN users u ON o.user_id = u.id "
                "GROUP BY CUBE(u.name)"
            ),
            [],
        )

    def test_grouping_sets_columns_count_as_grouped(self):
        """Members nest inside Paren/Tuple wrappers; "()" contributes none."""
        self.assertEqual(
            self._errors(
                "SELECT u.name, SUM(o.total) "
                "FROM orders o JOIN users u ON o.user_id = u.id "
                "GROUP BY GROUPING SETS ((u.name), ())"
            ),
            [],
        )

    def test_plain_and_rollup_keys_combine(self):
        self.assertEqual(
            self._errors(
                "SELECT u.name, o.order_date, SUM(o.total) "
                "FROM orders o JOIN users u ON o.user_id = u.id "
                "GROUP BY u.name, ROLLUP(o.order_date)"
            ),
            [],
        )

    def test_column_missing_from_rollup_is_still_flagged(self):
        """Recognising ROLLUP must not stop the rule doing its job."""
        errors = self._errors(
            "SELECT u.name, o.order_date, SUM(o.total) "
            "FROM orders o JOIN users u ON o.user_id = u.id "
            "GROUP BY ROLLUP(u.name)"
        )

        self.assertTrue(any("order_date" in e for e in errors), errors)


class TestJoinsWithoutOnClause(unittest.TestCase):
    """USING, NATURAL and the comma form are joins, not cross products."""

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def _errors(self, sql):
        result = self.validator.validate(sql)
        return [i.message for i in result.issues if i.severity == OBQCSeverity.ERROR]

    def test_comma_join_with_where_condition(self):
        """The pre-SQL-92 form states its join in WHERE."""
        self.assertEqual(
            self._errors(
                "SELECT u.name, o.total FROM orders o, users u WHERE o.user_id = u.id"
            ),
            [],
        )

    def test_comma_join_across_three_tables(self):
        self.assertEqual(
            self._errors(
                "SELECT u.name, i.quantity "
                "FROM orders o, users u, order_items i "
                "WHERE o.user_id = u.id AND i.order_id = o.id"
            ),
            [],
        )

    def test_using_join(self):
        result = self.validator.validate(
            "SELECT quantity FROM orders JOIN order_items USING (id)"
        )

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_natural_join(self):
        result = self.validator.validate("SELECT total FROM orders NATURAL JOIN users")

        self.assertTrue(result.is_valid, [i.message for i in result.issues])

    def test_genuine_cross_product_is_still_flagged(self):
        errors = self._errors("SELECT u.name, o.total FROM orders o, users u")

        self.assertTrue(any("Cartesian" in e for e in errors), errors)

    def test_partially_joined_scope_is_flagged(self):
        """Every table must be tied in, not just some pair.

        One qualified equality used to excuse the whole FROM clause, so a
        third table joined to nothing rode along as a cross product.
        """
        errors = self._errors(
            "SELECT o.total FROM orders o, users u, shipments s "
            "WHERE o.user_id = u.id"
        )

        self.assertTrue(any("Cartesian" in e for e in errors), errors)
        self.assertTrue(any("shipments" in e for e in errors), errors)

    def test_the_unjoined_table_is_named(self):
        """The joined pair is not the problem, so it is not reported."""
        errors = self._errors(
            "SELECT o.total FROM orders o JOIN users u ON o.user_id = u.id, "
            "shipments s"
        )

        self.assertTrue(any("shipments" in e for e in errors), errors)
        self.assertFalse(any("users" in e for e in errors), errors)

    def test_theta_join_is_not_a_cross_product(self):
        """An ON clause need not be an equality to be a join.

        Reading connectivity as pairs of qualified equalities rejected
        ordinary SQL and, because this is an ERROR, blocked it.
        """
        self.assertEqual(
            self._errors(
                "SELECT o.total FROM orders o JOIN shipments s ON s.cost > o.total"
            ),
            [],
        )

    def test_on_clause_with_an_unqualified_side_is_a_join(self):
        self.assertEqual(
            self._errors(
                "SELECT users.id FROM users JOIN orders ON users.id = user_id"
            ),
            [],
        )

    def test_on_clause_naming_no_other_table_is_still_a_join(self):
        """An explicit ON is a stated join whatever the predicate says."""
        self.assertEqual(
            self._errors(
                "SELECT o.total FROM orders o JOIN shipments s ON s.cost > 10"
            ),
            [],
        )

    def test_cross_table_where_comparison_connects(self):
        """The comma form's condition need not be an equality either."""
        self.assertEqual(
            self._errors(
                "SELECT o.total FROM orders o, shipments s WHERE s.cost > o.total"
            ),
            [],
        )

    def test_unconditioned_self_join_is_flagged(self):
        """Two aliases of one table are two nodes, not one."""
        errors = self._errors("SELECT a.id FROM orders a, orders b")

        self.assertTrue(any("Cartesian" in e for e in errors), errors)

    def test_conditioned_self_join_is_not_flagged(self):
        self.assertEqual(
            self._errors("SELECT a.id FROM orders a, orders b WHERE a.id = b.id"),
            [],
        )

    def test_chain_of_conditions_connects_every_table(self):
        """Connectivity is transitive: c reaches a through b."""
        self.assertEqual(
            self._errors(
                "SELECT u.name FROM orders o, users u, order_items i "
                "WHERE o.user_id = u.id AND i.order_id = o.id"
            ),
            [],
        )

    def test_using_join_connects_to_what_precedes_it(self):
        """USING names no qualifiers, so it joins to the preceding items."""
        self.assertEqual(
            self._errors(
                "SELECT o.total FROM orders o JOIN users u ON o.user_id = u.id "
                "JOIN order_items USING (id)"
            ),
            [],
        )

    def test_where_filter_on_one_table_is_not_a_join(self):
        """A predicate must relate two tables to stand in for a join."""
        errors = self._errors(
            "SELECT u.name, o.total FROM orders o, users u WHERE o.total > 100"
        )

        self.assertTrue(any("Cartesian" in e for e in errors), errors)


class TestAliasQualifiedColumns(unittest.TestCase):
    """A column qualified by a table alias resolves to that table."""

    def setUp(self):
        self.graph, self.base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(self.graph, self.base_uri)

    def _errors(self, sql):
        result = self.validator.validate(sql)
        return [i.message for i in result.issues if i.severity == OBQCSeverity.ERROR]

    def test_bogus_column_behind_an_alias_is_caught(self):
        """Unresolved qualifiers made the aliased spelling escape the check."""
        errors = self._errors("SELECT o.nonexistent FROM orders o")

        self.assertTrue(any("nonexistent" in e for e in errors), errors)
        self.assertTrue(any("orders" in e for e in errors), errors)

    def test_real_column_behind_an_alias_passes(self):
        self.assertEqual(self._errors("SELECT o.total FROM orders o"), [])

    def test_alias_resolves_through_a_subquery(self):
        errors = self._errors(
            "SELECT u.name FROM users u "
            "WHERE u.id IN (SELECT o.nonexistent FROM orders o)"
        )

        self.assertTrue(any("nonexistent" in e for e in errors), errors)

    def test_outer_alias_is_visible_to_a_correlated_subquery(self):
        self.assertEqual(
            self._errors(
                "SELECT u.name FROM users u "
                "WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)"
            ),
            [],
        )

    def test_derived_table_alias_is_not_resolved_against_the_ontology(self):
        """A subquery's output columns are not any table's columns."""
        self.assertEqual(
            self._errors(
                "SELECT x.revenue FROM (SELECT SUM(total) AS revenue FROM orders) x"
            ),
            [],
        )

    def test_reported_column_keeps_the_alias_the_query_wrote(self):
        result = self.validator.validate("SELECT o.total FROM orders o")

        self.assertIn("o.total", result.parsed_columns)


if __name__ == "__main__":
    unittest.main()

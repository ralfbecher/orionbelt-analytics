"""Tests for OBQC (Ontology-Based Query Check) validator."""

import unittest

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

from src.obqc_validator import OBQCIssue, OBQCIssueType, OBQCSeverity, OBQCValidator


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
        """order_items -> orders -> users: every hop is a lookup."""
        result = self.validator.validate(
            "SELECT users.name, SUM(orders.total) "
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
            "SELECT u.name, COUNT(*) AS n, SUM(o.total) AS lifetime_value "
            "FROM public.order_items oi "
            "JOIN public.orders o ON oi.order_id = o.id "
            "JOIN public.users u ON o.user_id = u.id "
            "GROUP BY u.name ORDER BY SUM(o.total) DESC"
        )

        self.assertFalse(
            result.fan_trap_risk,
            [i.message for i in result.issues],
        )

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

    def test_unrelated_join_is_not_counted(self):
        """Tables with no ontology relationship cannot be judged to fan out."""
        result = self.validator.validate(
            "SELECT SUM(orders.total) FROM orders "
            "JOIN users ON orders.user_id = users.id "
            "JOIN shipments ON shipments.order_id = orders.id"
        )

        # shipments fans out from orders; users does not. One fan-out only.
        self.assertFalse(
            result.fan_trap_risk,
            [i.message for i in result.issues],
        )


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


if __name__ == "__main__":
    unittest.main()

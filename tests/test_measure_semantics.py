"""Tests for per-column measure semantics.

Additivity is a property of a column, not of the table holding it. A
denormalized table carries additive measures next to attributes, so a
table-level role cannot express what SUM() is allowed to touch -- which is
why oba:tableType stayed advisory and this went on columns.

Classification is deterministic, in two tiers. Structural findings follow
from the data type or key membership and are certain, so they block. Name
patterns are heuristics and only warn: a pattern that misreads one schema's
naming must not refuse to run its queries.
"""

import unittest

from rdflib import Graph, Namespace

from src.constants import OBA_NAMESPACE
from src.database_manager import ColumnInfo, TableInfo
from src.obqc_validator import OBQCIssueType, OBQCSeverity, OBQCValidator
from src.ontology_generator import OntologyGenerator

BASE_URI = "http://test.com/ontology/"
OBA = Namespace(OBA_NAMESPACE)


def _col(name, data_type, pk=False, fk=False):
    return ColumnInfo(
        name=name,
        data_type=data_type,
        is_nullable=True,
        is_primary_key=pk,
        is_foreign_key=fk,
    )


# One denormalized table holding facts and dimensions side by side.
MIXED_TABLE = TableInfo(
    name="sales",
    schema="public",
    columns=[
        _col("sale_id", "INTEGER", pk=True),
        _col("customer_id", "INTEGER", fk=True),
        _col("region_name", "TEXT"),
        _col("sale_date", "DATE"),
        _col("amount", "DECIMAL"),
        _col("quantity", "INTEGER"),
        _col("unit_price", "DECIMAL"),
        _col("account_balance", "DECIMAL"),
        _col("mystery_number", "DECIMAL"),
    ],
    primary_keys=["sale_id"],
    foreign_keys=[],
)


class TestStructuralClassification(unittest.TestCase):
    """Tier 1: certain, from schema metadata alone."""

    def setUp(self):
        self.gen = OntologyGenerator(BASE_URI)

    def _classify(self, column):
        return self.gen.classify_measure(column)

    def test_primary_key_is_an_attribute(self):
        kind, basis, _ = self._classify(_col("sale_id", "INTEGER", pk=True))
        self.assertEqual((kind, basis), ("attribute", "structural"))

    def test_foreign_key_is_an_attribute(self):
        kind, basis, _ = self._classify(_col("customer_id", "INTEGER", fk=True))
        self.assertEqual((kind, basis), ("attribute", "structural"))

    def test_text_column_is_an_attribute(self):
        kind, basis, _ = self._classify(_col("region_name", "TEXT"))
        self.assertEqual((kind, basis), ("attribute", "structural"))

    def test_date_column_is_an_attribute(self):
        kind, basis, _ = self._classify(_col("sale_date", "DATE"))
        self.assertEqual((kind, basis), ("attribute", "structural"))

    def test_structural_wins_over_a_measure_sounding_name(self):
        """A key named like a measure is still a key."""
        kind, basis, _ = self._classify(_col("amount_id", "INTEGER", pk=True))
        self.assertEqual((kind, basis), ("attribute", "structural"))


class TestNamePatternClassification(unittest.TestCase):
    """Tier 2: heuristic, and marked as such."""

    def setUp(self):
        self.gen = OntologyGenerator(BASE_URI)

    def _kind(self, name, data_type="DECIMAL"):
        result = self.gen.classify_measure(_col(name, data_type))
        return result[0] if result else None

    def test_additive_measures(self):
        for name in ("amount", "total_revenue", "quantity", "line_cost"):
            self.assertEqual(self._kind(name), "additive", name)

    def test_non_additive_measures(self):
        for name in ("unit_price", "tax_rate", "discount_pct", "profit_margin"):
            self.assertEqual(self._kind(name), "non_additive", name)

    def test_semi_additive_measures(self):
        for name in ("account_balance", "stock_level", "inventory_on_hand"):
            self.assertEqual(self._kind(name), "semi_additive", name)

    def test_non_additive_beats_additive_on_overlap(self):
        """unit_cost contains "cost" but a per-unit value is not additive."""
        self.assertEqual(self._kind("unit_cost"), "non_additive")

    def test_basis_is_recorded_as_name_pattern(self):
        _, basis, reason = self.gen.classify_measure(_col("unit_price", "DECIMAL"))
        self.assertEqual(basis, "name_pattern")
        self.assertIn("unit_price", reason)

    def test_unmatched_numeric_column_is_left_unset(self):
        """Not "assume additive" -- consumers read these as fact."""
        self.assertIsNone(self.gen.classify_measure(_col("mystery_number", "DECIMAL")))


class TestOntologyEmission(unittest.TestCase):
    def setUp(self):
        ttl = OntologyGenerator(BASE_URI).generate_from_schema([MIXED_TABLE])
        self.ttl = ttl
        self.g = Graph()
        self.g.parse(data=ttl, format="turtle")
        self.ns = Namespace(BASE_URI)

    def _measure(self, column):
        return self.g.value(self.ns[f"sales_{column}"], OBA.measureType)

    def test_one_table_carries_several_measure_kinds(self):
        """The whole point: a table is not one role."""
        kinds = {
            str(self._measure(c))
            for c in ("sale_id", "amount", "unit_price", "account_balance")
        }
        self.assertEqual(
            kinds, {"attribute", "additive", "non_additive", "semi_additive"}
        )

    def test_basis_and_reason_are_emitted(self):
        uri = self.ns["sales_unit_price"]
        self.assertEqual(str(self.g.value(uri, OBA.measureBasis)), "name_pattern")
        self.assertIsNotNone(self.g.value(uri, OBA.measureReason))

    def test_unclassified_column_gets_no_triples(self):
        uri = self.ns["sales_mystery_number"]
        self.assertIsNone(self.g.value(uri, OBA.measureType))
        self.assertIsNone(self.g.value(uri, OBA.measureBasis))

    def test_shacl_conformance(self):
        from src.shacl_validator import shacl_available, validate_ontology

        if not shacl_available():
            self.skipTest("pyshacl not installed")
        result = validate_ontology(self.ttl)
        self.assertTrue(result["conforms"], result.get("report"))


class TestOBQCMeasureRule(unittest.TestCase):
    """Judged without any join: SUM(unit_price) is wrong from one table."""

    def setUp(self):
        ttl = OntologyGenerator(BASE_URI).generate_from_schema([MIXED_TABLE])
        g = Graph()
        g.parse(data=ttl, format="turtle")
        self.validator = OBQCValidator()
        self.validator.load_ontology(g, BASE_URI)

    def _findings(self, sql):
        return [
            i
            for i in self.validator.validate(sql).issues
            if i.issue_type is OBQCIssueType.INVALID_AGGREGATION
        ]

    def test_summing_an_additive_measure_is_clean(self):
        self.assertEqual(self._findings("SELECT SUM(amount) FROM sales"), [])

    def test_summing_a_key_is_a_blocking_error(self):
        found = self._findings("SELECT SUM(sale_id) FROM sales")
        self.assertEqual(found[0].severity, OBQCSeverity.ERROR)

    def test_summing_a_text_column_is_a_blocking_error(self):
        found = self._findings("SELECT SUM(region_name) FROM sales")
        self.assertEqual(found[0].severity, OBQCSeverity.ERROR)

    def test_summing_a_non_additive_measure_only_warns(self):
        """A name pattern must not refuse to run a query."""
        found = self._findings("SELECT SUM(unit_price) FROM sales")
        self.assertEqual(found[0].severity, OBQCSeverity.WARNING)

    def test_semi_additive_sum_is_allowed(self):
        """Summable across entities; only across time is it wrong, and the
        query does not say."""
        self.assertEqual(self._findings("SELECT SUM(account_balance) FROM sales"), [])

    def test_avg_of_a_non_additive_measure_is_fine(self):
        self.assertEqual(self._findings("SELECT AVG(unit_price) FROM sales"), [])

    def test_min_max_and_count_are_not_judged(self):
        for sql in (
            "SELECT MAX(unit_price) FROM sales",
            "SELECT MIN(unit_price) FROM sales",
            "SELECT COUNT(sale_id) FROM sales",
        ):
            self.assertEqual(self._findings(sql), [], sql)

    def test_unclassified_column_is_never_reported(self):
        self.assertEqual(self._findings("SELECT SUM(mystery_number) FROM sales"), [])

    def test_qualified_and_aliased_references_resolve(self):
        self.assertTrue(self._findings("SELECT SUM(s.sale_id) FROM sales s"))

    def test_reason_travels_with_the_finding(self):
        found = self._findings("SELECT SUM(unit_price) FROM sales")
        self.assertIn("unit_price", found[0].suggestion)
        self.assertIn("override", found[0].suggestion)


class TestCTEShadowingIsNotJudged(unittest.TestCase):
    """A WITH alias may shadow a base table, and its columns are the query's
    own. Resolving against the same-named ontology table blocked valid CTEs."""

    def setUp(self):
        ttl = OntologyGenerator(BASE_URI).generate_from_schema([MIXED_TABLE])
        g = Graph()
        g.parse(data=ttl, format="turtle")
        self.validator = OBQCValidator()
        self.validator.load_ontology(g, BASE_URI)

    def _findings(self, sql):
        return [
            i
            for i in self.validator.validate(sql).issues
            if i.issue_type is OBQCIssueType.INVALID_AGGREGATION
        ]

    def test_qualified_reference_to_a_shadowing_cte_is_not_judged(self):
        self.assertEqual(
            self._findings(
                "WITH sales AS (SELECT 1 AS sale_id) "
                "SELECT SUM(sales.sale_id) FROM sales"
            ),
            [],
        )

    def test_unqualified_reference_with_a_cte_in_scope_is_not_judged(self):
        self.assertEqual(
            self._findings(
                "WITH sales AS (SELECT 1 AS sale_id) SELECT SUM(sale_id) FROM sales"
            ),
            [],
        )

    def test_the_real_table_is_still_judged_outside_the_cte(self):
        """Shadowing must not disable the rule everywhere."""
        self.assertTrue(self._findings("SELECT SUM(sale_id) FROM sales"))


class TestInferredForeignKeys(unittest.TestCase):
    """Several supported engines carry no FK metadata at all."""

    def _validator(self, tables):
        ttl = OntologyGenerator(BASE_URI).generate_from_schema(tables)
        g = Graph()
        g.parse(data=ttl, format="turtle")
        validator = OBQCValidator()
        validator.load_ontology(g, BASE_URI)
        return validator

    def test_inferred_fk_is_an_attribute(self):
        """customer_id -> customers is inferred, though never declared."""
        sales = TableInfo(
            name="sales",
            schema="public",
            columns=[_col("id", "INTEGER", pk=True), _col("customer_id", "INTEGER")],
            primary_keys=["id"],
            foreign_keys=[],
        )
        customers = TableInfo(
            name="customers",
            schema="public",
            columns=[_col("id", "INTEGER", pk=True), _col("name", "TEXT")],
            primary_keys=["id"],
            foreign_keys=[],
        )
        validator = self._validator([sales, customers])
        found = [
            i
            for i in validator.validate("SELECT SUM(customer_id) FROM sales").issues
            if i.issue_type is OBQCIssueType.INVALID_AGGREGATION
        ]
        self.assertTrue(found)
        self.assertEqual(found[0].severity, OBQCSeverity.ERROR)
        self.assertIn("Inferred foreign key", found[0].suggestion)


class TestNumericTypeMatchingIsWholeWord(unittest.TestCase):
    """ "int" is a substring of POINT, INTERVAL and GEOPOINT."""

    def setUp(self):
        self.gen = OntologyGenerator(BASE_URI)

    def test_geometry_types_are_attributes(self):
        for sql_type in ("POINT", "GEOPOINT", "POLYGON", "GEOGRAPHY"):
            result = self.gen.classify_measure(_col("location", sql_type))
            self.assertIsNotNone(result, sql_type)
            self.assertEqual((result[0], result[1]), ("attribute", "structural"))

    def test_interval_is_an_attribute(self):
        result = self.gen.classify_measure(_col("duration", "INTERVAL"))
        self.assertEqual(result[0], "attribute")

    def test_real_numeric_types_still_match(self):
        for sql_type in (
            "INTEGER",
            "BIGINT",
            "DECIMAL(18,2)",
            "DOUBLE PRECISION",
            "NUMERIC",
        ):
            result = self.gen.classify_measure(_col("amount", sql_type))
            self.assertEqual(result[0], "additive", sql_type)


class TestTableTypeAllowsMixed(unittest.TestCase):
    def test_mixed_is_a_valid_table_type(self):
        from src.shacl_validator import shacl_available, validate_ontology

        if not shacl_available():
            self.skipTest("pyshacl not installed")

        ttl = OntologyGenerator(BASE_URI).generate_from_schema([MIXED_TABLE])
        ttl += (
            "\n<http://test.com/ontology/sales> "
            '<https://ralforion.com/ns/oba#tableType> "mixed" .\n'
        )
        result = validate_ontology(ttl)
        self.assertTrue(result["conforms"], result.get("report"))


if __name__ == "__main__":
    unittest.main()

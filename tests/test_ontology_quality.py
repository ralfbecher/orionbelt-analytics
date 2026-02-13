"""Tests for ontology quality improvements."""

import unittest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

from src.ontology_generator import (
    OntologyGenerator,
    OntologyQualityReport,
    InferredRelationship,
    DenormalizedField
)
from src.database_manager import TableInfo, ColumnInfo


class TestInferredRelationships(unittest.TestCase):
    """Test suite for implicit FK relationship inference."""

    def setUp(self):
        self.generator = OntologyGenerator()

    def _create_test_tables(self):
        """Create test tables for relationship inference testing."""
        # Countries table (target of inferred FK)
        countries = TableInfo(
            name="countries",
            schema="public",
            columns=[
                ColumnInfo(
                    name="countryid",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="countryname",
                    data_type="VARCHAR(100)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["countryid"],
            foreign_keys=[]
        )

        # Suppliers table (has implicit FK to countries)
        suppliers = TableInfo(
            name="suppliers",
            schema="public",
            columns=[
                ColumnInfo(
                    name="supplierid",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="suppliername",
                    data_type="VARCHAR(100)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                ),
                # This should be inferred as FK to countries
                ColumnInfo(
                    name="suppliercountryid",
                    data_type="INTEGER",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False  # Not declared as FK
                )
            ],
            primary_keys=["supplierid"],
            foreign_keys=[]  # No declared FKs
        )

        # Orders table with declared FK
        orders = TableInfo(
            name="orders",
            schema="public",
            columns=[
                ColumnInfo(
                    name="orderid",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="supplierid",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=True,
                    foreign_key_table="suppliers",
                    foreign_key_column="supplierid"
                )
            ],
            primary_keys=["orderid"],
            foreign_keys=[{
                "column": "supplierid",
                "referenced_table": "suppliers",
                "referenced_column": "supplierid"
            }]
        )

        return [countries, suppliers, orders]

    def test_infer_implicit_relationship_suffix_id(self):
        """Test inference of FK from column name ending in table name + 'id'."""
        tables = self._create_test_tables()
        self.generator._build_table_lookup(tables)

        inferred = self.generator._infer_implicit_relationships(tables)

        # Should find suppliercountryid -> countries
        country_rels = [r for r in inferred if r.target_table == "countries"]
        self.assertEqual(len(country_rels), 1)
        self.assertEqual(country_rels[0].source_table, "suppliers")
        self.assertEqual(country_rels[0].column, "suppliercountryid")
        self.assertEqual(country_rels[0].target_column, "countryid")

    def test_skip_declared_fk(self):
        """Test that declared FKs are not duplicated as inferred."""
        tables = self._create_test_tables()
        self.generator._build_table_lookup(tables)

        inferred = self.generator._infer_implicit_relationships(tables)

        # Should NOT find orders.supplierid -> suppliers (already declared)
        supplier_rels = [
            r for r in inferred
            if r.source_table == "orders" and r.target_table == "suppliers"
        ]
        self.assertEqual(len(supplier_rels), 0)

    def test_skip_primary_key(self):
        """Test that primary key columns are not inferred as FKs."""
        tables = self._create_test_tables()
        self.generator._build_table_lookup(tables)

        inferred = self.generator._infer_implicit_relationships(tables)

        # Should NOT find supplierid as FK (it's a PK)
        self_rels = [r for r in inferred if r.source_table == r.target_table]
        self.assertEqual(len(self_rels), 0)

    def test_confidence_calculation(self):
        """Test that confidence levels are calculated correctly."""
        tables = self._create_test_tables()
        self.generator._build_table_lookup(tables)

        inferred = self.generator._infer_implicit_relationships(tables)

        # suppliercountryid -> countries should have high confidence
        # (INT type matches, column ends with 'id')
        country_rel = next(r for r in inferred if r.target_table == "countries")
        self.assertIn(country_rel.confidence, ["high", "medium"])

    def test_inferred_relationship_added_to_ontology(self):
        """Test that inferred relationships are added to the ontology graph."""
        tables = self._create_test_tables()

        # Generate ontology with inferred relationships
        self.generator.generate_from_schema(tables, include_inferred_relationships=True)

        # Check that the inferred relationship exists
        db_ns = self.generator.db_ns
        base = self.generator.base_uri

        rel_uri = base["suppliers_has_countries"]

        # Should be an ObjectProperty
        self.assertIn(
            (rel_uri, RDF.type, OWL.ObjectProperty),
            self.generator.graph
        )

        # Should be marked as inferred
        self.assertIn(
            (rel_uri, db_ns.isInferredRelationship, Literal(True)),
            self.generator.graph
        )


class TestSemanticTypeMapping(unittest.TestCase):
    """Test suite for semantic-aware type mapping."""

    def setUp(self):
        self.generator = OntologyGenerator()

    def test_quantity_column_override(self):
        """Test that quantity columns get xsd:integer even if stored as DOUBLE."""
        # Quantity column stored as DOUBLE
        xsd_type, override = self.generator._map_sql_to_xsd(
            "DOUBLE PRECISION", "orderquantity", "orders"
        )

        self.assertEqual(xsd_type, XSD.integer)
        self.assertIsNotNone(override)
        self.assertEqual(override["overridden_xsd_type"], "xsd:integer")
        self.assertIn("quantity", override["reason"])

    def test_qty_abbreviation_override(self):
        """Test that 'qty' abbreviation triggers integer override."""
        xsd_type, override = self.generator._map_sql_to_xsd(
            "NUMERIC(10,2)", "product_qty", "inventory"
        )

        self.assertEqual(xsd_type, XSD.integer)
        self.assertIsNotNone(override)

    def test_count_column_override(self):
        """Test that count columns get xsd:integer."""
        xsd_type, override = self.generator._map_sql_to_xsd(
            "FLOAT", "item_count", "summary"
        )

        self.assertEqual(xsd_type, XSD.integer)
        self.assertIsNotNone(override)

    def test_non_quantity_double_unchanged(self):
        """Test that non-quantity DOUBLE columns remain xsd:double."""
        xsd_type, override = self.generator._map_sql_to_xsd(
            "DOUBLE PRECISION", "unit_price", "products"
        )

        self.assertEqual(xsd_type, XSD.double)
        self.assertIsNone(override)

    def test_integer_columns_no_override(self):
        """Test that already-integer columns don't get overrides."""
        xsd_type, override = self.generator._map_sql_to_xsd(
            "INTEGER", "quantity", "orders"
        )

        self.assertEqual(xsd_type, XSD.integer)
        self.assertIsNone(override)  # No override needed

    def test_type_override_in_ontology(self):
        """Test that type overrides are recorded in ontology generation."""
        table = TableInfo(
            name="orders",
            schema="public",
            columns=[
                ColumnInfo(
                    name="orderquantity",
                    data_type="DOUBLE PRECISION",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=[],
            foreign_keys=[]
        )

        self.generator.generate_from_schema([table])

        # Check quality report
        report = self.generator.get_quality_report()
        self.assertIsNotNone(report)
        self.assertEqual(len(report.type_overrides), 1)
        self.assertEqual(report.type_overrides[0]["column"], "orderquantity")


class TestDenormalizedFieldDetection(unittest.TestCase):
    """Test suite for denormalized field detection."""

    def setUp(self):
        self.generator = OntologyGenerator()

    def _create_test_tables(self):
        """Create test tables with denormalized fields."""
        clients = TableInfo(
            name="clients",
            schema="public",
            columns=[
                ColumnInfo(
                    name="clientid",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="clientname",
                    data_type="VARCHAR(100)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["clientid"],
            foreign_keys=[]
        )

        # Shipments with denormalized client name
        shipments = TableInfo(
            name="shipments",
            schema="public",
            columns=[
                ColumnInfo(
                    name="shipmentid",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="clientid",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=True,
                    foreign_key_table="clients",
                    foreign_key_column="clientid"
                ),
                # Denormalized field - stores client name redundantly
                ColumnInfo(
                    name="shipmentclient",
                    data_type="VARCHAR(100)",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["shipmentid"],
            foreign_keys=[{
                "column": "clientid",
                "referenced_table": "clients",
                "referenced_column": "clientid"
            }]
        )

        return [clients, shipments]

    def test_detect_denormalized_field(self):
        """Test detection of denormalized text fields."""
        tables = self._create_test_tables()

        denormalized = self.generator._detect_denormalized_fields(tables)

        # Should find shipmentclient as denormalized
        self.assertEqual(len(denormalized), 1)
        self.assertEqual(denormalized[0].table, "shipments")
        self.assertEqual(denormalized[0].column, "shipmentclient")
        self.assertEqual(denormalized[0].likely_source_table, "clients")

    def test_skip_non_text_columns(self):
        """Test that non-text columns are not flagged as denormalized."""
        tables = TableInfo(
            name="orders",
            schema="public",
            columns=[
                ColumnInfo(
                    name="clientcount",  # Contains 'client' but is numeric
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=[],
            foreign_keys=[]
        )

        clients = TableInfo(
            name="clients",
            schema="public",
            columns=[],
            primary_keys=[],
            foreign_keys=[]
        )

        denormalized = self.generator._detect_denormalized_fields([tables, clients])

        # Should not flag clientcount (it's an integer, not denormalized text)
        self.assertEqual(len(denormalized), 0)

    def test_denormalized_annotation_in_ontology(self):
        """Test that denormalized fields are annotated in the ontology."""
        tables = self._create_test_tables()

        self.generator.generate_from_schema(tables, annotate_denormalized=True)

        # Check that the denormalized annotation exists
        db_ns = self.generator.db_ns
        base = self.generator.base_uri

        prop_uri = base["shipments_shipmentclient"]

        self.assertIn(
            (prop_uri, db_ns.isDenormalized, Literal(True)),
            self.generator.graph
        )
        self.assertIn(
            (prop_uri, db_ns.likelySourceTable, Literal("clients")),
            self.generator.graph
        )


class TestEnrichmentCompleteness(unittest.TestCase):
    """Test suite for enrichment completeness validation."""

    def setUp(self):
        self.generator = OntologyGenerator()

    def test_completeness_with_no_descriptions(self):
        """Test completeness report when no descriptions are present."""
        table = TableInfo(
            name="users",
            schema="public",
            columns=[
                ColumnInfo(
                    name="userid",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="username",
                    data_type="VARCHAR(50)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["userid"],
            foreign_keys=[]
        )

        self.generator.generate_from_schema([table])
        report = self.generator.validate_enrichment_completeness()

        # Should report all items as missing descriptions
        self.assertEqual(report["total_classes"], 1)
        self.assertEqual(len(report["classes_without_description"]), 1)
        self.assertEqual(report["total_properties"], 2)
        self.assertEqual(len(report["properties_without_description"]), 2)
        self.assertLess(report["overall_coverage"], 1.0)

    def test_completeness_with_descriptions(self):
        """Test completeness report when descriptions are present."""
        table = TableInfo(
            name="users",
            schema="public",
            columns=[
                ColumnInfo(
                    name="userid",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False,
                    comment="Unique user identifier"
                )
            ],
            primary_keys=["userid"],
            foreign_keys=[],
            comment="Table storing user information"
        )

        self.generator.generate_from_schema([table])
        report = self.generator.validate_enrichment_completeness()

        # Should report items with descriptions as covered
        self.assertEqual(len(report["classes_without_description"]), 0)
        self.assertEqual(len(report["properties_without_description"]), 0)
        self.assertEqual(report["class_coverage"], 1.0)
        self.assertEqual(report["property_coverage"], 1.0)


class TestQualityReport(unittest.TestCase):
    """Test suite for the OntologyQualityReport dataclass."""

    def test_quality_report_to_dict(self):
        """Test serialization of quality report to dictionary."""
        report = OntologyQualityReport(
            inferred_relationships=[
                InferredRelationship(
                    source_table="suppliers",
                    column="countryid",
                    target_table="countries",
                    target_column="id",
                    confidence="high",
                    pattern_matched="suffix_id"
                )
            ],
            denormalized_fields=[
                DenormalizedField(
                    table="orders",
                    column="clientname",
                    likely_source_table="clients",
                    data_type="VARCHAR(100)",
                    warning="Test warning"
                )
            ],
            type_overrides=[
                {"column": "quantity", "overridden_xsd_type": "xsd:integer"}
            ],
            warnings=["Test warning"]
        )

        result = report.to_dict()

        self.assertEqual(result["summary"]["inferred_relationship_count"], 1)
        self.assertEqual(result["summary"]["denormalized_field_count"], 1)
        self.assertEqual(result["summary"]["type_override_count"], 1)
        self.assertEqual(result["summary"]["warning_count"], 1)

    def test_empty_quality_report(self):
        """Test empty quality report serialization."""
        report = OntologyQualityReport()
        result = report.to_dict()

        self.assertEqual(result["summary"]["inferred_relationship_count"], 0)
        self.assertEqual(result["summary"]["denormalized_field_count"], 0)


class TestTPCDSPatterns(unittest.TestCase):
    """Test suite for TPC-DS style naming patterns (_sk suffix)."""

    def setUp(self):
        self.generator = OntologyGenerator()

    def _create_tpcds_tables(self):
        """Create TPC-DS style test tables."""
        customer = TableInfo(
            name="CUSTOMER",
            schema="TPCDS",
            columns=[
                ColumnInfo(
                    name="c_customer_sk",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="c_customer_id",
                    data_type="VARCHAR(16)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["c_customer_sk"],
            foreign_keys=[]
        )

        item = TableInfo(
            name="ITEM",
            schema="TPCDS",
            columns=[
                ColumnInfo(
                    name="i_item_sk",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=True,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="i_item_id",
                    data_type="VARCHAR(16)",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=["i_item_sk"],
            foreign_keys=[]
        )

        store_sales = TableInfo(
            name="STORE_SALES",
            schema="TPCDS",
            columns=[
                ColumnInfo(
                    name="ss_sold_date_sk",
                    data_type="INTEGER",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="ss_customer_sk",
                    data_type="INTEGER",
                    is_nullable=True,
                    is_primary_key=False,
                    is_foreign_key=False
                ),
                ColumnInfo(
                    name="ss_item_sk",
                    data_type="INTEGER",
                    is_nullable=False,
                    is_primary_key=False,
                    is_foreign_key=False
                )
            ],
            primary_keys=[],
            foreign_keys=[]
        )

        return [customer, item, store_sales]

    def test_tpcds_sk_pattern_detection(self):
        """Test detection of TPC-DS style _sk foreign keys."""
        tables = self._create_tpcds_tables()
        self.generator._build_table_lookup(tables)

        inferred = self.generator._infer_implicit_relationships(tables)

        # Should find ss_customer_sk -> CUSTOMER
        customer_rels = [r for r in inferred if r.target_table == "CUSTOMER"]
        self.assertGreaterEqual(len(customer_rels), 1)
        self.assertEqual(customer_rels[0].source_table, "STORE_SALES")
        self.assertEqual(customer_rels[0].column, "ss_customer_sk")

        # Should find ss_item_sk -> ITEM
        item_rels = [r for r in inferred if r.target_table == "ITEM"]
        self.assertGreaterEqual(len(item_rels), 1)
        self.assertEqual(item_rels[0].source_table, "STORE_SALES")
        self.assertEqual(item_rels[0].column, "ss_item_sk")

    def test_tpcds_relationships_in_ontology(self):
        """Test that TPC-DS relationships are added to the ontology."""
        tables = self._create_tpcds_tables()

        self.generator.generate_from_schema(tables, include_inferred_relationships=True)

        db_ns = self.generator.db_ns
        base = self.generator.base_uri

        # Check STORE_SALES -> CUSTOMER relationship
        rel_uri = base["STORE_SALES_has_CUSTOMER"]
        self.assertIn(
            (rel_uri, RDF.type, OWL.ObjectProperty),
            self.generator.graph
        )


class TestTableLookup(unittest.TestCase):
    """Test suite for table name lookup with pluralization handling."""

    def setUp(self):
        self.generator = OntologyGenerator()

    def test_singular_to_plural(self):
        """Test finding plural table from singular name."""
        tables = [
            TableInfo(name="countries", schema="public", columns=[], primary_keys=[], foreign_keys=[])
        ]
        self.generator._build_table_lookup(tables)

        # Should find 'countries' from 'country'
        result = self.generator._find_matching_table("country")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "countries")

    def test_plural_to_singular(self):
        """Test finding singular table from plural name."""
        tables = [
            TableInfo(name="user", schema="public", columns=[], primary_keys=[], foreign_keys=[])
        ]
        self.generator._build_table_lookup(tables)

        # Should find 'user' from 'users'
        result = self.generator._find_matching_table("users")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "user")

    def test_direct_match(self):
        """Test direct name matching."""
        tables = [
            TableInfo(name="products", schema="public", columns=[], primary_keys=[], foreign_keys=[])
        ]
        self.generator._build_table_lookup(tables)

        result = self.generator._find_matching_table("products")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "products")

    def test_no_match(self):
        """Test when no matching table exists."""
        tables = [
            TableInfo(name="orders", schema="public", columns=[], primary_keys=[], foreign_keys=[])
        ]
        self.generator._build_table_lookup(tables)

        result = self.generator._find_matching_table("customers")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)

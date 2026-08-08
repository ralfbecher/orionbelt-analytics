"""Tests for modelling database views in the ontology.

Views are emitted as oba:View, not owl:Class, and their columns as
oba:ViewColumn, not owl:DatatypeProperty. That typing is load-bearing rather
than cosmetic: every consumer that reads tables looks for owl:Class carrying
oba:tableName, so a view modelled as an ordinary class would be picked up as
a base table by each of them and would have to be excluded again, rule by
rule, forever. These tests pin the separation at both ends -- what is emitted,
and what OBQC sees.
"""

import unittest

from rdflib import Graph, Namespace
from rdflib.namespace import OWL, RDF, RDFS

from src.constants import OBA_NAMESPACE
from src.database_manager import ColumnInfo, TableInfo, ViewInfo
from src.graphrag.manager import _annotate_view_sources
from src.obqc_validator import OBQCSeverity, OBQCValidator
from src.ontology_generator import OntologyGenerator

BASE_URI = "http://test.com/ontology/"
OBA = Namespace(OBA_NAMESPACE)
NS = Namespace(BASE_URI)


def _col(name, data_type="INTEGER", **kw):
    return ColumnInfo(
        name=name,
        data_type=data_type,
        is_nullable=kw.get("is_nullable", True),
        is_primary_key=kw.get("is_primary_key", False),
        is_foreign_key=kw.get("is_foreign_key", False),
    )


TABLES = [
    TableInfo(
        name="sales",
        schema="public",
        columns=[_col("id"), _col("clientid"), _col("amount", "DECIMAL")],
        primary_keys=["id"],
        foreign_keys=[],
    ),
    TableInfo(
        name="clients",
        schema="public",
        columns=[_col("clientid"), _col("name", "TEXT")],
        primary_keys=["clientid"],
        foreign_keys=[],
    ),
]

VIEW = ViewInfo(
    name="v_revenue",
    schema="public",
    definition="SELECT c.name, SUM(s.amount) AS total_revenue FROM sales s "
    "JOIN clients c ON s.clientid = c.clientid GROUP BY c.name",
    columns=[_col("name", "TEXT"), _col("total_revenue", "DECIMAL")],
    source_tables=["sales", "clients"],
)


def _generate(views=None):
    ttl = OntologyGenerator(BASE_URI).generate_from_schema(
        TABLES, views_info=views if views is not None else [VIEW]
    )
    graph = Graph()
    graph.parse(data=ttl, format="turtle")
    return ttl, graph


class TestViewTyping(unittest.TestCase):
    """The separation is structural: views are simply not classes."""

    def setUp(self):
        self.ttl, self.g = _generate()
        self.view_uri = NS["v_revenue"]

    def test_view_is_typed_oba_view(self):
        self.assertIn((self.view_uri, RDF.type, OBA.View), self.g)

    def test_view_is_not_an_owl_class(self):
        """An owl:Class would be picked up by every table consumer."""
        self.assertNotIn((self.view_uri, RDF.type, OWL.Class), self.g)

    def test_view_carries_view_name_not_table_name(self):
        self.assertEqual(str(self.g.value(self.view_uri, OBA.viewName)), "v_revenue")
        self.assertIsNone(self.g.value(self.view_uri, OBA.tableName))

    def test_view_columns_are_typed_oba_view_column(self):
        cols = set(self.g.subjects(RDF.type, OBA.ViewColumn))
        self.assertEqual(len(cols), 2)
        for col in cols:
            self.assertNotIn((col, RDF.type, OWL.DatatypeProperty), self.g)
            self.assertIsNone(self.g.value(col, OBA.tableName))
            self.assertEqual(str(self.g.value(col, OBA.viewName)), "v_revenue")

    def test_view_declares_no_primary_key(self):
        """A view has no key, and asserting one invites join validation."""
        self.assertIsNone(self.g.value(self.view_uri, OBA.primaryKey))

    def test_definition_and_label_are_carried(self):
        self.assertIn(
            "total_revenue", str(self.g.value(self.view_uri, OBA.viewDefinition))
        )
        self.assertEqual(str(self.g.value(self.view_uri, RDFS.label)), "v_revenue")

    def test_tables_are_still_owl_classes(self):
        self.assertIn((NS["sales"], RDF.type, OWL.Class), self.g)


class TestDerivedFrom(unittest.TestCase):
    def test_lineage_points_at_source_tables(self):
        _, g = _generate()
        sources = set(g.objects(NS["v_revenue"], OBA.derivedFrom))
        self.assertEqual(sources, {NS["sales"], NS["clients"]})

    def test_unknown_sources_are_omitted_not_guessed(self):
        """A dangling link would be consumed as fact by anything reasoning
        over provenance, so an unresolvable source is left out entirely."""
        view = ViewInfo(
            name="v_external",
            schema="public",
            source_tables=["sales", "table_in_another_schema"],
        )
        _, g = _generate([view])
        sources = set(g.objects(NS["v_external"], OBA.derivedFrom))
        self.assertEqual(sources, {NS["sales"]})

    def test_no_lineage_emits_no_links(self):
        view = ViewInfo(name="v_opaque", schema="public", source_tables=[])
        _, g = _generate([view])
        self.assertEqual(list(g.objects(NS["v_opaque"], OBA.derivedFrom)), [])


class TestViewSourcesExcludeSelf(unittest.TestCase):
    """DuckDB returns the whole CREATE VIEW statement."""

    def test_create_target_is_not_a_source(self):
        views = _annotate_view_sources(
            [
                {
                    "name": "v_rev",
                    "definition": "CREATE VIEW v_rev AS SELECT id FROM sales",
                }
            ],
            dialect="duckdb",
        )
        self.assertEqual(views[0]["referenced_tables"], ["sales"])
        self.assertNotIn("v_rev", views[0]["referenced_tables"])

    def test_dialect_is_honoured(self):
        """A Snowflake body read as PostgreSQL fails to parse, and the view
        then silently carries no sources at all."""
        snowflake_body = "SELECT v:customer.name::string AS nm FROM raw_events"
        as_snowflake = _annotate_view_sources(
            [{"name": "v", "definition": snowflake_body}], dialect="snowflake"
        )
        self.assertEqual(as_snowflake[0]["referenced_tables"], ["raw_events"])


class TestOBQCReadsViewsFromOntology(unittest.TestCase):
    def setUp(self):
        _, g = _generate()
        self.validator = OBQCValidator()
        self.validator.load_ontology(g, BASE_URI)

    def _errors(self, sql):
        return [
            i.message
            for i in self.validator.validate(sql).issues
            if i.severity == OBQCSeverity.ERROR
        ]

    def test_views_do_not_leak_into_the_table_cache(self):
        """The whole point of the typing: table rules cannot see views."""
        self.assertNotIn("v_revenue", self.validator._schema_cache.tables)
        self.assertEqual(
            sorted(self.validator._schema_cache.tables), ["clients", "sales"]
        )

    def test_view_is_registered_with_its_catalog_columns(self):
        self.assertEqual(
            self.validator._known_views["v_revenue"], {"name", "total_revenue"}
        )

    def test_querying_the_view_is_allowed(self):
        self.assertEqual(self._errors("SELECT total_revenue FROM v_revenue"), [])

    def test_bad_view_column_is_caught(self):
        self.assertTrue(self._errors("SELECT v_revenue.nope FROM v_revenue"))

    def test_joining_the_view_is_still_blocked(self):
        errors = self._errors("SELECT * FROM v_revenue v JOIN sales s ON s.id = v.name")
        self.assertTrue(any("cannot be joined" in e for e in errors), errors)

    def test_a_table_only_ontology_leaves_views_registered(self):
        """Loading an ontology without views must not silently drop a set the
        session had already discovered."""
        self.validator.load_views({"v_session": {"a"}})
        ttl = OntologyGenerator(BASE_URI).generate_from_schema(TABLES)
        graph = Graph()
        graph.parse(data=ttl, format="turtle")
        self.validator.load_ontology(graph, BASE_URI)
        self.assertIn("v_session", self.validator._known_views)


class TestShaclConformance(unittest.TestCase):
    def test_generated_ontology_with_views_conforms(self):
        from src.shacl_validator import shacl_available, validate_ontology

        if not shacl_available():
            self.skipTest("pyshacl not installed")

        ttl, _ = _generate()
        result = validate_ontology(ttl)
        self.assertTrue(result["conforms"], result.get("report"))


if __name__ == "__main__":
    unittest.main()

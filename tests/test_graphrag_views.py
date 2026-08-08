"""Tests for database view discovery and GraphRAG indexing.

Views are indexed for search but deliberately kept out of the ontology: a view
pre-joins its sources, so an OWL class for it would restate concepts the base
tables already model. These tests pin both halves -- that views reach the
index, and that they carry the vocabulary that made indexing them worthwhile.
"""

import unittest

import numpy as np

from src.database_manager import ViewInfo
from src.drivers.base import DatabaseDriver
from src.graphrag.embedder import SchemaEmbedder
from src.graphrag.manager import _annotate_view_sources
from src.session import SchemaCache

TABLES = [
    {
        "name": "sales",
        "columns": [
            {"name": "salesamount", "data_type": "DECIMAL"},
            {"name": "unitcost", "data_type": "DECIMAL"},
            {"name": "clientid", "data_type": "INTEGER"},
        ],
        "foreign_keys": [],
    },
    {
        "name": "clients",
        "columns": [
            {"name": "clientid", "data_type": "INTEGER"},
            {"name": "name", "data_type": "TEXT"},
        ],
        "foreign_keys": [],
    },
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


class TestViewSourceExtraction(unittest.TestCase):
    """_annotate_view_sources recovers base tables from a view body."""

    def test_extracts_joined_tables(self):
        views = [
            {
                "name": "v_revenue_by_client",
                "definition": (
                    "SELECT c.name, SUM(s.amount) AS total_revenue "
                    "FROM sales s JOIN clients c ON s.clientid = c.clientid "
                    "GROUP BY c.name"
                ),
            }
        ]
        result = _annotate_view_sources(views)
        self.assertEqual(result[0]["referenced_tables"], ["clients", "sales"])

    def test_cte_names_are_not_sources(self):
        """A WITH alias is defined by the view itself, not a base table."""
        views = [
            {
                "name": "v_margin",
                "definition": (
                    "WITH base AS (SELECT id, amount FROM sales) "
                    "SELECT id, amount FROM base"
                ),
            }
        ]
        result = _annotate_view_sources(views)
        self.assertEqual(result[0]["referenced_tables"], ["sales"])
        self.assertNotIn("base", result[0]["referenced_tables"])

    def test_unparseable_definition_degrades(self):
        """View bodies come back dialect-specific; a parse failure is not fatal."""
        views = [{"name": "v_broken", "definition": "NOT SQL AT ALL {{{"}]
        result = _annotate_view_sources(views)
        self.assertEqual(result[0]["referenced_tables"], [])

    def test_missing_definition_degrades(self):
        """PostgreSQL returns NULL bodies to non-owners; index the view anyway."""
        views = [{"name": "v_no_body", "definition": None}]
        result = _annotate_view_sources(views)
        self.assertEqual(result[0]["referenced_tables"], [])

    def test_none_returns_empty_list(self):
        self.assertEqual(_annotate_view_sources(None), [])

    def test_existing_sources_are_not_overwritten(self):
        views = [
            {
                "name": "v",
                "definition": "SELECT 1 FROM t",
                "referenced_tables": ["kept"],
            }
        ]
        self.assertEqual(
            _annotate_view_sources(views)[0]["referenced_tables"], ["kept"]
        )


class TestViewEmbedding(unittest.TestCase):
    """Views become searchable elements of their own type."""

    def setUp(self):
        self.embedder = SchemaEmbedder(embedding_model="tfidf")

    def test_view_element_type_and_metadata(self):
        self.embedder.batch_embed_schema(TABLES, [])  # fit the vectorizer
        element = self.embedder.create_view_embedding(
            view_name="v_revenue",
            definition="SELECT SUM(amount) AS total_revenue FROM sales",
            referenced_tables=["sales"],
        )
        self.assertEqual(element.element_type, "view")
        self.assertEqual(element.name, "v_revenue")
        self.assertTrue(element.metadata["is_view"])
        self.assertEqual(element.metadata["referenced_tables"], ["sales"])

    def test_batch_embed_returns_views_bucket(self):
        views = [{"name": "v_revenue", "definition": "SELECT 1 FROM sales"}]
        result = self.embedder.batch_embed_schema(TABLES, _annotate_view_sources(views))
        self.assertEqual(len(result["views"]), 1)
        self.assertEqual(len(result["tables"]), 2)

    def test_views_are_optional(self):
        """Existing callers pass no views and must keep working."""
        result = self.embedder.batch_embed_schema(TABLES)
        self.assertEqual(result["views"], [])


class TestViewVocabularyReachability(unittest.TestCase):
    """The point of indexing views: vocabulary the schema does not contain."""

    QUERY = "which clients have the best profit margin"
    VIEWS = [
        {
            "name": "v_profit_margin",
            "definition": (
                "SELECT clientid, (salesamount - unitcost) / salesamount "
                "AS profit_margin FROM sales"
            ),
        }
    ]

    def _top_hit(self, views):
        embedder = SchemaEmbedder(embedding_model="tfidf")
        out = embedder.batch_embed_schema(TABLES, _annotate_view_sources(views))
        elements = out["tables"] + out["columns"] + out["views"]
        query_vec = embedder._embed_text(self.QUERY)
        ranked = sorted(
            ((_cosine(query_vec, e.embedding), e) for e in elements),
            key=lambda pair: -pair[0],
        )
        return ranked[0][1], query_vec

    def test_view_becomes_the_top_hit(self):
        """Without the view, "profit margin" matches nothing that means it."""
        without, _ = self._top_hit(None)
        self.assertNotEqual(without.element_type, "view")

        with_views, _ = self._top_hit([dict(v) for v in self.VIEWS])
        self.assertEqual(with_views.element_type, "view")
        self.assertEqual(with_views.name, "v_profit_margin")

    def test_view_body_enters_the_tfidf_vocabulary(self):
        """Regression: the fitting corpus must use the same underscore
        splitting as create_view_embedding. Fitting on "profit_margin" while
        embedding "profit margin" leaves the term unmatchable either way, and
        the query vector stays as sparse as it was without views at all."""
        _, without_vec = self._top_hit(None)
        _, with_vec = self._top_hit([dict(v) for v in self.VIEWS])
        self.assertGreater(
            int(np.count_nonzero(with_vec)), int(np.count_nonzero(without_vec))
        )


class TestDriverViewContract(unittest.TestCase):
    """get_views is concrete, so adding it broke no existing driver."""

    def test_base_driver_returns_empty_mapping(self):
        class MinimalDriver(DatabaseDriver):
            db_type = "minimal"

            def connect(self, **params):
                return True

            def get_schemas(self):
                return []

            def get_tables(self, schema_name=None):
                return []

            def analyze_table(self, table_name, schema_name=None):
                return None

            def validate_sql_syntax(self, sql_query, validation_result):
                return validation_result

            def execute_sql_query(self, sql_query, limit=1000):
                return {}

            def sample_table_data(self, table_name, schema_name=None, limit=10):
                return []

            def test_connection(self):
                return True

            def disconnect(self):
                return None

        self.assertEqual(MinimalDriver().get_views(), {})


class TestViewInfo(unittest.TestCase):
    def test_from_dict_roundtrip(self):
        view = ViewInfo.from_dict(
            {"name": "v", "schema": "public", "definition": "SELECT 1"}
        )
        self.assertEqual(view.name, "v")
        self.assertEqual(view.schema, "public")
        self.assertEqual(view.definition, "SELECT 1")

    def test_definition_is_optional(self):
        self.assertIsNone(ViewInfo.from_dict({"name": "v"}).definition)


class TestSchemaCacheViews(unittest.TestCase):
    """Views cache separately from tables so they never reach the ontology."""

    def test_cache_and_retrieve(self):
        cache = SchemaCache()
        views = [ViewInfo(name="v", schema="public")]
        cache.cache_views("public", views)
        self.assertEqual(cache.get_cached_views("public"), views)

    def test_missing_schema_returns_empty_list(self):
        self.assertEqual(SchemaCache().get_cached_views("nope"), [])

    def test_views_do_not_appear_in_the_table_cache(self):
        cache = SchemaCache()
        cache.cache_views("public", [ViewInfo(name="v", schema="public")])
        self.assertIsNone(cache.get_cached_schema("public"))

    def test_clear_all_drops_views(self):
        """Views must not outlive the tables they were discovered with."""
        cache = SchemaCache()
        cache.cache_views("public", [ViewInfo(name="v", schema="public")])
        cache.clear()
        self.assertEqual(cache.get_all_cached_views(), [])

    def test_clear_one_schema_drops_only_its_views(self):
        cache = SchemaCache()
        cache.cache_views("public", [ViewInfo(name="pub_v", schema="public")])
        cache.cache_views("other", [ViewInfo(name="other_v", schema="other")])
        cache.clear("public")
        remaining = [v.name for v in cache.get_all_cached_views()]
        self.assertEqual(remaining, ["other_v"])

    def test_get_all_spans_schemas(self):
        cache = SchemaCache()
        cache.cache_views("a", [ViewInfo(name="va", schema="a")])
        cache.cache_views("b", [ViewInfo(name="vb", schema="b")])
        self.assertEqual(
            sorted(v.name for v in cache.get_all_cached_views()), ["va", "vb"]
        )


if __name__ == "__main__":
    unittest.main()

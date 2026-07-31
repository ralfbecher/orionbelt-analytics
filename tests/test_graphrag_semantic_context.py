"""Tests for client-supplied semantic context in the GraphRAG index.

Schema search can only match vocabulary the schema itself contains. These
tests pin the gap that motivates the feature -- a concept expressed only in
business terms is unreachable from abbreviated column names -- and verify that
indexing context closes it without disturbing the schema elements themselves.
"""

import os
import sys
from pathlib import Path

import pytest

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from graphrag.manager import GraphRAGManager  # noqa: E402

TABLES = [
    {
        "name": "sales",
        "schema": "public",
        "columns": [
            {
                "name": "salesamount",
                "data_type": "DECIMAL",
                "is_nullable": False,
                "is_primary_key": False,
                "is_foreign_key": False,
                "foreign_key_table": None,
            },
            {
                "name": "unitcost",
                "data_type": "DECIMAL",
                "is_nullable": False,
                "is_primary_key": False,
                "is_foreign_key": False,
                "foreign_key_table": None,
            },
        ],
        "foreign_keys": [],
    },
    {
        "name": "banks",
        "schema": "public",
        "columns": [
            {
                "name": "balanceamt",
                "data_type": "DECIMAL",
                "is_nullable": False,
                "is_primary_key": False,
                "is_foreign_key": False,
                "foreign_key_table": None,
            }
        ],
        "foreign_keys": [],
    },
]


def _minilm_is_cached() -> bool:
    """Whether MiniLM is usable without triggering a ~79MB download."""
    if os.getenv("ORIONBELT_TEST_EMBEDDING_DOWNLOAD") == "1":
        return True
    cache = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
    return cache.exists()


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """An initialized manager backed by the JSON store, isolated per test."""
    monkeypatch.setenv("GRAPHRAG_EMBEDDING_MODEL", "tfidf")
    monkeypatch.setattr("graphrag.manager.CHROMADB_AVAILABLE", False)
    mgr = GraphRAGManager(connection_id="test", schema_name="public")
    mgr.initialize_from_schema(tables_info=TABLES, schema_name="public")
    return mgr


class TestValidation:
    """Bad input is rejected before anything is indexed."""

    def test_requires_initialization(self, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_EMBEDDING_MODEL", "tfidf")
        monkeypatch.setattr("graphrag.manager.CHROMADB_AVAILABLE", False)
        mgr = GraphRAGManager(connection_id="test", schema_name="public")

        with pytest.raises(RuntimeError, match="not initialized"):
            mgr.add_semantic_context("sales.salesamount", "revenue")

    @pytest.mark.parametrize(
        ("target", "context"),
        [
            ("", "revenue per line item"),
            ("   ", "revenue per line item"),
            ("sales.salesamount", ""),
            ("sales.salesamount", "   "),
        ],
    )
    def test_blank_arguments_rejected(self, manager, target, context):
        with pytest.raises(ValueError):
            manager.add_semantic_context(target, context)


class TestIndexing:
    """What gets written to the store."""

    def test_returns_summary(self, manager):
        result = manager.add_semantic_context(
            "sales.salesamount", "Revenue per line item."
        )

        assert result["element_id"] == "semantic_context:sales.salesamount"
        assert result["target"] == "sales.salesamount"
        assert result["characters"] == len("Revenue per line item.")

    def test_stored_as_its_own_element_type(self, manager):
        manager.add_semantic_context("sales.salesamount", "Revenue per line item.")

        stored = manager.vector_store.get_by_id("semantic_context:sales.salesamount")

        assert stored is not None
        assert stored.element_type == "semantic_context"
        assert stored.metadata["target"] == "sales.salesamount"
        assert stored.metadata["source"] == "client"

    def test_does_not_modify_the_schema_element(self, manager):
        """Re-running discovery must not be able to clobber the context, and
        the column's own description must stay as discovery wrote it."""
        before = manager.vector_store.get_by_id("sales.salesamount")
        assert before is not None
        original = before.description

        manager.add_semantic_context("sales.salesamount", "Profit margin driver.")

        after = manager.vector_store.get_by_id("sales.salesamount")
        assert after is not None
        assert after.description == original

    def test_target_is_embedded_with_the_prose(self, manager):
        """Searching the column name should still reach its context."""
        manager.add_semantic_context("sales.salesamount", "Profit margin driver.")

        stored = manager.vector_store.get_by_id("semantic_context:sales.salesamount")

        assert stored is not None
        assert "sales salesamount" in stored.description
        assert "Profit margin driver." in stored.description

    def test_whitespace_is_trimmed(self, manager):
        result = manager.add_semantic_context(
            "  sales.unitcost  ", "  Cost of goods sold.  "
        )

        assert result["target"] == "sales.unitcost"
        stored = manager.vector_store.get_by_id("semantic_context:sales.unitcost")
        assert stored is not None
        assert stored.metadata["context"] == "Cost of goods sold."


@pytest.mark.skipif(
    not _minilm_is_cached(),
    reason=(
        "MiniLM model not cached; set ORIONBELT_TEST_EMBEDDING_DOWNLOAD=1 to "
        "allow the ~79MB download"
    ),
)
class TestSearchReachability:
    """The point of the feature: making an unreachable concept findable."""

    @pytest.fixture
    def semantic_manager(self, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_EMBEDDING_MODEL", "minilm")
        monkeypatch.setattr("graphrag.manager.CHROMADB_AVAILABLE", False)
        mgr = GraphRAGManager(connection_id="test", schema_name="public")
        mgr.initialize_from_schema(tables_info=TABLES, schema_name="public")
        return mgr

    def test_context_becomes_the_top_hit_for_its_concept(self, semantic_manager):
        """'profit margin' appears in no column name, so only the added
        context can carry it.

        The raw columns do score something on their own -- the model relates
        cost and amount to margin -- so the assertion is that the context wins
        by a clear margin, not merely that it appears.
        """
        query = "which product lines have the best profit margin"

        before = semantic_manager.search_schema(query, top_k=3)
        best_schema_score = before[0]["similarity_score"]

        semantic_manager.add_semantic_context(
            "sales.salesamount",
            "Revenue per line item. Profit margin is salesamount minus "
            "unitcost. Used for profitability analysis.",
        )
        after = semantic_manager.search_schema(query, top_k=3)

        assert after[0]["element"]["id"] == "semantic_context:sales.salesamount"
        assert after[0]["similarity_score"] > best_schema_score * 1.5

    def test_context_can_be_filtered_out(self, semantic_manager):
        semantic_manager.add_semantic_context(
            "sales.salesamount", "Profit margin driver."
        )

        results = semantic_manager.search_schema(
            "profit margin", top_k=5, element_type="column"
        )

        assert all(r["element"]["type"] == "column" for r in results)

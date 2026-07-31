"""Tests for mirroring applied semantic names into the GraphRAG index.

Semantic name suggestions carry exactly what schema search lacks -- the
business word for an abbreviated identifier and a sentence explaining it. They
were applied to the ontology only, so GraphRAG (whose vectors come from raw
schema metadata) never saw them and a user asking in the enriched vocabulary
still matched nothing.
"""

import pytest

from src.handlers.ontology_semantic import semantic_context_entries


class TestSemanticContextEntries:
    """Flattening suggestions into (target, context) pairs."""

    def test_class_becomes_a_table_target(self):
        entries = semantic_context_entries(
            {
                "classes": [
                    {
                        "original_name": "fct_sls",
                        "suggested_name": "Sales Fact",
                        "description": "One row per sold line item.",
                    }
                ]
            }
        )

        assert entries == [("fct_sls", "Sales Fact. One row per sold line item.")]

    def test_property_is_qualified_by_its_table(self):
        """Unqualified columns would collapse two tables' like-named fields."""
        entries = semantic_context_entries(
            {
                "properties": [
                    {
                        "original_name": "amt",
                        "table_name": "sales",
                        "suggested_name": "Sales Amount",
                        "description": "Revenue per line item.",
                    }
                ]
            }
        )

        assert entries == [("sales.amt", "Sales Amount. Revenue per line item.")]

    def test_property_without_table_stays_unqualified(self):
        entries = semantic_context_entries(
            {
                "properties": [
                    {"original_name": "amt", "suggested_name": "Amount"},
                ]
            }
        )

        assert entries == [("amt", "Amount")]

    def test_all_three_kinds_are_collected(self):
        entries = semantic_context_entries(
            {
                "classes": [{"original_name": "t1", "suggested_name": "Table One"}],
                "properties": [{"original_name": "c1", "suggested_name": "Column One"}],
                "relationships": [{"original_name": "r1", "suggested_name": "Rel One"}],
            }
        )

        assert [t for t, _ in entries] == ["t1", "c1", "r1"]

    def test_description_only_is_kept(self):
        """A description alone still adds vocabulary worth indexing."""
        entries = semantic_context_entries(
            {
                "classes": [
                    {
                        "original_name": "fct_sls",
                        "description": "One row per sold line item.",
                    }
                ]
            }
        )

        assert entries == [("fct_sls", "One row per sold line item.")]

    @pytest.mark.parametrize(
        "suggestion",
        [
            pytest.param({"suggested_name": "X"}, id="no-original-name"),
            pytest.param({"original_name": "  "}, id="blank-original-name"),
            pytest.param({"original_name": "t1"}, id="nothing-to-say"),
            pytest.param(
                {"original_name": "t1", "suggested_name": "", "description": ""},
                id="empty-strings",
            ),
        ],
    )
    def test_entries_carrying_no_vocabulary_are_skipped(self, suggestion):
        assert semantic_context_entries({"classes": [suggestion]}) == []

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({}, id="empty"),
            pytest.param({"classes": None}, id="null-list"),
            pytest.param({"classes": ["not-a-dict"]}, id="non-dict-entry"),
            pytest.param({"unknown_kind": [{"original_name": "x"}]}, id="unknown-key"),
        ],
    )
    def test_malformed_payloads_do_not_raise(self, payload):
        """Suggestions are LLM-authored, so the shape is not guaranteed."""
        assert semantic_context_entries(payload) == []

    def test_whitespace_is_trimmed(self):
        entries = semantic_context_entries(
            {
                "properties": [
                    {
                        "original_name": "  amt  ",
                        "table_name": "  sales  ",
                        "suggested_name": "  Sales Amount  ",
                        "description": "  Revenue.  ",
                    }
                ]
            }
        )

        assert entries == [("sales.amt", "Sales Amount. Revenue.")]

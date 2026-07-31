"""Tests for mirroring applied semantic names into the GraphRAG index.

Semantic name suggestions carry exactly what schema search lacks -- the
business word for an abbreviated identifier and a sentence explaining it. They
were applied to the ontology only, so GraphRAG (whose vectors come from raw
schema metadata) never saw them and a user asking in the enriched vocabulary
still matched nothing.

What gets mirrored must be what the ontology *accepted*, not what the caller
asked for: the generator silently skips names it cannot find, while the index
accepts any target, so propagating the request would invent schema elements.
"""

import pytest

from src.database_manager import ColumnInfo, TableInfo
from src.handlers.ontology_semantic import semantic_context_entries
from src.ontology_generator import OntologyGenerator


class TestSemanticContextEntries:
    """Flattening applied suggestions into (target, context) pairs."""

    def test_class_becomes_a_table_target(self):
        entries = semantic_context_entries(
            [
                {
                    "original_name": "fct_sls",
                    "suggested_name": "Sales Fact",
                    "description": "One row per sold line item.",
                }
            ]
        )

        assert entries == [("fct_sls", "Sales Fact. One row per sold line item.")]

    def test_property_is_qualified_by_its_table(self):
        """Unqualified columns would collapse two tables' like-named fields."""
        entries = semantic_context_entries(
            [
                {
                    "original_name": "amt",
                    "table_name": "sales",
                    "suggested_name": "Sales Amount",
                    "description": "Revenue per line item.",
                }
            ]
        )

        assert entries == [("sales.amt", "Sales Amount. Revenue per line item.")]

    def test_property_without_table_stays_unqualified(self):
        entries = semantic_context_entries(
            [{"original_name": "amt", "suggested_name": "Amount"}]
        )

        assert entries == [("amt", "Amount")]

    def test_order_is_preserved(self):
        entries = semantic_context_entries(
            [
                {"original_name": "t1", "suggested_name": "Table One"},
                {"original_name": "c1", "suggested_name": "Column One"},
                {"original_name": "r1", "suggested_name": "Rel One"},
            ]
        )

        assert [t for t, _ in entries] == ["t1", "c1", "r1"]

    def test_description_only_is_kept(self):
        """A description alone still adds vocabulary worth indexing."""
        entries = semantic_context_entries(
            [
                {
                    "original_name": "fct_sls",
                    "description": "One row per sold line item.",
                }
            ]
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
        assert semantic_context_entries([suggestion]) == []

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param([], id="empty"),
            pytest.param(None, id="none"),
            pytest.param(["not-a-dict"], id="non-dict-entry"),
        ],
    )
    def test_malformed_payloads_do_not_raise(self, payload):
        """Suggestions originate from an LLM, so the shape is not guaranteed."""
        assert semantic_context_entries(payload) == []

    def test_whitespace_is_trimmed(self):
        entries = semantic_context_entries(
            [
                {
                    "original_name": "  amt  ",
                    "table_name": "  sales  ",
                    "suggested_name": "  Sales Amount  ",
                    "description": "  Revenue.  ",
                }
            ]
        )

        assert entries == [("sales.amt", "Sales Amount. Revenue.")]


@pytest.fixture
def generator():
    """A generator holding a two-column 'sales' ontology."""
    gen = OntologyGenerator()
    gen.generate_from_schema(
        [
            TableInfo(
                name="sales",
                schema="public",
                columns=[
                    ColumnInfo(
                        name="amt",
                        data_type="DECIMAL",
                        is_nullable=False,
                        is_primary_key=False,
                        is_foreign_key=False,
                    ),
                    ColumnInfo(
                        name="salesid",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=True,
                        is_foreign_key=False,
                    ),
                ],
                primary_keys=["salesid"],
                foreign_keys=[],
            )
        ]
    )
    return gen


class TestAppliedSemanticNames:
    """Only names the ontology matched may be propagated."""

    def test_applied_names_are_reported(self, generator):
        generator.apply_semantic_names(
            {
                "classes": [
                    {
                        "original_name": "sales",
                        "suggested_name": "Sale",
                        "description": "A sold line item.",
                    }
                ]
            }
        )

        applied = generator.applied_semantic_names()

        assert [a["original_name"] for a in applied] == ["sales"]
        assert applied[0]["suggested_name"] == "Sale"

    def test_unmatched_class_is_not_reported(self, generator):
        """The generator skips it silently; indexing it would invent a table."""
        generator.apply_semantic_names(
            {
                "classes": [
                    {"original_name": "does_not_exist", "suggested_name": "Ghost"}
                ]
            }
        )

        assert generator.applied_semantic_names() == []

    def test_property_under_the_wrong_table_is_not_reported(self, generator):
        """The reported failure: a table_name typo'd to a table that has no
        such column must not become a searchable context."""
        generator.apply_semantic_names(
            {
                "properties": [
                    {
                        "original_name": "amt",
                        "table_name": "sale",  # typo: the table is 'sales'
                        "suggested_name": "Sales Amount",
                    }
                ]
            }
        )

        assert generator.applied_semantic_names() == []
        assert semantic_context_entries(generator.applied_semantic_names()) == []

    def test_matched_property_is_reported_with_its_table(self, generator):
        generator.apply_semantic_names(
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

        assert semantic_context_entries(generator.applied_semantic_names()) == [
            ("sales.amt", "Sales Amount. Revenue per line item.")
        ]

    def test_mixed_batch_reports_only_the_matches(self, generator):
        generator.apply_semantic_names(
            {
                "classes": [
                    {"original_name": "sales", "suggested_name": "Sale"},
                    {"original_name": "nope", "suggested_name": "Ghost"},
                ]
            }
        )

        assert [a["original_name"] for a in generator.applied_semantic_names()] == [
            "sales"
        ]

    def test_reset_between_applies(self, generator):
        """A later apply must not report the previous one's names."""
        generator.apply_semantic_names(
            {"classes": [{"original_name": "sales", "suggested_name": "Sale"}]}
        )
        generator.apply_semantic_names(
            {"classes": [{"original_name": "nope", "suggested_name": "Ghost"}]}
        )

        assert generator.applied_semantic_names() == []

    def test_empty_before_any_apply(self):
        assert OntologyGenerator().applied_semantic_names() == []

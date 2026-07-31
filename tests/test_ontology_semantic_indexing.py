"""Tests for mirroring applied semantic names into the GraphRAG index.

Semantic name suggestions carry exactly what schema search lacks -- the
business word for an abbreviated identifier and a sentence explaining it. They
were applied to the ontology only, so GraphRAG (whose vectors come from raw
schema metadata) never saw them and a user asking in the enriched vocabulary
still matched nothing.

Two things decide whether a mirrored entry is real rather than invented:

- it must be a name the ontology *accepted*, not merely one that was requested;
- it must be keyed by the raw SQL name, not the URI-safe name the suggestion
  carries, because URIs are built with ``_clean_name`` and ``order-items``
  becomes ``order_items``.
"""

import pytest

from src.database_manager import ColumnInfo, TableInfo
from src.handlers.ontology_semantic import semantic_context_entries
from src.ontology_generator import OntologyGenerator


class TestSemanticContextEntries:
    """Building (target, context) pairs from applied suggestions."""

    def test_table_only_entry_targets_the_table(self):
        entries = semantic_context_entries(
            [
                {
                    "table_name": "fct_sls",
                    "suggested_name": "Sales Fact",
                    "description": "One row per sold line item.",
                }
            ]
        )

        assert entries == [("fct_sls", "Sales Fact. One row per sold line item.")]

    def test_column_entry_is_qualified_by_its_table(self):
        """Unqualified columns would collapse two tables' like-named fields."""
        entries = semantic_context_entries(
            [
                {
                    "table_name": "sales",
                    "column_name": "amt",
                    "suggested_name": "Sales Amount",
                    "description": "Revenue per line item.",
                }
            ]
        )

        assert entries == [("sales.amt", "Sales Amount. Revenue per line item.")]

    def test_relationship_entry_uses_the_pair_of_tables(self):
        """Matches the id GraphRAG builds for a relationship element."""
        entries = semantic_context_entries(
            [
                {
                    "table_name": "sales",
                    "related_table": "customers",
                    "suggested_name": "Buyer",
                }
            ]
        )

        assert entries == [("sales__to__customers", "Buyer")]

    def test_entry_without_a_resolvable_target_is_skipped(self):
        """No annotation to anchor it: inventing a target is the failure this
        whole path exists to avoid."""
        entries = semantic_context_entries(
            [{"original_name": "orphan", "suggested_name": "Orphan"}]
        )

        assert entries == []

    def test_order_is_preserved(self):
        entries = semantic_context_entries(
            [
                {"table_name": "t1", "suggested_name": "Table One"},
                {"table_name": "t1", "column_name": "c1", "suggested_name": "Col One"},
                {"table_name": "t1", "related_table": "t2", "suggested_name": "Rel"},
            ]
        )

        assert [t for t, _ in entries] == ["t1", "t1.c1", "t1__to__t2"]

    def test_description_only_is_kept(self):
        """A description alone still adds vocabulary worth indexing."""
        entries = semantic_context_entries(
            [
                {
                    "table_name": "fct_sls",
                    "description": "One row per sold line item.",
                }
            ]
        )

        assert entries == [("fct_sls", "One row per sold line item.")]

    @pytest.mark.parametrize(
        "suggestion",
        [
            pytest.param({"suggested_name": "X"}, id="no-target"),
            pytest.param({"table_name": "  "}, id="blank-table"),
            pytest.param({"table_name": "t1"}, id="nothing-to-say"),
            pytest.param(
                {"table_name": "t1", "suggested_name": "", "description": ""},
                id="empty-strings",
            ),
        ],
    )
    def test_entries_carrying_no_vocabulary_or_target_are_skipped(self, suggestion):
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
                    "table_name": "  sales  ",
                    "column_name": "  amt  ",
                    "suggested_name": "  Sales Amount  ",
                    "description": "  Revenue.  ",
                }
            ]
        )

        assert entries == [("sales.amt", "Sales Amount. Revenue.")]


def _generator(table="sales", column="amt"):
    """A generator holding a one-column ontology for *table*."""
    gen = OntologyGenerator()
    gen.generate_from_schema(
        [
            TableInfo(
                name=table,
                schema="public",
                columns=[
                    ColumnInfo(
                        name=column,
                        data_type="DECIMAL",
                        is_nullable=False,
                        is_primary_key=False,
                        is_foreign_key=False,
                    ),
                    ColumnInfo(
                        name="rowid",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=True,
                        is_foreign_key=False,
                    ),
                ],
                primary_keys=["rowid"],
                foreign_keys=[],
            )
        ]
    )
    return gen


@pytest.fixture
def generator():
    return _generator()


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

        assert [a["table_name"] for a in applied] == ["sales"]
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
        """A table_name typo'd to a table that has no such column must not
        become a searchable context."""
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

        assert [a["table_name"] for a in generator.applied_semantic_names()] == [
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


class TestRawNamesSurviveUriCleaning:
    """Targets must key off the SQL name, not the URI-safe one.

    Class and property URIs are built with ``_clean_name``, so a table called
    ``order-items`` becomes ``order_items`` -- and suggestions, being generated
    from the ontology, carry that cleaned form. Recording it would index a
    context for a table that does not exist.
    """

    def test_class_target_uses_the_original_table_name(self):
        gen = _generator(table="order-items", column="qty-ordered")

        gen.apply_semantic_names(
            {
                "classes": [
                    {"original_name": "order_items", "suggested_name": "Order Line"}
                ]
            }
        )

        assert semantic_context_entries(gen.applied_semantic_names()) == [
            ("order-items", "Order Line")
        ]

    def test_property_target_uses_the_original_names(self):
        gen = _generator(table="order-items", column="qty-ordered")

        gen.apply_semantic_names(
            {
                "properties": [
                    {
                        "original_name": "qty_ordered",
                        "table_name": "order_items",
                        "suggested_name": "Quantity",
                    }
                ]
            }
        )

        assert semantic_context_entries(gen.applied_semantic_names()) == [
            ("order-items.qty-ordered", "Quantity")
        ]

    def test_dotted_table_name_is_preserved(self):
        """_clean_name also rewrites dots, which would split the target."""
        gen = _generator(table="sales.fact", column="amt")

        gen.apply_semantic_names(
            {"classes": [{"original_name": "sales_fact", "suggested_name": "Sales"}]}
        )

        assert semantic_context_entries(gen.applied_semantic_names()) == [
            ("sales.fact", "Sales")
        ]

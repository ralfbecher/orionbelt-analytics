"""Tests for OBQC's handling of database views.

Views are real objects that a query may legitimately name, but they are
deliberately absent from the ontology. Before views were registered, OBQC
rejected every query against one -- "Table 'v_x' not found in ontology" -- the
same false-positive shape already fixed for catalog tables (#83) and CTEs
(#87), and OBQC errors block execution.

The governing rule for columns: check them only when the view's output is
knowable from its definition. Guessing wrong in the permissive direction costs
a missed error; guessing wrong the other way blocks a correct query.
"""

import unittest

from src.obqc_validator import OBQCSeverity, OBQCValidator, derive_view_columns
from tests.test_obqc_validator import create_sample_ontology_graph

VIEWS = {
    "v_active_users": "SELECT id, name FROM users WHERE active = true",
    "v_star": "SELECT * FROM users",
    "v_unparseable": "NOT SQL AT ALL {{{",
    "v_no_body": None,
}


def _errors(result):
    return [i for i in result.issues if i.severity == OBQCSeverity.ERROR]


class TestDeriveViewColumns(unittest.TestCase):
    def test_explicit_projection(self):
        self.assertEqual(
            derive_view_columns("SELECT id, name FROM users"), {"id", "name"}
        )

    def test_aliases_are_the_output_names(self):
        self.assertEqual(
            derive_view_columns("SELECT SUM(amount) AS total_revenue FROM sales"),
            {"total_revenue"},
        )

    def test_star_is_not_derivable(self):
        """The output of SELECT * depends on tables resolved at creation."""
        self.assertEqual(derive_view_columns("SELECT * FROM users"), set())

    def test_qualified_star_is_not_derivable(self):
        self.assertEqual(
            derive_view_columns("SELECT u.*, 1 AS extra FROM users u"), set()
        )

    def test_unparseable_is_not_derivable(self):
        self.assertEqual(derive_view_columns("NOT SQL {{{"), set())

    def test_missing_definition_is_not_derivable(self):
        self.assertEqual(derive_view_columns(None), set())

    def test_case_is_normalised(self):
        self.assertEqual(
            derive_view_columns("SELECT ID, NaMe FROM users"), {"id", "name"}
        )


class OBQCViewTestCase(unittest.TestCase):
    def setUp(self):
        graph, base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(graph, base_uri)
        self.validator.load_views_from_definitions(dict(VIEWS))


class TestViewsAreNotBlocked(OBQCViewTestCase):
    def test_query_against_a_view_is_allowed(self):
        """Regression: this was an ERROR, and OBQC errors block execution."""
        result = self.validator.validate("SELECT id, name FROM v_active_users")
        self.assertEqual(_errors(result), [])
        self.assertTrue(result.is_valid)

    def test_view_joined_to_a_base_table_is_allowed(self):
        result = self.validator.validate(
            "SELECT v.name, o.id FROM v_active_users v "
            "JOIN orders o ON o.user_id = v.id"
        )
        self.assertEqual(_errors(result), [])

    def test_unknown_table_is_still_an_error(self):
        """Registering views must not turn the existence rule off."""
        result = self.validator.validate("SELECT * FROM totally_unknown_thing")
        messages = [i.message for i in _errors(result)]
        self.assertTrue(any("totally_unknown_thing" in m for m in messages), messages)


class TestViewColumnChecking(OBQCViewTestCase):
    def test_qualified_valid_column_passes(self):
        result = self.validator.validate("SELECT v_active_users.id FROM v_active_users")
        self.assertEqual(_errors(result), [])

    def test_qualified_bad_column_is_reported_against_the_view(self):
        result = self.validator.validate(
            "SELECT v_active_users.nope FROM v_active_users"
        )
        messages = [i.message for i in _errors(result)]
        self.assertTrue(any("not found in view" in m for m in messages), messages)

    def test_unqualified_bad_column_is_reported(self):
        result = self.validator.validate("SELECT nope FROM v_active_users")
        self.assertTrue(_errors(result))

    def test_unqualified_valid_column_resolves_through_the_view(self):
        result = self.validator.validate("SELECT name FROM v_active_users")
        self.assertEqual(_errors(result), [])


class TestUnderivableViewsAreUnchecked(OBQCViewTestCase):
    """When the output list is not knowable, checking it would invent errors."""

    def test_star_view_columns_are_not_checked(self):
        result = self.validator.validate("SELECT anything_at_all FROM v_star")
        self.assertEqual(_errors(result), [])

    def test_unparseable_view_columns_are_not_checked(self):
        result = self.validator.validate("SELECT whatever FROM v_unparseable")
        self.assertEqual(_errors(result), [])

    def test_bodyless_view_columns_are_not_checked(self):
        """PostgreSQL withholds the body from non-owners; still not an error."""
        result = self.validator.validate("SELECT x FROM v_no_body")
        self.assertEqual(_errors(result), [])

    def test_qualified_column_on_a_star_view_is_not_checked(self):
        result = self.validator.validate("SELECT v_star.anything FROM v_star")
        self.assertEqual(_errors(result), [])


class TestViewRegistration(unittest.TestCase):
    def setUp(self):
        graph, base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(graph, base_uri)

    def test_no_views_registered_keeps_previous_behaviour(self):
        result = self.validator.validate("SELECT * FROM v_active_users")
        self.assertTrue(_errors(result))

    def test_names_are_matched_case_insensitively(self):
        self.validator.load_views_from_definitions(
            {"V_Active_Users": "SELECT id FROM users"}
        )
        result = self.validator.validate("SELECT id FROM v_active_users")
        self.assertEqual(_errors(result), [])

    def test_definitions_are_reparsed_only_on_change(self):
        """Re-registering the same views is a no-op; a changed set is not."""
        self.validator.load_views_from_definitions({"v": "SELECT id FROM users"})
        first = self.validator._known_views
        self.validator.load_views_from_definitions({"v": "SELECT id FROM users"})
        self.assertIs(self.validator._known_views, first)

        self.validator.load_views_from_definitions({"v": "SELECT name FROM users"})
        self.assertEqual(self.validator._known_views["v"], {"name"})

    def test_views_discovered_later_are_picked_up(self):
        self.assertTrue(_errors(self.validator.validate("SELECT id FROM v_late")))
        self.validator.load_views_from_definitions({"v_late": "SELECT id FROM users"})
        self.assertEqual(_errors(self.validator.validate("SELECT id FROM v_late")), [])


if __name__ == "__main__":
    unittest.main()

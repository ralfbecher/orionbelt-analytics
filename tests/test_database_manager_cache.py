"""Tests for DatabaseManager metadata-cache invalidation.

Cache keys are built from the operation and schema name only -- nothing in
them identifies the connection. So swapping the driver without clearing the
cache answers the new connection's questions with the old connection's
metadata for the whole TTL, and the caller cannot tell.

This was found for get_views() but was never specific to views: get_tables()
had the same hole since long before views existed.
"""

import unittest

from src.database_manager import DatabaseManager


class FakeDriver:
    """Minimal driver stand-in exposing only what the cached methods call."""

    db_type = "fake"

    def __init__(self, tables, views):
        self._tables = tables
        self._views = views

    def get_tables(self, schema_name=None):
        return self._tables

    def get_views(self, schema_name=None):
        return self._views

    def test_connection(self):
        return True

    def disconnect(self):
        pass


class TestDriverSwapInvalidatesCache(unittest.TestCase):
    def setUp(self):
        self.dm = DatabaseManager()
        # Bypass engine/health checks; this is about caching, not connecting.
        self.dm._ensure_connection = lambda: None
        self.dm._dremio_rest_connection = None
        self.dm._activate_driver(FakeDriver(["old_table"], {"v_old": "SELECT 1"}))

    def test_views_do_not_survive_a_driver_swap(self):
        self.assertEqual([v.name for v in self.dm.get_views("public")], ["v_old"])

        self.dm._activate_driver(FakeDriver(["new_table"], {}))

        self.assertEqual(self.dm.get_views("public"), [])

    def test_tables_do_not_survive_a_driver_swap(self):
        """The same hole, predating views entirely."""
        self.assertEqual(self.dm.get_tables("public"), ["old_table"])

        self.dm._activate_driver(FakeDriver(["new_table"], {}))

        self.assertEqual(self.dm.get_tables("public"), ["new_table"])

    def test_new_connection_views_replace_the_old_ones(self):
        self.dm.get_views("public")

        self.dm._activate_driver(FakeDriver([], {"v_new": "SELECT 2"}))

        self.assertEqual([v.name for v in self.dm.get_views("public")], ["v_new"])

    def test_cache_still_serves_repeat_calls_on_one_connection(self):
        """Invalidation must not defeat the cache it is protecting."""
        first = self.dm.get_tables("public")

        # Mutate the driver behind the manager's back: a second call that hits
        # the cache cannot see this, which is what proves it was cached.
        self.dm._driver._tables = ["changed_underneath"]

        self.assertEqual(self.dm.get_tables("public"), first)

    def test_every_schema_is_invalidated_not_just_one(self):
        self.dm.get_views("public")
        self.dm.get_views("other")

        self.dm._activate_driver(FakeDriver([], {}))

        self.assertEqual(self.dm.get_views("public"), [])
        self.assertEqual(self.dm.get_views("other"), [])


class TestDisconnectClearsCache(unittest.TestCase):
    def test_disconnect_clears_metadata_cache(self):
        dm = DatabaseManager()
        dm._ensure_connection = lambda: None
        dm._dremio_rest_connection = None
        dm._activate_driver(FakeDriver(["t"], {}))
        dm.get_tables("public")
        self.assertTrue(dm._metadata_cache)

        dm.disconnect()

        self.assertEqual(dm._metadata_cache, {})


if __name__ == "__main__":
    unittest.main()

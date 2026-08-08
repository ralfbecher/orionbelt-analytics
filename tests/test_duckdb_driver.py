"""Tests for the DuckDB driver against a real in-process database.

DuckDB is the one supported backend that needs no server, so these run for
real rather than against mocks. There were no DuckDB tests at all until the
driver was found unable to connect: `connect_args={"timeout": ...}` is a
TypeError inside duckdb.connect(), so every connection failed and nothing
noticed.
"""

import unittest

from src.drivers.duckdb import DuckDBDriver


class DuckDBDriverTestCase(unittest.TestCase):
    """Shared in-memory DuckDB fixture."""

    def setUp(self):
        self.driver = DuckDBDriver()
        connected = self.driver.connect(database_path=":memory:")
        self.assertTrue(connected, "DuckDB driver failed to connect")

    def tearDown(self):
        self.driver.disconnect()

    def _exec(self, sql: str) -> None:
        from sqlalchemy import text

        assert self.driver.engine is not None
        with self.driver.engine.begin() as conn:
            conn.execute(text(sql))


class TestDuckDBConnection(DuckDBDriverTestCase):
    def test_connect_in_memory(self):
        """Regression: this returned False for every path, memory or file."""
        self.assertIsNotNone(self.driver.engine)
        self.assertTrue(self.driver.test_connection())

    def test_connect_to_file(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "probe.duckdb")
            driver = DuckDBDriver()
            try:
                self.assertTrue(driver.connect(database_path=path))
                self.assertTrue(driver.test_connection())
            finally:
                driver.disconnect()


class TestDuckDBTablesAndViews(DuckDBDriverTestCase):
    def setUp(self):
        super().setUp()
        self._exec("CREATE TABLE sales (id INTEGER, clientid INTEGER, amount DECIMAL)")
        self._exec("CREATE TABLE clients (clientid INTEGER, name TEXT)")
        self._exec(
            "CREATE VIEW v_revenue_by_client AS "
            "SELECT c.name, SUM(s.amount) AS total_revenue "
            "FROM sales s JOIN clients c ON s.clientid = c.clientid GROUP BY c.name"
        )

    def test_get_tables_excludes_views(self):
        tables = self.driver.get_tables("main")
        self.assertIn("sales", tables)
        self.assertIn("clients", tables)
        self.assertNotIn("v_revenue_by_client", tables)

    def test_get_views_returns_definition(self):
        views = self.driver.get_views("main")
        self.assertIn("v_revenue_by_client", views)
        definition = views["v_revenue_by_client"]
        self.assertIsNotNone(definition)
        self.assertIn("total_revenue", definition)

    def test_get_views_empty_when_none_defined(self):
        driver = DuckDBDriver()
        try:
            driver.connect(database_path=":memory:")
            self.assertEqual(driver.get_views("main"), {})
        finally:
            driver.disconnect()

    def test_analyze_table(self):
        info = self.driver.analyze_table("sales", "main")
        self.assertIsNotNone(info)
        self.assertEqual(info.name, "sales")
        self.assertEqual({c.name for c in info.columns}, {"id", "clientid", "amount"})


class TestDuckDBQuerying(DuckDBDriverTestCase):
    def setUp(self):
        super().setUp()
        self._exec("CREATE TABLE t (id INTEGER, label TEXT)")
        self._exec("INSERT INTO t VALUES (1, 'one'), (2, 'two')")

    def test_execute_sql_query(self):
        result = self.driver.execute_sql_query("SELECT id, label FROM t ORDER BY id")
        self.assertTrue(result.get("success"), result)
        self.assertEqual(len(result["data"]), 2)

    def test_sample_table_data(self):
        rows = self.driver.sample_table_data("t", "main", limit=1)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()

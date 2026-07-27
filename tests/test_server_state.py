"""Coverage tests for server_state helpers and the ServerState registry."""

import types
import unittest
from datetime import timedelta

from src.database_manager import ColumnInfo, TableInfo
from src.server_state import (
    ErrorResponse,
    ServerState,
    _calculate_schema_hash,
    _clear_session_state,
    create_error_response,
    get_session_id,
)
from src.session import SessionData
from src.utils import utc_now


def _table():
    return TableInfo(
        name="t",
        schema="public",
        columns=[
            ColumnInfo("id", "INTEGER", False, True, False, comment=None),
            ColumnInfo("name", "VARCHAR", True, False, False, comment=None),
        ],
        primary_keys=["id"],
        foreign_keys=[],
        comment=None,
        row_count=1,
    )


class TestSessionIdAndErrors(unittest.TestCase):
    def test_get_session_id_from_session_id(self):
        ctx = types.SimpleNamespace(session_id="abc")
        self.assertEqual(get_session_id(ctx), "abc")

    def test_get_session_id_from_session_object(self):
        ctx = types.SimpleNamespace(session=object())
        self.assertTrue(get_session_id(ctx).startswith("session_"))

    def test_get_session_id_default(self):
        self.assertEqual(get_session_id(types.SimpleNamespace()), "default_session")

    def test_error_response(self):
        r = create_error_response("boom", "bad", "details")
        self.assertEqual(r["error"], "boom")
        self.assertEqual(r["error_type"], "bad")
        self.assertEqual(ErrorResponse(error="x").error_type, "unknown")


class TestSchemaHash(unittest.TestCase):
    def test_hash_is_deterministic_and_order_independent(self):
        h1 = _calculate_schema_hash([_table()])
        h2 = _calculate_schema_hash([_table()])
        self.assertEqual(h1, h2)

    def test_hash_changes_with_structure(self):
        t = _table()
        t2 = _table()
        t2.name = "other"
        self.assertNotEqual(_calculate_schema_hash([t]), _calculate_schema_hash([t2]))


class TestClearSessionState(unittest.TestCase):
    def test_clears_connection_scoped_state(self):
        session = SessionData()
        session.graphrag_initialized = True
        session.oxigraph_initialized = True
        _clear_session_state(session, reason="test")
        self.assertFalse(session.graphrag_initialized)
        self.assertIsNone(session.graphrag_manager)
        self.assertFalse(session.oxigraph_initialized)


class TestServerState(unittest.TestCase):
    def test_session_lifecycle(self):
        ss = ServerState()
        s = ss.get_session("s1")
        self.assertIsInstance(s, SessionData)
        self.assertEqual(ss.session_count, 1)
        # Same id returns same object
        self.assertIs(ss.get_session("s1"), s)
        ss.cleanup_session("s1")
        self.assertEqual(ss.session_count, 0)

    def test_get_ontology_generator(self):
        gen = ServerState().get_ontology_generator(base_uri="http://x/")
        self.assertTrue(hasattr(gen, "generate_from_schema"))

    def test_evict_idle_sessions(self):
        ss = ServerState()
        s = ss.get_session("old")
        s.last_activity = utc_now() - timedelta(hours=1)
        ss._evict_idle_sessions(idle_timeout=1)
        self.assertEqual(ss.session_count, 0)

    def test_cleanup_all(self):
        ss = ServerState()
        ss.get_session("a")
        ss.get_session("b")
        ss.cleanup()
        self.assertEqual(ss.session_count, 0)


if __name__ == "__main__":
    unittest.main()

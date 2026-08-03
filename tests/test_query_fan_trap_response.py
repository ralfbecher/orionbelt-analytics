"""The fan-trap verdict must reach the caller as a field, not as prose.

An agent keys off ``success``. When a query silently returned inflated numbers,
the only signal was an English sentence in ``warnings`` -- and on the success
path not even that, since ``fan_trap_risk`` was set only on the failure branch.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.handler_context import HandlerContext
from src.handlers.query import execute_sql_query
from src.obqc_validator import OBQCValidator
from tests.test_obqc_validator import create_sample_ontology_graph

FAN_TRAP_SQL = (
    "SELECT SUM(orders.total) FROM orders "
    "JOIN order_items ON order_items.order_id = orders.id"
)
CLEAN_SQL = "SELECT users.name FROM users"


def _services(validator):
    """A HandlerContext whose db always succeeds, so only OBQC decides."""
    db_manager = MagicMock()
    db_manager.has_engine.return_value = True
    db_manager.connection_info = {"type": "postgresql"}
    db_manager.execute_sql_query.return_value = {
        "success": True,
        "data": [{"sum": 1}],
        "columns": ["sum"],
        "row_count": 1,
        "execution_time_ms": 1,
    }

    return (
        HandlerContext(
            get_session_db_manager=lambda ctx: db_manager,
            get_session_obqc_validator=lambda ctx: validator,
            get_session_data=lambda ctx: SimpleNamespace(
                graphrag_initialized=False, graphrag_manager=None
            ),
        ),
        db_manager,
    )


class TestFanTrapResponseField(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        graph, base_uri = create_sample_ontology_graph()
        self.validator = OBQCValidator()
        self.validator.load_ontology(graph, base_uri)
        self.ctx = MagicMock()
        self.ctx.info = AsyncMock()

    async def test_fan_trap_blocks_and_reports_structurally(self):
        services, db_manager = _services(self.validator)

        result = await execute_sql_query(
            self.ctx, FAN_TRAP_SQL, 100, True, None, services
        )

        self.assertFalse(result["success"])
        self.assertTrue(result["obqc_fan_trap"]["detected"])
        self.assertTrue(result["obqc_fan_trap"]["blocking"])
        self.assertEqual(
            result["obqc_fan_trap"]["findings"][0]["fan_out_table"], "order_items"
        )
        db_manager.execute_sql_query.assert_not_called()

    async def test_allow_fan_out_executes_and_still_reports(self):
        services, db_manager = _services(self.validator)

        result = await execute_sql_query(
            self.ctx, FAN_TRAP_SQL, 100, True, None, services, allow_fan_out=True
        )

        self.assertTrue(result["success"])
        db_manager.execute_sql_query.assert_called_once()
        self.assertTrue(result["obqc_fan_trap"]["detected"])
        self.assertFalse(result["obqc_fan_trap"]["blocking"])
        self.assertTrue(
            any("FAN-TRAP" in w for w in result["warnings"]), result["warnings"]
        )

    async def test_clean_query_carries_a_negative_verdict(self):
        """ "No fan-trap" must be readable, not merely the absence of a warning."""
        services, _ = _services(self.validator)

        result = await execute_sql_query(self.ctx, CLEAN_SQL, 100, True, None, services)

        self.assertTrue(result["success"])
        self.assertEqual(
            result["obqc_fan_trap"],
            {
                "evaluated": True,
                "detected": False,
                "blocking": True,
                "findings": [],
            },
        )

    async def test_early_failures_still_carry_the_verdict(self):
        """Every response has the field, so a caller needs no special case.

        These paths returned before it was attached, so a client reading
        `obqc_fan_trap` had to guard against its absence.
        """
        no_connection = MagicMock()
        no_connection.has_engine.return_value = False
        services = HandlerContext(get_session_db_manager=lambda ctx: no_connection)

        cases = {
            "no connection": (CLEAN_SQL, 100, True),
            "bad limit": (CLEAN_SQL, 0, True),
            "empty sql": ("   ", 100, True),
            "checklist not completed": (CLEAN_SQL, 100, False),
        }
        for label, (sql, limit, checklist) in cases.items():
            with self.subTest(case=label):
                connected = MagicMock()
                connected.has_engine.return_value = label != "no connection"
                result = await execute_sql_query(
                    self.ctx,
                    sql,
                    limit,
                    checklist,
                    None,
                    (
                        HandlerContext(get_session_db_manager=lambda ctx: connected)
                        if label != "no connection"
                        else services
                    ),
                )

                self.assertFalse(result["success"])
                self.assertIn("obqc_fan_trap", result)
                self.assertFalse(result["obqc_fan_trap"]["evaluated"])

    async def test_no_ontology_reports_not_evaluated(self):
        """ "Not checked" must not read as "checked and clean".

        Without an ontology OBQC never runs, but the query still executes and
        succeeds -- exactly the case where a caller keying off the field would
        otherwise be told there is no fan-trap.
        """
        services, db_manager = _services(None)

        result = await execute_sql_query(
            self.ctx, FAN_TRAP_SQL, 100, True, None, services
        )

        self.assertTrue(result["success"])
        db_manager.execute_sql_query.assert_called_once()
        self.assertFalse(result["obqc_fan_trap"]["evaluated"])
        self.assertFalse(result["obqc_fan_trap"]["detected"])

    async def test_string_allow_fan_out_is_coerced(self):
        """MCP clients sometimes send booleans as strings."""
        services, db_manager = _services(self.validator)

        result = await execute_sql_query(
            self.ctx, FAN_TRAP_SQL, 100, "true", None, services, allow_fan_out="true"
        )

        self.assertTrue(result["success"])
        db_manager.execute_sql_query.assert_called_once()


if __name__ == "__main__":
    unittest.main()

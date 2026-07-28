"""Regression test for RDF-store auto-restore through HandlerContext.

Guards against a regression where connect_database() passed a bare
``get_oxigraph_store`` function into _restore_workspace_core() instead of the
full HandlerContext, causing ``services.get_oxigraph_store(ctx)`` to raise
AttributeError and silently fail RDF restore.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.handler_context import HandlerContext
from src.handlers.workspace import _restore_workspace_core
from src.session import SessionData


class TestWorkspaceRdfRestore(unittest.IsolatedAsyncioTestCase):
    async def test_rdf_store_restored_via_handler_context(self):
        workspace = {
            "schemas": {"public": {}},  # no schema files -> per-schema loop is a no-op
            "rdf_store": {"initialized": True},
            "models": {},
        }

        mock_store = object()
        get_oxigraph_store = MagicMock(return_value=mock_store)
        services = HandlerContext(get_oxigraph_store=get_oxigraph_store)
        session = SessionData()
        ctx = MagicMock()

        with (
            patch("src.handlers.workspace.VersionMetadataManager") as MgrCls,
            patch("src.handlers.workspace.OXIGRAPH_AVAILABLE", True),
        ):
            MgrCls.return_value.get_workspace.return_value = workspace
            result = await _restore_workspace_core(
                ctx, session, "conn-123", None, services
            )

        # The context was used correctly (no AttributeError swallowed into failed).
        get_oxigraph_store.assert_called_once_with(ctx)
        self.assertIn("RDF store (initialized)", result["restored"])
        self.assertFalse(
            any("RDF store" in f for f in result["failed"]),
            f"RDF restore should not fail: {result['failed']}",
        )

    async def test_passing_bare_function_would_fail_restore(self):
        # Demonstrates the bug the fix prevents: a bare callable (not a
        # HandlerContext) makes services.get_oxigraph_store raise AttributeError,
        # which is recorded as a failed RDF restore.
        workspace = {
            "schemas": {"public": {}},
            "rdf_store": {"initialized": True},
            "models": {},
        }
        bare_function = MagicMock()  # not a HandlerContext
        del bare_function.get_oxigraph_store  # force AttributeError on attr access
        session = SessionData()

        with (
            patch("src.handlers.workspace.VersionMetadataManager") as MgrCls,
            patch("src.handlers.workspace.OXIGRAPH_AVAILABLE", True),
        ):
            MgrCls.return_value.get_workspace.return_value = workspace
            result = await _restore_workspace_core(
                MagicMock(), session, "conn-123", None, bare_function
            )

        self.assertNotIn("RDF store (initialized)", result["restored"])
        self.assertTrue(any("RDF store" in f for f in result["failed"]))


if __name__ == "__main__":
    unittest.main()

"""Tests for session idle timeout and eviction."""

import asyncio
import contextlib
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.main import ServerState
from src.session import SessionData
from src.utils import utc_now


class TestSessionDataActivity:
    """Tests for SessionData activity tracking."""

    def test_created_with_timestamps(self):
        before = utc_now()
        session = SessionData()
        after = utc_now()

        assert before <= session.created_at <= after
        assert before <= session.last_activity <= after

    def test_touch_updates_last_activity(self):
        session = SessionData()
        original = session.last_activity
        # Backdate to guarantee visible difference
        session.last_activity = original - timedelta(seconds=10)
        session.touch()
        assert session.last_activity > original - timedelta(seconds=10)

    def test_touch_does_not_change_created_at(self):
        session = SessionData()
        created = session.created_at
        session.last_activity = utc_now() - timedelta(seconds=10)
        session.touch()
        assert session.created_at == created


class TestServerStateEviction:
    """Tests for ServerState idle eviction logic."""

    def _make_state(self) -> ServerState:
        state = ServerState()
        # Prevent background task from starting (no event loop in tests)
        state._ensure_eviction_task = lambda: None
        return state

    def test_get_session_touches_activity(self):
        state = self._make_state()
        session = state.get_session("s1")
        # Backdate and access again
        session.last_activity = utc_now() - timedelta(minutes=5)
        before_touch = session.last_activity
        state.get_session("s1")
        assert session.last_activity > before_touch

    async def test_evict_idle_sessions_removes_stale(self):
        state = self._make_state()
        state.get_session("active")
        stale = state.get_session("stale")
        # Backdate the stale session
        stale.last_activity = utc_now() - timedelta(minutes=45)

        await state._evict_idle_sessions(idle_timeout=1800)  # 30 min

        assert "active" in state._sessions
        assert "stale" not in state._sessions

    async def test_evict_idle_sessions_keeps_all_when_fresh(self):
        state = self._make_state()
        state.get_session("s1")
        state.get_session("s2")

        await state._evict_idle_sessions(idle_timeout=1800)

        assert state.session_count == 2

    async def test_evict_idle_sessions_removes_all_stale(self):
        state = self._make_state()
        for name in ["a", "b", "c"]:
            s = state.get_session(name)
            s.last_activity = utc_now() - timedelta(hours=2)

        await state._evict_idle_sessions(idle_timeout=1800)

        assert state.session_count == 0

    def test_session_count_property(self):
        state = self._make_state()
        assert state.session_count == 0
        state.get_session("s1")
        assert state.session_count == 1
        state.get_session("s2")
        assert state.session_count == 2
        state.cleanup_session("s1")
        assert state.session_count == 1


class TestCleanupSession:
    """Tests for cleanup_session resource teardown."""

    def _make_state(self) -> ServerState:
        state = ServerState()
        state._ensure_eviction_task = lambda: None
        return state

    def test_cleanup_disconnects_db(self):
        state = self._make_state()
        session = state.get_session("s1")
        mock_db = MagicMock()
        session.db_manager = mock_db

        state.cleanup_session("s1")

        mock_db.disconnect.assert_called_once()
        assert "s1" not in state._sessions

    def test_cleanup_closes_oxigraph(self):
        state = self._make_state()
        session = state.get_session("s1")
        mock_store = MagicMock()
        session.rdf_store.oxigraph_store = mock_store

        state.cleanup_session("s1")

        mock_store.close.assert_called_once()

    def test_cleanup_handles_db_error(self):
        state = self._make_state()
        session = state.get_session("s1")
        mock_db = MagicMock()
        mock_db.disconnect.side_effect = RuntimeError("connection lost")
        session.db_manager = mock_db

        # Should not raise
        state.cleanup_session("s1")
        assert "s1" not in state._sessions

    def test_cleanup_handles_oxigraph_error(self):
        state = self._make_state()
        session = state.get_session("s1")
        mock_store = MagicMock()
        mock_store.close.side_effect = RuntimeError("store error")
        session.rdf_store.oxigraph_store = mock_store

        # Should not raise
        state.cleanup_session("s1")
        assert "s1" not in state._sessions

    def test_cleanup_nonexistent_session_is_noop(self):
        state = self._make_state()
        state.cleanup_session("nonexistent")  # Should not raise


class TestEvictionLoop:
    """Tests for the async eviction loop."""

    @pytest.mark.asyncio
    async def test_eviction_loop_disabled_when_timeout_zero(self):
        state = ServerState()
        mock_config = MagicMock()
        mock_config.session_idle_timeout = 0
        mock_config.session_scan_interval = 1

        with patch("src.config.config_manager") as mock_cm:
            mock_cm.get_server_config.return_value = mock_config
            # Should return immediately without looping
            await state._eviction_loop()

    @pytest.mark.asyncio
    async def test_eviction_loop_runs_and_evicts(self):
        state = ServerState()
        state._ensure_eviction_task = lambda: None
        session = state.get_session("stale")
        session.last_activity = utc_now() - timedelta(hours=1)

        mock_config = MagicMock()
        mock_config.session_idle_timeout = 60
        mock_config.session_scan_interval = 0.1  # Fast scan for test

        async def run_loop():
            with patch("src.config.config_manager") as mock_cm:
                mock_cm.get_server_config.return_value = mock_config
                task = asyncio.create_task(state._eviction_loop())
                await asyncio.sleep(0.3)  # Let it run a couple cycles
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        await run_loop()
        assert "stale" not in state._sessions

    def test_cleanup_cancels_eviction_task(self):
        state = ServerState()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        state._eviction_task = mock_task

        state.cleanup()

        mock_task.cancel.assert_called_once()


class TestBackgroundTaskTeardown:
    """Every in-flight init task must be stopped before the session is released.

    Two defects this pins:
      * a single `_init_task` slot lost earlier tasks whenever discover_schema
        overlapped, leaving them running against a released session;
      * teardown only *scheduled* cancellation, then closed the Oxigraph store
        and dropped the session in the same synchronous block -- so a cancelled
        task could still be mid save_state when its store closed.
    """

    async def test_all_init_tasks_are_tracked_not_just_the_newest(self):
        """Overlapping background inits must all be retained."""
        state = ServerState()
        session = state.get_session("s1")

        async def noop() -> None:
            await asyncio.sleep(3600)

        tasks = [asyncio.create_task(noop()) for _ in range(3)]
        for task in tasks:
            session.graphrag.track_init_task(task)

        assert len(session.graphrag.init_tasks) == 3, "earlier tasks were dropped"

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def test_finished_tasks_stop_being_tracked(self):
        """The set must not grow without bound over a long-lived session."""
        state = ServerState()
        session = state.get_session("s1")

        async def quick() -> None:
            return None

        task = asyncio.create_task(quick())
        session.graphrag.track_init_task(task)
        await task
        await asyncio.sleep(0)  # let the done-callback run

        assert session.graphrag.init_tasks == set()

    async def test_aclose_waits_for_tasks_before_releasing_resources(self):
        """The store must not close while a background task is still running."""
        state = ServerState()
        session = state.get_session("s1")

        store_closed = False
        still_running_at_close = False

        class FakeStore:
            def close(self) -> None:
                nonlocal store_closed
                store_closed = True

        session.rdf_store.oxigraph_store = FakeStore()

        started = asyncio.Event()

        async def slow_init() -> None:
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                nonlocal still_running_at_close
                # If teardown did not await us, the store is already closed by
                # the time this cleanup runs.
                still_running_at_close = store_closed
                raise

        task = asyncio.create_task(slow_init())
        session.graphrag.track_init_task(task)
        await started.wait()

        await state.aclose_session("s1")

        assert task.done(), "teardown returned with the task still running"
        assert store_closed, "store was never closed"
        assert (
            not still_running_at_close
        ), "Oxigraph store closed while a background task was still running"
        assert state.session_count == 0

    async def test_aclose_is_safe_for_an_unknown_session(self):
        """Closing twice, or closing something evicted already, must not raise."""
        state = ServerState()
        await state.aclose_session("never-existed")
        assert state.session_count == 0

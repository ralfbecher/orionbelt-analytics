"""Async/sync bridge utilities for OrionBelt Analytics.

Provides a single utility for running async code from synchronous contexts,
replacing the 8+ duplicated ThreadPoolExecutor patterns in database_manager.py.
"""

import asyncio
import logging
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .constants import CONNECTION_TIMEOUT

logger = logging.getLogger(__name__)


def run_async[T](coro: Coroutine[Any, Any, T], timeout: int = CONNECTION_TIMEOUT) -> T:
    """Run an async coroutine from a synchronous context.

    Handles the common case in MCP servers where tool handlers are sync
    but need to call async code (e.g., Dremio REST client).

    If an event loop is already running (MCP server context), runs the
    coroutine in a separate thread. Otherwise creates a new event loop.

    Args:
        coro: The coroutine to execute
        timeout: Maximum seconds to wait for completion

    Returns:
        The coroutine's return value

    Raises:
        TimeoutError: If execution exceeds timeout
        Exception: Any exception raised by the coroutine
    """
    try:
        asyncio.get_running_loop()
        # Already in async context (MCP server) - run in separate thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=timeout)
    except RuntimeError:
        # No event loop running - safe to use asyncio.run directly
        return asyncio.run(coro)

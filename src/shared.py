"""Shared instances and utilities for OrionBelt Analytics.

WARNING: The global database manager in this module is DEPRECATED.
Use get_session_db_manager(ctx) from main.py for proper session isolation.
"""

import logging
import warnings
from typing import Optional, Dict, Any

from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)

# DEPRECATED: Global database manager - violates session isolation!
# Use get_session_db_manager(ctx) from main.py instead.
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance.

    DEPRECATED: This function creates a SHARED database manager that violates
    session isolation. Use get_session_db_manager(ctx) from main.py instead.

    This global instance can cause:
    - Cross-session data leakage
    - Connection state bleeding between users
    - Unpredictable behavior in multi-user scenarios

    Returns:
        DatabaseManager: A shared (non-isolated) database manager instance.
    """
    warnings.warn(
        "get_db_manager() is deprecated and violates session isolation. "
        "Use get_session_db_manager(ctx) from main.py instead.",
        DeprecationWarning,
        stacklevel=2
    )
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        logger.warning(
            "DEPRECATION: Global database manager created. "
            "This violates session isolation - use get_session_db_manager(ctx) instead."
        )
    return _db_manager


def create_error_response(message: str, error_type: str, details: str = None) -> Dict[str, Any]:
    """Create a standardized error response."""
    error_response = {
        "success": False,
        "error": message,
        "error_type": error_type
    }
    if details:
        error_response["details"] = details
    return error_response
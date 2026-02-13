"""SQL query validation and execution tools."""

import logging
from typing import Dict, Any

from ..shared import get_db_manager, create_error_response

logger = logging.getLogger(__name__)


def validate_sql_syntax(sql_query: str) -> Dict[str, Any]:
    """Validate SQL query syntax. Full documentation in main.py."""
    try:
        db_manager = get_db_manager()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        validation_result = db_manager.validate_sql_syntax(sql_query)
        logger.info(f"SQL validation completed: {'valid' if validation_result['is_valid'] else 'invalid'}")
        return validation_result
        
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"SQL validation error: {str(e)}",
            "error_type": "validation_error"
        }


def execute_sql_query(
    sql_query: str,
    limit: int = 1000,
    checklist_completed: bool = False
) -> Dict[str, Any]:
    """Execute SQL query implementation. Full documentation in main.py."""
    try:
        # Handle string "True"/"False" from LLMs that send strings instead of booleans
        if isinstance(checklist_completed, str):
            checklist_completed = checklist_completed.lower() in ('true', '1', 'yes')

        db_manager = get_db_manager()

        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        result = db_manager.execute_sql_query(sql_query, limit)
        if result['success']:
            logger.info(f"SQL query executed successfully: {result.get('row_count', 0)} rows returned")
        else:
            logger.warning(f"SQL query failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return create_error_response(
            f"Failed to execute SQL query: {str(e)}",
            "execution_error"
        )
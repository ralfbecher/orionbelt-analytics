"""SQL validation and execution handler implementations."""

import logging
import re
from typing import Any

from fastmcp import Context

from ..exceptions import ConnectionError, ParameterError, ValidationError
from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)


def _extract_query_intent(sql: str) -> str:
    """Extract natural language intent from SQL query for context retrieval.

    Args:
        sql: SQL query string

    Returns:
        Natural language description of query intent
    """
    # Normalize whitespace
    sql = " ".join(sql.split())

    # Extract table names
    tables = []
    from_matches = re.findall(r"FROM\s+(?:[\w.]+\.)?(\w+)", sql, re.IGNORECASE)
    tables.extend(from_matches)
    join_matches = re.findall(r"JOIN\s+(?:[\w.]+\.)?(\w+)", sql, re.IGNORECASE)
    tables.extend(join_matches)

    # Deduplicate
    seen = set()
    unique_tables = []
    for t in tables:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_tables.append(t)

    # Extract aggregation functions
    aggs = re.findall(r"\b(SUM|AVG|COUNT|MAX|MIN)\s*\(", sql, re.IGNORECASE)
    aggs = list({a.upper() for a in aggs})

    # Extract WHERE conditions
    where_match = re.search(
        r"WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|$)", sql, re.IGNORECASE
    )
    conditions = []
    if where_match:
        where_clause = where_match.group(1)
        cond_cols = re.findall(
            r"\b(\w+)\s*(?:=|>|<|LIKE|IN)", where_clause, re.IGNORECASE
        )
        conditions = list(set(cond_cols[:3]))

    # Build intent string
    if aggs and unique_tables:
        intent = f"aggregate {', '.join(aggs)} from {', '.join(unique_tables)}"
    elif unique_tables:
        intent = f"query {', '.join(unique_tables)}"
    else:
        intent = "database query"

    if conditions:
        intent += f" filtered by {', '.join(conditions)}"

    return intent


async def validate_sql_syntax(
    ctx: Context,
    sql_query: str,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Validate SQL syntax, security, and fan-trap risks before execution.

    Args:
        ctx: FastMCP context
        sql_query: SQL SELECT statement to validate
        get_session_db_manager: Function to get session db manager
        get_session_obqc_validator: Function to get OBQC validator
    """
    try:
        db_manager = services.get_session_db_manager(ctx)

        if not db_manager.has_engine():
            return {
                "is_valid": False,
                "error": "No database connection established. Cannot perform full validation without schema information.",
                "error_type": "connection_error",
                "suggestions": [
                    "Use connect_database tool first to enable comprehensive validation",
                    "Basic syntax validation can still be performed, but schema validation requires a connection",
                ],
                "warnings": [
                    "Schema-level validation disabled without database connection"
                ],
                "database_dialect": "unknown",
            }

        if not sql_query or not sql_query.strip():
            return {
                "is_valid": False,
                "error": "SQL query cannot be empty.",
                "error_type": "parameter_error",
                "suggestions": [
                    "Provide a valid SELECT statement or schema introspection query"
                ],
                "database_dialect": "unknown",
            }

        validation_result: dict[str, Any] = db_manager.validate_sql_syntax(
            sql_query.strip()
        )

        if "warnings" not in validation_result:
            validation_result["warnings"] = []
        if "suggestions" not in validation_result:
            validation_result["suggestions"] = []

        # OBQC validation
        obqc_validator = services.get_session_obqc_validator(ctx)
        if obqc_validator:
            db_type = db_manager.connection_info.get("type", "postgresql")
            obqc_result = obqc_validator.validate(sql_query.strip(), dialect=db_type)

            validation_result.update(obqc_result.to_dict())

            if not obqc_result.is_valid:
                validation_result["is_valid"] = False
                if not validation_result.get("error"):
                    validation_result["error"] = (
                        "OBQC validation failed - see obqc_issues for details"
                    )
                validation_result["error_type"] = (
                    validation_result.get("error_type") or "obqc_error"
                )

            for issue in obqc_result.issues:
                if issue.severity.value == "warning":
                    msg = f"[OBQC] {issue.message}"
                    if issue.suggestion:
                        msg += f" - {issue.suggestion}"
                    validation_result["warnings"].append(msg)
                elif issue.severity.value == "error" and issue.suggestion:
                    validation_result["suggestions"].append(
                        f"[OBQC] {issue.suggestion}"
                    )

            if obqc_result.fan_trap_risk:
                validation_result["warnings"].append(
                    "[OBQC] FAN-TRAP RISK: Query aggregates across multiple 1:many relationships"
                )
                validation_result["suggestions"].append(
                    "Consider UNION ALL pattern: aggregate each fact table separately, then combine"
                )

            logger.debug(
                f"OBQC validation: valid={obqc_result.is_valid}, issues={len(obqc_result.issues)}"
            )
        else:
            validation_result["obqc_valid"] = None
            validation_result["obqc_issues"] = []
            validation_result["warnings"].append(
                "OBQC validation skipped - no ontology loaded. "
                "Use generate_ontology or load_my_ontology for semantic validation."
            )

        if validation_result.get("is_valid"):
            logger.info(
                f"SQL validation successful: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}"
            )
            validation_result["next_tool"] = "execute_sql_query"
            await ctx.info(
                "SQL validation passed; next call should be execute_sql_query"
            )
        else:
            logger.info(
                f"SQL validation failed: {validation_result.get('error', 'Unknown validation error')}"
            )
            await ctx.info(
                "SQL validation failed; fix the query and try validate_sql_syntax again"
            )

        return validation_result

    except Exception as e:
        logger.error(f"SQL validation error: {e}")
        return {
            "is_valid": False,
            "error": f"Validation system error: {e!s}",
            "error_type": "internal_error",
            "suggestions": [
                "Check if the database connection is stable",
                "Verify the SQL query contains valid UTF-8 characters",
                "Try breaking down complex queries into smaller parts",
            ],
            "database_dialect": "unknown",
        }


async def execute_sql_query(
    ctx: Context,
    sql_query: str,
    limit: int,
    checklist_completed: bool | str,
    query_intent: str | None,
    services: "HandlerContext",
    allow_fan_out: bool | str = False,
) -> dict[str, Any]:
    """Execute SQL query with built-in validation and fan-trap protection.

    Args:
        ctx: FastMCP context
        sql_query: SQL SELECT statement
        limit: Maximum rows to return
        checklist_completed: Pre-execution checklist confirmation
        query_intent: Natural language query description
        allow_fan_out: Run the query even if OBQC finds a fan-trap. The finding
            is still reported, as a warning rather than a blocking error.
        get_session_data: Function to get session data
        get_session_db_manager: Function to get session db manager
        get_session_obqc_validator: Function to get OBQC validator
        create_error_response: Error response helper
    """
    try:
        if isinstance(checklist_completed, str):
            checklist_completed = checklist_completed.lower() in ("true", "1", "yes")
        if isinstance(allow_fan_out, str):
            allow_fan_out = allow_fan_out.lower() in ("true", "1", "yes")

        # Structured fan-trap verdict, carried onto every response this tool
        # returns, so a caller reading fields never needs a special case for
        # the paths that bail out early. "evaluated" separates "checked and
        # clean" from "never checked" -- without it, a query run with no
        # ontology loaded reported detected=false and read as a clean bill of
        # health.
        fan_trap_report: dict[str, Any] = {
            "evaluated": False,
            "detected": False,
            # Nothing was checked, so nothing blocked on this account.
            "blocking": False,
            "findings": [],
        }

        def _with_verdict(response: dict[str, Any]) -> dict[str, Any]:
            response["obqc_fan_trap"] = fan_trap_report
            return response

        db_manager = services.get_session_db_manager(ctx)

        if not db_manager.has_engine():
            return _with_verdict(
                ConnectionError(
                    "No database connection established. Please use connect_database tool first to establish a connection to PostgreSQL, Snowflake, or Dremio.",
                    details="Available connection methods: connect_database('postgresql'), connect_database('snowflake'), connect_database('dremio')",
                ).to_response()
            )

        if limit <= 0 or limit > 5000:
            return _with_verdict(
                ParameterError(
                    f"Invalid limit value '{limit}'. Must be between 1 and 5000.",
                    details="Use a reasonable limit to prevent memory exhaustion while allowing comprehensive analysis.",
                ).to_response()
            )

        if not sql_query or not sql_query.strip():
            return _with_verdict(
                ParameterError(
                    "SQL query cannot be empty.",
                    details="Provide a valid SELECT statement or schema introspection query.",
                ).to_response()
            )

        if not checklist_completed:
            return _with_verdict(
                ValidationError(
                    "ERROR: PRE-EXECUTION CHECKLIST NOT COMPLETED.\nSee tool description for required steps.",
                    details="You must complete the pre-execution checklist before executing SQL queries. Review the tool documentation for required steps.",
                ).to_response()
            )

        # OBQC validation (fan-trap detection, ontology-aware checks)
        obqc_warnings = []
        obqc_validator = services.get_session_obqc_validator(ctx)
        if obqc_validator:
            db_type = db_manager.connection_info.get("type", "postgresql")
            obqc_result = obqc_validator.validate(
                sql_query.strip(), dialect=db_type, allow_fan_out=allow_fan_out
            )
            fan_trap_report = obqc_result.to_dict()["obqc_fan_trap"]

            if not obqc_result.is_valid:
                error_details = []
                for issue in obqc_result.issues:
                    detail = issue.message
                    if issue.suggestion:
                        detail += f" — {issue.suggestion}"
                    error_details.append(detail)
                return {
                    "success": False,
                    "data": [],
                    "columns": [],
                    "row_count": 0,
                    "execution_time_ms": None,
                    "error": "OBQC validation failed",
                    "error_type": "obqc_error",
                    "obqc_issues": error_details,
                    "fan_trap_risk": obqc_result.fan_trap_risk,
                    "obqc_fan_trap": fan_trap_report,
                    "warnings": [],
                    "query_plan": None,
                    "limit_applied": False,
                }

            for issue in obqc_result.issues:
                if issue.severity.value == "warning":
                    msg = f"[OBQC] {issue.message}"
                    if issue.suggestion:
                        msg += f" — {issue.suggestion}"
                    obqc_warnings.append(msg)

            if obqc_result.fan_trap_risk:
                # Only reachable with allow_fan_out: otherwise a fan-trap is an
                # error and the query never got here.
                obqc_warnings.append(
                    "[OBQC] FAN-TRAP RISK accepted via allow_fan_out: aggregates "
                    "read across a 1:many join and are inflated. See "
                    "obqc_fan_trap for the tables involved."
                )

            logger.debug(
                f"OBQC validation: valid={obqc_result.is_valid}, "
                f"issues={len(obqc_result.issues)}"
            )

        # Auto-inject GraphRAG context if available
        session = services.get_session_data(ctx)
        if session.graphrag_initialized and session.graphrag_manager:
            try:
                if query_intent:
                    logger.info(f"Using provided query intent: '{query_intent}'")
                    intent_to_use = query_intent
                else:
                    intent_to_use = _extract_query_intent(sql_query)
                    logger.info(f"Auto-extracted intent from SQL: '{intent_to_use}'")

                context = session.graphrag_manager.get_query_context(
                    query=intent_to_use, max_tables=3, max_columns=15
                )

                if context and "relevant_tables" in context:
                    table_count = len(context["relevant_tables"])
                    logger.info(
                        f"Auto-retrieved context: {table_count} relevant tables"
                    )
            except Exception as e:
                logger.debug(f"Context auto-retrieval failed (non-critical): {e}")

        result: dict[str, Any] = db_manager.execute_sql_query(sql_query.strip(), limit)

        # Merge OBQC warnings into result
        if obqc_warnings:
            existing_warnings = result.get("warnings", [])
            result["warnings"] = existing_warnings + obqc_warnings

        # Always present, so "no fan-trap" is an answer the caller can read
        # rather than the absence of a sentence in warnings.
        result["obqc_fan_trap"] = fan_trap_report

        if result.get("success"):
            logger.info(
                f"SQL query executed successfully: {result.get('row_count', 0)} rows returned in {result.get('execution_time_ms', 0)}ms"
            )
            row_count = result.get("row_count", 0)
            if row_count > 0:
                result["next_tool"] = "generate_chart"
                await ctx.info(
                    f"SQL query executed successfully with {row_count} rows; next call should be generate_chart for visualization"
                )
            else:
                await ctx.info("SQL query executed successfully but returned no rows")
        else:
            logger.warning(
                f"SQL query execution failed: {result.get('error', 'Unknown error')}"
            )
            await ctx.info("SQL query execution failed; review error and try again")

        return result

    except Exception as e:
        logger.error(f"Critical error in SQL execution: {e}")
        err: dict[str, Any] = services.create_error_response(
            f"Internal server error during SQL execution: {e!s}",
            "internal_error",
            "This may indicate a system-level issue. Please check server logs and try again.",
        )
        # The failure may have happened before, during or after validation, so
        # the honest verdict here is "not evaluated".
        err["obqc_fan_trap"] = {
            "evaluated": False,
            "detected": False,
            "blocking": False,
            "findings": [],
        }
        return err

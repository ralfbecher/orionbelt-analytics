"""Custom exception hierarchy for OrionBelt Analytics.

Provides structured error handling with consistent error types
that replace ad-hoc string-based error_type fields.
"""

from enum import StrEnum


class ErrorType(StrEnum):
    """Enumeration of all error types for consistent error reporting."""

    VALIDATION = "validation_error"
    PARAMETER = "parameter_error"
    CONNECTION = "connection_error"
    DATABASE = "database_error"
    SECURITY = "security_error"
    SYNTAX = "syntax_error"
    FORBIDDEN = "forbidden_operation"
    INTERNAL = "internal_error"
    RDF = "rdf_error"
    STORE = "store_not_initialized"
    DEPENDENCY = "dependency_error"
    OBQC = "obqc_error"


class OrionBeltError(Exception):
    """Base exception for all OrionBelt Analytics errors."""

    error_type: ErrorType = ErrorType.INTERNAL

    def __init__(
        self,
        message: str,
        details: str | None = None,
        suggestions: list[str] | None = None,
    ):
        self.message = message
        self.details = details
        self.suggestions = suggestions or []
        super().__init__(message)

    def to_response(self) -> dict:
        """Convert exception to a standardized error response dict."""
        response = {
            "success": False,
            "error": self.message,
            "error_type": self.error_type.value,
        }
        if self.details:
            response["details"] = self.details
        if self.suggestions:
            response["suggestions"] = self.suggestions
        return response


class ConnectionError(OrionBeltError):
    """Database connection failures."""

    error_type = ErrorType.CONNECTION


class DatabaseError(OrionBeltError):
    """Database operation failures (query execution, schema analysis)."""

    error_type = ErrorType.DATABASE


class ValidationError(OrionBeltError):
    """Input validation failures."""

    error_type = ErrorType.VALIDATION


class ParameterError(OrionBeltError):
    """Missing or invalid tool parameters."""

    error_type = ErrorType.PARAMETER


class RDFError(OrionBeltError):
    """RDF/ontology store errors."""

    error_type = ErrorType.RDF


class StoreNotInitializedError(OrionBeltError):
    """RDF or vector store not initialized."""

    error_type = ErrorType.STORE


class DependencyError(OrionBeltError):
    """Missing optional dependency."""

    error_type = ErrorType.DEPENDENCY

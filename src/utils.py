"""Utility functions for OrionBelt Analytics."""

import logging
import sys
from typing import Any, Dict, Optional
from urllib.parse import urlparse


def setup_logging(log_level: str = "INFO", structured: bool = False) -> logging.Logger:
    """
    Setup logging configuration for the application.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured logging format (JSON)

    Returns:
        Logger instance for the root logger
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter based on structured flag
    if structured:
        # Structured format for production (could be JSON in the future)
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        # Simple format for development and startup
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)

    return root_logger


def sanitize_for_logging(data: Any) -> Any:
    """
    Sanitize sensitive data for logging by redacting passwords, secrets, and API keys.

    Args:
        data: Data structure (dict, list, or primitive) to sanitize

    Returns:
        Sanitized copy of the data with sensitive fields redacted
    """
    if isinstance(data, dict):
        sanitized = {}
        sensitive_keys = {
            'password', 'passwd', 'pwd', 'secret', 'api_key', 'apikey',
            'token', 'auth', 'authorization', 'credentials', 'private_key'
        }

        for key, value in data.items():
            # Check if key name suggests sensitive data
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '***REDACTED***'
            elif isinstance(value, (dict, list)):
                # Recursively sanitize nested structures
                sanitized[key] = sanitize_for_logging(value)
            else:
                sanitized[key] = value

        return sanitized
    elif isinstance(data, list):
        return [sanitize_for_logging(item) for item in data]
    else:
        # Return primitives unchanged
        return data


def validate_uri(uri: str) -> bool:
    """
    Validate that a string is a valid HTTP/HTTPS URI.

    Args:
        uri: URI string to validate

    Returns:
        True if valid HTTP/HTTPS URI, False otherwise
    """
    if not uri:
        return False

    try:
        parsed = urlparse(uri)
        # Check scheme is http or https and has a netloc (domain)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except Exception:
        return False


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes into human-readable string with appropriate unit.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 KB", "2.3 MB")
    """
    if num_bytes == 0:
        return "0 B"

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    size = float(num_bytes)

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    # Format with 1 decimal place for units beyond bytes
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"

"""Centralized path management for OrionBelt Analytics.

Single source of truth for all file/directory paths used across the project.
Replaces scattered Path construction and hardcoded paths.
"""

import os
from pathlib import Path

from .constants import DEFAULT_OUTPUT_DIR

# Project root: parent of the src/ directory
PROJECT_ROOT = Path(__file__).parent.parent

# Output directory for generated files (configurable via OUTPUT_DIR env var)
OUTPUT_DIR = PROJECT_ROOT / os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)


def ensure_output_dir() -> Path:
    """Get the output directory, creating it if needed."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def get_env_file_path() -> Path | None:
    """Find the .env file using standard resolution order.

    Resolution order:
    1. Relative to project root (src/../.env)
    2. Current working directory
    """
    candidates = [
        PROJECT_ROOT / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_oxigraph_store_dir(connection_id: str | None = None) -> Path:
    """Get Oxigraph store directory, scoped per connection.

    Args:
        connection_id: Database connection fingerprint.
                      If None, uses legacy global store (backward compat).

    Returns:
        Path to Oxigraph store directory
    """
    if connection_id:
        store_dir = OUTPUT_DIR / "oxigraph" / connection_id / "store"
    else:
        store_dir = OUTPUT_DIR / "oxigraph_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def get_connection_dir(connection_id: str) -> Path:
    """Get the base directory for connection-scoped data."""
    conn_dir = OUTPUT_DIR / connection_id
    conn_dir.mkdir(parents=True, exist_ok=True)
    return conn_dir


def get_skills_dir() -> Path:
    """Get the skills documentation directory."""
    return PROJECT_ROOT / ".claude" / "skills"


def get_models_dir(connection_id: str) -> Path:
    """Get the semantic models directory for a connection.

    Args:
        connection_id: Database connection fingerprint.

    Returns:
        Path to models directory for this connection
    """
    models_dir = OUTPUT_DIR / connection_id / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_charts_dir(connection_id: str) -> Path:
    """Get the charts directory for a connection.

    Args:
        connection_id: Database connection fingerprint.

    Returns:
        Path to charts directory for this connection
    """
    charts_dir = OUTPUT_DIR / connection_id / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir

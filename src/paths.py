"""Centralized path management for OrionBelt Analytics.

Single source of truth for all file/directory paths used across the project.
Replaces scattered Path construction and hardcoded paths.
"""

import os
from pathlib import Path

from .constants import DEFAULT_OUTPUT_DIR

# Project root: parent of the src/ directory
PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_output_dir() -> Path:
    """Resolve OUTPUT_DIR, refusing values that would make cleanup destructive.

    Startup cleanup deletes loose files directly under this directory, and with
    AUTO_CLEANUP_ON_STARTUP=true it removes every subdirectory lacking a
    metadata.json. That is safe for a dedicated output directory and
    catastrophic for anything else: pathlib collapses "" and "." onto
    PROJECT_ROOT, so a blanked-out ``OUTPUT_DIR=`` in .env would target the
    installation itself -- unlinking .env and pyproject.toml, then removing
    src/, tests/ and .git/.

    Rejected at import rather than defaulted, because silently substituting a
    different directory than the operator configured is its own surprise.

    Returns:
        The configured output directory.

    Raises:
        ValueError: If the value is blank, or resolves to PROJECT_ROOT or any
            of its parents.
    """
    raw = os.getenv("OUTPUT_DIR")
    if raw is None:
        return PROJECT_ROOT / DEFAULT_OUTPUT_DIR

    if not raw.strip():
        raise ValueError(
            "OUTPUT_DIR is set but empty. Remove it to use the default "
            f"('{DEFAULT_OUTPUT_DIR}') or give it a real path; an empty value "
            "resolves to the project root, which startup cleanup would delete."
        )

    candidate = Path(raw.strip())
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    root = PROJECT_ROOT.resolve()

    if resolved == root or resolved in root.parents:
        raise ValueError(
            f"OUTPUT_DIR={raw!r} resolves to {resolved}, which contains the "
            "installation. Startup cleanup deletes loose files and unrecognized "
            "directories there. Point it at a dedicated directory such as "
            f"'{DEFAULT_OUTPUT_DIR}' or '/var/lib/orionbelt'."
        )
    return resolved


# Output directory for generated files (configurable via OUTPUT_DIR env var)
OUTPUT_DIR = _resolve_output_dir()

# Directories directly under OUTPUT_DIR that are NOT per-connection workspaces.
# They hold satellite stores keyed by connection one level deeper
# (chromadb/{connection_id}, oxigraph/{connection_id}/store), plus the legacy
# global Oxigraph store. Anything walking OUTPUT_DIR looking for workspaces must
# skip these -- they have no metadata.json, so treating them as workspaces makes
# them look orphaned and gets every connection's vectors and triples deleted.
NON_WORKSPACE_DIRS = frozenset({"chromadb", "oxigraph", "oxigraph_store"})


def ensure_output_dir() -> Path:
    """Get the output directory, creating it and any missing parents.

    ``parents=True`` because OUTPUT_DIR may legitimately be nested (say
    ``output/data`` or ``/var/lib/orionbelt/data``) -- the resolver accepts
    those, so creation has to as well or a fresh deployment fails with
    FileNotFoundError on the first write.

    Returns:
        The output directory, guaranteed to exist.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


def get_connection_store_dirs(connection_id: str) -> list[Path]:
    """Satellite store directories belonging to a single connection.

    These live outside the connection's workspace directory, so removing a
    workspace does not remove them -- they have to be cleaned explicitly or the
    vectors and triples for a deleted connection linger forever.

    Args:
        connection_id: Database connection fingerprint.

    Returns:
        Paths that may or may not exist, one per satellite store.
    """
    return [
        OUTPUT_DIR / "chromadb" / connection_id,
        OUTPUT_DIR / "oxigraph" / connection_id,
    ]


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

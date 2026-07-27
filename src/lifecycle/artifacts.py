"""Pruning of superseded per-schema artifacts.

Every ``generate_ontology`` / ``discover_schema`` call writes a *new* file:
filenames carry a microsecond timestamp (see ``get_session_safe_filename``), so
each run produces a distinct name. Workspace metadata records only the latest
filename per schema, which leaves every earlier file unreferenced and, until
now, permanently on disk. On a large schema an ontology TTL is megabytes, so
regenerating a handful of times quietly costs hundreds.

``AUTO_CLEANUP_ON_STARTUP`` does not help: it deletes whole workspaces by age,
never files inside a workspace that is still in use.

This module prunes the older files in the same *family* as the one just
written -- same artifact kind, same connection, same schema -- keeping the most
recent ``keep`` of them. It is deliberately conservative: if a filename does not
match the expected shape, nothing is deleted.
"""

import asyncio
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Trailing timestamp produced by get_session_safe_filename ("%Y%m%d_%H%M%S%f")
# and by the background ontology path ("%Y%m%d_%H%M%S"). Note it contains an
# underscore itself, so stripping "the last _ component" would be wrong.
_TIMESTAMP_SUFFIX = re.compile(r"_\d{8}_\d{6,12}$")

DEFAULT_KEEP_VERSIONS = 3


def get_keep_versions() -> int:
    """How many generations of each artifact to retain, including the current one.

    Returns:
        Value of ARTIFACT_KEEP_VERSIONS, or the default. Values below 1 are
        clamped to 1 so the file in use is never a deletion candidate.
    """
    raw = os.getenv("ARTIFACT_KEEP_VERSIONS")
    if raw is None:
        return DEFAULT_KEEP_VERSIONS
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            f"Invalid ARTIFACT_KEEP_VERSIONS={raw!r}; using {DEFAULT_KEEP_VERSIONS}"
        )
        return DEFAULT_KEEP_VERSIONS


def family_glob(filename: str) -> str | None:
    """Return a glob matching every generation of *filename*'s artifact family.

    ``ontology_ab12cd34_public_20260727_132056962941.ttl`` becomes
    ``ontology_ab12cd34_public_*.ttl`` -- same kind, connection and schema, any
    timestamp.

    Args:
        filename: Base name of a freshly written artifact.

    Returns:
        A glob pattern, or None if the name has no recognizable timestamp, in
        which case the caller must not prune.
    """
    path = Path(filename)
    stem, suffix = path.stem, path.suffix
    if not _TIMESTAMP_SUFFIX.search(stem):
        return None
    return f"{_TIMESTAMP_SUFFIX.sub('', stem)}_*{suffix}"


def prune_superseded_sync(current_file: Path, keep: int | None = None) -> list[Path]:
    """Delete older artifacts superseded by *current_file*.

    Args:
        current_file: The artifact just written. Never deleted, whatever its
            modification time says.
        keep: Generations to retain including *current_file*. Defaults to
            :func:`get_keep_versions`.

    Returns:
        The paths actually deleted.
    """
    keep = get_keep_versions() if keep is None else max(1, keep)
    pattern = family_glob(current_file.name)
    if pattern is None:
        logger.debug(f"No timestamp in {current_file.name}; skipping prune")
        return []

    directory = current_file.parent
    try:
        siblings = [p for p in directory.glob(pattern) if p.is_file()]
    except OSError as e:
        logger.warning(f"Could not scan {directory} for superseded artifacts: {e}")
        return []

    # Newest first, with the current file pinned to the front so it survives
    # even if another generation has a newer mtime.
    others = [p for p in siblings if p != current_file]
    others.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ordered = [current_file, *others]

    removed: list[Path] = []
    for stale in ordered[keep:]:
        try:
            stale.unlink()
            removed.append(stale)
        except OSError as e:
            logger.warning(f"Failed to prune superseded artifact {stale}: {e}")

    if removed:
        logger.info(
            f"Pruned {len(removed)} superseded artifact(s) matching {pattern} "
            f"(keeping {keep})"
        )
    return removed


async def prune_superseded_artifacts(
    current_file: Path, keep: int | None = None
) -> list[Path]:
    """Async wrapper for :func:`prune_superseded_sync`.

    Globbing and unlinking are blocking, and callers are request handlers, so
    the work runs in a worker thread.

    Args:
        current_file: The artifact just written.
        keep: Generations to retain including *current_file*.

    Returns:
        The paths actually deleted.
    """
    return await asyncio.to_thread(prune_superseded_sync, current_file, keep)

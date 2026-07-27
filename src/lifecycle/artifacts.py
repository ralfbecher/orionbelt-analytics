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
from collections.abc import Iterable
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


def family_key(filename: str) -> tuple[str, str] | None:
    """Identify the artifact family *filename* belongs to.

    ``ontology_ab12cd34_public_20260727_132056962941.ttl`` maps to
    ``("ontology_ab12cd34_public", ".ttl")`` -- same kind, connection and
    schema, any timestamp.

    Deliberately not a glob. Schema names may legally contain ``*``, ``?`` or
    ``[``, and those survive the ``schema_safe`` sanitizing (which only
    replaces spaces and dots). A glob built from such a name matches sibling
    schemas: a prune for ``sales*`` would delete ``sales_eu``'s artifacts.
    Comparing parsed keys removes that class of bug entirely rather than
    relying on escaping.

    Args:
        filename: Base name of an artifact.

    Returns:
        ``(family, extension)``, or None if the name carries no recognizable
        timestamp, in which case the caller must not prune.
    """
    path = Path(filename)
    stem, suffix = path.stem, path.suffix
    if not _TIMESTAMP_SUFFIX.search(stem):
        return None
    return _TIMESTAMP_SUFFIX.sub("", stem), suffix


def prune_superseded_sync(
    current_file: Path,
    keep: int | None = None,
    protect: Iterable[Path | str] = (),
) -> list[Path]:
    """Delete older artifacts superseded by *current_file*.

    Args:
        current_file: The artifact just written. Never deleted, whatever its
            modification time says.
        keep: Generations to retain including *current_file*. Defaults to
            :func:`get_keep_versions`.
        protect: Additional names never to delete -- in particular whatever
            workspace metadata still points at. Until metadata durably names
            the new file, deleting the old one would leave restore chasing a
            file that no longer exists.

    Returns:
        The paths actually deleted.
    """
    keep = get_keep_versions() if keep is None else max(1, keep)
    key = family_key(current_file.name)
    if key is None:
        logger.debug(f"No timestamp in {current_file.name}; skipping prune")
        return []

    directory = current_file.parent
    protected = {Path(p).name for p in protect} | {current_file.name}
    try:
        siblings = [
            entry
            for entry in directory.iterdir()
            if entry.is_file()
            and entry.name not in protected
            and family_key(entry.name) == key
        ]
    except OSError as e:
        logger.warning(f"Could not scan {directory} for superseded artifacts: {e}")
        return []

    # Newest first, with the current file pinned to the front so it survives
    # even if another generation has a newer mtime.
    siblings.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ordered = [current_file, *siblings]

    removed: list[Path] = []
    for stale in ordered[keep:]:
        try:
            stale.unlink()
            removed.append(stale)
        except OSError as e:
            logger.warning(f"Failed to prune superseded artifact {stale}: {e}")

    if removed:
        logger.info(
            f"Pruned {len(removed)} superseded artifact(s) from family "
            f"{key[0]}{key[1]} (keeping {keep})"
        )
    return removed


async def prune_superseded_artifacts(
    current_file: Path,
    keep: int | None = None,
    protect: Iterable[Path | str] = (),
) -> list[Path]:
    """Async wrapper for :func:`prune_superseded_sync`.

    Globbing and unlinking are blocking, and callers are request handlers, so
    the work runs in a worker thread.

    Args:
        current_file: The artifact just written.
        keep: Generations to retain including *current_file*.
        protect: Additional names never to delete (e.g. the artifact workspace
            metadata still references).

    Returns:
        The paths actually deleted.
    """
    return await asyncio.to_thread(prune_superseded_sync, current_file, keep, protect)

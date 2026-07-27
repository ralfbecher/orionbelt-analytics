"""Optional in-process SHACL validation of generated/loaded ontologies.

Validates an OBA ontology against the published shapes in
``ontology/oba-shacl.ttl`` using ``pyshacl``. Both the dependency and the shapes
file are treated as optional: if ``pyshacl`` is not installed or the shapes file
is not found, validation degrades to a no-op (``available=False``) instead of
raising. This keeps ontology generation/import working everywhere while enabling
real conformance checking wherever the pieces are present.
"""

from __future__ import annotations

import logging
from typing import Any

from .paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

# SHACL shapes live in the repo's ontology/ directory (force-included in the
# wheel). Resolve lazily so a missing file is a graceful skip, not an error.
_SHAPES_PATH = PROJECT_ROOT / "ontology" / "oba-shacl.ttl"


def shacl_available() -> bool:
    """True if pyshacl is importable and the shapes file exists."""
    if not _SHAPES_PATH.exists():
        return False
    try:
        import pyshacl  # noqa: F401
    except Exception:
        return False
    return True


def validate_ontology(ontology_ttl: str) -> dict[str, Any]:
    """Validate ontology Turtle against the OBA SHACL shapes.

    Returns a structured report. When validation cannot run (missing dependency
    or shapes file), returns ``{"available": False, ...}`` and never raises.

    Args:
        ontology_ttl: The ontology in Turtle format.

    Returns:
        Dict with keys: ``available`` (bool), ``conforms`` (bool|None),
        ``violations`` (int), ``report`` (str summary).
    """
    if not _SHAPES_PATH.exists():
        return {
            "available": False,
            "conforms": None,
            "violations": 0,
            "report": f"SHACL shapes not found at {_SHAPES_PATH}; validation skipped.",
        }

    try:
        from pyshacl import validate as _pyshacl_validate
    except Exception:
        return {
            "available": False,
            "conforms": None,
            "violations": 0,
            "report": "pyshacl not installed; SHACL validation skipped.",
        }

    try:
        # NB: the OBA vocabulary (oba.ttl) is intentionally NOT merged as an
        # ont_graph. The RelationshipShape targets every owl:ObjectProperty, and
        # merging the vocab would pull in meta-level term declarations (e.g.
        # oba:joinsTo) as focus nodes, producing spurious violations. The shapes
        # target by RDF types already present in the generated data graph, so the
        # vocabulary is not needed for correct validation.
        conforms, _results_graph, results_text = _pyshacl_validate(
            data_graph=ontology_ttl,
            shacl_graph=_SHAPES_PATH.read_text(encoding="utf-8"),
            data_graph_format="turtle",
            shacl_graph_format="turtle",
            inference="none",
            advanced=True,
            meta_shacl=False,
        )

        violations = (
            0 if conforms else max(results_text.count("Constraint Violation"), 1)
        )
        return {
            "available": True,
            "conforms": bool(conforms),
            "violations": violations,
            "report": results_text.strip(),
        }
    except Exception as e:  # never let validation break generation/import
        logger.warning("SHACL validation errored, skipping: %s", e)
        return {
            "available": False,
            "conforms": None,
            "violations": 0,
            "report": f"SHACL validation errored: {e}",
        }

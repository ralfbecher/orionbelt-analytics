"""Typed request-scoped services passed to handler functions.

Previously each tool wrapper in ``src/main.py`` threaded a handful of helper
callables into its handler as separate keyword arguments (``get_session_data``,
``create_error_response``, ``get_oxigraph_store``, ...). That made every wrapper
and every handler signature long and easy to get out of sync.

:class:`HandlerContext` bundles those services into one typed object. A wrapper
builds it once (capturing the current module-level helpers, so test patches on
``src.main`` / ``src.server_state`` are still honored) and passes ``services=``;
each handler reads what it needs as ``services.<name>``.

The callable fields are non-optional and default to :func:`_unset_service`, a
sentinel that raises if a handler reaches for a service its wrapper did not
provide. This keeps the fields statically callable (no ``Optional[Callable]``
"None not callable" noise) while still failing loudly on real misuse.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _unset_service(*_args: Any, **_kwargs: Any) -> Any:
    """Placeholder for a service the wrapper did not supply."""
    raise RuntimeError(
        "HandlerContext service was not provided by the caller. "
        "Build the context with the helper this handler needs."
    )


@dataclass(frozen=True)
class HandlerContext:
    """Request-scoped services available to handler functions.

    Every wrapper populates the services its handler uses via ``_services()``;
    any field left at :func:`_unset_service` raises if actually called.
    """

    get_session_data: Callable[..., Any] = _unset_service
    get_session_db_manager: Callable[..., Any] = _unset_service
    get_session_safe_filename: Callable[..., Any] = _unset_service
    get_session_obqc_validator: Callable[..., Any] = _unset_service
    get_oxigraph_store: Callable[..., Any] = _unset_service
    load_ontology_from_session: Callable[..., Any] = _unset_service
    create_error_response: Callable[..., Any] = _unset_service
    server_state: Any = None
    get_connection_fingerprint: Callable[..., Any] = _unset_service
    clear_session_state: Callable[..., Any] = _unset_service
    auto_initialize_graphrag_background: Callable[..., Any] = _unset_service
    add_resource: Callable[..., Any] = _unset_service

    def provides(self, name: str) -> bool:
        """Return True if the named service was supplied (not the sentinel)."""
        return getattr(self, name) is not _unset_service

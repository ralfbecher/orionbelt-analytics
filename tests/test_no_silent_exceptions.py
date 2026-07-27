"""Guard against silently swallowed exceptions.

`except Exception` is a legitimate and necessary pattern at the MCP tool
boundary: a handler must convert failures into an error response rather than
let an arbitrary exception escape into the protocol layer. Ruff's BLE001 flags
every one of those (167 at the time of writing), so enabling it would mean 167
`noqa` comments -- noise that buys no safety.

What actually matters is narrower: a broad handler must not *discard* what it
caught. This test encodes that invariant directly. A handler passes if it does
any of:

  * logs (logger.debug/info/warning/error/exception/critical), or
  * re-raises, or
  * references the bound exception (e.g. folds the message into a returned
    error payload).

A handler that does none of those has thrown information away, and the failure
becomes undiagnosable in production. That is the real defect BLE001 gestures at.
"""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


def _python_sources():
    """Every first-party module, including the root entry point."""
    yield from sorted((PROJECT_ROOT / "src").rglob("*.py"))
    yield PROJECT_ROOT / "server.py"


def _is_broad(handler: ast.ExceptHandler) -> bool:
    """True for `except:` and `except Exception:` (the catch-alls)."""
    exc = handler.type
    if exc is None:
        return True
    return isinstance(exc, ast.Name) and exc.id == "Exception"


def _handles_exception(handler: ast.ExceptHandler) -> bool:
    """True if the handler logs, re-raises, or uses the caught exception."""
    body = ast.Module(body=handler.body, type_ignores=[])

    for node in ast.walk(body):
        if isinstance(node, ast.Raise):
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in LOG_METHODS
        ):
            return True

    if handler.name:
        return any(
            isinstance(node, ast.Name) and node.id == handler.name
            for node in ast.walk(body)
        )
    return False


def test_no_broad_handler_discards_its_exception():
    """No `except Exception` may drop the error without a trace."""
    offenders = []

    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and _is_broad(node)
                and not _handles_exception(node)
            ):
                rel = path.relative_to(PROJECT_ROOT)
                offenders.append(f"{rel}:{node.lineno}")

    assert not offenders, (
        "Broad `except Exception` handlers that discard the exception "
        "(no log, no re-raise, never referenced):\n  "
        + "\n  ".join(offenders)
        + "\n\nLog it, re-raise it, or fold it into the returned error payload."
    )


def test_guard_detects_a_planted_silent_handler():
    """The guard must actually catch a violation, not vacuously pass."""
    planted = ast.parse(
        "try:\n" "    risky()\n" "except Exception:\n" "    fallback = 0\n"
    )
    handlers = [n for n in ast.walk(planted) if isinstance(n, ast.ExceptHandler)]
    assert len(handlers) == 1
    assert _is_broad(handlers[0])
    assert not _handles_exception(handlers[0]), "guard failed to flag a silent handler"


def test_guard_accepts_a_handler_that_logs():
    """A logging handler is accepted."""
    ok = ast.parse(
        "try:\n"
        "    risky()\n"
        "except Exception as exc:\n"
        "    logger.warning('failed: %s', exc)\n"
    )
    handler = next(n for n in ast.walk(ok) if isinstance(n, ast.ExceptHandler))
    assert _handles_exception(handler)

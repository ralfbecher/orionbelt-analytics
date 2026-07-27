"""Guard against blocking file I/O on the event loop.

Ruff's ASYNC230 only matches a literal ``open()`` lexically inside an
``async def``. That misses two large categories which stall the loop just as
badly:

  * ``Path.read_text() / write_text() / read_bytes() / write_bytes()``,
    ``json.load/dump``, ``shutil.rmtree`` called directly in an async body
  * a *sync* helper that does any of the above, called from an async function
    without ``asyncio.to_thread``

Both were present after the first pass at the ASYNC230 fix, which is why this
test exists: the lint rule is not a sufficient check, so the invariant is
enforced directly.

Legitimate escape hatch: work wrapped in ``asyncio.to_thread``. Those payloads
live in nested ``def``s, which this walker deliberately does not descend into.
"""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

BLOCKING_METHODS = {"read_text", "write_text", "read_bytes", "write_bytes"}
BLOCKING_MODULE_CALLS = {
    ("json", "dump"),
    ("json", "load"),
    ("shutil", "rmtree"),
    ("shutil", "copy"),
    ("shutil", "copytree"),
    ("shutil", "move"),
}


def _describe_blocking_call(node: ast.Call) -> str | None:
    """Name the blocking file operation this call performs, if any."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "open":
        return "open()"
    if isinstance(func, ast.Attribute):
        if func.attr in BLOCKING_METHODS:
            return f".{func.attr}()"
        if isinstance(func.value, ast.Name):
            pair = (func.value.id, func.attr)
            if pair in BLOCKING_MODULE_CALLS:
                return f"{pair[0]}.{pair[1]}()"
    return None


class _ShallowBlockingVisitor(ast.NodeVisitor):
    """Collect blocking calls, refusing to descend into nested functions.

    Nested defs are where ``asyncio.to_thread`` payloads live -- that code is
    explicitly off the loop and must not be reported.
    """

    def __init__(self):
        self.found: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_Call(self, node):
        described = _describe_blocking_call(node)
        if described:
            self.found.append((node.lineno, described))
        self.generic_visit(node)


def _index_functions():
    """Map (path, qualname) -> (node, path, is_async) for every function in src/.

    Keyed by path as well as name: several handler functions share a name with
    the thin ``@mcp.tool()`` wrapper in src/main.py, and a name-only key would
    let one silently overwrite the other -- dropping real code from the scan.
    """
    functions = {}

    class Indexer(ast.NodeVisitor):
        def __init__(self, path):
            self.path = path
            self.stack: list[str] = []

        def _record(self, node, is_async):
            self.stack.append(node.name)
            functions[(self.path, ".".join(self.stack))] = (node, self.path, is_async)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node):
            self._record(node, False)

        def visit_AsyncFunctionDef(self, node):
            self._record(node, True)

    for path in sorted(SRC.rglob("*.py")):
        Indexer(path).visit(ast.parse(path.read_text(encoding="utf-8")))
    return functions


def _blocking_in_body(node) -> list[tuple[int, str]]:
    """Blocking calls in this function's own body (not nested defs)."""
    visitor = _ShallowBlockingVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return visitor.found


def test_no_direct_blocking_io_in_async_functions():
    """An async function must not touch the filesystem synchronously."""
    offenders = []
    for (_path, qualname), (node, path, is_async) in sorted(_index_functions().items()):
        if not is_async:
            continue
        for lineno, op in _blocking_in_body(node):
            rel = path.relative_to(PROJECT_ROOT)
            offenders.append(f"{rel}:{lineno}  {op}  in async {qualname}")

    assert not offenders, (
        "Blocking file I/O inside async functions -- this stalls the event "
        "loop for every concurrent session:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse the helpers in src/utils.py or wrap in asyncio.to_thread()."
    )


def test_async_callers_do_not_invoke_blocking_sync_helpers():
    """A sync helper doing file I/O must be reached via asyncio.to_thread.

    This is the case ASYNC230 cannot see at all: the blocking call is one
    function away, so the lint rule reports nothing while the loop still
    stalls.
    """
    functions = _index_functions()
    # simple name -> called-from-async, for sync helpers that block
    blocking_helpers = {
        qualname.rsplit(".", 1)[-1]
        for (_p, qualname), (node, _path, is_async) in functions.items()
        if not is_async and _blocking_in_body(node)
    }

    offenders = []
    for (_path, qualname), (node, path, is_async) in sorted(functions.items()):
        if not is_async:
            continue
        source = ast.unparse(node)
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name not in blocking_helpers:
                continue
            # Accept it when the async function hands the helper to to_thread.
            if "to_thread" in source and name in source:
                continue
            rel = path.relative_to(PROJECT_ROOT)
            offenders.append(f"{rel}:{call.lineno}  {name}()  from async {qualname}")

    assert not offenders, (
        "Async functions calling blocking sync helpers without "
        "asyncio.to_thread:\n  " + "\n  ".join(offenders)
    )


def test_guard_detects_a_planted_blocking_call():
    """The guard must actually fire, not vacuously pass."""
    planted = ast.parse(
        "async def handler(path):\n    return path.read_text(encoding='utf-8')\n"
    )
    fn = planted.body[0]
    assert _blocking_in_body(fn) == [(2, ".read_text()")]


def test_guard_ignores_to_thread_payloads():
    """Work inside a nested def (the to_thread payload) is not a violation."""
    ok = ast.parse(
        "async def handler(path):\n"
        "    def _read():\n"
        "        return path.read_text(encoding='utf-8')\n"
        "    return await asyncio.to_thread(_read)\n"
    )
    fn = ok.body[0]
    assert _blocking_in_body(fn) == []

"""Global test fixtures for OrionBelt Analytics.

Patches OUTPUT_DIR to use pytest's tmp_path for all tests, preventing
test artifacts from polluting the real tmp/ directory.
"""

import asyncio
import contextlib

import pytest


@pytest.fixture(autouse=True)
async def drain_background_tasks():
    """Cancel fire-and-forget tasks a test leaves behind.

    Handlers spawn GraphRAG and RDF initialization as background tasks and
    stash the handle on the session. Production cancels those in
    ServerState.cleanup_session(), but tests drive the handlers with Mock
    sessions that never go through that path -- so without this the task
    survives the test and gets destroyed mid-await at loop teardown
    ("Task was destroyed but it is pending!"), which both adds noise and
    hides whether the background path completed.
    """
    yield
    current = asyncio.current_task()
    leftover = [
        task for task in asyncio.all_tasks() if task is not current and not task.done()
    ]
    for task in leftover:
        task.cancel()
    if leftover:
        await asyncio.gather(*leftover, return_exceptions=True)


@pytest.fixture(autouse=True)
def isolate_output_dir(tmp_path, monkeypatch):
    """Redirect OUTPUT_DIR to a temp directory for every test.

    This prevents tests from writing metadata.json, ontology files,
    and other artifacts into the project's real tmp/ directory.
    """
    test_output = tmp_path / "output"
    test_output.mkdir()

    # Patch the canonical source
    monkeypatch.setattr("src.paths.OUTPUT_DIR", test_output)

    # Patch every module that imports OUTPUT_DIR at module level
    targets = [
        "src.handlers.connection",
        "src.handlers.schema",
        "src.handlers.ontology",
        "src.handlers.ontology_generation",
        "src.handlers.ontology_semantic",
        "src.handlers.ontology_artifacts",
        "src.handlers.workspace",
        "src.handlers.rdf",
        "src.handlers.graphrag",
        "src.workspace",
        "src.graphrag.vector_store_chromadb",
    ]
    for mod in targets:
        # Module may not be imported yet -- nothing to patch in that case.
        with contextlib.suppress(AttributeError):
            monkeypatch.setattr(f"{mod}.OUTPUT_DIR", test_output)

    return test_output

"""Tests for the non-blocking file I/O helpers in src.utils.

These back the ASYNC230 fix: handlers must not call blocking open() on the
event loop. The helpers have to behave exactly like the blocking calls they
replaced -- same encoding, same JSON formatting, same exceptions.
"""

import asyncio
import json

import pytest

from src.utils import read_json_file, read_text_file, write_json_file, write_text_file

UNICODE_SAMPLE = "hello äöü — ünicode ✓"


async def test_text_round_trip(tmp_path):
    """Text written by the helper reads back identically."""
    target = tmp_path / "ontology.ttl"
    await write_text_file(target, UNICODE_SAMPLE)
    assert await read_text_file(target) == UNICODE_SAMPLE


async def test_text_helpers_accept_str_paths(tmp_path):
    """Call sites pass both Path and str; both must work."""
    target = tmp_path / "ontology.ttl"
    await write_text_file(str(target), UNICODE_SAMPLE)
    assert await read_text_file(str(target)) == UNICODE_SAMPLE


async def test_json_round_trip(tmp_path):
    """JSON payloads survive the encode/decode hop unchanged."""
    target = tmp_path / "schema.json"
    payload = {"tables": [{"name": "users", "row_count": 3}], "note": "ü"}
    await write_json_file(target, payload)
    assert await read_json_file(target) == payload


async def test_json_formatting_matches_previous_json_dump(tmp_path):
    """Preserve indent=2 and ensure_ascii=False from the replaced json.dump call.

    The on-disk format is user-visible (workspace files are inspected and
    diffed), so the migration must not silently reformat it.
    """
    target = tmp_path / "schema.json"
    await write_json_file(target, {"note": "ü", "n": 1})
    raw = target.read_text(encoding="utf-8")

    assert raw.startswith("{\n  ")  # indent=2
    assert "ü" in raw  # ensure_ascii=False, not ü
    assert raw == json.dumps({"note": "ü", "n": 1}, indent=2, ensure_ascii=False)


async def test_missing_file_raises_like_blocking_open(tmp_path):
    """Errors propagate identically to the blocking calls these replaced."""
    with pytest.raises(FileNotFoundError):
        await read_text_file(tmp_path / "absent.ttl")
    with pytest.raises(FileNotFoundError):
        await read_json_file(tmp_path / "absent.json")


async def test_invalid_json_raises(tmp_path):
    """A corrupt workspace file still surfaces a JSONDecodeError."""
    target = tmp_path / "broken.json"
    await write_text_file(target, "{not json")
    with pytest.raises(json.JSONDecodeError):
        await read_json_file(target)


async def test_does_not_block_the_event_loop(tmp_path):
    """The whole point: other tasks keep running during file I/O.

    Writes a payload large enough to be non-trivial while a ticker task runs,
    and asserts the ticker was scheduled meanwhile -- which cannot happen if
    the write holds the loop.
    """
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    spinner = asyncio.create_task(ticker())
    try:
        target = tmp_path / "big.json"
        await write_json_file(target, {"rows": [{"i": i} for i in range(50_000)]})
        assert await read_json_file(target) is not None
    finally:
        spinner.cancel()

    assert ticks > 0, "event loop was blocked during file I/O"

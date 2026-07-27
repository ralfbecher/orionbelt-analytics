"""Tests for timezone-aware timestamp handling (DTZ005/DTZ006 migration).

Workspace metadata persists timestamps as ISO strings and later compares them
against the current time to decide retention. Once the writer switched to
timezone-aware UTC, any workspace already on disk still held *naive* strings --
and comparing naive against aware raises TypeError. These tests pin the
backward-compatible read path that prevents that regression.
"""

from datetime import UTC, datetime, timedelta

from src.utils import parse_timestamp, utc_now

# A timestamp as written by the pre-migration code: no offset.
LEGACY_NAIVE_ISO = "2020-01-01T12:00:00"


def test_utc_now_is_timezone_aware():
    """Everything written from now on must carry tzinfo."""
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_parse_timestamp_coerces_naive_to_utc():
    """Legacy naive workspace values are read as UTC rather than rejected."""
    parsed = parse_timestamp(LEGACY_NAIVE_ISO)
    assert parsed.tzinfo is not None
    assert parsed == datetime(2020, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_parse_timestamp_preserves_existing_offset():
    """An already-aware value must survive untouched."""
    aware = "2020-01-01T12:00:00+02:00"
    assert parse_timestamp(aware) == datetime.fromisoformat(aware)


def test_parse_timestamp_round_trips_what_utc_now_writes():
    """The write path and read path agree."""
    written = utc_now().isoformat()
    assert parse_timestamp(written) == datetime.fromisoformat(written)


def test_legacy_naive_timestamp_is_comparable_after_migration():
    """The actual regression: retention cleanup on a pre-existing workspace.

    Reading a legacy naive value with a bare ``datetime.fromisoformat`` and
    comparing it to an aware "now" raises TypeError, which would break startup
    cleanup for every workspace written before the migration. ``parse_timestamp``
    is what keeps that working.
    """
    # Bare parse is genuinely broken against an aware now -- this is the bug.
    naive = datetime.fromisoformat(LEGACY_NAIVE_ISO)
    try:
        naive < utc_now()
        raise AssertionError("expected TypeError from naive/aware comparison")
    except TypeError:
        pass

    # The shim makes the same on-disk value usable.
    coerced = parse_timestamp(LEGACY_NAIVE_ISO)
    assert coerced < utc_now()
    assert (utc_now() - coerced).days > 0


def test_retention_math_works_on_legacy_values():
    """Age computation over a legacy timestamp yields a sane positive age."""
    cutoff = utc_now() - timedelta(days=30)
    assert parse_timestamp(LEGACY_NAIVE_ISO) < cutoff

    recent = (utc_now() - timedelta(days=1)).isoformat()
    assert parse_timestamp(recent) > cutoff

"""Tests for pruning of superseded per-schema artifacts.

Artifact filenames carry a microsecond timestamp, so every regeneration writes
a new file while workspace metadata records only the latest. Before pruning,
every earlier generation stayed on disk unreferenced -- megabytes per ontology
on a large schema, and never reclaimed, because AUTO_CLEANUP_ON_STARTUP only
removes whole workspaces by age, not files inside a live one.
"""

from pathlib import Path

import pytest

from src.lifecycle.artifacts import (
    DEFAULT_KEEP_VERSIONS,
    family_key,
    get_keep_versions,
    prune_superseded_artifacts,
    prune_superseded_sync,
)

CONN = "ab12cd34"


def _write(directory: Path, name: str, mtime: float | None = None) -> Path:
    path = directory / name
    path.write_text("x", encoding="utf-8")
    if mtime is not None:
        import os

        os.utime(path, (mtime, mtime))
    return path


def _generation(schema: str, stamp: str, kind: str = "ontology", ext: str = ".ttl"):
    return f"{kind}_{CONN}_{schema}_{stamp}{ext}"


class TestFamilyKey:
    """The key must isolate one artifact kind, connection and schema."""

    def test_strips_the_full_timestamp_including_its_underscore(self):
        """The timestamp is %Y%m%d_%H%M%S%f -- it contains an underscore."""
        assert family_key("ontology_ab12cd34_public_20260727_132056962941.ttl") == (
            "ontology_ab12cd34_public",
            ".ttl",
        )

    def test_handles_the_shorter_background_timestamp(self):
        """The background ontology path uses %Y%m%d_%H%M%S (no microseconds)."""
        assert family_key("ontology_public_20260727_132056.ttl") == (
            "ontology_public",
            ".ttl",
        )

    def test_returns_none_when_there_is_no_timestamp(self):
        """Unrecognized names must not be pruned against."""
        assert family_key("ontology_upload.ttl") is None
        assert family_key("metadata.json") is None

    def test_schema_names_with_glob_metacharacters_stay_distinct(self):
        """Schema names may contain *, ? or [ -- they must not match siblings.

        schema_safe only replaces spaces and dots, so these survive into the
        filename. A glob built from such a name matched other schemas: a prune
        for "sales*" deleted "sales_eu"'s artifacts. Keys compare exactly.
        """
        star = family_key("ontology_ab12cd34_sales*_20260727_132056962941.ttl")
        sibling = family_key("ontology_ab12cd34_sales_eu_20260727_132056962941.ttl")
        assert star != sibling


class TestPruning:
    def test_keeps_current_plus_previous_generations(self, tmp_path):
        """Older generations beyond `keep` are removed."""
        files = [
            _write(
                tmp_path, _generation("public", f"20260727_1320569629{i:02d}"), mtime=i
            )
            for i in range(6)
        ]
        current = files[-1]

        removed = prune_superseded_sync(current, keep=3)

        assert current.exists()
        surviving = sorted(p.name for p in tmp_path.glob("ontology_*"))
        assert len(surviving) == 3
        assert len(removed) == 3

    def test_never_deletes_the_current_file_even_if_it_is_oldest(self, tmp_path):
        """The freshly written file is pinned regardless of mtime.

        Clock skew or a slow write must not make the file just referenced by
        metadata a deletion candidate.
        """
        current = _write(
            tmp_path, _generation("public", "20260727_132056000000"), mtime=1
        )
        for i in range(2, 6):
            _write(
                tmp_path, _generation("public", f"20260727_1320560000{i:02d}"), mtime=i
            )

        prune_superseded_sync(current, keep=2)

        assert current.exists(), "current artifact was pruned"

    def test_does_not_touch_other_schemas(self, tmp_path):
        """Pruning is scoped to one schema's family."""
        for i in range(4):
            _write(tmp_path, _generation("public", f"2026072{i}_132056962941"), mtime=i)
        other = [
            _write(tmp_path, _generation("sales", f"2026072{i}_132056962941"), mtime=i)
            for i in range(4)
        ]
        current = tmp_path / _generation("public", "20260723_132056962941")

        prune_superseded_sync(current, keep=1)

        assert all(p.exists() for p in other), "another schema's artifacts were pruned"

    def test_does_not_touch_other_artifact_kinds(self, tmp_path):
        """An ontology prune must not remove schema JSON or R2RML."""
        current = _write(
            tmp_path, _generation("public", "20260727_132056962941"), mtime=9
        )
        _write(tmp_path, _generation("public", "20260726_132056962941"), mtime=1)
        keep_json = _write(
            tmp_path, _generation("public", "20260726_132056962941", "schema", ".json")
        )
        keep_r2rml = _write(
            tmp_path, _generation("public", "20260726_132056962941", "r2rml", ".ttl")
        )

        prune_superseded_sync(current, keep=1)

        assert keep_json.exists()
        assert keep_r2rml.exists()

    def test_glob_metacharacters_in_schema_do_not_match_siblings(self, tmp_path):
        """Regression: pruning schema "sales*" must not delete "sales_eu".

        Reproduced against the previous glob-based implementation.
        """
        victim = _write(
            tmp_path, _generation("sales_eu", "20260101_120000000000"), mtime=1
        )
        for i in range(4):
            _write(tmp_path, _generation("sales*", f"2026010{i}_120000000000"), mtime=i)
        current = tmp_path / _generation("sales*", "20260103_120000000000")

        removed = prune_superseded_sync(current, keep=1)

        assert victim.exists(), "another schema's artifact was pruned"
        assert victim not in removed

    def test_protected_names_are_never_deleted(self, tmp_path):
        """Whatever metadata still references must survive any prune.

        Pruning runs after the metadata write, but if that write failed the old
        filename is still the one on disk that restore will look for.
        """
        still_referenced = _write(
            tmp_path, _generation("public", "20260101_120000000000"), mtime=1
        )
        for i in range(2, 5):
            _write(tmp_path, _generation("public", f"2026010{i}_120000000000"), mtime=i)
        current = tmp_path / _generation("public", "20260104_120000000000")

        removed = prune_superseded_sync(
            current, keep=1, protect=[still_referenced.name]
        )

        assert still_referenced.exists(), "the referenced artifact was pruned"
        assert still_referenced not in removed

    def test_protect_accepts_paths_as_well_as_names(self, tmp_path):
        """Callers hold either form; both must work."""
        referenced = _write(
            tmp_path, _generation("public", "20260101_120000000000"), mtime=1
        )
        current = _write(
            tmp_path, _generation("public", "20260102_120000000000"), mtime=2
        )

        prune_superseded_sync(current, keep=1, protect=[referenced])

        assert referenced.exists()

    def test_unrecognized_name_prunes_nothing(self, tmp_path):
        """Without a parseable timestamp the function must be inert."""
        current = _write(tmp_path, "ontology_upload.ttl")
        _write(tmp_path, "ontology_other.ttl")

        assert prune_superseded_sync(current, keep=1) == []
        assert len(list(tmp_path.glob("*.ttl"))) == 2

    def test_keep_is_clamped_to_at_least_one(self, tmp_path):
        """keep=0 must not delete the file currently in use."""
        current = _write(tmp_path, _generation("public", "20260727_132056962941"))
        prune_superseded_sync(current, keep=0)
        assert current.exists()

    async def test_async_wrapper_prunes(self, tmp_path):
        """The handler-facing entry point works off the event loop."""
        for i in range(4):
            _write(tmp_path, _generation("public", f"2026072{i}_132056962941"), mtime=i)
        current = tmp_path / _generation("public", "20260723_132056962941")

        removed = await prune_superseded_artifacts(current, keep=2)

        assert len(removed) == 2
        assert current.exists()


class TestKeepVersionsSetting:
    def test_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv("ARTIFACT_KEEP_VERSIONS", raising=False)
        assert get_keep_versions() == DEFAULT_KEEP_VERSIONS

    def test_reads_the_env_var(self, monkeypatch):
        monkeypatch.setenv("ARTIFACT_KEEP_VERSIONS", "7")
        assert get_keep_versions() == 7

    @pytest.mark.parametrize("value", ["0", "-3"])
    def test_clamps_to_one(self, monkeypatch, value):
        """Retaining zero generations would delete the file in use."""
        monkeypatch.setenv("ARTIFACT_KEEP_VERSIONS", value)
        assert get_keep_versions() == 1

    def test_falls_back_on_garbage(self, monkeypatch):
        monkeypatch.setenv("ARTIFACT_KEEP_VERSIONS", "not-a-number")
        assert get_keep_versions() == DEFAULT_KEEP_VERSIONS

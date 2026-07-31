"""Regression tests for Oxigraph SPARQL read/write and store correctness.

Cover the paths that broke against pyoxigraph 0.5.x (issue #38):
- ``add_triple`` (default and named graph) — must build a ``Quad``.
- ``query_sparql`` SELECT — ``QuerySolution`` no longer exposes ``.items()``.
- ``query_sparql_ask`` ASK — ``QueryBoolean`` is not iterable.
- ``add_knowledge`` — exercises ``add_triple`` plus metadata triples.

Plus the follow-up RDF correctness fixes:
- ``schema_graph_uri`` — single source of truth so store/export/auto-persist
  agree on the named-graph URI (F3).
- ``delete_graph`` — version cleanup must actually drop triples (F2).
- ``query_sparql_construct`` — serialize via ``RdfFormat.TURTLE`` (F5).
- ``query_sparql`` best-effort timeout watchdog (F1).

They run against the real, pinned pyoxigraph (no mocks) so an API drift like
the 0.3.x -> 0.5.x break is caught instead of silently shipping.
"""

import shutil
import tempfile
import time
from pathlib import Path

import pytest

from src.oxigraph_store import (
    OXIGRAPH_AVAILABLE,
    OxigraphStoreManager,
    schema_graph_uri,
)

pytestmark = pytest.mark.skipif(not OXIGRAPH_AVAILABLE, reason="Oxigraph not available")

EX = "http://example.com/"


@pytest.fixture
def store():
    """A fresh on-disk Oxigraph store, cleaned up after the test."""
    temp_dir = Path(tempfile.mkdtemp())
    mgr = OxigraphStoreManager(store_path=temp_dir)
    yield mgr
    mgr.close()  # release RocksDB file handles before removing the dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestAddTripleAndSelect:
    """add_triple (write) + query_sparql (SELECT read) round-trips."""

    def test_add_uri_triple_default_graph_select(self, store):
        store.add_triple(f"{EX}s", f"{EX}p", f"{EX}o")

        results = store.query_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")

        assert results == [{"s": f"{EX}s", "p": f"{EX}p", "o": f"{EX}o"}]

    def test_add_literal_triple_select(self, store):
        store.add_triple(
            f"{EX}customers",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "Customer Master Data",
            object_is_literal=True,
        )

        results = store.query_sparql(
            "SELECT ?label WHERE { ?s "
            "<http://www.w3.org/2000/01/rdf-schema#label> ?label }"
        )

        assert results == [{"label": "Customer Master Data"}]

    def test_add_triple_named_graph_select(self, store):
        graph = f"{EX}graph1"
        store.add_triple(f"{EX}s", f"{EX}p", f"{EX}o", graph_uri=graph)

        # The triple is in the named graph, not the default graph.
        in_graph = store.query_sparql(
            f"SELECT ?s WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"
        )
        assert in_graph == [{"s": f"{EX}s"}]

    def test_unwrapped_pattern_spans_named_graphs(self, store):
        """Patterns outside GRAPH see named-graph triples, not an empty default.

        Ontologies are always loaded into a named graph, so without the union
        default graph an unwrapped pattern silently returns zero rows.
        """
        store.add_triple(f"{EX}a", f"{EX}p", f"{EX}o", graph_uri=f"{EX}g1")
        store.add_triple(f"{EX}b", f"{EX}p", f"{EX}o", graph_uri=f"{EX}g2")

        results = store.query_sparql(f"SELECT ?s WHERE {{ ?s <{EX}p> ?o }}")

        assert sorted(r["s"] for r in results) == [f"{EX}a", f"{EX}b"]

    def test_ask_and_construct_span_named_graphs(self, store):
        """ASK and CONSTRUCT use the same union default graph as SELECT."""
        store.add_triple(f"{EX}a", f"{EX}p", f"{EX}o", graph_uri=f"{EX}g1")

        assert store.query_sparql_ask(f"ASK {{ ?s <{EX}p> ?o }}") is True

        turtle = store.query_sparql_construct(
            f"CONSTRUCT {{ ?s <{EX}p> ?o }} WHERE {{ ?s <{EX}p> ?o }}"
        )
        assert f"{EX}a" in turtle

    def test_from_clause_still_isolates_one_graph(self, store):
        """FROM <g1> must not leak g2, i.e. the union must not override it.

        The union default graph is what lets an unwrapped pattern find triples
        at all, but applying it unconditionally overrode explicit dataset
        selection and broke every FROM-scoped helper.
        """
        store.add_triple(f"{EX}a", f"{EX}p", f"{EX}o", graph_uri=f"{EX}g1")
        store.add_triple(f"{EX}b", f"{EX}p", f"{EX}o", graph_uri=f"{EX}g2")

        results = store.query_sparql(
            f"SELECT ?s FROM <{EX}g1> WHERE {{ ?s <{EX}p> ?o }}"
        )

        assert [r["s"] for r in results] == [f"{EX}a"]

    def test_from_named_still_isolates_one_graph(self, store):
        """FROM NAMED + GRAPH ?g must not see graphs outside the dataset."""
        store.add_triple(f"{EX}a", f"{EX}p", f"{EX}o", graph_uri=f"{EX}g1")
        store.add_triple(f"{EX}b", f"{EX}p", f"{EX}o", graph_uri=f"{EX}g2")

        results = store.query_sparql(
            f"SELECT ?s FROM NAMED <{EX}g1> WHERE {{ GRAPH ?g {{ ?s <{EX}p> ?o }} }}"
        )

        assert [r["s"] for r in results] == [f"{EX}a"]

    def test_ask_and_construct_respect_from(self, store):
        """ASK and CONSTRUCT honour explicit dataset selection too."""
        store.add_triple(f"{EX}b", f"{EX}p", f"{EX}o", graph_uri=f"{EX}g2")

        # g1 is empty, so a query scoped to it must find nothing.
        assert store.query_sparql_ask(f"ASK FROM <{EX}g1> {{ ?s <{EX}p> ?o }}") is False

        turtle = store.query_sparql_construct(
            f"CONSTRUCT {{ ?s <{EX}p> ?o }} FROM <{EX}g1> WHERE {{ ?s <{EX}p> ?o }}"
        )
        assert f"{EX}b" not in turtle

    def test_from_inside_literal_or_comment_does_not_suppress_union(self, store):
        """Only a real FROM keyword counts -- not one in a literal or comment."""
        store.add_triple(
            f"{EX}a",
            f"{EX}p",
            "FROM <x>",
            object_is_literal=True,
            graph_uri=f"{EX}g1",
        )

        # 'FROM' appears in the literal and in a comment, but the query selects
        # no dataset, so the union must still apply or this returns nothing.
        results = store.query_sparql(
            f'# scoped? no FROM here\nSELECT ?s WHERE {{ ?s <{EX}p> "FROM <x>" }}'
        )

        assert [r["s"] for r in results] == [f"{EX}a"]

    def test_list_tables_sparql_is_scoped_to_its_schema(self, store):
        """The FROM-based helper must not report tables from other schemas."""
        table = "https://ralforion.com/ns/oba#Table"
        name = "https://ralforion.com/ns/oba#tableName"
        for graph, label in ((f"{EX}g1", "table_g1"), (f"{EX}g2", "table_g2")):
            store.add_triple(
                f"{EX}{label}",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                table,
                graph_uri=graph,
            )
            store.add_triple(
                f"{EX}{label}", name, label, object_is_literal=True, graph_uri=graph
            )

        assert store.list_tables_sparql(f"{EX}g1") == ["table_g1"]

    def test_select_unbound_variable_is_omitted(self, store):
        store.add_triple(f"{EX}s", f"{EX}p", f"{EX}o")

        results = store.query_sparql(
            "SELECT ?s ?missing WHERE { ?s ?p ?o OPTIONAL { ?s "
            f"<{EX}none> ?missing }} }}"
        )

        assert results == [{"s": f"{EX}s"}]

    def test_select_empty_result(self, store):
        results = store.query_sparql("SELECT ?s WHERE { ?s ?p ?o }")
        assert results == []


class TestAsk:
    """query_sparql_ask round-trips."""

    def test_ask_true(self, store):
        store.add_triple(f"{EX}s", f"{EX}p", f"{EX}o")
        assert store.query_sparql_ask("ASK { ?s ?p ?o }") is True

    def test_ask_false(self, store):
        assert store.query_sparql_ask("ASK { ?s ?p ?o }") is False


class TestAddKnowledge:
    """add_knowledge writes the main triple plus metadata triples."""

    def test_add_knowledge_roundtrip(self, store):
        subject = f"{EX}pattern/sales"
        predicate = f"{EX}schema#hasSQL"
        sql = "SELECT customer_id FROM orders"

        # add_knowledge writes into its default named graph.
        kg = "http://example.com/knowledge"
        store.add_knowledge(
            subject,
            predicate,
            sql,
            metadata={"confidence": "0.95"},
        )

        results = store.query_sparql(
            f"SELECT ?o WHERE {{ GRAPH <{kg}> {{ <{subject}> <{predicate}> ?o }} }}"
        )
        assert results == [{"o": sql}]

        # Metadata is stored as an additional triple on the same subject.
        meta = store.query_sparql(
            f"SELECT ?c WHERE {{ GRAPH <{kg}> {{ <{subject}> "
            f"<{EX}metadata#confidence> ?c }} }}"
        )
        assert meta == [{"c": "0.95"}]


class TestSchemaGraphUri:
    """The shared graph-URI helper that keeps store/export/auto-persist aligned."""

    def test_plain_name(self):
        assert schema_graph_uri("public") == "http://example.com/schema/public"

    def test_spaces_and_dots_are_made_safe(self):
        # Spaces/dots must be normalized identically everywhere, or a manual
        # store writes a graph the export can't find.
        assert schema_graph_uri("my db.public") == (
            "http://example.com/schema/my_db_public"
        )

    def test_store_and_export_use_the_same_graph(self, store):
        # Round-trip: load into the canonical graph, export from the same URI.
        graph = schema_graph_uri("sales.reporting")
        store.add_triple(f"{EX}t", f"{EX}p", f"{EX}o", graph_uri=graph)

        exported = store.export_graph(graph)

        assert f"{EX}t" in exported


class TestDeleteGraph:
    """delete_graph removes a named graph's triples (used by version cleanup)."""

    def test_delete_removes_triples(self, store):
        graph = schema_graph_uri("doomed")
        store.add_triple(f"{EX}s", f"{EX}p", f"{EX}o", graph_uri=graph)
        assert store.export_graph(graph)  # non-empty before

        store.delete_graph(graph)

        results = store.query_sparql(
            f"SELECT ?s WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"
        )
        assert results == []

    def test_delete_missing_graph_is_noop(self, store):
        # Cleanup may call this for a graph that was never persisted.
        store.delete_graph(schema_graph_uri("never-existed"))


class TestConstruct:
    """CONSTRUCT serialization (F5: RdfFormat.TURTLE, not the deprecated string)."""

    def test_construct_returns_turtle(self, store):
        store.add_triple(f"{EX}s", f"{EX}p", f"{EX}o")

        turtle = store.query_sparql_construct(
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"
        )

        assert f"{EX}s" in turtle and f"{EX}o" in turtle

    def test_construct_empty_result_is_empty_string(self, store):
        turtle = store.query_sparql_construct(
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"
        )
        assert turtle == ""


class TestQueryTimeout:
    """Best-effort SELECT timeout watchdog (F1)."""

    def test_no_timeout_returns_normally(self, store):
        store.add_triple(f"{EX}s", f"{EX}p", f"{EX}o")
        assert store.query_sparql(
            "SELECT ?s WHERE { ?s ?p ?o }", timeout_seconds=None
        ) == [{"s": f"{EX}s"}]

    def test_within_timeout_returns(self, store):
        store.add_triple(f"{EX}s", f"{EX}p", f"{EX}o")
        assert store.query_sparql(
            "SELECT ?s WHERE { ?s ?p ?o }", timeout_seconds=30
        ) == [{"s": f"{EX}s"}]

    def test_timeout_raises(self, store, monkeypatch):
        # Simulate a slow query by making the SELECT core block past the timeout.
        def slow_select(_query):
            time.sleep(2)
            return []

        monkeypatch.setattr(store, "_execute_select", slow_select)

        with pytest.raises(TimeoutError):
            store.query_sparql("SELECT ?s WHERE { ?s ?p ?o }", timeout_seconds=1)


class TestStoreLock:
    """A live RocksDB LOCK must never be auto-deleted (F1)."""

    def test_second_open_raises_and_preserves_lock(self):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            mgr = OxigraphStoreManager(store_path=temp_dir)
            lock = temp_dir / "LOCK"
            assert lock.exists()

            # A second open while the first holds the store must fail, NOT delete
            # the active lock (which would invite two-process corruption).
            with pytest.raises(OSError):
                OxigraphStoreManager(store_path=temp_dir)

            assert lock.exists()  # lock left intact
            mgr.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

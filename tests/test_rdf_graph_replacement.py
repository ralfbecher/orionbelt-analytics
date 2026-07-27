"""A schema's named graph holds one generation of its ontology, not a union.

pyoxigraph's ``store.load(..., to_graph=...)`` unions into the target graph.
Nothing cleared it, so every regeneration accumulated: drop a table, regenerate,
and the dropped table stayed queryable through query_sparql() and RDF exports
forever. No concurrency needed -- a single sequential regeneration was enough.

That also made guarding only the metadata flag insufficient. A superseded
request whose load had already landed left the graph holding the wrong
ontology while the flag correctly named the newer one.
"""

import re

import pytest

from src.lifecycle.metadata import ontology_is_current, update_workspace_section
from src.oxigraph_store import OxigraphStoreManager

OWL_CLASS = "<http://www.w3.org/2002/07/owl#Class>"
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
GRAPH = "http://example.com/schema/public"


def _ttl(*names: str) -> str:
    return "\n".join(
        f"<http://example.com/{n}> {RDF_TYPE} {OWL_CLASS} ." for n in names
    )


def _classes(store: OxigraphStoreManager, graph: str = GRAPH) -> set[str]:
    rows = store.query_sparql(
        f"SELECT ?c WHERE {{ GRAPH <{graph}> {{ ?c a {OWL_CLASS} }} }}"
    )
    return set(re.findall(r"example\.com/(\w+)", str(rows)))


@pytest.fixture
def store():
    return OxigraphStoreManager()


def test_regeneration_drops_entities_that_no_longer_exist(store):
    """The headline case: a dropped table must stop being queryable."""
    store.load_ontology(_ttl("Customers", "DroppedTable"), GRAPH, "public")
    store.load_ontology(_ttl("Customers"), GRAPH, "public")

    assert _classes(store) == {"Customers"}, "stale entity survived regeneration"


def test_regeneration_reports_the_graph_size_not_the_delta(store):
    """The old count was a store delta, so an unchanged reload reported 0."""
    assert store.load_ontology(_ttl("A", "B"), GRAPH, "public") == 2
    assert (
        store.load_ontology(_ttl("A", "B"), GRAPH, "public") == 2
    ), "reloading the same ontology should still report its size"
    assert store.load_ontology(_ttl("A"), GRAPH, "public") == 1


def test_replacement_is_scoped_to_one_schema_graph(store):
    """Loading one schema must not disturb another."""
    other = "http://example.com/schema/sales"
    store.load_ontology(_ttl("Orders"), other, "sales")
    store.load_ontology(_ttl("Customers", "Temp"), GRAPH, "public")
    store.load_ontology(_ttl("Customers"), GRAPH, "public")

    assert _classes(store) == {"Customers"}
    assert _classes(store, other) == {"Orders"}, "another schema's graph was cleared"


async def test_superseded_generation_is_refused_before_it_can_load(tmp_path):
    """The currency check must gate the RDF write, not just the flag.

    load_ontology replaces the graph, so a stale request reaching the load
    would overwrite the newer generation's triples -- leaving the flag honest
    and the graph wrong.
    """
    cid, schema = "rdfguard", "public"
    await update_workspace_section(
        cid, tmp_path, schema, "ontology", {"ontology_file": "B.ttl"}
    )

    assert await ontology_is_current(cid, tmp_path, schema, "B.ttl") is True
    assert await ontology_is_current(cid, tmp_path, schema, "A.ttl") is False


async def test_first_generation_is_allowed_through(tmp_path):
    """Nothing recorded yet means this caller is the first -- do not block it."""
    assert await ontology_is_current("fresh", tmp_path, "public", "A.ttl") is True


class TestFailedLoadPreservesPreviousGeneration:
    """A failed refresh must not destroy the last queryable ontology.

    Making the load a replacement introduced this: the target graph was cleared
    before the new TTL had parsed, so a malformed ontology left the graph empty
    and SPARQL answering nothing. The new data is now staged and swapped in only
    after it loads successfully.
    """

    def test_malformed_turtle_leaves_the_existing_graph_intact(self, store):
        store.load_ontology(_ttl("Customers"), GRAPH, "public")

        with pytest.raises(Exception):
            store.load_ontology("this is not valid turtle @@@", GRAPH, "public")

        assert _classes(store) == {
            "Customers"
        }, "a failed load destroyed the previous ontology"

    def test_graph_survives_repeated_failures(self, store):
        """Retrying a bad ontology must not erode the good one."""
        store.load_ontology(_ttl("Customers", "Orders"), GRAPH, "public")

        for _ in range(5):
            with pytest.raises(Exception):
                store.load_ontology("@@@ broken", GRAPH, "public")

        assert _classes(store) == {"Customers", "Orders"}

    def test_staging_graphs_are_not_left_behind(self, store):
        """Clearing a graph's quads does not unregister it in pyoxigraph.

        Without an explicit remove_graph the store accumulated one empty
        staging graph per load, which show up in named_graphs() and in
        cross-graph SPARQL.
        """
        store.load_ontology(_ttl("Customers"), GRAPH, "public")
        for _ in range(5):
            with pytest.raises(Exception):
                store.load_ontology("@@@ broken", GRAPH, "public")
            store.load_ontology(_ttl("Customers"), GRAPH, "public")

        graphs = [str(g) for g in store.store.named_graphs()]
        assert not [
            g for g in graphs if "__staging" in g
        ], f"staging graphs leaked: {graphs}"
        assert len(graphs) == 1

    def test_successful_load_still_replaces(self, store):
        """Staging must not accidentally turn replacement back into append."""
        store.load_ontology(_ttl("Customers", "DroppedTable"), GRAPH, "public")
        store.load_ontology(_ttl("Customers"), GRAPH, "public")
        assert _classes(store) == {"Customers"}

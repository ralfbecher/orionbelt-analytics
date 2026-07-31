"""
Oxigraph RDF Store Manager

Provides persistent RDF storage with SPARQL 1.1 query support using Oxigraph.
Stores ontologies, schema metadata, and accumulated knowledge across sessions.
"""

import contextlib
import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

if TYPE_CHECKING:
    from pyoxigraph import (
        Literal,
        NamedNode,
        Quad,
        QuerySolutions,
        QueryTriples,
        RdfFormat,
        Store,
    )

    OXIGRAPH_AVAILABLE: bool
else:
    try:
        from pyoxigraph import Literal, NamedNode, Quad, RdfFormat, Store

        OXIGRAPH_AVAILABLE = True
    except ImportError:
        OXIGRAPH_AVAILABLE = False
        Store = None
        NamedNode = None
        RdfFormat = None
        Literal = None
        Quad = None

logger = logging.getLogger(__name__)

# Single source of truth for the named-graph URI a schema's RDF is stored under.
# Manual persistence, auto-persistence, export/download, and SPARQL helpers must
# all agree on this, or e.g. a manual store writes to a graph the export can't
# find (see issue: graph-URI mismatch).
SCHEMA_GRAPH_PREFIX = "http://example.com/schema/"

# Private namespace for the transient graphs load_ontology() stages replacements
# in. Deliberately unrelated to any caller-supplied graph URI: deriving it from
# one produced invalid IRIs for URIs that already contained a fragment.
_STAGING_IRI_PREFIX = "urn:orionbelt:staging:"


def schema_graph_uri(schema_name: str) -> str:
    """Return the canonical named-graph URI for a schema's RDF.

    Args:
        schema_name: Schema identifier (may contain spaces/dots).

    Returns:
        Graph URI of the form ``http://example.com/schema/<safe-name>``.
    """
    schema_safe = schema_name.replace(" ", "_").replace(".", "_")
    return f"{SCHEMA_GRAPH_PREFIX}{schema_safe}"


def _escape_sparql_literal(value: str) -> str:
    """Escape a string for safe use as a SPARQL literal value.

    Prevents SPARQL injection by escaping special characters.
    """
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _escape_sparql_iri(value: str) -> str:
    """Escape a string for safe use as a SPARQL IRI.

    Prevents SPARQL injection by rejecting dangerous characters in IRIs.
    """
    forbidden = set('<>"{}|\\^`')
    return "".join(c for c in value if c not in forbidden)


# Blanks every region a bare "FROM" can appear in without being the dataset
# keyword, so the scan below sees only real syntax. Each alternative matters:
# a literal or comment may contain the word; an IRI path may end in ".../FROM";
# and -- less obviously -- `?from`, `oba:from` and `"x"@from` all put FROM
# between two word boundaries, so \bFROM\b alone matched them and wrongly
# suppressed the union default graph.
#
# Order matters. Literals come first so a '#' inside a string is not read as a
# comment, and variables/prefixed names come after IRIs so `<...>` wins.
_SPARQL_NON_KEYWORD_REGIONS = re.compile(
    r"'''.*?'''"  # long single-quoted literal
    r'|""".*?"""'  # long double-quoted literal
    r"|'(?:[^'\\\n]|\\.)*'"  # short single-quoted literal
    r'|"(?:[^"\\\n]|\\.)*"'  # short double-quoted literal
    r"|<[^<>\"{}|^`\\]*>"  # IRI reference
    r"|#[^\n]*"  # comment to end of line
    r"|[?$][A-Za-z0-9_]+"  # variable: ?from, $from
    r"|@[A-Za-z0-9-]+"  # language tag: "x"@from
    r"|[A-Za-z0-9_.\-]*:[A-Za-z0-9_.\-]*",  # prefixed name: oba:from, _:from
    re.DOTALL,
)

# A real dataset clause is the standalone keyword; by this point the tokens that
# merely contain it have been blanked out.
_SPARQL_FROM_KEYWORD = re.compile(r"\bFROM\b", re.IGNORECASE)


def _declares_dataset(sparql_query: str) -> bool:
    """Report whether a query selects its own RDF dataset via FROM / FROM NAMED.

    Such a query has already said exactly which graphs it wants, so the caller
    must not widen it (see :meth:`OxigraphStoreManager.query_sparql`).

    This is a lexical scan, not a parse -- pyoxigraph exposes no query AST to
    Python. It is deliberately biased towards *not* detecting a clause: a false
    positive silently narrows a query to an empty default graph and returns
    nothing, which is the failure this whole mechanism exists to prevent.

    Args:
        sparql_query: SPARQL query string.

    Returns:
        True if a standalone ``FROM`` keyword appears outside literals, IRIs,
        comments, variables, language tags and prefixed names.
    """
    return bool(
        _SPARQL_FROM_KEYWORD.search(_SPARQL_NON_KEYWORD_REGIONS.sub(" ", sparql_query))
    )


class OxigraphStoreManager:
    """Manages persistent RDF storage using Oxigraph."""

    def __init__(self, store_path: Path | None = None):
        """
        Initialize Oxigraph store manager.

        Args:
            store_path: Path to persistent store directory (None for in-memory)

        Raises:
            ImportError: If pyoxigraph is not installed
        """
        if not OXIGRAPH_AVAILABLE:
            raise ImportError(
                "pyoxigraph not installed. Install with: pip install pyoxigraph"
            )

        self.store_path = store_path

        if store_path:
            store_path.mkdir(parents=True, exist_ok=True)
            try:
                self.store = Store(str(store_path))
            except OSError as e:
                # A RocksDB LOCK in the store dir means another process currently
                # holds the store open. It is NOT stale just because it exists, and
                # auto-deleting it doesn't unblock this open — it only invites two
                # processes to corrupt the same database. Surface the error with
                # recovery guidance and let the operator remove the lock manually
                # if they are certain no other process is using the store.
                logger.error(
                    f"Failed to open Oxigraph store at {store_path}: {e}. "
                    "If you are certain no other process is using this store, "
                    f"remove {store_path / 'LOCK'} manually and retry."
                )
                raise
            logger.info(f"Initialized Oxigraph persistent store at: {store_path}")
        else:
            self.store = Store()
            logger.info("Initialized Oxigraph in-memory store")

        # Track loaded ontologies
        self._loaded_ontologies: dict[str, str] = {}  # schema_name -> graph_uri

    def load_ontology(self, ontology_ttl: str, graph_uri: str, schema_name: str) -> int:
        """
        Load ontology into the store.

        The named graph is **replaced**, not appended to. store.load() unions
        into the target graph, so repeated generations accumulated: regenerate
        after dropping a table and the dropped table stayed queryable forever,
        because nothing ever removed its triples. A named graph represents one
        schema's current ontology, so loading a new generation must supersede
        the old one.

        The replacement is staged. Clearing the target first meant a malformed
        TTL destroyed the last good ontology: the clear had already happened
        when the parser raised, leaving the graph empty and SPARQL answering
        nothing. So the new data is parsed into a temporary graph and swapped
        in only once it has loaded successfully -- a failed load leaves the
        previous generation exactly as it was. pyoxigraph's Store has no
        transaction API, so this is the available approximation.

        Args:
            ontology_ttl: Ontology in Turtle format
            graph_uri: Named graph URI for this ontology
            schema_name: Schema identifier

        Returns:
            Number of triples in the graph after loading.

        Raises:
            Exception: Whatever the parser raises for malformed input. The
                existing graph is left untouched in that case.
        """
        try:
            target = NamedNode(graph_uri)
            # Staging IRI comes from a private URN namespace rather than being
            # derived from graph_uri. Appending "#..." to the caller's URI
            # produced a second fragment for any graph URI that already had one
            # -- e.g. http://example.com/schema#public, which is perfectly valid
            # and reachable through the user-facing graph_uri tool parameter --
            # and pyoxigraph rejects that with "Invalid IRI code point '#'".
            # A uuid alone gives the uniqueness staging needs; the name never
            # escapes this method.
            staging = NamedNode(f"{_STAGING_IRI_PREFIX}{uuid4().hex}")

            def _clear(graph: NamedNode) -> None:
                """Remove a graph's quads, leaving the graph itself in place."""
                for quad in list(self.store.quads_for_pattern(None, None, None, graph)):
                    self.store.remove(quad)

            def _drop_staging() -> None:
                """Remove the staging graph entirely.

                Emptying it is not enough: pyoxigraph tracks graph existence
                separately from its contents, so a cleared staging graph still
                appears in named_graphs() and one would accumulate per load.
                """
                _clear(staging)
                with contextlib.suppress(Exception):
                    self.store.remove_graph(staging)

            # Parse into staging first; a failure here must not touch target.
            try:
                # Use RdfFormat.TURTLE for newer versions, fall back to strings
                # for older versions
                try:
                    # Try with RdfFormat object (pyoxigraph >= 0.4.0)
                    self.store.load(
                        ontology_ttl.encode("utf-8"),
                        format=RdfFormat.TURTLE,
                        base_iri=graph_uri,
                        to_graph=staging,
                    )
                except (TypeError, AttributeError):
                    # Fallback for older pyoxigraph versions
                    try:
                        self.store.load(
                            ontology_ttl.encode("utf-8"),
                            format="text/turtle",  # type: ignore[arg-type]
                            base_iri=graph_uri,
                            to_graph=staging,
                        )
                    except TypeError:
                        # Final fallback for very old versions using mime_type
                        self.store.load(  # type: ignore[call-arg]
                            ontology_ttl.encode("utf-8"),
                            mime_type="text/turtle",
                            base_iri=graph_uri,
                            to_graph=staging,
                        )
            except Exception:
                _drop_staging()
                logger.warning(
                    f"Ontology load failed for graph <{graph_uri}>; "
                    "previous generation left in place"
                )
                raise

            # Swap: the new generation parsed cleanly, so it may supersede.
            staged = list(self.store.quads_for_pattern(None, None, None, staging))
            try:
                _clear(target)
                for quad in staged:
                    self.store.add(
                        Quad(quad.subject, quad.predicate, quad.object, target)
                    )
            finally:
                _drop_staging()

            triples_loaded = len(staged)

            self._loaded_ontologies[schema_name] = graph_uri

            logger.info(
                f"Loaded ontology for schema '{schema_name}' into graph <{graph_uri}>: "
                f"{triples_loaded} triples"
            )

            return triples_loaded

        except Exception as e:
            logger.exception(f"Failed to load ontology: {e}")
            raise

    def query_sparql(
        self, sparql_query: str, timeout_seconds: int | None = 30
    ) -> list[dict[str, Any]]:
        """
        Execute SPARQL query.

        Patterns outside a ``GRAPH`` clause are matched against the union of all
        named graphs, so callers do not have to know which graph a schema was
        loaded into. Use ``GRAPH ?g { ... }`` to scope to one schema or to bind
        the source graph.

        A query that selects its own dataset with ``FROM`` / ``FROM NAMED`` is
        left alone: it has already stated which graphs it wants, and widening it
        would leak triples across schemas (``FROM <g1>`` would also return g2).

        Args:
            sparql_query: SPARQL query string
            timeout_seconds: Query timeout (None for no timeout)

        Returns:
            List of result bindings (each binding is a dict)

        Example:
            ```python
            results = store.query_sparql('''
                PREFIX oba: <https://ralforion.com/ns/oba#>
                SELECT ?table ?column
                WHERE {
                    ?table oba:hasColumn ?column .
                    ?column oba:dataType "INTEGER"
                }
                LIMIT 10
            ''')
            ```

        Raises:
            TimeoutError: If the query runs longer than ``timeout_seconds``.
                pyoxigraph exposes no native query cancellation to Python, so the
                timeout is best-effort: the caller is unblocked, but the orphaned
                query keeps running in the background until it finishes on its own.
        """
        if timeout_seconds is None:
            return self._execute_select(sparql_query)

        result: list[list[dict[str, Any]]] = []
        error: list[BaseException] = []

        def _runner() -> None:
            try:
                result.append(self._execute_select(sparql_query))
            except BaseException as exc:
                error.append(exc)

        worker = threading.Thread(target=_runner, name="sparql-query", daemon=True)
        worker.start()
        worker.join(timeout_seconds)

        if worker.is_alive():
            logger.warning(
                f"SPARQL query exceeded {timeout_seconds}s timeout; abandoning the "
                "wait (the query keeps running in the background until it completes)"
            )
            raise TimeoutError(f"SPARQL query exceeded {timeout_seconds}s timeout")
        if error:
            raise error[0]
        return result[0]

    def _execute_select(self, sparql_query: str) -> list[dict[str, Any]]:
        """Execute a SELECT query and materialize its bindings (no timeout).

        Args:
            sparql_query: SPARQL SELECT query string.

        Returns:
            List of result bindings (each binding is a dict keyed by variable
            name, with unbound variables omitted).
        """
        try:
            results = []

            # SELECT queries yield QuerySolutions; narrow the query() union so the
            # iteration type-checks (other query forms are handled by sibling methods).
            solutions = cast(
                "QuerySolutions",
                self.store.query(
                    sparql_query,
                    use_default_graph_as_union=not _declares_dataset(sparql_query),
                ),
            )
            variables = solutions.variables
            for solution in solutions:
                binding: dict[str, Any] = {}
                for var in variables:
                    term = solution[var]
                    if term is None:
                        # Variable is unbound in this solution; omit it.
                        continue
                    # Key by the bare variable name (no leading "?").
                    if hasattr(term, "value"):
                        binding[var.value] = term.value
                    else:
                        binding[var.value] = str(term)
                results.append(binding)

            logger.info(f"SPARQL query returned {len(results)} results")
            return results

        except Exception as e:
            logger.exception(f"SPARQL query failed: {e}")
            raise

    def query_sparql_ask(self, sparql_query: str) -> bool:
        """
        Execute SPARQL ASK query.

        Patterns outside a ``GRAPH`` clause are matched against the union of all
        named graphs unless the query selects its own dataset with ``FROM`` /
        ``FROM NAMED`` (see :meth:`query_sparql`).

        Args:
            sparql_query: SPARQL ASK query

        Returns:
            Boolean result

        Example:
            ```python
            exists = store.query_sparql_ask('''
                PREFIX oba: <https://ralforion.com/ns/oba#>
                ASK {
                    ?table oba:hasColumn ?column .
                    ?column oba:dataType "INTEGER"
                }
            ''')
            ```
        """
        try:
            # ASK queries yield a QueryBoolean (pyoxigraph >= 0.4) or a plain bool
            # (older versions); both support bool().
            return bool(
                self.store.query(
                    sparql_query,
                    use_default_graph_as_union=not _declares_dataset(sparql_query),
                )
            )
        except Exception as e:
            logger.exception(f"SPARQL ASK query failed: {e}")
            raise

    def query_sparql_construct(self, sparql_query: str) -> str:
        """
        Execute SPARQL CONSTRUCT query.

        Patterns outside a ``GRAPH`` clause are matched against the union of all
        named graphs unless the query selects its own dataset with ``FROM`` /
        ``FROM NAMED`` (see :meth:`query_sparql`).

        Args:
            sparql_query: SPARQL CONSTRUCT query

        Returns:
            Constructed RDF graph in Turtle format

        Example:
            ```python
            ttl = store.query_sparql_construct('''
                PREFIX oba: <https://ralforion.com/ns/oba#>
                CONSTRUCT {
                    ?table a oba:IntegerTable
                }
                WHERE {
                    ?table oba:hasColumn ?column .
                    ?column oba:dataType "INTEGER"
                }
            ''')
            ```
        """
        try:
            # CONSTRUCT yields QueryTriples; narrow the query() union so serialize()
            # resolves to the RDF (not results) overload.
            results = cast(
                "QueryTriples",
                self.store.query(
                    sparql_query,
                    use_default_graph_as_union=not _declares_dataset(sparql_query),
                ),
            )
            # serialize() yields bytes (or None for an empty result), so decode to
            # satisfy the str return contract.
            serialized = results.serialize(format=RdfFormat.TURTLE)
            return serialized.decode("utf-8") if serialized is not None else ""
        except Exception as e:
            logger.exception(f"SPARQL CONSTRUCT query failed: {e}")
            raise

    def add_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        graph_uri: str | None = None,
        object_is_literal: bool = False,
    ) -> None:
        """
        Add a single RDF triple to the store.

        Args:
            subject: Subject URI
            predicate: Predicate URI
            object: Object URI or literal value
            graph_uri: Optional named graph URI
            object_is_literal: If True, object is treated as literal value

        Example:
            ```python
            # Add a schema metadata triple
            store.add_triple(
                subject="http://example.com/schema/customers",
                predicate="http://www.w3.org/2000/01/rdf-schema#label",
                object="Customer Master Data",
                object_is_literal=True
            )
            ```
        """
        try:
            subj = NamedNode(subject)
            pred = NamedNode(predicate)
            obj = Literal(object) if object_is_literal else NamedNode(object)

            if graph_uri:
                self.store.add(Quad(subj, pred, obj, NamedNode(graph_uri)))
            else:
                self.store.add(Quad(subj, pred, obj))

            logger.debug(f"Added triple: <{subject}> <{predicate}> {object}")

        except Exception as e:
            logger.exception(f"Failed to add triple: {e}")
            raise

    def add_knowledge(
        self,
        subject: str,
        predicate: str,
        object: str,
        metadata: dict[str, Any] | None = None,
        graph_uri: str = "http://example.com/knowledge",
    ) -> None:
        """
        Add learned knowledge to the store with metadata.

        Args:
            subject: Subject URI
            predicate: Predicate URI
            object: Object value
            metadata: Optional metadata (added as additional triples)
            graph_uri: Knowledge graph URI

        Example:
            ```python
            # Document a learned query pattern
            store.add_knowledge(
                subject="http://example.com/pattern/sales_by_customer",
                predicate="http://example.com/schema#hasSQL",
                object="SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
                metadata={
                    "learned_from": "user_query",
                    "timestamp": "2026-02-26T16:00:00Z",
                    "confidence": 0.95
                }
            )
            ```
        """
        try:
            # Add main triple
            self.add_triple(
                subject, predicate, object, graph_uri, object_is_literal=True
            )

            # Add metadata triples
            if metadata:
                for key, value in metadata.items():
                    meta_predicate = f"http://example.com/metadata#{key}"
                    self.add_triple(
                        subject,
                        meta_predicate,
                        str(value),
                        graph_uri,
                        object_is_literal=True,
                    )

            logger.info(f"Added knowledge: {subject} -> {predicate}")

        except Exception as e:
            logger.exception(f"Failed to add knowledge: {e}")
            raise

    def get_ontology_stats(self, graph_uri: str | None = None) -> dict[str, Any]:
        """
        Get statistics about stored ontologies.

        Args:
            graph_uri: Optional specific graph to query

        Returns:
            Statistics dictionary
        """
        try:
            if graph_uri:
                # Count triples in specific graph
                safe_uri = _escape_sparql_iri(graph_uri)
                query = f"""
                    SELECT (COUNT(*) AS ?count)
                    WHERE {{
                        GRAPH <{safe_uri}> {{
                            ?s ?p ?o .
                        }}
                    }}
                """
                results = list(cast("QuerySolutions", self.store.query(query)))
                triple_count = int(results[0]["count"].value) if results else 0

                return {"graph_uri": graph_uri, "triple_count": triple_count}
            else:
                # Overall statistics
                total_triples = len(self.store)

                # Count named graphs
                query = """
                    SELECT DISTINCT ?g
                    WHERE {
                        GRAPH ?g { ?s ?p ?o }
                    }
                """
                graphs = list(cast("QuerySolutions", self.store.query(query)))

                return {
                    "total_triples": total_triples,
                    "named_graphs": len(graphs),
                    "graphs": [str(g["g"]) for g in graphs],
                    "loaded_ontologies": dict(self._loaded_ontologies),
                }

        except Exception as e:
            logger.exception(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def list_tables_sparql(self, schema_graph: str) -> list[str]:
        """
        List all tables from an ontology using SPARQL.

        Args:
            schema_graph: Graph URI containing the ontology

        Returns:
            List of table names
        """
        safe_graph = _escape_sparql_iri(schema_graph)
        query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX oba: <https://ralforion.com/ns/oba#>

            SELECT DISTINCT ?tableName
            FROM <{safe_graph}>
            WHERE {{
                ?table a oba:Table .
                ?table oba:tableName ?tableName .
            }}
            ORDER BY ?tableName
        """

        results = self.query_sparql(query)
        return [r["tableName"] for r in results]

    def find_columns_by_type(
        self, data_type: str, schema_graph: str | None = None
    ) -> list[dict[str, str]]:
        """
        Find columns by data type using SPARQL.

        Args:
            data_type: SQL data type (e.g., "INTEGER", "VARCHAR")
            schema_graph: Optional graph to search

        Returns:
            List of {table, column, type} dicts
        """
        safe_graph = _escape_sparql_iri(schema_graph) if schema_graph else ""
        graph_clause = f"FROM <{safe_graph}>" if schema_graph else ""
        safe_type = _escape_sparql_literal(data_type)

        query = f"""
            PREFIX oba: <https://ralforion.com/ns/oba#>

            SELECT ?tableName ?columnName ?dataType
            {graph_clause}
            WHERE {{
                ?column a oba:Column .
                ?column oba:tableName ?tableName .
                ?column oba:columnName ?columnName .
                ?column oba:dataType ?dataType .
                FILTER (LCASE(STR(?dataType)) = LCASE("{safe_type}"))
            }}
            ORDER BY ?tableName ?columnName
        """

        results = self.query_sparql(query)
        return [
            {"table": r["tableName"], "column": r["columnName"], "type": r["dataType"]}
            for r in results
        ]

    def export_graph(self, graph_uri: str, format: str = "turtle") -> str:
        """
        Export a named graph.

        Args:
            graph_uri: Graph to export
            format: Export format ("turtle", "ntriples", "rdfxml")

        Returns:
            Serialized RDF
        """
        safe_uri = _escape_sparql_iri(graph_uri)
        query = f"""
            CONSTRUCT {{ ?s ?p ?o }}
            WHERE {{
                GRAPH <{safe_uri}> {{
                    ?s ?p ?o
                }}
            }}
        """

        return self.query_sparql_construct(query)

    def delete_graph(self, graph_uri: str) -> None:
        """Remove a named graph and all of its triples from the store.

        Used when an ontology version is cleaned up so stale triples don't linger
        in Oxigraph. Removing a graph that doesn't exist is a no-op.

        Args:
            graph_uri: Named graph URI to delete.
        """
        try:
            self.store.remove_graph(NamedNode(graph_uri))
            # Drop any schema -> graph tracking that pointed at this graph.
            self._loaded_ontologies = {
                schema: uri
                for schema, uri in self._loaded_ontologies.items()
                if uri != graph_uri
            }
            logger.info(f"Deleted named graph <{graph_uri}>")
        except Exception as e:
            logger.exception(f"Failed to delete graph {graph_uri}: {e}")
            raise

    def close(self) -> None:
        """Close the store (flush to disk if persistent)."""
        if hasattr(self.store, "close"):
            self.store.close()
        logger.info("Closed Oxigraph store")

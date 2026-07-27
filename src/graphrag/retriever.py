"""
Graph Retriever - Graph-based traversal and relationship discovery

Provides intelligent schema navigation using the graph structure
of foreign key relationships and semantic similarity.
"""

import logging
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Graph-based retrieval for schema navigation."""

    def __init__(self) -> None:
        """Initialize the graph retriever."""
        self.graph = nx.DiGraph()
        self._tables_info: dict[str, dict[str, Any]] = {}

    def build_graph(self, tables_info: list[dict[str, Any]]) -> None:
        """
        Build graph from schema information.

        Args:
            tables_info: List of table metadata with columns and foreign keys
        """
        self.graph.clear()
        self._tables_info = {}

        # Add nodes (tables)
        for table in tables_info:
            table_name = table["name"]
            self._tables_info[table_name] = table

            self.graph.add_node(
                table_name,
                node_type="table",
                column_count=len(table.get("columns", [])),
                has_comment=bool(table.get("comment")),
                comment=table.get("comment", ""),
            )

        # Add edges (foreign key relationships)
        for table in tables_info:
            table_name = table["name"]

            for fk in table.get("foreign_keys", []):
                referenced_table = fk["referenced_table"]

                if referenced_table in self.graph:
                    self.graph.add_edge(
                        table_name,
                        referenced_table,
                        edge_type="foreign_key",
                        column=fk["column"],
                        referenced_column=fk["referenced_column"],
                    )

        logger.info(
            f"Built graph with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )

    def add_to_graph(self, tables_info: list[dict[str, Any]]) -> None:
        """Add tables and relationships to the existing graph (accumulative).

        Unlike build_graph(), this does NOT clear the graph first.
        Duplicate nodes are updated, new edges are added.

        Args:
            tables_info: List of table metadata with columns and foreign keys
        """
        added_nodes = 0
        added_edges = 0

        for table in tables_info:
            table_name = table["name"]
            self._tables_info[table_name] = table

            if table_name not in self.graph:
                added_nodes += 1
            self.graph.add_node(
                table_name,
                node_type="table",
                column_count=len(table.get("columns", [])),
                has_comment=bool(table.get("comment")),
                comment=table.get("comment", ""),
            )

        for table in tables_info:
            table_name = table["name"]
            for fk in table.get("foreign_keys", []):
                referenced_table = fk["referenced_table"]
                if referenced_table in self.graph:
                    if not self.graph.has_edge(table_name, referenced_table):
                        added_edges += 1
                    self.graph.add_edge(
                        table_name,
                        referenced_table,
                        edge_type="foreign_key",
                        column=fk["column"],
                        referenced_column=fk["referenced_column"],
                    )

        logger.info(
            f"Added to graph: +{added_nodes} nodes, +{added_edges} edges "
            f"(total: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges)"
        )

    def find_join_path(
        self, from_table: str, to_table: str, max_hops: int = 12
    ) -> list[dict[str, Any]] | None:
        """
        Find join path between two tables.

        Args:
            from_table: Source table
            to_table: Target table
            max_hops: Maximum number of joins allowed (default: 12)

        Returns:
            List of join specifications or None if no path exists
        """
        if from_table not in self.graph or to_table not in self.graph:
            return None

        try:
            # Use bidirectional search (considers both directions)
            # Also try undirected view to find mixed-direction paths (e.g., A → B ← C)
            path_forward = None
            path_backward = None
            path_undirected = None

            try:
                path_forward = nx.shortest_path(
                    self.graph, source=from_table, target=to_table
                )
            except nx.NetworkXNoPath:
                pass

            try:
                path_backward = nx.shortest_path(
                    self.graph, source=to_table, target=from_table
                )
            except nx.NetworkXNoPath:
                pass

            # Try undirected view for mixed-direction paths
            try:
                undirected_graph = self.graph.to_undirected()
                path_undirected = nx.shortest_path(
                    undirected_graph, source=from_table, target=to_table
                )
            except nx.NetworkXNoPath:
                pass

            # Choose shortest path among all options
            candidates = []
            if path_forward:
                candidates.append(path_forward)
            if path_backward:
                candidates.append(list(reversed(path_backward)))
            if path_undirected:
                candidates.append(path_undirected)

            if not candidates:
                return None

            # Pick shortest path
            path = min(candidates, key=len)

            if len(path) - 1 > max_hops:
                logger.warning(
                    f"Path from {from_table} to {to_table} requires {len(path) - 1} hops (max: {max_hops})"
                )
                return None

            # Build join specifications
            joins = []
            for i in range(len(path) - 1):
                left_table = path[i]
                right_table = path[i + 1]

                # Get edge data (FK relationship)
                if self.graph.has_edge(left_table, right_table):
                    edge_data = self.graph[left_table][right_table]
                    joins.append(
                        {
                            "from_table": left_table,
                            "to_table": right_table,
                            "from_column": edge_data["column"],
                            "to_column": edge_data["referenced_column"],
                            "join_type": "INNER",
                        }
                    )
                elif self.graph.has_edge(right_table, left_table):
                    edge_data = self.graph[right_table][left_table]
                    joins.append(
                        {
                            "from_table": left_table,
                            "to_table": right_table,
                            "from_column": edge_data["referenced_column"],
                            "to_column": edge_data["column"],
                            "join_type": "INNER",
                        }
                    )

            return joins

        except Exception as e:
            logger.error(f"Error finding join path: {e}")
            return None

    def get_related_tables(
        self, table_name: str, max_distance: int = 1, direction: str = "both"
    ) -> dict[int, list[str]]:
        """
        Get tables related to a given table.

        Args:
            table_name: Source table
            max_distance: Maximum graph distance (hops)
            direction: "outgoing" (FK from table), "incoming" (FK to table), "both"

        Returns:
            Dictionary with lists of related tables by distance
        """
        if table_name not in self.graph:
            return {}

        related: defaultdict[int, list[str]] = defaultdict(list)

        if direction in ["outgoing", "both"]:
            # Tables this table references (FK from)
            for target in self.graph.successors(table_name):
                related[1].append(target)

            # Multi-hop outgoing
            if max_distance > 1:
                visited = {table_name}
                queue = deque([(table_name, 0)])

                while queue:
                    current, dist = queue.popleft()

                    if dist >= max_distance:
                        continue

                    for neighbor in self.graph.successors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            related[dist + 1].append(neighbor)
                            queue.append((neighbor, dist + 1))

        if direction in ["incoming", "both"]:
            # Tables that reference this table (FK to)
            for source in self.graph.predecessors(table_name):
                if source not in related[1]:  # Avoid duplicates if "both"
                    related[1].append(source)

            # Multi-hop incoming
            if max_distance > 1:
                visited = {table_name}
                queue = deque([(table_name, 0)])

                while queue:
                    current, dist = queue.popleft()

                    if dist >= max_distance:
                        continue

                    for neighbor in self.graph.predecessors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            if neighbor not in related[dist + 1]:
                                related[dist + 1].append(neighbor)
                            queue.append((neighbor, dist + 1))

        return dict(related)

    def _directed_closure(
        self,
        table_name: str,
        max_hops: int | None,
        neighbor_fn: Callable[[str], Any],
    ) -> dict[str, Any]:
        """Cycle-safe directed transitive closure from a table.

        Edges in the FK DiGraph point finer grain -> coarser grain (a table to
        the table it references). Following ``successors`` walks the many-to-one
        (dimension) direction; following ``predecessors`` walks the one-to-many
        (measure) direction.

        Args:
            table_name: Anchor table
            max_hops: Maximum hops (None = unbounded full closure)
            neighbor_fn: ``self.graph.successors`` or ``self.graph.predecessors``

        Returns:
            Dict with ``exists``, ordered ``tables`` list, and ``by_hop`` mapping.
        """
        if table_name not in self.graph:
            return {"exists": False, "tables": [], "by_hop": {}}

        visited = {table_name}
        order: list[str] = []
        by_hop: dict[int, list[str]] = {}
        queue = deque([(table_name, 0)])

        while queue:
            current, dist = queue.popleft()
            if max_hops is not None and dist >= max_hops:
                continue
            for neighbor in neighbor_fn(current):
                if neighbor not in visited:
                    visited.add(neighbor)  # cycle-safe: each node enqueued once
                    order.append(neighbor)
                    by_hop.setdefault(dist + 1, []).append(neighbor)
                    queue.append((neighbor, dist + 1))

        return {"exists": True, "tables": order, "by_hop": by_hop}

    def reachable_from(
        self, table_name: str, max_hops: int | None = None
    ) -> dict[str, Any]:
        """Dimension-capable tables for a query anchored on ``table_name``.

        Directed many-to-one closure (``successors``): coarser-grain tables that
        can be joined from the anchor without row multiplication (every hop is
        functional), so their columns are safe to GROUP BY / filter on.
        """
        return self._directed_closure(table_name, max_hops, self.graph.successors)

    def measurable_from(
        self, table_name: str, max_hops: int | None = None
    ) -> dict[str, Any]:
        """Measure-capable tables for a query anchored on ``table_name``.

        Directed one-to-many closure (``predecessors``): finer-grain tables that
        fan out the anchor, so their values can only be aggregated into measures
        (SUM/COUNT/...), never used as dimensions at this grain.
        """
        return self._directed_closure(table_name, max_hops, self.graph.predecessors)

    def detect_fan_traps(self, tables: list[str]) -> list[dict[str, Any]]:
        """
        Detect potential fan-trap scenarios in a set of tables.

        Args:
            tables: List of table names

        Returns:
            List of fan-trap warnings
        """
        warnings = []

        for table in tables:
            if table not in self.graph:
                continue

            # Count outgoing FKs
            outgoing_fks = list(self.graph.successors(table))

            if len(outgoing_fks) > 1:
                warnings.append(
                    {
                        "bridge_table": table,
                        "referenced_tables": outgoing_fks,
                        "warning": f"Table '{table}' connects to multiple tables - potential fan-trap",
                        "recommendation": "Use separate CTEs or UNION approach if aggregating across these relationships",
                    }
                )

        return warnings

    def get_table_metadata(self, table_name: str) -> dict[str, Any] | None:
        """
        Get full metadata for a table.

        Args:
            table_name: Table name

        Returns:
            Table metadata dictionary or None
        """
        return self._tables_info.get(table_name)

    def get_graph_summary(self) -> dict[str, Any]:
        """
        Get summary statistics of the schema graph.

        Returns:
            Summary dictionary
        """
        # Find central tables (high degree centrality)
        centrality = nx.degree_centrality(self.graph.to_undirected())
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]

        # Find hub tables (many outgoing FKs)
        out_degrees = dict(self.graph.out_degree())
        top_hubs = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]

        # Find reference tables (many incoming FKs)
        in_degrees = dict(self.graph.in_degree())
        top_references = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        return {
            "total_tables": self.graph.number_of_nodes(),
            "total_relationships": self.graph.number_of_edges(),
            "top_central_tables": [
                {"table": t, "centrality": c} for t, c in top_central
            ],
            "top_hub_tables": [
                {"table": t, "outgoing_fks": d} for t, d in top_hubs if d > 0
            ],
            "top_reference_tables": [
                {"table": t, "incoming_fks": d} for t, d in top_references if d > 0
            ],
            "avg_connections_per_table": sum(dict(self.graph.degree()).values())
            / max(self.graph.number_of_nodes(), 1),
        }

    def load_graph(self, tables_info: list[dict[str, Any]]) -> bool:
        """Rebuild graph from previously saved tables_info.

        This is equivalent to build_graph() but named distinctly to indicate
        it's for restore-from-disk scenarios.

        Args:
            tables_info: List of table metadata dicts (from saved state)

        Returns:
            True if graph was rebuilt successfully
        """
        if not tables_info:
            return False
        self.build_graph(tables_info)
        return True

    def export_graph_for_visualization(self) -> dict[str, Any]:
        """
        Export graph in format suitable for visualization.

        Returns:
            Dictionary with nodes and edges
        """
        nodes = [
            {
                "id": node[0],
                "label": node[0],
                "type": "table",
                "column_count": node[1].get("column_count", 0),
            }
            for node in self.graph.nodes(data=True)
        ]

        edges = [
            {
                "from": edge[0],
                "to": edge[1],
                "label": f"{edge[2].get('column')} → {edge[2].get('referenced_column')}",
                "type": "foreign_key",
            }
            for edge in self.graph.edges(data=True)
        ]

        return {"nodes": nodes, "edges": edges}

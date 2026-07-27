"""
Community Detector - Identifies logical groupings in database schemas

Uses graph algorithms to detect communities (logical clusters) of related tables,
helping organize large schemas into understandable domains.
"""

import logging
from collections import defaultdict
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detects logical communities/clusters in database schemas."""

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize community detector.

        Args:
            graph: NetworkX graph of schema relationships
        """
        self.graph = graph
        self.communities: dict[int, set[str]] = {}
        self.table_to_community: dict[str, int] = {}

    def detect_communities(self, method: str = "louvain") -> dict[int, set[str]]:
        """
        Detect communities in the schema graph.

        Args:
            method: Detection method ("louvain", "connected_components", "label_propagation")

        Returns:
            Dictionary mapping community_id to set of table names
        """
        if method == "louvain":
            return self._detect_louvain()
        elif method == "connected_components":
            return self._detect_connected_components()
        elif method == "label_propagation":
            return self._detect_label_propagation()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _detect_louvain(self) -> dict[int, set[str]]:
        """
        Detect communities using Louvain method.

        Returns:
            Dictionary of communities
        """
        try:
            import community as community_louvain

            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()

            # Apply Louvain algorithm
            partition = community_louvain.best_partition(undirected)

            # Organize into communities
            communities = defaultdict(set)
            for table, community_id in partition.items():
                communities[community_id].add(table)
                self.table_to_community[table] = community_id

            self.communities = dict(communities)
            logger.info(
                f"Detected {len(self.communities)} communities using Louvain method"
            )

            return self.communities

        except ImportError:
            logger.warning(
                "python-louvain not installed, falling back to connected components"
            )
            return self._detect_connected_components()

    def _detect_connected_components(self) -> dict[int, set[str]]:
        """
        Detect communities using connected components.

        Returns:
            Dictionary of communities
        """
        # Convert to undirected
        undirected = self.graph.to_undirected()

        # Find connected components
        components = list(nx.connected_components(undirected))

        communities = {}
        for idx, component in enumerate(components):
            communities[idx] = component
            for table in component:
                self.table_to_community[table] = idx

        self.communities = communities
        logger.info(
            f"Detected {len(self.communities)} communities using connected components"
        )

        return self.communities

    def _detect_label_propagation(self) -> dict[int, set[str]]:
        """
        Detect communities using label propagation.

        Returns:
            Dictionary of communities
        """
        # Convert to undirected
        undirected = self.graph.to_undirected()

        # Apply label propagation
        communities_gen = nx.community.label_propagation_communities(undirected)
        communities_list = list(communities_gen)

        communities = {}
        for idx, community in enumerate(communities_list):
            communities[idx] = community
            for table in community:
                self.table_to_community[table] = idx

        self.communities = communities
        logger.info(
            f"Detected {len(self.communities)} communities using label propagation"
        )

        return self.communities

    def load_communities(self, data: dict[str, Any]) -> bool:
        """Restore communities from previously saved state.

        Args:
            data: Dict with "summaries" list (from save_state output)

        Returns:
            True if communities were restored successfully
        """
        summaries = data.get("summaries", [])
        if not summaries:
            return False

        self.communities = {}
        self.table_to_community = {}

        for summary in summaries:
            community_id = summary["community_id"]
            tables = set(summary.get("tables", []))
            self.communities[community_id] = tables
            for table in tables:
                self.table_to_community[table] = community_id

        logger.info(f"Restored {len(self.communities)} communities from saved state")
        return True

    def get_community(self, table_name: str) -> int | None:
        """
        Get community ID for a table.

        Args:
            table_name: Table name

        Returns:
            Community ID or None if not found
        """
        return self.table_to_community.get(table_name)

    def get_community_tables(self, community_id: int) -> set[str]:
        """
        Get all tables in a community.

        Args:
            community_id: Community identifier

        Returns:
            Set of table names
        """
        return self.communities.get(community_id, set())

    def get_community_summary(self, community_id: int) -> dict[str, Any]:
        """
        Get summary information about a community.

        Args:
            community_id: Community identifier

        Returns:
            Summary dictionary
        """
        tables = self.communities.get(community_id, set())

        if not tables:
            return {}

        # Calculate subgraph for this community
        subgraph = self.graph.subgraph(tables)

        # Find most connected table (potential "central" table)
        degrees = dict(subgraph.degree())
        central_table = max(degrees.items(), key=lambda x: x[1])[0] if degrees else None

        # Count relationships within community
        internal_edges = subgraph.number_of_edges()

        return {
            "community_id": community_id,
            "table_count": len(tables),
            "tables": sorted(tables),
            "internal_relationships": internal_edges,
            "central_table": central_table,
            "avg_connections": sum(degrees.values()) / len(degrees) if degrees else 0,
        }

    def get_all_summaries(self) -> list[dict[str, Any]]:
        """
        Get summaries for all communities.

        Returns:
            List of community summaries
        """
        summaries = []
        for community_id in sorted(self.communities.keys()):
            summary = self.get_community_summary(community_id)
            summaries.append(summary)

        return summaries

    def suggest_domain_names(self) -> dict[int, str]:
        """
        Suggest domain names for communities based on table names.

        Returns:
            Dictionary mapping community_id to suggested domain name
        """
        domain_names = {}

        for community_id, tables in self.communities.items():
            # Extract common prefixes/suffixes
            table_names = list(tables)

            if len(table_names) == 1:
                domain_names[community_id] = table_names[0].replace("_", " ").title()
                continue

            # Find common words in table names
            word_counts: defaultdict[str, int] = defaultdict(int)
            for table in table_names:
                words = table.lower().split("_")
                for word in words:
                    if len(word) > 2:  # Ignore very short words
                        word_counts[word] += 1

            # Find most common word
            if word_counts:
                common_word = max(word_counts.items(), key=lambda x: x[1])[0]
                domain_names[community_id] = common_word.title() + " Domain"
            else:
                domain_names[community_id] = f"Domain {community_id + 1}"

        return domain_names

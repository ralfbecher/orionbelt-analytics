"""
Tests for workspace persistence and restore functionality.

Covers:
1. VersionMetadataManager workspace extension (read/write)
2. Async write serialization (locking)
3. Workspace detection (detect_workspace)
4. Workspace summary formatting
5. GraphRAG load_state
6. TableInfo.from_dict deserialization
7. cleanup_workspace tool registration
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

import src.main as main_module
from src.database_manager import ColumnInfo, TableInfo
from src.graphrag.community_detector import CommunityDetector
from src.graphrag.retriever import GraphRetriever
from src.lifecycle.metadata import (
    VersionMetadataManager,
    update_workspace_rdf,
    update_workspace_section,
)
from src.main import mcp
from src.workspace import detect_workspace, format_workspace_summary


def _get_tool_fn(name):
    """Get the underlying function for a tool."""
    obj = getattr(main_module, name)
    return getattr(obj, "fn", obj)


class TestMetadataWorkspaceExtension:
    """Test VersionMetadataManager workspace methods."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir)
        self.connection_id = "test_conn_1234567"

    def test_get_workspace_empty(self):
        """Workspace is None when no workspace section exists."""
        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        assert mgr.get_workspace() is None

    def test_update_workspace_creates_section(self):
        """update_workspace creates workspace section if missing."""
        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        mgr.update_workspace(
            "public",
            "schema",
            {
                "schema_file": "schema_public.json",
                "table_count": 10,
            },
        )

        workspace = mgr.get_workspace()
        assert workspace is not None
        assert "schemas" in workspace
        assert "public" in workspace["schemas"]
        assert workspace["schemas"]["public"]["schema"]["table_count"] == 10
        assert "updated_at" in workspace

    def test_update_workspace_multiple_schemas(self):
        """Multiple schemas can be stored in workspace."""
        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        mgr.update_workspace("public", "schema", {"table_count": 10})
        mgr.update_workspace("analytics", "schema", {"table_count": 5})

        workspace = mgr.get_workspace()
        assert "public" in workspace["schemas"]
        assert "analytics" in workspace["schemas"]

    def test_update_workspace_multiple_sections(self):
        """Multiple sections per schema work correctly."""
        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        mgr.update_workspace("public", "schema", {"table_count": 10})
        mgr.update_workspace("public", "ontology", {"ontology_file": "ont.ttl"})
        mgr.update_workspace("public", "graphrag", {"initialized": True})

        schema = mgr.get_workspace_schema("public")
        assert schema["schema"]["table_count"] == 10
        assert schema["ontology"]["ontology_file"] == "ont.ttl"
        assert schema["graphrag"]["initialized"] is True

    def test_get_workspace_schema_nonexistent(self):
        """get_workspace_schema returns None for missing schema."""
        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        assert mgr.get_workspace_schema("missing") is None

    def test_update_workspace_connection(self):
        """update_workspace_connection sets db_type and db_name."""
        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        mgr.update_workspace_connection(db_type="postgresql", db_name="mydb")

        workspace = mgr.get_workspace()
        assert workspace["db_type"] == "postgresql"
        assert workspace["db_name"] == "mydb"

    def test_update_workspace_rdf_store(self):
        """update_workspace_rdf_store sets rdf_store section."""
        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        mgr.update_workspace_rdf_store(
            {
                "initialized": True,
                "graph_uris": ["http://example.com/g1"],
            }
        )

        workspace = mgr.get_workspace()
        assert workspace["rdf_store"]["initialized"] is True
        assert len(workspace["rdf_store"]["graph_uris"]) == 1

    def test_workspace_persists_to_disk(self):
        """Workspace data survives re-reading from disk."""
        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        mgr.update_workspace("public", "schema", {"table_count": 42})

        # Re-read from disk
        mgr2 = VersionMetadataManager(self.connection_id, self.output_dir)
        schema = mgr2.get_workspace_schema("public")
        assert schema["schema"]["table_count"] == 42

    def test_workspace_update_overwrites_section(self):
        """Updating a section replaces it entirely."""
        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        mgr.update_workspace("public", "schema", {"table_count": 10, "old_key": "x"})
        mgr.update_workspace("public", "schema", {"table_count": 20})

        schema = mgr.get_workspace_schema("public")
        assert schema["schema"]["table_count"] == 20
        assert "old_key" not in schema["schema"]


class TestAsyncWriteSerialization:
    """Test async locking for concurrent workspace writes."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir)
        self.connection_id = "test_lock_conn123"

    @pytest.mark.asyncio
    async def test_concurrent_writes_dont_corrupt(self):
        """Multiple concurrent writes produce valid JSON."""
        tasks = [
            update_workspace_section(
                connection_id=self.connection_id,
                output_dir=self.output_dir,
                schema_name=f"schema_{i}",
                section="schema",
                data={"table_count": i * 10},
            )
            for i in range(5)
        ]

        await asyncio.gather(*tasks)

        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        workspace = mgr.get_workspace()

        assert workspace is not None
        assert len(workspace["schemas"]) == 5
        for i in range(5):
            assert (
                workspace["schemas"][f"schema_{i}"]["schema"]["table_count"] == i * 10
            )

    @pytest.mark.asyncio
    async def test_concurrent_rdf_write(self):
        """Concurrent RDF store updates don't corrupt."""
        await update_workspace_rdf(
            connection_id=self.connection_id,
            output_dir=self.output_dir,
            data={"initialized": True, "graph_uris": ["uri1"]},
        )

        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        workspace = mgr.get_workspace()
        assert workspace["rdf_store"]["initialized"] is True


class TestWorkspaceDetection:
    """Test detect_workspace and format_workspace_summary."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir)
        self.connection_id = "detect_test_12345"

    @patch("src.workspace.OUTPUT_DIR")
    def test_detect_workspace_empty(self, mock_output):
        """Returns None when no workspace exists."""
        mock_output.__truediv__ = lambda s, x: self.output_dir / x
        mock_output.mkdir = MagicMock()
        result = detect_workspace("nonexistent_conn")
        assert result is None

    def test_detect_workspace_with_valid_files(self):
        """Returns workspace data when files exist on disk."""
        # Set up workspace metadata + actual files
        mgr = VersionMetadataManager(self.connection_id, self.output_dir)
        mgr.update_workspace_connection("postgresql", "testdb")
        mgr.update_workspace(
            "public",
            "schema",
            {
                "schema_file": "schema_test.json",
                "table_count": 5,
                "analyzed_at": "2026-04-14T10:00:00",
            },
        )

        # Create the actual file in the connection dir
        conn_dir = self.output_dir / self.connection_id
        conn_dir.mkdir(parents=True, exist_ok=True)
        (conn_dir / "schema_test.json").write_text('{"tables": []}')

        with (
            patch("src.workspace.OUTPUT_DIR", self.output_dir),
            patch("src.workspace.get_connection_dir", return_value=conn_dir),
        ):
            result = detect_workspace(self.connection_id)

        assert result is not None
        assert result["db_type"] == "postgresql"
        assert "public" in result["schemas"]
        assert result["schemas"]["public"]["schema"]["available"] is True

    def test_format_workspace_summary(self):
        """Summary formatting produces expected output."""
        workspace = {
            "db_type": "postgresql",
            "db_name": "mydb",
            "schemas": {
                "public": {
                    "schema": {"available": True, "table_count": 42},
                    "ontology": {
                        "available": True,
                        "enriched": True,
                        "persisted_to_rdf": True,
                    },
                    "graphrag": {"available": True, "embedding_count": 312},
                }
            },
            "rdf_store": {"initialized": True, "graph_uris": ["uri1"]},
        }

        summary = format_workspace_summary(workspace)
        assert "postgresql" in summary
        assert "mydb" in summary
        assert "42 tables" in summary
        assert "enriched" in summary
        assert "RDF store" in summary
        assert "312 embeddings" in summary
        assert "Auto-restore was not available" in summary


class TestGraphRetrieverLoadGraph:
    """Test GraphRetriever.load_graph."""

    def test_load_graph_empty(self):
        """load_graph returns False on empty data."""
        retriever = GraphRetriever()
        assert retriever.load_graph([]) is False
        assert retriever.graph.number_of_nodes() == 0

    def test_load_graph_rebuilds(self):
        """load_graph rebuilds the graph from tables_info."""
        tables = [
            {
                "name": "orders",
                "columns": [],
                "foreign_keys": [
                    {
                        "column": "customer_id",
                        "referenced_table": "customers",
                        "referenced_column": "id",
                    }
                ],
            },
            {"name": "customers", "columns": [], "foreign_keys": []},
        ]
        retriever = GraphRetriever()
        assert retriever.load_graph(tables) is True
        assert retriever.graph.number_of_nodes() == 2
        assert retriever.graph.number_of_edges() == 1


class TestCommunityDetectorLoadCommunities:
    """Test CommunityDetector.load_communities."""

    def _make_detector(self):
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(["t1", "t2", "t3"])
        return CommunityDetector(g)

    def test_load_communities_empty(self):
        """load_communities returns False on empty data."""
        detector = self._make_detector()
        assert detector.load_communities({"summaries": []}) is False

    def test_load_communities_restores(self):
        """load_communities restores community mappings."""
        detector = self._make_detector()
        data = {
            "summaries": [
                {"community_id": 0, "tables": ["t1", "t2"]},
                {"community_id": 1, "tables": ["t3"]},
            ]
        }
        assert detector.load_communities(data) is True
        assert detector.get_community("t1") == 0
        assert detector.get_community("t3") == 1
        assert len(detector.communities) == 2


class TestTableInfoFromDict:
    """Test TableInfo.from_dict deserialization."""

    def test_from_dict_basic(self):
        """Basic dict deserialization produces valid TableInfo."""
        data = {
            "name": "users",
            "schema": "public",
            "columns": [
                {
                    "name": "id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "is_primary_key": True,
                    "is_foreign_key": False,
                },
                {
                    "name": "email",
                    "data_type": "varchar",
                    "is_nullable": True,
                    "is_primary_key": False,
                    "is_foreign_key": False,
                },
            ],
            "primary_keys": ["id"],
            "foreign_keys": [],
            "comment": "User accounts",
        }

        table = TableInfo.from_dict(data)
        assert table.name == "users"
        assert table.schema == "public"
        assert len(table.columns) == 2
        assert isinstance(table.columns[0], ColumnInfo)
        assert table.columns[0].name == "id"
        assert table.primary_keys == ["id"]
        assert table.comment == "User accounts"

    def test_from_dict_with_defaults(self):
        """Missing optional fields get None defaults."""
        data = {
            "name": "events",
            "columns": [],
            "primary_keys": [],
            "foreign_keys": [],
        }

        table = TableInfo.from_dict(data)
        assert table.schema == ""
        assert table.comment is None
        assert table.row_count is None
        assert table.sample_data is None

    def test_from_dict_with_column_info_instances(self):
        """from_dict handles already-deserialized ColumnInfo objects."""
        col = ColumnInfo(
            name="id",
            data_type="int",
            is_nullable=False,
            is_primary_key=True,
            is_foreign_key=False,
        )
        data = {
            "name": "test",
            "schema": "s",
            "columns": [col],
            "primary_keys": ["id"],
            "foreign_keys": [],
        }

        table = TableInfo.from_dict(data)
        assert table.columns[0] is col


class TestToolRegistration:
    """Test cleanup_workspace is properly registered."""

    @pytest.mark.asyncio
    async def test_cleanup_workspace_registered(self):
        """Verify cleanup_workspace tool exists in MCP."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "cleanup_workspace" in tool_names

    @pytest.mark.asyncio
    async def test_restore_workspace_removed(self):
        """Verify restore_workspace tool no longer exists (replaced by auto-restore)."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "restore_workspace" not in tool_names


class TestGraphRAGManagerLoadState:
    """Test GraphRAGManager.load_state."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir)

    def test_load_state_missing_dir(self):
        """load_state returns False when connection dir doesn't exist."""
        from src.graphrag.manager import GraphRAGManager

        mgr = GraphRAGManager(
            connection_id="nonexistent",
            schema_name="public",
        )
        assert mgr.load_state(self.output_dir) is False

    def test_save_and_load_state_roundtrip(self):
        """save_state followed by load_state restores graph."""
        from src.graphrag.manager import GraphRAGManager

        # Initialize with test data
        mgr = GraphRAGManager(
            connection_id="roundtrip_test",
            schema_name="public",
        )

        tables = [
            {
                "name": "orders",
                "columns": [{"name": "id", "data_type": "int"}],
                "foreign_keys": [
                    {
                        "column": "customer_id",
                        "referenced_table": "customers",
                        "referenced_column": "id",
                    }
                ],
                "comment": "Order records",
            },
            {
                "name": "customers",
                "columns": [
                    {"name": "id", "data_type": "int"},
                    {"name": "name", "data_type": "varchar"},
                ],
                "foreign_keys": [],
                "comment": "Customer accounts",
            },
        ]

        mgr.initialize_from_schema(tables, schema_name="public")
        mgr.save_state(self.output_dir)

        # Create new manager and load state
        mgr2 = GraphRAGManager(
            connection_id="roundtrip_test",
            schema_name="public",
        )
        result = mgr2.load_state(self.output_dir)
        assert result is True
        assert mgr2._initialized is True
        assert mgr2.graph_retriever.graph.number_of_nodes() == 2
        assert mgr2.graph_retriever.graph.number_of_edges() == 1


class TestMultiSchemaSessionState:
    """Test that SessionData isolates per-schema state (ontology, GraphRAG)."""

    def test_set_current_schema_creates_state(self):
        """set_current_schema creates a SchemaState and sets it as active."""
        from src.session import SessionData

        session = SessionData()
        assert session.current_schema is None

        ss = session.set_current_schema("public")
        assert session.current_schema == "public"
        assert ss.schema_name == "public"
        assert ss.ontology.ontology_file is None

    def test_schema_isolation_ontology(self):
        """Ontology state is isolated per schema."""
        from src.session import SessionData

        session = SessionData()

        # Set up "public" schema ontology
        session.set_current_schema("public")
        session.ontology_file = "ontology_public.ttl"
        session.ontology_enriched = True

        # Switch to "analytics"
        session.set_current_schema("analytics")
        session.ontology_file = "ontology_analytics.ttl"
        session.ontology_enriched = False

        # Verify isolation
        session.set_current_schema("public")
        assert session.ontology_file == "ontology_public.ttl"
        assert session.ontology_enriched is True

        session.set_current_schema("analytics")
        assert session.ontology_file == "ontology_analytics.ttl"
        assert session.ontology_enriched is False

    def test_graphrag_is_connection_scoped(self):
        """GraphRAG state is connection-scoped (shared across schemas)."""
        from unittest.mock import Mock

        from src.session import SessionData

        session = SessionData()
        manager = Mock(name="graphrag_manager")
        session.graphrag_manager = manager
        session.graphrag_initialized = True

        # GraphRAG is the same regardless of current schema
        session.set_current_schema("public")
        assert session.graphrag_manager is manager
        assert session.graphrag_initialized is True

        session.set_current_schema("analytics")
        assert session.graphrag_manager is manager
        assert session.graphrag_initialized is True

    def test_schema_isolation_schema_file(self):
        """Schema file reference is isolated per schema."""
        from src.session import SessionData

        session = SessionData()

        session.set_current_schema("public")
        session.schema_file = "schema_public.json"

        session.set_current_schema("analytics")
        session.schema_file = "schema_analytics.json"

        session.set_current_schema("public")
        assert session.schema_file == "schema_public.json"

        session.set_current_schema("analytics")
        assert session.schema_file == "schema_analytics.json"

    def test_rdf_store_is_connection_scoped(self):
        """RDF store remains connection-scoped (shared across schemas)."""
        from unittest.mock import Mock

        from src.session import SessionData

        session = SessionData()
        store = Mock(name="oxigraph_store")
        session.oxigraph_store = store

        session.set_current_schema("public")
        assert session.oxigraph_store is store

        session.set_current_schema("analytics")
        assert session.oxigraph_store is store

    def test_schema_cache_is_multi_schema(self):
        """SchemaCache can hold multiple schemas simultaneously."""
        from src.session import SessionData

        session = SessionData()
        public_tables = [Mock(name="t1")]
        analytics_tables = [Mock(name="t2"), Mock(name="t3")]

        session.cache_schema_analysis("public", public_tables)
        session.cache_schema_analysis("analytics", analytics_tables)

        assert len(session.get_cached_schema("public")) == 1
        assert len(session.get_cached_schema("analytics")) == 2

    def test_clear_all_schema_states(self):
        """clear_all_schema_states removes all per-schema state."""
        from src.session import SessionData

        session = SessionData()

        session.set_current_schema("public")
        session.ontology_file = "ontology_public.ttl"

        session.set_current_schema("analytics")
        session.ontology_file = "ontology_analytics.ttl"

        session.clear_all_schema_states()
        assert session.current_schema is None
        assert session.schema_names == []
        assert session.ontology_file is None
        assert session.graphrag_manager is None

    def test_get_schema_state_specific(self):
        """get_schema_state can access non-current schema state."""
        from src.session import SessionData

        session = SessionData()

        session.set_current_schema("public")
        session.ontology_file = "pub.ttl"

        session.set_current_schema("analytics")

        # Access public's state while analytics is current
        pub_state = session.get_schema_state("public")
        assert pub_state is not None
        assert pub_state.ontology.ontology_file == "pub.ttl"

    def test_schema_names_lists_all(self):
        """schema_names lists all schemas with active state."""
        from src.session import SessionData

        session = SessionData()
        session.set_current_schema("public")
        session.set_current_schema("analytics")
        session.set_current_schema("staging")

        assert sorted(session.schema_names) == ["analytics", "public", "staging"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

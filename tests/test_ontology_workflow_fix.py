"""
Test that ontology generation works correctly after discover_schema(lightweight=True).

This test verifies the bug fix for Phase 2 where lightweight mode wasn't caching
TableInfo objects, causing generate_ontology() to fail or re-query the database.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import src.main as main_module
from src.database_manager import ColumnInfo, DatabaseManager, TableInfo


def create_mock_context():
    """Create a mock MCP context with async methods."""
    ctx = Mock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()
    return ctx


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager with sample data."""
    manager = MagicMock(spec=DatabaseManager)

    # Mock tables
    manager.get_tables.return_value = ["customers", "orders", "order_items"]

    # Mock table analysis
    def mock_analyze_table(table_name, schema_name=None):
        if table_name == "customers":
            return TableInfo(
                name="customers",
                schema="public",
                columns=[
                    ColumnInfo(
                        name="customer_id",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=True,
                        is_foreign_key=False,
                    ),
                    ColumnInfo(
                        name="name",
                        data_type="VARCHAR",
                        is_nullable=False,
                        is_primary_key=False,
                        is_foreign_key=False,
                    ),
                ],
                primary_keys=["customer_id"],
                foreign_keys=[],
                row_count=100,
            )
        elif table_name == "orders":
            return TableInfo(
                name="orders",
                schema="public",
                columns=[
                    ColumnInfo(
                        name="order_id",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=True,
                        is_foreign_key=False,
                    ),
                    ColumnInfo(
                        name="customer_id",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=False,
                        is_foreign_key=True,
                        foreign_key_table="customers",
                        foreign_key_column="customer_id",
                    ),
                ],
                primary_keys=["order_id"],
                foreign_keys=[
                    {
                        "column": "customer_id",
                        "referenced_table": "customers",
                        "referenced_column": "customer_id",
                    }
                ],
                row_count=500,
            )
        elif table_name == "order_items":
            return TableInfo(
                name="order_items",
                schema="public",
                columns=[
                    ColumnInfo(
                        name="item_id",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=True,
                        is_foreign_key=False,
                    ),
                    ColumnInfo(
                        name="order_id",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=False,
                        is_foreign_key=True,
                        foreign_key_table="orders",
                        foreign_key_column="order_id",
                    ),
                ],
                primary_keys=["item_id"],
                foreign_keys=[
                    {
                        "column": "order_id",
                        "referenced_table": "orders",
                        "referenced_column": "order_id",
                    }
                ],
                row_count=1500,
            )
        return None

    manager.analyze_table.side_effect = mock_analyze_table
    manager.has_engine.return_value = True
    manager.prefetch_schema_constraints.return_value = None

    return manager


@pytest.fixture
def mock_context():
    """Create a mock MCP context with async methods."""
    return create_mock_context()


@pytest.mark.asyncio
async def test_lightweight_caches_for_ontology(mock_context, mock_db_manager, tmp_path):
    """Test that lightweight mode caches full TableInfo for generate_ontology()."""

    with (
        patch("src.main.get_session_db_manager", return_value=mock_db_manager),
        patch("src.main.get_session_data") as mock_session_data,
        patch(
            "src.handlers.ontology_generation.ensure_output_dir", return_value=tmp_path
        ),
        patch("src.main.get_session_safe_filename", return_value="test"),
    ):
        # Mock session data
        session = Mock()
        session.get_cached_schema.return_value = None  # No cache initially
        session.schema_file = None
        session.ontology_file = None
        cached_tables = []

        def cache_schema(schema_name, tables):
            cached_tables.extend(tables)

        session.cache_schema_analysis = Mock(side_effect=cache_schema)
        mock_session_data.return_value = session

        # Step 1: Call discover_schema in lightweight mode
        # Use .fn accessor to handle both FunctionTool (under coverage) and plain function
        analyze_fn = getattr(
            main_module.discover_schema, "fn", main_module.discover_schema
        )
        result = await analyze_fn(mock_context, schema_name="public", lightweight=True)

        # Verify lightweight result structure
        assert result["mode"] == "lightweight"
        assert result["table_count"] == 3
        assert "table_names" in result
        assert "relationships" in result
        assert result["next_step"] == "generate_ontology"

        # CRITICAL: Verify that cache_schema_analysis was called with full TableInfo objects
        session.cache_schema_analysis.assert_called_once()
        call_args = session.cache_schema_analysis.call_args
        cached_schema_name = call_args[0][0]
        cached_table_objects = call_args[0][1]

        assert cached_schema_name == "public"
        assert len(cached_table_objects) == 3
        assert all(isinstance(t, TableInfo) for t in cached_table_objects)

        # Verify the cached objects have full metadata
        customers = next(t for t in cached_table_objects if t.name == "customers")
        assert len(customers.columns) == 2
        assert customers.columns[0].name == "customer_id"


@pytest.mark.asyncio
async def test_ontology_uses_lightweight_cache(mock_context, mock_db_manager, tmp_path):
    """Test that generate_ontology() works with data cached by lightweight mode."""

    with (
        patch("src.main.get_session_db_manager", return_value=mock_db_manager),
        patch("src.main.get_session_data") as mock_session_data,
        patch(
            "src.handlers.ontology_generation.ensure_output_dir", return_value=tmp_path
        ),
        patch("src.main.get_session_safe_filename", return_value="test"),
        patch("src.main._server_state") as mock_server_state,
    ):
        # Prepare cached tables (simulating what lightweight mode would cache)
        cached_tables = [
            TableInfo(
                name="customers",
                schema="public",
                columns=[
                    ColumnInfo(
                        name="customer_id",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=True,
                        is_foreign_key=False,
                    )
                ],
                primary_keys=["customer_id"],
                foreign_keys=[],
                row_count=100,
            ),
            TableInfo(
                name="orders",
                schema="public",
                columns=[
                    ColumnInfo(
                        name="order_id",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=True,
                        is_foreign_key=False,
                    ),
                    ColumnInfo(
                        name="customer_id",
                        data_type="INTEGER",
                        is_nullable=False,
                        is_primary_key=False,
                        is_foreign_key=True,
                        foreign_key_table="customers",
                        foreign_key_column="customer_id",
                    ),
                ],
                primary_keys=["order_id"],
                foreign_keys=[
                    {
                        "column": "customer_id",
                        "referenced_table": "customers",
                        "referenced_column": "customer_id",
                    }
                ],
                row_count=500,
            ),
        ]

        # Mock session with cached data
        session = Mock()
        session.get_cached_schema.return_value = cached_tables
        session.get_last_analyzed_schema.return_value = "public"
        session.ontology_file = None
        session.obqc_validator = None
        mock_session_data.return_value = session

        # Mock ontology generator
        mock_generator = Mock()
        mock_generator.generate_from_schema.return_value = (
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."
        )
        mock_generator.graph = Mock()
        mock_generator.graph.parse = Mock()
        mock_generator.base_uri = "http://example.org/"
        mock_generator.oba_ns = "https://ralforion.com/ns/oba#"
        mock_generator.extract_names_for_review.return_value = {
            "tables": [],
            "columns": [],
            "cryptic_count": 0,
        }
        mock_server_state.get_ontology_generator.return_value = mock_generator

        # Step 2: Call generate_ontology WITHOUT schema_info parameter
        # Use .fn accessor to handle both FunctionTool (under coverage) and plain function
        generate_fn = getattr(
            main_module.generate_ontology, "fn", main_module.generate_ontology
        )
        await generate_fn(mock_context)

        # Verify it used the cached data
        mock_generator.generate_from_schema.assert_called_once()
        call_args = mock_generator.generate_from_schema.call_args[0][0]
        assert len(call_args) == 2  # Should have used the 2 cached tables

        # Verify database was NOT queried again
        # (analyze_table should not be called since we used cache)
        mock_db_manager.get_tables.assert_not_called()


@pytest.mark.asyncio
async def test_full_workflow_lightweight_to_ontology(
    mock_context, mock_db_manager, tmp_path
):
    """Integration test: lightweight analyze -> generate ontology."""

    with (
        patch("src.main.get_session_db_manager", return_value=mock_db_manager),
        patch("src.main.get_session_data") as mock_session_data,
        patch(
            "src.handlers.ontology_generation.ensure_output_dir", return_value=tmp_path
        ),
        patch("src.main.get_session_safe_filename", return_value="test"),
        patch("src.main._server_state") as mock_server_state,
    ):
        # Real session-like behavior
        cached_data = {}

        class FakeSession:
            def __init__(self):
                self.schema_file = None
                self.ontology_file = None
                self.obqc_validator = None
                self.connection_id = None
                self.graphrag_initialized = False
                self.graphrag_manager = None
                self.graphrag = Mock()
                self._cache = cached_data
                self._views = {}
                self._current_schema = None
                self.ontology_enriched = False
                self.loaded_ontology = None
                self.loaded_ontology_path = None
                self.r2rml_file = None

            def get_cached_schema(self, schema_name):
                return self._cache.get(schema_name)

            def cache_schema_analysis(self, schema_name, tables):
                self._cache[schema_name] = tables

            def cache_views(self, schema_name, views):
                # Views are cached apart from tables: they feed GraphRAG only
                # and must never reach the ontology this test generates.
                self._views[schema_name] = views

            def get_cached_views(self, schema_name):
                return self._views.get(schema_name, [])

            def get_last_analyzed_schema(self):
                return "public" if "public" in self._cache else None

            def set_current_schema(self, schema_name):
                self._current_schema = schema_name
                return self

            @property
            def current_schema(self):
                return self._current_schema

        session = FakeSession()
        mock_session_data.return_value = session

        # Mock ontology generator
        mock_generator = Mock()
        mock_generator.generate_from_schema.return_value = (
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."
        )
        mock_generator.graph = Mock()
        mock_generator.graph.parse = Mock()
        mock_generator.base_uri = "http://example.org/"
        mock_generator.oba_ns = "https://ralforion.com/ns/oba#"
        mock_generator.extract_names_for_review.return_value = {
            "tables": [],
            "columns": [],
            "cryptic_count": 0,
        }
        mock_server_state.get_ontology_generator.return_value = mock_generator

        # Use .fn accessor to handle both FunctionTool (under coverage) and plain function
        analyze_fn = getattr(
            main_module.discover_schema, "fn", main_module.discover_schema
        )
        generate_fn = getattr(
            main_module.generate_ontology, "fn", main_module.generate_ontology
        )

        # Step 1: discover_schema(lightweight=True)
        schema_result = await analyze_fn(
            mock_context, schema_name="public", lightweight=True
        )

        assert schema_result["mode"] == "lightweight"
        assert schema_result["table_count"] == 3
        assert "public" in cached_data
        assert len(cached_data["public"]) == 3

        # Step 2: generate_ontology() - should use cache
        await generate_fn(mock_context, schema_name="public")

        # Verify ontology was generated
        mock_generator.generate_from_schema.assert_called_once()

        # Verify it used exactly the cached tables
        used_tables = mock_generator.generate_from_schema.call_args[0][0]
        assert used_tables == cached_data["public"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

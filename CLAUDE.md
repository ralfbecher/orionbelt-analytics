# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OrionBelt Analytics** - A FastMCP server that analyzes database schemas (PostgreSQL, Snowflake, Dremio, ClickHouse) and generates RDF/OWL ontologies with embedded SQL mappings. Enables Text-to-SQL with fan-trap prevention and relationship-aware query construction.

## Build & Development Commands

```bash
# Install dependencies (Python 3.13+ required)
uv sync

# Start MCP server (default: localhost:9000)
uv run server.py

# Run tests with coverage
pytest

# Run a single test file
pytest tests/test_ontology_generator.py -v

# Run a specific test
pytest tests/test_ontology_generator.py::test_function_name -v

# Code formatting
black src/ tests/
isort src/ tests/

# Linting and type checking
flake8 src/
mypy src/

# Run all pre-commit checks
pre-commit run --all-files

# Security scan
bandit -r src/
```

## Architecture

### Core Components

- **`server.py`** - Entry point with startup logging and signal handling
- **`src/main.py`** (~2800 lines) - FastMCP server with 13 MCP tools as `@mcp.tool()` async decorators
- **`src/database_manager.py`** (~1900 lines) - Connection pooling, schema analysis, SQLAlchemy integration
- **`src/ontology_generator.py`** (~900 lines) - RDF/OWL generation using rdflib with `db:` namespace annotations
- **`src/r2rml_generator.py`** (~360 lines) - W3C R2RML mapping generation
- **`src/security.py`** (~350 lines) - SQL injection prevention, fan-trap detection, credential encryption
- **`src/chart_utils.py`** (~450 lines) - Plotly chart generation (kaleido for PNG export)

### Key Patterns

**Per-Session State Isolation**: Each MCP session maintains isolated `SessionData` with its own `DatabaseManager`, preventing cross-session interference:
```python
class SessionData:
    db_manager: DatabaseManager
    schema_file: Optional[Path]
    ontology_file: Optional[Path]
    r2rml_file: Optional[Path]
```

**Fan-Trap Prevention**: Multi-step validation prevents data multiplication errors:
1. `analyze_schema()` extracts FK relationships
2. Pattern detection in `execute_sql_query()`
3. Suggests UNION ALL patterns for multi-fact aggregation

**Ontology Triple Storage**: RDF graphs link back to SQL:
- `ns:TableName` → OWL:Class with `db:tableName`, `db:primaryKey`
- `ns:relationship` → OWL:ObjectProperty with `db:joinCondition`
- Uses `rdfs:comment` for business descriptions

### MCP Tools (in `src/main.py`)

`connect_database`, `list_schemas`, `reset_cache`, `analyze_schema`, `generate_ontology`, `suggest_semantic_names`, `apply_semantic_names`, `load_my_ontology`, `sample_table_data`, `validate_sql_syntax`, `execute_sql_query`, `generate_chart`, `get_server_info`

## Configuration

Key environment variables (see `.env.template`):
- `MCP_TRANSPORT` - `http` (default) or `sse`
- `MCP_SERVER_PORT` - Server port (default: 9000)
- `ONTOLOGY_BASE_URI` - RDF base URI
- `R2RML_BASE_IRI` - R2RML mapping IRI
- `OUTPUT_DIR` - Generated files directory (default: `tmp`)
- Database configs: `POSTGRES_*`, `SNOWFLAKE_*`, `DREMIO_*`, `CLICKHOUSE_*`

## Code Standards

- Line length: 88 characters (black)
- Type hints required for all public functions (mypy strict)
- Google-style docstrings
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

## Implementation Guidelines

**IMPORTANT: Plan thoroughly before coding!**

When implementing changes, ALWAYS:

1. **Trace all code paths**: Before modifying a function, identify ALL places it's called from and ALL branches within it (success path, error path, fallback path, cache hit/miss, etc.)

2. **Consider database-specific behavior**: This codebase supports PostgreSQL, Snowflake, Dremio, and ClickHouse. Each has different:
   - Case sensitivity (Snowflake uses UPPERCASE identifiers; ClickHouse is case-sensitive)
   - Query syntax (SHOW commands, information_schema differences, system.* tables for ClickHouse)
   - Caching behavior (schema-level vs table-level queries)
   - Constraint model (ClickHouse has no FKs; PRIMARY KEY = sparse index, ORDER BY defines sort)

3. **Check cache implications**: When adding caching:
   - What happens on cache HIT?
   - What happens on cache MISS (fallback)?
   - Is the fallback path equally efficient?
   - Are cache keys consistent across related functions?

4. **Handle session state**: Changes to `SessionData` must consider:
   - When state is set (which tool sets it?)
   - When state is used (which tools depend on it?)
   - When state should be cleared (reconnect, new session?)

5. **Test edge cases**: Don't just test the happy path. Consider:
   - Empty results
   - Missing parameters (None/empty string)
   - Case sensitivity mismatches
   - Network/query failures

**Avoid incremental fixes** - solve problems completely the first time by understanding the full scope before writing code.

## Working Style

- **Verify before done**: Run tests and check logs to prove changes work before marking complete
- **Fix bugs directly**: When given a bug report, investigate and fix it - don't ask for hand-holding
- **Re-plan when stuck**: If implementation hits unexpected complexity, stop and re-plan rather than pushing through with hacky fixes
- **Challenge your work**: Before presenting a solution, ask "would a staff engineer approve this?"

## Dependencies

Core: `fastmcp>=2.14.4`, `sqlalchemy>=2.0.36`, `rdflib>=7.1.0`, `pydantic>=2.10.0`, `clickhouse-sqlalchemy>=0.3.0`

Visualization: `plotly>=6.1.1`, `kaleido>=1.2.0`, `pandas>=2.2.0`

MCP-UI: `mcp-ui-server>=1.0.0` (for interactive chart rendering in Claude Desktop)

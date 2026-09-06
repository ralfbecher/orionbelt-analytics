# Development Guide

[<- Back to README](../README.md)

This guide covers the development workflow for OrionBelt® Analytics -- prerequisites, project layout, architecture, testing, and contribution guidelines.

---

## Prerequisites

- **Python 3.13 or higher** -- required by the project and enforced in `pyproject.toml`
- **uv** package manager -- handles virtual environment creation, dependency resolution, and script execution in a single tool. Install from [github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- **Git** -- for version control and conventional commit workflow

Optional but recommended:

- **pre-commit** -- runs formatting and linting checks before each commit (installed as a dev dependency)

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/ralforion/orionbelt-analytics
cd orionbelt-analytics
uv sync
```

`uv sync` creates a virtual environment with Python 3.13, installs all production and dev dependencies, and locks the versions.

### 2. Configure environment

```bash
cp .env.template .env
```

Edit `.env` with your database credentials and server settings. Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MCP_TRANSPORT` | `http` | Transport mode (`http` or `sse`) |
| `MCP_SERVER_HOST` | `localhost` | Server bind address |
| `MCP_SERVER_PORT` | `9000` | Server port |
| `ONTOLOGY_BASE_URI` | `http://example.com/ontology/` | RDF namespace base |
| `R2RML_BASE_IRI` | `http://mycompany.com/` | R2RML mapping IRI |
| `OUTPUT_DIR` | `tmp` | Directory for generated files |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `AUTO_GRAPHRAG` | `true` | Auto-initialize GraphRAG on schema analysis |
| `SESSION_IDLE_TIMEOUT_SECONDS` | `1800` | Idle session eviction (0 to disable) |

Database-specific variables (`POSTGRES_*`, `SNOWFLAKE_*`, `CLICKHOUSE_*`, `DREMIO_*`, `BIGQUERY_*`, `DUCKDB_*`, `DATABRICKS_*`, `MYSQL_*`) are documented in `.env.template` and in [docs/configuration.md](configuration.md).

### 3. Start the server

```bash
uv run server.py
```

The server starts on `http://localhost:9000` by default, using streamable HTTP transport.

---

## Running Tests

Tests use **pytest** with `asyncio_mode = "auto"` and automatic coverage reporting. Configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`.

```bash
# Run the full test suite with coverage
uv run pytest

# Run a single test file
uv run pytest tests/test_ontology_generator.py -v

# Run a specific test by name
uv run pytest tests/test_ontology_generator.py::test_function_name -v

# Run tests matching a keyword expression
uv run pytest -k "ontology and not server"

# Run with detailed coverage report
uv run pytest --cov=src --cov-report=term-missing

# Run with HTML coverage output
uv run pytest --cov=src --cov-report=html
```

Default `addopts` in `pyproject.toml` already includes `-ra -q --cov=src --cov-report=term-missing`, so a bare `uv run pytest` produces a coverage summary.

Test files follow the `test_*.py` convention in the `tests/` directory, covering ontology generation, database management, security, OBQC validation, GraphRAG integration, OWL axioms, Oxigraph persistence, session eviction, workspaces, and server tools.

---

## Code Quality

### Formatting

```bash
# Auto-format with black (line length 88, Python 3.13 target)
black src/ tests/ server.py

# Sort imports (configured for black compatibility)
isort src/ tests/ server.py
```

### Linting

```bash
# Ruff -- fast linter and formatter
ruff check src/ tests/ server.py

# Flake8
flake8 src/

# Find unused code
vulture src/
```

### Type checking

```bash
# mypy with strict mode (configured in pyproject.toml)
mypy src/
```

mypy is configured with `disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any`, `strict_equality`, and other strict options. All public functions require type hints.

### Security scanning

```bash
# Static security analysis (skips assert_used in tests)
bandit -r src/

# Dependency vulnerability scanning, against the tree that actually ships
uv export --no-dev --no-emit-project --format requirements-txt \
  | uvx pip-audit --disable-pip -r -
```

Dependabot security updates scan the whole of `uv.lock`, so its alert list is
wider than the shipped surface — it includes the dev group. The `pip-audit`
command above is the narrower check: what a user installing the wheel is
actually exposed to.

### Workflow action pins

Every `uses:` in `.github/workflows/` is pinned to a 40-character commit SHA
with a `# vX.Y.Z` comment naming the release it was cut from. That is why the
workflows are full of hex. A git tag is a movable label, so `actions/checkout@v7`
runs whatever commit that label points at when the job starts, and the owner of
the action can repoint it at any time. A SHA cannot move.

Pinning fixes *which* code runs, but it also makes every reference unreadable,
and nothing otherwise keeps a hash and the comment beside it in agreement. A
pull request could swap the hash for one taken from a fork -- reachable under
the real repository's URL, because forks share object storage with their parent
-- leave the comment saying `# v7.0.1`, and produce a diff that looks exactly
like a routine Dependabot bump. Only resolving the tag tells the two apart:

```bash
# Verify every pin: SHA present, owner allowed, comment names an exact patch
# release, and that release really is the commit pinned.
./scripts/check-action-pins.sh

# Same, minus the upstream `git ls-remote` lookups -- useful offline or when
# you only want the format checks.
./scripts/check-action-pins.sh --offline
```

The comment must name an exact patch release (`# v7.0.1`), never a major tag:
`v7` moves with every upstream release and would turn the check red for a pin
that is still perfectly good. Tags are resolved with `git ls-remote`, so the
check needs no token and is not rate limited. It fails closed -- a lookup that
could not be made at all is a failure, not a pass.

The script's `ALLOWED_OWNERS` list carries the most weight of its three checks,
because a SHA matching its own tag says nothing about whether the action
belongs here at all: `evil/action@<sha> # v1.0.0` verifies perfectly well
against evil's own tag. Adding an owner is a deliberate decision, not a
formality. The `pins` job in `ci.yml` runs the check on every pull request, and
both tag-publishing workflows run it as the first step after checkout, before
any other action -- an action that has already executed could have rewritten
the workspace, this script included.

What it does not do: it does not judge whether an action is safe (a pinned SHA
of a genuinely malicious release passes), it does not stop a downgrade to a
real but ancient release, and it says nothing about upstream being compromised.
`actions/checkout` runs before the check and is therefore unverified -- an
irreducible bootstrap dependency, since nothing can be checked before the tree
it fetches exists.

To resolve a SHA yourself when bumping an action:

```bash
git ls-remote https://github.com/actions/checkout 'refs/tags/v7.0.1^{}'
```

### Pre-commit hooks

```bash
# Install hooks (one-time setup)
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

Pre-commit runs formatting, linting, and security checks automatically before each commit.

---

## Project Structure

```
orionbelt-analytics/
|-- server.py                        # Entry point -- starts FastMCP server
|-- pyproject.toml                   # Dependencies, tool configs, build metadata
|-- .env.template                    # Environment variable reference
|
|-- src/
|   |-- __init__.py                  # Package version (__version__)
|   |-- main.py                      # FastMCP server setup, @mcp.tool() registration
|   |-- session.py                   # SessionData -- per-session + per-schema state isolation
|   |-- config.py                    # Configuration management, .env loading
|   |-- constants.py                 # Shared constants
|   |-- paths.py                     # Centralized path resolution (output, skills, config)
|   |-- utils.py                     # Logging setup, shared utilities
|   |-- async_utils.py               # Async helper functions
|   |-- serialization.py             # Data serialization helpers
|   |-- exceptions.py                # Custom exception classes
|   |-- workspace.py                 # Workspace management
|   |
|   |-- database_manager.py          # Connection pooling, schema analysis (SQLAlchemy)
|   |-- ontology_generator.py        # RDF/OWL generation (rdflib, oba: namespace)
|   |-- oxigraph_store.py            # Persistent RDF store, SPARQL 1.1 queries
|   |-- obqc_validator.py            # Ontology Basic Quality Criteria (sqlglot)
|   |-- r2rml_generator.py           # W3C R2RML mapping generation
|   |-- security.py                  # SQL injection prevention, fan-trap detection,
|   |                                #   credential encryption (AES-128-CBC)
|   |-- chart_utils.py               # Plotly chart generation, kaleido PNG export
|   |-- dremio_client.py             # Dremio-specific client logic
|   |
|   |-- handlers/                    # MCP tool implementations (handler layer)
|   |   |-- connection.py            #   connect_database, list_schemas
|   |   |-- schema.py                #   discover_schema, get_table_details, reset_cache
|   |   |-- ontology.py              #   generate_ontology, suggest/apply semantic names,
|   |   |                            #     load_my_ontology
|   |   |-- query.py                 #   validate_sql_syntax, execute_sql_query,
|   |   |                            #     sample_table_data
|   |   |-- chart.py                 #   generate_chart
|   |   |-- rdf.py                   #   SPARQL tools, RDF store operations
|   |   |-- graphrag.py              #   GraphRAG initialization, context retrieval
|   |   +-- workspace.py             #   cleanup_workspace, save/get/list_semantic_model
|   |
|   |-- drivers/                     # Database-specific drivers (abstract base pattern)
|   |   |-- base.py                  #   BaseDriver -- abstract interface
|   |   |-- postgresql.py            #   PostgreSQL driver
|   |   |-- mysql.py                 #   MySQL driver
|   |   |-- snowflake.py             #   Snowflake (UPPERCASE identifiers, case-sensitive)
|   |   |-- clickhouse.py            #   ClickHouse (no FKs, ORDER BY sort, system.*)
|   |   |-- dremio.py                #   Dremio driver
|   |   |-- bigquery.py              #   BigQuery driver
|   |   |-- duckdb.py                #   DuckDB / MotherDuck driver
|   |   +-- databricks.py            #   Databricks SQL driver
|   |
|   |-- graphrag/                    # Graph-based RAG for schema intelligence
|   |   |-- manager.py               #   Orchestrator (auto-init by discover_schema)
|   |   |-- embedder.py              #   Vector embeddings for schema elements
|   |   |-- retriever.py             #   Graph traversal, relationship discovery (12 hops)
|   |   |-- community_detector.py    #   Schema clustering via community detection
|   |   |-- vector_store.py          #   Abstract vector store interface
|   |   +-- vector_store_chromadb.py  #   ChromaDB vector storage implementation
|   |
|   |-- lifecycle/                   # Artifact lifecycle management
|   |   |-- metadata.py              #   Version tracking, retention policies
|   |   +-- cleanup.py               #   Data cleanup for tmp/ outputs
|   |
|   |-- tools/                       # Tool utilities
|   |   +-- chart.py                 #   Chart tool helpers
|   |
|   +-- apps/                        # MCP Apps (interactive UI)
|       +-- chart_viewer.html        #   Embedded chart viewer template
|
|-- tests/                           # pytest test suite (test_*.py files)
|-- ontology/                        # Ontology specs and SHACL shapes
|-- integrations/                    # AI framework integration examples
|-- docs/                            # Documentation
+-- .claude/skills/                  # MCP skill resources
```

---

## Architecture

### Three-layer pattern

The codebase follows a strict three-layer separation:

```
server.py
  +-- src/main.py          (1) Registration layer -- @mcp.tool() async decorators
        +-- src/handlers/  (2) Handler layer     -- tool implementation logic
              +-- src/*.py  (3) Service layer     -- domain modules
```

**Registration layer** (`src/main.py`): Thin wrappers that define MCP tool signatures with `@mcp.tool()` decorators. Each wrapper extracts session state and delegates to the corresponding handler. No business logic lives here.

**Handler layer** (`src/handlers/*.py`): Contains the actual tool implementation. Handlers receive utilities (database manager, session data, configuration) as parameters and orchestrate calls to service-layer modules. Each file groups related tools:

- `connection.py` -- database connections
- `schema.py` -- schema analysis
- `ontology.py` -- ontology generation and semantic names
- `query.py` -- SQL validation and execution
- `chart.py` -- chart generation
- `rdf.py` -- SPARQL and RDF store operations
- `graphrag.py` -- GraphRAG initialization and context retrieval
- `info.py` -- server info

**Service layer** (`src/database_manager.py`, `src/ontology_generator.py`, etc.): Pure domain logic with no MCP dependencies. These modules are independently testable and handle database connectivity, RDF generation, security validation, and so on.

### Database driver pattern

An abstract base driver (`src/drivers/base.py`) defines the interface that each database implementation must satisfy. Per-database drivers handle dialect differences:

```
BaseDriver (src/drivers/base.py)
  |-- PostgreSQLDriver     Standard SQL, information_schema
  |-- MySQLDriver          MySQL-specific syntax, utf8mb4
  |-- SnowflakeDriver      UPPERCASE identifiers, case-sensitive quoting
  |-- ClickHouseDriver     No FKs, system.* tables, ORDER BY = sort key
  |-- DremioDriver         PostgreSQL wire protocol, path-based sources
  |-- BigQueryDriver       Project/dataset scoping, default credentials
  |-- DuckDBDriver         Local files, :memory:, MotherDuck cloud
  +-- DatabricksDriver     Unity Catalog, HTTP path, access tokens
```

Adding a new database requires implementing the `BaseDriver` interface and registering the driver in `database_manager.py`.

### Per-session state isolation

Each MCP session maintains its own `SessionData` instance (`src/session.py`), which contains:

- **ConnectionState** -- active database manager, connection ID (session-scoped)
- **SchemaCache** -- cached schema analysis results (multi-schema, dict-based)
- **SchemaState** (per schema) -- bundles:
  - **OntologyState** -- generated ontology content, file paths, OBQC validator
  - **schema_file** -- saved schema JSON reference
- **GraphRAGState** -- GraphRAG manager instance, initialization status (connection-scoped, accumulative)
- **RDFStoreState** -- Oxigraph store instance (connection-scoped, multi-schema via named graphs)

Ontology state is isolated per schema via `SchemaState`. GraphRAG and the Oxigraph RDF store are connection-scoped: each `discover_schema()` call accumulates tables into the same graph and vector store, enabling cross-schema join path discovery and unified semantic search. Switching schemas (e.g., `discover_schema("analytics")` after `discover_schema("public")`) does not destroy the previous schema's ontology state, and both schemas' tables are searchable in GraphRAG simultaneously.

This isolation prevents cross-session interference when multiple clients connect to the server simultaneously. Idle sessions are automatically evicted based on `SESSION_IDLE_TIMEOUT_SECONDS`.

### Key design patterns

**Fan-trap prevention**: Multi-step validation pipeline that detects dangerous join patterns in SQL queries (1:many relationships with aggregation). The system suggests safe alternatives like UNION ALL or separate CTEs.

**Ontology triple storage**: RDF graphs embed SQL metadata using the `oba:` (OrionBelt Analytics) namespace. Tables become OWL classes with `oba:tableName` and `oba:primaryKey` annotations; relationships become OWL object properties with `oba:sqlJoinCondition`. The Oxigraph store provides persistent SPARQL 1.1 query access.

**GraphRAG auto-initialization**: When `discover_schema()` runs, GraphRAG is automatically initialized in the background (controlled by `AUTO_GRAPHRAG`). The graph structure supports up to 12-hop traversal for relationship discovery, and ChromaDB stores vector embeddings for semantic similarity search.

**MCP resources**: Skills stored in `.claude/skills/` (fan-trap prevention, SQL best practices, chart examples, analytical workflow) are exposed as MCP resources via `@mcp.resource()` decorators.

---

## Python Scripting

While OrionBelt Analytics is designed to run as an MCP server, advanced users can import its service-layer modules directly for batch processing or automation:

```python
from src.database_manager import DatabaseManager
from src.ontology_generator import OntologyGenerator

db = DatabaseManager()
db.connect_postgresql("localhost", 5432, "mydb", "user", "pass")
schema = db.get_schema_info("public")

generator = OntologyGenerator()
ontology_ttl = generator.generate_ontology(schema)
```

Service-layer modules (`database_manager`, `ontology_generator`, `security`, `oxigraph_store`, etc.) have no MCP dependencies and can be used independently. Handler-layer and registration-layer code depends on FastMCP and session state and is not intended for standalone use. For most workflows, running the MCP server is the recommended approach.

---

## Contributing

### Commit messages

This project follows **conventional commits**:

```
feat:     New feature
fix:      Bug fix
docs:     Documentation only
test:     Adding or updating tests
refactor: Code restructuring without behavior change
chore:    Build, CI, or tooling changes
```

Examples:

```
feat: add MySQL driver with utf8mb4 support
fix: correct heatmap axis mapping for weekday ordering
refactor: extract driver pattern from database_manager
test: add coverage for fan-trap detection edge cases
docs: update configuration reference for BigQuery
```

### Before submitting changes

1. **Run the test suite** and confirm no regressions:

   ```bash
   uv run pytest
   ```

2. **Format and lint** your code:

   ```bash
   black src/ tests/ server.py
   isort src/ tests/ server.py
   ruff check src/ tests/ server.py
   mypy src/
   ```

   Or run all checks at once:

   ```bash
   pre-commit run --all-files
   ```

3. **Add tests** for new functionality. Place test files in `tests/` following the `test_*.py` naming convention. Use `pytest-asyncio` for async tests (auto mode is configured).

4. **Type-annotate** all public functions. The project enforces strict mypy settings including `disallow_untyped_defs` and `disallow_incomplete_defs`.

5. **Consider database-specific behavior** when modifying schema analysis or query logic. Each database has different case sensitivity rules, query syntax, constraint models, and caching behavior. See the driver pattern section above.

6. **Run security scanning** for changes touching SQL handling or credentials:

   ```bash
   bandit -r src/
   ```

7. **Verify action pins** if you touched anything under `.github/workflows/`:

   ```bash
   ./scripts/check-action-pins.sh
   ```

   The `pins` job in CI runs the same check and blocks the merge.

### Code standards

- **Line length**: 88 characters (black)
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style
- **Imports**: Sorted by isort with black-compatible profile
- **Async**: Use `async def` for MCP tool handlers; pytest uses `asyncio_mode = "auto"`

# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

OrionBelt Analytics is an MCP server (FastMCP) that analyzes relational database schemas, generates RDF/OWL ontologies with embedded SQL mappings, and provides relationship-aware Text-to-SQL with fan-trap prevention, GraphRAG schema discovery, SPARQL access, and Plotly charting. Python 3.13+, managed with `uv`.

## Commands

```bash
uv sync                                  # create venv + install all deps (prod + dev)
uv run server.py                         # start the MCP server (http://localhost:9000)

uv run pytest                            # full suite (coverage is on by default via addopts)
uv run pytest tests/test_obqc_validator.py -v          # single file
uv run pytest tests/test_obqc_validator.py::test_name -v   # single test
uv run pytest -k "ontology and not server"             # keyword expression

black src/ tests/ server.py && isort src/ tests/ server.py   # format
ruff check src/ tests/ server.py         # lint (server.py is the root entry point)
mypy src/                                # strict type check (disallow_untyped_defs etc.)
bandit -r src/                           # security scan (run for SQL/credential changes)
uv run python scripts/gen-third-party-notices.py   # refresh THIRD_PARTY_NOTICES.md after a dependency change
pre-commit run --all-files               # all of the above as configured hooks
```

Config comes from `.env` (copy from `.env.template`). At minimum set credentials for one database. Key vars: `MCP_TRANSPORT` (http/sse), `MCP_SERVER_PORT` (9000), `AUTO_GRAPHRAG`, `SESSION_IDLE_TIMEOUT_SECONDS`, `AUTO_CLEANUP_ON_STARTUP`.

## Architecture

**Strict three-layer separation** — keep these boundaries when adding or changing tools:

1. **Registration** (`src/main.py`): thin `@mcp.tool()` async wrappers. Extract session state, delegate to a handler. No business logic here.
2. **Handlers** (`src/handlers/*.py`): tool implementation. Receive db manager / session data / config as parameters and orchestrate service modules. Grouped by domain (`connection`, `schema`, `ontology`, `query`, `chart`, `rdf`, `graphrag`, `workspace`).
3. **Service** (`src/database_manager.py`, `src/ontology_generator.py`, `src/security.py`, `src/oxigraph_store.py`, `src/obqc_validator.py`, …): pure domain logic with **no MCP dependencies** — independently testable and importable for batch use.

**Database drivers** (`src/drivers/`): `BaseDriver` (`base.py`) defines the abstract interface; one driver per dialect (postgresql, mysql, snowflake, clickhouse, dremio, bigquery, duckdb, databricks). Adding a database = implement `BaseDriver` + register in `database_manager.py`. Dialect quirks matter (e.g. Snowflake UPPERCASE/case-sensitive identifiers, ClickHouse has no FKs and uses `ORDER BY` as sort key) — account for case sensitivity, constraint models, and query syntax when touching schema or query logic.

**Per-session / per-schema state** (`src/session.py`): each MCP session owns a `SessionData`. Ontology state is isolated **per schema** (`SchemaState` → `OntologyState`), but **GraphRAG and the Oxigraph RDF store are connection-scoped and accumulative** — successive `discover_schema()` calls add tables to the *same* graph/vector store/named-graphs, so cross-schema join discovery and unified search work, and switching schemas does not destroy a prior schema's ontology. Idle sessions are auto-evicted.

**OBQC (Ontology-Based Query Check)** — the core differentiator. `src/obqc_validator.py` deterministically validates every SQL statement (parsed with `sqlglot`) against the loaded ontology: table/column existence, join validity, type compatibility, GROUP BY correctness, and **fan-trap detection** (aggregation across 1:many joins that silently multiplies rows). Runs inside `execute_sql_query` — **errors block execution, warnings attach to the response** for the LLM to self-correct. No LLM calls; fully deterministic.

**Ontology / RDF**: `ontology_generator.py` emits OWL where tables → classes and FKs → object properties, annotated in the `oba:` namespace (`oba:tableName`, `oba:primaryKey`, `oba:sqlJoinCondition`) so SQL is recoverable from the graph. `oxigraph_store.py` persists triples for SPARQL 1.1. The shipped vocabulary lives in `ontology/oba.ttl` + `ontology/oba-shacl.ttl` (force-included into the wheel; SHACL conformance via `src/shacl_validator.py`).

**GraphRAG** (`src/graphrag/`): `manager.py` orchestrates graph traversal (`retriever.py`, up to 12 hops), embeddings (`embedder.py`), community detection, and a ChromaDB vector store. Auto-initialized by `discover_schema()` when `AUTO_GRAPHRAG=true`.

**MCP resources**: skill files under `.claude/skills/` (fan-trap prevention, SQL best practices, chart examples, workflows) are exposed via `@mcp.resource()` in `main.py`.

`docs/development.md` has the full layout, design rationale, and contributing guide — read it for deeper context. Per-feature docs: `docs/obqc.md`, `docs/graphrag.md`, `docs/fan-trap-prevention.md`, `docs/configuration.md`, `docs/tools-reference.md`.

## Conventions

- **Conventional commits** (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- Changing the **production dependency set** means regenerating attribution:
  `uv run python scripts/gen-third-party-notices.py`. CI (`--check`) enforces it.
  Version bumps alone produce no diff by design; adding a package, or one under
  copyleft terms, does — and a new copyleft dependency fails the check until it
  has a written notice. See `THIRD_PARTY_NOTICES.md`.
- Type hints required on all public functions (strict mypy); Google-style docstrings; 88-char lines; async handlers.
- Tests in `tests/` as `test_*.py`; `pytest-asyncio` in `auto` mode (no `@pytest.mark.asyncio` needed).
- Releases are **squash-only** (merge commits disabled). Tagging `v*.*.*` publishes
  both artefacts: `docker-publish.yml` builds the multi-arch image, and
  `pypi-publish.yml` builds, verifies and uploads the wheel. **The PyPI upload is
  irreversible** — a version can never be re-uploaded — so it is gated behind the
  `pypi` GitHub environment (Trusted Publishing, no stored token) and preceded by
  `twine check` plus assertions on tag/version agreement, licence metadata and the
  force-included ontology files. See
  `scripts/bump-version.sh` and the release-process note in memory before cutting
  a release.

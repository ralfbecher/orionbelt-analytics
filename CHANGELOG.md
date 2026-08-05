# Changelog

All notable changes to OrionBelt Analytics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.8.0] - 2026-08-05

Feature release centred on OBQC correctness and GraphRAG schema search. **OBQC
now blocks fan-traps instead of warning about them** — see Breaking Changes.
Six OBQC false positives that blocked valid SQL are fixed, schema search moved
from TF-IDF to semantic embeddings, and the per-version retention mechanism
records the data it was always meant to clean up. Tool surface grows from 28 to
30 tools (`add_semantic_context`, `cleanup_old_versions`).

### Breaking Changes
- **A detected fan-trap now fails the query.** `execute_sql_query` returns
  `success: false` with structured fan-trap data instead of executing and
  attaching a warning. Pass `allow_fan_out=true` to run one anyway. The single-
  child case was previously invisible entirely: the rule counted fan-out joins
  and said nothing below two, so `SUM(sales.amount)` across one 1:many join
  returned a silently multiplied total with zero warnings. (#88)

### Added
- **Semantic embeddings for GraphRAG schema search.** Schema elements were
  embedded with TF-IDF, which matches literal terms only — a query like "which
  products are most profitable and get returned the most" produced an all-zero
  vector (no stemming, `profitable` in no schema text), collapsing the ranking to
  index insertion order. Search now uses semantic embeddings, so questions match
  meaning rather than exact tokens. (#79)
- **`add_semantic_context(target, context)`** — clients can write business
  vocabulary into the semantic index for a table or column, so questions about
  "profit", "margin" or "churn" have something to land on when the real column is
  named `unitcost`. On a sales schema, the profit-margin query's top hit moved
  from `sales.unitcost` (0.261) to the annotated `sales.salesamount` (0.567).
  (#82)
- **Applied semantic names are indexed into GraphRAG.** `apply_semantic_names`
  wrote business vocabulary into the ontology and nowhere else, so the enrichment
  was invisible to search; each applied suggestion is now mirrored into the
  semantic index tagged `source="ontology"`. (#81)
- **Per-version retention now records versions.** `DataCleanupManager` had the
  cleanup half only — every reference to the version list was a read, so
  `get_versions_to_cleanup()` returned `[]` unconditionally and the mechanism
  provably did nothing. `discover_schema`, `generate_ontology` and GraphRAG init
  now open and fill in version records, opening a version archives its
  predecessor, and pre-existing workspaces are seeded rather than orphaned. Adds
  the `cleanup_old_versions` tool. (#73, closes #68)
- `docs/obqc-overview.md`, indexed from the README. (#86)

### Fixed
- **OBQC blocked database catalog queries.** Every referenced table had to be in
  the ontology, which describes user data only, so
  `SELECT table_name FROM information_schema.tables` failed with "Table 'tables'
  not found in ontology". The qualifier was being discarded by `_extract_tables`,
  so a catalog reference was indistinguishable from a user table even in
  principle; qualifiers are now kept and known catalog schemas exempted. (#83)
- **Six OBQC false positives that blocked valid SQL**: CTE names reported as
  missing tables; `ROLLUP`/`CUBE`/`GROUPING SETS` flagging every selected column
  as absent from the GROUP BY; `USING`/`NATURAL`/comma-joins reported as
  Cartesian products; window functions counted as grouping aggregates; and date
  literals warned as `date vs string` type mismatches. Alias-qualified columns
  were never resolved (a false *negative*). (#87)
- **SELECT aliases referenced from ORDER BY.** `SELECT SUM(total) AS revenue FROM
  orders ORDER BY revenue DESC` was rejected with "Column 'revenue' not found in
  any referenced table" — the most common analytical query shape could not run at
  all. Aliases used from `ORDER BY`, `GROUP BY`, `HAVING` and `QUALIFY` now
  resolve. (#84)
- **Validation rules now apply per SELECT, not query-wide.** `parsed_tables`,
  `parsed_columns` and `has_aggregation` were flat query-wide state gathered with
  `find_all` over the whole tree, so a subquery poisoned the outer scope:
  `SELECT id FROM users WHERE id IN (SELECT user_id FROM orders)` was rejected as
  a Cartesian product (two tables, zero joins in total). Present since the
  initial commit. (#86)
- **Fan-out judged per join direction.** The heuristic asked whether the joined
  table was on the "many" side of *any* relationship anywhere in the schema
  rather than of the join at hand, and counted relationships instead of joins —
  so a correct `sales → clients → countries` chain drew "2 one-to-many joins with
  aggregation". (#85)
- **SPARQL queries the union of named graphs by default.** Ontologies always load
  into a named graph, but all three pyoxigraph call sites left
  `use_default_graph_as_union` at `False`, so an unwrapped pattern matched the
  empty default graph — zero rows, no error, and the tool description and
  docstring both taught the failing shape. SELECT, ASK and CONSTRUCT now span
  every loaded schema. (#77)
- Tier B audit: error handling, async correctness and timestamp handling. (#67)

### Changed
- `fastmcp` bumped to 3.4.5 (lockfile only; the existing
  `fastmcp[apps]>=3.3.1,<3.5` constraint already permitted it). (#76)
- Dependency refresh across the `python-minor-patch` group. (#57, #59, #89)
- Removed the dead `MCP_SHUTDOWN_TIMEOUT` setting. `server.py` wrote it into the
  environment and nothing — not fastmcp, mcp, uvicorn or starlette — ever read it
  back; the comment's "default 5" did not exist either (fastmcp hardcodes
  `timeout_graceful_shutdown=2`). Both `.env.template` and the docs presented it
  as a working knob. (#78)

### Internal
- Cleanup computes its survivor set once per run instead of per deletion
  candidate, dropping an O(candidates × schemas × versions) scan and removing an
  order dependency in the ChromaDB ownership guard. (#75)
- `test_evict_idle_sessions` had never executed — `TestServerState` was a plain
  `unittest.TestCase` holding an `async def`, so unittest discarded the coroutine
  and reported a pass. Switched to `IsolatedAsyncioTestCase`. (#74)
- Ruff modernization rules adopted with an explicit lint select; all findings
  fixed. (#65, #70)
- Dev dependencies migrated to PEP 735 dependency groups; black 26.5, pytest 9.x,
  isort 8.0.1, mypy 2.3.0, pre-commit 4.6.1; unused `testcontainers` dropped.
  (#58, #60, #61, #64, #71, #72, #90, #91)

## [1.7.2] - 2026-07-15

Bug-fix release. Charting broke for anyone installing on pandas 3, which a fresh
install already resolves — the dependency floors are unbounded, so this affected
1.7.1 as published. No MCP tool surface changes (still 26 tools, same signatures).

### Fixed
- **Line-chart datetime axes lost `type="date"` under pandas 3.** The
  time-series axis enhancement was gated on a literal `datetime64[ns]` dtype;
  pandas 3 parses to `datetime64[us]`, so the check silently stopped matching and
  the axis rendered untyped — no error, just a wrong axis. Now matched with
  `pd.api.types.is_datetime64_any_dtype`, which is resolution-agnostic (and was
  already used for the same purpose elsewhere). (#54)

### Changed
- Allow `fastmcp[apps]` 3.4.x (`>=3.3.1,<3.5`). This is the only dependency
  constraint that moves for installers; the pandas/sqlglot/cryptography floors are
  unchanged. (#47)
- Docker image moves to **Python 3.14**. The base image tag is now the single
  source of truth for the image's interpreter (`UV_PYTHON` pinned to it), so a
  base bump can no longer leave uv building the virtualenv against a different
  Python. (#50)
- OBQC accepts sqlglot 30's `Expr`. `Expr` is a new base class of `Expression`
  and `parse_one` returns the wider type; the validator helpers now take
  `exp.Expr`. Types only — no behavior change. (#52)
- Dependency refresh: sqlglot 30.12.0, pandas 3.0.3, cryptography 49.0.0, plus 20
  minor/patch bumps. (#45, #49, #52, #55)

### Internal
- Dependabot now watches uv, GitHub Actions, and Docker. (#42)
- CI tests on both Python 3.13 and 3.14, covering the supported `requires-python`
  range rather than a single pinned version. (#50)
- Docker image changes are gated by a build **and smoke test** (venv interpreter
  resolves, `src.main` imports, server boots and serves) — the image previously
  was not built at all until a release tag, so a broken one surfaced as a failed
  publish. (#51, #53)

## [1.7.1] - 2026-06-26

Bug-fix release centered on the RDF/SPARQL store, which had broken against the
resolved pyoxigraph version. No MCP tool surface changes (still 26 tools, same
signatures).

### Fixed
- **RDF/SPARQL store repaired for pyoxigraph 0.5.x.** `query_sparql` (SELECT),
  `query_sparql_ask` (ASK), and `add_triple`/`add_knowledge` raised at runtime
  because they used 0.3.x APIs removed in the resolved 0.5.x. Migrated to `Quad`,
  `QuerySolutions.variables`, and `bool(QueryBoolean)`. (#39)
- **Ontology version cleanup now deletes RDF triples.** Added
  `OxigraphStoreManager.delete_graph()`; cleanup previously called a nonexistent
  method and left stale triples in the store. (#39)
- **Consistent RDF named-graph URIs.** Manual storage, auto-persistence,
  semantic-name persistence, and export/download now share a single
  `schema_graph_uri()` helper, so a manually stored ontology is no longer
  invisible to RDF export. (#39)
- **`query_sparql` timeout is now enforced (best-effort).** `timeout_seconds`
  previously had no effect; it now unblocks the caller when the timeout elapses
  (the underlying query may keep running, as pyoxigraph exposes no native
  cancellation). (#39)
- **Safe Oxigraph store open.** A locked store no longer has its RocksDB `LOCK`
  auto-deleted on open failure (which risked two-process corruption); the error
  propagates with recovery guidance. (#39)
- CONSTRUCT serialization uses `RdfFormat.TURTLE` instead of a deprecated MIME
  string. (#39)

### Changed
- Pinned `pyoxigraph>=0.5,<0.6` and upgraded to 0.5.9. (#39)
- `src/` now passes strict mypy; the mypy check is a blocking CI gate (was
  advisory). (#37)
- Author email updated to info@ralforion.com. (#33)

### Added
- Deterministic MCP tool-surface audit workflow (`mcp-xray`) in CI, rendering its
  report to the run summary. (#35, #36)
- Docker Hub badges and publish workflow. (#34)

## [1.7.0] - 2026-06-22

Architecture-review release: correctness fixes, stronger SQL safety, and a large
internal refactor of the registration layer. No MCP tool surface changes (still
26 tools, same signatures).

### Added
- **Driver/dialect registry** (`src/drivers/registry.py`) — a single source of
  truth mapping each `db_type` to its driver class and sqlglot dialect.
  `SUPPORTED_DB_TYPES`, the driver classes, and OBQC's dialect map now all derive
  from `constants.DB_SQLGLOT_DIALECTS`, so they can no longer drift apart.
- **Parser-based SQL safety gate** (`security.analyze_sql_statement`) — sqlglot
  validation is now the primary read-only/single-statement check, run ahead of
  the regex first-filter.
- **Typed `HandlerContext`** — handlers receive one request-scoped services
  object instead of many helper keyword arguments.
- CI workflow enforcing ruff, black, isort and pytest (mypy advisory).

### Changed
- **OBQC dialect parity** — `bigquery`, `duckdb`, `databricks` and `mysql` are
  now validated with their own sqlglot dialect instead of silently falling back
  to the PostgreSQL dialect.
- SQL validation now catches write/DDL operations hidden after a CTE
  (`WITH x AS (...) INSERT ...`) and no longer mis-flags a semicolon inside a
  string literal (`WHERE name = 'a;b'`) as multiple statements.
- **Refactor (no behavior change):** split the 1,237-line `main.py` into
  `server_state.py`, `resources.py` and `tool_types.py`; decomposed the
  1,466-line `handlers/ontology.py` into focused
  generation/semantic/io/artifacts modules.
- Repository formatted with black + isort; tooling aligned to Python 3.13.

### Fixed
- **AUTO_ONTOLOGY** background generation called a non-existent generator method
  and silently produced nothing; it now generates the ontology correctly.
- Artifact downloads (`download_artifact`) honor an explicit `schema_name` and
  read that schema's per-schema state instead of the currently active schema.
- RDF-store auto-restore on `connect_database` no longer fails with an
  AttributeError on RDF-enabled workspaces.
- Startup banner reports the real registered tool count (was a stale `23`).

## [1.6.0] - 2026-06-15

### Added
- **Graph reasoning surface over the generated ontology** (`design/PLAN_graph_reasoning.md`):
  - **`oba:joinsTo` shared join predicate.** Every declared or inferred many-to-one foreign key now also emits a single directed `oba:joinsTo` edge between the table classes (finer grain → coarser grain). A single SPARQL property path `?from oba:joinsTo+ ?to` answers directed reachability across all FKs without enumerating each per-FK object property. Declared in `ontology/oba.ttl` (vocabulary bumped 0.1 → 0.2).
  - **`reachable_from` MCP tool** — the dimension-capable tables for a query anchored on a table (many-to-one closure): coarser-grain tables joinable without row multiplication, safe to GROUP BY / filter on.
  - **`measurable_from` MCP tool** — the measure-capable tables (one-to-many closure, the inverse): finer-grain tables that fan out the anchor and must only be aggregated, never used as dimensions at this grain. Both closures are cycle-safe.
  - **`plan_composite_query` MCP tool** — advisory Composite Fact Layer (CFL) decomposition: detects whether the requested facts are independent grains (disjoint siblings) requiring a `UNION ALL` composite, and returns the leg roots, conformed (shared) GROUP BY dimensions, and per-leg NULL-pad sets. Advisory only — OBA does not compile SQL; defer compilation to OrionBelt Semantic Layer when connected.
  - **Optional in-process SHACL validation.** When `pyshacl` is installed, `generate_ontology` validates the generated ontology against `ontology/oba-shacl.ttl` and surfaces violations as a non-blocking warning (gated by `OBA_SHACL_VALIDATE`, default on). Gracefully no-ops when the dependency or shapes file is absent.

### Changed
- **OBQC fan-trap detection is now grounded in the ontology's own `owl:disjointWith` axioms** (sibling facts sharing a dimension) instead of only re-deriving the risk from the relationship heuristic. When disjoint facts appear together in an aggregating query, OBQC cites the actual disjoint tables and recommends a Composite Fact Layer (UNION ALL). The relationship heuristic is retained as a fallback when no disjointness axioms are present.

## [1.5.3] - 2026-06-10

### Changed
- **Constrained MCP tool string parameters at the input boundary.** Enumerated parameters now use `Literal` types (`db_type`, `cache_type`, `artifact_type`, `source`, `chart_type`, `chart_style`, `sort_order`, `output_format`, `element_type`), free-text and identifier parameters carry `max_length` bounds, and filename/model-name parameters reject path separators — so invalid, oversized, or path-traversing arguments are rejected before reaching a handler, and the constraints are published in each tool's JSON schema. Handlers keep their existing runtime validation as defense-in-depth. `save_semantic_model` additionally reduces `model_name` to a bare filename component so it can never escape the models directory.

### Fixed
- **Aligned all documentation with the registered tool set.** Removed references to four tools that were advertised but never registered (`validate_sql_syntax`, `download_ontology`, `list_tables_sparql`, `diagnose_connection_issue`) across the README, `docs/`, Claude skills, and all 8 integration guides, replacing them with registered equivalents (`execute_sql_query`'s built-in validation, `download_artifact`, `query_sparql`). Completed `docs/tools-reference.md` to cover all 23 tools (was 16). Corrected stale tool counts (README 32, startup banner 22 → 23).
- **Resolved all pre-existing lint findings** (ruff: unused imports, f-strings without placeholders, unused locals, redundant imports; intentional late imports annotated with `# noqa: E402`). `ruff check .` is now clean.

### Added
- `scripts/bump-version.sh` — bumps the version across all files, inserts a CHANGELOG stub, and runs `uv lock` in one step.
- `scripts/publish-docker.sh` accepts a generic `DOCKERHUB_PAT` (legacy `DOCKERHUB_RALFORION_PAT` still works).

## [1.5.2] - 2026-06-08

### Removed
- **`get_server_info` MCP tool** -- removed as redundant. Server metadata (name, version, supported databases, capabilities) is already provided via the MCP `initialize` handshake and the server `instructions`, and the live tool list via `tools/list`. The genuinely-unique capability descriptions were folded into the server instructions.

### Changed
- **Condensed server `instructions`** (~58 -> ~30 lines) -- dropped the duplicated database list and `Version:` footer (already supplied via `FastMCP(version=...)`), merged overlapping capability sections.
- **Slimmed `generate_chart`** from 13 to 11 parameters -- removed `width`/`height`. Interactive charts are responsive and size to their container; static PNG export now uses fixed 800x600 constants. Docstring condensed.

## [1.5.1] - 2026-06-06

### Fixed
- **MCP handshake advertised the wrong version** -- `FastMCP()` was constructed without an explicit `version`, so the `initialize` response's `serverInfo.version` fell back to the FastMCP package version (e.g. `3.2.4`) instead of the application version. The constructor now receives `version=__version__`.

### Changed
- Upgraded `fastmcp[apps]` from `>=3.2.4` to `>=3.3.1,<3.4`.

## [1.5.0] - 2026-05-03

### Added
- **MCP sampling for `suggest_semantic_names`** -- when the connected client advertises the sampling capability, the server now calls back through the host LLM via `ctx.sample()` to pre-fill rename suggestions for cryptic identifiers, returning a `suggestions` dict alongside the cryptic-name lists in a single tool call.
  - Gated on a new `ENABLE_SAMPLING` env flag (default `true`). Set to `false` to force the legacy manual-review path everywhere.
  - Clients without sampling support (e.g. Claude Desktop) silently fall back to the legacy response shape — no breaking change.
  - Sampling requests, results, and failures are logged with elapsed time and item counts for observability.

### Fixed
- **MCP session crash on client disconnect during `suggest_semantic_names`** -- a notification (`ctx.info`) write that hit `anyio.ClosedResourceError` because the client had already closed the streamable-HTTP session was caught by the handler's outer `except` and triggered a second doomed write, bringing down the entire FastMCP TaskGroup. Notifications are now sent through `safe_ctx_info` (failures swallowed at debug level), and `ClosedResourceError`-class disconnects re-raise cleanly so the framework tears the session down instead of writing into a dead transport.
- **Sampling response parsing** -- replaced pydantic-ai's `result_type=Dict[str, str]` (which forces the model to call an injected `final_response` tool, fragile on large responses) with explicit JSON parsing that accepts bare JSON, ```json fences, and prose-embedded JSON.

## [1.2.0] - 2026-04-05

### Changed
- **Migrated to official FastMCP Apps standard** - Replaced `mcp-ui-server` community library with native FastMCP Apps support
  - Charts now use official `ui://` resource URI pattern with `AppConfig`
  - Chart viewer configured with `ResourceCSP` for CDN security (Plotly, unpkg)
  - Full compatibility with Claude Desktop, Claude.ai, ChatGPT, VS Code, and Goose
  - Cleaner implementation using standard MCP Apps protocol
- Upgraded `fastmcp` dependency from `>=3.1.0` to `fastmcp[apps]>=3.1.0`

### Removed
- Dependency on `mcp-ui-server>=1.0.0` (replaced by official FastMCP Apps)

## [1.1.3] - 2026-03-28

### Fixed
- Clear OBQC validator on connection change to prevent cross-database validation
- Clear Oxigraph store on connection change to prevent cross-connection RDF contamination
- Extend `_reconnect()` for BigQuery, DuckDB, Databricks, and MySQL backends
- Add DREMIO_URI + DREMIO_PAT auth support to main connect_database handler

### Added
- 11 regression tests for code review findings

## [1.1.2] - 2026-03-27

### Fixed
- Fix Dremio routing bug calling connect_postgresql instead of connect_dremio
- Add SPARQL injection escaping for pyoxigraph f-string queries
- Register atexit cleanup handler for session/store teardown
- Clean tmp/ subdirectories on startup, not just files
- Align execute_sql_query row limit from 10000 to 5000 (matching docstring)
- Fix GraphRAG init task tracking via session attribute instead of monkey-patch
- Remove circular import in chart handler
- Use typed exceptions in RDF handler
- Fix all failing tests (pytest shebang, FastMCP API changes, mock fixtures)

### Added
- Expose all 8 database drivers in connection handler (BigQuery, DuckDB, Databricks, MySQL)
- Extract _table_info_to_dict() utility for GraphRAG handler
- Update get_server_info to list all 8 databases, 28 tools, 10 features

### Changed
- Downgrade verbose schema analysis logs from INFO to DEBUG
- Remove unused ThreadPoolExecutor from DatabaseManager
- Remove stale Python 3.10-3.12 classifiers from pyproject.toml
- Remove outdated test_integration.py

## [1.1.1] - 2026-03-27

### Fixed
- Lowered `pandas` requirement from `>=3.0.0` to `>=2.2.3` to resolve dependency conflict with `databricks-sql-connector` (which caps pandas at `<2.4.0`)

### Changed
- Published to PyPI as `orionbelt-analytics`

## [1.1.0] - 2026-03-22

### Added
- **MySQL Support** - Full support for MySQL 8.0+ and MariaDB 10.5+
  - MySQL 5.7 reached EOL in October 2023 (no longer supported)
  - MySQL 8.0+ provides CTEs, window functions, and improved performance
- New MySQL database driver: `mysql.py` with PyMySQL connector
- Connection method: `connect_mysql()` with charset configuration (default: utf8mb4)
- MySQL configuration section in `.env.template` with troubleshooting guide
- MySQL system schema exclusions: `information_schema`, `mysql`, `performance_schema`, `sys`
- MySQL badge and documentation in README
- Connection pooling with automatic reconnection for MySQL (pool_pre_ping=True)

### Changed
- Supported databases expanded from 7 to 8
- README updated with MySQL in all database lists
- Version bumped to 1.1.0
- Project keywords expanded to include `mysql`

### Dependencies
- Added `pymysql>=1.1.0` for MySQL connectivity (pure Python, cross-platform)

### Database Support Summary
OrionBelt Analytics v1.1.0 now supports:
1. PostgreSQL
2. **MySQL** (NEW)
3. Snowflake
4. ClickHouse
5. Dremio
6. BigQuery
7. DuckDB/MotherDuck
8. Databricks SQL

## [1.0.0] - 2026-03-16

### Added
- **BigQuery Support** - Full support for Google BigQuery with service account authentication
- **DuckDB/MotherDuck Support** - Local DuckDB files and MotherDuck cloud database support
- **Databricks SQL Support** - Databricks SQL Warehouse and Unity Catalog integration
- New database drivers: `bigquery.py`, `duckdb.py`, `databricks.py`
- Connection methods: `connect_bigquery()`, `connect_duckdb()`, `connect_databricks()`
- Configuration templates for all new databases in `.env.template`
- Comprehensive troubleshooting guides for BigQuery, DuckDB/MotherDuck, and Databricks
- Database-specific system schema exclusions for all vendors
- Updated README with badges, examples, and documentation for all 7 databases
- Connection test examples for new databases

### Changed
- **FastMCP upgraded to 3.1+** - Updated from 3.0.2 to >=3.1.0 for latest features
- Development Status upgraded to "Production/Stable" (was "Beta")
- README badge updated to reflect FastMCP 3.1+
- Copyright year updated to 2025-2026
- "Better Together" section now lists all 7 supported databases
- Key dependencies documentation updated to include all database connectors
- Project keywords expanded to include bigquery, duckdb, databricks

### Dependencies
- Added `sqlalchemy-bigquery>=1.11.0` for BigQuery support
- Added `duckdb>=1.1.0` and `duckdb-engine>=0.13.0` for DuckDB support
- Added `databricks-sql-connector>=3.5.0` for Databricks support
- Updated `fastmcp>=3.1.0` (was >=3.0.2)

### Compatibility
- Fully compatible with OrionBelt Semantic Layer v1.0.0
- Fully compatible with OrionBelt Semantic Layer MCP v1.0.0
- All three platform components now support the same 7 database vendors

### Database Support Summary
OrionBelt Analytics v1.0.0 now supports:
1. PostgreSQL
2. Snowflake
3. ClickHouse
4. Dremio
5. **BigQuery** (NEW)
6. **DuckDB/MotherDuck** (NEW)
7. **Databricks SQL** (NEW)

## [0.7.0] - 2024

### Added
- SPARQL query support with 7 SPARQL tools
- GraphRAG integration for schema discovery
- Comprehensive ontology generation
- RDF/OWL support with Oxigraph storage

### Changed
- Enhanced documentation
- Improved repository structure

## [0.6.0] - 2024

### Added
- Initial release with PostgreSQL, Snowflake, ClickHouse, and Dremio support
- FastMCP 3.0 integration
- Ontology generation
- Schema analysis tools

[1.0.0]: https://github.com/ralforion/orionbelt-analytics/releases/tag/v1.0.0
[0.7.0]: https://github.com/ralforion/orionbelt-analytics/releases/tag/v0.7.0
[0.6.0]: https://github.com/ralforion/orionbelt-analytics/releases/tag/v0.6.0

[<- Back to README](../README.md)

# MCP Tools Reference

Complete reference for all OrionBelt® Analytics MCP tools. These tools are invoked by AI clients (Claude, etc.) through the Model Context Protocol -- they are not Python functions.

---

## Recommended Workflows

### Standard Analysis Workflow

1. **connect_database** -- establish a secure database connection
2. **list_schemas** -- discover available schemas
3. **discover_schema** -- extract schema structure with relationships (auto-generates R2RML)
4. **generate_ontology** -- create semantic ontology with `oba:` annotations
5. **suggest_semantic_names** -- identify cryptic/abbreviated names for review
6. **apply_semantic_names** -- apply LLM-suggested improvements
7. **execute_sql_query** -- run validated SQL with fan-trap protection
8. **generate_chart** -- visualize results

### Resuming a Previous Session

1. **connect_database** -- reconnect (auto-restores workspace if one exists)
2. Continue with **execute_sql_query**, **generate_chart**, etc.

### Quick Data Exploration

1. **connect_database** -- connect
2. **discover_schema** -- lightweight mode (default) for fast overview
3. **sample_table_data** -- preview actual data
4. **execute_sql_query** -- run queries

### GraphRAG-Assisted Analysis (large schemas)

1. **connect_database** -- connect
2. **discover_schema** -- auto-initializes GraphRAG in the background
3. **graphrag_query_context** -- narrow the schema to just the tables/columns relevant to the question (85-95% fewer tokens than the full schema)
4. *(optional)* **graphrag_find_join_path** -- resolve how two specific tables join, or **graphrag_search** to explore the schema by keyword
5. **execute_sql_query** -- run SQL informed by the focused context
6. **generate_chart** -- visualize results

### Semantic Exploration & Export (SPARQL / RDF)

1. **connect_database** -- connect
2. **discover_schema** -> **generate_ontology** -- generating the ontology auto-persists it to the Oxigraph RDF store (use **store_ontology_in_rdf** only if `auto_persist` was disabled)
3. **query_sparql** -- explore classes, properties, and relationships semantically
4. *(optional)* **add_rdf_knowledge** -- layer custom metadata triples onto the ontology
5. **download_artifact** -- export the ontology or R2RML mapping as a Turtle file for external tools (Protégé, Ontop, D2RQ)

---

## Tool Reference

### 1. connect_database

Connect to a database using credentials from environment variables.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `db_type` | string | Yes | Database type: `postgresql`, `mysql`, `snowflake`, `clickhouse`, `dremio`, `bigquery`, `duckdb`, `databricks` |

**Returns:** Connection status message. If a previous workspace exists for this connection, includes a workspace summary with available artifacts.

**Key Features:**
- Credentials are read from environment variables (e.g., `POSTGRES_HOST`, `SNOWFLAKE_ACCOUNT`), not passed as parameters
- Automatically detects existing workspaces from prior sessions
- Clears session state when switching to a different database connection
- Generates a connection fingerprint for workspace scoping

**Environment Variables by Database:**

| Database | Required Variables |
|----------|-------------------|
| PostgreSQL | `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DATABASE`, `POSTGRES_USERNAME`, `POSTGRES_PASSWORD` |
| MySQL | `MYSQL_HOST`, `MYSQL_DATABASE`, `MYSQL_USERNAME`, `MYSQL_PASSWORD` |
| Snowflake | `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USERNAME`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE` |
| ClickHouse | `CLICKHOUSE_HOST`, `CLICKHOUSE_DATABASE` |
| Dremio | `DREMIO_URI` + `DREMIO_PAT` (preferred), or `DREMIO_HOST` + `DREMIO_PORT` + `DREMIO_USERNAME` + `DREMIO_PASSWORD` |
| BigQuery | `BIGQUERY_PROJECT_ID` |
| DuckDB | None required (defaults to in-memory); optional: `DUCKDB_DATABASE_PATH`, `MOTHERDUCK_TOKEN` |
| Databricks | `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_ACCESS_TOKEN` |

---

### 2. list_schemas

List available schemas from the connected database.

**Parameters:** None

**Returns:** Array of schema name strings.

**Key Features:**
- Requires `connect_database` to be called first
- Useful for multi-schema databases to identify which schema to analyze

---

### 3. reset_cache

Clear cached schema and/or ontology data to force re-analysis.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `cache_type` | string | No | `null` | Type of cache to reset: `"schema"`, `"ontology"`, or `"all"`. Omitting it (null) is treated as `"all"`. |

**Returns:** Dictionary with `status`, `cleared_caches` (list of cleared types), `message`, and `next_steps`.

**Key Features:**
- Resetting `schema` clears cached table metadata, schema file, and R2RML file references
- Resetting `ontology` clears the ontology file, loaded ontology content, and OBQC validator
- Use this when the database schema has changed and you need fresh analysis

---

### 4. discover_schema

Analyze database schema and return table metadata with relationships. Automatically generates W3C R2RML mappings and triggers GraphRAG initialization in the background.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `schema_name` | string | No | Default schema | Schema to analyze |
| `lightweight` | boolean | No | `true` | If true, return minimal data (table names, FK relationships, fan-trap warnings). If false, return full schema with all column details. |

**Returns:** Dictionary containing:
- `table_count` -- number of tables found
- `tables` -- table summaries (lightweight) or full details
- `relationships` -- foreign key relationships between tables
- `fan_trap_warnings` -- tables with multiple FK relationships (potential data multiplication risk)
- `schema_file` -- path to saved schema JSON (full mode)
- `r2rml_file` -- path to generated R2RML mapping (full mode)
- `next_step` -- recommended next tool to call

**Key Features:**
- Requires `connect_database` first
- Results are cached for the session -- calling again returns cached data immediately
- Lightweight mode (default) saves significant tokens by returning only table names and relationships
- Use `get_table_details` to drill into specific tables after lightweight analysis
- Automatically generates R2RML mappings in full mode
- Auto-initializes GraphRAG in the background (configurable via `AUTO_GRAPHRAG` env var)
- Detects fan-trap risks: tables connecting to multiple other tables via foreign keys

---

### 5. get_table_details

Get detailed metadata for a single table, including all columns, data types, keys, and constraints.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `table_name` | string | Yes | Name of the table to analyze |
| `schema_name` | string | No | Schema containing the table |

**Returns:** Dictionary containing:
- `columns` -- array of column details (name, data type, nullability, key status, comments)
- `primary_keys` -- list of primary key columns
- `foreign_keys` -- list of foreign key relationships
- `row_count` -- approximate row count
- `comment` -- table-level comment if available

**Key Features:**
- Requires `connect_database` first
- Ideal companion to lightweight `discover_schema` -- get full details for specific tables only
- Returns foreign key targets with referenced table and column names

---

### 6. generate_ontology

Generate an RDF/OWL ontology from the database schema with `oba:` (OrionBelt Analytics) namespace annotations that link ontology classes directly to SQL tables and columns.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `schema_name` | string | No | Last analyzed schema | Schema to generate ontology for |
| `schema_info` | string | No | None | Pre-analyzed schema JSON (usually not needed -- uses cached schema automatically) |
| `base_uri` | string | No | `"http://example.com/ontology/"` | Base URI for the ontology namespace |
| `auto_persist` | boolean | No | `true` | Automatically store in Oxigraph RDF database |
| `graph_uri` | string | No | Auto-generated | Custom named graph URI for RDF storage |

**Returns:** Status message with ontology file path, table count, and (if auto-persisted) triple count and graph URI.

**Key Features:**
- Automatically uses cached schema from `discover_schema` -- no need to pass schema data
- Returns cached result if ontology was already generated this session
- Generates OWL classes for tables with `oba:tableName`, `oba:primaryKey` annotations
- Generates OWL ObjectProperties for relationships with `oba:sqlJoinCondition`
- Generates OWL DatatypeProperties for columns with `oba:columnName`, `oba:sqlDataType`
- Auto-persists to Oxigraph RDF store for SPARQL querying (when `auto_persist` is true)
- Analyzes generated names and reports how many may need semantic review
- Saves ontology as `.ttl` (Turtle) file in connection-scoped output directory

---

### 7. suggest_semantic_names

Extract and analyze names from a generated ontology to identify abbreviations, cryptic identifiers, and names that would benefit from human-readable alternatives.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ontology_file` | string | No | Ontology filename (auto-detected from session if not provided) |

**Returns:** Dictionary containing:
- `classes_for_review` -- table-level names that appear cryptic
- `properties_for_review` -- column-level names needing improvement
- `relationships_for_review` -- relationship names to clarify
- `summary` -- counts of items needing review
- `instructions` -- guidance on how to provide better names

**Key Features:**
- Uses the cached ontology from `generate_ontology` automatically
- Identifies abbreviated names (e.g., `acctbal` -> `AccountBalance`)
- Provides the foundation for the `apply_semantic_names` step
- Does not modify the ontology -- only extracts names for review
- When the client supports MCP sampling (and `ENABLE_SAMPLING=true`), the response also pre-fills a `suggestions` dict via the host LLM, ready to pass straight to `apply_semantic_names`

---

### 8. apply_semantic_names

Apply LLM-suggested semantic name improvements to an existing ontology, replacing cryptic identifiers with business-friendly labels.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `suggestions` | string | Yes | -- | JSON object with `classes`, `properties`, and/or `relationships` arrays |
| `ontology_file` | string | No | Auto-detected | Ontology filename to update |
| `save_to_file` | boolean | No | `true` | Whether to save the updated ontology |

The `suggestions` parameter expects a JSON structure like:

```json
{
  "classes": [
    {
      "original_name": "acctbal",
      "suggested_name": "AccountBalance",
      "description": "Account balance records"
    }
  ],
  "properties": [
    {
      "original_name": "bankid",
      "table_name": "acctbal",
      "suggested_name": "Bank Identifier"
    }
  ],
  "relationships": [
    {
      "original_name": "acctbal_to_banks",
      "suggested_name": "Account Bank Relationship"
    }
  ]
}
```

**Returns:** Status message confirming applied changes.

**Key Features:**
- Updates `rdfs:label` annotations on OWL classes, properties, and relationships
- Optionally re-persists updated ontology to Oxigraph RDF store
- Saves updated `.ttl` file to the connection-scoped output directory

---

### 9. load_my_ontology

Load a custom `.ttl` (Turtle) ontology, either from inline content or from the import folder, bypassing the automated generation pipeline.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `import_folder` | string | No | `"./import"` | Path to the folder containing `.ttl` files (used only when `ontology_content` is not provided) |
| `auto_persist` | boolean | No | `true` | Store in Oxigraph RDF database |
| `graph_uri` | string | No | Auto-generated | Custom graph URI for RDF storage |
| `ontology_content` | string | No | None | TTL content passed directly (e.g. a `.ttl` file dropped into the chat) |
| `file_name` | string | No | None | Original file name to associate with `ontology_content` |

**Returns:** Dictionary with ontology information including class count, property count, and storage status.

**Key Features:**
- Accepts inline TTL via `ontology_content`, or reads the newest `.ttl` file in `import_folder` when no content is passed
- Enables OBQC (Ontology Basic Quality Criteria) validation for subsequent SQL queries
- Useful for loading externally curated or hand-crafted ontologies
- Supports the same auto-persist workflow as `generate_ontology`

---

### 10. download_artifact

Download a generated artifact -- the ontology or the R2RML mapping -- as a Turtle (`.ttl`) file with its full content.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `artifact_type` | string | Yes | -- | `"ontology"` or `"r2rml"` |
| `schema_name` | string | No | Last analyzed schema | Schema the artifact belongs to |
| `source` | string | No | `"rdf"` | Where to read the ontology from: `"rdf"` (Oxigraph store) or `"file"` (tmp folder). Applies to ontologies only; R2RML is always read from file. |

**Returns:** Dictionary containing:
- `success` -- boolean result
- `content` -- the artifact's full TTL text
- `file_path` / `file_name` / `file_size` -- saved file location and size
- `source` -- where the ontology was read from (`"rdf"` or `"file"`; ontology only)
- `triple_count`, `graph_uri` -- included when exporting an ontology from the RDF store
- `base_iri`, `schema_name`, `usage_examples` -- included for R2RML downloads
- On failure: `error`, `error_type`, and (often) a `hint`

**Key Features:**
- Use `artifact_type="ontology"` to retrieve the generated/loaded ontology (replaces the older standalone download tool)
- Use `artifact_type="r2rml"` to retrieve the W3C R2RML mapping generated by `discover_schema`
- Ontology source defaults to the RDF store; fall back to `source="file"` if the store export fails
- Intended for backups, version control, and importing into external RDF tooling (Protégé, Ontop, D2RQ)

---

### 11. sample_table_data

Safely sample rows from a specific table for data exploration and quality assessment.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `table_name` | string | Yes | -- | Name of the table to sample |
| `schema_name` | string | No | Default schema | Schema containing the table |
| `limit` | integer | No | `10` | Maximum rows to return (max: 100) |

**Returns:** Array of row dictionaries.

**Key Features:**
- Requires `connect_database` first
- Enforces a maximum of 100 rows for safety
- Invalid or out-of-range limits are silently corrected to 10
- Useful for understanding data format before writing queries

---

### 12. execute_sql_query

Execute a SQL query with built-in validation, fan-trap protection, and automatic GraphRAG context enrichment.

> **Note:** There is no separate validation tool. `execute_sql_query` runs structural validation, SQL-injection checks, and ontology-aware semantic checks (OBQC, including fan-trap detection) automatically before executing. Queries that fail validation are rejected with `error`, `error_type`, `warnings`, and `suggestions` fields rather than being run.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sql_query` | string | Yes | -- | SQL SELECT statement with fully qualified identifiers (`schema.table.column`) |
| `limit` | integer | No | `1000` | Maximum rows to return (max: 5,000) |
| `checklist_completed` | boolean | No | `false` | Confirmation that the pre-execution checklist has been completed |
| `query_intent` | string | No | Auto-extracted | Natural language description of what the query aims to retrieve |
| `allow_fan_out` | boolean | No | `false` | Execute even when OBQC detects a fan-trap. The finding is still reported, as a warning rather than a blocking error. |

**Returns:** Dictionary containing:
- `success` -- boolean execution result
- `columns` -- array of column names
- `rows` -- array of result rows
- `row_count` -- number of rows returned
- `execution_time_ms` -- query execution time in milliseconds
- `obqc_fan_trap` -- `{detected, blocking, findings}`, present on every response
- `next_tool` -- suggests `generate_chart` when results contain data

**Key Features:**
- Requires `connect_database` first
- **Pre-execution checklist must be confirmed** (`checklist_completed: true`) or the query is rejected
- Read-only enforcement -- only SELECT statements and schema introspection queries are allowed
- SQL injection prevention
- Query timeout protection
- Result size capped at 5,000 rows
- Automatically retrieves GraphRAG context for relevant tables when available
- Fan-trap detection **blocks** queries whose aggregates read across a 1:many join (a single such join is enough); `allow_fan_out: true` runs one anyway
- `query_intent` enables better GraphRAG context retrieval; if omitted, intent is auto-extracted from the SQL

---

### 13. generate_chart

Generate interactive Plotly charts rendered via MCP Apps, or export as static PNG images.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data_source` | array | Yes | -- | JSON array of objects, e.g., `[{"name": "A", "value": 10}]`. Pass as an array, not a string. |
| `chart_type` | string | Yes | -- | Chart type: `bar`, `line`, `scatter`, or `heatmap` |
| `x_column` | string | Yes | -- | Column name for the X-axis |
| `y_column` | string or array | No | None | Column name(s) for the Y-axis. Pass an array for multi-series charts. |
| `color_column` | string | No | None | Column for grouping/coloring. For heatmaps: the numeric value column for color intensity. |
| `title` | string | No | None | Chart title |
| `chart_style` | string | No | `"grouped"` | Layout style for bar charts: `stacked` or `grouped` |
| `sort_by` | string | No | None | Column to sort by (auto-sorted per chart type if omitted) |
| `sort_order` | string | No | None | Sort direction: `ascending` or `descending` |
| `output_format` | string | No | `"interactive"` | `"interactive"` (renders via MCP Apps) or `"image"` (saves PNG file) |

**Returns:**
- Interactive mode: `"Chart generated: ui://orionbelt/chart/<uuid>"` -- the chart is registered as a dynamic MCP Apps resource
- Image mode: `"Chart saved to: <file_path>"` -- path to the saved PNG file

**Key Features:**
- Interactive charts are rendered via FastMCP Apps as self-contained HTML with Plotly.js; they are responsive and size to their container (no width/height parameters)
- PNG export uses Kaleido for server-side rendering at a fixed 800x600
- Supports multi-series Y-axis by passing an array of column names
- Heatmap charts use `x_column` for X-axis, `y_column` for Y-axis, and `color_column` for cell values
- Heatmap axes are sorted by ordinal order for weekdays and time-of-day categories
- Charts are saved to the connection-scoped output directory

---

### 14. cleanup_workspace

Delete all workspace files for the current database connection and clear session state. The database connection remains active.

**Parameters:** None

**Returns:** Markdown-formatted summary of what was removed.

**Key Features:**
- Removes the workspace directory (`tmp/{connection_id}/`), Oxigraph RDF store, and ChromaDB vector store
- Clears all in-memory session state (schema cache, ontology, GraphRAG, RDF store)
- Database connection stays active -- call `discover_schema()` to start fresh
- Safe: only affects the current connection's workspace, not other sessions

---

### 15. save_semantic_model

Save a semantic model definition (e.g., OBML YAML) to the workspace for reuse across sessions.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_yaml` | string | Yes | -- | The model definition in YAML format |
| `model_name` | string | Yes | -- | Name to identify this model (e.g., "sales_analytics") |
| `schema_name` | string | No | Auto-detected | Database schema this model is based on |

**Returns:** Dictionary with `success`, `model_name`, `schema_name`, `file`, and `message`.

**Key Features:**
- Stores model YAML in `tmp/{connection_id}/models/{name}.yaml`
- Tracks models in workspace metadata for auto-restore discovery
- Model content is treated as opaque -- no parsing or validation of the YAML structure
- Enables cross-session model persistence for use with external Semantic Layer tools

---

### 16. get_semantic_model

Retrieve a stored semantic model YAML by name.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | string | Yes | -- | Name of the model to retrieve |

**Returns:** Dictionary with `success`, `model_name`, `schema_name`, `saved_at`, and `model_yaml`.

**Key Features:**
- Returns the full YAML content of a previously saved model
- Use this to pass model content to a Semantic Layer's `load_model` tool

---

### 17. list_semantic_models

List all stored semantic models for the current database connection.

**Parameters:** None

**Returns:** Dictionary with `models` array (each entry has `model_name`, `schema_name`, `saved_at`) and `count`.

---

### 18. graphrag_search

Search the schema using natural language via GraphRAG semantic search, or return a schema overview. GraphRAG is auto-initialized by `discover_schema`.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Conditional | None | Natural language search query. Required unless `overview=true`. |
| `top_k` | integer | No | `5` | Number of results to return |
| `element_type` | string | No | None | Filter results by type: `"table"`, `"column"`, or `"relationship"` |
| `overview` | boolean | No | `false` | If true, return schema statistics and community clustering instead of search results |

**Returns:**
- Search mode: `success`, `query`, `result_count`, `results`
- Overview mode (`overview=true`): `success`, `overview` (schema statistics and communities)

**Key Features:**
- Requires GraphRAG to be initialized -- call `discover_schema` first
- Returns a `graphrag_not_initialized` error if GraphRAG is unavailable
- `query` is required when `overview=false`; otherwise a `parameter_error` is returned

---

### 19. graphrag_query_context

Get an optimized, minimal schema context for SQL generation, selecting only the tables and columns relevant to a natural-language query.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | -- | Natural language description of what you want to query |
| `max_tables` | integer | No | `5` | Maximum tables to include in the context |
| `max_columns` | integer | No | `20` | Maximum columns to include in the context |

**Returns:** Dictionary containing:
- `success` -- boolean result
- `query` -- the original query
- `context` -- relevant tables, columns, relationships, and a `token_estimate`
- `usage_guidance` -- note on how to apply the context

**Key Features:**
- Requires GraphRAG initialization (`discover_schema` first)
- Reduces SQL-generation token usage by an estimated 85-95% versus passing the full schema
- Ideal precursor to `execute_sql_query` for large schemas

---

### 20. graphrag_find_join_path

Discover a join path between two tables using GraphRAG graph traversal.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `from_table` | string | Yes | -- | Source table name |
| `to_table` | string | Yes | -- | Target table name |
| `max_hops` | integer | No | `3` | Maximum number of joins allowed in the path |

**Returns:**
- On success: `success`, `from`, `to`, `hops`, `path` (ordered table list), and `joins` (per-hop join specifications)
- When no path is found: `success: false`, `from`, `to`, and a `message`

**Key Features:**
- Requires GraphRAG initialization (`discover_schema` first)
- Helps construct multi-table joins without manually reasoning over foreign keys
- Returns the concrete join conditions for each hop

---

### 21. store_ontology_in_rdf

Persist the current session's ontology in the Oxigraph RDF store so it can be queried with SPARQL. Usually unnecessary -- `generate_ontology` auto-persists by default -- but useful after loading or editing an ontology with auto-persist disabled.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `schema_name` | string | No | Last analyzed schema | Schema whose ontology to store |
| `graph_uri` | string | No | Auto-generated | Named graph URI for the triples |

**Returns:** Status message string with the schema, graph URI, and triple count.

**Key Features:**
- Requires an ontology to have been generated (`generate_ontology`) first
- Requires `pyoxigraph` to be installed
- After storing, query the graph with `query_sparql`

---

### 22. query_sparql

Execute a SPARQL query against the RDF ontology store to explore classes, properties, relationships, and semantic metadata.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sparql_query` | string | Yes | -- | A complete SPARQL query (`SELECT`, `ASK`, or `CONSTRUCT` -- auto-detected) |
| `timeout_seconds` | integer | No | `30` | Best-effort query timeout in seconds (applies to `SELECT`). Unblocks the caller after the timeout with an error; the underlying query may keep running in the background, since pyoxigraph exposes no native query cancellation. |

**Returns:** Dictionary containing `success`, `query_type`, the echoed `query`, and:
- `SELECT`: `result_count` and `results` (variable bindings)
- `ASK`: `result` (boolean)
- `CONSTRUCT`: `result` (Turtle string)

**Key Features:**
- Requires an ontology to be loaded (`generate_ontology` or `load_my_ontology`) and `pyoxigraph` installed
- Common prefixes (`rdf`, `rdfs`, `owl`, `xsd`) are available by default; the `oba:` namespace is `https://ralforion.com/ns/oba#`
- Query type is auto-detected from the query string -- no separate parameter

---

### 23. add_rdf_knowledge

Add a custom triple (subject-predicate-object) to the RDF store to enrich the ontology with bespoke metadata.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | Yes | Subject URI |
| `predicate` | string | Yes | Predicate URI |
| `object` | string | Yes | Object value (literal or URI) |
| `metadata` | object | No | Optional metadata dictionary |

**Returns:** Confirmation message string echoing the added triple.

**Key Features:**
- Requires the Oxigraph store to be initialized and `pyoxigraph` installed
- Useful for layering business annotations onto a generated ontology
- Added triples are queryable via `query_sparql`

---

### 24. cleanup_old_versions

Prune old ontology and GraphRAG versions for one schema according to the retention policy. Unlike `cleanup_workspace`, this keeps the current generation and recent history -- it deletes only *archived* versions that have aged out.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `schema_name` | string | No | Last analyzed | Schema whose version history to prune |
| `dry_run` | boolean | No | `true` | Report what would be deleted without deleting it |

**Returns:** Dictionary with `schema`, `dry_run`, `retention_policy`, per-area `graphrag` and `ontology` reports, and `versions` -- the schema's remaining history.

**Key Features:**
- Defaults to a dry run, so the first call is always safe to make
- A version must exceed both the keep-count and the age threshold; at least 2 are always kept
- Retention comes from `GRAPHRAG_KEEP_VERSIONS`, `GRAPHRAG_MAX_AGE_DAYS`, `ONTOLOGY_KEEP_VERSIONS`, `ONTOLOGY_MAX_AGE_DAYS` -- see [Configuration](configuration.md#per-version-retention)
- Named graphs and ChromaDB collections are only deleted when no surviving version still references them, since generations share both

---

> **Note:** Server metadata (name, version, supported databases, capabilities) is provided
> automatically via the MCP `initialize` handshake and the server `instructions`, and the live
> tool list via `tools/list` -- so no dedicated `get_server_info` tool is needed.

---

## Security Model

All tools operate within these security constraints:

- **Read-only SQL** -- only SELECT statements and schema introspection queries are permitted
- **SQL injection prevention** -- queries are scanned for injection patterns before execution
- **Query timeout protection** -- queries honor a configurable timeout. For SPARQL this is best-effort: the caller is released when the timeout elapses, but the underlying query may keep running in the background (pyoxigraph exposes no native query cancellation), so a timeout bounds caller latency, not server CPU
- **Result size limits** -- maximum 5,000 rows per query
- **Credential isolation** -- database credentials are read from environment variables, never passed as tool parameters
- **Session isolation** -- each MCP session maintains independent state (connections, caches, artifacts)
- **Idle session eviction** -- sessions are automatically cleaned up after a configurable idle timeout

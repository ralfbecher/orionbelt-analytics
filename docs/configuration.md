[<- Back to README](../README.md)

# Configuration Reference

This document provides the full configuration reference for OrionBelt® Analytics, covering all environment variables, transport modes, and per-database troubleshooting guidance.

## Environment Variables

OrionBelt Analytics is configured via a `.env` file in the project root. Copy the template to get started:

```bash
cp .env.template .env
```

All database parameters are optional when calling `connect_database` -- the server falls back to `.env` values when tool parameters are not provided.

### Full `.env` Reference

```env
# =================================================================
# OrionBelt Analytics Configuration
# =================================================================

# Server Configuration
# -------------------
# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Automatic Infrastructure Management
# --------------------------------------
# Phase 1: Auto-initialize GraphRAG in background when schema is analyzed
# Set to false to disable automatic initialization
AUTO_GRAPHRAG=true

# Phase 2: Auto-generate ontology in background after GraphRAG completes
# Conservative default (false) - enable after testing
# When enabled: ontology is automatically generated and stored in Oxigraph RDF store
AUTO_ONTOLOGY=false

# Superseded artifact pruning
# Artifact filenames carry a timestamp, so every generate_ontology /
# discover_schema writes a NEW file while metadata records only the latest.
# This is how many generations of each artifact (per connection, per schema)
# to keep; older ones are deleted as soon as a new one is written. Minimum 1
# -- the file currently referenced is never a deletion candidate.
ARTIFACT_KEEP_VERSIONS=3

# Startup Workspace Cleanup
# -------------------------
# Runs on every startup regardless of the setting below:
#   - stale loose files directly under the output dir are deleted
#   - each workspace's charts/ directory is deleted (chart images are ephemeral)
#
# AUTO_CLEANUP_ON_STARTUP then controls whether whole workspaces are deleted.
# See "Startup workspace cleanup" below for the exact semantics.
AUTO_CLEANUP_ON_STARTUP=false

# Age threshold for AUTO_CLEANUP_ON_STARTUP=true, measured from the workspace's
# last update (not file mtime). Ignored for the false and all modes.
WORKSPACE_MAX_AGE_DAYS=30

# MCP Sampling
# ------------
# Allow the server to call back through the client's LLM (MCP sampling) for
# tasks like generating semantic-rename suggestions inside suggest_semantic_names.
# Requires a sampling-capable client (e.g. OrionBelt Chat). Clients without
# sampling support (e.g. Claude Desktop) silently fall back to the legacy
# manual-review path. Set to false to force the legacy path everywhere.
ENABLE_SAMPLING=true

# MCP Transport Configuration
# Options: http, sse (Server-Sent Events)
# - http: Standard HTTP transport (streamable, default)
# - sse: Server-Sent Events for (legacy)
MCP_TRANSPORT=http

# MCP Server configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=9000

# Shutdown timeout for graceful connection closure (seconds)
# Lower values = faster shutdown but may interrupt active requests
# Higher values = cleaner shutdown but slower Ctrl+C response
MCP_SHUTDOWN_TIMEOUT=2

# Session Idle Timeout
# Sessions idle longer than this are automatically evicted (resources freed).
# Set to 0 to disable idle eviction entirely.
SESSION_IDLE_TIMEOUT_SECONDS=1800
# How often to scan for idle sessions (seconds).
SESSION_SCAN_INTERVAL_SECONDS=60

# Master password for encrypting database credentials in memory
# This should be a strong, unique password for your deployment
# If not set, credentials will be stored in plain text (NOT recommended for production)
MCP_MASTER_PASSWORD=MySecurePassword123!@

# Ontology settings
ONTOLOGY_BASE_URI=http://example.com/ontology/

# R2RML Mapping settings
# Base IRI for R2RML subject templates (schema name will be appended)
R2RML_BASE_IRI=http://mycompany.com/

# Output directory for generated files (schema JSON, ontology TTL, R2RML, etc.)
# Relative to project root. Default: tmp
#
# PERSISTENCE NOTE:
# - Workspace data (schema, ontology, GraphRAG, semantic models) persists across server restarts
# - Chart images are cleaned on each restart (ephemeral)
# - GraphRAG vector stores: OUTPUT_DIR/chromadb/{connection_id}/
# - RDF ontology stores: OUTPUT_DIR/oxigraph/{connection_id}/store/
# - Semantic models: OUTPUT_DIR/{connection_id}/models/
# - The default tmp/ directory is NOT persistent across deployments or container rebuilds
#
# For production deployments:
# - Use a persistent directory (e.g., /var/lib/orionbelt, /data/orionbelt)
# - Mount as a volume in containerized environments
# - Ensure proper backup of OUTPUT_DIR
# - Consider AUTO_CLEANUP_ON_STARTUP / WORKSPACE_MAX_AGE_DAYS (see above)
#
OUTPUT_DIR=tmp

# -----------------------------------------------------------------
# PostgreSQL Configuration
# -----------------------------------------------------------------
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=mydb
POSTGRES_USERNAME=user
POSTGRES_PASSWORD=password

# -----------------------------------------------------------------
# MySQL Configuration
# -----------------------------------------------------------------
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=mydb
MYSQL_USERNAME=root
MYSQL_PASSWORD=your_password
# Optional: Character set (default: utf8mb4 for full Unicode support)
MYSQL_CHARSET=utf8mb4

# -----------------------------------------------------------------
# Snowflake Configuration
# -----------------------------------------------------------------
SNOWFLAKE_ACCOUNT=your-account         # e.g. CLYKFLK-KA74251
SNOWFLAKE_USERNAME=user
SNOWFLAKE_PASSWORD=password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=MYDB
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_ROLE=PUBLIC

# -----------------------------------------------------------------
# ClickHouse Configuration
# -----------------------------------------------------------------
# NOTE: ClickHouse has no foreign key constraints. PRIMARY KEY is a
# sparse index (not a uniqueness constraint) and ORDER BY defines the
# physical sort order on disk. OrionBelt Analytics handles these
# differences transparently via the ClickHouse driver.
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_DATABASE=default
CLICKHOUSE_USERNAME=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_PROTOCOL=http
CLICKHOUSE_SECURE=false

# -----------------------------------------------------------------
# Dremio Configuration
# -----------------------------------------------------------------
DREMIO_HOST=localhost
DREMIO_PORT=31010
DREMIO_USERNAME=your_username
DREMIO_PASSWORD=your_password

# -----------------------------------------------------------------
# BigQuery Configuration
# -----------------------------------------------------------------
BIGQUERY_PROJECT_ID=your-gcp-project-id
BIGQUERY_DATASET=your_dataset
BIGQUERY_CREDENTIALS_PATH=/path/to/service-account-key.json
# Alternatively, use GOOGLE_APPLICATION_CREDENTIALS environment variable

# -----------------------------------------------------------------
# DuckDB/MotherDuck Configuration
# -----------------------------------------------------------------
DUCKDB_DATABASE_PATH=:memory:  # or /path/to/file.db
# For MotherDuck cloud:
# DUCKDB_DATABASE_PATH=md:your_database
# MOTHERDUCK_TOKEN=your_motherduck_token

# -----------------------------------------------------------------
# Databricks SQL Configuration
# -----------------------------------------------------------------
DATABRICKS_SERVER_HOSTNAME=your-workspace.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your_warehouse_id
DATABRICKS_ACCESS_TOKEN=your_access_token
DATABRICKS_CATALOG=hive_metastore
DATABRICKS_SCHEMA=default
```

### Variable Reference Table

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `AUTO_GRAPHRAG` | `true` | Auto-initialize GraphRAG when schema is analyzed |
| `AUTO_ONTOLOGY` | `false` | Auto-generate ontology after GraphRAG completes |
| `ARTIFACT_KEEP_VERSIONS` | `3` | Generations of each ontology / schema / R2RML file to keep per schema. Older ones are pruned when a new one is written. Minimum 1 |
| `AUTO_CLEANUP_ON_STARTUP` | `false` | Delete whole workspaces at startup: `false` (keep all), `true` (orphaned, or older than `WORKSPACE_MAX_AGE_DAYS`), `all` (delete every workspace). See [Startup workspace cleanup](#startup-workspace-cleanup) |
| `WORKSPACE_MAX_AGE_DAYS` | `30` | Age threshold for `AUTO_CLEANUP_ON_STARTUP=true`, measured from the workspace's last update. Ignored in the other modes |
| `GRAPHRAG_KEEP_VERSIONS` | `3` | Archived GraphRAG snapshots kept per schema. See [Per-version retention](#per-version-retention) |
| `GRAPHRAG_MAX_AGE_DAYS` | `30` | Age past which an archived GraphRAG snapshot may be deleted |
| `ONTOLOGY_KEEP_VERSIONS` | `5` | Archived ontology versions kept per schema |
| `ONTOLOGY_MAX_AGE_DAYS` | `60` | Age past which an archived ontology version may be deleted |
| `ONTOLOGY_BASE_URI` | `http://example.com/ontology/` | Base URI for generated RDF ontologies |
| `R2RML_BASE_IRI` | `http://mycompany.com/` | Base IRI for R2RML subject templates |
| `OUTPUT_DIR` | `tmp` | Directory for generated files (relative to project root) |
| `MCP_TRANSPORT` | `http` | MCP transport mode: `http` or `sse` |
| `MCP_SERVER_HOST` | `localhost` | Host address the server binds to |
| `MCP_SERVER_PORT` | `9000` | Port the server listens on |
| `MCP_SHUTDOWN_TIMEOUT` | `2` | Seconds to wait for graceful shutdown |
| `SESSION_IDLE_TIMEOUT_SECONDS` | `1800` | Idle timeout before session eviction (0 to disable) |
| `SESSION_SCAN_INTERVAL_SECONDS` | `60` | How often to scan for idle sessions |
| `MCP_MASTER_PASSWORD` | *(unset)* | Master password for encrypting credentials in memory |

### Startup workspace cleanup

`AUTO_CLEANUP_ON_STARTUP` controls how much of the output directory survives a restart. It operates on **whole workspaces** -- one per database connection, under `OUTPUT_DIR/{connection_id}/` -- and honours a custom `OUTPUT_DIR`.

**Always runs, whatever the setting is:**

- stale loose files sitting directly in `OUTPUT_DIR` (not inside a connection directory) are deleted
- every workspace's `charts/` directory is deleted -- chart images are ephemeral and are regenerated on demand

Workspace data itself is only touched by the modes below.

| Mode | What is deleted |
|------|-----------------|
| `false` *(default)* | Nothing beyond the two steps above. Every workspace is kept, so auto-restore works after a restart. |
| `true` | A workspace is deleted if it has **no `metadata.json`** (orphaned directory), or if its recorded `workspace.updated_at` is older than `WORKSPACE_MAX_AGE_DAYS`. A workspace whose metadata exists but has no `updated_at` is **kept**. |
| `all` | Every workspace directory, unconditionally -- a full fresh start. |

**Deletion is all-or-nothing per connection.** Removing a workspace removes its schema cache, ontology TTL, R2RML mapping, saved semantic models, *and* the satellite stores held outside the workspace directory:

```
OUTPUT_DIR/{connection_id}/            # workspace: metadata.json, schema, ontology, models/
OUTPUT_DIR/chromadb/{connection_id}/   # GraphRAG vectors
OUTPUT_DIR/oxigraph/{connection_id}/   # RDF / SPARQL store
```

There is no partial or per-artifact cleanup. The `chromadb/` and `oxigraph/` parent directories are shared across connections and are never themselves treated as workspaces.

**Age comes from metadata, not the filesystem.** The `true` mode reads `workspace.updated_at` out of `metadata.json` rather than looking at file modification times, so touching files on disk does not keep a workspace alive.

### Per-version retention

Startup cleanup deletes whole workspaces; this deletes individual **versions** inside a workspace that is still in use.

Every `discover_schema` call opens a version for that schema, and `generate_ontology` and GraphRAG initialization fill in their halves as they complete. The result is a history under `schemas.{name}.versions` in `metadata.json` recording, per generation: the schema fingerprint and table/column counts, the ontology TTL file, its named graph and triple count, and the GraphRAG vector count and snapshot files. Opening a new version archives the one before it -- only the newest is `active`.

Retention applies to **archived** versions only, so the current generation is never a candidate:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPHRAG_KEEP_VERSIONS` | `3` | Archived GraphRAG snapshots to keep per schema |
| `GRAPHRAG_MAX_AGE_DAYS` | `30` | Age past which an archived GraphRAG snapshot is eligible for deletion |
| `ONTOLOGY_KEEP_VERSIONS` | `5` | Archived ontology versions to keep per schema |
| `ONTOLOGY_MAX_AGE_DAYS` | `60` | Age past which an archived ontology version is eligible for deletion |

A version must exceed **both** the count and the age threshold to be deleted, and at least `min_versions` (2) are always kept regardless. These are read at cleanup time and override the copy recorded in `metadata.json`, so changing them takes effect on workspaces that already exist. Invalid or sub-1 values are logged and ignored rather than failing startup.

Cleanup is **manual**: call the `cleanup_old_versions` tool, which defaults to `dry_run=true` so you can see what would go before anything does. It reports the effective policy, what was (or would be) deleted, and the schema's remaining history.

Two deletions are deliberately guarded rather than unconditional:

- **Named graphs.** Successive generations of a schema reuse the same graph URI, so the graph is only dropped from Oxigraph when no surviving version still references it.
- **ChromaDB collections.** GraphRAG is connection-scoped and accumulative by design -- one collection holds every schema's vectors so cross-schema search and join discovery work -- so versions share a collection rather than each owning one. The collection is deleted only when no surviving version references it, which in practice means it stays as long as the schema has a live version. Per-version GraphRAG *files* are pruned normally.

Ontology, schema and R2RML **files** are additionally pruned by count as they are written, independently of this, via `ARTIFACT_KEEP_VERSIONS`.

### Security Notes

- **Master password**: Used to encrypt database credentials in memory via AES-128-CBC with HMAC. Creates a persistent salt file at `~/.mcp_credential_salt`. Without this setting, credentials are stored in plain text.
- **File permissions**: Restrict `.env` to owner-only access: `chmod 600 .env`
- **Version control**: Never commit `.env` -- add it to `.gitignore`.
- **Production**: Consider using environment variables directly, or a secrets management service (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault).
- **Credential rotation**: Implement rotation policies for database passwords and API tokens.

---

## MCP Sampling

`ENABLE_SAMPLING` (default `true`) controls whether the server is allowed to call back through the client's LLM via the MCP `sampling/createMessage` capability.

**Where it is used:** `suggest_semantic_names`. When sampling is available, the server asks the host LLM to produce rename suggestions for cryptic identifiers and returns them as a `suggestions: {old_name: new_name}` map alongside the existing cryptic-name lists. Without sampling, the response shape is unchanged: the host LLM is expected to inspect the cryptic lists and call `apply_semantic_names` with its own suggestions.

**Capability detection is implicit.** The server attempts `ctx.sample(...)` and falls back to the legacy path on any failure — including the case where the client never advertised the capability. There is no separate handshake to configure.

**Client compatibility:**

| Client | Sampling support | Behaviour |
|---|---|---|
| OrionBelt Chat | Yes (with `sampling.tools`) | `suggestions` field is populated; one tool call instead of two |
| Claude Desktop | No | Falls back silently to manual review path |
| Claude Code | No | Falls back silently to manual review path |
| Generic pydantic-ai clients | Depends on agent wiring | Works when `agent.set_mcp_sampling_model()` (or equivalent) is called |

**Disabling:** set `ENABLE_SAMPLING=false` to force the legacy path even when the client supports sampling. Useful for cost control, deterministic regression testing, or when a particular host LLM produces poor rename suggestions.

**Logging:** sampling activity is logged at INFO/WARNING with elapsed time and item counts -- look for lines starting with `MCP sampling:` in the server log to verify the path the request took.

---

## Transport Configuration

The server supports two MCP transport modes, configured via `MCP_TRANSPORT`:

### `http` (default, recommended)

Streamable HTTP transport for modern MCP clients. This is the standard transport for FastMCP servers and provides better performance and reliability.

```env
MCP_TRANSPORT=http
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=9000
```

**Claude Desktop** configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "OrionBelt-Analytics": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:9000/mcp",
        "--transport",
        "http-only"
      ]
    }
  }
}
```

### `sse` (legacy)

Server-Sent Events transport for backward compatibility with older MCP clients. Use this mode for LibreChat integration or other clients that do not support streamable HTTP.

```env
MCP_TRANSPORT=sse
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=9000
```

**LibreChat** configuration (`librechat.yaml`):

```yaml
mcpServers:
  OrionBelt-Analytics:
    url: "http://host.docker.internal:9000/sse"
    timeout: 60000
    startup: true
```

**Note:** LibreChat requires SSE transport. Set `MCP_TRANSPORT=sse` before starting the server when using LibreChat.

### Transport Validation

The server automatically validates the `MCP_TRANSPORT` value on startup. If an invalid value is provided, it defaults to `http` and logs a warning.

---

## Troubleshooting

### PostgreSQL

- **Connection refused**: Verify the PostgreSQL server is running and accepting connections on the configured host and port.
- **Authentication failure**: Check that `POSTGRES_USERNAME` and `POSTGRES_PASSWORD` are correct and the user has `CONNECT` privilege on the target database.
- **Database does not exist**: Ensure `POSTGRES_DATABASE` refers to an existing database. Create it with `CREATE DATABASE mydb;` if needed.
- **SSL issues**: If the server requires SSL, make sure your connection string or driver configuration includes the appropriate SSL mode.
- **Firewall / network**: Confirm that the PostgreSQL port (default 5432) is open and reachable from the machine running OrionBelt Analytics.

### Snowflake

- **Account format**: The `SNOWFLAKE_ACCOUNT` value must match the account identifier shown in your Snowflake web UI URL. Common formats include:
  - `CLYKFLK-KA74251` (org-account)
  - `account.region` (e.g., `xy12345.us-east-1`)
  - `account.region.cloud` (e.g., `xy12345.us-east-1.aws`)
- **Role permissions**: Ensure your user has access to the specified `SNOWFLAKE_ROLE`. The role must have the necessary grants on the warehouse, database, and schema.
- **Warehouse**: The warehouse specified by `SNOWFLAKE_WAREHOUSE` must be running (not suspended) and accessible to the configured role.
- **Case sensitivity**: Snowflake uses UPPERCASE identifiers by default. Database, schema, and table names in `SNOWFLAKE_DATABASE` and `SNOWFLAKE_SCHEMA` should typically be uppercase unless they were created with double-quoted lowercase names.

### BigQuery

- **Authentication**: BigQuery requires a service account JSON key file. Set `BIGQUERY_CREDENTIALS_PATH` to the absolute path of the key file, or set the standard `GOOGLE_APPLICATION_CREDENTIALS` environment variable instead.
- **Project ID**: Find your project ID in the GCP Console under project settings. Set it via `BIGQUERY_PROJECT_ID`.
- **Dataset access**: The `BIGQUERY_DATASET` is optional and can be specified per query. When set, it limits schema analysis to that dataset.
- **Permissions**: Ensure the service account has at least the `BigQuery Data Viewer` role (`roles/bigquery.dataViewer`) for read access, and `BigQuery Job User` (`roles/bigquery.jobUser`) to run queries.
- **Billing**: BigQuery queries incur costs. Verify that billing is enabled on the GCP project.

### DuckDB / MotherDuck

- **Local file**: Use an absolute path for `DUCKDB_DATABASE_PATH` (e.g., `/data/analytics.db`). Use `:memory:` for a transient in-memory database.
- **MotherDuck cloud**: Prefix the database name with `md:` (e.g., `DUCKDB_DATABASE_PATH=md:my_database`) and provide your token via `MOTHERDUCK_TOKEN`.
- **Token**: Obtain your MotherDuck access token from the MotherDuck web UI under Settings > Access Tokens.
- **File locking**: DuckDB uses file-level locking. Only one process can write to a database file at a time. If you see lock errors, close other connections first.
- **Read-only mode**: Pass `read_only=true` when connecting if you only need to query data and want to avoid lock contention.

### Databricks SQL

- **Server hostname**: Use the workspace URL without the `https://` prefix (e.g., `your-workspace.cloud.databricks.com`). Find this in the Databricks workspace settings.
- **HTTP path**: Copy from the SQL Warehouse connection details page in the Databricks UI (e.g., `/sql/1.0/warehouses/your_warehouse_id`).
- **Access token**: Generate a personal access token in User Settings > Developer > Access Tokens. Tokens can be scoped to specific permissions.
- **Unity Catalog**: Set `DATABRICKS_CATALOG` to the Unity Catalog name (e.g., `main`). For legacy Hive metastore, use `hive_metastore`.
- **Schema**: Set `DATABRICKS_SCHEMA` to the target schema (default: `default`).
- **Warehouse state**: The SQL Warehouse must be running. Serverless warehouses start automatically; classic warehouses may need manual start.

### MySQL

- **Connection refused**: Verify that MySQL is running (`sudo systemctl status mysql` or `brew services list`).
- **Access denied**: Check username, password, and user privileges. Grant access with `GRANT ALL ON mydb.* TO 'user'@'%';` and `FLUSH PRIVILEGES;`.
- **Unknown database**: Ensure the database specified in `MYSQL_DATABASE` exists. Create it with `CREATE DATABASE mydb;`.
- **Character encoding**: Use `MYSQL_CHARSET=utf8mb4` (the default) for full Unicode support, including supplementary characters and emoji.
- **Connection timeouts**: Check firewall rules and network connectivity. For remote servers, confirm the MySQL `bind-address` allows external connections.
- **Too many connections**: Increase `max_connections` in the MySQL configuration or reduce the connection pool size.

### ClickHouse

- **No foreign keys**: ClickHouse does not support foreign key constraints. OrionBelt Analytics uses the ClickHouse driver, which queries `system.*` tables for schema metadata and handles the absence of FKs gracefully.
- **PRIMARY KEY vs ORDER BY**: In ClickHouse, `PRIMARY KEY` defines a sparse index (not a uniqueness constraint) and `ORDER BY` defines the physical sort order on disk. These are fundamentally different from RDBMS semantics.
- **Protocol**: Set `CLICKHOUSE_PROTOCOL` to `http` (default, port 8123) or `native` (port 9000). The HTTP interface is recommended for most use cases.
- **Secure connections**: Set `CLICKHOUSE_SECURE=true` when connecting to ClickHouse Cloud or any TLS-enabled instance.
- **Default credentials**: ClickHouse ships with `CLICKHOUSE_USERNAME=default` and an empty password. For production, create a dedicated user with appropriate permissions.

### Dremio

- **Host**: Use the Dremio coordinator node hostname or IP address.
- **Port**: The default PostgreSQL wire protocol port for Dremio is `31010`.
- **Protocol**: Dremio uses the PostgreSQL wire protocol; no additional drivers are needed beyond the standard PostgreSQL connector.
- **SSL**: SSL is enabled by default. Disable with `ssl=False` in the `connect_database` call if your Dremio instance does not use TLS.
- **Permissions**: Ensure your Dremio user has access to the target spaces, folders, and datasets.

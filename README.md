<!-- mcp-name: io.github.ralforion/orionbelt-analytics -->
<p align="center">
  <img src="https://raw.githubusercontent.com/ralforion/orionbelt-analytics/main/assets/ORIONBELT_Logo.png" alt="OrionBelt Logo" width="400">
</p>

<h1 align="center">OrionBelt® Analytics</h1>

<p align="center"><strong>The Ontology-based MCP server for your Text-2-SQL convenience.</strong></p>

[![Version 2.0.0](https://img.shields.io/badge/version-2.0.0-purple.svg)](https://github.com/ralforion/orionbelt-analytics/releases)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](https://github.com/ralforion/orionbelt-analytics/blob/main/LICENSE)
[![FastMCP](https://img.shields.io/badge/FastMCP-3.3.1+-blue)](https://github.com/jlowin/fastmcp)
[![RDF/OWL](https://img.shields.io/badge/RDF%2FOWL-Ontology-orange)](https://www.w3.org/OWL/)

[![BigQuery](https://img.shields.io/badge/BigQuery-669DF6.svg?logo=googlebigquery&logoColor=white)](https://cloud.google.com/bigquery)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8.svg?logo=snowflake&logoColor=white)](https://www.snowflake.com)
[![ClickHouse](https://img.shields.io/badge/ClickHouse-FFCC01.svg?logo=clickhouse&logoColor=black)](https://clickhouse.com)
[![Dremio](https://img.shields.io/badge/Dremio-31B48D.svg)](https://www.dremio.com)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621.svg?logo=databricks&logoColor=white)](https://www.databricks.com)
[![DuckDB](https://img.shields.io/badge/DuckDB-FFF000.svg?logo=duckdb&logoColor=black)](https://duckdb.org)
[![MySQL](https://img.shields.io/badge/MySQL-4479A1.svg?logo=mysql&logoColor=white)](https://www.mysql.com)

[![Docker Hub](https://img.shields.io/docker/v/ralforion/orionbelt-analytics?logo=docker&logoColor=white&label=Docker%20Hub&color=2496ED&sort=semver)](https://hub.docker.com/r/ralforion/orionbelt-analytics/tags)
[![Docker pulls](https://img.shields.io/docker/pulls/ralforion/orionbelt-analytics?logo=docker&logoColor=white&color=2496ED)](https://hub.docker.com/r/ralforion/orionbelt-analytics)
[![Image size](https://img.shields.io/docker/image-size/ralforion/orionbelt-analytics/latest?logo=docker&logoColor=white&color=2496ED)](https://hub.docker.com/r/ralforion/orionbelt-analytics)

OrionBelt Analytics is an MCP server that analyzes relational database schemas and generates RDF/OWL ontologies with embedded SQL mappings. It provides relationship-aware Text-to-SQL with automatic fan-trap prevention, GraphRAG for intelligent schema discovery, and interactive charting -- all accessible through any MCP-compatible AI client.

## The OrionBelt Ecosystem

| Project                                                                                    | Purpose                                                                                                           |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **OrionBelt Analytics** (this)                                                             | Schema analysis, ontology generation, GraphRAG, Text-to-SQL                                                       |
| [**OrionBelt Semantic Layer**](https://github.com/ralfbecher/orionbelt-semantic-layer)     | Declarative YAML models compiled into dialect-specific, fan-trap-free SQL                                         |
| [**OrionBelt Ontology Builder**](https://github.com/ralfbecher/orionbelt-ontology-builder) | Visual OWL ontology editor with reasoning and graph visualization ([live demo](https://orionbelt.streamlit.app/)) |
| [**OrionBelt Chat**](https://github.com/ralfbecher/orionbelt-chat)                         | AI chat UI for Analytics + Semantic Layer (Chainlit, multiple LLM providers)                                      |

Run Analytics and Semantic Layer side-by-side in Claude Desktop for schema-aware ontology generation **and** guaranteed-correct SQL compilation.

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/ralforion/orionbelt-analytics/main/assets/architecture.png" alt="OrionBelt Analytics Architecture" width="900">
</p>

- **8 database connectors** -- PostgreSQL, MySQL, Snowflake, ClickHouse, Dremio, BigQuery, DuckDB/MotherDuck, Databricks SQL
- **RDF/OWL ontology generation** with `oba:` namespace SQL annotations and W3C R2RML mappings
- **GraphRAG** -- graph traversal (up to 12 hops) + ChromaDB vector embeddings for semantic schema discovery
- **SPARQL 1.1** query interface via persistent Oxigraph RDF store
- **OBQC validation** -- deterministic SQL checks against the ontology (table/column existence, join validity, type mismatches, fan-traps)
- **Interactive charting** -- Plotly charts with MCP-UI rendering in Claude Desktop
- **Multi-schema support** -- analyze multiple schemas simultaneously; ontology and GraphRAG state are isolated per schema
- **Workspace persistence** -- reconnect to the same database and restore your previous session
- **MCP sampling** -- when the connected client supports sampling (e.g. [OrionBelt Chat](https://github.com/ralfbecher/orionbelt-chat)), `suggest_semantic_names` asks the host LLM to pre-fill rename suggestions for cryptic identifiers via `sampling/createMessage`, collapsing the previous review-then-apply flow into a single tool call. Clients without sampling support (e.g. Claude Desktop) silently fall back to the manual review path

## OBQC -- Ontology-Based Query Check

A key differentiator of OrionBelt is **OBQC** (Ontology-Based Query Check), a deterministic, rule-based SQL validator that catches errors _before_ queries reach the database. Unlike LLM-only approaches that rely on the model "getting it right," OBQC cross-references every generated SQL statement against the loaded RDF/OWL ontology to enforce structural correctness.

**What OBQC validates:**

| Check                       | What it catches                                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| **Table existence**         | References to tables that don't exist in the schema                                               |
| **Column existence**        | References to columns not present in their table, ambiguous unqualified columns                   |
| **Join validity**           | Missing join conditions (Cartesian products), join columns that don't match declared foreign keys |
| **Type compatibility**      | WHERE/ON comparisons between incompatible types (e.g. string vs. integer)                         |
| **Aggregation correctness** | SELECT columns missing from GROUP BY when aggregates are used                                     |
| **Fan-trap detection**      | Aggregations across multiple one-to-many joins that silently multiply results                     |

**How it works:**

1. `generate_ontology` or `load_my_ontology` creates/loads an ontology with `oba:` namespace annotations that map OWL classes and properties to actual database tables, columns, types, and foreign keys.
2. When `execute_sql_query` is called, OBQC parses the SQL with [sqlglot](https://github.com/tobymao/sqlglot) and validates every table, column, join, and aggregation against the ontology's schema model.
3. Issues are returned with severity levels (error, warning, info) alongside the query results, so the LLM can self-correct before the user sees wrong data.

OBQC is fully deterministic -- no LLM calls, no probabilistic reasoning. It acts as a safety net that complements the LLM's SQL generation with hard structural guarantees. Errors **block query execution**; warnings are attached to the response for the LLM to act on. See [OBQC documentation](docs/obqc.md) for the full rule reference, severity behavior, and annotation requirements.

## Quick Start

### 1. Install

```bash
git clone https://github.com/ralforion/orionbelt-analytics
cd orionbelt-analytics
uv sync
```

> Requires **Python 3.13+** and [**uv**](https://github.com/astral-sh/uv).

### 2. Configure

```bash
cp .env.template .env
```

Edit `.env` with your database credentials. At minimum, set the variables for one database (e.g. `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DATABASE`, `POSTGRES_USERNAME`, `POSTGRES_PASSWORD`).

See [docs/configuration.md](docs/configuration.md) for all environment variables, transport options, and troubleshooting.

### 3. Run

```bash
uv run server.py
```

The server starts on `http://localhost:9000` (HTTP transport, configurable via `MCP_SERVER_PORT`).

## Connect Your AI Client

### Claude Desktop

Start the server, then add to your `claude_desktop_config.json`:

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

### Claude Code

```bash
claude mcp add orionbelt-analytics http://localhost:9000/mcp
```

### LibreChat

Set `MCP_TRANSPORT=sse` in `.env`, restart the server, then add to `librechat.yaml`:

```yaml
mcpServers:
  OrionBelt-Analytics:
    url: "http://host.docker.internal:9000/sse"
    timeout: 60000
    startup: true
```

### Other Frameworks

OrionBelt works with LangChain, OpenAI Agents SDK, CrewAI, Google ADK, Vercel AI SDK, n8n, and ChatGPT Custom GPTs. See [docs/integrations.md](docs/integrations.md) for setup examples.

## Tools

OrionBelt exposes 26 MCP tools. Here is a summary by category:

### Connection & Schema

| Tool                | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| `connect_database`  | Connect to any supported database using `.env` credentials             |
| `list_schemas`      | List available schemas in the connected database                       |
| `reset_cache`       | Clear cached schema and ontology data for the current session          |
| `discover_schema`   | Analyze schema structure with automatic GraphRAG + ontology generation |
| `get_table_details` | Get detailed column, key, and constraint info for a specific table     |
| `cleanup_workspace` | Delete all workspace files for the current connection and start fresh  |

### Ontology & Semantic

| Tool                     | Description                                                           |
| ------------------------ | --------------------------------------------------------------------- |
| `generate_ontology`      | Generate RDF/OWL ontology from schema with SQL mapping annotations    |
| `suggest_semantic_names` | Detect abbreviations and cryptic names for business-friendly renaming |
| `apply_semantic_names`   | Apply LLM-suggested semantic names and descriptions to ontology       |
| `load_my_ontology`       | Load a custom `.ttl` ontology file from an import folder              |
| `download_artifact`      | Download ontology or R2RML mapping as a Turtle file                   |

### Query & Visualization

| Tool                | Description                                                                |
| ------------------- | -------------------------------------------------------------------------- |
| `sample_table_data` | Preview table data with row limit and injection protection                 |
| `execute_sql_query` | Execute SQL with OBQC validation, security checks, and fan-trap detection  |
| `generate_chart`    | Generate Plotly charts (bar, line, scatter, heatmap) with MCP-UI rendering |

### GraphRAG

| Tool                      | Description                                                               |
| ------------------------- | ------------------------------------------------------------------------- |
| `graphrag_search`         | Semantic search + schema overview (auto-initialized by `discover_schema`) |
| `graphrag_query_context`  | Get optimized context for SQL generation (85-95% token reduction)         |
| `graphrag_find_join_path` | Discover join paths between tables via graph traversal                    |
| `reachable_from`          | Dimension-capable tables for an anchor grain (many-to-one closure)        |
| `measurable_from`         | Measure-capable tables for an anchor grain (one-to-many closure)          |
| `plan_composite_query`    | Advise a fan-trap-safe Composite Fact Layer (UNION ALL) decomposition     |

### SPARQL & RDF

| Tool                    | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| `store_ontology_in_rdf` | Persist ontology in Oxigraph for SPARQL access                  |
| `query_sparql`          | Execute SPARQL queries (SELECT, ASK, CONSTRUCT — auto-detected) |
| `add_rdf_knowledge`     | Add custom metadata triples to the RDF store                    |

### Semantic Models

| Tool                   | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `save_semantic_model`  | Save a semantic model (e.g., OBML YAML) to the workspace   |
| `get_semantic_model`   | Retrieve a stored semantic model by name                   |
| `list_semantic_models` | List all stored semantic models for the current connection |

For full parameter details, return values, and examples, see [docs/tools-reference.md](docs/tools-reference.md).

## Typical Workflows

**Full analysis session:**

```
connect_database("postgresql") -> discover_schema("public") -> generate_ontology() -> execute_sql_query(...)
```

**Quick data exploration:**

```
connect_database("duckdb") -> list_schemas() -> sample_table_data("events")
```

**Query with visualization:**

```
execute_sql_query(query) -> generate_chart(data, "bar", ...)
```

`execute_sql_query` runs OBQC validation, security checks, and fan-trap detection before executing — no separate validation step is needed.

**Resume a previous session (auto-restores workspace):**

```
connect_database("postgresql") -> execute_sql_query(...)
```

## Documentation

| Document                                           | Contents                                                                      |
| -------------------------------------------------- | ----------------------------------------------------------------------------- |
| [Tools Reference](docs/tools-reference.md)         | Full parameter docs, return values, and usage examples                        |
| [Configuration](docs/configuration.md)             | Environment variables, transport setup, troubleshooting                       |
| [GraphRAG](docs/graphrag.md)                       | Graph-based schema intelligence and OBML workflow                             |
| [OBQC Overview](docs/obqc-overview.md)             | Short explanation of how OBQC works inside OrionBelt Analytics                |
| [OBQC](docs/obqc.md)                               | Validation rules, severity levels, blocking behavior, annotation requirements |
| [Fan-Trap Prevention](docs/fan-trap-prevention.md) | The fan-trap problem, detection, and safe SQL patterns                        |
| [Integrations](docs/integrations.md)               | LangChain, OpenAI, CrewAI, Google ADK, Vercel, n8n, ChatGPT                   |
| [Development](docs/development.md)                 | Project structure, testing, contributing                                      |

## License

Copyright 2025-2026 [RALFORION d.o.o.](https://ralforion.com)

Licensed under the [Business Source License 1.1](LICENSE). The Licensed Work will convert to Apache License 2.0 on 2030-03-16.

By contributing to this project, you agree to the [Contributor License Agreement](CLA.md).

For commercial licensing inquiries, contact: licensing@ralforion.com

---

<p align="center">
  <a href="https://ralforion.com">
    <img src="https://raw.githubusercontent.com/ralforion/orionbelt-analytics/main/assets/RALFORION_doo_Logo.png" alt="RALFORION d.o.o." width="200">
  </a>
</p>

<p align="center">
  Copyright © 2026 RALFORION d.o.o.<br>
  OrionBelt® is a registered trademark of RALFORION d.o.o.
</p>

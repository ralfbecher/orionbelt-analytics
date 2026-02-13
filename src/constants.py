"""Constants for OrionBelt Analytics."""

# Database connection constants
DEFAULT_POSTGRES_PORT = 5432
DEFAULT_SNOWFLAKE_SCHEMA = "PUBLIC"
DEFAULT_DREMIO_PORT = 9047  # Dremio REST API port
DEFAULT_CLICKHOUSE_PORT = 8123  # ClickHouse HTTP protocol port

# Data sampling limits
MIN_SAMPLE_LIMIT = 1
MAX_SAMPLE_LIMIT = 1000
DEFAULT_SAMPLE_LIMIT = 10
MAX_ENRICHMENT_SAMPLES = 3

# Connection and timeout settings
CONNECTION_TIMEOUT = 30
QUERY_TIMEOUT = 60

# Ontology generation constants
DEFAULT_BASE_URI = "http://example.com/ontology/"
ONTOLOGY_TITLE = "Database Schema Ontology"
ONTOLOGY_DESCRIPTION = "Ontology generated from database schema"

# R2RML mapping constants
DEFAULT_R2RML_BASE_IRI = "http://mycompany.com/"

# Output directory for generated files (schema, ontology, r2rml, etc.)
DEFAULT_OUTPUT_DIR = "tmp"

# Identifier validation pattern
IDENTIFIER_PATTERN = r'^[a-zA-Z_][a-zA-Z0-9_-]*$'

# Supported database types
SUPPORTED_DB_TYPES = ["postgresql", "snowflake", "dremio", "clickhouse"]

# System schemas to exclude
POSTGRES_SYSTEM_SCHEMAS = ["information_schema", "pg_catalog", "pg_toast"]
SNOWFLAKE_SYSTEM_SCHEMAS = ["INFORMATION_SCHEMA", "SNOWFLAKE", "SNOWFLAKE_SAMPLE_DATA"]
DREMIO_SYSTEM_SCHEMAS = ["INFORMATION_SCHEMA", "sys"]
CLICKHOUSE_SYSTEM_SCHEMAS = ["system", "INFORMATION_SCHEMA", "information_schema"]
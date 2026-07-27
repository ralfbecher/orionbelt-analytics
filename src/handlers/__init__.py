"""Handler modules for OrionBelt Analytics MCP tools.

Each module contains the implementation logic for a group of related tools.
The @mcp.tool() decorators remain in main.py for FastMCP registration.

Modules:
    connection - Database connection and diagnostics
    schema     - Schema analysis, table details, cache management
    ontology   - Ontology generation, semantic names, loading
    query      - SQL validation and execution
    chart      - Chart generation (interactive and static)
    rdf        - Oxigraph RDF store and SPARQL operations
    graphrag   - GraphRAG initialization and search
    workspace  - Workspace restore from previous sessions
"""

from . import chart, connection, graphrag, ontology, query, rdf, schema, workspace

__all__ = [
    "chart",
    "connection",
    "graphrag",
    "ontology",
    "query",
    "rdf",
    "schema",
    "workspace",
]

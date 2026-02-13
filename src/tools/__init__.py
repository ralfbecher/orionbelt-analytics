"""
MCP Tools for the Database Ontology MCP Server.

This package contains all MCP tools organized by functionality.
Each tool is in its own module for better maintainability.
"""

# Import all tools for easy access
from .connection import connect_database, diagnose_connection_issue
from .schema import list_schemas, get_analysis_context, sample_table_data
from .ontology import generate_ontology, load_ontology_from_file
from .query import validate_sql_syntax, execute_sql_query
from .chart import generate_chart
from .info import get_server_info

__all__ = [
    'connect_database',
    'diagnose_connection_issue',
    'list_schemas',
    'get_analysis_context',
    'sample_table_data',
    'generate_ontology',
    'load_ontology_from_file',
    'validate_sql_syntax',
    'execute_sql_query',
    'generate_chart',
    'get_server_info'
]
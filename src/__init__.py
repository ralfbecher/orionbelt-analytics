"""OrionBelt Analytics - Ontology-based MCP server for your Text-2-SQL convenience."""

__version__ = "0.5.0"
__author__ = "OrionBelt Analytics Contributors"
__email__ = "contributors@example.com"
__description__ = "OrionBelt Analytics - the Ontology-based MCP server for your Text-2-SQL convenience"
__name__ = "OrionBelt Analytics"

# Export main components for easier imports
from .database_manager import DatabaseManager, TableInfo, ColumnInfo
from .ontology_generator import OntologyGenerator
from .config import config_manager
from .constants import SUPPORTED_DB_TYPES

__all__ = [
    "DatabaseManager",
    "TableInfo", 
    "ColumnInfo",
    "OntologyGenerator",
    "config_manager",
    "SUPPORTED_DB_TYPES",
    "__version__",
    "__name__",
    "__description__",
]

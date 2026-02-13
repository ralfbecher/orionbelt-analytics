"""Server information tool."""

import logging
from typing import Dict, Any

from .. import __version__, __name__ as SERVER_NAME, __description__
from ..config import config_manager

logger = logging.getLogger(__name__)


def get_server_info() -> Dict[str, Any]:
    """Get server information implementation. Full documentation in main.py."""
    server_config = config_manager.get_server_config()
    
    return {
        "name": SERVER_NAME,
        "version": __version__, 
        "description": __description__,
        "supported_databases": ["PostgreSQL", "Snowflake", "Dremio", "ClickHouse"],
        "features": [
            "🌟 Single main analysis tool with automatic ontology generation",
            "🚀 5-step workflow (Connect → Analyze → Validate → Execute → Visualize)",
            "🎯 Self-sufficient ontologies with direct SQL references",
            "🔧 Ready-to-use JOIN conditions and business context",
            "📈 Interactive charting with Plotly and Matplotlib/Seaborn support",
            "⚡ Performance with consolidated functionality",
            "🎨 Clean, focused interface with 11 essential tools"
        ],
        "tools": [
            "connect_database",        # Connect to PostgreSQL, Snowflake or Dremio database
            "list_schemas",           # List available database schemas  
            "get_analysis_context",   # 🚀 MAIN TOOL: Complete analysis with automatic ontology
            "generate_ontology",      # Generate ontology manually (if needed)
            "load_ontology_from_file", # Load saved/edited ontology from tmp folder
            "sample_table_data",      # Sample data from specific tables
            "validate_sql_syntax",    # Validate SQL before execution
            "execute_sql_query",      # Execute validated SQL queries
            "generate_chart",         # 📈 Generate interactive charts from data
            "diagnose_connection_issue", # 🔍 Diagnose and troubleshoot connection problems
            "get_server_info"         # Server information and capabilities
        ],
        "workflow": {
            "recommended_steps": [
                "1. connect_database() - Connect to your database",
                "2. get_analysis_context() - Get complete schema + ontology + relationships", 
                "3. validate_sql_syntax() - Validate your SQL queries",
                "4. execute_sql_query() - Execute validated queries",
                "5. generate_chart() - Visualize results with interactive charts"
            ],
            "main_tool": "get_analysis_context",
            "main_tool_benefits": [
                "Automatic ontology generation with SQL references",
                "Complete schema analysis in one call",
                "Relationship mapping and fan-trap warnings", 
                "Business context for data understanding"
            ]
        },
        "configuration": {
            "log_level": server_config.log_level,
            "ontology_base_uri": server_config.ontology_base_uri,
            "max_query_limit": 5000,
            "supported_formats": ["turtle", "html", "png", "json"],
            "supported_chart_types": ["bar", "line", "scatter", "heatmap"],
            "supported_chart_libraries": ["plotly", "matplotlib"]
        }
    }
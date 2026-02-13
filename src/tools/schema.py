"""Schema analysis and data sampling tools."""

import logging
from typing import Dict, Optional, Any

from ..shared import get_db_manager, create_error_response

logger = logging.getLogger(__name__)


def list_schemas() -> Dict[str, Any]:
    """List database schemas. Full documentation in main.py."""
    db_manager = get_db_manager()

    if not db_manager.has_engine():
        return create_error_response(
            "No database connection established",
            "connection_error",
            "Use connect_database tool first"
        )

    try:
        schemas = db_manager.get_schemas()
        logger.debug(f"Retrieved {len(schemas)} schemas")
        return {
            "success": True,
            "schemas": schemas,
            "count": len(schemas)
        }
    except Exception as e:
        logger.error(f"Failed to list schemas: {e}")
        return create_error_response(
            "Failed to retrieve schema list",
            "database_error",
            str(e)
        )


def get_analysis_context(
    schema_name: Optional[str] = None
) -> Dict[str, Any]:
    """Get analysis context implementation. Full documentation in main.py."""
    try:
        db_manager = get_db_manager()
        
        # Check connection
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        logger.info(f"Generating complete analysis context for schema: {schema_name or 'default'}")
        
        # Get schema analysis (inline implementation)
        try:
            tables = db_manager.get_tables(schema_name)
            logger.debug(f"Found {len(tables)} tables in schema '{schema_name or 'default'}'")
            
            # Sequential table analysis 
            all_table_info = []
            sample_data_dict = {}
            for table_name in tables:
                try:
                    table_info = db_manager.analyze_table(table_name, schema_name)
                    if table_info:
                        # Convert dataclass to dict for JSON serialization
                        # Separate schema structure from sample data
                        schema_table = {
                            "name": table_info.name,
                            "schema": table_info.schema,
                            "columns": [
                                {
                                    "name": col.name,
                                    "data_type": col.data_type,
                                    "is_nullable": col.is_nullable,
                                    "is_primary_key": col.is_primary_key,
                                    "is_foreign_key": col.is_foreign_key,
                                    "foreign_key_table": col.foreign_key_table,
                                    "foreign_key_column": col.foreign_key_column,
                                    "comment": col.comment
                                } for col in table_info.columns
                            ],
                            "primary_keys": table_info.primary_keys,
                            "foreign_keys": table_info.foreign_keys,
                            "comment": table_info.comment,
                            "row_count": table_info.row_count
                        }
                        all_table_info.append(schema_table)
                        
                        # Store sample data separately if available
                        if table_info.sample_data:
                            sample_data_dict[table_info.name] = table_info.sample_data
                except Exception as e:
                    logger.warning(f"Failed to analyze table {table_name}: {e}")
            
            schema_data = {
                "schema": schema_name or "default",
                "table_count": len(all_table_info),
                "tables": all_table_info
            }
            
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
        
        # Get relationships (inline implementation)
        relationships = {}
        table_names = [table['name'] for table in all_table_info]
        
        for table_name in table_names:
            table_data = next((t for t in all_table_info if t['name'] == table_name), None)
            if table_data and table_data.get('foreign_keys'):
                relationships[table_name] = table_data['foreign_keys']
            
        result = {
            "schema_analysis": schema_data,
            "sample_data": sample_data_dict,
            "relationships": relationships,
            "sql_hints": {
                "workflow": [
                    "1. Review the schema_analysis to understand table structure",
                    "2. Check relationships for potential fan-traps before JOINs",
                    "3. Use generate_semantic_descriptions(schema_analysis) for business context",
                    "4. Use generate_ontology with semantic descriptions for enriched ontology",
                    "5. Validate SQL syntax before execution",
                    "6. Execute queries with appropriate limits"
                ],
                "data_separation": [
                    "schema_analysis: Pure schema structure (for generate_semantic_descriptions)",
                    "sample_data: Actual data samples (optional for additional context)",
                    "Use schema_analysis for semantic analysis, sample_data only if needed"
                ]
            }
        }
        
        # Add relationship warnings for analysis
        fan_trap_warnings = []
        for table, fks in relationships.items():
            if len(fks) > 1:
                referenced_tables = [fk['referenced_table'] for fk in fks]
                fan_trap_warnings.append({
                    "table": table,
                    "warning": f"Table {table} connects to multiple tables - potential fan-trap risk",
                    "referenced_tables": referenced_tables,
                    "recommendation": "Use separate CTEs or UNION approach for multi-fact aggregations"
                })
        
        if fan_trap_warnings:
            result["sql_hints"]["fan_trap_warnings"] = fan_trap_warnings
            
        logger.info(f"Generated analysis context: {len(schema_data.get('tables', []))} tables")
                   
        return result
        
    except Exception as e:
        logger.error(f"Error generating analysis context: {e}")
        return create_error_response(
            f"Failed to generate analysis context: {str(e)}",
            "internal_error"
        )


def sample_table_data(
    table_name: str,
    schema_name: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """Sample table data implementation. Full documentation in main.py."""
    db_manager = get_db_manager()

    if not db_manager.has_engine():
        return create_error_response(
            "No database connection established",
            "connection_error",
            "Use connect_database tool first"
        )

    try:
        sample_data = db_manager.sample_table_data(table_name, schema_name, limit)
        logger.info(f"Retrieved {len(sample_data)} sample rows from {table_name}")
        return {
            "success": True,
            "table_name": table_name,
            "schema_name": schema_name,
            "sample_data": sample_data,
            "row_count": len(sample_data),
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Failed to sample table data: {e}")
        return create_error_response(
            f"Failed to sample data from table '{table_name}'",
            "database_error",
            str(e)
        )
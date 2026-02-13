"""Ontology generation and management tools."""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from ..config import config_manager
from ..shared import get_db_manager, create_error_response
from ..ontology_generator import OntologyGenerator

logger = logging.getLogger(__name__)


def generate_ontology(
    schema_name: Optional[str] = None,
    base_uri: Optional[str] = None,
    semantic_descriptions: Optional[str] = None
) -> Dict[str, Any]:
    """Generate ontology implementation. Full documentation in main.py."""
    try:
        db_manager = get_db_manager()
        server_config = config_manager.get_server_config()
        
        if not db_manager.has_engine():
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Please use connect_database tool first to establish a connection"
            )
        
        try:
            tables = db_manager.get_tables(schema_name)
            if not tables:
                return create_error_response(
                    f"No tables found in schema '{schema_name or 'default'}'",
                    "data_error"
                )
            
            # Analyze tables
            tables_info = []
            for table_name in tables:
                try:
                    table_info = db_manager.analyze_table(table_name, schema_name)
                    if table_info:
                        tables_info.append(table_info)
                except Exception as e:
                    logger.warning(f"Failed to analyze table {table_name}: {e}")
            
            if not tables_info:
                return create_error_response(
                    f"Could not analyze any tables in schema '{schema_name or 'default'}'",
                    "data_error"
                )
            
            # Generate ontology
            uri = base_uri or server_config.ontology_base_uri
            generator = OntologyGenerator(base_uri=uri)
            ontology_ttl = generator.generate_from_schema(tables_info)
            
            # Track whether descriptions were applied
            enriched = False
            
            # Apply semantic descriptions if provided
            if semantic_descriptions:
                logger.info("Applying semantic descriptions to ontology")
                try:
                    # Handle JSON string or dictionary
                    descriptions_dict = semantic_descriptions
                    if isinstance(semantic_descriptions, str):
                        import json
                        try:
                            descriptions_dict = json.loads(semantic_descriptions)
                            logger.debug("Parsed JSON string semantic descriptions")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse semantic descriptions JSON: {e}. Using basic ontology.")
                            descriptions_dict = None
                    
                    if descriptions_dict and isinstance(descriptions_dict, dict):
                        # Check if it has the expected structure (tables, columns, relationships)
                        if any(key in descriptions_dict for key in ['tables', 'columns', 'relationships']):
                            generator.apply_semantic_descriptions(descriptions_dict)
                            ontology_ttl = generator.serialize_ontology()
                            enriched = True
                            logger.info("Successfully applied semantic descriptions to ontology")
                        else:
                            logger.warning("Semantic descriptions provided but not in expected format. Using basic ontology.")
                            logger.debug(f"Received format keys: {list(descriptions_dict.keys())}")
                    else:
                        logger.warning("Semantic descriptions should be a dictionary or valid JSON string. Using basic ontology.")
                except Exception as e:
                    logger.warning(f"Failed to apply semantic descriptions: {e}. Using basic ontology.")
            
            # Save ontology to tmp folder for user access
            ontology_file_path = None
            try:
                TMP_DIR = Path(__file__).parent.parent.parent / "tmp"
                TMP_DIR.mkdir(exist_ok=True)  # Ensure tmp directory exists
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                schema_safe = (schema_name or "default").replace(" ", "_").replace(".", "_")
                ontology_filename = f"ontology_{schema_safe}_{timestamp}.ttl"
                ontology_file_path = TMP_DIR / ontology_filename
                
                with open(ontology_file_path, 'w', encoding='utf-8') as f:
                    f.write(ontology_ttl)
                
                logger.info(f"Generated ontology for schema '{schema_name or 'default'}': {len(tables_info)} tables")
                logger.info(f"Saved ontology to: {ontology_file_path}")
                
            except Exception as e:
                logger.warning(f"Failed to save ontology to file: {e}")
                ontology_file_path = None
            
            # Return structured response
            return {
                "success": True,
                "ontology": ontology_ttl,
                "file_path": str(ontology_file_path) if ontology_file_path else None,
                "schema": schema_name or "default",
                "table_count": len(tables_info),
                "enriched": enriched,
                "base_uri": uri,
                "generation_info": {
                    "timestamp": datetime.now().isoformat(),
                    "has_semantic_descriptions": bool(semantic_descriptions),
                    "applied_enrichment": enriched
                }
            }
            
        except RuntimeError as e:
            return create_error_response(
                "No database connection established",
                "connection_error",
                "Use connect_database tool first"
            )
    except Exception as e:
        logger.error(f"Error generating ontology: {e}")
        return create_error_response(
            f"Failed to generate ontology: {str(e)}",
            "internal_error"
        )


def load_ontology_from_file(
    file_path: str
) -> Dict[str, Any]:
    """Load ontology from file implementation. Full documentation in main.py."""
    try:
        # Get tmp directory
        TMP_DIR = Path(__file__).parent.parent.parent / "tmp"
        
        # Handle relative paths (assume they're in tmp folder)
        if not os.path.isabs(file_path):
            full_path = TMP_DIR / file_path
        else:
            full_path = Path(file_path)
        
        # Check if file exists
        if not full_path.exists():
            return create_error_response(
                f"Ontology file not found: {full_path}",
                "file_not_found",
                f"Available files in tmp folder: {[f.name for f in TMP_DIR.glob('*.ttl')] if TMP_DIR.exists() else 'tmp folder not found'}"
            )
        
        # Check file extension
        if not full_path.suffix.lower() == '.ttl':
            return create_error_response(
                f"Invalid file format. Expected .ttl file, got: {full_path.suffix}",
                "invalid_format",
                "Only Turtle (.ttl) format ontology files are supported"
            )
        
        # Read ontology file
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                ontology_content = f.read()
        except Exception as e:
            return create_error_response(
                f"Failed to read ontology file: {str(e)}",
                "read_error"
            )
        
        # Validate that it's a valid Turtle format (basic check)
        if not ontology_content.strip():
            return create_error_response(
                "Ontology file is empty",
                "empty_file"
            )
        
        # Basic validation - check for common RDF/Turtle patterns
        turtle_patterns = ['@prefix', 'PREFIX', 'rdf:', 'rdfs:', 'owl:', '<', '>']
        if not any(pattern in ontology_content for pattern in turtle_patterns):
            logger.warning(f"File {full_path} may not be a valid Turtle ontology - no RDF patterns found")
        
        # Get file statistics
        file_stats = full_path.stat()
        file_size = file_stats.st_size
        modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # Count some basic statistics
        line_count = len(ontology_content.splitlines())
        triple_count = ontology_content.count(' .') + ontology_content.count(' ;\n') + ontology_content.count(' ,\n')
        
        logger.info(f"Successfully loaded ontology from: {full_path} ({file_size} bytes, {line_count} lines)")
        
        return {
            "success": True,
            "ontology": ontology_content,
            "file_info": {
                "file_path": str(full_path),
                "filename": full_path.name,
                "file_size_bytes": file_size,
                "file_size_human": f"{file_size / 1024:.1f} KB" if file_size > 1024 else f"{file_size} B",
                "line_count": line_count,
                "estimated_triple_count": triple_count,
                "last_modified": modified_time
            },
            "usage_hints": [
                "Use this ontology content for analytical context and SQL generation",
                "Extract table.column references for accurate SQL queries",
                "Look for business descriptions to understand data meaning",
                "Check relationship annotations for proper JOIN conditions",
                "Use the ontology as documentation for the database schema"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error loading ontology from file: {e}")
        return create_error_response(
            f"Failed to load ontology: {str(e)}",
            "load_error"
        )
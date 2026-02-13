"""R2RML mapping generator for creating RDB to RDF mappings from database schemas.

This module generates R2RML (RDB to RDF Mapping Language) mappings that define
how relational database tables map to RDF triples. The generated mappings follow
the W3C R2RML specification: https://www.w3.org/TR/r2rml/

R2RML mappings enable:
- Converting relational data to RDF format
- Semantic integration of database content
- SPARQL querying over relational databases
- Linked Data publication from databases
"""

import logging
import re
from typing import List, Dict, Optional, Any

from .database_manager import TableInfo, ColumnInfo

logger = logging.getLogger(__name__)


class R2RMLGenerator:
    """Generates R2RML mappings from database schema information.

    R2RML (RDB to RDF Mapping Language) is a W3C standard for expressing
    customized mappings from relational databases to RDF datasets.

    This generator creates TriplesMap definitions that include:
    - Subject maps with IRI templates based on primary keys
    - Predicate-object maps for column values
    - Reference object maps for foreign key relationships
    """

    def __init__(
        self,
        base_iri: str = "http://example.com/",
        database_name: str = "database"
    ):
        """Initialize the R2RML generator.

        Args:
            base_iri: Base IRI for generated RDF resources (must end with /)
            database_name: Name of the database (used in comments)
        """
        self.base_iri = base_iri if base_iri.endswith('/') else base_iri + '/'
        self.database_name = database_name

    def generate_from_schema(
        self,
        tables_info: List[TableInfo],
        schema_name: Optional[str] = None
    ) -> str:
        """Generate R2RML mapping from database schema information.

        Args:
            tables_info: List of TableInfo objects containing schema metadata
            schema_name: Optional schema name to include in table references

        Returns:
            R2RML mapping in Turtle format
        """
        if not tables_info:
            logger.warning("No tables provided for R2RML generation")
            return ""

        lines = []

        # Add prefixes
        lines.extend(self._generate_prefixes())
        lines.append("")

        # Build a lookup for tables by name for FK resolution
        table_lookup = {t.name: t for t in tables_info}

        # Generate TriplesMap for each table
        for table_info in tables_info:
            lines.extend(self._generate_triples_map(table_info, schema_name, table_lookup))
            lines.append("")

        return "\n".join(lines)

    def _generate_prefixes(self) -> List[str]:
        """Generate R2RML prefix declarations."""
        prefixes = [
            "@prefix rr: <http://www.w3.org/ns/r2rml#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            f"@prefix base: <{self.base_iri}> .",
        ]
        return prefixes

    def _generate_triples_map(
        self,
        table_info: TableInfo,
        schema_name: Optional[str],
        table_lookup: Dict[str, TableInfo]
    ) -> List[str]:
        """Generate a TriplesMap for a single table.

        Args:
            table_info: TableInfo object for the table
            schema_name: Optional schema name
            table_lookup: Dictionary mapping table names to TableInfo objects

        Returns:
            List of Turtle lines for the TriplesMap
        """
        lines = []

        # Generate mapping name (safe identifier)
        map_name = self._safe_identifier(table_info.name.capitalize()) + "Mapping"
        table_name = table_info.name
        effective_schema = schema_name or table_info.schema

        # Full table reference with schema
        if effective_schema and effective_schema != "public":
            full_table_ref = f"{effective_schema}.{table_name}"
        else:
            full_table_ref = table_name

        # Generate subject template based on primary keys
        subject_template = self._generate_subject_template(table_info, effective_schema)

        # Class name for rr:class
        class_name = self._safe_identifier(table_name.capitalize())

        lines.append(f"<#{map_name}>")
        lines.append("    a rr:TriplesMap ;")

        # Logical table with SQL query (R2RML)
        lines.append("    rr:logicalTable [")
        lines.append("        a rr:LogicalTable ;")
        lines.append(f'        rr:sqlQuery "SELECT * FROM {full_table_ref} " ;')
        lines.append("        rr:sqlVersion rr:SQL2008")
        lines.append("    ] ;")

        # Subject map
        lines.append(f'    rr:subjectMap [ rr:template "{subject_template}" ; rr:class base:{class_name} ] ;')

        # Build FK lookup for this table
        fk_columns = {}
        for fk in table_info.foreign_keys:
            fk_columns[fk['column']] = fk

        # Predicate-object maps for each column
        for column in table_info.columns:
            fk = fk_columns.get(column.name)
            if fk:
                # Foreign key column - generate object property with IRI reference
                lines.extend(self._generate_fk_predicate_object_map(
                    column, fk, effective_schema, table_lookup
                ))
            else:
                # Regular column - generate data property with literal value
                lines.extend(self._generate_predicate_object_map(column, table_name))

        # Remove trailing semicolon from last predicate-object map and add period
        if lines and lines[-1].endswith(" ;"):
            lines[-1] = lines[-1][:-2] + " ."
        elif lines and lines[-1].strip() == "] ;":
            lines[-1] = "    ] ."

        return lines

    def _generate_subject_template(
        self,
        table_info: TableInfo,
        schema_name: Optional[str]
    ) -> str:
        """Generate subject IRI template based on primary keys.

        Args:
            table_info: TableInfo object
            schema_name: Optional schema name

        Returns:
            Subject template string with placeholders for PK values
        """
        table_name = table_info.name
        pk_columns = table_info.primary_keys

        if len(pk_columns) == 1:
            # Single primary key
            return f"{self.base_iri}{table_name}/{{{pk_columns[0]}}}"
        elif len(pk_columns) > 1:
            # Composite primary key - join with hyphen
            pk_template = "-".join([f"{{{pk}}}" for pk in pk_columns])
            return f"{self.base_iri}{table_name}/{pk_template}"
        else:
            # No primary key defined - use all columns as fallback (not ideal)
            # Try to find an 'id' column
            id_col = next(
                (c.name for c in table_info.columns if c.name.lower() in ['id', 'rowid', '_id']),
                None
            )
            if id_col:
                return f"{self.base_iri}{table_name}/{{{id_col}}}"
            else:
                # Last resort: use first column
                first_col = table_info.columns[0].name if table_info.columns else "id"
                logger.warning(
                    f"Table '{table_name}' has no primary key, using column '{first_col}' for subject template"
                )
                return f"{self.base_iri}{table_name}/{{{first_col}}}"

    def _generate_predicate_object_map(
        self,
        column: ColumnInfo,
        table_name: str
    ) -> List[str]:
        """Generate predicate-object map for a regular column.

        Args:
            column: ColumnInfo object for the column
            table_name: Name of the containing table

        Returns:
            List of Turtle lines for the predicate-object map
        """
        lines = []
        predicate_name = self._safe_identifier(column.name)
        xsd_type = self._map_sql_to_xsd(column.data_type)

        lines.append("    rr:predicateObjectMap [")
        lines.append(f"        rr:predicate base:{predicate_name} ;")
        lines.append("        rr:objectMap [")
        lines.append(f'            rr:column "{column.name}" ;')
        lines.append(f"            rr:datatype {xsd_type} ;")
        lines.append("        ] ;")
        lines.append("    ] ;")

        return lines

    def _generate_fk_predicate_object_map(
        self,
        column: ColumnInfo,
        fk: Dict[str, str],
        schema_name: Optional[str],
        table_lookup: Dict[str, TableInfo]
    ) -> List[str]:
        """Generate predicate-object map for a foreign key column.

        Foreign key columns are mapped as object properties that reference
        IRIs of the related table's resources.

        Args:
            column: ColumnInfo object for the FK column
            fk: Foreign key info dict with 'referenced_table' and 'referenced_column'
            schema_name: Optional schema name
            table_lookup: Dict mapping table names to TableInfo

        Returns:
            List of Turtle lines for the FK predicate-object map
        """
        lines = []
        predicate_name = self._safe_identifier(column.name)
        ref_table = fk['referenced_table']
        ref_column = fk['referenced_column']

        # Build the referenced IRI template
        # Use the FK column value to construct the referenced resource IRI
        ref_template = f"{self.base_iri}{ref_table}/{{{column.name}}}"

        lines.append("    rr:predicateObjectMap [")
        lines.append(f"        rr:predicate base:{predicate_name} ;")
        lines.append("        rr:objectMap [")
        lines.append(f'            rr:template "{ref_template}" ;')
        lines.append("            rr:termType rr:IRI ;")
        lines.append("        ] ;")
        lines.append("    ] ;")

        return lines

    def _map_sql_to_xsd(self, sql_type: str) -> str:
        """Map SQL data types to XSD Schema datatypes.

        Args:
            sql_type: SQL data type string

        Returns:
            XSD datatype as prefixed string (e.g., "xsd:integer")
        """
        sql_type = sql_type.lower()

        # Integer types
        if "tinyint" in sql_type:
            return "xsd:byte"
        if any(t in sql_type for t in ["integer", "int", "int4", "int2", "int8", "serial", "bigint", "smallint"]):
            return "xsd:integer"

        # Numeric types
        if any(t in sql_type for t in ["numeric", "decimal", "money"]):
            return "xsd:decimal"
        if any(t in sql_type for t in ["float4", "float8", "real", "float"]):
            return "xsd:float"
        if "double" in sql_type:
            return "xsd:double"

        # Temporal types
        if any(t in sql_type for t in ["timestamp", "datetime"]):
            return "xsd:dateTime"
        if sql_type.startswith("date"):
            return "xsd:date"
        if sql_type.startswith("time"):
            return "xsd:time"

        # Boolean types
        if any(t in sql_type for t in ["bool", "boolean"]):
            return "xsd:boolean"

        # Binary types
        if any(t in sql_type for t in ["bytea", "binary", "varbinary", "blob"]):
            return "xsd:base64Binary"

        # String types (default)
        return "xsd:string"

    def _safe_identifier(self, name: str) -> str:
        """Convert a name to a safe identifier for use in IRIs.

        Args:
            name: Original name

        Returns:
            Safe identifier string
        """
        if not name:
            return "unnamed"

        # Replace non-alphanumeric characters with underscores
        safe = re.sub(r'[^A-Za-z0-9_]', '_', name)

        # Ensure it starts with a letter or underscore
        if safe and not (safe[0].isalpha() or safe[0] == '_'):
            safe = '_' + safe

        return safe or "unnamed"


def generate_r2rml_mapping(
    tables_info: List[TableInfo],
    base_iri: str = "http://example.com/",
    database_name: str = "database",
    schema_name: Optional[str] = None
) -> str:
    """Convenience function to generate R2RML mapping.

    Args:
        tables_info: List of TableInfo objects containing schema metadata
        base_iri: Base IRI for generated RDF resources
        database_name: Name of the database (used in comments)
        schema_name: Optional schema name to include in table references

    Returns:
        R2RML mapping in Turtle format (W3C R2RML standard)
    """
    generator = R2RMLGenerator(
        base_iri=base_iri,
        database_name=database_name
    )
    return generator.generate_from_schema(tables_info, schema_name)

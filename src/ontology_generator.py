"""Ontology generator for creating RDF graphs from database schemas."""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

from .constants import DEFAULT_BASE_URI, ONTOLOGY_TITLE, ONTOLOGY_DESCRIPTION
from .database_manager import TableInfo, ColumnInfo

logger = logging.getLogger(__name__)

# Define namespaces
EX = Namespace("http://example.com/ontology/")


@dataclass
class InferredRelationship:
    """Represents a relationship inferred from naming patterns."""
    source_table: str
    column: str
    target_table: str
    target_column: str
    confidence: str  # 'high', 'medium', 'low'
    pattern_matched: str


@dataclass
class DenormalizedField:
    """Represents a detected denormalized field."""
    table: str
    column: str
    likely_source_table: str
    data_type: str
    warning: str


@dataclass
class OntologyQualityReport:
    """Report on ontology generation quality."""
    inferred_relationships: List[InferredRelationship] = field(default_factory=list)
    denormalized_fields: List[DenormalizedField] = field(default_factory=list)
    type_overrides: List[Dict[str, str]] = field(default_factory=list)
    missing_descriptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "inferred_relationships": [
                {
                    "source_table": r.source_table,
                    "column": r.column,
                    "target_table": r.target_table,
                    "target_column": r.target_column,
                    "confidence": r.confidence,
                    "pattern_matched": r.pattern_matched
                }
                for r in self.inferred_relationships
            ],
            "denormalized_fields": [
                {
                    "table": d.table,
                    "column": d.column,
                    "likely_source_table": d.likely_source_table,
                    "warning": d.warning
                }
                for d in self.denormalized_fields
            ],
            "type_overrides": self.type_overrides,
            "missing_descriptions": self.missing_descriptions,
            "warnings": self.warnings,
            "summary": {
                "inferred_relationship_count": len(self.inferred_relationships),
                "denormalized_field_count": len(self.denormalized_fields),
                "type_override_count": len(self.type_overrides),
                "missing_description_count": len(self.missing_descriptions),
                "warning_count": len(self.warnings)
            }
        }

class OntologyGenerator:
    """Generates an ontology from a database schema with comprehensive database annotations."""

    # Patterns for inferring FK relationships from column names
    FK_PATTERNS = [
        # Pattern: suppliercountryid → country/countries
        (r'^(.+?)(?:_)?id$', 'suffix_id'),
        # Pattern: id_country → country
        (r'^id[_]?(.+)$', 'prefix_id'),
        # Pattern: fk_country → country
        (r'^fk[_]?(.+)$', 'prefix_fk'),
        # Pattern: country_fk → country
        (r'^(.+?)[_]?fk$', 'suffix_fk'),
        # Pattern: customer_sk → customer (TPC-DS style surrogate keys)
        (r'^(.+?)(?:_)?sk$', 'suffix_sk'),
        # Pattern: ss_customer_sk → customer (TPC-DS with table prefix)
        (r'^[a-z]{1,3}_(.+?)(?:_)?sk$', 'tpcds_sk'),
    ]

    # Column name patterns that indicate quantities (should be integers)
    QUANTITY_PATTERNS = ['quantity', 'qty', 'count', 'cnt', 'number', 'num', 'amount', 'amt']

    # Patterns for denormalized text fields
    DENORM_SUFFIXES = ['name', 'title', 'description', 'desc', 'label']

    def __init__(self, base_uri: str = DEFAULT_BASE_URI):
        self.graph = Graph()
        self.base_uri = Namespace(base_uri)

        # Define custom namespace for database-specific annotations
        self.db_ns = Namespace(f"{base_uri}db/")

        self.graph.bind("ns", self.base_uri)
        self.graph.bind("db", self.db_ns)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)

        # Quality tracking
        self.quality_report: Optional[OntologyQualityReport] = None
        self._table_lookup: Dict[str, TableInfo] = {}
        self._pk_columns: Dict[str, List[str]] = {}  # table -> primary key columns

    def load_from_file(self, file_path: str) -> None:
        """Load an existing ontology from a Turtle file.

        Args:
            file_path: Path to the .ttl file to load
        """
        self.graph = Graph()
        self.graph.parse(file_path, format="turtle")

        # Re-bind namespaces
        self.graph.bind("ns", self.base_uri)
        self.graph.bind("db", self.db_ns)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)

        logger.info(f"Loaded ontology from {file_path} with {len(self.graph)} triples")

    def load_from_string(self, turtle_content: str) -> None:
        """Load an existing ontology from a Turtle string.

        Args:
            turtle_content: Turtle format ontology content
        """
        self.graph = Graph()
        self.graph.parse(data=turtle_content, format="turtle")

        # Re-bind namespaces
        self.graph.bind("ns", self.base_uri)
        self.graph.bind("db", self.db_ns)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)

        logger.info(f"Loaded ontology from string with {len(self.graph)} triples")

    def generate_from_schema(
        self,
        tables_info: List[TableInfo],
        include_inferred_relationships: bool = True,
        annotate_denormalized: bool = True
    ) -> str:
        """Generate an ontology from a list of table information with quality enhancements.

        Args:
            tables_info: List of table information from schema analysis
            include_inferred_relationships: Whether to add relationships inferred from naming patterns
            annotate_denormalized: Whether to annotate detected denormalized fields

        Returns:
            Serialized ontology in Turtle format
        """
        # Initialize quality report
        self.quality_report = OntologyQualityReport()

        # Build table lookup for relationship inference
        self._build_table_lookup(tables_info)

        # Add ontology metadata
        ontology_uri = self.base_uri[""]
        self.graph.add((ontology_uri, RDF.type, OWL.Ontology))
        self.graph.add((ontology_uri, RDFS.label, Literal(ONTOLOGY_TITLE)))
        self.graph.add((ontology_uri, RDFS.comment, Literal(ONTOLOGY_DESCRIPTION)))

        # Add all tables and their columns/relationships
        for table_info in tables_info:
            self._add_table_to_ontology(table_info)

        # Infer implicit relationships from naming patterns
        if include_inferred_relationships:
            inferred = self._infer_implicit_relationships(tables_info)
            self.quality_report.inferred_relationships = inferred

            for rel in inferred:
                self._add_inferred_relationship_to_ontology(rel)
                logger.info(
                    f"Added inferred relationship: {rel.source_table}.{rel.column} -> "
                    f"{rel.target_table} (confidence: {rel.confidence})"
                )

        # Detect and annotate denormalized fields
        if annotate_denormalized:
            denormalized = self._detect_denormalized_fields(tables_info)
            self.quality_report.denormalized_fields = denormalized

            for denorm in denormalized:
                self._add_denormalized_field_annotation(denorm)
                logger.info(f"Annotated denormalized field: {denorm.table}.{denorm.column}")

        # Log quality summary
        if self.quality_report.inferred_relationships:
            logger.info(
                f"Quality: Added {len(self.quality_report.inferred_relationships)} "
                f"inferred relationships"
            )
        if self.quality_report.denormalized_fields:
            logger.warning(
                f"Quality: Detected {len(self.quality_report.denormalized_fields)} "
                f"potentially denormalized fields"
            )
        if self.quality_report.type_overrides:
            logger.info(
                f"Quality: Applied {len(self.quality_report.type_overrides)} "
                f"semantic type overrides"
            )

        return self.graph.serialize(format="turtle")

    def get_quality_report(self) -> Optional[OntologyQualityReport]:
        """Get the quality report from the last generation.

        Returns:
            Quality report or None if no generation has occurred
        """
        return self.quality_report

    def _add_table_to_ontology(self, table_info: TableInfo):
        """Add a single table and its columns to the ontology with comprehensive database annotations."""
        # Create proper URI for table class
        table_uri = self.base_uri[self._clean_name(table_info.name)]

        # Define table as a class
        self.graph.add((table_uri, RDF.type, OWL.Class))
        self.graph.add((table_uri, RDFS.label, Literal(table_info.name)))
        
        # Add comprehensive database-specific annotations
        self.graph.add((table_uri, self.db_ns.tableName, Literal(table_info.name)))
        self.graph.add((table_uri, self.db_ns.schemaName, Literal(table_info.schema)))
        
        if table_info.row_count is not None:
            self.graph.add((table_uri, self.db_ns.rowCount, Literal(table_info.row_count)))
            
        if table_info.comment:
            self.graph.add((table_uri, RDFS.comment, Literal(table_info.comment)))
            
        # Add primary key information
        if table_info.primary_keys:
            for pk in table_info.primary_keys:
                self.graph.add((table_uri, self.db_ns.primaryKey, Literal(pk)))

        # Add columns as properties
        for column in table_info.columns:
            self._add_column_to_ontology(table_uri, column, table_info.name)

        # Define relationships
        for fk in table_info.foreign_keys:
            self._add_relationship_to_ontology(table_uri, fk, table_info.name)

    def _add_column_to_ontology(self, table_uri: URIRef, column: ColumnInfo, table_name: str):
        """Add a column as a data property to the ontology with comprehensive database annotations."""
        # Create proper property URI
        prop_name = f"{self._clean_name(table_name)}_{self._clean_name(column.name)}"
        prop_uri = self.base_uri[prop_name]

        # Foreign key columns need both data property (for the value) and object property (for relationship)
        # Always create the data property for the column
        self.graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))

        self.graph.add((prop_uri, RDFS.domain, table_uri))
        self.graph.add((prop_uri, RDFS.label, Literal(column.name)))

        # Add comprehensive database-specific annotations
        self.graph.add((prop_uri, self.db_ns.columnName, Literal(column.name)))
        self.graph.add((prop_uri, self.db_ns.tableName, Literal(table_name)))
        self.graph.add((prop_uri, self.db_ns.sqlDataType, Literal(column.data_type)))
        self.graph.add((prop_uri, self.db_ns.isNullable, Literal(column.is_nullable)))
        self.graph.add((prop_uri, self.db_ns.isPrimaryKey, Literal(column.is_primary_key)))
        self.graph.add((prop_uri, self.db_ns.isForeignKey, Literal(column.is_foreign_key)))

        # Add SQL query generation hints
        full_column_ref = f"{table_name}.{column.name}"
        self.graph.add((prop_uri, self.db_ns.sqlReference, Literal(full_column_ref)))

        # Map SQL data types to proper XSD types (with semantic awareness)
        xsd_type, type_override = self._map_sql_to_xsd(
            column.data_type, column.name, table_name
        )
        if xsd_type:
            self.graph.add((prop_uri, RDFS.range, xsd_type))

        # Track type overrides for quality report
        if type_override and self.quality_report:
            self.quality_report.type_overrides.append(type_override)
            # Also add annotation to the property
            self.graph.add((
                prop_uri,
                self.db_ns.typeOverrideReason,
                Literal(type_override["reason"])
            ))

        # Note: Primary key and nullability constraints are already captured
        # in the metadata annotations (db:isPrimaryKey, db:isNullable).
        # We don't create OWL restriction classes as that would incorrectly
        # make table classes subclasses of restrictions.

        if column.comment:
            self.graph.add((prop_uri, RDFS.comment, Literal(column.comment)))

    def _add_relationship_to_ontology(self, table_uri: URIRef, fk: Dict[str, str], table_name: str):
        """Add a foreign key relationship as an object property with comprehensive database annotations."""
        referenced_table = fk.get('referenced_table')
        if not referenced_table:
            logger.warning(f"FK from {table_name} missing referenced_table: {fk}")
            return

        # Create descriptive relationship name
        rel_name = f"{self._clean_name(table_name)}_has_{self._clean_name(referenced_table)}"
        prop_uri = self.base_uri[rel_name]
        referenced_table_uri = self.base_uri[self._clean_name(referenced_table)]

        referenced_column = fk.get('referenced_column', 'id')
        fk_column = fk.get('column', '')
        referenced_schema = fk.get('referenced_schema')

        self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
        self.graph.add((prop_uri, RDFS.domain, table_uri))
        self.graph.add((prop_uri, RDFS.range, referenced_table_uri))
        self.graph.add((prop_uri, RDFS.label, Literal(f"{table_name} has {referenced_table}")))

        # Add comprehensive database-specific annotations for foreign keys
        self.graph.add((prop_uri, self.db_ns.foreignKeyColumn, Literal(fk_column)))
        self.graph.add((prop_uri, self.db_ns.referencedTable, Literal(referenced_table)))
        self.graph.add((prop_uri, self.db_ns.referencedColumn, Literal(referenced_column)))
        if referenced_schema:
            self.graph.add((prop_uri, self.db_ns.referencedSchema, Literal(referenced_schema)))

        # Add SQL join condition
        join_condition = f"{table_name}.{fk_column} = {referenced_table}.{referenced_column}"
        self.graph.add((prop_uri, self.db_ns.sqlJoinCondition, Literal(join_condition)))

        # Add relationship type annotation
        self.graph.add((prop_uri, self.db_ns.relationshipType, Literal("many_to_one")))

        # Add inverse relationship
        inverse_rel_name = f"{self._clean_name(referenced_table)}_referenced_by_{self._clean_name(table_name)}"
        inverse_prop_uri = self.base_uri[inverse_rel_name]
        self.graph.add((inverse_prop_uri, RDF.type, OWL.ObjectProperty))
        self.graph.add((inverse_prop_uri, RDFS.domain, referenced_table_uri))
        self.graph.add((inverse_prop_uri, RDFS.range, table_uri))
        self.graph.add((inverse_prop_uri, RDFS.label, Literal(f"{referenced_table} referenced by {table_name}")))

        # Add database annotations for inverse relationship
        self.graph.add((inverse_prop_uri, self.db_ns.relationshipType, Literal("one_to_many")))

        # Link them as inverses
        self.graph.add((prop_uri, OWL.inverseOf, inverse_prop_uri))

        logger.debug(f"Added declared FK relationship: {table_name}.{fk_column} -> {referenced_table}.{referenced_column}")

    def _clean_name(self, name: str) -> str:
        """Clean a name to make it suitable for URIs."""
        if not name:
            return 'unnamed'

        # Replace spaces and special characters with underscores
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)

        # Ensure it starts with a letter or underscore
        if cleaned and not (cleaned[0].isalpha() or cleaned[0] == '_'):
            cleaned = '_' + cleaned

        return cleaned or 'unnamed'

    def _build_table_lookup(self, tables_info: List[TableInfo]) -> None:
        """Build lookup structures for tables and their primary keys."""
        self._table_lookup = {}
        self._pk_columns = {}

        for table in tables_info:
            # Store with multiple key variations for matching
            table_lower = table.name.lower()
            self._table_lookup[table_lower] = table

            # Also store singular/plural variants
            if table_lower.endswith('s'):
                self._table_lookup[table_lower.rstrip('s')] = table
            if table_lower.endswith('es'):
                self._table_lookup[table_lower[:-2]] = table
            if table_lower.endswith('ies'):
                self._table_lookup[table_lower[:-3] + 'y'] = table

            # Store primary key columns
            self._pk_columns[table.name] = table.primary_keys

    def _find_matching_table(self, name: str) -> Optional[TableInfo]:
        """Find a table that matches the given name (handles pluralization)."""
        name_lower = name.lower()

        # Direct match
        if name_lower in self._table_lookup:
            return self._table_lookup[name_lower]

        # Try adding common plural suffixes
        for suffix in ['s', 'es', 'ies']:
            candidate = name_lower + suffix
            if candidate in self._table_lookup:
                return self._table_lookup[candidate]

        # Try removing plural suffixes
        if name_lower.endswith('ies') and len(name_lower) > 3:
            candidate = name_lower[:-3] + 'y'
            if candidate in self._table_lookup:
                return self._table_lookup[candidate]
        if name_lower.endswith('es') and len(name_lower) > 2:
            candidate = name_lower[:-2]
            if candidate in self._table_lookup:
                return self._table_lookup[candidate]
        if name_lower.endswith('s') and len(name_lower) > 1:
            candidate = name_lower[:-1]
            if candidate in self._table_lookup:
                return self._table_lookup[candidate]

        # Try removing 'y' and adding 'ies'
        if name_lower.endswith('y'):
            candidate = name_lower[:-1] + 'ies'
            if candidate in self._table_lookup:
                return self._table_lookup[candidate]

        return None

    def _infer_implicit_relationships(
        self, tables_info: List[TableInfo]
    ) -> List[InferredRelationship]:
        """Detect likely FK relationships from column naming patterns.

        This method analyzes column names to find implicit relationships that
        aren't declared as formal foreign keys in the database schema.

        Args:
            tables_info: List of table information

        Returns:
            List of inferred relationships with confidence levels
        """
        inferred: List[InferredRelationship] = []
        existing_fks: Set[Tuple[str, str]] = set()

        # Build set of existing declared FKs to avoid duplicates
        for table in tables_info:
            for fk in table.foreign_keys:
                existing_fks.add((table.name.lower(), fk['column'].lower()))

        # Build table name variations for matching (include singular/plural forms)
        table_name_variations: Dict[str, TableInfo] = {}
        for t in tables_info:
            name_lower = t.name.lower()
            table_name_variations[name_lower] = t

            # Add singular forms for matching in column names
            if name_lower.endswith('ies'):
                table_name_variations[name_lower[:-3] + 'y'] = t
            elif name_lower.endswith('es') and len(name_lower) > 2:
                table_name_variations[name_lower[:-2]] = t
            elif name_lower.endswith('s') and len(name_lower) > 1:
                table_name_variations[name_lower[:-1]] = t

        # Sort by length (longer first) to match most specific names first
        sorted_variations = sorted(
            table_name_variations.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        logger.debug(f"FK inference: Table name variations: {list(table_name_variations.keys())}")

        for table in tables_info:
            for col in table.columns:
                # Skip if already has declared FK
                if (table.name.lower(), col.name.lower()) in existing_fks:
                    continue

                # Skip primary key columns (they reference themselves)
                if col.is_primary_key:
                    continue

                col_lower = col.name.lower()
                found_match = False

                # Log columns that look like potential FKs
                if 'id' in col_lower and not col.is_primary_key:
                    logger.debug(f"FK inference: Checking column {table.name}.{col.name}")

                # First try: look for embedded table names (or variations) in the column
                for target_name, target_table in sorted_variations:
                    if target_table.name.lower() == table.name.lower():
                        continue  # Skip self-references

                    # Skip very short table names (too many false positives)
                    if len(target_name) < 4:
                        continue

                    # Check if table name variation is embedded in column name
                    # Pattern: <prefix><tablename>id or <prefix><tablename>_id or <tablename>_sk
                    if target_name in col_lower:
                        # Check if followed by 'id', '_id', 'sk', '_sk' or empty
                        idx = col_lower.find(target_name)
                        suffix = col_lower[idx + len(target_name):]
                        if suffix in ['id', '_id', 'sk', '_sk', '']:
                            target_pk = target_table.primary_keys
                            target_column = target_pk[0] if target_pk else 'id'

                            confidence = self._calculate_fk_confidence(
                                col, target_table, target_name
                            )

                            inferred.append(InferredRelationship(
                                source_table=table.name,
                                column=col.name,
                                target_table=target_table.name,
                                target_column=target_column,
                                confidence=confidence,
                                pattern_matched='embedded_table_name'
                            ))
                            found_match = True
                            break

                if found_match:
                    continue

                # Second try: use regex patterns for standard FK naming
                for pattern, pattern_name in self.FK_PATTERNS:
                    match = re.match(pattern, col_lower)
                    if match:
                        # Extract the potential table name from the pattern
                        potential_table = match.group(1)

                        # Skip very short extractions (likely false positives)
                        if len(potential_table) < 3:
                            continue

                        # Try to find a matching table
                        target_table = self._find_matching_table(potential_table)

                        if target_table and target_table.name != table.name:
                            # Determine the likely target column (usually PK)
                            target_pk = target_table.primary_keys
                            target_column = target_pk[0] if target_pk else 'id'

                            # Determine confidence level
                            confidence = self._calculate_fk_confidence(
                                col, target_table, potential_table
                            )

                            inferred.append(InferredRelationship(
                                source_table=table.name,
                                column=col.name,
                                target_table=target_table.name,
                                target_column=target_column,
                                confidence=confidence,
                                pattern_matched=pattern_name
                            ))
                            break  # Only match first pattern

        return inferred

    def _calculate_fk_confidence(
        self, column: ColumnInfo, target_table: TableInfo, extracted_name: str
    ) -> str:
        """Calculate confidence level for an inferred FK relationship."""
        score = 0

        # Higher confidence if column type matches target PK type
        target_pk_cols = [c for c in target_table.columns if c.is_primary_key]
        if target_pk_cols:
            target_pk_type = target_pk_cols[0].data_type.lower()
            col_type = column.data_type.lower()
            if 'int' in col_type and 'int' in target_pk_type:
                score += 2
            elif col_type == target_pk_type:
                score += 2

        # Higher confidence if exact table name match (vs plural/singular variation)
        if extracted_name.lower() == target_table.name.lower():
            score += 2
        elif extracted_name.lower() + 's' == target_table.name.lower():
            score += 1

        # Higher confidence for common FK naming patterns
        col_lower = column.name.lower()
        if col_lower.endswith('id') or col_lower.endswith('_id'):
            score += 1

        if score >= 4:
            return 'high'
        elif score >= 2:
            return 'medium'
        return 'low'

    def _detect_denormalized_fields(
        self, tables_info: List[TableInfo]
    ) -> List[DenormalizedField]:
        """Detect likely denormalized text fields that duplicate data from other tables.

        Denormalized fields store redundant copies of data (like customer names in orders)
        which can lead to data inconsistency. This detects such patterns.

        Args:
            tables_info: List of table information

        Returns:
            List of detected denormalized fields
        """
        denormalized: List[DenormalizedField] = []

        # Build table name variations for matching (singular and plural)
        table_name_variations: Dict[str, str] = {}
        for t in tables_info:
            name_lower = t.name.lower()
            table_name_variations[name_lower] = t.name

            # Add singular forms
            if name_lower.endswith('ies'):
                table_name_variations[name_lower[:-3] + 'y'] = t.name
            elif name_lower.endswith('es'):
                table_name_variations[name_lower[:-2]] = t.name
            elif name_lower.endswith('s'):
                table_name_variations[name_lower[:-1]] = t.name

        for table in tables_info:
            for col in table.columns:
                col_lower = col.name.lower()
                col_type = col.data_type.lower()

                # Only consider text/varchar columns
                if not any(t in col_type for t in ['char', 'text', 'string', 'varchar']):
                    continue

                # Check if column name contains another table name (or variation)
                for ref_name, original_table_name in table_name_variations.items():
                    # Skip self-references and very short names
                    if original_table_name.lower() == table.name.lower():
                        continue
                    if len(ref_name) < 4:
                        continue

                    # Check if table name variation is embedded in column name
                    if ref_name in col_lower:
                        # Also check for common denorm suffixes or if column equals ref_name
                        has_denorm_suffix = any(
                            suffix in col_lower for suffix in self.DENORM_SUFFIXES
                        )

                        # Check if this looks like a denormalized name field
                        # (column contains table ref + optionally a text suffix)
                        suffix_after_ref = col_lower[col_lower.find(ref_name) + len(ref_name):]
                        is_denorm_pattern = (
                            has_denorm_suffix or
                            col_lower == ref_name or
                            suffix_after_ref == '' or
                            suffix_after_ref in ['name', 'title', 'desc', 'description']
                        )

                        if is_denorm_pattern:
                            denormalized.append(DenormalizedField(
                                table=table.name,
                                column=col.name,
                                likely_source_table=original_table_name,
                                data_type=col.data_type,
                                warning=(
                                    f"Column '{col.name}' appears to store denormalized data "
                                    f"from '{original_table_name}'. Consider joining to source "
                                    f"table instead of storing duplicate text."
                                )
                            ))
                            break  # Found a match, no need to check more variations

        return denormalized

    def _add_inferred_relationship_to_ontology(
        self, inferred_rel: InferredRelationship
    ) -> None:
        """Add an inferred relationship to the ontology with appropriate annotations.

        Args:
            inferred_rel: The inferred relationship to add
        """
        source_table_uri = self.base_uri[self._clean_name(inferred_rel.source_table)]
        target_table_uri = self.base_uri[self._clean_name(inferred_rel.target_table)]

        # Create relationship name
        rel_name = (
            f"{self._clean_name(inferred_rel.source_table)}_has_"
            f"{self._clean_name(inferred_rel.target_table)}"
        )
        prop_uri = self.base_uri[rel_name]

        # Check if relationship already exists
        if (prop_uri, RDF.type, OWL.ObjectProperty) in self.graph:
            return  # Already exists, skip

        self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
        self.graph.add((prop_uri, RDFS.domain, source_table_uri))
        self.graph.add((prop_uri, RDFS.range, target_table_uri))
        self.graph.add((prop_uri, RDFS.label, Literal(
            f"{inferred_rel.source_table} has {inferred_rel.target_table}"
        )))

        # Add database-specific annotations
        self.graph.add((prop_uri, self.db_ns.foreignKeyColumn, Literal(inferred_rel.column)))
        self.graph.add((prop_uri, self.db_ns.referencedTable, Literal(inferred_rel.target_table)))
        self.graph.add((prop_uri, self.db_ns.referencedColumn, Literal(inferred_rel.target_column)))

        # Add SQL join condition
        join_condition = (
            f"{inferred_rel.source_table}.{inferred_rel.column} = "
            f"{inferred_rel.target_table}.{inferred_rel.target_column}"
        )
        self.graph.add((prop_uri, self.db_ns.sqlJoinCondition, Literal(join_condition)))

        # Mark as inferred (not declared FK)
        self.graph.add((prop_uri, self.db_ns.isInferredRelationship, Literal(True)))
        self.graph.add((prop_uri, self.db_ns.inferenceConfidence, Literal(inferred_rel.confidence)))
        self.graph.add((prop_uri, self.db_ns.inferencePattern, Literal(inferred_rel.pattern_matched)))
        self.graph.add((prop_uri, self.db_ns.relationshipType, Literal("many_to_one")))

        # Add inverse relationship
        inverse_rel_name = (
            f"{self._clean_name(inferred_rel.target_table)}_referenced_by_"
            f"{self._clean_name(inferred_rel.source_table)}"
        )
        inverse_prop_uri = self.base_uri[inverse_rel_name]

        if (inverse_prop_uri, RDF.type, OWL.ObjectProperty) not in self.graph:
            self.graph.add((inverse_prop_uri, RDF.type, OWL.ObjectProperty))
            self.graph.add((inverse_prop_uri, RDFS.domain, target_table_uri))
            self.graph.add((inverse_prop_uri, RDFS.range, source_table_uri))
            self.graph.add((inverse_prop_uri, RDFS.label, Literal(
                f"{inferred_rel.target_table} referenced by {inferred_rel.source_table}"
            )))
            self.graph.add((inverse_prop_uri, self.db_ns.relationshipType, Literal("one_to_many")))
            self.graph.add((inverse_prop_uri, self.db_ns.isInferredRelationship, Literal(True)))

            # Link as inverses
            self.graph.add((prop_uri, OWL.inverseOf, inverse_prop_uri))

    def _add_denormalized_field_annotation(self, denorm: DenormalizedField) -> None:
        """Add annotation to a denormalized field in the ontology.

        Args:
            denorm: The denormalized field info
        """
        prop_name = f"{self._clean_name(denorm.table)}_{self._clean_name(denorm.column)}"
        prop_uri = self.base_uri[prop_name]

        if (prop_uri, RDF.type, OWL.DatatypeProperty) in self.graph:
            self.graph.add((prop_uri, self.db_ns.isDenormalized, Literal(True)))
            self.graph.add((
                prop_uri,
                self.db_ns.likelySourceTable,
                Literal(denorm.likely_source_table)
            ))
            self.graph.add((
                prop_uri,
                self.db_ns.denormalizationWarning,
                Literal(denorm.warning)
            ))

    def validate_enrichment_completeness(self) -> Dict[str, Any]:
        """Check that all classes and properties have semantic descriptions.

        Returns:
            Dictionary with completeness statistics and lists of unenriched items
        """
        classes_without_desc = []
        properties_without_desc = []
        relationships_without_desc = []
        total_classes = 0
        total_props = 0
        total_rels = 0

        # Check classes (OWL Classes that have db:tableName annotation)
        for subject in self.graph.subjects(RDF.type, OWL.Class):
            # Skip the OWL.Class itself and anonymous nodes
            if subject == OWL.Class or not isinstance(subject, URIRef):
                continue

            # Only count classes that have db:tableName (i.e., actual table classes)
            table_names = list(self.graph.objects(subject, self.db_ns.tableName))
            if not table_names:
                continue

            total_classes += 1
            has_comment = any(self.graph.objects(subject, RDFS.comment))
            if not has_comment:
                classes_without_desc.append(str(table_names[0]))

        # Check datatype properties
        for subject in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            if not isinstance(subject, URIRef):
                continue

            total_props += 1
            has_comment = any(self.graph.objects(subject, RDFS.comment))
            if not has_comment:
                table_name = None
                col_name = None
                for tn in self.graph.objects(subject, self.db_ns.tableName):
                    table_name = str(tn)
                for cn in self.graph.objects(subject, self.db_ns.columnName):
                    col_name = str(cn)
                if table_name and col_name:
                    properties_without_desc.append(f"{table_name}.{col_name}")

        # Check object properties
        for subject in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            if not isinstance(subject, URIRef):
                continue

            total_rels += 1
            has_comment = any(self.graph.objects(subject, RDFS.comment))
            # Also check for relationship description
            has_rel_desc = any(self.graph.objects(subject, self.db_ns.relationshipDescription))
            if not has_comment and not has_rel_desc:
                for label in self.graph.objects(subject, RDFS.label):
                    relationships_without_desc.append(str(label))
                    break

        return {
            "total_classes": total_classes,
            "classes_without_description": classes_without_desc,
            "class_coverage": (
                (total_classes - len(classes_without_desc)) / max(1, total_classes)
            ),
            "total_properties": total_props,
            "properties_without_description": properties_without_desc,
            "property_coverage": (
                (total_props - len(properties_without_desc)) / max(1, total_props)
            ),
            "total_relationships": total_rels,
            "relationships_without_description": relationships_without_desc,
            "relationship_coverage": (
                (total_rels - len(relationships_without_desc)) / max(1, total_rels)
            ),
            "overall_coverage": (
                (total_classes + total_props + total_rels -
                 len(classes_without_desc) - len(properties_without_desc) -
                 len(relationships_without_desc)) /
                max(1, total_classes + total_props + total_rels)
            )
        }

    def _map_sql_to_xsd(
        self, sql_type: str, column_name: str = "", table_name: str = ""
    ) -> Tuple[Optional[URIRef], Optional[Dict[str, str]]]:
        """Map SQL data types to XSD Schema datatypes with semantic awareness.

        This method considers column naming patterns to make smarter type decisions.
        For example, columns named 'quantity' should be integers even if stored as DOUBLE.

        Args:
            sql_type: The SQL data type string
            column_name: Optional column name for semantic type inference
            table_name: Optional table name for context

        Returns:
            Tuple of (XSD type URI, optional type override info dict)
        """
        sql_type_lower = sql_type.lower()
        col_lower = column_name.lower() if column_name else ""
        type_override = None

        # Semantic override: quantities should be integers even if stored as float/double
        if col_lower and any(p in col_lower for p in self.QUANTITY_PATTERNS):
            if any(t in sql_type_lower for t in ["float", "double", "decimal", "numeric", "real"]):
                type_override = {
                    "table": table_name,
                    "column": column_name,
                    "original_sql_type": sql_type,
                    "original_xsd_type": "xsd:double",
                    "overridden_xsd_type": "xsd:integer",
                    "reason": f"Column name '{column_name}' indicates quantity/count (should be integer)"
                }
                return XSD.integer, type_override

        # Integer types - check tinyint first before checking for "int"
        if "tinyint" in sql_type_lower:
            return XSD.byte, None
        if any(t in sql_type_lower for t in ["int", "serial", "bigint", "smallint"]):
            return XSD.integer, None

        # String types
        if any(t in sql_type_lower for t in ["char", "text", "varchar", "string"]):
            return XSD.string, None
        if "clob" in sql_type_lower or "blob" in sql_type_lower:
            return XSD.string, None

        # Temporal types
        if "timestamp" in sql_type_lower or "datetime" in sql_type_lower:
            return XSD.dateTime, None
        if sql_type_lower.startswith("date"):
            return XSD.date, None
        if sql_type_lower.startswith("time"):
            return XSD.time, None

        # Numeric types
        if any(t in sql_type_lower for t in ["float", "real"]):
            return XSD.float, None
        if any(t in sql_type_lower for t in ["double", "double precision"]):
            return XSD.double, None
        if any(t in sql_type_lower for t in ["decimal", "numeric", "number", "money"]):
            return XSD.decimal, None

        # Boolean types
        if any(t in sql_type_lower for t in ["bool", "boolean", "bit"]):
            return XSD.boolean, None

        # Binary types
        if any(t in sql_type_lower for t in ["binary", "varbinary", "bytea"]):
            return XSD.base64Binary, None

        # UUID types
        if "uuid" in sql_type_lower:
            return XSD.string, None

        # JSON types
        if any(t in sql_type_lower for t in ["json", "jsonb"]):
            return XSD.string, None

        # Default to string for unknown types
        logger.warning(f"Unknown SQL type '{sql_type}', mapping to xsd:string")
        return XSD.string, None

    def apply_semantic_descriptions(self, descriptions: Dict[str, Any]):
        """Apply LLM-generated semantic descriptions to the ontology.
        
        This method allows applying rich, context-aware descriptions generated
        by the LLM through the generate_semantic_descriptions tool.
        
        Args:
            descriptions: Dictionary containing semantic descriptions for:
                - tables: Business descriptions for each table
                - columns: Business meanings for each column
                - relationships: Descriptions of foreign key relationships
        """
        logger.info("Applying LLM-generated semantic descriptions to ontology")
        
        # Apply table descriptions
        if "tables" in descriptions:
            for table_name, table_desc in descriptions["tables"].items():
                table_uri = self.base_uri[self._clean_name(table_name)]
                if (table_uri, RDF.type, OWL.Class) in self.graph:
                    if "business_description" in table_desc:
                        # Remove existing comment if present
                        self.graph.remove((table_uri, RDFS.comment, None))
                        # Add new rich description as rdfs:comment
                        self.graph.add((table_uri, RDFS.comment,
                                      Literal(table_desc["business_description"])))
                    
                    if "table_type" in table_desc:
                        self.graph.add((table_uri, self.db_ns.tableType, 
                                      Literal(table_desc["table_type"])))
                    
                    if "usage_notes" in table_desc:
                        self.graph.add((table_uri, self.db_ns.usageNotes, 
                                      Literal(table_desc["usage_notes"])))
        
        # Apply column descriptions
        if "columns" in descriptions:
            for column_ref, column_desc in descriptions["columns"].items():
                # Parse table.column format
                if "." in column_ref:
                    table_name, column_name = column_ref.split(".", 1)
                    prop_name = f"{self._clean_name(table_name)}_{self._clean_name(column_name)}"
                    prop_uri = self.base_uri[prop_name]
                    
                    if ((prop_uri, RDF.type, OWL.DatatypeProperty) in self.graph or
                        (prop_uri, RDF.type, OWL.ObjectProperty) in self.graph):
                        
                        if "business_description" in column_desc:
                            # Remove existing comment if present
                            self.graph.remove((prop_uri, RDFS.comment, None))
                            # Add new rich description as rdfs:comment
                            self.graph.add((prop_uri, RDFS.comment,
                                          Literal(column_desc["business_description"])))
                        
                        if "data_characteristics" in column_desc:
                            self.graph.add((prop_uri, self.db_ns.dataCharacteristics, 
                                          Literal(column_desc["data_characteristics"])))
                        
                        if "business_rules" in column_desc:
                            self.graph.add((prop_uri, self.db_ns.businessRules, 
                                          Literal(column_desc["business_rules"])))
        
        # Apply relationship descriptions
        if "relationships" in descriptions:
            for rel_key, rel_desc in descriptions["relationships"].items():
                # Parse relationship key format
                # Expected format: "from_table.column -> to_table.column"
                if " -> " in rel_key:
                    from_part, to_part = rel_key.split(" -> ")
                    from_table = from_part.split(".")[0] if "." in from_part else from_part
                    to_table = to_part.split(".")[0] if "." in to_part else to_part
                    
                    rel_name = f"{self._clean_name(from_table)}_has_{self._clean_name(to_table)}"
                    rel_uri = self.base_uri[rel_name]
                    
                    if (rel_uri, RDF.type, OWL.ObjectProperty) in self.graph:
                        if "description" in rel_desc:
                            self.graph.add((rel_uri, self.db_ns.relationshipDescription, 
                                          Literal(rel_desc["description"])))
                        
                        if "cardinality" in rel_desc:
                            self.graph.add((rel_uri, self.db_ns.cardinality, 
                                          Literal(rel_desc["cardinality"])))
                        
                        if "business_rule" in rel_desc:
                            self.graph.add((rel_uri, self.db_ns.businessRule, 
                                          Literal(rel_desc["business_rule"])))
        
        logger.info("Finished applying semantic descriptions")
    
    def serialize_ontology(self) -> str:
        """Serialize the current ontology to Turtle format."""
        return self.graph.serialize(format="turtle")

    def get_enrichment_data(self, tables_info: List[TableInfo], sample_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate enrichment data structure for LLM processing.

        Args:
            tables_info: List of table information
            sample_data: Dictionary mapping table names to sample rows

        Returns:
            Dictionary with schema_data and instructions for LLM enrichment
        """
        schema_data = []

        for table_info in tables_info:
            table_data = {
                "table_name": table_info.name,
                "schema": table_info.schema,
                "row_count": table_info.row_count,
                "columns": []
            }

            # Add column information
            for column in table_info.columns:
                column_data = {
                    "name": column.name,
                    "data_type": column.data_type,
                    "is_nullable": column.is_nullable,
                    "is_primary_key": column.is_primary_key,
                    "is_foreign_key": column.is_foreign_key
                }
                if column.comment:
                    column_data["comment"] = column.comment
                if column.is_foreign_key and column.foreign_key_table:
                    column_data["foreign_key_table"] = column.foreign_key_table
                    column_data["foreign_key_column"] = column.foreign_key_column
                table_data["columns"].append(column_data)

            # Add sample data if available (limit to 3 rows)
            if table_info.name in sample_data and sample_data[table_info.name]:
                table_data["sample_data"] = sample_data[table_info.name][:3]

            schema_data.append(table_data)

        # Generate instructions for LLM enrichment
        instructions = {
            "task": "Enrich the database schema with semantic descriptions",
            "expected_format": {
                "classes": [
                    {
                        "original_name": "table_name",
                        "suggested_name": "SemanticName",
                        "description": "Business description"
                    }
                ],
                "properties": [
                    {
                        "table_name": "table_name",
                        "original_name": "column_name",
                        "suggested_name": "semanticPropertyName",
                        "description": "Business meaning"
                    }
                ],
                "relationships": [
                    {
                        "from_table": "source_table",
                        "to_table": "target_table",
                        "suggested_name": "semanticRelationshipName",
                        "description": "Relationship meaning"
                    }
                ]
            },
            "guidelines": [
                "Use clear, business-oriented terminology",
                "Provide meaningful descriptions based on table and column names and sample data",
                "Suggest appropriate semantic names that reflect business concepts",
                "For relationships, describe the business meaning of the association"
            ]
        }

        return {
            "schema_data": schema_data,
            "instructions": instructions
        }

    def apply_enrichment(self, enrichment_suggestions: Dict[str, List[Dict[str, Any]]]) -> None:
        """Apply enrichment suggestions to the ontology.

        Args:
            enrichment_suggestions: Dictionary containing enrichment suggestions for:
                - classes: Class-level enrichments (table names, descriptions)
                - properties: Property-level enrichments (column names, descriptions)
                - relationships: Relationship enrichments (foreign key descriptions)
        """
        logger.info("Applying enrichment suggestions to ontology")

        # Apply class enrichments
        if "classes" in enrichment_suggestions:
            for class_enrichment in enrichment_suggestions["classes"]:
                original_name = class_enrichment.get("original_name")
                suggested_name = class_enrichment.get("suggested_name")
                description = class_enrichment.get("description")

                if not original_name:
                    continue

                # Find the original class URI
                original_uri = self.base_uri[self._clean_name(original_name)]

                # Add suggested name as label if provided
                if suggested_name:
                    # Remove old label if exists
                    self.graph.remove((original_uri, RDFS.label, None))
                    self.graph.add((original_uri, RDFS.label, Literal(suggested_name)))

                # Add description as comment if provided
                if description:
                    # Remove old comment if exists
                    self.graph.remove((original_uri, RDFS.comment, None))
                    self.graph.add((original_uri, RDFS.comment, Literal(description)))

        # Apply property enrichments
        if "properties" in enrichment_suggestions:
            for prop_enrichment in enrichment_suggestions["properties"]:
                table_name = prop_enrichment.get("table_name")
                original_name = prop_enrichment.get("original_name")
                suggested_name = prop_enrichment.get("suggested_name")
                description = prop_enrichment.get("description")

                if not table_name or not original_name:
                    continue

                # Find the original property URI
                prop_name = f"{self._clean_name(table_name)}_{self._clean_name(original_name)}"
                prop_uri = self.base_uri[prop_name]

                # Add suggested name as label if provided
                if suggested_name:
                    self.graph.remove((prop_uri, RDFS.label, None))
                    self.graph.add((prop_uri, RDFS.label, Literal(suggested_name)))

                # Add description as comment if provided
                if description:
                    self.graph.remove((prop_uri, RDFS.comment, None))
                    self.graph.add((prop_uri, RDFS.comment, Literal(description)))

        # Apply relationship enrichments
        if "relationships" in enrichment_suggestions:
            for rel_enrichment in enrichment_suggestions["relationships"]:
                from_table = rel_enrichment.get("from_table")
                to_table = rel_enrichment.get("to_table")
                suggested_name = rel_enrichment.get("suggested_name")
                description = rel_enrichment.get("description")

                if not from_table or not to_table:
                    continue

                # Find the original relationship URI
                rel_name = f"{self._clean_name(from_table)}_has_{self._clean_name(to_table)}"
                rel_uri = self.base_uri[rel_name]

                # Add suggested name as label if provided
                if suggested_name:
                    self.graph.remove((rel_uri, RDFS.label, None))
                    self.graph.add((rel_uri, RDFS.label, Literal(suggested_name)))

                # Add description as comment if provided
                if description:
                    self.graph.remove((rel_uri, RDFS.comment, None))
                    self.graph.add((rel_uri, RDFS.comment, Literal(description)))

        logger.info("Finished applying enrichment suggestions")

    def enrich_with_llm(self, tables_info: List[TableInfo], sample_data: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate basic ontology with optional LLM enrichment.

        This is a placeholder method that generates a basic ontology.
        LLM enrichment is handled by the MCP tools layer.

        Args:
            tables_info: List of table information
            sample_data: Sample data for tables

        Returns:
            Serialized ontology in Turtle format
        """
        logger.info("Generating basic ontology (LLM enrichment handled by MCP tools)")
        return self.generate_from_schema(tables_info)

    def extract_names_for_review(self) -> Dict[str, Any]:
        """Extract all class, property, and relationship names from the ontology for LLM review.

        This method analyzes the current ontology graph and extracts all names that might
        need improvement - abbreviations, cryptic identifiers, or technical names that
        could be made more business-friendly.

        Returns:
            Dictionary containing:
            - classes: List of class info with original names, labels, and metadata
            - properties: List of property info with original names and context
            - relationships: List of relationship info
            - analysis_hints: Patterns detected that suggest names need improvement
        """
        classes = []
        properties = []
        relationships = []
        analysis_hints = []

        # Extract classes (tables)
        for subject in self.graph.subjects(RDF.type, OWL.Class):
            if subject == OWL.Class:
                continue

            class_info = {
                "uri": str(subject),
                "local_name": str(subject).split("/")[-1] if "/" in str(subject) else str(subject),
                "current_label": None,
                "table_name": None,
                "schema_name": None,
                "row_count": None,
                "comment": None
            }

            # Get current label
            for label in self.graph.objects(subject, RDFS.label):
                class_info["current_label"] = str(label)

            # Get database annotations
            for table_name in self.graph.objects(subject, self.db_ns.tableName):
                class_info["table_name"] = str(table_name)
            for schema_name in self.graph.objects(subject, self.db_ns.schemaName):
                class_info["schema_name"] = str(schema_name)
            for row_count in self.graph.objects(subject, self.db_ns.rowCount):
                class_info["row_count"] = int(row_count)
            for comment in self.graph.objects(subject, RDFS.comment):
                class_info["comment"] = str(comment)

            # Analyze if name looks cryptic
            name = class_info["current_label"] or class_info["local_name"]
            class_info["needs_review"] = self._analyze_name_quality(name)

            classes.append(class_info)

        # Extract data properties (columns)
        for subject in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            prop_info = {
                "uri": str(subject),
                "local_name": str(subject).split("/")[-1] if "/" in str(subject) else str(subject),
                "current_label": None,
                "column_name": None,
                "table_name": None,
                "sql_data_type": None,
                "is_primary_key": False,
                "is_foreign_key": False,
                "comment": None
            }

            # Get current label
            for label in self.graph.objects(subject, RDFS.label):
                prop_info["current_label"] = str(label)

            # Get database annotations
            for col_name in self.graph.objects(subject, self.db_ns.columnName):
                prop_info["column_name"] = str(col_name)
            for table_name in self.graph.objects(subject, self.db_ns.tableName):
                prop_info["table_name"] = str(table_name)
            for sql_type in self.graph.objects(subject, self.db_ns.sqlDataType):
                prop_info["sql_data_type"] = str(sql_type)
            for is_pk in self.graph.objects(subject, self.db_ns.isPrimaryKey):
                prop_info["is_primary_key"] = str(is_pk).lower() == "true"
            for is_fk in self.graph.objects(subject, self.db_ns.isForeignKey):
                prop_info["is_foreign_key"] = str(is_fk).lower() == "true"
            for comment in self.graph.objects(subject, RDFS.comment):
                prop_info["comment"] = str(comment)

            # Analyze if name looks cryptic
            name = prop_info["current_label"] or prop_info["local_name"]
            prop_info["needs_review"] = self._analyze_name_quality(name)

            properties.append(prop_info)

        # Extract object properties (relationships)
        for subject in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            rel_info = {
                "uri": str(subject),
                "local_name": str(subject).split("/")[-1] if "/" in str(subject) else str(subject),
                "current_label": None,
                "foreign_key_column": None,
                "referenced_table": None,
                "relationship_type": None,
                "comment": None
            }

            # Get current label
            for label in self.graph.objects(subject, RDFS.label):
                rel_info["current_label"] = str(label)

            # Get database annotations
            for fk_col in self.graph.objects(subject, self.db_ns.foreignKeyColumn):
                rel_info["foreign_key_column"] = str(fk_col)
            for ref_table in self.graph.objects(subject, self.db_ns.referencedTable):
                rel_info["referenced_table"] = str(ref_table)
            for rel_type in self.graph.objects(subject, self.db_ns.relationshipType):
                rel_info["relationship_type"] = str(rel_type)
            for comment in self.graph.objects(subject, RDFS.comment):
                rel_info["comment"] = str(comment)

            # Analyze if name looks cryptic
            name = rel_info["current_label"] or rel_info["local_name"]
            rel_info["needs_review"] = self._analyze_name_quality(name)

            relationships.append(rel_info)

        # Generate analysis hints
        cryptic_classes = [c for c in classes if c.get("needs_review", {}).get("is_cryptic")]
        cryptic_props = [p for p in properties if p.get("needs_review", {}).get("is_cryptic")]
        cryptic_rels = [r for r in relationships if r.get("needs_review", {}).get("is_cryptic")]

        if cryptic_classes:
            analysis_hints.append(f"Found {len(cryptic_classes)} class names that may need improvement")
        if cryptic_props:
            analysis_hints.append(f"Found {len(cryptic_props)} property names that may need improvement")
        if cryptic_rels:
            analysis_hints.append(f"Found {len(cryptic_rels)} relationship names that may need improvement")

        return {
            "classes": classes,
            "properties": properties,
            "relationships": relationships,
            "analysis_hints": analysis_hints,
            "summary": {
                "total_classes": len(classes),
                "total_properties": len(properties),
                "total_relationships": len(relationships),
                "classes_needing_review": len(cryptic_classes),
                "properties_needing_review": len(cryptic_props),
                "relationships_needing_review": len(cryptic_rels)
            }
        }

    def _analyze_name_quality(self, name: str) -> Dict[str, Any]:
        """Analyze if a name looks like an abbreviation or cryptic identifier.

        Args:
            name: The name to analyze

        Returns:
            Dictionary with analysis results
        """
        if not name:
            return {"is_cryptic": True, "reasons": ["Empty name"]}

        reasons = []
        is_cryptic = False

        # Check for very short names (likely abbreviations)
        if len(name) <= 3:
            is_cryptic = True
            reasons.append("Very short name (≤3 chars) - likely abbreviation")

        # Check for all uppercase (common for abbreviations)
        if name.isupper() and len(name) > 1:
            is_cryptic = True
            reasons.append("All uppercase - likely acronym")

        # Check for underscore-separated abbreviations (e.g., cust_id, ord_dt)
        parts = name.split("_")
        short_parts = [p for p in parts if len(p) <= 3 and p.isalpha()]
        if len(short_parts) > len(parts) / 2:
            is_cryptic = True
            reasons.append("Contains multiple abbreviations")

        # Check for common cryptic patterns
        cryptic_patterns = [
            (r"_id$", "Ends with '_id' - consider more descriptive name"),
            (r"_dt$", "Ends with '_dt' (date abbreviation)"),
            (r"_cd$", "Ends with '_cd' (code abbreviation)"),
            (r"_no$", "Ends with '_no' (number abbreviation)"),
            (r"_nm$", "Ends with '_nm' (name abbreviation)"),
            (r"_amt$", "Ends with '_amt' (amount abbreviation)"),
            (r"_qty$", "Ends with '_qty' (quantity abbreviation)"),
            (r"_flg$", "Ends with '_flg' (flag abbreviation)"),
            (r"_ind$", "Ends with '_ind' (indicator abbreviation)"),
            (r"_num$", "Ends with '_num' (number abbreviation)"),
            (r"_cnt$", "Ends with '_cnt' (count abbreviation)"),
            (r"_desc$", "Ends with '_desc' (description abbreviation)"),
            (r"_typ$", "Ends with '_typ' (type abbreviation)"),
            (r"_cat$", "Ends with '_cat' (category abbreviation)"),
            (r"_sts$", "Ends with '_sts' (status abbreviation)"),
            (r"^pk_", "Starts with 'pk_' (primary key prefix)"),
            (r"^fk_", "Starts with 'fk_' (foreign key prefix)"),
            (r"^tbl_", "Starts with 'tbl_' (table prefix)"),
            (r"^vw_", "Starts with 'vw_' (view prefix)"),
        ]

        for pattern, reason in cryptic_patterns:
            if re.search(pattern, name.lower()):
                is_cryptic = True
                reasons.append(reason)

        # Check for numeric suffixes that might indicate versions or partitions
        if re.search(r"\d+$", name):
            reasons.append("Contains numeric suffix")

        # Check for mixed case that looks like system-generated names
        if re.match(r"^[a-z]+[A-Z]", name) and "_" not in name:
            # camelCase is OK, but some system names are cryptic camelCase
            pass

        return {
            "is_cryptic": is_cryptic,
            "reasons": reasons,
            "confidence": "high" if len(reasons) >= 2 else "medium" if len(reasons) == 1 else "low"
        }

    def apply_semantic_names(self, name_suggestions: Dict[str, Any]) -> str:
        """Apply suggested semantic names to the ontology.

        This method takes LLM-generated name suggestions and updates the ontology
        labels to use more business-friendly terminology.

        Args:
            name_suggestions: Dictionary containing:
                - classes: List of {original_name, suggested_name, description}
                - properties: List of {original_name, suggested_name, description}
                - relationships: List of {original_name, suggested_name, description}

        Returns:
            Updated ontology in Turtle format
        """
        logger.info("Applying semantic name suggestions to ontology")
        changes_made = 0

        # Apply class name suggestions
        if "classes" in name_suggestions:
            for suggestion in name_suggestions["classes"]:
                original = suggestion.get("original_name")
                suggested = suggestion.get("suggested_name")
                description = suggestion.get("description")

                if not original:
                    continue

                # Find the class URI
                class_uri = self.base_uri[self._clean_name(original)]

                # Check if this class exists
                if (class_uri, RDF.type, OWL.Class) in self.graph:
                    if suggested:
                        # Update the label
                        self.graph.remove((class_uri, RDFS.label, None))
                        self.graph.add((class_uri, RDFS.label, Literal(suggested)))
                        # Also add a semantic name annotation
                        self.graph.add((class_uri, self.db_ns.semanticName, Literal(suggested)))
                        changes_made += 1

                    if description:
                        # Add or update description as rdfs:comment
                        self.graph.remove((class_uri, RDFS.comment, None))
                        self.graph.add((class_uri, RDFS.comment, Literal(description)))

        # Apply property name suggestions
        if "properties" in name_suggestions:
            for suggestion in name_suggestions["properties"]:
                original = suggestion.get("original_name")
                suggested = suggestion.get("suggested_name")
                description = suggestion.get("description")
                table_name = suggestion.get("table_name")

                if not original:
                    continue

                # Find the property URI - might need table context
                if table_name:
                    prop_name = f"{self._clean_name(table_name)}_{self._clean_name(original)}"
                else:
                    prop_name = self._clean_name(original)

                prop_uri = self.base_uri[prop_name]

                # Check if this property exists (as data or object property)
                if ((prop_uri, RDF.type, OWL.DatatypeProperty) in self.graph or
                    (prop_uri, RDF.type, OWL.ObjectProperty) in self.graph):
                    if suggested:
                        # Update the label
                        self.graph.remove((prop_uri, RDFS.label, None))
                        self.graph.add((prop_uri, RDFS.label, Literal(suggested)))
                        # Also add a semantic name annotation
                        self.graph.add((prop_uri, self.db_ns.semanticName, Literal(suggested)))
                        changes_made += 1

                    if description:
                        # Add or update description as rdfs:comment
                        self.graph.remove((prop_uri, RDFS.comment, None))
                        self.graph.add((prop_uri, RDFS.comment, Literal(description)))

        # Apply relationship name suggestions
        if "relationships" in name_suggestions:
            for suggestion in name_suggestions["relationships"]:
                original = suggestion.get("original_name")
                suggested = suggestion.get("suggested_name")
                description = suggestion.get("description")

                if not original:
                    continue

                # Find the relationship URI
                rel_uri = self.base_uri[self._clean_name(original)]

                # Check if this relationship exists
                if (rel_uri, RDF.type, OWL.ObjectProperty) in self.graph:
                    if suggested:
                        # Update the label
                        self.graph.remove((rel_uri, RDFS.label, None))
                        self.graph.add((rel_uri, RDFS.label, Literal(suggested)))
                        # Also add a semantic name annotation
                        self.graph.add((rel_uri, self.db_ns.semanticName, Literal(suggested)))
                        changes_made += 1

                    if description:
                        # Add or update description as rdfs:comment
                        self.graph.remove((rel_uri, RDFS.comment, None))
                        self.graph.add((rel_uri, RDFS.comment, Literal(description)))

        logger.info(f"Applied {changes_made} semantic name changes to ontology")

        return self.graph.serialize(format="turtle")




"""Ontology-Based Query Check (OBQC) validator for semantic SQL validation.

This module provides deterministic, rule-based validation of SQL queries against
an RDF/OWL ontology, detecting schema violations, type mismatches, invalid joins,
and fan-trap patterns without using LLM.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

logger = logging.getLogger(__name__)


class OBQCIssueType(Enum):
    """Categories of OBQC validation issues."""

    TABLE_NOT_FOUND = "table_not_found"
    COLUMN_NOT_FOUND = "column_not_found"
    TYPE_MISMATCH = "type_mismatch"
    INVALID_JOIN = "invalid_join"
    MISSING_JOIN_CONDITION = "missing_join_condition"
    FAN_TRAP_DETECTED = "fan_trap_detected"
    AGGREGATION_WITHOUT_GROUP = "aggregation_without_group"
    NON_AGGREGATED_COLUMN = "non_aggregated_column"
    AMBIGUOUS_COLUMN = "ambiguous_column"


class OBQCSeverity(Enum):
    """Severity levels for OBQC issues."""

    ERROR = "error"  # Query will fail or produce incorrect results
    WARNING = "warning"  # Query may have issues or suboptimal patterns
    INFO = "info"  # Informational note about query structure


@dataclass
class OBQCIssue:
    """Single OBQC validation issue."""

    issue_type: OBQCIssueType
    severity: OBQCSeverity
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)


@dataclass
class OBQCResult:
    """Complete OBQC validation result."""

    is_valid: bool
    issues: List[OBQCIssue] = field(default_factory=list)
    parsed_tables: List[str] = field(default_factory=list)
    parsed_columns: List[str] = field(default_factory=list)
    parsed_joins: List[Dict[str, Any]] = field(default_factory=list)
    has_aggregation: bool = False
    has_group_by: bool = False
    fan_trap_risk: bool = False
    ontology_compatible: bool = True  # Whether ontology has db: annotations

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "obqc_valid": self.is_valid,
            "obqc_ontology_compatible": self.ontology_compatible,
            "obqc_issues": [
                {
                    "type": issue.issue_type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "location": issue.location,
                    "suggestion": issue.suggestion,
                    "related_entities": issue.related_entities,
                }
                for issue in self.issues
            ],
            "parsed_tables": self.parsed_tables,
            "parsed_columns": self.parsed_columns,
            "parsed_joins": self.parsed_joins,
            "has_aggregation": self.has_aggregation,
            "has_group_by": self.has_group_by,
            "fan_trap_risk": self.fan_trap_risk,
            "obqc_error_count": sum(
                1 for i in self.issues if i.severity == OBQCSeverity.ERROR
            ),
            "obqc_warning_count": sum(
                1 for i in self.issues if i.severity == OBQCSeverity.WARNING
            ),
        }


@dataclass
class ColumnSchema:
    """Schema information for a single column."""

    name: str
    table_name: str
    sql_data_type: str
    xsd_type: Optional[URIRef] = None
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    fk_referenced_table: Optional[str] = None
    fk_referenced_column: Optional[str] = None


@dataclass
class TableSchema:
    """Schema information for a single table."""

    name: str
    schema_name: str
    columns: Dict[str, ColumnSchema] = field(default_factory=dict)
    primary_keys: List[str] = field(default_factory=list)


@dataclass
class RelationshipInfo:
    """Information about a foreign key relationship."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str  # "many_to_one" or "one_to_many"
    join_condition: str


@dataclass
class OntologySchema:
    """Cached schema information extracted from ontology."""

    tables: Dict[str, TableSchema] = field(default_factory=dict)
    relationships: Dict[str, RelationshipInfo] = field(default_factory=dict)


class OBQCValidator:
    """Ontology-Based Query Check validator.

    Validates SQL queries against an RDF/OWL ontology to detect:
    - Schema violations (missing tables/columns)
    - Type mismatches in comparisons
    - Invalid or missing join conditions
    - Fan-trap patterns with aggregation
    - GROUP BY completeness
    """

    # Dialect mapping for sqlglot
    DIALECT_MAP = {
        "postgresql": "postgres",
        "snowflake": "snowflake",
        "dremio": "trino",  # Dremio uses Trino-compatible syntax
        "clickhouse": "clickhouse",
    }

    def __init__(self) -> None:
        self._schema_cache: Optional[OntologySchema] = None
        self._graph: Optional[Graph] = None
        self._base_uri: Optional[Namespace] = None
        self._db_ns: Optional[Namespace] = None
        self._is_compatible: bool = False  # Whether ontology has db: annotations

    def load_ontology(self, ontology_graph: Graph, base_uri: str) -> None:
        """Load and cache schema from ontology graph.

        Args:
            ontology_graph: The rdflib Graph containing the ontology
            base_uri: The base URI namespace (e.g., "http://example.com/ontology/")
        """
        self._graph = ontology_graph
        self._base_uri = Namespace(base_uri)
        self._db_ns = Namespace(f"{base_uri}db/")
        self._schema_cache = self._extract_schema_from_ontology()

        # Check if ontology has required db: annotations for OBQC
        self._is_compatible = self._check_ontology_compatibility()

        if self._is_compatible:
            logger.info(
                f"OBQC loaded ontology with {len(self._schema_cache.tables)} tables, "
                f"{sum(len(t.columns) for t in self._schema_cache.tables.values())} columns"
            )
        else:
            logger.warning(
                "OBQC: Ontology lacks db: namespace annotations - semantic validation disabled. "
                "Use generate_ontology to create a compatible ontology."
            )

    def _check_ontology_compatibility(self) -> bool:
        """Check if ontology has required db: namespace annotations for OBQC.

        Returns:
            True if ontology has db:tableName annotations, False otherwise
        """
        if self._schema_cache is None:
            return False

        # Ontology is compatible if it has at least one table with db:tableName
        # and at least one column with db:columnName
        has_tables = len(self._schema_cache.tables) > 0
        has_columns = any(
            len(table.columns) > 0 for table in self._schema_cache.tables.values()
        )

        return has_tables and has_columns

    @property
    def is_compatible(self) -> bool:
        """Whether the loaded ontology is compatible with OBQC validation."""
        return self._is_compatible

    def _extract_schema_from_ontology(self) -> OntologySchema:
        """Extract schema information from ontology graph."""
        schema = OntologySchema()

        if self._graph is None or self._db_ns is None:
            return schema

        # Extract tables (owl:Class with db:tableName)
        for subject in self._graph.subjects(RDF.type, OWL.Class):
            if subject == OWL.Class:
                continue
            table_name = self._get_literal(subject, self._db_ns.tableName)
            if table_name:
                schema_name = (
                    self._get_literal(subject, self._db_ns.schemaName) or "public"
                )
                table_schema = TableSchema(name=table_name, schema_name=schema_name)

                # Get primary keys
                for pk in self._graph.objects(subject, self._db_ns.primaryKey):
                    table_schema.primary_keys.append(str(pk))

                schema.tables[table_name.lower()] = table_schema

        # Extract columns (owl:DatatypeProperty with db:columnName)
        for subject in self._graph.subjects(RDF.type, OWL.DatatypeProperty):
            column_name = self._get_literal(subject, self._db_ns.columnName)
            table_name = self._get_literal(subject, self._db_ns.tableName)

            if column_name and table_name:
                table_key = table_name.lower()
                if table_key in schema.tables:
                    col_schema = ColumnSchema(
                        name=column_name,
                        table_name=table_name,
                        sql_data_type=self._get_literal(subject, self._db_ns.sqlDataType)
                        or "VARCHAR",
                        is_nullable=self._get_bool(subject, self._db_ns.isNullable, True),
                        is_primary_key=self._get_bool(
                            subject, self._db_ns.isPrimaryKey, False
                        ),
                        is_foreign_key=self._get_bool(
                            subject, self._db_ns.isForeignKey, False
                        ),
                    )

                    # Get XSD type from rdfs:range
                    for range_val in self._graph.objects(subject, RDFS.range):
                        col_schema.xsd_type = range_val
                        break

                    schema.tables[table_key].columns[column_name.lower()] = col_schema

        # Extract relationships (owl:ObjectProperty with db:foreignKeyColumn)
        for subject in self._graph.subjects(RDF.type, OWL.ObjectProperty):
            fk_column = self._get_literal(subject, self._db_ns.foreignKeyColumn)
            ref_table = self._get_literal(subject, self._db_ns.referencedTable)
            ref_column = self._get_literal(subject, self._db_ns.referencedColumn)
            rel_type = self._get_literal(subject, self._db_ns.relationshipType)
            join_cond = self._get_literal(subject, self._db_ns.sqlJoinCondition)

            if fk_column and ref_table:
                # Determine from_table from domain
                from_table = None
                for domain in self._graph.objects(subject, RDFS.domain):
                    from_table = self._get_literal(domain, self._db_ns.tableName)
                    break

                if from_table:
                    rel_key = f"{from_table}.{fk_column}->{ref_table}.{ref_column}"
                    schema.relationships[rel_key] = RelationshipInfo(
                        from_table=from_table,
                        from_column=fk_column,
                        to_table=ref_table,
                        to_column=ref_column or "id",
                        relationship_type=rel_type or "many_to_one",
                        join_condition=join_cond
                        or f"{from_table}.{fk_column} = {ref_table}.{ref_column}",
                    )

                    # Update column FK info
                    table_key = from_table.lower()
                    col_key = fk_column.lower()
                    if (
                        table_key in schema.tables
                        and col_key in schema.tables[table_key].columns
                    ):
                        col = schema.tables[table_key].columns[col_key]
                        col.is_foreign_key = True
                        col.fk_referenced_table = ref_table
                        col.fk_referenced_column = ref_column

        return schema

    def _get_literal(self, subject: URIRef, predicate: URIRef) -> Optional[str]:
        """Get string value of a literal predicate."""
        if self._graph is None:
            return None
        for obj in self._graph.objects(subject, predicate):
            return str(obj)
        return None

    def _get_bool(self, subject: URIRef, predicate: URIRef, default: bool) -> bool:
        """Get boolean value of a literal predicate."""
        val = self._get_literal(subject, predicate)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes")

    def validate(self, sql_query: str, dialect: str = "postgresql") -> OBQCResult:
        """Validate SQL query against loaded ontology.

        Args:
            sql_query: The SQL query to validate
            dialect: Database dialect ("postgresql", "snowflake", "dremio")

        Returns:
            OBQCResult with validation findings
        """
        result = OBQCResult(is_valid=True)

        if not self._schema_cache:
            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.TABLE_NOT_FOUND,
                    severity=OBQCSeverity.WARNING,
                    message="No ontology loaded - OBQC validation skipped",
                    suggestion="Load ontology using generate_ontology or load_my_ontology",
                )
            )
            return result

        # Check if ontology has required db: namespace annotations
        if not self._is_compatible:
            result.ontology_compatible = False
            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.TABLE_NOT_FOUND,
                    severity=OBQCSeverity.INFO,
                    message="Ontology lacks db: namespace annotations - OBQC validation skipped",
                    suggestion=(
                        "The loaded ontology does not contain db:tableName/db:columnName annotations. "
                        "Use generate_ontology to create a compatible ontology from your database schema."
                    ),
                )
            )
            return result

        # Parse SQL using sqlglot
        try:
            sqlglot_dialect = self.DIALECT_MAP.get(dialect.lower(), "postgres")
            parsed = sqlglot.parse_one(sql_query, dialect=sqlglot_dialect)
        except ParseError as e:
            result.is_valid = False
            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.TABLE_NOT_FOUND,
                    severity=OBQCSeverity.ERROR,
                    message=f"SQL parse error: {str(e)}",
                    location="Query",
                )
            )
            return result

        # Extract query components
        self._extract_tables(parsed, result)
        self._extract_columns(parsed, result)
        self._extract_joins(parsed, result)
        self._extract_aggregations(parsed, result)

        # Run validation rules
        self._validate_tables(result)
        self._validate_columns(result)
        self._validate_joins(result)
        self._validate_type_compatibility(parsed, result)
        self._validate_aggregation_context(parsed, result)
        self._detect_fan_trap(result)

        # Set overall validity
        result.is_valid = not any(
            issue.severity == OBQCSeverity.ERROR for issue in result.issues
        )

        return result

    def _extract_tables(self, parsed: exp.Expression, result: OBQCResult) -> None:
        """Extract all table references from parsed query."""
        for table in parsed.find_all(exp.Table):
            table_name = table.name
            if table_name and table_name not in result.parsed_tables:
                result.parsed_tables.append(table_name)

    def _extract_columns(self, parsed: exp.Expression, result: OBQCResult) -> None:
        """Extract all column references from parsed query."""
        for column in parsed.find_all(exp.Column):
            col_ref = column.name
            if column.table:
                col_ref = f"{column.table}.{column.name}"
            if col_ref and col_ref not in result.parsed_columns:
                result.parsed_columns.append(col_ref)

    def _extract_joins(self, parsed: exp.Expression, result: OBQCResult) -> None:
        """Extract join information from parsed query."""
        for join in parsed.find_all(exp.Join):
            join_info: Dict[str, Any] = {
                "type": join.kind or "INNER",
                "table": None,
                "on_condition": None,
            }

            # Get joined table
            if join.this and isinstance(join.this, exp.Table):
                join_info["table"] = join.this.name

            # Get ON condition
            if join.args.get("on"):
                join_info["on_condition"] = join.args["on"].sql()

            result.parsed_joins.append(join_info)

    def _extract_aggregations(self, parsed: exp.Expression, result: OBQCResult) -> None:
        """Detect aggregate functions and GROUP BY."""
        # Check for aggregate functions
        agg_types = (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)
        result.has_aggregation = any(parsed.find_all(*agg_types))

        # Check for GROUP BY
        for select in parsed.find_all(exp.Select):
            if select.args.get("group"):
                result.has_group_by = True
                break

    def _validate_tables(self, result: OBQCResult) -> None:
        """Rule: Check all referenced tables exist in ontology."""
        if self._schema_cache is None:
            return

        for table_name in result.parsed_tables:
            if table_name.lower() not in self._schema_cache.tables:
                available_tables = list(self._schema_cache.tables.keys())[:10]
                result.issues.append(
                    OBQCIssue(
                        issue_type=OBQCIssueType.TABLE_NOT_FOUND,
                        severity=OBQCSeverity.ERROR,
                        message=f"Table '{table_name}' not found in ontology",
                        location="FROM/JOIN clause",
                        suggestion=f"Available tables: {', '.join(available_tables)}",
                        related_entities=[table_name],
                    )
                )

    def _validate_columns(self, result: OBQCResult) -> None:
        """Rule: Check all referenced columns exist in their respective tables."""
        if self._schema_cache is None:
            return

        for col_ref in result.parsed_columns:
            if "." in col_ref:
                parts = col_ref.split(".", 1)
                table_name, col_name = parts[0], parts[1]
                table_key = table_name.lower()
                col_key = col_name.lower()

                if table_key in self._schema_cache.tables:
                    table_schema = self._schema_cache.tables[table_key]
                    if col_key not in table_schema.columns:
                        available_cols = list(table_schema.columns.keys())[:10]
                        result.issues.append(
                            OBQCIssue(
                                issue_type=OBQCIssueType.COLUMN_NOT_FOUND,
                                severity=OBQCSeverity.ERROR,
                                message=f"Column '{col_name}' not found in table '{table_name}'",
                                location="Column reference",
                                suggestion=f"Available columns: {', '.join(available_cols)}",
                                related_entities=[col_ref],
                            )
                        )
            else:
                # Unqualified column - check for ambiguity
                col_key = col_ref.lower()
                found_in_tables: List[str] = []

                # Only check tables that are actually in the query
                for table_name in result.parsed_tables:
                    table_key = table_name.lower()
                    if table_key in self._schema_cache.tables:
                        if col_key in self._schema_cache.tables[table_key].columns:
                            found_in_tables.append(table_name)

                if len(found_in_tables) == 0 and len(result.parsed_tables) > 0:
                    result.issues.append(
                        OBQCIssue(
                            issue_type=OBQCIssueType.COLUMN_NOT_FOUND,
                            severity=OBQCSeverity.ERROR,
                            message=f"Column '{col_ref}' not found in any referenced table",
                            location="Column reference",
                            suggestion="Qualify column with table name (e.g., table.column)",
                        )
                    )
                elif len(found_in_tables) > 1:
                    result.issues.append(
                        OBQCIssue(
                            issue_type=OBQCIssueType.AMBIGUOUS_COLUMN,
                            severity=OBQCSeverity.WARNING,
                            message=f"Column '{col_ref}' is ambiguous - exists in: {', '.join(found_in_tables)}",
                            location="Column reference",
                            suggestion="Qualify column with table name (e.g., table.column)",
                        )
                    )

    def _validate_joins(self, result: OBQCResult) -> None:
        """Rule: Validate joins use declared FK relationships."""
        if len(result.parsed_tables) < 2:
            return  # No joins needed for single table

        if len(result.parsed_tables) > 1 and len(result.parsed_joins) == 0:
            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.MISSING_JOIN_CONDITION,
                    severity=OBQCSeverity.ERROR,
                    message="Multiple tables without explicit JOIN (Cartesian product)",
                    location="FROM clause",
                    suggestion="Add explicit JOIN ... ON conditions",
                )
            )
            return

        for join_info in result.parsed_joins:
            join_table = join_info.get("table")
            on_condition = join_info.get("on_condition")

            if not on_condition:
                result.issues.append(
                    OBQCIssue(
                        issue_type=OBQCIssueType.MISSING_JOIN_CONDITION,
                        severity=OBQCSeverity.ERROR,
                        message=f"JOIN with '{join_table}' has no ON condition",
                        location="JOIN clause",
                        suggestion="Add ON condition based on foreign key relationship",
                    )
                )
                continue

            # Check if join condition matches a declared relationship
            if not self._is_valid_join_condition(
                join_table, on_condition, result.parsed_tables
            ):
                suggested = self._get_suggested_join(join_table, result.parsed_tables)
                result.issues.append(
                    OBQCIssue(
                        issue_type=OBQCIssueType.INVALID_JOIN,
                        severity=OBQCSeverity.WARNING,
                        message="JOIN condition may not match declared FK relationship",
                        location=f"JOIN {join_table}",
                        suggestion=suggested or "Verify join matches foreign key constraint",
                        related_entities=[join_table] if join_table else [],
                    )
                )

    def _is_valid_join_condition(
        self, join_table: Optional[str], on_condition: str, all_tables: List[str]
    ) -> bool:
        """Check if join condition matches a declared relationship."""
        if self._schema_cache is None or join_table is None:
            return True  # Can't validate without schema

        on_lower = on_condition.lower()
        join_table_lower = join_table.lower()

        for rel_info in self._schema_cache.relationships.values():
            # Check if relationship involves the join table
            if (
                rel_info.from_table.lower() == join_table_lower
                or rel_info.to_table.lower() == join_table_lower
            ):
                # Check if condition references the FK columns
                if (
                    rel_info.from_column.lower() in on_lower
                    and rel_info.to_column.lower() in on_lower
                ):
                    return True
        return False

    def _get_suggested_join(
        self, join_table: Optional[str], all_tables: List[str]
    ) -> Optional[str]:
        """Get suggested join condition from ontology relationships."""
        if self._schema_cache is None or join_table is None:
            return None

        join_table_lower = join_table.lower()
        all_tables_lower = [t.lower() for t in all_tables]

        for rel_info in self._schema_cache.relationships.values():
            if rel_info.from_table.lower() == join_table_lower:
                if rel_info.to_table.lower() in all_tables_lower:
                    return f"Suggested: {rel_info.join_condition}"
            elif rel_info.to_table.lower() == join_table_lower:
                if rel_info.from_table.lower() in all_tables_lower:
                    return f"Suggested: {rel_info.join_condition}"
        return None

    def _validate_type_compatibility(
        self, parsed: exp.Expression, result: OBQCResult
    ) -> None:
        """Rule: Check type compatibility in comparisons."""
        comparison_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)

        for comp in parsed.find_all(*comparison_types):
            left = comp.left
            right = comp.right

            left_type = self._infer_type(left)
            right_type = self._infer_type(right)

            if left_type and right_type:
                if not self._types_compatible(left_type, right_type):
                    result.issues.append(
                        OBQCIssue(
                            issue_type=OBQCIssueType.TYPE_MISMATCH,
                            severity=OBQCSeverity.WARNING,
                            message=f"Type mismatch: {self._type_name(left_type)} vs {self._type_name(right_type)}",
                            location="WHERE/ON clause",
                            suggestion="Ensure compared values have compatible types",
                        )
                    )

    def _infer_type(self, expr: exp.Expression) -> Optional[str]:
        """Infer the XSD type of an expression from ontology."""
        if self._schema_cache is None:
            return None

        if isinstance(expr, exp.Column):
            table = expr.table
            column = expr.name

            if table:
                table_key = table.lower()
                col_key = column.lower()
                if table_key in self._schema_cache.tables:
                    cols = self._schema_cache.tables[table_key].columns
                    if col_key in cols:
                        xsd = cols[col_key].xsd_type
                        return str(xsd) if xsd else None
            else:
                # Search all referenced tables for this column
                for table_schema in self._schema_cache.tables.values():
                    if column.lower() in table_schema.columns:
                        xsd = table_schema.columns[column.lower()].xsd_type
                        return str(xsd) if xsd else None

        elif isinstance(expr, exp.Literal):
            if expr.is_int:
                return str(XSD.integer)
            elif expr.is_number:
                return str(XSD.decimal)
            elif expr.is_string:
                return str(XSD.string)

        return None

    def _type_name(self, xsd_uri: str) -> str:
        """Extract readable type name from XSD URI."""
        if "#" in xsd_uri:
            return xsd_uri.split("#")[-1]
        return xsd_uri.split("/")[-1]

    def _types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two XSD types are compatible for comparison."""

        def get_type_category(xsd_uri: str) -> str:
            uri_lower = xsd_uri.lower()
            if any(t in uri_lower for t in ["integer", "decimal", "float", "double", "byte"]):
                return "numeric"
            elif "string" in uri_lower:
                return "string"
            elif any(t in uri_lower for t in ["date", "time", "datetime"]):
                return "temporal"
            elif "boolean" in uri_lower:
                return "boolean"
            return "unknown"

        cat1 = get_type_category(type1)
        cat2 = get_type_category(type2)

        # Same category or unknown are compatible
        return cat1 == cat2 or cat1 == "unknown" or cat2 == "unknown"

    def _validate_aggregation_context(
        self, parsed: exp.Expression, result: OBQCResult
    ) -> None:
        """Rule: Validate GROUP BY completeness for aggregation queries."""
        if not result.has_aggregation:
            return

        for select in parsed.find_all(exp.Select):
            expressions = select.args.get("expressions", [])

            # Get GROUP BY columns
            group_by_cols: Set[str] = set()
            if select.args.get("group"):
                for group_expr in select.args["group"].expressions:
                    if isinstance(group_expr, exp.Column):
                        col_name = group_expr.name.lower()
                        if group_expr.table:
                            col_name = f"{group_expr.table.lower()}.{col_name}"
                        group_by_cols.add(col_name)

            # Check each SELECT expression
            for expr in expressions:
                col_name = None
                col_table = None

                if isinstance(expr, exp.Column):
                    col_name = expr.name
                    col_table = expr.table
                elif isinstance(expr, exp.Alias) and isinstance(expr.this, exp.Column):
                    col_name = expr.this.name
                    col_table = expr.this.table

                if col_name:
                    # Build qualified name
                    qualified = col_name.lower()
                    if col_table:
                        qualified = f"{col_table.lower()}.{col_name.lower()}"

                    # Check if it's in GROUP BY
                    if qualified not in group_by_cols and col_name.lower() not in group_by_cols:
                        # Check if it's inside an aggregate function
                        is_aggregated = self._is_inside_aggregate(expr, select)

                        if not is_aggregated:
                            if not result.has_group_by:
                                result.issues.append(
                                    OBQCIssue(
                                        issue_type=OBQCIssueType.NON_AGGREGATED_COLUMN,
                                        severity=OBQCSeverity.ERROR,
                                        message=f"Column '{col_name}' in SELECT with aggregation but no GROUP BY",
                                        location="SELECT clause",
                                        suggestion=f"Add GROUP BY {col_name} or wrap in aggregate",
                                    )
                                )
                            else:
                                result.issues.append(
                                    OBQCIssue(
                                        issue_type=OBQCIssueType.NON_AGGREGATED_COLUMN,
                                        severity=OBQCSeverity.ERROR,
                                        message=f"Column '{col_name}' not in GROUP BY clause",
                                        location="SELECT clause",
                                        suggestion=f"Add '{col_name}' to GROUP BY or use aggregate",
                                    )
                                )

    def _is_inside_aggregate(
        self, expr: exp.Expression, select: exp.Select
    ) -> bool:
        """Check if expression is inside an aggregate function."""
        agg_types = (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)

        for agg in select.find_all(*agg_types):
            for col in agg.find_all(exp.Column):
                if isinstance(expr, exp.Column):
                    if col.name == expr.name:
                        if (col.table is None and expr.table is None) or col.table == expr.table:
                            return True
                elif isinstance(expr, exp.Alias) and isinstance(expr.this, exp.Column):
                    if col.name == expr.this.name:
                        return True
        return False

    def _detect_fan_trap(self, result: OBQCResult) -> None:
        """Rule: Detect potential fan-trap patterns."""
        if not result.has_aggregation:
            return

        if len(result.parsed_tables) < 2:
            return

        if self._schema_cache is None:
            return

        # Count one-to-many relationships in the join graph
        one_to_many_count = 0
        involved_tables: List[str] = []

        for join_info in result.parsed_joins:
            join_table = join_info.get("table")
            if join_table:
                join_table_lower = join_table.lower()
                for rel_info in self._schema_cache.relationships.values():
                    # A table is on the "many" side in these cases
                    if rel_info.relationship_type == "one_to_many":
                        if rel_info.to_table.lower() == join_table_lower:
                            one_to_many_count += 1
                            involved_tables.append(join_table)
                    elif rel_info.relationship_type == "many_to_one":
                        if rel_info.from_table.lower() == join_table_lower:
                            one_to_many_count += 1
                            involved_tables.append(join_table)

        if one_to_many_count >= 2:
            result.fan_trap_risk = True
            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.FAN_TRAP_DETECTED,
                    severity=OBQCSeverity.WARNING,
                    message=f"Potential fan-trap: {one_to_many_count} one-to-many joins with aggregation",
                    location="Query structure",
                    suggestion=(
                        "Use UNION ALL pattern for separate aggregations per fact table, "
                        "or use CTEs to pre-aggregate before joining"
                    ),
                    related_entities=list(set(involved_tables)),
                )
            )

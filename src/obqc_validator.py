"""Ontology-Based Query Check (OBQC) validator for semantic SQL validation.

This module provides deterministic, rule-based validation of SQL queries against
an RDF/OWL ontology, detecting schema violations, type mismatches, invalid joins,
and fan-trap patterns without using LLM.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

import sqlglot
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from rdflib.term import Node
from sqlglot import exp
from sqlglot.errors import ParseError

from .constants import DB_SQLGLOT_DIALECTS, OBA_NAMESPACE

logger = logging.getLogger(__name__)


class OBQCIssueType(Enum):
    """Categories of OBQC validation issues."""

    TABLE_NOT_FOUND = "table_not_found"
    COLUMN_NOT_FOUND = "column_not_found"
    TYPE_MISMATCH = "type_mismatch"
    INVALID_JOIN = "invalid_join"
    MISSING_JOIN_CONDITION = "missing_join_condition"
    FAN_TRAP_DETECTED = "fan_trap_detected"
    NON_AGGREGATED_COLUMN = "non_aggregated_column"
    AMBIGUOUS_COLUMN = "ambiguous_column"


class OBQCSeverity(Enum):
    """Severity levels for OBQC issues."""

    ERROR = "error"  # Query will fail or produce incorrect results
    WARNING = "warning"  # Query may have issues or suboptimal patterns
    INFO = "info"  # Informational note about query structure


# Schemas holding database metadata rather than user data, per dialect. Tables
# here are never part of a generated ontology -- that describes the user's
# schema -- so requiring them to appear in it blocks legitimate catalog queries
# such as "SELECT table_name FROM information_schema.tables".
#
# This is not a security decision: src/security.py separately blocks the
# privilege-bearing views (information_schema.*_privileges, pg_catalog.pg_authid
# and friends) and permits the rest, so exempting these from the ontology rule
# follows the security policy instead of widening it.
CATALOG_SCHEMAS = frozenset(
    {
        "information_schema",  # ANSI: PostgreSQL, MySQL, Snowflake, Databricks, ...
        "pg_catalog",  # PostgreSQL
        "pg_toast",  # PostgreSQL
        "performance_schema",  # MySQL runtime statistics
        "sys",  # Dremio metadata views
        "system",  # ClickHouse
        "snowflake",  # Snowflake (snowflake.account_usage)
        "sqlite_schema",  # SQLite
        "sqlite_master",  # SQLite (legacy name)
    }
)

# Deliberately NOT listed above: MySQL's "mysql" schema. It is not a metadata
# catalog but the server's own data -- mysql.user holds account names and
# password hashes, mysql.db and mysql.tables_priv hold grants. Exempting it
# would have let those through on the grounds that they are "catalog tables",
# which they are not. src/security.py blocks them outright.


# Where a SELECT alias may be referenced, keyed by the database type callers
# pass to validate() (not the sqlglot parsing dialect -- dremio parses as
# trino, but is configured here under its own name).
#
# ORDER BY and GROUP BY see the select list's output names everywhere. HAVING
# is where the dialects diverge, checked against vendor documentation:
#
#   postgresql  no   an output name may be used in GROUP BY and ORDER BY but
#                    not in WHERE or HAVING (PostgreSQL SELECT reference)
#   mysql       yes  permitted in HAVING
#   clickhouse  yes  "reference aggregation results from SELECT clause in
#                    HAVING clause by their alias" (HAVING clause docs)
#   snowflake   yes  "expressions in the SELECT list can be referred to by the
#                    column alias defined in the list" (HAVING docs)
#   databricks  yes  resolvable in HAVING, though a real column of the same
#                    name wins over the alias (name-resolution docs)
#   bigquery    yes  aliases are visible to GROUP BY, HAVING and ORDER BY
#   duckdb      yes  verified directly against duckdb 1.5.5 -- it accepts an
#                    alias in every clause, WHERE included
#   dremio      ?    Trino's SELECT docs describe GROUP BY and ORDER BY in
#                    terms of input columns and ordinals without stating
#                    whether an output alias resolves. Left permissive; see
#                    below for why the unknown case errs that way.
#
# The default is permissive because OBQC errors block execution. Wrongly
# allowing an alias costs nothing -- the database rejects the query itself,
# with a better message. Wrongly forbidding one stops a query that would have
# run. So a clause is only closed off where documentation says it is closed,
# which today means PostgreSQL's HAVING.
DEFAULT_ALIAS_VISIBLE_CLAUSES = ("order", "group", "having", "qualify")

ALIAS_VISIBLE_CLAUSES = {
    "postgresql": ("order", "group", "qualify"),
    # DuckDB resolves aliases laterally, including in WHERE.
    "duckdb": ("order", "group", "having", "qualify", "where"),
}

# Dialects where an output name is only recognised as a whole sort or group
# key, never inside a larger expression. PostgreSQL's ORDER BY documentation
# is explicit that an output name may be used as a sort key but that anything
# more than a bare name is evaluated as an expression over input columns, so
# "ORDER BY t + 1" fails there while DuckDB -- checked against 1.5.5 --
# accepts it. Listed rather than defaulted for the usual reason: forbidding it
# where it is legal would block a query the database would have run.
ALIAS_STANDALONE_ONLY = frozenset({"postgresql"})

# A string literal holding a date or timestamp. SQL has no date literal syntax
# in common use -- every dialect accepts a string and converts it -- so typing
# these as strings reported "order_date >= '2024-01-01'" as a type mismatch.
TEMPORAL_LITERAL = re.compile(
    r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$"
)


@dataclass
class OBQCIssue:
    """Single OBQC validation issue."""

    issue_type: OBQCIssueType
    severity: OBQCSeverity
    message: str
    location: str | None = None
    suggestion: str | None = None
    related_entities: list[str] = field(default_factory=list)


@dataclass
class OBQCResult:
    """Complete OBQC validation result."""

    is_valid: bool
    issues: list[OBQCIssue] = field(default_factory=list)
    parsed_tables: list[str] = field(default_factory=list)
    # Bare names of tables referenced through a database catalog schema
    # (information_schema and friends). They are legitimately absent from the
    # ontology, which describes user data, so ontology-existence rules skip them.
    catalog_tables: set[str] = field(default_factory=set)
    # Lower-cased names declared by a WITH clause. A CTE is defined by the query
    # itself, so it is legitimately absent from the ontology and its projected
    # columns are not describable by it either; ontology-existence rules skip
    # references to these names (but not the tables inside the CTE bodies).
    cte_names: set[str] = field(default_factory=set)
    parsed_columns: list[str] = field(default_factory=list)
    # Lower-cased SELECT aliases referenced from ORDER BY / GROUP BY / HAVING.
    # They resolve to select-list output, not to any table's column.
    select_aliases: set[str] = field(default_factory=set)
    # (column reference, tables it may resolve against) per occurrence. Name
    # resolution is scoped to the SELECT a column appears in plus its enclosing
    # ones, so a subquery's tables cannot answer for the outer query.
    column_scopes: list[tuple[str, tuple[tuple[str, ...], ...]]] = field(
        default_factory=list
    )
    parsed_joins: list[dict[str, Any]] = field(default_factory=list)
    # Tables of each SELECT that applies an aggregate, in its own scope. Only
    # these can be multiplied by that SELECT's joins, so fan-trap rules judge
    # them rather than every table named anywhere in the query.
    aggregating_scopes: list[tuple[str, ...]] = field(default_factory=list)
    has_aggregation: bool = False
    has_group_by: bool = False
    fan_trap_risk: bool = False
    ontology_compatible: bool = True  # Whether ontology has oba: annotations

    # Keys of parsed_joins that belong in a response. Everything else the
    # extractor records is bookkeeping for the validation rules and must not
    # reach a caller -- notably scope_id, which identifies a SELECT within one
    # parse and means nothing outside it.
    PUBLIC_JOIN_KEYS: ClassVar[tuple[str, ...]] = (
        "type",
        "table",
        "on_condition",
        "on_tables",
    )

    def to_dict(self) -> dict[str, Any]:
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
            "parsed_joins": [
                {k: j[k] for k in self.PUBLIC_JOIN_KEYS if k in j}
                for j in self.parsed_joins
            ],
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
    xsd_type: URIRef | None = None
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    fk_referenced_table: str | None = None
    fk_referenced_column: str | None = None


@dataclass
class TableSchema:
    """Schema information for a single table."""

    name: str
    schema_name: str
    columns: dict[str, ColumnSchema] = field(default_factory=dict)
    primary_keys: list[str] = field(default_factory=list)


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

    tables: dict[str, TableSchema] = field(default_factory=dict)
    relationships: dict[str, RelationshipInfo] = field(default_factory=dict)


class OBQCValidator:
    """Ontology-Based Query Check validator.

    Validates SQL queries against an RDF/OWL ontology to detect:
    - Schema violations (missing tables/columns)
    - Type mismatches in comparisons
    - Invalid or missing join conditions
    - Fan-trap patterns with aggregation
    - GROUP BY completeness
    """

    # Dialect mapping for sqlglot, sourced from the canonical metadata in
    # constants so it always covers exactly SUPPORTED_DB_TYPES (no drift, no
    # silent fallback to postgres for a supported database).
    DIALECT_MAP = DB_SQLGLOT_DIALECTS

    def __init__(self) -> None:
        self._schema_cache: OntologySchema | None = None
        self._graph: Graph | None = None
        self._base_uri: Namespace | None = None
        self._oba_ns: Namespace | None = None
        self._is_compatible: bool = False  # Whether ontology has oba: annotations
        # Fan-trap topology read straight from the ontology axioms (Phase 2):
        # pairs of lower-cased table names declared owl:disjointWith each other
        # (sibling facts sharing a dimension — the canonical fan-trap shape).
        self._disjoint_pairs: set[frozenset] = set()

    def load_ontology(self, ontology_graph: Graph, base_uri: str) -> None:
        """Load and cache schema from ontology graph.

        Args:
            ontology_graph: The rdflib Graph containing the ontology
            base_uri: The base URI namespace (e.g., "http://example.com/ontology/")
        """
        self._graph = ontology_graph
        self._base_uri = Namespace(base_uri)
        self._oba_ns = Namespace(OBA_NAMESPACE)
        self._schema_cache = self._extract_schema_from_ontology()
        self._disjoint_pairs = self._extract_disjoint_pairs()

        # Check if ontology has required oba: annotations for OBQC
        self._is_compatible = self._check_ontology_compatibility()

        if self._is_compatible:
            logger.info(
                f"OBQC loaded ontology with {len(self._schema_cache.tables)} tables, "
                f"{sum(len(t.columns) for t in self._schema_cache.tables.values())} columns"
            )
        else:
            logger.warning(
                "OBQC: Ontology lacks oba: namespace annotations - semantic validation disabled. "
                "Use generate_ontology to create a compatible ontology."
            )

    def _check_ontology_compatibility(self) -> bool:
        """Check if ontology has required oba: namespace annotations for OBQC.

        Returns:
            True if ontology has oba:tableName annotations, False otherwise
        """
        if self._schema_cache is None:
            return False

        # Ontology is compatible if it has at least one table with oba:tableName
        # and at least one column with oba:columnName
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

        if self._graph is None or self._oba_ns is None:
            return schema

        # Extract tables (owl:Class with oba:tableName)
        for subject in self._graph.subjects(RDF.type, OWL.Class):
            if subject == OWL.Class:
                continue
            table_name = self._get_literal(subject, self._oba_ns.tableName)
            if table_name:
                schema_name = (
                    self._get_literal(subject, self._oba_ns.schemaName) or "public"
                )
                table_schema = TableSchema(name=table_name, schema_name=schema_name)

                # Get primary keys
                for pk in self._graph.objects(subject, self._oba_ns.primaryKey):
                    table_schema.primary_keys.append(str(pk))

                schema.tables[table_name.lower()] = table_schema

        # Extract columns (owl:DatatypeProperty with oba:columnName)
        for subject in self._graph.subjects(RDF.type, OWL.DatatypeProperty):
            column_name = self._get_literal(subject, self._oba_ns.columnName)
            table_name = self._get_literal(subject, self._oba_ns.tableName)

            if column_name and table_name:
                table_key = table_name.lower()
                if table_key in schema.tables:
                    col_schema = ColumnSchema(
                        name=column_name,
                        table_name=table_name,
                        sql_data_type=self._get_literal(
                            subject, self._oba_ns.sqlDataType
                        )
                        or "VARCHAR",
                        is_nullable=self._get_bool(
                            subject, self._oba_ns.isNullable, True
                        ),
                        is_primary_key=self._get_bool(
                            subject, self._oba_ns.isPrimaryKey, False
                        ),
                        is_foreign_key=self._get_bool(
                            subject, self._oba_ns.isForeignKey, False
                        ),
                    )

                    # Get XSD type from rdfs:range
                    for range_val in self._graph.objects(subject, RDFS.range):
                        if isinstance(range_val, URIRef):
                            col_schema.xsd_type = range_val
                            break

                    schema.tables[table_key].columns[column_name.lower()] = col_schema

        # Extract relationships (owl:ObjectProperty with oba:foreignKeyColumn)
        for subject in self._graph.subjects(RDF.type, OWL.ObjectProperty):
            fk_column = self._get_literal(subject, self._oba_ns.foreignKeyColumn)
            ref_table = self._get_literal(subject, self._oba_ns.referencedTable)
            ref_column = self._get_literal(subject, self._oba_ns.referencedColumn)
            rel_type = self._get_literal(subject, self._oba_ns.relationshipType)
            join_cond = self._get_literal(subject, self._oba_ns.sqlJoinCondition)

            if fk_column and ref_table:
                # Determine from_table from domain
                from_table = None
                for domain in self._graph.objects(subject, RDFS.domain):
                    from_table = self._get_literal(domain, self._oba_ns.tableName)
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

    def _extract_disjoint_pairs(self) -> set[frozenset]:
        """Read owl:disjointWith axioms as pairs of lower-cased table names.

        The generator emits owl:disjointWith between sibling fact tables that
        share a dimension but have no FK between them — exactly the fan-trap
        topology. Reading it here lets OBQC ground fan-trap detection in the
        ontology instead of re-deriving it from the relationship heuristic.
        """
        pairs: set[frozenset] = set()
        if self._graph is None or self._oba_ns is None:
            return pairs

        for a_uri, b_uri in self._graph.subject_objects(OWL.disjointWith):
            a_name = self._get_literal(a_uri, self._oba_ns.tableName)
            b_name = self._get_literal(b_uri, self._oba_ns.tableName)
            if a_name and b_name and a_name.lower() != b_name.lower():
                pairs.add(frozenset((a_name.lower(), b_name.lower())))

        return pairs

    def _get_literal(self, subject: Node, predicate: URIRef) -> str | None:
        """Get string value of a literal predicate."""
        if self._graph is None:
            return None
        for obj in self._graph.objects(subject, predicate):
            return str(obj)
        return None

    def _get_bool(self, subject: Node, predicate: URIRef, default: bool) -> bool:
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

        # Check if ontology has required oba: namespace annotations
        if not self._is_compatible:
            result.ontology_compatible = False
            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.TABLE_NOT_FOUND,
                    severity=OBQCSeverity.INFO,
                    message="Ontology lacks oba: namespace annotations - OBQC validation skipped",
                    suggestion=(
                        "The loaded ontology does not contain oba:tableName/oba:columnName annotations. "
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
                    message=f"SQL parse error: {e!s}",
                    location="Query",
                )
            )
            return result

        # Extract query components. CTE names first: the rules below need to
        # know which references name a WITH alias rather than a real table.
        self._extract_ctes(parsed, result)
        self._extract_tables(parsed, result)
        self._extract_columns(parsed, result, dialect)
        self._extract_joins(parsed, result)
        self._extract_aggregations(parsed, result)

        # Run validation rules
        self._validate_tables(result)
        self._validate_columns(result)
        self._validate_joins(parsed, result)
        self._validate_type_compatibility(parsed, result)
        self._validate_aggregation_context(parsed, result)
        self._detect_fan_trap(result)

        # Set overall validity
        result.is_valid = not any(
            issue.severity == OBQCSeverity.ERROR for issue in result.issues
        )

        return result

    def _extract_ctes(self, parsed: exp.Expr, result: OBQCResult) -> None:
        """Record every name declared by a WITH clause.

        A CTE is a table the query defines for itself, so the ontology never
        describes it. Without this, ``WITH recent AS (...) SELECT ... FROM
        recent`` was rejected outright: ``recent`` was reported as a table not
        found in the ontology, an error, which blocks execution.

        Names are collected across the whole tree, so CTEs declared inside a
        subquery count too. Scoping them to the exact SELECT that declared them
        would be more precise, but the cost of over-collecting is only that a
        real table sharing a CTE's name goes unchecked, while under-collecting
        blocks a valid query.

        Args:
            parsed: Parsed query.
            result: Result to record the names on.
        """
        for cte in parsed.find_all(exp.CTE):
            name = cte.alias_or_name
            if name:
                result.cte_names.add(name.lower())

    def _extract_tables(self, parsed: exp.Expr, result: OBQCResult) -> None:
        """Extract all table references, noting which come from a catalog schema."""
        # Bare names that also appear as a non-catalog reference. Catalog
        # membership is tracked by bare name -- sqlglot gives no other handle --
        # so a user table sharing a catalog table's name would otherwise be
        # exempted along with it, hiding an unknown table called e.g. "tables".
        shadowed: set[str] = set()

        for table in parsed.find_all(exp.Table):
            table_name = table.name
            if not table_name:
                continue
            if table_name not in result.parsed_tables:
                result.parsed_tables.append(table_name)

            # Without the qualifier a catalog reference is indistinguishable
            # from a user table: sqlglot reports information_schema.tables as
            # simply "tables". Both positions are checked -- ``db`` carries the
            # schema in ``information_schema.tables`` and
            # ``mydb.information_schema.tables``, while ``catalog`` carries it
            # in Snowflake's ``snowflake.account_usage.query_history``.
            qualifiers = {q.lower() for q in (table.db, table.catalog) if q}
            if qualifiers & CATALOG_SCHEMAS:
                result.catalog_tables.add(table_name)
            else:
                shadowed.add(table_name)

        # A name used both ways is ambiguous, and the ontology rule is the only
        # thing that would catch the non-catalog use, so it keeps applying.
        result.catalog_tables -= shadowed

    def _extract_columns(
        self, parsed: exp.Expr, result: OBQCResult, dialect: str = "postgresql"
    ) -> None:
        """Extract column references, excluding legal SELECT-alias references."""
        alias_refs = self._select_alias_references(parsed, result, dialect)
        scope_cache: dict[int, tuple[tuple[str, ...], ...]] = {}

        for column in parsed.find_all(exp.Column):
            # Alias references resolve to the select list, not to a table, so
            # they are not column references at all. Dropping them here rather
            # than excusing their names later keeps the exemption tied to the
            # exact node: a name is only excused where it really is an alias
            # use, in the query scope that declared it.
            if id(column) in alias_refs:
                continue
            col_ref = column.name
            if column.table:
                col_ref = f"{column.table}.{column.name}"
            if not col_ref:
                continue
            if col_ref not in result.parsed_columns:
                result.parsed_columns.append(col_ref)

            owner = column.find_ancestor(exp.Select)

            # Validation needs the table, not the alias the query happened to
            # write. The reference is reported as written (above); only the
            # form the rules consume is resolved.
            scoped_ref = col_ref
            if column.table:
                real_table = self._resolve_qualifier(owner, column.table)
                if real_table:
                    scoped_ref = f"{real_table}.{column.name}"

            # Which tables the name could resolve against, which is a property
            # of where it appears. Resolving against every table in the query
            # let a subquery's table answer for the outer SELECT: "SELECT
            # quantity FROM users WHERE id IN (SELECT order_id FROM
            # order_items)" found quantity in order_items and reported nothing.
            scope = self._scope_tables(owner, scope_cache) if owner else ()
            entry = (scoped_ref, scope)
            if entry not in result.column_scopes:
                result.column_scopes.append(entry)

    def _scope_tables(
        self, select: exp.Select, cache: dict[int, tuple[tuple[str, ...], ...]]
    ) -> tuple[tuple[str, ...], ...]:
        """Tables a name in *select* may resolve against, innermost level first.

        Returned as levels rather than one flat set because SQL resolves a name
        in the innermost scope that provides it and stops. Flattening made
        ``SELECT name FROM users WHERE EXISTS (SELECT 1 FROM orders WHERE id =
        user_id)`` look ambiguous between orders.id and users.id, when the
        inner ``id`` is simply orders.id.

        Enclosing levels are still included: a correlated subquery legally
        references them, so dropping them would report those names as missing.

        Args:
            select: The SELECT a column appears in.
            cache: Memo keyed by ``id(select)``.

        Returns:
            One tuple of table names per scope level, innermost first.
        """
        cached = cache.get(id(select))
        if cached is not None:
            return cached

        own = tuple(
            dict.fromkeys(
                t.name
                for t in select.find_all(exp.Table)
                if t.name and t.find_ancestor(exp.Select) is select
            )
        )
        # A CTE body is not a nested scope of the query that declares it: it
        # may reference earlier CTEs and real tables, never the outer FROM.
        # Treating it as nested let the outer query's tables answer for names
        # inside the CTE -- and, once WITH aliases became undescribable, let a
        # mere reference to the CTE excuse any bogus name in its own body.
        parent = None if self._is_cte_body(select) else select.parent_select
        outer = self._scope_tables(parent, cache) if parent is not None else ()

        levels = (own, *outer) if own else outer
        cache[id(select)] = levels
        return levels

    @staticmethod
    def _is_cte_body(select: exp.Select) -> bool:
        """Whether *select* is the body of a WITH clause definition.

        Args:
            select: The SELECT to classify.

        Returns:
            True if the nearest enclosing construct is a CTE definition rather
            than an enclosing query.
        """
        node = select.parent
        while node is not None and not isinstance(node, exp.Select):
            if isinstance(node, exp.CTE):
                return True
            node = node.parent
        return False

    def _select_alias_references(
        self, parsed: exp.Expr, result: OBQCResult, dialect: str
    ) -> set[int]:
        """Identify column nodes that legally reference a SELECT alias.

        ``SELECT SUM(total) AS revenue FROM orders ORDER BY revenue`` is valid
        SQL, but ``revenue`` is not a column of any table, so the column rule
        reported it missing and -- being an error -- blocked the query.

        Resolution is per SELECT and per clause. An alias is visible only to
        clauses evaluated after the select list, and only within the query that
        declared it: a subquery's alias must not excuse a bogus name in the
        outer SELECT.

        Args:
            parsed: Parsed query.
            result: Result to record resolved alias names on (reporting only).
            dialect: Database dialect, which decides where aliases are visible.

        Returns:
            ``id()`` of every column node that is an alias reference.
        """
        clauses = ALIAS_VISIBLE_CLAUSES.get(
            dialect.lower(), DEFAULT_ALIAS_VISIBLE_CLAUSES
        )
        alias_refs: set[int] = set()

        for select in parsed.find_all(exp.Select):
            aliases = {
                projection.alias.lower()
                for projection in select.expressions
                if isinstance(projection, exp.Alias) and projection.alias
            }
            if not aliases:
                continue

            standalone_only = dialect.lower() in ALIAS_STANDALONE_ONLY

            for clause in clauses:
                node = select.args.get(clause)
                if node is None:
                    continue
                for column in node.find_all(exp.Column):
                    # A qualified reference names a real table, so it is not an
                    # alias use and stays subject to the normal column rule.
                    if column.table or column.name.lower() not in aliases:
                        continue
                    # A nested SELECT re-declares its own scope; its clauses
                    # are resolved on its own iteration, not this one.
                    if column.find_ancestor(exp.Select) is not select:
                        continue
                    if standalone_only and not self._is_standalone_key(column, node):
                        continue
                    alias_refs.add(id(column))
                    result.select_aliases.add(column.name.lower())

        return alias_refs

    def _is_real_column(self, name: str, result: OBQCResult) -> bool:
        """Whether *name* is a column of some table the query references.

        Args:
            name: Bare or qualified column name, lower-cased.
            result: Result carrying the query's tables.

        Returns:
            True if any queried table declares the column.
        """
        if self._schema_cache is None:
            return False

        bare = name.split(".")[-1]
        for table_name in result.parsed_tables:
            table = self._schema_cache.tables.get(table_name.lower())
            if table and bare in table.columns:
                return True
        return False

    def _is_standalone_key(self, column: exp.Column, clause_node: exp.Expr) -> bool:
        """Whether *column* is a whole sort/group key rather than part of one.

        ``ORDER BY t`` refers to the output name; ``ORDER BY t + 1`` is an
        expression, which strict dialects evaluate over input columns only.

        Args:
            column: Candidate alias reference.
            clause_node: The ORDER BY / GROUP BY node containing it.

        Returns:
            True if the column is one of the clause's top-level keys.
        """
        keys = (
            self._group_by_keys(clause_node)
            if isinstance(clause_node, exp.Group)
            else clause_node.expressions
        )
        for key in keys:
            # ORDER BY keys are wrapped in Ordered (carrying ASC/DESC etc.);
            # GROUP BY keys are the expressions themselves.
            target = key.this if isinstance(key, exp.Ordered) else key
            if target is column:
                return True
        return False

    def _extract_joins(self, parsed: exp.Expr, result: OBQCResult) -> None:
        """Extract join information from parsed query."""
        # Alias maps are built per SELECT. Table aliases are scoped to the
        # query that declares them, and a subquery may reuse an outer one: a
        # single map over the whole tree let "FROM order_items u" inside an
        # EXISTS overwrite the outer "FROM users u", so the outer join's ON
        # condition resolved to order_items and its fan-out was judged against
        # the wrong table -- silently dropping a fan-trap warning.
        for scope_index, select in enumerate(parsed.find_all(exp.Select)):
            alias_map = self._build_alias_map(select)
            # Whether the SELECT owning these joins aggregates. Fan-out only
            # inflates a total if the aggregation happens over these joins --
            # an aggregate in some unrelated subquery does not.
            scope_aggregates = self._select_aggregates(select)
            if scope_aggregates:
                result.aggregating_scopes.append(
                    tuple(
                        dict.fromkeys(
                            t.name
                            for t in select.find_all(exp.Table)
                            if t.name and t.find_ancestor(exp.Select) is select
                        )
                    )
                )

            # The comma form's join conditions live in WHERE, so a join here
            # may carry no ON and still be conditioned.
            where_joins = self._where_joins_tables(select)

            for join in select.args.get("joins") or []:
                join_info: dict[str, Any] = {
                    "type": join.kind or "INNER",
                    "table": None,
                    "on_condition": None,
                    "scope_aggregates": scope_aggregates,
                    # Identifies the owning SELECT so fan-out is counted within
                    # one query rather than pooled across unrelated ones. A
                    # traversal index, not id(select): object addresses vary
                    # per run, and this value must never reach a response.
                    "scope_id": scope_index,
                    # Real table names referenced by the ON condition. Fan-trap
                    # detection needs to know which table this join attaches
                    # to, and the ON condition is the only place that says so.
                    "on_tables": [],
                }

                # Get joined table
                if join.this and isinstance(join.this, exp.Table):
                    join_info["table"] = join.this.name

                # Whether the join is conditioned at all -- by ON, by USING or
                # NATURAL, or by a cross-table predicate in WHERE. Recorded so
                # the missing-condition rule does not demand an ON that these
                # forms never have. None of them yields a pair of qualified
                # columns in the join itself, so they are not judged for
                # fan-out.
                join_info["has_condition"] = (
                    self._join_is_qualified(join) or where_joins
                )

                # Get ON condition
                on_clause = join.args.get("on")
                if on_clause is not None:
                    join_info["on_condition"] = on_clause.sql()
                    on_tables: list[str] = []
                    for column in on_clause.find_all(exp.Column):
                        if not column.table:
                            continue
                        # Columns are qualified by alias far more often than by
                        # table name, so resolve through the alias map.
                        resolved = alias_map.get(column.table.lower(), column.table)
                        if resolved not in on_tables:
                            on_tables.append(resolved)
                    join_info["on_tables"] = on_tables

                result.parsed_joins.append(join_info)

    def _resolve_qualifier(
        self, select: exp.Select | None, qualifier: str
    ) -> str | None:
        """Resolve a column's qualifier to the table it names.

        ``FROM sales s`` makes ``s.amount`` a reference to ``sales``. Rules
        that look the qualifier up in the ontology directly found nothing and
        said nothing, so ``SELECT s.bogus FROM sales s`` passed while the
        unaliased spelling was correctly rejected -- and aliased comparisons,
        which is most real SQL, were never type-checked at all.

        Enclosing scopes are searched too, since a correlated subquery may
        qualify with an outer alias. A CTE body ends the search: it cannot see
        the query that declares it.

        Args:
            select: The SELECT the reference appears in.
            qualifier: The alias or table name written before the dot.

        Returns:
            The real table name, or None if the qualifier names something the
            ontology cannot describe (a derived table, an unknown alias).
        """
        key = qualifier.lower()
        node = select
        while node is not None:
            resolved = self._build_alias_map(node).get(key)
            if resolved is not None:
                return resolved
            node = None if self._is_cte_body(node) else node.parent_select
        return None

    def _build_alias_map(self, select: exp.Expr) -> dict[str, str]:
        """Map one SELECT's table aliases (and bare names) to real table names.

        Tables belonging to a nested subquery are excluded: their aliases live
        in that subquery's scope and would otherwise shadow same-named ones
        here.

        Args:
            select: The SELECT whose scope to map.

        Returns:
            Lower-cased alias -> table name. Bare names map to themselves so
            callers can look up unaliased references the same way.
        """
        alias_map: dict[str, str] = {}
        for table in select.find_all(exp.Table):
            if not table.name:
                continue
            if table.find_ancestor(exp.Select) is not select:
                continue
            alias_map[table.name.lower()] = table.name
            if table.alias:
                alias_map[table.alias.lower()] = table.name
        return alias_map

    def _extract_aggregations(self, parsed: exp.Expr, result: OBQCResult) -> None:
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
            # Catalog metadata is not described by the ontology and never will
            # be; demanding it appear there blocks catalog queries outright.
            if table_name in result.catalog_tables:
                continue
            # A WITH alias is defined by the query, not by the ontology.
            if table_name.lower() in result.cte_names:
                continue
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

        for col_ref, scope in result.column_scopes:
            if "." in col_ref:
                parts = col_ref.split(".", 1)
                table_name, col_name = parts[0], parts[1]
                table_key = table_name.lower()
                col_key = col_name.lower()

                # Columns of a CTE come from its select list, which the
                # ontology does not describe. The name also shadows any real
                # table it collides with, so the ontology's columns for that
                # table are not the right thing to check against either.
                if table_key in result.cte_names:
                    continue

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

                # Innermost level that provides the name wins; SQL stops there,
                # so tables further out are not candidates and cannot make it
                # ambiguous.
                found_in_tables: list[str] = []

                for level in scope:
                    matches = [
                        table_name
                        for table_name in level
                        if (
                            table_name.lower() in self._schema_cache.tables
                            and col_key
                            in self._schema_cache.tables[table_name.lower()].columns
                        )
                    ]
                    if matches:
                        found_in_tables = matches
                        break

                # Columns of a catalog table cannot be resolved -- the ontology
                # does not describe them. If the query touches one at all, an
                # unqualified name that matches no user table might still be
                # its column, so there is nothing to report. Only when every
                # table in the query is describable can a missing name be
                # called missing.
                # Unresolved names are judged against every level, since any of
                # them could legitimately have provided the name.
                # A CTE in scope is undescribable for the same reason: its
                # output columns are whatever its select list produced, so an
                # unqualified name that matches no ontology table may well be
                # one of them.
                visible = [t for level in scope for t in level]
                describable_tables = [
                    t
                    for t in visible
                    if t not in result.catalog_tables
                    and t.lower() not in result.cte_names
                ]
                all_tables_describable = len(describable_tables) == len(visible)

                if (
                    len(found_in_tables) == 0
                    and len(describable_tables) > 0
                    and all_tables_describable
                ):
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

    def _flag_cartesian_products(self, parsed: exp.Expr, result: OBQCResult) -> bool:
        """Report SELECTs that combine tables without any join condition.

        Judged per SELECT. Counting tables and joins across the whole query
        made every subquery look like a cross product: "SELECT id FROM users
        WHERE id IN (SELECT user_id FROM orders)" has two tables and no joins
        in total, so it was rejected outright even though each SELECT is
        perfectly ordinary.

        Args:
            parsed: Parsed query.
            result: Result to append issues to.

        Returns:
            True if a cross product was reported.
        """
        found = False

        for select in parsed.find_all(exp.Select):
            tables = [
                t
                for t in select.find_all(exp.Table)
                if t.name and t.find_ancestor(exp.Select) is select
            ]
            if len(tables) < 2:
                continue

            # Comma-separated FROM items arrive as joins carrying no ON, so a
            # cross product is a scope whose joins all lack one.
            #
            # "Lacks one" is not the same as "has no ON": USING and NATURAL
            # state the join just as explicitly, and the older test read them
            # as cross products -- "FROM sales JOIN clients USING (client_id)"
            # was rejected outright.
            joins = select.args.get("joins") or []
            if joins and any(self._join_is_qualified(join) for join in joins):
                continue

            # The comma form puts its condition in WHERE, where it is a join in
            # everything but syntax: "FROM sales s, clients c WHERE s.client_id
            # = c.id" is the same query as the JOIN ... ON spelling, and was
            # rejected as a cross product.
            if self._where_joins_tables(select):
                continue

            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.MISSING_JOIN_CONDITION,
                    severity=OBQCSeverity.ERROR,
                    message="Multiple tables without explicit JOIN (Cartesian product)",
                    location="FROM clause",
                    suggestion="Add explicit JOIN ... ON conditions",
                )
            )
            found = True

        return found

    @staticmethod
    def _join_is_qualified(join: exp.Join) -> bool:
        """Whether *join* states how the two sides line up.

        ON, USING and NATURAL are three spellings of the same thing. Only a
        join with none of them produces a cross product.

        Args:
            join: The JOIN to inspect.

        Returns:
            True if the join carries a condition.
        """
        return bool(
            join.args.get("on")
            or join.args.get("using")
            # sqlglot records NATURAL as the join *method*, not its kind.
            or (join.args.get("method") or "").upper() == "NATURAL"
        )

    def _where_joins_tables(self, select: exp.Select) -> bool:
        """Whether this SELECT's WHERE equates columns of two different tables.

        The pre-SQL-92 comma form writes its join conditions in WHERE, so a
        scope with no ON clause may still be fully joined.

        Qualified names are what make this decidable: two columns qualified by
        different aliases are a cross-table predicate. Unqualified names are
        ignored -- resolving them would take the full scope, and guessing wrong
        would either excuse a real cross product or block a valid query.

        Args:
            select: The SELECT whose WHERE clause to inspect.

        Returns:
            True if some equality compares columns of two distinct tables.
        """
        where = select.args.get("where")
        if where is None:
            return False

        for eq in where.find_all(exp.EQ):
            # An EQ inside a nested subquery belongs to that subquery's scope.
            if eq.find_ancestor(exp.Select) is not select:
                continue
            left, right = eq.this, eq.expression
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            if left.table and right.table and left.table.lower() != right.table.lower():
                return True

        return False

    def _validate_joins(self, parsed: exp.Expr, result: OBQCResult) -> None:
        """Rule: Validate joins use declared FK relationships."""
        if self._flag_cartesian_products(parsed, result):
            # A cross product is reported once per query; the per-join checks
            # below would restate it as a missing ON condition.
            return

        if len(result.parsed_tables) < 2:
            return  # No joins needed for single table

        for join_info in result.parsed_joins:
            join_table = join_info.get("table")
            on_condition = join_info.get("on_condition")

            if not on_condition:
                if join_info.get("has_condition"):
                    # USING / NATURAL / comma-form: joined, just not with an ON.
                    continue
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

            # A CTE has no declared FK relationships -- it is not in the
            # ontology at all -- so the check below could only ever say "may
            # not match", on every join to a WITH alias.
            if join_table and join_table.lower() in result.cte_names:
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
                        suggestion=suggested
                        or "Verify join matches foreign key constraint",
                        related_entities=[join_table] if join_table else [],
                    )
                )

    def _is_valid_join_condition(
        self, join_table: str | None, on_condition: str, all_tables: list[str]
    ) -> bool:
        """Check if join condition matches a declared relationship."""
        if self._schema_cache is None or join_table is None:
            return True  # Can't validate without schema

        on_lower = on_condition.lower()
        join_table_lower = join_table.lower()

        for rel_info in self._schema_cache.relationships.values():
            # Relationship must involve the join table AND the condition must
            # reference both of its FK columns.
            if (
                rel_info.from_table.lower() == join_table_lower
                or rel_info.to_table.lower() == join_table_lower
            ) and (
                rel_info.from_column.lower() in on_lower
                and rel_info.to_column.lower() in on_lower
            ):
                return True
        return False

    def _get_suggested_join(
        self, join_table: str | None, all_tables: list[str]
    ) -> str | None:
        """Get suggested join condition from ontology relationships."""
        if self._schema_cache is None or join_table is None:
            return None

        join_table_lower = join_table.lower()
        all_tables_lower = [t.lower() for t in all_tables]

        for rel_info in self._schema_cache.relationships.values():
            if (
                rel_info.from_table.lower() == join_table_lower
                and rel_info.to_table.lower() in all_tables_lower
            ):
                return f"Suggested: {rel_info.join_condition}"
            if (
                rel_info.to_table.lower() == join_table_lower
                and rel_info.from_table.lower() in all_tables_lower
            ):
                return f"Suggested: {rel_info.join_condition}"
        return None

    def _validate_type_compatibility(
        self, parsed: exp.Expr, result: OBQCResult
    ) -> None:
        """Rule: Check type compatibility in comparisons."""
        comparison_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)

        for comp in parsed.find_all(*comparison_types):
            left = comp.left
            right = comp.right

            # The scope is what makes an alias resolvable, so it travels with
            # the expression: "WHERE s.amount = c.name" is only checkable once
            # s and c are known to be sales and clients.
            scope = comp.find_ancestor(exp.Select)
            left_type = self._infer_type(left, scope)
            right_type = self._infer_type(right, scope)

            if (
                left_type
                and right_type
                and not self._types_compatible(left_type, right_type)
            ):
                result.issues.append(
                    OBQCIssue(
                        issue_type=OBQCIssueType.TYPE_MISMATCH,
                        severity=OBQCSeverity.WARNING,
                        message=f"Type mismatch: {self._type_name(left_type)} vs {self._type_name(right_type)}",
                        location="WHERE/ON clause",
                        suggestion="Ensure compared values have compatible types",
                    )
                )

    def _infer_type(
        self, expr: exp.Expr, scope: exp.Select | None = None
    ) -> str | None:
        """Infer the XSD type of an expression from ontology.

        Args:
            expr: The expression to type.
            scope: SELECT the expression sits in, used to resolve a column's
                table alias. Without it, only unaliased references type.

        Returns:
            The XSD type URI as a string, or None if it cannot be determined.
        """
        if self._schema_cache is None:
            return None

        if isinstance(expr, exp.Column):
            table = expr.table
            column = expr.name

            if table:
                table = self._resolve_qualifier(scope, table) or table
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
                # Every dialect writes date and timestamp values as string
                # literals, so "order_date >= '2024-01-01'" is idiomatic rather
                # than a mismatch. Judged on the text: a string that is not a
                # date still compares as a string, so "order_date = 'hello'"
                # is still reported.
                if TEMPORAL_LITERAL.match(expr.this):
                    return str(XSD.dateTime)
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
            if any(
                t in uri_lower
                for t in ["integer", "decimal", "float", "double", "byte"]
            ):
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

    def _own_aggregates(self, select: exp.Select) -> list[Any]:
        """Aggregate calls belonging to this SELECT's own scope.

        Args:
            select: The SELECT to inspect.

        Returns:
            Aggregate expressions whose nearest enclosing SELECT is *select*.
        """
        agg_types = (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)
        return [
            agg
            for agg in select.find_all(*agg_types)
            if agg.find_ancestor(exp.Select) is select
        ]

    def _grouping_aggregates(self, select: exp.Select) -> list[Any]:
        """This SELECT's aggregates that collapse rows into groups.

        A windowed aggregate does not: ``SUM(total) OVER (PARTITION BY region)``
        computes a value per row and leaves the row count alone, so it imposes
        no GROUP BY at all. Counting it as one made every other selected column
        look ungrouped, and rejected valid window queries outright.

        Args:
            select: The SELECT to inspect.

        Returns:
            Aggregate expressions of this scope that are not windowed.
        """
        return [
            agg
            for agg in self._own_aggregates(select)
            if agg.find_ancestor(exp.Window) is None
        ]

    def _select_aggregates(self, select: exp.Select) -> bool:
        """Whether this SELECT itself applies an aggregate function.

        Aggregates inside a nested subquery belong to that subquery, not here.
        Windowed aggregates count: they read the joined rows, so duplicated
        rows corrupt them exactly as they corrupt a grouped total.

        Args:
            select: The SELECT to inspect.

        Returns:
            True if an aggregate call sits in this SELECT's own scope.
        """
        return bool(self._own_aggregates(select))

    @staticmethod
    def _group_by_keys(group: exp.Group) -> list[exp.Expression]:
        """Every grouping key of a GROUP BY, including grouping-set constructs.

        sqlglot does not put ROLLUP / CUBE / GROUPING SETS members in
        ``Group.expressions``; they hang off separate ``rollup``, ``cube`` and
        ``grouping_sets`` args. Reading only ``expressions`` therefore saw
        ``GROUP BY ROLLUP(country, client)`` as grouping by nothing at all, and
        reported both selected columns as not in the GROUP BY -- an error,
        which blocked every rollup query.

        A column named anywhere in a grouping set is a legal non-aggregated
        selection: super-aggregate rows null it out rather than making it
        ambiguous, which is what the rule is guarding against.

        Args:
            group: The GROUP BY node.

        Returns:
            The grouping keys, with grouping-set nesting flattened away.
        """
        keys: list[exp.Expression] = list(group.expressions)

        for arg in ("rollup", "cube", "grouping_sets"):
            for construct in group.args.get(arg) or []:
                # A grouping set nests its members in Paren/Tuple wrappers, and
                # "()" (the grand total) simply contributes none.
                keys.extend(construct.find_all(exp.Column))

        return keys

    def _validate_aggregation_context(
        self, parsed: exp.Expr, result: OBQCResult
    ) -> None:
        """Rule: Validate GROUP BY completeness for aggregation queries."""
        if not result.has_aggregation:
            return

        for select in parsed.find_all(exp.Select):
            # Aggregation is a property of one SELECT. The query-wide flag above
            # is true if an aggregate appears anywhere, so a subquery's SUM used
            # to make the outer SELECT look like it aggregates -- and every
            # plain column in it was reported as missing from a GROUP BY that
            # the query never needed.
            #
            # Windowed aggregates are excluded: they group nothing, so they
            # cannot be what makes a column need grouping.
            if not self._grouping_aggregates(select):
                continue

            expressions = select.args.get("expressions", [])

            # Aliases declared by this select, mapped to the column they stand
            # for. GROUP BY may name either, and the two must be treated as the
            # same key: "SELECT user_id AS uid ... GROUP BY uid" groups by
            # user_id, but comparing the alias against the source column name
            # reported user_id as not grouped.
            alias_to_column: dict[str, str] = {}
            for projection in expressions:
                if (
                    isinstance(projection, exp.Alias)
                    and projection.alias
                    and isinstance(projection.this, exp.Column)
                ):
                    source = projection.this
                    qualified = source.name.lower()
                    if source.table:
                        qualified = f"{source.table.lower()}.{qualified}"
                    alias_to_column[projection.alias.lower()] = qualified

            # Get GROUP BY columns
            group_by_cols: set[str] = set()
            if select.args.get("group"):
                for group_expr in self._group_by_keys(select.args["group"]):
                    if isinstance(group_expr, exp.Column):
                        gb_col_name = group_expr.name.lower()
                        if group_expr.table:
                            gb_col_name = f"{group_expr.table.lower()}.{gb_col_name}"
                        group_by_cols.add(gb_col_name)

                        # Record the underlying column too, so grouping by an
                        # alias satisfies the check on its source column.
                        #
                        # Only when the name is not itself a real column: a name
                        # that is both an input column and an output alias
                        # resolves to the input column, so "SELECT total AS
                        # user_id ... GROUP BY user_id" groups by orders.user_id
                        # and leaves total ungrouped. Verified against DuckDB,
                        # which rejects exactly that query, and documented for
                        # PostgreSQL.
                        source_col = alias_to_column.get(gb_col_name)
                        if source_col and not self._is_real_column(gb_col_name, result):
                            group_by_cols.add(source_col)
                            group_by_cols.add(source_col.split(".")[-1])

            # Check each SELECT expression
            for expr in expressions:
                col_name: str | None = None
                col_table: str | None = None

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
                    if (
                        qualified not in group_by_cols
                        and col_name.lower() not in group_by_cols
                    ):
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

    def _is_inside_aggregate(self, expr: exp.Expression, select: exp.Select) -> bool:
        """Check if expression is inside an aggregate function of this SELECT.

        Aggregates in a nested subquery are that subquery's; counting them here
        made an outer column look aggregated because some inner aggregate
        happened to mention the same name. A windowed aggregate does not excuse
        a column either -- it collapses nothing, so a bare column beside it
        still needs grouping.
        """
        for agg in self._grouping_aggregates(select):
            for col in agg.find_all(exp.Column):
                if (
                    isinstance(expr, exp.Column)
                    and col.name == expr.name
                    and col.table == expr.table
                ):
                    return True
                if (
                    isinstance(expr, exp.Alias)
                    and isinstance(expr.this, exp.Column)
                    and col.name == expr.this.name
                ):
                    return True
        return False

    def _join_fans_out(self, join_table: str, anchor_table: str) -> bool:
        """Whether joining *join_table* onto *anchor_table* multiplies rows.

        True when the ontology puts *join_table* on the "many" side of the
        relationship between the two: one anchor row can match many joined
        rows, so any measure taken from the anchor side is repeated.

        Args:
            join_table: Table introduced by the JOIN.
            anchor_table: Table its ON condition attaches to.

        Returns:
            True if the join can duplicate anchor rows. False when the joined
            table is the "one" side (a dimension lookup), or when the two are
            not related in the ontology at all.
        """
        if self._schema_cache is None:
            return False

        joined = join_table.lower()
        anchor = anchor_table.lower()

        for rel in self._schema_cache.relationships.values():
            from_t = rel.from_table.lower()
            to_t = rel.to_table.lower()

            # many_to_one is stored from the child: from_table is the many side.
            if rel.relationship_type == "many_to_one":
                many, one = from_t, to_t
            elif rel.relationship_type == "one_to_many":
                many, one = to_t, from_t
            else:
                continue

            if {many, one} == {joined, anchor} and many == joined:
                return True

        return False

    def _detect_fan_trap(self, result: OBQCResult) -> None:
        """Rule: Detect potential fan-trap patterns.

        Prefers the ontology's own ``owl:disjointWith`` axioms (sibling facts
        sharing a dimension — the canonical fan-trap shape) so OBQC and the
        ontology agree by construction. Falls back to the relationship heuristic
        when no disjointness axioms are present (e.g. minimal imports).
        """
        # Only joins whose own SELECT aggregates can inflate a total. The
        # query-wide flag is true if an aggregate appears anywhere, so an
        # unrelated subquery's COUNT(*) used to raise a fan-trap warning about
        # outer joins that aggregate nothing.
        aggregating_joins = [
            j for j in result.parsed_joins if j.get("scope_aggregates")
        ]
        if not aggregating_joins:
            return

        if len(result.parsed_tables) < 2:
            return

        if self._schema_cache is None:
            return

        # --- Axiom-grounded path: disjoint sibling facts in one SELECT --------
        #
        # Scoped like the heuristic below. Reading the disjoint pair off every
        # table named in the query flagged a fact that only appears inside a
        # semi-join filter: "... FROM customers JOIN orders ... WHERE EXISTS
        # (SELECT 1 FROM returns ...)" aggregates orders alone, and returns
        # cannot multiply its rows from inside the subquery.
        disjoint_hits: set[frozenset] = set()
        for scope in result.aggregating_scopes:
            queried = {t.lower() for t in scope}
            disjoint_hits |= {pair for pair in self._disjoint_pairs if pair <= queried}
        if disjoint_hits:
            result.fan_trap_risk = True
            involved = sorted({t for pair in disjoint_hits for t in pair})
            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.FAN_TRAP_DETECTED,
                    severity=OBQCSeverity.WARNING,
                    message=(
                        "Potential fan-trap: query aggregates across tables the ontology "
                        f"declares disjoint (sibling facts sharing a dimension): "
                        f"{', '.join(involved)}"
                    ),
                    location="Query structure",
                    suggestion=(
                        "These facts are at different grains sharing a common dimension. "
                        "Aggregate each fact separately and combine with UNION ALL "
                        "(Composite Fact Layer), or pre-aggregate in CTEs before joining."
                    ),
                    related_entities=involved,
                )
            )
            return

        # --- Heuristic fallback: count fan-out joins (no disjointness axioms)
        #
        # A join multiplies rows only when the table being joined sits on the
        # "many" side *of that join*. The direction is what matters, and it is
        # only meaningful relative to the table the join attaches to.
        #
        # The previous version asked whether the joined table was on the many
        # side of any relationship anywhere in the schema, and counted once per
        # matching relationship rather than once per join. A dimension table
        # with its own foreign keys therefore scored a fan-out for merely
        # existing: "sales JOIN clients JOIN countries" -- many sales to one
        # client to one country, where no row is ever duplicated -- was warned
        # about because clients happens to reference countries.
        # Counted within each aggregating SELECT, not pooled across them. Rows
        # are multiplied by joins in the same query, so two subqueries that
        # fan out once each are two safe aggregations -- summing them reported
        # a fan-trap that exists in neither.
        fan_outs_by_scope: dict[Any, list[str]] = {}

        # Only joins sitting in an aggregating SELECT; a subquery's joins
        # cannot multiply rows the outer query aggregates.
        for join_info in aggregating_joins:
            join_table = join_info.get("table")
            if not join_table:
                continue

            anchors = [
                t
                for t in join_info.get("on_tables", [])
                if t.lower() != join_table.lower()
            ]
            if not anchors:
                # No ON condition to anchor against (CROSS JOIN, or a
                # condition naming no other table); nothing to judge.
                continue

            if any(self._join_fans_out(join_table, anchor) for anchor in anchors):
                scope_id = join_info.get("scope_id")
                fan_outs_by_scope.setdefault(scope_id, []).append(join_table)

        worst_scope = max(fan_outs_by_scope.values(), key=len, default=[])
        one_to_many_count = len(worst_scope)
        involved_tables = worst_scope

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

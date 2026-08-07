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
    VIEW_NOT_JOINABLE = "view_not_joinable"


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
# in common use -- every dialect accepts a string and converts it -- so reading
# these as plain strings reported "order_date >= '2024-01-01'" as a mismatch.
# Only consulted against a temporal column: on its own such a literal is still
# a string, and "email = '2024-01-01'" is an ordinary string comparison.
TEMPORAL_LITERAL = re.compile(
    r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$"
)

# The fan-trap findings OBQC can report, strongest first. Named here so the
# rules, the response and the documentation cannot drift apart -- each has gone
# stale separately while this was built.
KIND_MEASURE_ACROSS_FAN_OUT = "measure_across_fan_out"
KIND_DISJOINT_FACTS = "disjoint_facts"
KIND_MULTIPLE_FAN_OUT_JOINS = "multiple_fan_out_joins"
KIND_CONDITIONAL_ROW_COUNT = "conditional_row_count"

FAN_TRAP_KINDS = (
    KIND_MEASURE_ACROSS_FAN_OUT,
    KIND_DISJOINT_FACTS,
    KIND_MULTIPLE_FAN_OUT_JOINS,
    KIND_CONDITIONAL_ROW_COUNT,
)

# Comparison operators that can relate two tables. A join condition is not
# always an equality: "ON a.starts < b.ends" is an ordinary theta join.
COMPARISON_TYPES = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)

# XSD type names the ontology uses for dates and times.
TEMPORAL_XSD_TYPES = frozenset({"date", "datetime", "time", "gyear", "gyearmonth"})


def derive_view_columns(definition: str | None, dialect: str = "postgres") -> set[str]:
    """Derive a view's output column names from its definition.

    Returns an empty set whenever the answer is not certain, and the caller
    reads that as "do not check this view's columns". Being wrong in the
    permissive direction costs a missed error; being wrong the other way
    blocks a correct query, which is the failure this whole exemption exists
    to prevent.

    Uncertain means: no definition (PostgreSQL withholds the body from
    non-owners), a body that does not parse (view SQL comes back in
    dialect-specific forms), or a ``SELECT *`` whose output depends on tables
    resolved at creation time.

    Args:
        definition: The view's SQL body, if the backend exposed it.
        dialect: sqlglot dialect name for parsing.

    Returns:
        Lower-cased output column names, or an empty set when not derivable.
    """
    if not definition:
        return set()

    try:
        parsed = sqlglot.parse_one(definition, dialect=dialect)
    except Exception as e:
        logger.debug(f"Could not parse view definition ({e}); columns unchecked.")
        return set()

    select = parsed.find(exp.Select)
    if select is None:
        return set()

    columns: set[str] = set()
    for projection in select.expressions:
        # A star anywhere means the output list is not knowable from the text.
        if isinstance(projection, exp.Star) or projection.find(exp.Star):
            return set()

        name = projection.alias_or_name
        if not name:
            # An unnamed expression (e.g. a bare literal) leaves the column
            # name up to the database, so the set would be incomplete.
            return set()
        columns.add(name.lower())

    return columns


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
    # Lower-cased WITH aliases the query referred to, for reporting. The rules
    # do not consult this: whether a reference is a CTE is a property of that
    # reference, not of its name, and is decided where the reference appears.
    cte_names: set[str] = field(default_factory=set)
    # Table references the ontology is expected to describe: every name in
    # parsed_tables except those that resolved to a CTE where they appear.
    checked_tables: list[str] = field(default_factory=list)
    parsed_columns: list[str] = field(default_factory=list)
    # Lower-cased SELECT aliases referenced from ORDER BY / GROUP BY / HAVING.
    # They resolve to select-list output, not to any table's column.
    select_aliases: set[str] = field(default_factory=set)
    # (column reference, tables it may resolve against) per occurrence. Name
    # resolution is scoped to the SELECT a column appears in plus its enclosing
    # ones, so a subquery's tables cannot answer for the outer query.
    column_scopes: list[tuple[str, tuple[tuple[tuple[str, bool], ...], ...]]] = field(
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
    # One entry per fan-trap finding, as data rather than prose: a consumer
    # that keys off fields (an agent reading "success") must be able to see a
    # corrupted aggregate without parsing an English sentence out of warnings.
    fan_trap_findings: list[dict[str, Any]] = field(default_factory=list)
    # Whether the fan-trap verdict actually stopped the query. Computed from
    # the issues that were raised, so it starts false: a run that returns
    # before the rules execute blocked nothing, and saying otherwise
    # contradicted the field's own meaning.
    fan_trap_blocking: bool = False
    # Whether the fan-trap rules actually ran. False means "not checked", which
    # is not the same answer as "nothing found" -- without it, a query validated
    # with no ontology loaded reported detected=false and read as a clean bill
    # of health.
    fan_trap_evaluated: bool = False
    # Whether allow_fan_out actually downgraded a finding that would otherwise
    # have blocked. Distinct from fan_trap_risk, which is also true for
    # findings that never block, so a caller who passed nothing is not told
    # they accepted a risk.
    fan_trap_overridden: bool = False
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
            "obqc_fan_trap": {
                "evaluated": self.fan_trap_evaluated,
                "detected": self.fan_trap_risk,
                "blocking": self.fan_trap_blocking,
                "findings": self.fan_trap_findings,
            },
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
        # id() of table nodes in the query under validation that name a CTE
        # rather than a real table. Per-parse state, reset by validate().
        self._cte_references: set[int] = set()
        # Database views: lower-cased name -> lower-cased output columns.
        # Views are deliberately absent from the ontology (a view pre-joins its
        # sources, so a class for it would restate what the base tables already
        # model), but they are real objects a query may legitimately name. Held
        # here so existence checks pass without the ontology having to describe
        # them. An empty column set means "columns not derivable" -- see
        # derive_view_columns.
        self._known_views: dict[str, set[str]] = {}
        # Signature of the view set last registered, so definitions are
        # re-parsed only when they change.
        self._view_signature: tuple[tuple[str, str], ...] | None = None

    def load_views(self, views: dict[str, set[str]]) -> None:
        """Register database views so queries against them are not rejected.

        Without this, every view query fails: OBQC requires each referenced
        table to appear in the ontology, and views never do. That is the same
        false-positive shape already fixed for catalog tables and CTEs -- a
        correct query blocked because the validator had no way to know the
        object exists.

        Args:
            views: Mapping of view name to its output column names. An empty
                set registers the view's existence while leaving its columns
                unchecked, which is the safe reading when the definition could
                not be parsed or selects ``*``.
        """
        self._known_views = {
            name.lower(): {col.lower() for col in columns}
            for name, columns in views.items()
        }
        logger.debug(f"OBQC registered {len(self._known_views)} views")

    def load_views_from_definitions(
        self, definitions: dict[str, str | None], dialect: str = "postgresql"
    ) -> None:
        """Register views, deriving each one's columns from its definition.

        Safe to call on every validation: the definitions are re-parsed only
        when the set of views actually changed, so a session that discovers a
        schema after its validator was built still picks the views up without
        paying to parse them again on each query.

        Args:
            definitions: Mapping of view name to SQL body (None when withheld).
            dialect: Database dialect name, mapped to sqlglot's.
        """
        signature = tuple(
            sorted((name, body or "") for name, body in definitions.items())
        )
        if signature == self._view_signature:
            return

        sqlglot_dialect = self.DIALECT_MAP.get(dialect, "postgres")
        self.load_views(
            {
                name: derive_view_columns(body, sqlglot_dialect)
                for name, body in definitions.items()
            }
        )
        self._view_signature = signature

    def _view_columns_unknown(self, table_key: str) -> bool:
        """True when *table_key* is a view whose columns could not be derived."""
        return table_key in self._known_views and not self._known_views[table_key]

    def _table_provides_column(self, table_key: str, col_key: str) -> bool:
        """True when *table_key* -- an ontology table or a view -- has *col_key*."""
        if table_key in self._known_views:
            return col_key in self._known_views[table_key]
        if self._schema_cache is None:
            return False
        table = self._schema_cache.tables.get(table_key)
        return table is not None and col_key in table.columns

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

    def validate(
        self,
        sql_query: str,
        dialect: str = "postgresql",
        allow_fan_out: bool = False,
    ) -> OBQCResult:
        """Validate SQL query against loaded ontology.

        Args:
            sql_query: The SQL query to validate
            dialect: Database dialect ("postgresql", "snowflake", "dremio")
            allow_fan_out: Downgrade fan-trap findings from blocking errors to
                warnings. For a caller that has judged the fan-out harmless or
                wants the multiplied rows on purpose; the finding is still
                reported either way.

        Returns:
            OBQCResult with validation findings
        """
        result = OBQCResult(is_valid=True)
        self._cte_references = set()

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

        # Past the guards: the rules below really run, so a "not detected"
        # verdict from here on means the query was checked and came back clean.
        result.fan_trap_evaluated = True

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
        self._detect_fan_trap(result, blocking=not allow_fan_out)

        # Set overall validity
        result.is_valid = not any(
            issue.severity == OBQCSeverity.ERROR for issue in result.issues
        )

        # Whether the fan-trap verdict actually stopped the query, rather than
        # just what the caller asked for: an ambiguous conditional count is
        # reported without blocking even when blocking is on.
        result.fan_trap_blocking = any(
            issue.issue_type == OBQCIssueType.FAN_TRAP_DETECTED
            and issue.severity == OBQCSeverity.ERROR
            for issue in result.issues
        )

        return result

    def _extract_ctes(self, parsed: exp.Expr, result: OBQCResult) -> None:
        """Record which table references resolve to a WITH alias.

        A CTE is a table the query defines for itself, so the ontology never
        describes it. Without this, ``WITH recent AS (...) SELECT ... FROM
        recent`` was rejected outright: ``recent`` was reported as a table not
        found in the ontology, an error, which blocks execution.

        The decision belongs to each *reference*, not to the name. A name can
        be a CTE in one scope and a real table in another, and neither reading
        may leak into the other:

        - Collecting names globally let a CTE hide a real table elsewhere in
          the query, so ``SELECT users.nonexistent FROM users WHERE EXISTS
          (WITH users AS (...) SELECT 1 FROM users)`` reported nothing.
        - Dropping the name from the exemption when it is used both ways fixed
          that but broke the other half: the inner CTE's own columns were then
          checked against the real table, and a valid query was blocked.

        So the exemption is recorded against the table node, and ``cte_names``
        stays purely informational.

        Args:
            parsed: Parsed query.
            result: Result to record the names on.
        """
        for table in parsed.find_all(exp.Table):
            name = table.name
            if name and name.lower() in self._visible_ctes(table):
                result.cte_names.add(name.lower())
                self._cte_references.add(id(table))

    def _is_cte_reference(self, table: exp.Table | None) -> bool:
        """Whether this table node resolves to a WITH alias rather than a table.

        Args:
            table: The reference to classify, or None.

        Returns:
            True if a CTE of that name was in scope at the reference.
        """
        return table is not None and id(table) in self._cte_references

    @staticmethod
    def _visible_ctes(node: exp.Expression) -> set[str]:
        """Lower-cased WITH aliases in scope at *node*.

        Only enclosing WITH clauses are visible. One declared in a sibling
        subquery is not in scope here, which is exactly what makes a name
        usable as a CTE in one place and a real table in another.

        Position within a WITH matters too. A CTE sees the siblings declared
        *before* it and, only when the WITH is RECURSIVE, itself; the query
        body sees all of them. Exposing every name to every reference below
        the WITH skipped validation that should have happened, and matched no
        database: ``WITH orders AS (SELECT nonexistent FROM orders) ...`` reads
        the real table inside the body, and a forward reference to a later
        sibling is an error rather than a CTE.

        Args:
            node: The table reference to resolve from.

        Returns:
            CTE names visible at that position.
        """

        def names_of(ctes: list[exp.Expression]) -> set[str]:
            return {cte.alias_or_name.lower() for cte in ctes if cte.alias_or_name}

        names: set[str] = set()
        previous: exp.Expr | None = None
        current: exp.Expr | None = node

        while current is not None:
            if isinstance(current, exp.With):
                # Reached from inside one of its own CTE definitions: only the
                # ones declared earlier are in scope, plus this one if the
                # WITH is recursive.
                siblings = list(current.expressions)
                if previous is not None and any(cte is previous for cte in siblings):
                    index = next(i for i, cte in enumerate(siblings) if cte is previous)
                    end = index + 1 if current.args.get("recursive") else index
                    names |= names_of(siblings[:end])
                else:
                    names |= names_of(siblings)
            else:
                # A node owning a WITH: its aliases are visible in the body.
                # Found by node type rather than by arg name, since sqlglot
                # spells the key "with" in some versions and "with_" in
                # others, and a lookup by the wrong name silently finds none.
                # A WITH we arrived *through* is skipped -- it was judged
                # above, by position.
                for value in current.args.values():
                    for item in value if isinstance(value, list) else [value]:
                        if isinstance(item, exp.With) and item is not previous:
                            names |= names_of(list(item.expressions))

            previous = current
            current = current.parent

        return names

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

            # A reference the ontology is expected to describe. Judged per
            # reference, so the same name can be a CTE in one scope and a real
            # table needing to exist in another.
            if (
                not self._is_cte_reference(table)
                and table_name not in result.checked_tables
            ):
                result.checked_tables.append(table_name)

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
        scope_cache: dict[int, tuple[tuple[tuple[str, bool], ...], ...]] = {}

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
                source = self._resolve_qualifier_table(owner, column.table)
                if self._is_cte_reference(source):
                    # The qualifier names a CTE here, so its columns come from
                    # that CTE's select list and the ontology cannot judge
                    # them. Dropped at the node, so the same name qualifying a
                    # real table elsewhere is still checked.
                    continue
                if source is not None and source.name:
                    scoped_ref = f"{source.name}.{column.name}"

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
        self,
        select: exp.Select,
        cache: dict[int, tuple[tuple[tuple[str, bool], ...], ...]],
    ) -> tuple[tuple[tuple[str, bool], ...], ...]:
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
            One tuple per scope level, innermost first, each holding
            ``(table name, is a CTE reference)`` pairs.
        """
        cached = cache.get(id(select))
        if cached is not None:
            return cached

        own = tuple(
            dict.fromkeys(
                (t.name, self._is_cte_reference(t))
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
            # Whether any of them is one duplication would corrupt. MAX and
            # COUNT(DISTINCT ...) read the same answer off multiplied rows, so
            # a query using only those is safe across any join shape -- and was
            # being blocked outright by the rules below.
            scope_sensitive = bool(self._duplication_sensitive_aggregates(select))
            if scope_sensitive:
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

            # Tables whose columns this SELECT adds up. A join can only report
            # a wrong number if it multiplies rows a measure is taken from.
            measure_tables = self._measure_tables(select, alias_map)
            # Tables a constant-valued conditional aggregate counts by. Judged
            # separately: those are ambiguous rather than provably wrong.
            counted_tables = self._counted_tables(select, alias_map)

            for join in select.args.get("joins") or []:
                join_info: dict[str, Any] = {
                    "type": join.kind or "INNER",
                    "table": None,
                    "on_condition": None,
                    "scope_aggregates": scope_aggregates,
                    # Whether this scope's aggregates can be corrupted at all.
                    "scope_sensitive": scope_sensitive,
                    # Identifies the owning SELECT so fan-out is counted within
                    # one query rather than pooled across unrelated ones. A
                    # traversal index, not id(select): object addresses vary
                    # per run, and this value must never reach a response.
                    "scope_id": scope_index,
                    # Real table names referenced by the ON condition. Fan-trap
                    # detection needs to know which table this join attaches
                    # to, and the ON condition is the only place that says so.
                    "on_tables": [],
                    # Tables this SELECT sums or averages over, lower-cased.
                    "measure_tables": measure_tables,
                    "counted_tables": counted_tables,
                }

                # How to name this join in a message. A joined subquery has
                # no table name, and the missing-condition error read "JOIN
                # with 'None' has no ON condition".
                joined_item = join.this
                join_info["label"] = (
                    (
                        getattr(joined_item, "alias", "")
                        or getattr(joined_item, "name", "")
                    )
                    if joined_item is not None
                    else ""
                ) or "subquery"

                # Get joined table
                if join.this and isinstance(join.this, exp.Table):
                    join_info["table"] = join.this.name
                    # Judged at the reference: the FK rule cannot speak about a
                    # CTE, but the same name may be a real table elsewhere.
                    join_info["table_is_cte"] = self._is_cte_reference(join.this)

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
                    join_info["on_tables"] = self._condition_tables(
                        on_clause, alias_map
                    )
                elif join_info["table"]:
                    # A comma join states the same relationship in WHERE.
                    # Reading anchors only from ON let the identical query
                    # escape fan-trap detection by being written the older way:
                    # "FROM orders o, order_items i WHERE i.order_id = o.id"
                    # inflates SUM(o.total) exactly as the JOIN ... ON spelling
                    # does, and returned fan_trap_risk=False.
                    join_info["on_tables"] = self._where_anchors(
                        select, join.this, alias_map
                    )

                result.parsed_joins.append(join_info)

    @staticmethod
    def _condition_tables(condition: exp.Expr, alias_map: dict[str, str]) -> list[str]:
        """Real table names a join condition refers to.

        Args:
            condition: An ON clause, or a WHERE predicate acting as one.
            alias_map: The owning scope's alias -> table name map.

        Returns:
            Distinct table names, in the order the condition names them.
        """
        tables: list[str] = []
        for column in condition.find_all(exp.Column):
            if not column.table:
                continue
            # Columns are qualified by alias far more often than by table
            # name, so resolve through the alias map.
            resolved = alias_map.get(column.table.lower(), column.table)
            if resolved not in tables:
                tables.append(resolved)
        return tables

    def _where_anchors(
        self, select: exp.Select, joined: exp.Expr, alias_map: dict[str, str]
    ) -> list[str]:
        """Tables that *joined* is tied to by an equality in this SELECT's WHERE.

        The comma form's counterpart to reading anchors off an ON clause.

        Args:
            select: The SELECT owning the join and the WHERE.
            joined: The table introduced by the comma join.
            alias_map: The scope's alias -> table name map.

        Returns:
            Table names the join attaches to, including the joined table
            itself, matching what an ON clause would have yielded.
        """
        if not isinstance(joined, exp.Table):
            return []

        where = select.args.get("where")
        if where is None:
            return []

        key = (joined.alias or joined.name).lower()
        anchors: list[str] = []

        for eq in where.find_all(exp.EQ):
            # An equality inside a nested subquery belongs to that scope.
            if eq.find_ancestor(exp.Select) is not select:
                continue
            left, right = eq.this, eq.expression
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            qualifiers = {side.table.lower() for side in (left, right) if side.table}
            if key not in qualifiers or len(qualifiers) < 2:
                continue
            for name in self._condition_tables(eq, alias_map):
                if name not in anchors:
                    anchors.append(name)

        return anchors

    def _duplication_sensitive_aggregates(self, select: exp.Select) -> list[Any]:
        """This SELECT's aggregates whose value repeated rows would change.

        MIN and MAX read the same answer off a duplicated set, and so does
        COUNT(DISTINCT ...). SUM, AVG and a plain COUNT do not. ``COUNT(*)``
        counts rows, so it belongs here even though it names no table -- across
        two fan-out joins it returns the product of the two children.

        Args:
            select: The SELECT to inspect.

        Returns:
            Aggregate expressions of this scope that duplication corrupts.
        """
        sensitive = (exp.Sum, exp.Avg, exp.Count)
        return [
            agg
            for agg in self._own_aggregates(select)
            if isinstance(agg, sensitive)
            # sqlglot models DISTINCT as a node wrapping the argument, not as a
            # flag on the call: COUNT(DISTINCT id) is Count(this=Distinct(...)).
            and not isinstance(agg.this, exp.Distinct)
        ]

    @classmethod
    def _value_columns(cls, expr: exp.Expr | None) -> list[exp.Column]:
        """Columns that contribute to *expr*'s value, not to a condition in it.

        A conditional aggregate reads its measure from the branches, never from
        the test: in ``CASE WHEN a.flag THEN b.amount ELSE 0 END`` the value is
        ``b.amount`` and ``a.flag`` only decides whether it is taken. Treating
        both alike attributed the measure to the wrong table.

        Args:
            expr: Expression to walk, or None.

        Returns:
            The value-producing column references, in source order.
        """
        if expr is None:
            return []

        if isinstance(expr, exp.Column):
            return [expr]

        if isinstance(expr, exp.Case):
            # CASE <operand> WHEN ... : the operand is half of a comparison,
            # so it is a condition like the WHEN tests are.
            columns: list[exp.Column] = []
            for branch in expr.args.get("ifs") or []:
                columns += cls._value_columns(branch.args.get("true"))
            return columns + cls._value_columns(expr.args.get("default"))

        if isinstance(expr, exp.If):
            return cls._value_columns(expr.args.get("true")) + cls._value_columns(
                expr.args.get("false")
            )

        columns = []
        for value in expr.args.values():
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, exp.Expression):
                    columns += cls._value_columns(item)
        return columns

    @classmethod
    def _condition_columns(cls, agg: exp.Expr) -> list[exp.Column]:
        """Columns an aggregate only tests, rather than reads a value from.

        Includes a trailing ``FILTER (WHERE ...)``, which lives on the parent
        node and is the same construct as a CASE test written another way.

        Args:
            agg: The aggregate call.

        Returns:
            Its condition-only column references.
        """
        value_ids = {id(column) for column in cls._value_columns(agg)}
        columns = [c for c in agg.find_all(exp.Column) if id(c) not in value_ids]

        parent = agg.parent
        if isinstance(parent, exp.Filter):
            where = parent.args.get("expression")
            if isinstance(where, exp.Expression):
                columns += list(where.find_all(exp.Column))

        return columns

    def _counted_tables(
        self, select: exp.Select, alias_map: dict[str, str]
    ) -> set[str]:
        """Tables a constant-valued conditional aggregate counts rows by.

        ``SUM(CASE WHEN orders.total > 100 THEN 1 ELSE 0 END)`` is COUNT(*)
        over the rows where that holds. It has no measure column, so the
        measure rule cannot see it, yet a join that repeats orders repeats the
        count too -- three for a single qualifying order.

        What it *should* count is not decidable from the SQL. The same shape is
        a ubiquitous correct idiom in a star join: over ``orders JOIN users``,
        ``SUM(CASE WHEN users.segment = 'SMB' THEN 1 ELSE 0 END)`` counts
        orders, which is exactly right, and reads as an inflated count of users
        only if that is what you meant. Both are "count the fine-grained rows
        matching a coarse predicate", so callers get a warning rather than a
        block.

        Only conditions naming a single table qualify. One that also names the
        child counts at the child's grain, which no join corrupts, and an
        unconditional COUNT(*) names nothing at all.

        Args:
            select: The SELECT to inspect.
            alias_map: This scope's alias -> table name map.

        Returns:
            Lower-cased table names such counts are conditioned on.
        """
        counted: set[str] = set()

        for agg in self._duplication_sensitive_aggregates(select):
            if self._value_columns(agg):
                continue
            condition_tables: set[str] = set()
            for column in self._condition_columns(agg):
                if column.table:
                    resolved = alias_map.get(column.table.lower())
                    if resolved:
                        condition_tables.add(resolved.lower())
                    continue

                # Unqualified, exactly as the value side handles it: attribute
                # only when a single table in scope declares the name. Reading
                # qualified names alone missed the same risky query written
                # without the prefix.
                owners = self._tables_declaring(column.name, alias_map)
                if len(owners) == 1:
                    condition_tables.add(owners[0])

            if len(condition_tables) == 1:
                counted.add(next(iter(condition_tables)))

        return counted

    def _measure_tables(
        self, select: exp.Select, alias_map: dict[str, str]
    ) -> set[str]:
        """Tables whose columns this SELECT aggregates in a duplication-sensitive way.

        A fan-out join only produces a wrong number when a measure is taken
        from the side it multiplies. ``SUM(sales.amount)`` over a join that
        repeats each sale is wrong; ``SUM(shipments.weight)`` over the same
        join is right, because the repeated rows *are* the shipments.

        Only aggregates that duplication changes are counted. MIN and MAX are
        indifferent to repeated rows, and so is COUNT(DISTINCT ...); SUM, AVG
        and a plain COUNT of a column are not. ``COUNT(*)`` names no table and
        so attributes to none -- counting joined rows is usually the intent.

        Within an aggregate, only the columns that produce its *value* count.
        A column tested in a condition contributes nothing to the total, so
        ``SUM(CASE WHEN orders.total > 100 THEN order_items.quantity ELSE 0
        END)`` measures order_items and merely filters on orders -- reading
        every column under the aggregate blamed orders and blocked a safe
        conditional aggregate, which is the shape the fan-trap guidance itself
        recommends.

        Args:
            select: The SELECT to inspect.
            alias_map: This scope's alias -> table name map.

        Returns:
            Lower-cased table names whose columns are summed or averaged.
        """
        tables: set[str] = set()

        for agg in self._duplication_sensitive_aggregates(select):
            for column in self._value_columns(agg):
                if column.table:
                    resolved = alias_map.get(column.table.lower())
                    if resolved:
                        tables.add(resolved.lower())
                    continue

                # Unqualified: attribute it only when exactly one table in
                # scope declares the name. Spreading an ambiguous name over
                # every candidate would blame tables the measure may not come
                # from, and a fan-trap finding blocks the query.
                owners = self._tables_declaring(column.name, alias_map)
                if len(owners) == 1:
                    tables.add(owners[0])

        return tables

    def _tables_declaring(self, column: str, alias_map: dict[str, str]) -> list[str]:
        """In-scope tables that declare *column*, lower-cased.

        Args:
            column: Unqualified column name.
            alias_map: This scope's alias -> table name map.

        Returns:
            Distinct lower-cased table names owning the column.
        """
        if self._schema_cache is None:
            return []

        key = column.lower()
        owners = set()
        for table in alias_map.values():
            schema = self._schema_cache.tables.get(table.lower())
            if schema and key in schema.columns:
                owners.add(table.lower())
        return sorted(owners)

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
        source = self._resolve_qualifier_table(select, qualifier)
        return source.name if source is not None and source.name else None

    def _resolve_qualifier_table(
        self, select: exp.Select | None, qualifier: str
    ) -> exp.Table | None:
        """The table node a column's qualifier names.

        The node rather than the name, because the two answer different
        questions: whether a reference is a CTE is a property of *that*
        reference, and a name alone cannot say -- the same name may be a WITH
        alias in one scope and a real table in another.

        Args:
            select: The SELECT the reference appears in.
            qualifier: The alias or table name written before the dot.

        Returns:
            The table node, or None when the qualifier names something else
            (a derived table, an unknown alias).
        """
        key = qualifier.lower()
        node = select
        while node is not None:
            for table in node.find_all(exp.Table):
                if not table.name or table.find_ancestor(exp.Select) is not node:
                    continue
                if key in {table.name.lower(), (table.alias or "").lower()}:
                    return table
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

        # CTE references are already excluded: a WITH alias is defined by the
        # query, not by the ontology.
        for table_name in result.checked_tables:
            # Catalog metadata is not described by the ontology and never will
            # be; demanding it appear there blocks catalog queries outright.
            if table_name in result.catalog_tables:
                continue
            # A view is a real object the ontology deliberately omits, for the
            # same reason: requiring it there would block every view query.
            if table_name.lower() in self._known_views:
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

                # A view's columns come from its definition, not the ontology.
                # When they were derivable, check against them; when they were
                # not, checking anything would invent errors.
                if table_key in self._known_views:
                    view_columns = self._known_views[table_key]
                    if view_columns and col_key not in view_columns:
                        available_cols = sorted(view_columns)[:10]
                        result.issues.append(
                            OBQCIssue(
                                issue_type=OBQCIssueType.COLUMN_NOT_FOUND,
                                severity=OBQCSeverity.ERROR,
                                message=(
                                    f"Column '{col_name}' not found in view "
                                    f"'{table_name}'"
                                ),
                                location="Column reference",
                                suggestion=f"Available columns: {', '.join(available_cols)}",
                                related_entities=[col_ref],
                            )
                        )
                    continue

                # A qualifier naming a CTE was already dropped at extraction,
                # at the reference itself, so anything reaching here is a real
                # table.
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
                        for table_name, is_cte in level
                        if not is_cte
                        and self._table_provides_column(table_name.lower(), col_key)
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
                # one of them. Judged per reference: a name that is a CTE here
                # may be a real table in another scope.
                # A view whose columns could not be derived is undescribable in
                # exactly the sense this check means: an unqualified name that
                # matches no ontology table may well be one of its outputs.
                visible = [pair for level in scope for pair in level]
                describable_tables = [
                    name
                    for name, is_cte in visible
                    if not is_cte
                    and name not in result.catalog_tables
                    and not self._view_columns_unknown(name.lower())
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

            # Every table has to be tied to the rest, so this is a
            # connectivity question rather than a count of conditions. Asking
            # only whether *some* condition existed passed a scope that was
            # partly joined: "FROM orders o, users u, shipments s WHERE
            # o.user_id = u.id" leaves shipments a cross product, and one
            # qualified equality anywhere used to excuse the whole FROM.
            unjoined = self._unjoined_tables(select, tables)
            if not unjoined:
                continue

            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.MISSING_JOIN_CONDITION,
                    severity=OBQCSeverity.ERROR,
                    message=(
                        "Multiple tables without explicit JOIN (Cartesian product): "
                        f"{', '.join(unjoined)} not joined to the rest of the query"
                    ),
                    location="FROM clause",
                    suggestion="Add explicit JOIN ... ON conditions",
                )
            )
            found = True

        return found

    def _flag_view_joins(self, parsed: exp.Expr, result: OBQCResult) -> None:
        """Report SELECTs that join a view to anything else.

        A view is a single entity: it has already applied its own joins and
        grain, and the ontology describes none of that. So none of the
        machinery that makes a join checkable is available for one -- no
        primary key, no foreign keys, no declared cardinality, no place in the
        fan-trap topology. A join to a view is therefore unvalidatable, and an
        unvalidatable join between an aggregate view and a fact table is
        exactly the shape that silently multiplies rows.

        Blocking is the same judgement OBQC already makes for fan-traps:
        refuse the query rather than return numbers nobody can check. Query
        the view on its own, or join the base tables it derives from.

        Judged per SELECT, following the rule in _flag_cartesian_products: a
        view used in one scope must not condemn a join in another.

        Args:
            parsed: Parsed query.
            result: Result to append issues to.
        """
        if not self._known_views:
            return

        for select in parsed.find_all(exp.Select):
            tables = [
                t
                for t in select.find_all(exp.Table)
                if t.name
                and t.find_ancestor(exp.Select) is select
                and id(t) not in self._cte_references
            ]
            if len(tables) < 2:
                continue

            views_used = sorted(
                {t.name for t in tables if t.name.lower() in self._known_views}
            )
            if not views_used:
                continue

            others = sorted(
                {t.name for t in tables if t.name.lower() not in self._known_views}
            )
            # A view joined only to other views is equally unvalidatable.
            joined_to = others or views_used[1:]

            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.VIEW_NOT_JOINABLE,
                    severity=OBQCSeverity.ERROR,
                    message=(
                        f"View '{views_used[0]}' cannot be joined: a view is a "
                        f"single entity whose joins and grain are already fixed, "
                        f"and the ontology does not describe them "
                        f"(joined with: {', '.join(joined_to)})"
                    ),
                    location="FROM clause",
                    suggestion=(
                        f"Query '{views_used[0]}' on its own, or join the base "
                        "tables it derives from so the join can be validated."
                    ),
                    related_entities=views_used,
                )
            )

    def _unjoined_tables(
        self, select: exp.Select, tables: list[exp.Table]
    ) -> list[str]:
        """FROM items of *select* that no condition ties to the others.

        The scope's tables are nodes and its join conditions are edges; a query
        is fully joined when they form one connected component. Anything left
        in a separate component is multiplied against the rest.

        Identity is the alias where there is one, so a self-join stays two
        nodes -- collapsing ``FROM orders a, orders b`` to a single "orders"
        would make an unconditioned self-join look connected to itself.

        Args:
            select: The SELECT whose FROM to judge.
            tables: Its own table references.

        Returns:
            Names of the tables in the smaller components, empty if the scope
            is fully joined.
        """
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: str, b: str) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_a] = root_b

        # Node identity, and the display name to report it under.
        label = {(t.alias or t.name).lower(): t.name for t in tables}
        for key in label:
            find(key)

        def connect_where(where: exp.Expression | None) -> None:
            """Union the qualifiers of each cross-table comparison in WHERE."""
            if where is None:
                return
            for comp in where.find_all(*COMPARISON_TYPES):
                # A comparison inside a nested subquery is that scope's.
                if comp.find_ancestor(exp.Select) is not select:
                    continue
                left, right = comp.this, comp.expression
                if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                    continue
                if left.table and right.table:
                    union(left.table.lower(), right.table.lower())

        # The comma form writes its conditions in WHERE, where they join just
        # as effectively as an ON clause. Any comparison counts, not just
        # equality: "WHERE a.starts < b.ends" relates the two tables too.
        connect_where(select.args.get("where"))

        preceding: list[str] = []
        first = next(iter(label), None)
        if first is not None:
            preceding.append(first)

        for join in select.args.get("joins") or []:
            joined = join.this
            if not isinstance(joined, exp.Table):
                continue
            key = (joined.alias or joined.name).lower()
            find(key)

            on_clause = join.args.get("on")
            if on_clause is not None:
                # An explicit ON is a statement about *this* join, whatever
                # shape the predicate takes. Reading it as pairs of qualified
                # equalities rejected ordinary SQL: "JOIN shipments s ON s.cost
                # > o.total" is a theta join, and "JOIN orders ON users.id =
                # user_id" leaves one side unqualified -- both were reported as
                # Cartesian products and blocked.
                qualifiers = {
                    column.table.lower()
                    for column in on_clause.find_all(exp.Column)
                    if column.table
                }
                others = qualifiers - {key}
                if others:
                    for other in others:
                        union(key, other)
                else:
                    # The ON names nothing else to attach to (a constant
                    # predicate, or only this table's own columns). It is still
                    # an explicit join, so it joins to what came before it.
                    for earlier in preceding:
                        union(key, earlier)
            elif self._join_is_qualified(join):
                # USING and NATURAL name no qualifiers, so there is nothing to
                # read a pair off; they join this item to what came before it.
                for earlier in preceding:
                    union(key, earlier)
            preceding.append(key)

        components: dict[str, list[str]] = {}
        for key, name in label.items():
            components.setdefault(find(key), []).append(name)

        if len(components) < 2:
            return []

        # Report the odd ones out rather than the whole FROM: the largest
        # component is the query, the rest are what fell off it.
        largest = max(components.values(), key=len)
        return sorted(
            name
            for group in components.values()
            if group is not largest
            for name in group
        )

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
        """Whether this SELECT's WHERE relates columns of two different tables.

        The pre-SQL-92 comma form writes its join conditions in WHERE, so a
        scope with no ON clause may still be fully joined.

        Qualified names are what make this decidable: two columns qualified by
        different aliases are a cross-table predicate. Unqualified names are
        ignored -- resolving them would take the full scope, and guessing wrong
        would either excuse a real cross product or block a valid query.

        Args:
            select: The SELECT whose WHERE clause to inspect.

        Returns:
            True if some comparison relates columns of two distinct tables.
        """
        where = select.args.get("where")
        if where is None:
            return False

        # The same operators the connectivity rule accepts, so a scope it
        # judges joined is never then asked for a missing ON clause.
        for eq in where.find_all(*COMPARISON_TYPES):
            # A comparison in a nested subquery belongs to that scope.
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
        # Runs before the cross-product check returns: a view joined without
        # an ON condition is both, and the view finding is the one that
        # explains why no ON condition could have made it valid.
        self._flag_view_joins(parsed, result)

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
                        message=(
                            f"JOIN with '{join_info.get('label') or join_table}' "
                            "has no ON condition"
                        ),
                        location="JOIN clause",
                        suggestion="Add ON condition based on foreign key relationship",
                    )
                )
                continue

            # A CTE has no declared FK relationships -- it is not in the
            # ontology at all -- so the check below could only ever say "may
            # not match", on every join to a WITH alias.
            if join_info.get("table_is_cte"):
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
        for comp in parsed.find_all(*COMPARISON_TYPES):
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
                and not self._is_date_literal_comparison(left, right)
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

    def _is_date_literal_comparison(self, left: exp.Expr, right: exp.Expr) -> bool:
        """Whether this is a temporal value written the only way SQL allows.

        No dialect in common use has a date literal syntax, so a date is
        written as a string and converted: ``order_date >= '2024-01-01'`` is
        idiomatic, not a mismatch.

        Decided from the pair, not from the literal alone. Typing every
        ISO-looking string as temporal fixed date columns but broke string
        ones -- ``email = '2024-01-01'`` is a perfectly ordinary string
        comparison, and was reported as "string vs dateTime".

        Args:
            left: Left operand of the comparison.
            right: Right operand.

        Returns:
            True if one side is a temporal column and the other a string
            literal holding a date or timestamp.
        """
        for column, literal in ((left, right), (right, left)):
            if not isinstance(literal, exp.Literal) or not literal.is_string:
                continue
            if not TEMPORAL_LITERAL.match(literal.this):
                continue
            xsd = self._infer_type(column, column.find_ancestor(exp.Select))
            if xsd and self._type_name(xsd).lower() in TEMPORAL_XSD_TYPES:
                return True
        return False

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

    def _inflated_measures(
        self,
        result: OBQCResult,
        key: str = "measure_tables",
        kind: str = "measure_across_fan_out",
    ) -> list[dict[str, Any]]:
        """Joins that repeat rows a measure in the same SELECT is taken from.

        The single-child fan trap: ``FROM sales JOIN shipments ON
        shipments.sale_id = sales.id`` with ``SUM(sales.amount)`` returns a
        total inflated by every sale that shipped more than once, and an inner
        join silently drops the ones that never shipped. One fan-out join is
        enough to corrupt the number, so no count threshold applies.

        A measure taken from the *many* side is fine -- ``SUM(shipments.cost)``
        over the same join sums each shipment once -- so what matters is the
        direction between the measure's table and the one joined to it, not
        which table the query happened to put in FROM. ``FROM order_items JOIN
        orders`` summing ``orders.total`` inflates exactly as ``FROM orders
        JOIN order_items`` does: both produce one row per item. Each join is
        therefore judged from both ends.

        Args:
            result: Result carrying the extracted joins.
            key: Which per-scope table set to read -- the measures an
                aggregate takes its value from, or the tables a constant-valued
                conditional aggregate counts by.
            kind: Value for the finding's ``kind`` field.

        Returns:
            One finding per (measure table, fan-out table) pair, deduplicated.
        """
        findings: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for join_info in result.parsed_joins:
            if not join_info.get("scope_aggregates"):
                continue

            join_table = join_info.get("table")
            measures = join_info.get(key) or set()
            if not join_table or not measures:
                continue

            anchors = [
                t
                for t in join_info.get("on_tables", [])
                if t.lower() != join_table.lower()
            ]

            for anchor in anchors:
                # Both ends of the edge: whichever side holds the measure, the
                # other one inflates it if the ontology puts it on the many
                # side.
                for measure_table, other in (
                    (anchor, join_table),
                    (join_table, anchor),
                ):
                    if measure_table.lower() not in measures:
                        continue
                    if not self._join_fans_out(other, measure_table):
                        continue

                    pair = (measure_table.lower(), other.lower())
                    if pair in seen:
                        continue
                    seen.add(pair)
                    findings.append(
                        {
                            "kind": kind,
                            "measure_table": measure_table,
                            "fan_out_table": other,
                            "tables": sorted({measure_table, other}),
                        }
                    )

        return findings

    def _detect_fan_trap(self, result: OBQCResult, blocking: bool = True) -> None:
        """Rule: Detect fan-trap patterns.

        Three findings, strongest first:

        1. **Measure across a fan-out join.** The ontology says the joined
           table is on the "many" side of the table a measure is taken from,
           so every row of that measure is repeated and the total is inflated.
           This holds for a *single* join -- ``sales JOIN shipments`` summing
           ``sales.amount`` reports a wrong number with no second fact table in
           sight, and used to pass in silence because the count heuristic below
           needed two fan-outs before it said anything.
        2. **Disjoint sibling facts**, from the ontology's own
           ``owl:disjointWith`` axioms: the canonical fan-trap shape.
        3. **Two or more fan-out joins** in one aggregating SELECT, the
           heuristic fallback for ontologies with no disjointness axioms.

        Args:
            result: Result to record findings on.
            blocking: Whether a finding blocks the query (ERROR) or merely
                annotates it (WARNING).
        """
        severity = OBQCSeverity.ERROR if blocking else OBQCSeverity.WARNING

        def record(finding: dict[str, Any]) -> None:
            """Note a finding of a kind that blocks unless allow_fan_out was set."""
            result.fan_trap_risk = True
            result.fan_trap_findings.append(finding)
            if not blocking:
                result.fan_trap_overridden = True

        # Only joins whose own SELECT aggregates can inflate a total. The
        # query-wide flag is true if an aggregate appears anywhere, so an
        # unrelated subquery's COUNT(*) used to raise a fan-trap warning about
        # outer joins that aggregate nothing.
        # Only joins in a SELECT whose aggregates duplication can corrupt. A
        # scope aggregating solely with MAX, MIN or COUNT(DISTINCT ...) reads
        # the same answer however many times its rows are repeated, so no join
        # shape makes it wrong -- and blocking it contradicted the rule that
        # those aggregates survive a fan-out.
        aggregating_joins = [j for j in result.parsed_joins if j.get("scope_sensitive")]
        if not aggregating_joins:
            return

        if len(result.parsed_tables) < 2:
            return

        if self._schema_cache is None:
            return

        # --- Measure multiplied by a fan-out join ----------------------------
        #
        # The most direct evidence there is: a table is summed, and a join in
        # the same SELECT repeats its rows. One such join is enough.
        inflated = self._inflated_measures(result)
        if inflated:
            for finding in inflated:
                record(finding)
                measure = anchor = finding["measure_table"]
                fanning = finding["fan_out_table"]
                result.issues.append(
                    OBQCIssue(
                        issue_type=OBQCIssueType.FAN_TRAP_DETECTED,
                        severity=severity,
                        message=(
                            f"Fan-trap: aggregating '{measure}' across the join to "
                            f"'{fanning}', which the ontology puts on the many side of "
                            f"'{anchor}'. Each {anchor} row is repeated once per "
                            f"matching {fanning} row, so the total is inflated."
                        ),
                        location="Query structure",
                        suggestion=(
                            f"Pre-aggregate {fanning} in a CTE and join the one row per "
                            f"{anchor} that produces, or aggregate each fact separately "
                            "and combine with UNION ALL (Composite Fact Layer). "
                            "COUNT(DISTINCT ...) also reads correctly across a fan-out."
                        ),
                        related_entities=sorted({measure, fanning, anchor}),
                    )
                )
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
            involved = sorted({t for pair in disjoint_hits for t in pair})
            record({"kind": KIND_DISJOINT_FACTS, "tables": involved})
            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.FAN_TRAP_DETECTED,
                    severity=severity,
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

        # Ambiguous counts, reported but never blocking -- see _counted_tables
        # for why the same shape is both a bug and a common correct idiom.
        counts = self._inflated_measures(
            result, key="counted_tables", kind=KIND_CONDITIONAL_ROW_COUNT
        )
        for finding in counts:
            result.fan_trap_risk = True
            result.fan_trap_findings.append(finding)
            counted, fanning = finding["measure_table"], finding["fan_out_table"]
            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.FAN_TRAP_DETECTED,
                    severity=OBQCSeverity.WARNING,
                    message=(
                        f"Conditional count over '{counted}', whose rows the join to "
                        f"'{fanning}' repeats. This counts {fanning} rows matching a "
                        f"{counted} condition, not {counted} rows -- correct if that "
                        "is what you meant, inflated if it is not."
                    ),
                    location="Query structure",
                    suggestion=(
                        f"To count {counted} rows, use COUNT(DISTINCT {counted}.<key>) "
                        f"or filter with EXISTS instead of joining {fanning}."
                    ),
                    related_entities=sorted({counted, fanning}),
                )
            )

        worst_scope = max(fan_outs_by_scope.values(), key=len, default=[])
        one_to_many_count = len(worst_scope)
        involved_tables = worst_scope

        if one_to_many_count >= 2:
            record(
                {
                    "kind": KIND_MULTIPLE_FAN_OUT_JOINS,
                    "tables": sorted(set(involved_tables)),
                    "join_count": one_to_many_count,
                }
            )
            result.issues.append(
                OBQCIssue(
                    issue_type=OBQCIssueType.FAN_TRAP_DETECTED,
                    severity=severity,
                    message=f"Potential fan-trap: {one_to_many_count} one-to-many joins with aggregation",
                    location="Query structure",
                    suggestion=(
                        "Use UNION ALL pattern for separate aggregations per fact table, "
                        "or use CTEs to pre-aggregate before joining"
                    ),
                    related_entities=list(set(involved_tables)),
                )
            )

"""Ontology generator for creating RDF graphs from database schemas."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, XSD
from wordfreq import word_frequency

from .constants import (
    DEFAULT_BASE_URI,
    OBA_NAMESPACE,
    ONTOLOGY_DESCRIPTION,
    ONTOLOGY_TITLE,
)
from .database_manager import ColumnInfo, TableInfo

# Minimum word frequency threshold (in Zipf scale ~1e-6).
# Words below this are considered non-English / abbreviations.
_WORD_FREQ_THRESHOLD = 1e-6

# Common DB column suffixes that are not English words but are well-understood.
# These should NOT be flagged as cryptic on their own.
_KNOWN_DB_SUFFIXES = {"id", "pk", "fk", "idx", "seq"}

# Whole-word matcher for numeric SQL types, built from
# OntologyGenerator.NUMERIC_SQL_TYPES below. Word boundaries matter: "int" is
# a substring of POINT, INTERVAL and GEOPOINT, so substring matching reads a
# geometry column as numeric.
NUMERIC_TYPE_RE = re.compile(
    r"\b(?:"
    r"int|integer|bigint|smallint|tinyint|mediumint"
    r"|serial|bigserial|smallserial"
    r"|decimal|numeric|dec|fixed"
    r"|float|double|real"
    r"|money|smallmoney|number"
    r")\b"
)

logger = logging.getLogger(__name__)


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

    inferred_relationships: list[InferredRelationship] = field(default_factory=list)
    denormalized_fields: list[DenormalizedField] = field(default_factory=list)
    type_overrides: list[dict[str, str]] = field(default_factory=list)
    missing_descriptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "inferred_relationships": [
                {
                    "source_table": r.source_table,
                    "column": r.column,
                    "target_table": r.target_table,
                    "target_column": r.target_column,
                    "confidence": r.confidence,
                    "pattern_matched": r.pattern_matched,
                }
                for r in self.inferred_relationships
            ],
            "denormalized_fields": [
                {
                    "table": d.table,
                    "column": d.column,
                    "likely_source_table": d.likely_source_table,
                    "warning": d.warning,
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
                "warning_count": len(self.warnings),
            },
        }


class OntologyGenerator:
    """Generates an ontology from a database schema with comprehensive database annotations."""

    # Patterns for inferring FK relationships from column names
    FK_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        # Pattern: suppliercountryid → country/countries
        (r"^(.+?)(?:_)?id$", "suffix_id"),
        # Pattern: id_country → country
        (r"^id[_]?(.+)$", "prefix_id"),
        # Pattern: fk_country → country
        (r"^fk[_]?(.+)$", "prefix_fk"),
        # Pattern: country_fk → country
        (r"^(.+?)[_]?fk$", "suffix_fk"),
        # Pattern: customer_sk → customer (TPC-DS style surrogate keys)
        (r"^(.+?)(?:_)?sk$", "suffix_sk"),
        # Pattern: ss_customer_sk → customer (TPC-DS with table prefix)
        (r"^[a-z]{1,3}_(.+?)(?:_)?sk$", "tpcds_sk"),
    ]

    # Column name patterns that indicate quantities (should be integers)
    QUANTITY_PATTERNS: ClassVar[list[str]] = [
        "quantity",
        "qty",
        "count",
        "cnt",
        "number",
        "num",
        "amount",
        "amt",
    ]

    # Column-name patterns for measure semantics. Ordered most specific
    # first at the point of use: "unit_price" is non-additive even though it
    # contains "price", and "total_amount" is additive even though a bare
    # "total" could be either.
    #
    # Never meaningful to SUM: a rate, a ratio, a per-unit value. AVG and
    # MIN/MAX of these are perfectly ordinary, so only SUM is at issue.
    NON_ADDITIVE_PATTERNS: ClassVar[list[str]] = [
        "unit_price",
        "unitprice",
        "unit_cost",
        "unitcost",
        "_rate",
        "rate_",
        "_pct",
        "_percent",
        "percentage",
        "_ratio",
        "ratio_",
        "margin",
        "_avg",
        "avg_",
        "average",
        "_price",
        "price_",
        "discount_pct",
        "tax_rate",
        "exchange_rate",
        "score",
        "rating",
        "temperature",
        "latitude",
        "longitude",
    ]

    # Summable across entities but not across time: a snapshot of a level
    # rather than a flow. Summing balances over months double-counts.
    SEMI_ADDITIVE_PATTERNS: ClassVar[list[str]] = [
        "balance",
        "_level",
        "level_",
        "stock",
        "on_hand",
        "onhand",
        "inventory",
        "headcount",
        "occupancy",
        "_snapshot",
    ]

    # Meaningful to SUM across every dimension.
    ADDITIVE_PATTERNS: ClassVar[list[str]] = [
        "amount",
        "_amt",
        "amt_",
        "quantity",
        "_qty",
        "qty_",
        "revenue",
        "sales",
        "_total",
        "total_",
        "_sum",
        "sum_",
        "_count",
        "count_",
        "cost",
        "profit",
        "spend",
        "volume",
        "weight",
    ]

    # SQL types that can hold a measure at all. Anything else is an
    # attribute by construction -- SUM() of a string, a date or a geometry
    # is not a question about additivity, it is impossible.
    #
    # Matched with NUMERIC_TYPE_RE (whole words) rather than by substring:
    # "int" is a substring of POINT, INTERVAL and GEOPOINT.
    NUMERIC_SQL_TYPES: ClassVar[list[str]] = [
        "int",
        "integer",
        "bigint",
        "smallint",
        "tinyint",
        "mediumint",
        "serial",
        "bigserial",
        "smallserial",
        "decimal",
        "numeric",
        "float",
        "double",
        "real",
        "money",
        "smallmoney",
        "number",
        "dec",
        "fixed",
    ]

    # Patterns for denormalized text fields
    DENORM_SUFFIXES: ClassVar[list[str]] = [
        "name",
        "title",
        "description",
        "desc",
        "label",
    ]

    def __init__(self, base_uri: str = DEFAULT_BASE_URI):
        self.graph = Graph()
        self.base_uri = Namespace(base_uri)

        # OBA (OrionBelt Analytics) namespace for database schema annotations
        self.oba_ns = Namespace(OBA_NAMESPACE)

        # (table, column) pairs this generator inferred to be foreign keys.
        # Populated by generate_from_schema before any column is written, so
        # classify_measure can treat an inferred key as the identifier it is.
        self._inferred_fk_columns: set[tuple[str, str]] = set()

        self.graph.bind("ns", self.base_uri)
        self.graph.bind("oba", self.oba_ns)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)

        # Quality tracking
        self.quality_report: OntologyQualityReport | None = None
        self._table_lookup: dict[str, TableInfo] = {}
        self._pk_columns: dict[str, list[str]] = {}  # table -> primary key columns
        # Suggestions the last apply_semantic_names() actually matched.
        self._applied_semantic_names: list[dict[str, str]] = []

    def load_from_file(self, file_path: str) -> None:
        """Load an existing ontology from a Turtle file.

        Args:
            file_path: Path to the .ttl file to load
        """
        self.graph = Graph()
        self.graph.parse(file_path, format="turtle")

        # Re-bind namespaces
        self.graph.bind("ns", self.base_uri)
        self.graph.bind("oba", self.oba_ns)
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
        self.graph.bind("oba", self.oba_ns)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)

        logger.info(f"Loaded ontology from string with {len(self.graph)} triples")

    def generate_from_schema(
        self,
        tables_info: list[TableInfo],
        include_inferred_relationships: bool = True,
        annotate_denormalized: bool = True,
        views_info: list[Any] | None = None,
    ) -> str:
        """Generate an ontology from a list of table information with quality enhancements.

        Args:
            tables_info: List of table information from schema analysis
            include_inferred_relationships: Whether to add relationships inferred from naming patterns
            annotate_denormalized: Whether to annotate detected denormalized fields
            views_info: Optional database views. Emitted as oba:View, typed
                apart from owl:Class so table consumers never see them, and
                excluded from every table-driven step below (relationship
                inference, denormalization, disjointness, property chains) --
                a view has no keys for any of those to reason about.

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

        # Relationships are inferred *before* the columns are written,
        # because measure classification needs them: an inferred foreign key
        # is an identifier just as much as a declared one, and the engines
        # that most need inference (ClickHouse has no FKs at all; BigQuery
        # and Dremio rarely carry them) are exactly the ones where every key
        # would otherwise go unrecognised and SUM(customer_id) pass unseen.
        # The relationship triples themselves are still added further down,
        # once the classes they connect exist.
        inferred = (
            self._infer_implicit_relationships(tables_info)
            if include_inferred_relationships
            else []
        )
        self._inferred_fk_columns = {
            (rel.source_table.lower(), rel.column.lower()) for rel in inferred
        }

        # Add all tables and their columns/relationships
        for table_info in tables_info:
            self._add_table_to_ontology(table_info)

        # Views are added after the tables so oba:derivedFrom can point at
        # classes that already exist, and before nothing else: every step
        # below this reasons about keys and relationships, which a view has
        # none of, so views are excluded from all of them by simply not being
        # in tables_info.
        known_tables = {t.name.lower(): t.name for t in tables_info}
        for view_info in views_info or []:
            self._add_view_to_ontology(view_info, known_tables)

        # Materialize the relationships inferred above, now that the classes
        # they connect exist.
        if include_inferred_relationships:
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
                logger.info(
                    f"Annotated denormalized field: {denorm.table}.{denorm.column}"
                )

        # Generate OWL axioms from relationship structure
        self._add_disjoint_axioms(tables_info)
        self._add_property_chain_axioms(tables_info)

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

    def get_quality_report(self) -> OntologyQualityReport | None:
        """Get the quality report from the last generation.

        Returns:
            Quality report or None if no generation has occurred
        """
        return self.quality_report

    def classify_measure(
        self, column: ColumnInfo, table_name: str | None = None
    ) -> tuple[str, str, str] | None:
        """Classify how *column* behaves under aggregation.

        Deterministic, in two tiers of decreasing certainty, because the
        consumer weighs them differently: a structural finding can justify
        refusing a query, a name-pattern finding can only warn.

        Tier 1, structural and certain. A key is an identifier, not a
        measure -- ``SUM(order_id)`` is nonsense however numeric it is. A
        non-numeric column cannot be summed at all. Both follow from schema
        metadata alone, with no interpretation.

        Tier 2, column-name patterns. A unit price or a rate is never
        meaningful to SUM; a balance is summable across accounts but not
        across time. Heuristic, so it is recorded with its reason and marked
        ``name_pattern`` for the consumer to discount.

        A numeric non-key column matching no pattern returns None -- left
        unannotated rather than assumed additive. Consumers read these as
        fact, so an absent value is safer than a wrong one.

        Args:
            column: The column to classify.
            table_name: Owning table, needed to recognise a key this
                generator inferred rather than read from the catalog.

        Returns:
            ``(measure_type, basis, reason)``, or None when undeterminable.
        """
        name = (column.name or "").lower()
        sql_type = (column.data_type or "").lower()

        # -- Tier 1: structural -------------------------------------------
        if column.is_primary_key:
            return (
                "attribute",
                "structural",
                "Primary key: an identifier, not a measure",
            )
        if column.is_foreign_key:
            return (
                "attribute",
                "structural",
                "Foreign key: an identifier, not a measure",
            )
        # A key this generator inferred is as much an identifier as a
        # declared one. Several supported engines carry no FK metadata at
        # all -- ClickHouse has no foreign keys, BigQuery and Dremio rarely
        # declare them -- so without this their keys would all read as
        # unclassified numerics and SUM(customer_id) would pass unseen.
        if table_name and (table_name.lower(), name) in self._inferred_fk_columns:
            return (
                "attribute",
                "structural",
                "Inferred foreign key: an identifier, not a measure",
            )
        # Whole-word match, not substring. "int" is contained in POINT,
        # INTERVAL and GEOPOINT, all of which would otherwise be read as
        # numeric and then fall through unclassified -- leaving SUM(location)
        # on a POINT column entirely unreported.
        if not NUMERIC_TYPE_RE.search(sql_type):
            return (
                "attribute",
                "structural",
                f"Non-numeric SQL type '{column.data_type}' cannot be aggregated as a measure",
            )
        if any(t in sql_type for t in ("bool", "date", "time", "interval")):
            return (
                "attribute",
                "structural",
                f"Temporal or boolean SQL type '{column.data_type}' is not a measure",
            )

        # -- Tier 2: name patterns ----------------------------------------
        # Most specific first: "unit_price" is non-additive despite
        # containing neither an additive nor a semi-additive token, and
        # "total_amount" must not be read as non-additive via "_avg".
        for pattern in self.NON_ADDITIVE_PATTERNS:
            if pattern in name:
                return (
                    "non_additive",
                    "name_pattern",
                    f"Column name contains '{pattern}': a per-unit value, rate or "
                    "ratio is not meaningful to SUM (AVG and MIN/MAX still are)",
                )
        for pattern in self.SEMI_ADDITIVE_PATTERNS:
            if pattern in name:
                return (
                    "semi_additive",
                    "name_pattern",
                    f"Column name contains '{pattern}': a level or snapshot is "
                    "summable across entities but not across time",
                )
        for pattern in self.ADDITIVE_PATTERNS:
            if pattern in name:
                return (
                    "additive",
                    "name_pattern",
                    f"Column name contains '{pattern}': a flow quantity, "
                    "summable across every dimension",
                )

        # Numeric, not a key, no pattern matched. Unknowable without
        # guessing, so say nothing.
        return None

    def _add_view_to_ontology(
        self, view_info: Any, known_tables: dict[str, str]
    ) -> None:
        """Add a database view to the ontology as an oba:View.

        Typed oba:View rather than owl:Class, and annotated oba:viewName
        rather than oba:tableName, so every consumer that reads tables --
        OBQC's schema extraction, the SHACL TableShape, the relationship and
        fan-trap reasoning -- cannot see it. Excluding views rule by rule
        would work only until the next rule is written; this makes the
        separation structural.

        No primary key and no foreign keys are emitted, because a view has
        neither. oba:derivedFrom records provenance only, and is asserted
        solely for sources that resolve to a table already in the ontology --
        a guessed lineage would be consumed as fact by anything reasoning
        over it, so an unknown source is omitted instead.

        Args:
            view_info: A ViewInfo (or compatible object) to add.
            known_tables: Lower-cased name -> real name for tables already in
                the ontology, used to resolve oba:derivedFrom targets.
        """
        view_uri = self.base_uri[self._clean_name(view_info.name)]

        self.graph.add((view_uri, RDF.type, self.oba_ns.View))
        self.graph.add((view_uri, RDFS.label, Literal(view_info.name)))
        self.graph.add((view_uri, self.oba_ns.viewName, Literal(view_info.name)))
        self.graph.add(
            (view_uri, self.oba_ns.schemaName, Literal(view_info.schema or "public"))
        )

        definition = getattr(view_info, "definition", None)
        if definition:
            self.graph.add((view_uri, self.oba_ns.viewDefinition, Literal(definition)))

        comment = getattr(view_info, "comment", None)
        if comment:
            self.graph.add((view_uri, RDFS.comment, Literal(comment)))

        columns = getattr(view_info, "columns", None) or []
        for column in columns:
            self._add_view_column_to_ontology(view_uri, column, view_info.name)
        if columns:
            self.graph.add(
                (view_uri, self.oba_ns.viewColumnCount, Literal(len(columns)))
            )

        for source in getattr(view_info, "source_tables", None) or []:
            real_name = known_tables.get(str(source).lower())
            if real_name is None:
                # The view reads something this ontology does not describe
                # (another schema, a nested view). Asserting a dangling link
                # would be worse than asserting nothing.
                logger.debug(
                    f"View '{view_info.name}' reads unknown source '{source}'; "
                    "omitting oba:derivedFrom for it."
                )
                continue
            self.graph.add(
                (
                    view_uri,
                    self.oba_ns.derivedFrom,
                    self.base_uri[self._clean_name(real_name)],
                )
            )

    def _add_view_column_to_ontology(
        self, view_uri: URIRef, column: Any, view_name: str
    ) -> None:
        """Add a single view column as an oba:ViewColumn.

        Typed apart from owl:DatatypeProperty for the same reason the view is
        typed apart from owl:Class: column extraction keys on that type plus
        oba:tableName, and a view column must not answer to either.

        Args:
            view_uri: URI of the owning view.
            column: A ColumnInfo (or compatible object).
            view_name: Name of the owning view.
        """
        prop_uri = self.base_uri[
            f"{self._clean_name(view_name)}_{self._clean_name(column.name)}"
        ]

        self.graph.add((prop_uri, RDF.type, self.oba_ns.ViewColumn))
        self.graph.add((prop_uri, RDFS.label, Literal(column.name)))
        self.graph.add((prop_uri, self.oba_ns.columnName, Literal(column.name)))
        self.graph.add((prop_uri, self.oba_ns.viewName, Literal(view_name)))
        self.graph.add((prop_uri, RDFS.domain, view_uri))

        data_type = getattr(column, "data_type", None)
        if data_type:
            self.graph.add((prop_uri, self.oba_ns.sqlDataType, Literal(data_type)))

        is_nullable = getattr(column, "is_nullable", None)
        if is_nullable is not None:
            self.graph.add(
                (
                    prop_uri,
                    self.oba_ns.isNullable,
                    Literal("true" if is_nullable else "false"),
                )
            )

    def _add_table_to_ontology(self, table_info: TableInfo) -> None:
        """Add a single table and its columns to the ontology with comprehensive database annotations."""
        # Create proper URI for table class
        table_uri = self.base_uri[self._clean_name(table_info.name)]

        # Define table as a class
        self.graph.add((table_uri, RDF.type, OWL.Class))
        self.graph.add((table_uri, RDFS.label, Literal(table_info.name)))

        # Add comprehensive database-specific annotations
        self.graph.add((table_uri, self.oba_ns.tableName, Literal(table_info.name)))
        self.graph.add((table_uri, self.oba_ns.schemaName, Literal(table_info.schema)))

        if table_info.row_count is not None:
            self.graph.add(
                (table_uri, self.oba_ns.rowCount, Literal(table_info.row_count))
            )

        if table_info.comment:
            self.graph.add((table_uri, RDFS.comment, Literal(table_info.comment)))

        # Add primary key information
        if table_info.primary_keys:
            for pk in table_info.primary_keys:
                self.graph.add((table_uri, self.oba_ns.primaryKey, Literal(pk)))

        # Add columns as properties
        for column in table_info.columns:
            self._add_column_to_ontology(table_uri, column, table_info.name)

        # Define relationships
        for fk in table_info.foreign_keys:
            self._add_relationship_to_ontology(table_uri, fk, table_info.name)

    def _add_column_to_ontology(
        self, table_uri: URIRef, column: ColumnInfo, table_name: str
    ) -> None:
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
        self.graph.add((prop_uri, self.oba_ns.columnName, Literal(column.name)))
        self.graph.add((prop_uri, self.oba_ns.tableName, Literal(table_name)))
        self.graph.add((prop_uri, self.oba_ns.sqlDataType, Literal(column.data_type)))
        self.graph.add((prop_uri, self.oba_ns.isNullable, Literal(column.is_nullable)))

        # Measure semantics: what SUM() may meaningfully touch. Per column,
        # because a denormalized table holds additive measures next to
        # attributes and a table-level role cannot express that. Omitted
        # entirely when it could not be determined without guessing.
        measure = self.classify_measure(column, table_name)
        if measure is not None:
            measure_type, basis, reason = measure
            self.graph.add((prop_uri, self.oba_ns.measureType, Literal(measure_type)))
            self.graph.add((prop_uri, self.oba_ns.measureBasis, Literal(basis)))
            self.graph.add((prop_uri, self.oba_ns.measureReason, Literal(reason)))
        self.graph.add(
            (prop_uri, self.oba_ns.isPrimaryKey, Literal(column.is_primary_key))
        )
        self.graph.add(
            (prop_uri, self.oba_ns.isForeignKey, Literal(column.is_foreign_key))
        )

        # Add SQL query generation hints
        full_column_ref = f"{table_name}.{column.name}"
        self.graph.add((prop_uri, self.oba_ns.sqlReference, Literal(full_column_ref)))

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
            self.graph.add(
                (
                    prop_uri,
                    self.oba_ns.typeOverrideReason,
                    Literal(type_override["reason"]),
                )
            )

        # Note: Primary key and nullability constraints are already captured
        # in the metadata annotations (oba:isPrimaryKey, oba:isNullable).
        # We don't create OWL restriction classes as that would incorrectly
        # make table classes subclasses of restrictions.

        if column.comment:
            self.graph.add((prop_uri, RDFS.comment, Literal(column.comment)))

    def _add_relationship_to_ontology(
        self, table_uri: URIRef, fk: dict[str, str], table_name: str
    ) -> None:
        """Add a foreign key relationship as an object property with comprehensive database annotations."""
        referenced_table = fk.get("referenced_table")
        if not referenced_table:
            logger.warning(f"FK from {table_name} missing referenced_table: {fk}")
            return

        # Create descriptive relationship name
        rel_name = (
            f"{self._clean_name(table_name)}_has_{self._clean_name(referenced_table)}"
        )
        prop_uri = self.base_uri[rel_name]
        referenced_table_uri = self.base_uri[self._clean_name(referenced_table)]

        referenced_column = fk.get("referenced_column", "id")
        fk_column = fk.get("column", "")
        referenced_schema = fk.get("referenced_schema")

        self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
        # Many-to-one: each source row references at most one target row
        self.graph.add((prop_uri, RDF.type, OWL.FunctionalProperty))
        self.graph.add((prop_uri, RDFS.domain, table_uri))
        self.graph.add((prop_uri, RDFS.range, referenced_table_uri))
        self.graph.add(
            (prop_uri, RDFS.label, Literal(f"{table_name} has {referenced_table}"))
        )

        # Add comprehensive database-specific annotations for foreign keys
        self.graph.add((prop_uri, self.oba_ns.foreignKeyColumn, Literal(fk_column)))
        self.graph.add(
            (prop_uri, self.oba_ns.referencedTable, Literal(referenced_table))
        )
        self.graph.add(
            (prop_uri, self.oba_ns.referencedColumn, Literal(referenced_column))
        )
        if referenced_schema:
            self.graph.add(
                (prop_uri, self.oba_ns.referencedSchema, Literal(referenced_schema))
            )

        # Add SQL join condition
        join_condition = (
            f"{table_name}.{fk_column} = {referenced_table}.{referenced_column}"
        )
        self.graph.add(
            (prop_uri, self.oba_ns.sqlJoinCondition, Literal(join_condition))
        )

        # Add relationship type annotation
        self.graph.add((prop_uri, self.oba_ns.relationshipType, Literal("many_to_one")))

        # Shared traversable join edge (finer grain -> coarser grain) so a single
        # SPARQL property path `oba:joinsTo+` answers directed reachability across
        # all FKs without per-property paths. Many-to-one direction only.
        self.graph.add((table_uri, self.oba_ns.joinsTo, referenced_table_uri))

        # Add inverse relationship
        inverse_rel_name = f"{self._clean_name(referenced_table)}_referenced_by_{self._clean_name(table_name)}"
        inverse_prop_uri = self.base_uri[inverse_rel_name]
        self.graph.add((inverse_prop_uri, RDF.type, OWL.ObjectProperty))
        self.graph.add((inverse_prop_uri, RDFS.domain, referenced_table_uri))
        self.graph.add((inverse_prop_uri, RDFS.range, table_uri))
        self.graph.add(
            (
                inverse_prop_uri,
                RDFS.label,
                Literal(f"{referenced_table} referenced by {table_name}"),
            )
        )

        # Add database annotations for inverse relationship
        self.graph.add(
            (inverse_prop_uri, self.oba_ns.relationshipType, Literal("one_to_many"))
        )

        # Link them as inverses (both directions for explicit traversal)
        self.graph.add((prop_uri, OWL.inverseOf, inverse_prop_uri))
        self.graph.add((inverse_prop_uri, OWL.inverseOf, prop_uri))

        logger.debug(
            f"Added declared FK relationship: {table_name}.{fk_column} -> {referenced_table}.{referenced_column}"
        )

    def _clean_name(self, name: str) -> str:
        """Clean a name to make it suitable for URIs."""
        if not name:
            return "unnamed"

        # Replace spaces and special characters with underscores
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        # Ensure it starts with a letter or underscore
        if cleaned and not (cleaned[0].isalpha() or cleaned[0] == "_"):
            cleaned = "_" + cleaned

        return cleaned or "unnamed"

    def _build_table_lookup(self, tables_info: list[TableInfo]) -> None:
        """Build lookup structures for tables and their primary keys."""
        self._table_lookup = {}
        self._pk_columns = {}

        for table in tables_info:
            # Store with multiple key variations for matching
            table_lower = table.name.lower()
            self._table_lookup[table_lower] = table

            # Also store singular/plural variants
            if table_lower.endswith("s"):
                self._table_lookup[table_lower.rstrip("s")] = table
            if table_lower.endswith("es"):
                self._table_lookup[table_lower[:-2]] = table
            if table_lower.endswith("ies"):
                self._table_lookup[table_lower[:-3] + "y"] = table

            # Store primary key columns
            self._pk_columns[table.name] = table.primary_keys

    def _find_matching_table(self, name: str) -> TableInfo | None:
        """Find a table that matches the given name (handles pluralization)."""
        name_lower = name.lower()

        # Direct match
        if name_lower in self._table_lookup:
            return self._table_lookup[name_lower]

        # Try adding common plural suffixes
        for suffix in ["s", "es", "ies"]:
            candidate = name_lower + suffix
            if candidate in self._table_lookup:
                return self._table_lookup[candidate]

        # Try removing plural suffixes
        if name_lower.endswith("ies") and len(name_lower) > 3:
            candidate = name_lower[:-3] + "y"
            if candidate in self._table_lookup:
                return self._table_lookup[candidate]
        if name_lower.endswith("es") and len(name_lower) > 2:
            candidate = name_lower[:-2]
            if candidate in self._table_lookup:
                return self._table_lookup[candidate]
        if name_lower.endswith("s") and len(name_lower) > 1:
            candidate = name_lower[:-1]
            if candidate in self._table_lookup:
                return self._table_lookup[candidate]

        # Try removing 'y' and adding 'ies'
        if name_lower.endswith("y"):
            candidate = name_lower[:-1] + "ies"
            if candidate in self._table_lookup:
                return self._table_lookup[candidate]

        return None

    def _infer_implicit_relationships(
        self, tables_info: list[TableInfo]
    ) -> list[InferredRelationship]:
        """Detect likely FK relationships from column naming patterns.

        This method analyzes column names to find implicit relationships that
        aren't declared as formal foreign keys in the database schema.

        Args:
            tables_info: List of table information

        Returns:
            List of inferred relationships with confidence levels
        """
        inferred: list[InferredRelationship] = []
        existing_fks: set[tuple[str, str]] = set()

        # Build set of existing declared FKs to avoid duplicates
        for table in tables_info:
            for fk in table.foreign_keys:
                existing_fks.add((table.name.lower(), fk["column"].lower()))

        # Build table name variations for matching (include singular/plural forms)
        table_name_variations: dict[str, TableInfo] = {}
        for t in tables_info:
            name_lower = t.name.lower()
            table_name_variations[name_lower] = t

            # Add singular forms for matching in column names
            if name_lower.endswith("ies"):
                table_name_variations[name_lower[:-3] + "y"] = t
            elif name_lower.endswith("es") and len(name_lower) > 2:
                table_name_variations[name_lower[:-2]] = t
            elif name_lower.endswith("s") and len(name_lower) > 1:
                table_name_variations[name_lower[:-1]] = t

        # Sort by length (longer first) to match most specific names first
        sorted_variations = sorted(
            table_name_variations.items(), key=lambda x: len(x[0]), reverse=True
        )

        logger.debug(
            f"FK inference: Table name variations: {list(table_name_variations.keys())}"
        )

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
                if "id" in col_lower and not col.is_primary_key:
                    logger.debug(
                        f"FK inference: Checking column {table.name}.{col.name}"
                    )

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
                        suffix = col_lower[idx + len(target_name) :]
                        if suffix in ["id", "_id", "sk", "_sk", ""]:
                            target_pk = target_table.primary_keys
                            target_column = target_pk[0] if target_pk else "id"

                            confidence = self._calculate_fk_confidence(
                                col, target_table, target_name
                            )

                            inferred.append(
                                InferredRelationship(
                                    source_table=table.name,
                                    column=col.name,
                                    target_table=target_table.name,
                                    target_column=target_column,
                                    confidence=confidence,
                                    pattern_matched="embedded_table_name",
                                )
                            )
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
                        matched_table = self._find_matching_table(potential_table)

                        if matched_table and matched_table.name != table.name:
                            # Determine the likely target column (usually PK)
                            target_pk = matched_table.primary_keys
                            target_column = target_pk[0] if target_pk else "id"

                            # Determine confidence level
                            confidence = self._calculate_fk_confidence(
                                col, matched_table, potential_table
                            )

                            inferred.append(
                                InferredRelationship(
                                    source_table=table.name,
                                    column=col.name,
                                    target_table=matched_table.name,
                                    target_column=target_column,
                                    confidence=confidence,
                                    pattern_matched=pattern_name,
                                )
                            )
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
            if (
                "int" in col_type and "int" in target_pk_type
            ) or col_type == target_pk_type:
                score += 2

        # Higher confidence if exact table name match (vs plural/singular variation)
        if extracted_name.lower() == target_table.name.lower():
            score += 2
        elif extracted_name.lower() + "s" == target_table.name.lower():
            score += 1

        # Higher confidence for common FK naming patterns
        col_lower = column.name.lower()
        if col_lower.endswith(("id", "_id")):
            score += 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        return "low"

    def _detect_denormalized_fields(
        self, tables_info: list[TableInfo]
    ) -> list[DenormalizedField]:
        """Detect likely denormalized text fields that duplicate data from other tables.

        Denormalized fields store redundant copies of data (like customer names in orders)
        which can lead to data inconsistency. This detects such patterns.

        Args:
            tables_info: List of table information

        Returns:
            List of detected denormalized fields
        """
        denormalized: list[DenormalizedField] = []

        # Build table name variations for matching (singular and plural)
        table_name_variations: dict[str, str] = {}
        for t in tables_info:
            name_lower = t.name.lower()
            table_name_variations[name_lower] = t.name

            # Add singular forms
            if name_lower.endswith("ies"):
                table_name_variations[name_lower[:-3] + "y"] = t.name
            elif name_lower.endswith("es"):
                table_name_variations[name_lower[:-2]] = t.name
            elif name_lower.endswith("s"):
                table_name_variations[name_lower[:-1]] = t.name

        for table in tables_info:
            for col in table.columns:
                col_lower = col.name.lower()
                col_type = col.data_type.lower()

                # Only consider text/varchar columns
                if not any(
                    t in col_type for t in ["char", "text", "string", "varchar"]
                ):
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
                        suffix_after_ref = col_lower[
                            col_lower.find(ref_name) + len(ref_name) :
                        ]
                        is_denorm_pattern = (
                            has_denorm_suffix
                            or col_lower == ref_name
                            or suffix_after_ref == ""
                            or suffix_after_ref
                            in ["name", "title", "desc", "description"]
                        )

                        if is_denorm_pattern:
                            denormalized.append(
                                DenormalizedField(
                                    table=table.name,
                                    column=col.name,
                                    likely_source_table=original_table_name,
                                    data_type=col.data_type,
                                    warning=(
                                        f"Column '{col.name}' appears to store denormalized data "
                                        f"from '{original_table_name}'. Consider joining to source "
                                        f"table instead of storing duplicate text."
                                    ),
                                )
                            )
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
        # Many-to-one: each source row references at most one target row
        self.graph.add((prop_uri, RDF.type, OWL.FunctionalProperty))
        self.graph.add((prop_uri, RDFS.domain, source_table_uri))
        self.graph.add((prop_uri, RDFS.range, target_table_uri))
        self.graph.add(
            (
                prop_uri,
                RDFS.label,
                Literal(f"{inferred_rel.source_table} has {inferred_rel.target_table}"),
            )
        )

        # Add database-specific annotations
        self.graph.add(
            (prop_uri, self.oba_ns.foreignKeyColumn, Literal(inferred_rel.column))
        )
        self.graph.add(
            (prop_uri, self.oba_ns.referencedTable, Literal(inferred_rel.target_table))
        )
        self.graph.add(
            (
                prop_uri,
                self.oba_ns.referencedColumn,
                Literal(inferred_rel.target_column),
            )
        )

        # Add SQL join condition
        join_condition = (
            f"{inferred_rel.source_table}.{inferred_rel.column} = "
            f"{inferred_rel.target_table}.{inferred_rel.target_column}"
        )
        self.graph.add(
            (prop_uri, self.oba_ns.sqlJoinCondition, Literal(join_condition))
        )

        # Mark as inferred (not declared FK)
        self.graph.add((prop_uri, self.oba_ns.isInferredRelationship, Literal(True)))
        self.graph.add(
            (
                prop_uri,
                self.oba_ns.inferenceConfidence,
                Literal(inferred_rel.confidence),
            )
        )
        self.graph.add(
            (
                prop_uri,
                self.oba_ns.inferencePattern,
                Literal(inferred_rel.pattern_matched),
            )
        )
        self.graph.add((prop_uri, self.oba_ns.relationshipType, Literal("many_to_one")))

        # Shared traversable join edge (finer grain -> coarser grain); see
        # _add_relationship_to_ontology. Many-to-one direction only.
        self.graph.add((source_table_uri, self.oba_ns.joinsTo, target_table_uri))

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
            self.graph.add(
                (
                    inverse_prop_uri,
                    RDFS.label,
                    Literal(
                        f"{inferred_rel.target_table} referenced by {inferred_rel.source_table}"
                    ),
                )
            )
            self.graph.add(
                (inverse_prop_uri, self.oba_ns.relationshipType, Literal("one_to_many"))
            )
            self.graph.add(
                (inverse_prop_uri, self.oba_ns.isInferredRelationship, Literal(True))
            )

            # Link as inverses (both directions for explicit traversal)
            self.graph.add((prop_uri, OWL.inverseOf, inverse_prop_uri))
            self.graph.add((inverse_prop_uri, OWL.inverseOf, prop_uri))

    def _add_denormalized_field_annotation(self, denorm: DenormalizedField) -> None:
        """Add annotation to a denormalized field in the ontology.

        Args:
            denorm: The denormalized field info
        """
        prop_name = (
            f"{self._clean_name(denorm.table)}_{self._clean_name(denorm.column)}"
        )
        prop_uri = self.base_uri[prop_name]

        if (prop_uri, RDF.type, OWL.DatatypeProperty) in self.graph:
            self.graph.add((prop_uri, self.oba_ns.isDenormalized, Literal(True)))
            self.graph.add(
                (
                    prop_uri,
                    self.oba_ns.likelySourceTable,
                    Literal(denorm.likely_source_table),
                )
            )
            self.graph.add(
                (prop_uri, self.oba_ns.denormalizationWarning, Literal(denorm.warning))
            )

    def _add_disjoint_axioms(self, tables_info: list[TableInfo]) -> None:
        """Add owl:disjointWith axioms between sibling tables that share a dimension.

        Two tables are considered siblings when they both hold foreign keys to
        the same referenced table but have no FK relationship between each other.
        Declaring them disjoint formalizes that their row populations are distinct,
        which directly supports fan-trap detection in Text-to-SQL.

        Args:
            tables_info: List of table information
        """
        # Build mapping: referenced_table -> set of source tables that FK into it
        dimension_to_sources: dict[str, set[str]] = {}
        for table in tables_info:
            for fk in table.foreign_keys:
                ref = fk.get("referenced_table", "")
                if ref:
                    dimension_to_sources.setdefault(ref, set()).add(table.name)

        # Also include inferred relationships already in the graph
        for subj in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            rel_type = self.graph.value(subj, self.oba_ns.relationshipType)
            if str(rel_type) != "many_to_one":
                continue
            ref_table = self.graph.value(subj, self.oba_ns.referencedTable)
            domain = self.graph.value(subj, RDFS.domain)
            if ref_table and domain:
                source_name = self.graph.value(domain, self.oba_ns.tableName)
                if source_name:
                    dimension_to_sources.setdefault(str(ref_table), set()).add(
                        str(source_name)
                    )

        # Build set of tables that have a direct FK relationship between them.
        # Must mirror dimension_to_sources, which includes inferred relationships:
        # if a sibling pair is connected by an *inferred* FK, declaring them
        # disjoint would contradict the relationship and feed OBQC bad input.
        # So derive the exclusion set from both declared FKs and the inferred
        # many-to-one relationships already materialized in the graph.
        fk_pairs: set[tuple[str, str]] = set()
        for table in tables_info:
            for fk in table.foreign_keys:
                ref = fk.get("referenced_table", "")
                if ref:
                    fk_pairs.add((table.name, ref))
                    fk_pairs.add((ref, table.name))

        for subj in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            rel_type = self.graph.value(subj, self.oba_ns.relationshipType)
            if str(rel_type) != "many_to_one":
                continue
            ref_table = self.graph.value(subj, self.oba_ns.referencedTable)
            domain = self.graph.value(subj, RDFS.domain)
            if ref_table and domain:
                source_name = self.graph.value(domain, self.oba_ns.tableName)
                if source_name:
                    fk_pairs.add((str(source_name), str(ref_table)))
                    fk_pairs.add((str(ref_table), str(source_name)))

        # Declare disjoint pairs: siblings sharing a dimension, no FK between them
        declared: set[tuple[str, str]] = set()
        for sources in dimension_to_sources.values():
            source_list = sorted(sources)
            for i, a in enumerate(source_list):
                for b in source_list[i + 1 :]:
                    if (a, b) in fk_pairs or (a, b) in declared:
                        continue
                    uri_a = self.base_uri[self._clean_name(a)]
                    uri_b = self.base_uri[self._clean_name(b)]
                    # Only add if both are actual classes in the graph
                    if (uri_a, RDF.type, OWL.Class) in self.graph and (
                        uri_b,
                        RDF.type,
                        OWL.Class,
                    ) in self.graph:
                        self.graph.add((uri_a, OWL.disjointWith, uri_b))
                        declared.add((a, b))
                        declared.add((b, a))
                        logger.info(
                            f"Added owl:disjointWith: {a} <-> {b} "
                            f"(sibling tables sharing a dimension)"
                        )

    def _add_property_chain_axioms(self, tables_info: list[TableInfo]) -> None:
        """Add owl:propertyChainAxiom for transitive multi-hop join paths.

        When A→B and B→C are FK relationships but A→C has no direct FK,
        a derived property is created with a property-chain axiom linking the
        two hops.  This gives SPARQL reasoners and the query planner explicit
        multi-hop traversal paths.

        Args:
            tables_info: List of table information
        """
        # Collect all many-to-one relationships currently in the graph
        # as (source_table_name, target_table_name, property_uri)
        relationships: list[tuple[str, str, URIRef]] = []
        for subj in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            if not isinstance(subj, URIRef):
                continue
            rel_type = self.graph.value(subj, self.oba_ns.relationshipType)
            if str(rel_type) != "many_to_one":
                continue
            domain = self.graph.value(subj, RDFS.domain)
            range_ = self.graph.value(subj, RDFS.range)
            if domain and range_:
                src = self.graph.value(domain, self.oba_ns.tableName)
                tgt = self.graph.value(range_, self.oba_ns.tableName)
                if src and tgt:
                    relationships.append((str(src), str(tgt), subj))

        # Build lookup: (source, target) -> property URI
        rel_lookup: dict[tuple[str, str], URIRef] = {}
        for rel_src, rel_tgt, rel_uri in relationships:
            rel_lookup[(rel_src, rel_tgt)] = rel_uri

        # Find 2-hop chains A→B→C where A→C has no direct relationship
        declared: set[tuple[str, str]] = set()
        for src_ab, tgt_ab, uri_ab in relationships:
            for src_bc, tgt_bc, uri_bc in relationships:
                if tgt_ab != src_bc:
                    continue  # B must match
                a, c = src_ab, tgt_bc
                if a == c:
                    continue  # Skip cycles
                if (a, c) in rel_lookup:
                    continue  # Direct FK exists, no chain needed
                if (a, c) in declared:
                    continue  # Already created

                # Create the derived chain property
                chain_name = (
                    f"{self._clean_name(a)}_via_"
                    f"{self._clean_name(tgt_ab)}_has_"
                    f"{self._clean_name(c)}"
                )
                chain_uri = self.base_uri[chain_name]

                uri_a = self.base_uri[self._clean_name(a)]
                uri_c = self.base_uri[self._clean_name(c)]

                self.graph.add((chain_uri, RDF.type, OWL.ObjectProperty))
                self.graph.add((chain_uri, RDFS.domain, uri_a))
                self.graph.add((chain_uri, RDFS.range, uri_c))
                self.graph.add(
                    (chain_uri, RDFS.label, Literal(f"{a} via {tgt_ab} has {c}"))
                )

                # Build the RDF list for the property chain
                chain_list = Collection(self.graph, BNode(), [uri_ab, uri_bc])
                self.graph.add(
                    (
                        chain_uri,
                        OWL.propertyChainAxiom,
                        chain_list.uri,
                    )
                )

                declared.add((a, c))
                logger.info(f"Added owl:propertyChainAxiom: {a} -> {tgt_ab} -> {c}")

    def validate_enrichment_completeness(self) -> dict[str, Any]:
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

        # Check classes (OWL Classes that have oba:tableName annotation)
        for subject in self.graph.subjects(RDF.type, OWL.Class):
            # Skip the OWL.Class itself and anonymous nodes
            if subject == OWL.Class or not isinstance(subject, URIRef):
                continue

            # Only count classes that have oba:tableName (i.e., actual table classes)
            table_names = list(self.graph.objects(subject, self.oba_ns.tableName))
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
                for tn in self.graph.objects(subject, self.oba_ns.tableName):
                    table_name = str(tn)
                for cn in self.graph.objects(subject, self.oba_ns.columnName):
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
            has_rel_desc = any(
                self.graph.objects(subject, self.oba_ns.relationshipDescription)
            )
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
                (
                    total_classes
                    + total_props
                    + total_rels
                    - len(classes_without_desc)
                    - len(properties_without_desc)
                    - len(relationships_without_desc)
                )
                / max(1, total_classes + total_props + total_rels)
            ),
        }

    def _map_sql_to_xsd(
        self, sql_type: str, column_name: str = "", table_name: str = ""
    ) -> tuple[URIRef | None, dict[str, str] | None]:
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
        if (
            col_lower
            and any(p in col_lower for p in self.QUANTITY_PATTERNS)
            and any(
                t in sql_type_lower
                for t in ["float", "double", "decimal", "numeric", "real"]
            )
        ):
            type_override = {
                "table": table_name,
                "column": column_name,
                "original_sql_type": sql_type,
                "original_xsd_type": "xsd:double",
                "overridden_xsd_type": "xsd:integer",
                "reason": f"Column name '{column_name}' indicates quantity/count (should be integer)",
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

    def apply_semantic_descriptions(self, descriptions: dict[str, Any]) -> None:
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
                        self.graph.add(
                            (
                                table_uri,
                                RDFS.comment,
                                Literal(table_desc["business_description"]),
                            )
                        )

                    if "table_type" in table_desc:
                        self.graph.add(
                            (
                                table_uri,
                                self.oba_ns.tableType,
                                Literal(table_desc["table_type"]),
                            )
                        )

                    if "usage_notes" in table_desc:
                        self.graph.add(
                            (
                                table_uri,
                                self.oba_ns.usageNotes,
                                Literal(table_desc["usage_notes"]),
                            )
                        )

        # Apply column descriptions
        if "columns" in descriptions:
            for column_ref, column_desc in descriptions["columns"].items():
                # Parse table.column format
                if "." in column_ref:
                    table_name, column_name = column_ref.split(".", 1)
                    prop_name = f"{self._clean_name(table_name)}_{self._clean_name(column_name)}"
                    prop_uri = self.base_uri[prop_name]

                    if (prop_uri, RDF.type, OWL.DatatypeProperty) in self.graph or (
                        prop_uri,
                        RDF.type,
                        OWL.ObjectProperty,
                    ) in self.graph:
                        if "business_description" in column_desc:
                            # Remove existing comment if present
                            self.graph.remove((prop_uri, RDFS.comment, None))
                            # Add new rich description as rdfs:comment
                            self.graph.add(
                                (
                                    prop_uri,
                                    RDFS.comment,
                                    Literal(column_desc["business_description"]),
                                )
                            )

                        if "data_characteristics" in column_desc:
                            self.graph.add(
                                (
                                    prop_uri,
                                    self.oba_ns.dataCharacteristics,
                                    Literal(column_desc["data_characteristics"]),
                                )
                            )

                        if "business_rules" in column_desc:
                            self.graph.add(
                                (
                                    prop_uri,
                                    self.oba_ns.businessRules,
                                    Literal(column_desc["business_rules"]),
                                )
                            )

        # Apply relationship descriptions
        if "relationships" in descriptions:
            for rel_key, rel_desc in descriptions["relationships"].items():
                # Parse relationship key format
                # Expected format: "from_table.column -> to_table.column"
                if " -> " in rel_key:
                    from_part, to_part = rel_key.split(" -> ")
                    from_table = (
                        from_part.split(".")[0] if "." in from_part else from_part
                    )
                    to_table = to_part.split(".")[0] if "." in to_part else to_part

                    rel_name = f"{self._clean_name(from_table)}_has_{self._clean_name(to_table)}"
                    rel_uri = self.base_uri[rel_name]

                    if (rel_uri, RDF.type, OWL.ObjectProperty) in self.graph:
                        if "description" in rel_desc:
                            self.graph.add(
                                (
                                    rel_uri,
                                    self.oba_ns.relationshipDescription,
                                    Literal(rel_desc["description"]),
                                )
                            )

                        if "cardinality" in rel_desc:
                            self.graph.add(
                                (
                                    rel_uri,
                                    self.oba_ns.cardinality,
                                    Literal(rel_desc["cardinality"]),
                                )
                            )

                        if "business_rule" in rel_desc:
                            self.graph.add(
                                (
                                    rel_uri,
                                    self.oba_ns.businessRule,
                                    Literal(rel_desc["business_rule"]),
                                )
                            )

        logger.info("Finished applying semantic descriptions")

    def serialize_ontology(self) -> str:
        """Serialize the current ontology to Turtle format."""
        return self.graph.serialize(format="turtle")

    def get_enrichment_data(
        self, tables_info: list[TableInfo], sample_data: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Generate enrichment data structure for LLM processing.

        Args:
            tables_info: List of table information
            sample_data: Dictionary mapping table names to sample rows

        Returns:
            Dictionary with schema_data and instructions for LLM enrichment
        """
        schema_data = []

        for table_info in tables_info:
            table_data: dict[str, Any] = {
                "table_name": table_info.name,
                "schema": table_info.schema,
                "row_count": table_info.row_count,
                "columns": [],
            }

            # Add column information
            for column in table_info.columns:
                column_data = {
                    "name": column.name,
                    "data_type": column.data_type,
                    "is_nullable": column.is_nullable,
                    "is_primary_key": column.is_primary_key,
                    "is_foreign_key": column.is_foreign_key,
                }
                if column.comment:
                    column_data["comment"] = column.comment
                if column.is_foreign_key and column.foreign_key_table:
                    column_data["foreign_key_table"] = column.foreign_key_table
                    column_data["foreign_key_column"] = column.foreign_key_column
                table_data["columns"].append(column_data)

            # Add sample data if available (limit to 3 rows)
            if sample_data.get(table_info.name):
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
                        "description": "Business description",
                    }
                ],
                "properties": [
                    {
                        "table_name": "table_name",
                        "original_name": "column_name",
                        "suggested_name": "semanticPropertyName",
                        "description": "Business meaning",
                    }
                ],
                "relationships": [
                    {
                        "from_table": "source_table",
                        "to_table": "target_table",
                        "suggested_name": "semanticRelationshipName",
                        "description": "Relationship meaning",
                    }
                ],
            },
            "guidelines": [
                "Use clear, business-oriented terminology",
                "Provide meaningful descriptions based on table and column names and sample data",
                "Suggest appropriate semantic names that reflect business concepts",
                "For relationships, describe the business meaning of the association",
            ],
        }

        return {"schema_data": schema_data, "instructions": instructions}

    def apply_enrichment(
        self, enrichment_suggestions: dict[str, list[dict[str, Any]]]
    ) -> None:
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
                prop_name = (
                    f"{self._clean_name(table_name)}_{self._clean_name(original_name)}"
                )
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
                rel_name = (
                    f"{self._clean_name(from_table)}_has_{self._clean_name(to_table)}"
                )
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

    def enrich_with_llm(
        self, tables_info: list[TableInfo], sample_data: dict[str, list[dict[str, Any]]]
    ) -> str:
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

    def extract_names_for_review(self, compact: bool = False) -> dict[str, Any]:
        """Extract all class, property, and relationship names from the ontology for LLM review.

        This method analyzes the current ontology graph and extracts all names that might
        need improvement - abbreviations, cryptic identifiers, or technical names that
        could be made more business-friendly.

        Args:
            compact: If True, return full metadata only for cryptic items.
                Non-cryptic items are listed as name-only entries to reduce
                token usage for LLM clients with limited context windows.

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

            class_info: dict[str, Any] = {
                "uri": str(subject),
                "local_name": (
                    str(subject).split("/")[-1] if "/" in str(subject) else str(subject)
                ),
                "current_label": None,
                "table_name": None,
                "schema_name": None,
                "row_count": None,
                "comment": None,
            }

            # Get current label
            for label in self.graph.objects(subject, RDFS.label):
                class_info["current_label"] = str(label)

            # Get database annotations
            for table_name in self.graph.objects(subject, self.oba_ns.tableName):
                class_info["table_name"] = str(table_name)
            for schema_name in self.graph.objects(subject, self.oba_ns.schemaName):
                class_info["schema_name"] = str(schema_name)
            for row_count in self.graph.objects(subject, self.oba_ns.rowCount):
                class_info["row_count"] = int(str(row_count))
            for comment in self.graph.objects(subject, RDFS.comment):
                class_info["comment"] = str(comment)

            # Analyze if name looks cryptic
            name = class_info["current_label"] or class_info["local_name"]
            class_info["needs_review"] = self._analyze_name_quality(name)

            classes.append(class_info)

        # Extract data properties (columns)
        for subject in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            prop_info: dict[str, Any] = {
                "uri": str(subject),
                "local_name": (
                    str(subject).split("/")[-1] if "/" in str(subject) else str(subject)
                ),
                "current_label": None,
                "column_name": None,
                "table_name": None,
                "sql_data_type": None,
                "is_primary_key": False,
                "is_foreign_key": False,
                "comment": None,
            }

            # Get current label
            for label in self.graph.objects(subject, RDFS.label):
                prop_info["current_label"] = str(label)

            # Get database annotations
            for col_name in self.graph.objects(subject, self.oba_ns.columnName):
                prop_info["column_name"] = str(col_name)
            for table_name in self.graph.objects(subject, self.oba_ns.tableName):
                prop_info["table_name"] = str(table_name)
            for sql_type in self.graph.objects(subject, self.oba_ns.sqlDataType):
                prop_info["sql_data_type"] = str(sql_type)
            for is_pk in self.graph.objects(subject, self.oba_ns.isPrimaryKey):
                prop_info["is_primary_key"] = str(is_pk).lower() == "true"
            for is_fk in self.graph.objects(subject, self.oba_ns.isForeignKey):
                prop_info["is_foreign_key"] = str(is_fk).lower() == "true"
            for comment in self.graph.objects(subject, RDFS.comment):
                prop_info["comment"] = str(comment)

            # Analyze if name looks cryptic
            name = prop_info["current_label"] or prop_info["local_name"]
            prop_info["needs_review"] = self._analyze_name_quality(name)

            properties.append(prop_info)

        # Extract object properties (relationships)
        for subject in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            rel_info: dict[str, Any] = {
                "uri": str(subject),
                "local_name": (
                    str(subject).split("/")[-1] if "/" in str(subject) else str(subject)
                ),
                "current_label": None,
                "foreign_key_column": None,
                "referenced_table": None,
                "relationship_type": None,
                "comment": None,
            }

            # Get current label
            for label in self.graph.objects(subject, RDFS.label):
                rel_info["current_label"] = str(label)

            # Get database annotations
            for fk_col in self.graph.objects(subject, self.oba_ns.foreignKeyColumn):
                rel_info["foreign_key_column"] = str(fk_col)
            for ref_table in self.graph.objects(subject, self.oba_ns.referencedTable):
                rel_info["referenced_table"] = str(ref_table)
            for rel_type in self.graph.objects(subject, self.oba_ns.relationshipType):
                rel_info["relationship_type"] = str(rel_type)
            for comment in self.graph.objects(subject, RDFS.comment):
                rel_info["comment"] = str(comment)

            # Analyze if name looks cryptic
            name = rel_info["current_label"] or rel_info["local_name"]
            rel_info["needs_review"] = self._analyze_name_quality(name)

            relationships.append(rel_info)

        # Generate analysis hints
        cryptic_classes = [
            c for c in classes if c.get("needs_review", {}).get("is_cryptic")
        ]
        cryptic_props = [
            p for p in properties if p.get("needs_review", {}).get("is_cryptic")
        ]
        cryptic_rels = [
            r for r in relationships if r.get("needs_review", {}).get("is_cryptic")
        ]

        if cryptic_classes:
            analysis_hints.append(
                f"Found {len(cryptic_classes)} class names that may need improvement"
            )
        if cryptic_props:
            analysis_hints.append(
                f"Found {len(cryptic_props)} property names that may need improvement"
            )
        if cryptic_rels:
            analysis_hints.append(
                f"Found {len(cryptic_rels)} relationship names that may need improvement"
            )

        # In compact mode, return full metadata only for cryptic items.
        # Non-cryptic items are reduced to name-only entries so LLM clients
        # with limited context windows can still see all names at a glance.
        if compact:

            def _compact_class(c: dict[str, Any]) -> dict[str, Any]:
                if c.get("needs_review", {}).get("is_cryptic"):
                    return c
                return {
                    "local_name": c["local_name"],
                    "table_name": c.get("table_name"),
                }

            def _compact_prop(p: dict[str, Any]) -> dict[str, Any]:
                if p.get("needs_review", {}).get("is_cryptic"):
                    return p
                return {
                    "local_name": p["local_name"],
                    "table_name": p.get("table_name"),
                }

            def _compact_rel(r: dict[str, Any]) -> dict[str, Any]:
                if r.get("needs_review", {}).get("is_cryptic"):
                    return r
                return {"local_name": r["local_name"]}

            classes = [_compact_class(c) for c in classes]
            properties = [_compact_prop(p) for p in properties]
            relationships = [_compact_rel(r) for r in relationships]

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
                "relationships_needing_review": len(cryptic_rels),
            },
        }

    @staticmethod
    def _is_known_word(token: str) -> bool:
        """Check if a token is a recognised English word using word frequency data.

        Args:
            token: Lowercase alphabetic token to check.

        Returns:
            True if the token appears in the wordfreq corpus above the threshold.
        """
        if not token or not token.isalpha():
            return False
        if token in _KNOWN_DB_SUFFIXES:
            return True
        return word_frequency(token, "en") >= _WORD_FREQ_THRESHOLD

    def _analyze_name_quality(self, name: str) -> dict[str, Any]:
        """Analyze if a name looks like an abbreviation or cryptic identifier.

        Uses a combination of structural heuristics (length, casing, suffix
        patterns) **and** NLP word-frequency lookup via *wordfreq* to catch
        concatenated abbreviations (e.g. ``acctbal``, ``custaddr``) that pure
        regex would miss.

        Args:
            name: The name to analyze

        Returns:
            Dictionary with analysis results
        """
        if not name:
            return {"is_cryptic": True, "reasons": ["Empty name"]}

        reasons: list[str] = []
        is_cryptic = False

        # --- structural checks ------------------------------------------

        # Very short names (likely abbreviations)
        if len(name) <= 3 and name.lower() not in _KNOWN_DB_SUFFIXES:
            is_cryptic = True
            reasons.append("Very short name (≤3 chars) - likely abbreviation")

        # All uppercase (common for abbreviations/acronyms)
        if name.isupper() and len(name) > 1:
            is_cryptic = True
            reasons.append("All uppercase - likely acronym")

        # Common cryptic suffix/prefix patterns
        cryptic_patterns = [
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

        # Numeric suffix (versions / partitions)
        if re.search(r"\d+$", name):
            reasons.append("Contains numeric suffix")

        # --- NLP word-frequency check ------------------------------------
        # Split on underscores, then check each token against the corpus.
        parts = [p for p in name.lower().split("_") if p.isalpha()]
        if parts:
            non_words = [p for p in parts if not self._is_known_word(p)]
            # If more than half the tokens are not real words → cryptic
            if len(non_words) > len(parts) / 2:
                is_cryptic = True
                reasons.append(
                    f"Contains non-dictionary tokens: {', '.join(non_words)}"
                )
            elif non_words:
                # Some tokens unrecognised — mention but don't auto-flag
                reasons.append(f"Possible abbreviations: {', '.join(non_words)}")

        return {
            "is_cryptic": is_cryptic,
            "reasons": reasons,
            "confidence": (
                "high"
                if len(reasons) >= 2
                else "medium" if len(reasons) == 1 else "low"
            ),
        }

    def apply_semantic_names(self, name_suggestions: dict[str, Any]) -> str:
        """Apply suggested semantic names to the ontology.

        This method takes LLM-generated name suggestions and updates the ontology
        labels to use more business-friendly terminology.

        Args:
            name_suggestions: Dictionary containing:
                - classes: List of {original_name, suggested_name, description}
                - properties: List of {original_name, suggested_name, description}
                - relationships: List of {original_name, suggested_name, description}

        Returns:
            Updated ontology in Turtle format. The subset that was actually
            applied is available afterwards from
            :meth:`applied_semantic_names`.
        """
        logger.info("Applying semantic name suggestions to ontology")
        changes_made = 0

        # Suggestions naming something that is not in the ontology are skipped
        # silently, so the requested list is not evidence of what happened. Record
        # what was actually applied, for callers that must not act on a name the
        # ontology rejected -- indexing a typo'd target into schema search would
        # invent an element that does not exist.
        self._applied_semantic_names = []

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
                    if suggested or description:
                        self._applied_semantic_names.append(
                            {
                                "original_name": original,
                                "suggested_name": suggested or "",
                                "description": description or "",
                                # Raw SQL name from the ontology, not the
                                # caller's string: suggestions are generated
                                # from the ontology and so carry _clean_name()
                                # output ("order_items"), while the schema --
                                # and anything keyed by it -- uses the original
                                # ("order-items").
                                "table_name": self._raw_table_name(class_uri),
                                "column_name": "",
                                "related_table": "",
                            }
                        )
                    if suggested:
                        # Update the label
                        self.graph.remove((class_uri, RDFS.label, None))
                        self.graph.add((class_uri, RDFS.label, Literal(suggested)))
                        # Also add a semantic name annotation
                        self.graph.add(
                            (class_uri, self.oba_ns.semanticName, Literal(suggested))
                        )
                        changes_made += 1

                    if description:
                        # Add or update description as rdfs:comment
                        self.graph.remove((class_uri, RDFS.comment, None))
                        self.graph.add((class_uri, RDFS.comment, Literal(description)))

        # Apply property name suggestions (two-pass to detect and disambiguate duplicates)
        if "properties" in name_suggestions:
            # First pass: resolve URIs and collect proposed changes
            proposed_changes = []
            for suggestion in name_suggestions["properties"]:
                original = suggestion.get("original_name")
                suggested = suggestion.get("suggested_name")
                description = suggestion.get("description")
                table_name = suggestion.get("table_name")

                if not original:
                    continue

                # Find the property URI - might need table context
                if table_name:
                    prop_name = (
                        f"{self._clean_name(table_name)}_{self._clean_name(original)}"
                    )
                else:
                    prop_name = self._clean_name(original)

                prop_uri = self.base_uri[prop_name]

                # Check if this property exists (as data or object property)
                if (prop_uri, RDF.type, OWL.DatatypeProperty) in self.graph or (
                    prop_uri,
                    RDF.type,
                    OWL.ObjectProperty,
                ) in self.graph:
                    proposed_changes.append(
                        {
                            "prop_uri": prop_uri,
                            "suggested": suggested,
                            "description": description,
                            "table_name": table_name,
                            "original_name": original,
                        }
                    )

            # Detect duplicate suggested labels and disambiguate with table context
            if proposed_changes:
                # Collect existing labels from properties NOT being changed
                changing_uris = {
                    c["prop_uri"] for c in proposed_changes if c["suggested"]
                }
                existing_labels: dict[str, URIRef] = {}
                for rdf_type in (OWL.DatatypeProperty, OWL.ObjectProperty):
                    for s, _, _ in self.graph.triples((None, RDF.type, rdf_type)):
                        if isinstance(s, URIRef) and s not in changing_uris:
                            for label in self.graph.objects(s, RDFS.label):
                                existing_labels[str(label).lower()] = s

                # Group proposed changes by suggested label (case-insensitive)
                label_groups: dict[str, list] = {}
                for change in proposed_changes:
                    if change["suggested"]:
                        key = change["suggested"].lower()
                        label_groups.setdefault(key, []).append(change)

                # Disambiguate labels that appear more than once or conflict with existing
                for label_key, changes in label_groups.items():
                    needs_disambig = len(changes) > 1 or label_key in existing_labels
                    if needs_disambig:
                        for change in changes:
                            if change["table_name"]:
                                # Prefer the class's rdfs:label (may have been enriched above)
                                class_uri = self.base_uri[
                                    self._clean_name(change["table_name"])
                                ]
                                class_label = None
                                for label in self.graph.objects(class_uri, RDFS.label):
                                    class_label = str(label)
                                    break
                                qualifier = class_label or change["table_name"]
                                change["suggested"] = (
                                    f"{change['suggested']} ({qualifier})"
                                )

            # Second pass: apply all changes
            for change in proposed_changes:
                if change["suggested"] or change["description"]:
                    self._applied_semantic_names.append(
                        {
                            "original_name": change["original_name"],
                            # The disambiguated label, not the requested one --
                            # this is what the ontology now says.
                            "suggested_name": change["suggested"] or "",
                            "description": change["description"] or "",
                            # Raw SQL names from the ontology rather than the
                            # caller's, which carry _clean_name() output.
                            "table_name": self._raw_table_name(change["prop_uri"]),
                            "column_name": self._raw_column_name(change["prop_uri"]),
                            "related_table": "",
                        }
                    )
                if change["suggested"]:
                    self.graph.remove((change["prop_uri"], RDFS.label, None))
                    self.graph.add(
                        (change["prop_uri"], RDFS.label, Literal(change["suggested"]))
                    )
                    self.graph.add(
                        (
                            change["prop_uri"],
                            self.oba_ns.semanticName,
                            Literal(change["suggested"]),
                        )
                    )
                    changes_made += 1

                if change["description"]:
                    self.graph.remove((change["prop_uri"], RDFS.comment, None))
                    self.graph.add(
                        (
                            change["prop_uri"],
                            RDFS.comment,
                            Literal(change["description"]),
                        )
                    )

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
                    if suggested or description:
                        # A relationship is identified by the pair of tables it
                        # joins, taken from its domain and range.
                        domain = self.graph.value(rel_uri, RDFS.domain)
                        range_ = self.graph.value(rel_uri, RDFS.range)
                        self._applied_semantic_names.append(
                            {
                                "original_name": original,
                                "suggested_name": suggested or "",
                                "description": description or "",
                                "table_name": self._raw_table_name(domain),
                                "column_name": "",
                                "related_table": self._raw_table_name(range_),
                            }
                        )
                    if suggested:
                        # Update the label
                        self.graph.remove((rel_uri, RDFS.label, None))
                        self.graph.add((rel_uri, RDFS.label, Literal(suggested)))
                        # Also add a semantic name annotation
                        self.graph.add(
                            (rel_uri, self.oba_ns.semanticName, Literal(suggested))
                        )
                        changes_made += 1

                    if description:
                        # Add or update description as rdfs:comment
                        self.graph.remove((rel_uri, RDFS.comment, None))
                        self.graph.add((rel_uri, RDFS.comment, Literal(description)))

        logger.info(f"Applied {changes_made} semantic name changes to ontology")

        return self.graph.serialize(format="turtle")

    def _raw_table_name(self, uri: Any) -> str:
        """Return the original SQL table name a URI is annotated with.

        Class and property URIs are built from :meth:`_clean_name`, which
        rewrites characters that are illegal in a URI local name -- so the URI
        cannot be turned back into the table name it came from. ``oba:tableName``
        preserves the original, and is the only reliable way back to the schema.

        Args:
            uri: Class or property URI, or None.

        Returns:
            The annotated table name, or "" when the URI is None or carries no
            annotation (an ontology loaded from elsewhere may have none).
        """
        if uri is None:
            return ""
        value = self.graph.value(uri, self.oba_ns.tableName)
        return str(value) if value is not None else ""

    def _raw_column_name(self, uri: Any) -> str:
        """Return the original SQL column name a property URI is annotated with.

        Args:
            uri: Property URI, or None.

        Returns:
            The annotated column name, or "" when absent (object properties
            representing relationships carry no column).
        """
        if uri is None:
            return ""
        value = self.graph.value(uri, self.oba_ns.columnName)
        return str(value) if value is not None else ""

    def applied_semantic_names(self) -> list[dict[str, str]]:
        """Return the suggestions the last apply actually matched.

        :meth:`apply_semantic_names` skips anything naming a class, property or
        relationship the ontology does not contain, so the requested list
        overstates what happened. Callers that propagate names elsewhere must
        use this instead, or they will act on names the ontology rejected.

        Returns:
            One dict per applied suggestion with ``original_name``,
            ``suggested_name``, ``description``, and the raw SQL identity taken
            from the ontology's own annotations: ``table_name``,
            ``column_name`` and ``related_table`` (the far side of a
            relationship). Each is "" where it does not apply or the ontology
            carries no annotation. Empty list if apply has not run.

            The raw names matter because URIs are built with
            :meth:`_clean_name`, so a suggestion's ``original_name`` is the
            cleaned form and does not identify the table in the database.
        """
        return list(self._applied_semantic_names)

# OBA-Core 0.1

Status: Draft

## 1. Abstract
OBA, the OrionBelt Analytics vocabulary, is an RDF-based annotation vocabulary for physical database schema metadata. `OBA-Core 0.1` defines a minimal set of OWL annotation properties that describe tables, columns, foreign-key relationships, inferred relationships, and LLM-generated semantic enrichments on OWL ontologies generated from database introspection.

OBA uses:
- OWL annotation properties for schema metadata
- RDFS for labels and descriptions
- XSD for typed literals
- SHACL for optional validation
- a custom `oba:` namespace for physical-schema annotations

OBA complements OBSL (OrionBelt Semantic Layer), which operates at the business/logical level. Together they cover the full stack from physical SQL to business semantics.

## 2. Goals
OBA-Core is designed to:
- formally define the annotation vocabulary used by OrionBelt Analytics
- enable SHACL-based validation of generated ontologies
- provide a stable, dereferenceable namespace for physical-schema metadata
- complement OBSL with physical-layer annotations for cross-graph federation
- support Text-to-SQL generation by preserving SQL-ready metadata (join conditions, data types, column references)

OBA-Core is not intended to represent:
- business-level concepts (dimensions, measures, metrics --- that is OBSL's domain)
- R2RML or other W3C mapping vocabularies
- query execution plans or optimizer hints
- vector embeddings or GraphRAG structures

## 3. Profiles

### 3.1 OBA-Core 0.1
`OBA-Core 0.1` includes:
- table annotations (name, schema, database, primary keys, row count)
- column annotations (name, data type, nullability, key flags, SQL reference)
- relationship annotations (FK column, target table/column, join condition, cardinality)
- inference metadata (confidence, pattern)
- denormalization flags
- LLM-applied semantic enrichments (semantic names, table types, business rules)

### 3.2 Future extensions
Possible future additions:
- index metadata (index type, columns, uniqueness)
- constraint details (CHECK, UNIQUE, EXCLUDE)
- partitioning information
- view definitions and materialization status
- cross-database lineage links

Only `OBA-Core 0.1` is defined by this document.

## 4. Namespaces
Required prefixes:

```ttl
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix oba:  <https://ralforion.com/ns/oba#> .
```

The `oba:` namespace is:
- `https://ralforion.com/ns/oba#`
- Hosted at: [https://ralforion.com/ns/oba/](https://ralforion.com/ns/oba/)

## 5. Design Principles

### 5.1 Annotation-only vocabulary
OBA defines only `owl:AnnotationProperty` instances. It does not introduce new OWL classes. The structural pattern of the generated ontology is:
- `owl:Class` = database table
- `owl:DatatypeProperty` = database column
- `owl:ObjectProperty` = foreign-key relationship

OBA properties annotate these standard OWL entities with physical-schema metadata.

### 5.2 Self-contained columns
Each column property carries `oba:tableName` and `oba:sqlReference` so that individual triples are self-contained for SQL generation. This intentional redundancy means a SPARQL query can generate a `SELECT` clause from column triples alone, without traversing `rdfs:domain` links.

### 5.3 Dual primary-key representation
Primary keys are represented in two complementary ways:
- `oba:primaryKey` on `owl:Class` --- one triple per PK column (for table-level PK enumeration)
- `oba:isPrimaryKey` boolean on `owl:DatatypeProperty` --- per-column flag (for column-level filtering)

### 5.4 Validation via SHACL
Domain/range constraints are expressed in SHACL shapes, not OWL restrictions. This keeps the vocabulary definition clean and separates structural assertions from validation rules.

## 6. Core Properties

### 6.1 Table Properties (on `owl:Class`)

Required:
- `rdfs:label` --- display name
- `oba:tableName` --- `xsd:string`, exact SQL table name
- `oba:schemaName` --- `xsd:string`, database schema

Optional:
- `oba:database` --- `xsd:string`, database or catalog name
- `oba:primaryKey` --- `xsd:string`, PK column name (repeating, one triple per column)
- `oba:rowCount` --- `xsd:integer`, approximate row count

Cardinality:
- at least one `rdfs:label`
- exactly one `oba:tableName`
- exactly one `oba:schemaName`
- at most one `oba:database`
- zero or more `oba:primaryKey`
- at most one `oba:rowCount`

### 6.2 Column Properties (on `owl:DatatypeProperty`)

Required:
- `rdfs:label` --- display name
- `oba:columnName` --- `xsd:string`, exact SQL column name
- `oba:tableName` --- `xsd:string`, owning table name
- `oba:sqlDataType` --- `xsd:string`, raw SQL type (e.g., `VARCHAR(255)`)
- `oba:sqlReference` --- `xsd:string`, fully-qualified `table.column`
- `oba:isPrimaryKey` --- `xsd:boolean`
- `oba:isForeignKey` --- `xsd:boolean`
- `oba:isNullable` --- `xsd:boolean`

Optional:
- `oba:typeOverrideReason` --- `xsd:string`, reason for XSD type override

Cardinality:
- at least one `rdfs:label`
- exactly one of each other required property
- at most one `oba:typeOverrideReason`

### 6.3 Relationship Properties (on `owl:ObjectProperty`)

OBA generates two object properties per foreign key: a **forward** property (FK-holding side, `many_to_one`) and an **inverse** property (referenced side, `one_to_many`), linked via `owl:inverseOf`.

FK-specific annotations (`foreignKeyColumn`, `referencedTable`, `referencedColumn`, `sqlJoinCondition`) appear **only on the forward property**. Inverse properties derive FK metadata by following `owl:inverseOf`. This avoids the inconsistency of FK annotations contradicting the inverse property's domain/range direction.

**All relationships (forward and inverse):**

Required:
- `rdfs:label` --- display name
- `oba:relationshipType` --- `xsd:string`, `"many_to_one"` or `"one_to_many"`
- `rdfs:domain` --- source class
- `rdfs:range` --- target class

**Forward relationships only (FK-holding side):**

Required:
- `oba:foreignKeyColumn` --- `xsd:string`, FK column in source table
- `oba:referencedTable` --- `xsd:string`, target table name
- `oba:referencedColumn` --- `xsd:string`, target column
- `oba:sqlJoinCondition` --- `xsd:string`, SQL `JOIN ON` clause

Optional:
- `oba:referencedSchema` --- `xsd:string`, target schema (cross-schema FKs only)

Cardinality:
- exactly one of each required property
- at most one `oba:referencedSchema`

### 6.4 Inference Properties (on `owl:ObjectProperty`, inferred relationships only)

Optional (present only when relationship is inferred, not declared):
- `oba:isInferredRelationship` --- `xsd:boolean`, always `true` when present
- `oba:inferenceConfidence` --- `xsd:string`, `"high"`, `"medium"`, or `"low"`
- `oba:inferencePattern` --- `xsd:string`, pattern name

Cardinality:
- at most one of each
- if `oba:isInferredRelationship` is `true`, `oba:inferenceConfidence` and `oba:inferencePattern` SHOULD be present

### 6.5 Denormalization Properties (on `owl:DatatypeProperty`, flagged columns only)

Optional (present only when denormalization is detected):
- `oba:isDenormalized` --- `xsd:boolean`, always `true` when present
- `oba:likelySourceTable` --- `xsd:string`, probable source table
- `oba:denormalizationWarning` --- `xsd:string`, human-readable warning

Cardinality:
- at most one of each
- if `oba:isDenormalized` is `true`, `oba:likelySourceTable` SHOULD be present

### 6.7 Measure Semantics (on `owl:DatatypeProperty`)

How a column behaves under aggregation. Placed on columns, not tables,
because additivity is a property of the column: a denormalized table holds
additive measures beside attributes, and no table-level role can say what
`SUM()` may touch.

Optional (present only when it could be determined without guessing):
- `oba:measureType` --- `xsd:string`, one of `additive`, `semi_additive`,
  `non_additive`, `attribute` (7.5)
- `oba:measureBasis` --- `xsd:string`, one of `structural`, `name_pattern`,
  `declared` (7.6)
- `oba:measureReason` --- `xsd:string`, justification, so a heuristic
  classification can be audited and corrected

Cardinality:
- at most one of each
- if `oba:measureType` is present, `oba:measureBasis` SHOULD be present

Consumers MUST weigh a finding by its basis: a `structural` classification
is certain and may justify rejecting a query, whereas a `name_pattern`
classification is a heuristic and SHOULD only warn.

A column carrying no `oba:measureType` has not been classified. Consumers
MUST NOT read that as `additive`.

### 6.6 Semantic Enrichment Properties (LLM-applied)

These properties are added during LLM-driven semantic enrichment workflows. They are always optional.

On `owl:Class` (tables):
- `oba:semanticName` --- `xsd:string`, business-friendly name
- `oba:tableType` --- `xsd:string`, `"fact"`, `"dimension"`, `"lookup"`, or
  `"mixed"`. Advisory only. A denormalized table holds additive measures
  beside dimension attributes, which is what `"mixed"` records; what may
  actually be aggregated is decided per column by `oba:measureType` (6.7).
- `oba:usageNotes` --- `xsd:string`, usage guidance

On `owl:DatatypeProperty` (columns):
- `oba:semanticName` --- `xsd:string`, business-friendly name
- `oba:dataCharacteristics` --- `xsd:string`, value patterns
- `oba:businessRules` --- `xsd:string`, business rules

On `owl:ObjectProperty` (relationships):
- `oba:semanticName` --- `xsd:string`, business-friendly name
- `oba:relationshipDescription` --- `xsd:string`, business description
- `oba:cardinality` --- `xsd:string`, LLM-assigned cardinality
- `oba:businessRule` --- `xsd:string`, governing business rule

Cardinality:
- at most one of each enrichment property per entity

## 7. Controlled Value Sets

### 7.1 Relationship Types
- `many_to_one`
- `one_to_many`

### 7.2 Inference Confidence Levels
- `high`
- `medium`
- `low`

### 7.3 Inference Patterns
- `embedded_table_name`
- `suffix_id`
- `prefix_id`
- `prefix_fk`
- `suffix_fk`
- `suffix_sk`
- `tpcds_sk`

### 7.4 Table Types (LLM-assigned)
- `fact`
- `dimension`
- `lookup`
- `mixed` --- both, as a denormalized table usually is

Advisory. Aggregation is governed per column by `oba:measureType`, not by
this value.

### 7.5 Measure Types
- `additive` --- meaningful to SUM across every dimension
- `semi_additive` --- summable across entities but not across time
- `non_additive` --- never meaningful to SUM (AVG and MIN/MAX still are)
- `attribute` --- not a measure at all

### 7.6 Measure Bases
- `structural` --- follows from the SQL type or key membership; certain
- `name_pattern` --- inferred from the column name; a heuristic
- `declared` --- supplied by a client or semantic enrichment

## 8. Relationship to OBSL

OBA and OBSL are independent vocabularies at different abstraction levels:

| | OBA | OBSL |
|---|---|---|
| **Level** | Physical (SQL schema) | Logical (business model) |
| **Annotates** | `owl:Class`, `owl:DatatypeProperty`, `owl:ObjectProperty` | `obsl:DataObject`, `obsl:Column`, `obsl:Dimension`, etc. |
| **Generated by** | Database introspection | OBML YAML authoring |
| **Key use case** | Text-to-SQL, schema quality validation | BI query generation, governance |

Cross-graph linking is possible via shared identifiers:
- An OBSL `DataObject` references a table by `obsl:code` + `obsl:database` + `obsl:schema`
- An OBA `owl:Class` describes the same table via `oba:tableName` + `oba:database` + `oba:schemaName`
- SPARQL federation can match on these values to bridge the physical and logical layers

No `owl:imports` exists between OBA and OBSL --- they are designed to be used together but do not depend on each other.

## 9. URI Strategy

Instance URIs use the `base_uri` (configurable, default `http://example.com/ontology/`):

- `{base_uri}` --- ontology header
- `{base_uri}{TableName}` --- table class (e.g., `ns:orders`)
- `{base_uri}{table}_{column}` --- column property (e.g., `ns:orders_customer_id`)
- `{base_uri}{table}_has_{target}` --- FK relationship (e.g., `ns:orders_has_customers`)
- `{base_uri}{target}_referenced_by_{table}` --- inverse relationship

The `oba:` vocabulary namespace is fixed and independent of instance URIs.

## 10. Validation

SHACL shapes are provided in `oba-shacl.ttl` to enforce:
- required properties on table classes, column properties, and relationship properties
- datatype constraints (`xsd:string`, `xsd:boolean`, `xsd:integer`)
- controlled value sets for `oba:relationshipType`, `oba:inferenceConfidence`, `oba:tableType`
- cardinality constraints (minCount, maxCount)
- structural requirements (`rdfs:domain`, `rdfs:range` presence on columns and relationships)
- cross-validation of FK annotations against graph structure via SPARQL-based constraints:
  - `oba:referencedTable` must match `oba:tableName` of the `rdfs:range` class
  - `oba:foreignKeyColumn` must be a known column of the `rdfs:domain` class

## 11. Versioning

OBA-Core 0.1 is the initial release. Non-breaking additions (new optional properties) do not bump the version. Breaking changes (property removal, renaming, range changes) bump the minor version.

The ontology uses `owl:versionIRI` and `owl:versionInfo` in its header.

## 12. Finalized Core Surface

Annotation properties --- Table (on `owl:Class`):
- `oba:tableName`
- `oba:schemaName`
- `oba:database`
- `oba:primaryKey`
- `oba:rowCount`

Annotation properties --- Column (on `owl:DatatypeProperty`):
- `oba:columnName`
- `oba:tableName`
- `oba:sqlDataType`
- `oba:sqlReference`
- `oba:isPrimaryKey`
- `oba:isForeignKey`
- `oba:isNullable`
- `oba:typeOverrideReason`

Annotation properties --- Relationship (on `owl:ObjectProperty`):
- `oba:foreignKeyColumn`
- `oba:referencedTable`
- `oba:referencedColumn`
- `oba:referencedSchema`
- `oba:sqlJoinCondition`
- `oba:relationshipType`

Annotation properties --- Inference (on `owl:ObjectProperty`):
- `oba:isInferredRelationship`
- `oba:inferenceConfidence`
- `oba:inferencePattern`

Annotation properties --- Denormalization (on `owl:DatatypeProperty`):
- `oba:isDenormalized`
- `oba:likelySourceTable`
- `oba:denormalizationWarning`

Annotation properties --- Semantic Enrichment (on any OWL entity):
- `oba:semanticName`
- `oba:tableType`
- `oba:usageNotes`
- `oba:dataCharacteristics`
- `oba:businessRules`
- `oba:relationshipDescription`
- `oba:cardinality`
- `oba:businessRule`

# OBQC in OrionBelt Analytics

OBQC means **Ontology-Based Query Check**. In OrionBelt Analytics (OBA), it is
the deterministic safety layer that checks SQL against the loaded RDF/OWL
ontology before a query is allowed to reach the database.

OBQC does not call an LLM. It parses SQL, compares the query structure with the
ontology, and returns structured errors or warnings that the calling assistant
can use to correct the SQL.

## Why OBA Uses OBQC

LLMs can generate plausible SQL that is still structurally wrong. Typical
failures include misspelled columns, joins that do not follow foreign keys,
missing `GROUP BY` columns, and analytical fan-traps that silently multiply
totals.

OBQC gives OBA a deterministic check after SQL generation and before execution:

- **Errors block execution.** The query is not sent to the database.
- **Warnings allow execution.** The warning is returned with the result so the
  assistant can explain or revise the query.
- **No ontology means limited validation.** OBA can still do syntax and security
  checks, but semantic validation needs an OrionBelt ontology.

## How It Works

1. OBA connects to a database and discovers the schema.
2. `generate_ontology()` or `load_my_ontology()` creates or loads an ontology
   with `oba:` annotations for tables, columns, SQL types, primary keys, foreign
   keys, join conditions, and relationship direction.
3. OBA creates a session-local `OBQCValidator` from that ontology.
4. When `execute_sql_query()` receives SQL, OBQC parses it with `sqlglot`.
5. OBQC extracts the referenced tables, columns, joins, aggregations, aliases,
   and join anchors.
6. OBQC validates those parts against the ontology cache.
7. If the query has no blocking errors, OBA executes it and attaches any OBQC
   warnings to the response.

`execute_sql_query()` is the only entry point that runs OBQC. The separate
security and dialect-syntax checks in `DatabaseManager.validate_sql_syntax()`
are unrelated to it and carry no ontology awareness.

In the main execution path, `execute_sql_query()` runs OBQC before database
execution. If OBQC returns an error, OBA returns an `obqc_error` response instead
of executing the SQL.

## What OBQC Checks

| Check | Purpose |
| --- | --- |
| Table existence | Every referenced table must exist in the ontology. |
| Column existence | Qualified and unqualified columns must resolve to real columns. |
| Join validity | Joins should match declared ontology relationships. |
| Type compatibility | `WHERE` and `ON` comparisons should use compatible types. |
| Aggregation correctness | Non-aggregated selected columns must be in `GROUP BY`. |
| Fan-trap risk | Aggregations across multiple fan-out joins are flagged. |

Three things sit outside those rules, because treating them as violations
blocked correct SQL:

- **Database catalog schemas are exempt from the table rule.** `information_schema`,
  `pg_catalog`, `system` and friends describe the database itself, so they are
  never in an ontology of user data. MySQL's `mysql` schema is deliberately not
  exempt -- it holds accounts and grants rather than metadata, and
  `src/security.py` blocks those tables outright.
- **`SELECT` aliases are not columns.** `ORDER BY revenue` over
  `SUM(total) AS revenue` resolves to the select list. Where an alias may be
  referenced varies by database, so OBQC follows each dialect: PostgreSQL
  accepts one in `GROUP BY` and `ORDER BY` but not `HAVING`, while DuckDB
  resolves aliases in every clause including `WHERE`.
- **Rules apply per `SELECT`.** Tables, columns and aggregation in a subquery
  belong to that subquery. A name in one scope is not resolved against another,
  except that a correlated subquery may still see its enclosing query's tables.

## Fan-Trap Protection

A fan-trap happens when a query aggregates after joining across multiple
one-to-many paths. The SQL may run successfully, but totals can be inflated
because rows are multiplied before aggregation.

OBQC checks fan-traps in two ways:

- It first uses ontology axioms such as `owl:disjointWith` for sibling fact
  tables that share a dimension.
- If those axioms are not present, it uses relationship metadata and join
  direction to count fan-out joins in the actual query.

The direction matters: joining from a fact table to a dimension is usually a
lookup, while joining from a dimension to multiple fact tables can multiply
rows. OBQC judges fan-out per join, not just by asking whether a table is on the
many side somewhere in the schema.

## Example Flow

```text
connect_database()
  -> discover_schema()
  -> generate_ontology()
  -> execute_sql_query()
       -> OBQC parses SQL
       -> OBQC checks ontology rules
       -> errors block, warnings attach
       -> database execution only if valid
```

For detailed rule behavior, severity handling, and required ontology
annotations, see [OBQC -- Ontology-Based Query Check](obqc.md). For analytical
data multiplication examples, see [Fan-Trap Prevention](fan-trap-prevention.md).

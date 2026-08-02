[<- Back to README](../README.md)

# Fan-Trap Prevention

Fan-traps are one of the most common causes of silently incorrect SQL results. They inflate aggregated values without raising errors, making them particularly dangerous in analytical queries. OrionBelt® Analytics detects fan-trap risks automatically at multiple stages -- during schema analysis, query validation, and GraphRAG context retrieval.

## What Is a Fan-Trap?

A fan-trap occurs when a query joins a parent table to two or more child tables through one-to-many relationships and then applies aggregation functions (SUM, COUNT, AVG). The multiple one-to-many joins create a Cartesian product between the child tables, multiplying rows before the aggregation runs.

Consider an `orders` table with two child tables:

```
orders (1) ---> order_items (many)
orders (1) ---> shipments   (many)
```

If order #100 has 3 items and 2 shipments, joining all three tables produces 6 rows for that order (3 x 2). Any SUM on `order_items.amount` is now tripled by the shipment rows, and any COUNT on shipments is tripled by the item rows.

### A Concrete Example

Suppose the data looks like this:

| order_id | item_amount |
|----------|-------------|
| 100      | 50          |
| 100      | 30          |
| 100      | 20          |

| order_id | shipment_id |
|----------|-------------|
| 100      | S1          |
| 100      | S2          |

The correct total for order #100 is 100 (50 + 30 + 20). But a naive join produces:

```sql
SELECT o.id, SUM(oi.item_amount) AS total
FROM orders o
JOIN order_items oi ON o.id = oi.order_id
JOIN shipments s ON o.id = s.order_id
GROUP BY o.id;
```

Result: **200** (each item row is duplicated per shipment). The query executes without errors, returns a plausible-looking number, and silently corrupts every downstream report that uses it.

---

## How OrionBelt Detects Fan-Traps

OrionBelt Analytics applies fan-trap detection at three layers, each catching different scenarios.

### Layer 1: Schema Analysis (`discover_schema`)

When `discover_schema()` inspects foreign keys, it flags any table that references multiple other tables as a potential fan-trap bridge:

```python
# From src/handlers/schema.py
if len(table_info.foreign_keys) > 1:
    fan_trap_warnings.append({
        "table": table_name,
        "warning": f"Table {table_name} connects to multiple tables "
                   "- potential fan-trap risk",
        "referenced_tables": referenced_tables,
        "recommendation": "Use separate CTEs or UNION approach "
                          "for multi-fact aggregations",
    })
```

These warnings appear in the `discover_schema()` output so the LLM (or user) is aware of risky join paths before writing any SQL.

### Layer 2: OBQC Query Validation (built into `execute_sql_query`)

The Ontology Basic Quality Criteria (OBQC) validator runs automatically inside `execute_sql_query` before the query is run. It parses SQL queries using sqlglot and checks them against the ontology's relationship metadata. **A detected fan-trap is a blocking error**: the query is refused rather than answered with an inflated number.

The strongest finding needs only a *single* one-to-many join. If an aggregate reads a measure from the "one" side of a join, every one of its rows is repeated:

```python
# From src/obqc_validator.py -- _inflated_measures()
for measure_table, other in ((anchor, join_table), (join_table, anchor)):
    if measure_table.lower() not in measures:
        continue
    if not self._join_fans_out(other, measure_table):
        continue
    findings.append({
        "kind": "measure_across_fan_out",
        "measure_table": measure_table,
        "fan_out_table": other,
        ...
    })
```

Both ends of every join are checked, so `FROM order_items JOIN orders` summing `orders.total` is caught exactly like `FROM orders JOIN order_items`. Aggregates that duplication cannot change (`MIN`, `MAX`, `COUNT(DISTINCT ...)`, `COUNT(*)`) are left alone, as is a measure taken from the many side.

Two weaker findings follow: sibling facts the ontology declares `owl:disjointWith`, and the older heuristic of two or more one-to-many joins in one aggregating `SELECT`.

The validator knows which relationships are one-to-many because the ontology stores `oba:relationshipType` annotations on every OWL ObjectProperty.

Every response carries an `obqc_fan_trap` object (`{detected, blocking, findings}`) so a caller can read the verdict as data instead of parsing `warnings`. To run a flagged query anyway, pass `allow_fan_out=True` -- the finding is still reported, as a warning.

### Layer 3: GraphRAG Context Retrieval

When GraphRAG retrieves context for a natural-language question, it examines the graph structure to detect tables with multiple outgoing foreign keys:

```python
# From src/graphrag/retriever.py
outgoing_fks = list(self.graph.successors(table))
if len(outgoing_fks) > 1:
    warnings.append({
        "bridge_table": table,
        "referenced_tables": outgoing_fks,
        "warning": f"Table '{table}' connects to multiple tables "
                   "- potential fan-trap",
        "recommendation": "Use separate CTEs or UNION approach "
                          "if aggregating across these relationships"
    })
```

These warnings are included in the GraphRAG context returned to the LLM, guiding it to generate safe SQL patterns.

---

## Dangerous SQL Patterns

The following patterns are common sources of fan-trap errors. All of them join a parent table to multiple child tables and then aggregate.

### Direct Multi-Child Join with SUM

```sql
-- DANGEROUS: inflates both SUM and COUNT
SELECT
    c.customer_name,
    SUM(s.amount) AS total_sales,
    COUNT(sh.shipment_id) AS shipment_count
FROM customers c
LEFT JOIN sales s ON c.id = s.customer_id
LEFT JOIN shipments sh ON c.id = sh.customer_id
GROUP BY c.customer_name;
```

If a customer has 5 sales and 3 shipments, each sale row appears 3 times (once per shipment). `total_sales` is tripled; `shipment_count` is quintupled.

### Chain Join Through a Bridge Table

```sql
-- DANGEROUS: orders is the bridge between order_items and shipments
SELECT
    o.id AS order_id,
    SUM(oi.amount) AS item_total,
    SUM(sh.shipping_cost) AS shipping_total
FROM orders o
JOIN order_items oi ON o.id = oi.order_id
JOIN shipments sh ON o.id = sh.order_id
GROUP BY o.id;
```

### Aggregation After Multiple LEFT JOINs

```sql
-- DANGEROUS: two 1:many LEFT JOINs from the same parent
SELECT
    p.product_name,
    AVG(r.rating) AS avg_rating,
    SUM(s.quantity) AS total_sold
FROM products p
LEFT JOIN reviews r ON p.id = r.product_id
LEFT JOIN sale_items s ON p.id = s.product_id
GROUP BY p.product_name;
```

Products with many reviews and many sales will have heavily inflated results for both measures.

---

## Safe SQL Patterns

### Pattern 1: UNION ALL (Recommended)

Combine fact tables vertically before aggregating. This eliminates the Cartesian product entirely because each fact type contributes its own rows independently.

```sql
WITH unified_facts AS (
    SELECT
        customer_id,
        amount AS measure_value,
        'sale' AS fact_type
    FROM sales

    UNION ALL

    SELECT
        customer_id,
        return_amount AS measure_value,
        'return' AS fact_type
    FROM returns
)
SELECT
    customer_id,
    SUM(CASE WHEN fact_type = 'sale' THEN measure_value ELSE 0 END) AS total_sales,
    SUM(CASE WHEN fact_type = 'return' THEN measure_value ELSE 0 END) AS total_returns
FROM unified_facts
GROUP BY customer_id;
```

When the measures share the same unit (e.g., monetary amounts), a simpler variant works:

```sql
WITH unified_facts AS (
    SELECT customer_id, sales_amount, 0 AS returns
    FROM sales
    UNION ALL
    SELECT customer_id, 0, return_amount
    FROM returns
)
SELECT
    customer_id,
    SUM(sales_amount) AS total_sales,
    SUM(returns) AS total_returns
FROM unified_facts
GROUP BY customer_id;
```

### Pattern 2: Separate Aggregation with CTEs

Pre-aggregate each fact table independently, then join the results. Since each CTE produces one row per key, the final join is one-to-one and safe.

```sql
WITH sales_totals AS (
    SELECT
        customer_id,
        SUM(amount) AS total_sales,
        COUNT(*) AS sale_count
    FROM sales
    GROUP BY customer_id
),
shipment_totals AS (
    SELECT
        customer_id,
        COUNT(*) AS shipment_count,
        SUM(shipping_cost) AS total_shipping
    FROM shipments
    GROUP BY customer_id
)
SELECT
    s.customer_id,
    s.total_sales,
    s.sale_count,
    COALESCE(sh.shipment_count, 0) AS shipment_count,
    COALESCE(sh.total_shipping, 0) AS total_shipping
FROM sales_totals s
LEFT JOIN shipment_totals sh ON s.customer_id = sh.customer_id;
```

### Pattern 3: Correlated Subqueries

Use scalar subqueries to fetch aggregated values from each child table without joining them together.

```sql
SELECT
    o.id AS order_id,
    o.order_date,
    (SELECT SUM(oi.amount) FROM order_items oi WHERE oi.order_id = o.id) AS item_total,
    (SELECT COUNT(*) FROM shipments sh WHERE sh.order_id = o.id) AS shipment_count
FROM orders o;
```

This approach is clear and correct, though it may be less efficient than CTEs on large datasets.

### Pattern 4: Window Functions with Pre-Aggregation

When you need to preserve row-level granularity alongside aggregated values, pre-aggregate the "many" side in a subquery:

```sql
SELECT DISTINCT
    o.id AS order_id,
    SUM(oi.amount) OVER (PARTITION BY o.id) AS item_total,
    sh_agg.shipment_count
FROM orders o
JOIN order_items oi ON o.id = oi.order_id
LEFT JOIN (
    SELECT order_id, COUNT(*) AS shipment_count
    FROM shipments
    GROUP BY order_id
) sh_agg ON o.id = sh_agg.order_id;
```

The shipments are pre-aggregated in the subquery, so joining them back to orders does not multiply the order_items rows.

---

## How Ontology Annotations Help

OrionBelt generates an OWL ontology where each foreign key relationship is stored as an `owl:ObjectProperty` with custom `oba:` (OrionBelt Analytics) namespace annotations:

- **`oba:relationshipType`** -- Records whether the relationship is `"one_to_many"` or `"many_to_one"`. This is the primary signal used by the OBQC validator to count how many one-to-many joins a query traverses.

- **`oba:sqlJoinCondition`** -- Stores the exact SQL join condition (e.g., `orders.id = order_items.order_id`). This ensures the LLM generates correct join predicates and does not invent invalid join paths.

- **Inverse relationships** -- Every foreign key generates both the forward (`many_to_one`) and inverse (`one_to_many`) OWL properties, linked via `owl:inverseOf`. This gives the validator and the LLM a complete picture of directionality.

Together, these annotations let the OBQC validator make a deterministic decision: if a query joins two or more tables that sit on the "many" side of one-to-many relationships and applies aggregation, it is a fan-trap candidate.

### Example Ontology Fragment (Turtle)

```turtle
:orders_has_order_items a owl:ObjectProperty ;
    rdfs:domain :Orders ;
    rdfs:range :OrderItems ;
    oba:relationshipType "one_to_many" ;
    oba:sqlJoinCondition "orders.id = order_items.order_id" ;
    owl:inverseOf :order_items_belongs_to_orders .

:orders_has_shipments a owl:ObjectProperty ;
    rdfs:domain :Orders ;
    rdfs:range :Shipments ;
    oba:relationshipType "one_to_many" ;
    oba:sqlJoinCondition "orders.id = shipments.order_id" ;
    owl:inverseOf :shipments_belongs_to_orders .
```

When the OBQC validator sees a query joining `orders`, `order_items`, and `shipments` with `SUM()`, it looks up both ObjectProperties, finds two `one_to_many` relationships, and raises the `FAN_TRAP_DETECTED` warning.

---

## Result Validation Checklist

Even with automatic detection, verify your results:

1. **Compare against source tables.** Run `SELECT SUM(amount) FROM order_items` independently and compare to your joined query.

2. **Check row counts.** If `orders` has 1,000 rows but your join produces 50,000, something is being multiplied.

3. **Rely on `execute_sql_query`'s built-in checks.** It runs OBQC validation before executing and will flag fan-trap risks (or reject the query) in the response.

4. **Review the `fan_trap_warnings` from `discover_schema()`.** They list which tables have multiple foreign keys and are therefore fan-trap candidates.

5. **Look at business expectations.** If a customer's total suddenly jumps by an order of magnitude after adding a second join, investigate.

---

## Quick Reference

| Scenario | Risk Level | Recommended Pattern |
|---|---|---|
| Single fact table, one join | None | Direct aggregation |
| Two tables, 1:1 relationship | None | Direct join |
| Parent + one child (1:many) | Low | Direct join (single child is safe) |
| Parent + two children (1:many each) | **High** | UNION ALL or separate CTEs |
| Bridge table connecting multiple tables | **High** | Pre-aggregate before joining |
| Multiple fact tables, same measure type | **High** | UNION ALL |
| Multiple fact tables, different measures | **High** | Separate CTE aggregation |

---

## Workflow Summary

```
1. connect_database()
2. discover_schema()        --> fan_trap_warnings in output
3. generate_ontology()     --> oba:relationshipType annotations stored
4. execute_sql_query()     --> OBQC fan-trap checks run automatically, then query executes
```

When in doubt, use UNION ALL or separate aggregation CTEs. Fan-traps cause silent data corruption -- queries succeed but return wrong numbers.

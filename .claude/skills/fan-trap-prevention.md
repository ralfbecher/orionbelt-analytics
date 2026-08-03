# Fan-Trap Prevention Guide

**Skill for OrionBelt Analytics MCP Server**

## What is a Fan-Trap?

A fan-trap occurs when you aggregate a measure across a 1:many join, so each measure row is repeated once per matching child row and the total comes back inflated.

**A single 1:many join is enough.** The classic multi-fact shape is the worst case, not the threshold.

### Example — one join is already wrong:
```
sales (1) → shipments (many)
```
```sql
-- ❌ each sale's amount is added once per shipment
SELECT SUM(public.sales.amount)
FROM public.sales
JOIN public.shipments ON public.shipments.sale_id = public.sales.id;
```
The inner join also drops sales that never shipped, so the number is wrong in both directions.

Which table you put in `FROM` makes no difference: `FROM shipments JOIN sales` produces the same one-row-per-shipment result.

### Example — the multi-fact shape:
```
orders (1) → order_items (many)
orders (1) → shipments (many)
```

**❌ WRONG APPROACH:**
```sql
SELECT SUM(public.order_items.amount)
FROM public.orders
JOIN public.order_items ON public.orders.id = public.order_items.order_id
JOIN public.shipments ON public.orders.id = public.shipments.order_id;
```

**Result:** Inflated totals due to Cartesian product multiplication

**✅ CORRECT APPROACH:** Use UNION ALL to combine facts, then aggregate

---

## Detection Checklist

Before writing multi-table queries with aggregation:

1. **Review foreign_keys** from `discover_schema()` FIRST
2. **Identify relationship patterns:**
   - Safe: 1:1 relationships (customers → customer_profiles)
   - Safe: measure taken from the **many** side (`SUM(order_items.qty)` across orders → order_items)
   - Fan-trap: measure taken from the **one** side across a 1:many join (`SUM(orders.total)` with order_items joined)
   - Fan-trap: multiple 1:many from the same parent
3. **Conditional aggregates measure the branch, not the test.** `SUM(CASE WHEN orders.total > 100 THEN order_items.quantity ELSE 0 END)` measures `order_items`; the `orders` column only filters. Putting the parent's measure in the branch does still inflate.
4. **Let `execute_sql_query()` validate** — OBQC runs before execution and a fan-trap is a **blocking error**, not a warning. Read the `obqc_fan_trap` field on the response: `{detected, blocking, findings}`, where each finding names the `measure_table` being inflated and the `fan_out_table` doing it. `evaluated: false` means OBQC never ran (no ontology loaded) — treat that as unknown, not as safe.
5. **Fix the query, don't force it.** Pre-aggregate the fanning table in a CTE, or use UNION ALL. `allow_fan_out=True` exists for the rare case where the multiplied rows are genuinely wanted — it does not make the numbers right.
6. **Validate results** against source tables

### Conditional row counts

`SUM(CASE WHEN <table> … THEN 1 ELSE 0 END)` and `COUNT(*) FILTER (WHERE <table> …)` count *rows*, and a join repeats the rows of whatever the condition names. OBQC **warns without blocking** here, because the SQL cannot say which count you meant — over `orders JOIN order_items` a condition on `orders` counts items, not orders; over `orders JOIN users` a condition on `users` counts orders, which is usually exactly right.

If you meant the coarser count, use `COUNT(DISTINCT <table>.<key>)` or filter with `EXISTS` instead of joining.

### Aggregates that survive a fan-out

`MIN`, `MAX` and `COUNT(DISTINCT ...)` read the same answer off repeated rows, so OBQC never blocks a query that aggregates only with those — no join shape makes them wrong.

`SUM`, `AVG` and `COUNT(col)` are corrupted by duplication and are what the checks look for.

`COUNT(*)` sits between the two: counting the joined rows is usually what you meant across a single 1:many join, so it is not blocked there — but across **two** fan-out joins it returns the product of the two children, which is meaningless, and is blocked.

---

## Safe Query Patterns

### PATTERN 1 - UNION ALL (RECOMMENDED)

**Best for:** Combining multiple fact tables with consistent measures

```sql
WITH unified_facts AS (
    SELECT
        public.fact1.key,
        public.fact1.category,
        public.fact1.amount as measure_value,
        'type1' as fact_type
    FROM public.fact1

    UNION ALL

    SELECT
        public.fact2.key,
        public.fact2.category,
        public.fact2.quantity as measure_value,
        'type2' as fact_type
    FROM public.fact2
)
SELECT
    key,
    category,
    SUM(measure_value) as total_measure,
    COUNT(DISTINCT fact_type) as num_fact_types
FROM unified_facts
GROUP BY key, category;
```

**Benefits:**
- No data multiplication
- Unified data model for consistent aggregation
- Easy to extend with additional fact types
- Better performance with fewer table scans

### PATTERN 2 - SEPARATE AGGREGATION

**Use when:** UNION approach is not suitable (different measures)

```sql
WITH fact1_totals AS (
    SELECT
        public.fact1.key,
        SUM(public.fact1.amount) as total_amount
    FROM public.fact1
    GROUP BY public.fact1.key
),
fact2_totals AS (
    SELECT
        public.fact2.key,
        SUM(public.fact2.quantity) as total_quantity
    FROM public.fact2
    GROUP BY public.fact2.key
)
SELECT
    f1.key,
    f1.total_amount,
    COALESCE(f2.total_quantity, 0) as total_quantity
FROM fact1_totals f1
LEFT JOIN fact2_totals f2 ON f1.key = f2.key;
```

### PATTERN 3 - DISTINCT AGGREGATION (USE CAREFULLY)

**Warning:** Only use when you fully understand the data relationships

```sql
SELECT
    public.fact1.key,
    SUM(DISTINCT public.fact1.amount) as total_amount,
    SUM(public.fact2.quantity) as total_quantity
FROM public.fact1
LEFT JOIN public.fact2 ON public.fact1.id = public.fact2.fact1_id
GROUP BY public.fact1.key;
```

**Caution:** DISTINCT can mask issues and give false confidence

### PATTERN 4 - WINDOW FUNCTIONS

**For:** Complex analytical queries with preserved granularity

```sql
SELECT DISTINCT
    public.fact1.key,
    SUM(public.fact1.amount) OVER (PARTITION BY public.fact1.key) as total_amount,
    f2.pre_aggregated_quantity
FROM public.fact1
LEFT JOIN (
    SELECT
        public.fact2.key,
        SUM(public.fact2.qty) as pre_aggregated_quantity
    FROM public.fact2
    GROUP BY public.fact2.key
) f2 ON public.fact1.key = f2.key;
```

---

## Common Problematic Combinations

**Patterns requiring careful review:**

- `public.sales LEFT JOIN public.shipments + SUM(public.sales.amount)`
- `public.orders LEFT JOIN public.order_items LEFT JOIN public.products + SUM(public.orders.total)`
- `public.customers LEFT JOIN public.transactions LEFT JOIN public.transaction_items + aggregation`
- Queries joining parent→child1 + parent→child2 with SUM/COUNT

---

## Relationship Examples

### Safe (1:1 relationships):
```
customers → customer_profiles (1:1)
employees → employee_details (1:1)
```

### Requires care (1:many):
```
customers → orders (1:many)
products → inventory_records (1:many)
```

### High risk (fan-trap potential):
```
orders → order_items (1:many) + orders → shipments (1:many)
customers → orders (1:many) + customers → support_tickets (1:many)
```

**For high-risk patterns:** Always use UNION approach or separate aggregation CTEs

---

## Fan-Trap Solutions

If you suspect fan-trap in existing query:

1. **Split into UNION approach** (recommended)
2. **Use separate aggregations** with CTEs
3. **Add DISTINCT in SUM()** as temporary fix (not ideal)
4. **Validate results** against source tables
5. **Aggregate fact tables separately** before joining

**Critical:** Fan-traps cause silent data corruption - queries execute successfully but return inflated results!

---

## Result Validation

**Verify results make business sense:**

- Compare totals with business expectations
- Cross-check: `SELECT SUM(public.base_table.amount) FROM public.base_table` vs your query result
- Ensure row counts are reasonable
- High/unexpected results may indicate fan-trap multiplication

---

## Validation Checklist

For queries with 2+ tables and aggregation:

- [ ] Schema analyzed with `discover_schema()`
- [ ] Relationships reviewed (check foreign_keys)
- [ ] Fan-trap patterns identified
- [ ] Query run through `execute_sql_query()` (OBQC validation runs automatically)
- [ ] Safe aggregation pattern selected
- [ ] Results validated against business expectations

---

## Quick Reference

| Scenario | Solution |
|----------|----------|
| Multiple fact tables, same measures | UNION ALL (Pattern 1) |
| Multiple fact tables, different measures | Separate aggregation (Pattern 2) |
| Single fact table | Direct aggregation (no fan-trap risk) |
| 1:1 relationships only | Direct JOIN (safe) |
| Parent + multiple child tables | UNION or separate CTEs |

---

**Always remember:** When in doubt, use UNION ALL or separate aggregations!

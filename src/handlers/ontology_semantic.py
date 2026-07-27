"""Semantic-naming handlers: suggest and apply business-friendly names."""

import json
import logging
import re
from datetime import datetime
from typing import Any

from fastmcp import Context

from ..config import config_manager
from ..handler_context import HandlerContext
from ..lifecycle.metadata import update_workspace_section
from ..ontology_generator import OntologyGenerator
from ..oxigraph_store import OXIGRAPH_AVAILABLE, schema_graph_uri
from ..paths import OUTPUT_DIR, ensure_output_dir, get_connection_dir
from ..utils import is_client_disconnect, safe_ctx_info, write_text_file
from .ontology_generation import _build_minimal_graph_summary

logger = logging.getLogger(__name__)


async def _maybe_sample_rename_suggestions(
    ctx: Context,
    cryptic_classes: list,
    cryptic_props_by_table: dict[str, list],
    cryptic_relationships: list,
) -> dict[str, Any] | None:
    """Ask the host LLM (via MCP sampling) for ontology-shaped rename suggestions.

    Returns the structured payload that ``apply_semantic_names`` consumes
    natively::

        {
          "classes":       [{"original_name", "suggested_name", "description"}],
          "properties":    [{"original_name", "suggested_name", "description",
                             "table_name"}],
          "relationships": [{"original_name", "suggested_name", "description"}],
        }

    Returns ``None`` if sampling is disabled, the client doesn't support it,
    or the call fails — caller falls back to the legacy review-then-apply
    payload.
    """
    if not config_manager.get_server_config().enable_sampling:
        logger.info(
            "MCP sampling disabled (ENABLE_SAMPLING=false) — using legacy review path"
        )
        return None

    if not (cryptic_classes or cryptic_props_by_table or cryptic_relationships):
        return {"classes": [], "properties": [], "relationships": []}

    items: list[str] = [f"CLASS  {c}" for c in cryptic_classes]
    for table, cols in cryptic_props_by_table.items():
        items.extend(f"PROP   {table}.{col}" for col in cols)
    items.extend(f"REL    {r}" for r in cryptic_relationships)

    logger.info(
        "MCP sampling: requesting rename suggestions for %d items "
        "(%d classes, %d properties, %d relationships)",
        len(items),
        len(cryptic_classes),
        sum(len(v) for v in cryptic_props_by_table.values()),
        len(cryptic_relationships),
    )
    started = datetime.now()

    prompt = _build_rename_prompt(items)

    try:
        result = await ctx.sample(
            messages=prompt,
            system_prompt=(
                "You are an expert ontology and information-architecture designer. "
                "Produce concise, business-friendly OWL labels — not literal "
                "column names. Respond with one JSON object only."
            ),
            temperature=0.2,
            max_tokens=8000,
        )
    except Exception as e:
        logger.warning(
            "MCP sampling unavailable or failed after %.2fs (%s: %s) — "
            "falling back to manual review path",
            (datetime.now() - started).total_seconds(),
            type(e).__name__,
            str(e)[:200],
        )
        return None

    elapsed = (datetime.now() - started).total_seconds()
    raw_text = getattr(result, "text", None) or ""
    parsed = _parse_rename_json(raw_text)
    suggestions = _normalize_structured_suggestions(parsed)
    total = (
        len(suggestions.get("classes") or [])
        + len(suggestions.get("properties") or [])
        + len(suggestions.get("relationships") or [])
    )
    if total == 0:
        logger.info(
            "MCP sampling returned no usable suggestions (%.2fs, %d chars text)",
            elapsed,
            len(raw_text),
        )
        return None

    logger.info(
        "MCP sampling: received %d suggestions (%d classes, %d properties, "
        "%d relationships) in %.2fs (model=%s)",
        total,
        len(suggestions.get("classes") or []),
        len(suggestions.get("properties") or []),
        len(suggestions.get("relationships") or []),
        elapsed,
        getattr(result, "model", "unknown"),
    )
    return suggestions


def _build_rename_prompt(items: list[str]) -> str:
    """Compose the sampling prompt with concrete naming rules and a worked example."""
    return (
        "You are renaming cryptic identifiers in a SQL-derived OWL ontology so "
        "the resulting labels read like domain language, not table columns.\n\n"
        "Each PROP item is qualified as `table.column`. The table is the OWL "
        "class the property belongs to — its name is implicit context, so "
        "REMOVE redundant prefixes from the property name.\n\n"
        "Naming rules (apply strictly):\n"
        "1. CLASSES → singular PascalCase. Example: `clientcomplaints` → "
        "`ClientComplaint`.\n"
        "2. PROPERTIES → camelCase, no underscores, no table prefix. "
        "Examples: `purchases.purchaseamount` → `amount`; "
        "`sales.salesdate` → `date` (or `placedOn`); "
        "`clients.clientname` → `name`.\n"
        "3. FOREIGN-KEY columns become object-property names that READ LIKE THE "
        "RELATED ENTITY — drop the trailing `Id` and the source-table prefix. "
        "Examples: `sales.salesclient` → `client`; "
        "`purchases.purchaseproduct` → `product`; "
        "`purchases.purchasechanid` → `channel`.\n"
        "4. PRIMARY-KEY identifiers stay short: `clients.clientid` → `id`.\n"
        "5. Acronyms remain uppercase: `iban` → `IBAN`, `url` → `URL`, "
        "`vat` → `VAT`. Trailing identifier suffix is `Id` (camelCase), not `ID`.\n"
        "6. Date/time columns prefer verb-form participles when context suggests "
        "an event: `purchasedate` → `placedOn`; `returndate` → `returnedOn`; "
        "`shipmentdate` → `shippedOn`. Plain time fields stay as-is: "
        "`createdat` → `createdAt`.\n"
        "7. Add a one-sentence rdfs:comment-style `description` for every item — "
        "what the concept means in business terms.\n"
        "8. If you are not confident about an item, OMIT it (do not invent).\n\n"
        "Output format — a single JSON object, no prose, no code fences, no "
        "wrapping. Keys are exactly `classes`, `properties`, `relationships`.\n"
        "- `classes[i]`        : {original_name, suggested_name, description}\n"
        "- `properties[i]`     : {original_name, suggested_name, description, "
        "table_name}\n"
        "- `relationships[i]`  : {original_name, suggested_name, description}\n"
        "  `original_name` is the bare identifier (part after the table dot for "
        "PROP items).\n"
        "  `table_name` is the table for PROP items — REQUIRED to disambiguate "
        "columns that share a name across tables.\n\n"
        "Worked example. Input:\n"
        "  CLASS  clientcomplaints\n"
        "  PROP   purchases.purchaseamount\n"
        "  PROP   purchases.purchasechanid\n"
        "  PROP   sales.salesclient\n"
        "  PROP   acctbal.iban\n"
        "Output:\n"
        '{"classes":[{"original_name":"clientcomplaints",'
        '"suggested_name":"ClientComplaint",'
        '"description":"A complaint filed by a client."}],'
        '"properties":['
        '{"original_name":"purchaseamount","suggested_name":"amount",'
        '"description":"Total monetary amount of the purchase.",'
        '"table_name":"purchases"},'
        '{"original_name":"purchasechanid","suggested_name":"channel",'
        '"description":"Sales channel through which the purchase was placed.",'
        '"table_name":"purchases"},'
        '{"original_name":"salesclient","suggested_name":"client",'
        '"description":"The client who placed the sale.",'
        '"table_name":"sales"},'
        '{"original_name":"iban","suggested_name":"IBAN",'
        '"description":"International Bank Account Number for the account.",'
        '"table_name":"acctbal"}],'
        '"relationships":[]}\n\n'
        "Now produce suggestions for the items below. Items:\n" + "\n".join(items)
    )


def _normalize_structured_suggestions(parsed: dict[str, Any] | None) -> dict[str, list]:
    """Validate and clean a structured suggestions payload.

    Drops items missing required fields, strips suggestions that match the
    original verbatim, and guarantees the three top-level keys exist.
    """
    out: dict[str, list] = {"classes": [], "properties": [], "relationships": []}
    if not isinstance(parsed, dict):
        return out

    def _clean_item(item: Any, *, require_table: bool) -> dict[str, str] | None:
        if not isinstance(item, dict):
            return None
        original = str(item.get("original_name") or "").strip()
        suggested = str(item.get("suggested_name") or "").strip()
        if not original or not suggested or original == suggested:
            return None
        cleaned: dict[str, str] = {
            "original_name": original,
            "suggested_name": suggested,
        }
        description = item.get("description")
        if description:
            cleaned["description"] = str(description).strip()
        table_name = item.get("table_name")
        if table_name:
            cleaned["table_name"] = str(table_name).strip()
        elif require_table:
            return None
        return cleaned

    for item in parsed.get("classes") or []:
        c = _clean_item(item, require_table=False)
        if c:
            out["classes"].append(c)
    for item in parsed.get("properties") or []:
        p = _clean_item(item, require_table=True)
        if p:
            out["properties"].append(p)
    for item in parsed.get("relationships") or []:
        r = _clean_item(item, require_table=False)
        if r:
            out["relationships"].append(r)

    return out


def _parse_rename_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a sampling text response.

    Handles three common shapes: a bare JSON object, a JSON object inside
    ```json fences, and a JSON object embedded in surrounding prose. Returns
    the parsed dict or None if nothing parses.
    """
    if not text:
        return None

    candidates: list[str] = []

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        candidates.append(fence_match.group(1))

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and parsed:
            return parsed

    return None


async def suggest_semantic_names(
    ctx: Context,
    ontology_file: str | None,
    services: "HandlerContext",
) -> dict[str, Any]:
    """Extract and analyze names from a generated ontology."""
    try:
        try:
            if ontology_file:
                session = services.get_session_data(ctx)
                file_dir = (
                    get_connection_dir(session.connection_id)
                    if session.connection_id
                    else ensure_output_dir()
                )
                ontology_path = file_dir / ontology_file
                if not ontology_path.exists():
                    return {
                        "error": f"Ontology file not found: {ontology_file}",
                        "error_type": "file_not_found",
                        "hint": "Check the filename from generate_ontology response",
                    }
                generator = OntologyGenerator()
                generator.load_from_file(str(ontology_path))
                source_filename = ontology_file
                logger.info(f"Loaded ontology from provided file: {ontology_file}")
            else:
                generator, source_filename = services.load_ontology_from_session(ctx)
        except ValueError as e:
            return {
                "error": str(e),
                "error_type": "session_error",
                "hint": "Pass ontology_file parameter from generate_ontology response",
            }

        extraction_result = generator.extract_names_for_review(compact=True)

        # Build compact review lists — only cryptic items, grouped to save tokens
        cryptic_classes = [
            c["local_name"]
            for c in extraction_result["classes"]
            if c.get("needs_review", {}).get("is_cryptic")
        ]

        # Group cryptic properties by table for compact output
        cryptic_props_by_table: dict[str, list] = {}
        for p in extraction_result["properties"]:
            if p.get("needs_review", {}).get("is_cryptic"):
                table = p.get("table_name") or "unknown"
                cryptic_props_by_table.setdefault(table, []).append(
                    p.get("column_name") or p["local_name"]
                )

        cryptic_relationships = [
            r["local_name"]
            for r in extraction_result["relationships"]
            if r.get("needs_review", {}).get("is_cryptic")
        ]

        total_cryptic = (
            len(cryptic_classes)
            + sum(len(v) for v in cryptic_props_by_table.values())
            + len(cryptic_relationships)
        )
        summary = extraction_result["summary"]

        sampled_suggestions = await _maybe_sample_rename_suggestions(
            ctx,
            cryptic_classes=cryptic_classes,
            cryptic_props_by_table=cryptic_props_by_table,
            cryptic_relationships=cryptic_relationships,
        )

        if sampled_suggestions and any(
            sampled_suggestions.get(k)
            for k in ("classes", "properties", "relationships")
        ):
            sampled_total = sum(
                len(sampled_suggestions.get(k) or [])
                for k in ("classes", "properties", "relationships")
            )
            await safe_ctx_info(
                ctx,
                f"Found {total_cryptic} cryptic names; "
                f"server pre-filled {sampled_total} suggestions via MCP sampling — "
                f"review and call apply_semantic_names",
            )
            return {
                "ontology_file": source_filename,
                "summary": summary,
                "cryptic_classes": cryptic_classes,
                "cryptic_properties_by_table": cryptic_props_by_table,
                "cryptic_relationships": cryptic_relationships,
                "suggestions": sampled_suggestions,
                "suggestions_source": "mcp_sampling",
                "next_step": (
                    "Suggestions are in apply_semantic_names native format "
                    "({classes, properties, relationships} arrays). Pass the "
                    "`suggestions` value through to apply_semantic_names "
                    "verbatim, or edit individual entries first."
                ),
                "next_tool": "apply_semantic_names",
            }

        await safe_ctx_info(
            ctx,
            f"Found {total_cryptic} cryptic names to review; "
            f"next call should be apply_semantic_names with your suggestions",
        )

        return {
            "ontology_file": source_filename,
            "summary": summary,
            "cryptic_classes": cryptic_classes,
            "cryptic_properties_by_table": cryptic_props_by_table,
            "cryptic_relationships": cryptic_relationships,
            "next_step": "Review the cryptic names and call apply_semantic_names with your suggestions",
            "next_tool": "apply_semantic_names",
        }

    except Exception as e:
        if is_client_disconnect(e):
            logger.warning(
                "MCP client closed the session during suggest_semantic_names; "
                "skipping error response (transport already closed)"
            )
            raise
        logger.error(f"Error extracting names for review: {e}")
        return {
            "error": f"Failed to extract names: {e!s}",
            "error_type": "internal_error",
        }


async def apply_semantic_names(
    ctx: Context,
    suggestions: str | dict[str, Any],
    ontology_file: str | None,
    save_to_file: bool,
    services: "HandlerContext",
) -> str | dict[str, Any]:
    """Apply LLM-suggested semantic names to an existing ontology."""
    try:
        session = services.get_session_data(ctx)
        try:
            if ontology_file:
                conn_dir = (
                    get_connection_dir(session.connection_id)
                    if session.connection_id
                    else ensure_output_dir()
                )
                ontology_path = conn_dir / ontology_file
                if not ontology_path.exists():
                    err: dict[str, Any] = services.create_error_response(
                        f"Ontology file not found: {ontology_file}", "file_not_found"
                    )
                    return err
                generator = OntologyGenerator()
                generator.load_from_file(str(ontology_path))
                logger.info(f"Loaded ontology from provided file: {ontology_file}")
            else:
                generator, _ = services.load_ontology_from_session(ctx)
        except ValueError as e:
            err = services.create_error_response(
                f"{e!s} - pass ontology_file parameter from generate_ontology response",
                "session_error",
            )
            return err

        try:
            if isinstance(suggestions, str):
                name_suggestions = json.loads(suggestions)
            else:
                name_suggestions = suggestions
        except json.JSONDecodeError as e:
            err = services.create_error_response(
                f"Invalid JSON in suggestions parameter: {e!s}",
                "parameter_error",
                "Ensure suggestions is valid JSON with classes, properties, and relationships arrays",
            )
            return err

        if not isinstance(name_suggestions, dict):
            err = services.create_error_response(
                "Suggestions must be a JSON object with 'classes', 'properties', and/or 'relationships' arrays",
                "parameter_error",
            )
            return err

        updated_ontology = generator.apply_semantic_names(name_suggestions)

        new_ontology_filename = None
        if save_to_file:
            try:
                conn_dir = (
                    get_connection_dir(session.connection_id)
                    if session.connection_id
                    else ensure_output_dir()
                )
                new_ontology_filename = (
                    services.get_session_safe_filename(ctx, "ontology", "semantic")
                    + ".ttl"
                )
                ontology_file_path = conn_dir / new_ontology_filename

                await write_text_file(ontology_file_path, updated_ontology)

                logger.info(f"Saved semantic ontology to: {ontology_file_path}")
                session.ontology_file = new_ontology_filename
                session.ontology_enriched = True
                session.obqc_validator = None

                # Update workspace: mark ontology as enriched
                if session.connection_id:
                    try:
                        schema_name = session.get_last_analyzed_schema() or "default"
                        await update_workspace_section(
                            connection_id=session.connection_id,
                            output_dir=OUTPUT_DIR,
                            schema_name=schema_name,
                            section="ontology",
                            data={
                                "ontology_file": new_ontology_filename,
                                "enriched": True,
                                "persisted_to_rdf": False,
                                "generated_at": datetime.now().isoformat(),
                            },
                        )
                    except Exception as e:
                        logger.warning(f"Failed to write workspace metadata: {e}")
            except Exception as e:
                logger.warning(f"Failed to save ontology to file: {e}")

        classes_updated = len(name_suggestions.get("classes", []))
        properties_updated = len(name_suggestions.get("properties", []))
        relationships_updated = len(name_suggestions.get("relationships", []))
        total_updated = classes_updated + properties_updated + relationships_updated

        await ctx.info(f"Applied {total_updated} semantic name changes to ontology")

        result = "# Semantic Names Applied Successfully\n\n"
        result += f"- Classes updated: {classes_updated}\n"
        result += f"- Properties updated: {properties_updated}\n"
        result += f"- Relationships updated: {relationships_updated}\n"
        if new_ontology_filename:
            result += f"\n## ontology_file: {new_ontology_filename}\n"
            result += f"\nThe ontology file '{new_ontology_filename}' has been saved and is now the active ontology in session context.\n"

        # Auto-persist to Oxigraph to avoid returning the full TTL
        persisted = False
        if OXIGRAPH_AVAILABLE and services.get_oxigraph_store is not None:
            try:
                store = services.get_oxigraph_store(ctx)
                if store:
                    session = services.get_session_data(ctx)
                    schema_name = (
                        session.current_schema
                        or session.get_last_analyzed_schema()
                        or "default"
                    )
                    graph_uri = schema_graph_uri(schema_name)
                    triple_count = store.load_ontology(
                        updated_ontology, graph_uri, schema_name
                    )
                    result += f"\nPersisted to Oxigraph: {triple_count:,} triples in <{graph_uri}>"
                    result += f"\nToken savings: ~{len(updated_ontology) // 4} tokens saved by auto-persisting to RDF store!"
                    result += '\nUse query_sparql() to explore or download_artifact(artifact_type="ontology") to get the TTL file.'
                    persisted = True
            except Exception as e:
                logger.warning(
                    f"Auto-persist semantic ontology to Oxigraph failed: {e}"
                )

        if not persisted:
            # Without Oxigraph, return a minimal graph summary instead of full TTL
            result += _build_minimal_graph_summary(updated_ontology)

        return result

    except Exception as e:
        logger.error(f"Error applying semantic names: {e}")
        err = services.create_error_response(
            f"Failed to apply semantic names: {e!s}", "internal_error"
        )
        return err

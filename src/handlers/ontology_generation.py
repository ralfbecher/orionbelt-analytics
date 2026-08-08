"""Ontology generation handler: build OWL/RDF from a database schema."""

import asyncio
import json
import logging
import os
from functools import partial
from typing import Any, cast

from fastmcp import Context

from ..constants import DB_SQLGLOT_DIALECTS
from ..database_manager import ColumnInfo, TableInfo
from ..graphrag.manager import _annotate_view_sources
from ..handler_context import HandlerContext
from ..lifecycle.artifacts import artifact_family_lock, prune_superseded_artifacts
from ..lifecycle.metadata import (
    get_active_version_number,
    mark_ontology_persisted,
    ontology_is_current,
    update_schema_version,
    update_workspace_rdf,
    update_workspace_section,
)
from ..oxigraph_store import OXIGRAPH_AVAILABLE, schema_graph_uri
from ..paths import OUTPUT_DIR, ensure_output_dir, get_connection_dir
from ..utils import utc_now, write_text_file

logger = logging.getLogger(__name__)


def _views_for_ontology(session: Any, schema_name: str | None) -> list[Any]:
    """Discovered views for *schema_name*, with their lineage resolved.

    Lineage is filled in here rather than at discovery because it is only the
    ontology that consumes it, and it is parsed with the connection's own
    dialect: a Snowflake body read as PostgreSQL fails to parse, and the view
    then silently carries no sources at all.

    Views are an enrichment, not a prerequisite: an ontology describing the
    tables is useful, and one that fails to generate is not. So anything that
    goes wrong resolving them degrades to "no views" rather than propagating.

    Args:
        session: The session holding the discovery cache.
        schema_name: Schema being generated, or None for the default.

    Returns:
        ViewInfo objects, empty when nothing was discovered or resolution
        failed.
    """
    try:
        views = session.get_cached_views(schema_name or "")
        if not views:
            return []

        dialect = None
        if getattr(session, "db_manager", None) is not None:
            db_type = session.db_manager.connection_info.get("type")
            dialect = DB_SQLGLOT_DIALECTS.get(db_type) if db_type else None

        annotated = _annotate_view_sources(
            [{"name": v.name, "definition": v.definition} for v in views],
            dialect=dialect,
        )
        for view, entry in zip(views, annotated, strict=False):
            view.source_tables = entry.get("referenced_tables", [])

        return cast(list[Any], views)
    except Exception as e:
        logger.warning(
            f"Could not resolve views for the ontology ({e}); "
            "generating it without them."
        )
        return []


def _build_minimal_graph_summary(ontology_ttl: str) -> str:
    """Build a compact class-and-relationship summary from a Turtle ontology.

    Parses the TTL with rdflib and returns a human-readable summary listing
    class names and relationship edges — enough for an LLM to understand the
    schema structure without consuming the full serialization.
    """
    from rdflib import Graph, Namespace
    from rdflib.namespace import OWL, RDF, RDFS

    g = Graph()
    g.parse(data=ontology_ttl, format="turtle")

    oba_ns = None
    for prefix, ns in g.namespaces():
        if prefix == "oba":
            oba_ns = Namespace(str(ns))
            break

    # Collect class names
    class_names = []
    for subj in g.subjects(RDF.type, OWL.Class):
        if subj == OWL.Class:
            continue
        label = None
        for lbl in g.objects(subj, RDFS.label):
            label = str(lbl)
        if not label:
            label = str(subj).split("/")[-1]
        row_count = None
        if oba_ns:
            for rc in g.objects(subj, oba_ns.rowCount):
                row_count = str(rc)
        entry = label
        if row_count:
            entry += f" ({row_count} rows)"
        class_names.append(entry)

    # Collect relationships
    rels = []
    for subj in g.subjects(RDF.type, OWL.ObjectProperty):
        label = None
        for lbl in g.objects(subj, RDFS.label):
            label = str(lbl)
        if not label:
            label = str(subj).split("/")[-1]
        # Get source table from rdfs:domain (oba:tableName is not set on ObjectProperties)
        domain_name = None
        for dom in g.objects(subj, RDFS.domain):
            domain_name = str(dom).split("/")[-1]
        ref_table = None
        if oba_ns:
            for rt in g.objects(subj, oba_ns.referencedTable):
                ref_table = str(rt)
        if domain_name and ref_table:
            rels.append(f"  {domain_name} -> {ref_table}  ({label})")

    lines = ["\n## Minimal Graph Summary\n"]
    lines.append(f"### Classes ({len(class_names)})")
    lines.extend(f"  - {cn}" for cn in sorted(class_names))
    lines.append(f"\n### Relationships ({len(rels)})")
    lines.extend(sorted(set(rels)))
    lines.append(
        '\nUse download_artifact(artifact_type="ontology") to retrieve the full Turtle serialization.'
    )
    return "\n".join(lines)


async def generate_ontology(
    ctx: Context,
    schema_info: str | None,
    schema_name: str | None,
    base_uri: str,
    auto_persist: bool,
    graph_uri: str | None,
    services: "HandlerContext",
) -> str | dict[str, Any]:
    """Generate an RDF ontology from database schema.

    Extracts implementation from main.py's generate_ontology tool.
    """
    # Resolve effective schema and set current schema for state isolation
    session = services.get_session_data(ctx)
    effective_schema_for_state = schema_name
    if not effective_schema_for_state:
        effective_schema_for_state = session.get_last_analyzed_schema()
    if effective_schema_for_state:
        session.set_current_schema(effective_schema_for_state)

    # Check if ontology is already generated
    if session.ontology_file:
        if session.ontology_enriched:
            await ctx.info("Ontology CACHED and already enriched — ready to use")
            return (
                f"# ONTOLOGY ALREADY CACHED AND ENRICHED\n\n"
                f"Ontology file: {session.ontology_file}\n\n"
                f"Semantic names have already been applied. The ontology is ready to use.\n\n"
                f"Do NOT call generate_ontology, discover_schema, or suggest_semantic_names again.\n\n"
                f"## READY TO USE:\n"
                f"- query_sparql() for semantic queries\n"
                f"- execute_sql_query() for data queries (includes built-in validation)"
            )
        await ctx.info("Ontology CACHED - call suggest_semantic_names() for enrichment")
        return (
            f"# ONTOLOGY ALREADY CACHED\n\n"
            f"Ontology file: {session.ontology_file}\n\n"
            f"Do NOT call generate_ontology or discover_schema again!\n\n"
            f"## FOR ENRICHMENT:\n"
            f"Call suggest_semantic_names() NOW - it will use the cached ontology automatically.\n\n"
            f"That's the ONLY tool you need to call for enrichment!"
        )

    # Validate base_uri
    if not base_uri.endswith("/"):
        base_uri += "/"

    tables_info = []

    if schema_info:
        # Use provided schema information
        try:
            schema_data = (
                json.loads(schema_info) if isinstance(schema_info, str) else schema_info
            )

            if "tables" in schema_data:
                for table_data in schema_data["tables"]:
                    columns = []
                    for col_data in table_data.get("columns", []):
                        column = ColumnInfo(
                            name=col_data["name"],
                            data_type=col_data["data_type"],
                            is_nullable=col_data.get("is_nullable", True),
                            is_primary_key=col_data.get("is_primary_key", False),
                            is_foreign_key=col_data.get("is_foreign_key", False),
                            foreign_key_table=col_data.get("foreign_key_table"),
                            foreign_key_column=col_data.get("foreign_key_column"),
                            comment=col_data.get("comment"),
                        )
                        columns.append(column)

                    table = TableInfo(
                        name=table_data["name"],
                        schema=table_data.get("schema", schema_name or "default"),
                        columns=columns,
                        primary_keys=table_data.get("primary_keys", []),
                        foreign_keys=table_data.get("foreign_keys", []),
                        comment=table_data.get("comment"),
                        row_count=table_data.get("row_count"),
                    )
                    tables_info.append(table)

            logger.info(f"Using provided schema info: {len(tables_info)} tables")

        except Exception as e:
            err: dict[str, Any] = services.create_error_response(
                f"Failed to parse schema_info parameter: {e!s}", "parameter_error"
            )
            return err
    else:
        # Try to use cached schema analysis
        session = services.get_session_data(ctx)

        effective_schema = schema_name
        if not effective_schema:
            effective_schema = session.get_last_analyzed_schema()
            if effective_schema:
                logger.info(f"Using last analyzed schema: {effective_schema}")

        cached_tables = session.get_cached_schema(effective_schema or "")

        if cached_tables:
            schema_name = effective_schema
            tables_info = cached_tables
            logger.info(
                f"Using CACHED schema from discover_schema: {len(tables_info)} tables (no re-query needed)"
            )
            await ctx.info(
                f"Using cached schema: {len(tables_info)} tables - no database queries needed"
            )
        else:
            schema_name = effective_schema or schema_name
            db_manager = services.get_session_db_manager(ctx)

            if not db_manager.has_engine():
                err = services.create_error_response(
                    "No database connection established and no schema_info provided. Please use connect_database tool first or provide schema_info parameter.",
                    "connection_error",
                )
                return err

            try:
                tables = db_manager.get_tables(schema_name)
                logger.info(
                    f"Found {len(tables)} tables in schema '{schema_name or 'default'}': {tables}"
                )

                if schema_name:
                    db_manager.prefetch_schema_constraints(schema_name)

                for table_name in tables:
                    try:
                        table_info = db_manager.analyze_table(table_name, schema_name)
                        if table_info:
                            tables_info.append(table_info)
                    except Exception as e:
                        logger.error(f"Failed to analyze table {table_name}: {e}")

                session.cache_schema_analysis(schema_name or "", tables_info)

            except Exception as e:
                err = services.create_error_response(
                    f"Failed to get tables from database: {e!s}", "database_error"
                )
                return err

    if not tables_info:
        err = services.create_error_response(
            "No tables found to generate ontology from", "data_error"
        )
        return err

    # Bound before the slow work below, not after it. Generation plus SHACL
    # validation are the long poles in this handler, and a discover_schema
    # landing during them opens a newer version -- which would then be credited
    # with this run's TTL file and treated as current for RDF auto-persist.
    session = services.get_session_data(ctx)
    target_version: int | None = None
    if session.connection_id:
        try:
            target_version = await get_active_version_number(
                session.connection_id, OUTPUT_DIR, schema_name or "default"
            )
        except Exception as e:
            logger.warning(f"Failed to read active version: {e}")

    generator = services.server_state.get_ontology_generator(base_uri=base_uri)
    views_info = _views_for_ontology(session, schema_name)
    # rdflib work is CPU-bound (~876 ms per 2.65 MB of Turtle); off the loop
    # so it does not freeze every concurrent session.
    ontology_ttl = await asyncio.to_thread(
        partial(generator.generate_from_schema, tables_info, views_info=views_info)
    )

    # Optional SHACL conformance check (Phase 4). Default on, gated by setting;
    # never hard-fails generation — surfaces violations as a warning only.
    if os.getenv("OBA_SHACL_VALIDATE", "true").lower() == "true":
        try:
            from ..shacl_validator import validate_ontology

            shacl = await asyncio.to_thread(validate_ontology, ontology_ttl)
            if shacl["available"] and not shacl["conforms"]:
                logger.warning(
                    "SHACL: generated ontology has %d violation(s):\n%s",
                    shacl["violations"],
                    shacl["report"],
                )
                await ctx.info(
                    f"SHACL validation: {shacl['violations']} violation(s) in the "
                    "generated ontology (non-blocking; see server logs for detail)."
                )
            elif shacl["available"]:
                logger.info("SHACL: generated ontology conforms to oba-shacl shapes.")
        except Exception as e:
            logger.warning(f"SHACL validation step skipped: {e}")

    # Save ontology to connection-scoped output folder
    ontology_filename = None
    try:
        session = services.get_session_data(ctx)
        conn_dir = (
            get_connection_dir(session.connection_id)
            if session.connection_id
            else ensure_output_dir()
        )

        schema_safe = (schema_name or "default").replace(" ", "_").replace(".", "_")

        # Serialize the whole produce -> record -> prune sequence for this
        # family. Two overlapping generate_ontology calls for one schema would
        # otherwise each write a file, and the loser's prune would delete the
        # winner's just-written artifact.
        async with artifact_family_lock(
            conn_dir, f"ontology_{session.connection_id or 'default'}_{schema_safe}"
        ):
            ontology_filename = (
                services.get_session_safe_filename(ctx, "ontology", schema_safe)
                + ".ttl"
            )
            ontology_file_path = conn_dir / ontology_filename

            await write_text_file(ontology_file_path, ontology_ttl)

            logger.info(
                f"Generated ontology for schema '{schema_name or 'default'}': {len(tables_info)} tables"
            )
            logger.info(f"Saved ontology to: {ontology_file_path}")

            previous_ontology_file = session.ontology_file
            session.ontology_file = ontology_filename
            session.obqc_validator = None

            # Write workspace metadata for ontology section
            if session.connection_id:
                try:
                    await update_workspace_section(
                        connection_id=session.connection_id,
                        output_dir=OUTPUT_DIR,
                        schema_name=schema_name or "default",
                        section="ontology",
                        data={
                            "ontology_file": ontology_filename,
                            "enriched": False,
                            "graph_uri": graph_uri,
                            "persisted_to_rdf": False,
                            "generated_at": utc_now().isoformat(),
                        },
                    )
                    # Recorded against the version captured before generation
                    # started, so this TTL lands on the generation it was built
                    # from. The triple count is not known until the RDF load
                    # below, so it is filled in there against the same version.
                    await update_schema_version(
                        connection_id=session.connection_id,
                        output_dir=OUTPUT_DIR,
                        schema_name=schema_name or "default",
                        updates={
                            "ontology_ttl_file": ontology_filename,
                            "ontology_graph_uri": graph_uri or "",
                        },
                        version=target_version,
                    )
                except Exception as e:
                    logger.warning(f"Failed to write workspace metadata: {e}")

            # Prune only once metadata durably names the new file; the previously
            # referenced one is protected so a failed metadata write cannot leave
            # restore chasing a deleted artifact.
            await prune_superseded_artifacts(
                ontology_file_path,
                protect=[previous_ontology_file] if previous_ontology_file else [],
            )

        await ctx.info(
            "Ontology generation complete; next call should be suggest_semantic_names to improve cryptic names"
        )

        # Analyze for cryptic names
        generator = services.server_state.get_ontology_generator(base_uri=base_uri)
        await asyncio.to_thread(
            generator.graph.parse, data=ontology_ttl, format="turtle"
        )
        generator.graph.bind("ns", generator.base_uri)
        generator.graph.bind("oba", generator.oba_ns)
        name_analysis = generator.extract_names_for_review()

        cryptic_count = (
            name_analysis["summary"]["classes_needing_review"]
            + name_analysis["summary"]["properties_needing_review"]
            + name_analysis["summary"]["relationships_needing_review"]
        )

        # Auto-persist to RDF store.
        #
        # Serialized per schema and gated on still being the current
        # generation, because load_ontology() *replaces* the named graph: a
        # stale request that reached this point would overwrite a newer
        # generation's triples. Guarding only the metadata flag left the flag
        # honest while the graph held the wrong ontology.
        #
        # The lock is taken here rather than extending the artifact family lock
        # over the whole handler: the Oxigraph load is the expensive part and
        # only conflicts with other loads into the same schema graph.
        if auto_persist and OXIGRAPH_AVAILABLE:
            try:
                store = services.get_oxigraph_store(ctx)
                if store:
                    if not graph_uri:
                        graph_uri = schema_graph_uri(schema_name or "default")

                    persist_schema = schema_name or "default"
                    async with artifact_family_lock(
                        conn_dir,
                        f"rdf_{session.connection_id or 'default'}"
                        f"_{persist_schema}",
                    ):
                        still_current = not session.connection_id or (
                            await ontology_is_current(
                                session.connection_id,
                                OUTPUT_DIR,
                                persist_schema,
                                ontology_filename,
                            )
                        )
                        if still_current:
                            triple_count = await asyncio.to_thread(
                                store.load_ontology,
                                ontology_ttl,
                                graph_uri,
                                persist_schema,
                            )
                            logger.info(
                                f"Auto-persisted ontology to Oxigraph: "
                                f"{triple_count} triples in graph <{graph_uri}>"
                            )
                        else:
                            logger.info(
                                "Skipped RDF auto-persist: a newer ontology "
                                f"generation superseded {ontology_filename}"
                            )

                    # Update workspace: mark ontology as persisted + write rdf_store
                    if still_current and session.connection_id:
                        try:
                            # Re-checked inside the helper: the currency test
                            # above happened before the load, and a newer
                            # generation can land while it runs.
                            marked = await mark_ontology_persisted(
                                connection_id=session.connection_id,
                                output_dir=OUTPUT_DIR,
                                schema_name=schema_name or "default",
                                ontology_file=ontology_filename,
                                graph_uri=graph_uri,
                            )
                            if not marked:
                                logger.info(
                                    "Skipped persisted_to_rdf flag: a newer "
                                    "ontology generation superseded "
                                    f"{ontology_filename}"
                                )
                            await update_workspace_rdf(
                                connection_id=session.connection_id,
                                output_dir=OUTPUT_DIR,
                                data={
                                    "initialized": True,
                                    "graph_uris": [graph_uri],
                                    "initialized_at": utc_now().isoformat(),
                                },
                            )
                            # Only now is the triple count known. Gated on
                            # `marked` for the same reason that flag is: if a
                            # newer generation superseded this one, its counts
                            # are the ones the version should carry.
                            if marked:
                                await update_schema_version(
                                    connection_id=session.connection_id,
                                    output_dir=OUTPUT_DIR,
                                    schema_name=schema_name or "default",
                                    updates={
                                        "ontology_triple_count": triple_count,
                                        "ontology_graph_uri": graph_uri,
                                    },
                                    version=target_version,
                                )
                        except Exception as e:
                            logger.warning(f"Failed to write workspace metadata: {e}")

                    result = f"""Ontology generated and stored successfully!

Schema: {schema_name or "default"}
Tables: {len(tables_info)}
Ontology file: {ontology_filename}
Storage location: {conn_dir}/
Graph URI: <{graph_uri}>
Triples stored: {triple_count:,}

Ontology is now persistent in Oxigraph RDF database.
Use query_sparql() to explore the schema graph.
Use download_artifact(artifact_type='ontology', schema_name='{schema_name or "default"}') to get the TTL file.

Token savings: ~{len(ontology_ttl)//4} tokens saved by auto-persisting to RDF store!"""

                    if cryptic_count > 0:
                        result += f"""

SEMANTIC NAME RESOLUTION RECOMMENDED
Found {cryptic_count} names that may need review:
  Classes needing review: {name_analysis['summary']['classes_needing_review']}
  Properties needing review: {name_analysis['summary']['properties_needing_review']}
  Relationships needing review: {name_analysis['summary']['relationships_needing_review']}

To improve ontology for business users:
1. Call suggest_semantic_names() (ontology is CACHED)
2. Review suggestions and provide alternatives
3. Call apply_semantic_names() with your suggestions"""

                    return result
            except Exception as e:
                logger.warning(
                    f"Auto-persist to Oxigraph failed: {e}, falling back to full TTL return"
                )

        # Fallback: return minimal graph summary instead of full TTL
        result = f"Ontology generated and saved to file: {ontology_filename}\n"
        result += f"Schema: {schema_name or 'default'}, Tables: {len(tables_info)}\n"
        result += await asyncio.to_thread(_build_minimal_graph_summary, ontology_ttl)

        if cryptic_count > 0:
            result += "\n\nSEMANTIC NAME RESOLUTION RECOMMENDED"
            result += f"\nFound {cryptic_count} names that may be abbreviations or cryptic identifiers."
            result += "\n1. Call suggest_semantic_names() - NO parameters needed, ontology is CACHED"
            result += (
                "\n2. Review the suggestions and provide business-friendly alternatives"
            )
            result += "\n3. Call apply_semantic_names() with your suggestions"

        return result

    except Exception as e:
        logger.warning(f"Failed to save ontology to file: {e}")
        await ctx.info(
            "Ontology file save failed but ontology generated; next call should be suggest_semantic_names to improve cryptic names"
        )
        return await asyncio.to_thread(_build_minimal_graph_summary, ontology_ttl)

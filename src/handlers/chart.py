"""Chart generation handler implementation."""

import asyncio
import logging
from typing import Any, cast
from uuid import uuid4

from fastmcp import Context
from fastmcp.utilities.types import Image

from ..chart_utils import save_image_to_tmp
from ..config import config_manager
from ..handler_context import HandlerContext

logger = logging.getLogger(__name__)

# Static PNG export dimensions. Interactive charts are responsive (autosize) and
# ignore these; they only size the PNG fallback/export.
PNG_WIDTH = 800
PNG_HEIGHT = 600


async def generate_chart(
    ctx: Context,
    data_source: list[dict[str, Any]] | str,
    chart_type: str,
    x_column: str,
    y_column: str | list[str] | None,
    color_column: str | None,
    title: str | None,
    chart_style: str,
    sort_by: str | None,
    sort_order: str | None,
    output_format: str,
    services: "HandlerContext",
) -> str | list[str | Image]:
    """Generate interactive or static charts from query results.

    This handler delegates to tools.chart.generate_chart for the core charting
    logic and handles MCP Apps resource registration.
    """
    import json

    from ..tools.chart import generate_chart as generate_chart_impl

    # Parse data_source if it's a string
    if data_source and isinstance(data_source, str):
        raw_data_source = data_source
        logger.warning(
            "data_source was sent as string instead of JSON array - attempting to parse"
        )
        try:
            parsed = json.loads(raw_data_source)
            if isinstance(parsed, list):
                data_source = parsed
                logger.info("Successfully parsed data_source from JSON string format")
        except (json.JSONDecodeError, ValueError):
            try:
                import ast

                parsed = ast.literal_eval(raw_data_source)
                if isinstance(parsed, list):
                    data_source = parsed
                    logger.info(
                        "Successfully parsed data_source from Python literal string format"
                    )
            except (ValueError, SyntaxError) as e:
                await ctx.info("Chart generation failed - data_source format error")
                raise RuntimeError(
                    f"data_source must be valid JSON (array of objects), not a string. "
                    f"Received string: {raw_data_source[:100]}... "
                    f"Expected format: [{{'key': 'value'}}, ...] "
                    f"Parse error: {e!s}"
                )

    # Parse y_column if it's a JSON string list
    if y_column and isinstance(y_column, str):
        try:
            parsed = json.loads(y_column)
            if isinstance(parsed, list):
                y_column = parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Call implementation
    logger.info(
        f"generate_chart called with output_format={output_format}, chart_type={chart_type}"
    )
    chart_data = cast(list[dict[str, Any]], data_source)
    result = generate_chart_impl(
        chart_data,
        chart_type,
        x_column,
        y_column,
        color_column,
        title,
        chart_style,
        PNG_WIDTH,
        PNG_HEIGHT,
        sort_by,
        sort_order,
        output_format=output_format,
    )
    logger.info(
        f"generate_chart_impl returned: {type(result)}, has_error={isinstance(result, dict) and 'error' in result}"
    )

    # Check for errors
    if isinstance(result, dict) and result.get("error"):
        await ctx.info("Chart generation failed")
        raise RuntimeError(result.get("error", "Chart generation failed"))

    # Handle interactive output — generate HTML and register as ui:// resource
    if output_format == "interactive":
        if isinstance(result, dict) and "figure" in result:
            fig = result["figure"]
            metadata = result.get("metadata", {})
            data_points = metadata.get("data_points", 0)
            chart_type_display = metadata.get("chart_type", chart_type)

            # Generate self-contained HTML with Plotly.js from CDN
            # Remove fixed dimensions so the chart is fully responsive in its container
            fig.update_layout(width=None, height=None, autosize=True)
            html = fig.to_html(include_plotlyjs="cdn", full_html=True)

            # Register as FastMCP Apps resources:
            # - HTML for Claude Desktop / MCP Apps clients
            # - JSON for direct consumption by OB Chat (avoids HTML parsing)
            chart_id = uuid4()
            chart_uri = f"ui://orionbelt/chart/{chart_id}"
            chart_json_uri = f"ui://orionbelt/chart-json/{chart_id}"
            if services.provides("add_resource"):
                from fastmcp.resources import TextResource

                services.add_resource(
                    TextResource(
                        uri=cast(Any, chart_uri),
                        name=f"Chart: {title or chart_type_display}",
                        text=html,
                        mime_type="text/html",
                    )
                )
                services.add_resource(
                    TextResource(
                        uri=cast(Any, chart_json_uri),
                        name=f"Chart JSON: {title or chart_type_display}",
                        text=fig.to_json(),
                        mime_type="application/json",
                    )
                )
                logger.info(
                    f"Registered chart resources: {chart_uri}, {chart_json_uri}"
                )

            # Export a static PNG: saved to disk and returned inline as ImageContent
            image_inline = None
            file_uri = ""
            try:
                image_id = str(uuid4())
                image_bytes = fig.to_image(
                    format="png", width=PNG_WIDTH, height=PNG_HEIGHT
                )
                connection_id = services.get_session_data(ctx).connection_id
                image_file_path = await asyncio.to_thread(
                    save_image_to_tmp,
                    image_bytes,
                    image_id,
                    "png",
                    connection_id=connection_id,
                )
                if image_file_path:
                    file_uri = f"\nStatic image: file://{image_file_path}"
                image_inline = Image(data=image_bytes, format="png")
            except Exception as e:
                logger.debug(f"PNG export failed: {e}")

            await ctx.info(
                f"Interactive {chart_type_display} chart with {data_points} data points"
            )
            text_result = (
                f"Chart generated: {chart_uri}{file_uri}\nChart JSON: {chart_json_uri}"
            )
            return_binary = config_manager.get_server_config().chart_return_binary
            if image_inline and return_binary:
                return [text_result, image_inline]
            return text_result
        else:
            await ctx.info("Chart generation failed")
            raise RuntimeError(
                "Chart generation failed: unexpected result format for interactive mode"
            )

    # Handle image output
    if isinstance(result, tuple) and len(result) == 2:
        image_bytes, image_id = result

        connection_id = services.get_session_data(ctx).connection_id

        image_file_path = await asyncio.to_thread(
            save_image_to_tmp,
            image_bytes,
            image_id,
            "png",
            connection_id=connection_id,
        )

        if not image_file_path:
            await ctx.info("Chart generation failed to save file")
            raise RuntimeError("Failed to save chart image to file")

        await ctx.info(f"Chart image saved: {image_file_path}")
        return_binary = config_manager.get_server_config().chart_return_binary
        if return_binary:
            return [
                f"Chart saved to: {image_file_path}",
                Image(data=image_bytes, format="png"),
            ]
        return f"Chart saved to: {image_file_path}"
    else:
        await ctx.info("Chart generation failed")
        raise RuntimeError("Chart generation failed: unexpected result format")

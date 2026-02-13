"""Chart generation tool."""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from ..chart_utils import create_plotly_chart, save_image_to_tmp

logger = logging.getLogger(__name__)


def generate_chart(
    data_source: List[Dict[str, Any]],
    chart_type: str,
    x_column: str,
    y_column: Optional[Union[str, List[str]]] = None,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    chart_style: str = "grouped",
    width: int = 800,
    height: int = 600,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    output_format: str = "image"
) -> Union[Tuple[bytes, str], Dict[str, Any]]:
    """Generate chart and return either Plotly data or image bytes.

    Args:
        data_source: List of dictionaries containing the data
        chart_type: Type of chart (bar, line, scatter, heatmap)
        x_column: Column name for X-axis
        y_column: Column name(s) for Y-axis. Can be:
            - String: single measure (all chart types)
            - List of strings: multiple measures (line charts only - creates multi-line comparison)
        color_column: Column for color grouping (for stacked/grouped bar charts)
        title: Chart title
        chart_style: "grouped" or "stacked" (for bar charts)
        width: Chart width in pixels
        height: Chart height in pixels
        sort_by: Column to sort by. If None, uses automatic sorting based on chart type:
            - Bar/grouped/stacked: sorts by measure (y_column) descending
            - Line: sorts by dimension (x_column) ascending
            - Heatmap: sorts x_column ascending, y_column descending
        sort_order: 'ascending' or 'descending'. If None, uses automatic order based on chart type.
        output_format: "interactive" returns Plotly JSON data for MCP Apps rendering.
                      "image" returns PNG bytes for static display (requires kaleido).

    Returns:
        For output_format="interactive": Dict with 'traces', 'layout', 'config' for Plotly
        For output_format="image": Tuple of (image_bytes, chart_id)
        Dict with 'error' key on failure
    """
    try:
        # Check for visualization libraries
        missing_libs = []

        try:
            import pandas as pd
        except ImportError:
            missing_libs.append("pandas")

        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            missing_libs.append("plotly")

        if missing_libs:
            return {"error": f"❌ Missing required visualization libraries: {', '.join(missing_libs)}. Install with: pip install {' '.join(missing_libs)}"}

        # Validate input data
        if not data_source:
            return {"error": "❌ No data provided for charting"}

        data = data_source

        # Convert to pandas DataFrame
        df = pd.DataFrame(data)

        # Validate required columns
        if x_column not in df.columns:
            return {"error": f"❌ X-axis column '{x_column}' not found in data. Available columns: {list(df.columns)}"}

        # Auto-detect time series data and switch to line chart
        def is_time_based_column(column_data):
            """Detect if a column contains time-based data."""
            # Check if already datetime dtype
            if pd.api.types.is_datetime64_any_dtype(column_data):
                return True

            # Sample first few non-null values to check
            sample = column_data.dropna().head(10)
            if len(sample) == 0:
                return False

            # Check for common time-based patterns
            if column_data.dtype == 'object' or column_data.dtype == 'string':
                for val in sample:
                    val_str = str(val).lower()
                    # Check for common date/time patterns
                    time_indicators = [
                        'date', 'time', 'year', 'month', 'day', 'hour',
                        'timestamp', 'period', 'quarter', 'week'
                    ]
                    # Check if value contains date separators or time indicators
                    if any(sep in str(val) for sep in ['-', '/', ':', 'T']) or \
                       any(ind in val_str for ind in time_indicators):
                        try:
                            # Try to parse as datetime
                            pd.to_datetime(val)
                            return True
                        except (ValueError, TypeError):
                            pass

            return False

        # Detect time series and auto-switch to line chart
        if x_column in df.columns and is_time_based_column(df[x_column]):
            original_chart_type = chart_type
            if chart_type == "bar":
                chart_type = "line"
                logger.info(f"Auto-switched from bar to line chart due to time-based x-axis: {x_column}")

            # Auto-set sorting for time series if not specified
            if sort_by is None:
                sort_by = x_column
                sort_order = "ascending"
                logger.info(f"Auto-sorting time series by {x_column} ascending")

        # Validate y_column(s)
        if chart_type in ["bar", "line", "scatter"] and y_column:
            if isinstance(y_column, list):
                # Multiple measures (only supported for line charts)
                if chart_type != "line":
                    return {"error": f"❌ Multiple y_columns only supported for line charts, not {chart_type}"}
                missing_cols = [col for col in y_column if col not in df.columns]
                if missing_cols:
                    return {"error": f"❌ Y-axis columns {missing_cols} not found in data. Available columns: {list(df.columns)}"}
            else:
                # Single measure
                if y_column not in df.columns:
                    return {"error": f"❌ Y-axis column '{y_column}' not found in data. Available columns: {list(df.columns)}"}

        # Generate title if not provided
        if not title:
            if chart_type == "bar":
                y_label = y_column if isinstance(y_column, str) else 'Count'
                title = f"{y_label} by {x_column}"
                if chart_style == "stacked" and color_column:
                    title += f" (stacked by {color_column})"
            elif chart_type == "line":
                if isinstance(y_column, list):
                    title = f"Comparison of {', '.join(y_column)} over {x_column}"
                else:
                    title = f"{y_column} over {x_column}"
            elif chart_type == "scatter":
                title = f"{y_column} vs {x_column}"
            elif chart_type == "heatmap":
                title = f"Heatmap of {x_column}" + (f" and {y_column}" if y_column else "")
            else:
                title = f"Chart of {x_column}"

        # Create chart ID for file naming and logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in title.replace(" ", "_") if c.isalnum() or c in "_-")
        chart_id = f"{chart_type}_{safe_title}_{timestamp}"

        # Handle interactive output format (return Plotly JSON data)
        if output_format == "interactive":
            try:
                fig = create_plotly_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width, height, sort_by, sort_order)

                # Convert figure to JSON-serializable dict
                fig_dict = fig.to_dict()

                # Return Plotly data for MCP Apps rendering
                result = {
                    "traces": fig_dict.get("data", []),
                    "layout": fig_dict.get("layout", {}),
                    "config": {
                        "displayModeBar": True,
                        "responsive": True,
                        "scrollZoom": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                        "toImageButtonOptions": {
                            "format": "png",
                            "filename": chart_id,
                            "scale": 2
                        }
                    },
                    "metadata": {
                        "chart_id": chart_id,
                        "chart_type": chart_type,
                        "data_points": len(df),
                        "library": "plotly"
                    }
                }

                logger.info(f"Created interactive {chart_type} chart with {len(df)} data points")
                return result

            except ImportError as e:
                logger.warning(f"Plotly not available for interactive mode: {e}")
                return {"error": "❌ Plotly required for interactive charts. Install with: pip install plotly"}

        # Handle image output format (return PNG bytes using Plotly + kaleido)
        try:
            fig = create_plotly_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width, height, sort_by, sort_order)
            image_bytes = fig.to_image(format='png', width=width, height=height, scale=2)
        except Exception as e:
            if "kaleido" in str(e).lower():
                return {"error": "❌ kaleido required for PNG export. Install with: pip install kaleido"}
            raise

        logger.info(f"Created {chart_type} chart image with {len(df)} data points")

        # Also save to tmp directory for backward compatibility
        save_image_to_tmp(image_bytes, chart_id, 'png')
        return (image_bytes, chart_id)

    except Exception as e:
        logger.error(f"Chart creation error: {e}")
        return {"error": f"❌ Chart generation failed: {str(e)}"}
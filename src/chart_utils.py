"""Chart generation utilities for the Database Ontology MCP Server."""

import logging
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Temporary directory for file storage
TMP_DIR = Path(__file__).parent.parent / "tmp"


def format_measure_name(measure: str) -> str:
    """Format measure name by removing underscores and applying title case.

    Args:
        measure: Raw measure name (e.g., "total_revenue", "profit_margin_pct")

    Returns:
        Formatted measure name (e.g., "Total Revenue", "Profit Margin Pct")
    """
    return measure.replace('_', ' ').title()


def get_quarter_sort_order(values, ascending=True):
    """Get chronological sort order for quarterly values.

    Supports multiple formats:
    - Q1 2024, Q2 2024, etc.
    - 2024 Q1, 2024 Q2, etc.
    - 2024/Q1, 2024/Q2, etc.
    - 2024-Q1, 2024-Q2, etc.
    - Q1-2024, Q2-2024, etc.

    Args:
        values: List or array of quarter strings
        ascending: Whether to sort in ascending order

    Returns:
        List of values in chronological order, or None if not quarterly data
    """
    import re
    import pandas as pd

    # Check if all values match any quarterly pattern
    values_str = [str(v) for v in values if pd.notna(v)]

    # Multiple patterns to support different formats
    patterns = [
        r'^Q([1-4])\s*[\-/]?\s*(\d{4})$',  # Q1 2024, Q1-2024, Q1/2024
        r'^(\d{4})\s*[\-/]?\s*Q([1-4])$',  # 2024 Q1, 2024-Q1, 2024/Q1
    ]

    def parse_quarter(q_str):
        """Parse quarter string and return (year, quarter, original_string)."""
        q_str_clean = str(q_str).strip()

        for pattern in patterns:
            match = re.match(pattern, q_str_clean, re.IGNORECASE)
            if match:
                g1, g2 = match.groups()
                # Check which group is the quarter (1-4) and which is the year (4 digits)
                if len(g1) == 4:  # Format: 2024 Q1
                    year, quarter = int(g1), int(g2)
                else:  # Format: Q1 2024
                    quarter, year = int(g1), int(g2)
                return (year, quarter, q_str_clean)

        return None

    # Try to parse all values
    parsed = []
    for v in values_str:
        result = parse_quarter(v)
        if result is None:
            return None  # Not all values are quarterly data
        parsed.append(result)

    # Sort by year and quarter
    parsed.sort(key=lambda x: (x[0], x[1]), reverse=not ascending)

    return [x[2] for x in parsed]


def _clean_dataframe_for_plotly(df):
    """Create a clean DataFrame copy safe for Plotly operations.

    The "cannot insert X, already exists" error occurs when a pandas DataFrame
    has an index.name that matches an existing column name. When reset_index()
    is called (either explicitly or internally by pandas/Plotly), pandas tries
    to insert the index as a new column with that name, causing the conflict.

    This function creates a completely fresh DataFrame from the data records,
    which eliminates any index metadata that could cause conflicts.

    Args:
        df: Input DataFrame (may have problematic index.name)

    Returns:
        Clean DataFrame with default RangeIndex and no index.name
    """
    import pandas as pd

    # Create a fresh DataFrame from records - this completely removes any
    # index metadata (name, type, etc.) that could conflict with columns
    clean_df = pd.DataFrame(df.to_dict('records'))
    return clean_df


def create_plotly_chart(df, chart_type, x_column, y_column, color_column, title, chart_style, width=800, height=600, sort_by=None, sort_order=None):
    """Create Plotly chart based on type.

    Args:
        y_column: Can be a string (single measure) or list of strings (multiple measures for line charts)
        sort_by: Column to sort by. If None, uses automatic sorting based on chart type:
            - Bar/grouped/stacked: sorts by measure (y_column) descending
            - Line: sorts by dimension (x_column) ascending
            - Heatmap: sorts x_column ascending, y_column descending
        sort_order: 'ascending' or 'descending'. If None, uses automatic order based on chart type.
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    # Create a completely clean DataFrame to avoid pandas index/column conflicts
    # (fixes "cannot insert X, already exists" errors when df.index.name matches a column)
    df = _clean_dataframe_for_plotly(df)

    if chart_type == "bar":
        # Check if x_column contains chronological data (quarters, months, or dates)
        x_values_lower = df[x_column].astype(str).str.lower().unique()
        month_names_full = ['january', 'february', 'march', 'april', 'may', 'june',
                           'july', 'august', 'september', 'october', 'november', 'december']
        month_names_short = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

        is_month_column = any(val in month_names_full for val in x_values_lower) or \
                         any(val in month_names_short for val in x_values_lower)
        is_quarterly_x = get_quarter_sort_order(df[x_column].unique()) is not None

        # Try to detect datetime
        is_datetime = False
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(df[x_column])
            is_datetime = True
        except (ValueError, TypeError):
            pass

        is_chronological = is_quarterly_x or is_month_column or is_datetime

        # Determine sorting for bar charts
        # Chronological data: sort ascending by x_column (time order)
        # Non-chronological data: sort descending by y_column (measure, largest first)
        if is_chronological:
            effective_sort_by = sort_by if sort_by is not None else x_column
            effective_sort_order = sort_order if sort_order is not None else 'ascending'
        else:
            effective_sort_by = sort_by if sort_by else y_column
            effective_sort_order = sort_order if sort_order else 'descending'

        sort_ascending = (effective_sort_order == 'ascending')

        if chart_style == "stacked" and color_column:
            # Stacked bar chart with two dimensions: x_column (categories) and color_column (stack groups)
            # Aggregate data first to handle duplicates
            agg_df = df.groupby([x_column, color_column], as_index=False)[y_column].sum()
            agg_df = _clean_dataframe_for_plotly(agg_df)  # Clean for pandas index conflicts
            # Check for quarterly data first
            quarter_order = get_quarter_sort_order(agg_df[x_column].unique(), ascending=sort_ascending)
            if quarter_order:
                category_order = quarter_order
            elif effective_sort_by == y_column:
                total_by_category = agg_df.groupby(x_column)[y_column].sum().sort_values(ascending=sort_ascending)
                category_order = total_by_category.index.tolist()
            else:
                # Sort by x_column or another column
                total_by_category = agg_df.groupby(x_column)[effective_sort_by].sum().sort_values(ascending=sort_ascending)
                category_order = total_by_category.index.tolist()
            fig = px.bar(agg_df, x=x_column, y=y_column, color=color_column, title=title,
                        barmode='stack', category_orders={x_column: category_order},
                        labels={x_column: format_measure_name(x_column),
                                y_column: format_measure_name(y_column)})
        elif color_column:
            # Grouped bar chart - aggregate data first to handle duplicates
            agg_df = df.groupby([x_column, color_column], as_index=False)[y_column].sum()
            agg_df = _clean_dataframe_for_plotly(agg_df)  # Clean for pandas index conflicts
            # Check for quarterly data first
            quarter_order = get_quarter_sort_order(agg_df[x_column].unique(), ascending=sort_ascending)
            if quarter_order:
                category_order = quarter_order
            elif effective_sort_by == y_column:
                total_by_category = agg_df.groupby(x_column)[y_column].sum().sort_values(ascending=sort_ascending)
                category_order = total_by_category.index.tolist()
            else:
                # Sort by x_column or another column
                total_by_category = agg_df.groupby(x_column)[effective_sort_by].sum().sort_values(ascending=sort_ascending)
                category_order = total_by_category.index.tolist()
            fig = px.bar(agg_df, x=x_column, y=y_column, color=color_column, title=title,
                        barmode='group', category_orders={x_column: category_order},
                        labels={x_column: format_measure_name(x_column),
                                y_column: format_measure_name(y_column)})
            # Make bars wider for grouped charts
            fig.update_layout(bargap=0.15, bargroupgap=0.05)
        else:
            # Simple bar chart - aggregate first, then sort by measure (default) or specified column
            agg_df = df.groupby(x_column, as_index=False)[y_column].sum()
            agg_df = _clean_dataframe_for_plotly(agg_df)  # Clean for pandas index conflicts
            # Check for quarterly data first
            quarter_order = get_quarter_sort_order(agg_df[x_column].unique(), ascending=sort_ascending)
            if quarter_order:
                category_order = quarter_order
            else:
                # Sort by aggregated measure (default) or specified column
                sorted_df = agg_df.sort_values(by=effective_sort_by, ascending=sort_ascending)
                category_order = sorted_df[x_column].tolist()
            fig = px.bar(agg_df, x=x_column, y=y_column, title=title,
                        category_orders={x_column: category_order},
                        labels={x_column: format_measure_name(x_column),
                                y_column: format_measure_name(y_column)})
    elif chart_type == "line":
        # Work with a clean copy to avoid modifying the original dataframe
        # and prevent Plotly 6.x index/column conflicts
        sorted_df = _clean_dataframe_for_plotly(df)

        # Determine sorting for line charts
        # Default: sort by dimension (x_column) ascending
        effective_sort_by = sort_by if sort_by else x_column
        effective_sort_order = sort_order if sort_order else 'ascending'
        sort_ascending = (effective_sort_order == 'ascending')

        # Define month names and abbreviations for chronological sorting
        month_names_full = ['january', 'february', 'march', 'april', 'may', 'june',
                           'july', 'august', 'september', 'october', 'november', 'december']
        month_names_short = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

        # Check if the x_column contains month values (full or abbreviated)
        x_values_lower = sorted_df[x_column].astype(str).str.lower().unique()
        is_month_column = (
            any(val in month_names_full for val in x_values_lower) or
            any(val in month_names_short for val in x_values_lower)
        )

        # Check if the x_column contains quarterly values (e.g., "Q1 2024", "Q2 2024")
        import re
        x_values_str = sorted_df[x_column].astype(str).unique()
        is_quarter_column = all(
            re.match(r'^Q[1-4]\s*\d{4}$', str(val).strip(), re.IGNORECASE)
            for val in x_values_str if pd.notna(val)
        )

        if is_quarter_column and effective_sort_by == x_column:
            # Extract quarter and year for proper sorting
            def parse_quarter(q_str):
                match = re.match(r'^Q([1-4])\s*(\d{4})$', str(q_str).strip(), re.IGNORECASE)
                if match:
                    qtr = int(match.group(1))
                    year = int(match.group(2))
                    return (year, qtr)
                return (0, 0)  # Fallback for invalid formats

            # Add a temporary column for sorting (use unique name to avoid conflicts)
            temp_col = '_qtr_sort_order'
            if temp_col in sorted_df.columns:
                sorted_df = sorted_df.drop(columns=[temp_col])
            sorted_df[temp_col] = sorted_df[x_column].apply(parse_quarter)

            # Sort by the temporary column
            sorted_df = sorted_df.sort_values(by=temp_col, ascending=sort_ascending)

            # Remove the temporary column
            sorted_df = sorted_df.drop(columns=[temp_col])
        elif is_month_column and effective_sort_by == x_column:
            # Create a mapping for month sorting
            month_order = {month: i for i, month in enumerate(month_names_full)}
            month_order.update({month: i for i, month in enumerate(month_names_short)})

            # Add a temporary column for sorting
            sorted_df['_month_order'] = sorted_df[x_column].astype(str).str.lower().map(month_order)

            # Sort by the temporary column
            sorted_df = sorted_df.sort_values(by='_month_order', ascending=sort_ascending)

            # Remove the temporary column
            sorted_df = sorted_df.drop(columns=['_month_order'])
        else:
            # Try to convert x_column to datetime if it looks like a date
            # This ensures proper chronological sorting
            try:
                sorted_df[x_column] = pd.to_datetime(sorted_df[x_column])
            except (ValueError, TypeError):
                # Not a datetime column, use as-is
                pass

            # Sort data by x-axis column (default) or specified column for proper line chart ordering
            sorted_df = sorted_df.sort_values(by=effective_sort_by, ascending=sort_ascending)

        # Support multiple measures for line charts
        if isinstance(y_column, list):
            # Multiple measures - separate into regular and percentage measures
            # Strip spaces and check case-insensitively for percentage measures
            pct_measures = [m for m in y_column if m in sorted_df.columns and (
                m.strip().lower().endswith('_pct') or
                m.strip().lower().endswith('_percent') or
                m.strip().lower().endswith('_percentage')
            )]
            value_measures = [m for m in y_column if m in sorted_df.columns and m not in pct_measures]

            fig = go.Figure()

            # Plot value measures on primary y-axis
            for measure in value_measures:
                fig.add_trace(go.Scatter(
                    x=sorted_df[x_column],
                    y=sorted_df[measure],
                    mode='lines+markers',
                    name=format_measure_name(measure),
                    yaxis='y',
                    showlegend=True
                ))

            # Plot percentage measures on secondary y-axis if we have both types
            for measure in pct_measures:
                fig.add_trace(go.Scatter(
                    x=sorted_df[x_column],
                    y=sorted_df[measure],
                    mode='lines+markers',
                    name=format_measure_name(measure),
                    yaxis='y2' if value_measures else 'y',
                    line=dict(dash='dash') if value_measures else None,
                    showlegend=True
                ))

            # Configure layout with dual y-axes if needed
            layout_config = {
                'title': title,
                'xaxis_title': format_measure_name(x_column),
                'showlegend': True
            }

            if pct_measures and value_measures:
                # Dual y-axis configuration
                layout_config['yaxis'] = dict(
                    title=", ".join([format_measure_name(m) for m in value_measures]),
                    side='left'
                )
                layout_config['yaxis2'] = dict(
                    title=", ".join([format_measure_name(m) for m in pct_measures]),
                    side='right',
                    overlaying='y',
                    titlefont=dict(color='gray'),
                    tickfont=dict(color='gray'),
                    ticksuffix='%'
                )
            elif pct_measures:
                layout_config['yaxis_title'] = ", ".join([format_measure_name(m) for m in pct_measures])
            else:
                layout_config['yaxis_title'] = ", ".join([format_measure_name(m) for m in value_measures])

            fig.update_layout(**layout_config)
        else:
            # Single measure with optional color grouping
            fig = px.line(sorted_df, x=x_column, y=y_column, color=color_column, title=title,
                         labels={x_column: format_measure_name(x_column),
                                 y_column: format_measure_name(y_column)})

        # Enhance for time series
        if sorted_df[x_column].dtype in ['datetime64[ns]']:
            fig.update_xaxes(title=format_measure_name(x_column), type='date')
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title,
                        size_max=15,
                        labels={x_column: format_measure_name(x_column),
                                y_column: format_measure_name(y_column)})
    elif chart_type == "heatmap":
        if y_column:
            # Pivot table heatmap
            pivot_df = df.pivot_table(index=x_column, columns=y_column, aggfunc='size', fill_value=0)

            # Sort heatmap: x_column ascending (index), y_column descending (columns)
            # Default behavior, can be overridden
            if not sort_by or sort_by == x_column:
                # Sort index (x_column) - default ascending
                x_sort_order = sort_order if sort_order else 'ascending'
                pivot_df = pivot_df.sort_index(ascending=(x_sort_order == 'ascending'))

            # Sort columns (y_column) descending by default
            if y_column:
                # For columns, we always want descending unless explicitly overridden
                pivot_df = pivot_df.sort_index(axis=1, ascending=False)
        else:
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=['number']).columns
            pivot_df = df[numeric_cols].corr()

            # Sort correlation heatmap indices
            if sort_order:
                pivot_df = pivot_df.sort_index(ascending=(sort_order == 'ascending'))
                pivot_df = pivot_df.sort_index(axis=1, ascending=(sort_order == 'ascending'))

        fig = px.imshow(pivot_df, title=title, aspect="auto")
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    # Check if x-axis labels are long and need rotation
    if chart_type in ["bar", "line", "scatter"]:
        # Use the appropriate dataframe based on chart type
        check_df = sorted_df if chart_type == "line" else df
        x_labels = check_df[x_column].astype(str).unique()
        max_label_length = max([len(str(label)) for label in x_labels]) if len(x_labels) > 0 else 0

        if max_label_length > 10 or len(x_labels) > 8:
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
    
    # Apply consistent styling
    # Determine if we should show legend
    show_legend = bool(color_column or (chart_type == "line" and isinstance(y_column, list)))

    # For line charts with multiple measures, use horizontal legend between title and plot
    if chart_type == "line" and isinstance(y_column, list):
        legend_config = dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,  # Position at top of plot area (below title)
            xanchor="center",
            x=0.5
        )
        margin_config = dict(b=100, t=120, l=60, r=60)  # Extra top margin for title and legend space
    else:
        legend_config = dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        )
        margin_config = dict(b=100, t=60, l=60, r=200)  # Extra right margin for legend

    fig.update_layout(
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=margin_config,
        showlegend=show_legend,
        legend=legend_config,
        width=width,
        height=height
    )
    
    return fig


def save_image_to_tmp(image_bytes: bytes, chart_id: str, format: str) -> Optional[str]:
    """Save image bytes to temporary directory and return file path."""
    try:
        image_filename = f"{chart_id}.{format}"
        image_file_path = TMP_DIR / image_filename
        
        with open(image_file_path, 'wb') as f:
            f.write(image_bytes)
        
        logger.debug(f"Saved {format.upper()} chart to: {image_file_path}")
        return str(image_file_path)
    except Exception as e:
        logger.warning(f"Failed to save {format.upper()} file: {e}")
        return None
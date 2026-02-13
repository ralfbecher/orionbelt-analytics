"""Generate a test chart with monthly revenue data to verify month sorting."""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chart_utils import create_plotly_chart

# Test data with months in your example format
data = [
    {"month": "Jan", "revenue": 35000},
    {"month": "Feb", "revenue": 42000},
    {"month": "Mar", "revenue": 38000},
    {"month": "Apr", "revenue": 51000},
    {"month": "May", "revenue": 48000},
    {"month": "Jun", "revenue": 55000},
    {"month": "Jul", "revenue": 62000},
    {"month": "Aug", "revenue": 58000},
    {"month": "Sep", "revenue": 67000},
    {"month": "Oct", "revenue": 71000},
    {"month": "Nov", "revenue": 69000},
    {"month": "Dec", "revenue": 78000}
]

# Create DataFrame
df = pd.DataFrame(data)

print("Generating Monthly Revenue Trend chart...")
print(f"Data includes {len(df)} months from {df['month'].iloc[0]} to {df['month'].iloc[-1]}")
print()

# Create the chart
fig = create_plotly_chart(
    df=df,
    chart_type="line",
    x_column="month",
    y_column="revenue",
    color_column=None,
    title="Monthly Revenue Trend",
    chart_style=None,
    width=800,
    height=600
)

# Save as HTML (interactive)
html_path = Path(__file__).parent / "monthly_revenue_chart.html"
fig.write_html(str(html_path))
print(f"✓ Saved interactive chart to: {html_path}")

# Try to save as PNG (static image) if kaleido is available
try:
    png_path = Path(__file__).parent / "monthly_revenue_chart.png"
    fig.write_image(str(png_path))
    print(f"✓ Saved static image to: {png_path}")
except Exception as e:
    print(f"  Note: PNG export not available (kaleido not installed)")

print()
print("Chart generation complete!")
print(f"X-axis order: {list(fig.data[0].x)}")
print()
print(f"Open the HTML file in your browser to view the chart:")

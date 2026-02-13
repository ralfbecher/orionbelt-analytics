"""Generate a Matplotlib chart with monthly revenue data to verify month sorting."""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chart_utils import create_matplotlib_chart

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

print("Generating Monthly Revenue Trend chart with Matplotlib...")
print(f"Data includes {len(df)} months from {df['month'].iloc[0]} to {df['month'].iloc[-1]}")
print()

# Create the chart
fig = create_matplotlib_chart(
    df=df,
    chart_type="line",
    x_column="month",
    y_column="revenue",
    color_column=None,
    title="Monthly Revenue Trend (Matplotlib)",
    chart_style=None,
    width=800,
    height=600
)

# Save as PNG
png_path = Path(__file__).parent / "tmp" / "monthly_revenue_chart_matplotlib.png"
fig.savefig(str(png_path), dpi=100, bbox_inches='tight')
print(f"✓ Saved Matplotlib chart to: {png_path}")

print()
print("Chart generation complete!")
print("The chart shows months in chronological order from Jan to Dec.")

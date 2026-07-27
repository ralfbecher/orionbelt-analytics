"""Row serialization utilities for OrionBelt Analytics.

Extracts the duplicated row-to-dict serialization logic that was present
in both sample_table_data() and execute_sql_query() within database_manager.py.
"""

import decimal
from collections.abc import Sequence
from typing import Any


def serialize_row(row: Sequence[Any], columns: list[str]) -> dict[str, Any]:
    """Serialize a database row into a JSON-compatible dict.

    Handles conversion of non-serializable types:
    - Decimal -> float
    - datetime/date -> ISO format string
    - bytes/bytearray -> hex string (sample) or size marker (query)
    - Complex objects -> str()
    - dict/list -> pass through (JSON/array types)
    - None -> None

    Args:
        row: Sequence of column values from database result
        columns: Column names in order matching row values

    Returns:
        Dictionary mapping column names to serialized values
    """
    row_dict: dict[str, Any] = {}
    for i, value in enumerate(row):
        column_name = columns[i]
        if value is None:
            row_dict[column_name] = None
        elif isinstance(value, decimal.Decimal):
            row_dict[column_name] = float(value)
        elif hasattr(value, "isoformat"):
            row_dict[column_name] = value.isoformat()
        elif isinstance(value, (bytes, bytearray)):
            row_dict[column_name] = value.hex()
        elif isinstance(value, (dict, list)):
            row_dict[column_name] = value
        elif hasattr(value, "__dict__"):
            row_dict[column_name] = str(value)
        else:
            row_dict[column_name] = value
    return row_dict


def serialize_rows(
    rows: Sequence[Sequence[Any]], columns: list[str]
) -> list[dict[str, Any]]:
    """Serialize multiple database rows into JSON-compatible dicts.

    Args:
        rows: Sequence of row tuples from database result
        columns: Column names in order

    Returns:
        List of dictionaries with serialized values
    """
    return [serialize_row(row, columns) for row in rows]

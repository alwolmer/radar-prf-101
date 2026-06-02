from __future__ import annotations

from datetime import date, datetime
from typing import Any


def format_number_pt_br(value: Any, decimals: int = 0) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if decimals == 0:
        rendered = f"{numeric:,.0f}"
    else:
        rendered = f"{numeric:,.{decimals}f}"
    return rendered.replace(",", "X").replace(".", ",").replace("X", ".")


def format_date_pt_br(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.strftime("%d/%m/%Y")
    parsed = datetime.fromisoformat(str(value)).date()
    return parsed.strftime("%d/%m/%Y")


def format_period_pt_br(start: Any, end: Any | None = None) -> str:
    if end is None:
        return format_date_pt_br(start)
    return f"{format_date_pt_br(start)} a {format_date_pt_br(end)}"

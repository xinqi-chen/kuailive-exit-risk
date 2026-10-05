from __future__ import annotations

import pandas as pd


def normalize_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or not text.isdecimal():
        raise ValueError(f"non-integer entity ID: {text!r}")
    return str(int(text))


def timestamp_ms(series: pd.Series, *, name: str) -> pd.Series:
    """Parse a single known timestamp column. Naive released timestamps assume UTC."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    elif pd.api.types.is_integer_dtype(series.dtype):
        values = series.dropna()
        if not values.empty and not values.between(1_700_000_000_000, 1_900_000_000_000).all():
            raise ValueError(f"{name}: integer timestamp is not plausible epoch milliseconds")
        return series.astype("Int64")
    else:
        text = series.astype("string")
        numeric = text.dropna().str.fullmatch(r"\d+").all()
        if numeric:
            return timestamp_ms(pd.to_numeric(text, errors="coerce").astype("Int64"), name=name)
        parsed = pd.to_datetime(text, utc=True, errors="coerce", format="mixed")
    return (parsed.dt.as_unit("ns").astype("int64") // 1_000_000).where(parsed.notna(), pd.NA).astype("Int64")

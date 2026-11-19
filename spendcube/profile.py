"""Quick look at an unfamiliar file, to write its config: fill rate, distinct values, formats, samples."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .ingest import read_csv_text


def _pattern(value: str) -> str:
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"\d+", "9", value))


def profile(path: str, sep: str = ",", sheet=None, top: int = 4) -> pd.DataFrame:
    df = read_csv_text(Path(path), sep=sep, sheet=sheet)
    rows = []
    for col in df.columns:
        values = df[col].dropna()
        patterns = values.map(_pattern).value_counts()
        rows.append({
            "column": col,
            "filled": f"{len(values) / max(len(df), 1):.0%}",
            "distinct": values.nunique(),
            "formats": "  ".join(f"{p}({n})" for p, n in patterns.head(3).items()),
            "top values": "  |  ".join(f"{v[:30]}({n})" for v, n in values.value_counts().head(top).items()),
        })
    print(f"{path}: {len(df)} rows, {len(df.columns)} columns")
    return pd.DataFrame(rows)

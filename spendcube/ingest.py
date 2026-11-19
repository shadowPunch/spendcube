"""Read source files and map them onto the canonical record layout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CANONICAL_FIELDS, CompanyConfig, SourceConfig


def read_csv_text(path: Path, sep: str = ",", encoding: str = "utf-8-sig", sheet=None) -> pd.DataFrame:
    """Read every column as text; blank or whitespace-only cells become missing."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet or 0, dtype=str, keep_default_na=False, na_values=[""])
    else:
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, na_values=[""], encoding=encoding)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    return df


def _column(raw: pd.DataFrame, spec: str | list[str], join: bool = False) -> pd.Series:
    if isinstance(spec, str):
        return raw[spec]
    if join:
        return raw[spec].fillna("").agg("-".join, axis=1)
    out = raw[spec[0]]
    for col in spec[1:]:
        out = out.fillna(raw[col])
    return out


def read_source(name: str, src: SourceConfig, data_dir: Path) -> pd.DataFrame:
    raw = read_csv_text(data_dir / src.file, src.sep, src.encoding, src.sheet)
    missing = src.source_columns() - set(raw.columns)
    if missing:
        raise ValueError(f"{src.file}: columns not found: {sorted(missing)}; file has {list(raw.columns)}")

    out = pd.DataFrame(index=raw.index)
    for field in CANONICAL_FIELDS:
        out[field] = _column(raw, src.columns[field], join=field == "record_id") if field in src.columns else pd.NA
    for col in src.context:
        out[col] = raw[col]
    if "currency" not in src.columns and src.currency:
        out["currency"] = src.currency
    out["is_credit"] = (raw[src.credit.column].str.upper().isin([v.upper() for v in src.credit.values]).fillna(False)
                        if src.credit else False)

    out["record_id"] = src.id_prefix + out["record_id"].astype("string")
    out["source"] = name
    # Rows identical on the duplicate columns are the same record exported twice.
    id_cols = [src.columns["record_id"]] if isinstance(src.columns["record_id"], str) else src.columns["record_id"]
    if not src.exact_duplicates:
        out["row_hash"] = range(len(raw))  # every row unique
    else:
        compare = raw[src.duplicate_columns] if src.duplicate_columns else raw.drop(columns=id_cols)
        out["row_hash"] = pd.util.hash_pandas_object(compare, index=False).values
    return out


def ingest(cfg: CompanyConfig, data_dir: Path) -> pd.DataFrame:
    frames = [read_source(name, src, data_dir) for name, src in cfg.sources.items()]
    df = pd.concat(frames, ignore_index=True)
    if df["record_id"].duplicated().any():  # keep going: every input row still gets an output row
        dupes = df.loc[df["record_id"].duplicated(), "record_id"].head(3).tolist()
        print(f"[ingest] WARNING: {df['record_id'].duplicated().sum()} repeated record IDs, e.g. {dupes}")
    df["data_flags"] = ""
    return df


def add_flag(df: pd.DataFrame, mask, flag: str) -> None:
    """Record a data-quality flag on the selected rows (flags are ';'-separated)."""
    df.loc[mask, "data_flags"] = df.loc[mask, "data_flags"] + flag + ";"

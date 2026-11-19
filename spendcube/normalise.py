"""Parse amounts, tax rates and dates; convert spend to the reporting currency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CompanyConfig
from .ingest import add_flag, read_csv_text


def parse_amount(values: pd.Series, decimal: str = ".") -> pd.Series:
    """'5,266.63', '-1,234.5', '(12.00)', 'R 99' -> float. Unparseable -> NaN."""
    text = values.astype("string").str.strip()
    negative = text.str.contains(r"^\(.*\)$|^-|-$", regex=True).fillna(False)
    text = text.str.replace(r"[^\d,.]", "", regex=True)
    if decimal == ",":
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        text = text.str.replace(",", "", regex=False)
    amount = pd.to_numeric(text, errors="coerce").astype(float)
    return amount.where(~negative, -amount)


def parse_rate(values: pd.Series) -> pd.Series:
    """Tax rate as a fraction from '15', '15%', '0.15' or '0'."""
    text = values.astype("string").str.replace("%", "", regex=False).str.strip()
    rate = pd.to_numeric(text, errors="coerce").astype(float)
    return rate.where(rate <= 1, rate / 100)


def parse_dates(values: pd.Series, formats: list[str]) -> pd.Series:
    """Try each configured format in order; the first that parses wins."""
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    text = values.astype("string")
    for fmt in formats:
        todo = parsed.isna() & text.notna()
        if not todo.any():
            break
        parsed[todo] = pd.to_datetime(text[todo], format=fmt, errors="coerce")
    return parsed


def _source_rows(df: pd.DataFrame, cfg: CompanyConfig):
    for name, src in cfg.sources.items():
        yield src, (df["source"] == name)


def normalise(df: pd.DataFrame, cfg: CompanyConfig, data_dir: Path, settings: dict) -> pd.DataFrame:
    df = df.copy()
    df["amount_num"] = np.nan
    df["net_amount"] = np.nan
    df["date_parsed"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    df["fallback_parsed"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    df["fx_line_rate"] = pd.to_numeric(df["fx_rate"].astype("string").str.replace(",", ".", regex=False),
                                       errors="coerce").astype(float)
    for src, rows in _source_rows(df, cfg):
        amount = parse_amount(df.loc[rows, "amount"], src.decimal)
        credit = df.loc[rows, "is_credit"].astype(bool)
        amount = amount.where(~credit, -amount.abs())
        df.loc[rows, "amount_num"] = amount
        if src.amount_is == "gross" and "tax_amount" in src.columns:
            tax = parse_amount(df.loc[rows, "tax_amount"], src.decimal).fillna(0.0)
            df.loc[rows, "net_amount"] = amount - tax.where(~credit, -tax.abs())
        elif src.amount_is == "gross":
            rate = parse_rate(df.loc[rows, "tax_rate"]).fillna(src.default_tax_rate)
            df.loc[rows, "net_amount"] = amount / (1 + rate)
        else:
            df.loc[rows, "net_amount"] = amount
        if src.rate_direction == "reporting_per_unit":
            df.loc[rows, "fx_line_rate"] = 1 / df.loc[rows, "fx_line_rate"]
        df.loc[rows, "date_parsed"] = parse_dates(df.loc[rows, "date"], src.date_formats)
        df.loc[rows, "fallback_parsed"] = parse_dates(df.loc[rows, "fallback_date"], src.date_formats)

    add_flag(df, df["amount_num"].isna(), "AMOUNT_UNPARSEABLE")
    add_flag(df, df["date"].notna() & df["date_parsed"].isna(), "DATE_UNPARSEABLE")

    # A date far from the record's fallback date (e.g. an invoice dated 2018 posted in 2024)
    # is kept in the data but not trusted for currency conversion.
    gap = (df["date_parsed"] - df["fallback_parsed"]).abs().dt.days
    implausible = gap > settings["max_date_gap_days"]
    add_flag(df, implausible, "DATE_IMPLAUSIBLE")
    df["effective_date"] = df["date_parsed"].where(~implausible).fillna(df["fallback_parsed"])
    add_flag(df, df["effective_date"].isna(), "DATE_MISSING")

    df["quantity_num"] = parse_amount(df["quantity"])
    df = convert_currency(df, cfg, data_dir, settings)
    df["spend_value"] = df["spend_value"].fillna(0.0)
    return df


def load_fx(cfg: CompanyConfig, data_dir: Path) -> pd.DataFrame:
    fx = read_csv_text(data_dir / cfg.fx.file)
    table = pd.DataFrame({
        "currency": fx[cfg.fx.currency_column].str.upper(),
        "month": pd.to_datetime(fx[cfg.fx.month_column], format="%Y-%m").astype("datetime64[ns]"),
        "table_rate": pd.to_numeric(fx[cfg.fx.rate_column], errors="coerce"),
    })
    if cfg.fx.rate_direction == "reporting_per_unit":
        table["table_rate"] = 1 / table["table_rate"]
    return table.dropna().sort_values("month")


def convert_currency(df: pd.DataFrame, cfg: CompanyConfig, data_dir: Path, settings: dict) -> pd.DataFrame:
    """Spend in the reporting currency.

    Order: reporting-currency lines as they are; else the line's own rate if it is
    plausible; else the FX table for the month (nearest month if that one is missing).
    Rates are units of line currency per one unit of reporting currency.
    """
    currency = df["currency"].astype("string").str.upper().fillna(cfg.reporting_currency)
    is_reporting = currency == cfg.reporting_currency
    line_rate = df["fx_line_rate"]
    table_rate = pd.Series(np.nan, index=df.index)

    if cfg.fx is not None:
        fx = load_fx(cfg, data_dir)
        month = df["effective_date"].dt.to_period("M").dt.to_timestamp().astype("datetime64[ns]")
        lookup = pd.DataFrame({"currency": currency.astype(str), "month": month, "row": df.index})
        fx["currency"] = fx["currency"].astype(str)
        dated = lookup.dropna(subset=["month"]).sort_values("month")
        merged = pd.merge_asof(dated, fx, on="month", by="currency", direction="nearest")
        table_rate.loc[merged["row"].to_numpy()] = merged["table_rate"].to_numpy()
        # Records without any usable date fall back to the currency's median rate.
        median = fx.groupby("currency")["table_rate"].median()
        undated = table_rate.isna() & lookup["month"].isna()
        table_rate[undated] = currency[undated].map(median)

    ratio = line_rate / table_rate
    max_ratio = settings["max_fx_ratio"]
    plausible = (line_rate > 0) & (table_rate.isna() | ratio.between(1 / max_ratio, max_ratio))
    rate = pd.Series(np.where(plausible, line_rate, table_rate), index=df.index)
    rate[is_reporting] = 1.0

    non_reporting = ~is_reporting
    add_flag(df, non_reporting & line_rate.notna() & ~plausible, "FX_RATE_REPLACED")
    add_flag(df, non_reporting & line_rate.isna() & rate.notna(), "FX_FROM_TABLE")
    add_flag(df, non_reporting & rate.isna(), "FX_MISSING")

    df["currency_clean"] = currency
    df["fx_used"] = rate
    df["spend_value"] = df["net_amount"] / rate
    return df

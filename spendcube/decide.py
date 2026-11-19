"""Decide what counts as spend and give a reason code for everything excluded.

Rules apply in a fixed order and the first that fires sets the code:
status (CANCELLED, NOT_APPROVED) -> DUPLICATE -> company rules -> generic terms.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import CompanyConfig

CONFIDENCE = {"status": 0.99, "exact_duplicate": 0.99, "rule": 0.95, "term": 0.85}


def status_codes(df: pd.DataFrame, cfg: CompanyConfig) -> pd.Series:
    """CANCELLED / NOT_APPROVED from each source's status rules; NA when approved or no status."""
    code = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, src in cfg.sources.items():
        if src.status is None:
            continue
        rows = df["source"] == name
        status = df.loc[rows, "status"].astype("string").str.strip().str.lower()
        approved = status.isin([s.lower() for s in src.status.approved])
        cancelled = status.isin([s.lower() for s in src.status.cancelled])
        code[rows & cancelled.reindex(df.index, fill_value=False)] = "CANCELLED"
        not_ok = status.notna() & ~approved & ~cancelled
        code[rows & not_ok.reindex(df.index, fill_value=False)] = "NOT_APPROVED"
    return code


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[name].astype("string")


def rule_codes(df: pd.DataFrame, cfg: CompanyConfig) -> pd.Series:
    code = pd.Series(pd.NA, index=df.index, dtype="string")
    for rule in cfg.exclusion_rules:
        hit = _column(df, rule.column).str.contains(rule.pattern, flags=re.IGNORECASE, regex=True).fillna(False)
        code[hit & code.isna()] = rule.reason
    return code


def term_hits(df: pd.DataFrame, terms: dict, text_columns: list[str]) -> pd.Series:
    """Generic non-procurement terms (taxes, intercompany, financing) in any text column.

    Each group has `any` phrases and `unless` phrases; e.g. 'sales tax' is not a hit
    when the line also says 'advisory', because tax advice is a real purchase.
    """
    text = pd.Series("", index=df.index, dtype="string")
    for col in text_columns:
        text = text + " " + _column(df, col).fillna("")
    text = " " + text.str.lower().str.replace(r"\s+", " ", regex=True) + " "
    hit = pd.Series(False, index=df.index)
    for group in terms["groups"]:
        any_hit = np.logical_or.reduce([text.str.contains(p, regex=False).fillna(False) for p in group["any"]])
        unless = np.logical_or.reduce([text.str.contains(p, regex=False).fillna(False)
                                       for p in group.get("unless", [])] or [np.zeros(len(df), bool)])
        hit |= pd.Series(np.asarray(any_hit, bool) & ~np.asarray(unless, bool), index=df.index)
    supplier = " " + _column(df, "supplier_name").fillna("").str.lower() + " "
    for p in terms.get("supplier_any", []):
        hit |= supplier.str.contains(p, regex=False).fillna(False)
    return hit


def decide(df: pd.DataFrame, cfg: CompanyConfig, terms: dict, duplicate_of: pd.Series,
           duplicate_confidence: pd.Series) -> pd.DataFrame:
    """Add include, exclusion_reason and exclusion_confidence columns."""
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    conf = pd.Series(np.nan, index=df.index)

    status = status_codes(df, cfg)
    reason = reason.fillna(status)
    conf[status.notna()] = CONFIDENCE["status"]

    dup = reason.isna() & duplicate_of.notna()
    reason[dup] = "DUPLICATE"
    conf[dup] = duplicate_confidence[dup]

    rules = rule_codes(df, cfg)
    hit = reason.isna() & rules.notna()
    reason[hit] = rules[hit]
    conf[hit] = CONFIDENCE["rule"]

    text_columns = ["description", *cfg.classification_context]
    terms_hit = reason.isna() & term_hits(df, terms, text_columns)
    reason[terms_hit] = "NON_PROCUREMENT"
    conf[terms_hit] = CONFIDENCE["term"]

    out = df.copy()
    out["exclusion_reason"] = reason
    out["include"] = reason.isna().astype(int)
    out["exclusion_confidence"] = conf
    return out

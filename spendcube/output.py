"""Build and check the output file: one row per input record, in the template's columns."""

from __future__ import annotations

import pandas as pd

from .config import EXCLUSION_CODES
from .taxonomy import Taxonomy

COLUMNS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
           "l1", "l2", "l3", "l4", "confidence"]


def build_output(df: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """df needs: record_id, include, exclusion_reason, supplier_key, item_key, spend_value,
    leaf_id, confidence (classification), exclusion_confidence."""
    included = df["include"] == 1
    out = pd.DataFrame({
        "record_id": df["record_id"],
        "include": df["include"].astype(int),
        "exclusion_reason": df["exclusion_reason"].fillna(""),
        "supplier_key": df.get("supplier_key_out", df["supplier_key"]),
        "item_key": df.get("item_key_out", df["item_key"]),
        "spend": df["spend_value"].where(included, 0.0).round(2),
    })
    levels = taxonomy.level_columns(df["leaf_id"].where(included))
    out = out.join(levels)
    out["confidence"] = df["confidence"].where(included, df["exclusion_confidence"]).fillna(0.0).clip(0, 1).round(3)
    return out[COLUMNS]


def check_output(out: pd.DataFrame, n_records: int, taxonomy: Taxonomy) -> list[str]:
    """Contract checks; returns a list of problems (empty when the file is valid)."""
    problems = []
    if len(out) != n_records:
        problems.append(f"{len(out)} rows for {n_records} input records")
    if out["record_id"].duplicated().any():
        problems.append("duplicate record_id values")
    bad_codes = set(out["exclusion_reason"]) - set(EXCLUSION_CODES) - {""}
    if bad_codes:
        problems.append(f"unknown exclusion codes {bad_codes}")
    if ((out["include"] == 0) & (out["exclusion_reason"] == "")).any():
        problems.append("excluded rows without a reason")
    if ((out["include"] == 0) & (out["spend"] != 0)).any():
        problems.append("excluded rows with non-zero spend")
    inc = out[out["include"] == 1]
    paths = {leaf.path for leaf in taxonomy.leaves}
    observed = inc[["l1", "l2", "l3", "l4"]].apply(lambda r: tuple(v for v in r if v), axis=1)
    unknown = set(observed) - paths
    if unknown:
        problems.append(f"{len(unknown)} category paths not in the tree, e.g. {next(iter(unknown))}")
    if not out["confidence"].between(0, 1).all():
        problems.append("confidence outside 0-1")
    return problems

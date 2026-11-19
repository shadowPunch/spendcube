"""Count each purchase once across two systems.

1. Collapse re-exports inside the secondary system (same invoice exported twice).
2. Match secondary records to primary records one-to-one, in tiers of decreasing
   strictness. Amounts must agree to within a cent: near-amounts on the same PO in
   this data are separate deliveries, not the same one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .text import desc_key, invoice_core

TIER_CONFIDENCE = {1: 0.99, 2: 0.97, 3: 0.9}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["po"] = df["po_ref"].astype("string").str.extract(r"(\d{6,})", expand=False)
    line = df["po_line"].astype("string").str.extract(r"(\d+)", expand=False)
    # A PO reference like '4510000018/10.0' carries the PO line after the slash.
    from_ref = df["po_ref"].astype("string").str.extract(r"\d{6,}\s*/\s*(\d+)", expand=False)
    out["po_line"] = line.fillna(from_ref).str.lstrip("0")
    out["cents"] = (df["net_amount"] * 100).round().astype("Int64")
    out["qty"] = df["quantity_num"].round(4)
    out["supplier"] = df["supplier_key"]
    out["text"] = df["description"].map(desc_key)
    return out


def collapse_reexports(df: pd.DataFrame, approved: pd.Series, min_similarity: float) -> pd.Series:
    """Return, for each re-exported record, the index of the record it duplicates (else NA).

    Same supplier, invoice number, net amount and quantity, with similar text (OCR
    varies between exports). An approved record is preferred as the survivor.
    """
    p = _prepare(df)
    p["invoice"] = df["invoice_ref"].map(invoice_core)
    p["approved"] = approved
    p["position"] = df.index
    p["record_id"] = df["record_id"]
    dup_of = pd.Series(pd.NA, index=df.index, dtype="Int64")
    keys = ["supplier", "invoice", "cents", "qty"]
    candidates = p[p.duplicated(keys, keep=False) & (p["invoice"] != "")]
    # The lowest record ID survives, so the result does not depend on file order.
    order = ["approved", "record_id", "position"]
    for _, grp in candidates.sort_values(order, ascending=[False, True, True]).groupby(keys):
        survivors: list[tuple[int, str]] = []  # (index, text)
        for idx, row in grp.iterrows():
            twin = next((rid for rid, text in survivors
                         if fuzz.ratio(text, row["text"]) >= min_similarity), None)
            if twin is None:
                survivors.append((idx, row["text"]))
            else:
                dup_of[idx] = twin
    return dup_of


def match_systems(primary: pd.DataFrame, secondary: pd.DataFrame, min_text_similarity: float) -> pd.DataFrame:
    """One-to-one matches of secondary records to primary records.

    Tier 1: same PO, PO line, quantity and net amount.
    Tier 2: same PO and net amount.
    Tier 3: same supplier and net amount with similar text (PO missing or mistyped).
    Returns columns: secondary_index, primary_index, tier, cent_diff, similarity.
    """
    a, b = _prepare(primary), _prepare(secondary)
    a["primary_index"], b["secondary_index"] = a.index, b.index
    b = b.dropna(subset=["cents"])
    # Recomputing net from gross can be a cent off, so also try +/- 1 cent.
    shifted = pd.concat([b.assign(cent_diff=d, cents=b["cents"] + d) for d in (0, -1, 1)])

    tiers = [
        (1, ["po", "po_line", "cents", "qty"]),
        (2, ["po", "cents"]),
        (3, ["supplier", "cents"]),
    ]
    found = []
    for tier, keys in tiers:
        m = shifted.dropna(subset=keys).merge(a.dropna(subset=keys), on=keys, suffixes=("_s", "_p"))
        if m.empty:
            continue
        m["similarity"] = [fuzz.ratio(x, y) for x, y in zip(m["text_s"], m["text_p"])]
        if tier == 3:
            m = m[m["similarity"] >= min_text_similarity]
        m["tier"] = tier
        found.append(m[["secondary_index", "primary_index", "tier", "cent_diff", "similarity"]])
    if not found:
        return pd.DataFrame(columns=["secondary_index", "primary_index", "tier", "cent_diff", "similarity"])

    cand = pd.concat(found, ignore_index=True)
    cand["abs_cent"] = cand["cent_diff"].abs()
    cand = cand.sort_values(["tier", "abs_cent", "similarity", "primary_index"],
                            ascending=[True, True, False, True])
    used_p, used_s, rows = set(), set(), []
    for row in cand.itertuples(index=False):
        if row.primary_index in used_p or row.secondary_index in used_s:
            continue
        used_p.add(row.primary_index)
        used_s.add(row.secondary_index)
        rows.append(row)
    result = pd.DataFrame(rows).drop(columns="abs_cent")
    result["confidence"] = result["tier"].map(TIER_CONFIDENCE)
    return result.astype({"secondary_index": np.int64, "primary_index": np.int64})

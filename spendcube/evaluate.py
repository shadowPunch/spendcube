"""Measure the pipeline against labels, and test the matcher on synthetic corruptions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dedupe import supplier_keys
from .ingest import read_csv_text
from .config import CompanyConfig
from .match import match_systems
from .text import desc_key


def load_labels(cfg: CompanyConfig, data_dir) -> pd.DataFrame | None:
    if cfg.labels is None:
        return None
    lab = read_csv_text(data_dir / cfg.labels.file)
    out = pd.DataFrame({"record_id": cfg.labels.id_prefix + lab[cfg.labels.id_column]})
    for i, col in enumerate(cfg.labels.category_columns, start=1):
        out[f"gold_l{i}"] = lab[col]
    for name in ("supplier_key", "item_key", "include", "reason"):
        col = getattr(cfg.labels, f"{name}_column")
        if col:
            out[f"gold_{name}"] = lab[col]
    return out


def _pairs(counts: pd.Series) -> float:
    return float((counts * (counts - 1) / 2).sum())


def pairwise_scores(pred: pd.Series, gold: pd.Series) -> dict:
    """Pairwise precision/recall/F1 of a grouping: do records that belong together end up together?"""
    frame = pd.DataFrame({"p": pred.to_numpy(), "g": gold.to_numpy()}).dropna()
    tp = _pairs(frame.groupby(["p", "g"]).size())
    pred_pairs, gold_pairs = _pairs(frame.groupby("p").size()), _pairs(frame.groupby("g").size())
    precision = tp / pred_pairs if pred_pairs else 1.0
    recall = tp / gold_pairs if gold_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n": len(frame)}


def grouping_report(df: pd.DataFrame, labels: pd.DataFrame, key: str) -> dict:
    """Pairwise scores for supplier_key or item_key, with and without the largest gold group.

    In A's labels the largest supplier group and item group are catch-alls (about 120
    unrelated suppliers under one key), so the score without them is also reported.
    """
    gold_col = f"gold_{key}"
    if labels is None or gold_col not in labels:
        return {}
    column = f"{key}_out" if f"{key}_out" in df else key  # score the keys that go in the output
    m = df[["record_id", column]].rename(columns={column: key}).merge(
        labels[["record_id", gold_col]], on="record_id").dropna()
    report = {f"{key}_{k}": v for k, v in pairwise_scores(m[key], m[gold_col]).items()}
    biggest = m[gold_col].value_counts().index[0]
    rest = m[m[gold_col] != biggest]
    report.update({f"{key}_excl_largest_{k}": v for k, v in pairwise_scores(rest[key], rest[gold_col]).items()})
    return report


def classification_report(df: pd.DataFrame, labels: pd.DataFrame, levels: int) -> dict:
    """Accuracy per tree level on included, labelled lines, by line count and by spend."""
    if labels is None or "gold_l1" not in labels:
        return {}
    m = df.merge(labels, on="record_id")
    m = m[(m["include"] == 1) & m["gold_l1"].notna()]
    if m.empty:
        return {}
    weight = m["spend_value"].abs()
    report = {"cls_n": len(m)}
    correct = pd.Series(True, index=m.index)
    for level in range(1, levels + 1):
        gold, pred = m[f"gold_l{level}"].fillna(""), m[f"l{level}"].fillna("")
        correct &= gold == pred  # a level counts only if every level above it is right too
        report[f"acc_l{level}"] = float(correct.mean())
        report[f"acc_l{level}_spend"] = float((correct * weight).sum() / weight.sum())
    return report


def exclusion_report(df: pd.DataFrame, labels: pd.DataFrame) -> dict:
    if labels is None or "gold_include" not in labels:
        return {}
    m = df[["record_id", "include"]].merge(labels[["record_id", "gold_include"]], on="record_id")
    gold, pred = m["gold_include"].astype(int), m["include"].astype(int)
    tp_ex = int(((pred == 0) & (gold == 0)).sum())
    return {
        "excl_accuracy": float((gold == pred).mean()),
        "excl_precision": tp_ex / max(int((pred == 0).sum()), 1),
        "excl_recall": tp_ex / max(int((gold == 0).sum()), 1),
        "excl_n": len(m),
    }


# --- synthetic matching test -------------------------------------------------

_OCR = {"S": "5", "O": "0", "I": "1", "B": "8"}


def _corrupt_text(text: str, rng: np.random.Generator) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in _OCR and rng.random() < 0.08:
            chars[i] = _OCR[ch]
    if len(chars) > 10 and rng.random() < 0.3:
        del chars[rng.integers(len(chars))]
    return "".join(chars).replace(" ", "", 1) if rng.random() < 0.2 else "".join(chars)


def _corrupt_name(name: str, rng: np.random.Generator) -> str:
    roll = rng.random()
    if roll < 0.2 and len(name) > 6:
        cut = int(rng.integers(2, len(name) - 2))
        return name[:cut] + " " + name[cut:]
    if roll < 0.4:
        return name.title()
    if roll < 0.6:
        return name.replace("(PTY) LTD", "").strip()
    return name


def synthetic_match_test(records: pd.DataFrame, primary: str, settings: dict, n: int = 3000,
                         seed: int = 0) -> dict:
    """Make noisy invoice copies of real primary records and check the matcher finds them.

    Noise mirrors what A's invoice tool shows: PO missing or reformatted, OCR errors in
    text, supplier typed differently, gross amounts. A match counts as correct when it
    lands on the source record or on an identical record (same PO, line and amount).
    """
    rng = np.random.default_rng(seed)
    prim = records[records["source"] == primary]
    sample = prim.sample(min(n, len(prim)), random_state=seed)
    fake = pd.DataFrame(index=pd.RangeIndex(len(sample)) + records.index.max() + 1)
    fake["record_id"] = "FAKE:" + sample["record_id"].to_numpy()
    po = sample["po_ref"].astype("string").to_numpy()
    roll = rng.random(len(sample))
    fake["po_ref"] = np.where(roll < 0.18, None, np.where(roll < 0.4, "PO#" + po, po))
    fake["po_line"] = None
    fake["description"] = [_corrupt_text(t, rng) if isinstance(t, str) else t for t in sample["description"]]
    fake["supplier_name"] = [_corrupt_name(s, rng) if isinstance(s, str) else s for s in sample["supplier_name"]]
    fake["supplier_id"] = None
    rate = np.where(rng.random(len(sample)) < 0.05, 0.0, 0.15)
    gross = (sample["net_amount"].to_numpy() * (1 + rate)).round(2)
    fake["net_amount"] = gross / (1 + rate)
    fake["quantity_num"] = sample["quantity_num"].to_numpy()

    combined = pd.concat([prim[fake.columns.drop("record_id").tolist() + ["record_id"]], fake])
    combined["supplier_key"] = supplier_keys(combined, settings)
    found = match_systems(combined.loc[prim.index], combined.loc[fake.index], settings["match_min_text_similarity"])

    truth = pd.Series(sample.index.to_numpy(), index=fake.index)
    amount = prim["net_amount"].round(2).astype(str)
    # Same receipt: same PO, line and amount. Equivalent purchase: same supplier, amount and text
    # (counting, spend and category come out the same whichever twin is matched).
    same = prim["po_ref"].astype(str) + "|" + prim["po_line"].astype(str) + "|" + amount
    equivalent = combined.loc[prim.index, "supplier_key"] + "|" + amount + "|" + prim["description"].map(desc_key)
    hit, true = found["primary_index"], truth.loc[found["secondary_index"]]
    same_hit = same.loc[hit].to_numpy() == same.loc[true].to_numpy()
    correct = int(same_hit.sum())
    equiv = int((same_hit | (equivalent.loc[hit].to_numpy() == equivalent.loc[true].to_numpy())).sum())
    return {
        "synthetic_n": len(fake),
        "synthetic_recall": correct / len(fake),
        "synthetic_precision": correct / max(len(found), 1),
        "synthetic_equivalent_precision": equiv / max(len(found), 1),
        "synthetic_unmatched": len(fake) - len(found),
        **{f"synthetic_tier{t}": int(c) for t, c in found["tier"].value_counts().items()},
    }

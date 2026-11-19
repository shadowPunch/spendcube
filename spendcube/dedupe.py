"""Supplier and item de-duplication.

Both work the same way: records are linked through shared identifiers (an ID, a
normalised name or description, a non-generic item code), the links are closed
with union-find, and each connected component becomes one key.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .text import desc_key, numbers, supplier_norm


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, node: str) -> None:
        self.parent.setdefault(node, node)

    def find(self, node: str) -> str:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, a: str, b: str) -> None:
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _labels(groups: pd.Series, display: pd.Series, prefix: str) -> pd.Series:
    """Readable, unique label per group: its most frequent display text."""
    frame = pd.DataFrame({"g": groups, "d": display.fillna("")})
    mode = frame.groupby("g")["d"].agg(lambda s: s.value_counts().index[0])
    mode = mode.where(mode != "", prefix + mode.index.to_series().astype(str))
    dup = mode.duplicated(keep=False)
    mode[dup] = mode[dup] + " #" + mode[dup].groupby(mode[dup]).cumcount().add(1).astype(str)
    return groups.map(mode)


def similar_pairs(keys: list[str], threshold: float, min_len: int, chunk: int = 2000) -> Iterable[tuple[int, int]]:
    """Index pairs of near-identical keys (ratio >= threshold), both at least min_len long."""
    idx = [i for i, k in enumerate(keys) if len(k) >= min_len]
    if len(idx) < 2:
        return
    subset = [keys[i] for i in idx]
    for start in range(0, len(subset), chunk):
        block = process.cdist(subset[start:start + chunk], subset, scorer=fuzz.ratio,
                              score_cutoff=threshold, dtype=np.uint8, workers=-1)
        rows, cols = np.nonzero(block)
        for r, c in zip(rows, cols):
            i, j = idx[start + r], idx[c]
            if i < j and numbers(keys[i]) == numbers(keys[j]):
                yield i, j


def supplier_keys(df: pd.DataFrame, settings: dict) -> pd.Series:
    """One key per real supplier, from shared supplier IDs and near-identical names."""
    norm = df["supplier_name"].map(supplier_norm)
    ids = df["supplier_id"].astype("string")
    name_node = ("N:" + norm).where(norm != "")
    id_node = ("I:" + ids).where(ids.notna())

    uf = UnionFind()
    for node in pd.concat([name_node, id_node]).dropna().unique():
        uf.add(node)
    for a, b in pd.DataFrame({"a": name_node, "b": id_node}).dropna().drop_duplicates().itertuples(index=False):
        uf.union(a, b)

    names = sorted(norm[norm != ""].unique())
    for i, j in similar_pairs(names, settings["supplier_fuzzy_threshold"], settings["supplier_fuzzy_min_len"]):
        uf.union("N:" + names[i], "N:" + names[j])

    node = name_node.fillna(id_node)
    groups = node.map(lambda n: uf.find(n) if isinstance(n, str) else "UNKNOWN")
    display = df["supplier_name"].str.upper().str.split().str.join(" ").fillna(ids)
    return _labels(groups, display, "SUPPLIER ")


def uninformative(keys: pd.Series, phrases: list[str]) -> pd.Series:
    """Descriptions that say nothing about the item ('Product', 'No Object', blank, a bare number)."""
    phrase_keys = [desc_key(p) for p in phrases]
    letters = keys.str.count(r"[A-Z]")

    def junk(key: str) -> bool:
        return process.extractOne(key, phrase_keys, scorer=fuzz.ratio, score_cutoff=85) is not None

    unique = pd.Series(keys.unique())
    junk_map = dict(zip(unique, unique.map(junk)))
    return (letters < 3) | keys.map(junk_map)


def generic_codes(code: pd.Series, dkey: pd.Series, informative: pd.Series, supplier: pd.Series,
                  settings: dict) -> set[str]:
    """Item codes used as catch-alls: many unrelated descriptions or many unrelated suppliers."""
    frame = pd.DataFrame({"code": code, "d": dkey.where(informative), "s": supplier}).dropna(subset=["code"])
    stats = frame.groupby("code").agg(
        descs=("d", "nunique"),
        top_share=("d", lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else 1.0),
        suppliers=("s", "nunique"),
    )
    many_descs = (stats["descs"] >= settings["generic_code_min_descs"]) & (
        stats["top_share"] < settings["generic_code_max_top_share"])
    many_suppliers = stats["suppliers"] >= settings["generic_code_min_suppliers"]
    return set(stats.index[many_descs | many_suppliers])


def item_keys(df: pd.DataFrame, settings: dict, fallback_context: list[str],
              links: Iterable[tuple[int, int]] = ()) -> pd.Series:
    """One key per real item.

    Records are linked by the same non-generic code or the same normalised description,
    and a fuzzy pass attaches noisy descriptions to a clean one from the same supplier
    when all numbers agree. `links` are (duplicate, original) record positions: the
    duplicate takes the original's item. They are one-way on purpose, so one noisy
    invoice text never merges two items.
    """
    dkey = df["description"].map(desc_key)
    informative = ~uninformative(dkey, settings["uninformative_descriptions"])
    code = df["item_code"].astype("string").str.upper()
    generic = generic_codes(code, dkey, informative, df["supplier_key"], settings)
    code = code.where(~code.isin(generic))

    code_node = ("C:" + code).where(code.notna())
    desc_node = ("D:" + dkey).where(informative)
    context = df[fallback_context].astype("string").fillna("").agg("|".join, axis=1) if fallback_context else ""
    fallback_node = "U:" + df["supplier_key"] + "|" + context
    node = code_node.fillna(desc_node).fillna(fallback_node)

    uf = UnionFind()
    for n in node.unique():
        uf.add(n)
    for a, b in pd.DataFrame({"a": code_node, "b": desc_node}).dropna().drop_duplicates().itertuples(index=False):
        uf.union(a, b)
    _fuzzy_attach(uf, dkey, desc_node, code_node, df["supplier_key"], settings)

    groups = node.map(uf.find).to_numpy(dtype=object)
    parent = dict(links)

    def resolve(i: int) -> int:
        seen = set()
        while i in parent and i not in seen:  # follow duplicate -> original chains
            seen.add(i)
            i = parent[i]
        return i

    for i in parent:
        groups[i] = groups[resolve(i)]
    groups = pd.Series(groups, index=df.index)
    display = df["description"].astype("string").str.upper().str.split().str.join(" ")
    display = display.where(informative, "UNSPECIFIED " + df["supplier_key"])
    return _labels(groups, display, "ITEM ")


def _fuzzy_attach(uf: UnionFind, dkey: pd.Series, desc_node: pd.Series, code_node: pd.Series,
                  supplier: pd.Series, settings: dict) -> None:
    """Attach descriptions without an item code to a near-identical description of the same supplier."""
    threshold = settings["item_fuzzy_threshold"]
    frame = pd.DataFrame({"k": dkey, "node": desc_node, "coded": code_node.notna(), "s": supplier}).dropna(subset=["node"])
    for _, grp in frame.groupby("s"):
        keys = grp.drop_duplicates("k")
        if len(keys) < 2:
            continue
        choices = keys["k"].tolist()
        coded = set(keys.loc[keys["coded"], "k"]) | set(grp.loc[grp["coded"], "k"])
        for query in keys.loc[~keys["k"].isin(coded), "k"]:
            if len(query) < settings["item_fuzzy_min_len"]:
                continue
            hits = process.extract(query, choices, scorer=fuzz.ratio, score_cutoff=threshold, limit=3)
            for match, _, _ in hits:
                if match != query and numbers(match) == numbers(query):
                    uf.union("D:" + query, "D:" + match)
                    break


SHARED_SUPPLIER = "UNMAPPED SUPPLIER"
SHARED_ITEM = "UNMAPPED ITEM"


def output_keys(df: pd.DataFrame, conventions, names_with_id: frozenset[str] = frozenset()
                ) -> tuple[pd.Series, pd.Series]:
    """Supplier and item keys for the output file, after the company's key conventions.

    `names_with_id` holds normalised supplier names seen with a supplier ID in an earlier
    delivery: a new delivery may carry no IDs at all for a supplier the client does know.
    """
    supplier, item = df["supplier_key"].copy(), df["item_key"].copy()
    if conventions is None:
        return supplier, item
    if conventions.shared_supplier_without_id:
        known = df["supplier_id"].notna() | df["supplier_name"].map(supplier_norm).isin(names_with_id)
        has_id = known.groupby(df["supplier_key"]).transform("any")
        supplier = supplier.where(has_id, SHARED_SUPPLIER)
    if conventions.shared_item:
        rule = conventions.shared_item
        column = df[rule.column] if rule.column in df else pd.Series(pd.NA, index=df.index)
        hit = column.astype("string").str.contains(rule.pattern, case=False, regex=True).astype("Float64")
        share = hit.groupby(df["item_key"]).mean()  # records without the column don't vote
        item = item.where(~(df["item_key"].map(share) > 0.5).fillna(False), SHARED_ITEM)
    return supplier, item

"""Classify included lines into the company's tree, to its lowest level.

The unit of work is the item signature (supplier group + item group): each distinct
thing is classified once and the answer is copied to all of its lines.

Each signature takes the first route that applies:
1. known  - a labelled example with the same supplier and description (or the same
            description) exists: take its majority category.
2. llm    - the LLM labels the signatures that carry most of the spend.
3. local  - a TF-IDF + logistic-regression model trained on routes 1-2 labels the rest.
            Its features include the supplier and account text, so it also acts as a
            supplier prior for vague descriptions ('Parts', blank).
With no labels and no LLM, a lexical match between item text and leaf names seeds the
local model instead (the free fallback).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline

from .config import CompanyConfig
from .dedupe import uninformative
from .llm import LLM
from .taxonomy import Taxonomy
from .text import desc_key, desc_norm, supplier_norm

KNOWN_EXACT_CONFIDENCE = 0.97
KNOWN_DESC_CONFIDENCE = 0.9
MAX_PROMPT_EXAMPLES = 60
FALLBACK_MIN_SIMILARITY = 0.15
SUPPLIER_PROFILE_ITEMS = 6


# --- features ---------------------------------------------------------------

def record_features(df: pd.DataFrame, cfg: CompanyConfig, settings: dict) -> pd.DataFrame:
    """Per-record text features shared by examples and signatures."""
    feats = pd.DataFrame(index=df.index)
    name = df["supplier_name"].map(supplier_norm)
    feats["supplier"] = name.where(name != "", df["supplier_id"].astype("string")).fillna("")
    feats["supplier_display"] = df["supplier_name"].astype("string").fillna("").str.split().str.join(" ")
    feats["desc"] = df["description"].astype("string").fillna("").str.split().str.join(" ")
    feats["desc_key"] = df["description"].map(desc_key)
    feats["informative"] = ~uninformative(feats["desc_key"], settings["uninformative_descriptions"])
    ctx = cfg.classification_context
    feats["context"] = (df[ctx].astype("string").apply(
        lambda row: " | ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v)), axis=1)
        if ctx else "")
    # Vague descriptions are told apart by their account context instead.
    feats["match_key"] = feats["desc_key"].where(feats["informative"], "U|" + feats["context"])
    return feats


def examples_from_labels(df: pd.DataFrame, feats: pd.DataFrame, labels: pd.DataFrame,
                         taxonomy: Taxonomy) -> pd.DataFrame:
    """Labelled records as training examples: features plus leaf ID, included lines only."""
    level_cols = [c for c in labels.columns if c.startswith("gold_l")]
    m = df[["record_id"]].join(feats).merge(labels, on="record_id")
    if "gold_include" in m:
        m = m[m["gold_include"].astype(str) == "1"]
    m["leaf"] = [getattr(taxonomy.leaf_for_labels(list(r)), "id", None) for r in m[level_cols].itertuples(index=False)]
    return m.dropna(subset=["leaf"])[["supplier", "desc", "desc_key", "informative", "context", "match_key", "leaf"]]


def build_signatures(inc: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One row per (supplier_key, item_key) with its most common features and its spend."""
    frame = inc[["supplier_key", "item_key"]].join(feats)
    frame["spend"] = inc["spend_value"].abs()
    mode = lambda s: s.value_counts().index[0] if s.notna().any() else ""  # noqa: E731
    sigs = frame.groupby(["supplier_key", "item_key"], sort=False).agg(
        supplier=("supplier", mode), supplier_display=("supplier_display", mode), desc=("desc", mode), desc_key=("desc_key", mode),
        informative=("informative", "max"), context=("context", mode), match_key=("match_key", mode),
        spend=("spend", "sum"), lines=("spend", "size"),
    ).reset_index()
    sigs["sig"] = np.arange(len(sigs))
    line_sig = frame.merge(sigs[["supplier_key", "item_key", "sig"]], on=["supplier_key", "item_key"], how="left")["sig"]
    line_sig.index = inc.index
    return sigs, line_sig


# --- routes -----------------------------------------------------------------

def known_labels(sigs: pd.DataFrame, examples: pd.DataFrame) -> pd.DataFrame:
    """Signatures that have labelled twins: same supplier and key, else same informative key."""
    if examples.empty:
        return pd.DataFrame(columns=["sig", "leaf", "confidence"])

    def majority(keys: list[str], base: float) -> pd.DataFrame:
        stats = examples.groupby(keys)["leaf"].agg(
            leaf=lambda s: s.value_counts().index[0], purity=lambda s: s.value_counts(normalize=True).iloc[0])
        hit = sigs.merge(stats.reset_index(), on=keys)
        return pd.DataFrame({"sig": hit["sig"], "leaf": hit["leaf"], "confidence": base * hit["purity"]})

    exact = majority(["supplier", "match_key"], KNOWN_EXACT_CONFIDENCE)
    by_desc = majority(["desc_key"], KNOWN_DESC_CONFIDENCE)
    by_desc = by_desc[by_desc["sig"].isin(sigs.loc[sigs["informative"], "sig"]) & ~by_desc["sig"].isin(exact["sig"])]
    return pd.concat([exact, by_desc], ignore_index=True)


def _schema(leaf_ids: list[str]) -> dict:
    item = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "leaf": {"type": "string", "enum": leaf_ids},
            "confidence": {"type": "number"},
            "non_procurement": {"type": "boolean"},
        },
        "required": ["id", "leaf", "confidence", "non_procurement"],
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {"results": {"type": "array", "items": item}},
            "required": ["results"], "additionalProperties": False}


def _prompt_header(taxonomy: Taxonomy, examples: pd.DataFrame) -> str:
    names = taxonomy.answer_labels()
    tree = "\n".join(f"{names[leaf.id]}  =  {leaf.text}" for leaf in taxonomy.leaves)
    parts = [
        "You classify one company's purchase lines into that company's spend category tree.",
        "Choose the single best leaf for each item and answer with its name exactly as listed before the '='. Descriptions may be abbreviated, OCR-damaged or in any "
        "language; use the supplier, what that supplier mainly sells, and the other fields when the "
        "description is vague.",
        "A leaf whose name repeats its parent's name is that branch's general category: prefer it unless a "
        "more specific sibling clearly and explicitly applies.",
        "confidence is your probability (0 to 1) that the leaf is right.",
        "Set non_procurement to true only if the line is clearly not a purchase from a third-party supplier "
        "(a tax, intercompany charge, revenue, royalty, financing cost or similar).",
        "Answer from the text alone. Do not run commands or read files.",
        f"\nCategory tree (leaf name  =  full path):\n{tree}",
    ]
    shots = _prompt_examples(examples, names)
    if shots:
        parts.append(f"\nLabelled examples from this company (description | context -> leaf):\n{shots}")
    return "\n".join(parts)


def _prompt_examples(examples: pd.DataFrame, names: dict[str, str]) -> str:
    """Up to two well-supported examples per leaf, so the model sees the company's conventions."""
    if examples.empty:
        return ""
    ex = examples[examples["informative"]]
    counts = ex.groupby(["leaf", "desc", "context"]).size().rename("n").reset_index()
    picked = counts.sort_values("n", ascending=False).groupby("leaf").head(2).head(MAX_PROMPT_EXAMPLES)
    return "\n".join(f"{r.desc} | {r.context} -> {names[r.leaf]}" for r in picked.sort_values("leaf").itertuples())


def _supplier_label(row) -> str:
    return row.supplier_display or row.supplier_key


def _card(row) -> dict:
    card = {"id": f"s{row.sig}", "description": row.desc or "(blank)", "supplier": _supplier_label(row)}
    if row.context:
        card["context"] = row.context
    return card


def supplier_profiles(sigs: pd.DataFrame, top: int = SUPPLIER_PROFILE_ITEMS) -> dict[str, str]:
    """What each supplier mainly sells: its top informative descriptions by spend.

    Supplier names are often anonymised or uninformative, so this is the LLM's best clue for
    vague lines such as 'Mechanical maintenance' or 'Parts'.
    """
    ranked = sigs[sigs["informative"]].sort_values("spend", ascending=False)
    ranked = ranked.assign(label=[_supplier_label(r) for r in ranked.itertuples()])
    return {label: "; ".join(d[:45] for d in grp["desc"].head(top))
            for label, grp in ranked.groupby("label", sort=False)}


def llm_labels(sigs: pd.DataFrame, taxonomy: Taxonomy, examples: pd.DataFrame, llm: LLM,
               settings: dict, profiles: dict[str, str] | None = None) -> pd.DataFrame:
    """Label signatures with the LLM in batches; failed batches are simply left out."""
    cfg = settings["llm"]
    header = _prompt_header(taxonomy, examples)
    profiles = profiles or {}
    names = taxonomy.answer_labels()
    leaf_of_name = {name: lid for lid, name in names.items()}
    schema = _schema(list(names.values()))
    # Highest spend first, so if LLM calls stop partway only the low-spend tail misses out;
    # within a batch, similar items sit together.
    ranked, size = sigs.sort_values("spend", ascending=False), cfg["batch_size"]
    batches = [ranked.iloc[i:i + size].sort_values(["supplier", "desc"]) for i in range(0, len(ranked), size)]

    def run(batch: pd.DataFrame) -> list[dict]:
        cards = [_card(r) for r in batch.itertuples()]
        suppliers = dict.fromkeys(c["supplier"] for c in cards)
        known = "\n".join(f"- {name}: {profiles[name]}" for name in suppliers if profiles.get(name))
        context = f"\n\nSuppliers in this batch and their main items by spend:\n{known}" if known else ""
        items = "\n".join(json.dumps(c, ensure_ascii=False) for c in cards)
        answer = llm.complete_json(f"{header}{context}\n\nItems (one JSON object per line):\n{items}", schema)
        return answer["results"] if answer else []

    with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
        results = [r for batch in pool.map(run, batches) for r in batch]
    out = pd.DataFrame(results, columns=["id", "leaf", "confidence", "non_procurement"])
    out["leaf"] = out["leaf"].map(leaf_of_name)
    out = out[out["id"].str.fullmatch(r"s\d+") & out["leaf"].notna()]
    out["sig"] = out["id"].str[1:].astype(int)
    out = out[out["sig"].isin(sigs["sig"])].drop_duplicates("sig")
    out["confidence"] = out["confidence"].clip(0, 1)
    return out[["sig", "leaf", "confidence", "non_procurement"]]


def _model() -> object:
    features = ColumnTransformer([
        ("desc", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1), "desc_text"),
        ("meta", TfidfVectorizer(analyzer="word", token_pattern=r"[^\s|]+", sublinear_tf=True), "meta_text"),
    ])
    return make_pipeline(features, LogisticRegression(C=5.0, max_iter=2000))


def _model_input(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "desc_text": frame["desc"].map(desc_norm).where(frame["informative"], ""),
        "meta_text": frame["context"].str.lower() + " sup_" + frame["supplier"].str.lower(),
    })


def local_labels(train: pd.DataFrame, sigs: pd.DataFrame) -> pd.DataFrame:
    """Train on labelled rows (features + leaf) and predict every signature."""
    if train["leaf"].nunique() < 2:
        leaf = train["leaf"].iloc[0] if len(train) else None
        return pd.DataFrame({"sig": sigs["sig"], "leaf": leaf, "confidence": 0.3 if leaf else 0.0})
    # Identical rows (one per labelled line) collapse to one weighted row: same fit, less memory.
    rows = train.groupby(["supplier", "desc", "informative", "context", "leaf"], dropna=False).size()
    rows = rows.rename("weight").reset_index()
    model = _model()
    model.fit(_model_input(rows), rows["leaf"], logisticregression__sample_weight=rows["weight"].to_numpy())
    proba = model.predict_proba(_model_input(sigs))
    classes = model.classes_
    return pd.DataFrame({"sig": sigs["sig"].to_numpy(), "leaf": classes[proba.argmax(1)], "confidence": proba.max(1)})


def lexical_seed(sigs: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """Free fallback: character n-gram similarity between item text and leaf paths."""
    leaf_text = [desc_norm(leaf.text) for leaf in taxonomy.leaves]
    item_text = (sigs["desc"].map(desc_norm) + " " + sigs["context"].str.upper()).tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(leaf_text + item_text)
    sim = cosine_similarity(vec.transform(item_text), vec.transform(leaf_text))
    best = sim.argmax(1)
    seed = pd.DataFrame({"sig": sigs["sig"].to_numpy(), "leaf": [taxonomy.leaves[i].id for i in best],
                         "confidence": sim.max(1)})
    return seed[seed["confidence"] >= FALLBACK_MIN_SIMILARITY]


# --- orchestration ----------------------------------------------------------

def select_for_llm(sigs: pd.DataFrame, done: set, settings: dict) -> pd.DataFrame:
    """Informative, not-yet-known signatures in spend order until coverage or the item cap is reached.

    Coverage counts only informative spend: lines with blank or vague descriptions never go to
    the LLM, so counting them made the target unreachable and ran selection up to the cap.
    """
    cfg = settings["llm"]
    eligible = sigs[sigs["informative"]]
    total = eligible["spend"].sum()
    covered = eligible.loc[eligible["sig"].isin(done), "spend"].sum()
    todo = eligible[~eligible["sig"].isin(done)].sort_values("spend", ascending=False)
    need = cfg["spend_coverage"] * total - covered
    take = int((todo["spend"].cumsum().shift(fill_value=0) < need).sum()) if need > 0 else 0
    return todo.head(min(take, cfg["max_items"]))


def classify(inc: pd.DataFrame, feats: pd.DataFrame, taxonomy: Taxonomy, examples: pd.DataFrame,
             llm: LLM | None, settings: dict) -> tuple[pd.DataFrame, dict]:
    """Leaf ID, confidence and route for every included line, plus route statistics."""
    sigs, line_sig = build_signatures(inc, feats)
    known = known_labels(sigs, examples)
    known["route"] = "known"

    llm_out = pd.DataFrame(columns=["sig", "leaf", "confidence", "non_procurement"])
    if llm is not None:
        chosen = select_for_llm(sigs, set(known["sig"]), settings)
        if len(chosen):
            llm_out = llm_labels(chosen, taxonomy, examples, llm, settings, supplier_profiles(sigs))
    llm_out["route"] = "llm"

    decided = pd.concat([known, llm_out.drop(columns="non_procurement")], ignore_index=True)
    train = pd.concat([
        examples[["supplier", "desc", "informative", "context", "leaf"]],
        sigs.merge(decided[["sig", "leaf"]], on="sig")[["supplier", "desc", "informative", "context", "leaf"]],
    ], ignore_index=True)
    if train.empty:
        seed = lexical_seed(sigs, taxonomy)
        train = sigs.merge(seed[["sig", "leaf"]], on="sig")[["supplier", "desc", "informative", "context", "leaf"]]
    local = local_labels(train, sigs) if len(train) else pd.DataFrame(columns=["sig", "leaf", "confidence"])
    local = local[~local["sig"].isin(decided["sig"])].assign(route="local")

    per_sig = pd.concat([decided, local], ignore_index=True).drop_duplicates("sig").set_index("sig")
    llm_flag = llm_out.set_index("sig")["non_procurement"] if len(llm_out) else pd.Series(dtype=bool)
    lines = pd.DataFrame({
        "leaf_id": line_sig.map(per_sig["leaf"]),
        "confidence": line_sig.map(per_sig["confidence"]).astype(float),
        "route": line_sig.map(per_sig["route"]),
        "llm_non_procurement": line_sig.map(llm_flag).eq(True),
        "llm_confidence": line_sig.map(llm_out.set_index("sig")["confidence"]) if len(llm_out) else np.nan,
    }, index=inc.index)

    spend = sigs.set_index("sig")["spend"]
    stats = {"signatures": len(sigs)}
    for route, grp in per_sig.groupby("route"):
        stats[f"route_{route}_signatures"] = len(grp)
        stats[f"route_{route}_spend_share"] = float(spend.loc[grp.index].sum() / max(spend.sum(), 1e-9))
    return lines, stats

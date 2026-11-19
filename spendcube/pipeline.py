"""End-to-end orchestration: config + files -> spend cube."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .classify import classify, examples_from_labels, record_features
from .config import CompanyConfig, load_config, load_yaml
from .decide import decide, status_codes
from .dedupe import item_keys, output_keys, supplier_keys
from .evaluate import classification_report, exclusion_report, grouping_report, load_labels
from .ingest import ingest
from .llm import REPO_ROOT, make_llm
from .match import collapse_reexports, match_systems
from .normalise import normalise
from .output import build_output, check_output
from .taxonomy import Taxonomy
from .text import supplier_norm
from .tracking import Tracker

REEXPORT_CONFIDENCE = 0.95
EXACT_DUPLICATE_CONFIDENCE = 0.99
LLM_CACHE = REPO_ROOT / ".cache" / "llm"


@dataclass
class Prepared:
    records: pd.DataFrame
    matches: pd.DataFrame
    stats: dict = field(default_factory=dict)


def find_duplicates(df: pd.DataFrame, cfg: CompanyConfig, settings: dict):
    """Return (duplicate_of, confidence, matches, item_links).

    Exact re-exports within a source, re-exports within the secondary system, and
    secondary records matched to the primary system. Only approved records take part
    in matching, so a cancelled invoice never uses up a goods receipt.
    """
    dup_of = pd.Series(pd.NA, index=df.index, dtype="string")
    conf = pd.Series(np.nan, index=df.index)
    links: list[tuple[int, int]] = []

    # Positions, not record IDs, link duplicates: IDs are not guaranteed unique in new files.
    first = pd.Series(df.index, index=df.index).groupby([df["source"], df["row_hash"]]).transform("first")
    exact = df.duplicated(["source", "row_hash"])
    dup_of[exact] = df.loc[first[exact], "record_id"].to_numpy()
    conf[exact] = EXACT_DUPLICATE_CONFIDENCE
    links += [(i, int(first[i])) for i in df.index[exact]]

    matches = pd.DataFrame()
    if cfg.matching is None:
        return dup_of, conf, matches, links

    approved = status_codes(df, cfg).isna()
    secondary = df[(df["source"] == cfg.matching.secondary) & ~exact]
    reexport = collapse_reexports(secondary, approved[secondary.index], settings["reexport_min_similarity"])
    hit = reexport.dropna()
    dup_of[hit.index] = df.loc[hit.to_numpy(), "record_id"].to_numpy()
    conf[hit.index] = REEXPORT_CONFIDENCE
    links += [(i, int(j)) for i, j in hit.items()]

    candidates = secondary[reexport.isna() & approved[secondary.index]]
    primary = df[(df["source"] == cfg.matching.primary) & ~exact]
    matches = match_systems(primary, candidates, settings["match_min_text_similarity"])
    if not matches.empty:
        dup_of[matches["secondary_index"].to_numpy()] = df.loc[matches["primary_index"], "record_id"].to_numpy()
        conf[matches["secondary_index"].to_numpy()] = matches["confidence"].to_numpy()
        links += list(zip(matches["secondary_index"], matches["primary_index"]))
    return dup_of, conf, matches, links


def names_with_id(cfg: CompanyConfig, reference_dir: Path | None) -> frozenset[str]:
    """Normalised supplier names that carry a supplier ID in the reference delivery."""
    if reference_dir is None or cfg.key_conventions is None:
        return frozenset()
    ref = ingest(cfg, reference_dir)
    return frozenset(ref.loc[ref["supplier_id"].notna(), "supplier_name"].map(supplier_norm)) - {""}


def prepare(cfg: CompanyConfig, data_dir: Path, reference_dir: Path | None = None) -> Prepared:
    """Everything before classification: ingest, clean, count once, decide, de-duplicate."""
    settings, terms = load_yaml("defaults.yaml"), load_yaml("non_procurement_terms.yaml")
    df = ingest(cfg, data_dir)
    df = normalise(df, cfg, data_dir, settings)
    df["supplier_key"] = supplier_keys(df, settings)
    dup_of, dup_conf, matches, links = find_duplicates(df, cfg, settings)
    df = decide(df, cfg, terms, dup_of, dup_conf)
    df["duplicate_of"] = dup_of
    df["item_key"] = item_keys(df, settings, cfg.classification_context, links)
    df["supplier_key_out"], df["item_key_out"] = output_keys(df, cfg.key_conventions,
                                                             names_with_id(cfg, reference_dir))
    stats = {
        "records": len(df),
        "suppliers": int(df["supplier_key_out"].nunique()),
        "items": int(df["item_key_out"].nunique()),
        **{f"flag_{k.lower()}": int(v) for k, v in
           df["data_flags"].str.split(";").explode().loc[lambda s: s != ""].value_counts().items()},
    }
    if not matches.empty:
        stats.update({f"matched_tier{t}": int(n) for t, n in matches["tier"].value_counts().items()})
    return Prepared(df, matches, stats)


def load_examples(cfg: CompanyConfig, reference_dir: Path | None, taxonomy: Taxonomy, settings: dict) -> pd.DataFrame:
    """Labelled training examples from a data folder that holds the company's labels file."""
    empty = pd.DataFrame(columns=["supplier", "desc", "desc_key", "informative", "context", "match_key", "leaf"])
    if cfg.labels is None or reference_dir is None or not (reference_dir / cfg.labels.file).exists():
        return empty
    ref = ingest(cfg, reference_dir)
    return examples_from_labels(ref, record_features(ref, cfg, settings), load_labels(cfg, reference_dir), taxonomy)


def classify_records(df: pd.DataFrame, cfg: CompanyConfig, taxonomy: Taxonomy, examples: pd.DataFrame,
                     llm, settings: dict) -> tuple[pd.DataFrame, dict]:
    """Classify included records; the LLM's confident non-procurement flags become exclusions."""
    feats = record_features(df, cfg, settings)
    inc = df[df["include"] == 1]
    lines, stats = classify(inc, feats.loc[inc.index], taxonomy, examples, llm, settings)
    df = df.join(lines)
    flagged = df["llm_non_procurement"].fillna(False).astype(bool) & (
        df["llm_confidence"] >= settings["llm_exclusion_min_confidence"])
    df.loc[flagged, "include"] = 0
    df.loc[flagged, "exclusion_reason"] = "NON_PROCUREMENT"
    df.loc[flagged, "exclusion_confidence"] = df.loc[flagged, "llm_confidence"] * 0.9
    stats["llm_excluded_lines"] = int(flagged.sum())
    return df, stats


def summarise(df: pd.DataFrame) -> dict:
    inc = df["include"] == 1
    return {
        "included": int(inc.sum()),
        "spend_total": float(df.loc[inc, "spend_value"].sum()),
        **{f"excluded_{k.lower()}": int(v) for k, v in df["exclusion_reason"].value_counts().items()},
    }


def label_metrics(df: pd.DataFrame, cfg: CompanyConfig, data_dir: Path, taxonomy: Taxonomy,
                  only: pd.Series | None = None) -> dict:
    """Scores against the labels in data_dir, if any (optionally on a subset of records)."""
    labels = load_labels(cfg, data_dir) if cfg.labels and (data_dir / cfg.labels.file).exists() else None
    if labels is None:
        return {}
    if only is not None:
        labels = labels[labels["record_id"].isin(only)]
    df = df.join(taxonomy.level_columns(df["leaf_id"].where(df["include"] == 1)))
    return {**classification_report(df, labels, taxonomy.depth), **exclusion_report(df, labels),
            **grouping_report(df, labels, "supplier_key"), **grouping_report(df, labels, "item_key")}


def llm_metrics(llm) -> dict:
    if llm is None:
        return {}
    metrics = {f"llm_{k}": v for k, v in llm.usage.as_dict().items()}
    if llm.disabled_reason:
        metrics["llm_disabled_reason"] = llm.disabled_reason
        print(f"[llm] WARNING: LLM calls stopped during this run: {llm.disabled_reason}")
    return metrics


def _write(df: pd.DataFrame, taxonomy: Taxonomy, out_path: Path) -> list[str]:
    out = build_output(df, taxonomy)
    problems = check_output(out, len(df), taxonomy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    details = df[["record_id", "source", "route", "duplicate_of", "data_flags", "currency_clean", "fx_used",
                  "net_amount"]]
    details_path = REPO_ROOT / "runs" / f"{out_path.stem}_details.csv"
    details_path.parent.mkdir(exist_ok=True)
    details.to_csv(details_path, index=False)
    return problems


def run(config_path: str, data_dir: str, out_path: str, reference_dir: str | None = None,
        hide_labels: bool = False, name: str | None = None) -> dict:
    """Produce the output file for one company. Returns the run metrics."""
    load_dotenv(REPO_ROOT / ".env")
    start = time.time()
    cfg, settings = load_config(config_path), load_yaml("defaults.yaml")
    data, out = Path(data_dir), Path(out_path)
    tags = [f"company-{cfg.company}", "zero-label" if hide_labels else "with-labels"]
    tracker = Tracker("run", name or out.stem, {"config": cfg.model_dump(), "settings": settings,
                                                "data_dir": data.name, "hide_labels": hide_labels}, tags)
    try:
        reference = None if hide_labels else Path(reference_dir) if reference_dir else data
        prep = prepare(cfg, data, reference)
        taxonomy = Taxonomy.from_csv(data / cfg.taxonomy, cfg.taxonomy_columns, cfg.taxonomy_separator)
        examples = load_examples(cfg, reference, taxonomy, settings)
        llm = make_llm(settings, LLM_CACHE)
        df, class_stats = classify_records(prep.records, cfg, taxonomy, examples, llm, settings)
        problems = _write(df, taxonomy, out)
        metrics = {**prep.stats, **summarise(df), **class_stats, "examples": len(examples),
                   "llm_backend": getattr(llm, "name", "none"), "runtime_seconds": time.time() - start,
                   **llm_metrics(llm), **label_metrics(df, cfg, data, taxonomy), "output_problems": len(problems)}
        if llm and llm.usage.calls:
            metrics["llm_tokens_per_1000_lines"] = (
                (llm.usage.input_tokens + llm.usage.output_tokens) / max(metrics["included"], 1) * 1000)
        tracker.log(metrics)
        for problem in problems:
            print(f"[output] PROBLEM: {problem}")
        tracker.finish(failed=bool(problems))
        return metrics
    except BaseException:
        tracker.finish(failed=True)
        raise


def holdout(config_path: str, data_dir: str, fraction: float = 0.2, seed: int = 0,
            name: str | None = None) -> dict:
    """Hide the labels of a random fifth of the labelled items and score the classifier on them.

    Splitting by item (not by line) measures new items, the hard case; lines of known items
    are found by the 'known' route.
    """
    load_dotenv(REPO_ROOT / ".env")
    start = time.time()
    cfg, settings = load_config(config_path), load_yaml("defaults.yaml")
    data = Path(data_dir)
    tracker = Tracker("holdout", name or f"{cfg.company}-holdout", {"config": cfg.model_dump(), "settings": settings,
                                                                    "fraction": fraction, "seed": seed},
                      [f"company-{cfg.company}", "holdout"])
    try:
        prep = prepare(cfg, data)
        df = prep.records
        taxonomy = Taxonomy.from_csv(data / cfg.taxonomy, cfg.taxonomy_columns, cfg.taxonomy_separator)
        labels = load_labels(cfg, data)
        labelled = df[df["record_id"].isin(labels["record_id"])]
        items = labelled["item_key"].drop_duplicates().sample(frac=fraction, random_state=seed)
        test_ids = labelled.loc[labelled["item_key"].isin(items), "record_id"]
        train_labels = labels[~labels["record_id"].isin(test_ids)]
        examples = examples_from_labels(df, record_features(df, cfg, settings), train_labels, taxonomy)
        llm = make_llm(settings, LLM_CACHE)
        df, class_stats = classify_records(df, cfg, taxonomy, examples, llm, settings)
        test_metrics = label_metrics(df, cfg, data, taxonomy, only=test_ids)
        metrics = {**class_stats, "test_records": len(test_ids), "train_examples": len(examples),
                   "runtime_seconds": time.time() - start, **llm_metrics(llm),
                   **{k: v for k, v in test_metrics.items() if k.startswith(("cls_", "acc_", "excl_"))}}
        test = df[df["record_id"].isin(test_ids) & (df["include"] == 1)]
        metrics.update({f"test_route_{r}_lines": int(n) for r, n in test["route"].value_counts().items()})
        tracker.log(metrics)
        tracker.finish()
        return metrics
    except BaseException:
        tracker.finish(failed=True)
        raise

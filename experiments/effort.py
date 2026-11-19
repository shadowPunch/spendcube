"""Does more reasoning help on an unseen tree? Label a company's top-spend items with its labels
hidden, once per reasoning effort, and score the answers against the labels.

One W&B run per effort (tag effort-experiment). Answers are cached, so a repeat makes no new calls.

    uv run python experiments/effort.py --config configs/A.yaml --data-dir data/company_A
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from spendcube.classify import build_signatures, llm_labels, record_features, supplier_profiles
from spendcube.config import load_config, load_yaml
from spendcube.evaluate import classification_report, load_labels
from spendcube.llm import REPO_ROOT, make_llm
from spendcube.pipeline import LLM_CACHE, llm_metrics, prepare
from spendcube.taxonomy import Taxonomy
from spendcube.tracking import Tracker

NO_EXAMPLES = pd.DataFrame(columns=["supplier", "desc", "desc_key", "informative", "context", "match_key", "leaf"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--efforts", nargs="+", default=["low", "medium", "high"])
    parser.add_argument("--top", type=int, default=200, help="items by spend, sent as one batch")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")

    cfg, data, base = load_config(args.config), Path(args.data_dir), load_yaml("defaults.yaml")
    df = prepare(cfg, data).records
    taxonomy = Taxonomy.from_csv(data / cfg.taxonomy, cfg.taxonomy_columns, cfg.taxonomy_separator)
    included = df[df["include"] == 1]
    sigs, line_sig = build_signatures(included, record_features(df, cfg, base).loc[included.index])
    top = sigs[sigs["informative"]].nlargest(args.top, "spend")
    lines = included.assign(sig=line_sig)
    lines = lines[lines["sig"].isin(top["sig"])][["record_id", "spend_value", "sig"]]
    labels = load_labels(cfg, data)
    levels = sum(c.startswith("gold_l") for c in labels.columns)
    share = float(top["spend"].sum() / sigs["spend"].sum())

    for effort in args.efforts:
        settings = copy.deepcopy(base)
        settings["llm"].update(reasoning_effort=effort, batch_size=args.top, concurrency=1, timeout_seconds=1800)
        name = f"effort-{cfg.company}-top{args.top}-{effort}"
        tracker = Tracker("eval", name, {"company": cfg.company, "effort": effort, "top_items": args.top,
                                         "spend_share": share, "llm": settings["llm"]}, ["effort-experiment"])
        llm = make_llm(settings, LLM_CACHE)
        if llm is None:
            tracker.finish(failed=True)
            raise SystemExit("no LLM configured: set the API key in .env")
        start = time.time()
        answers = llm_labels(top, taxonomy, NO_EXAMPLES, llm, settings, supplier_profiles(sigs))
        pred = lines.merge(answers[["sig", "leaf"]], on="sig", how="left")
        scored = pd.concat([pred.assign(include=1), taxonomy.level_columns(pred["leaf"])], axis=1)
        metrics = {"items": args.top, "answered": len(answers), "lines": len(lines), "spend_share": share,
                   "seconds": time.time() - start, **llm_metrics(llm),
                   **classification_report(scored, labels, levels)}
        tracker.log(metrics)
        tracker.finish(failed=len(answers) == 0)
        print(json.dumps({"effort": effort, **{k: round(v, 4) if isinstance(v, float) else v
                                               for k, v in metrics.items()}}))


if __name__ == "__main__":
    main()

"""Command line: `run` builds a company's output file, `holdout` measures it, `profile` inspects a new file."""

from __future__ import annotations

import argparse
import json

from .pipeline import holdout, run


def _print(metrics: dict) -> None:
    shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    print(json.dumps(shown, indent=1, default=str))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="spendcube", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="build the output file for one company")
    p_run.add_argument("--config", required=True, help="company YAML config")
    p_run.add_argument("--data-dir", required=True, help="folder with the company's files")
    p_run.add_argument("--out", required=True, help="output CSV path")
    p_run.add_argument("--reference-dir", help="folder with the company's labelled delivery, for a new delivery of a known company")
    p_run.add_argument("--hide-labels", action="store_true", help="ignore labels: the zero-label path")
    p_run.add_argument("--name", help="W&B run name")

    p_hold = sub.add_parser("holdout", help="score classification on items whose labels are hidden")
    p_hold.add_argument("--config", required=True)
    p_hold.add_argument("--data-dir", required=True)
    p_hold.add_argument("--fraction", type=float, default=0.2)
    p_hold.add_argument("--seed", type=int, default=0)
    p_hold.add_argument("--name")

    p_prof = sub.add_parser("profile", help="summarise an unfamiliar file's columns, to write its config")
    p_prof.add_argument("file")
    p_prof.add_argument("--sep", default=",")
    p_prof.add_argument("--sheet")

    args = parser.parse_args(argv)
    if args.command == "profile":
        import pandas as pd

        from .profile import profile

        with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 200):
            print(profile(args.file, args.sep, args.sheet).to_string(index=False))
    elif args.command == "run":
        _print(run(args.config, args.data_dir, args.out, args.reference_dir, args.hide_labels, args.name))
    else:
        _print(holdout(args.config, args.data_dir, args.fraction, args.seed, args.name))


if __name__ == "__main__":
    main()

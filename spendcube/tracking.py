"""Weights & Biases tracking. Only configs and metrics are sent, never data rows.

SPENDCUBE_WANDB=0 turns tracking off (for reviewers or offline use). When tracking is
on and W&B cannot start, the run stops with a clear message instead of running untracked.
"""

from __future__ import annotations

import os

import pandas as pd


class Tracker:
    def __init__(self, job_type: str, name: str, config: dict, tags: list[str]):
        self.enabled = os.environ.get("SPENDCUBE_WANDB", "1") != "0"
        self.run = None
        if not self.enabled:
            return
        try:
            import wandb

            self.run = wandb.init(project=os.environ.get("SPENDCUBE_WANDB_PROJECT", "spendcube"),
                                  job_type=job_type, name=name, config=config, tags=tags)
        except Exception as exc:
            raise SystemExit(f"W&B tracking could not start ({exc}). Fix W&B access or set "
                             "SPENDCUBE_WANDB=0 to run untracked.") from exc

    def log(self, metrics: dict) -> None:
        if self.run:
            self.run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            self.run.summary.update(metrics)

    def table(self, key: str, frame: pd.DataFrame) -> None:
        if self.run:
            import wandb

            self.run.log({key: wandb.Table(dataframe=frame.reset_index(drop=True))})

    def finish(self, failed: bool = False) -> None:
        if self.run:
            self.run.finish(exit_code=1 if failed else 0)

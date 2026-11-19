"""Category tree read at run time. No category names live in code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .ingest import read_csv_text

MAX_LEVELS = 4


@dataclass(frozen=True)
class Leaf:
    id: str  # short stable ID ('C07') the classifier answers with
    path: tuple[str, ...]  # names from level 1 down to this leaf

    @property
    def text(self) -> str:
        return " > ".join(self.path)


class Taxonomy:
    def __init__(self, leaves: list[Leaf], depth: int):
        self.leaves = leaves
        self.depth = depth
        self.by_id = {leaf.id: leaf for leaf in leaves}
        self.by_path = {leaf.path: leaf for leaf in leaves}

    @classmethod
    def from_csv(cls, path: Path, columns: list[str] | None = None, separator: str | None = None) -> Taxonomy:
        """One row per path, name columns in level order (or one column split by `separator`).

        A path that is a prefix of another is an inner node, not a leaf.
        """
        df = read_csv_text(path)
        if columns:
            df = df[columns]
        if separator:
            df = df.iloc[:, 0].str.split(separator, expand=True, regex=False)
        depth = min(len(df.columns), MAX_LEVELS)
        paths = []
        for row in df.iloc[:, :depth].itertuples(index=False):
            names = [n.strip() for n in row if isinstance(n, str) and n.strip()]
            if names and tuple(names) not in paths:
                paths.append(tuple(names))
        prefixes = {p[:k] for p in paths for k in range(1, len(p))}
        leaf_paths = [p for p in paths if p not in prefixes]
        width = len(str(len(leaf_paths)))
        leaves = [Leaf(f"C{i:0{width}d}", p) for i, p in enumerate(leaf_paths, start=1)]
        return cls(leaves, depth)

    def answer_labels(self) -> dict[str, str]:
        """Leaf ID -> a unique, readable name for the LLM to answer with.

        Opaque IDs ('C02') get mixed up in long answers; the leaf's own name does not. A name
        used by two leaves gets its parent's name in front, then the full path if still shared.
        """
        labels = {}
        for depth_from_leaf in range(1, self.depth + 1):
            for leaf in self.leaves:
                if leaf.id not in labels:
                    labels[leaf.id] = " > ".join(leaf.path[-depth_from_leaf:])
            names = list(labels.values())
            clashes = {n for n in names if names.count(n) > 1}
            if not clashes:
                return labels
            labels = {lid: name for lid, name in labels.items() if name not in clashes}
        return {leaf.id: leaf.text for leaf in self.leaves}

    def leaf_for_labels(self, labels: list) -> Leaf | None:
        names = tuple(n for n in labels if isinstance(n, str) and n)
        return self.by_path.get(names)

    def level_columns(self, leaf_ids: pd.Series) -> pd.DataFrame:
        """Output columns l1..l4 for each leaf ID; blank below the leaf and for missing IDs."""
        cols = {}
        for level in range(MAX_LEVELS):
            names = {lid: (leaf.path[level] if level < len(leaf.path) else "") for lid, leaf in self.by_id.items()}
            cols[f"l{level + 1}"] = leaf_ids.map(names).fillna("")
        return pd.DataFrame(cols, index=leaf_ids.index)

"""Load fixed train/unseen workloads without importing PyTorch."""

import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import Problem


@dataclass(frozen=True)
class Workload:
    name: str
    split: str
    problem: Problem


def load_workloads(path: str | Path, dtype: str = "float32", split: str = "all") -> list[Workload]:
    if split not in {"all", "train", "unseen"}:
        raise ValueError("split must be all, train or unseen")
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("workload file must contain a nonempty JSON list")
    workloads, names, shapes = [], set(), set()
    for entry in entries:
        item = dict(entry)
        name, item_split = item.pop("name"), item.pop("split")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("workload names must be nonempty and unique")
        if item_split not in {"train", "unseen"}:
            raise ValueError("workload split must be train or unseen")
        problem = Problem(dtype=dtype, **item)
        if problem in shapes:
            raise ValueError("duplicate shape across workloads (train/unseen leakage)")
        names.add(name)
        shapes.add(problem)
        if split == "all" or split == item_split:
            workloads.append(Workload(name, item_split, problem))
    if not workloads:
        raise ValueError("no workloads selected")
    return workloads

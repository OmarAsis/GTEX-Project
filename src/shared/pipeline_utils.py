"""Small file and edge-table helpers shared by analysis stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: Any, *, allow_nan: bool = True) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=allow_nan) + "\n")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def edge_keys(edges: Any) -> set[tuple[str, str]]:
    return set(zip(edges["gene_1"].astype(str), edges["gene_2"].astype(str)))


def top_edge_keys(edges: Any, count: int) -> set[tuple[str, str]]:
    ranked = edges.assign(abs_r=edges["r"].abs()).sort_values(
        ["abs_r", "gene_1", "gene_2"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(count)
    return edge_keys(ranked)

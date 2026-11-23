"""Recompute 100 resampling runs and validate stability outputs."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedShuffleSplit

from pipeline.stability_recomputation import (
    SEED,
    THRESHOLD,
    _edge_keys,
    _independent_edges,
    _independent_labels,
    _independent_metrics,
)

from shared.pipeline_utils import write_json
from shared.paths import PROCESSED

OUTPUT = PROCESSED / "stability"
N_REPEATS = 100
TOP_HUBS = 15
PRESERVATION_THRESHOLD = 0.5
CHECKS = 0


def _check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def _match_modules(reference: pd.Series, observed: pd.Series) -> dict[str, str | None]:
    ref_modules = sorted(set(reference) - {"M0"})
    obs_modules = sorted(set(observed) - {"M0"})
    if not ref_modules or not obs_modules:
        return {}
    ref_sets = {module: set(reference.index[reference == module]) for module in ref_modules}
    obs_sets = {module: set(observed.index[observed == module]) for module in obs_modules}
    scores = np.zeros((len(ref_modules), len(obs_modules)), dtype=np.float64)
    for i, ref_module in enumerate(ref_modules):
        for j, obs_module in enumerate(obs_modules):
            union = ref_sets[ref_module] | obs_sets[obs_module]
            scores[i, j] = len(ref_sets[ref_module] & obs_sets[obs_module]) / len(union) if union else 0.0
    rows, columns = linear_sum_assignment(-scores)
    mapping = {module: None for module in obs_modules}
    for row, column in zip(rows, columns):
        mapping[obs_modules[column]] = ref_modules[row]
    return mapping


def _module_rows(repeat: int, split: str, reference: pd.Series,
                 observed: pd.Series, mapping: dict[str, str | None]) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for ref_module in sorted(set(reference) - {"M0"}, key=lambda value: int(value[1:])):
        ref_genes = set(reference.index[reference == ref_module])
        split_module = next((module for module, target in mapping.items() if target == ref_module), None)
        split_genes = set(observed.index[observed == split_module]) if split_module else set()
        overlap = len(ref_genes & split_genes)
        union = ref_genes | split_genes
        jaccard = overlap / len(union) if union else 1.0
        rows.append({
            "repeat": repeat, "split": split, "full_module_id": ref_module,
            "matched_split_module_id": split_module,
            "full_module_size": len(ref_genes), "split_module_size": len(split_genes),
            "overlap_genes": overlap, "jaccard": jaccard,
            "precision": overlap / len(split_genes) if split_genes else 0.0,
            "recall": overlap / len(ref_genes) if ref_genes else 0.0,
            "preserved_jaccard_ge_0_5": bool(jaccard >= PRESERVATION_THRESHOLD),
        })
    mapped = observed.map(lambda value: mapping.get(value) if value != "M0" else "M0").fillna("UNMATCHED")
    return pd.DataFrame(rows), mapped


def _hub_rows(repeat: int, split: str, full_membership: pd.DataFrame,
              observed: pd.Series, mapping: dict[str, str | None], edges: pd.DataFrame) -> pd.DataFrame:
    degree: Counter[str] = Counter()
    for edge in edges.loc[edges["r"] >= THRESHOLD].itertuples(index=False):
        gene_a, gene_b = str(edge.gene_1), str(edge.gene_2)
        module_a, module_b = observed.loc[gene_a], observed.loc[gene_b]
        if module_a == module_b and module_a != "M0":
            degree[gene_a] += 1
            degree[gene_b] += 1
    indexed = full_membership.set_index("gene_id")
    rows = []
    for ref_module in sorted(set(full_membership["module_id"]) - {"M0"}, key=lambda value: int(value[1:])):
        ref_hubs = set(
            indexed.loc[indexed["module_id"] == ref_module]
            .sort_values(["within_module_degree", "gene_id"], ascending=[False, True], kind="mergesort")
            .head(TOP_HUBS).index.astype(str)
        )
        split_module = next((module for module, target in mapping.items() if target == ref_module), None)
        if split_module is None:
            split_hubs: set[str] = set()
        else:
            candidates = [str(gene) for gene in observed.index[observed == split_module]]
            candidates.sort(key=lambda gene: (-degree.get(gene, 0), gene))
            split_hubs = set(candidates[:TOP_HUBS])
        overlap = len(ref_hubs & split_hubs)
        union = ref_hubs | split_hubs
        rows.append({
            "repeat": repeat, "split": split, "full_module_id": ref_module,
            "matched_split_module_id": split_module,
            "full_top_hubs": len(ref_hubs), "split_top_hubs": len(split_hubs),
            "shared_top_hubs": overlap,
            "top_hub_overlap_fraction": overlap / len(ref_hubs) if ref_hubs else 0.0,
            "top_hub_jaccard": overlap / len(union) if union else 1.0,
        })
    return pd.DataFrame(rows)


def _compare_table(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str],
                   numeric: list[str], booleans: list[str], label: str) -> None:
    left = actual.sort_values(keys, kind="mergesort", na_position="last").reset_index(drop=True)
    right = expected.sort_values(keys, kind="mergesort", na_position="last").reset_index(drop=True)
    _check(len(left) == len(right), f"{label}: row count differs")
    for key in keys:
        _check(left[key].fillna("<NA>").astype(str).equals(right[key].fillna("<NA>").astype(str)), f"{label}: key differs ({key})")
    for column in numeric:
        _check(np.allclose(left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float), rtol=1e-10, atol=1e-12), f"{label}: value differs ({column})")
    for column in booleans:
        _check(left[column].astype(bool).equals(right[column].astype(bool)), f"{label}: flag differs ({column})")


def _aggregate(values: pd.Series) -> dict[str, float | int]:
    values = values.dropna().astype(float)
    return {
        "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
        "median": float(values.median()), "p05": float(values.quantile(0.05)),
        "p95": float(values.quantile(0.95)), "min": float(values.min()),
        "max": float(values.max()), "n": int(len(values)),
    }


def run() -> dict:
    global CHECKS
    CHECKS = 0
    summary = json.loads((OUTPUT / "stability_experiments.json").read_text())
    expression = pd.read_parquet(PROCESSED / "muscle_logcpm_corrected_top5000.parquet")
    metadata = pd.read_parquet(PROCESSED / "muscle_metadata.parquet")
    full_edges = pd.read_parquet(PROCESSED / "network_edges_pearson_abs0.65.parquet")
    full_membership = pd.read_parquet(PROCESSED / "module_membership_louvain_res1.0.parquet")
    repeats = pd.read_parquet(OUTPUT / "stability_repeat_metrics.parquet")
    preservation_saved = pd.read_parquet(OUTPUT / "module_preservation_repeats.parquet")
    hubs_saved = pd.read_parquet(OUTPUT / "hub_stability_repeats.parquet")
    assignments_saved = pd.read_parquet(OUTPUT / "gene_module_assignments_repeats.parquet")
    splits_saved = pd.read_parquet(OUTPUT / "sample_split_assignments_repeats.parquet")
    edge_frequency_saved = pd.read_parquet(OUTPUT / "edge_appearance_frequency.parquet")
    consistency_saved = pd.read_parquet(OUTPUT / "gene_module_consistency.parquet")
    edge_bins_saved = pd.read_parquet(OUTPUT / "full_edge_frequency_by_strength.parquet")
    genes = expression.index.astype(str).tolist()
    sample_ids = expression.columns.astype(str).to_numpy()
    strata = metadata["DTHHRDY"].astype("string").fillna("Missing").astype(str).to_numpy()
    reference = full_membership.set_index("gene_id")["module_id"].astype("string").reindex(genes)

    _check(summary["status"] == "PASS", "Repeated experiment summary is not marked PASS")
    _check(summary["design"]["repeats"] == N_REPEATS, "Unexpected number of repeats")
    _check(expression.shape == (5_000, 803) and metadata.index.equals(expression.columns), "Locked input differs")
    _check(len(repeats) == N_REPEATS and repeats["repeat"].astype(int).tolist() == list(range(1, N_REPEATS + 1)), "Repeat metrics are incomplete")
    _check(len(splits_saved) == N_REPEATS * len(sample_ids), "Sample split table has wrong size")
    _check(len(assignments_saved) == 2 * N_REPEATS * len(genes), "Gene assignment table has wrong size")
    module_count = int(reference[reference != "M0"].nunique())
    _check(len(preservation_saved) == 2 * N_REPEATS * module_count, "Module preservation table has wrong size")
    _check(len(hubs_saved) == len(preservation_saved), "Hub table is not complete")

    split_groups = {(int(r), str(s)): group for (r, s), group in splits_saved.groupby(["repeat", "split"], sort=False)}
    assignment_groups = {(int(r), str(s)): group for (r, s), group in assignments_saved.groupby(["repeat", "split"], sort=False)}
    preservation_groups = {(int(r), str(s)): group for (r, s), group in preservation_saved.groupby(["repeat", "split"], sort=False)}
    hub_groups = {(int(r), str(s)): group for (r, s), group in hubs_saved.groupby(["repeat", "split"], sort=False)}
    metric_index = repeats.set_index("repeat")
    edge_presence: Counter[tuple[str, str]] = Counter()
    consistency_count: Counter[str] = Counter()
    expected_preservation: list[pd.DataFrame] = []
    expected_hubs: list[pd.DataFrame] = []
    checked_gene_assignments = 0
    splitter = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=0.5, random_state=SEED)

    for repeat, (idx_a, idx_b) in enumerate(splitter.split(np.zeros(len(sample_ids)), strata), start=1):
        labels_by_split: dict[str, pd.Series] = {}
        edges_by_split: dict[str, pd.DataFrame] = {}
        for split, sample_indices in [("A", idx_a), ("B", idx_b)]:
            sample_subset = sample_ids[sample_indices].tolist()
            sample_group = split_groups[(repeat, split)]
            _check(set(sample_group["sample_id"].astype(str)) == set(sample_subset), f"Sample split mismatch: {repeat}/{split}")
            edges = _independent_edges(expression.loc[:, sample_subset].to_numpy(dtype=np.float64), genes)
            labels = _independent_labels(genes, edges)
            saved_genes = assignment_groups[(repeat, split)].set_index("gene_id").reindex(genes)
            saved_labels = saved_genes["module_id"].astype("string")
            _check(saved_labels.equals(labels), f"Louvain assignments differ: {repeat}/{split}")
            mapping = _match_modules(reference, labels)
            mapped = labels.map(lambda value: mapping.get(value) if value != "M0" else "M0").fillna("UNMATCHED").astype("string")
            saved_mapped = saved_genes["module_matched_to_full"].astype("string").reset_index(drop=True)
            _check(saved_mapped.equals(mapped.reset_index(drop=True)), f"Reference module matching differs: {repeat}/{split}")
            saved_reference = saved_genes["module_full_data"].astype("string").reset_index(drop=True)
            _check(saved_reference.equals(reference.astype("string").reset_index(drop=True)), f"Full-data labels differ: {repeat}/{split}")
            checked_gene_assignments += len(genes)

            module_rows, _ = _module_rows(repeat, split, reference, labels, mapping)
            hub_rows = _hub_rows(repeat, split, full_membership, labels, mapping, edges)
            expected_preservation.append(module_rows)
            expected_hubs.append(hub_rows)
            _compare_table(
                preservation_groups[(repeat, split)], module_rows,
                ["full_module_id", "matched_split_module_id"],
                ["full_module_size", "split_module_size", "overlap_genes", "jaccard", "precision", "recall"],
                ["preserved_jaccard_ge_0_5"], f"Preservation {repeat}/{split}",
            )
            _compare_table(
                hub_groups[(repeat, split)], hub_rows,
                ["full_module_id", "matched_split_module_id"],
                ["full_top_hubs", "split_top_hubs", "shared_top_hubs", "top_hub_overlap_fraction", "top_hub_jaccard"],
                [], f"Hub stability {repeat}/{split}",
            )
            for gene, ref_module, test_module in zip(genes, reference.to_numpy(), mapped.to_numpy()):
                if ref_module == test_module:
                    consistency_count[gene] += 1
            edge_presence.update(_edge_keys(edges))
            labels_by_split[split] = labels
            edges_by_split[split] = edges

        expected_metrics = _independent_metrics(
            edges_by_split["A"], edges_by_split["B"], labels_by_split["A"], labels_by_split["B"],
            reference, len(full_edges),
        )
        observed_metrics = metric_index.loc[repeat]
        _check(set(expected_metrics).issubset(set(observed_metrics.index)), f"Metrics missing: repeat {repeat}")
        for key, expected_value in expected_metrics.items():
            observed_value = observed_metrics[key]
            if expected_value is None:
                _check(pd.isna(observed_value), f"Expected null metric {key}, repeat {repeat}")
            elif isinstance(expected_value, (int, np.integer)):
                _check(int(observed_value) == int(expected_value), f"Metric differs: {key}, repeat {repeat}")
            else:
                _check(np.isclose(float(observed_value), float(expected_value), rtol=1e-10, atol=1e-12), f"Metric differs: {key}, repeat {repeat}")

    _check(checked_gene_assignments == 2 * N_REPEATS * len(genes), "Gene assignments were not fully recomputed")
    expected_preservation_all = pd.concat(expected_preservation, ignore_index=True)
    expected_hubs_all = pd.concat(expected_hubs, ignore_index=True)

    full_keys = _edge_keys(full_edges)
    all_keys = full_keys | set(edge_presence)
    indexed_frequency = edge_frequency_saved.set_index(["gene_1", "gene_2"])
    _check(set(indexed_frequency.index) == all_keys, "Edge frequency table has wrong edge universe")
    for edge in all_keys:
        observed = int(indexed_frequency.loc[edge, "appearances_in_half_networks"])
        _check(observed == edge_presence.get(edge, 0), f"Edge presence differs for {edge}")
    reference_frequency = edge_frequency_saved.loc[edge_frequency_saved["in_full_data_network"]].copy()
    _check(len(reference_frequency) == len(full_edges), "Full-data edge count differs in recurrence table")
    _check(np.allclose(reference_frequency["appearance_frequency"], reference_frequency["appearances_in_half_networks"] / 200), "Edge frequency denominator should be 200")
    full_r = {(str(row.gene_1), str(row.gene_2)): float(row.r) for row in full_edges.itertuples(index=False)}
    for edge in full_keys:
        _check(np.isclose(float(indexed_frequency.loc[edge, "r_full_data"]), full_r[edge], rtol=1e-12, atol=1e-12), f"Full-data edge weight differs: {edge}")
    reference_frequency["strength_bin"] = pd.qcut(
        reference_frequency["abs_r_full_data"].rank(method="first"), q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    expected_bins = reference_frequency.groupby("strength_bin", observed=True).agg(
        edge_count=("gene_1", "size"),
        mean_frequency=("appearance_frequency", "mean"),
        median_frequency=("appearance_frequency", "median"),
        p05_frequency=("appearance_frequency", lambda values: values.quantile(0.05)),
        p95_frequency=("appearance_frequency", lambda values: values.quantile(0.95)),
    ).reset_index()
    _compare_table(
        edge_bins_saved, expected_bins, ["strength_bin"],
        ["edge_count", "mean_frequency", "median_frequency", "p05_frequency", "p95_frequency"],
        [], "Full-edge strength quintiles",
    )

    consistency = consistency_saved.set_index("gene_id").reindex(genes)
    expected_counts = np.asarray([consistency_count.get(gene, 0) for gene in genes], dtype=int)
    _check(np.array_equal(consistency["same_module_half_networks"].to_numpy(dtype=int), expected_counts), "Gene consistency count differs")
    _check((consistency["half_network_count"] == 200).all(), "Gene consistency denominator differs")
    _check(np.allclose(consistency["module_consistency_frequency"], expected_counts / 200), "Gene consistency frequencies differ")
    assigned_mask = reference.to_numpy() != "M0"
    _check(np.array_equal(consistency["is_assigned_full_data"].to_numpy(dtype=bool), assigned_mask), "Assigned-gene indicator differs")

    for column in ["thresholded_edge_jaccard", "top_edge_overlap_fraction", "shared_edge_weight_pearson_r", "shared_edge_sign_concordance"]:
        _check(summary["edge_metrics"][column] == _aggregate(repeats[column]), f"Edge summary differs: {column}")
    for column in ["module_ari_all_genes_m0_as_label", "module_ari_assigned_in_both", "assigned_gene_jaccard", "best_match_module_jaccard_macro"]:
        _check(summary["module_metrics"][column] == _aggregate(repeats[column]), f"Module summary differs: {column}")
    _check(summary["module_preservation"]["module_half_network_observations"] == len(expected_preservation_all), "Module summary row count differs")
    _check(np.isclose(summary["module_preservation"]["median_best_match_jaccard"], expected_preservation_all["jaccard"].median()), "Module preservation median differs")
    _check(np.isclose(summary["hub_stability"]["median_top_hub_jaccard"], expected_hubs_all["top_hub_jaccard"].median()), "Hub median differs")
    assigned_frequency = expected_counts[assigned_mask] / 200
    residual_frequency = expected_counts[~assigned_mask] / 200
    assigned_summary = summary["gene_consistency"]["assigned_full_data_modules"]
    residual_summary = summary["gene_consistency"]["full_data_m0_residual"]
    _check(assigned_summary["genes"] == int(assigned_mask.sum()), "Assigned-gene consistency summary size differs")
    _check(np.isclose(assigned_summary["mean_frequency"], assigned_frequency.mean()), "Assigned-gene consistency mean differs")
    _check(np.isclose(assigned_summary["median_frequency"], np.median(assigned_frequency)), "Assigned-gene consistency median differs")
    _check(assigned_summary["genes_always_consistent"] == int((assigned_frequency == 1).sum()), "Always-consistent assigned-gene count differs")
    _check(assigned_summary["genes_never_consistent"] == int((assigned_frequency == 0).sum()), "Never-consistent assigned-gene count differs")
    _check(residual_summary["genes"] == int((~assigned_mask).sum()), "M0 summary size differs")
    _check(np.isclose(residual_summary["mean_frequency"], residual_frequency.mean()), "M0 consistency mean differs")
    _check(np.isclose(residual_summary["median_frequency"], np.median(residual_frequency)), "M0 consistency median differs")
    _check(summary["edge_frequency"]["full_data_edges"] == len(full_edges), "Edge recurrence summary size differs")

    figures = {
        "stability_metric_distributions.png": (1_000, 700),
        "module_and_hub_preservation.png": (900, 900),
        "edge_recurrence.png": (1_000, 400),
        "gene_module_consistency.png": (700, 400),
    }
    image_results = []
    for name, (min_width, min_height) in figures.items():
        path = OUTPUT / "figures" / name
        _check(path.is_file() and path.stat().st_size > 10_000, f"Figure missing or too small: {name}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            _check(image.width >= min_width and image.height >= min_height, f"Figure dimensions too small: {name}")
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            _check(float(pixels.std()) > 3.0, f"Figure appears blank: {name}")
            image_results.append({"figure": name, "width": image.width, "height": image.height})

    result = {
        "status": "PASS",
        "independent_checks": CHECKS,
        "repeats_recomputed": N_REPEATS,
        "half_networks_recomputed": 2 * N_REPEATS,
        "gene_assignments_recomputed": checked_gene_assignments,
        "edge_universe_checked": len(all_keys),
        "module_preservation_rows_checked": len(expected_preservation_all),
        "hub_rows_checked": len(expected_hubs_all),
        "figures": image_results,
    }
    write_json(OUTPUT / "stability_experiments_verification.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()

"""Repeat stratified split-half network and module stability experiments (v2)."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from pipeline.stability_common import (
    MIN_MODULE_SIZE,
    RANDOM_STATE,
    RESOLUTION,
    THRESHOLD,
    TOP_K,
    _compare_networks,
    _correlation_matrix,
    _module_labels,
    _stratum_labels,
    _threshold_edges,
)
from shared.paths import PROCESSED

os.environ.setdefault("MPLCONFIGDIR", str(PROCESSED / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from shared.pipeline_utils import write_json

OUTPUT = PROCESSED / "stability"
N_REPEATS = 100
TOP_HUBS = 15
PRESERVATION_JACCARD_THRESHOLD = 0.5
FIGURES = OUTPUT / "figures"


def _module_matches(reference: pd.Series, observed: pd.Series) -> dict[str, str | None]:
    """One-to-one maximum-Jaccard mapping from observed modules to full modules."""
    from scipy.optimize import linear_sum_assignment

    reference_modules = sorted(set(reference) - {"M0"})
    observed_modules = sorted(set(observed) - {"M0"})
    if not reference_modules or not observed_modules:
        return {}
    ref_sets = {module: set(reference.index[reference == module]) for module in reference_modules}
    obs_sets = {module: set(observed.index[observed == module]) for module in observed_modules}
    scores = np.zeros((len(reference_modules), len(observed_modules)), dtype=np.float64)
    for i, ref_module in enumerate(reference_modules):
        for j, obs_module in enumerate(observed_modules):
            union = ref_sets[ref_module] | obs_sets[obs_module]
            scores[i, j] = len(ref_sets[ref_module] & obs_sets[obs_module]) / len(union) if union else 0.0
    rows, columns = linear_sum_assignment(-scores)
    mapping = {module: None for module in observed_modules}
    for row, column in zip(rows, columns):
        mapping[observed_modules[column]] = reference_modules[row]
    return mapping


def _preservation_rows(
    repeat: int,
    split: str,
    reference: pd.Series,
    observed: pd.Series,
    mapping: dict[str, str | None],
) -> tuple[list[dict[str, Any]], pd.Series]:
    rows: list[dict[str, Any]] = []
    observed_modules = sorted(set(observed) - {"M0"})
    for ref_module in sorted(set(reference) - {"M0"}, key=lambda value: int(value[1:])):
        reference_genes = set(reference.index[reference == ref_module])
        matched = next((module for module, target in mapping.items() if target == ref_module), None)
        if matched is None:
            overlap = 0
            jaccard = precision = recall = 0.0
            observed_size = 0
        else:
            observed_genes = set(observed.index[observed == matched])
            overlap = len(reference_genes & observed_genes)
            union = reference_genes | observed_genes
            jaccard = overlap / len(union) if union else 1.0
            precision = overlap / len(observed_genes) if observed_genes else 0.0
            recall = overlap / len(reference_genes) if reference_genes else 0.0
            observed_size = len(observed_genes)
        rows.append({
            "repeat": repeat,
            "split": split,
            "full_module_id": ref_module,
            "matched_split_module_id": matched,
            "full_module_size": len(reference_genes),
            "split_module_size": observed_size,
            "overlap_genes": overlap,
            "jaccard": float(jaccard),
            "precision": float(precision),
            "recall": float(recall),
            "preserved_jaccard_ge_0_5": bool(jaccard >= PRESERVATION_JACCARD_THRESHOLD),
        })
    mapped_labels = observed.map(
        lambda module: mapping.get(module) if module != "M0" else "M0"
    ).fillna("UNMATCHED")
    return rows, mapped_labels


def _within_module_degree(edges: pd.DataFrame, labels: pd.Series) -> dict[str, int]:
    degree: Counter[str] = Counter()
    positive = edges.loc[edges["r"] >= THRESHOLD]
    for row in positive.itertuples(index=False):
        module_a = labels.loc[str(row.gene_1)]
        module_b = labels.loc[str(row.gene_2)]
        if module_a == module_b and module_a != "M0":
            degree[str(row.gene_1)] += 1
            degree[str(row.gene_2)] += 1
    return dict(degree)


def _hub_rows(
    repeat: int,
    split: str,
    reference_membership: pd.DataFrame,
    observed: pd.Series,
    mapping: dict[str, str | None],
    edges: pd.DataFrame,
) -> list[dict[str, Any]]:
    degree = _within_module_degree(edges, observed)
    result = []
    ref_index = reference_membership.set_index("gene_id")
    for ref_module in sorted(set(reference_membership["module_id"]) - {"M0"}, key=lambda value: int(value[1:])):
        reference_hubs = set(
            ref_index.loc[ref_index["module_id"] == ref_module]
            .sort_values(["within_module_degree", "gene_id"], ascending=[False, True], kind="mergesort")
            .head(TOP_HUBS)
            .index.astype(str)
        )
        observed_module = next((module for module, target in mapping.items() if target == ref_module), None)
        if observed_module is None:
            observed_hubs: set[str] = set()
        else:
            genes = [str(gene) for gene in observed.index[observed == observed_module]]
            genes.sort(key=lambda gene: (-degree.get(gene, 0), gene))
            observed_hubs = set(genes[:TOP_HUBS])
        overlap = len(reference_hubs & observed_hubs)
        union = reference_hubs | observed_hubs
        result.append({
            "repeat": repeat,
            "split": split,
            "full_module_id": ref_module,
            "matched_split_module_id": observed_module,
            "full_top_hubs": len(reference_hubs),
            "split_top_hubs": len(observed_hubs),
            "shared_top_hubs": overlap,
            "top_hub_overlap_fraction": overlap / len(reference_hubs) if reference_hubs else 0.0,
            "top_hub_jaccard": overlap / len(union) if union else 1.0,
        })
    return result


def _distribution_summary(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().astype(float)
    return {
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)),
        "median": float(clean.median()),
        "p05": float(clean.quantile(0.05)),
        "p95": float(clean.quantile(0.95)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "n": int(len(clean)),
    }


def _plot_distributions(repeats: pd.DataFrame, preservation: pd.DataFrame,
                        hubs: pd.DataFrame, genes: pd.DataFrame,
                        reference_frequency: pd.DataFrame,
                        edge_bins: pd.DataFrame) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    metrics = [
        ("thresholded_edge_jaccard", "Thresholded edge Jaccard"),
        ("top_edge_overlap_fraction", "Top-edge overlap (K=5,000)"),
        ("shared_edge_weight_pearson_r", "Shared-edge weight correlation"),
        ("best_match_module_jaccard_macro", "Best-match module Jaccard"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (column, title) in zip(axes.ravel(), metrics):
        values = repeats[column].dropna().to_numpy(dtype=float)
        ax.hist(values, bins=20, color="#4778a8", alpha=0.9, edgecolor="white")
        median = float(np.median(values))
        ax.axvline(median, color="#c65338", linewidth=1.5, label=f"Median {median:.3f}")
        ax.set(xlabel="Score", ylabel="Split count", title=title)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Network and module stability across {N_REPEATS} stratified half-splits", y=1.01)
    fig.tight_layout()
    name = "stability_metric_distributions.png"
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    names.append(f"stability/figures/{name}")

    module_summary = preservation.groupby("full_module_id")["jaccard"].median().rename("module_jaccard")
    hub_summary = hubs.groupby("full_module_id")["top_hub_jaccard"].median().rename("hub_jaccard")
    summary = pd.concat([module_summary, hub_summary], axis=1).sort_index(
        key=lambda index: index.map(lambda value: int(value[1:]))
    )
    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(9, max(7, 0.35 * len(summary))))
    ax.scatter(summary["module_jaccard"], y - 0.12, color="#4778a8", label="Module membership", s=30)
    ax.scatter(summary["hub_jaccard"], y + 0.12, color="#d57a45", label="Top-15 hubs", s=30)
    ax.set_yticks(y, summary.index)
    ax.set_xlim(0, 1)
    ax.set(xlabel="Median Jaccard across 200 half-networks", ylabel="Full-data module",
           title="Module preservation and hub consistency")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    name = "module_and_hub_preservation.png"
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    names.append(f"stability/figures/{name}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    frequency = reference_frequency["appearance_frequency"].to_numpy(dtype=float)
    axes[0].hist(frequency, bins=np.linspace(0, 1, 21), color="#4778a8", edgecolor="white")
    axes[0].set(xlabel="Fraction of 200 half-networks containing edge", ylabel="Full-data edge count",
                title="Full-data edge appearance frequency")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].errorbar(edge_bins["strength_bin"], edge_bins["median_frequency"],
                     yerr=[edge_bins["median_frequency"] - edge_bins["p05_frequency"],
                           edge_bins["p95_frequency"] - edge_bins["median_frequency"]],
                     fmt="o-", color="#4778a8", capsize=3)
    axes[1].set(xlabel="Full-data |r| quintile", ylabel="Half-network appearance frequency",
                ylim=(0, 1), title="Median recurrence; bars show edge-level 5th–95th percentiles")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    name = "edge_recurrence.png"
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    names.append(f"stability/figures/{name}")

    assigned = genes.loc[genes["module_full_data"] != "M0", "module_consistency_frequency"]
    residual = genes.loc[genes["module_full_data"] == "M0", "module_consistency_frequency"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, values, title, color in [
        (axes[0], assigned, "Genes assigned to full-data modules", "#528b72"),
        (axes[1], residual, "Full-data M0 residual genes", "#8697a8"),
    ]:
        ax.hist(values, bins=np.linspace(0, 1, 21), color=color, edgecolor="white")
        ax.set(xlabel="Fraction of half-networks matching the full-data module", title=title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Gene count")
    fig.suptitle("Gene-level module membership consistency")
    fig.tight_layout()
    name = "gene_module_consistency.png"
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    names.append(f"stability/figures/{name}")
    return names


def run() -> dict[str, Any]:
    expression = pd.read_parquet(PROCESSED / "muscle_logcpm_corrected_top5000.parquet")
    metadata = pd.read_parquet(PROCESSED / "muscle_metadata.parquet")
    full_edges = pd.read_parquet(PROCESSED / "network_edges_pearson_abs0.65.parquet")
    reference_membership = pd.read_parquet(PROCESSED / "module_membership_louvain_res1.0.parquet")
    if expression.shape != (5_000, 803) or not metadata.index.equals(expression.columns):
        raise ValueError("Input matrix or metadata differs from the locked v2 dataset")
    if not expression.index.is_unique or not np.isfinite(expression.to_numpy(dtype=np.float64)).all():
        raise ValueError("Locked expression matrix contains duplicate genes or invalid values")

    genes = expression.index.astype(str).tolist()
    strata = _stratum_labels(metadata).to_numpy()
    sample_ids = expression.columns.astype(str).to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=0.5, random_state=RANDOM_STATE)
    reference_labels = reference_membership.set_index("gene_id")["module_id"].astype("string").reindex(genes)
    if reference_labels.isna().any():
        raise ValueError("Full-data module map does not cover the analysis genes")

    repeat_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    hub_rows: list[dict[str, Any]] = []
    assignment_rows: list[pd.DataFrame] = []
    split_rows: list[pd.DataFrame] = []
    edge_presence: Counter[tuple[str, str]] = Counter()
    module_agreement_count = Counter()
    half_network_count = 0

    for repeat, (idx_a, idx_b) in enumerate(splitter.split(np.zeros(len(sample_ids)), strata), start=1):
        split_labels: dict[str, pd.Series] = {}
        split_edges: dict[str, pd.DataFrame] = {}
        for split, sample_indices in [("A", idx_a), ("B", idx_b)]:
            selected_samples = sample_ids[sample_indices].tolist()
            values = expression.loc[:, selected_samples].to_numpy(dtype=np.float64, copy=False)
            correlation = _correlation_matrix(values)
            edges = _threshold_edges(correlation, genes)
            del correlation
            labels = _module_labels(genes, edges)
            split_edges[split] = edges
            split_labels[split] = labels
            edge_presence.update(zip(edges["gene_1"].astype(str), edges["gene_2"].astype(str)))
            matching = _module_matches(reference_labels, labels)
            preservation, mapped_labels = _preservation_rows(repeat, split, reference_labels, labels, matching)
            preservation_rows.extend(preservation)
            hub_rows.extend(_hub_rows(repeat, split, reference_membership, labels, matching, edges))
            for gene, reference_module, mapped_module in zip(genes, reference_labels.to_numpy(), mapped_labels.to_numpy()):
                if reference_module == mapped_module:
                    module_agreement_count[gene] += 1
            assignment_rows.append(pd.DataFrame({
                "repeat": repeat,
                "split": split,
                "gene_id": genes,
                "module_id": labels.to_numpy(),
                "module_matched_to_full": mapped_labels.to_numpy(),
                "module_full_data": reference_labels.to_numpy(),
            }))
            split_rows.append(pd.DataFrame({
                "repeat": repeat,
                "split": split,
                "sample_id": selected_samples,
                "DTHHRDY_stratum": strata[sample_indices],
            }))
            half_network_count += 1

        metrics = _compare_networks(
            split_edges["A"], split_edges["B"], split_labels["A"], split_labels["B"],
            reference_labels, len(full_edges),
        )
        metrics["repeat"] = repeat
        repeat_rows.append(metrics)
        if repeat % 10 == 0:
            print(f"Completed {repeat}/{N_REPEATS} splits ({half_network_count} half-networks)", flush=True)

    repeats = pd.DataFrame(repeat_rows).sort_values("repeat").reset_index(drop=True)
    preservation = pd.DataFrame(preservation_rows)
    hubs = pd.DataFrame(hub_rows)
    assignments = pd.concat(assignment_rows, ignore_index=True)
    split_assignments = pd.concat(split_rows, ignore_index=True)
    if len(repeats) != N_REPEATS or half_network_count != 2 * N_REPEATS:
        raise AssertionError("The repeated experiment did not complete all requested half-networks")

    ref_pairs = set(zip(full_edges["gene_1"].astype(str), full_edges["gene_2"].astype(str)))
    full_r = {
        (str(row.gene_1), str(row.gene_2)): float(row.r)
        for row in full_edges.itertuples(index=False)
    }
    all_pairs = ref_pairs | set(edge_presence)
    edge_rows = [
        {
            "gene_1": edge[0],
            "gene_2": edge[1],
            "in_full_data_network": edge in ref_pairs,
            "r_full_data": full_r.get(edge),
            "abs_r_full_data": abs(full_r[edge]) if edge in full_r else None,
            "appearances_in_half_networks": int(edge_presence.get(edge, 0)),
            "appearance_frequency": float(edge_presence.get(edge, 0) / half_network_count),
        }
        for edge in sorted(all_pairs)
    ]
    edge_frequency = pd.DataFrame(edge_rows)
    reference_frequency = edge_frequency.loc[edge_frequency["in_full_data_network"]].copy()
    reference_frequency["strength_bin"] = pd.qcut(
        reference_frequency["abs_r_full_data"].rank(method="first"), q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    edge_bins = reference_frequency.groupby("strength_bin", observed=True).agg(
        edge_count=("gene_1", "size"),
        mean_frequency=("appearance_frequency", "mean"),
        median_frequency=("appearance_frequency", "median"),
        p05_frequency=("appearance_frequency", lambda values: values.quantile(0.05)),
        p95_frequency=("appearance_frequency", lambda values: values.quantile(0.95)),
    ).reset_index()
    module_consistency = pd.DataFrame({
        "gene_id": genes,
        "module_full_data": reference_labels.to_numpy(),
        "same_module_half_networks": [int(module_agreement_count.get(gene, 0)) for gene in genes],
    })
    module_consistency["is_assigned_full_data"] = module_consistency["module_full_data"] != "M0"
    module_consistency["half_network_count"] = half_network_count
    module_consistency["module_consistency_frequency"] = (
        module_consistency["same_module_half_networks"] / half_network_count
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    repeats.to_parquet(OUTPUT / "stability_repeat_metrics.parquet", index=False)
    preservation.to_parquet(OUTPUT / "module_preservation_repeats.parquet", index=False)
    hubs.to_parquet(OUTPUT / "hub_stability_repeats.parquet", index=False)
    assignments.to_parquet(OUTPUT / "gene_module_assignments_repeats.parquet", index=False)
    split_assignments.to_parquet(OUTPUT / "sample_split_assignments_repeats.parquet", index=False)
    edge_frequency.to_parquet(OUTPUT / "edge_appearance_frequency.parquet", index=False)
    module_consistency.to_parquet(OUTPUT / "gene_module_consistency.parquet", index=False)
    edge_bins.to_parquet(OUTPUT / "full_edge_frequency_by_strength.parquet", index=False)
    figures = _plot_distributions(
        repeats, preservation, hubs, module_consistency, reference_frequency, edge_bins
    )

    summary = {
        "status": "PASS",
        "design": {
            "repeats": N_REPEATS,
            "half_networks": half_network_count,
            "stratification": "DTHHRDY including Missing",
            "split_seed": RANDOM_STATE,
            "matrix_shape": list(expression.shape),
            "threshold_abs_r": THRESHOLD,
            "top_edge_k": TOP_K,
            "module_edges": "positive Pearson r >= 0.65",
            "louvain_resolution": RESOLUTION,
            "louvain_seed": RANDOM_STATE,
            "module_minimum_size": MIN_MODULE_SIZE,
            "top_hubs_per_module": TOP_HUBS,
            "module_preservation_jaccard_threshold": PRESERVATION_JACCARD_THRESHOLD,
            "preprocessing_scope": "fixed full-data preprocessing and fixed top-5,000 gene universe; conditional stability only",
        },
        "edge_metrics": {
            column: _distribution_summary(repeats[column])
            for column in ["thresholded_edge_jaccard", "top_edge_overlap_fraction",
                           "shared_edge_weight_pearson_r", "shared_edge_sign_concordance"]
        },
        "module_metrics": {
            column: _distribution_summary(repeats[column])
            for column in ["module_ari_all_genes_m0_as_label", "module_ari_assigned_in_both",
                           "assigned_gene_jaccard", "best_match_module_jaccard_macro"]
        },
        "module_preservation": {
            "median_best_match_jaccard": float(preservation["jaccard"].median()),
            "median_recall": float(preservation["recall"].median()),
            "fraction_jaccard_at_least_0_5": float(preservation["preserved_jaccard_ge_0_5"].mean()),
            "module_count": int(preservation["full_module_id"].nunique()),
            "module_half_network_observations": int(len(preservation)),
        },
        "hub_stability": {
            "median_top_hub_overlap_fraction": float(hubs["top_hub_overlap_fraction"].median()),
            "median_top_hub_jaccard": float(hubs["top_hub_jaccard"].median()),
            "module_count": int(hubs["full_module_id"].nunique()),
            "module_half_network_observations": int(len(hubs)),
        },
        "gene_consistency": {
            "assigned_full_data_modules": {
                "genes": int(module_consistency["is_assigned_full_data"].sum()),
                "mean_frequency": float(module_consistency.loc[module_consistency["is_assigned_full_data"], "module_consistency_frequency"].mean()),
                "median_frequency": float(module_consistency.loc[module_consistency["is_assigned_full_data"], "module_consistency_frequency"].median()),
                "genes_always_consistent": int(module_consistency.loc[module_consistency["is_assigned_full_data"], "module_consistency_frequency"].eq(1).sum()),
                "genes_never_consistent": int(module_consistency.loc[module_consistency["is_assigned_full_data"], "module_consistency_frequency"].eq(0).sum()),
            },
            "full_data_m0_residual": {
                "genes": int((~module_consistency["is_assigned_full_data"]).sum()),
                "mean_frequency": float(module_consistency.loc[~module_consistency["is_assigned_full_data"], "module_consistency_frequency"].mean()),
                "median_frequency": float(module_consistency.loc[~module_consistency["is_assigned_full_data"], "module_consistency_frequency"].median()),
            },
        },
        "edge_frequency": {
            "full_data_edges": int(reference_frequency.shape[0]),
            "all_unique_edges_seen": int(edge_frequency.shape[0]),
            "mean_full_edge_frequency": float(reference_frequency["appearance_frequency"].mean()),
            "q05_full_edge_frequency": float(reference_frequency["appearance_frequency"].quantile(0.05)),
            "q95_full_edge_frequency": float(reference_frequency["appearance_frequency"].quantile(0.95)),
            "strength_quintiles": edge_bins.to_dict(orient="records"),
        },
        "outputs": {
            "repeated_metrics": "stability_repeat_metrics.parquet",
            "module_preservation": "module_preservation_repeats.parquet",
            "hub_stability": "hub_stability_repeats.parquet",
            "gene_assignments": "gene_module_assignments_repeats.parquet",
            "gene_consistency": "gene_module_consistency.parquet",
            "edge_frequency": "edge_appearance_frequency.parquet",
            "edge_strength_bins": "full_edge_frequency_by_strength.parquet",
            "sample_splits": "sample_split_assignments_repeats.parquet",
            "figures": figures,
        },
    }
    write_json(OUTPUT / "stability_experiments.json", summary, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return summary


if __name__ == "__main__":
    run()

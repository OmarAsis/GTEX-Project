"""Detect and characterize v2 co-expression modules with weighted Louvain."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from shared.pipeline_utils import write_json
from shared.paths import PROCESSED

THRESHOLD = 0.65
RESOLUTION = 1.0
MIN_MODULE_SIZE = 10
RANDOM_STATE = 42

def _partition_mapping(
    communities: list[set[str]],
    gene_order: list[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    ordered = sorted(communities, key=lambda group: (-len(group), min(group)))
    ordered_modules = [group for group in ordered if len(group) >= MIN_MODULE_SIZE]
    community_by_gene: dict[str, str] = {}
    module_by_gene: dict[str, str] = {}
    size_by_community: dict[str, int] = {}
    for position, group in enumerate(ordered, start=1):
        community_id = f"C{position:04d}"
        size_by_community[community_id] = len(group)
        module_id = None
        if len(group) >= MIN_MODULE_SIZE:
            module_position = ordered_modules.index(group) + 1
            module_id = f"M{module_position:03d}"
        for gene in group:
            community_by_gene[gene] = community_id
            module_by_gene[gene] = module_id or "M0"
    if set(community_by_gene) != set(gene_order):
        raise AssertionError("Partition does not cover the expression matrix gene set")
    return community_by_gene, module_by_gene, size_by_community


def _correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    return float(np.corrcoef(values_a, values_b)[0, 1])


def _module_eigengene(
    expression: pd.DataFrame,
    module_genes: list[str],
) -> tuple[np.ndarray, float, np.ndarray]:
    values = expression.loc[module_genes].to_numpy(dtype=np.float64)
    gene_means = values.mean(axis=1, keepdims=True)
    gene_sd = values.std(axis=1, ddof=1, keepdims=True)
    if (gene_sd == 0).any() or not np.isfinite(gene_sd).all():
        raise ValueError("A module gene has zero or invalid variance")
    standardized = ((values - gene_means) / gene_sd).T
    pca = PCA(n_components=1, svd_solver="randomized", random_state=RANDOM_STATE)
    eigengene = pca.fit_transform(standardized).ravel()
    kme = np.array([_correlation(values[i], eigengene) for i in range(len(module_genes))])
    if np.nanmean(kme) < 0:
        eigengene *= -1
        kme *= -1
    return eigengene, float(pca.explained_variance_ratio_[0]), kme


def run() -> dict[str, Any]:
    expression = pd.read_parquet(PROCESSED / "muscle_logcpm_corrected_top5000.parquet")
    metadata = pd.read_parquet(PROCESSED / "muscle_metadata.parquet")
    symbols = pd.read_parquet(PROCESSED / "gene_symbols.parquet")["symbol"].reindex(expression.index)
    edges = pd.read_parquet(
        PROCESSED / f"network_edges_pearson_abs{THRESHOLD:.2f}.parquet"
    )
    if expression.shape != (5_000, 803):
        raise ValueError(f"Expected locked 5,000 x 803 expression matrix, got {expression.shape}")
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("Expression gene and sample identifiers must be unique")
    if not metadata.index.equals(expression.columns):
        raise ValueError("Metadata does not align with expression samples")
    if expression.isna().any().any() or not np.isfinite(expression.to_numpy()).all():
        raise ValueError("Expression input contains missing or non-finite values")
    endpoints = edges[["gene_1", "gene_2"]].astype(str).to_numpy()
    known_genes = set(expression.index.astype(str))
    if not set(endpoints.ravel()).issubset(known_genes):
        raise ValueError("Network edge table contains genes outside the locked expression matrix")
    if not np.isfinite(edges["r"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Network edge table contains non-finite correlations")
    if not (edges["r"].abs() >= THRESHOLD).all():
        raise ValueError("An edge is below the absolute correlation threshold")
    positive_edges = edges.loc[edges["r"] >= THRESHOLD]

    graph = nx.Graph()
    graph.add_nodes_from(expression.index.astype(str))
    graph.add_edges_from(
        (str(row.gene_1), str(row.gene_2), {"weight": float(row.r)})
        for row in positive_edges.itertuples(index=False)
    )
    if graph.number_of_edges() != len(positive_edges):
        raise AssertionError("Positive graph edge count differs from the unique edge table")

    gene_order = expression.index.astype(str).tolist()
    communities = list(
        nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=RESOLUTION,
            seed=RANDOM_STATE,
        )
    )
    community_by_gene, module_by_gene, size_by_community = _partition_mapping(
        communities, gene_order
    )
    module_sizes = [size for size in size_by_community.values() if size >= MIN_MODULE_SIZE]
    assigned_genes = sum(module_sizes)
    modularity = nx.community.modularity(
        graph, communities, weight="weight", resolution=RESOLUTION
    )
    selected_resolution_result = {
        "resolution": RESOLUTION,
        "modularity_q": float(modularity),
        "community_count_all_sizes": int(len(communities)),
        "module_count_size_at_least_10": int(len(module_sizes)),
        "genes_in_modules_size_at_least_10": int(assigned_genes),
        "fraction_of_all_genes_in_modules": float(assigned_genes / len(gene_order)),
        "largest_module_size": int(max(module_sizes)) if module_sizes else 0,
        "module_size_range": [int(min(module_sizes)), int(max(module_sizes))] if module_sizes else [],
    }
    selected_module_ids = sorted(
        {module for module in module_by_gene.values() if module != "M0"},
        key=lambda module: int(module[1:]),
    )
    graph_degree = dict(graph.degree())
    graph_strength = dict(graph.degree(weight="weight"))
    within_degree = {gene: 0 for gene in gene_order}
    within_strength = {gene: 0.0 for gene in gene_order}
    module_internal_edges: dict[str, list[float]] = {module: [] for module in selected_module_ids}
    module_between_edges: dict[str, list[float]] = {module: [] for module in selected_module_ids}
    module_m0_edges: dict[str, list[float]] = {module: [] for module in selected_module_ids}
    for gene_1, gene_2, data in graph.edges(data=True):
        module_1, module_2 = module_by_gene[gene_1], module_by_gene[gene_2]
        weight = float(data["weight"])
        if module_1 == module_2 and module_1 != "M0":
            within_degree[gene_1] += 1
            within_degree[gene_2] += 1
            within_strength[gene_1] += weight
            within_strength[gene_2] += weight
            module_internal_edges[module_1].append(weight)
        elif module_1 != "M0" and module_2 != "M0":
            module_between_edges[module_1].append(weight)
            module_between_edges[module_2].append(weight)
        else:
            if module_1 != "M0":
                module_m0_edges[module_1].append(weight)
            if module_2 != "M0":
                module_m0_edges[module_2].append(weight)

    eigengene_vectors: dict[str, np.ndarray] = {}
    module_pve: dict[str, float] = {}
    module_kme: dict[str, dict[str, float]] = {}
    for module in selected_module_ids:
        genes = [gene for gene in gene_order if module_by_gene[gene] == module]
        eigengene, pve, kme = _module_eigengene(expression, genes)
        eigengene_vectors[module] = eigengene
        module_pve[module] = pve
        module_kme[module] = dict(zip(genes, kme.astype(float)))
    eigengenes = pd.DataFrame(eigengene_vectors, index=expression.columns)
    eigengenes.to_parquet(
        PROCESSED / f"module_eigengenes_louvain_res{RESOLUTION:.1f}.parquet"
    )
    eigengene_correlation = eigengenes.corr()
    eigengene_correlation.to_parquet(
        PROCESSED / f"module_eigengene_correlations_louvain_res{RESOLUTION:.1f}.parquet"
    )

    gene_values = expression.to_numpy(dtype=np.float64)
    gene_z = (gene_values - gene_values.mean(axis=1, keepdims=True)) / gene_values.std(axis=1, ddof=1, keepdims=True)
    eigengene_values = eigengenes.to_numpy(dtype=np.float64)
    eigengene_z = (eigengene_values - eigengene_values.mean(axis=0, keepdims=True)) / eigengene_values.std(axis=0, ddof=1, keepdims=True)
    all_kme = gene_z @ eigengene_z / (gene_values.shape[1] - 1)
    best_positions = np.argmax(np.abs(all_kme), axis=1)
    mapping_rows = []
    for position, gene in enumerate(gene_order):
        module = module_by_gene[gene]
        own_kme = float(module_kme[module][gene]) if module != "M0" else None
        best_module = str(eigengenes.columns[best_positions[position]])
        best_kme = float(abs(all_kme[position, best_positions[position]]))
        mapping_rows.append({
            "gene_id": gene,
            "symbol": symbols.get(gene, None),
            "community_id": community_by_gene[gene],
            "module_id": module,
            "community_size": size_by_community[community_by_gene[gene]],
            "degree": int(graph_degree[gene]),
            "weighted_degree": float(graph_strength[gene]),
            "within_module_degree": int(within_degree[gene]),
            "within_module_strength": float(within_strength[gene]),
            "fraction_edges_within_module": float(within_degree[gene] / graph_degree[gene]) if graph_degree[gene] and module != "M0" else None,
            "kME": own_kme,
            "best_matching_module": best_module,
            "best_abs_kME": best_kme,
            "boundary_gene": bool(module != "M0" and best_module != module),
            "weak_kME_below_0_5": bool(module != "M0" and own_kme is not None and abs(own_kme) < 0.5),
        })
    membership = pd.DataFrame(mapping_rows)
    membership.to_parquet(
        PROCESSED / f"module_membership_louvain_res{RESOLUTION:.1f}.parquet",
        index=False,
    )

    module_summaries = []
    for module in selected_module_ids:
        genes = membership.loc[membership["module_id"] == module]
        weights_inside = module_internal_edges[module]
        weights_between = module_between_edges[module]
        weights_to_m0 = module_m0_edges[module]
        top_genes = (
            genes.sort_values(
                ["within_module_degree", "within_module_strength", "gene_id"],
                ascending=[False, False, True],
            )
            .head(3)
        )
        module_summaries.append({
            "module_id": module,
            "n_genes": int(len(genes)),
            "n_internal_edges": int(len(weights_inside)),
            "internal_density": float(len(weights_inside) / (len(genes) * (len(genes) - 1) / 2)),
            "mean_intramodule_degree": float(genes["within_module_degree"].mean()),
            "mean_intramodule_strength": float(genes["within_module_strength"].mean()),
            "mean_fraction_edges_within_module": float(genes["fraction_edges_within_module"].dropna().mean()),
            "mean_r_within_module": float(np.mean(weights_inside)) if weights_inside else None,
            "mean_r_between_other_modules": float(np.mean(weights_between)) if weights_between else None,
            "mean_r_to_m0_genes": float(np.mean(weights_to_m0)) if weights_to_m0 else None,
            "mean_kME": float(genes["kME"].mean()),
            "boundary_fraction": float(genes["boundary_gene"].mean()),
            "weak_kME_fraction": float(genes["weak_kME_below_0_5"].mean()),
            "eigengene_variance_explained": module_pve[module],
            "top_hubs": [
                {"symbol": None if pd.isna(row.symbol) else str(row.symbol), "gene_id": row.gene_id,
                 "within_module_degree": int(row.within_module_degree),
                 "kME": float(row.kME)}
                for row in top_genes.itertuples(index=False)
            ],
        })
    module_summary = pd.DataFrame(module_summaries).sort_values("n_genes", ascending=False)
    module_summary.to_parquet(
        PROCESSED / f"module_summary_louvain_res{RESOLUTION:.1f}.parquet",
        index=False,
    )

    result = {
        "status": "PASS",
        "input": {
            "matrix": "muscle_logcpm_corrected_top5000.parquet",
            "shape": list(expression.shape),
            "orientation": "rows=genes, columns=samples",
            "network": "positive Pearson edges only",
            "threshold_r": THRESHOLD,
            "threshold_status": "provisional pending resampling and enrichment",
        },
        "method": {
            "algorithm": "NetworkX weighted Louvain",
            "weight": "positive Pearson r",
            "random_state": RANDOM_STATE,
            "reported_module_minimum_size": MIN_MODULE_SIZE,
            "resolution": RESOLUTION,
            "selected_resolution_for_characterization": RESOLUTION,
        },
        "selected_resolution": {
            **selected_resolution_result,
            "module_ids": selected_module_ids,
            "modules": module_summaries,
            "genes_assigned_to_modules": int(membership["module_id"].ne("M0").sum()),
            "genes_in_m0": int(membership["module_id"].eq("M0").sum()),
            "boundary_genes": int(membership["boundary_gene"].sum()),
            "weak_kme_genes": int(membership["weak_kME_below_0_5"].sum()),
            "maximum_abs_module_eigengene_correlation": float(
                eigengene_correlation.where(~np.eye(len(eigengene_correlation), dtype=bool)).abs().max().max()
            ) if len(eigengene_correlation) > 1 else None,
        },
        "outputs": {
            "membership": f"module_membership_louvain_res{RESOLUTION:.1f}.parquet",
            "summary": f"module_summary_louvain_res{RESOLUTION:.1f}.parquet",
            "eigengenes": f"module_eigengenes_louvain_res{RESOLUTION:.1f}.parquet",
            "eigengene_correlations": f"module_eigengene_correlations_louvain_res{RESOLUTION:.1f}.parquet",
        },
    }
    output = PROCESSED / "module_analysis.json"
    write_json(output, result, allow_nan=False)
    print(json.dumps({
        "status": result["status"],
        "selected_resolution": {
            key: result["selected_resolution"][key]
            for key in ["resolution", "module_ids", "genes_assigned_to_modules", "genes_in_m0", "boundary_genes", "weak_kme_genes", "maximum_abs_module_eigengene_correlation"]
        },
        "modules": module_summaries,
        "outputs": result["outputs"],
    }, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    run()

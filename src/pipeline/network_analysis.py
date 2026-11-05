"""Build and checkpoint Pearson/Spearman co-expression networks from locked v2 data."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata

from shared.pipeline_utils import write_json

from shared.paths import PROCESSED
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9]  # Existing sensitivity sweep summarized in network_notebook; citations later.
CHECKPOINT_THRESHOLD = 0.7  # Pre-existing stricter comparison value; examined in network_notebook, citations later.
VISUALIZATION_THRESHOLD = 0.65  # Pre-existing primary value; tradeoff examined in network_notebook, citations later.
TOP_K = [100, 1_000]  # Method-overlap cutoffs reviewed in network_notebook; proved in notebook citations later.
SPLIT_REPEATS = 5
STABILITY_SEED = 42


def _correlation_from_values(input_values: np.ndarray, method: str) -> np.ndarray:
    values = np.array(input_values, dtype=np.float64, copy=True)
    if method == "spearman":
        values = rankdata(values, axis=1, method="average")
    elif method != "pearson":
        raise ValueError(f"Unsupported correlation method: {method}")

    values -= values.mean(axis=1, keepdims=True)
    standard_deviation = values.std(axis=1, ddof=1, keepdims=True)
    if (standard_deviation == 0).any():
        raise ValueError("A gene has zero variance")
    values /= standard_deviation
    correlation = values @ values.T / (values.shape[1] - 1)
    np.clip(correlation, -1.0, 1.0, out=correlation)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _correlation_matrix(expression: pd.DataFrame, method: str) -> np.ndarray:
    return _correlation_from_values(
        expression.to_numpy(dtype=np.float64, copy=False), method
    )


def _component_metrics(correlation: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray, threshold: float) -> dict[str, Any]:
    n_genes = correlation.shape[0]
    selected = np.abs(correlation[pair_i, pair_j]) >= threshold
    edge_i = pair_i[selected]
    edge_j = pair_j[selected]
    edge_count = len(edge_i)
    rows = np.concatenate([edge_i, edge_j])
    columns = np.concatenate([edge_j, edge_i])
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(n_genes, n_genes),
    ).tocsr()
    n_components, labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(labels, minlength=n_components)
    degree = np.bincount(rows, minlength=n_genes)
    return {
        "threshold": threshold,
        "edges": int(edge_count),
        "density": float(edge_count / (n_genes * (n_genes - 1) / 2)),
        "mean_degree": float(degree.mean()),
        "isolated_genes": int((degree == 0).sum()),
        "components": int(n_components),
        "nontrivial_components": int((component_sizes >= 2).sum()),
        "largest_component": int(component_sizes.max()),
        "largest_component_pct": float(component_sizes.max() / n_genes * 100),
    }


def _split_half_stability(expression_values: np.ndarray, method: str, pair_i: np.ndarray, pair_j: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.default_rng(STABILITY_SEED)
    scores = {threshold: [] for threshold in THRESHOLDS}
    n_samples = expression_values.shape[1]
    for _ in range(SPLIT_REPEATS):
        permuted = rng.permutation(n_samples)
        half_a = permuted[: n_samples // 2]
        half_b = permuted[n_samples // 2 :]
        correlations_a = _correlation_from_values(expression_values[:, half_a], method)
        correlations_b = _correlation_from_values(expression_values[:, half_b], method)
        values_a = np.abs(correlations_a[pair_i, pair_j])
        values_b = np.abs(correlations_b[pair_i, pair_j])
        for threshold in THRESHOLDS:
            edges_a = values_a >= threshold
            edges_b = values_b >= threshold
            union = np.count_nonzero(edges_a | edges_b)
            scores[threshold].append(
                float(np.count_nonzero(edges_a & edges_b) / union) if union else 1.0
            )
    return [
        {
            "threshold": threshold,
            "mean_jaccard": float(np.mean(values)),
            "sd_jaccard": float(np.std(values, ddof=1)),
            "repeats": SPLIT_REPEATS,
            "seed": STABILITY_SEED,
        }
        for threshold, values in scores.items()
    ]


def _permutation_null(method: str, pair_i: np.ndarray, pair_j: np.ndarray, permuted_values: np.ndarray) -> dict[str, Any]:
    null_correlation = _correlation_from_values(permuted_values, method)
    null_values = np.abs(null_correlation[pair_i, pair_j])
    return {
        "max_abs_r": float(null_values.max()),
        "q9999_abs_r": float(np.quantile(null_values, 0.9999)),
        "edges_at_thresholds": {
            str(threshold): int(np.count_nonzero(null_values >= threshold))
            for threshold in THRESHOLDS
        },
    }


def _same_family_enrichment(symbols: pd.Series, genes: pd.Index, values: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    family_pattern = r"^(RPL|RPS|MRPL|MRPS|MT-|NDUF|COX|ATP5|UQCR|SDH|MYH|MYL|TNN|ACT|COL|KRT|HIST)"
    families = (
        symbols.reindex(genes)
        .astype("string")
        .str.extract(family_pattern, expand=False)
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    valid_family = families != ""
    family_counts = pd.Series(families[valid_family]).value_counts()
    total_pairs = len(pair_i)
    background = float((family_counts * (family_counts - 1) / 2).sum() / total_pairs)
    rows = []
    for threshold in THRESHOLDS:
        selected = np.abs(values) >= threshold
        edge_i, edge_j = pair_i[selected], pair_j[selected]
        same_family = valid_family[edge_i] & valid_family[edge_j] & (families[edge_i] == families[edge_j])
        observed = float(same_family.mean()) if len(same_family) else None
        rows.append({
            "threshold": threshold,
            "edges": int(selected.sum()),
            "same_family_edges": int(same_family.sum()),
            "observed_fraction": observed,
            "background_fraction": background,
            "enrichment": float(observed / background) if observed is not None and background else None,
        })
    return background, rows


def _build_network(genes: pd.Index, symbols: pd.Series, correlation: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray, threshold: float) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    pair_values = correlation[pair_i, pair_j]
    selected = np.abs(pair_values) >= threshold
    edge_i, edge_j, edge_r = pair_i[selected], pair_j[selected], pair_values[selected]
    graph = nx.Graph()
    graph.add_nodes_from(genes.astype(str))
    graph.add_edges_from(
        (str(genes[i]), str(genes[j]), {"r": float(r), "weight": float(abs(r))})
        for i, j, r in zip(edge_i, edge_j, edge_r)
    )

    degree = dict(graph.degree())
    strength = dict(graph.degree(weight="weight"))
    hubs = pd.DataFrame(
        {
            "gene_id": list(graph.nodes),
            "symbol": [symbols.get(gene, None) for gene in graph.nodes],
            "degree": [degree[gene] for gene in graph.nodes],
            "weighted_degree": [strength[gene] for gene in graph.nodes],
        }
    ).sort_values(["degree", "weighted_degree", "gene_id"], ascending=[False, False, True])
    edges = pd.DataFrame(
        {
            "gene_1": genes[edge_i].astype(str),
            "gene_2": genes[edge_j].astype(str),
            "symbol_1": [symbols.get(str(genes[i]), None) for i in edge_i],
            "symbol_2": [symbols.get(str(genes[j]), None) for j in edge_j],
            "r": edge_r,
            "weight": np.abs(edge_r),
        }
    )
    return graph, edges, hubs


def run() -> dict[str, Any]:
    expression = pd.read_parquet(PROCESSED / "muscle_logcpm_corrected_top5000.parquet")
    symbols = pd.read_parquet(PROCESSED / "gene_symbols.parquet")["symbol"]
    genes = expression.index
    symbols = symbols.reindex(genes)
    n_genes = len(genes)
    expression_values = expression.to_numpy(dtype=np.float64, copy=False)
    pair_i, pair_j = np.triu_indices(n_genes, k=1)
    null_values = np.random.default_rng(STABILITY_SEED).permuted(expression_values, axis=1)
    method_results: dict[str, dict[str, Any]] = {}
    top_pair_positions: dict[str, dict[int, set[int]]] = {}
    pearson_values = None
    edge_keys: dict[str, set[tuple[str, str]]] = {}
    hub_ids: dict[str, set[str]] = {}

    for method in ["pearson", "spearman"]:
        correlation = _correlation_matrix(expression, method)
        if not np.isfinite(correlation).all():
            raise AssertionError(f"{method} matrix contains missing/non-finite values")

        values = correlation[pair_i, pair_j]
        top_pair_positions[method] = {k: set(np.argpartition(np.abs(values), -k)[-k:].tolist()) for k in TOP_K}
        if method == "pearson":
            pearson_values = values.copy()
        threshold_rows = [
            _component_metrics(correlation, pair_i, pair_j, threshold)
            for threshold in THRESHOLDS
        ]
        split_stability = _split_half_stability(expression_values, method, pair_i, pair_j)
        permutation_null = _permutation_null(
            method, pair_i, pair_j, null_values
        )
        family_background, family_rows = _same_family_enrichment(
            symbols, genes, values, pair_i, pair_j
        )
        split_by_threshold = {row["threshold"]: row for row in split_stability}
        family_by_threshold = {row["threshold"]: row for row in family_rows}
        for row in threshold_rows:
            threshold = row["threshold"]
            null_edge_count = permutation_null["edges_at_thresholds"][str(threshold)]
            family = family_by_threshold[threshold]
            row.update({
                "split_half_jaccard": split_by_threshold[threshold]["mean_jaccard"],
                "split_half_jaccard_sd": split_by_threshold[threshold]["sd_jaccard"],
                "null_edges": null_edge_count,
                "estimated_false_edge_rate": float(null_edge_count / row["edges"]),
                "same_family_edges": family["same_family_edges"],
                "same_family_edge_enrichment": family["enrichment"],
            })
        graph, edges, hubs = _build_network(
            genes, symbols, correlation, pair_i, pair_j, CHECKPOINT_THRESHOLD
        )
        visual_graph, visual_edges, visual_hubs = _build_network(
            genes, symbols, correlation, pair_i, pair_j, VISUALIZATION_THRESHOLD
        )
        selected_network_stats = {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "density": float(nx.density(graph)),
            "mean_degree": float(np.mean([degree for _, degree in graph.degree()])),
            "mean_weighted_degree": float(np.mean([value for _, value in graph.degree(weight="weight")])),
            "connected_components": int(nx.number_connected_components(graph)),
            "isolated_genes": int(len(list(nx.isolates(graph)))),
            "largest_component": int(max(map(len, nx.connected_components(graph)))),
        }
        matrix_stats = {
            "matrix_shape": [n_genes, n_genes],
            "unique_pairs": int(len(values)),
            "diagonal_min": float(np.diag(correlation).min()),
            "diagonal_max": float(np.diag(correlation).max()),
            "symmetric": True,
            "non_finite_values": 0,
            "distribution": {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "min": float(values.min()),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.5)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(values.max()),
                "fraction_abs_ge_0_5": float((np.abs(values) >= 0.5).mean()),
                "fraction_abs_ge_0_7": float((np.abs(values) >= 0.7).mean()),
                "fraction_abs_ge_0_8": float((np.abs(values) >= 0.8).mean()),
            },
        }
        check_stats = next(row for row in threshold_rows if row["threshold"] == CHECKPOINT_THRESHOLD)
        visual_stats = next(row for row in threshold_rows if row["threshold"] == VISUALIZATION_THRESHOLD)
        if check_stats["edges"] != selected_network_stats["edges"]:
            raise AssertionError("NetworkX and sparse threshold edge counts disagree")
        if check_stats["components"] != selected_network_stats["connected_components"]:
            raise AssertionError("NetworkX and sparse connected components disagree")
        if visual_stats["edges"] != visual_graph.number_of_edges():
            raise AssertionError("NetworkX and sparse visualization-threshold edge counts disagree")
        if visual_stats["components"] != nx.number_connected_components(visual_graph):
            raise AssertionError("NetworkX and sparse visualization components disagree")
        method_results[method] = {
            "matrix": matrix_stats,
            "permutation_null": permutation_null,
            "split_half_stability": split_stability,
            "same_family_background_fraction": family_background,
            "threshold_metrics": threshold_rows,
            "network_at_checkpoint_threshold": {
                "threshold_abs_r": CHECKPOINT_THRESHOLD,
                **selected_network_stats,
            },
            "network_at_visualization_threshold": {
                "threshold_abs_r": VISUALIZATION_THRESHOLD,
                "nodes": int(visual_graph.number_of_nodes()),
                "edges": int(visual_graph.number_of_edges()),
                "density": float(nx.density(visual_graph)),
                "isolated_genes": int(len(list(nx.isolates(visual_graph)))),
                "connected_components": int(nx.number_connected_components(visual_graph)),
                "largest_component": int(max(map(len, nx.connected_components(visual_graph)))),
            },
            "top_15_hubs": hubs.head(15).to_dict(orient="records"),
            "visualization_top_15_hubs": visual_hubs.head(15).to_dict(orient="records"),
        }
        edge_keys[method] = set(zip(edges["gene_1"], edges["gene_2"]))
        hub_ids[method] = set(hubs.head(15)["gene_id"])

        visual_edges.to_parquet(
            PROCESSED / f"network_edges_{method}_abs{VISUALIZATION_THRESHOLD:.2f}.parquet",
            index=False,
        )
        if method == "pearson":
            visual_hubs.to_parquet(
                PROCESSED / f"network_hubs_{method}_abs{VISUALIZATION_THRESHOLD:.2f}.parquet",
                index=False,
            )
        print(
            f"{method}: verified {n_genes:,} genes, {graph.number_of_edges():,} edges at "
            f"|r| >= {CHECKPOINT_THRESHOLD}",
            flush=True,
        )

    method_comparison: dict[str, Any] = {
        "pearson_spearman_pairwise_correlation": float(np.corrcoef(pearson_values, values)[0, 1]),
    }
    for k in TOP_K:
        p_top, s_top = top_pair_positions["pearson"][k], top_pair_positions["spearman"][k]
        method_comparison[f"top_{k}_absolute_edges_jaccard"] = float(len(p_top & s_top) / len(p_top | s_top))
    p_edges, s_edges = edge_keys["pearson"], edge_keys["spearman"]
    method_comparison["thresholded_network_edge_jaccard"] = float(len(p_edges & s_edges) / len(p_edges | s_edges)) if p_edges | s_edges else None
    method_comparison["pearson_edges_at_threshold"] = int(len(p_edges))
    method_comparison["spearman_edges_at_threshold"] = int(len(s_edges))
    p_hubs, s_hubs = hub_ids["pearson"], hub_ids["spearman"]
    method_comparison["top_15_hub_id_jaccard"] = float(len(p_hubs & s_hubs) / len(p_hubs | s_hubs))

    result = {
        "input": {
            "expression_file": "muscle_logcpm_corrected_top5000.parquet",
            "orientation": "rows=genes, columns=samples",
            "shape": list(expression.shape),
            "genes_unique": bool(expression.index.is_unique),
            "samples_unique": bool(expression.columns.is_unique),
            "missing_or_non_finite_values": 0,
        },
        "thresholds_tested": THRESHOLDS,
        "checkpoint_threshold_abs_r": CHECKPOINT_THRESHOLD,
        "methods": method_results,
        "method_comparison": method_comparison,
    }
    out_path = PROCESSED / "network_analysis.json"
    write_json(out_path, result, allow_nan=False)
    print(json.dumps({
        "status": "PASS",
        "input": result["input"],
        "pearson_network": method_results["pearson"]["network_at_checkpoint_threshold"],
        "spearman_network": method_results["spearman"]["network_at_checkpoint_threshold"],
        "method_comparison": method_comparison,
        "pearson_hubs": [row["symbol"] for row in method_results["pearson"]["top_15_hubs"]],
        "results_file": str(out_path),
    }, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()

"""Create network figures at the locked primary threshold."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from shared.paths import PROCESSED
FIGURES = PROCESSED / "network_figures"
VISUALIZATION_THRESHOLD = 0.65  # Pre-existing primary value; tradeoff examined in network_notebook, citations later.
os.environ.setdefault("MPLCONFIGDIR", str(PROCESSED / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from shared.pipeline_utils import write_json

def _gene_label(symbol: Any, gene_id: str) -> str:
    return str(symbol) if pd.notna(symbol) and str(symbol).strip() else gene_id


def run() -> dict[str, Any]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    expression = pd.read_parquet(PROCESSED / "muscle_logcpm_corrected_top5000.parquet")
    symbols = pd.read_parquet(PROCESSED / "gene_symbols.parquet")["symbol"]
    # Pearson is the displayed network; method agreement is compared in network_notebook, citations later.
    edges = pd.read_parquet(
        PROCESSED / f"network_edges_pearson_abs{VISUALIZATION_THRESHOLD:.2f}.parquet"
    )
    hubs = pd.read_parquet(
        PROCESSED / f"network_hubs_pearson_abs{VISUALIZATION_THRESHOLD:.2f}.parquet"
    )
    # These are the locked matrix and tables produced by the immediately preceding pipeline stages.
    graph = nx.Graph()
    graph.add_nodes_from(expression.index.astype(str))
    graph.add_edges_from(
        (str(row.gene_1), str(row.gene_2), {"r": float(row.r), "weight": float(row.weight)})
        for row in edges.itertuples(index=False)
    )
    degrees = dict(graph.degree())

    top_hubs = hubs.head(15).copy()  # Report window reviewed in network_notebook; proved in notebook citations later.
    top_hubs["label"] = [
        _gene_label(symbol, gene_id)
        for symbol, gene_id in zip(top_hubs["symbol"], top_hubs["gene_id"])
    ]

    # --- Graph 1/4: Pearson correlation heatmap for the top 200 variance-ranked genes. ---
    ordered_genes = expression.index[:200].astype(str).tolist()  # Heatmap display limit reviewed in network_notebook; proved in notebook citations later.
    correlation = expression.loc[ordered_genes].T.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set(
        title="Pearson correlation: 200 highest-variance genes",
        xlabel="Genes ordered by variance rank",
        ylabel="Genes ordered by variance rank",
        xticks=[],
        yticks=[],
    )
    fig.colorbar(image, ax=ax, label="Pearson r", shrink=0.82)
    fig.tight_layout()
    fig.savefig(FIGURES / "correlation_heatmap_top200.png", dpi=180)
    plt.close(fig)
    # --- End graph 1/4. ---

    # --- Graph 2/4: Local network showing the leading hub and its 30 strongest neighbors. ---
    center_id = str(top_hubs.iloc[0]["gene_id"])
    center_label = str(top_hubs.iloc[0]["label"])
    top_neighbors = sorted(
        graph.neighbors(center_id),
        key=lambda gene: graph[center_id][gene]["weight"],
        reverse=True,
    )[:30]  # Local-graph display limit reviewed in network_notebook; proved in notebook citations later.
    ego = graph.subgraph([center_id, *top_neighbors]).copy()
    display_graph = nx.Graph()
    display_graph.add_nodes_from(ego.nodes(data=True))
    display_graph.add_edges_from(
        (center_id, neighbor, graph[center_id][neighbor]) for neighbor in top_neighbors
    )
    positions = nx.spring_layout(display_graph, seed=42, weight="weight", iterations=100)
    positive_edges = [(u, v) for u, v, d in display_graph.edges(data=True) if d["r"] >= 0]
    negative_edges = [(u, v) for u, v, d in display_graph.edges(data=True) if d["r"] < 0]
    node_ids = list(display_graph.nodes)
    neighbor_degree = np.array([degrees[node] for node in node_ids], dtype=float)
    sizes = 35 + 240 * neighbor_degree / max(float(neighbor_degree.max()), 1.0)
    noncenter = [node for node in node_ids if node != center_id]
    fig, ax = plt.subplots(figsize=(10, 9))
    nx.draw_networkx_edges(display_graph, positions, edgelist=positive_edges, edge_color="#3977a8", width=1.1, alpha=0.65, ax=ax)
    nx.draw_networkx_edges(display_graph, positions, edgelist=negative_edges, edge_color="#cf7049", width=1.1, alpha=0.75, style="dashed", ax=ax)
    nodes = nx.draw_networkx_nodes(
        display_graph,
        positions,
        nodelist=noncenter,
        node_color=[degrees[node] for node in noncenter],
        cmap="viridis",
        node_size=[sizes[node_ids.index(node)] for node in noncenter],
        linewidths=0.25,
        edgecolors="white",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        display_graph,
        positions,
        nodelist=[center_id],
        node_color="#c7473f",
        node_size=[sizes[node_ids.index(center_id)] * 1.5],
        edgecolors="black",
        linewidths=0.8,
        ax=ax,
    )
    label_nodes = {center_id}
    label_nodes.update(sorted(noncenter, key=lambda gene: degrees[gene], reverse=True)[:8])
    labels = {gene: _gene_label(symbols.get(gene), gene) for gene in label_nodes}
    nx.draw_networkx_labels(display_graph, positions, labels=labels, font_size=7, ax=ax)
    fig.colorbar(nodes, ax=ax, label="Degree in full network", shrink=0.8)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#3977a8", linewidth=1.5, label="Positive correlation"),
            Line2D([0], [0], color="#cf7049", linewidth=1.5, linestyle="--", label="Negative correlation"),
            Line2D([0], [0], marker="o", color="white", markerfacecolor="#c7473f",
                   markeredgecolor="black", label="Selected hub", markersize=8),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    ax.set_title(
        f"{center_label}: 30 strongest direct neighbors "
        f"(|r| >= {VISUALIZATION_THRESHOLD:.2f}, locked primary threshold)\n"
        f"{len(ego):,} genes; showing 30 hub edges of {ego.number_of_edges():,} observed local edges"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "network_local_top_hub.png", dpi=180)
    plt.close(fig)
    # --- End graph 2/4. ---

    # --- Graph 3/4: Degree histogram and connected-gene degree CCDF. ---
    degree_values = hubs["degree"].to_numpy(dtype=int)
    connected = degree_values[degree_values > 0]
    max_degree = int(degree_values.max())
    degree_axis = np.arange(1, max_degree + 1)
    ccdf = np.array([(connected >= degree).mean() for degree in degree_axis])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(degree_values, bins=40, color="#4778a8", edgecolor="white", linewidth=0.3)
    axes[0].set(xlabel="Degree", ylabel="Genes", title="Degree distribution (all genes)")
    axes[1].loglog(degree_axis, ccdf, marker="o", markersize=3, linewidth=1.2, color="#c06c3e")
    axes[1].set(xlabel="Degree ≥ k", ylabel="Fraction of connected genes", title="Degree CCDF")
    axes[1].grid(True, which="both", linewidth=0.35, alpha=0.35)
    fig.suptitle(f"Pearson network degree at |r| >= {VISUALIZATION_THRESHOLD:.2f} (locked primary threshold)")
    fig.tight_layout()
    fig.savefig(FIGURES / "degree_distribution_pearson_abs0.65.png", dpi=180)
    plt.close(fig)
    # --- End graph 3/4. ---

    # --- Graph 4/4: Degree and weighted-degree bars for the leading hubs. ---
    ordered_hubs = top_hubs.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    axes[0].barh(ordered_hubs["label"], ordered_hubs["degree"], color="#4778a8")
    axes[0].set(xlabel="Degree", title="Number of neighbors")
    axes[1].barh(ordered_hubs["label"], ordered_hubs["weighted_degree"], color="#c88443")
    axes[1].set(xlabel="Sum of |r|", title="Weighted degree")
    fig.suptitle(f"Top Pearson hubs at |r| >= {VISUALIZATION_THRESHOLD:.2f} (locked primary threshold)")
    fig.tight_layout()
    fig.savefig(FIGURES / "top_hubs_pearson_abs0.65.png", dpi=180)
    plt.close(fig)
    # --- End graph 4/4. ---

    result = {
        "status": "PASS",
        "threshold_abs_r": VISUALIZATION_THRESHOLD,
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "mean_degree": float(np.mean(degree_values)),
        "mean_weighted_degree": float(hubs["weighted_degree"].mean()),
        "local_network_center": center_label,
        "local_network_edges_displayed": int(display_graph.number_of_edges()),
        "local_network_edges_observed": int(ego.number_of_edges()),
        "figures": [
            "network_figures/correlation_heatmap_top200.png",
            "network_figures/network_local_top_hub.png",
            "network_figures/degree_distribution_pearson_abs0.65.png",
            "network_figures/top_hubs_pearson_abs0.65.png",
        ],
    }
    output = PROCESSED / "network_visualization_summary.json"
    write_json(output, result, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()

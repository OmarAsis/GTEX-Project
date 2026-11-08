"""Create the locked v2 module, enrichment, network, and hub figures."""

from __future__ import annotations

import json
import os
from pathlib import Path
import textwrap
from typing import Any

import numpy as np
import pandas as pd

from shared.paths import PROCESSED
OUTPUT = PROCESSED / "module_visualizations"
NETWORK_THRESHOLD = 0.65
TOP_NETWORK_NODES = 18
LABEL_NETWORK_HUBS = 6
TOP_HEATMAP_GENES_PER_MODULE = 25
REPRESENTATIVE_MODULES = ["M001", "M002", "M004", "M006", "M009", "M019"]
os.environ.setdefault("MPLCONFIGDIR", str(PROCESSED / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from shared.pipeline_utils import write_json

def _module_order(values: pd.Series) -> list[str]:
    return sorted(values.unique(), key=lambda value: int(value[1:]))


def _save_heatmap(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    membership: pd.DataFrame,
    eigengenes: pd.DataFrame,
) -> dict[str, Any]:
    module_ids = _module_order(membership.loc[membership["module_id"] != "M0", "module_id"])
    ordered_members = membership.loc[membership["module_id"] != "M0"].copy()
    ordered_members["module_order"] = ordered_members["module_id"].map({m: i for i, m in enumerate(module_ids)})
    ordered_members = ordered_members.sort_values(
        ["module_order", "kME", "within_module_degree", "gene_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    eligible_gene_count = len(ordered_members)
    ordered_members = ordered_members.groupby("module_id", sort=False, group_keys=False).head(
        TOP_HEATMAP_GENES_PER_MODULE
    )
    genes = ordered_members["gene_id"].astype(str).tolist()
    values = expression.loc[genes].to_numpy(dtype=np.float64)
    means = values.mean(axis=1, keepdims=True)
    sds = values.std(axis=1, ddof=1, keepdims=True)
    if (sds == 0).any() or not np.isfinite(sds).all():
        raise ValueError("A module heatmap gene has zero or invalid variance")
    z = np.clip((values - means) / sds, -3, 3)

    sample_matrix = eigengenes.loc[expression.columns].to_numpy(dtype=np.float64)
    distances = pdist(sample_matrix, metric="correlation")
    if not np.isfinite(distances).all():
        raise ValueError("Sample clustering distances from module eigengenes are non-finite")
    sample_order_positions = leaves_list(linkage(distances, method="average"))
    sample_order = expression.columns[sample_order_positions]
    z = z[:, sample_order_positions]

    metadata_ordered = metadata.loc[sample_order]
    death_class = metadata_ordered["DTHHRDY"].astype("string").fillna("missing")
    categories = sorted(death_class.unique().tolist())
    palette = plt.get_cmap("tab10")
    category_colors = {category: palette(i % 10) for i, category in enumerate(categories)}
    strip = np.asarray(
        [[category_colors[value] for value in death_class]], dtype=np.float64
    ).reshape(1, len(death_class), 4)

    fig, (ax_strip, ax) = plt.subplots(
        2, 1, figsize=(17, 12), sharex=True,
        gridspec_kw={"height_ratios": [0.12, 5], "hspace": 0.04},
    )
    ax_strip.imshow(strip, aspect="auto", interpolation="nearest")
    ax_strip.set_yticks([])
    ax_strip.set_title("DTHHRDY", loc="left", fontsize=9)
    ax_strip.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    image = ax.imshow(z, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto", interpolation="nearest")
    module_sizes = ordered_members["module_id"].value_counts().reindex(module_ids)
    boundaries = module_sizes.cumsum().to_numpy()[:-1]
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="black", linewidth=0.55, alpha=0.9)
    centers = module_sizes.cumsum() - module_sizes / 2
    ax.set_yticks(centers.to_numpy())
    ax.set_yticklabels(module_ids, fontsize=8)
    ax.set(xlabel="Samples (hierarchical order from module eigengenes)",
           ylabel=f"Top {TOP_HEATMAP_GENES_PER_MODULE} genes per module, ordered by kME")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    fig.colorbar(image, ax=ax, label="Gene expression z-score", shrink=0.7, pad=0.01)
    legend = [Patch(facecolor=category_colors[c], label=f"DTHHRDY {c}") for c in categories]
    fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 6),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Representative v2 module expression heatmap (gene-wise z-score, clipped at ±3)",
                 y=0.99)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, hspace=0.08)
    name = "module_expression_heatmap.png"
    fig.savefig(OUTPUT / name, dpi=180)
    plt.close(fig)
    return {
        "figure": name,
        "genes": len(genes),
        "eligible_module_genes": eligible_gene_count,
        "top_genes_per_module": TOP_HEATMAP_GENES_PER_MODULE,
        "samples": len(sample_order),
        "modules": module_ids,
        "sample_order_method": "average-linkage hierarchical clustering on module eigengene profiles using correlation distance",
        "death_class_categories": categories,
    }


def _save_module_sizes(module_summary: pd.DataFrame) -> str:
    ordered = module_summary.sort_values("n_genes", ascending=True, kind="mergesort")
    fig, ax = plt.subplots(figsize=(9, max(6, 0.33 * len(ordered))))
    ax.barh(ordered["module_id"], ordered["n_genes"], color="#4778a8")
    ax.set(xlabel="Genes", ylabel="Module", title="V2 Louvain module sizes (size ≥ 10)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    name = "module_sizes.png"
    fig.savefig(OUTPUT / name, dpi=180)
    plt.close(fig)
    return name


def _save_eigengene_similarity(eigengene_correlations: pd.DataFrame) -> str:
    ordered = eigengene_correlations
    if len(ordered) > 1:
        tree = linkage(pdist(ordered.to_numpy(), metric="correlation"), method="average")
        order = leaves_list(tree)
        ordered = ordered.iloc[order, order]
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(ordered)), max(6, 0.38 * len(ordered))))
    image = ax.imshow(ordered.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered)), ordered.columns, rotation=90)
    ax.set_yticks(range(len(ordered)), ordered.index)
    ax.set(title="Module eigengene similarity", xlabel="Module", ylabel="Module")
    fig.colorbar(image, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    name = "module_eigengene_similarity.png"
    fig.savefig(OUTPUT / name, dpi=180)
    plt.close(fig)
    return name


def _save_expression_patterns(
    eigengenes: pd.DataFrame,
    metadata: pd.DataFrame,
) -> str:
    sample_order = metadata.sort_values(["DTHHRDY", "SMTSISCH"], na_position="last").index
    expression_pattern = eigengenes.loc[sample_order].T
    sample_groups = metadata.loc[sample_order, "DTHHRDY"].astype(str).to_numpy()
    boundaries = np.flatnonzero(sample_groups[1:] != sample_groups[:-1]) + 1
    fig, ax = plt.subplots(figsize=(12, max(5, 0.32 * len(expression_pattern))))
    image = ax.imshow(expression_pattern.to_numpy(), cmap="coolwarm", vmin=-3, vmax=3,
                       aspect="auto", interpolation="nearest")
    for boundary in boundaries:
        ax.axvline(boundary - 0.5, color="black", linewidth=0.35, alpha=0.65)
    ax.set(
        title="Module eigengene patterns across samples (ordered by death classification)",
        xlabel="Samples; vertical lines separate DTHHRDY categories",
        ylabel="Module",
        yticks=range(len(expression_pattern)),
        yticklabels=expression_pattern.index,
    )
    fig.colorbar(image, ax=ax, label="Eigengene score", shrink=0.8)
    fig.tight_layout()
    name = "module_expression_patterns.png"
    fig.savefig(OUTPUT / name, dpi=180)
    plt.close(fig)
    return name


def _save_enrichment_dotplot(enrichment: pd.DataFrame) -> dict[str, Any]:
    significant = enrichment.loc[enrichment["significant_global_all_tests"]].copy()
    selected_rows = []
    for module, group in significant.groupby("module_id", sort=False):
        selected_rows.append(group.sort_values("fdr_global_all_tests").head(2))
    top = pd.concat(selected_rows, ignore_index=True) if selected_rows else significant.head(0)
    if top.empty:
        raise ValueError("No globally significant enrichment terms are available to plot")
    module_ids = sorted(top["module_id"].unique(), key=lambda value: int(value[1:]))
    module_x = {module: position for position, module in enumerate(module_ids)}
    top = top.sort_values(["module_id", "fdr_global_all_tests"],
                          key=lambda col: col.map(lambda value: int(value[1:])) if col.name == "module_id" else col,
                          ascending=[True, True], kind="mergesort").reset_index(drop=True)
    top["y"] = np.arange(len(top))
    top["minus_log10_q"] = -np.log10(np.maximum(top["fdr_global_all_tests"].to_numpy(), np.finfo(float).tiny))
    labels = [
        textwrap.fill(f"{row.module_id} · {row.library}: {row.term}", width=78)
        for row in top.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(20, max(10, 0.42 * len(top))))
    scatter = ax.scatter(
        [module_x[module] for module in top["module_id"]],
        top["y"],
        s=35 + 13 * np.sqrt(top["overlap"].to_numpy(dtype=float)),
        c=np.minimum(top["minus_log10_q"].to_numpy(dtype=float), 40),
        cmap="viridis",
        norm=Normalize(vmin=0, vmax=40),
        edgecolors="white",
        linewidths=0.4,
    )
    ax.set_yticks(top["y"])
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xticks(range(len(module_ids)), module_ids, rotation=45, ha="right")
    ax.set(xlabel="Module", ylabel="Top terms per module (global FDR ranking)",
           title="Enrichment dot plot: top two globally adjusted terms per module")
    ax.grid(axis="x", alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="−log10(global FDR), clipped at 40", pad=0.01)
    fig.subplots_adjust(left=0.48, right=0.92, top=0.96, bottom=0.09)
    name = "enrichment_dotplot_top_terms.png"
    fig.savefig(OUTPUT / name, dpi=180)
    plt.close(fig)
    return {"figure": name, "modules": len(module_ids), "terms_plotted": int(len(top)),
            "max_terms_per_module": 2, "fdr_scope": "all tested module x library x term hypotheses"}


def _save_representative_networks(
    membership: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict[str, Any]:
    positive = edges.loc[edges["r"] >= NETWORK_THRESHOLD]
    graph = nx.Graph()
    graph.add_edges_from(
        (str(row.gene_1), str(row.gene_2), {"weight": float(row.r)})
        for row in positive.itertuples(index=False)
    )
    graph.add_nodes_from(membership["gene_id"].astype(str))
    indexed = membership.set_index("gene_id")
    fig, axes = plt.subplots(3, 2, figsize=(15, 17))
    normalize = Normalize(vmin=0, vmax=1)
    mappable = plt.cm.ScalarMappable(norm=normalize, cmap="viridis")
    network_summaries = []
    for ax, module in zip(axes.ravel(), REPRESENTATIVE_MODULES):
        module_genes = membership.loc[membership["module_id"] == module]
        chosen = module_genes.sort_values(
            ["within_module_degree", "within_module_strength", "gene_id"],
            ascending=[False, False, True], kind="mergesort",
        ).head(TOP_NETWORK_NODES)["gene_id"].astype(str).tolist()
        subgraph = graph.subgraph(chosen).copy()
        positions = nx.spring_layout(subgraph, seed=42, weight="weight", k=0.8)
        rows = indexed.loc[chosen]
        degrees = rows["within_module_degree"].to_numpy(dtype=float)
        maximum = max(float(degrees.max()), 1.0) if len(degrees) else 1.0
        sizes = 90 + 280 * degrees / maximum
        kme = rows["kME"].fillna(0).to_numpy(dtype=float)
        for gene_1, gene_2, data in subgraph.edges(data=True):
            nx.draw_networkx_edges(
                subgraph, positions, edgelist=[(gene_1, gene_2)], ax=ax,
                width=0.35 + 3.0 * (float(data["weight"]) - NETWORK_THRESHOLD),
                alpha=0.35, edge_color="#667085",
            )
        nodes = nx.draw_networkx_nodes(
            subgraph, positions, nodelist=chosen, node_size=sizes,
            node_color=kme, cmap="viridis", vmin=0, vmax=1,
            edgecolors="white", linewidths=0.5, ax=ax,
        )
        labels_to_show = rows.sort_values(
            ["within_module_degree", "gene_id"], ascending=[False, True], kind="mergesort"
        ).head(LABEL_NETWORK_HUBS)
        labels = {
            str(gene_id): str(row["symbol"]) if pd.notna(row["symbol"]) else str(gene_id)
            for gene_id, row in labels_to_show.iterrows()
        }
        nx.draw_networkx_labels(subgraph, positions, labels=labels, font_size=7, ax=ax)
        ax.set_title(
            f"{module}: {len(module_genes)} genes; top {len(chosen)} shown; "
            f"{subgraph.number_of_edges()} induced edges at r ≥ {NETWORK_THRESHOLD:.2f}",
            fontsize=9,
        )
        ax.axis("off")
        network_summaries.append({
            "module_id": module,
            "module_genes": int(len(module_genes)),
            "nodes_shown": int(subgraph.number_of_nodes()),
            "edges_shown": int(subgraph.number_of_edges()),
            "top_labelled_hubs": list(labels.values()),
        })
    fig.subplots_adjust(left=0.04, right=0.88, top=0.95, bottom=0.04,
                        wspace=0.08, hspace=0.27)
    colorbar_axis = fig.add_axes([0.92, 0.29, 0.018, 0.48])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis)
    colorbar.set_label("Gene kME with own module eigengene")
    fig.suptitle("Representative positive co-expression module networks", y=0.995)
    name = "representative_module_networks.png"
    fig.savefig(OUTPUT / name, dpi=180)
    plt.close(fig)
    return {"figure": name, "threshold_r": NETWORK_THRESHOLD,
            "top_nodes_per_module": TOP_NETWORK_NODES, "networks": network_summaries}


def _save_module_hubs(membership: pd.DataFrame) -> dict[str, Any]:
    module_ids = _module_order(membership.loc[membership["module_id"] != "M0", "module_id"])
    selected = []
    for module in module_ids:
        selected.append(
            membership.loc[membership["module_id"] == module]
            .sort_values(["within_module_degree", "within_module_strength", "gene_id"],
                         ascending=[False, False, True], kind="mergesort")
            .head(3)
            .assign(hub_rank=lambda frame: np.arange(1, len(frame) + 1))
        )
    hubs = pd.concat(selected, ignore_index=True)
    hubs["label"] = hubs.apply(
        lambda row: f"{row.module_id} · #{int(row.hub_rank)} · {row.symbol if pd.notna(row.symbol) else row.gene_id}",
        axis=1,
    )
    rank_colors = {1: "#234e70", 2: "#4778a8", 3: "#90b5d8"}
    fig, ax = plt.subplots(figsize=(11, max(11, 0.24 * len(hubs))))
    ax.barh(hubs["label"], hubs["within_module_degree"],
            color=[rank_colors[int(rank)] for rank in hubs["hub_rank"]])
    ax.invert_yaxis()
    ax.set(xlabel="Edges within the assigned module", ylabel="Module hub",
           title="Top three intramodule-degree hubs in every v2 module")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(handles=[Patch(color=rank_colors[i], label=f"Rank {i}") for i in rank_colors],
              loc="lower right", frameon=False)
    fig.tight_layout()
    name = "module_top_hubs.png"
    fig.savefig(OUTPUT / name, dpi=180)
    plt.close(fig)
    return {"figure": name, "modules": len(module_ids), "hubs_plotted": int(len(hubs)),
            "top_hubs": hubs[["module_id", "hub_rank", "symbol", "gene_id", "within_module_degree"]].to_dict(orient="records")}


def run() -> dict[str, Any]:
    expression = pd.read_parquet(PROCESSED / "muscle_logcpm_corrected_top5000.parquet")
    metadata = pd.read_parquet(PROCESSED / "muscle_metadata.parquet")
    membership = pd.read_parquet(PROCESSED / "module_membership_louvain_res1.0.parquet")
    module_summary = pd.read_parquet(PROCESSED / "module_summary_louvain_res1.0.parquet")
    eigengenes = pd.read_parquet(PROCESSED / "module_eigengenes_louvain_res1.0.parquet")
    eigengene_correlations = pd.read_parquet(
        PROCESSED / "module_eigengene_correlations_louvain_res1.0.parquet"
    )
    edges = pd.read_parquet(PROCESSED / "network_edges_pearson_abs0.65.parquet")
    enrichment = pd.read_parquet(PROCESSED / "enrichment" / "module_enrichment_all_tests.parquet")
    if expression.shape != (5_000, 803):
        raise ValueError(f"Expected locked matrix 5,000 x 803, got {expression.shape}")
    if not metadata.index.equals(expression.columns):
        raise ValueError("Sample metadata does not align with expression columns")
    if not eigengenes.index.equals(expression.columns):
        raise ValueError("Module eigengenes do not align with expression samples")
    if membership["gene_id"].nunique() != 5_000:
        raise ValueError("Every selected gene must have exactly one module assignment")
    if set(enrichment["module_id"].unique()) != set(module_summary["module_id"]):
        raise ValueError("Enrichment results do not match the v2 module IDs")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    heatmap = _save_heatmap(expression, metadata, membership, eigengenes)
    size_plot = _save_module_sizes(module_summary)
    eigengene_similarity = _save_eigengene_similarity(eigengene_correlations)
    expression_patterns = _save_expression_patterns(eigengenes, metadata)
    enrichment_plot = _save_enrichment_dotplot(enrichment)
    networks = _save_representative_networks(membership, edges)
    hubs = _save_module_hubs(membership)
    figures = [
        f"module_visualizations/{heatmap['figure']}",
        f"module_visualizations/{size_plot}",
        f"module_visualizations/{eigengene_similarity}",
        f"module_visualizations/{expression_patterns}",
        f"module_visualizations/{enrichment_plot['figure']}",
        f"module_visualizations/{networks['figure']}",
        f"module_visualizations/{hubs['figure']}",
    ]
    report = {
        "status": "PASS",
        "input": {
            "matrix": "muscle_logcpm_corrected_top5000.parquet",
            "shape": list(expression.shape),
            "module_membership": "module_membership_louvain_res1.0.parquet",
            "network_edges": "network_edges_pearson_abs0.65.parquet",
            "enrichment": "enrichment/module_enrichment_all_tests.parquet",
            "module_count": int(module_summary["module_id"].nunique()),
            "module_genes": int(membership["module_id"].ne("M0").sum()),
            "threshold_r": NETWORK_THRESHOLD,
            "threshold_and_resolution_status": "provisional; later resampling and method comparison remain",
        },
        "heatmap": heatmap,
        "module_size_plot": size_plot,
        "eigengene_similarity_plot": eigengene_similarity,
        "expression_patterns_plot": expression_patterns,
        "enrichment_dotplot": enrichment_plot,
        "representative_networks": networks,
        "hub_plot": hubs,
        "figures": figures,
    }
    write_json(OUTPUT / "module_visualizations.json", report, allow_nan=False)
    print(json.dumps({
        "status": report["status"],
        "input": report["input"],
        "heatmap": report["heatmap"],
        "enrichment_dotplot": report["enrichment_dotplot"],
        "representative_networks": report["representative_networks"],
        "hub_plot": {key: value for key, value in report["hub_plot"].items() if key != "top_hubs"},
        "figures": report["figures"],
    }, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    run()

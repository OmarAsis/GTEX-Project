"""Run reproducible ORA for locked v2 co-expression modules."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.paths import PROCESSED
OUTPUT = PROCESSED / "enrichment"
FIGURES = PROCESSED / "enrichment_figures"
MATRIX_NAME = "muscle_logcpm_corrected_top5000.parquet"
MIN_TERM_SIZE = 5
MAX_TERM_SIZE = 500
MIN_OVERLAP_TO_REPORT = 3
FDR_ALPHA = 0.05
os.environ.setdefault("MPLCONFIGDIR", str(PROCESSED / "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import gseapy as gp
from scipy import sparse
from scipy.stats import hypergeom, spearmanr

from shared.pipeline_utils import write_json

LIBRARIES = {
    "GO_BP": {
        "name": "GO_Biological_Process_2023",
        "source": "Gene Ontology Consortium data served by Enrichr",
        "url": "https://geneontology.org/docs/go-citation-policy/",
    },
    "GO_MF": {
        "name": "GO_Molecular_Function_2023",
        "source": "Gene Ontology Consortium data served by Enrichr",
        "url": "https://geneontology.org/docs/go-citation-policy/",
    },
    "GO_CC": {
        "name": "GO_Cellular_Component_2023",
        "source": "Gene Ontology Consortium data served by Enrichr",
        "url": "https://geneontology.org/docs/go-citation-policy/",
    },
    "KEGG": {
        "name": "KEGG_2021_Human",
        "source": "KEGG human pathway library served by Enrichr",
        "url": "https://www.kegg.jp/kegg/",
    },
    "Reactome": {
        "name": "Reactome_2022",
        "source": "Reactome pathway library served by Enrichr",
        "url": "https://reactome.org/download/",
    },
    "Hallmark": {
        "name": "MSigDB_Hallmark_2020",
        "source": "MSigDB Hallmark collection served by Enrichr",
        "url": "https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp",
    },
}


def _load_gene_sets(library_name: str) -> tuple[dict[str, set[str]], dict[str, Any]]:
    cache_file = Path.home() / ".cache" / "gseapy" / f"Enrichr.{library_name}.gmt"
    if cache_file.is_file():
        gene_sets = gp.read_gmt(str(cache_file))
        source_type = "GSEApy local Enrichr cache"
        digest = hashlib.sha256(cache_file.read_bytes()).hexdigest()
        file_bytes = cache_file.stat().st_size
    else:
        gene_sets = gp.get_library(library_name, organism="Human")
        source_type = "GSEApy/Enrichr library download"
        file_bytes = None
        if cache_file.is_file():
            digest = hashlib.sha256(cache_file.read_bytes()).hexdigest()
            file_bytes = cache_file.stat().st_size
        else:
            digest = None
    normalized = {str(term): set(map(str, genes)) for term, genes in gene_sets.items()}
    if not normalized:
        raise ValueError(f"Gene-set library {library_name} is empty")
    return normalized, {
        "library_name": library_name,
        "source_type": source_type,
        "source_url": "https://maayanlab.cloud/Enrichr/",
        "source_file_sha256": digest,
        "source_file_bytes": file_bytes,
        "terms_loaded": int(len(normalized)),
    }


def _term_matrix(
    gene_sets: dict[str, set[str]], background: list[str]
) -> tuple[list[str], sparse.csr_matrix, np.ndarray, list[set[str]]]:
    positions = {symbol: position for position, symbol in enumerate(background)}
    terms: list[str] = []
    term_genes: list[set[str]] = []
    row_indices: list[int] = []
    column_indices: list[int] = []
    for term in sorted(gene_sets):
        genes = gene_sets[term] & positions.keys()
        if MIN_TERM_SIZE <= len(genes) <= MAX_TERM_SIZE:
            term_position = len(terms)
            terms.append(term)
            term_genes.append(genes)
            for gene in genes:
                row_indices.append(term_position)
                column_indices.append(positions[gene])
    matrix = sparse.csr_matrix(
        (np.ones(len(row_indices), dtype=np.int8), (row_indices, column_indices)),
        shape=(len(terms), len(background)),
    )
    sizes = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    return terms, matrix, sizes, term_genes


def _run_ora_for_library(
    library_key: str,
    terms: list[str],
    matrix: sparse.csr_matrix,
    term_sizes: np.ndarray,
    term_genes: list[set[str]],
    background: list[str],
    module_symbols: dict[str, list[str]],
) -> pd.DataFrame:
    background_size = len(background)
    rows: list[dict[str, Any]] = []
    for module_id, symbols in module_symbols.items():
        members = set(symbols) & set(background)
        overlap_counts = np.asarray(matrix @ np.isin(background, list(members))).ravel().astype(int)
        module_size = len(members)
        p_values = hypergeom.sf(overlap_counts - 1, background_size, term_sizes, module_size)
        if not np.isfinite(p_values).all():
            raise AssertionError(f"Non-finite enrichment p-value in {library_key}/{module_id}")
        for i, count in enumerate(overlap_counts):
            overlaps = sorted(members & term_genes[i])
            rows.append({
                "module_id": module_id,
                "library": library_key,
                "library_name": LIBRARIES[library_key]["name"],
                "term": terms[i],
                "background_genes": background_size,
                "module_genes": module_size,
                "term_genes_in_background": int(term_sizes[i]),
                "overlap": int(count),
                "overlap_genes": ";".join(overlaps),
                "p_value": float(p_values[i]),
                "fold_enrichment": float(
                    (count / module_size) / (term_sizes[i] / background_size)
                ) if module_size else None,
            })
    return pd.DataFrame(rows)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, in the original input order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("BH input must be a finite one-dimensional vector in [0, 1]")
    if not len(values):
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    adjusted = values[order] * len(values) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _write_figures(
    module_summary: pd.DataFrame,
    significance_counts: pd.DataFrame,
    library_keys: list[str],
) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_files: list[str] = []
    counts = significance_counts.reindex(columns=library_keys).fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.36 * len(counts))))
    image = ax.imshow(counts.to_numpy(), cmap="magma", aspect="auto")
    ax.set_xticks(range(len(counts.columns)), counts.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(counts.index)), counts.index)
    ax.set(title="Enrichment terms after global FDR correction", xlabel="Gene-set library", ylabel="Module")
    fig.colorbar(image, ax=ax, label="Terms (global FDR < 0.05; overlap ≥ 3)")
    fig.tight_layout()
    name = "global_fdr_terms_by_module_library.png"
    fig.savefig(FIGURES / name, dpi=180)
    plt.close(fig)
    figure_files.append(f"enrichment_figures/{name}")

    total_hits = counts.sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(module_summary["module_genes"], total_hits.reindex(module_summary["module_id"]), color="#4778a8")
    label_rows = module_summary.loc[module_summary["global_fdr_significant_terms"] >= 35]
    for row in label_rows.itertuples(index=False):
        ax.annotate(row.module_id, (row.module_genes, total_hits.get(row.module_id, 0)),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set(xscale="log", xlabel="Module genes (log scale)", ylabel="Globally FDR-significant terms",
           title="Enrichment yield and module size (labels: ≥35 hits)")
    fig.tight_layout()
    name = "global_fdr_hits_vs_module_size.png"
    fig.savefig(FIGURES / name, dpi=180)
    plt.close(fig)
    figure_files.append(f"enrichment_figures/{name}")
    return figure_files


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def run() -> dict[str, Any]:
    expression = pd.read_parquet(PROCESSED / MATRIX_NAME)
    symbols = pd.read_parquet(PROCESSED / "gene_symbols.parquet")["symbol"].reindex(expression.index)
    membership = pd.read_parquet(PROCESSED / "module_membership_louvain_res1.0.parquet")
    if expression.shape != (5_000, 803):
        raise ValueError(f"Expected locked 5,000 x 803 matrix, got {expression.shape}")
    if not expression.index.is_unique or symbols.isna().any():
        raise ValueError("Selected expression genes must have unique IDs and a symbol mapping")
    if symbols.duplicated().any():
        raise ValueError("Selected expression gene symbols are not one-to-one; resolve aliases before ORA")
    if membership["gene_id"].nunique() != len(expression.index):
        raise ValueError("Module membership must map every selected gene exactly once")
    if set(membership["gene_id"]) != set(expression.index.astype(str)):
        raise ValueError("Module membership IDs do not match the selected expression matrix")

    background = sorted(symbols.astype(str).unique().tolist())
    gene_to_symbol = symbols.astype(str).to_dict()
    module_ids = sorted(
        membership.loc[membership["module_id"] != "M0", "module_id"].unique(),
        key=lambda value: int(value[1:]),
    )
    module_symbols = {
        module_id: sorted({gene_to_symbol[gene] for gene in membership.loc[membership["module_id"] == module_id, "gene_id"]})
        for module_id in module_ids
    }
    if any(len(values) < 10 for values in module_symbols.values()):
        raise ValueError("A characterized module has fewer than 10 unique symbols")
    if len(set().union(*(set(values) for values in module_symbols.values()))) > len(background):
        raise AssertionError("Module symbol union exceeds the selected background")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    library_metadata: dict[str, Any] = {}
    library_gene_unions: dict[str, set[str]] = {}
    all_results: list[pd.DataFrame] = []
    for library_key, config in LIBRARIES.items():
        gene_sets, metadata = _load_gene_sets(config["name"])
        library_gene_unions[library_key] = set().union(
            *(set(genes) for genes in gene_sets.values())
        ) & set(background)
        terms, matrix, term_sizes, term_genes = _term_matrix(gene_sets, background)
        if not terms:
            raise ValueError(f"No {library_key} terms passed the {MIN_TERM_SIZE}–{MAX_TERM_SIZE} background-size filter")
        library_metadata[library_key] = {
            **metadata,
            "source": config["source"],
            "citation_url": config["url"],
            "terms_tested_per_module": len(terms),
            "genes_annotated_in_background": int(len(library_gene_unions[library_key])),
        }
        result = _run_ora_for_library(
            library_key, terms, matrix, term_sizes, term_genes, background, module_symbols
        )
        all_results.append(result)

    results = pd.concat(all_results, ignore_index=True)
    results["fdr_module_library"] = results.groupby(["module_id", "library"], sort=False)["p_value"].transform(_bh_fdr)
    results["fdr_library_all_modules"] = results.groupby("library", sort=False)["p_value"].transform(_bh_fdr)
    results["fdr_global_all_tests"] = _bh_fdr(results["p_value"].to_numpy())
    results["reported_overlap"] = results["overlap"] >= MIN_OVERLAP_TO_REPORT
    results["significant_module_library"] = results["reported_overlap"] & (results["fdr_module_library"] < FDR_ALPHA)
    results["significant_library_all_modules"] = results["reported_overlap"] & (results["fdr_library_all_modules"] < FDR_ALPHA)
    results["significant_global_all_tests"] = results["reported_overlap"] & (results["fdr_global_all_tests"] < FDR_ALPHA)
    results = results.sort_values(["p_value", "library", "module_id", "term"], kind="mergesort").reset_index(drop=True)

    all_results_path = OUTPUT / "module_enrichment_all_tests.parquet"
    results.to_parquet(all_results_path, index=False)
    reportable = results.loc[results["reported_overlap"]].copy()
    reportable.to_csv(OUTPUT / "module_enrichment_reportable.csv", index=False)
    global_hits = results.loc[results["significant_global_all_tests"]].copy()
    global_hits.to_csv(OUTPUT / "module_enrichment_global_fdr_hits.csv", index=False)

    library_summary_rows = []
    for library_key in LIBRARIES:
        sub = results.loc[results["library"] == library_key]
        library_summary_rows.append({
            "library": library_key,
            "library_name": LIBRARIES[library_key]["name"],
            "terms_tested_per_module": library_metadata[library_key]["terms_tested_per_module"],
            "modules_tested": len(module_ids),
            "tests_total": len(sub),
            "raw_p_lt_0_05_all_tests": int((sub["p_value"] < FDR_ALPHA).sum()),
            "raw_p_lt_0_05_overlap_ge_3": int((sub["reported_overlap"] & (sub["p_value"] < FDR_ALPHA)).sum()),
            "nominal_expected_raw_hits_if_uniform": float(FDR_ALPHA * len(sub)),
            "module_library_fdr_lt_0_05_overlap_ge_3": int(sub["significant_module_library"].sum()),
            "library_fdr_lt_0_05_overlap_ge_3": int(sub["significant_library_all_modules"].sum()),
            "global_fdr_lt_0_05_overlap_ge_3": int(sub["significant_global_all_tests"].sum()),
        })
    library_summary = pd.DataFrame(library_summary_rows)
    library_summary.to_parquet(OUTPUT / "enrichment_library_summary.parquet", index=False)

    module_summary_rows = []
    for module_id in module_ids:
        sub = results.loc[results["module_id"] == module_id]
        reportable_sub = sub.loc[sub["reported_overlap"]]
        top_terms = global_hits.loc[global_hits["module_id"] == module_id].sort_values("fdr_global_all_tests").head(5)
        module_summary_rows.append({
            "module_id": module_id,
            "module_genes": len(module_symbols[module_id]),
            "tests_total": len(sub),
            "reportable_terms_overlap_ge_3": len(reportable_sub),
            "module_library_fdr_significant_terms": int(sub["significant_module_library"].sum()),
            "library_global_fdr_significant_terms": int(sub["significant_library_all_modules"].sum()),
            "global_fdr_significant_terms": int(sub["significant_global_all_tests"].sum()),
            "best_global_fdr": float(sub["fdr_global_all_tests"].min()),
            "top_global_terms": [
                {"library": row.library, "term": row.term, "overlap": int(row.overlap),
                 "fold_enrichment": float(row.fold_enrichment), "fdr": float(row.fdr_global_all_tests),
                 "overlap_genes": row.overlap_genes}
                for row in top_terms.itertuples(index=False)
            ],
        })
    module_summary = pd.DataFrame(module_summary_rows)

    all_annotated = set().union(*library_gene_unions.values())
    module_summary["module_symbols_annotated_any_library"] = module_summary["module_id"].map(
        {module_id: len(set(values) & all_annotated) for module_id, values in module_symbols.items()}
    )
    module_summary["module_annotation_fraction"] = (
        module_summary["module_symbols_annotated_any_library"] / module_summary["module_genes"]
    )
    module_summary.to_parquet(OUTPUT / "enrichment_module_summary.parquet", index=False)

    size_term_correlation = spearmanr(
        module_summary["module_genes"], module_summary["global_fdr_significant_terms"]
    )
    significance_counts = (
        global_hits.groupby(["module_id", "library"]).size().unstack(fill_value=0)
        .reindex(index=module_ids, columns=list(LIBRARIES), fill_value=0)
    )
    figure_files = _write_figures(module_summary, significance_counts, list(LIBRARIES))
    report = {
        "status": "PASS",
        "input": {
            "matrix": MATRIX_NAME,
            "shape": list(expression.shape),
            "orientation": "rows=genes, columns=samples",
            "module_assignments": "module_membership_louvain_res1.0.parquet",
            "module_resolution": 1.0,
            "threshold_r": 0.65,
            "threshold_and_resolution_status": "provisional pending stability and comparative analysis",
        },
        "method": {
            "test": "one-sided hypergeometric over-representation test",
            "background": "all 5,000 unique selected gene symbols",
            "background_size": len(background),
            "module_count": len(module_ids),
            "min_term_size_in_background": MIN_TERM_SIZE,
            "max_term_size_in_background": MAX_TERM_SIZE,
            "minimum_reported_overlap": MIN_OVERLAP_TO_REPORT,
            "background_includes_selected_genes_unannotated_in_a_library": True,
            "fdr_alpha": FDR_ALPHA,
            "correction_scopes": [
                "within each module x library",
                "within each library across all modules",
                "across every tested module x library x term",
            ],
            "p_value_correction_applied_before_overlap_reporting_filter": True,
            "raw_gene_set_files_copied_into_project": False,
            "gseapy_version": gp.__version__,
        },
        "libraries": library_metadata,
        "totals": {
            "tests_total": int(len(results)),
            "reportable_terms_overlap_ge_3": int(results["reported_overlap"].sum()),
            "raw_p_lt_0_05_all_tests": int((results["p_value"] < FDR_ALPHA).sum()),
            "raw_p_lt_0_05_overlap_ge_3": int((results["reported_overlap"] & (results["p_value"] < FDR_ALPHA)).sum()),
            "module_library_fdr_hits": int(results["significant_module_library"].sum()),
            "library_fdr_hits": int(results["significant_library_all_modules"].sum()),
            "global_fdr_hits": int(results["significant_global_all_tests"].sum()),
            "nominal_expected_raw_hits_if_uniform": float(FDR_ALPHA * len(results)),
            "nominal_expectation_note": "Reference only; discrete hypergeometric p-values are not exactly uniform.",
        },
        "module_size_vs_global_fdr_term_count_spearman": {
            "rho": float(size_term_correlation.statistic),
            "p_value": float(size_term_correlation.pvalue),
            "interpretation": "descriptive only; module size affects power and library coverage varies",
        },
        "outputs": {
            "all_tests": str(all_results_path.relative_to(PROCESSED)),
            "reportable_terms": "enrichment/module_enrichment_reportable.csv",
            "global_fdr_hits": "enrichment/module_enrichment_global_fdr_hits.csv",
            "library_summary": "enrichment/enrichment_library_summary.parquet",
            "module_summary": "enrichment/enrichment_module_summary.parquet",
            "figures": figure_files,
        },
    }
    write_json(OUTPUT / "enrichment_analysis.json", _safe_json(report), allow_nan=False)
    print(json.dumps({
        "status": report["status"],
        "input": report["input"],
        "method": report["method"],
        "library_summary": library_summary.to_dict(orient="records"),
        "module_summary": module_summary[[
            "module_id", "module_genes", "module_symbols_annotated_any_library",
            "global_fdr_significant_terms", "best_global_fdr",
        ]].to_dict(orient="records"),
        "totals": report["totals"],
        "module_size_vs_global_fdr_term_count_spearman": report["module_size_vs_global_fdr_term_count_spearman"],
        "outputs": report["outputs"],
    }, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    run()

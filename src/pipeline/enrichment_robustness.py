"""Background sensitivity check for v2 module enrichment."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from pipeline.enrichment_analysis import (
    LIBRARIES,
    _apply_fdr_scopes,
    _load_gene_sets,
    _run_ora_for_library,
    _term_matrix,
)

from shared.pipeline_utils import write_json
from shared.paths import PROCESSED

OUTPUT = PROCESSED / "enrichment"


def run() -> dict[str, Any]:
    expression = pd.read_parquet(PROCESSED / "muscle_logcpm_corrected_top5000.parquet")
    all_symbols = pd.read_parquet(PROCESSED / "gene_symbols.parquet")["symbol"]
    selected_symbols = all_symbols.reindex(expression.index)
    membership = pd.read_parquet(PROCESSED / "module_membership_louvain_res1.0.parquet")
    primary = pd.read_parquet(OUTPUT / "module_enrichment_all_tests.parquet")
    if selected_symbols.isna().any() or selected_symbols.duplicated().any():
        raise ValueError("Selected expression genes need unique symbol mappings")
    if not membership["gene_id"].isin(expression.index.astype(str)).all():
        raise ValueError("Module assignments contain genes outside the locked expression matrix")

    selected_background = sorted(selected_symbols.astype(str).unique())
    expressed_background = sorted(all_symbols.astype(str).unique())
    if not set(selected_background).issubset(expressed_background):
        raise AssertionError("Selected symbols are not a subset of the expressed-gene background")
    gene_to_symbol = all_symbols.astype(str).to_dict()
    module_ids = sorted(
        membership.loc[membership["module_id"] != "M0", "module_id"].unique(),
        key=lambda value: int(value[1:]),
    )
    full_module_symbols = {
        module: sorted({gene_to_symbol[gene] for gene in membership.loc[membership["module_id"] == module, "gene_id"]})
        for module in module_ids
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gene_sets_by_library = {
        key: _load_gene_sets(config["name"])[0] for key, config in LIBRARIES.items()
    }

    expressed_results = []
    for library_key, gene_sets in gene_sets_by_library.items():
        terms, matrix, sizes, term_genes = _term_matrix(gene_sets, expressed_background)
        expressed_results.append(_run_ora_for_library(
            library_key, terms, matrix, sizes, term_genes, expressed_background, full_module_symbols
        ))
    expressed = _apply_fdr_scopes(pd.concat(expressed_results, ignore_index=True))
    expressed.to_parquet(OUTPUT / "module_enrichment_expressed_background.parquet", index=False)
    background_rows = []
    for module in module_ids:
        for library in LIBRARIES:
            primary_sub = primary.loc[
                (primary["module_id"] == module)
                & (primary["library"] == library)
                & primary["significant_module_library"]
            ]
            expressed_sub = expressed.loc[
                (expressed["module_id"] == module)
                & (expressed["library"] == library)
                & expressed["significant_module_library"]
            ]
            primary_terms, expressed_terms = set(primary_sub["term"]), set(expressed_sub["term"])
            union = primary_terms | expressed_terms
            background_rows.append({
                "module_id": module,
                "library": library,
                "selected_background_significant_terms": len(primary_terms),
                "expressed_background_significant_terms": len(expressed_terms),
                "shared_significant_terms": len(primary_terms & expressed_terms),
                "jaccard": len(primary_terms & expressed_terms) / len(union) if union else None,
            })
    background_summary = pd.DataFrame(background_rows)
    background_summary.to_parquet(OUTPUT / "enrichment_background_sensitivity.parquet", index=False)

    broad_sig_total = int(expressed["significant_module_library"].sum())
    selected_sig_total = int(primary["significant_module_library"].sum())
    report = {
        "status": "PASS",
        "method": {
            "background_sensitivity": {
                "primary_background_symbols": len(selected_background),
                "alternative_background_symbols": len(expressed_background),
                "gene_lists_and_module_assignments_held_fixed": True,
                "significance_scope": "BH within module x library, overlap >= 3",
            },
        },
        "summary": {
            "selected_background_module_library_fdr_hits": selected_sig_total,
            "expressed_background_module_library_fdr_hits": broad_sig_total,
        },
        "outputs": {
            "expressed_background_tests": "enrichment/module_enrichment_expressed_background.parquet",
            "background_sensitivity": "enrichment/enrichment_background_sensitivity.parquet",
        },
    }
    write_json(OUTPUT / "enrichment_robustness.json", report, allow_nan=False)
    print(json.dumps({
        "status": report["status"],
        "method": report["method"],
        "summary": report["summary"],
        "background_sensitivity": background_summary.groupby("library")[[
            "selected_background_significant_terms", "expressed_background_significant_terms"
        ]].sum().reset_index().to_dict(orient="records"),
        "outputs": report["outputs"],
    }, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    run()

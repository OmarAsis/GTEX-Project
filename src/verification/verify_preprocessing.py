"""Verify the saved, locked preprocessing artifacts and EDA figures."""

from __future__ import annotations

import json

import pandas as pd

from shared.paths import PROCESSED


FIGURES = (
    "expression_distribution.png",
    "sample_distributions.png",
    "sample_correlation_heatmap.png",
    "sample_correlation_distribution.png",
    "pca_before_after_correction.png",
    "pca_variance_explained.png",
)


def verify() -> dict[str, object]:
    top = pd.read_parquet(PROCESSED / "muscle_logcpm_corrected_top5000.parquet")
    metadata = pd.read_parquet(PROCESSED / "muscle_metadata.parquet")
    symbols = pd.read_parquet(PROCESSED / "gene_symbols.parquet")["symbol"]
    params = json.loads((PROCESSED / "preprocessing_params.json").read_text())

    samples = params["samples"]["n_final"]
    genes = params["gene_filter"]["n_after_filter"]
    selected = params["gene_selection"]["n_selected"]
    assert (samples, genes, selected) == (803, 17_224, 5_000)
    assert (top.shape, metadata.shape, symbols.shape) == (
        (selected, samples), (samples, 64), (genes,)
    )
    assert top.index.is_unique and top.columns.is_unique
    assert top.columns.equals(metadata.index)
    assert symbols.index.is_unique and top.index.isin(symbols.index).all()
    assert top.index.str.match(r"^ENSG\d+$").all()

    figures_dir = PROCESSED / "eda_figures"
    invalid_figures = [
        name for name in FIGURES
        if not (figures_dir / name).is_file() or (figures_dir / name).stat().st_size == 0
    ]
    assert not invalid_figures, f"Missing or empty EDA figures: {invalid_figures}"

    result = {
        "status": "PASS",
        "top_variance_matrix": list(top.shape),
        "metadata": list(metadata.shape),
        "symbols": list(symbols.shape),
        "samples_kept": samples,
        "eda_figures_verified": len(FIGURES),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))

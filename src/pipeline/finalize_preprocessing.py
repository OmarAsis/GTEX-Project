"""Run GTEx preprocessing QC and save the final analysis matrices."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from shared.paths import CONFIG, PROJECT_ROOT, PROCESSED
from shared.pipeline_utils import read_json, write_json

os.environ.setdefault("MPLCONFIGDIR", str(PROCESSED / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = PROJECT_ROOT / "data"
PARAMS = read_json(CONFIG / "preprocessing_params.json")
TOP_N_GENES = PARAMS["gene_selection"]["top_n"]
# Numeric covariates chosen based on notebook; citation to be added later.
COVARIATE_SETTINGS = PARAMS["covariate_correction"]
NUMERIC_COVARIATES = COVARIATE_SETTINGS["numeric"]
# Categorical covariate chosen based on notebook; citation to be added later.
CATEGORICAL_COVARIATES = COVARIATE_SETTINGS["categorical"]
COVARIATE_MISSING_CATEGORY = COVARIATE_SETTINGS["missing_category"]
# PCA scree range chosen based on notebook; citation to be added later.
PCA_DIAGNOSTICS = PARAMS["pca_diagnostics"]
PCA_DIAGNOSTIC_COMPONENTS = PCA_DIAGNOSTICS["components_for_scree_plot"]
# PCA check component count chosen based on notebook; citation to be added later.
PCA_CHECK_COMPONENTS = PCA_DIAGNOSTICS["components_checked_against_covariates"]
# Seed 42 is the project repeatability seed used by module and resampling stages, not an EDA result.
PCA_RANDOM_STATE = PCA_DIAGNOSTICS["random_state"]


def _add_sample_traits(metadata: pd.DataFrame, phenotypes: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    sample_ids = pd.Index(metadata.index.astype(str))
    donor_ids = sample_ids.str.extract(r"^(GTEX-[^-]+)", expand=False)
    if donor_ids.isna().any():
        raise ValueError("Could not derive donor IDs for all sample IDs")

    subjects = phenotypes.set_index("SUBJID")
    raw_sex = subjects["SEX"].reindex(donor_ids.to_numpy())
    if raw_sex.isna().any() or not raw_sex.isin([1, 2]).all():
        raise ValueError("GTEx donor sex is missing or has an unexpected encoding")
    metadata["SEX"] = (raw_sex.to_numpy() == 1).astype(np.int8)

    # DTHHRDY was chosen based on notebook; citation to be added later.
    death = subjects["DTHHRDY"].reindex(donor_ids.to_numpy())
    metadata["DTHHRDY"] = death.astype("string").fillna("Missing").to_numpy()
    return metadata


def _pca(matrix: pd.DataFrame) -> tuple[PCA, np.ndarray]:
    model = PCA(
        n_components=PCA_DIAGNOSTIC_COMPONENTS,
        svd_solver="randomized",
        random_state=PCA_RANDOM_STATE,
    )
    scores = model.fit_transform(matrix.T.to_numpy(dtype=np.float64, copy=False))
    return model, scores


def _save_eda_figures(
    expression: pd.DataFrame,
    sample_corr: pd.DataFrame,
    pca_before: PCA,
    scores_before: np.ndarray,
    scores_after: np.ndarray,
    metadata: pd.DataFrame,
    out_dir: Path,
) -> None:
    # EDA graph-only section begins.
    out_dir.mkdir(parents=True, exist_ok=True)
    X = expression.to_numpy(dtype=np.float64, copy=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(X.ravel(), bins=100, color="#4778a8", edgecolor="white", linewidth=0.25)
    ax.set(xlabel="log1p(CPM)", ylabel="Gene-sample observations", title="Expression distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "expression_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(expression.mean(axis=0), bins=30, color="#4778a8")
    axes[0].set(xlabel="Sample mean log1p(CPM)", ylabel="Samples", title="Sample means")
    axes[1].hist(expression.median(axis=0), bins=30, color="#d18b47")
    axes[1].set(xlabel="Sample median log1p(CPM)", ylabel="Samples", title="Sample medians")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_distributions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(sample_corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set(title="Sample-sample correlation", xlabel="Samples", ylabel="Samples")
    fig.colorbar(image, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    off_diagonal = sample_corr.to_numpy()[np.triu_indices(len(sample_corr), k=1)]
    corr_quantiles = np.quantile(off_diagonal, [0.05, 0.5, 0.95])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(off_diagonal, bins=60, color="#4778a8", edgecolor="white", linewidth=0.25)
    ax.axvline(corr_quantiles[1], color="#c44e52", linestyle="--", linewidth=1.5,
               label=f"Median = {corr_quantiles[1]:.3f}")
    ax.set(xlabel="Pearson r between samples", ylabel="Sample pairs",
           title="Pairwise sample-correlation distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_correlation_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sex_values = metadata["SEX"].to_numpy()
    for sex, label, color in [(0, "Female", "#e58c42"), (1, "Male", "#3478b8")]:
        selected = sex_values == sex
        axes[0].scatter(scores_before[selected, 0], scores_before[selected, 1],
                        c=color, label=label, s=10, alpha=0.65)
    axes[0].set(title="PCA before covariate correction", xlabel="PC1", ylabel="PC2")
    axes[0].legend(title="Sex", frameon=False)
    ischemic = pd.to_numeric(metadata["SMTSISCH"], errors="coerce")
    points = axes[1].scatter(scores_after[:, 0], scores_after[:, 1], c=ischemic, cmap="viridis", s=10, alpha=0.65)
    axes[1].set(title="PCA after covariate correction", xlabel="PC1", ylabel="PC2")
    fig.colorbar(points, ax=axes[1], label="Ischemic time")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_before_after_correction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    components = np.arange(1, PCA_DIAGNOSTIC_COMPONENTS + 1)
    ax.plot(components, pca_before.explained_variance_ratio_[: len(components)], marker="o", color="#4778a8")
    ax.set(xlabel="Principal component", ylabel="Variance explained", title="PCA scree plot")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_variance_explained.png", dpi=180)
    plt.close(fig)
    # EDA graph-only section ends.


def run() -> dict[str, Any]:
    expression = pd.read_parquet(PROCESSED / "muscle_logcpm_filtered.parquet")
    metadata = pd.read_parquet(PROCESSED / "muscle_metadata_clean.parquet")
    symbols = pd.read_parquet(PROCESSED / "gene_symbols.parquet")["symbol"]
    phenotypes = pd.read_csv(
        DATA / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t"
    )
    print(f"Loaded clean matrix {expression.shape}; metadata {metadata.shape}", flush=True)

    if not metadata.index.equals(expression.columns):
        raise ValueError("Sample metadata and expression columns are misaligned")
    if not symbols.index.equals(expression.index):
        raise ValueError("Gene symbols and expression rows are misaligned")

    metadata = _add_sample_traits(metadata, phenotypes)
    sample_corr = expression.corr(method="pearson")
    print("Computed sample-sample correlations", flush=True)
    mean_sample_corr = (sample_corr.sum(axis=1) - 1) / (len(sample_corr) - 1)

    pca_before, scores_before = _pca(expression)
    print("Computed pre-correction PCA", flush=True)
    numeric_covariates = metadata.loc[:, NUMERIC_COVARIATES].apply(pd.to_numeric, errors="raise")
    if numeric_covariates.isna().any().any():
        raise ValueError("A required numeric covariate is missing")
    covariates = numeric_covariates
    for name in CATEGORICAL_COVARIATES:
        categories = metadata[name].astype("string").fillna(COVARIATE_MISSING_CATEGORY)
        dummies = pd.get_dummies(
            categories, prefix=name, drop_first=True, dtype=np.float64
        )
        covariates = pd.concat([covariates, dummies], axis=1)
    covariate_sd = covariates.std(axis=0, ddof=1)
    if covariates.isna().any().any() or (covariate_sd == 0).any():
        raise ValueError("Covariates contain missing values or a constant column")
    covariates_z = (covariates - covariates.mean(axis=0)) / covariate_sd
    design = np.column_stack(
        [np.ones(len(covariates_z)), covariates_z.to_numpy(dtype=np.float64)]
    )
    Y = expression.to_numpy(dtype=np.float64, copy=False).T
    beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
    corrected_values = Y - design @ beta + Y.mean(axis=0)
    print("Regressed covariates from expression matrix", flush=True)
    corrected = pd.DataFrame(
        corrected_values.T, index=expression.index, columns=expression.columns
    )
    if not np.isfinite(corrected_values).all():
        raise AssertionError("Covariate-corrected expression has invalid values")

    pca_after, scores_after = _pca(corrected)
    print("Computed post-correction PCA", flush=True)
    pc_names = [f"PC{i + 1}" for i in range(PCA_CHECK_COMPONENTS)]
    pcs_after = pd.DataFrame(
        scores_after[:, :PCA_CHECK_COMPONENTS], index=corrected.columns, columns=pc_names
    )
    corrected_covariate_correlations = {
        pc: {
            name: float(numeric_covariates[name].corr(pcs_after[pc]))
            for name in NUMERIC_COVARIATES
        }
        for pc in pcs_after.columns
    }
    top_expression = corrected.iloc[
        np.argsort(-corrected_values.var(axis=0, ddof=1), kind="stable")[:TOP_N_GENES]
    ]
    if top_expression.shape != (TOP_N_GENES, len(expression.columns)):
        raise AssertionError("Final top-variance matrix has an unexpected shape")
    print(f"Selected top {TOP_N_GENES} genes", flush=True)

    final_params: dict[str, Any] = {
        "samples": {
            "tissue": "Muscle - Skeletal",
            "sample_type": "RNASEQ",
            "n_final": int(len(expression.columns)),
        },
        "normalization": "CPM (counts / library size * 1e6)",
        "gene_filter": {
            "min_cpm": 1.0,
            "min_samples": 10,
            "n_after_filter": int(expression.shape[0]),
        },
        "transformation": "log1p(CPM)",
        "covariate_correction": {
            "numeric": NUMERIC_COVARIATES,
            "categorical": CATEGORICAL_COVARIATES,
            "missing_category": COVARIATE_MISSING_CATEGORY,
            "categorical_encoding": "dummy-coded with the first level omitted",
            "method": "joint least-squares regression with standardized covariates; gene means restored",
        },
        "gene_selection": {
            "method": "highest variance after covariate correction",
            "top_n": TOP_N_GENES,
            "n_selected": int(top_expression.shape[0]),
        },
        "pca_components": PCA_DIAGNOSTIC_COMPONENTS,
        "pca_random_state": PCA_RANDOM_STATE,
        "final_matrix_shape": [int(n) for n in top_expression.shape],
    }

    eda: dict[str, Any] = {
        "sample_correlation": {
            "mean_range": [float(mean_sample_corr.min()), float(mean_sample_corr.max())],
        },
        "pca_before_correction": {
            "variance_explained_first_3": [
                float(value) for value in pca_before.explained_variance_ratio_[:3]
            ],
        },
        "pca_after_correction": {
            "variance_explained_first_3": [
                float(value) for value in pca_after.explained_variance_ratio_[:3]
            ],
            "correlations_with_corrected_covariates": corrected_covariate_correlations,
        },
        "final_matrix_shape": [int(value) for value in top_expression.shape],
    }

    figures = PROCESSED / "eda_figures"
    _save_eda_figures(expression, sample_corr, pca_before, scores_before, scores_after, metadata, figures)
    print("Saved EDA figures", flush=True)
    top_expression.to_parquet(PROCESSED / "muscle_logcpm_corrected_top5000.parquet")
    metadata.to_parquet(PROCESSED / "muscle_metadata.parquet")
    symbols.to_frame("symbol").to_parquet(PROCESSED / "gene_symbols.parquet")
    write_json(PROCESSED / "preprocessing_params.json", final_params, allow_nan=False)
    write_json(PROCESSED / "eda_summary.json", eda, allow_nan=False)
    print("Saved the selected matrix, metadata, symbols and summaries", flush=True)
    return {"parameters": final_params, "eda": eda, "figures_dir": str(figures)}


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": "PASS",
        "final_matrix_shape": result["parameters"]["final_matrix_shape"],
        "samples_kept": result["parameters"]["samples"]["n_final"],
        "pca_figures": 2,
    }, indent=2))

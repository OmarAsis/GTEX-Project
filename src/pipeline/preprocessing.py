"""Clean GTEx skeletal-muscle counts into a numerical expression matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.paths import PROCESSED, PROJECT_ROOT
from shared.pipeline_utils import write_json

COUNTS_FILE_PATTERN = "GTEx_Analysis_*_v8_RNASeQCv1.1.9_gene_reads.gct.gz"
ATTRIBUTES_FILENAME = "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"


def standardize_gene_ids(ids: pd.Series) -> pd.Series:
    """Trim IDs and remove Ensembl version suffixes while preserving other IDs."""
    result = ids.astype("string").str.strip()
    return result.str.replace(r"\.\d+$", "", regex=True)


def _first_nonmissing(values: pd.Series) -> Any:
    nonmissing = values.dropna()
    return nonmissing.iloc[0] if not nonmissing.empty else pd.NA


def preprocess_gtex_muscle(
    data_dir: str | Path,
    *,
    min_cpm: float = 1.0,
    min_samples: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, Any]]:
    """Return log1p(CPM), aligned sample metadata, symbols, and an audit record.

    The result is genes × samples. Missing count cells remove the affected gene;
    they are never silently interpreted as zero. Duplicate canonical gene IDs
    are collapsed by summing their raw counts before CPM normalization.
    """
    data_dir = Path(data_dir)
    count_files = sorted(data_dir.glob(COUNTS_FILE_PATTERN))
    if len(count_files) != 1:
        raise FileNotFoundError(
            f"Expected one count matrix matching {COUNTS_FILE_PATTERN!r}; found {len(count_files)}"
        )
    counts_path = count_files[0]
    attributes_path = data_dir / ATTRIBUTES_FILENAME

    attributes = pd.read_csv(attributes_path, sep="\t")
    if attributes["SAMPID"].isna().any() or not attributes["SAMPID"].is_unique:
        raise ValueError("Sample attributes must have one nonmissing row per SAMPID")

    usable_metadata = attributes.loc[
        attributes["SMTSD"].eq("Muscle - Skeletal")
        & attributes["SMAFRZE"].eq("RNASEQ")
    ].copy()
    usable_ids = set(usable_metadata["SAMPID"])

    raw_columns = pd.read_csv(
        #first two rows are metadata-note to self
        counts_path, sep="\t", skiprows=2, nrows=0
    ).columns.tolist()
    count_sample_columns = raw_columns[2:]
    if len(count_sample_columns) != len(set(count_sample_columns)):
        raise ValueError("The raw count matrix has duplicate sample columns")
    missing_samples = sorted(usable_ids.difference(count_sample_columns))
    if missing_samples:
        raise ValueError(
            f"{len(missing_samples)} eligible metadata samples are absent from counts"
        )
    sample_ids = [sample for sample in count_sample_columns if sample in usable_ids]
    if not sample_ids:
        raise ValueError("No skeletal-muscle RNASEQ samples matched the count matrix")

    raw = pd.read_csv(
        counts_path,
        sep="\t",
        skiprows=2,
        usecols=["Name", "Description", *sample_ids],
        dtype={"Name": "string", "Description": "string"},
    )
    counts = raw.loc[:, sample_ids].apply(pd.to_numeric, errors="raise")
    gene_ids = standardize_gene_ids(raw["Name"])

    valid_ids = gene_ids.notna() & gene_ids.ne("")
    missing_id_rows = int((~valid_ids).sum())
    missing_count_rows = int(counts.isna().any(axis=1).sum())
    valid_rows = valid_ids & ~counts.isna().any(axis=1)
    counts = counts.loc[valid_rows].copy()
    gene_ids = gene_ids.loc[valid_rows]
    descriptions = raw.loc[valid_rows, "Description"].astype("string")

    count_values = counts.to_numpy()
    if not np.isfinite(count_values).all():
        raise ValueError("Raw counts contain infinite or non-finite values")
    if (count_values < 0).any():
        raise ValueError("Raw counts contain negative values")

    duplicate_rows = int(gene_ids.duplicated(keep="first").sum())
    descriptions_by_gene = descriptions.groupby(gene_ids, sort=False).agg(
        _first_nonmissing
    )
    counts.index = pd.Index(gene_ids, name="gene_id")
    counts = counts.groupby(level=0, sort=False).sum()
    counts = counts.loc[descriptions_by_gene.index]
    symbols = descriptions_by_gene.reindex(counts.index).rename("symbol")

    library_sizes = counts.sum(axis=0)
    zero_library_samples = library_sizes.index[library_sizes.eq(0)].tolist()
    if zero_library_samples:
        counts = counts.drop(columns=zero_library_samples)
        library_sizes = library_sizes.drop(index=zero_library_samples)
    cpm = counts.div(library_sizes, axis=1).mul(1_000_000)
    if not np.allclose(cpm.sum(axis=0).to_numpy(), 1_000_000):
        raise AssertionError("CPM columns do not sum to one million")

    keep_genes = (cpm >= min_cpm).sum(axis=1) >= min_samples
    filtered_cpm = cpm.loc[keep_genes]
    log_expression = np.log1p(filtered_cpm)

    symbols = symbols.reindex(log_expression.index)

    metadata = usable_metadata.set_index("SAMPID").reindex(log_expression.columns)
    if not metadata.index.equals(log_expression.columns):
        raise AssertionError("Sample metadata is not aligned to expression columns")
    if metadata.isna().all(axis=1).any():
        raise AssertionError("An expression sample has no matching metadata row")

    donor_ids = pd.Index(log_expression.columns).str.extract(
        r"^(GTEX-[^-]+)", expand=False
    )
    if donor_ids.isna().any():
        raise ValueError("Could not derive a GTEx donor ID for every sample")

    audit: dict[str, Any] = {
        "source_counts": counts_path.name,
        "source_metadata": attributes_path.name,
        "sample_selection": {
            "tissue": "Muscle - Skeletal",
            "sample_type": "RNASEQ",
            "n_samples": int(log_expression.shape[1]),
            "n_unique_donors": int(donor_ids.nunique()),
            "zero_library_samples_removed": zero_library_samples,
        },
        "gene_ids": {
            "standardization": "trim whitespace; remove trailing Ensembl version suffix",
            "missing_or_empty_ids_removed": missing_id_rows,
            "duplicate_rows_aggregated_by_sum": duplicate_rows,
            "unique_genes_after_canonicalization": int(counts.shape[0]),
        },
        "missing_counts": {
            "genes_with_missing_values_removed": missing_count_rows,
            "imputation": None,
        },
        "normalization": "CPM (counts / sample library size * 1e6)",
        "gene_filter": {
            "min_cpm": min_cpm,
            "min_samples": min_samples,
            "genes_after_filter": int(log_expression.shape[0]),
        },
        "transformation": "log1p(CPM)",
        "matrix_orientation": "genes x samples",
        "shape": [int(n) for n in log_expression.shape],
        "checks": {
            "gene_ids_unique": bool(log_expression.index.is_unique),
            "sample_ids_unique": bool(log_expression.columns.is_unique),
            "metadata_aligned": bool(metadata.index.equals(log_expression.columns)),
            "missing_expression_values": int(log_expression.isna().sum().sum()),
        },
    }
    return log_expression, metadata, symbols, audit


def save_outputs(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    symbols: pd.Series,
    audit: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expression.to_parquet(output_dir / "muscle_logcpm_filtered.parquet")
    metadata.to_parquet(output_dir / "muscle_metadata_clean.parquet")
    # The final preprocessing stage preserves the same full gene universe, so
    # this canonical map can be consumed directly and avoids a duplicate file.
    symbols.to_frame().to_parquet(output_dir / "gene_symbols.parquet")
    write_json(output_dir / "preprocessing_stage1_params.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument("--min-cpm", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=10)
    args = parser.parse_args()

    if args.min_cpm < 0 or args.min_samples < 1:
        parser.error("--min-cpm must be >= 0 and --min-samples must be >= 1")
    expression, metadata, symbols, audit = preprocess_gtex_muscle(
        args.data_dir, min_cpm=args.min_cpm, min_samples=args.min_samples
    )
    save_outputs(expression, metadata, symbols, audit, args.output_dir)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

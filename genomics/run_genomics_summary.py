#!/usr/bin/env python3
"""
Genomics summary: variant missingness and mutation distribution per chromosome,
then genome-wide combined plot. Run with sbatch for parallel chromosome processing.

Usage:
  python run_genomics_summary.py --dir /path/to/ukb --fig-dir genomics_figures --workers 8
  sbatch run_genomics_summary.sbatch
"""
import argparse
import gc
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display for batch/slurm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Defaults (override with CLI)
DEFAULT_DIR = "/../../orcd/pool/003/dbertsim_shared/ukb/"
DEFAULT_FIG_DIR = "figures"
CHROMS = ["Y","X"] + list(range(1, 23))
CHROMS = [19]

def variant_missingness_distribution(
    chrom,
    data_dir,
):
    """Compute missingness/mutation stats and optionally save figures. Returns summary DataFrame."""
    files = sorted(glob.glob(os.path.join(data_dir, f"bgen/ch{chrom}/c{chrom}_b0_*parquet")))
    print(f"Found {len(files)} files for chromosome {chrom}", flush=True)

    # Load sample map ONCE per chromosome and restrict to unique sample_idx
    sample_map_path = os.path.join(data_dir, f"bgen/ch{chrom}/c{chrom}_b0_v1_samples.csv")
    if not os.path.exists(sample_map_path):
        print(f"Sample map not found for chromosome {chrom}: {sample_map_path}", flush=True)
        return None

    sample_map = pd.read_csv(sample_map_path)
    assert (sample_map["ID_1"] == sample_map["ID_2"]).all(), "ID_1 vs ID_2 mismatch"
    sample_map = sample_map.rename(columns={"ID_1": "eid"})

    sample_idx = sample_map["sample_idx"].unique()
    n_samples = len(sample_idx)

    chrom_str = str(chrom)
    is_Y = chrom_str.upper() == "Y"

    all_missing_fracs = []
    all_zero_fracs = []
    all_one_fracs = []
    all_two_fracs = []

    for path in files:
        # Read only required columns to reduce I/O and memory
        df = pd.read_parquet(path, columns=["sample_idx", "variant_idx", "dosage"])

        # Keep only samples present in the sample map
        df = df.loc[df["sample_idx"].isin(sample_idx)]
        if df.empty:
            del df
            gc.collect()
            continue

        # Long-format aggregation: compute per-variant fractions directly
        dosage = df["dosage"]
        variant_ids = df["variant_idx"]

        # Counts per variant (how many samples have this variant recorded)
        counts = df.groupby("variant_idx")["dosage"].size()

        # Boolean masks for each genotype class
        is_missing = dosage.isna()
        is_zero = dosage == 0
        is_one = dosage == 1
        is_two = dosage == 2

        missing_counts = is_missing.groupby(variant_ids).sum()
        zero_counts = is_zero.groupby(variant_ids).sum()
        one_counts = is_one.groupby(variant_ids).sum()
        two_counts = is_two.groupby(variant_ids).sum()

        # Samples without a row for a given variant were previously treated as 0
        n_absent = n_samples - counts

        missing_frac = (missing_counts.to_numpy()) / n_samples
        zero_frac = (zero_counts.to_numpy() + n_absent.to_numpy()) / n_samples
        one_frac = (one_counts.to_numpy()) / n_samples
        two_frac = (two_counts.to_numpy()) / n_samples

        all_missing_fracs.append(missing_frac)
        all_zero_fracs.append(zero_frac)
        all_one_fracs.append(one_frac)
        all_two_fracs.append(two_frac)

        del df
        gc.collect()

    if not all_missing_fracs:
        print("No data after filtering on eids.", flush=True)
        return None

    all_missing_fracs = np.concatenate(all_missing_fracs)
    all_zero_fracs = np.concatenate(all_zero_fracs)
    all_one_fracs = np.concatenate(all_one_fracs)
    all_two_fracs = np.concatenate(all_two_fracs)

    n_variants = len(all_missing_fracs)

    # Bucket edges and histograms (0.0–0.1, ..., 0.9–1.0)
    bins = np.linspace(0, 1, 11)
    counts_missing, edges = np.histogram(all_missing_fracs, bins=bins)
    counts_zero, _ = np.histogram(all_zero_fracs, bins=bins)
    counts_one, _ = np.histogram(all_one_fracs, bins=bins)
    counts_two, _ = np.histogram(all_two_fracs, bins=bins)

    prop_missing = counts_missing / n_variants
    prop_zero = counts_zero / n_variants
    prop_one = counts_one / n_variants
    prop_two = counts_two / n_variants

    # Identify variants with zero mutations
    if is_Y:
        has_zero_mut = all_one_fracs == 0
    else:
        has_zero_mut = (all_one_fracs == 0) & (all_two_fracs == 0)
    n_zero_mut = int(has_zero_mut.sum())

    # One row per chromosome: flatten bucketed proportions into columns
    bucket_labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges) - 1)]

    data = {
        "chrom": [chrom],
        "n_variants": [n_variants],
        "n_variants_no_mutations": [n_zero_mut],
    }

    for label, vals in [
        ("missing", prop_missing),
        ("zero_mut", prop_zero),
        ("one_mut", prop_one),
        ("two_mut", prop_two),
    ]:
        # For Y chromosome, two_mut columns are not meaningful; set to NaN
        if is_Y and label == "two_mut":
            vals = np.full_like(vals, np.nan, dtype=float)
        for b_label, v in zip(bucket_labels, vals):
            col = f"prop_variants_{label}_{b_label}"
            data[col] = [float(v)]

    summary = pd.DataFrame(data)

    summary.to_csv(f"{DEFAULT_FIG_DIR}/ch{chrom}_summary.csv", index=False)

    return summary

def process_chrom(args):
    """Worker: process one chromosome. Must be top-level for pickling."""
    chrom, data_dir, fig_dir = args
    return variant_missingness_distribution(chrom, data_dir)


def main():
    parser = argparse.ArgumentParser(description="Genomics summary: per-chrom and combined missingness/mutation plots")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers (chromosomes)")
    args = parser.parse_args()

    data_dir = DEFAULT_DIR
    fig_dir = DEFAULT_FIG_DIR
    n_workers = min(args.workers, len(CHROMS))

    os.makedirs(fig_dir, exist_ok=True)

    # Parallel chromosome processing
    task_args = [(c, data_dir, fig_dir) for c in CHROMS]
    summary_list = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_chrom, a): a[0] for a in task_args}
        for future in as_completed(futures):
            chrom = futures[future]
            try:
                s = future.result()
                if s is not None:
                    summary_list.append(s)
                    print(f"Chromosome {chrom} done.", flush=True)
            except Exception as e:
                print(f"Chromosome {chrom} failed: {e}", flush=True)

    summary_df = pd.concat(summary_list, ignore_index=True) if summary_list else pd.DataFrame()
    if summary_df.empty:
        print("No summary data; skipping combined plot and CSV.", flush=True)
        sys.exit(1)

    # summary_csv = os.path.join(fig_dir, "summary_df.csv")
    # summary_df.to_csv(summary_csv, index=False)
    # print(f"Saved {summary_csv}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

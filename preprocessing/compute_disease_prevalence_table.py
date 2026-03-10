#!/usr/bin/env python3
"""
Compute disease prevalence table by time frame for internal (England) and external (Scotland/Wales).

For each disease:
- prevalence_0yr: baseline prevalence (proportion with disease at baseline)
- prevalence_1yr, 2yr, 5yr, 10yr: among those WITHOUT disease at baseline, proportion
  diagnosed within that time frame (excludes prevalent cases from denominator)

Output columns: disease, prevalence_0yr_internal, prevalence_1yr_internal, ..., prevalence_10yr_external
"""

import argparse
import os

import pandas as pd

DISEASES = [
    "acute_kidney_injury",
    "alzheimers_disease",
    "atrial_fibrillation",
    "chronic_kidney_disease",
    "copd",
    "end_stage_renal_disease",
    "heart_failure",
    "hypertensive_heart_kidney_diseases",
    "ischemic_heart_disease",
    "liver_disease",
    "lower_respiratory_disease",
    "other_dementia",
    "parkinsons",
    "peripheral_vascular_disease",
    "stroke",
    "type_1_diabetes",
    "type_2_diabetes",
]

TIME_FRAMES = [0, 1, 2, 5, 10]


def compute_prevalence_for_disease(df: pd.DataFrame, disease: str) -> dict[str, float]:
    """
    Compute prevalence at 0, 1, 2, 5, 10yr for one disease in one dataset.

    - 0yr: proportion with disease at baseline (all participants)
    - 1/2/5/10yr: among those WITHOUT disease at baseline, proportion diagnosed within that horizon
    """
    if disease not in df.columns:
        return {f"prevalence_{tf}yr": float("nan") for tf in TIME_FRAMES}

    time_col = f"{disease}_time_to_diagnosis"
    if time_col not in df.columns:
        return {f"prevalence_{tf}yr": float("nan") for tf in TIME_FRAMES}

    result = {}

    # 0yr: baseline prevalence = proportion with disease==1
    result["prevalence_0yr"] = (df[disease] == 1).sum()

    # For >0yr: restrict to disease==0 (at-risk cohort), then proportion diagnosed within horizon
    ttd = df[time_col]

    for tf in [1, 2, 5, 10]:
        # Positive: diagnosed within tf years (ttd notna and ttd <= tf)
        n_pos = (ttd.notna() & (ttd <= tf) & (ttd > 30 / 365.25)).sum()  # exclude cases diagnosed within 30 days
        result[f"prevalence_{tf}yr"] = n_pos

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute disease prevalence table by time frame (internal=England, external=Scotland/Wales)"
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default=None,
        help="Path to folder with ukb_disease_england.csv and ukb_disease_scotland_wales.csv. "
        "Default: /orcd/pool/003/dbertsim_shared/ukb/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="disease_prevalence_table.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    labels_path = args.labels_path or "/orcd/pool/003/dbertsim_shared/ukb"
    labels_path = os.path.abspath(labels_path)

    england_path = os.path.join(labels_path, "ukb_disease_england.csv")
    scotland_wales_path = os.path.join(labels_path, "ukb_disease_scotland_wales.csv")

    if not os.path.isfile(england_path):
        raise FileNotFoundError(f"England labels not found: {england_path}")
    if not os.path.isfile(scotland_wales_path):
        raise FileNotFoundError(f"Scotland/Wales labels not found: {scotland_wales_path}")

    df_england = pd.read_csv(england_path, low_memory=False)
    df_scotland_wales = pd.read_csv(scotland_wales_path, low_memory=False)

    rows = []
    for disease in DISEASES:
        prev_internal = compute_prevalence_for_disease(df_england, disease)
        prev_external = compute_prevalence_for_disease(df_scotland_wales, disease)

        row = {"disease": disease}
        for tf in TIME_FRAMES:
            row[f"prevalence_{tf}yr_internal"] = prev_internal[f"prevalence_{tf}yr"]
        for tf in TIME_FRAMES:
            row[f"prevalence_{tf}yr_external"] = prev_external[f"prevalence_{tf}yr"]
        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Reorder columns: disease, then internal 0-10yr, then external 0-10yr
    internal_cols = [f"prevalence_{tf}yr_internal" for tf in TIME_FRAMES]
    external_cols = [f"prevalence_{tf}yr_external" for tf in TIME_FRAMES]
    out_df = out_df[["disease"] + internal_cols + external_cols]

    out_path = args.output
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

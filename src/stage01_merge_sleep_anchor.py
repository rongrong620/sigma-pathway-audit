# src/stage01_merge_sleep_anchor.py
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def merge_on_time(anchor_df, other_df, anchor_date_col, other_date_col, tolerance, suffix):
    """
    For each row in anchor_df, find the closest record in other_df
    for the same participant_id within ±tolerance.
    """
    anchor_df = anchor_df.sort_values(["participant_id", anchor_date_col])
    other_df  = other_df.sort_values(["participant_id", other_date_col])

    merged = pd.merge_asof(
        anchor_df,
        other_df,
        by="participant_id",
        left_on=anchor_date_col,
        right_on=other_date_col,
        tolerance=tolerance,
        direction="nearest",
        suffixes=("", f"_{suffix}")
    )
    return merged

def run_stage01(data_dir: Path, out_dir: Path, window_days: int = 180) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tolerance = pd.Timedelta(days=window_days)

    # Load
    s1 = pd.read_csv(data_dir / "s1_body_composition_and_adiposity.csv", parse_dates=["body_comp_date"])
    s2 = pd.read_csv(data_dir / "s2_sleep_and_sleep_disordered_breathing.csv", parse_dates=["sleep_date"])
    s3 = pd.read_csv(data_dir / "s3_autonomic_regulation_and_nocturnal_physiology.csv")
    s4 = pd.read_csv(data_dir / "s4_hemodynamic_and_cardiovascular_function.csv", parse_dates=["hemodynamic_date"])
    s5 = pd.read_csv(data_dir / "s5_vascular_structure_and_atherosclerosis.csv", parse_dates=["vascular_date"])
    s6 = pd.read_csv(data_dir / "s6_gut_microbiome_functional_ecology.csv", parse_dates=["gut_date"])
    cov = pd.read_csv(data_dir / "covariates.csv", parse_dates=["baseline_date"])

    # Merge pipeline (sleep anchor)
    df = s2.copy()
    df = merge_on_time(df, s1, "sleep_date", "body_comp_date", tolerance, "s1")
    df = df.merge(s3, on="participant_id", how="left", suffixes=("", "_s3"))
    df = merge_on_time(df, s4, "sleep_date", "hemodynamic_date", tolerance, "s4")
    df = merge_on_time(df, s5, "sleep_date", "vascular_date", tolerance, "s5")
    df = merge_on_time(df, s6, "sleep_date", "gut_date", tolerance, "s6")
    df = merge_on_time(df, cov, "sleep_date", "baseline_date", tolerance, "cov")

    # Diagnostics
    for col in ["body_comp_date", "hemodynamic_date", "vascular_date", "gut_date", "baseline_date"]:
        if col in df.columns:
            df[f"delta_{col}"] = (df["sleep_date"] - df[col]).abs().dt.days

    out_file = out_dir / f"analysis_table_sleep_anchored_{window_days}d.csv"
    df.to_csv(out_file, index=False)

    logger.info("Stage01 saved: %s", out_file)
    logger.info("Final sample size: %d", len(df))
    return out_file
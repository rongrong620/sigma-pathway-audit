import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def run_stage02(
    in_file: Path,
    out_dir: Path,
    parse_dates: List[str] = ("sleep_date",),
    covariates: List[str] = ("age", "sex", "bmi"),
    extreme_min: float = -1e6,
    extreme_max: float = 1e6,
) -> Dict[str, Any]:
    """
    Stage02: High-level descriptive sanity checks and numeric summary export.

    Outputs:
      - descriptive_numeric_summary.csv
      - descriptive_overview.csv
      - (optional) missingness_rates.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(in_file, parse_dates=list(parse_dates))

    n_rows = len(df)
    if "participant_id" not in df.columns:
        raise ValueError("participant_id column not found in input table.")

    n_participants = df["participant_id"].nunique()
    visits_per_person = df["participant_id"].value_counts()

    logger.info("=" * 80)
    logger.info("BASIC DATA OVERVIEW")
    logger.info("=" * 80)
    logger.info("Input file: %s", str(in_file))
    logger.info("Total rows (sleep-anchored visits): %d", n_rows)
    logger.info("Unique participants: %d", n_participants)
    logger.info("Visits per participant (summary):\n%s", visits_per_person.describe().to_string())

    # ----------------------------
    # Temporal alignment diagnostics
    # ----------------------------
    logger.info("=" * 80)
    logger.info("TEMPORAL ALIGNMENT (Δ days relative to sleep)")
    logger.info("=" * 80)

    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    for col in delta_cols:
        desc = df[col].describe()
        logger.info("%s:\n%s", col, desc.to_string())

    # ----------------------------
    # Missingness overview
    # ----------------------------
    logger.info("=" * 80)
    logger.info("MISSINGNESS OVERVIEW")
    logger.info("=" * 80)

    missing_rate = df.isna().mean().sort_values(ascending=False)
    logger.info("Top 20 variables by missing rate:\n%s", missing_rate.head(20).to_string())
    logger.info("Bottom 20 variables by missing rate:\n%s", missing_rate.tail(20).to_string())

    # Optional: save full missingness vector (recommended for reproducibility)
    missing_file = out_dir / "missingness_rates.csv"
    missing_rate.rename("missing_rate").to_csv(missing_file)
    logger.info("Saved missingness rates to: %s", str(missing_file))

    # ----------------------------
    # Covariate sanity check
    # ----------------------------
    logger.info("=" * 80)
    logger.info("COVARIATE DISTRIBUTIONS")
    logger.info("=" * 80)

    cov_desc = {}
    for cov in covariates:
        if cov in df.columns:
            cov_desc[cov] = df[cov].describe()
            logger.info("%s:\n%s", cov, cov_desc[cov].to_string())
        else:
            logger.warning("%s: NOT FOUND", cov)

    # ----------------------------
    # Numeric variable summary
    # ----------------------------
    logger.info("=" * 80)
    logger.info("NUMERIC VARIABLE RANGE CHECK (selected)")
    logger.info("=" * 80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c for c in numeric_cols
        if not c.startswith("delta_") and c != "participant_id"
    ]

    summary = df[numeric_cols].agg(["count", "mean", "std", "min", "max"]).T

    summary_file = out_dir / "descriptive_numeric_summary.csv"
    summary.to_csv(summary_file)
    logger.info("Saved full numeric summary to: %s", str(summary_file))

    extreme = summary[(summary["min"] < extreme_min) | (summary["max"] > extreme_max)]
    if len(extreme) > 0:
        logger.warning("Variables with potentially extreme ranges (showing up to 20):\n%s",
                       extreme.head(20).to_string())
    else:
        logger.info("No variables exceeded extreme thresholds [%s, %s].", extreme_min, extreme_max)

    # ----------------------------
    # High-level overview table
    # ----------------------------
    overview = pd.DataFrame({
        "metric": [
            "total_rows",
            "unique_participants",
            "median_visits_per_participant"
        ],
        "value": [
            n_rows,
            n_participants,
            float(visits_per_person.median())
        ]
    })

    overview_file = out_dir / "descriptive_overview.csv"
    overview.to_csv(overview_file, index=False)
    logger.info("Saved overview summary to: %s", str(overview_file))

    logger.info("=" * 80)
    logger.info("SANITY CHECK COMPLETED")
    logger.info("=" * 80)

    return {
        "n_rows": n_rows,
        "n_participants": n_participants,
        "median_visits_per_participant": float(visits_per_person.median()),
        "delta_cols": delta_cols,
        "outputs": {
            "overview": str(overview_file),
            "numeric_summary": str(summary_file),
            "missingness": str(missing_file),
        }
    }
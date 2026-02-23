import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import statsmodels.api as sm
from itertools import combinations
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

def aggregate_node_median(
    df: pd.DataFrame,
    vars_: List[str],
    min_nonmissing: int
) -> pd.Series:
    """
    Row-wise median aggregation with minimum non-missing requirement.
    """
    # fast-ish row-wise apply; keeps your original logic exactly
    return df[vars_].apply(
        lambda x: np.nanmedian(x) if x.notna().sum() >= min_nonmissing else np.nan,
        axis=1
    )

def residualize(
    y: pd.Series,
    cov_df: pd.DataFrame,
    min_pairwise_n: int
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Residualize y on covariates using OLS.
    Returns (residuals, valid_mask) or (None, None) if insufficient N.
    """
    valid = y.notna() & cov_df.notna().all(axis=1)
    if int(valid.sum()) < min_pairwise_n:
        return None, None
    X = sm.add_constant(cov_df.loc[valid])
    model = sm.OLS(y.loc[valid], X).fit()

    resid = pd.Series(index=y.index, dtype="float64")
    resid.loc[valid] = model.resid
    return resid, valid

def run_stage03_node(
    in_file: Path,
    out_dir: Path,
    node_map: Dict[str, List[str]],
    covariates: List[str] = None,
    min_nonmissing_per_node: int = 2,
    min_pairwise_n: int = 200,
    out_file_name: str = "node_level_spearman_correlation.csv"
) -> Dict[str, Any]:
    """
    Stage03b: Node-level Spearman correlation (covariate-adjusted residual-based).

    Steps:
      1) Aggregate each node by row-wise median across its variables (>=min_nonmissing_per_node).
      2) Residualize node scores on covariates using OLS.
      3) Compute pairwise Spearman correlations between residualized node scores.

    Output:
      - node_level_spearman_correlation.csv (long pairs)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if covariates is None:
        covariates = ["age", "sex", "bmi"]

    df = pd.read_csv(in_file)
    for c in covariates:
        if c not in df.columns:
            raise ValueError(f"Covariate column not found: {c}")

    # --- construct node-level table ---
    node_df = pd.DataFrame(index=df.index)
    for node, vars_ in node_map.items():
        present = [v for v in vars_ if v in df.columns]
        if len(present) == 0:
            logger.warning("Node '%s' skipped: none of its variables exist in input.", node)
            node_df[node] = np.nan
            continue
        if len(present) < len(vars_):
            missing = sorted(set(vars_) - set(present))
            logger.warning("Node '%s': missing variables in input: %s", node, missing)

        node_df[node] = aggregate_node_median(df, present, min_nonmissing_per_node)

    cov_df = df[covariates].copy()

    # --- pairwise node correlation ---
    results = []
    node_names = list(node_map.keys())

    for node_a, node_b in combinations(node_names, 2):
        y1 = node_df[node_a]
        y2 = node_df[node_b]

        r1, idx1 = residualize(y1, cov_df, min_pairwise_n)
        r2, idx2 = residualize(y2, cov_df, min_pairwise_n)

        if r1 is None or r2 is None:
            continue

        valid = idx1 & idx2
        n = int(valid.sum())
        if n < min_pairwise_n:
            continue

        rho, pval = spearmanr(r1.loc[valid], r2.loc[valid])

        results.append({
            "node_A": node_a,
            "node_B": node_b,
            "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
            "p_value": float(pval) if pd.notna(pval) else np.nan,
            "N": n
        })

    res_df = pd.DataFrame(results)
    if len(res_df) > 0:
        res_df = res_df.sort_values(
            by="spearman_rho",
            key=lambda s: s.abs(),
            ascending=False
        ).reset_index(drop=True)

    out_file = out_dir / out_file_name
    res_df.to_csv(out_file, index=False)

    logger.info("Saved node-level correlations: %s (%d pairs)", str(out_file), len(res_df))

    return {
        "output": str(out_file),
        "n_pairs": int(len(res_df)),
        "n_nodes": int(len(node_names)),
        "covariates": covariates,
        "min_nonmissing_per_node": int(min_nonmissing_per_node),
        "min_pairwise_n": int(min_pairwise_n),
    }
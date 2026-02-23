# src/stage03_variable_variable_corr.py
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import statsmodels.api as sm
import logging
from typing import List, Dict, Optional, Tuple
import itertools

logger = logging.getLogger(__name__)

def residualize_series(y: pd.Series, cov_df: pd.DataFrame, min_pairwise_n: int) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    OLS residualization of y on cov_df.
    Returns (residuals, valid_mask) or (None, None) if insufficient N.
    valid_mask is a boolean Series aligned with y (True = used).
    """
    valid = y.notna() & cov_df.notna().all(axis=1)
    if int(valid.sum()) < min_pairwise_n:
        return None, None
    X = sm.add_constant(cov_df.loc[valid])
    model = sm.OLS(y.loc[valid], X).fit()
    resid = pd.Series(index=y.index, dtype="float64")
    resid.loc[valid] = model.resid
    return resid, valid

def compute_pairwise_spearman(resid_dict: Dict[str, pd.Series],
                              valid_masks: Dict[str, pd.Series],
                              var_pairs: List[Tuple[str, str]],
                              min_pairwise_n: int) -> pd.DataFrame:
    """
    Compute pairwise Spearman for provided variable pairs.
    resid_dict: {var: residual_series}
    valid_masks: {var: boolean mask series}
    var_pairs: list of (x, y) variable name tuples
    """
    results = []
    for x, y in var_pairs:
        r_x = resid_dict.get(x)
        r_y = resid_dict.get(y)
        if r_x is None or r_y is None:
            continue
        valid = valid_masks[x] & valid_masks[y]
        n = int(valid.sum())
        if n < min_pairwise_n:
            continue
        rho, p = spearmanr(r_x.loc[valid], r_y.loc[valid])
        results.append({
            "var1": x,
            "var2": y,
            "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
            "p_value": float(p) if pd.notna(p) else np.nan,
            "N": n
        })
    return pd.DataFrame(results)

def run_stage03(
    in_file: Path,
    out_dir: Path,
    covariates: List[str] = None,
    variables: Optional[List[str]] = None,
    exclude_prefixes: List[str] = ("delta_",),
    min_pairwise_n: int = 200,
    pairwise_only: Optional[List[Tuple[str,str]]] = None
) -> Dict[str, str]:
    """
    General variable-variable correlation stage.

    - If `variables` is None -> infer numeric variables from the table excluding covariates, ids, and exclude_prefixes.
    - If `pairwise_only` is provided, compute only those pairs (list of (x,y)).
    - Otherwise compute all unique unordered pairs among `variables`.

    Outputs (saved under out_dir):
      - full_corr_long.csv    : long-format pairs + rho + p + N
      - full_corr_matrix.csv  : square correlation matrix (rho), NaN where insufficient N
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Stage03 input: %s", in_file)

    df = pd.read_csv(in_file)
    if covariates is None:
        covariates = ["age", "sex", "bmi"]

    # choose variables
    if variables is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # drop participant_id if numeric, and drop covariates and excluded prefixes
        vars_filtered = []
        for c in numeric_cols:
            if c == "participant_id" or c in covariates:
                continue
            if any(c.startswith(pref) for pref in exclude_prefixes):
                continue
            vars_filtered.append(c)
        variables = sorted(vars_filtered)
    else:
        # ensure provided variables exist
        variables = [v for v in variables if v in df.columns]
    logger.info("Number of variables to analyze: %d", len(variables))

    # residualize each variable once
    cov_df = df[covariates].copy()
    resid_dict = {}
    valid_masks = {}
    for v in variables:
        y = df[v]
        resid, valid = residualize_series(y, cov_df, min_pairwise_n)
        if resid is None:
            logger.warning("Variable %s skipped (insufficient non-missing pairs after covariate filter).", v)
            continue
        resid_dict[v] = resid
        valid_masks[v] = valid

    analyzed_vars = sorted(resid_dict.keys())
    logger.info("Variables residualized successfully: %d", len(analyzed_vars))

    if pairwise_only:
        var_pairs = [(x, y) for x, y in pairwise_only if x in analyzed_vars and y in analyzed_vars]
    else:
        # all unique unordered pairs
        var_pairs = list(itertools.combinations(analyzed_vars, 2))

    logger.info("Number of pairs to compute: %d", len(var_pairs))

    long_df = compute_pairwise_spearman(resid_dict, valid_masks, var_pairs, min_pairwise_n)
    long_out = out_dir / "full_corr_long.csv"
    long_df.to_csv(long_out, index=False)
    logger.info("Saved long-format correlations: %s (%d pairs)", str(long_out), len(long_df))

    # build square matrix (rho), fill NaN if pair missing
    matrix_df = pd.DataFrame(index=analyzed_vars, columns=analyzed_vars, dtype="float64")
    for _, row in long_df.iterrows():
        i = row["var1"]
        j = row["var2"]
        rho = row["spearman_rho"]
        matrix_df.at[i, j] = rho
        matrix_df.at[j, i] = rho
    # diagonal = 1
    for v in analyzed_vars:
        matrix_df.at[v, v] = 1.0

    matrix_out = out_dir / "full_corr_matrix.csv"
    matrix_df.to_csv(matrix_out, index=True)
    logger.info("Saved matrix-format correlations: %s", str(matrix_out))

    return {
        "long": str(long_out),
        "matrix": str(matrix_out),
    }
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
import statsmodels.api as sm
from tqdm import tqdm
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# -----------------------------
# Utilities (same logic)
# -----------------------------
def invnorm_transform(s: pd.Series) -> pd.Series:
    x = s.to_numpy()
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if mask.sum() < 3:
        return pd.Series(out, index=s.index)
    r = rankdata(x[mask], method="average")
    n = mask.sum()
    p = (r - 0.5) / n
    out[mask] = norm.ppf(p)
    return pd.Series(out, index=s.index)

def fit_ols(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc, missing="drop").fit()

def bootstrap_ci(arr: np.ndarray, alpha: float = 0.05):
    lo = np.nanquantile(arr, alpha/2)
    hi = np.nanquantile(arr, 1 - alpha/2)
    return lo, hi

def bootstrap_p_two_sided(arr: np.ndarray):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    p_pos = np.mean(arr > 0)
    p_neg = np.mean(arr < 0)
    return 2 * min(p_pos, p_neg)

def normalize_sex(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object:
        sx = s.astype(str).str.strip().str.lower()
        s = sx.map({"m": 1, "male": 1, "1": 1, "f": 0, "female": 0, "0": 0})
    if pd.api.types.is_numeric_dtype(s):
        uniq = sorted([u for u in s.dropna().unique().tolist()])
        if uniq == [1, 2]:
            s = s.map({1: 0, 2: 1})
    return s

# -----------------------------
# Node score builder
# -----------------------------
def build_node_scores(
    df_raw: pd.DataFrame,
    node_map: Dict[str, List[str]],
    covariates: List[str],
    agg: str = "mean",
    min_nonmiss_frac: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df_raw.copy()
    df["sex"] = normalize_sex(df["sex"])

    # required columns check
    all_vars = set(covariates)
    for _, cols in node_map.items():
        all_vars.update(cols)

    missing_vars = [c for c in sorted(all_vars) if c not in df.columns]
    if missing_vars:
        raise ValueError("Missing required columns:\n" + "\n".join(missing_vars))

    df_t = df.copy()
    # invnorm node vars
    for _, cols in node_map.items():
        for c in cols:
            df_t[c] = invnorm_transform(df_t[c])
    # invnorm age/bmi
    for c in ["age", "bmi"]:
        df_t[c] = invnorm_transform(df_t[c])

    node_scores = pd.DataFrame(index=df_t.index)
    missing_info: Dict[str, Any] = {}

    for node, cols in node_map.items():
        sub = df_t[cols]
        nonmiss = sub.notna().sum(axis=1)
        need = int(np.ceil(len(cols) * min_nonmiss_frac))

        if agg == "mean":
            score = sub.mean(axis=1, skipna=True)
        elif agg == "median":
            score = sub.median(axis=1, skipna=True)
        else:
            raise ValueError(f"Unsupported agg: {agg}")

        score[nonmiss < need] = np.nan
        node_scores[node] = score

        missing_info[node] = {
            "n_vars": int(len(cols)),
            "min_nonmiss_required": int(need),
            "n_nonmissing_scores": int(score.notna().sum()),
        }

    # covariates
    for c in covariates:
        node_scores[c] = df_t[c] if c in ["age", "bmi"] else df["sex"]

    return node_scores, missing_info

# -----------------------------
# Mediation runners
# -----------------------------
def run_simple(node_scores: pd.DataFrame, path: dict, covariates: List[str], rng: np.random.Generator, n_boot: int) -> dict:
    Xn, Mn, Yn = path["X"], path["M1"], path["Y"]
    dat = node_scores[[Xn, Mn, Yn] + covariates].dropna()
    n = len(dat)

    out = {"path_id": path["path_id"], "type": "simple", "X": Xn, "M1": Mn, "M2": "", "Y": Yn, "n": int(n)}
    if n < 50:
        out["note"] = "n<50 after complete-case filtering; unstable."
        return out

    X = dat[Xn]; M = dat[Mn]; Y = dat[Yn]; C = dat[covariates]

    m_a = fit_ols(M, pd.concat([X.rename("X"), C], axis=1))
    a = m_a.params.get("X", np.nan)

    m_c = fit_ols(Y, pd.concat([X.rename("X"), C], axis=1))
    c = m_c.params.get("X", np.nan)

    m_b = fit_ols(Y, pd.concat([X.rename("X"), M.rename("M"), C], axis=1))
    b = m_b.params.get("M", np.nan)
    c_prime = m_b.params.get("X", np.nan)

    ind = a * b

    ind_bs = np.full(n_boot, np.nan)
    dir_bs = np.full(n_boot, np.nan)
    tot_bs = np.full(n_boot, np.nan)

    idx = np.arange(n)
    for i in range(n_boot):
        sidx = rng.choice(idx, size=n, replace=True)
        dd = dat.iloc[sidx]

        Xb = dd[Xn]; Mb = dd[Mn]; Yb = dd[Yn]; Cb = dd[covariates]

        ma = fit_ols(Mb, pd.concat([Xb.rename("X"), Cb], axis=1))
        mc = fit_ols(Yb, pd.concat([Xb.rename("X"), Cb], axis=1))
        mb = fit_ols(Yb, pd.concat([Xb.rename("X"), Mb.rename("M"), Cb], axis=1))

        ind_bs[i] = ma.params.get("X", np.nan) * mb.params.get("M", np.nan)
        dir_bs[i] = mb.params.get("X", np.nan)
        tot_bs[i] = mc.params.get("X", np.nan)

    ind_lo, ind_hi = bootstrap_ci(ind_bs)
    dir_lo, dir_hi = bootstrap_ci(dir_bs)
    tot_lo, tot_hi = bootstrap_ci(tot_bs)

    out.update({
        "a": a, "b": b,
        "indirect_ab": ind,
        "indirect_ci_lo": ind_lo, "indirect_ci_hi": ind_hi,
        "indirect_p_boot": bootstrap_p_two_sided(ind_bs),
        "direct_cprime": c_prime,
        "direct_ci_lo": dir_lo, "direct_ci_hi": dir_hi,
        "direct_p_boot": bootstrap_p_two_sided(dir_bs),
        "total_c": c,
        "total_ci_lo": tot_lo, "total_ci_hi": tot_hi,
        "total_p_boot": bootstrap_p_two_sided(tot_bs),
        "prop_mediated": (ind / c) if np.isfinite(c) and c != 0 else np.nan,
    })
    return out

def run_serial(node_scores: pd.DataFrame, path: dict, covariates: List[str], rng: np.random.Generator, n_boot: int) -> dict:
    Xn, M1n, M2n, Yn = path["X"], path["M1"], path["M2"], path["Y"]
    dat = node_scores[[Xn, M1n, M2n, Yn] + covariates].dropna()
    n = len(dat)

    out = {"path_id": path["path_id"], "type": "serial", "X": Xn, "M1": M1n, "M2": M2n, "Y": Yn, "n": int(n)}
    if n < 50:
        out["note"] = "n<50 after complete-case filtering; unstable."
        return out

    X = dat[Xn]; M1 = dat[M1n]; M2 = dat[M2n]; Y = dat[Yn]; C = dat[covariates]

    m_m1 = fit_ols(M1, pd.concat([X.rename("X"), C], axis=1))
    a1 = m_m1.params.get("X", np.nan)

    m_m2 = fit_ols(M2, pd.concat([X.rename("X"), M1.rename("M1"), C], axis=1))
    a2  = m_m2.params.get("X", np.nan)
    d21 = m_m2.params.get("M1", np.nan)

    m_y = fit_ols(Y, pd.concat([X.rename("X"), M1.rename("M1"), M2.rename("M2"), C], axis=1))
    b1 = m_y.params.get("M1", np.nan)
    b2 = m_y.params.get("M2", np.nan)
    c_prime = m_y.params.get("X", np.nan)

    m_c = fit_ols(Y, pd.concat([X.rename("X"), C], axis=1))
    c = m_c.params.get("X", np.nan)

    ind1 = a1 * b1
    ind2 = a2 * b2
    ind3 = a1 * d21 * b2
    ind_total = ind1 + ind2 + ind3

    ind1_bs = np.full(n_boot, np.nan)
    ind2_bs = np.full(n_boot, np.nan)
    ind3_bs = np.full(n_boot, np.nan)
    indt_bs = np.full(n_boot, np.nan)
    dir_bs  = np.full(n_boot, np.nan)
    tot_bs  = np.full(n_boot, np.nan)

    idx = np.arange(n)
    for i in range(n_boot):
        sidx = rng.choice(idx, size=n, replace=True)
        dd = dat.iloc[sidx]

        Xb = dd[Xn]; M1b = dd[M1n]; M2b = dd[M2n]; Yb = dd[Yn]; Cb = dd[covariates]

        mm1 = fit_ols(M1b, pd.concat([Xb.rename("X"), Cb], axis=1))
        mm2 = fit_ols(M2b, pd.concat([Xb.rename("X"), M1b.rename("M1"), Cb], axis=1))
        my  = fit_ols(Yb,  pd.concat([Xb.rename("X"), M1b.rename("M1"), M2b.rename("M2"), Cb], axis=1))
        mc  = fit_ols(Yb,  pd.concat([Xb.rename("X"), Cb], axis=1))

        a1b  = mm1.params.get("X", np.nan)
        a2b  = mm2.params.get("X", np.nan)
        d21b = mm2.params.get("M1", np.nan)
        b1b  = my.params.get("M1", np.nan)
        b2b  = my.params.get("M2", np.nan)

        i1 = a1b * b1b
        i2 = a2b * b2b
        i3 = a1b * d21b * b2b
        it = i1 + i2 + i3

        ind1_bs[i] = i1
        ind2_bs[i] = i2
        ind3_bs[i] = i3
        indt_bs[i] = it
        dir_bs[i]  = my.params.get("X", np.nan)
        tot_bs[i]  = mc.params.get("X", np.nan)

    out.update({
        "a1": a1, "d21": d21, "a2": a2, "b1": b1, "b2": b2,
        "ind_X_M1_Y": ind1,
        "ind_X_M1_Y_ci_lo": bootstrap_ci(ind1_bs)[0],
        "ind_X_M1_Y_ci_hi": bootstrap_ci(ind1_bs)[1],
        "ind_X_M1_Y_p_boot": bootstrap_p_two_sided(ind1_bs),

        "ind_X_M2_Y": ind2,
        "ind_X_M2_Y_ci_lo": bootstrap_ci(ind2_bs)[0],
        "ind_X_M2_Y_ci_hi": bootstrap_ci(ind2_bs)[1],
        "ind_X_M2_Y_p_boot": bootstrap_p_two_sided(ind2_bs),

        "ind_X_M1_M2_Y": ind3,
        "ind_X_M1_M2_Y_ci_lo": bootstrap_ci(ind3_bs)[0],
        "ind_X_M1_M2_Y_ci_hi": bootstrap_ci(ind3_bs)[1],
        "ind_X_M1_M2_Y_p_boot": bootstrap_p_two_sided(ind3_bs),

        "indirect_total": ind_total,
        "indirect_total_ci_lo": bootstrap_ci(indt_bs)[0],
        "indirect_total_ci_hi": bootstrap_ci(indt_bs)[1],
        "indirect_total_p_boot": bootstrap_p_two_sided(indt_bs),

        "direct_cprime": c_prime,
        "direct_ci_lo": bootstrap_ci(dir_bs)[0],
        "direct_ci_hi": bootstrap_ci(dir_bs)[1],
        "direct_p_boot": bootstrap_p_two_sided(dir_bs),

        "total_c": c,
        "total_ci_lo": bootstrap_ci(tot_bs)[0],
        "total_ci_hi": bootstrap_ci(tot_bs)[1],
        "total_p_boot": bootstrap_p_two_sided(tot_bs),

        "prop_mediated_total": (ind_total / c) if np.isfinite(c) and c != 0 else np.nan,
    })
    return out

def run_all_paths(node_scores: pd.DataFrame, paths: List[dict], seed: int, n_boot: int, covariates: List[str], label: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in tqdm(paths, desc=f"Running mediation [{label}]"):
        if p["type"] == "simple":
            rows.append(run_simple(node_scores, p, covariates, rng, n_boot))
        else:
            rows.append(run_serial(node_scores, p, covariates, rng, n_boot))
    return pd.DataFrame(rows)

def reverse_paths(paths: List[dict]) -> List[dict]:
    rev = []
    for p in paths:
        q = p.copy()
        q["path_id"] = p["path_id"] + "_REV"
        q["X"], q["Y"] = p["Y"], p["X"]
        rev.append(q)
    return rev

# -----------------------------
# Stage04 runner
# -----------------------------
def run_stage04_part1(
    in_file: Path,
    out_dir: Path,
    node_map: Dict[str, List[str]],
    paths: List[dict],
    covariates: List[str] = None,
    seed_base: int = 20260105,
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """
    Stage04 (part1): all-in-one mediation runner producing:
      - 3 node-score tables (mean/median + missingness thresholds)
      - 6 mediation result tables (main + reverse across the 3 sensitivity sets)
      - 1 metadata JSON
    """
    if covariates is None:
        covariates = ["age", "sex", "bmi"]

    out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(in_file)

    # Build node-score datasets
    node_mean_05, miss_mean_05 = build_node_scores(df_raw, node_map, covariates, agg="mean",   min_nonmiss_frac=0.5)
    node_median_05, miss_median_05 = build_node_scores(df_raw, node_map, covariates, agg="median", min_nonmiss_frac=0.5)
    node_mean_07, miss_mean_07 = build_node_scores(df_raw, node_map, covariates, agg="mean",   min_nonmiss_frac=0.7)

    node_files = {
        "node_scores_mean_nonmiss0p5.csv": node_mean_05,
        "node_scores_median_nonmiss0p5.csv": node_median_05,
        "node_scores_mean_nonmiss0p7.csv": node_mean_07,
    }
    for fn, dfx in node_files.items():
        dfx.to_csv(out_dir / fn, index=False)

    # Prepare path sets
    paths_main = paths
    paths_rev  = reverse_paths(paths)

    # Run analyses
    res_main_mean05 = run_all_paths(node_mean_05,   paths_main, seed_base + 1, n_boot, covariates, "MAIN mean nonmiss0.5")
    res_rev_mean05  = run_all_paths(node_mean_05,   paths_rev,  seed_base + 2, n_boot, covariates, "REV  mean nonmiss0.5")

    res_main_median05 = run_all_paths(node_median_05, paths_main, seed_base + 3, n_boot, covariates, "MAIN median nonmiss0.5")
    res_rev_median05  = run_all_paths(node_median_05, paths_rev,  seed_base + 4, n_boot, covariates, "REV  median nonmiss0.5")

    res_main_mean07 = run_all_paths(node_mean_07,   paths_main, seed_base + 5, n_boot, covariates, "MAIN mean nonmiss0.7")
    res_rev_mean07  = run_all_paths(node_mean_07,   paths_rev,  seed_base + 6, n_boot, covariates, "REV  mean nonmiss0.7")

    result_tables = {
        "mediation_main_mean_nonmiss0p5.csv": res_main_mean05,
        "mediation_reverse_mean_nonmiss0p5.csv": res_rev_mean05,
        "mediation_main_median_nonmiss0p5.csv": res_main_median05,
        "mediation_reverse_median_nonmiss0p5.csv": res_rev_median05,
        "mediation_main_mean_nonmiss0p7.csv": res_main_mean07,
        "mediation_reverse_mean_nonmiss0p7.csv": res_rev_mean07,
    }
    for fn, dfx in result_tables.items():
        dfx.to_csv(out_dir / fn, index=False)

    meta = {
        "input_file": str(in_file),
        "out_dir": str(out_dir),
        "seed_base": seed_base,
        "n_boot": n_boot,
        "covariates": covariates,
        "node_score_files": {k: str(out_dir / k) for k in node_files.keys()},
        "result_tables": {k: str(out_dir / k) for k in result_tables.keys()},
        "node_missing_info": {
            "mean_nonmiss0p5": miss_mean_05,
            "median_nonmiss0p5": miss_median_05,
            "mean_nonmiss0p7": miss_mean_07,
        },
        "notes": [
            "Reverse paths are generated by swapping X and Y (mediator order unchanged).",
            "Variables and age/bmi are rank-inverse-normal transformed; sex is normalized to 0/1 when possible.",
            "Complete-case filtering is applied per-path after node score construction."
        ]
    }

    meta_file = out_dir / "run_metadata_allinone.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Stage04(part1) completed. Outputs in: %s", str(out_dir))

    return {
        "out_dir": str(out_dir),
        "node_score_files": meta["node_score_files"],
        "result_tables": meta["result_tables"],
        "metadata": str(meta_file),
    }
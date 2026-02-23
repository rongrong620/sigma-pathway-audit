from pathlib import Path
import pandas as pd
from subgroup_runner import run_subgroup_analysis

BASE_DIR = Path("/home/ec2-user/ISMB2026")

df_all = pd.read_csv(BASE_DIR / "analysis" / "analysis_table_all_sleep_anchored_6mo.csv")

SUBGROUPS = {
    "male": df_all["sex"] == 1,
    "female": df_all["sex"] == 0,
}

run_subgroup_analysis(
    df_all=df_all,
    subgroup_dict=SUBGROUPS,
    subgroup_name="sex",
    node_map=NODE_MAP,
    data_file=BASE_DIR / "analysis" / "analysis_table_all_sleep_anchored_6mo.csv",
    path_file=BASE_DIR / "analysis" / "SIGMA_pathways.csv",
    out_base=BASE_DIR / "analysis" / "subgroup",
    covariates_full=["age", "sex", "bmi"],
    seed=20260105,
    n_boot=2000,
    min_n=200
)
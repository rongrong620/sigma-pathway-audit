# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import json
from mediation_core import build_node_scores, run_all_paths, reverse_paths


def load_sigma_paths(path_file: Path):
    df = pd.read_csv(path_file)
    paths = []
    skipped = []

    for _, row in df.iterrows():
        if pd.notna(row.get("M3")) and str(row["M3"]).strip() != "":
            skipped.append(row["task_id"])
            continue

        p = {
            "path_id": row["task_id"],
            "type": row["task_type"].strip().lower(),
            "X": row["X"],
            "M1": row["M1"],
            "Y": row["Y"],
            "M2": row["M2"] if row["task_type"].strip().lower() == "serial" else "",
        }
        paths.append(p)

    return paths, skipped


def sanitize_covariates(df, covariates):
    used = []
    for c in covariates:
        if c not in df.columns:
            continue
        x = df[c].dropna()
        if x.empty or x.nunique() <= 1:
            continue
        used.append(c)
    return used


def run_subgroup_analysis(
    df_all,
    subgroup_dict,
    subgroup_name,
    node_map,
    data_file,
    path_file,
    out_base,
    covariates_full,
    seed,
    n_boot,
    min_n
):
    OUT_BASE = out_base / subgroup_name
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    paths_all, skipped_tasks = load_sigma_paths(path_file)

    meta = {
        "input_data": str(data_file),
        "path_file": str(path_file),
        "skipped_tasks_due_to_M3": skipped_tasks,
        "subgroups": {}
    }

    for sg_name, sg_mask in subgroup_dict.items():

        df_sub = df_all.loc[sg_mask].copy()
        n_sub = df_sub.shape[0]

        print(f"\nRunning subgroup: {sg_name} (n={n_sub})")

        if n_sub < min_n:
            meta["subgroups"][sg_name] = {"n": int(n_sub), "skipped": True}
            continue

        OUT_DIR = OUT_BASE / sg_name
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        covars_used = sanitize_covariates(df_sub, covariates_full)

        node_mean_05, _ = build_node_scores(
            df_sub, node_map, covars_used,
            agg="mean", min_nonmiss_frac=0.5
        )
        node_median_05, _ = build_node_scores(
            df_sub, node_map, covars_used,
            agg="median", min_nonmiss_frac=0.5
        )
        node_mean_07, _ = build_node_scores(
            df_sub, node_map, covars_used,
            agg="mean", min_nonmiss_frac=0.7
        )

        settings = {
            "main_mean05": (node_mean_05, paths_all, seed + 1),
            "rev_mean05": (node_mean_05, reverse_paths(paths_all), seed + 2),
            "main_median05": (node_median_05, paths_all, seed + 3),
            "rev_median05": (node_median_05, reverse_paths(paths_all), seed + 4),
            "main_mean07": (node_mean_07, paths_all, seed + 5),
            "rev_mean07": (node_mean_07, reverse_paths(paths_all), seed + 6),
        }

        saved = []

        for name, (node_scores, paths, sd) in settings.items():

            df_res, df_boot = run_all_paths(
                node_scores,
                paths,
                sd,
                n_boot,
                name,
                covars_used,
                return_boot=True
            )

            csv_path = OUT_DIR / f"{name}.csv"
            df_res.to_csv(csv_path, index=False)

            boot_path = OUT_DIR / f"{name}_boot.parquet"
            df_boot.to_parquet(boot_path, index=False)

            saved.append(str(csv_path))
            saved.append(str(boot_path))

        meta["subgroups"][sg_name] = {
            "n": int(n_sub),
            "covariates_used": covars_used,
            "files": saved
        }

    with open(OUT_BASE / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{subgroup_name} subgroup analysis finished.")
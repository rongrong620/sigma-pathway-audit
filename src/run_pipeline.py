# run_pipeline.py

import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

def run(script_name):
    print(f"\nRunning {script_name} ...")
    subprocess.run(["python", str(SRC_DIR / script_name)], check=True)

if __name__ == "__main__":

    # ---------- Main pipeline ----------
    run("stage01_merge_sleep_anchor.py")
    run("stage02_descriptive_sanity.py")
    run("stage03_node_level_corr.py")
    run("stage03_variable_variable_corr.py")
    run("stage04_mediation.py")

    # ---------- Subgroups ----------
    run("run_subgroup_sex.py")
    run("run_subgroup_age.py")
    run("run_subgroup_bmi.py")
    run("run_subgroup_menopause.py")

    print("\nAll pipeline steps completed successfully.")
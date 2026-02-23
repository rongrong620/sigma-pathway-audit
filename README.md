# Sigma Pathway Audit

Biology-guided causal pathway auditing and mediation analysis pipeline.

This repository contains a reproducible codebase for structured causal pathway validation, including:

- Node-level correlation analysis
- Variable-level correlation analysis
- Mediation analysis (bootstrap-based inference)
- Direction reversal sensitivity
- Aggregation sensitivity (mean vs median)
- Missingness threshold sensitivity
- Subgroup analyses (age, BMI, sex, menopause)

## Repository Structure
src/
├── stage01_merge_sleep_anchor.py
├── stage02_descriptive_sanity.py
├── stage03_node_level_corr.py
├── stage03_variable_variable_corr.py
├── stage04_mediation.py
├── mediation_paths.py
├── node_map.py
├── run_pipeline.py
├── subgroup_runner.py
└── run_subgroup_*.py

## Pipeline Overview

1. Data merging and preprocessing  
2. Sanity checks and descriptive validation  
3. Correlation analysis (node and variable levels)  
4. Mediation modeling with bootstrap inference  
5. Sensitivity analyses  
6. Subgroup-specific pathway auditing  

## Reproducibility

Recommended environment:
- Python 3.10+
- numpy, pandas, scipy, statsmodels, tqdm

## License

Academic research use.# sigma-pathway-audit

Biology-guided causal pathway auditing & mediation analysis pipeline.

## Contents
- `src/`: stage scripts (correlation + mediation), path definitions, runners
- sensitivity analyses: direction reversal, aggregation (mean/median), missingness thresholds

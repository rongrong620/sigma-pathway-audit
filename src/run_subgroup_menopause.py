df_female = df_all[df_all["sex"] == 0].copy()

SUBGROUPS = {
    "pre_menopause": df_female["age"] < 50,
    "post_menopause": df_female["age"] >= 55,
}

run_subgroup_analysis(
    df_all=df_female,
    subgroup_dict=SUBGROUPS,
    subgroup_name="menopause",
    ...
)
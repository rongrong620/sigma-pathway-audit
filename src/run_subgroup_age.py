SUBGROUPS = {
    "age_18_49": (df_all["age"] >= 18) & (df_all["age"] < 50),
    "age_50_64": (df_all["age"] >= 50) & (df_all["age"] < 65),
    "age_65_plus": df_all["age"] >= 65,
}
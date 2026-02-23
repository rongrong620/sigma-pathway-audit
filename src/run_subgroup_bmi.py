q1 = df_all["bmi"].quantile(0.25)
q3 = df_all["bmi"].quantile(0.75)

SUBGROUPS = {
    "low_bmi": df_all["bmi"] <= q1,
    "mid_bmi": (df_all["bmi"] > q1) & (df_all["bmi"] < q3),
    "high_bmi": df_all["bmi"] >= q3,
}
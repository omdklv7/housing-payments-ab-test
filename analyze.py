#!/usr/bin/env python3
"""Reproducible A/B analysis for utility bill payment form.

Usage:
    python src/analyze.py --input data/data.xlsx

No raw experiment records are committed to this repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

STEPS = ["step1_opened", "step2_entered", "step3_confirmed", "step4_success"]
GROUPS = ("A", "B")


def calculate(input_file: Path, output_dir: Path, figures_dir: Path) -> dict:
    users = pd.read_excel(input_file, sheet_name="Пользователи")
    payments = pd.read_excel(input_file, sheet_name="Платежи")
    raw_rows = {"users": len(users), "payments": len(payments)}

    for df in (users, payments):
        df["group"] = df["group"].astype("string").str.strip()

    assert users["user_id"].notna().all(), "User table contains missing user_id"
    assert not users["user_id"].duplicated().any(), "User IDs are not unique"
    assert not payments["payment_id"].duplicated().any(), "Payment IDs are not unique"

    # Allocation is by user. Cross-group records are excluded conservatively.
    matched = payments.merge(
        users[["user_id", "group"]], on="user_id", how="left",
        suffixes=("_payment", "_user"), validate="many_to_one"
    )
    group_mismatch = matched.loc[
        matched["group_user"].isin(GROUPS)
        & matched["group_payment"].isin(GROUPS)
        & matched["group_user"].ne(matched["group_payment"]), "user_id"
    ].dropna().unique().tolist()
    mixed_groups = payments.groupby("user_id")["group"].nunique(dropna=True)
    ambiguous_ids = set(group_mismatch) | set(mixed_groups[mixed_groups > 1].index)

    users_clean = users.loc[
        users["group"].isin(GROUPS) & ~users["user_id"].isin(ambiguous_ids)
    ].copy()
    payments_clean = payments.loc[
        payments["user_id"].isin(users_clean["user_id"])
    ].copy()
    check = payments_clean.merge(users_clean[["user_id", "group"]], on="user_id", how="left", suffixes=("_payment", "_user"), validate="many_to_one")
    assert check["group_payment"].eq(check["group_user"]).all(), "Group mismatch remains"

    users_clean["age"] = pd.to_numeric(users_clean["age"], errors="coerce")
    users_clean.loc[~users_clean["age"].between(18, 100), "age"] = np.nan
    users_clean["city"] = users_clean["city"].replace(["NaN", "nan", "Неизвестен", ""], np.nan)
    for step in STEPS:
        payments_clean[step] = pd.to_datetime(payments_clean[step], errors="coerce")

    bad_order = pd.Series(False, index=payments_clean.index)
    for before, after in zip(STEPS[:-1], STEPS[1:]):
        bad_order |= payments_clean[after].notna() & (
            payments_clean[before].isna() | (payments_clean[after] < payments_clean[before])
        )
    if bad_order.any():
        raise ValueError(f"Out-of-order timestamps: {int(bad_order.sum())}")
    start, end = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-01")
    bad_period = payments_clean["step1_opened"].isna() | ~payments_clean["step1_opened"].between(start, end, inclusive="left")
    if bad_period.any():
        raise ValueError(f"Outside experiment period: {int(bad_period.sum())}")

    attempt_counts = payments_clean.groupby("group")[STEPS].count()
    user_funnel = (payments_clean.groupby(["group", "user_id"])[STEPS]
                   .agg(lambda s: s.notna().any()).reset_index())
    user_counts = user_funnel.groupby("group")[STEPS].sum().astype(int)

    success_a, n_a = int(user_counts.loc["A", STEPS[-1]]), int(user_counts.loc["A", STEPS[0]])
    success_b, n_b = int(user_counts.loc["B", STEPS[-1]]), int(user_counts.loc["B", STEPS[0]])
    p_a, p_b = success_a / n_a, success_b / n_b
    z, p_val = proportions_ztest([success_b, success_a], [n_b, n_a], alternative="two-sided")
    low, high = confint_proportions_2indep(success_b, n_b, success_a, n_a, compare="diff", method="newcomb")

    first_attempts = payments_clean.sort_values(["user_id", "step1_opened", "payment_id"]).drop_duplicates("user_id").copy()
    first_attempts["success"] = first_attempts[STEPS[-1]].notna()
    first_stats = first_attempts.groupby("group")["success"].agg(["sum", "count"])
    f_a, f_b = int(first_stats.loc["A", "sum"]), int(first_stats.loc["B", "sum"])
    fn_a, fn_b = int(first_stats.loc["A", "count"]), int(first_stats.loc["B", "count"])
    first_z, first_p = proportions_ztest([f_b, f_a], [fn_b, fn_a])
    first_low, first_high = confint_proportions_2indep(f_b, fn_b, f_a, fn_a, compare="diff", method="newcomb")

    per_user = payments_clean.groupby(["group", "user_id"]).agg(
        attempts=("payment_id", "count"), successes=(STEPS[-1], "count")
    ).reset_index()
    averages = per_user.groupby("group")[["attempts", "successes"]].mean()

    segments = users_clean[["user_id", "group", "device_type", "city", "age"]].merge(
        user_funnel[["user_id", STEPS[0], STEPS[-1]]], on="user_id", how="left", validate="one_to_one"
    )
    segments = segments[segments[STEPS[0]] == True].copy()
    segments["age_group"] = pd.cut(segments["age"], bins=[18,25,35,45,101], right=False, labels=["18–24","25–34","35–44","45+"])
    seg_results = {}
    for col in ("device_type", "city", "age_group"):
        grouped = segments.groupby([col,"group"], dropna=False, observed=True)[STEPS[-1]].agg(["sum", "count"]).reset_index()
        records=[]
        for _, r in grouped.iterrows():
            records.append({"segment": None if pd.isna(r[col]) else str(r[col]), "group":str(r["group"]), "n":int(r["count"]), "successful":int(r["sum"]), "conversion":round(float(r["sum"] / r["count"]), 6)})
        seg_results[col]=records

    # Per-user median across their attempts reaching each transition.
    time_results = []
    for label, a, b in zip(["1→2", "2→3", "3→4"], STEPS[:-1], STEPS[1:]):
        d = payments_clean.dropna(subset=[a,b]).copy()
        d["minutes"] = (d[b]-d[a]).dt.total_seconds()/60
        d = d[d["minutes"] >= 0]
        medians = d.groupby(["group","user_id"])["minutes"].median().reset_index()
        for group in GROUPS:
            x = medians.loc[medians["group"]==group, "minutes"]
            time_results.append({"transition":label, "group":group, "users":len(x), "median_minutes":float(x.median()), "mean_minutes":round(float(x.mean()),3)})

    # Visual for the repository README.
    stages = ["Открытие", "Реквизиты", "Подтверждение", "Оплата"]
    fig, ax = plt.subplots(figsize=(9.3, 4.8))
    for group in GROUPS:
        vals = user_counts.loc[group, STEPS].to_numpy(dtype=float) / user_counts.loc[group, STEPS[0]] * 100
        ax.plot(stages, vals, marker="o", linewidth=2.5, label=f"Группа {group}")
        for ix,v in enumerate(vals):
            ax.annotate(f"{v:.1f}%", (ix,v), xytext=(0,9 if group=="B" else -16), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylim(55,108)
    ax.set_ylabel("Доля от открывших форму, %")
    ax.set_title("Пользовательская воронка оплаты ЖКХ — A/B-тест")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir/"funnel_ab.png", dpi=180)
    plt.close(fig)

    result = {
        "experiment_dates": "2024-04-01 — 2024-04-30",
        "source_rows": raw_rows,
        "clean_rows": {"users":len(users_clean), "payments":len(payments_clean)},
        "excluded_ambiguous_user_count": len(ambiguous_ids),
        "allocation_A_B": users_clean["group"].value_counts().to_dict(),
        "attempt_funnel": {g:attempt_counts.loc[g, STEPS].astype(int).tolist() for g in GROUPS},
        "user_funnel": {g:user_counts.loc[g, STEPS].astype(int).tolist() for g in GROUPS},
        "primary": {
            "conversion_A": p_a, "conversion_B": p_b,
            "effect_pp":100*(p_b-p_a), "relative_uplift_pct":100*(p_b/p_a-1),
            "z_stat":float(z), "p_value":float(p_val), "ci_95_pp":[100*low,100*high]
        },
        "first_attempt": {
            "A": {"successes":f_a,"users":fn_a,"rate":f_a/fn_a},
            "B": {"successes":f_b,"users":fn_b,"rate":f_b/fn_b},
            "effect_pp":100*(f_b/fn_b-f_a/fn_a), "z_stat":float(first_z),
            "p_value":float(first_p), "ci_95_pp":[100*first_low,100*first_high]
        },
        "activity": {g:{"avg_attempts":float(averages.loc[g,"attempts"]),"avg_successes":float(averages.loc[g,"successes"])} for g in GROUPS},
        "segments":seg_results,
        "time":time_results,
        "notes": ["Randomization unit: user", "Age outside 18–100 is retained as missing, not grounds for exclusion", "Timing comparison is conditional on reaching the downstream stage", "Time and segmentation results are exploratory"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/data.xlsx"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    result = calculate(args.input, args.results, args.figures)
    p = result["primary"]
    print(f"Users: {result['clean_rows']['users']}, payment attempts: {result['clean_rows']['payments']}")
    print(f"A: {p['conversion_A']:.2%}, B: {p['conversion_B']:.2%}")
    print(f"Effect: +{p['effect_pp']:.2f} pp, uplift: +{p['relative_uplift_pct']:.2f}%")
    print(f"p-value: {p['p_value']:.3g}, 95% CI: {p['ci_95_pp']}")


if __name__ == "__main__":
    main()

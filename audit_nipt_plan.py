from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr
from scipy.stats import pearsonr, spearmanr


DATA = Path("附件.xlsx")
THRESHOLD = 0.04


def parse_ga(value: object) -> float:
    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d+)w(?:\+(\d+))?", text)
    if not match:
        raise ValueError(f"Unrecognized gestational age: {value!r}")
    weeks = int(match.group(1))
    days = int(match.group(2) or 0)
    return weeks + days / 7.0


def parse_date(value: object) -> pd.Timestamp:
    if isinstance(value, (int, np.integer)):
        return pd.to_datetime(str(int(value)), format="%Y%m%d")
    if isinstance(value, float) and value.is_integer():
        return pd.to_datetime(str(int(value)), format="%Y%m%d")
    return pd.to_datetime(value)


def stable_log_interval_prob(z_left: np.ndarray, z_right: np.ndarray) -> np.ndarray:
    log_left = log_ndtr(z_left)
    log_right = log_ndtr(z_right)
    ratio = np.exp(np.minimum(log_left - log_right, -np.finfo(float).eps))
    return log_right + np.log1p(-ratio)


def fit_lognormal_aft(intervals: pd.DataFrame) -> tuple[np.ndarray, object]:
    bmi = intervals["first_bmi"].to_numpy(float)
    bmi_mean = bmi.mean()
    bmi_sd = bmi.std(ddof=0)
    x = (bmi - bmi_mean) / bmi_sd
    left = intervals["left"].to_numpy(float)
    right = intervals["right"].to_numpy(float)
    left_censored = left == 0
    right_censored = np.isinf(right)
    interval_censored = ~(left_censored | right_censored)

    def objective(params: np.ndarray) -> float:
        mu = params[0] + params[1] * x
        sigma = np.exp(params[2])
        ll = np.empty(len(intervals), dtype=float)
        z_right = (np.log(right[left_censored]) - mu[left_censored]) / sigma
        ll[left_censored] = log_ndtr(z_right)
        z_left = (np.log(left[right_censored]) - mu[right_censored]) / sigma
        ll[right_censored] = log_ndtr(-z_left)
        z_left_i = (np.log(left[interval_censored]) - mu[interval_censored]) / sigma
        z_right_i = (np.log(right[interval_censored]) - mu[interval_censored]) / sigma
        ll[interval_censored] = stable_log_interval_prob(z_left_i, z_right_i)
        if not np.all(np.isfinite(ll)):
            return 1e100
        return -float(ll.sum())

    starts = [
        np.array([2.05, 0.12, np.log(0.53)]),
        np.array([2.3, 0.0, np.log(0.3)]),
        np.array([2.0, 0.2, np.log(0.8)]),
    ]
    fits = [minimize(objective, start, method="BFGS") for start in starts]
    fit = min(fits, key=lambda item: item.fun)
    params = fit.x.copy()
    return np.array([params[0], params[1], np.exp(params[2]), bmi_mean, bmi_sd]), fit


def fit_parametric_aft(intervals: pd.DataFrame, distribution: str) -> tuple[np.ndarray, object]:
    bmi = intervals["first_bmi"].to_numpy(float)
    bmi_mean = bmi.mean()
    bmi_sd = bmi.std(ddof=0)
    x = (bmi - bmi_mean) / bmi_sd
    left = intervals["left"].to_numpy(float)
    right = intervals["right"].to_numpy(float)
    left_censored = left == 0
    right_censored = np.isinf(right)
    interval_censored = ~(left_censored | right_censored)

    def cdf(z: np.ndarray) -> np.ndarray:
        if distribution == "lognormal":
            return ndtr(z)
        if distribution == "loglogistic":
            return expit(z)
        if distribution == "weibull":
            return -np.expm1(-np.exp(np.clip(z, -700, 50)))
        raise ValueError(distribution)

    def survival(z: np.ndarray) -> np.ndarray:
        if distribution == "lognormal":
            return ndtr(-z)
        if distribution == "loglogistic":
            return expit(-z)
        if distribution == "weibull":
            return np.exp(-np.exp(np.clip(z, -700, 50)))
        raise ValueError(distribution)

    def objective(params: np.ndarray) -> float:
        mu = params[0] + params[1] * x
        sigma = np.exp(params[2])
        ll = np.empty(len(intervals), dtype=float)
        z_right = (np.log(right[left_censored]) - mu[left_censored]) / sigma
        ll[left_censored] = np.log(np.clip(cdf(z_right), 1e-300, 1))
        z_left = (np.log(left[right_censored]) - mu[right_censored]) / sigma
        ll[right_censored] = np.log(np.clip(survival(z_left), 1e-300, 1))
        z_left_i = (np.log(left[interval_censored]) - mu[interval_censored]) / sigma
        z_right_i = (np.log(right[interval_censored]) - mu[interval_censored]) / sigma
        probability = cdf(z_right_i) - cdf(z_left_i)
        ll[interval_censored] = np.log(np.clip(probability, 1e-300, 1))
        return -float(ll.sum())

    starts = [
        np.array([2.05, 0.12, np.log(0.53)]),
        np.array([2.3, 0.0, np.log(0.3)]),
        np.array([2.0, 0.2, np.log(0.8)]),
    ]
    fits = [
        minimize(objective, start, method="L-BFGS-B", bounds=[(0, 5), (-3, 3), (-5, 2)])
        for start in starts
    ]
    fit = min(fits, key=lambda item: item.fun)
    return np.array([fit.x[0], fit.x[1], np.exp(fit.x[2]), bmi_mean, bmi_sd]), fit


def segmented_quantile_dp(
    bmi: np.ndarray,
    tau: np.ndarray,
    *,
    alpha: float,
    groups: int,
    min_size: int,
    min_width: float,
    quantile_method: str,
) -> list[tuple[int, int, float, float]]:
    order = np.argsort(bmi, kind="stable")
    bmi = bmi[order]
    tau = tau[order]
    n = len(bmi)
    cost = np.full((n, n), np.inf)
    time = np.full((n, n), np.nan)
    for start in range(n):
        for end in range(start + min_size - 1, n):
            if bmi[end] - bmi[start] < min_width:
                continue
            values = tau[start : end + 1]
            t_group = float(np.quantile(values, alpha, method=quantile_method))
            residual = values - t_group
            check = residual * (alpha - (residual < 0).astype(float))
            cost[start, end] = float(check.sum())
            time[start, end] = t_group
    dp = np.full((groups, n), np.inf)
    previous = np.full((groups, n), -1, dtype=int)
    dp[0] = cost[0]
    for group_idx in range(1, groups):
        for end in range(n):
            for previous_end in range(end):
                candidate = dp[group_idx - 1, previous_end] + cost[previous_end + 1, end]
                if candidate < dp[group_idx, end]:
                    dp[group_idx, end] = candidate
                    previous[group_idx, end] = previous_end
    segments: list[tuple[int, int, float, float]] = []
    end = n - 1
    for group_idx in range(groups - 1, -1, -1):
        previous_end = previous[group_idx, end] if group_idx else -1
        start = previous_end + 1
        values = tau[start : end + 1]
        t_group = float(time[start, end])
        coverage = float(np.mean(values <= t_group))
        segments.append((start, end, t_group, coverage))
        end = previous_end
    segments.reverse()
    return segments


def main() -> None:
    male = pd.read_excel(DATA, sheet_name="男胎检测数据")
    female = pd.read_excel(DATA, sheet_name="女胎检测数据")
    male.columns = male.columns.str.strip()
    female.columns = female.columns.str.strip()
    male["ga"] = male["检测孕周"].map(parse_ga)
    female["ga"] = female["检测孕周"].map(parse_ga)
    male["date_norm"] = male["检测日期"].map(parse_date)
    female["date_norm"] = female["检测日期"].map(parse_date)

    strict_visit_key = ["孕妇代码", "date_norm", "检测抽血次数", "ga"]
    strict_visit_sizes = male.groupby(strict_visit_key, dropna=False).size()
    visit_key = ["孕妇代码", "检测抽血次数", "ga"]
    visit_sizes = male.groupby(visit_key, dropna=False).size()
    repeat_sizes = visit_sizes[visit_sizes > 1]
    male_visits = (
        male.sort_values(["孕妇代码", "ga", "date_norm"])
        .groupby(visit_key, as_index=False, dropna=False)
        .agg(
            date_norm=("date_norm", "min"),
            y=("Y染色体浓度", "median"),
            bmi=("孕妇BMI", "median"),
            age=("年龄", "first"),
            height=("身高", "first"),
            weight=("体重", "median"),
            n_tech=("Y染色体浓度", "size"),
        )
        .sort_values(["孕妇代码", "ga", "date_norm"])
    )

    pooled_ss = 0.0
    pooled_df = 0
    pair_diffs: list[float] = []
    for _, group in male.groupby(visit_key, dropna=False):
        if len(group) <= 1:
            continue
        values = group["Y染色体浓度"].to_numpy(float)
        pooled_ss += float(((values - values.mean()) ** 2).sum())
        pooled_df += len(values) - 1
        pair_diffs.extend(abs(a - b) for a, b in itertools.combinations(values, 2))

    print("=== BASIC STRUCTURE ===")
    print(
        {
            "male_rows": len(male),
            "male_women": male["孕妇代码"].nunique(),
            "male_visits": len(male_visits),
            "repeat_visit_groups": len(repeat_sizes),
            "repeat_group_sizes": repeat_sizes.value_counts().sort_index().to_dict(),
            "strict_date_key_visits": len(strict_visit_sizes),
            "strict_date_key_repeat_groups": int((strict_visit_sizes > 1).sum()),
            "female_rows": len(female),
            "female_women": female["孕妇代码"].nunique(),
        }
    )
    print(
        {
            "pooled_tech_sd": np.sqrt(pooled_ss / pooled_df),
            "pairwise_abs_diff_median": float(np.median(pair_diffs)),
            "y_min": male["Y染色体浓度"].min(),
            "y_max": male["Y染色体浓度"].max(),
            "y_nonpositive": int((male["Y染色体浓度"] <= 0).sum()),
            "ga_min": male["ga"].min(),
            "ga_max": male["ga"].max(),
        }
    )

    print("\n=== Q1 QUICK CHECKS ===")
    y = male["Y染色体浓度"].to_numpy(float)
    bmi = male["孕妇BMI"].to_numpy(float)
    ga = male["ga"].to_numpy(float)
    print("pearson_y_bmi", pearsonr(y, bmi))
    print("pearson_y_ga", pearsonr(y, ga))
    print("spearman_y_bmi", spearmanr(y, bmi))
    print("spearman_y_ga", spearmanr(y, ga))
    design = np.column_stack([np.ones(len(male)), bmi, ga])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    r2 = 1 - np.square(y - pred).sum() / np.square(y - y.mean()).sum()
    print("ols_intercept_bmi_ga_r2", *coef, r2)

    print("\n=== THRESHOLD/CENSORING AUDIT ===")
    interval_rows: list[dict[str, float | str]] = []
    recross_women: list[str] = []
    multi_transition_women: list[str] = []
    credible_conflict_women: list[str] = []
    first_positive_then_negative_counts: list[int] = []
    sigma_tech = np.sqrt(pooled_ss / pooled_df)
    for code, group in male_visits.groupby("孕妇代码", sort=False):
        group = group.sort_values(["ga", "date_norm"])
        states = (group["y"].to_numpy(float) >= THRESHOLD).astype(int)
        transitions = int(np.count_nonzero(np.diff(states)))
        if transitions > 1:
            multi_transition_women.append(code)
        positive_positions = np.flatnonzero(states == 1)
        later_negative_count = 0
        if len(positive_positions):
            first_pos = int(positive_positions[0])
            later_negative_count = int(np.count_nonzero(states[first_pos + 1 :] == 0))
            if later_negative_count:
                recross_women.append(code)
        first_positive_then_negative_counts.append(later_negative_count)

        first = group.iloc[0]
        first_bmi = float(first["bmi"])
        if states[0] == 1:
            left, right, censor_type = 0.0, float(first["ga"]), "left"
        elif len(positive_positions):
            first_pos = int(positive_positions[0])
            left = float(group.iloc[first_pos - 1]["ga"])
            right = float(group.iloc[first_pos]["ga"])
            censor_type = "interval"
        else:
            left, right, censor_type = float(group.iloc[-1]["ga"]), np.inf, "right"
        interval_rows.append(
            {
                "code": code,
                "first_bmi": first_bmi,
                "left": left,
                "right": right,
                "type": censor_type,
            }
        )

        se = sigma_tech / np.sqrt(group["n_tech"].to_numpy(float))
        credible_neg = group.loc[group["y"].to_numpy(float) + 1.96 * se < THRESHOLD, "ga"]
        credible_pos = group.loc[group["y"].to_numpy(float) - 1.96 * se >= THRESHOLD, "ga"]
        if len(credible_neg) and len(credible_pos) and credible_neg.max() >= credible_pos.min():
            credible_conflict_women.append(code)

    intervals = pd.DataFrame(interval_rows)
    print("censor_counts", intervals["type"].value_counts().to_dict())
    print(
        {
            "positive_then_later_negative_women": len(recross_women),
            "multiple_state_transition_women": len(multi_transition_women),
            "credible_last_negative_not_before_first_positive": len(credible_conflict_women),
            "later_negative_visits_after_first_positive": int(sum(first_positive_then_negative_counts)),
        }
    )
    print("recross_examples", recross_women[:20])

    aft_params, aft_fit = fit_lognormal_aft(intervals)
    a0, beta, sigma, bmi_mean, bmi_sd = aft_params
    print("\n=== LOGNORMAL AFT REPRODUCTION ===")
    print(
        {
            "success": bool(aft_fit.success),
            "message": str(aft_fit.message),
            "neg_loglik": float(aft_fit.fun),
            "a0": a0,
            "beta_std_bmi": beta,
            "sigma": sigma,
            "bmi_mean": bmi_mean,
            "bmi_sd": bmi_sd,
        }
    )
    z90 = 1.2815515655446004
    for target_bmi in [28.5, 30.5, 36.0, 40.0]:
        mu = a0 + beta * (target_bmi - bmi_mean) / bmi_sd
        print("aft_q90", target_bmi, float(np.exp(mu + sigma * z90)))

    print("aft_distribution_sensitivity")
    quantile_error = {
        "lognormal": z90,
        "loglogistic": np.log(0.9 / 0.1),
        "weibull": np.log(-np.log(0.1)),
    }
    for distribution in ["lognormal", "loglogistic", "weibull"]:
        params_dist, fit_dist = fit_parametric_aft(intervals, distribution)
        d0, db, ds, dm, dsd = params_dist
        times = {}
        for target_bmi in [28.5, 30.5, 36.0, 40.0]:
            mu = d0 + db * (target_bmi - dm) / dsd
            times[target_bmi] = float(np.exp(mu + ds * quantile_error[distribution]))
        print(
            distribution,
            {
                "aic": float(2 * 3 + 2 * fit_dist.fun),
                "parameters": [float(d0), float(db), float(ds)],
                "q90": times,
            },
        )

    print("\n=== DP QUANTILE FEASIBILITY ===")
    bmi_array = intervals["first_bmi"].to_numpy(float)
    z80 = 0.8416212335729143
    tau80 = np.exp(a0 + beta * (bmi_array - bmi_mean) / bmi_sd + sigma * z80)
    order = np.argsort(bmi_array, kind="stable")
    sorted_bmi = bmi_array[order]
    for method in ["linear", "higher"]:
        segments = segmented_quantile_dp(
            bmi_array,
            tau80,
            alpha=0.80,
            groups=4,
            min_size=20,
            min_width=1.0,
            quantile_method=method,
        )
        print("quantile_method", method)
        for start, end, t_group, coverage in segments:
            print(
                {
                    "bmi_min": float(sorted_bmi[start]),
                    "bmi_max": float(sorted_bmi[end]),
                    "n": end - start + 1,
                    "time": t_group,
                    "coverage": coverage,
                }
            )

    first_bmi = intervals["first_bmi"]
    print("\n=== SUPPORT/EXTRAPOLATION ===")
    print(first_bmi.describe(percentiles=[0.01, 0.05, 0.1, 0.9, 0.95, 0.99]).to_dict())
    for cutoff in [20, 28, 32, 36, 40, 42]:
        print("first_bmi_ge", cutoff, int((first_bmi >= cutoff).sum()))
    first_ga = male_visits.groupby("孕妇代码", sort=False).first()["ga"]
    print("first_ga", first_ga.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict())

    print("\n=== Q4 LABEL/QUALITY AUDIT ===")
    label_text = female["染色体的非整倍体"].fillna("").astype(str)
    for chrom in [13, 18, 21]:
        target = label_text.str.contains(f"T{chrom}", regex=False)
        z_col = f"{chrom}号染色体的Z值"
        z = female[z_col]
        positive_sens = float((z[target] >= 3).mean()) if target.any() else np.nan
        abs_sens = float((z[target].abs() >= 3).mean()) if target.any() else np.nan
        print(
            f"T{chrom}",
            {
                "positive_records": int(target.sum()),
                "positive_women": int(female.loc[target, "孕妇代码"].nunique()),
                "z_ge_3_sensitivity": positive_sens,
                "abs_z_ge_3_sensitivity": abs_sens,
                "positive_z_range": [float(z[target].min()), float(z[target].max())],
            },
        )
    abnormal = label_text.ne("")
    woman_label_counts = female.assign(abnormal=abnormal).groupby("孕妇代码")["abnormal"].nunique()
    print(
        {
            "abnormal_records": int(abnormal.sum()),
            "abnormal_women": int(female.loc[abnormal, "孕妇代码"].nunique()),
            "women_with_record_level_label_disagreement": int((woman_label_counts > 1).sum()),
            "health_values": female["胎儿是否健康"].value_counts(dropna=False).to_dict(),
            "gc_outside_40_60": int(((female["GC含量"] < 0.40) | (female["GC含量"] > 0.60)).sum()),
        }
    )
    quality_cols = [
        "原始读段数",
        "在参考基因组上比对的比例",
        "重复读段的比例",
        "唯一比对的读段数",
        "GC含量",
        "被过滤掉读段数的比例",
    ]
    print("quality_spearman_corr")
    print(female[quality_cols].corr(method="spearman").round(3).to_string())


if __name__ == "__main__":
    main()

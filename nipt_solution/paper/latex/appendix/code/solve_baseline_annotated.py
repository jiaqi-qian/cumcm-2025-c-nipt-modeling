# -*- coding: utf-8 -*-
# =============================================================================
# solve_baseline.py  问题一至问题四的基线求解主流程
# 功能：串联数据准备 -> 问题一分段线性混合模型 -> 问题二区间删失 AFT 与
#       动态规划分组 -> 问题三多因素 AFT 比较 -> 问题四女胎标签审计，
#       全部中间结果以 CSV/JSON 落盘，保证可复现。
# 说明：数据文件默认取相对路径“附件.xlsx”，不含本地绝对路径。
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from nipt_core import (
    SEED,
    PreparedData,
    classify_visit_states,
    construct_intervals,
    determined_binary_at_time,
    exact_dp_policy,
    fit_aft,
    prepare_data,
    rounded_policy_table,
    sha256_file,
    support_status,
    to_jsonable,
    week_day,
    write_json,
)



# ---------- 问题一：分段线性随机截距混合模型（logit 变换浓度） ----------
def fit_q1_model(
    visits: pd.DataFrame,
    *,
    knot: float | None = None,
    interaction: bool = False,
    age_height: bool = False,
    random_slope: bool = False,
):
    frame = visits.copy()
    frame["z_y"] = np.log(frame["y"] / (1 - frame["y"]))
    centers = {
        "ga": float(frame["ga"].mean()),
        "first_bmi": float(frame["first_bmi"].mean()),
        "age": float(frame["age"].mean()),
        "height": float(frame["height"].mean()),
    }
    frame["ga_c"] = frame["ga"] - centers["ga"]
    frame["first_bmi_c"] = frame["first_bmi"] - centers["first_bmi"]
    frame["age_c"] = frame["age"] - centers["age"]
    frame["height_c"] = frame["height"] - centers["height"]
    columns = ["ga_c", "first_bmi_c", "delta_bmi"]
    if knot is not None:
        name = f"ga_hinge_{int(knot)}"
        frame[name] = np.maximum(frame["ga"] - knot, 0)
        columns.append(name)
    if interaction:
        frame["ga_bmi_interaction"] = frame["ga_c"] * frame["first_bmi_c"]
        columns.append("ga_bmi_interaction")
    if age_height:
        columns.extend(["age_c", "height_c"])
    x = sm.add_constant(frame[columns], has_constant="add")
    exog_re = None
    if random_slope:
        exog_re = sm.add_constant(frame[["ga_c"]], has_constant="add")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sm.MixedLM(
            frame["z_y"],
            x,
            groups=frame["孕妇代码"],
            exog_re=exog_re,
        ).fit(
            reml=False,
            method="lbfgs",
            maxiter=1500,
            disp=False,
        )
    warning_text = [str(item.message) for item in caught]
    return result, frame, columns, centers, warning_text



# ---------- 问题一求解：候选结构比较、系数与预测网格输出 ----------
def q1_solve(data: PreparedData, out: Path) -> dict:
    visits = data.male_visits.copy()
    candidates: list[dict] = []
    fitted: dict[str, tuple] = {}
    specs = [("linear", None, False, False)]
    specs.extend((f"ga_hinge_{k}", float(k), False, False) for k in range(13, 19))
    specs.extend(
        [
            ("interaction", None, True, False),
            ("age_height", None, False, True),
        ]
    )
    for name, knot, interaction, age_height in specs:
        try:
            fit_tuple = fit_q1_model(
                visits,
                knot=knot,
                interaction=interaction,
                age_height=age_height,
            )
            result = fit_tuple[0]
            fitted[name] = fit_tuple
            candidates.append(
                {
                    "model": name,
                    "aic": result.aic,
                    "bic": result.bic,
                    "loglik": result.llf,
                    "converged": bool(result.converged),
                    "warnings": " | ".join(fit_tuple[-1]),
                }
            )
        except Exception as exc:
            candidates.append(
                {
                    "model": name,
                    "aic": np.nan,
                    "bic": np.nan,
                    "loglik": np.nan,
                    "converged": False,
                    "warnings": repr(exc),
                }
            )
    comparison = pd.DataFrame(candidates).sort_values("bic")
    comparison.to_csv(out / "q1_model_comparison.csv", index=False)
    chosen_name = str(comparison.dropna(subset=["bic"]).iloc[0]["model"])
    chosen, frame, columns, centers, warning_text = fitted[chosen_name]

    conf = chosen.conf_int()
    coefficient_rows = []
    for name in chosen.fe_params.index:
        coefficient_rows.append(
            {
                "term": name,
                "estimate_logit": chosen.fe_params[name],
                "std_error": chosen.bse_fe[name],
                "z": chosen.fe_params[name] / chosen.bse_fe[name],
                "p_value": chosen.pvalues[name],
                "ci_low": conf.loc[name, 0],
                "ci_high": conf.loc[name, 1],
                "odds_ratio_per_unit": np.exp(chosen.fe_params[name]),
            }
        )
    coefficient_table = pd.DataFrame(coefficient_rows)
    coefficient_table.to_csv(out / "q1_coefficients.csv", index=False)

    random_intercept_var = float(chosen.cov_re.iloc[0, 0])
    residual_var = float(chosen.scale)
    icc = random_intercept_var / (random_intercept_var + residual_var)

    grid_rows = []
    for bmi in [28, 30, 32, 34, 36, 40]:
        for ga in [10, 11, 12, 14, 16, 18, 20, 22, 24, 25]:
            row = {
                "ga": ga,
                "first_bmi": bmi,
                "delta_bmi": 0.0,
                "age": centers["age"],
                "height": centers["height"],
            }
            values = {
                "ga_c": ga - centers["ga"],
                "first_bmi_c": bmi - centers["first_bmi"],
                "delta_bmi": 0.0,
                "age_c": 0.0,
                "height_c": 0.0,
                "ga_bmi_interaction": (ga - centers["ga"])
                * (bmi - centers["first_bmi"]),
            }
            for k in range(13, 19):
                values[f"ga_hinge_{k}"] = max(ga - k, 0)
            x = np.array([1.0] + [values[col] for col in columns])
            mean_logit = float(x @ chosen.fe_params.to_numpy())
            mean_y = float(expit(mean_logit))
            conditional_sd = np.sqrt(residual_var)
            marginal_sd = np.sqrt(residual_var + random_intercept_var)
            grid_rows.append(
                {
                    **row,
                    "predicted_y": mean_y,
                    "predicted_logit": mean_logit,
                    "conditional_low": float(expit(mean_logit - 1.96 * conditional_sd)),
                    "conditional_high": float(expit(mean_logit + 1.96 * conditional_sd)),
                    "new_woman_low": float(expit(mean_logit - 1.96 * marginal_sd)),
                    "new_woman_high": float(expit(mean_logit + 1.96 * marginal_sd)),
                }
            )
    prediction_grid = pd.DataFrame(grid_rows)
    prediction_grid.to_csv(out / "q1_effect_grid.csv", index=False)

    random_slope = {}
    try:
        knot = float(chosen_name.rsplit("_", 1)[-1]) if "hinge" in chosen_name else None
        rs_fit, _, _, _, rs_warnings = fit_q1_model(
            visits,
            knot=knot,
            interaction=chosen_name == "interaction",
            age_height=chosen_name == "age_height",
            random_slope=True,
        )
        random_slope = {
            "aic": rs_fit.aic,
            "bic": rs_fit.bic,
            "loglik": rs_fit.llf,
            "converged": bool(rs_fit.converged),
            "cov_re": rs_fit.cov_re.to_numpy(),
            "residual_var": rs_fit.scale,
            "warnings": rs_warnings,
        }
    except Exception as exc:
        random_slope = {"error": repr(exc)}

    summary = {
        "chosen_model": chosen_name,
        "centers": centers,
        "n_visits": len(visits),
        "n_women": int(visits["孕妇代码"].nunique()),
        "aic": chosen.aic,
        "bic": chosen.bic,
        "loglik": chosen.llf,
        "converged": bool(chosen.converged),
        "warnings": warning_text,
        "random_intercept_var": random_intercept_var,
        "random_intercept_sd": np.sqrt(random_intercept_var),
        "residual_var_logit": residual_var,
        "residual_sd_logit": np.sqrt(residual_var),
        "icc_logit": icc,
        "random_slope_sensitivity": random_slope,
    }
    write_json(out / "q1_summary.json", summary)
    return summary


def interval_summary(intervals: pd.DataFrame) -> dict:
    return {
        "n_total": len(intervals),
        "counts": intervals["type"].value_counts(dropna=False).to_dict(),
        "state_conflicts": int(intervals["state_conflict"].sum()),
        "uncertain_visits": int(intervals["n_uncertain"].sum()),
        "uninformative_women": int((intervals["type"] == "uninformative").sum()),
    }


def choose_policy_k(cost_table: pd.DataFrame) -> int:
    usable = cost_table.dropna(subset=["total_cost"]).sort_values("groups")
    cost1 = float(usable.loc[usable["groups"] == 1, "total_cost"].iloc[0])
    best = float(usable["total_cost"].min())
    denominator = max(cost1 - best, 1e-12)
    for _, row in usable.iterrows():
        if int(row["groups"]) < 2:
            continue
        captured = (cost1 - float(row["total_cost"])) / denominator
        if captured >= 0.90:
            return int(row["groups"])
    return int(usable.iloc[-1]["groups"])



# ---------- 问题二求解：多判定模式 -> AFT -> 动态规划分组排程 ----------
def q2_solve(data: PreparedData, out: Path) -> dict:
    modes: dict[str, pd.DataFrame] = {}
    hard_states = classify_visit_states(
        data.male_visits, data.sigma_tech, mode="hard"
    )
    modes["hard"] = construct_intervals(hard_states, data.male_baseline)
    credible_states = classify_visit_states(
        data.male_visits,
        data.sigma_tech,
        mode="credible",
        eta=0.025,
    )
    modes["credible_eta025"] = construct_intervals(
        credible_states, data.male_baseline
    )
    for eta in [0.01, 0.05, 0.10]:
        states = classify_visit_states(
            data.male_visits,
            data.sigma_tech,
            mode="credible",
            eta=eta,
        )
        modes[f"credible_eta{str(eta).replace('.', '')}"] = construct_intervals(
            states, data.male_baseline
        )
    for name, intervals in modes.items():
        intervals.to_csv(out / f"q2_intervals_{name}.csv", index=False)

    fit_rows: list[dict] = []
    fit_map = {}
    for mode_name in ["hard", "credible_eta025"]:
        intervals = modes[mode_name]
        for distribution in ["lognormal", "loglogistic", "weibull"]:
            fit = fit_aft(
                intervals,
                ["first_bmi"],
                distribution=distribution,
                nonnegative_features=["first_bmi"],
            )
            fit_map[(mode_name, distribution)] = fit
            fit_rows.append({"mode": mode_name, **fit.to_dict()})
    aft_comparison = pd.DataFrame(fit_rows)
    aft_comparison.to_csv(out / "q2_aft_comparison.csv", index=False)

    q_grid_rows = []
    for (mode_name, distribution), fit in fit_map.items():
        grid = pd.DataFrame({"first_bmi": [28.5, 30.5, 32, 34, 36, 40, 42, 46.875]})
        for rho in [0.80, 0.90, 0.95]:
            values = fit.quantile(rho, grid)
            for bmi, value in zip(grid["first_bmi"], values):
                q_grid_rows.append(
                    {
                        "mode": mode_name,
                        "distribution": distribution,
                        "rho": rho,
                        "bmi": bmi,
                        "quantile_week": value,
                        "week_day": week_day(value),
                        "within_10_25": 10 <= value <= 25,
                    }
                )
    pd.DataFrame(q_grid_rows).to_csv(out / "q2_aft_quantile_grid.csv", index=False)

    primary_fit = min(
        [
            fit_map[("hard", distribution)]
            for distribution in ["lognormal", "loglogistic", "weibull"]
        ],
        key=lambda item: item.aic,
    )
    credible_fit = min(
        [
            fit_map[("credible_eta025", distribution)]
            for distribution in ["lognormal", "loglogistic", "weibull"]
        ],
        key=lambda item: item.aic,
    )

    baseline = data.male_baseline.sort_values("孕妇代码").reset_index(drop=True)
    bmi = baseline["first_bmi"].to_numpy(float)
    policies_summary = []
    chosen_tables = {}
    for model_name, fit in [("hard", primary_fit), ("credible", credible_fit)]:
        for rho in [0.80, 0.90]:
            tau = fit.quantile(rho, baseline)
            cost_rows = []
            policy_map = {}
            for groups in range(1, 7):
                try:
                    policy = exact_dp_policy(
                        bmi,
                        tau,
                        groups=groups,
                        alpha=rho,
                        min_size=20,
                    )
                    policy_map[groups] = policy
                    cost_rows.append(
                        {
                            "model": model_name,
                            "rho": rho,
                            "groups": groups,
                            "total_cost": policy.total_cost,
                            "cutpoints": json.dumps(policy.cutpoints),
                        }
                    )
                except Exception:
                    cost_rows.append(
                        {
                            "model": model_name,
                            "rho": rho,
                            "groups": groups,
                            "total_cost": np.nan,
                            "cutpoints": "[]",
                        }
                    )
            costs = pd.DataFrame(cost_rows)
            selected_k = choose_policy_k(costs)
            selected = policy_map[selected_k]
            table, rounded = rounded_policy_table(
                bmi,
                tau,
                fit,
                baseline,
                cutpoints=selected.cutpoints,
                alpha=rho,
            )
            table.insert(0, "model", model_name)
            table.insert(1, "rho", rho)
            table.insert(2, "selected_k", selected_k)
            table["raw_cutpoints"] = json.dumps(selected.cutpoints)
            table["rounded_cutpoints"] = json.dumps(rounded)
            key = f"{model_name}_rho{int(rho * 100)}"
            table.to_csv(out / f"q2_policy_{key}.csv", index=False)
            costs.to_csv(out / f"q2_policy_costs_{key}.csv", index=False)
            chosen_tables[key] = table
            policies_summary.append(
                {
                    "key": key,
                    "selected_k": selected_k,
                    "raw_cutpoints": selected.cutpoints,
                    "rounded_cutpoints": rounded,
                    "total_cost": selected.total_cost,
                    "out_of_window_individuals": int(np.sum(tau > 25)),
                }
            )

    state_sensitivity = {
        name: interval_summary(intervals) for name, intervals in modes.items()
    }
    summary = {
        "intervals": state_sensitivity,
        "primary_point_model": primary_fit.to_dict(),
        "credible_point_model": credible_fit.to_dict(),
        "policies": policies_summary,
    }
    write_json(out / "q2_summary.json", summary)
    return summary



# ---------- 问题三求解：多因素 AFT 候选集与样本内校准 ----------
def q3_solve(data: PreparedData, out: Path) -> dict:
    hard_states = classify_visit_states(
        data.male_visits, data.sigma_tech, mode="hard"
    )
    intervals = construct_intervals(hard_states, data.male_baseline)
    candidate_features = {
        "bmi_only": ["first_bmi"],
        "bmi_age_height": ["first_bmi", "age", "height"],
        "bmi_full": [
            "first_bmi",
            "age",
            "height",
            "gravidity",
            "parity",
            "ivf_iui",
        ],
        "weight_height_full": [
            "first_weight",
            "height",
            "age",
            "gravidity",
            "parity",
            "ivf_iui",
        ],
    }
    fits = {}
    rows = []
    calibration_rows = []
    for name, features in candidate_features.items():
        nonnegative = ["first_bmi"] if "first_bmi" in features else []
        fit = fit_aft(
            intervals,
            features,
            distribution="lognormal",
            nonnegative_features=nonnegative,
            ridge=0.0,
        )
        fits[name] = fit
        rows.append({"model": name, **fit.to_dict()})
        for time_point in [12, 14, 16, 18, 20]:
            known, labels = determined_binary_at_time(intervals, time_point)
            subset = intervals.loc[known].copy()
            probabilities = fit.cdf(time_point, subset)
            calibration_rows.append(
                {
                    "model": name,
                    "time": time_point,
                    "n_known": int(known.sum()),
                    "prevalence": float(labels.mean()),
                    "mean_predicted": float(probabilities.mean()),
                    "brier_in_sample": float(np.mean(np.square(labels - probabilities))),
                }
            )
    comparison = pd.DataFrame(rows).sort_values("bic")
    comparison.to_csv(out / "q3_aft_comparison.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        out / "q3_time_calibration_in_sample.csv", index=False
    )
    best_name = str(comparison.iloc[0]["model"])
    summary = {
        "best_bic_model": best_name,
        "bmi_only": fits["bmi_only"].to_dict(),
        "best": fits[best_name].to_dict(),
        "bic_improvement_vs_bmi": fits["bmi_only"].bic - fits[best_name].bic,
        "aic_improvement_vs_bmi": fits["bmi_only"].aic - fits[best_name].aic,
    }
    write_json(out / "q3_summary.json", summary)
    return summary



# ---------- 问题四前置：女胎阳性标签与 Z 值阈值审计 ----------
def female_audit(data: PreparedData, out: Path) -> dict:
    female = data.female_raw.copy()
    label_text = female["染色体的非整倍体"].fillna("").astype(str)
    audit = {
        "records": len(female),
        "women": int(female["孕妇代码"].nunique()),
        "abnormal_records": int(label_text.ne("").sum()),
        "abnormal_women": int(
            female.loc[label_text.ne(""), "孕妇代码"].nunique()
        ),
        "gc_outside_40_60": int(
            ((female["GC含量"] < 0.40) | (female["GC含量"] > 0.60)).sum()
        ),
        "labels": {},
    }
    rows = []
    for chrom in [13, 18, 21]:
        y = label_text.str.contains(f"T{chrom}", regex=False)
        z = female[f"{chrom}号染色体的Z值"].to_numpy(float)
        audit["labels"][f"T{chrom}"] = {
            "positive_records": int(y.sum()),
            "positive_women": int(female.loc[y, "孕妇代码"].nunique()),
            "z_ge_3_sensitivity": float(np.mean(z[y] >= 3)),
            "abs_z_ge_3_sensitivity": float(np.mean(np.abs(z[y]) >= 3)),
            "positive_z_min": float(np.min(z[y])),
            "positive_z_max": float(np.max(z[y])),
        }
        rows.append({"label": f"T{chrom}", **audit["labels"][f"T{chrom}"]})
    pd.DataFrame(rows).to_csv(out / "q4_label_audit.csv", index=False)
    write_json(out / "q4_audit_summary.json", audit)
    return audit



# ---------- 主入口：解析参数、串联四问、落盘全部结果 ----------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("附件.xlsx"))
    parser.add_argument(
        "--output", type=Path, default=Path("nipt_solution/outputs/baseline")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = prepare_data(args.data)
    write_json(args.output / "data_audit.json", data.technical_summary)
    support_status(data.male_baseline["first_bmi"].to_numpy()).to_csv(
        args.output / "male_bmi_support.csv", index=False
    )
    summaries = {
        "metadata": {
            "seed": SEED,
            "data": str(args.data),
            "data_sha256": sha256_file(args.data),
            "python": sys.version,
        },
        "q1": q1_solve(data, args.output),
        "q2": q2_solve(data, args.output),
        "q3": q3_solve(data, args.output),
        "q4_audit": female_audit(data, args.output),
    }
    summaries["metadata"]["elapsed_seconds"] = time.time() - started
    write_json(args.output / "baseline_manifest.json", summaries)
    print(json.dumps(to_jsonable(summaries), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

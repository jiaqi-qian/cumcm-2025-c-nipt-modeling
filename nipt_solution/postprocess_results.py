from __future__ import annotations

import argparse
import ast
import json
import os
import platform
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nipt_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nipt_xdg_cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from nipt_core import (
    exact_dp_policy,
    exact_upper_quantile,
    per_group_row_weights,
    prepare_data,
    sha256_file,
    to_jsonable,
    weighted_mean,
    week_day,
    write_json,
)
from run_q4_models import prepare_female


def empirical_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.median(array)),
        "q975": float(np.quantile(array, 0.975)),
        "positive_fraction": float(np.mean(array > 0)),
    }


def parse_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    return [float(item) for item in parsed]


def round_half(value: float) -> float:
    return float(np.round(value * 2) / 2)


def ceil_to_day(value: float) -> float:
    return float(np.ceil(value * 7 - 1e-12) / 7)


def q1_outputs(
    baseline: Path,
    stability: Path,
    data_path: Path,
    out: Path,
) -> dict[str, Any]:
    coefficients = pd.read_csv(baseline / "q1_coefficients.csv")
    bootstrap = pd.read_csv(stability / "q1_cluster_bootstrap.csv")
    successful = bootstrap.loc[bootstrap["success"].astype(bool)].copy()
    rows = []
    term_map = {
        "const": "beta_const",
        "ga_c": "beta_ga_c",
        "first_bmi_c": "beta_first_bmi_c",
        "delta_bmi": "beta_delta_bmi",
        "ga_hinge_18": "beta_ga_hinge_18",
    }
    for _, row in coefficients.iterrows():
        term = str(row["term"])
        values = successful[term_map[term]].to_numpy(float)
        summary = empirical_summary(values)
        rows.append(
            {
                **row.to_dict(),
                "bootstrap_q025": summary["q025"],
                "bootstrap_median": summary["median"],
                "bootstrap_q975": summary["q975"],
                "bootstrap_sign_fraction": float(
                    np.mean(np.sign(values) == np.sign(row["estimate_logit"]))
                ),
            }
        )
    total_post = (
        successful["beta_ga_c"]
        + successful["beta_ga_hinge_18"]
    ).to_numpy(float)
    point = float(
        coefficients.set_index("term").loc["ga_c", "estimate_logit"]
        + coefficients.set_index("term").loc[
            "ga_hinge_18", "estimate_logit"
        ]
    )
    post_summary = empirical_summary(total_post)
    rows.append(
        {
            "term": "ga_slope_after_18_total",
            "estimate_logit": point,
            "std_error": np.nan,
            "z": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "odds_ratio_per_unit": float(np.exp(point)),
            "bootstrap_q025": post_summary["q025"],
            "bootstrap_median": post_summary["median"],
            "bootstrap_q975": post_summary["q975"],
            "bootstrap_sign_fraction": float(np.mean(total_post > 0)),
        }
    )
    coefficient_final = pd.DataFrame(rows)
    coefficient_final.to_csv(out / "q1_final_coefficients.csv", index=False)

    cv = pd.read_csv(stability / "q1_group_cv.csv")
    seed_metrics = (
        cv.groupby(["seed", "model"])[["rmse", "mae", "r2"]]
        .mean()
        .reset_index()
    )
    cv_summary = (
        seed_metrics.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    cv_summary.to_csv(out / "q1_cv_seed_summary.csv", index=False)
    comparisons = []
    pairs = [
        ("mixed_linear", "mixed_hinge18"),
        ("mixed_hinge18", "ridge"),
        ("mixed_hinge18", "random_forest"),
        ("ridge", "random_forest"),
    ]
    for reference, candidate in pairs:
        for metric in ["rmse", "mae", "r2"]:
            pivot = seed_metrics.pivot(
                index="seed", columns="model", values=metric
            )
            if metric in {"rmse", "mae"}:
                gain = pivot[reference] - pivot[candidate]
            else:
                gain = pivot[candidate] - pivot[reference]
            summary = empirical_summary(gain.to_numpy(float))
            comparisons.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "metric": metric,
                    "improvement_mean": summary["mean"],
                    "improvement_q025": summary["q025"],
                    "improvement_q975": summary["q975"],
                    "positive_fraction": summary["positive_fraction"],
                    "relative_improvement_percent": (
                        100
                        * summary["mean"]
                        / float(pivot[reference].mean())
                        if metric in {"rmse", "mae"}
                        else np.nan
                    ),
                }
            )
    comparison_table = pd.DataFrame(comparisons)
    comparison_table.to_csv(out / "q1_cv_paired_improvements.csv", index=False)

    grid = pd.read_csv(baseline / "q1_effect_grid.csv")
    selected_grid = grid.loc[
        grid["ga"].isin([12, 16, 18, 20, 24])
        & grid["first_bmi"].isin([28, 32, 36, 40])
    ].copy()
    selected_grid.to_csv(out / "q1_selected_effect_predictions.csv", index=False)
    rf_gain = comparison_table.loc[
        (comparison_table["reference"] == "ridge")
        & (comparison_table["candidate"] == "random_forest")
        & (comparison_table["metric"] == "rmse")
    ].iloc[0]
    model_comparison = pd.read_csv(baseline / "q1_model_comparison.csv").set_index(
        "model"
    )
    lrt_stat = float(
        2
        * (
            model_comparison.loc["ga_hinge_18", "loglik"]
            - model_comparison.loc["linear", "loglik"]
        )
    )
    q1_base_summary = json.loads((baseline / "q1_summary.json").read_text())
    data_audit = json.loads((baseline / "data_audit.json").read_text())
    prepared = prepare_data(data_path)
    technical_groups = (
        prepared.male_raw.groupby(
            ["孕妇代码", "检测抽血次数", "ga"], as_index=False
        )["Y染色体浓度"]
        .agg(["count", "var"])
        .reset_index()
    )
    technical_groups = technical_groups.loc[technical_groups["count"] > 1].copy()
    technical_groups["ss"] = (
        (technical_groups["count"] - 1) * technical_groups["var"]
    )
    technical_groups["df"] = technical_groups["count"] - 1
    contributions = technical_groups.groupby("孕妇代码")[["ss", "df"]].sum()
    codes = prepared.male_raw["孕妇代码"].unique()
    contributions = contributions.reindex(codes, fill_value=0.0)
    rng = np.random.default_rng(20250904)
    technical_bootstrap = []
    for iteration in range(500):
        sampled = rng.integers(0, len(codes), size=len(codes))
        total_ss = float(contributions["ss"].to_numpy()[sampled].sum())
        total_df = float(contributions["df"].to_numpy()[sampled].sum())
        technical_bootstrap.append(
            {
                "iteration": iteration,
                "sigma_tech": float(np.sqrt(total_ss / total_df)),
            }
        )
    technical_bootstrap_table = pd.DataFrame(technical_bootstrap)
    technical_bootstrap_table.to_csv(
        out / "q1_technical_sd_bootstrap.csv", index=False
    )
    technical_summary = empirical_summary(
        technical_bootstrap_table["sigma_tech"].to_numpy(float)
    )
    feature_names = [
        "ga",
        "bmi",
        "first_bmi",
        "delta_bmi",
        "age",
        "height",
        "gravidity",
        "parity",
        "ivf",
        "iui",
    ]
    visits = prepared.male_visits
    x = visits[feature_names].to_numpy(float)
    y = visits["y"].to_numpy(float)
    weights = per_group_row_weights(visits["孕妇代码"].to_numpy(str))
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=10.0).fit(
        scaler.transform(x), y, sample_weight=weights
    )
    forest = RandomForestRegressor(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=8,
        max_features=0.7,
        random_state=20250904,
        n_jobs=1,
    ).fit(x, y, sample_weight=weights)
    auxiliary_terms = []
    for feature, coefficient, importance in zip(
        feature_names, ridge.coef_, forest.feature_importances_
    ):
        auxiliary_terms.extend(
            [
                {
                    "model": "ridge",
                    "feature": feature,
                    "quantity": "standardized_coefficient",
                    "value": coefficient,
                    "status": "descriptive_full_data_fit",
                },
                {
                    "model": "random_forest",
                    "feature": feature,
                    "quantity": "impurity_importance",
                    "value": importance,
                    "status": "descriptive_full_data_fit",
                },
            ]
        )
    pd.DataFrame(auxiliary_terms).to_csv(
        out / "q1_auxiliary_model_terms.csv", index=False
    )
    return {
        "bootstrap_success": int(len(successful)),
        "pre18_slope": float(
            coefficient_final.set_index("term").loc["ga_c", "estimate_logit"]
        ),
        "post18_total_slope": point,
        "bmi_between_slope": float(
            coefficient_final.set_index("term").loc[
                "first_bmi_c", "estimate_logit"
            ]
        ),
        "within_bmi_significant": not (
            coefficient_final.set_index("term").loc[
                "delta_bmi", "bootstrap_q025"
            ]
            <= 0
            <= coefficient_final.set_index("term").loc[
                "delta_bmi", "bootstrap_q975"
            ]
        ),
        "hinge_lrt_statistic": lrt_stat,
        "hinge_lrt_p_value": float(chi2.sf(lrt_stat, 1)),
        "icc_logit": float(q1_base_summary["icc_logit"]),
        "technical_sd_y": float(data_audit["sigma_tech"]),
        "technical_sd_bootstrap_q025": technical_summary["q025"],
        "technical_sd_bootstrap_q975": technical_summary["q975"],
        "random_slope_boundary_warning": bool(
            q1_base_summary["random_slope_sensitivity"].get("warnings")
        ),
        "rf_vs_ridge_rmse_gain_percent": float(
            rf_gain["relative_improvement_percent"]
        ),
        "rf_vs_ridge_seed_q025": float(rf_gain["improvement_q025"]),
        "ml_decision": (
            "retain_mixed_model_for_inference_and_extrapolation; "
            "use_random_forest_as_local_prediction_audit"
        ),
    }


def bootstrap_aft_probability(
    rows: pd.DataFrame,
    bmi: np.ndarray,
    time: float,
    *,
    distribution: str,
) -> np.ndarray:
    intercept = rows["intercept"].to_numpy(float)[:, None]
    beta = rows["beta_bmi_std"].to_numpy(float)[:, None]
    mean = rows["bmi_mean"].to_numpy(float)[:, None]
    sd = rows["bmi_sd"].to_numpy(float)[:, None]
    sigma = rows["sigma"].to_numpy(float)[:, None]
    eta = intercept + beta * (bmi[None, :] - mean) / sd
    z = (np.log(time) - eta) / sigma
    if distribution == "lognormal":
        return norm.cdf(z)
    if distribution == "loglogistic":
        return expit(z)
    raise ValueError(distribution)


def fixed_cut_group_ids(bmi: np.ndarray, cuts: list[float]) -> np.ndarray:
    return np.digitize(bmi, bins=np.asarray(cuts), right=False)


def q2_outputs(
    baseline: Path,
    stability: Path,
    sensitivity: Path,
    data_path: Path,
    out: Path,
) -> tuple[dict[str, Any], dict[float, list[float]]]:
    prepared = prepare_data(data_path)
    aft_comparison = pd.read_csv(baseline / "q2_aft_comparison.csv")
    aft_comparison.to_csv(out / "q2_aft_distribution_comparison.csv", index=False)
    pd.read_csv(baseline / "q2_aft_quantile_grid.csv").to_csv(
        out / "q2_aft_quantile_grid.csv", index=False
    )
    pd.read_csv(baseline / "male_bmi_support.csv").to_csv(
        out / "q2_bmi_support.csv", index=False
    )
    q2_base_summary = json.loads((baseline / "q2_summary.json").read_text())
    baseline_women = prepared.male_baseline.sort_values(
        "孕妇代码"
    ).reset_index(drop=True)
    bmi = baseline_women["first_bmi"].to_numpy(float)
    parameter_rows = []
    stability_rows = []
    final_policy_rows = []
    cost_rows = []
    assignment_rows = []
    final_cuts: dict[float, list[float]] = {}

    for mode in ["ordinary", "measurement", "credible"]:
        boot = pd.read_csv(stability / f"q2_bootstrap_{mode}.csv")
        successful = boot.loc[boot["success"].astype(bool)].reset_index(drop=True)
        raw_beta = (
            successful["beta_bmi_std"].to_numpy(float)
            / successful["bmi_sd"].to_numpy(float)
        )
        for quantity, values in {
            "beta_bmi_raw": raw_beta,
            "time_ratio_per_bmi": np.exp(raw_beta),
            "sigma": successful["sigma"].to_numpy(float),
        }.items():
            parameter_rows.append(
                {"mode": mode, "quantity": quantity, **empirical_summary(values)}
            )

        for rho in [0.80, 0.90]:
            key = f"rho{int(rho * 100)}"
            archive = np.load(stability / f"q2_tau_{mode}_{key}.npz")
            tau = archive["tau"].astype(float)
            archive_bmi = archive["bmi"].astype(float)
            if not np.allclose(archive_bmi, bmi):
                raise ValueError(f"BMI alignment mismatch for {mode} {rho}")
            if len(successful) != len(tau):
                raise ValueError(f"Bootstrap row mismatch for {mode} {rho}")
            cut_lists = successful[f"{key}_cuts"].map(parse_list)
            time_lists = successful[f"{key}_times"].map(parse_list)
            valid_cut = [values for values in cut_lists if len(values) == 3]
            valid_time = [values for values in time_lists if len(values) == 4]
            for index in range(3):
                values = np.asarray([item[index] for item in valid_cut])
                stability_rows.append(
                    {
                        "mode": mode,
                        "rho": rho,
                        "quantity": f"optimized_cut_{index + 1}",
                        **empirical_summary(values),
                    }
                )
            for index in range(4):
                values = np.asarray([item[index] for item in valid_time])
                stability_rows.append(
                    {
                        "mode": mode,
                        "rho": rho,
                        "quantity": f"optimized_time_{index + 1}",
                        **empirical_summary(values),
                    }
                )

            median_tau = np.median(tau, axis=0)
            robust_tau = np.quantile(tau, 0.95, axis=0)
            for tau_type, values in [
                ("bootstrap_median", median_tau),
                ("individual_q95", robust_tau),
            ]:
                costs_for_type: dict[int, float] = {}
                for groups in range(1, 7):
                    try:
                        policy = exact_dp_policy(
                            bmi,
                            values,
                            groups=groups,
                            alpha=rho,
                            min_size=20,
                        )
                        cost = policy.total_cost
                        cuts = policy.cutpoints
                    except Exception:
                        cost = np.nan
                        cuts = []
                    costs_for_type[groups] = cost
                    cost_rows.append(
                        {
                            "mode": mode,
                            "rho": rho,
                            "tau_type": tau_type,
                            "groups": groups,
                            "total_cost": cost,
                            "cutpoints": json.dumps(cuts),
                        }
                    )
                finite = {
                    groups: value
                    for groups, value in costs_for_type.items()
                    if np.isfinite(value)
                }
                if 1 in finite and len(finite) > 1:
                    best = min(finite.values())
                    denominator = max(finite[1] - best, 1e-12)
                    for row in cost_rows[-6:]:
                        if np.isfinite(row["total_cost"]):
                            row["fraction_of_max_reduction"] = (
                                finite[1] - row["total_cost"]
                            ) / denominator

            if mode != "measurement":
                continue
            point_policy = exact_dp_policy(
                bmi,
                median_tau,
                groups=4,
                alpha=rho,
                min_size=20,
            )
            rounded = [round_half(item) for item in point_policy.cutpoints]
            final_cuts[rho] = rounded
            group_id = fixed_cut_group_ids(bmi, rounded)
            bootstrap_assignments = np.stack(
                [fixed_cut_group_ids(bmi, cut) for cut in valid_cut]
            )
            for index, code in enumerate(baseline_women["孕妇代码"]):
                values, counts = np.unique(
                    bootstrap_assignments[:, index], return_counts=True
                )
                modal_index = int(np.argmax(counts))
                assignment_rows.append(
                    {
                        "rho": rho,
                        "code": code,
                        "bmi": bmi[index],
                        "final_group": int(group_id[index] + 1),
                        "match_final_fraction": float(
                            np.mean(
                                bootstrap_assignments[:, index]
                                == group_id[index]
                            )
                        ),
                        "modal_group": int(values[modal_index] + 1),
                        "modal_group_probability": float(
                            counts[modal_index] / len(bootstrap_assignments)
                        ),
                    }
                )
            distribution = "lognormal"
            for group in range(4):
                mask = group_id == group
                replicate_times = np.asarray(
                    [
                        max(10.0, exact_upper_quantile(row[mask], rho))
                        for row in tau
                    ]
                )
                point_week = float(np.median(replicate_times))
                conservative_week_raw = float(
                    np.quantile(replicate_times, 0.95)
                )
                conservative_week = ceil_to_day(conservative_week_raw)
                strict_individual_q95_week_raw = max(
                    10.0, exact_upper_quantile(robust_tau[mask], rho)
                )
                strict_individual_q95_week = ceil_to_day(
                    strict_individual_q95_week_raw
                )
                probabilities = bootstrap_aft_probability(
                    successful,
                    bmi[mask],
                    conservative_week,
                    distribution=distribution,
                )
                probabilities_at_25 = bootstrap_aft_probability(
                    successful,
                    bmi[mask],
                    25.0,
                    distribution=distribution,
                )
                shortfall = np.maximum(
                    robust_tau[mask] - conservative_week, 0.0
                )
                tail_count = max(1, int(np.ceil(0.10 * mask.sum())))
                final_policy_rows.append(
                    {
                        "rho": rho,
                        "group": group + 1,
                        "bmi_rule_left": np.nan if group == 0 else rounded[group - 1],
                        "bmi_rule_right": np.nan if group == 3 else rounded[group],
                        "bmi_interval": (
                            f"BMI < {rounded[0]:g}"
                            if group == 0
                            else (
                                f"{rounded[-1]:g} <= BMI"
                                if group == 3
                                else f"{rounded[group - 1]:g} <= BMI < {rounded[group]:g}"
                            )
                        ),
                        "observed_bmi_min": float(np.min(bmi[mask])),
                        "observed_bmi_max": float(np.max(bmi[mask])),
                        "n": int(mask.sum()),
                        "point_week": point_week,
                        "point_week_day": week_day(point_week),
                        "bootstrap_q025_week": float(
                            np.quantile(replicate_times, 0.025)
                        ),
                        "bootstrap_q975_week": float(
                            np.quantile(replicate_times, 0.975)
                        ),
                        "recommended_week_raw": conservative_week_raw,
                        "recommended_week": conservative_week,
                        "recommended_week_day": week_day(conservative_week),
                        "bootstrap_guarantee_probability": float(
                            np.mean(replicate_times <= conservative_week)
                        ),
                        "bootstrap_window_guarantee_probability_at_25": float(
                            np.mean(replicate_times <= 25.0)
                        ),
                        "strict_individual_q95_week": strict_individual_q95_week,
                        "strict_individual_q95_week_raw": strict_individual_q95_week_raw,
                        "strict_individual_q95_week_day": week_day(
                            strict_individual_q95_week
                        ),
                        "median_tau_coverage_at_recommendation": float(
                            np.mean(median_tau[mask] <= conservative_week)
                        ),
                        "individual_q95_coverage_at_recommendation": float(
                            np.mean(robust_tau[mask] <= conservative_week)
                        ),
                        "mean_attainment_probability": float(
                            np.mean(probabilities)
                        ),
                        "mean_attainment_probability_at_25": float(
                            np.mean(probabilities_at_25)
                        ),
                        "individual_q95_after_25": int(
                            np.sum(robust_tau[mask] > 25)
                        ),
                        "recommendation_after_25": bool(conservative_week > 25),
                        "operational_action": (
                            "window_infeasible_individualized_retest"
                            if conservative_week > 25
                            else "schedule_at_recommended_week"
                        ),
                        "cvar90_q95_shortfall": float(
                            np.sort(shortfall)[-tail_count:].mean()
                        ),
                        "support_status": (
                            "high_bmi_tail" if group == 3 else "main_support"
                        ),
                    }
                )

    pd.DataFrame(parameter_rows).to_csv(
        out / "q2_bootstrap_parameter_summary.csv", index=False
    )
    stability_table = pd.DataFrame(stability_rows)
    stability_table.to_csv(out / "q2_cut_time_stability.csv", index=False)
    policy_table = pd.DataFrame(final_policy_rows)
    policy_table.to_csv(out / "q2_final_policies.csv", index=False)
    assignment_table = pd.DataFrame(assignment_rows)
    assignment_table.to_csv(
        out / "q2_group_assignment_stability.csv", index=False
    )
    pd.DataFrame(cost_rows).to_csv(out / "q2_k_cost_curves.csv", index=False)
    error_impact_rows = []
    for rho, cuts in final_cuts.items():
        group_id = fixed_cut_group_ids(bmi, cuts)
        for mode in ["ordinary", "measurement", "credible"]:
            matrix = np.load(
                stability / f"q2_tau_{mode}_rho{int(rho * 100)}.npz"
            )["tau"].astype(float)
            for group in range(4):
                mask = group_id == group
                times = np.asarray(
                    [
                        max(10.0, exact_upper_quantile(row[mask], rho))
                        for row in matrix
                    ]
                )
                error_impact_rows.append(
                    {
                        "rho": rho,
                        "group": group + 1,
                        "mode": mode,
                        "q025_week": float(np.quantile(times, 0.025)),
                        "median_week": float(np.median(times)),
                        "q95_week": float(np.quantile(times, 0.95)),
                        "q975_week": float(np.quantile(times, 0.975)),
                    }
                )
    error_impact = pd.DataFrame(error_impact_rows)
    ordinary = error_impact.loc[error_impact["mode"] == "ordinary"].set_index(
        ["rho", "group"]
    )
    for index in error_impact.index:
        row = error_impact.loc[index]
        reference = ordinary.loc[(row["rho"], row["group"])]
        error_impact.loc[index, "median_change_vs_ordinary"] = (
            row["median_week"] - reference["median_week"]
        )
        error_impact.loc[index, "q95_change_vs_ordinary"] = (
            row["q95_week"] - reference["q95_week"]
        )
    error_impact.to_csv(out / "q2_error_impact_fixed_policy.csv", index=False)
    sensitivity_table = pd.read_csv(
        sensitivity / "q2_one_factor_sensitivity.csv"
    )
    sensitivity_table.to_csv(out / "q2_one_factor_sensitivity.csv", index=False)

    main_cost = pd.DataFrame(cost_rows)
    main_cost = main_cost.loc[
        (main_cost["mode"] == "measurement")
        & (main_cost["tau_type"] == "bootstrap_median")
    ]
    k4_capture = {
        str(int(rho * 100)): float(
            main_cost.loc[
                (main_cost["rho"] == rho) & (main_cost["groups"] == 4),
                "fraction_of_max_reduction",
            ].iloc[0]
        )
        for rho in [0.80, 0.90]
    }
    return (
        {
            "primary_error_method": "measurement_perturbation_cluster_bootstrap",
            "point_distribution": q2_base_summary["primary_point_model"][
                "distribution"
            ],
            "point_interval_counts": q2_base_summary["intervals"]["hard"][
                "counts"
            ],
            "credible_interval_counts": q2_base_summary["intervals"][
                "credible_eta025"
            ]["counts"],
            "bootstrap_replicates_per_mode": int(
                pd.read_csv(stability / "q2_bootstrap_measurement.csv").shape[0]
            ),
            "selected_groups": 4,
            "cut_selection": (
                "exact K=4 DP on the individual bootstrap-median reliable times; "
                "then round to 0.5 BMI and revalidate"
            ),
            "k4_fraction_of_max_loss_reduction": k4_capture,
            "cuts": {str(int(k * 100)): v for k, v in final_cuts.items()},
            "mean_group_assignment_stability": {
                str(int(rho * 100)): float(
                    assignment_table.loc[
                        assignment_table["rho"] == rho,
                        "match_final_fraction",
                    ].mean()
                )
                for rho in [0.80, 0.90]
            },
            "policies": policy_table.to_dict(orient="records"),
            "credible_state_warning": (
                "credible reclassification creates heavy left censoring; "
                "use as sensitivity, not the point estimator"
            ),
        },
        final_cuts,
    )


def solve_usable_quantile(
    eta: np.ndarray,
    sigma: float,
    rho: float,
    failure_probability: float,
    delay: float,
) -> np.ndarray:
    low = np.full_like(eta, 0.1, dtype=float)
    high = np.full_like(eta, 60.0, dtype=float)
    for _ in range(80):
        middle = (low + high) / 2
        f_now = norm.cdf((np.log(middle) - eta) / sigma)
        delayed_time = np.maximum(middle - delay, 1e-6)
        f_delayed = norm.cdf((np.log(delayed_time) - eta) / sigma)
        usable = (1 - failure_probability) * f_now + failure_probability * f_delayed
        low = np.where(usable < rho, middle, low)
        high = np.where(usable >= rho, middle, high)
    return high


def q3_outputs(
    baseline: Path,
    stability: Path,
    q3_ml: Path,
    data_path: Path,
    final_cuts: dict[float, list[float]],
    out: Path,
) -> dict[str, Any]:
    in_sample = pd.read_csv(baseline / "q3_aft_comparison.csv")
    in_sample.to_csv(
        out / "q3_aft_in_sample_comparison.csv", index=False
    )
    effect_rows = []
    for _, row in in_sample.iterrows():
        features = ast.literal_eval(str(row["feature_cols"]))
        coefficients = np.fromstring(
            str(row["coefficients"]).replace("[", "").replace("]", ""),
            sep=" ",
        )
        scales = np.fromstring(
            str(row["scales"]).replace("[", "").replace("]", ""),
            sep=" ",
        )
        for index, feature in enumerate(features):
            raw = float(coefficients[index + 1] / scales[index])
            effect_rows.append(
                {
                    "model": row["model"],
                    "feature": feature,
                    "standardized_coefficient": coefficients[index + 1],
                    "raw_log_time_coefficient": raw,
                    "time_ratio_per_unit": float(np.exp(raw)),
                    "status": "in_sample_explanatory_only",
                }
            )
    pd.DataFrame(effect_rows).to_csv(
        out / "q3_in_sample_time_ratios.csv", index=False
    )
    pd.read_csv(baseline / "q3_time_calibration_in_sample.csv").to_csv(
        out / "q3_time_calibration_in_sample.csv", index=False
    )
    cv = pd.read_csv(stability / "q3_repeated_group_cv.csv")
    seed_metrics = (
        cv.groupby(["seed", "model"])[["nll", "mean_brier"]]
        .mean()
        .reset_index()
    )
    summary = (
        seed_metrics.groupby("model")[["nll", "mean_brier"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(out / "q3_aft_cv_seed_summary.csv", index=False)
    pivot_nll = seed_metrics.pivot(index="seed", columns="model", values="nll")
    pivot_brier = seed_metrics.pivot(
        index="seed", columns="model", values="mean_brier"
    )
    rows = []
    for candidate in [
        "bmi_age_height",
        "bmi_full",
        "weight_height_full",
    ]:
        for metric, pivot in [("nll", pivot_nll), ("mean_brier", pivot_brier)]:
            gain = pivot["bmi_only"] - pivot[candidate]
            result = empirical_summary(gain.to_numpy(float))
            rows.append(
                {
                    "candidate": candidate,
                    "metric": metric,
                    "improvement_mean": result["mean"],
                    "improvement_q025": result["q025"],
                    "improvement_q975": result["q975"],
                    "positive_fraction": result["positive_fraction"],
                    "relative_improvement_percent": float(
                        100 * result["mean"] / pivot["bmi_only"].mean()
                    ),
                }
            )
    increment = pd.DataFrame(rows)
    increment.to_csv(out / "q3_aft_paired_increment.csv", index=False)

    ml_metrics = pd.read_csv(q3_ml / "q3_ml_seed_metrics.csv")
    ml_summary = (
        ml_metrics.groupby("model")[["roc_auc", "pr_auc", "brier"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    ml_summary.to_csv(out / "q3_ml_seed_summary.csv", index=False)
    ml_increment = pd.read_csv(q3_ml / "q3_ml_increment.csv")
    ml_increment.to_csv(out / "q3_ml_increment.csv", index=False)

    bmi_full_nll = increment.loc[
        (increment["candidate"] == "bmi_full")
        & (increment["metric"] == "nll")
    ].iloc[0]
    bmi_full_brier = increment.loc[
        (increment["candidate"] == "bmi_full")
        & (increment["metric"] == "mean_brier")
    ].iloc[0]
    aft_promoted = bool(
        bmi_full_nll["improvement_q025"] > 0
        and bmi_full_nll["relative_improvement_percent"] >= 1.0
        and bmi_full_brier["improvement_q025"] >= 0
    )
    rf_rows = ml_increment.loc[ml_increment["model_type"] == "rf"].set_index(
        "metric"
    )
    rf_promoted = bool(
        rf_rows.loc["roc_auc", "q025"] > 0
        and rf_rows.loc["pr_auc", "q025"] > 0
        and rf_rows.loc["brier", "q025"] > 0
    )

    prepared = prepare_data(data_path)
    women = prepared.male_baseline.sort_values("孕妇代码").reset_index(drop=True)
    bmi = women["first_bmi"].to_numpy(float)
    q2_summary = json.loads((baseline / "q2_summary.json").read_text())
    fit = q2_summary["primary_point_model"]
    coefficient = np.asarray(fit["coefficients"], dtype=float)
    mean = float(fit["means"][0])
    scale = float(fit["scales"][0])
    sigma = float(fit["sigma"])
    eta = coefficient[0] + coefficient[1] * (bmi - mean) / scale
    scenario_rows = []
    for rho in [0.80, 0.90]:
        group_id = fixed_cut_group_ids(bmi, final_cuts[rho])
        base_times: dict[int, float] = {}
        for failure_probability in [0.0, 0.05, 0.10, 0.15]:
            for delay in [1.0, 2.0]:
                individual = solve_usable_quantile(
                    eta,
                    sigma,
                    rho,
                    failure_probability,
                    delay,
                )
                for group in range(4):
                    mask = group_id == group
                    raw_time = max(
                        10.0, exact_upper_quantile(individual[mask], rho)
                    )
                    time = ceil_to_day(raw_time)
                    if failure_probability == 0 and delay == 1:
                        base_times[group] = time
                    scenario_rows.append(
                        {
                            "rho": rho,
                            "failure_probability": failure_probability,
                            "retest_delay_weeks": delay,
                            "group": group + 1,
                            "recommended_week_raw": raw_time,
                            "recommended_week": time,
                            "week_day": week_day(time),
                            "delay_vs_no_failure": time
                            - base_times.get(group, time),
                            "after_25": bool(time > 25),
                        }
                    )
    pd.DataFrame(scenario_rows).to_csv(
        out / "q3_failure_scenarios.csv", index=False
    )
    return {
        "aft_multi_promoted": aft_promoted,
        "rf_multi_promoted": rf_promoted,
        "rf_multi_signal": bool(
            rf_rows.loc["roc_auc", "q025"] > 0
            and rf_rows.loc["brier", "q025"] > 0
        ),
        "final_model": "multi_factor_aft" if aft_promoted else "bmi_only_aft",
        "final_grouping": "reoptimize" if aft_promoted else "retain_q2",
        "bmi_full_nll_improvement": bmi_full_nll.to_dict(),
        "bmi_full_brier_improvement": bmi_full_brier.to_dict(),
        "ml_increment": ml_increment.to_dict(orient="records"),
    }


def q4_outputs(
    q4: Path,
    baseline: Path,
    data_path: Path,
    out: Path,
) -> dict[str, Any]:
    pd.read_csv(baseline / "q4_label_audit.csv").to_csv(
        out / "q4_label_audit.csv", index=False
    )
    write_json(
        out / "q4_audit_summary.json",
        json.loads((baseline / "q4_audit_summary.json").read_text()),
    )
    prepared_data = prepare_data(data_path)
    female = prepare_female(prepared_data.female_raw)
    audit_rows = []
    for label in ["T13", "T18", "T21"]:
        y = female[f"y_{label}"].to_numpy(int)
        z = female[f"z{int(label[1:])}"].to_numpy(float)
        for rule, prediction in [
            ("z_ge_3", z >= 3),
            ("abs_z_ge_3", np.abs(z) >= 3),
        ]:
            audit_rows.append(
                {
                    "label": label,
                    "rule": rule,
                    "positive_records": int(y.sum()),
                    "positive_women": int(
                        female.loc[y == 1, "code"].nunique()
                    ),
                    "women_with_label_change": int(
                        female.groupby("code")[f"y_{label}"].nunique().gt(1).sum()
                    ),
                    "sensitivity": float(np.mean(prediction[y == 1])),
                    "specificity": float(np.mean(~prediction[y == 0])),
                    "precision": (
                        float(np.mean(y[prediction]))
                        if prediction.sum() > 0
                        else np.nan
                    ),
                    "predicted_positive_records": int(prediction.sum()),
                }
            )
    pd.DataFrame(audit_rows).to_csv(
        out / "q4_extended_z_rule_audit.csv", index=False
    )
    label_matrix = female[["y_T13", "y_T18", "y_T21"]].to_numpy(int)
    combinations = []
    for row in label_matrix:
        active = [
            label
            for label, value in zip(["T13", "T18", "T21"], row)
            if value
        ]
        combinations.append("+".join(active) if active else "normal")
    pd.Series(combinations).value_counts().rename_axis("label_combination").reset_index(
        name="records"
    ).to_csv(out / "q4_label_combinations.csv", index=False)
    female_audit = female.copy()
    female_audit["label_combination"] = combinations
    write_json(
        out / "q4_extended_audit_summary.json",
        {
            "records": len(female),
            "women": int(female["code"].nunique()),
            "women_with_any_label_change": int(
                female_audit.groupby("code")["label_combination"]
                .nunique()
                .gt(1)
                .sum()
            ),
            "gc_outside_40_60": int(
                ((female["gc"] < 0.4) | (female["gc"] > 0.6)).sum()
            ),
            "raw_unique_reads_spearman": float(
                prepared_data.female_raw[["原始读段数", "唯一比对的读段数"]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
        },
    )
    metrics = pd.read_csv(q4 / "q4_seed_metrics.csv")
    summary_rows = []
    for (label, model), group in metrics.groupby(["label", "model"]):
        for metric in ["roc_auc", "pr_auc", "brier", "logloss"]:
            summary_rows.append(
                {
                    "label": label,
                    "model": model,
                    "metric": metric,
                    **empirical_summary(group[metric].to_numpy(float)),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        out / "q4_seed_metric_intervals.csv", index=False
    )
    paired_rows = []
    for label, label_metrics in metrics.groupby("label"):
        for candidate in ["elastic_full", "random_forest", "firth"]:
            if candidate not in set(label_metrics["model"]):
                continue
            for metric in ["roc_auc", "pr_auc", "brier", "logloss"]:
                pivot = label_metrics.pivot(
                    index="seed_index", columns="model", values=metric
                )
                if metric in {"brier", "logloss"}:
                    gain = pivot["z_only"] - pivot[candidate]
                else:
                    gain = pivot[candidate] - pivot["z_only"]
                paired_rows.append(
                    {
                        "label": label,
                        "reference": "z_only",
                        "candidate": candidate,
                        "metric": metric,
                        **empirical_summary(gain.to_numpy(float)),
                    }
                )
    pd.DataFrame(paired_rows).to_csv(
        out / "q4_paired_improvements_vs_z.csv", index=False
    )
    promotion = pd.read_csv(q4 / "q4_ml_promotion.csv")
    promotion.to_csv(out / "q4_ml_promotion.csv", index=False)
    selected = json.loads((q4 / "q4_selected_models.json").read_text())

    record_order = prepared_data.female_raw[
        ["序号", "孕妇代码", "date_norm"]
    ].rename(columns={"序号": "row_id", "孕妇代码": "code"})
    record_order = record_order.sort_values(["code", "date_norm", "row_id"])
    first_ids = set(record_order.groupby("code").first()["row_id"].astype(int))
    last_ids = set(record_order.groupby("code").last()["row_id"].astype(int))
    seed_predictions = pd.read_csv(q4 / "q4_seed_oof_predictions.csv")
    sensitivity_seed_rows = []
    for label in ["T13", "T18", "T21"]:
        subset = seed_predictions.loc[
            (seed_predictions["label"] == label)
            & (seed_predictions["model"] == selected[label])
        ].copy()
        for seed_index, seed_frame in subset.groupby("seed_index"):
            for scope, mask in [
                ("first_record", seed_frame["row_id"].isin(first_ids)),
                ("last_record", seed_frame["row_id"].isin(last_ids)),
            ]:
                scoped = seed_frame.loc[mask]
                y_scope = scoped["y"].to_numpy(int)
                p_scope = scoped["probability"].to_numpy(float)
                sensitivity_seed_rows.append(
                    {
                        "label": label,
                        "model": selected[label],
                        "seed_index": int(seed_index),
                        "scope": scope,
                        "roc_auc": float(roc_auc_score(y_scope, p_scope)),
                        "pr_auc": float(
                            average_precision_score(y_scope, p_scope)
                        ),
                        "brier": float(np.mean(np.square(y_scope - p_scope))),
                    }
                )
            ever = seed_frame.groupby("code", as_index=False).agg(
                y=("y", "max"), probability=("probability", "max")
            )
            y_ever = ever["y"].to_numpy(int)
            p_ever = ever["probability"].to_numpy(float)
            sensitivity_seed_rows.append(
                {
                    "label": label,
                    "model": selected[label],
                    "seed_index": int(seed_index),
                    "scope": "ever_positive_max_probability",
                    "roc_auc": float(roc_auc_score(y_ever, p_ever)),
                    "pr_auc": float(average_precision_score(y_ever, p_ever)),
                    "brier": float(np.mean(np.square(y_ever - p_ever))),
                }
            )
    sensitivity_seed = pd.DataFrame(sensitivity_seed_rows)
    sensitivity_rows = []
    for (label, model, scope), group in sensitivity_seed.groupby(
        ["label", "model", "scope"]
    ):
        for metric in ["roc_auc", "pr_auc", "brier"]:
            sensitivity_rows.append(
                {
                    "label": label,
                    "model": model,
                    "scope": scope,
                    "metric": metric,
                    **empirical_summary(group[metric].to_numpy(float)),
                }
            )
    pd.DataFrame(sensitivity_rows).to_csv(
        out / "q4_woman_level_sensitivity.csv", index=False
    )

    nested = pd.read_csv(q4 / "q4_nested_final_metrics.csv")
    parameter_rows = []
    for _, row in nested.iterrows():
        for fold, parameters in enumerate(json.loads(row["fold_parameters"])):
            parameter_rows.append(
                {
                    "label": row["label"],
                    "seed_index": int(row["seed_index"]),
                    "fold": fold,
                    "parameters": json.dumps(parameters, sort_keys=True),
                }
            )
    parameter_frequency = (
        pd.DataFrame(parameter_rows)
        .groupby(["label", "parameters"])
        .size()
        .rename("folds")
        .reset_index()
    )
    parameter_frequency["fraction_within_label"] = parameter_frequency.groupby(
        "label"
    )["folds"].transform(lambda values: values / values.sum())
    parameter_frequency.to_csv(
        out / "q4_nested_parameter_frequency.csv", index=False
    )
    nested_rows = []
    for label, group in nested.groupby("label"):
        for metric in [
            "roc_auc",
            "pr_auc",
            "brier",
            "logloss",
            "coverage",
            "sensitivity_all",
            "specificity_all",
            "selective_sensitivity",
            "selective_specificity",
            "selective_accuracy",
            "quality_retest_rate",
        ]:
            nested_rows.append(
                {
                    "label": label,
                    "model": selected[label],
                    "metric": metric,
                    **empirical_summary(group[metric].to_numpy(float)),
                }
            )
    nested_summary = pd.DataFrame(nested_rows)
    nested_summary.to_csv(out / "q4_nested_metric_intervals.csv", index=False)

    decisions = pd.read_csv(q4 / "q4_final_record_decisions.csv")
    decisions.to_csv(out / "q4_final_record_decisions.csv", index=False)
    labels = female[["row_id", "code", "y_T13", "y_T18", "y_T21"]]
    merged = decisions.merge(labels, on=["row_id", "code"], validate="one_to_one")
    weights = per_group_row_weights(merged["code"].to_numpy(str))
    performance_rows = []
    calibration_rows = []
    for label in ["T13", "T18", "T21"]:
        y = merged[f"y_{label}"].to_numpy(int)
        probability = merged[f"probability_median_{label}"].to_numpy(float)
        bins = pd.qcut(
            probability,
            q=5,
            labels=False,
            duplicates="drop",
        )
        for bin_index in sorted(pd.Series(bins).dropna().unique()):
            mask = np.asarray(bins == bin_index)
            calibration_rows.append(
                {
                    "label": label,
                    "bin": int(bin_index) + 1,
                    "records": int(mask.sum()),
                    "mean_predicted_probability": weighted_mean(
                        probability[mask], weights[mask]
                    ),
                    "observed_positive_rate": weighted_mean(
                        y[mask], weights[mask]
                    ),
                }
            )
        decision = merged[f"decision_{label}"].astype(str).to_numpy()
        positive = decision == "异常"
        negative = decision == "正常"
        classified = positive | negative
        positive_weight = weights[y == 1].sum()
        negative_weight = weights[y == 0].sum()
        classified_positive_weight = weights[(y == 1) & classified].sum()
        classified_negative_weight = weights[(y == 0) & classified].sum()
        classified_weight = weights[classified].sum()
        correct = weights[
            ((y == 1) & positive) | ((y == 0) & negative)
        ].sum()
        performance_rows.append(
            {
                "label": label,
                "model": selected[label],
                "records": len(merged),
                "positive_records": int(y.sum()),
                "coverage": float(weights[classified].sum() / weights.sum()),
                "retest_rate": float(weights[~classified].sum() / weights.sum()),
                "sensitivity_all": float(
                    weights[(y == 1) & positive].sum() / positive_weight
                ),
                "specificity_all": float(
                    weights[(y == 0) & negative].sum() / negative_weight
                ),
                "selective_sensitivity": (
                    float(
                        weights[(y == 1) & positive].sum()
                        / classified_positive_weight
                    )
                    if classified_positive_weight > 0
                    else np.nan
                ),
                "selective_specificity": (
                    float(
                        weights[(y == 0) & negative].sum()
                        / classified_negative_weight
                    )
                    if classified_negative_weight > 0
                    else np.nan
                ),
                "selective_accuracy": (
                    float(correct / classified_weight)
                    if classified_weight > 0
                    else np.nan
                ),
                "definitive_positive_records": int(positive.sum()),
                "definitive_negative_records": int(negative.sum()),
                "retest_records": int((~classified).sum()),
            }
        )
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(out / "q4_final_selective_performance.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        out / "q4_calibration_curve.csv", index=False
    )
    merged.to_csv(out / "q4_final_record_decisions_with_labels.csv", index=False)
    pd.read_csv(q4 / "q4_final_decision_summary.csv").to_csv(
        out / "q4_final_decision_summary.csv", index=False
    )
    pd.read_csv(q4 / "q4_final_model_terms.csv").to_csv(
        out / "q4_final_model_terms.csv", index=False
    )
    write_json(
        out / "q4_operational_parameters.json",
        json.loads((q4 / "q4_operational_parameters.json").read_text()),
    )
    write_json(
        out / "q4_selected_hyperparameters.json",
        json.loads((q4 / "q4_selected_hyperparameters.json").read_text()),
    )
    qualification: dict[str, str] = {}
    operational_models: dict[str, str] = {}
    for label in ["T13", "T18", "T21"]:
        roc_row = nested_summary.loc[
            (nested_summary["label"] == label)
            & (nested_summary["metric"] == "roc_auc")
        ].iloc[0]
        if float(roc_row["q025"]) > 0.5:
            qualification[label] = "qualified_for_selective_prediction_only"
            operational_models[label] = selected[label]
        else:
            qualification[label] = "not_qualified_retest_all"
            operational_models[label] = "no_automatic_model"
    return {
        "stability_seeds": int(metrics["seed_index"].nunique()),
        "nested_seeds": int(nested["seed_index"].nunique()),
        "probability_interval_method": (
            "2.5%/97.5% split-stability quantiles over repeated nested "
            "grouped cross-validation; not a clinical confidence interval"
        ),
        "selected_validation_candidates": selected,
        "operational_models": operational_models,
        "qualification": qualification,
        "selected_models": selected,
        "promotion": promotion.to_dict(orient="records"),
        "selective_performance": performance.to_dict(orient="records"),
        "decision_counts": pd.read_csv(
            q4 / "q4_final_decision_summary.csv"
        ).to_dict(orient="records"),
    }


def make_figures(
    baseline: Path,
    out: Path,
    q1_summary: dict[str, Any],
) -> list[str]:
    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)
    created = []

    grid = pd.read_csv(baseline / "q1_effect_grid.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for bmi, group in grid.groupby("first_bmi"):
        ax.plot(group["ga"], 100 * group["predicted_y"], marker="o", label=f"BMI {bmi:g}")
    ax.axhline(4, color="black", linestyle="--", linewidth=1, label="4% threshold")
    ax.axvline(18, color="grey", linestyle=":", linewidth=1)
    ax.set(xlabel="Gestational week", ylabel="Predicted Y concentration (%)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "q1_effect_curves.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    created.append(str(path))

    policy = pd.read_csv(out / "q2_final_policies.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, rho in zip(axes, [0.80, 0.90]):
        group = policy.loc[policy["rho"] == rho]
        x = group["group"].to_numpy()
        y = group["recommended_week"].to_numpy()
        lower = y - group["bootstrap_q025_week"].to_numpy()
        upper = group["bootstrap_q975_week"].to_numpy() - y
        lower = np.maximum(lower, 0)
        upper = np.maximum(upper, 0)
        ax.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="o-", capsize=4)
        ax.axhline(25, color="red", linestyle="--", linewidth=1)
        ax.set(title=f"Reliability {rho:.0%}", xlabel="BMI group", xticks=x)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Recommended week (95% conservative)")
    fig.tight_layout()
    path = figure_dir / "q2_recommended_times.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    created.append(str(path))

    q3 = pd.read_csv(out / "q3_aft_paired_increment.csv")
    plot = q3.loc[q3["metric"] == "nll"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(plot))
    mean = plot["improvement_mean"].to_numpy()
    ax.bar(x, mean, color="#4C78A8")
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack(
            [
                mean - plot["improvement_q025"].to_numpy(),
                plot["improvement_q975"].to_numpy() - mean,
            ]
        ),
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, plot["candidate"], rotation=15, ha="right")
    ax.set_ylabel("NLL improvement vs BMI-only")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "q3_aft_increment.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    created.append(str(path))

    q4 = pd.read_csv(out / "q4_final_selective_performance.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(q4))
    ax.bar(x - 0.2, q4["coverage"], width=0.4, label="Coverage")
    ax.bar(x + 0.2, q4["selective_accuracy"], width=0.4, label="Selective accuracy")
    ax.set_xticks(x, q4["label"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "q4_selective_performance.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    created.append(str(path))

    intervals = pd.read_csv(out / "q4_seed_metric_intervals.csv")
    models = ["z_only", "elastic_full", "random_forest", "firth"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric, title in [
        (axes[0], "pr_auc", "PR-AUC (higher is better)"),
        (axes[1], "brier", "Brier score (lower is better)"),
    ]:
        subset = intervals.loc[intervals["metric"] == metric]
        width = 0.18
        base_x = np.arange(3)
        for model_index, model in enumerate(models):
            values = subset.loc[subset["model"] == model].set_index("label")
            if values.empty:
                continue
            means = np.asarray(
                [values.loc[label, "mean"] if label in values.index else np.nan for label in ["T13", "T18", "T21"]]
            )
            axis.bar(
                base_x + (model_index - 1.5) * width,
                means,
                width=width,
                label=model,
            )
        axis.set_xticks(base_x, ["T13", "T18", "T21"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = figure_dir / "q4_model_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    created.append(str(path))

    calibration = pd.read_csv(out / "q4_calibration_curve.csv")
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for label, group in calibration.groupby("label"):
        ax.plot(
            group["mean_predicted_probability"],
            group["observed_positive_rate"],
            marker="o",
            label=label,
        )
    maximum = max(
        0.1,
        float(
            max(
                calibration["mean_predicted_probability"].max(),
                calibration["observed_positive_rate"].max(),
            )
        ),
    )
    ax.plot([0, maximum], [0, maximum], color="black", linestyle="--")
    ax.set(
        xlim=(0, maximum),
        ylim=(0, maximum),
        xlabel="Median cross-fitted probability",
        ylabel="Observed positive rate",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "q4_calibration.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    created.append(str(path))
    return created


def validate_final_outputs(out: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    q1 = pd.read_csv(out / "q1_final_coefficients.csv")
    checks["q1_key_terms_present"] = {
        "ga_c",
        "first_bmi_c",
        "delta_bmi",
        "ga_hinge_18",
    }.issubset(set(q1["term"]))

    q2 = pd.read_csv(out / "q2_final_policies.csv")
    checks["q2_two_reliability_levels"] = set(q2["rho"]) == {0.8, 0.9}
    checks["q2_four_groups_each"] = bool(
        (q2.groupby("rho")["group"].nunique() == 4).all()
    )
    checks["q2_all_group_sizes_at_least_20"] = bool((q2["n"] >= 20).all())
    checks["q2_each_level_has_267_women"] = bool(
        (q2.groupby("rho")["n"].sum() == 267).all()
    )
    checks["q2_bootstrap_guarantee_at_least_95pct"] = bool(
        (q2["bootstrap_guarantee_probability"] >= 0.95 - 1e-12).all()
    )

    q3 = pd.read_csv(out / "q3_aft_paired_increment.csv")
    checks["q3_all_three_multi_candidates_compared"] = (
        q3["candidate"].nunique() == 3
    )

    q4 = pd.read_csv(out / "q4_final_record_decisions_with_labels.csv")
    checks["q4_605_unique_records"] = bool(
        len(q4) == 605 and q4["row_id"].nunique() == 605
    )
    for label in ["T13", "T18", "T21"]:
        lower = q4[f"probability_q025_{label}"]
        middle = q4[f"probability_median_{label}"]
        upper = q4[f"probability_q975_{label}"]
        checks[f"q4_{label}_probability_order"] = bool(
            ((0 <= lower) & (lower <= middle) & (middle <= upper) & (upper <= 1)).all()
        )
        checks[f"q4_{label}_threshold_order"] = bool(
            (
                q4[f"low_threshold_median_{label}"]
                <= q4[f"high_threshold_median_{label}"]
            ).all()
        )
    decision_counts = pd.read_csv(out / "q4_final_decision_summary.csv")
    checks["q4_decision_counts_sum_605"] = int(
        decision_counts["records"].sum()
    ) == 605
    report = {
        "all_passed": bool(all(checks.values())),
        "checks": checks,
    }
    write_json(out / "validation_report.json", report)
    if not report["all_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Final result validation failed: {failed}")
    return report


def write_results_readme(out: Path, summary: dict[str, Any]) -> None:
    q2_policies = pd.DataFrame(summary["q2"]["policies"])
    q4_perf = pd.DataFrame(summary["q4"]["selective_performance"])
    q4_perf["operational_status"] = q4_perf["label"].map(
        summary["q4"]["qualification"]
    )
    lines = [
        "# NIPT 全量求解结果索引（本轮仅计算）",
        "",
        "本目录保存下一轮撰文所需的数值结果、稳健性区间、逐记录判定和作图素材。",
        "",
        "## 核心计算结论",
        "",
        f"- Q1：18 周折点模型；18 周前 logit 斜率 {summary['q1']['pre18_slope']:.5f}，18 周后总斜率 {summary['q1']['post18_total_slope']:.5f}。",
        f"- Q1 ML 审计：随机森林相对 Ridge 的重复分组 CV RMSE 改善 {summary['q1']['rf_vs_ridge_rmse_gain_percent']:.2f}%。",
        f"- Q2：采用测量扰动 + 孕妇级 500 次 Bootstrap；固定 4 个 BMI 组，80%/90% 两档结果见 `q2_final_policies.csv`。",
        f"- Q3：最终模型为 {summary['q3']['final_model']}，分组决策为 {summary['q3']['final_grouping']}。",
        f"- Q4：验证候选为 {json.dumps(summary['q4']['selected_models'], ensure_ascii=False)}；实际可执行状态为 {json.dumps(summary['q4']['operational_models'], ensure_ascii=False)}。",
        "",
        "## Q2 最终时点表",
        "",
        q2_policies[
            [
                "rho",
                "group",
                "bmi_interval",
                "n",
                "point_week",
                "recommended_week_day",
                "bootstrap_q025_week",
                "bootstrap_q975_week",
                "operational_action",
            ]
        ].to_markdown(index=False),
        "",
        "## Q4 选择性判定表现",
        "",
        q4_perf[
            [
                "label",
                "model",
                "operational_status",
                "coverage",
                "sensitivity_all",
                "specificity_all",
                "selective_accuracy",
                "retest_records",
            ]
        ].to_markdown(index=False),
        "",
        "## 文件导航",
        "",
        "- `solution_summary.json`：全部核心结论和晋级决定。",
        "- `q1_*`：效应系数、Bootstrap 区间、重复分组 CV 和 ML 增益。",
        "- `q2_*`：AFT 参数、K 值损失曲线、切点/时点稳定性、两档最终方案和误差敏感性。",
        "- `q3_*`：多因素 AFT 配对增量、轻量 ML 审计和检测失败情景。",
        "- `q4_*`：模型晋级、嵌套验证、选择性风险、逐记录概率区间与最终判定。",
        "- `figures/`：下一轮可直接选用或重绘的结果图。",
        "",
        "注：这是计算结果索引，不是论文正文。",
    ]
    (out / "RESULTS_README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("附件.xlsx"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("nipt_solution/outputs/baseline")
    )
    parser.add_argument(
        "--stability",
        type=Path,
        default=Path("nipt_solution/outputs/male_stability"),
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=Path("nipt_solution/outputs/sensitivity"),
    )
    parser.add_argument(
        "--q3-ml", type=Path, default=Path("nipt_solution/outputs/q3_ml")
    )
    parser.add_argument(
        "--q4", type=Path, default=Path("nipt_solution/outputs/q4")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nipt_solution/outputs/final_results"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    q1 = q1_outputs(args.baseline, args.stability, args.data, args.output)
    q2, final_cuts = q2_outputs(
        args.baseline,
        args.stability,
        args.sensitivity,
        args.data,
        args.output,
    )
    q3 = q3_outputs(
        args.baseline,
        args.stability,
        args.q3_ml,
        args.data,
        final_cuts,
        args.output,
    )
    q4 = q4_outputs(args.q4, args.baseline, args.data, args.output)
    validation = validate_final_outputs(args.output)
    male_manifest = json.loads(
        (args.stability / "male_stability_manifest.json").read_text()
    )
    q3_ml_manifest = json.loads((args.q3_ml / "q3_ml_manifest.json").read_text())
    q4_manifest = json.loads((args.q4 / "q4_manifest.json").read_text())
    summary = {
        "metadata": {
            "data": str(args.data),
            "data_sha256": sha256_file(args.data),
            "python": platform.python_version(),
            "bootstrap": 500,
            "q3_random_seeds": 100,
            "q4_random_seeds": 100,
            "q4_nested_seeds": 20,
            "run_manifests": {
                "male_stability": male_manifest,
                "q3_ml": q3_ml_manifest,
                "q4": q4_manifest,
            },
        },
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "validation": validation,
    }
    figures = make_figures(args.baseline, args.output, q1)
    summary["figures"] = figures
    write_json(args.output / "solution_summary.json", summary)
    write_results_readme(args.output, to_jsonable(summary))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validation": validation,
                "q2_cuts": q2["cuts"],
                "q3_final_model": q3["final_model"],
                "q4_selected_models": q4["selected_models"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

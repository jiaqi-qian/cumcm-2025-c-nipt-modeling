from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nipt_core import (
    SEED,
    classify_visit_states,
    construct_intervals,
    determined_binary_at_time,
    evaluate_aft_loglik,
    exact_dp_policy,
    fit_aft,
    per_group_row_weights,
    prepare_data,
    shuffled_group_folds,
    to_jsonable,
    weighted_mean,
    write_json,
)
from solve_baseline import fit_q1_model


def fixed_q1_prediction(
    result,
    columns: list[str],
    centers: dict[str, float],
    test: pd.DataFrame,
    knot: float | None,
) -> np.ndarray:
    frame = test.copy()
    values = pd.DataFrame(index=frame.index)
    values["ga_c"] = frame["ga"] - centers["ga"]
    values["first_bmi_c"] = frame["first_bmi"] - centers["first_bmi"]
    values["delta_bmi"] = frame["delta_bmi"]
    values["age_c"] = frame["age"] - centers["age"]
    values["height_c"] = frame["height"] - centers["height"]
    values["ga_bmi_interaction"] = (
        values["ga_c"] * values["first_bmi_c"]
    )
    if knot is not None:
        values[f"ga_hinge_{int(knot)}"] = np.maximum(frame["ga"] - knot, 0)
    design = sm.add_constant(values[columns], has_constant="add")
    linear = design.to_numpy(float) @ result.fe_params.to_numpy(float)
    return expit(linear)


def regression_metrics(
    observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    error = observed - predicted
    mse = weighted_mean(np.square(error), weights)
    mae = weighted_mean(np.abs(error), weights)
    center = weighted_mean(observed, weights)
    total = weighted_mean(np.square(observed - center), weights)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": float(1 - mse / total),
    }


def q1_cv_one_seed(visits: pd.DataFrame, seed: int) -> list[dict]:
    rows = []
    folds = shuffled_group_folds(
        visits["孕妇代码"].to_numpy(), n_splits=5, seed=seed
    )
    feature_cols = [
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
    for fold_id, (train_index, test_index) in enumerate(folds):
        train = visits.iloc[train_index].copy()
        test = visits.iloc[test_index].copy()
        observed = test["y"].to_numpy(float)
        test_weights = per_group_row_weights(test["孕妇代码"].to_numpy())

        for model_name, knot in [("mixed_linear", None), ("mixed_hinge18", 18.0)]:
            try:
                fit, _, columns, centers, warning_text = fit_q1_model(
                    train, knot=knot
                )
                prediction = fixed_q1_prediction(
                    fit, columns, centers, test, knot
                )
                rows.append(
                    {
                        "seed": seed,
                        "fold": fold_id,
                        "model": model_name,
                        **regression_metrics(observed, prediction, test_weights),
                        "converged": bool(fit.converged),
                        "warnings": " | ".join(warning_text),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "seed": seed,
                        "fold": fold_id,
                        "model": model_name,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "r2": np.nan,
                        "converged": False,
                        "warnings": repr(exc),
                    }
                )

        x_train = train[feature_cols].to_numpy(float)
        x_test = test[feature_cols].to_numpy(float)
        y_train = train["y"].to_numpy(float)
        train_weights = per_group_row_weights(train["孕妇代码"].to_numpy())
        scaler = StandardScaler().fit(x_train)
        ridge = Ridge(alpha=10.0)
        ridge.fit(
            scaler.transform(x_train),
            y_train,
            sample_weight=train_weights,
        )
        ridge_prediction = np.clip(
            ridge.predict(scaler.transform(x_test)), 0, 1
        )
        rows.append(
            {
                "seed": seed,
                "fold": fold_id,
                "model": "ridge",
                **regression_metrics(observed, ridge_prediction, test_weights),
                "converged": True,
                "warnings": "",
            }
        )

        forest = RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=8,
            max_features=0.7,
            random_state=seed * 10 + fold_id,
            n_jobs=1,
        )
        forest.fit(x_train, y_train, sample_weight=train_weights)
        forest_prediction = np.clip(forest.predict(x_test), 0, 1)
        rows.append(
            {
                "seed": seed,
                "fold": fold_id,
                "model": "random_forest",
                **regression_metrics(observed, forest_prediction, test_weights),
                "converged": True,
                "warnings": "",
            }
        )
    return rows


def q1_bootstrap_once(visits: pd.DataFrame, iteration: int) -> dict:
    rng = np.random.default_rng(SEED + iteration)
    codes = visits["孕妇代码"].unique()
    sampled = rng.choice(codes, size=len(codes), replace=True)
    pieces = []
    for draw, code in enumerate(sampled):
        piece = visits.loc[visits["孕妇代码"] == code].copy()
        piece["孕妇代码"] = f"{code}__boot{draw}"
        pieces.append(piece)
    boot = pd.concat(pieces, ignore_index=True)
    try:
        fit, _, _, _, warning_text = fit_q1_model(boot, knot=18.0)
        values = {
            f"beta_{name}": float(value)
            for name, value in fit.fe_params.items()
        }
        random_var = float(fit.cov_re.iloc[0, 0])
        residual_var = float(fit.scale)
        return {
            "iteration": iteration,
            "success": True,
            **values,
            "random_intercept_var": random_var,
            "residual_var": residual_var,
            "icc": random_var / (random_var + residual_var),
            "warnings": " | ".join(warning_text),
        }
    except Exception as exc:
        return {
            "iteration": iteration,
            "success": False,
            "warnings": repr(exc),
        }


def q2_bootstrap_once(
    visits: pd.DataFrame,
    baseline: pd.DataFrame,
    sigma_tech: float,
    iteration: int,
    bootstrap_mode: str,
) -> tuple[dict, dict[float, np.ndarray]]:
    rng = np.random.default_rng(SEED + 100000 + iteration)
    if bootstrap_mode == "ordinary":
        states = classify_visit_states(visits, sigma_tech, mode="hard")
        distribution = "lognormal"
    elif bootstrap_mode == "measurement":
        states = classify_visit_states(
            visits,
            sigma_tech,
            mode="perturbed",
            rng=rng,
        )
        distribution = "lognormal"
    elif bootstrap_mode == "credible":
        states = classify_visit_states(
            visits,
            sigma_tech,
            mode="credible",
            eta=0.025,
        )
        distribution = "loglogistic"
    else:
        raise ValueError(bootstrap_mode)
    intervals = construct_intervals(states, baseline)
    usable = intervals.loc[
        intervals["type"].isin(["left", "interval", "right"])
    ].reset_index(drop=True)
    sampled_index = rng.integers(0, len(usable), size=len(usable))
    sample = usable.iloc[sampled_index].copy()
    try:
        fit = fit_aft(
            sample,
            ["first_bmi"],
            distribution=distribution,
            nonnegative_features=["first_bmi"],
        )
        predictions: dict[float, np.ndarray] = {}
        policy_payload: dict[str, object] = {}
        for rho in [0.80, 0.90]:
            tau = fit.quantile(rho, baseline)
            predictions[rho] = tau.astype(np.float32)
            try:
                policy = exact_dp_policy(
                    sample["first_bmi"].to_numpy(float),
                    fit.quantile(rho, sample),
                    groups=4,
                    alpha=rho,
                    min_size=20,
                )
                policy_payload[f"rho{int(rho * 100)}_cuts"] = policy.cutpoints
                policy_payload[f"rho{int(rho * 100)}_times"] = [
                    segment.operational_time for segment in policy.segments
                ]
                policy_payload[f"rho{int(rho * 100)}_cost"] = policy.total_cost
            except Exception:
                policy_payload[f"rho{int(rho * 100)}_cuts"] = []
                policy_payload[f"rho{int(rho * 100)}_times"] = []
                policy_payload[f"rho{int(rho * 100)}_cost"] = np.nan
        row = {
            "iteration": iteration,
            "mode": bootstrap_mode,
            "success": bool(fit.success),
            "intercept": fit.coefficients[0],
            "beta_bmi_std": fit.coefficients[1],
            "sigma": fit.sigma,
            "bmi_mean": fit.means[0],
            "bmi_sd": fit.scales[0],
            "n_left": int((usable["type"] == "left").sum()),
            "n_interval": int((usable["type"] == "interval").sum()),
            "n_right": int((usable["type"] == "right").sum()),
            "n_uninformative": int(
                (intervals["type"] == "uninformative").sum()
            ),
            **policy_payload,
        }
        return row, predictions
    except Exception as exc:
        return (
            {
                "iteration": iteration,
                "mode": bootstrap_mode,
                "success": False,
                "error": repr(exc),
            },
            {},
        )


def q3_cv_one_seed(
    intervals: pd.DataFrame, seed: int
) -> list[dict]:
    candidates = {
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
    folds = shuffled_group_folds(
        intervals["code"].to_numpy(), n_splits=5, seed=seed
    )
    rows = []
    for fold_id, (train_index, test_index) in enumerate(folds):
        train = intervals.iloc[train_index].copy()
        test = intervals.iloc[test_index].copy()
        for name, features in candidates.items():
            fit = fit_aft(
                train,
                features,
                distribution="lognormal",
                nonnegative_features=(
                    ["first_bmi"] if "first_bmi" in features else []
                ),
            )
            log_values = evaluate_aft_loglik(fit, test)
            row = {
                "seed": seed,
                "fold": fold_id,
                "model": name,
                "nll": -float(np.mean(log_values)),
                "success": bool(fit.success),
            }
            briers = []
            for time_point in [12, 14, 16, 18, 20]:
                known, labels = determined_binary_at_time(test, time_point)
                if known.sum() == 0:
                    continue
                probability = fit.cdf(time_point, test.loc[known])
                brier = float(np.mean(np.square(labels - probability)))
                row[f"brier_{time_point}"] = brier
                briers.append(brier)
            row["mean_brier"] = float(np.mean(briers))
            rows.append(row)
    return rows


def summarize_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").columns
    rows = []
    for column in numeric:
        if column == "iteration":
            continue
        values = frame[column].dropna().to_numpy(float)
        if not len(values):
            continue
        rows.append(
            {
                "quantity": column,
                "n": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "q025": float(np.quantile(values, 0.025)),
                "median": float(np.median(values)),
                "q975": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("附件.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nipt_solution/outputs/stability"),
    )
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--q1-cv-seeds", type=int, default=20)
    parser.add_argument("--q3-cv-seeds", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = prepare_data(args.data)

    q1_cv_nested = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(q1_cv_one_seed)(data.male_visits, SEED + seed)
        for seed in range(args.q1_cv_seeds)
    )
    q1_cv = pd.DataFrame([row for batch in q1_cv_nested for row in batch])
    q1_cv.to_csv(args.output / "q1_group_cv.csv", index=False)
    q1_cv_summary = (
        q1_cv.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    q1_cv_summary.to_csv(args.output / "q1_group_cv_summary.csv", index=False)

    q1_boot_rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(q1_bootstrap_once)(data.male_visits, iteration)
        for iteration in range(args.bootstrap)
    )
    q1_boot = pd.DataFrame(q1_boot_rows)
    q1_boot.to_csv(args.output / "q1_cluster_bootstrap.csv", index=False)
    summarize_bootstrap(q1_boot.loc[q1_boot["success"]]).to_csv(
        args.output / "q1_cluster_bootstrap_summary.csv", index=False
    )

    baseline = data.male_baseline.sort_values("孕妇代码").reset_index(drop=True)
    for mode in ["ordinary", "measurement", "credible"]:
        results = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(q2_bootstrap_once)(
                data.male_visits,
                baseline,
                data.sigma_tech,
                iteration,
                mode,
            )
            for iteration in range(args.bootstrap)
        )
        rows = pd.DataFrame([item[0] for item in results])
        rows.to_csv(args.output / f"q2_bootstrap_{mode}.csv", index=False)
        summarize_bootstrap(rows.loc[rows["success"]]).to_csv(
            args.output / f"q2_bootstrap_{mode}_summary.csv", index=False
        )
        for rho in [0.80, 0.90]:
            matrices = [
                item[1][rho]
                for item in results
                if bool(item[0].get("success")) and rho in item[1]
            ]
            matrix = np.stack(matrices)
            np.savez_compressed(
                args.output / f"q2_tau_{mode}_rho{int(rho * 100)}.npz",
                tau=matrix,
                code=baseline["孕妇代码"].to_numpy(str),
                bmi=baseline["first_bmi"].to_numpy(float),
            )

    hard = construct_intervals(
        classify_visit_states(data.male_visits, data.sigma_tech, mode="hard"),
        data.male_baseline,
    )
    q3_nested = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(q3_cv_one_seed)(hard, SEED + 2000 + seed)
        for seed in range(args.q3_cv_seeds)
    )
    q3 = pd.DataFrame([row for batch in q3_nested for row in batch])
    q3.to_csv(args.output / "q3_repeated_group_cv.csv", index=False)
    q3_summary = (
        q3.groupby("model")[["nll", "mean_brier"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    q3_summary.to_csv(args.output / "q3_repeated_group_cv_summary.csv", index=False)

    manifest = {
        "bootstrap": args.bootstrap,
        "q1_cv_seeds": args.q1_cv_seeds,
        "q3_cv_seeds": args.q3_cv_seeds,
        "jobs": args.jobs,
        "q1_bootstrap_success": int(q1_boot["success"].sum()),
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output / "male_stability_manifest.json", manifest)
    print(json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

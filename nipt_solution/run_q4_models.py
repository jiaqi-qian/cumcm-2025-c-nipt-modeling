from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from nipt_core import (
    SEED,
    per_group_row_weights,
    prepare_data,
    to_jsonable,
    weighted_mean,
    write_json,
)


LABELS = ["T13", "T18", "T21"]


def prepare_female(frame: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(index=frame.index)
    data["row_id"] = frame["序号"].astype(int)
    data["code"] = frame["孕妇代码"].astype(str)
    label_text = frame["染色体的非整倍体"].fillna("").astype(str)
    for chrom in [13, 18, 21]:
        data[f"y_T{chrom}"] = label_text.str.contains(
            f"T{chrom}", regex=False
        ).astype(int)
        data[f"z{chrom}"] = frame[f"{chrom}号染色体的Z值"].astype(float)
        data[f"gc{chrom}"] = frame[f"{chrom}号染色体的GC含量"].astype(float)
    data["zx"] = frame["X染色体的Z值"].astype(float)
    data["x_concentration"] = frame["X染色体浓度"].astype(float)
    data["ga"] = frame["ga"].astype(float)
    data["bmi"] = frame["孕妇BMI"].astype(float)
    data["age"] = frame["年龄"].astype(float)
    data["height"] = frame["身高"].astype(float)
    data["weight"] = frame["体重"].astype(float)
    data["gravidity"] = frame["gravidity_num"].astype(float)
    data["parity"] = frame["生产次数"].astype(float)
    data["ivf"] = frame["ivf"].astype(float)
    data["iui"] = frame["iui"].astype(float)
    data["log_unique_reads"] = np.log1p(
        frame["唯一比对的读段数"].astype(float)
    )
    data["map_rate"] = frame["在参考基因组上比对的比例"].astype(float)
    data["duplicate_rate"] = frame["重复读段的比例"].astype(float)
    data["filter_rate"] = frame["被过滤掉读段数的比例"].astype(float)
    data["gc"] = frame["GC含量"].astype(float)
    data["gc_abs_dev"] = np.abs(data["gc"] - 0.5)
    return data


QUALITY_BASE = [
    "log_unique_reads",
    "map_rate",
    "duplicate_rate",
    "filter_rate",
    "gc_abs_dev",
    "gc13",
    "gc18",
    "gc21",
]


@dataclass
class QualityModel:
    medians: np.ndarray
    scales: np.ndarray
    gc_centers: np.ndarray
    center: np.ndarray
    precision: np.ndarray
    warn_threshold: float
    retest_threshold: float


def quality_matrix(
    frame: pd.DataFrame,
    *,
    medians: np.ndarray | None = None,
    gc_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = frame[QUALITY_BASE].to_numpy(float).copy()
    if gc_centers is None:
        gc_centers = np.nanmedian(base[:, -3:], axis=0)
    base[:, -3:] = np.abs(base[:, -3:] - gc_centers)
    if medians is None:
        medians = np.nanmedian(base, axis=0)
    base = np.where(np.isfinite(base), base, medians)
    return base, medians, gc_centers


def fit_quality_model(train: pd.DataFrame) -> QualityModel:
    raw, medians, gc_centers = quality_matrix(train)
    mad = np.median(np.abs(raw - medians), axis=0)
    scales = 1.4826 * mad
    fallback = np.std(raw, axis=0, ddof=0)
    scales = np.where(scales < 1e-8, np.maximum(fallback, 1e-8), scales)
    z = np.clip((raw - medians) / scales, -6, 6)
    covariance = LedoitWolf().fit(z)
    distance = covariance.mahalanobis(z)
    return QualityModel(
        medians=medians,
        scales=scales,
        gc_centers=gc_centers,
        center=covariance.location_,
        precision=covariance.precision_,
        warn_threshold=float(np.quantile(distance, 0.95)),
        retest_threshold=float(np.quantile(distance, 0.99)),
    )


def quality_distance(model: QualityModel, frame: pd.DataFrame) -> np.ndarray:
    raw, _, _ = quality_matrix(
        frame,
        medians=model.medians,
        gc_centers=model.gc_centers,
    )
    raw = np.where(np.isfinite(raw), raw, model.medians)
    z = np.clip((raw - model.medians) / model.scales, -6, 6)
    centered = z - model.center
    return np.einsum("ij,jk,ik->i", centered, model.precision, centered)


def features_for_label(frame: pd.DataFrame, label: str) -> tuple[np.ndarray, list[str]]:
    chrom = int(label[1:])
    others = [item for item in [13, 18, 21] if item != chrom]
    local = frame.copy()
    local["z_target"] = local[f"z{chrom}"]
    local["z_hinge3"] = np.maximum(local["z_target"] - 3, 0)
    local["gc_target"] = local[f"gc{chrom}"]
    columns = [
        "z_target",
        "z_hinge3",
        f"z{others[0]}",
        f"z{others[1]}",
        "zx",
        "x_concentration",
        "gc_target",
        "log_unique_reads",
        "map_rate",
        "duplicate_rate",
        "filter_rate",
        "gc_abs_dev",
        "quality_distance",
        "ga",
        "bmi",
        "age",
        "height",
        "gravidity",
        "parity",
        "ivf",
        "iui",
    ]
    return local[columns].to_numpy(float), columns


def group_stratified_splits(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(y)), y, groups))


def fit_weights(groups: np.ndarray, y: np.ndarray, balance: bool = True) -> np.ndarray:
    weights = per_group_row_weights(groups)
    if balance:
        pos = float(weights[y == 1].sum())
        neg = float(weights[y == 0].sum())
        if pos > 0 and neg > 0:
            weights = weights * np.where(
                y == 1,
                (pos + neg) / (2 * pos),
                (pos + neg) / (2 * neg),
            )
    return weights / np.mean(weights)


@dataclass
class SklearnModel:
    imputer: SimpleImputer
    scaler: StandardScaler | None
    model: Any

    def predict(self, x: np.ndarray) -> np.ndarray:
        transformed = self.imputer.transform(x)
        if self.scaler is not None:
            transformed = self.scaler.transform(transformed)
        return self.model.predict_proba(transformed)[:, 1]


def fit_elastic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    c_value: float,
    l1_ratio: float,
    seed: int,
    z_only: bool = False,
) -> SklearnModel:
    if z_only:
        x = x[:, :2]
    imputer = SimpleImputer(strategy="median").fit(x)
    transformed = imputer.transform(x)
    scaler = StandardScaler().fit(transformed)
    transformed = scaler.transform(transformed)
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=c_value,
        l1_ratio=l1_ratio,
        max_iter=5000,
        tol=1e-6,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(transformed, y, sample_weight=weights)
    wrapped = SklearnModel(imputer, scaler, model)
    if z_only:
        original_predict = wrapped.predict

        def predict_z(array: np.ndarray) -> np.ndarray:
            return original_predict(array[:, :2])

        wrapped.predict = predict_z  # type: ignore[method-assign]
    return wrapped


def fit_forest(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    max_depth: int,
    min_leaf: int,
    seed: int,
    trees: int,
) -> SklearnModel:
    imputer = SimpleImputer(strategy="median").fit(x)
    transformed = imputer.transform(x)
    model = RandomForestClassifier(
        n_estimators=trees,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        max_features=0.7,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(transformed, y, sample_weight=weights)
    return SklearnModel(imputer, None, model)


@dataclass
class FirthModel:
    imputer: SimpleImputer
    scaler: StandardScaler
    coefficients: np.ndarray
    columns: np.ndarray
    success: bool

    def predict(self, x: np.ndarray) -> np.ndarray:
        selected = x[:, self.columns]
        transformed = self.scaler.transform(self.imputer.transform(selected))
        design = np.column_stack([np.ones(len(transformed)), transformed])
        return expit(design @ self.coefficients)


def fit_firth(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> FirthModel:
    columns = np.array([0, 1, 4, 12, 15], dtype=int)
    selected = x[:, columns]
    imputer = SimpleImputer(strategy="median").fit(selected)
    transformed = imputer.transform(selected)
    scaler = StandardScaler().fit(transformed)
    transformed = scaler.transform(transformed)
    design = np.column_stack([np.ones(len(transformed)), transformed])

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        probability = expit(eta)
        loglik = np.sum(
            weights * (y * eta - np.logaddexp(0.0, eta))
        )
        working = weights * probability * (1 - probability)
        information = design.T @ (working[:, None] * design)
        information += np.eye(information.shape[0]) * 1e-7
        sign, logdet = np.linalg.slogdet(information)
        if sign <= 0:
            return 1e100
        return -float(loglik + 0.5 * logdet)

    fit = minimize(
        objective,
        np.zeros(design.shape[1]),
        method="BFGS",
        options={"maxiter": 2000, "gtol": 1e-7},
    )
    return FirthModel(
        imputer=imputer,
        scaler=scaler,
        coefficients=fit.x,
        columns=columns,
        success=bool(fit.success or np.isfinite(fit.fun)),
    )


def metric_bundle(
    y: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    probability = np.clip(probability, 1e-8, 1 - 1e-8)
    return {
        "roc_auc": float(roc_auc_score(y, probability, sample_weight=weights)),
        "pr_auc": float(
            average_precision_score(y, probability, sample_weight=weights)
        ),
        "brier": weighted_mean(np.square(y - probability), weights),
        "logloss": weighted_mean(
            -(y * np.log(probability) + (1 - y) * np.log(1 - probability)),
            weights,
        ),
    }


def model_fit_predict(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    x_test: np.ndarray,
    *,
    params: dict[str, Any],
    seed: int,
    trees: int,
) -> tuple[np.ndarray, Any]:
    weights = fit_weights(groups_train, y_train, balance=True)
    if model_name == "elastic_full":
        model = fit_elastic(
            x_train,
            y_train,
            weights,
            c_value=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            seed=seed,
        )
    elif model_name == "z_only":
        model = fit_elastic(
            x_train,
            y_train,
            weights,
            c_value=0.3,
            l1_ratio=0.0,
            seed=seed,
            z_only=True,
        )
    elif model_name == "random_forest":
        model = fit_forest(
            x_train,
            y_train,
            weights,
            max_depth=int(params["max_depth"]),
            min_leaf=int(params["min_leaf"]),
            seed=seed,
            trees=trees,
        )
    elif model_name == "firth":
        model = fit_firth(x_train, y_train, per_group_row_weights(groups_train))
    else:
        raise ValueError(model_name)
    return model.predict(x_test), model


def cross_validated_prediction(
    data: pd.DataFrame,
    label: str,
    model_name: str,
    params: dict[str, Any],
    *,
    seed: int,
    trees: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = data[f"y_{label}"].to_numpy(int)
    groups = data["code"].to_numpy(str)
    n_splits = 3 if label == "T21" else 5
    splits = group_stratified_splits(
        y, groups, n_splits=n_splits, seed=seed
    )
    oof = np.full(len(data), np.nan)
    quality_distance_oof = np.full(len(data), np.nan)
    quality_retest = np.zeros(len(data), dtype=bool)
    for fold, (train_index, test_index) in enumerate(splits):
        train = data.iloc[train_index].copy()
        test = data.iloc[test_index].copy()
        quality = fit_quality_model(train)
        train["quality_distance"] = quality_distance(quality, train)
        test["quality_distance"] = quality_distance(quality, test)
        x_train, _ = features_for_label(train, label)
        x_test, _ = features_for_label(test, label)
        prediction, _ = model_fit_predict(
            model_name,
            x_train,
            y[train_index],
            groups[train_index],
            x_test,
            params=params,
            seed=seed * 10 + fold,
            trees=trees,
        )
        oof[test_index] = prediction
        quality_distance_oof[test_index] = test["quality_distance"].to_numpy()
        quality_retest[test_index] = (
            test["quality_distance"].to_numpy() > quality.retest_threshold
        )
    return oof, quality_distance_oof, quality_retest


def tune_parameters(
    data: pd.DataFrame,
    label: str,
    model_name: str,
    *,
    trees: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if model_name == "elastic_full":
        grid = [
            {"C": c_value, "l1_ratio": l1}
            for c_value in [0.03, 0.1, 0.3, 1.0, 3.0]
            for l1 in [0.0, 0.5, 1.0]
        ]
    elif model_name == "random_forest":
        grid = [
            {"max_depth": depth, "min_leaf": leaf}
            for depth in [3, 5, 7]
            for leaf in [5, 10, 20]
        ]
    else:
        raise ValueError(model_name)
    y = data[f"y_{label}"].to_numpy(int)
    groups = data["code"].to_numpy(str)
    weights = per_group_row_weights(groups)
    rows = []
    for grid_id, params in enumerate(grid):
        probability, _, _ = cross_validated_prediction(
            data,
            label,
            model_name,
            params,
            seed=SEED + grid_id,
            trees=trees,
        )
        rows.append(
            {
                "label": label,
                "model": model_name,
                "params": json.dumps(params, sort_keys=True),
                **metric_bundle(y, probability, weights),
            }
        )
    table = pd.DataFrame(rows).sort_values(["brier", "pr_auc"], ascending=[True, False])
    return json.loads(table.iloc[0]["params"]), table


def stability_one_seed(
    data: pd.DataFrame,
    parameters: dict[str, dict[str, dict[str, Any]]],
    seed_index: int,
    *,
    trees: int,
) -> tuple[list[dict], list[dict]]:
    seed = SEED + 10000 + seed_index
    metrics = []
    predictions = []
    for label in LABELS:
        models = ["z_only", "elastic_full", "random_forest"]
        if label == "T21":
            models.append("firth")
        y = data[f"y_{label}"].to_numpy(int)
        groups = data["code"].to_numpy(str)
        eval_weights = per_group_row_weights(groups)
        for model_name in models:
            params = parameters[label].get(model_name, {})
            probability, quality_distance_oof, quality_retest = (
                cross_validated_prediction(
                    data,
                    label,
                    model_name,
                    params,
                    seed=seed,
                    trees=trees,
                )
            )
            metrics.append(
                {
                    "seed_index": seed_index,
                    "seed": seed,
                    "label": label,
                    "model": model_name,
                    **metric_bundle(y, probability, eval_weights),
                    "quality_retest_rate": weighted_mean(
                        quality_retest.astype(float), eval_weights
                    ),
                }
            )
            for index in range(len(data)):
                predictions.append(
                    {
                        "seed_index": seed_index,
                        "label": label,
                        "model": model_name,
                        "row_id": int(data.iloc[index]["row_id"]),
                        "code": str(data.iloc[index]["code"]),
                        "y": int(y[index]),
                        "probability": float(probability[index]),
                        "quality_distance": float(quality_distance_oof[index]),
                        "quality_retest": int(quality_retest[index]),
                    }
                )
    return metrics, predictions


def fit_platt(
    probability: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> LogisticRegression:
    logits = np.log(
        np.clip(probability, 1e-6, 1 - 1e-6)
        / np.clip(1 - probability, 1e-6, 1 - 1e-6)
    ).reshape(-1, 1)
    model = LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000)
    model.fit(logits, y, sample_weight=weights)
    return model


def apply_platt(model: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    logits = np.log(
        np.clip(probability, 1e-6, 1 - 1e-6)
        / np.clip(1 - probability, 1e-6, 1 - 1e-6)
    ).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def thresholds_for_triage(
    y: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    sensitivity_target: float = 0.90,
    specificity_target: float = 0.95,
) -> tuple[float, float]:
    candidates = np.unique(probability)
    low_threshold = float(np.min(candidates))
    low_best_specificity = -1.0
    high_threshold = float(np.max(candidates))
    high_best_sensitivity = -1.0
    positive_weight = float(weights[y == 1].sum())
    negative_weight = float(weights[y == 0].sum())
    for threshold in candidates:
        prediction = probability >= threshold
        sensitivity = (
            float(weights[(y == 1) & prediction].sum()) / positive_weight
        )
        specificity = (
            float(weights[(y == 0) & (~prediction)].sum()) / negative_weight
        )
        if (
            sensitivity >= sensitivity_target
            and specificity > low_best_specificity
        ):
            low_best_specificity = specificity
            low_threshold = float(threshold)
        if (
            specificity >= specificity_target
            and sensitivity > high_best_sensitivity
        ):
            high_best_sensitivity = sensitivity
            high_threshold = float(threshold)
    if high_threshold < low_threshold:
        midpoint = (high_threshold + low_threshold) / 2
        low_threshold = midpoint
        high_threshold = midpoint
    return low_threshold, high_threshold


def nested_parameter_grid(model_name: str) -> list[dict[str, Any]]:
    """Predeclared grids used strictly inside each outer training fold."""
    if model_name == "elastic_full":
        return [
            {"C": c_value, "l1_ratio": l1_ratio}
            for c_value in [0.03, 0.1, 0.3, 1.0]
            for l1_ratio in [0.0, 0.5, 1.0]
        ]
    if model_name == "random_forest":
        return [
            {"max_depth": max_depth, "min_leaf": min_leaf}
            for max_depth in [3, 5, 7]
            for min_leaf in [5, 10, 20]
        ]
    return [{}]


def prepare_inner_folds(
    train: pd.DataFrame,
    label: str,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Fit every preprocessing object on the corresponding inner train fold."""
    y = train[f"y_{label}"].to_numpy(int)
    groups = train["code"].to_numpy(str)
    splits = group_stratified_splits(
        y,
        groups,
        n_splits=3,
        seed=seed,
    )
    prepared: list[dict[str, Any]] = []
    for inner_fold, (fit_index, valid_index) in enumerate(splits):
        fit_frame = train.iloc[fit_index].copy()
        valid_frame = train.iloc[valid_index].copy()
        quality = fit_quality_model(fit_frame)
        fit_frame["quality_distance"] = quality_distance(quality, fit_frame)
        valid_frame["quality_distance"] = quality_distance(
            quality, valid_frame
        )
        x_fit, _ = features_for_label(fit_frame, label)
        x_valid, _ = features_for_label(valid_frame, label)
        prepared.append(
            {
                "inner_fold": inner_fold,
                "fit_index": fit_index,
                "valid_index": valid_index,
                "x_fit": x_fit,
                "x_valid": x_valid,
            }
        )
    return y, groups, prepared


def select_inner_parameters(
    train: pd.DataFrame,
    label: str,
    model_name: str,
    *,
    seed: int,
    trees: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, float]]:
    """Tune and predict only within an outer training fold."""
    y, groups, inner_folds = prepare_inner_folds(
        train, label, seed=seed
    )
    evaluation_weights = per_group_row_weights(groups)
    candidates: list[tuple[dict[str, Any], np.ndarray, dict[str, float]]] = []
    for parameter_index, candidate in enumerate(
        nested_parameter_grid(model_name)
    ):
        oof = np.full(len(train), np.nan)
        for fold in inner_folds:
            fit_index = fold["fit_index"]
            valid_index = fold["valid_index"]
            prediction, _ = model_fit_predict(
                model_name,
                fold["x_fit"],
                y[fit_index],
                groups[fit_index],
                fold["x_valid"],
                params=candidate,
                seed=seed + parameter_index * 100 + fold["inner_fold"],
                trees=trees,
            )
            oof[valid_index] = prediction
        bundle = metric_bundle(y, oof, evaluation_weights)
        candidates.append((candidate, oof, bundle))
    candidates.sort(
        key=lambda item: (
            item[2]["brier"],
            item[2]["logloss"],
            -item[2]["pr_auc"],
        )
    )
    return candidates[0]


def nested_final_one_seed(
    data: pd.DataFrame,
    label: str,
    model_name: str,
    params: dict[str, Any],
    seed_index: int,
    *,
    trees: int,
) -> tuple[dict, list[dict]]:
    seed = SEED + 50000 + seed_index
    y = data[f"y_{label}"].to_numpy(int)
    groups = data["code"].to_numpy(str)
    outer_splits = group_stratified_splits(
        y, groups, n_splits=(3 if label == "T21" else 5), seed=seed
    )
    oof = np.full(len(data), np.nan)
    low_thresholds = np.full(len(data), np.nan)
    high_thresholds = np.full(len(data), np.nan)
    quality_retest = np.zeros(len(data), dtype=bool)
    fold_parameters: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(outer_splits):
        train = data.iloc[train_index].copy()
        test = data.iloc[test_index].copy()
        y_train = y[train_index]
        groups_train = groups[train_index]
        selected_params, inner_oof, _ = select_inner_parameters(
            train,
            label,
            model_name,
            seed=seed + fold * 1000 + 1,
            trees=trees,
        )
        fold_parameters.append(selected_params)
        calibration_weights = per_group_row_weights(groups_train)
        calibrator = fit_platt(inner_oof, y_train, calibration_weights)
        calibrated_inner = apply_platt(calibrator, inner_oof)
        low_threshold, high_threshold = thresholds_for_triage(
            y_train,
            calibrated_inner,
            calibration_weights,
            sensitivity_target=0.90,
            specificity_target=0.95,
        )
        quality = fit_quality_model(train)
        train["quality_distance"] = quality_distance(quality, train)
        test["quality_distance"] = quality_distance(quality, test)
        x_train, _ = features_for_label(train, label)
        x_test, _ = features_for_label(test, label)
        raw_test, _ = model_fit_predict(
            model_name,
            x_train,
            y_train,
            groups_train,
            x_test,
            params=selected_params,
            seed=seed + fold * 100,
            trees=trees,
        )
        oof[test_index] = apply_platt(calibrator, raw_test)
        low_thresholds[test_index] = low_threshold
        high_thresholds[test_index] = high_threshold
        quality_retest[test_index] = (
            test["quality_distance"].to_numpy() > quality.retest_threshold
        )

    weights = per_group_row_weights(groups)
    definitive_positive = (oof >= high_thresholds) & (~quality_retest)
    definitive_negative = (oof < low_thresholds) & (~quality_retest)
    classified = definitive_positive | definitive_negative
    positive_weight = float(weights[y == 1].sum())
    negative_weight = float(weights[y == 0].sum())
    classified_positive_weight = float(weights[(y == 1) & classified].sum())
    classified_negative_weight = float(weights[(y == 0) & classified].sum())
    classified_weight = float(weights[classified].sum())
    correct_classified_weight = float(
        weights[
            ((y == 1) & definitive_positive)
            | ((y == 0) & definitive_negative)
        ].sum()
    )
    metric = {
        "seed_index": seed_index,
        "label": label,
        "model": model_name,
        **metric_bundle(y, oof, weights),
        "low_threshold_mean": weighted_mean(low_thresholds, weights),
        "high_threshold_mean": weighted_mean(high_thresholds, weights),
        "coverage": weighted_mean(classified.astype(float), weights),
        "sensitivity_all": float(
            weights[(y == 1) & definitive_positive].sum() / positive_weight
        ),
        "specificity_all": float(
            weights[(y == 0) & definitive_negative].sum() / negative_weight
        ),
        "selective_sensitivity": (
            float(
                weights[(y == 1) & definitive_positive].sum()
                / classified_positive_weight
            )
            if classified_positive_weight > 0
            else np.nan
        ),
        "selective_specificity": (
            float(
                weights[(y == 0) & definitive_negative].sum()
                / classified_negative_weight
            )
            if classified_negative_weight > 0
            else np.nan
        ),
        "selective_accuracy": (
            correct_classified_weight / classified_weight
            if classified_weight > 0
            else np.nan
        ),
        "quality_retest_rate": weighted_mean(
            quality_retest.astype(float), weights
        ),
        "fold_parameters": json.dumps(fold_parameters, sort_keys=True),
    }
    rows = []
    for index in range(len(data)):
        rows.append(
            {
                "seed_index": seed_index,
                "label": label,
                "model": model_name,
                "row_id": int(data.iloc[index]["row_id"]),
                "code": str(data.iloc[index]["code"]),
                "y": int(y[index]),
                "probability": float(oof[index]),
                "low_threshold": float(low_thresholds[index]),
                "high_threshold": float(high_thresholds[index]),
                "quality_retest": int(quality_retest[index]),
            }
        )
    return metric, rows


def promotion_summary(metrics: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    rows = []
    selected: dict[str, str] = {}
    for label in LABELS:
        subset = metrics.loc[metrics["label"] == label]
        elastic = subset.loc[subset["model"] == "elastic_full"].set_index(
            "seed_index"
        )
        forest = subset.loc[subset["model"] == "random_forest"].set_index(
            "seed_index"
        )
        common = elastic.index.intersection(forest.index)
        pr_gain = forest.loc[common, "pr_auc"] - elastic.loc[common, "pr_auc"]
        brier_gain = elastic.loc[common, "brier"] - forest.loc[common, "brier"]
        row = {
            "label": label,
            "pr_gain_mean": float(pr_gain.mean()),
            "pr_gain_q025": float(pr_gain.quantile(0.025)),
            "pr_gain_q975": float(pr_gain.quantile(0.975)),
            "brier_gain_mean": float(brier_gain.mean()),
            "brier_gain_q025": float(brier_gain.quantile(0.025)),
            "brier_gain_q975": float(brier_gain.quantile(0.975)),
        }
        promote = (
            row["pr_gain_q025"] > 0 and row["brier_gain_q025"] > 0
        )
        if label == "T21":
            firth = subset.loc[subset["model"] == "firth"]
            elastic_summary = elastic[["roc_auc", "pr_auc", "brier"]].mean()
            firth_summary = firth[["roc_auc", "pr_auc", "brier"]].mean()
            if (
                firth_summary["roc_auc"] >= elastic_summary["roc_auc"]
                and firth_summary["pr_auc"] >= elastic_summary["pr_auc"]
            ):
                selected[label] = "firth"
                row["decision"] = "select_firth_no_tree"
            else:
                selected[label] = "elastic_full"
                row["decision"] = "select_regularized_logistic_no_tree"
        elif promote:
            selected[label] = "random_forest"
            row["decision"] = "promote_random_forest"
        else:
            selected[label] = "elastic_full"
            row["decision"] = "retain_elastic"
        rows.append(row)
    return selected, pd.DataFrame(rows)


def aggregate_final_decisions(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    selected: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        subset = predictions.loc[
            (predictions["label"] == label)
            & (predictions["model"] == selected[label])
        ]
        grouped = subset.groupby(["row_id", "code"], sort=False)
        for (row_id, code), group in grouped:
            probability = group["probability"].to_numpy(float)
            low_threshold = group["low_threshold"].to_numpy(float)
            high_threshold = group["high_threshold"].to_numpy(float)
            quality_rate = float(group["quality_retest"].mean())
            lower = float(np.quantile(probability, 0.025))
            median = float(np.median(probability))
            upper = float(np.quantile(probability, 0.975))
            low_threshold_median = float(np.median(low_threshold))
            high_threshold_median = float(np.median(high_threshold))
            if quality_rate >= 0.5:
                decision = "复检_质量"
            elif lower >= high_threshold_median:
                decision = "异常"
            elif upper < low_threshold_median:
                decision = "正常"
            else:
                decision = "复检_模型"
            rows.append(
                {
                    "row_id": int(row_id),
                    "code": code,
                    "label": label,
                    "model": selected[label],
                    "probability_q025": lower,
                    "probability_median": median,
                    "probability_q975": upper,
                    "low_threshold_median": low_threshold_median,
                    "high_threshold_median": high_threshold_median,
                    "quality_retest_frequency": quality_rate,
                    "decision": decision,
                }
            )
    long = pd.DataFrame(rows)
    wide = long.pivot(index=["row_id", "code"], columns="label")
    wide.columns = [f"{left}_{right}" for left, right in wide.columns]
    wide = wide.reset_index()
    decision_columns = [f"decision_{label}" for label in LABELS]

    def overall(row: pd.Series) -> str:
        values = [row[column] for column in decision_columns]
        abnormal = [
            label
            for label, value in zip(LABELS, values)
            if value == "异常"
        ]
        if abnormal:
            return "异常:" + "+".join(abnormal)
        if any(str(value).startswith("复检") for value in values):
            return "建议复检"
        return "正常"

    wide["overall_decision"] = wide.apply(overall, axis=1)
    return wide


def fit_operational_models(
    data: pd.DataFrame,
    selected: dict[str, str],
    parameters: dict[str, dict[str, dict[str, Any]]],
    *,
    trees: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit deployable full-data models after all validation is finished."""
    quality = fit_quality_model(data)
    full = data.copy()
    full["quality_distance"] = quality_distance(quality, full)
    term_rows: list[dict[str, Any]] = []
    operational: dict[str, Any] = {
        "quality": {
            "feature_names": QUALITY_BASE,
            "medians": quality.medians,
            "scales": quality.scales,
            "gc_centers": quality.gc_centers,
            "center": quality.center,
            "precision": quality.precision,
            "warn_threshold": quality.warn_threshold,
            "retest_threshold": quality.retest_threshold,
        },
        "labels": {},
    }
    for label in LABELS:
        y = full[f"y_{label}"].to_numpy(int)
        groups = full["code"].to_numpy(str)
        x, feature_names = features_for_label(full, label)
        model_name = selected[label]
        params = parameters[label].get(model_name, {})
        _, fitted = model_fit_predict(
            model_name,
            x,
            y,
            groups,
            x,
            params=params,
            seed=SEED + 99000 + int(label[1:]),
            trees=trees,
        )

        oof, _, _ = cross_validated_prediction(
            data,
            label,
            model_name,
            params,
            seed=SEED + 98000 + int(label[1:]),
            trees=trees,
        )
        calibration_weights = per_group_row_weights(groups)
        calibrator = fit_platt(oof, y, calibration_weights)
        calibrated = apply_platt(calibrator, oof)
        low_threshold, high_threshold = thresholds_for_triage(
            y,
            calibrated,
            calibration_weights,
            sensitivity_target=0.90,
            specificity_target=0.95,
        )
        label_payload: dict[str, Any] = {
            "model": model_name,
            "parameters": params,
            "feature_names": feature_names,
            "platt_intercept": float(calibrator.intercept_[0]),
            "platt_slope": float(calibrator.coef_[0, 0]),
            "normal_below": low_threshold,
            "abnormal_at_or_above": high_threshold,
            "cross_fitted_metrics": metric_bundle(
                y, calibrated, calibration_weights
            ),
        }
        if isinstance(fitted, SklearnModel) and hasattr(
            fitted.model, "coef_"
        ):
            coefficients = fitted.model.coef_[0]
            intercept = float(fitted.model.intercept_[0])
            label_payload["standardized_intercept"] = intercept
            for name, coefficient in zip(feature_names, coefficients):
                term_rows.append(
                    {
                        "label": label,
                        "model": model_name,
                        "term": name,
                        "value": float(coefficient),
                        "quantity": "standardized_coefficient",
                    }
                )
        elif isinstance(fitted, SklearnModel) and hasattr(
            fitted.model, "feature_importances_"
        ):
            for name, importance in zip(
                feature_names, fitted.model.feature_importances_
            ):
                term_rows.append(
                    {
                        "label": label,
                        "model": model_name,
                        "term": name,
                        "value": float(importance),
                        "quantity": "feature_importance",
                    }
                )
        elif isinstance(fitted, FirthModel):
            selected_names = [feature_names[index] for index in fitted.columns]
            term_rows.append(
                {
                    "label": label,
                    "model": model_name,
                    "term": "const",
                    "value": float(fitted.coefficients[0]),
                    "quantity": "standardized_coefficient",
                }
            )
            for name, coefficient in zip(
                selected_names, fitted.coefficients[1:]
            ):
                term_rows.append(
                    {
                        "label": label,
                        "model": model_name,
                        "term": name,
                        "value": float(coefficient),
                        "quantity": "standardized_coefficient",
                    }
                )
        operational["labels"][label] = label_payload
    return pd.DataFrame(term_rows), operational


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("附件.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nipt_solution/outputs/q4"),
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--nested-seeds", type=int, default=20)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--tune-trees", type=int, default=180)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prepared = prepare_data(args.data)
    data = prepare_female(prepared.female_raw)

    parameters: dict[str, dict[str, dict[str, Any]]] = {}
    tuning_tables = []
    for label in LABELS:
        parameters[label] = {"z_only": {}, "firth": {}}
        elastic, table = tune_parameters(
            data, label, "elastic_full", trees=args.tune_trees
        )
        parameters[label]["elastic_full"] = elastic
        tuning_tables.append(table)
        forest, table = tune_parameters(
            data, label, "random_forest", trees=args.tune_trees
        )
        parameters[label]["random_forest"] = forest
        tuning_tables.append(table)
    pd.concat(tuning_tables, ignore_index=True).to_csv(
        args.output / "q4_hyperparameter_tuning.csv", index=False
    )
    write_json(args.output / "q4_selected_hyperparameters.json", parameters)

    stability_results = Parallel(
        n_jobs=args.jobs, verbose=5, prefer="threads"
    )(
        delayed(stability_one_seed)(
            data,
            parameters,
            seed_index,
            trees=args.trees,
        )
        for seed_index in range(args.seeds)
    )
    metrics = pd.DataFrame(
        [row for result in stability_results for row in result[0]]
    )
    predictions = pd.DataFrame(
        [row for result in stability_results for row in result[1]]
    )
    metrics.to_csv(args.output / "q4_seed_metrics.csv", index=False)
    predictions.to_csv(args.output / "q4_seed_oof_predictions.csv", index=False)
    metric_summary = (
        metrics.groupby(["label", "model"])[
            ["roc_auc", "pr_auc", "brier", "logloss", "quality_retest_rate"]
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    metric_summary.to_csv(args.output / "q4_seed_metric_summary.csv", index=False)
    selected, promotion = promotion_summary(metrics)
    promotion.to_csv(args.output / "q4_ml_promotion.csv", index=False)
    write_json(args.output / "q4_selected_models.json", selected)

    nested_results = Parallel(
        n_jobs=args.jobs, verbose=5, prefer="threads"
    )(
        delayed(nested_final_one_seed)(
            data,
            label,
            selected[label],
            parameters[label][selected[label]],
            seed_index,
            trees=args.trees,
        )
        for seed_index in range(args.nested_seeds)
        for label in LABELS
    )
    nested_metrics = pd.DataFrame([item[0] for item in nested_results])
    nested_predictions = pd.DataFrame(
        [row for item in nested_results for row in item[1]]
    )
    nested_metrics.to_csv(args.output / "q4_nested_final_metrics.csv", index=False)
    nested_predictions.to_csv(
        args.output / "q4_nested_final_predictions.csv", index=False
    )
    decisions = aggregate_final_decisions(data, nested_predictions, selected)
    decisions.to_csv(args.output / "q4_final_record_decisions.csv", index=False)
    decision_summary = (
        decisions["overall_decision"].value_counts(dropna=False).rename_axis(
            "decision"
        ).reset_index(name="records")
    )
    decision_summary.to_csv(
        args.output / "q4_final_decision_summary.csv", index=False
    )
    terms, operational = fit_operational_models(
        data,
        selected,
        parameters,
        trees=args.trees,
    )
    terms.to_csv(args.output / "q4_final_model_terms.csv", index=False)
    write_json(
        args.output / "q4_operational_parameters.json", operational
    )

    manifest = {
        "seeds": args.seeds,
        "nested_seeds": args.nested_seeds,
        "trees": args.trees,
        "selected_models": selected,
        "parameters": parameters,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output / "q4_manifest.json", manifest)
    print(json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

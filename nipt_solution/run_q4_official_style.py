#!/usr/bin/env python3
"""Leakage-safe reconstruction of the official Q4 quality-stratified rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


SEED = 20260824
TARGETS = ["T13", "T18", "T21"]
ALL_TARGETS = ["ANY", *TARGETS]
VARIANTS = ["unweighted", "group_class_weighted"]
QUALITY_LEVELS = ["good", "typ", "marg"]
FEATURES = ["z_target", "gc_target", "logL", "M", "N", "O", "AA", "BMI", "ZX"]
PUBLISHED_THRESHOLDS = {
    "T13": {
        "good": (3.223, 3.477),
        "typ": (2.896, 3.145),
        "marg": (2.429, 2.671),
    },
    "T18": {
        "good": (2.904, 3.119),
        "typ": (2.721, 2.931),
        "marg": (2.527, 2.732),
    },
    "T21": {
        "good": (2.298, 2.310),
        "typ": (2.122, 2.133),
        "marg": (1.966, 1.976),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("附件.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nipt_solution/outputs/q4_official_style"),
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--outer-splits", type=int, default=4)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument(
        "--c-grid", type=str, default="0.03,0.1,0.3,1,3,10"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def parse_date(value: object) -> pd.Timestamp:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def load_data(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="女胎检测数据")
    raw.columns = raw.columns.str.strip()
    data = pd.DataFrame(index=raw.index)
    data["row_id"] = numeric(raw["序号"]).astype(int)
    data["code"] = raw["孕妇代码"].astype(str)
    data["date"] = raw["检测日期"].map(parse_date)
    labels = raw["染色体的非整倍体"].fillna("").astype(str)
    for chromosome in (13, 18, 21):
        data[f"y_T{chromosome}"] = labels.str.contains(
            f"T{chromosome}", regex=False
        ).astype(int)
        data[f"z{chromosome}"] = numeric(raw[f"{chromosome}号染色体的Z值"])
        data[f"gc{chromosome}"] = numeric(
            raw[f"{chromosome}号染色体的GC含量"]
        )
    data["y_ANY"] = data[[f"y_{target}" for target in TARGETS]].max(axis=1)
    data["L"] = numeric(raw["原始读段数"])
    data["M"] = numeric(raw["在参考基因组上比对的比例"])
    data["N"] = numeric(raw["重复读段的比例"])
    data["O"] = numeric(raw["唯一比对的读段数"])
    data["AA"] = numeric(raw["被过滤掉读段数的比例"])
    data["BMI"] = numeric(raw["孕妇BMI"])
    data["ZX"] = numeric(raw["X染色体的Z值"])
    data["logL"] = np.log(data["L"].clip(lower=1.0))
    return data.reset_index(drop=True)


def group_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    weights = np.array([1.0 / counts[item] for item in groups], dtype=float)
    return weights / weights.mean()


def fit_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    weights = group_weights(groups)
    negative_mass = weights[y == 0].sum()
    positive_mass = weights[y == 1].sum()
    total = negative_mass + positive_mass
    if negative_mass <= 0 or positive_mass <= 0:
        return weights
    weights *= np.where(
        y == 1,
        total / (2.0 * positive_mass),
        total / (2.0 * negative_mass),
    )
    return weights / weights.mean()


def grouped_splits(
    y: np.ndarray, groups: np.ndarray, requested: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    woman_labels = (
        pd.DataFrame({"group": groups, "y": y})
        .groupby("group", sort=False)["y"]
        .max()
    )
    count = max(
        2,
        min(
            requested,
            int(woman_labels.sum()),
            int(len(woman_labels) - woman_labels.sum()),
        ),
    )
    cv = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    result = list(cv.split(np.zeros(len(y)), y, groups))
    for train, test in result:
        if set(groups[train]).intersection(groups[test]):
            raise RuntimeError("Pregnant-woman leakage detected")
    return result


def target_number(target: str) -> int:
    return int(target[1:])


def raw_feature_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    chromosome = target_number(target)
    return pd.DataFrame(
        {
            "z_target": frame[f"z{chromosome}"],
            "gc_target": frame[f"gc{chromosome}"],
            "logL": frame["logL"],
            "M": frame["M"],
            "N": frame["N"],
            "O": frame["O"],
            "AA": frame["AA"],
            "BMI": frame["BMI"],
            "ZX": frame["ZX"],
        },
        index=frame.index,
    )


def robust_location_scale(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().astype(float)
    median = float(values.median()) if len(values) else 0.0
    mad = float((values - median).abs().median()) if len(values) else 0.0
    if not np.isfinite(mad) or mad <= 1e-12:
        q25, q75 = values.quantile([0.25, 0.75]) if len(values) else (0.0, 0.0)
        mad = float((q75 - q25) / 1.349)
    if not np.isfinite(mad) or mad <= 1e-12:
        mad = float(values.std(ddof=0)) if len(values) else 1.0
    if not np.isfinite(mad) or mad <= 1e-12:
        mad = 1.0
    return median, mad


@dataclass
class OfficialState:
    target: str
    qc_thresholds: dict[str, float]
    impute: dict[str, float]
    robust: dict[str, tuple[float, float]]
    gc_center: float
    qi_q20: float
    qi_q80: float


def qc_mask(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    mask = (
        frame["L"].ge(thresholds["L_low"])
        & frame["M"].ge(thresholds["M_low"])
        & frame["O"].ge(thresholds["O_low"])
        & frame["N"].le(thresholds["N_high"])
        & frame["AA"].le(thresholds["AA_high"])
    )
    return mask.fillna(False).to_numpy(bool)


def fit_state(frame: pd.DataFrame, target: str) -> OfficialState:
    thresholds = {
        "L_low": float(frame["L"].quantile(0.05)),
        "M_low": float(frame["M"].quantile(0.05)),
        "O_low": float(frame["O"].quantile(0.05)),
        "N_high": float(frame["N"].quantile(0.95)),
        "AA_high": float(frame["AA"].quantile(0.95)),
    }
    passed = qc_mask(frame, thresholds)
    reference = frame.loc[passed].copy()
    if len(reference) < 20:
        raise RuntimeError("Too few QC-passing training records")
    raw = raw_feature_frame(reference, target)
    impute = {
        column: float(raw[column].median()) if raw[column].notna().any() else 0.0
        for column in FEATURES
    }
    raw = raw.fillna(impute)
    gc_center = float(raw["gc_target"].median())
    raw["gcdev"] = (raw["gc_target"] - gc_center).abs()
    robust = {
        column: robust_location_scale(raw[column])
        for column in [*FEATURES, "gcdev"]
    }
    standardized = pd.DataFrame(index=raw.index)
    for column, (location, scale) in robust.items():
        standardized[column] = (raw[column] - location) / scale
    qi = (
        standardized["M"]
        + standardized["O"]
        + standardized["logL"]
        - standardized["N"]
        - standardized["AA"]
        - standardized["gcdev"]
    )
    return OfficialState(
        target=target,
        qc_thresholds=thresholds,
        impute=impute,
        robust=robust,
        gc_center=gc_center,
        qi_q20=float(qi.quantile(0.20)),
        qi_q80=float(qi.quantile(0.80)),
    )


def transform(
    frame: pd.DataFrame, state: OfficialState
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    raw = raw_feature_frame(frame, state.target).fillna(state.impute)
    raw["gcdev"] = (raw["gc_target"] - state.gc_center).abs()
    standardized = pd.DataFrame(index=raw.index)
    for column, (location, scale) in state.robust.items():
        standardized[column] = (raw[column] - location) / scale
    qi = (
        standardized["M"]
        + standardized["O"]
        + standardized["logL"]
        - standardized["N"]
        - standardized["AA"]
        - standardized["gcdev"]
    ).to_numpy(float)
    quality = np.full(len(frame), "typ", dtype=object)
    quality[qi < state.qi_q20] = "marg"
    quality[qi > state.qi_q80] = "good"
    passed = qc_mask(frame, state.qc_thresholds)
    return standardized[FEATURES].reset_index(drop=True), passed, quality, qi


def make_model(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=3000,
        tol=1e-7,
        random_state=seed,
    )


def fit_model(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    passed: np.ndarray,
    c_value: float,
    variant: str,
    seed: int,
) -> LogisticRegression:
    local = passed.copy()
    if len(np.unique(y[local])) < 2:
        raise RuntimeError("QC-passing training data lost a class")
    model = make_model(c_value, seed)
    if variant == "group_class_weighted":
        weights = fit_weights(y[local], groups[local])
        model.fit(X.loc[local], y[local], sample_weight=weights)
    else:
        model.fit(X.loc[local], y[local])
    return model


def inner_candidate_scores(
    frame: pd.DataFrame,
    target: str,
    variant: str,
    c_grid: list[float],
    seed: int,
    inner_splits: int,
) -> tuple[float, dict[float, float], np.ndarray, np.ndarray, np.ndarray]:
    y = frame[f"y_{target}"].to_numpy(int)
    groups = frame["code"].astype(str).to_numpy()
    scores = {c_value: np.full(len(frame), np.nan) for c_value in c_grid}
    passed_all = np.zeros(len(frame), dtype=bool)
    quality_all = np.full(len(frame), "typ", dtype=object)
    for fold, (train, test) in enumerate(
        grouped_splits(y, groups, inner_splits, seed)
    ):
        training = frame.iloc[train].reset_index(drop=True)
        validation = frame.iloc[test].reset_index(drop=True)
        state = fit_state(training, target)
        X_train, passed_train, _, _ = transform(training, state)
        X_test, passed_test, quality_test, _ = transform(validation, state)
        passed_all[test] = passed_test
        quality_all[test] = quality_test
        for c_value in c_grid:
            model = fit_model(
                X_train,
                y[train],
                groups[train],
                passed_train,
                c_value,
                variant,
                seed + fold * 1009 + int(round(c_value * 100)),
            )
            scores[c_value][test] = model.predict_proba(X_test)[:, 1]
    objectives: dict[float, float] = {}
    valid = passed_all & np.isfinite(scores[c_grid[0]])
    weights = group_weights(groups[valid])
    for c_value in c_grid:
        local_score = scores[c_value][valid]
        if len(np.unique(y[valid])) < 2:
            objectives[c_value] = -np.inf
            continue
        ap = average_precision_score(y[valid], local_score, sample_weight=weights)
        auc = roc_auc_score(y[valid], local_score, sample_weight=weights)
        objectives[c_value] = float(0.65 * ap + 0.35 * auc)
    best_c = max(c_grid, key=lambda item: (objectives[item], -abs(math.log10(item))))
    return best_c, objectives, scores[best_c], passed_all, quality_all


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    index = min(int(np.searchsorted(cumulative, q, side="left")), len(values) - 1)
    return float(values[index])


def choose_thresholds(
    y: np.ndarray,
    scores: np.ndarray,
    passed: np.ndarray,
    quality: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
    valid_negative = passed & (y == 0) & np.isfinite(scores)
    if valid_negative.sum() < 20:
        raise RuntimeError("Too few inner-OOF negative scores for thresholding")
    all_weights = group_weights(groups[valid_negative])
    pooled = {
        "lo": weighted_quantile(scores[valid_negative], all_weights, 0.99),
        "hi": weighted_quantile(scores[valid_negative], all_weights, 0.995),
    }
    result: dict[str, tuple[float, float]] = {}
    details: list[dict[str, Any]] = []
    for level in QUALITY_LEVELS:
        local = valid_negative & (quality == level)
        women = np.unique(groups[local])
        fallback = len(women) < 30 or local.sum() < 40
        if fallback:
            low, high = pooled["lo"], pooled["hi"]
        else:
            weights = group_weights(groups[local])
            low = weighted_quantile(scores[local], weights, 0.99)
            high = weighted_quantile(scores[local], weights, 0.995)
        high = max(low, high)
        result[level] = (float(low), float(high))
        details.append(
            {
                "quality": level,
                "low_threshold": low,
                "high_threshold": high,
                "negative_records": int(local.sum()),
                "negative_women": int(len(women)),
                "pooled_fallback": fallback,
            }
        )
    return result, details


def threshold_vectors(
    quality: np.ndarray, thresholds: dict[str, tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    low = np.array([thresholds[str(item)][0] for item in quality], dtype=float)
    high = np.array([thresholds[str(item)][1] for item in quality], dtype=float)
    return low, high


def apply_policy(
    scores: np.ndarray,
    passed: np.ndarray,
    quality: np.ndarray,
    thresholds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    low, high = threshold_vectors(quality, thresholds)
    low_call = passed & (scores > low)
    high_call = passed & (scores > high)
    negative = passed & (scores <= low)
    state = np.full(len(scores), "retest", dtype=object)
    state[negative] = "negative"
    state[high_call] = "positive"
    state[~passed] = "qc_retest"
    return pd.DataFrame(
        {
            "score": scores,
            "qc_pass": passed.astype(int),
            "quality": quality,
            "low_threshold": low,
            "high_threshold": high,
            "low_call": low_call.astype(int),
            "high_call": high_call.astype(int),
            "negative_call": negative.astype(int),
            "state": state,
        }
    )


def classification_metrics(
    y: np.ndarray, pred: np.ndarray, groups: np.ndarray
) -> dict[str, float]:
    weights = group_weights(groups)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1], sample_weight=weights
    ).ravel()
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    specificity = tn / max(tn + fp, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1e-12)
    return {
        "precision_w": float(precision),
        "recall_w": float(recall),
        "specificity_w": float(specificity),
        "f1_w": float(f1),
        "accuracy_w": float(accuracy),
        "precision_raw": float(precision_score(y, pred, zero_division=0)),
        "recall_raw": float(recall_score(y, pred, zero_division=0)),
        "n": int(len(y)),
        "positive": int(y.sum()),
        "predicted_positive": int(pred.sum()),
    }


def policy_metrics(
    y: np.ndarray,
    policy: pd.DataFrame,
    groups: np.ndarray,
) -> dict[str, float]:
    weights = group_weights(groups)
    negative = policy["negative_call"].to_numpy(int).astype(bool)
    positive = policy["high_call"].to_numpy(int).astype(bool)
    retest = ~(negative | positive)
    total = weights.sum()
    positive_mass = weights[y == 1].sum()
    negative_mass = weights[y == 0].sum()
    high_tp = weights[positive & (y == 1)].sum()
    high_fp = weights[positive & (y == 0)].sum()
    clear_tn = weights[negative & (y == 0)].sum()
    clear_fn = weights[negative & (y == 1)].sum()
    follow_up = ~negative
    follow_tp = weights[follow_up & (y == 1)].sum()
    follow_fp = weights[follow_up & (y == 0)].sum()
    return {
        "coverage_w": float(weights[negative | positive].sum() / total),
        "retest_rate_w": float(weights[retest].sum() / total),
        "qc_fail_rate_w": float(
            weights[policy["qc_pass"].to_numpy(int) == 0].sum() / total
        ),
        "negative_clear_rate_w": float(weights[negative].sum() / total),
        "positive_call_rate_w": float(weights[positive].sum() / total),
        "positive_precision_w": float(high_tp / max(high_tp + high_fp, 1e-12)),
        "positive_recall_all_w": float(high_tp / max(positive_mass, 1e-12)),
        "negative_npv_w": float(clear_tn / max(clear_tn + clear_fn, 1e-12)),
        "false_negative_clear_rate_w": float(clear_fn / max(positive_mass, 1e-12)),
        "followup_recall_w": float(follow_tp / max(positive_mass, 1e-12)),
        "followup_precision_w": float(
            follow_tp / max(follow_tp + follow_fp, 1e-12)
        ),
        "negative_specific_clear_w": float(clear_tn / max(negative_mass, 1e-12)),
        "retest_records": int(retest.sum()),
        "negative_records": int(negative.sum()),
        "positive_call_records": int(positive.sum()),
    }


def evaluate_policy(
    y: np.ndarray,
    policy: pd.DataFrame,
    groups: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    passed = policy["qc_pass"].to_numpy(int).astype(bool)
    for scope, mask in [("overall_qcfail_negative", np.ones(len(y), bool)), ("qc_pass", passed)]:
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            continue
        for mode, column in [("low", "low_call"), ("high", "high_call")]:
            rows.append(
                {
                    "scope": scope,
                    "mode": mode,
                    **classification_metrics(
                        y[mask],
                        policy.loc[mask, column].to_numpy(int),
                        groups[mask],
                    ),
                }
            )
    rows.append({"scope": "overall", "mode": "three_way", **policy_metrics(y, policy, groups)})
    return rows


def published_raw_policy(
    frame: pd.DataFrame,
    target: str,
    passed: np.ndarray,
    quality: np.ndarray,
) -> pd.DataFrame:
    thresholds = PUBLISHED_THRESHOLDS[target]
    scores = frame[f"z{target_number(target)}"].to_numpy(float)
    return apply_policy(scores, passed, quality, thresholds)


def evaluate_target_seed(
    data: pd.DataFrame,
    target: str,
    variant: str,
    c_grid: list[float],
    seed: int,
    outer_splits: int,
    inner_splits: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y = data[f"y_{target}"].to_numpy(int)
    groups = data["code"].astype(str).to_numpy()
    scores = np.zeros(len(data), dtype=float)
    passed = np.zeros(len(data), dtype=bool)
    quality = np.full(len(data), "typ", dtype=object)
    fold_used = np.zeros(len(data), dtype=int)
    low_threshold = np.zeros(len(data), dtype=float)
    high_threshold = np.zeros(len(data), dtype=float)
    tuning_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    raw_policy_parts: list[pd.DataFrame] = []
    for fold, (train, test) in enumerate(
        grouped_splits(y, groups, outer_splits, seed)
    ):
        training = data.iloc[train].reset_index(drop=True)
        testing = data.iloc[test].reset_index(drop=True)
        best_c, objectives, inner_score, inner_passed, inner_quality = (
            inner_candidate_scores(
                training,
                target,
                variant,
                c_grid,
                seed + fold * 10007,
                inner_splits,
            )
        )
        thresholds, details = choose_thresholds(
            y[train],
            inner_score,
            inner_passed,
            inner_quality,
            groups[train],
        )
        state = fit_state(training, target)
        X_train, train_passed, _, _ = transform(training, state)
        X_test, test_passed, test_quality, _ = transform(testing, state)
        model = fit_model(
            X_train,
            y[train],
            groups[train],
            train_passed,
            best_c,
            variant,
            seed + fold * 10007 + 5000,
        )
        test_score = model.predict_proba(X_test)[:, 1]
        test_policy = apply_policy(test_score, test_passed, test_quality, thresholds)
        scores[test] = test_score
        passed[test] = test_passed
        quality[test] = test_quality
        fold_used[test] = fold
        low_threshold[test] = test_policy["low_threshold"].to_numpy(float)
        high_threshold[test] = test_policy["high_threshold"].to_numpy(float)
        raw_local = published_raw_policy(testing, target, test_passed, test_quality)
        raw_local["index"] = test
        raw_policy_parts.append(raw_local)
        tuning_rows.append(
            {
                "target": target,
                "variant": variant,
                "seed": seed,
                "outer_fold": fold,
                "best_c": best_c,
                "candidate_objectives": json.dumps(objectives, sort_keys=True),
                "outer_train_records": len(train),
                "outer_test_records": len(test),
                "outer_train_women": len(np.unique(groups[train])),
                "outer_test_women": len(np.unique(groups[test])),
                "outer_train_qc_pass": int(train_passed.sum()),
                "outer_test_qc_pass": int(test_passed.sum()),
            }
        )
        for item in details:
            threshold_rows.append(
                {
                    "target": target,
                    "variant": variant,
                    "seed": seed,
                    "outer_fold": fold,
                    **item,
                }
            )
    policy = apply_policy(
        scores,
        passed,
        quality,
        {
            level: (0.0, 0.0) for level in QUALITY_LEVELS
        },
    )
    policy["low_threshold"] = low_threshold
    policy["high_threshold"] = high_threshold
    policy["low_call"] = (passed & (scores > low_threshold)).astype(int)
    policy["high_call"] = (passed & (scores > high_threshold)).astype(int)
    policy["negative_call"] = (passed & (scores <= low_threshold)).astype(int)
    state_values = np.full(len(data), "retest", dtype=object)
    state_values[policy["negative_call"].to_numpy(int) == 1] = "negative"
    state_values[policy["high_call"].to_numpy(int) == 1] = "positive"
    state_values[~passed] = "qc_retest"
    policy["state"] = state_values
    predictions = pd.DataFrame(
        {
            "target": target,
            "variant": variant,
            "seed": seed,
            "index": np.arange(len(data)),
            "row_id": data["row_id"],
            "code": data["code"],
            "y": y,
            "fold": fold_used,
            **{column: policy[column] for column in policy.columns},
        }
    )
    metric_rows = []
    for row in evaluate_policy(y, policy, groups):
        metric_rows.append(
            {"target": target, "variant": variant, "seed": seed, **row}
        )

    raw_policy = pd.concat(raw_policy_parts, ignore_index=True).sort_values("index")
    raw_policy = raw_policy.reset_index(drop=True)
    raw_metric_rows = []
    for row in evaluate_policy(y, raw_policy, groups):
        raw_metric_rows.append(
            {
                "target": target,
                "variant": "published_thresholds_on_raw_z",
                "seed": seed,
                **row,
            }
        )
    # The published raw-Z rule is independent of logistic-regression weighting;
    # emit it once per target/seed rather than duplicating it for both variants.
    if variant == "unweighted":
        metric_rows.extend(raw_metric_rows)
    return predictions, metric_rows, tuning_rows, threshold_rows


def combine_any(
    subtype_predictions: pd.DataFrame,
    data: pd.DataFrame,
    variant: str,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    local = subtype_predictions[
        (subtype_predictions["variant"] == variant)
        & (subtype_predictions["seed"] == seed)
    ].copy()
    parts = []
    for target in TARGETS:
        part = local[local["target"] == target].set_index("index")
        parts.append(part)
    combined = pd.DataFrame(index=np.arange(len(data)))
    combined["score"] = np.column_stack([part["score"] for part in parts]).max(axis=1)
    combined["qc_pass"] = np.column_stack([part["qc_pass"] for part in parts]).min(axis=1)
    combined["low_call"] = np.column_stack([part["low_call"] for part in parts]).max(axis=1)
    combined["high_call"] = np.column_stack([part["high_call"] for part in parts]).max(axis=1)
    combined["negative_call"] = np.column_stack(
        [part["negative_call"] for part in parts]
    ).min(axis=1)
    combined["quality"] = "combined"
    combined["low_threshold"] = np.nan
    combined["high_threshold"] = np.nan
    state = np.full(len(data), "retest", dtype=object)
    state[combined["negative_call"].to_numpy(int) == 1] = "negative"
    state[combined["high_call"].to_numpy(int) == 1] = "positive"
    state[combined["qc_pass"].to_numpy(int) == 0] = "qc_retest"
    combined["state"] = state
    y = data["y_ANY"].to_numpy(int)
    groups = data["code"].astype(str).to_numpy()
    predictions = pd.DataFrame(
        {
            "target": "ANY",
            "variant": variant,
            "seed": seed,
            "index": np.arange(len(data)),
            "row_id": data["row_id"],
            "code": data["code"],
            "y": y,
            "fold": -1,
            **{column: combined[column].to_numpy() for column in combined.columns},
        }
    )
    rows = [
        {"target": "ANY", "variant": variant, "seed": seed, **row}
        for row in evaluate_policy(y, combined, groups)
    ]
    return predictions, rows


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "variant", "scope", "mode"]
    excluded = set(keys + ["seed"])
    numeric_columns = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    rows: list[dict[str, Any]] = []
    for name, group in metrics.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, name))
        row["seeds"] = int(group["seed"].nunique())
        for column in numeric_columns:
            values = group[column].dropna()
            if len(values) == 0:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_q025"] = float(values.quantile(0.025))
            row[f"{column}_q975"] = float(values.quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def consensus_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = predictions.groupby(
        ["target", "variant", "index", "row_id", "code", "y"],
        as_index=False,
    )
    consensus = grouped.agg(
        mean_score=("score", "mean"),
        qc_pass_vote=("qc_pass", "mean"),
        low_vote=("low_call", "mean"),
        high_vote=("high_call", "mean"),
        negative_vote=("negative_call", "mean"),
    )
    consensus["qc_pass"] = (consensus["qc_pass_vote"] >= 0.5).astype(int)
    consensus["low_call"] = (consensus["low_vote"] >= 0.5).astype(int)
    consensus["high_call"] = (consensus["high_vote"] >= 0.5).astype(int)
    consensus["negative_call"] = (consensus["negative_vote"] >= 0.5).astype(int)
    return consensus


def bootstrap_consensus(
    consensus: pd.DataFrame, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (target, variant), local in consensus.groupby(["target", "variant"], sort=False):
        # Cache each woman's row indices once.  The original implementation
        # repeatedly filtered a DataFrame 147 times per bootstrap replicate;
        # this index-based version is mathematically identical and much faster.
        local = local.reset_index(drop=True)
        codes = local["code"].drop_duplicates().to_numpy()
        local_codes = local["code"].to_numpy()
        code_indices = [np.flatnonzero(local_codes == code) for code in codes]
        code_position = {code: position for position, code in enumerate(codes)}
        y_all = local["y"].to_numpy(int)
        qc_all = local["qc_pass"].to_numpy(int)
        low_all = local["low_call"].to_numpy(int)
        high_all = local["high_call"].to_numpy(int)
        negative_all = local["negative_call"].to_numpy(int)
        for replicate in range(replicates):
            sampled_codes = rng.choice(codes, size=len(codes), replace=True)
            sampled_positions = np.fromiter(
                (code_position[code] for code in sampled_codes),
                dtype=int,
                count=len(sampled_codes),
            )
            selected = [code_indices[position] for position in sampled_positions]
            sample_index = np.concatenate(selected)
            groups = np.repeat(
                np.arange(len(sampled_positions), dtype=int),
                [len(item) for item in selected],
            )
            y = y_all[sample_index]
            base = {"target": target, "variant": variant, "replicate": replicate}
            for mode, predictions in [
                ("low", low_all[sample_index]),
                ("high", high_all[sample_index]),
            ]:
                rows.append(
                    {
                        **base,
                        "scope": "overall_qcfail_negative",
                        "mode": mode,
                        **classification_metrics(y, predictions, groups),
                    }
                )
            policy = pd.DataFrame(
                {
                    "qc_pass": qc_all[sample_index],
                    "low_call": low_all[sample_index],
                    "high_call": high_all[sample_index],
                    "negative_call": negative_all[sample_index],
                }
            )
            rows.append(
                {
                    **base,
                    "scope": "overall",
                    "mode": "three_way",
                    **policy_metrics(y, policy, groups),
                }
            )
    frame = pd.DataFrame(rows)
    summary = summarize_metrics(frame.rename(columns={"replicate": "seed"}))
    summary = summary.rename(columns={"seeds": "replicates"})
    return frame, summary


def chronological_test(
    data: pd.DataFrame,
    target: str,
    variant: str,
    c_grid: list[float],
    inner_splits: int,
) -> list[dict[str, Any]]:
    first_dates = data.groupby("code")["date"].min().sort_values()
    cut = int(math.floor(0.8 * len(first_dates)))
    train_codes = set(first_dates.index[:cut])
    test_codes = set(first_dates.index[cut:])
    train_mask = data["code"].isin(train_codes).to_numpy()
    test_mask = data["code"].isin(test_codes).to_numpy()
    training = data.loc[train_mask].reset_index(drop=True)
    testing = data.loc[test_mask].reset_index(drop=True)
    y_train = training[f"y_{target}"].to_numpy(int)
    y_test = testing[f"y_{target}"].to_numpy(int)
    groups_train = training["code"].to_numpy(str)
    groups_test = testing["code"].to_numpy(str)
    best_c, objectives, inner_score, inner_passed, inner_quality = inner_candidate_scores(
        training,
        target,
        variant,
        c_grid,
        SEED + 990000 + target_number(target),
        inner_splits,
    )
    thresholds, _ = choose_thresholds(
        y_train,
        inner_score,
        inner_passed,
        inner_quality,
        groups_train,
    )
    state = fit_state(training, target)
    X_train, train_passed, _, _ = transform(training, state)
    X_test, test_passed, test_quality, _ = transform(testing, state)
    model = fit_model(
        X_train,
        y_train,
        groups_train,
        train_passed,
        best_c,
        variant,
        SEED + 995000 + target_number(target),
    )
    score = model.predict_proba(X_test)[:, 1]
    policy = apply_policy(score, test_passed, test_quality, thresholds)
    rows = []
    for item in evaluate_policy(y_test, policy, groups_test):
        rows.append(
            {
                "target": target,
                "variant": variant,
                "train_women": len(train_codes),
                "test_women": len(test_codes),
                "train_date_max": first_dates.iloc[cut - 1],
                "test_date_min": first_dates.iloc[cut],
                "best_c": best_c,
                "candidate_objectives": json.dumps(objectives, sort_keys=True),
                **item,
            }
        )
    return rows


def summarize_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in thresholds.groupby(
        ["target", "variant", "quality"], sort=False
    ):
        target, variant, quality = keys
        row: dict[str, Any] = {
            "target": target,
            "variant": variant,
            "quality": quality,
            "fold_estimates": len(group),
            "pooled_fallback_rate": float(group["pooled_fallback"].mean()),
        }
        for column in ["low_threshold", "high_threshold"]:
            values = group[column]
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = float(values.std(ddof=1))
            row[f"{column}_q025"] = float(values.quantile(0.025))
            row[f"{column}_q975"] = float(values.quantile(0.975))
            logits = np.log(np.clip(values, 1e-8, 1 - 1e-8) / np.clip(1 - values, 1e-8, 1))
            row[f"{column}_logit_mean"] = float(logits.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    started = time.time()
    c_grid = [float(item) for item in args.c_grid.split(",")]
    data = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for seed_index in range(args.seeds):
        seed = SEED + 100003 * seed_index
        seed_predictions: list[pd.DataFrame] = []
        for variant in VARIANTS:
            for target in TARGETS:
                predictions, metrics, tuning, thresholds = evaluate_target_seed(
                    data,
                    target,
                    variant,
                    c_grid,
                    seed,
                    args.outer_splits,
                    args.inner_splits,
                )
                prediction_parts.append(predictions)
                seed_predictions.append(predictions)
                metric_rows.extend(metrics)
                tuning_rows.extend(tuning)
                threshold_rows.extend(thresholds)
                key = next(
                    row
                    for row in metrics
                    if row["scope"] == "overall_qcfail_negative"
                    and row["mode"] == "low"
                    and row["variant"] == variant
                )
                print(
                    f"SEED {seed_index + 1:02d}/{args.seeds} {variant:<22} {target} "
                    f"P={key['precision_w']:.3f} R={key['recall_w']:.3f} F1={key['f1_w']:.3f}",
                    flush=True,
                )
            combined_source = pd.concat(seed_predictions, ignore_index=True)
            any_predictions, any_metrics = combine_any(
                combined_source, data, variant, seed
            )
            prediction_parts.append(any_predictions)
            metric_rows.extend(any_metrics)

    predictions_all = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    tuning = pd.DataFrame(tuning_rows)
    thresholds = pd.DataFrame(threshold_rows)
    summary = summarize_metrics(metrics)
    threshold_summary = summarize_thresholds(thresholds)
    consensus = consensus_predictions(
        predictions_all[predictions_all["variant"].isin(VARIANTS)]
    )
    bootstrap, bootstrap_summary = bootstrap_consensus(
        consensus, args.bootstrap, SEED + 800000
    )

    chronology_rows = []
    for variant in VARIANTS:
        for target in TARGETS:
            chronology_rows.extend(
                chronological_test(data, target, variant, c_grid, args.inner_splits)
            )
    chronology = pd.DataFrame(chronology_rows)

    data_audit = {
        "records": len(data),
        "women": int(data["code"].nunique()),
        "positive_records": {
            target: int(data[f"y_{target}"].sum()) for target in ALL_TARGETS
        },
        "official_slide_claim": {
            "records": 604,
            "T13_positive": 24,
            "T18_positive": 46,
            "T21_positive": 13,
            "QC_pass": 501,
        },
        "source_difference_flag": True,
    }
    manifest = {
        "method": "official-style QC + robust MAD scaling + QI strata + logistic regression + dual thresholds",
        "input_sha256": sha256_file(args.data),
        "seeds": args.seeds,
        "outer_cv": f"StratifiedGroupKFold({args.outer_splits}), group=pregnant woman",
        "inner_cv": f"StratifiedGroupKFold({args.inner_splits}), group=pregnant woman",
        "c_grid": c_grid,
        "low_threshold_target_specificity": 0.99,
        "high_threshold_target_specificity": 0.995,
        "threshold_source": "inner-OOF negative scores inside each outer training fold",
        "QC_source": "outer/inner training fold only",
        "robust_scaler_source": "QC-passing records in training fold only",
        "QI_cutpoint_source": "20th/80th percentile of training-fold QI",
        "published_fixed_threshold_test": "sensitivity audit on raw Excel Z; never selected as the main model",
        "bootstrap_replicates": args.bootstrap,
        "leakage_assertions": {
            "woman_overlap_between_train_test": 0,
            "test_data_used_for_QC_quantiles": False,
            "test_data_used_for_robust_scaling": False,
            "test_data_used_for_QI_cutpoints": False,
            "test_data_used_for_C_selection": False,
            "test_data_used_for_probability_thresholds": False,
            "labels_or_health_outcome_used_as_features": False,
        },
        "data_audit": data_audit,
        "runtime_seconds": time.time() - started,
    }

    predictions_all.to_csv(args.output / "q4_official_nested_predictions.csv", index=False)
    metrics.to_csv(args.output / "q4_official_seed_metrics.csv", index=False)
    summary.to_csv(args.output / "q4_official_metric_summary.csv", index=False)
    tuning.to_csv(args.output / "q4_official_tuning.csv", index=False)
    thresholds.to_csv(args.output / "q4_official_thresholds_by_fold.csv", index=False)
    threshold_summary.to_csv(
        args.output / "q4_official_threshold_summary.csv", index=False
    )
    consensus.to_csv(args.output / "q4_official_consensus_predictions.csv", index=False)
    bootstrap.to_csv(args.output / "q4_official_bootstrap_500.csv", index=False)
    bootstrap_summary.to_csv(
        args.output / "q4_official_bootstrap_summary.csv", index=False
    )
    chronology.to_csv(args.output / "q4_official_chronological_holdout.csv", index=False)
    write_json(args.output / "q4_official_manifest.json", manifest)


if __name__ == "__main__":
    main()

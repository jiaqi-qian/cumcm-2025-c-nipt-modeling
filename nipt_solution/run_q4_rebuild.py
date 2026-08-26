from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # Optional: only required for the XGBoost benchmark.
    XGBClassifier = None


SEED = 20260824
TARGETS = ["ANY", "T13", "T18", "T21"]
SUBTYPES = ["T13", "T18", "T21"]
MATERNAL_FEATURES = [
    "age",
    "height",
    "weight",
    "bmi",
    "gravidity",
    "parity",
    "ivf",
    "iui",
]
QC_FEATURES = [
    "log_raw_reads",
    "log_unique_reads",
    "unique_ratio",
    "map_rate",
    "duplicate_rate",
    "filter_rate",
    "gc",
    "gc13",
    "gc18",
    "gc21",
    "gc_abs_dev",
    "gcdev13",
    "gcdev18",
    "gcdev21",
]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ga(value: object) -> float:
    match = re.fullmatch(r"(\d+)w(?:\+(\d+))?", str(value).strip().lower())
    if not match:
        return np.nan
    return int(match.group(1)) + int(match.group(2) or 0) / 7.0


def parse_date(value: object) -> pd.Timestamp:
    text = str(value).strip()
    if re.fullmatch(r"\d{8}(?:\.0)?", text):
        text = text.split(".")[0]
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


@dataclass
class DataBundle:
    record: pd.DataFrame
    woman: pd.DataFrame
    feature_sets: dict[str, dict[str, list[str]]]
    audit: dict[str, Any]


def _history_features(data: pd.DataFrame, sources: list[str]) -> tuple[pd.DataFrame, list[str]]:
    ordered = data.sort_values(["code", "date", "ga", "row_id"]).copy()
    created: list[str] = []
    group = ordered.groupby("code", sort=False)
    ordered["prior_n"] = group.cumcount().astype(float)
    ordered["ga_since_first"] = ordered["ga"] - group["ga"].transform("min")
    created.extend(["prior_n", "ga_since_first"])
    for col in sources:
        g = ordered.groupby("code", sort=False)[col]
        prior = g.shift(1)
        first = g.transform("first")
        names = {
            f"{col}__previous": prior,
            f"{col}__delta": ordered[col] - prior,
            f"{col}__cummean": g.transform(lambda s: s.expanding().mean()),
            f"{col}__cumstd": g.transform(lambda s: s.expanding().std(ddof=0)),
            f"{col}__cummin": g.transform(lambda s: s.expanding().min()),
            f"{col}__cummax": g.transform(lambda s: s.expanding().max()),
            f"{col}__from_first": ordered[col] - first,
        }
        for name, values in names.items():
            ordered[name] = values
            created.append(name)
    ordered = ordered.sort_index()
    return ordered, created


def _woman_table(record: pd.DataFrame, compact_sources: list[str]) -> tuple[pd.DataFrame, list[str]]:
    ordered = record.sort_values(["code", "date", "ga", "row_id"]).copy()
    time_varying = [c for c in compact_sources if c not in MATERNAL_FEATURES]
    agg = ordered.groupby("code", sort=False)[time_varying].agg(
        ["mean", "std", "min", "max", "median", "first", "last"]
    )
    agg.columns = [f"{col}__{stat}" for col, stat in agg.columns]
    woman = agg.reset_index()
    maternal = ordered.groupby("code", sort=False)[MATERNAL_FEATURES].first().reset_index()
    woman = woman.merge(maternal, on="code", how="left", validate="one_to_one")
    counts = ordered.groupby("code", sort=False).size().rename("n_records").reset_index()
    woman = woman.merge(counts, on="code", how="left", validate="one_to_one")
    ga_range = (
        ordered.groupby("code", sort=False)["ga"].agg(["min", "max"]).reset_index()
    )
    ga_range["ga_range"] = ga_range["max"] - ga_range["min"]
    ga_range = ga_range.rename(columns={"min": "ga_min", "max": "ga_max"})
    woman = woman.merge(ga_range, on="code", how="left", validate="one_to_one")

    slope_sources = ["z13", "z18", "z21", "zx", "x_concentration", "gc", "filter_rate"]
    for col in slope_sources:
        rows: list[tuple[str, float]] = []
        for code, group in ordered.groupby("code", sort=False):
            valid = group[["ga", col]].dropna()
            if len(valid) >= 2 and valid["ga"].nunique() >= 2:
                slope = float(np.polyfit(valid["ga"], valid[col], 1)[0])
            else:
                slope = 0.0
            rows.append((code, slope))
        woman = woman.merge(
            pd.DataFrame(rows, columns=["code", f"{col}__slope"]),
            on="code",
            how="left",
            validate="one_to_one",
        )

    labels = ordered.groupby("code", sort=False)[[f"y_{t}" for t in TARGETS]].max()
    labels = labels.reset_index()
    first_date = ordered.groupby("code", sort=False)["date"].min().rename("first_date")
    woman = woman.merge(labels, on="code", how="left", validate="one_to_one")
    woman = woman.merge(first_date.reset_index(), on="code", how="left", validate="one_to_one")
    features = [
        col
        for col in woman.columns
        if col not in {"code", "first_date", *[f"y_{t}" for t in TARGETS]}
    ]
    return woman, features


def load_bundle(data_path: Path) -> DataBundle:
    raw = pd.read_excel(data_path, sheet_name="女胎检测数据")
    raw.columns = raw.columns.str.strip()
    data = pd.DataFrame(index=raw.index)
    data["row_id"] = numeric(raw["序号"]).astype(int)
    data["code"] = raw["孕妇代码"].astype(str)
    data["date"] = raw["检测日期"].map(parse_date)
    data["ga"] = raw["检测孕周"].map(parse_ga)
    data["draw"] = numeric(raw["检测抽血次数"])
    labels = raw["染色体的非整倍体"].fillna("").astype(str)
    for chrom in (13, 18, 21):
        data[f"y_T{chrom}"] = labels.str.contains(f"T{chrom}", regex=False).astype(int)
        data[f"z{chrom}"] = numeric(raw[f"{chrom}号染色体的Z值"])
        data[f"gc{chrom}"] = numeric(raw[f"{chrom}号染色体的GC含量"])
    data["y_ANY"] = data[["y_T13", "y_T18", "y_T21"]].max(axis=1)
    data["zx"] = numeric(raw["X染色体的Z值"])
    data["x_concentration"] = numeric(raw["X染色体浓度"])
    data["age"] = numeric(raw["年龄"])
    data["height"] = numeric(raw["身高"])
    data["weight"] = numeric(raw["体重"])
    data["bmi"] = numeric(raw["孕妇BMI"])
    data["gravidity"] = numeric(
        raw["怀孕次数"].astype(str).replace({"≥3": "3", ">=3": "3", "3及以上": "3"})
    )
    data["parity"] = numeric(raw["生产次数"])
    conception = raw["IVF妊娠"].fillna("").astype(str)
    data["ivf"] = conception.str.contains("IVF", regex=False).astype(float)
    data["iui"] = conception.str.contains("IUI", regex=False).astype(float)
    data["raw_reads"] = numeric(raw["原始读段数"])
    data["unique_reads"] = numeric(raw["唯一比对的读段数"])
    data["log_raw_reads"] = np.log1p(data["raw_reads"])
    data["log_unique_reads"] = np.log1p(data["unique_reads"])
    data["unique_ratio"] = data["unique_reads"] / data["raw_reads"].replace(0, np.nan)
    data["map_rate"] = numeric(raw["在参考基因组上比对的比例"])
    data["duplicate_rate"] = numeric(raw["重复读段的比例"])
    data["filter_rate"] = numeric(raw["被过滤掉读段数的比例"])
    data["gc"] = numeric(raw["GC含量"])
    data["gc_abs_dev"] = (data["gc"] - 0.5).abs()

    for chrom in (13, 18, 21):
        z = data[f"z{chrom}"]
        data[f"absz{chrom}"] = z.abs()
        data[f"sqz{chrom}"] = z**2
        for threshold in (1, 2, 3):
            data[f"pos{chrom}_{threshold}"] = (z - threshold).clip(lower=0)
            data[f"neg{chrom}_{threshold}"] = (-z - threshold).clip(lower=0)
        data[f"gcdev{chrom}"] = (data[f"gc{chrom}"] - data["gc"]).abs()
    for left, right in ((13, 18), (13, 21), (18, 21)):
        data[f"zprod{left}_{right}"] = data[f"z{left}"] * data[f"z{right}"]
        data[f"zdiff{left}_{right}"] = data[f"z{left}"] - data[f"z{right}"]
    abs_cols = ["absz13", "absz18", "absz21"]
    data["max_abs_z"] = data[abs_cols].max(axis=1)
    data["mean_abs_z"] = data[abs_cols].mean(axis=1)
    data["std_z"] = data[["z13", "z18", "z21"]].std(axis=1)
    data["n_abs_z_ge_2"] = (data[abs_cols] >= 2).sum(axis=1).astype(float)
    data["n_abs_z_ge_3"] = (data[abs_cols] >= 3).sum(axis=1).astype(float)

    z_engineered = [
        col
        for col in data.columns
        if re.fullmatch(r"(?:absz|sqz|pos|neg)\d+.*", col)
        or col.startswith("zprod")
        or col.startswith("zdiff")
    ] + ["max_abs_z", "mean_abs_z", "std_z", "n_abs_z_ge_2", "n_abs_z_ge_3"]
    core = list(dict.fromkeys([
        "z13",
        "z18",
        "z21",
        "zx",
        "x_concentration",
        "gc13",
        "gc18",
        "gc21",
        "ga",
        "draw",
        *MATERNAL_FEATURES,
        *QC_FEATURES,
    ]))
    engineered = list(dict.fromkeys([*core, *z_engineered]))
    history_sources = [
        "z13",
        "z18",
        "z21",
        "zx",
        "x_concentration",
        "map_rate",
        "duplicate_rate",
        "filter_rate",
        "gc",
        "gc13",
        "gc18",
        "gc21",
        "log_unique_reads",
    ]
    data, history = _history_features(data, history_sources)
    longitudinal = list(dict.fromkeys([*engineered, *history]))

    record_sets: dict[str, list[str]] = {
        "core": core,
        "engineered": engineered,
        "longitudinal": longitudinal,
        "no_maternal": [c for c in longitudinal if c not in MATERNAL_FEATURES],
        "no_qc": [
            c
            for c in longitudinal
            if c not in QC_FEATURES
            and not any(c.startswith(f"{q}__") for q in QC_FEATURES)
        ],
        "no_history": engineered,
    }
    for target in TARGETS:
        if target == "ANY":
            z_only = [
                "z13",
                "z18",
                "z21",
                "absz13",
                "absz18",
                "absz21",
                "max_abs_z",
                "n_abs_z_ge_2",
                "n_abs_z_ge_3",
            ]
        else:
            chrom = int(target[1:])
            z_only = [
                f"z{chrom}",
                f"absz{chrom}",
                f"sqz{chrom}",
                f"pos{chrom}_1",
                f"pos{chrom}_2",
                f"pos{chrom}_3",
                f"neg{chrom}_1",
                f"neg{chrom}_2",
                f"neg{chrom}_3",
            ]
        record_sets[f"z_only__{target}"] = z_only

    woman_sources = list(
        dict.fromkeys(
            [
                "z13",
                "z18",
                "z21",
                "absz13",
                "absz18",
                "absz21",
                "zx",
                "x_concentration",
                "ga",
                "draw",
                "map_rate",
                "duplicate_rate",
                "filter_rate",
                "gc",
                "gc13",
                "gc18",
                "gc21",
                "log_unique_reads",
                "unique_ratio",
                "max_abs_z",
                "mean_abs_z",
                "std_z",
                "n_abs_z_ge_2",
                "n_abs_z_ge_3",
                *MATERNAL_FEATURES,
            ]
        )
    )
    woman, woman_features = _woman_table(data, woman_sources)
    audit = {
        "records": len(data),
        "women": int(data["code"].nunique()),
        "date_min": data["date"].min(),
        "date_max": data["date"].max(),
        "positive_records": {t: int(data[f"y_{t}"].sum()) for t in TARGETS},
        "all_negative_accuracy": {
            t: float(1.0 - data[f"y_{t}"].mean()) for t in TARGETS
        },
        "all_negative_positive_recall": {t: 0.0 for t in TARGETS},
        "positive_women": {
            t: int(data.groupby("code")[f"y_{t}"].max().sum()) for t in TARGETS
        },
        "within_woman_label_changes": {
            t: int((data.groupby("code")[f"y_{t}"].nunique() > 1).sum())
            for t in TARGETS
        },
        "ignored_as_features": [
            "序号/row_id",
            "孕妇代码/code",
            "检测日期/date (only chronological robustness split)",
            "染色体的非整倍体/all labels",
            "胎儿是否健康",
        ],
    }
    return DataBundle(
        record=data,
        woman=woman,
        feature_sets={"record": record_sets, "woman": {"woman_compact": woman_features}},
        audit=audit,
    )


def dataset_for(
    bundle: DataBundle, track: str, feature_set: str, target: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    frame = bundle.record if track == "record" else bundle.woman
    key = f"z_only__{target}" if feature_set == "z_only" else feature_set
    columns = bundle.feature_sets[track][key]
    X = frame[columns].copy()
    y = frame[f"y_{target}"].to_numpy(int)
    groups = frame["code"].astype(str).to_numpy()
    meta_cols = ["code"]
    if track == "record":
        meta_cols = ["row_id", "code", "date", "ga", *[f"y_{t}" for t in TARGETS]]
    else:
        meta_cols = ["code", "first_date", *[f"y_{t}" for t in TARGETS]]
    return X, y, groups, frame[meta_cols].reset_index(drop=True), columns


def group_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    weights = np.array([1.0 / counts[item] for item in groups], dtype=float)
    return weights / weights.mean()


def fit_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    weights = group_weights(groups)
    mass0 = float(weights[y == 0].sum())
    mass1 = float(weights[y == 1].sum())
    if mass0 <= 0 or mass1 <= 0:
        return weights
    total = mass0 + mass1
    weights = weights * np.where(y == 1, total / (2 * mass1), total / (2 * mass0))
    return weights / weights.mean()


def split_count(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    table = pd.DataFrame({"group": groups, "y": y}).groupby("group")["y"].max()
    positives = int(table.sum())
    negatives = int(len(table) - positives)
    return max(2, min(requested, positives, negatives))


def grouped_splits(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    count = split_count(y, groups, n_splits)
    cv = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(y)), y, groups))
    for train, test in splits:
        overlap = set(groups[train]).intersection(groups[test])
        if overlap:
            raise RuntimeError(f"Group leakage detected: {sorted(overlap)[:3]}")
    return splits


def fixed_params(model: str) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "elastic": {"C": 0.2, "l1_ratio": 0.35},
        "lda": {"shrinkage": "auto"},
        "extra_trees": {
            "n_estimators": 250,
            "max_depth": None,
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "max_features": 0.7,
        },
        "random_forest": {
            "n_estimators": 250,
            "max_depth": 5,
            "min_samples_leaf": 4,
            "min_samples_split": 8,
            "max_features": 0.7,
        },
        "histgb": {
            "max_iter": 250,
            "learning_rate": 0.04,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 15,
            "l2_regularization": 3.0,
        },
        "lightgbm": {
            "n_estimators": 300,
            "learning_rate": 0.025,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 3.0,
        },
        "xgboost": {
            "n_estimators": 300,
            "learning_rate": 0.025,
            "max_depth": 3,
            "min_child_weight": 5.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 3.0,
            "gamma": 0.0,
        },
        "rbf_svm": {"C": 1.0, "gamma": "scale"},
    }
    return defaults[model].copy()


def suggest_params(trial: optuna.Trial, model: str) -> dict[str, Any]:
    if model == "elastic":
        return {
            "C": trial.suggest_float("C", 0.005, 20.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model == "lda":
        return {"shrinkage": trial.suggest_float("shrinkage", 0.0, 1.0)}
    if model in {"extra_trees", "random_forest"}:
        return {
            "n_estimators": 250,
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 24),
            "max_features": trial.suggest_float("max_features", 0.25, 1.0),
        }
    if model == "histgb":
        return {
            "max_iter": 300,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 3, 31),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 45),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 30.0, log=True),
        }
    if model == "lightgbm":
        return {
            "n_estimators": 250,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 3, 31),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 55),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
    if model == "xgboost":
        return {
            "n_estimators": 250,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 25.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
    if model == "rbf_svm":
        return {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
    raise KeyError(model)


def make_estimator(model: str, params: dict[str, Any], seed: int) -> BaseEstimator:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    if model == "elastic":
        classifier = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=3000,
            tol=1e-3,
            random_state=seed,
            **params,
        )
        return Pipeline([("impute", imputer), ("scale", StandardScaler()), ("clf", classifier)])
    if model == "lda":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", **params)
        return Pipeline([("impute", imputer), ("scale", StandardScaler()), ("clf", classifier)])
    if model == "extra_trees":
        classifier = ExtraTreesClassifier(random_state=seed, n_jobs=4, **params)
        return Pipeline([("impute", imputer), ("clf", classifier)])
    if model == "random_forest":
        classifier = RandomForestClassifier(random_state=seed, n_jobs=4, **params)
        return Pipeline([("impute", imputer), ("clf", classifier)])
    if model == "histgb":
        classifier = HistGradientBoostingClassifier(random_state=seed, **params)
        return Pipeline([("impute", imputer), ("clf", classifier)])
    if model == "lightgbm":
        classifier = LGBMClassifier(
            objective="binary",
            verbosity=-1,
            random_state=seed,
            n_jobs=4,
            deterministic=True,
            force_col_wise=True,
            **params,
        )
        return Pipeline([("impute", imputer), ("clf", classifier)])
    if model == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed in this environment")
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
            verbosity=0,
            **params,
        )
        return Pipeline([("impute", imputer), ("clf", classifier)])
    if model == "rbf_svm":
        classifier = SVC(kernel="rbf", probability=False, random_state=seed, **params)
        return Pipeline([("impute", imputer), ("scale", StandardScaler()), ("clf", classifier)])
    raise KeyError(model)


def fit_estimator(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> BaseEstimator:
    weights = fit_weights(y, groups)
    try:
        estimator.fit(X, y, clf__sample_weight=weights)
    except (TypeError, ValueError):
        estimator.fit(X, y)
    return estimator


def predict_score(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        score = estimator.predict_proba(X)[:, 1]
    else:
        raw = estimator.decision_function(X)
        score = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
    return np.clip(np.asarray(score, dtype=float), 1e-8, 1 - 1e-8)


def safe_auc(function: Callable[..., float], y: np.ndarray, score: np.ndarray, weights: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(function(y, score, sample_weight=weights))


def select_threshold(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> tuple[float, dict[str, float]]:
    weights = group_weights(groups)
    precision, recall, thresholds = precision_recall_curve(y, score, sample_weight=weights)
    if len(thresholds) == 0:
        return 0.5, {"inner_precision": 0.0, "inner_recall": 0.0, "inner_f1": 0.0}
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(f1, np.nanmax(f1), rtol=1e-10, atol=1e-12))
    if len(best) > 1:
        best = best[np.argmax(recall[:-1][best]) : np.argmax(recall[:-1][best]) + 1]
    index = int(best[0])
    return float(thresholds[index]), {
        "inner_precision": float(precision[index]),
        "inner_recall": float(recall[index]),
        "inner_f1": float(f1[index]),
    }


def metric_row(
    y: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    weights = group_weights(groups)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1], sample_weight=weights).ravel()
    specificity = float(tn / max(tn + fp, 1e-12))
    npv = float(tn / max(tn + fn, 1e-12))
    result = {
        "roc_auc_w": safe_auc(roc_auc_score, y, score, weights),
        "pr_auc_w": safe_auc(average_precision_score, y, score, weights),
        "precision_w": float(precision_score(y, pred, sample_weight=weights, zero_division=0)),
        "recall_w": float(recall_score(y, pred, sample_weight=weights, zero_division=0)),
        "specificity_w": specificity,
        "npv_w": npv,
        "f1_w": float(f1_score(y, pred, sample_weight=weights, zero_division=0)),
        "balanced_accuracy_w": float(balanced_accuracy_score(y, pred, sample_weight=weights)),
        "mcc_w": float(matthews_corrcoef(y, pred, sample_weight=weights)),
        "accuracy_w": float(accuracy_score(y, pred, sample_weight=weights)),
        "brier_w": float(brier_score_loss(y, score, sample_weight=weights)),
        "logloss_w": float(log_loss(y, score, sample_weight=weights, labels=[0, 1])),
        "precision_raw": float(precision_score(y, pred, zero_division=0)),
        "recall_raw": float(recall_score(y, pred, zero_division=0)),
        "f1_raw": float(f1_score(y, pred, zero_division=0)),
        "accuracy_raw": float(accuracy_score(y, pred)),
        "predicted_positive": int(pred.sum()),
        "n": int(len(y)),
        "positive": int(y.sum()),
    }
    return result


def inner_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model: str,
    params: dict[str, Any],
    seed: int,
    n_splits: int = 3,
) -> np.ndarray:
    predictions = np.zeros(len(y), dtype=float)
    for fold, (train, test) in enumerate(grouped_splits(y, groups, n_splits, seed)):
        estimator = make_estimator(model, params, seed + fold + 1)
        fit_estimator(estimator, X.iloc[train], y[train], groups[train])
        predictions[test] = predict_score(estimator, X.iloc[test])
    return predictions


def evaluate_fixed_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model: str,
    params: dict[str, Any],
    seed: int,
    outer_splits: int = 4,
    inner_splits: int = 3,
    return_predictions: bool = False,
) -> tuple[dict[str, float], pd.DataFrame | None]:
    score = np.zeros(len(y), dtype=float)
    pred = np.zeros(len(y), dtype=int)
    threshold_used = np.zeros(len(y), dtype=float)
    fold_used = np.zeros(len(y), dtype=int)
    inner_f1 = np.zeros(len(y), dtype=float)
    for fold, (train, test) in enumerate(grouped_splits(y, groups, outer_splits, seed)):
        inner_score = inner_oof(
            X.iloc[train].reset_index(drop=True),
            y[train],
            groups[train],
            model,
            params,
            seed + 1000 + fold * 17,
            inner_splits,
        )
        threshold, threshold_info = select_threshold(y[train], inner_score, groups[train])
        estimator = make_estimator(model, params, seed + 2000 + fold)
        fit_estimator(estimator, X.iloc[train], y[train], groups[train])
        score[test] = predict_score(estimator, X.iloc[test])
        pred[test] = (score[test] >= threshold).astype(int)
        threshold_used[test] = threshold
        fold_used[test] = fold
        inner_f1[test] = threshold_info["inner_f1"]
    metrics = metric_row(y, score, pred, groups)
    metrics.update(
        {
            "seed": int(seed),
            "mean_threshold": float(np.mean(threshold_used)),
            "sd_threshold": float(np.std(threshold_used, ddof=1)),
            "mean_inner_f1": float(np.mean(inner_f1)),
        }
    )
    predictions = None
    if return_predictions:
        predictions = pd.DataFrame(
            {
                "index": np.arange(len(y)),
                "seed": seed,
                "fold": fold_used,
                "y": y,
                "score": score,
                "threshold": threshold_used,
                "pred": pred,
                "code": groups,
            }
        )
    return metrics, predictions


def tune_on_training(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model: str,
    seed: int,
    trials: int,
    inner_splits: int,
) -> tuple[dict[str, Any], float, pd.DataFrame]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    history: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model)
        score = inner_oof(X, y, groups, model, params, seed + trial.number * 101, inner_splits)
        weights = group_weights(groups)
        ap = safe_auc(average_precision_score, y, score, weights)
        _, info = select_threshold(y, score, groups)
        value = 0.55 * ap + 0.45 * info["inner_f1"]
        history.append(
            {
                "trial": trial.number,
                "objective": value,
                "pr_auc_w": ap,
                **info,
                "params": json.dumps(jsonable(params), ensure_ascii=False, sort_keys=True),
            }
        )
        return float(value)

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max(1, trials), show_progress_bar=False)
    best_full_params = json.loads(
        next(
            item["params"]
            for item in history
            if int(item["trial"]) == int(study.best_trial.number)
        )
    )
    return best_full_params, float(study.best_value), pd.DataFrame(history)


def nested_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model: str,
    seed: int,
    trials: int,
    outer_splits: int = 4,
    inner_splits: int = 3,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    score = np.zeros(len(y), dtype=float)
    pred = np.zeros(len(y), dtype=int)
    threshold_used = np.zeros(len(y), dtype=float)
    fold_used = np.zeros(len(y), dtype=int)
    tuning_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(grouped_splits(y, groups, outer_splits, seed)):
        local_X = X.iloc[train].reset_index(drop=True)
        local_y = y[train]
        local_groups = groups[train]
        best_params, best_value, history = tune_on_training(
            local_X,
            local_y,
            local_groups,
            model,
            seed + fold * 10007,
            trials,
            inner_splits,
        )
        best_inner_score = inner_oof(
            local_X,
            local_y,
            local_groups,
            model,
            best_params,
            seed + 50000 + fold,
            inner_splits,
        )
        threshold, threshold_info = select_threshold(local_y, best_inner_score, local_groups)
        estimator = make_estimator(model, best_params, seed + 60000 + fold)
        fit_estimator(estimator, local_X, local_y, local_groups)
        score[test] = predict_score(estimator, X.iloc[test])
        pred[test] = (score[test] >= threshold).astype(int)
        threshold_used[test] = threshold
        fold_used[test] = fold
        tuning_rows.append(
            {
                "outer_seed": seed,
                "outer_fold": fold,
                "best_objective": best_value,
                "threshold": threshold,
                **threshold_info,
                "best_params": json.dumps(jsonable(best_params), ensure_ascii=False, sort_keys=True),
                "all_trials": history.to_json(orient="records", force_ascii=False),
            }
        )
    metrics = metric_row(y, score, pred, groups)
    metrics.update(
        {
            "seed": int(seed),
            "mean_threshold": float(np.mean(threshold_used)),
            "sd_threshold": float(np.std(threshold_used, ddof=1)),
        }
    )
    predictions = pd.DataFrame(
        {
            "index": np.arange(len(y)),
            "seed": seed,
            "fold": fold_used,
            "y": y,
            "score": score,
            "threshold": threshold_used,
            "pred": pred,
            "code": groups,
        }
    )
    return metrics, predictions, pd.DataFrame(tuning_rows)


def summarize_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metric_cols = [
        "roc_auc_w",
        "pr_auc_w",
        "precision_w",
        "recall_w",
        "specificity_w",
        "f1_w",
        "balanced_accuracy_w",
        "mcc_w",
        "accuracy_w",
        "precision_raw",
        "recall_raw",
        "f1_raw",
        "accuracy_raw",
        "predicted_positive",
        "mean_threshold",
    ]
    pieces: list[pd.DataFrame] = []
    grouped = frame.groupby(keys, dropna=False, sort=False)
    for name, group in grouped:
        name_tuple = name if isinstance(name, tuple) else (name,)
        row = {key: value for key, value in zip(keys, name_tuple)}
        row["seeds"] = int(group["seed"].nunique()) if "seed" in group else len(group)
        for col in metric_cols:
            if col not in group:
                continue
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_sd"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{col}_q025"] = float(group[col].quantile(0.025))
            row[f"{col}_q975"] = float(group[col].quantile(0.975))
        row["selection_score"] = (
            row.get("f1_w_mean", 0.0)
            - 0.25 * row.get("f1_w_sd", 0.0)
            + 0.10 * row.get("pr_auc_w_mean", 0.0)
        )
        pieces.append(pd.DataFrame([row]))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def run_screen(bundle: DataBundle, output: Path, seeds: int) -> None:
    combinations: list[tuple[str, str, str]] = []
    record_models = [
        "elastic",
        "lda",
        "extra_trees",
        "random_forest",
        "histgb",
        "lightgbm",
        "xgboost",
        "rbf_svm",
    ]
    for feature_set in ["engineered", "longitudinal"]:
        combinations.extend(("record", feature_set, model) for model in record_models)
    combinations.append(("record", "z_only", "elastic"))
    for model in ["elastic", "lda", "extra_trees", "lightgbm", "xgboost"]:
        combinations.append(("woman", "woman_compact", model))
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for track, feature_set, model in combinations:
            X, y, groups, _, columns = dataset_for(bundle, track, feature_set, target)
            for seed_index in range(seeds):
                seed = SEED + 997 * seed_index
                metrics, _ = evaluate_fixed_oof(
                    X,
                    y,
                    groups,
                    model,
                    fixed_params(model),
                    seed,
                    outer_splits=4,
                    inner_splits=3,
                )
                rows.append(
                    {
                        "target": target,
                        "track": track,
                        "feature_set": feature_set,
                        "model": model,
                        "n_features": len(columns),
                        **metrics,
                    }
                )
                print(
                    f"SCREEN {target:>3} {track:>6}/{feature_set:<12} {model:<14} "
                    f"seed={seed_index + 1}/{seeds} P={metrics['precision_w']:.3f} "
                    f"R={metrics['recall_w']:.3f} F1={metrics['f1_w']:.3f} AP={metrics['pr_auc_w']:.3f}",
                    flush=True,
                )
    metrics_frame = pd.DataFrame(rows)
    summary = summarize_metrics(metrics_frame, ["target", "track", "feature_set", "model", "n_features"])
    output.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output / "q4_screen_metrics.csv", index=False)
    summary.sort_values(["target", "selection_score"], ascending=[True, False]).to_csv(
        output / "q4_screen_summary.csv", index=False
    )
    write_json(
        output / "q4_screen_manifest.json",
        {
            "seed_base": SEED,
            "seeds": seeds,
            "outer_cv": "StratifiedGroupKFold(4), group=孕妇代码",
            "inner_threshold_cv": "StratifiedGroupKFold(3), group=孕妇代码",
            "threshold_objective": "weighted F1; threshold chosen using inner OOF only",
            "model_count": len(combinations),
            "data_audit": bundle.audit,
        },
    )


def choose_finalists(screen: pd.DataFrame, target: str, maximum: int = 3) -> list[dict[str, str]]:
    table = screen[(screen["target"] == target) & (screen["track"] == "record")].copy()
    table = table.sort_values("selection_score", ascending=False)
    selected: list[dict[str, str]] = []

    def add(row: pd.Series) -> None:
        candidate = {
            "track": str(row["track"]),
            "feature_set": str(row["feature_set"]),
            "model": str(row["model"]),
        }
        if candidate not in selected:
            selected.append(candidate)

    if not table.empty:
        add(table.iloc[0])
    for required in ["lightgbm", "elastic"]:
        local = table[table["model"] == required]
        if not local.empty:
            add(local.iloc[0])
    for _, row in table.iterrows():
        if len(selected) >= maximum:
            break
        add(row)
    return selected[:maximum]


def chronological_test(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
    model: str,
    seed: int,
    trials: int,
) -> dict[str, Any]:
    first_date = meta.groupby("code")["date"].min().sort_values()
    cut = int(math.floor(0.8 * len(first_date)))
    train_groups = set(first_date.index[:cut])
    test_groups = set(first_date.index[cut:])
    train = np.array([item in train_groups for item in groups])
    test = np.array([item in test_groups for item in groups])
    params, tuning_objective, _ = tune_on_training(
        X.loc[train].reset_index(drop=True),
        y[train],
        groups[train],
        model,
        seed,
        trials,
        3,
    )
    inner = inner_oof(
        X.loc[train].reset_index(drop=True),
        y[train],
        groups[train],
        model,
        params,
        seed + 1,
        3,
    )
    threshold, threshold_info = select_threshold(y[train], inner, groups[train])
    estimator = make_estimator(model, params, seed + 2)
    fit_estimator(estimator, X.loc[train], y[train], groups[train])
    score = predict_score(estimator, X.loc[test])
    pred = (score >= threshold).astype(int)
    result = metric_row(y[test], score, pred, groups[test])
    result.update(
        {
            "threshold": threshold,
            "train_women": len(train_groups),
            "test_women": len(test_groups),
            "train_date_max": first_date.iloc[cut - 1],
            "test_date_min": first_date.iloc[cut],
            "tuning_objective": tuning_objective,
            "params": json.dumps(jsonable(params), ensure_ascii=False, sort_keys=True),
            **threshold_info,
        }
    )
    return result


def bootstrap_consensus(
    decisions: pd.DataFrame, target: str, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    codes = decisions["code"].drop_duplicates().to_numpy()
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        sampled = rng.choice(codes, size=len(codes), replace=True)
        pieces: list[pd.DataFrame] = []
        for index, code in enumerate(sampled):
            local = decisions[decisions["code"] == code].copy()
            local["boot_code"] = f"{code}__{index}"
            pieces.append(local)
        sample = pd.concat(pieces, ignore_index=True)
        metrics = metric_row(
            sample["y"].to_numpy(int),
            sample["mean_score"].to_numpy(float),
            sample["consensus_pred"].to_numpy(int),
            sample["boot_code"].to_numpy(str),
        )
        rows.append({"target": target, "replicate": replicate, **metrics})
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for col in [
        "roc_auc_w",
        "pr_auc_w",
        "precision_w",
        "recall_w",
        "specificity_w",
        "f1_w",
        "balanced_accuracy_w",
        "mcc_w",
        "accuracy_w",
    ]:
        values = frame[col].dropna()
        summary_rows.append(
            {
                "target": target,
                "metric": col,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "q025": float(values.quantile(0.025)),
                "median": float(values.median()),
                "q975": float(values.quantile(0.975)),
                "replicates": int(len(values)),
            }
        )
    return frame, pd.DataFrame(summary_rows)


def feature_importance(
    estimator: BaseEstimator, columns: list[str], target: str, model: str
) -> pd.DataFrame:
    classifier = estimator.named_steps["clf"]
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_).reshape(-1))
    else:
        return pd.DataFrame(columns=["target", "model", "feature", "importance"])
    imputer = estimator.named_steps["impute"]
    names = list(columns)
    if getattr(imputer, "indicator_", None) is not None:
        names.extend(f"missing__{columns[i]}" for i in imputer.indicator_.features_)
    length = min(len(names), len(values))
    return pd.DataFrame(
        {
            "target": target,
            "model": model,
            "feature": names[:length],
            "importance": values[:length],
        }
    ).sort_values("importance", ascending=False)


def run_final(
    bundle: DataBundle,
    output: Path,
    nested_seeds: int,
    trials: int,
    stability_seeds: int,
    ablation_seeds: int,
    bootstrap_replicates: int,
) -> None:
    screen_path = output / "q4_screen_summary.csv"
    if not screen_path.exists():
        raise FileNotFoundError(f"Run --phase screen first: {screen_path}")
    screen = pd.read_csv(screen_path)
    nested_metric_rows: list[dict[str, Any]] = []
    nested_prediction_rows: list[pd.DataFrame] = []
    nested_tuning_rows: list[pd.DataFrame] = []
    finalists_payload: dict[str, Any] = {}

    for target in TARGETS:
        finalists = choose_finalists(screen, target, maximum=3)
        finalists_payload[target] = finalists
        for candidate in finalists:
            track = candidate["track"]
            feature_set = candidate["feature_set"]
            model = candidate["model"]
            candidate_id = f"{track}|{feature_set}|{model}"
            X, y, groups, _, columns = dataset_for(bundle, track, feature_set, target)
            for seed_index in range(nested_seeds):
                seed = SEED + 100003 * seed_index
                metrics, predictions, tuning = nested_oof(
                    X,
                    y,
                    groups,
                    model,
                    seed,
                    trials,
                    outer_splits=4,
                    inner_splits=3,
                )
                nested_metric_rows.append(
                    {
                        "target": target,
                        "candidate_id": candidate_id,
                        "track": track,
                        "feature_set": feature_set,
                        "model": model,
                        "n_features": len(columns),
                        **metrics,
                    }
                )
                predictions.insert(0, "candidate_id", candidate_id)
                predictions.insert(0, "target", target)
                nested_prediction_rows.append(predictions)
                tuning.insert(0, "candidate_id", candidate_id)
                tuning.insert(0, "target", target)
                nested_tuning_rows.append(tuning)
                print(
                    f"NESTED {target:>3} {candidate_id:<42} seed={seed_index + 1}/{nested_seeds} "
                    f"P={metrics['precision_w']:.3f} R={metrics['recall_w']:.3f} "
                    f"F1={metrics['f1_w']:.3f} AP={metrics['pr_auc_w']:.3f}",
                    flush=True,
                )

    nested_metrics = pd.DataFrame(nested_metric_rows)
    nested_predictions = pd.concat(nested_prediction_rows, ignore_index=True)
    nested_tuning = pd.concat(nested_tuning_rows, ignore_index=True)
    nested_summary = summarize_metrics(
        nested_metrics,
        ["target", "candidate_id", "track", "feature_set", "model", "n_features"],
    )
    nested_summary = nested_summary.sort_values(
        ["target", "selection_score"], ascending=[True, False]
    )
    winners = nested_summary.groupby("target", sort=False).first().reset_index()

    output.mkdir(parents=True, exist_ok=True)
    nested_metrics.to_csv(output / "q4_nested_metrics.csv", index=False)
    nested_predictions.to_csv(output / "q4_nested_predictions.csv", index=False)
    nested_tuning.to_csv(output / "q4_nested_tuning.csv", index=False)
    nested_summary.to_csv(output / "q4_nested_summary.csv", index=False)
    winners.to_csv(output / "q4_final_selection.csv", index=False)
    write_json(output / "q4_finalists.json", finalists_payload)

    stability_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    consensus_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[pd.DataFrame] = []
    bootstrap_summary_rows: list[pd.DataFrame] = []
    chronology_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    operational: dict[str, Any] = {}

    for _, winner in winners.iterrows():
        target = str(winner["target"])
        track = str(winner["track"])
        feature_set = str(winner["feature_set"])
        model = str(winner["model"])
        if track != "record":
            raise RuntimeError("Primary winner must be record-level; finalist filter failed")
        X, y, groups, meta, columns = dataset_for(bundle, track, feature_set, target)
        tuned_params, tuned_value, full_tuning = tune_on_training(
            X,
            y,
            groups,
            model,
            SEED + 700000,
            max(3, trials * 2),
            4,
        )
        full_inner_score = inner_oof(
            X, y, groups, model, tuned_params, SEED + 710000, 5
        )
        operational_threshold, threshold_info = select_threshold(
            y, full_inner_score, groups
        )
        estimator = make_estimator(model, tuned_params, SEED + 720000)
        fit_estimator(estimator, X, y, groups)
        importance_rows.append(feature_importance(estimator, columns, target, model))
        operational[target] = {
            "track": track,
            "feature_set": feature_set,
            "model": model,
            "n_features": len(columns),
            "params": tuned_params,
            "tuning_objective": tuned_value,
            "threshold": operational_threshold,
            "threshold_inner_metrics": threshold_info,
            "full_tuning_trials": json.loads(full_tuning.to_json(orient="records")),
        }
        stability_params = fixed_params(model)
        operational[target]["stability_parameter_source"] = (
            "predeclared fixed anchor; Optuna parameters are evaluated only in nested CV"
        )
        operational[target]["stability_params"] = stability_params

        score_sum = np.zeros(len(y), dtype=float)
        vote_sum = np.zeros(len(y), dtype=float)
        threshold_sum = np.zeros(len(y), dtype=float)
        for seed_index in range(stability_seeds):
            seed = SEED + 7919 * seed_index
            metrics, predictions = evaluate_fixed_oof(
                X,
                y,
                groups,
                model,
                stability_params,
                seed,
                outer_splits=4,
                inner_splits=3,
                return_predictions=True,
            )
            stability_rows.append(
                {
                    "target": target,
                    "track": track,
                    "feature_set": feature_set,
                    "model": model,
                    **metrics,
                }
            )
            score_sum += predictions["score"].to_numpy(float)
            vote_sum += predictions["pred"].to_numpy(float)
            threshold_sum += predictions["threshold"].to_numpy(float)
            if (seed_index + 1) % 10 == 0 or seed_index + 1 == stability_seeds:
                print(
                    f"STABILITY {target} {seed_index + 1}/{stability_seeds}",
                    flush=True,
                )
        locked = nested_predictions[
            (nested_predictions["target"] == target)
            & (nested_predictions["candidate_id"] == str(winner["candidate_id"]))
        ]
        locked_consensus = locked.groupby("index", sort=True).agg(
            y=("y", "first"),
            mean_score=("score", "mean"),
            mean_threshold=("threshold", "mean"),
            positive_vote_rate=("pred", "mean"),
        )
        decisions = meta.copy()
        decisions.insert(0, "target", target)
        decisions["y"] = locked_consensus["y"].to_numpy(int)
        decisions["mean_score"] = locked_consensus["mean_score"].to_numpy(float)
        decisions["mean_threshold"] = locked_consensus["mean_threshold"].to_numpy(float)
        decisions["positive_vote_rate"] = locked_consensus[
            "positive_vote_rate"
        ].to_numpy(float)
        decisions["consensus_pred"] = (decisions["positive_vote_rate"] >= 0.5).astype(int)
        decisions["stability_mean_score"] = score_sum / stability_seeds
        decisions["stability_vote_rate"] = vote_sum / stability_seeds
        decisions["stability_consensus_pred"] = (
            decisions["stability_vote_rate"] >= 0.5
        ).astype(int)
        consensus_rows.append(decisions)

        boot, boot_summary = bootstrap_consensus(
            decisions,
            target,
            bootstrap_replicates,
            SEED + 800000 + TARGETS.index(target),
        )
        bootstrap_rows.append(boot)
        bootstrap_summary_rows.append(boot_summary)

        chronology = chronological_test(
            X,
            y,
            groups,
            meta,
            model,
            SEED + 900000 + TARGETS.index(target),
            max(2, trials),
        )
        chronology_rows.append(
            {
                "target": target,
                "track": track,
                "feature_set": feature_set,
                "model": model,
                **chronology,
            }
        )

        variants = ["z_only", "core", "engineered", "longitudinal", "no_maternal", "no_qc"]
        for variant in variants:
            local_X, local_y, local_groups, _, local_columns = dataset_for(
                bundle, "record", variant, target
            )
            for seed_index in range(ablation_seeds):
                seed = SEED + 12347 * seed_index
                metrics, _ = evaluate_fixed_oof(
                    local_X,
                    local_y,
                    local_groups,
                    model,
                    stability_params,
                    seed,
                    outer_splits=4,
                    inner_splits=3,
                )
                ablation_rows.append(
                    {
                        "target": target,
                        "variant": variant,
                        "model": model,
                        "n_features": len(local_columns),
                        **metrics,
                    }
                )
            print(f"ABLATION {target} {variant}", flush=True)

    stability = pd.DataFrame(stability_rows)
    stability_summary = summarize_metrics(
        stability, ["target", "track", "feature_set", "model"]
    )
    ablation = pd.DataFrame(ablation_rows)
    ablation_summary = summarize_metrics(ablation, ["target", "variant", "model", "n_features"])
    decisions_all = pd.concat(consensus_rows, ignore_index=True)
    bootstrap_all = pd.concat(bootstrap_rows, ignore_index=True)
    bootstrap_summary_all = pd.concat(bootstrap_summary_rows, ignore_index=True)
    chronology = pd.DataFrame(chronology_rows)
    importance = pd.concat(importance_rows, ignore_index=True)

    women = (
        decisions_all.groupby(["target", "code"], as_index=False)
        .agg(
            true_ever=("y", "max"),
            max_mean_score=("mean_score", "max"),
            max_vote_rate=("positive_vote_rate", "max"),
            any_consensus_positive=("consensus_pred", "max"),
            positive_records=("consensus_pred", "sum"),
            records=("consensus_pred", "size"),
        )
    )
    wide = women.pivot(index="code", columns="target")
    wide.columns = [f"{metric}__{target}" for metric, target in wide.columns]
    wide = wide.reset_index()

    record_index = (
        decisions_all[["row_id", "code", "date", "ga"]]
        .drop_duplicates("row_id")
        .set_index("row_id")
    )
    for value_col, prefix in [
        ("mean_score", "score"),
        ("positive_vote_rate", "vote_rate"),
        ("consensus_pred", "call"),
        ("y", "true"),
    ]:
        pivot = decisions_all.pivot(index="row_id", columns="target", values=value_col)
        pivot.columns = [f"{prefix}__{target}" for target in pivot.columns]
        record_index = record_index.join(pivot, how="left")
    record_index["call__ANY_reconciled"] = record_index[
        ["call__ANY", "call__T13", "call__T18", "call__T21"]
    ].max(axis=1)

    def predicted_label(row: pd.Series) -> str:
        called = [target for target in SUBTYPES if int(row[f"call__{target}"]) == 1]
        if called:
            return "+".join(called)
        if int(row["call__ANY"]) == 1:
            return "异常待分型"
        return "正常"

    record_index["predicted_label"] = record_index.apply(predicted_label, axis=1)
    record_answers = record_index.reset_index()

    stability.to_csv(output / "q4_seed_stability_metrics.csv", index=False)
    stability_summary.to_csv(output / "q4_seed_stability_summary.csv", index=False)
    ablation.to_csv(output / "q4_ablation_metrics.csv", index=False)
    ablation_summary.to_csv(output / "q4_ablation_summary.csv", index=False)
    decisions_all.to_csv(output / "q4_record_decisions_oof.csv", index=False)
    record_answers.to_csv(output / "q4_final_record_answers.csv", index=False)
    wide.to_csv(output / "q4_woman_decisions_oof.csv", index=False)
    bootstrap_all.to_csv(output / "q4_cluster_bootstrap_500.csv", index=False)
    bootstrap_summary_all.to_csv(output / "q4_cluster_bootstrap_summary.csv", index=False)
    chronology.to_csv(output / "q4_chronological_holdout.csv", index=False)
    importance.to_csv(output / "q4_feature_importance.csv", index=False)
    write_json(output / "q4_operational_models.json", operational)

    manifest = {
        "completed_at_unix": time.time(),
        "seed_base": SEED,
        "nested_seeds": nested_seeds,
        "optuna_trials_per_outer_fold": trials,
        "stability_seeds": stability_seeds,
        "ablation_seeds": ablation_seeds,
        "cluster_bootstrap_replicates": bootstrap_replicates,
        "primary_metrics": "woman-balanced record-level precision/recall/F1",
        "outer_validation": "StratifiedGroupKFold by pregnant woman",
        "hyperparameter_tuning": "Optuna only inside each outer training fold",
        "threshold_selection": "F1 threshold only from inner OOF predictions",
        "record_answer_source": "winner's strictly nested OOF predictions averaged across nested seeds",
        "stability_parameter_source": "predeclared fixed anchor, not full-data Optuna tuning",
        "chronological_robustness": "last 20% of women by first collection date held out",
        "leakage_assertions": {
            "group_overlap": 0,
            "labels_in_features": False,
            "row_id_or_code_in_features": False,
            "future_measurements_in_record_longitudinal_features": False,
            "test_fold_used_for_threshold": False,
            "test_fold_used_for_optuna": False,
        },
        "data_audit": bundle.audit,
    }
    write_json(output / "q4_rebuild_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe Q4 model rebuild")
    parser.add_argument("--data", type=Path, default=Path("附件.xlsx"))
    parser.add_argument(
        "--output", type=Path, default=Path("nipt_solution/outputs/q4_rebuild")
    )
    parser.add_argument("--phase", choices=["screen", "final", "all"], default="all")
    parser.add_argument("--screen-seeds", type=int, default=2)
    parser.add_argument("--nested-seeds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--stability-seeds", type=int, default=100)
    parser.add_argument("--ablation-seeds", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    started = time.time()
    bundle = load_bundle(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "q4_data_audit.json", bundle.audit)
    if args.phase in {"screen", "all"}:
        run_screen(bundle, args.output, args.screen_seeds)
    if args.phase in {"final", "all"}:
        run_final(
            bundle,
            args.output,
            args.nested_seeds,
            args.trials,
            args.stability_seeds,
            args.ablation_seeds,
            args.bootstrap,
        )
    write_json(
        args.output / "q4_runtime.json",
        {
            "phase": args.phase,
            "seconds": time.time() - started,
            "input_sha256": sha256_file(args.data),
        },
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from nipt_core import (
    SEED,
    classify_visit_states,
    construct_intervals,
    per_group_row_weights,
    prepare_data,
    weighted_mean,
    write_json,
)


def build_discrete_data(intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in intervals.iterrows():
        for time_point in [12, 14, 16, 18, 20]:
            if row["right"] <= time_point:
                label = 1
            elif row["left"] >= time_point:
                label = 0
            else:
                continue
            rows.append(
                {
                    "code": row["code"],
                    "time": time_point,
                    "event_by_time": label,
                    "first_bmi": row["first_bmi"],
                    "age": row["age"],
                    "height": row["height"],
                    "gravidity": row["gravidity"],
                    "parity": row["parity"],
                    "ivf_iui": row["ivf_iui"],
                }
            )
    return pd.DataFrame(rows)


def fit_weights(groups: np.ndarray, y: np.ndarray) -> np.ndarray:
    weights = per_group_row_weights(groups)
    pos = weights[y == 1].sum()
    neg = weights[y == 0].sum()
    weights *= np.where(
        y == 1,
        (pos + neg) / (2 * pos),
        (pos + neg) / (2 * neg),
    )
    return weights / weights.mean()


def metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p, sample_weight=weights)),
        "pr_auc": float(
            average_precision_score(y, p, sample_weight=weights)
        ),
        "brier": weighted_mean(np.square(y - p), weights),
    }


def one_seed(data: pd.DataFrame, seed_index: int, trees: int) -> list[dict]:
    seed = SEED + 80000 + seed_index
    y = data["event_by_time"].to_numpy(int)
    groups = data["code"].to_numpy(str)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed
    )
    specs = {
        "logistic_bmi": (
            "logistic",
            ["time", "first_bmi"],
        ),
        "logistic_full": (
            "logistic",
            [
                "time",
                "first_bmi",
                "age",
                "height",
                "gravidity",
                "parity",
                "ivf_iui",
            ],
        ),
        "rf_bmi": (
            "rf",
            ["time", "first_bmi"],
        ),
        "rf_full": (
            "rf",
            [
                "time",
                "first_bmi",
                "age",
                "height",
                "gravidity",
                "parity",
                "ivf_iui",
            ],
        ),
    }
    predictions = {name: np.full(len(data), np.nan) for name in specs}
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(data)), y, groups)
    ):
        train_weights = fit_weights(groups[train_index], y[train_index])
        for name, (model_type, columns) in specs.items():
            x_train = data.iloc[train_index][columns].to_numpy(float)
            x_test = data.iloc[test_index][columns].to_numpy(float)
            imputer = SimpleImputer(strategy="median").fit(x_train)
            x_train = imputer.transform(x_train)
            x_test = imputer.transform(x_test)
            if model_type == "logistic":
                scaler = StandardScaler().fit(x_train)
                model = LogisticRegression(
                    C=0.3,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                )
                model.fit(
                    scaler.transform(x_train),
                    y[train_index],
                    sample_weight=train_weights,
                )
                probability = model.predict_proba(
                    scaler.transform(x_test)
                )[:, 1]
            else:
                model = RandomForestClassifier(
                    n_estimators=trees,
                    max_depth=5,
                    min_samples_leaf=8,
                    max_features=0.8,
                    n_jobs=1,
                    random_state=seed + fold,
                )
                model.fit(
                    x_train,
                    y[train_index],
                    sample_weight=train_weights,
                )
                probability = model.predict_proba(x_test)[:, 1]
            predictions[name][test_index] = probability
    eval_weights = per_group_row_weights(groups)
    return [
        {
            "seed_index": seed_index,
            "seed": seed,
            "model": name,
            **metrics(y, probability, eval_weights),
        }
        for name, probability in predictions.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("附件.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nipt_solution/outputs/q3_ml"),
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prepared = prepare_data(args.data)
    intervals = construct_intervals(
        classify_visit_states(
            prepared.male_visits, prepared.sigma_tech, mode="hard"
        ),
        prepared.male_baseline,
    )
    discrete = build_discrete_data(intervals)
    discrete.to_csv(args.output / "q3_discrete_time_data.csv", index=False)
    nested = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one_seed)(discrete, seed_index, args.trees)
        for seed_index in range(args.seeds)
    )
    results = pd.DataFrame([row for batch in nested for row in batch])
    results.to_csv(args.output / "q3_ml_seed_metrics.csv", index=False)
    summary = (
        results.groupby("model")[["roc_auc", "pr_auc", "brier"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(args.output / "q3_ml_summary.csv", index=False)
    comparisons = []
    for model_type in ["logistic", "rf"]:
        base = results.loc[
            results["model"] == f"{model_type}_bmi"
        ].set_index("seed_index")
        full = results.loc[
            results["model"] == f"{model_type}_full"
        ].set_index("seed_index")
        for metric in ["roc_auc", "pr_auc", "brier"]:
            if metric == "brier":
                improvement = base[metric] - full[metric]
            else:
                improvement = full[metric] - base[metric]
            comparisons.append(
                {
                    "model_type": model_type,
                    "metric": metric,
                    "improvement_mean": float(improvement.mean()),
                    "q025": float(improvement.quantile(0.025)),
                    "q975": float(improvement.quantile(0.975)),
                    "positive_fraction": float(np.mean(improvement > 0)),
                }
            )
    pd.DataFrame(comparisons).to_csv(
        args.output / "q3_ml_increment.csv", index=False
    )
    manifest = {
        "seeds": args.seeds,
        "trees": args.trees,
        "rows": len(discrete),
        "women": int(discrete["code"].nunique()),
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output / "q3_ml_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

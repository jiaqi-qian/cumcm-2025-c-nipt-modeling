#!/usr/bin/env python3
"""Fair 100-seed validation of the compact-core Q4 challengers."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_q4_rebuild as q4


CHALLENGERS = {
    "ANY": {"feature_set": "core", "model": "elastic"},
    "T13": {"feature_set": "core", "model": "extra_trees"},
    "T18": {"feature_set": "core", "model": "elastic"},
    "T21": {"feature_set": "core", "model": "lda"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("附件.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nipt_solution/outputs/q4_core_challengers"),
    )
    parser.add_argument(
        "--incumbent-metrics",
        type=Path,
        default=Path(
            "nipt_solution/outputs/q4_rebuild/q4_seed_stability_metrics.csv"
        ),
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--chronology-trials", type=int, default=20)
    return parser.parse_args()


def bootstrap_consensus_fast(
    decisions: pd.DataFrame, target: str, replicates: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    local = decisions.reset_index(drop=True)
    codes = local["code"].drop_duplicates().to_numpy()
    local_codes = local["code"].to_numpy()
    indices = [np.flatnonzero(local_codes == code) for code in codes]
    position = {code: index for index, code in enumerate(codes)}
    y_all = local["y"].to_numpy(int)
    score_all = local["mean_score"].to_numpy(float)
    pred_all = local["consensus_pred"].to_numpy(int)
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        sampled_codes = rng.choice(codes, size=len(codes), replace=True)
        chosen = [indices[position[code]] for code in sampled_codes]
        sample_index = np.concatenate(chosen)
        boot_groups = np.repeat(
            np.arange(len(chosen), dtype=int), [len(item) for item in chosen]
        )
        rows.append(
            {
                "target": target,
                "replicate": replicate,
                **q4.metric_row(
                    y_all[sample_index],
                    score_all[sample_index],
                    pred_all[sample_index],
                    boot_groups,
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "roc_auc_w",
        "pr_auc_w",
        "precision_w",
        "recall_w",
        "specificity_w",
        "f1_w",
        "balanced_accuracy_w",
        "mcc_w",
        "accuracy_w",
    ]
    rows: list[dict[str, Any]] = []
    for target, local in frame.groupby("target", sort=False):
        for metric in metrics:
            values = local[metric].dropna()
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "replicates": int(len(values)),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "q025": float(values.quantile(0.025)),
                    "median": float(values.median()),
                    "q975": float(values.quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def paired_comparison(
    challenger: pd.DataFrame, incumbent_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    incumbent = pd.read_csv(incumbent_path)
    metric_names = [
        "precision_w",
        "recall_w",
        "f1_w",
        "pr_auc_w",
        "roc_auc_w",
        "specificity_w",
        "accuracy_w",
    ]
    keep = ["target", "seed", *metric_names]
    left = challenger[keep].copy()
    right = incumbent[keep].copy()
    paired = left.merge(
        right,
        on=["target", "seed"],
        suffixes=("_challenger", "_incumbent"),
        validate="one_to_one",
    )
    for metric in metric_names:
        paired[f"delta_{metric}"] = (
            paired[f"{metric}_challenger"] - paired[f"{metric}_incumbent"]
        )
    rows: list[dict[str, Any]] = []
    for target, local in paired.groupby("target", sort=False):
        row: dict[str, Any] = {"target": target, "paired_seeds": len(local)}
        for metric in metric_names:
            values = local[f"delta_{metric}"]
            row[f"delta_{metric}_mean"] = float(values.mean())
            row[f"delta_{metric}_sd"] = float(values.std(ddof=1))
            row[f"delta_{metric}_q025"] = float(values.quantile(0.025))
            row[f"delta_{metric}_q975"] = float(values.quantile(0.975))
            row[f"challenger_win_rate_{metric}"] = float((values > 0).mean())
        rows.append(row)
    return paired, pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = q4.load_bundle(args.data)
    seed_rows: list[dict[str, Any]] = []
    decision_parts: list[pd.DataFrame] = []
    bootstrap_parts: list[pd.DataFrame] = []
    chronology_rows: list[dict[str, Any]] = []
    operational: dict[str, Any] = {}

    for target, specification in CHALLENGERS.items():
        feature_set = specification["feature_set"]
        model = specification["model"]
        X, y, groups, meta, columns = q4.dataset_for(
            bundle, "record", feature_set, target
        )
        params = q4.fixed_params(model)
        score_sum = np.zeros(len(y), dtype=float)
        vote_sum = np.zeros(len(y), dtype=float)
        threshold_sum = np.zeros(len(y), dtype=float)
        for seed_index in range(args.seeds):
            seed = q4.SEED + 7919 * seed_index
            metrics, predictions = q4.evaluate_fixed_oof(
                X,
                y,
                groups,
                model,
                params,
                seed,
                outer_splits=4,
                inner_splits=3,
                return_predictions=True,
            )
            seed_rows.append(
                {
                    "target": target,
                    "track": "record",
                    "feature_set": feature_set,
                    "model": model,
                    "n_features": len(columns),
                    **metrics,
                }
            )
            score_sum += predictions["score"].to_numpy(float)
            vote_sum += predictions["pred"].to_numpy(float)
            threshold_sum += predictions["threshold"].to_numpy(float)
            if (seed_index + 1) % 10 == 0 or seed_index + 1 == args.seeds:
                print(
                    f"CORE CHALLENGER {target} {seed_index + 1}/{args.seeds}",
                    flush=True,
                )

        decisions = meta.copy()
        decisions.insert(0, "target", target)
        decisions["y"] = y
        decisions["mean_score"] = score_sum / args.seeds
        decisions["mean_threshold"] = threshold_sum / args.seeds
        decisions["positive_vote_rate"] = vote_sum / args.seeds
        decisions["consensus_pred"] = (
            decisions["positive_vote_rate"] >= 0.5
        ).astype(int)
        decision_parts.append(decisions)
        bootstrap_parts.append(
            bootstrap_consensus_fast(
                decisions,
                target,
                args.bootstrap,
                q4.SEED + 840000 + q4.TARGETS.index(target),
            )
        )

        chronology = q4.chronological_test(
            X,
            y,
            groups,
            meta,
            model,
            q4.SEED + 940000 + q4.TARGETS.index(target),
            args.chronology_trials,
        )
        chronology_rows.append(
            {
                "target": target,
                "track": "record",
                "feature_set": feature_set,
                "model": model,
                **chronology,
            }
        )

        full_inner = q4.inner_oof(
            X, y, groups, model, params, q4.SEED + 950000, 5
        )
        threshold, threshold_info = q4.select_threshold(y, full_inner, groups)
        operational[target] = {
            "track": "record",
            "feature_set": feature_set,
            "model": model,
            "n_features": len(columns),
            "features": columns,
            "fixed_params": params,
            "operational_threshold": threshold,
            "threshold_source": "5-fold grouped OOF on full development data",
            "threshold_inner_metrics": threshold_info,
        }

    seed_metrics = pd.DataFrame(seed_rows)
    summary = q4.summarize_metrics(
        seed_metrics,
        ["target", "track", "feature_set", "model", "n_features"],
    )
    decisions_all = pd.concat(decision_parts, ignore_index=True)
    bootstrap_all = pd.concat(bootstrap_parts, ignore_index=True)
    bootstrap_table = bootstrap_summary(bootstrap_all)
    chronology = pd.DataFrame(chronology_rows)
    paired, paired_summary = paired_comparison(
        seed_metrics, args.incumbent_metrics
    )

    seed_metrics.to_csv(args.output / "q4_core_seed_metrics.csv", index=False)
    summary.to_csv(args.output / "q4_core_seed_summary.csv", index=False)
    decisions_all.to_csv(args.output / "q4_core_consensus_predictions.csv", index=False)
    bootstrap_all.to_csv(args.output / "q4_core_bootstrap_500.csv", index=False)
    bootstrap_table.to_csv(args.output / "q4_core_bootstrap_summary.csv", index=False)
    chronology.to_csv(args.output / "q4_core_chronological.csv", index=False)
    paired.to_csv(args.output / "q4_core_paired_seed_deltas.csv", index=False)
    paired_summary.to_csv(
        args.output / "q4_core_paired_seed_summary.csv", index=False
    )
    q4.write_json(args.output / "q4_core_operational_models.json", operational)
    q4.write_json(
        args.output / "q4_core_manifest.json",
        {
            "input_sha256": q4.sha256_file(args.data),
            "challengers": CHALLENGERS,
            "seeds": args.seeds,
            "outer_cv": "StratifiedGroupKFold(4), group=pregnant woman",
            "inner_cv": "StratifiedGroupKFold(3), group=pregnant woman",
            "threshold_source": "inner-OOF within each outer training fold",
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_unit": "pregnant woman",
            "chronological_holdout": "earliest 80% women train, latest 20% test",
            "fixed_parameter_source": "predeclared anchors from q4_rebuild",
            "leakage_assertions": {
                "woman_overlap_between_train_test": 0,
                "outer_test_used_for_preprocessing": False,
                "outer_test_used_for_threshold": False,
                "labels_or_health_outcome_used_as_features": False,
            },
            "runtime_seconds": time.time() - started,
        },
    )
    print(f"Wrote core-challenger results to {args.output}", flush=True)


if __name__ == "__main__":
    main()

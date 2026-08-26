#!/usr/bin/env python3
"""Robustness audit for the higher-F1 T21 elastic-net alternative."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from run_q4_rebuild import (
    SEED,
    bootstrap_consensus,
    chronological_test,
    dataset_for,
    evaluate_fixed_oof,
    fixed_params,
    load_bundle,
    summarize_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("附件.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--stability-seeds", type=int, default=100)
    parser.add_argument("--ablation-seeds", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--trials", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    bundle = load_bundle(args.data)
    target = "T21"
    model = "elastic"
    feature_set = "engineered"
    params = fixed_params(model)
    X, y, groups, meta, columns = dataset_for(bundle, "record", feature_set, target)

    rows: list[dict[str, object]] = []
    score_sum = np.zeros(len(y), dtype=float)
    vote_sum = np.zeros(len(y), dtype=float)
    threshold_sum = np.zeros(len(y), dtype=float)
    for seed_index in range(args.stability_seeds):
        seed = SEED + 7919 * seed_index
        metrics, predictions = evaluate_fixed_oof(
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
        assert predictions is not None
        rows.append(
            {
                "target": target,
                "track": "record",
                "feature_set": feature_set,
                "model": model,
                "n_features": len(columns),
                **metrics,
            }
        )
        order = predictions["index"].to_numpy(int)
        score_sum[order] += predictions["score"].to_numpy(float)
        vote_sum[order] += predictions["pred"].to_numpy(float)
        threshold_sum[order] += predictions["threshold"].to_numpy(float)
        if (seed_index + 1) % 10 == 0:
            print(f"T21 ELASTIC STABILITY {seed_index + 1}/{args.stability_seeds}", flush=True)

    metrics_frame = pd.DataFrame(rows)
    summary = summarize_metrics(
        metrics_frame,
        ["target", "track", "feature_set", "model", "n_features"],
    )
    decisions = meta.copy()
    decisions.insert(0, "target", target)
    decisions["y"] = y
    decisions["mean_score"] = score_sum / args.stability_seeds
    decisions["mean_threshold"] = threshold_sum / args.stability_seeds
    decisions["positive_vote_rate"] = vote_sum / args.stability_seeds
    decisions["consensus_pred"] = (decisions["positive_vote_rate"] >= 0.5).astype(int)
    boot, boot_summary = bootstrap_consensus(
        decisions,
        target,
        args.bootstrap,
        SEED + 880021,
    )
    chronology = pd.DataFrame(
        [
            {
                "target": target,
                "track": "record",
                "feature_set": feature_set,
                "model": model,
                **chronological_test(
                    X,
                    y,
                    groups,
                    meta,
                    model,
                    SEED + 990021,
                    args.trials,
                ),
            }
        ]
    )

    ablation_rows: list[dict[str, object]] = []
    for variant in ["z_only", "core", "engineered", "longitudinal", "no_maternal", "no_qc"]:
        local_X, local_y, local_groups, _, local_columns = dataset_for(
            bundle, "record", variant, target
        )
        for seed_index in range(args.ablation_seeds):
            seed = SEED + 12347 * seed_index
            metrics, _ = evaluate_fixed_oof(
                local_X,
                local_y,
                local_groups,
                model,
                params,
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
        print(f"T21 ELASTIC ABLATION {variant}", flush=True)
    ablation = pd.DataFrame(ablation_rows)
    ablation_summary = summarize_metrics(
        ablation, ["target", "variant", "model", "n_features"]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(args.output / "q4_t21_elastic_stability_metrics.csv", index=False)
    summary.to_csv(args.output / "q4_t21_elastic_stability_summary.csv", index=False)
    decisions.to_csv(args.output / "q4_t21_elastic_decisions_oof.csv", index=False)
    boot.to_csv(args.output / "q4_t21_elastic_bootstrap_500.csv", index=False)
    boot_summary.to_csv(args.output / "q4_t21_elastic_bootstrap_summary.csv", index=False)
    chronology.to_csv(args.output / "q4_t21_elastic_chronological.csv", index=False)
    ablation.to_csv(args.output / "q4_t21_elastic_ablation_metrics.csv", index=False)
    ablation_summary.to_csv(
        args.output / "q4_t21_elastic_ablation_summary.csv", index=False
    )


if __name__ == "__main__":
    main()

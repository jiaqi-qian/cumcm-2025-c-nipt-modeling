#!/usr/bin/env python3
"""Build an apples-labelled comparison of the official-style Q4 rerun."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


TARGETS = ["ANY", "T13", "T18", "T21"]
OFFICIAL_VARIANTS = ["unweighted", "group_class_weighted"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("nipt_solution/outputs/q4_official_style"),
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("nipt_solution/outputs/q4_rebuild"),
    )
    return parser.parse_args()


def group_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    weights = np.array([1.0 / counts[item] for item in groups], dtype=float)
    return weights / weights.mean()


def qsummary(values: pd.Series, prefix: str) -> dict[str, float]:
    values = values.dropna().astype(float)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_q025": float(values.quantile(0.025)),
        f"{prefix}_q975": float(values.quantile(0.975)),
    }


def rank_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (target, variant, seed), frame in predictions.groupby(
        ["target", "variant", "seed"], sort=False
    ):
        for scope, mask in [
            ("overall", np.ones(len(frame), dtype=bool)),
            ("qc_pass", frame["qc_pass"].to_numpy(int).astype(bool)),
        ]:
            local = frame.loc[mask]
            y = local["y"].to_numpy(int)
            if len(np.unique(y)) < 2:
                continue
            weights = group_weights(local["code"].astype(str).to_numpy())
            score = local["score"].to_numpy(float)
            rows.append(
                {
                    "target": target,
                    "variant": variant,
                    "seed": seed,
                    "scope": scope,
                    "pr_auc_w": average_precision_score(
                        y, score, sample_weight=weights
                    ),
                    "roc_auc_w": roc_auc_score(y, score, sample_weight=weights),
                }
            )
    per_seed = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for keys, frame in per_seed.groupby(["target", "variant", "scope"], sort=False):
        row: dict[str, object] = dict(zip(["target", "variant", "scope"], keys))
        row["seeds"] = int(frame["seed"].nunique())
        row.update(qsummary(frame["pr_auc_w"], "pr_auc_w"))
        row.update(qsummary(frame["roc_auc_w"], "roc_auc_w"))
        summary_rows.append(row)
    return per_seed, pd.DataFrame(summary_rows)


def get_one(frame: pd.DataFrame, **filters: object) -> pd.Series | None:
    local = frame
    for column, value in filters.items():
        local = local[local[column] == value]
    if local.empty:
        return None
    return local.iloc[0]


def value(row: pd.Series | None, key: str) -> float:
    if row is None or key not in row or pd.isna(row[key]):
        return float("nan")
    return float(row[key])


def build_binary_comparison(
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    bootstrap: pd.DataFrame,
    chronological: pd.DataFrame,
    previous: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    previous_by_target = previous.set_index("target")
    for target in TARGETS:
        if target in previous_by_target.index:
            old = previous_by_target.loc[target]
            rows.append(
                {
                    "target": target,
                    "approach": f"previous_{old['model']}_{old['feature_set']}",
                    "population": "all_records",
                    "decision_rule": "training-only tuned binary threshold",
                    "precision_w": old["stability100_precision_w_mean"],
                    "recall_w": old["stability100_recall_w_mean"],
                    "specificity_w": np.nan,
                    "f1_w": old["stability100_f1_w_mean"],
                    "pr_auc_w": old["stability100_pr_auc_w_mean"],
                    "roc_auc_w": old["stability100_roc_auc_w_mean"],
                    "bootstrap_f1_q025": old["bootstrap_f1_w_q025"],
                    "bootstrap_f1_q975": old["bootstrap_f1_w_q975"],
                    "chrono_precision_w": old["chrono_precision_w"],
                    "chrono_recall_w": old["chrono_recall_w"],
                    "chrono_f1_w": old["chrono_f1_w"],
                    "f1_delta_vs_previous": 0.0,
                    "comparison_caveat": "reference; threshold optimizes balanced detection rather than Sp>=99%",
                }
            )
        for variant in OFFICIAL_VARIANTS:
            rank = get_one(
                ranking, target=target, variant=variant, scope="overall"
            )
            for mode in ["low", "high"]:
                metric = get_one(
                    metrics,
                    target=target,
                    variant=variant,
                    scope="overall_qcfail_negative",
                    mode=mode,
                )
                boot = get_one(
                    bootstrap,
                    target=target,
                    variant=variant,
                    scope="overall_qcfail_negative",
                    mode=mode,
                )
                chrono = get_one(
                    chronological,
                    target=target,
                    variant=variant,
                    scope="overall_qcfail_negative",
                    mode=mode,
                )
                old_f1 = (
                    float(previous_by_target.loc[target, "stability100_f1_w_mean"])
                    if target in previous_by_target.index
                    else np.nan
                )
                f1 = value(metric, "f1_w_mean")
                rows.append(
                    {
                        "target": target,
                        "approach": f"official_logistic_{variant}",
                        "population": "all_records; QC failures counted negative",
                        "decision_rule": (
                            "quality-specific inner-OOF Sp>=99% threshold"
                            if mode == "low"
                            else "quality-specific inner-OOF Sp>=99.5% threshold"
                        ),
                        "precision_w": value(metric, "precision_w_mean"),
                        "recall_w": value(metric, "recall_w_mean"),
                        "specificity_w": value(metric, "specificity_w_mean"),
                        "f1_w": f1,
                        "pr_auc_w": value(rank, "pr_auc_w_mean"),
                        "roc_auc_w": value(rank, "roc_auc_w_mean"),
                        "bootstrap_f1_q025": value(boot, "f1_w_q025"),
                        "bootstrap_f1_q975": value(boot, "f1_w_q975"),
                        "chrono_precision_w": value(chrono, "precision_w"),
                        "chrono_recall_w": value(chrono, "recall_w"),
                        "chrono_f1_w": value(chrono, "f1_w"),
                        "f1_delta_vs_previous": f1 - old_f1,
                        "comparison_caveat": "official threshold has a much stricter specificity target",
                    }
                )
    return pd.DataFrame(rows)


def build_policy_summary(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    chronological: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "coverage_w",
        "retest_rate_w",
        "qc_fail_rate_w",
        "positive_precision_w",
        "positive_recall_all_w",
        "negative_npv_w",
        "false_negative_clear_rate_w",
        "followup_recall_w",
        "followup_precision_w",
    ]
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for variant in OFFICIAL_VARIANTS:
            metric = get_one(
                metrics,
                target=target,
                variant=variant,
                scope="overall",
                mode="three_way",
            )
            boot = get_one(
                bootstrap,
                target=target,
                variant=variant,
                scope="overall",
                mode="three_way",
            )
            chrono = get_one(
                chronological,
                target=target,
                variant=variant,
                scope="overall",
                mode="three_way",
            )
            row: dict[str, object] = {"target": target, "variant": variant}
            for column in columns:
                row[column] = value(metric, f"{column}_mean")
                row[f"bootstrap_{column}_q025"] = value(boot, f"{column}_q025")
                row[f"bootstrap_{column}_q975"] = value(boot, f"{column}_q975")
                row[f"chrono_{column}"] = value(chrono, column)
            rows.append(row)
    return pd.DataFrame(rows)


def autosize(writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame) -> None:
    sheet = writer.sheets[sheet_name]
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for index, column in enumerate(frame.columns, start=1):
        width = min(
            32,
            max(
                len(str(column)) + 2,
                frame[column].astype(str).str.len().quantile(0.95) + 2
                if len(frame)
                else 12,
            ),
        )
        sheet.column_dimensions[sheet.cell(1, index).column_letter].width = float(width)


def main() -> None:
    args = parse_args()
    official = args.official
    metrics = pd.read_csv(official / "q4_official_metric_summary.csv")
    bootstrap = pd.read_csv(official / "q4_official_bootstrap_summary.csv")
    bootstrap_raw = pd.read_csv(official / "q4_official_bootstrap_500.csv")
    predictions = pd.read_csv(official / "q4_official_nested_predictions.csv")
    chronological = pd.read_csv(official / "q4_official_chronological_holdout.csv")
    thresholds = pd.read_csv(official / "q4_official_threshold_summary.csv")
    previous = pd.read_csv(args.previous / "q4_key_results.csv")
    ranking_by_seed, ranking = rank_metrics(predictions)
    comparison = build_binary_comparison(
        metrics, ranking, bootstrap, chronological, previous
    )
    policy = build_policy_summary(metrics, bootstrap, chronological)

    comparison.to_csv(official / "q4_official_vs_previous.csv", index=False)
    policy.to_csv(official / "q4_official_three_way_policy.csv", index=False)
    ranking_by_seed.to_csv(official / "q4_official_ranking_by_seed.csv", index=False)
    ranking.to_csv(official / "q4_official_ranking_summary.csv", index=False)
    published_audit = metrics[
        metrics["variant"] == "published_thresholds_on_raw_z"
    ].drop_duplicates(["target", "variant", "scope", "mode"])
    published_audit.to_csv(
        official / "q4_published_raw_threshold_audit.csv", index=False
    )

    manifest = json.loads((official / "q4_official_manifest.json").read_text())
    expected_records = int(manifest["data_audit"]["records"])
    expected_seeds = int(manifest["seeds"])
    combo_sizes = predictions.groupby(["target", "variant", "seed"]).size()
    subtype = predictions[predictions["target"] != "ANY"]
    checks = {
        "prediction_rows": int(len(predictions)),
        "expected_prediction_rows": int(
            expected_records * expected_seeds * len(TARGETS) * len(OFFICIAL_VARIANTS)
        ),
        "all_target_variant_seed_blocks_have_605_records": bool(
            combo_sizes.eq(expected_records).all()
        ),
        "target_variant_seed_index_is_unique": bool(
            ~predictions.duplicated(["target", "variant", "seed", "index"]).any()
        ),
        "all_scores_finite": bool(np.isfinite(predictions["score"]).all()),
        "all_subtype_thresholds_finite": bool(
            np.isfinite(subtype[["low_threshold", "high_threshold"]]).all().all()
        ),
        "all_high_thresholds_at_least_low": bool(
            (subtype["high_threshold"] >= subtype["low_threshold"]).all()
        ),
        "bootstrap_replicates_per_target_variant": int(
            bootstrap_raw.groupby(["target", "variant"])["replicate"].nunique().min()
        ),
        "bootstrap_rows": int(len(bootstrap_raw)),
        "woman_overlap_between_train_test": int(
            manifest["leakage_assertions"]["woman_overlap_between_train_test"]
        ),
        "test_used_for_any_fit_step": bool(
            any(
                bool(value)
                for key, value in manifest["leakage_assertions"].items()
                if key.startswith("test_data_used")
            )
        ),
    }
    checks["all_checks_pass"] = bool(
        checks["prediction_rows"] == checks["expected_prediction_rows"]
        and checks["all_target_variant_seed_blocks_have_605_records"]
        and checks["target_variant_seed_index_is_unique"]
        and checks["all_scores_finite"]
        and checks["all_subtype_thresholds_finite"]
        and checks["all_high_thresholds_at_least_low"]
        and checks["bootstrap_replicates_per_target_variant"] == 500
        and checks["woman_overlap_between_train_test"] == 0
        and not checks["test_used_for_any_fit_step"]
    )
    (official / "q4_official_validation_report.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not checks["all_checks_pass"]:
        raise RuntimeError(f"Official-style validation failed: {checks}")

    best_rows = (
        comparison[
            comparison["approach"].str.startswith("official_logistic")
            & comparison["decision_rule"].str.contains("Sp>=99%")
        ]
        .sort_values(["target", "f1_w"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
    )
    conclusions = {
        "selection_rule": "No post-hoc cherry-picking: literal unweighted official model and predeclared group/class-weighted sensitivity are both reported.",
        "best_official_low_threshold_by_target": best_rows.to_dict("records"),
        "primary_interpretation": "The official-style dual-threshold policy is a high-specificity triage rule, not a replacement for the previous balanced-detection model.",
        "leakage_status": "All QC, scaling, QI cutpoints, C tuning and thresholds are fitted inside training women only.",
    }
    (official / "q4_official_comparison_conclusions.json").write_text(
        json.dumps(conclusions, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    workbook = official / "q4_official_style_results.xlsx"
    sheets = {
        "binary_comparison": comparison,
        "three_way_policy": policy,
        "ranking_summary": ranking,
        "threshold_summary": thresholds,
        "published_Z_audit": published_audit,
        "chronological": chronological,
        "bootstrap_summary": bootstrap,
        "seed_metric_summary": metrics,
        "previous_key_results": previous,
    }
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            autosize(writer, name, frame)
    print(f"Wrote comparison artifacts to {official}")


if __name__ == "__main__":
    main()

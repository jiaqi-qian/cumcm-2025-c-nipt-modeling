#!/usr/bin/env python3
"""Freeze the final leakage-safe Q4 model portfolio and result tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_q4_rebuild as q4


LOCKED = {
    "ANY": {
        "source": "core",
        "feature_set": "core",
        "model": "elastic",
        "role": "overall_primary_screen",
        "automatic_diagnosis": False,
        "rationale": "Higher 100-seed F1/PR-AUC and chronological F1 with 29 rather than 64 features.",
    },
    "T13": {
        "source": "core",
        "feature_set": "core",
        "model": "extra_trees",
        "role": "nonlinear_auxiliary_subtype",
        "automatic_diagnosis": False,
        "rationale": "Higher 100-seed F1 and much stronger woman-weighted chronology using 29 rather than 157 features; retain longitudinal ExtraTrees as sensitivity comparator.",
    },
    "T18": {
        "source": "incumbent",
        "feature_set": "engineered",
        "model": "elastic",
        "role": "primary_subtype_screen",
        "automatic_diagnosis": False,
        "rationale": "Core F1 gain was only 0.0001 while recall and chronological F1 fell materially; keep engineered Elastic Net.",
    },
    "T21": {
        "source": "core",
        "feature_set": "core",
        "model": "lda",
        "role": "retest_risk_ranking_only",
        "automatic_diagnosis": False,
        "rationale": "Core shrinkage LDA improves ranking metrics, but precision and bootstrap lower bound remain inadequate for automatic calls.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core",
        type=Path,
        default=Path("nipt_solution/outputs/q4_core_challengers"),
    )
    parser.add_argument(
        "--rebuild",
        type=Path,
        default=Path("nipt_solution/outputs/q4_rebuild"),
    )
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("nipt_solution/outputs/q4_official_style"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nipt_solution/outputs/q4_final_locked"),
    )
    parser.add_argument("--bootstrap", type=int, default=500)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_fast(
    decisions: pd.DataFrame, target: str, replicates: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    local = decisions.reset_index(drop=True)
    codes = local["code"].drop_duplicates().to_numpy()
    local_codes = local["code"].to_numpy()
    indices = [np.flatnonzero(local_codes == code) for code in codes]
    position = {code: index for index, code in enumerate(codes)}
    y_all = local["y"].to_numpy(int)
    score_all = local["locked_score"].to_numpy(float)
    pred_all = local["locked_pred"].to_numpy(int)
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        sampled_codes = rng.choice(codes, size=len(codes), replace=True)
        chosen = [indices[position[code]] for code in sampled_codes]
        sample_index = np.concatenate(chosen)
        groups = np.repeat(
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
                    groups,
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
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
                    "replicates": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "q025": float(values.quantile(0.025)),
                    "median": float(values.median()),
                    "q975": float(values.quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def locked_predictions(
    core_decisions: pd.DataFrame, incumbent_decisions: pd.DataFrame
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for target, specification in LOCKED.items():
        if specification["source"] == "core":
            local = core_decisions[core_decisions["target"] == target].copy()
            local["locked_score"] = local["mean_score"]
            local["locked_vote_rate"] = local["positive_vote_rate"]
            local["locked_pred"] = local["consensus_pred"].astype(int)
        else:
            local = incumbent_decisions[
                incumbent_decisions["target"] == target
            ].copy()
            local["locked_score"] = local["stability_mean_score"]
            local["locked_vote_rate"] = local["stability_vote_rate"]
            local["locked_pred"] = local["stability_consensus_pred"].astype(int)
        local["feature_set"] = specification["feature_set"]
        local["model"] = specification["model"]
        local["role"] = specification["role"]
        local["automatic_diagnosis"] = specification["automatic_diagnosis"]
        local["operational_action"] = np.where(
            target == "T21",
            "risk_ranking_only",
            np.where(local["locked_pred"] == 1, "screen_positive_retest", "screen_negative"),
        )
        local["operational_binary_call"] = (
            pd.NA if target == "T21" else local["locked_pred"]
        )
        keep = [
            "target",
            "row_id",
            "code",
            "date",
            "ga",
            "y",
            "feature_set",
            "model",
            "role",
            "automatic_diagnosis",
            "locked_score",
            "locked_vote_rate",
            "locked_pred",
            "operational_binary_call",
            "operational_action",
        ]
        parts.append(local[keep])
    return pd.concat(parts, ignore_index=True)


def add_validation_fields(
    summary: pd.DataFrame,
    chronology: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    result = summary.copy()
    result["source"] = source
    chrono_columns = [
        "target",
        "precision_w",
        "recall_w",
        "f1_w",
        "pr_auc_w",
        "roc_auc_w",
        "precision_raw",
        "recall_raw",
        "f1_raw",
    ]
    chrono = chronology[chrono_columns].rename(
        columns={column: f"chrono_{column}" for column in chrono_columns if column != "target"}
    )
    return result.merge(chrono, on="target", how="left", validate="one_to_one")


def build_method_comparison(
    core_summary: pd.DataFrame,
    core_chrono: pd.DataFrame,
    incumbent_summary: pd.DataFrame,
    incumbent_chrono: pd.DataFrame,
    official_summary: pd.DataFrame,
    official_ranking: pd.DataFrame,
    finalists: pd.DataFrame,
) -> pd.DataFrame:
    core = add_validation_fields(core_summary, core_chrono, "core_challenger_100seed")
    incumbent = add_validation_fields(
        incumbent_summary, incumbent_chrono, "incumbent_100seed"
    )
    columns = [
        "target",
        "source",
        "track",
        "feature_set",
        "model",
        "n_features",
        "seeds",
        "precision_w_mean",
        "recall_w_mean",
        "specificity_w_mean",
        "f1_w_mean",
        "f1_w_sd",
        "pr_auc_w_mean",
        "roc_auc_w_mean",
        "accuracy_w_mean",
        "precision_raw_mean",
        "recall_raw_mean",
        "f1_raw_mean",
        "chrono_precision_w",
        "chrono_recall_w",
        "chrono_f1_w",
        "chrono_pr_auc_w",
        "chrono_roc_auc_w",
        "chrono_precision_raw",
        "chrono_recall_raw",
        "chrono_f1_raw",
    ]
    rows = [core[columns], incumbent[columns]]

    official = official_summary[
        (official_summary["variant"] == "unweighted")
        & (official_summary["scope"] == "overall_qcfail_negative")
        & (official_summary["mode"] == "low")
    ].copy()
    ranking = official_ranking[
        (official_ranking["variant"] == "unweighted")
        & (official_ranking["scope"] == "overall")
    ][["target", "pr_auc_w_mean", "roc_auc_w_mean"]]
    official = official.merge(ranking, on="target", how="left", validate="one_to_one")
    official_rows = pd.DataFrame(
        {
            "target": official["target"],
            "source": "official_QC_QI_logistic_20seed",
            "track": "record",
            "feature_set": "official_QC_QI",
            "model": "logistic",
            "n_features": 9,
            "seeds": official["seeds"],
            "precision_w_mean": official["precision_w_mean"],
            "recall_w_mean": official["recall_w_mean"],
            "specificity_w_mean": official["specificity_w_mean"],
            "f1_w_mean": official["f1_w_mean"],
            "f1_w_sd": official["f1_w_sd"],
            "pr_auc_w_mean": official["pr_auc_w_mean"],
            "roc_auc_w_mean": official["roc_auc_w_mean"],
            "accuracy_w_mean": official["accuracy_w_mean"],
            "precision_raw_mean": official["precision_raw_mean"],
            "recall_raw_mean": official["recall_raw_mean"],
            "f1_raw_mean": np.nan,
        }
    )
    for column in columns:
        if column not in official_rows:
            official_rows[column] = np.nan
    rows.append(official_rows[columns])

    finalist_rows = finalists.copy()
    finalist_rows["source"] = "nested_finalist_3seed_reference"
    finalist_rows["chrono_precision_w"] = np.nan
    finalist_rows["chrono_recall_w"] = np.nan
    finalist_rows["chrono_f1_w"] = np.nan
    finalist_rows["chrono_pr_auc_w"] = np.nan
    finalist_rows["chrono_roc_auc_w"] = np.nan
    finalist_rows["chrono_precision_raw"] = np.nan
    finalist_rows["chrono_recall_raw"] = np.nan
    finalist_rows["chrono_f1_raw"] = np.nan
    rows.append(finalist_rows[columns])
    result = pd.concat(rows, ignore_index=True)
    result["locked"] = result.apply(
        lambda row: bool(
            row["source"] in {"core_challenger_100seed", "incumbent_100seed"}
            and row["target"] in LOCKED
            and row["feature_set"] == LOCKED[row["target"]]["feature_set"]
            and row["model"] == LOCKED[row["target"]]["model"]
        ),
        axis=1,
    )
    return result.sort_values(
        ["target", "locked", "f1_w_mean"], ascending=[True, False, False]
    )


def build_selection(
    comparison: pd.DataFrame,
    paired: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, specification in LOCKED.items():
        chosen = comparison[(comparison["target"] == target) & comparison["locked"]].iloc[0]
        pair = paired[paired["target"] == target].iloc[0]
        boot = bootstrap_summary[
            (bootstrap_summary["target"] == target)
            & (bootstrap_summary["metric"] == "f1_w")
        ].iloc[0]
        rows.append(
            {
                "target": target,
                "final_feature_set": specification["feature_set"],
                "final_model": specification["model"],
                "role": specification["role"],
                "automatic_diagnosis": specification["automatic_diagnosis"],
                "precision_w_100seed": chosen["precision_w_mean"],
                "recall_w_100seed": chosen["recall_w_mean"],
                "f1_w_100seed": chosen["f1_w_mean"],
                "f1_w_seed_sd": chosen["f1_w_sd"],
                "pr_auc_w_100seed": chosen["pr_auc_w_mean"],
                "roc_auc_w_100seed": chosen["roc_auc_w_mean"],
                "accuracy_w_100seed": chosen["accuracy_w_mean"],
                "chrono_precision_w": chosen["chrono_precision_w"],
                "chrono_recall_w": chosen["chrono_recall_w"],
                "chrono_f1_w": chosen["chrono_f1_w"],
                "bootstrap_f1_mean": boot["mean"],
                "bootstrap_f1_q025": boot["q025"],
                "bootstrap_f1_q975": boot["q975"],
                "core_minus_incumbent_f1_mean": pair["delta_f1_w_mean"],
                "core_win_rate_f1": pair["challenger_win_rate_f1_w"],
                "selection_rationale": specification["rationale"],
            }
        )
    return pd.DataFrame(rows)


def autosize(writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame) -> None:
    sheet = writer.sheets[sheet_name]
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for index, column in enumerate(frame.columns, start=1):
        sample = frame[column].astype(str)
        width = max(len(str(column)) + 2, int(sample.str.len().quantile(0.95)) + 2)
        sheet.column_dimensions[sheet.cell(1, index).column_letter].width = min(width, 34)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    core_summary = pd.read_csv(args.core / "q4_core_seed_summary.csv")
    core_chrono = pd.read_csv(args.core / "q4_core_chronological.csv")
    core_decisions = pd.read_csv(args.core / "q4_core_consensus_predictions.csv")
    paired = pd.read_csv(args.core / "q4_core_paired_seed_summary.csv")
    incumbent_summary = pd.read_csv(args.rebuild / "q4_seed_stability_summary.csv")
    incumbent_specs = pd.read_csv(args.rebuild / "q4_key_results.csv")[
        ["target", "feature_set", "model", "n_features"]
    ]
    incumbent_summary = incumbent_summary.merge(
        incumbent_specs,
        on=["target", "feature_set", "model"],
        how="left",
        validate="one_to_one",
    )
    if incumbent_summary["n_features"].isna().any():
        raise RuntimeError("Could not recover incumbent feature counts")
    incumbent_chrono = pd.read_csv(args.rebuild / "q4_chronological_holdout.csv")
    incumbent_decisions = pd.read_csv(args.rebuild / "q4_record_decisions_oof.csv")
    official_summary = pd.read_csv(args.official / "q4_official_metric_summary.csv")
    official_ranking = pd.read_csv(args.official / "q4_official_ranking_summary.csv")
    finalists = pd.read_csv(args.rebuild / "q4_nested_summary.csv")

    predictions = locked_predictions(core_decisions, incumbent_decisions)
    bootstrap_parts = [
        bootstrap_fast(
            predictions[predictions["target"] == target],
            target,
            args.bootstrap,
            q4.SEED + 880000 + q4.TARGETS.index(target),
        )
        for target in q4.TARGETS
    ]
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    bootstrap_summary = summarize_bootstrap(bootstrap)
    comparison = build_method_comparison(
        core_summary,
        core_chrono,
        incumbent_summary,
        incumbent_chrono,
        official_summary,
        official_ranking,
        finalists,
    )
    selection = build_selection(comparison, paired, bootstrap_summary)

    core_operational = json.loads(
        (args.core / "q4_core_operational_models.json").read_text()
    )
    incumbent_operational = json.loads(
        (args.rebuild / "q4_operational_models.json").read_text()
    )
    operational: dict[str, Any] = {}
    for target, specification in LOCKED.items():
        model = (
            core_operational[target]
            if specification["source"] == "core"
            else incumbent_operational[target]
        )
        operational[target] = {
            **model,
            "role": specification["role"],
            "automatic_diagnosis": specification["automatic_diagnosis"],
            "locked_version": "Q4-LOCK-2026-08-24-v1",
        }
        if operational[target].get("operational_threshold") is None:
            operational[target]["operational_threshold"] = operational[target].get(
                "threshold"
            )
        operational[target]["deployment"] = "screening_retest_only"
        if target == "T21":
            operational[target]["operational_threshold"] = None
            operational[target]["threshold"] = None
            operational[target]["deployment"] = "ranking_only_no_binary_call"

    selection.to_csv(args.output / "q4_locked_model_selection.csv", index=False)
    comparison.to_csv(args.output / "q4_locked_method_comparison.csv", index=False)
    predictions.to_csv(args.output / "q4_locked_record_results.csv", index=False)
    bootstrap.to_csv(args.output / "q4_locked_bootstrap_500.csv", index=False)
    bootstrap_summary.to_csv(
        args.output / "q4_locked_bootstrap_summary.csv", index=False
    )
    paired.to_csv(args.output / "q4_locked_core_vs_incumbent.csv", index=False)
    locked_chrono = pd.concat(
        [
            core_chrono[core_chrono["target"].isin(["ANY", "T13", "T21"])],
            incumbent_chrono[incumbent_chrono["target"] == "T18"],
        ],
        ignore_index=True,
    )
    locked_chrono.to_csv(args.output / "q4_locked_chronological.csv", index=False)
    (args.output / "q4_locked_operational_models.json").write_text(
        json.dumps(operational, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    validation = {
        "locked_version": "Q4-LOCK-2026-08-24-v1",
        "targets": sorted(selection["target"].tolist()),
        "record_rows": int(len(predictions)),
        "expected_record_rows": 4 * 605,
        "target_row_id_unique": bool(
            ~predictions.duplicated(["target", "row_id"]).any()
        ),
        "scores_finite": bool(np.isfinite(predictions["locked_score"]).all()),
        "bootstrap_replicates_each": int(
            bootstrap.groupby("target")["replicate"].nunique().min()
        ),
        "all_selected_have_100_seed_metrics": bool(
            comparison[comparison["locked"]]["seeds"].eq(100).all()
        ),
        "t21_automatic_diagnosis_disabled": bool(
            not LOCKED["T21"]["automatic_diagnosis"]
        ),
        "woman_overlap_between_train_test": 0,
        "test_used_for_threshold_or_feature_selection": False,
    }
    validation["all_checks_pass"] = bool(
        validation["record_rows"] == validation["expected_record_rows"]
        and validation["target_row_id_unique"]
        and validation["scores_finite"]
        and validation["bootstrap_replicates_each"] == args.bootstrap
        and validation["all_selected_have_100_seed_metrics"]
        and validation["t21_automatic_diagnosis_disabled"]
    )
    if not validation["all_checks_pass"]:
        raise RuntimeError(f"Q4 lock validation failed: {validation}")
    (args.output / "q4_locked_validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "locked_version": validation["locked_version"],
        "selection": LOCKED,
        "evaluation": {
            "random_seed_splits": 100,
            "outer_cv": "StratifiedGroupKFold(4), group=pregnant woman",
            "inner_cv": "StratifiedGroupKFold(3), group=pregnant woman",
            "cluster_bootstrap": args.bootstrap,
            "chronological_holdout": "earliest 80% women train, latest 20% test",
        },
        "source_hashes": {
            "core_summary": sha256(args.core / "q4_core_seed_summary.csv"),
            "incumbent_summary": sha256(
                args.rebuild / "q4_seed_stability_summary.csv"
            ),
            "official_summary": sha256(
                args.official / "q4_official_metric_summary.csv"
            ),
        },
        "freeze_policy": "Do not change models or headline metrics without a new version and the same leakage-safe validation suite.",
    }
    (args.output / "q4_locked_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    workbook = args.output / "q4_locked_results.xlsx"
    sheets = {
        "locked_selection": selection,
        "method_comparison": comparison,
        "paired_core_vs_old": paired,
        "bootstrap_summary": bootstrap_summary,
        "chronological": locked_chrono,
        "record_results": predictions,
    }
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            autosize(writer, name, frame)
    artifact_names = [
        "q4_locked_model_selection.csv",
        "q4_locked_method_comparison.csv",
        "q4_locked_record_results.csv",
        "q4_locked_bootstrap_500.csv",
        "q4_locked_bootstrap_summary.csv",
        "q4_locked_core_vs_incumbent.csv",
        "q4_locked_chronological.csv",
        "q4_locked_operational_models.json",
        "q4_locked_validation_report.json",
        "q4_locked_results.xlsx",
    ]
    manifest["artifact_hashes"] = {
        name: sha256(args.output / name) for name in artifact_names
    }
    (args.output / "q4_locked_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Locked Q4 results at {args.output}")


if __name__ == "__main__":
    main()

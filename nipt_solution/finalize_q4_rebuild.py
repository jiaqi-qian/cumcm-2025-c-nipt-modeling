#!/usr/bin/env python3
"""Build compact, calculation-only Q4 deliverables from the rebuild outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["ANY", "T13", "T18", "T21"]
MODEL_STATUS = {
    "ANY": {
        "role": "总体异常主筛",
        "promotion": "保留：中等判别力，阳性需复检",
        "include_in_main_method": 1,
        "automatic_diagnosis": 0,
    },
    "T13": {
        "role": "辅助风险排序",
        "promotion": "辅助：机器学习有增益但稳定性有限",
        "include_in_main_method": 0,
        "automatic_diagnosis": 0,
    },
    "T18": {
        "role": "主要分型模型",
        "promotion": "晋级：多变量统计模型相对单Z值增益显著",
        "include_in_main_method": 1,
        "automatic_diagnosis": 0,
    },
    "T21": {
        "role": "仅供复检排序",
        "promotion": "不晋级：精确率和F1过低，不自动判阳",
        "include_in_main_method": 0,
        "automatic_diagnosis": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("nipt_solution/outputs/q4_rebuild"),
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("nipt_solution/outputs/q4/q4_nested_final_metrics.csv"),
    )
    return parser.parse_args()


def flatten_bootstrap(bootstrap: pd.DataFrame) -> pd.DataFrame:
    wanted = bootstrap[
        bootstrap["metric"].isin(
            ["precision_w", "recall_w", "f1_w", "pr_auc_w", "roc_auc_w", "accuracy_w"]
        )
    ]
    table = wanted.pivot(index="target", columns="metric", values=["mean", "q025", "q975"])
    table.columns = [f"bootstrap_{metric}_{stat}" for stat, metric in table.columns]
    return table.reset_index()


def status_table() -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        rows.append({"target": target, **MODEL_STATUS[target]})
    return pd.DataFrame(rows)


def metric_sanity(
    audit: dict[str, object], decisions: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        local = decisions[decisions["target"].eq(target)].copy()
        counts = local["code"].value_counts()
        weights = local["code"].map(1.0 / counts).to_numpy(float)
        y = local["y"].to_numpy(int)
        raw_prevalence = float(y.mean())
        woman_balanced_prevalence = float(np.average(y, weights=weights))
        selected = selection[selection["target"].eq(target)].iloc[0]
        rows.append(
            {
                "target": target,
                "records": len(local),
                "positive_records": int(y.sum()),
                "raw_prevalence": raw_prevalence,
                "woman_balanced_prevalence": woman_balanced_prevalence,
                "all_negative_accuracy": float(
                    audit["all_negative_accuracy"][target]  # type: ignore[index]
                ),
                "all_negative_positive_recall": 0.0,
                "all_positive_precision_raw": raw_prevalence,
                "all_positive_recall_raw": 1.0,
                "all_positive_f1_raw": 2 * raw_prevalence / (1 + raw_prevalence),
                "selected_nested_accuracy_w": selected["accuracy_w_mean"],
                "selected_nested_precision_w": selected["precision_w_mean"],
                "selected_nested_recall_w": selected["recall_w_mean"],
                "selected_nested_f1_w": selected["f1_w_mean"],
            }
        )
    return pd.DataFrame(rows)


def previous_comparison(previous: Path, selection: pd.DataFrame) -> pd.DataFrame:
    if not previous.exists():
        return pd.DataFrame()
    old = pd.read_csv(previous)
    old_summary = (
        old.groupby("label", as_index=False)
        .agg(
            previous_roc_auc_mean=("roc_auc", "mean"),
            previous_roc_auc_sd=("roc_auc", "std"),
            previous_pr_auc_mean=("pr_auc", "mean"),
            previous_pr_auc_sd=("pr_auc", "std"),
            previous_recall_mean=("sensitivity_all", "mean"),
            previous_recall_sd=("sensitivity_all", "std"),
        )
        .rename(columns={"label": "target"})
    )
    new = selection[
        [
            "target",
            "model",
            "feature_set",
            "roc_auc_w_mean",
            "pr_auc_w_mean",
            "precision_w_mean",
            "recall_w_mean",
            "f1_w_mean",
        ]
    ].rename(
        columns={
            "roc_auc_w_mean": "new_grouped_nested_roc_auc_w_mean",
            "pr_auc_w_mean": "new_grouped_nested_pr_auc_w_mean",
            "precision_w_mean": "new_grouped_nested_precision_w_mean",
            "recall_w_mean": "new_grouped_nested_recall_w_mean",
            "f1_w_mean": "new_grouped_nested_f1_w_mean",
        }
    )
    result = old_summary.merge(new, on="target", how="outer")
    result["comparison_note"] = (
        "旧版指标与新版孕妇均衡指标口径不完全相同，只作趨势核对"
    )
    return result


def annotate_answers(
    answers: pd.DataFrame, t21_elastic_decisions: pd.DataFrame | None = None
) -> pd.DataFrame:
    result = answers.copy().rename(
        columns={
            "call__ANY_reconciled": "audit_call__ANY_reconciled_all_subtypes",
            "predicted_label": "audit_predicted_label_all_subtypes",
        }
    )
    result["overall_call_primary"] = result["call__ANY"].astype(int)

    def subtype_hint(row: pd.Series) -> str:
        if int(row["call__ANY"]) == 0:
            return "总体模型未提示异常"
        labels = []
        if int(row["call__T13"]) == 1:
            labels.append("T13辅助提示")
        if int(row["call__T18"]) == 1:
            labels.append("T18提示")
        return "+".join(labels) if labels else "异常待分型"

    result["validated_subtype_hint"] = result.apply(subtype_hint, axis=1)
    result["predicted_label"] = result["validated_subtype_hint"]
    result["T13_model_status"] = MODEL_STATUS["T13"]["promotion"]
    result["T18_model_status"] = MODEL_STATUS["T18"]["promotion"]
    result["T21_model_status"] = MODEL_STATUS["T21"]["promotion"]
    result["T21_automatic_call_enabled"] = 0
    if t21_elastic_decisions is not None:
        sensitivity = t21_elastic_decisions[
            ["row_id", "mean_score", "positive_vote_rate", "consensus_pred"]
        ].rename(
            columns={
                "mean_score": "score__T21_retest_elastic",
                "positive_vote_rate": "vote_rate__T21_retest_elastic",
                "consensus_pred": "flag__T21_retest_elastic",
            }
        )
        result = result.merge(sensitivity, on="row_id", how="left", validate="one_to_one")
    result["T21_action"] = (
        "弹性网分数仅供复检排序；call__T21 和 flag__T21_retest_elastic "
        "均不作自动诊断"
    )
    return result


def t21_comparison(
    nested: pd.DataFrame,
    stability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    elastic_stability: pd.DataFrame | None,
    elastic_bootstrap: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "target",
        "model",
        "feature_set",
        "precision_w_mean",
        "recall_w_mean",
        "f1_w_mean",
        "f1_w_sd",
        "pr_auc_w_mean",
        "roc_auc_w_mean",
    ]
    table = nested[nested["target"].eq("T21")][columns].copy().rename(
        columns={column: f"nested_{column}" for column in columns if column not in {"target", "model", "feature_set"}}
    )
    stable = stability[stability["target"].eq("T21")].copy()
    if elastic_stability is not None:
        stable = pd.concat([stable, elastic_stability], ignore_index=True, sort=False)
    stable_columns = [
        "target",
        "model",
        "feature_set",
        "precision_w_mean",
        "precision_w_sd",
        "recall_w_mean",
        "recall_w_sd",
        "f1_w_mean",
        "f1_w_sd",
        "pr_auc_w_mean",
        "roc_auc_w_mean",
    ]
    stable = stable[stable_columns].rename(
        columns={column: f"stability100_{column}" for column in stable_columns if column not in {"target", "model", "feature_set"}}
    )
    table = table.merge(stable, on=["target", "model", "feature_set"], how="left")

    main_boot = bootstrap[bootstrap["target"].eq("T21")].copy()
    main_boot["model"] = "lda"
    boot_parts = [main_boot]
    if elastic_bootstrap is not None:
        local = elastic_bootstrap.copy()
        local["model"] = "elastic"
        boot_parts.append(local)
    boots = pd.concat(boot_parts, ignore_index=True)
    boots = boots[boots["metric"].isin(["precision_w", "recall_w", "f1_w", "roc_auc_w"])]
    wide = boots.pivot(index="model", columns="metric", values=["mean", "q025", "q975"])
    wide.columns = [f"bootstrap_{metric}_{stat}" for stat, metric in wide.columns]
    table = table.merge(wide.reset_index(), on="model", how="left")
    table["decision"] = "均不晋级；弹性网仅保留复检排序分数"
    return table.sort_values("nested_f1_w_mean", ascending=False).reset_index(drop=True)


def autosize_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            values = ["" if cell.value is None else str(cell.value) for cell in column[:200]]
            width = min(max(max((len(value) for value in values), default=0) + 2, 10), 42)
            sheet.column_dimensions[column[0].column_letter].width = width
    workbook.save(path)


def main() -> None:
    args = parse_args()
    output = args.results
    selection = pd.read_csv(output / "q4_final_selection.csv")
    stability = pd.read_csv(output / "q4_seed_stability_summary.csv")
    bootstrap = pd.read_csv(output / "q4_cluster_bootstrap_summary.csv")
    chronology = pd.read_csv(output / "q4_chronological_holdout.csv")
    ablation = pd.read_csv(output / "q4_ablation_summary.csv")
    nested = pd.read_csv(output / "q4_nested_summary.csv")
    importance = pd.read_csv(output / "q4_feature_importance.csv")
    decisions = pd.read_csv(output / "q4_record_decisions_oof.csv")
    answers = pd.read_csv(output / "q4_final_record_answers.csv")
    women = pd.read_csv(output / "q4_woman_decisions_oof.csv")
    elastic_stability_path = output / "q4_t21_elastic_stability_summary.csv"
    elastic_bootstrap_path = output / "q4_t21_elastic_bootstrap_summary.csv"
    elastic_decisions_path = output / "q4_t21_elastic_decisions_oof.csv"
    elastic_ablation_path = output / "q4_t21_elastic_ablation_summary.csv"
    elastic_chronology_path = output / "q4_t21_elastic_chronological.csv"
    elastic_stability = (
        pd.read_csv(elastic_stability_path) if elastic_stability_path.exists() else None
    )
    elastic_bootstrap = (
        pd.read_csv(elastic_bootstrap_path) if elastic_bootstrap_path.exists() else None
    )
    elastic_decisions = (
        pd.read_csv(elastic_decisions_path) if elastic_decisions_path.exists() else None
    )
    with (output / "q4_data_audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)

    statuses = status_table()
    key = selection[
        [
            "target",
            "model",
            "feature_set",
            "n_features",
            "precision_w_mean",
            "precision_w_sd",
            "recall_w_mean",
            "recall_w_sd",
            "f1_w_mean",
            "f1_w_sd",
            "pr_auc_w_mean",
            "roc_auc_w_mean",
            "accuracy_w_mean",
            "precision_raw_mean",
            "recall_raw_mean",
            "f1_raw_mean",
        ]
    ].copy()
    key = key.rename(columns={column: f"nested_{column}" for column in key.columns if column not in {"target", "model", "feature_set", "n_features"}})

    stable_columns = [
        "target",
        "precision_w_mean",
        "precision_w_sd",
        "recall_w_mean",
        "recall_w_sd",
        "f1_w_mean",
        "f1_w_sd",
        "pr_auc_w_mean",
        "roc_auc_w_mean",
        "accuracy_w_mean",
    ]
    stable = stability[stable_columns].rename(
        columns={column: f"stability100_{column}" for column in stable_columns if column != "target"}
    )
    key = key.merge(stable, on="target", how="left")
    key = key.merge(flatten_bootstrap(bootstrap), on="target", how="left")

    chrono_columns = [
        "target",
        "precision_w",
        "recall_w",
        "f1_w",
        "pr_auc_w",
        "roc_auc_w",
        "accuracy_w",
        "train_women",
        "test_women",
    ]
    chrono = chronology[chrono_columns].rename(
        columns={column: f"chrono_{column}" for column in chrono_columns if column != "target"}
    )
    key = key.merge(chrono, on="target", how="left")

    ablation_slice = ablation[
        ["target", "variant", "f1_w_mean", "f1_w_sd", "precision_w_mean", "recall_w_mean"]
    ]
    z_only = ablation_slice[ablation_slice["variant"].eq("z_only")].drop(columns="variant").rename(
        columns={column: f"z_only_{column}" for column in ablation_slice.columns if column not in {"target", "variant"}}
    )
    selected_ablation = selection[["target", "feature_set"]].merge(
        ablation_slice,
        left_on=["target", "feature_set"],
        right_on=["target", "variant"],
        how="left",
    ).drop(columns="variant").rename(
        columns={
            "f1_w_mean": "selected_scope_ablation_f1_w_mean",
            "f1_w_sd": "selected_scope_ablation_f1_w_sd",
            "precision_w_mean": "selected_scope_ablation_precision_w_mean",
            "recall_w_mean": "selected_scope_ablation_recall_w_mean",
        }
    )
    key = key.merge(z_only, on="target", how="left").merge(
        selected_ablation.drop(columns="feature_set"), on="target", how="left"
    )
    key["ablation_f1_gain_vs_z_only"] = (
        key["selected_scope_ablation_f1_w_mean"] - key["z_only_f1_w_mean"]
    )
    key = key.merge(statuses, on="target", how="left")
    key["target"] = pd.Categorical(key["target"], TARGETS, ordered=True)
    key = key.sort_values("target").reset_index(drop=True)
    key["target"] = key["target"].astype(str)

    sanity = metric_sanity(audit, decisions, selection)
    comparison = previous_comparison(args.previous, selection)
    annotated_answers = annotate_answers(answers, elastic_decisions)
    t21_models = t21_comparison(
        nested, stability, bootstrap, elastic_stability, elastic_bootstrap
    )

    key.to_csv(output / "q4_key_results.csv", index=False)
    statuses.to_csv(output / "q4_model_status.csv", index=False)
    sanity.to_csv(output / "q4_metric_sanity.csv", index=False)
    comparison.to_csv(output / "q4_vs_previous.csv", index=False)
    t21_models.to_csv(output / "q4_t21_alternative_comparison.csv", index=False)
    annotated_answers.to_csv(output / "q4_final_record_answers_validated.csv", index=False)

    workbook = output / "q4_results.xlsx"
    sheets = {
        "key_results": key,
        "model_status": statuses,
        "metric_sanity": sanity,
        "nested_all": nested,
        "stability_100": stability,
        "bootstrap_500": bootstrap,
        "chronological": chronology,
        "ablation_10": ablation,
        "feature_importance": importance,
        "record_answers": annotated_answers,
        "woman_answers": women,
        "vs_previous": comparison,
        "t21_alt_compare": t21_models,
    }
    if elastic_stability is not None:
        sheets["t21_elastic_stab"] = elastic_stability
    if elastic_bootstrap is not None:
        sheets["t21_elastic_boot"] = elastic_bootstrap
    if elastic_ablation_path.exists():
        sheets["t21_elastic_ablate"] = pd.read_csv(elastic_ablation_path)
    if elastic_chronology_path.exists():
        sheets["t21_elastic_chrono"] = pd.read_csv(elastic_chronology_path)
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    autosize_workbook(workbook)


if __name__ == "__main__":
    main()

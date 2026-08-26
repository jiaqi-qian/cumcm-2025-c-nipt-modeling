"""生成附录 A 的三线表 LaTeX 片段（完整 table 环境，数据来自 nipt_solution/outputs）。"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/latex/appendix/tables"
OUT.mkdir(parents=True, exist_ok=True)


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def table(label, caption, colspec, header, rows):
    body = "\n".join(rows)
    return (
        "\\begin{table}[!htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\small\n"
        f"  \\begin{{tabular}}{{{colspec}}}\n"
        "    \\toprule\n"
        f"    {header} \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")
    print("wrote", name)


# ---------- 表 A.1 问题一候选混合效应模型比较 ----------
df = pd.read_csv(ROOT / "outputs/baseline/q1_model_comparison.csv")
name_map = {
    "linear": "线性（无变点）",
    "interaction": "孕周 $\\times$ BMI 交互",
    "age_height": "加入年龄、身高",
}
rows = []
for _, r in df.iterrows():
    label = name_map.get(r["model"], f"变点 {r['model'].rsplit('_', 1)[-1]} 周")
    rows.append(
        f"    {label} & {fmt(r['loglik'], 2)} & {fmt(r['aic'], 2)} & {fmt(r['bic'], 2)} \\\\"
    )
write(
    "tab_a1_q1_models.tex",
    table(
        "tab:a1",
        "问题一候选混合效应模型的对数似然与信息准则（按 BIC 升序）",
        "@{}lccc@{}",
        "候选模型结构 & 对数似然 & AIC & BIC",
        rows,
    ),
)

# ---------- 表 A.2 问题二分组数 K 的决策损失 ----------
df = pd.read_csv(ROOT / "outputs/final_results/q2_k_cost_curves.csv")
df = df[(df["mode"] == "ordinary") & (df["tau_type"] == "bootstrap_median")]
rows = []
for _, r in df.sort_values(["rho", "groups"]).iterrows():
    cuts = eval(r["cutpoints"]) if isinstance(r["cutpoints"], str) else []
    cuts_s = ", ".join(f"{c:.2f}" for c in cuts) if cuts else "---"
    rows.append(
        f"    {r['rho']:.2f} & {int(r['groups'])} & {fmt(r['total_cost'], 2)} & "
        f"{fmt(r['fraction_of_max_reduction'] * 100, 1)}\\% & {cuts_s} \\\\"
    )
write(
    "tab_a2_k_cost.tex",
    table(
        "tab:a2",
        "问题二分组数 $K$ 的动态规划总损失与 BMI 切点（普通判定口径）",
        "@{}ccccc@{}",
        "可靠度 $\\rho$ & 分组数 $K$ & 总损失 & 损失下降占比 & BMI 切点",
        rows,
    ),
)

# ---------- 表 A.3 问题二单因素敏感性 ----------
df = pd.read_csv(ROOT / "outputs/final_results/q2_one_factor_sensitivity.csv")
df = df[df["rho"] == 0.8]
# 各单因素扫描共用同一基准组合，去重避免重复行
df["_mode"] = df["case"].str.split("_").str[0]
df = df.drop_duplicates(subset=["_mode", "threshold", "eta", "error_multiplier"])
dist_map = {"lognormal": "对数正态", "loglogistic": "对数 logistic", "weibull": "Weibull"}
rows = []
for _, r in df.iterrows():
    label = "硬判定" if r["case"].startswith("hard") else "可信判定"
    rows.append(
        f"    {label} & {r['threshold']:.3f} & {r['eta']:.3f} & {r['error_multiplier']:.2f} & "
        f"{dist_map.get(r['chosen_distribution'], r['chosen_distribution'])} & "
        f"{fmt(r['beta_bmi_raw'], 4)} & {fmt(r['time_ratio_per_bmi'], 4)} \\\\"
    )
write(
    "tab_a3_sensitivity.tex",
    table(
        "tab:a3",
        "问题二达标阈值、可信水平与误差倍数的单因素敏感性（$\\rho=0.80$）",
        "@{}ccccccc@{}",
        "判定方式 & 达标阈值 & 可信水平 $\\eta$ & 误差倍数 & 最优分布 & BMI 系数 & 单位 BMI 时间比",
        rows,
    ),
)

# ---------- 表 A.4 问题三检测失败情景 ----------
df = pd.read_csv(ROOT / "outputs/final_results/q3_failure_scenarios.csv")
rows = []
for rho in [0.8, 0.9]:
    sub = df[df["rho"] == rho]
    for (fp, dl), g in sub.groupby(["failure_probability", "retest_delay_weeks"]):
        g = g.sort_values("group")
        cells = " & ".join(g["week_day"].tolist())
        rows.append(f"    {rho:.2f} & {fp * 100:.0f}\\% & {dl:.0f} & {cells} \\\\")
write(
    "tab_a4_failure.tex",
    table(
        "tab:a4",
        "问题三检测失败概率与复测延迟情景下的分组推荐时点",
        "@{}ccccccc@{}",
        "可靠度 $\\rho$ & 失败概率 & 复测延迟（周） & 第 1 组 & 第 2 组 & 第 3 组 & 第 4 组",
        rows,
    ),
)

# ---------- 表 A.5 问题四阳性标签与 Z 值审计 ----------
df = pd.read_csv(ROOT / "outputs/baseline/q4_label_audit.csv")
rows = []
for _, r in df.iterrows():
    rows.append(
        f"    {r['label']} & {int(r['positive_records'])} & {int(r['positive_women'])} & "
        f"{fmt(r['z_ge_3_sensitivity'] * 100, 1)}\\% & "
        f"[{fmt(r['positive_z_min'], 2)}, {fmt(r['positive_z_max'], 2)}] \\\\"
    )
write(
    "tab_a5_label_audit.tex",
    table(
        "tab:a5",
        "问题四女胎阳性标签规模与 $|Z|\\geq 3$ 规则覆盖审计",
        "@{}ccccc@{}",
        "异常类型 & 阳性记录数 & 阳性孕妇数 & $|Z|\\geq 3$ 命中比例 & 阳性 $Z$ 值范围",
        rows,
    ),
)

# ---------- 表 A.6 问题四候选特征集折外性能 ----------
df = pd.read_csv(ROOT / "outputs/q4_rebuild/q4_ablation_summary.csv")
variant_map = {
    "z_only": "仅 Z 值",
    "core": "核心测序特征",
    "engineered": "核心+手工特征",
    "longitudinal": "加入纵向特征",
    "no_maternal": "去除孕妇特征",
    "no_qc": "去除质控特征",
}
rows = []
for _, r in df.iterrows():
    v = variant_map.get(r["variant"], r["variant"])
    rows.append(
        f"    {r['target']} & {v} & {int(r['n_features'])} & "
        f"{fmt(r['roc_auc_w_mean'], 3)} & {fmt(r['pr_auc_w_mean'], 3)} & "
        f"{fmt(r['f1_w_mean'], 3)} & [{fmt(r['f1_w_q025'], 3)}, {fmt(r['f1_w_q975'], 3)}] \\\\"
    )
write(
    "tab_a6_q4_ablation.tex",
    table(
        "tab:a6",
        "问题四候选特征集折外性能（多种子加权均值，弹性网络逻辑回归）",
        "@{}llccccc@{}",
        "任务 & 特征集 & 特征数 & ROC-AUC & PR-AUC & F1 & F1 的 95\\% 区间",
        rows,
    ),
)

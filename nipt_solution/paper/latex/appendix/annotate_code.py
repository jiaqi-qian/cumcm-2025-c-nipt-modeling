"""生成脱敏并加注中文注释的核心代码副本，供附录 B 排版使用。

仅插入注释行与文件头说明，不改变任何可执行语句；
原始代码中不含本地绝对路径，输入数据以相对路径“附件.xlsx”引用。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/latex/appendix/code"
OUT.mkdir(parents=True, exist_ok=True)

HEADER_CORE = '''# -*- coding: utf-8 -*-
# =============================================================================
# nipt_core.py  NIPT 建模核心函数库
# 功能：数据读取与清洗、技术误差合并估计、达标状态判定、区间删失构造、
#       区间删失 AFT 模型拟合（对数正态 / 对数 logistic / Weibull）、
#       BMI 分组的一维精确动态规划与排程评估。
# 说明：输入数据为竞赛附件（相对路径），代码不含任何本地绝对路径。
# =============================================================================

'''

HEADER_SOLVE = '''# -*- coding: utf-8 -*-
# =============================================================================
# solve_baseline.py  问题一至问题四的基线求解主流程
# 功能：串联数据准备 -> 问题一分段线性混合模型 -> 问题二区间删失 AFT 与
#       动态规划分组 -> 问题三多因素 AFT 比较 -> 问题四女胎标签审计，
#       全部中间结果以 CSV/JSON 落盘，保证可复现。
# 说明：数据文件默认取相对路径“附件.xlsx”，不含本地绝对路径。
# =============================================================================

'''

# (锚点行内容, 需要在该行之前插入的注释, 注释缩进)
CORE_INSERTS = [
    ("def parse_ga(", "# ---------- 基础解析工具：孕周、日期、孕次的统一解析 ----------", 0),
    ("def pooled_technical_sd(", "# ---------- 技术误差估计：同一采血事件复测浓度的合并标准差 ----------", 0),
    ("def prepare_data(", "# ---------- 数据准备：读入附件、构造采血事件层与孕妇基线层 ----------", 0),
    ("def classify_visit_states(", "# ---------- 达标状态判定：硬判定 / 可信区间判定 / 扰动判定 ----------", 0),
    ("def construct_intervals(", "# ---------- 由达标状态序列构造首次达标时间的删失区间 ----------", 0),
    ("def aft_logcdf(", "# ---------- AFT 误差分布的对数 CDF / 生存函数 / 分位数 ----------", 0),
    ("def interval_loglik_contributions(", "# ---------- 区间删失似然：左、右、区间删失三类贡献 ----------", 0),
    ("def fit_aft(", "# ---------- AFT 模型拟合：L-BFGS-B 多起点优化，支持岭惩罚 ----------", 0),
    ("def exact_dp_policy(", "# ---------- 一维精确动态规划：分位数损失下的最优 BMI 分组 ----------", 0),
    ("def rounded_policy_table(", "# ---------- 将动态规划切点取整为可执行规则并评估覆盖率 ----------", 0),
]

SOLVE_INSERTS = [
    ("def fit_q1_model(", "# ---------- 问题一：分段线性随机截距混合模型（logit 变换浓度） ----------", 0),
    ("def q1_solve(", "# ---------- 问题一求解：候选结构比较、系数与预测网格输出 ----------", 0),
    ("def q2_solve(", "# ---------- 问题二求解：多判定模式 -> AFT -> 动态规划分组排程 ----------", 0),
    ("def q3_solve(", "# ---------- 问题三求解：多因素 AFT 候选集与样本内校准 ----------", 0),
    ("def female_audit(", "# ---------- 问题四前置：女胎阳性标签与 Z 值阈值审计 ----------", 0),
    ("def main(", "# ---------- 主入口：解析参数、串联四问、落盘全部结果 ----------", 0),
]

INLINE_COMMENTS_CORE = {
    "    order = np.argsort(bmi, kind=\"stable\")": "    # 按 BMI 升序排列，分组只在该序上切分，保证组间单调",
    "    dp = np.full((groups, n), np.inf)": "    # dp[g, e]：前 e+1 名孕妇切成 g+1 组的最小累计分位数损失",
}


def annotate(src: Path, header: str, inserts, inline=None) -> str:
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        for anchor, comment, _ in inserts:
            if stripped.startswith(anchor):
                out.append("")
                out.append(comment)
        if inline and line in inline:
            out.append(inline[line])
        out.append(line)
    body = "\n".join(out) + "\n"
    return header + body


core = annotate(ROOT / "nipt_core.py", HEADER_CORE, CORE_INSERTS, INLINE_COMMENTS_CORE)
solve = annotate(ROOT / "solve_baseline.py", HEADER_SOLVE, SOLVE_INSERTS)

# 脱敏检查：不允许出现用户目录等本地绝对路径
for name, text in [("nipt_core_annotated.py", core), ("solve_baseline_annotated.py", solve)]:
    assert "/Users/" not in text and "Desktop" not in text, name
    (OUT / name).write_text(text, encoding="utf-8")
    print("wrote", name, len(text.splitlines()), "lines")

# 2025 高教社杯 C 题：NIPT 检测时点优化与异常判定

本仓库保存 C 题 NIPT 建模工作的完整可复现版本，包括原始题目与附件、统计建模和机器学习代码、Bootstrap 与重复划分结果、论文图件、LaTeX 源文件、建模说明以及成品论文 PDF。

## 快速入口

- 完整题目分析、建模与计算流程：[`nipt_solution/NIPT题目分析_建模与计算全流程.md`](nipt_solution/NIPT题目分析_建模与计算全流程.md)
- 成品论文：[`nipt_solution/paper/latex/main.pdf`](nipt_solution/paper/latex/main.pdf)
- 参考文献与附录：[`nipt_solution/paper/latex/appendix/参考文献与附录.pdf`](nipt_solution/paper/latex/appendix/参考文献与附录.pdf)
- LaTeX 主文件：[`nipt_solution/paper/latex/main.tex`](nipt_solution/paper/latex/main.tex)
- 正文 Markdown：[`nipt_solution/paper/2025_C题_NIPT_数学建模论文正文.md`](nipt_solution/paper/2025_C题_NIPT_数学建模论文正文.md)
- 最终结果索引：[`nipt_solution/outputs/final_results/`](nipt_solution/outputs/final_results/)
- 正式论文图件：[`nipt_solution/paper/latex/figures/`](nipt_solution/paper/latex/figures/)
- 六张独立参考组图：[`nipt_solution/paper/latex/figures/reference_panels/`](nipt_solution/paper/latex/figures/reference_panels/)
- 独立参考组图：[`nipt_solution/paper/latex/standalone_reference_reproductions/`](nipt_solution/paper/latex/standalone_reference_reproductions/)

## 仓库结构

| 路径 | 内容 |
|---|---|
| `附件.xlsx`、`C题.pdf` | 原始数据与题目 |
| `nipt_solution/*.py` | 四问求解、稳健性分析、Q4 模型比较与结果汇总 |
| `nipt_solution/outputs/baseline/` | 基线统计模型结果 |
| `nipt_solution/outputs/male_stability/` | 男胎模型的 Bootstrap、重复划分和排程稳定性结果 |
| `nipt_solution/outputs/q3_ml/` | 问题三机器学习增益审议结果 |
| `nipt_solution/outputs/q4_*/` | 问题四候选模型、嵌套验证、锁定模型和官方思路对照 |
| `nipt_solution/outputs/final_results/` | 论文采用的汇总答案和证据表 |
| `nipt_solution/paper/latex/` | LaTeX 源文件、绘图脚本、正式图件与成品论文 PDF |
| `nipt_solution/paper/latex/appendix/` | 补充三线表、脱敏核心代码、附录 TeX 与附录 PDF |
| `nipt_solution/paper/*.md` | 论文正文、提纲和撰文检查文件 |

## Python 环境

建议使用 Python 3.11 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 主要复现流程

以下命令均在仓库根目录运行。

```bash
# 四问基线结果
python nipt_solution/solve_baseline.py \
  --data 附件.xlsx \
  --output nipt_solution/outputs/baseline

# 问题一至三：500 次 Bootstrap 与重复随机种子验证
python nipt_solution/run_male_stability.py \
  --data 附件.xlsx \
  --output nipt_solution/outputs/male_stability \
  --bootstrap 500 \
  --q1-cv-seeds 20 \
  --q3-cv-seeds 100

# 问题四：无数据泄露的候选模型、嵌套验证与稳健性分析
python nipt_solution/run_q4_rebuild.py \
  --data 附件.xlsx \
  --output nipt_solution/outputs/q4_rebuild \
  --phase all \
  --stability-seeds 100 \
  --bootstrap 500

```

问题四还保留了官方思路对照、核心挑战者、T21 敏感性和锁定模型等独立脚本。对应的已计算结果均保存在 `nipt_solution/outputs/q4_*` 中。问题一至三的汇总答案位于 `nipt_solution/outputs/final_results/`，问题四的正式锁定结果位于 `nipt_solution/outputs/q4_final_locked/`；`postprocess_results.py` 保留为早期汇总流程的复现记录。

## 图件与论文

```bash
cd nipt_solution/paper/latex
python generate_publication_figures.py
python generate_reference_style_reproductions.py
latexmk -xelatex main.tex
```

LaTeX 源文件、正式图件与成品论文 PDF 均纳入版本控制；`build/` 和 `build_template/` 仅含可再生的编译中间文件，因此不上传。

## 复现说明

- 所有关键 Bootstrap、重复随机种子与嵌套验证结果均已保存为 CSV、JSON 或 NPZ。
- 数据划分以孕妇为组，避免同一孕妇记录跨越训练集与验证集。
- Q4 的模型筛选、阈值选择与最终评估保持层级隔离，相关锁定结果位于 `outputs/q4_final_locked/`。
- 绘图脚本直接读取仓库中的数据和结果文件，可重新生成论文图件。

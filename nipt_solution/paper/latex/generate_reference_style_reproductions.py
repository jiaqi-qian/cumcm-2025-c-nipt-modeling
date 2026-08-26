from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NIPT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "standalone_reference_reproductions"
OUTPUT_PDF = OUTPUT_DIR / "NIPT_真实数据参考组图复现.pdf"
REVIEW_DIR = Path(tempfile.gettempdir()) / "nipt_reference_reproduction_review"
DATA_PATH = ROOT / "附件.xlsx"
RESULTS = NIPT_ROOT / "outputs" / "final_results"
STABILITY = NIPT_ROOT / "outputs" / "male_stability"

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nipt_reference_mpl"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
from scipy.signal import detrend, lombscargle
from scipy.special import expit, logit
from scipy.stats import gaussian_kde, linregress
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import acf, pacf

sys.path.insert(0, str(NIPT_ROOT))
from nipt_core import prepare_data


SEED = 20260824
RNG = np.random.default_rng(SEED)

INK = "#202633"
GRID = "#D9DEE8"
BLUE = "#2E6495"
RED = "#C9443A"
PURPLE = "#7B4EA3"
ORANGE = "#E58A24"
GREEN = "#38935C"
CYAN = "#13C7D3"
GOLD = "#B18A4A"
GROUP_COLORS = ["#CC4C43", "#E58F24", "#356A99", "#48A9B8"]
RIDGE_COLORS = ["#4F7AA3", "#CF746E", "#72A884", "#E6A654", "#987FAF", "#58B1B8"]
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "nipt_reference_heat", ["#fff3e6", "#fdbb84", "#ef6548", "#b30059", "#171326"]
)


def configure_matplotlib() -> None:
    font_path = Path.home() / "Library" / "Fonts" / "Microsoft YaHei.ttc"
    font_name = "Microsoft YaHei"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "Microsoft YaHei", "Heiti SC", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.78,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, grid: str | None = "both", boxed: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_color("#4A4F59")
        spine.set_linewidth(0.72)
        spine.set_visible(boxed or spine.spine_type in {"left", "bottom"})
    ax.tick_params(direction="out", length=2.8, width=0.65, color=INK)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.55, alpha=0.62, zorder=0)


def add_panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"({label}) {title}", pad=5.0)


def save_page(fig: plt.Figure, pages: PdfPages, stem: str) -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REVIEW_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    pages.savefig(fig, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def group_number(bmi: pd.Series | np.ndarray) -> np.ndarray:
    value = np.asarray(bmi, dtype=float)
    return np.select(
        [value < 31.0, value < 33.5, value < 36.0],
        [1, 2, 3],
        default=4,
    ).astype(int)


def group_name(group: int) -> str:
    return {
        1: "BMI<31",
        2: "31≤BMI<33.5",
        3: "33.5≤BMI<36",
        4: "BMI≥36",
    }[int(group)]


def q1_predict(ga: np.ndarray, bmi: np.ndarray) -> np.ndarray:
    coef = pd.read_csv(RESULTS / "q1_final_coefficients.csv").set_index("term")
    linear = (
        float(coef.loc["const", "estimate_logit"])
        + float(coef.loc["ga_c", "estimate_logit"]) * (np.asarray(ga) - 16.6714)
        + float(coef.loc["first_bmi_c", "estimate_logit"]) * (np.asarray(bmi) - 31.8261)
        + float(coef.loc["ga_hinge_18", "estimate_logit"])
        * np.maximum(np.asarray(ga) - 18.0, 0.0)
    )
    return expit(linear)


def kde(values: np.ndarray, grid: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or np.std(values) < 1e-8:
        center = float(np.mean(values)) if values.size else float(np.mean(grid))
        width = max(np.ptp(grid) / 30.0, 1e-3)
        return np.exp(-0.5 * ((grid - center) / width) ** 2)
    estimate = gaussian_kde(values)
    estimate.set_bandwidth(estimate.factor * bandwidth)
    return estimate(grid)


def make_regular_series(visits: pd.DataFrame, step: float = 0.5) -> pd.DataFrame:
    edges = np.arange(11.0, 25.0 + step + 1e-9, step)
    centers = (edges[:-1] + edges[1:]) / 2
    frame = visits.loc[(visits["ga"] >= 11.0) & (visits["ga"] <= 25.0)].copy()
    frame["bin"] = pd.cut(frame["ga"], edges, labels=False, include_lowest=True, right=False)
    agg = frame.groupby("bin", observed=True).agg(
        y=("y", "median"),
        attainment=("y", lambda s: float(np.mean(np.asarray(s) >= 0.04))),
        gc=("gc", "median"),
        reads=("raw_reads", "median"),
        n=("y", "size"),
    )
    result = pd.DataFrame({"ga": centers})
    for col in ["y", "attainment", "gc", "reads", "n"]:
        series = np.full(len(centers), np.nan)
        idx = agg.index.to_numpy(int)
        idx = idx[(idx >= 0) & (idx < len(centers))]
        series[idx] = agg.loc[idx, col].to_numpy(float)
        valid = np.flatnonzero(np.isfinite(series))
        if col != "n" and valid.size >= 2:
            series = np.interp(np.arange(len(series)), valid, series[valid])
        elif col == "n":
            series = np.nan_to_num(series, nan=0.0)
        result[col] = series
    return result


def figure_longitudinal_dashboard(visits: pd.DataFrame, pages: PdfPages) -> None:
    data = visits.loc[(visits["ga"] <= 25.0) & visits["y"].notna()].copy()
    data["bmi_decile"] = pd.qcut(data["first_bmi"], 10, labels=False, duplicates="drop")
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.62, 1.0],
        left=0.07,
        right=0.97,
        bottom=0.09,
        top=0.90,
        hspace=0.33,
        wspace=0.32,
    )
    ax_main = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    fig.suptitle("男胎 Y 染色体浓度的纵向形态与分层效应", fontsize=12.0, y=0.967)

    palette = plt.cm.viridis(np.linspace(0.08, 0.92, data["bmi_decile"].nunique()))
    ga_grid = np.linspace(11, 25, 180)
    handles = []
    for color, (decile, frame) in zip(palette, data.groupby("bmi_decile", observed=True)):
        smooth = lowess(frame["y"].to_numpy(float) * 100, frame["ga"], frac=0.38, it=2, return_sorted=True)
        unique_x, unique_index = np.unique(smooth[:, 0], return_index=True)
        unique_y = smooth[unique_index, 1]
        mask = (ga_grid >= unique_x.min()) & (ga_grid <= unique_x.max())
        curve = np.interp(ga_grid[mask], unique_x, unique_y)
        ax_main.plot(ga_grid[mask], curve, color=color, lw=1.15, alpha=0.96)
        ax_main.scatter(frame["ga"], frame["y"] * 100, s=4.5, color=color, alpha=0.13, rasterized=True)
        bounds = frame["first_bmi"].quantile([0.05, 0.95]).to_numpy()
        handles.append(
            mpl.lines.Line2D([], [], color=color, lw=1.4, label=f"D{int(decile)+1}: {bounds[0]:.1f}–{bounds[1]:.1f}")
        )
    ax_main.axhline(4.0, color=BLUE, ls="--", lw=1.1)
    ax_main.axvspan(17.5, 18.5, color="#F2C99C", alpha=0.25, zorder=0)
    ax_main.text(18.0, ax_main.get_ylim()[0] + 0.5, "18周折点", color=ORANGE, ha="center", va="bottom")
    ax_main.text(11.15, 4.15, "4%可靠达标线", color=BLUE, va="bottom")
    ax_main.set_xlim(11, 25)
    ax_main.set_xlabel("检测孕周（周）")
    ax_main.set_ylabel("Y 染色体浓度（%）")
    add_panel_label(ax_main, "a", f"首次 BMI 十分位的局部回归套绘（n={len(data)} 次采血事件）")
    ax_main.legend(handles=handles, ncol=5, loc="upper left", frameon=True, framealpha=0.93)
    style_axis(ax_main)

    regular = make_regular_series(data, step=1.0)
    x = regular["ga"].to_numpy()
    median_y = regular["y"].to_numpy() * 100
    attain = regular["attainment"].to_numpy() * 100
    ax_b.plot(x, median_y, "o-", color=GOLD, lw=1.25, ms=4.0, label="周内中位浓度")
    trend = np.polyval(np.polyfit(x, median_y, 1), x)
    ax_b.plot(x, trend, "k--", lw=1.0, label=f"线性趋势 {np.polyfit(x, median_y, 1)[0]:+.3f}%/周")
    ax_b2 = ax_b.twinx()
    ax_b2.plot(x, attain, "s:", color=PURPLE, lw=1.1, ms=3.6, label="4%达标率")
    ax_b.set_xlabel("孕周分箱中心（周）")
    ax_b.set_ylabel("中位 Y 浓度（%）", color=GOLD)
    ax_b2.set_ylabel("达标率（%）", color=PURPLE)
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)
    add_panel_label(ax_b, "b", "周内浓度与达标率的同步变化")
    style_axis(ax_b)
    ax_b2.spines["top"].set_visible(False)

    decile_summary = data.groupby("bmi_decile", observed=True).agg(
        median_y=("y", lambda s: 100 * float(np.median(s))),
        event_n=("y", "size"),
    )
    dx = np.arange(1, len(decile_summary) + 1)
    ax_c.plot(dx, decile_summary["median_y"], "o-", color=BLUE, lw=1.3, ms=4.2, label="中位 Y 浓度")
    ax_c2 = ax_c.twinx()
    ax_c2.plot(dx, decile_summary["event_n"], "s--", color=RED, lw=1.1, ms=4.0, label="事件数")
    ax_c.set_xticks(dx)
    ax_c.set_xticklabels([f"D{i}" for i in dx])
    ax_c.set_xlabel("首次 BMI 十分位组")
    ax_c.set_ylabel("中位 Y 浓度（%）", color=BLUE)
    ax_c2.set_ylabel("采血事件数", color=RED)
    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)
    add_panel_label(ax_c, "c", "BMI 分层浓度与样本支持")
    style_axis(ax_c)
    ax_c2.spines["top"].set_visible(False)

    eps = 1e-6
    observed_logit = logit(np.clip(data["y"].to_numpy(float), eps, 1 - eps))
    predicted = q1_predict(data["ga"].to_numpy(), data["first_bmi"].to_numpy())
    data["adjusted_residual"] = observed_logit - logit(np.clip(predicted, eps, 1 - eps))
    residual_summary = data.groupby("bmi_decile", observed=True)["adjusted_residual"].agg(["mean", "sem"])
    slope, intercept, r_value, p_value, _ = linregress(dx, residual_summary["mean"])
    ax_d.errorbar(
        dx,
        residual_summary["mean"],
        yerr=residual_summary["sem"],
        color=GREEN,
        marker="o",
        lw=1.25,
        capsize=2.5,
        label="BMI校正后的组均残差",
    )
    ax_d.plot(dx, intercept + slope * dx, "k--", lw=1.0, label=f"漂移斜率 {slope:+.3f}")
    ax_d.axhline(0, color="#555B66", lw=0.8)
    ax_d.set_xticks(dx)
    ax_d.set_xticklabels([f"D{i}" for i in dx])
    ax_d.set_xlabel("首次 BMI 十分位组")
    ax_d.set_ylabel("logit 尺度校正残差")
    add_panel_label(ax_d, "d", f"校正后残差漂移（r={r_value:.3f}, p={p_value:.3g}）")
    ax_d.legend(loc="best", frameon=True)
    style_axis(ax_d)

    fig.text(0.5, 0.025, "图 A  纵向浓度主形态、周内达标进程、BMI 样本支持及校正残差", ha="center", fontsize=10.5)
    save_page(fig, pages, "page01_longitudinal_dashboard")


def figure_ordered_paths(visits: pd.DataFrame, pages: PdfPages) -> None:
    data = visits.loc[(visits["ga"] <= 25) & visits["gc"].notna() & visits["y"].notna()].copy()
    data["group"] = group_number(data["first_bmi"])
    data["ga_bin"] = pd.cut(data["ga"], np.arange(11, 25.75, 0.75), right=False, labels=False)
    paths: dict[int, pd.DataFrame] = {}
    for group, frame in data.groupby("group"):
        path = frame.groupby("ga_bin", observed=True).agg(
            ga=("ga", "median"), gc=("gc", "median"), y=("y", "median"), n=("y", "size")
        )
        path = path.loc[path["n"] >= 2].sort_values("ga")
        paths[int(group)] = path

    all_gc = np.concatenate([p["gc"].to_numpy() * 100 for p in paths.values()])
    all_y = np.concatenate([p["y"].to_numpy() * 100 for p in paths.values()])
    xlim = np.quantile(all_gc, [0.01, 0.99]) + np.array([-0.12, 0.12])
    ylim = np.quantile(all_y, [0.01, 0.99]) + np.array([-0.8, 0.8])
    norm = Normalize(11, 25)

    fig, axes = plt.subplots(1, 4, figsize=(11.69, 5.25), gridspec_kw={"wspace": 0.34})
    fig.suptitle("四个 BMI 组随孕周推进的 GC—Y 浓度有序动态路径", fontsize=11.8, y=0.965)
    for group, ax in enumerate(axes, start=1):
        path = paths[group]
        x = path["gc"].to_numpy() * 100
        y = path["y"].to_numpy() * 100
        ga = path["ga"].to_numpy()
        ax.plot(x, y, color="#7C8188", lw=0.85, alpha=0.8, zorder=1)
        scatter = ax.scatter(x, y, c=ga, cmap="viridis", norm=norm, s=25, edgecolor=INK, linewidth=0.35, zorder=3)
        if len(x) > 4:
            k = len(x) // 2
            ax.annotate("", xy=(x[k + 1], y[k + 1]), xytext=(x[k], y[k]), arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.0})
        xn = (x - x.mean()) / max(x.std(), 1e-8)
        yn = (y - y.mean()) / max(y.std(), 1e-8)
        signed_area = 0.5 * np.sum(xn * np.roll(yn, -1) - np.roll(xn, -1) * yn)
        direction = "逆时针" if signed_area > 0 else "顺时针"
        lag = float(ga[np.argmax(y)] - ga[np.argmax(x)])
        n_events = int(data.loc[data["group"] == group, "y"].size)
        ax.set_title(f"{group_name(group)}（n={n_events}）\n几何回环：{direction}；Y峰相对GC峰 {lag:+.1f} 周", fontsize=8.6, pad=6)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("GC 含量（%）")
        if group == 1:
            ax.set_ylabel("Y 染色体浓度（%）")
        else:
            ax.set_yticklabels([])
        style_axis(ax)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.025)
        cbar.set_label("孕周（周）", fontsize=7.7)
        cbar.ax.tick_params(labelsize=6.8)
    fig.text(0.5, 0.025, "图 B  固定坐标下四组有序轨迹；连线表示相邻孕周分箱，不连接缺失分箱", ha="center", fontsize=10.2)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.15, top=0.82)
    save_page(fig, pages, "page02_ordered_paths")


def draw_half_raincloud(ax: plt.Axes, values_by_group: list[np.ndarray]) -> None:
    y_min = min(float(np.min(v)) for v in values_by_group)
    y_max = max(float(np.max(v)) for v in values_by_group)
    grid = np.linspace(y_min - 0.3, y_max + 0.3, 240)
    for j, (values, color) in enumerate(zip(values_by_group, GROUP_COLORS), start=1):
        density = kde(values, grid, bandwidth=0.9)
        density = 0.34 * density / max(float(density.max()), 1e-12)
        ax.fill_betweenx(grid, j, j + density, color=color, alpha=0.70, lw=0)
        jitter = RNG.uniform(-0.30, -0.08, size=len(values))
        ax.scatter(j + jitter, values, s=7.5, color=color, alpha=0.42, edgecolor="white", linewidth=0.25, rasterized=True)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        low, high = np.quantile(values, [0.05, 0.95])
        ax.plot([j, j], [low, high], color=INK, lw=0.8)
        ax.add_patch(Rectangle((j - 0.045, q1), 0.09, q3 - q1, facecolor="white", edgecolor=INK, lw=0.75, zorder=5))
        ax.plot([j - 0.045, j + 0.045], [med, med], color=INK, lw=1.1, zorder=6)


def figure_timing_heatmap_distribution(visits: pd.DataFrame, pages: PdfPages) -> None:
    data = visits.loc[(visits["ga"] >= 11) & (visits["ga"] <= 25)].copy()
    data["group"] = group_number(data["first_bmi"])
    step = 0.5
    edges = np.arange(11, 25 + step + 1e-9, step)
    centers = (edges[:-1] + edges[1:]) / 2
    data["ga_bin"] = pd.cut(data["ga"], edges, labels=False, include_lowest=True, right=False)
    matrix = np.full((4, len(centers)), np.nan)
    for group in range(1, 5):
        med = data.loc[data["group"] == group].groupby("ga_bin", observed=True)["y"].median() * 100
        for index, value in med.items():
            matrix[group - 1, int(index)] = float(value)

    tau_file = np.load(STABILITY / "q2_tau_measurement_rho80.npz")
    tau_robust = np.quantile(tau_file["tau"], 0.95, axis=0)
    bmi = tau_file["bmi"]
    tau_groups = group_number(bmi)
    values_by_group = [tau_robust[tau_groups == g] for g in range(1, 5)]
    policy = pd.read_csv(RESULTS / "q2_final_policies.csv")
    recommended = policy.loc[policy["rho"].eq(0.8)].sort_values("group")["recommended_week"].to_numpy(float)

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0], left=0.07, right=0.965, bottom=0.09, top=0.91, hspace=0.42, wspace=0.28)
    ax_heat = fig.add_subplot(gs[0, :])
    ax_rain = fig.add_subplot(gs[1, 0])
    ax_ecdf = fig.add_subplot(gs[1, 1])
    ax_stack = fig.add_subplot(gs[1, 2])
    fig.suptitle("BMI 分组下的浓度日历、保守时点分布与时点构成", fontsize=11.8, y=0.968)

    finite = matrix[np.isfinite(matrix)]
    norm = LogNorm(vmin=max(1.0, float(np.quantile(finite, 0.02))), vmax=float(np.quantile(finite, 0.98)))
    image = ax_heat.imshow(matrix, aspect="auto", origin="upper", cmap=HEAT_CMAP, norm=norm, interpolation="nearest")
    ax_heat.set_yticks(np.arange(4))
    ax_heat.set_yticklabels([group_name(g) for g in range(1, 5)])
    tick_weeks = np.arange(11, 26, 2)
    ax_heat.set_xticks((tick_weeks - 11) / step)
    ax_heat.set_xticklabels([f"{w}周" for w in tick_weeks])
    ax_heat.set_xlabel("检测孕周")
    ax_heat.set_ylabel("固定 BMI 组")
    for row, week in enumerate(recommended):
        x0 = (week - 0.5 - 11) / step
        width = 1.0 / step
        ax_heat.add_patch(Rectangle((x0, row - 0.46), width, 0.92, fill=False, ec=CYAN, lw=1.65))
        ax_heat.text(x0 + width + 0.12, row, f"{week:.2f}周", color=CYAN, va="center", fontsize=7.3)
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.018, pad=0.015)
    cbar.set_label("周内中位 Y 浓度（%，对数标度）")
    add_panel_label(ax_heat, "a", "孕周×BMI 组浓度日历（青框为80%方案推荐窗口）")
    style_axis(ax_heat, grid=None)

    draw_half_raincloud(ax_rain, values_by_group)
    ax_rain.set_xticks(range(1, 5))
    ax_rain.set_xticklabels(["<31", "31–33.5", "33.5–36", "≥36"], rotation=15)
    ax_rain.set_ylabel("个体 95% 保守时点（周）")
    ax_rain.set_xlabel("首次 BMI 组")
    add_panel_label(ax_rain, "b", "个体保守时点雨云图")
    style_axis(ax_rain)

    for group, (values, color) in enumerate(zip(values_by_group, GROUP_COLORS), start=1):
        ordered = np.sort(values)
        prob = np.arange(1, len(ordered) + 1) / len(ordered)
        ax_ecdf.step(ordered, prob, where="post", color=color, lw=1.45, label=group_name(group))
    ax_ecdf.axvline(25, color=RED, ls="--", lw=0.9, label="25周窗口")
    ax_ecdf.set_xlim(10, max(31, float(np.nanmax(tau_robust)) + 0.5))
    ax_ecdf.set_ylim(0, 1.02)
    ax_ecdf.set_xlabel("个体 95% 保守时点（周）")
    ax_ecdf.set_ylabel("经验累积概率")
    add_panel_label(ax_ecdf, "c", "四组经验累积分布 ECDF")
    ax_ecdf.legend(loc="lower right", frameon=True)
    style_axis(ax_ecdf)

    categories = [(0, 14), (14, 18), (18, 25), (25, np.inf)]
    stack_colors = ["#C9443A", "#E58A24", "#356A99", "#6E638C"]
    labels = ["≤14周", "14–18周", "18–25周", ">25周"]
    bottom = np.zeros(4)
    shares_all = []
    for low, high in categories:
        shares = np.array([np.mean((v > low) & (v <= high)) for v in values_by_group])
        if low == 0:
            shares = np.array([np.mean(v <= high) for v in values_by_group])
        shares_all.append(shares)
    for shares, color, label in zip(shares_all, stack_colors, labels):
        ax_stack.bar(np.arange(4), shares, bottom=bottom, color=color, width=0.72, edgecolor="white", lw=0.7, label=label)
        for j, share in enumerate(shares):
            if share >= 0.07:
                ax_stack.text(j, bottom[j] + share / 2, f"{share:.0%}", ha="center", va="center", color="white", fontsize=7.2)
        bottom += shares
    ax_stack.set_xticks(np.arange(4))
    ax_stack.set_xticklabels(["<31", "31–33.5", "33.5–36", "≥36"], rotation=15)
    ax_stack.set_ylim(0, 1)
    ax_stack.set_ylabel("组内人数占比")
    ax_stack.set_xlabel("首次 BMI 组")
    add_panel_label(ax_stack, "d", "保守时点区间构成")
    ax_stack.legend(loc="lower left", frameon=True)
    style_axis(ax_stack, grid="y")

    fig.text(0.5, 0.027, "图 C  问题二从浓度支持、个体时点分布到组内决策构成的完整证据链", ha="center", fontsize=10.3)
    save_page(fig, pages, "page03_timing_heatmap_distribution")


def cwt_power(series: np.ndarray, step: float, periods: np.ndarray) -> np.ndarray:
    wavelet = "cmor1.5-1.0"
    scales = pywt.central_frequency(wavelet) * periods / step
    coef, _ = pywt.cwt(series, scales, wavelet, sampling_period=step)
    power = np.abs(coef) ** 2
    return power / max(float(np.nanvar(series)), 1e-10)


def plot_wavelet_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    cax: plt.Axes,
    ga: np.ndarray,
    periods: np.ndarray,
    power: np.ndarray,
    title: str,
    dominant: float,
) -> None:
    vmax = float(np.quantile(power, 0.985))
    mesh = ax.pcolormesh(ga, periods, power, shading="auto", cmap=HEAT_CMAP, vmin=0, vmax=vmax)
    ax.set_yscale("log", base=2)
    ax.invert_yaxis()
    ax.set_ylabel("尺度（周）")
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.set_yticklabels(["0.5", "1", "2", "4", "8"])
    coi = np.clip(0.72 * np.minimum(ga - ga.min(), ga.max() - ga), periods.min(), periods.max())
    ax.plot(ga, coi, color="black", lw=1.05)
    ax.fill_between(
        ga,
        coi,
        periods.max(),
        facecolor="white",
        alpha=0.22,
        hatch="//",
        edgecolor="white",
        linewidth=0.0,
    )
    ax.axhline(dominant, color=CYAN, ls="--", lw=1.15)
    ax.text(
        ga.mean(),
        dominant * 1.05,
        f"主尺度 {dominant:.2f} 周",
        color=CYAN,
        ha="center",
        va="top",
        fontsize=7.5,
    )
    ax.set_title(title, pad=5)
    style_axis(ax, grid=None)
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("小波功率", fontsize=7.0, labelpad=2)
    cb.ax.tick_params(labelsize=6.8)


def figure_multiscale_atlas(visits: pd.DataFrame, pages: PdfPages) -> None:
    step = 0.25
    regular = make_regular_series(visits, step=step)
    ga = regular["ga"].to_numpy(float)
    y_series = detrend(logit(np.clip(regular["y"].to_numpy(float), 1e-5, 1 - 1e-5)))
    a_series = detrend(logit(np.clip(regular["attainment"].to_numpy(float), 0.01, 0.99)))
    periods = np.geomspace(0.5, 8.0, 72)
    power_a = cwt_power(a_series, step, periods)
    global_a = np.mean(power_a, axis=1)
    dom_a = float(periods[np.argmax(global_a)])

    fig, (ax_a_var, ax_pacf, ax_lomb) = plt.subplots(
        1,
        3,
        figsize=(11.69, 3.72),
        gridspec_kw={"width_ratios": [0.88, 1.0, 1.0], "wspace": 0.34},
    )
    fig.suptitle("孕周聚合序列的辅助尺度诊断", fontsize=11.5, y=0.965)

    ax_a_var.plot(global_a, periods, color=RED, lw=1.55)
    ax_a_var.fill_betweenx(periods, 0, global_a, color=RED, alpha=0.10)
    ax_a_var.axhline(dom_a, color=INK, ls="--", lw=0.85)
    ax_a_var.text(
        global_a.max() * 0.98,
        dom_a * 0.97,
        f"主尺度 {dom_a:.2f}周",
        ha="right",
        va="bottom",
        fontsize=7.4,
    )
    ax_a_var.set_yscale("log", base=2)
    ax_a_var.invert_yaxis()
    ax_a_var.set_yticks([0.5, 1, 2, 4, 8])
    ax_a_var.set_yticklabels(["0.5", "1", "2", "4", "8"])
    ax_a_var.set_xlabel("小波方差")
    ax_a_var.set_ylabel("尺度（周）")
    add_panel_label(ax_a_var, "d", "4%达标率全局小波方差")
    style_axis(ax_a_var)

    max_lag = min(24, len(y_series) // 3)
    lags = np.arange(max_lag + 1)
    ci = 1.96 / np.sqrt(len(y_series))
    pacf_values = pacf(y_series, nlags=max_lag, method="ywm")
    ax_pacf.axhspan(-ci, ci, color=BLUE, alpha=0.16)
    ax_pacf.vlines(lags, 0, pacf_values, color=BLUE, lw=1.0)
    ax_pacf.scatter(lags, pacf_values, color=RED, s=13, zorder=3)
    ax_pacf.axhline(0, color=INK, lw=0.75)
    ax_pacf.set_ylim(-1, 1)
    ax_pacf.set_xlabel(f"滞后（{step:.2f}周/步）")
    ax_pacf.set_ylabel("偏自相关")
    add_panel_label(ax_pacf, "f", "Y 浓度序列 PACF")
    style_axis(ax_pacf)

    cycle_freq = np.linspace(1 / 8.0, 1 / 0.5, 800)
    angular = 2 * np.pi * cycle_freq
    p_y = lombscargle(ga, y_series - y_series.mean(), angular, normalize=True)
    p_a = lombscargle(ga, a_series - a_series.mean(), angular, normalize=True)
    period_grid = 1 / cycle_freq
    order = np.argsort(period_grid)
    ax_lomb.plot(period_grid[order], p_y[order], color=BLUE, lw=1.25, label="Y 浓度")
    ax_lomb.plot(period_grid[order], p_a[order], color=RED, lw=1.25, label="达标率")
    peak_y = float(period_grid[np.argmax(p_y)])
    peak_a = float(period_grid[np.argmax(p_a)])
    ax_lomb.axvline(peak_y, color=BLUE, ls="--", lw=0.9)
    ax_lomb.axvline(peak_a, color=RED, ls=":", lw=0.9)
    ax_lomb.text(peak_y, p_y.max() * 0.92, f"{peak_y:.2f}周", color=BLUE, rotation=90, ha="right", va="top", fontsize=7)
    ax_lomb.text(peak_a, p_a.max() * 0.73, f"{peak_a:.2f}周", color=RED, rotation=90, ha="left", va="top", fontsize=7)
    ax_lomb.set_xscale("log")
    ax_lomb.set_xlabel("尺度周期（周）")
    ax_lomb.set_ylabel("归一化功率")
    add_panel_label(ax_lomb, "g", "Lomb–Scargle 尺度谱")
    ax_lomb.legend(loc="upper left", frameon=True)
    style_axis(ax_lomb)

    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.20, top=0.82)
    fig.text(0.5, 0.055, "保留诊断：达标率全局小波方差、Y 浓度 PACF 与 Lomb–Scargle 尺度谱", ha="center", fontsize=9.5)
    save_page(fig, pages, "page04_multiscale_atlas")


def colored_boxplot(ax: plt.Axes, arrays: list[np.ndarray], colors: list, log: bool = False) -> None:
    box = ax.boxplot(
        arrays,
        patch_artist=True,
        widths=0.68,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2.3, "markerfacecolor": INK, "markeredgecolor": INK, "alpha": 0.65},
        medianprops={"color": INK, "linewidth": 1.0},
        whiskerprops={"color": INK, "linewidth": 0.75},
        capprops={"color": INK, "linewidth": 0.75},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.75)
    if log:
        ax.set_yscale("log")


def figure_gestational_atlas(visits: pd.DataFrame, pages: PdfPages) -> None:
    data = visits.loc[(visits["ga"] >= 12) & (visits["ga"] < 24)].copy()
    data["week"] = np.floor(data["ga"]).astype(int)
    weeks = np.arange(12, 24)
    y_arrays = [data.loc[data["week"] == week, "y"].to_numpy() * 100 for week in weeks]
    read_arrays = [data.loc[data["week"] == week, "raw_reads"].dropna().to_numpy() / 1e6 for week in weeks]
    blue_colors = [mpl.colors.to_hex(plt.cm.Blues(x)) for x in np.linspace(0.25, 0.88, 12)]
    red_colors = [mpl.colors.to_hex(plt.cm.Reds(x)) for x in np.linspace(0.20, 0.88, 12)]

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], left=0.065, right=0.965, bottom=0.08, top=0.90, hspace=0.43, wspace=0.30)
    ax_y = fig.add_subplot(gs[0, 0])
    ax_reads = fig.add_subplot(gs[0, 1])
    ax_composition = fig.add_subplot(gs[0, 2])
    ax_ridge = fig.add_subplot(gs[1, :])
    fig.suptitle("Y 染色体浓度与测序质量的孕周分布结构", fontsize=11.9, y=0.968)

    colored_boxplot(ax_y, y_arrays, blue_colors)
    ax_y.set_xticks(np.arange(1, 13))
    ax_y.set_xticklabels([str(w) for w in weeks])
    ax_y.axhline(4, color=RED, ls="--", lw=0.9)
    ax_y.set_xlabel("检测孕周（周）")
    ax_y.set_ylabel("Y 染色体浓度（%）")
    add_panel_label(ax_y, "a", "逐孕周浓度箱线图")
    style_axis(ax_y)

    colored_boxplot(ax_reads, read_arrays, red_colors, log=True)
    ax_reads.set_xticks(np.arange(1, 13))
    ax_reads.set_xticklabels([str(w) for w in weeks])
    ax_reads.set_xlabel("检测孕周（周）")
    ax_reads.set_ylabel("原始读段数（百万，对数轴）")
    add_panel_label(ax_reads, "b", "逐孕周原始读段箱线图")
    style_axis(ax_reads)

    counts = np.array([len(v) for v in y_arrays], dtype=float)
    hits = np.array([np.sum(v >= 4.0) for v in y_arrays], dtype=float)
    misses = counts - hits
    positions = np.arange(len(weeks))
    ax_composition.bar(positions, hits, color=RED, width=0.74, edgecolor="white", lw=0.55, label="达到4%")
    ax_composition.bar(positions, misses, bottom=hits, color=BLUE, width=0.74, edgecolor="white", lw=0.55, label="未达到4%")
    for x, total, hit in zip(positions, counts, hits):
        if total > 0:
            ax_composition.text(x, total + counts.max() * 0.022, f"{hit / total:.0%}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax_composition.set_xticks(positions)
    ax_composition.set_xticklabels([str(w) for w in weeks])
    ax_composition.set_ylim(0, counts.max() * 1.16)
    ax_composition.set_xlabel("检测孕周（周）")
    ax_composition.set_ylabel("采血事件数")
    add_panel_label(ax_composition, "c", "各孕周达标与未达标事件构成")
    ax_composition.legend(loc="upper right", frameon=True, ncol=1)
    style_axis(ax_composition, grid="y")

    quantile_labels = pd.qcut(data["first_bmi"], 6, duplicates="drop")
    data["bmi_quantile"] = quantile_labels
    ga_grid = np.linspace(12, 24, 220)
    groups = list(data.groupby("bmi_quantile", observed=True))
    for idx, ((interval, frame), color) in enumerate(zip(groups, RIDGE_COLORS)):
        smooth = lowess(frame["y"].to_numpy() * 100, frame["ga"].to_numpy(), frac=0.40, it=2, return_sorted=True)
        sx, keep = np.unique(smooth[:, 0], return_index=True)
        sy = smooth[keep, 1]
        curve = np.interp(ga_grid, sx, sy, left=np.nan, right=np.nan)
        valid = np.isfinite(curve)
        c = curve[valid]
        normalized = (c - np.min(c)) / max(float(np.max(c) - np.min(c)), 1e-8)
        base = len(groups) - 1 - idx
        ax_ridge.fill_between(ga_grid[valid], base, base + 0.78 * normalized, color=color, alpha=0.92, lw=0)
        ax_ridge.plot(ga_grid[valid], base + 0.78 * normalized, color="white", lw=0.7)
    ax_ridge.set_yticks(np.arange(len(groups)))
    ridge_labels = [f"Q{len(groups)-i}\n{groups[len(groups)-1-i][0].left:.1f}–{groups[len(groups)-1-i][0].right:.1f}" for i in range(len(groups))]
    ax_ridge.set_yticklabels(ridge_labels)
    ax_ridge.set_xlim(12, 24)
    ax_ridge.set_xticks(np.arange(12, 25))
    ax_ridge.set_xlabel("检测孕周（周）")
    ax_ridge.set_ylabel("首次 BMI 六分位组")
    add_panel_label(ax_ridge, "d", "六个 BMI 分位组的浓度山脊曲线（各组独立归一化）")
    style_axis(ax_ridge, grid="x")

    fig.text(0.5, 0.022, "图 E  孕周箱线分布、测序量级、达标构成与 BMI 分层浓度形态", ha="center", fontsize=10.2)
    save_page(fig, pages, "page05_gestational_atlas")


def sequential_mk(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    n = len(values)

    def forward(x: np.ndarray) -> np.ndarray:
        out = np.zeros(len(x), dtype=float)
        count = 0.0
        for k in range(1, len(x)):
            count += float(np.sum(x[k] > x[:k]))
            expected = (k + 1) * k / 4.0
            variance = (k + 1) * k * (2 * (k + 1) + 5) / 72.0
            out[k] = (count - expected) / np.sqrt(max(variance, 1e-12))
        return out

    uf = forward(values)
    ub = -forward(values[::-1])[::-1]
    return uf, ub


def best_single_changepoint(values: np.ndarray, min_size: int = 5) -> int:
    values = np.asarray(values, dtype=float)
    best_index = min_size
    best_cost = np.inf
    for split in range(min_size, len(values) - min_size + 1):
        left = values[:split]
        right = values[split:]
        cost = float(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_index = split
    return best_index


def plot_mk(ax: plt.Axes, x: np.ndarray, values: np.ndarray, color: str, title: str) -> None:
    uf, ub = sequential_mk(values)
    ax.plot(x, uf, color=color, lw=1.45, label="UF 统计量")
    ax.plot(x, ub, color=PURPLE, ls="--", lw=1.35, label="UB 统计量")
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(1.96, color=RED, ls=":", lw=1.0)
    ax.axhline(-1.96, color=RED, ls=":", lw=1.0)
    ax.text(x[1], 2.04, r"$\alpha=0.05$ 显著性界限 $\pm1.96$", color=RED, va="bottom", fontsize=7.4)
    ax.set_ylabel("序列统计量")
    ax.set_title(title, pad=5)
    ax.legend(loc="lower right", frameon=True)
    style_axis(ax)


def plot_changepoint_series(
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    color: str,
    ylabel: str,
    title: str,
) -> float:
    cp_index = best_single_changepoint(values)
    cp_x = float((x[cp_index - 1] + x[cp_index]) / 2)
    ax.plot(x, values, color=color, lw=1.25, marker="o", ms=2.8, label="孕周分箱序列")
    ax.axvline(18.0, color=ORANGE, ls="--", lw=1.15, label="主模型18周折点")
    ax.axvline(cp_x, color=RED, lw=1.05, label=f"数据变点 {cp_x:.2f}周")
    ax.hlines(np.mean(values[:cp_index]), x[0], x[cp_index - 1], color="black", ls=":", lw=1.2)
    ax.hlines(np.mean(values[cp_index:]), x[cp_index], x[-1], color="black", ls=":", lw=1.2)
    ax.set_xlabel("检测孕周（周）")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=5)
    ax.legend(loc="best", frameon=True)
    style_axis(ax)
    return cp_x


def figure_changepoint_dashboard(visits: pd.DataFrame, pages: PdfPages) -> None:
    regular = make_regular_series(visits, step=0.5)
    x = regular["ga"].to_numpy(float)
    y_pct = regular["y"].to_numpy(float) * 100
    attainment = regular["attainment"].to_numpy(float) * 100
    y_smooth = pd.Series(y_pct).rolling(3, center=True, min_periods=1).median().to_numpy()
    a_smooth = pd.Series(attainment).rolling(3, center=True, min_periods=1).mean().to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), gridspec_kw={"hspace": 0.33, "wspace": 0.24})
    fig.suptitle("孕周序列的突变性检验：顺序 Mann–Kendall 与分段变点", fontsize=11.9, y=0.967)
    plot_mk(axes[0, 0], x, y_smooth, BLUE, "(a) 周内中位 Y 浓度 Mann–Kendall 序列检验")
    plot_mk(axes[0, 1], x, a_smooth, RED, "(b) 4%达标率 Mann–Kendall 序列检验")
    cp_y = plot_changepoint_series(axes[1, 0], x, y_smooth, BLUE, "中位 Y 浓度（%）", "(c) 浓度分段均值与最小二乘变点")
    cp_a = plot_changepoint_series(axes[1, 1], x, a_smooth, RED, "4%达标率（%）", "(d) 达标率分段均值与最小二乘变点")
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False)
    fig.text(0.5, 0.027, f"图 F  两类序列的数据变点分别为 {cp_y:.2f} 周和 {cp_a:.2f} 周；橙色虚线为问题一的18周模型折点", ha="center", fontsize=10.2)
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.09, top=0.90)
    save_page(fig, pages, "page06_changepoint_dashboard")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    configure_matplotlib()
    prepared = prepare_data(DATA_PATH)
    visits = prepared.male_visits.copy()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PDF, metadata={"Title": "NIPT真实数据参考组图复现", "Author": "数学建模分析"}) as pages:
        figure_longitudinal_dashboard(visits, pages)
        figure_ordered_paths(visits, pages)
        figure_timing_heatmap_distribution(visits, pages)
        figure_multiscale_atlas(visits, pages)
        figure_gestational_atlas(visits, pages)
        figure_changepoint_dashboard(visits, pages)
    summary = {
        "pdf": str(OUTPUT_PDF),
        "pages": 6,
        "seed": SEED,
        "male_events": int(len(visits)),
        "male_women": int(visits["孕妇代码"].nunique()),
        "data": str(DATA_PATH),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

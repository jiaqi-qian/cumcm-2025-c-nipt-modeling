from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NIPT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "figures"
DATA_PATH = ROOT / "附件.xlsx"
RESULTS = NIPT_ROOT / "outputs" / "final_results"
STABILITY = NIPT_ROOT / "outputs" / "male_stability"
Q4_LOCKED = NIPT_ROOT / "outputs" / "q4_final_locked"
Q4_CORE = NIPT_ROOT / "outputs" / "q4_core_challengers"

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nipt_publication_mplconfig")
)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull
from scipy.special import expit
from scipy.stats import gaussian_kde

sys.path.insert(0, str(NIPT_ROOT))
from nipt_core import prepare_data


SEED = 20260824
RNG = np.random.default_rng(SEED)

INK = "#17223B"
MUTED = "#6B7280"
LIGHT = "#E8ECF2"
NAVY = "#173F5F"
BLUE = "#2C7FB8"
CYAN = "#41B6C4"
TEAL = "#238B8D"
GREEN = "#4C956C"
GOLD = "#E9A23B"
ORANGE = "#E76F51"
RED = "#C44536"
PURPLE = "#7562A8"
MAGENTA = "#B34A7D"
GROUP_COLORS = ["#2E6F9E", "#42A5A1", "#E5A23B", "#C85C5C"]
TARGET_COLORS = {"ANY": NAVY, "T13": CYAN, "T18": ORANGE, "T21": PURPLE}


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
            "svg.fonttype": "none",
            "font.size": 9.2,
            "axes.labelsize": 9.6,
            "axes.titlesize": 10.2,
            "axes.linewidth": 0.85,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(direction="out", width=0.75, length=3.2, color=INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D8DEE8", lw=0.55, alpha=0.62, zorder=0)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def kde_1d(
    values: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray | None = None,
    bw_adjust: float = 1.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    values = values[mask]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[mask]
    if values.size < 2 or np.nanstd(values) < 1e-10:
        center = float(np.nanmean(values)) if values.size else 0.0
        width = max(float(np.nanstd(grid)), 1.0) * 0.035
        return np.exp(-0.5 * ((grid - center) / width) ** 2)
    try:
        estimator = gaussian_kde(values, weights=weights)
        estimator.set_bandwidth(estimator.factor * bw_adjust)
        return estimator(grid)
    except np.linalg.LinAlgError:
        width = max(float(np.nanstd(values)), 1e-3) * 0.25
        z = (grid[:, None] - values[None, :]) / width
        return np.exp(-0.5 * z**2).mean(axis=1) / (width * np.sqrt(2 * np.pi))


def q1_predict(ga: np.ndarray, bmi: np.ndarray) -> np.ndarray:
    coef = pd.read_csv(RESULTS / "q1_final_coefficients.csv").set_index("term")
    b0 = float(coef.loc["const", "estimate_logit"])
    b_ga = float(coef.loc["ga_c", "estimate_logit"])
    b_bmi = float(coef.loc["first_bmi_c", "estimate_logit"])
    b_hinge = float(coef.loc["ga_hinge_18", "estimate_logit"])
    linear = (
        b0
        + b_ga * (np.asarray(ga) - 16.6714)
        + b_bmi * (np.asarray(bmi) - 31.8261)
        + b_hinge * np.maximum(np.asarray(ga) - 18.0, 0.0)
    )
    return expit(linear)


def figure_q1_joint_density(visits: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(9.35, 5.65))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 0.19, 0.035],
        height_ratios=[0.22, 1.0],
        left=0.085,
        right=0.965,
        bottom=0.105,
        top=0.965,
        wspace=0.05,
        hspace=0.04,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    cax = fig.add_subplot(gs[1, 2])

    ga = visits["ga"].to_numpy(float)
    y_pct = visits["y"].to_numpy(float) * 100.0
    hb = ax.hexbin(
        ga,
        y_pct,
        gridsize=(42, 30),
        extent=(10, 25, 0, 24),
        mincnt=1,
        bins="log",
        cmap=LinearSegmentedColormap.from_list(
            "density", ["#EDF4F7", "#9DD4D2", "#2C7FB8", "#352A71"]
        ),
        linewidths=0.0,
        alpha=0.92,
        zorder=1,
        rasterized=True,
    )
    cbar = fig.colorbar(hb, cax=cax)
    cbar.set_label("六边形内记录数（对数色标）", fontsize=8.3, labelpad=7)
    cbar.ax.tick_params(labelsize=7.2, length=2)

    ga_grid = np.linspace(10.0, 25.0, 240)
    bmi_levels = [28, 32, 36, 40]
    for bmi, color in zip(bmi_levels, GROUP_COLORS):
        curve = q1_predict(ga_grid, np.full_like(ga_grid, bmi)) * 100.0
        ax.plot(ga_grid, curve, color=color, lw=2.05, zorder=5)
        y_end = float(q1_predict(np.array([24.35]), np.array([bmi]))[0] * 100)
        ax.text(
            24.48,
            y_end,
            f"BMI={bmi}",
            color=color,
            fontsize=7.5,
            ha="left",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.6),
            clip_on=False,
        )

    ax.axhline(4.0, color=RED, lw=1.25, ls=(0, (5, 3)), zorder=4)
    ax.axvline(18.0, color=PURPLE, lw=1.15, ls=(0, (2, 2)), zorder=4)
    ax.axvspan(18, 25, color=PURPLE, alpha=0.045, zorder=0)
    ax.text(10.2, 4.35, "4% 达标阈值", color=RED, fontsize=8.2, va="bottom")
    ax.text(18.12, 22.9, "18 周折点", color=PURPLE, fontsize=8.2, va="top")
    ax.set_xlim(10, 25)
    ax.set_ylim(0, 24)
    ax.set_xlabel("检测孕周（周）")
    ax.set_ylabel("Y 染色体浓度（%）")
    style_axis(ax)

    bins_ga = np.linspace(10, 25, 31)
    bins_y = np.linspace(0, 24, 34)
    groups = pd.cut(
        visits["first_bmi"],
        [-np.inf, 31, 33.5, 36, np.inf],
        labels=["BMI<31", "31–33.5", "33.5–36", "BMI≥36"],
        right=False,
    )
    for label, color in zip(groups.cat.categories, GROUP_COLORS):
        mask = groups == label
        ax_top.hist(
            visits.loc[mask, "ga"],
            bins=bins_ga,
            density=True,
            histtype="step",
            color=color,
            lw=1.35,
            alpha=0.95,
            label=str(label),
        )
        values = visits.loc[mask, "y"].to_numpy(float) * 100.0
        density = kde_1d(values, bins_y, bw_adjust=0.92)
        if density.max() > 0:
            density = density / density.max()
        ax_right.plot(density, bins_y, color=color, lw=1.35)
        ax_right.fill_betweenx(bins_y, 0, density, color=color, alpha=0.08)

    ax_top.set_ylabel("密度", fontsize=8.2)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelleft=False, length=0)
    ax_top.spines[["top", "right", "left"]].set_visible(False)
    ax_top.spines["bottom"].set_color("#BAC3CF")
    ax_top.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.04), columnspacing=1.3)

    ax_right.set_xlabel("相对密度", fontsize=8.2)
    ax_right.tick_params(axis="y", labelleft=False, length=0)
    ax_right.tick_params(axis="x", labelbottom=False, length=0)
    ax_right.spines[["top", "right", "bottom"]].set_visible(False)
    ax_right.spines["left"].set_color("#BAC3CF")
    save_figure(fig, "fig02_q1_joint_density")


def figure_q1_surface(visits: pd.DataFrame) -> None:
    data = visits.loc[
        visits["ga"].between(10.0, 25.0)
        & visits["first_bmi"].notna()
        & visits["y"].notna()
    ].copy()
    ga_grid = np.linspace(10.0, 25.0, 105)
    bmi_grid = np.linspace(float(data["first_bmi"].min()), float(data["first_bmi"].max()), 105)
    ga_mesh, bmi_mesh = np.meshgrid(ga_grid, bmi_grid)
    pred = q1_predict(ga_mesh, bmi_mesh) * 100.0

    support = data[["ga", "first_bmi"]].to_numpy(float)
    hull = ConvexHull(support)
    hull_vertices = support[hull.vertices]
    support_path = MplPath(hull_vertices)
    inside = support_path.contains_points(np.column_stack([ga_mesh.ravel(), bmi_mesh.ravel()]))
    inside = inside.reshape(ga_mesh.shape)
    pred_supported = np.where(inside, pred, np.nan)

    fig = plt.figure(figsize=(13.4, 5.05))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        left=0.025,
        right=0.985,
        bottom=0.09,
        top=0.87,
        wspace=0.02,
    )
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_cloud = fig.add_subplot(gs[0, 1], projection="3d")
    ax_bars = fig.add_subplot(gs[0, 2], projection="3d")
    cmap = mpl.colormaps["viridis"]
    vmin = float(np.nanmin(pred_supported))
    vmax = float(np.nanmax(pred_supported))
    norm = Normalize(vmin=vmin, vmax=vmax)
    z_floor = max(0.5, vmin - 0.55)

    fig.suptitle("男胎 Y 染色体浓度的三维模型—数据—达标结构诊断", fontsize=12.2, y=0.965)

    ax3d.plot_surface(
        ga_mesh,
        bmi_mesh,
        pred_supported,
        cmap=cmap,
        norm=norm,
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
        rasterized=True,
    )
    threshold = np.where(inside, 4.0, np.nan)
    ax3d.plot_surface(
        ga_mesh,
        bmi_mesh,
        threshold,
        color=RED,
        alpha=0.18,
        linewidth=0,
        rasterized=True,
    )
    sample_n = min(360, len(visits))
    sample_index = RNG.choice(len(visits), sample_n, replace=False)
    sample = visits.iloc[sample_index]
    ax3d.scatter(
        sample["ga"],
        sample["first_bmi"],
        sample["y"] * 100.0,
        s=7,
        c=sample["y"] * 100.0,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidths=0.15,
        alpha=0.55,
        rasterized=True,
    )
    ax3d.contour(
        ga_mesh,
        bmi_mesh,
        pred_supported,
        zdir="z",
        offset=z_floor,
        levels=8,
        cmap=cmap,
        linewidths=0.70,
    )
    closed_hull = np.vstack([hull_vertices, hull_vertices[0]])
    ax3d.plot(closed_hull[:, 0], closed_hull[:, 1], z_floor, color=INK, lw=1.0, ls=(0, (3, 2)))
    ax3d.set_xlabel("孕周（周）", labelpad=6)
    ax3d.set_ylabel("基线 BMI", labelpad=7)
    ax3d.set_zlabel("预测 Y 浓度（%）", labelpad=5)
    ax3d.set_xlim(10, 25)
    ax3d.set_ylim(float(data["first_bmi"].min()), float(data["first_bmi"].max()))
    ax3d.set_zlim(z_floor, max(14.0, vmax + 0.4))
    ax3d.view_init(elev=29, azim=-136)
    ax3d.set_box_aspect((1.15, 1.0, 0.72))
    ax3d.set_title("(a) 支持域内模型响应面与底面等值线", pad=8)
    ax3d.text(10.7, float(data["first_bmi"].max()) - 0.4, 4.18, "4% 阈值面", color=RED, fontsize=7.6)

    attained = data["y"].to_numpy(float) >= 0.04
    ax_cloud.scatter(
        data.loc[attained, "ga"],
        data.loc[attained, "first_bmi"],
        data.loc[attained, "y"] * 100.0,
        s=8.5,
        c=BLUE,
        marker="o",
        alpha=0.34,
        edgecolors="none",
        rasterized=True,
    )
    ax_cloud.scatter(
        data.loc[~attained, "ga"],
        data.loc[~attained, "first_bmi"],
        data.loc[~attained, "y"] * 100.0,
        s=12,
        c=RED,
        marker="x",
        alpha=0.74,
        linewidths=0.65,
        rasterized=True,
    )
    threshold_cloud = np.full_like(ga_mesh, 4.0)
    ax_cloud.plot_surface(
        ga_mesh[::5, ::5],
        bmi_mesh[::5, ::5],
        threshold_cloud[::5, ::5],
        color=GOLD,
        alpha=0.12,
        linewidth=0,
        rasterized=True,
    )
    ax_cloud.set_xlabel("孕周（周）", labelpad=6)
    ax_cloud.set_ylabel("基线 BMI", labelpad=7)
    ax_cloud.set_zlabel("观测 Y 浓度（%）", labelpad=5)
    ax_cloud.set_xlim(10, 25)
    ax_cloud.set_ylim(float(data["first_bmi"].min()), float(data["first_bmi"].max()))
    ax_cloud.set_zlim(0, float(np.ceil(data["y"].max() * 100.0 / 2.0) * 2.0))
    ax_cloud.view_init(elev=27, azim=-56)
    ax_cloud.set_box_aspect((1.12, 1.0, 0.75))
    ax_cloud.set_title(f"(b) 采血事件观测点云与 4% 阈值（n={len(data)}）", pad=8)
    ax_cloud.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=5.5, label="达到4%"),
            Line2D([0], [0], marker="x", color=RED, markersize=5.5, label="未达到4%"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fontsize=7.1,
    )

    binned = data.loc[data["ga"] >= 11.0].copy()
    binned["week"] = np.floor(binned["ga"]).astype(int)
    binned["bmi_group"] = pd.cut(
        binned["first_bmi"],
        [-np.inf, 31.0, 33.5, 36.0, np.inf],
        labels=[1, 2, 3, 4],
        right=False,
    ).astype(int)
    cell = binned.groupby(["week", "bmi_group"], observed=True).agg(
        n=("y", "size"),
        attainment=("y", lambda s: float(np.mean(np.asarray(s) >= 0.04))),
    ).reset_index()
    xpos = cell["week"].to_numpy(float) - 0.34
    ypos = cell["bmi_group"].to_numpy(float) - 0.29
    rates = cell["attainment"].to_numpy(float)
    bar_colors = mpl.colormaps["plasma"](Normalize(0.0, 1.0)(rates))
    ax_bars.bar3d(
        xpos,
        ypos,
        np.zeros(len(cell)),
        np.full(len(cell), 0.68),
        np.full(len(cell), 0.58),
        rates,
        color=bar_colors,
        edgecolor=mpl.colors.to_rgba("white", 0.80),
        linewidth=0.35,
        shade=True,
        alpha=0.94,
    )
    sparse = cell["n"].to_numpy(int) < 5
    ax_bars.scatter(
        cell.loc[sparse, "week"],
        cell.loc[sparse, "bmi_group"],
        cell.loc[sparse, "attainment"] + 0.035,
        marker="x",
        s=19,
        c=INK,
        linewidths=0.8,
        depthshade=False,
        label="n<5",
    )
    ax_bars.set_xlabel("整孕周", labelpad=6)
    ax_bars.set_ylabel("BMI 政策组", labelpad=7)
    ax_bars.set_zlabel("组内达标率", labelpad=5)
    ax_bars.set_xlim(10.5, 24.8)
    ax_bars.set_ylim(0.5, 4.7)
    ax_bars.set_zlim(0, 1.05)
    ax_bars.set_xticks([11, 13, 15, 17, 19, 21, 23])
    ax_bars.set_yticks([1, 2, 3, 4])
    ax_bars.set_yticklabels(["<31", "31–33.5", "33.5–36", "≥36"])
    ax_bars.view_init(elev=28, azim=-59)
    ax_bars.set_box_aspect((1.25, 0.78, 0.76))
    ax_bars.set_title("(c) 整孕周×BMI 组达标率柱阵", pad=8)
    ax_bars.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=7.1)

    for axis3d in (ax3d, ax_cloud, ax_bars):
        axis3d.tick_params(labelsize=7.2, pad=0.5)
        for axis in (axis3d.xaxis, axis3d.yaxis, axis3d.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor("#CBD2DC")
            axis._axinfo["grid"]["color"] = (0.72, 0.75, 0.80, 0.44)
            axis._axinfo["grid"]["linewidth"] = 0.48

    save_figure(fig, "fig03_q1_response_surface")


def figure_q1_evidence() -> None:
    coef = pd.read_csv(RESULTS / "q1_final_coefficients.csv").set_index("term")
    order = ["ga_c", "first_bmi_c", "delta_bmi", "ga_hinge_18", "ga_slope_after_18_total"]
    labels = ["18 周前孕周斜率", "基线 BMI", "孕期内 BMI 变化", "18 周后新增斜率", "18 周后总斜率"]
    colors = [BLUE, RED, MUTED, PURPLE, TEAL]

    summary = pd.read_csv(RESULTS / "q1_cv_seed_summary.csv", header=[0, 1])
    summary.columns = pd.MultiIndex.from_tuples(
        [("model", "name"), *list(summary.columns[1:])]
    )
    model_names = summary[("model", "name")].astype(str)
    rmse = summary[("rmse", "mean")].astype(float)
    mae = summary[("mae", "mean")].astype(float)
    r2 = summary[("r2", "mean")].astype(float)
    model_label = {
        "mixed_hinge18": "18 周分段混合模型",
        "mixed_linear": "线性混合模型",
        "ridge": "岭回归",
        "random_forest": "随机森林",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.75), gridspec_kw={"width_ratios": [1.0, 1.05]})
    fig.subplots_adjust(left=0.155, right=0.95, bottom=0.15, top=0.91, wspace=0.34)
    ax = axes[0]
    y_pos = np.arange(len(order))[::-1]
    for y, term, color in zip(y_pos, order, colors):
        row = coef.loc[term]
        mid = float(row["bootstrap_median"])
        lo = float(row["bootstrap_q025"])
        hi = float(row["bootstrap_q975"])
        ax.errorbar(mid, y, xerr=[[mid - lo], [hi - mid]], fmt="o", ms=6.2, color=color, ecolor=color, capsize=3, lw=1.5)
        ax.text(hi + 0.003, y, f"{mid:+.3f}", color=color, fontsize=7.8, va="center")
    ax.axvline(0, color=INK, lw=0.9, ls=(0, (4, 3)))
    ax.set_yticks(y_pos, labels)
    ax.set_xlabel("logit 尺度系数（500 次孕妇簇重抽样 95% 区间）")
    ax.set_title("(a) 纵向模型参数的方向与稳定性")
    ax.set_xlim(-0.069, 0.088)
    style_axis(ax, grid_axis="x")

    ax2 = axes[1]
    norm = Normalize(vmin=-0.10, vmax=0.15)
    cmap = mpl.colormaps["viridis"]
    markers = {"mixed_linear": "o", "mixed_hinge18": "s", "ridge": "D", "random_forest": "*"}
    for name, x, y, score in zip(model_names, rmse, mae, r2):
        ax2.scatter(
            x,
            y,
            s=165 if name == "random_forest" else 78,
            marker=markers.get(name, "o"),
            color=cmap(norm(score)),
            edgecolor=INK,
            linewidth=0.75,
            zorder=3,
        )
        text_offsets = {
            "mixed_linear": (-12, 17),
            "mixed_hinge18": (-12, -24),
            "ridge": (7, 13),
            "random_forest": (7, 14),
        }
        arrow = None
        if name in {"mixed_linear", "mixed_hinge18"}:
            arrow = dict(arrowstyle="-", color="#7B8492", lw=0.65)
        ax2.annotate(
            model_label.get(name, name),
            xy=(x, y),
            xytext=text_offsets.get(name, (7, 10)),
            textcoords="offset points",
            fontsize=7.5,
            ha="right" if name in {"mixed_linear", "mixed_hinge18"} else "left",
            va="center",
            arrowprops=arrow,
        )
    ridge_idx = int(np.flatnonzero(model_names.to_numpy() == "ridge")[0])
    rf_idx = int(np.flatnonzero(model_names.to_numpy() == "random_forest")[0])
    arrow = FancyArrowPatch(
        (rmse.iloc[ridge_idx], mae.iloc[ridge_idx]),
        (rmse.iloc[rf_idx], mae.iloc[rf_idx]),
        arrowstyle="-|>",
        mutation_scale=12,
        color=ORANGE,
        lw=1.45,
        connectionstyle="arc3,rad=-0.18",
    )
    ax2.add_patch(arrow)
    ax2.text(0.03155, 0.02555, "RMSE 降低 4.76%\nMAE 降低 7.04%", color=ORANGE, fontsize=8.1, ha="center")
    ax2.set_xlabel("样本外 RMSE（越小越好）")
    ax2.set_ylabel("样本外 MAE（越小越好）")
    ax2.set_title("(b) 预测误差平面与机器学习增量")
    ax2.set_xlim(0.03045, 0.03425)
    ax2.set_ylim(0.0240, 0.0280)
    style_axis(ax2)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.045, pad=0.03)
    cbar.set_label("样本外 $R^2$", fontsize=8.1)
    cbar.ax.tick_params(labelsize=7.2)
    save_figure(fig, "fig04_q1_parameter_ml_evidence")


def _draw_half_violin(
    ax: plt.Axes,
    values: np.ndarray,
    x: float,
    side: str,
    color: str,
    width: float = 0.28,
) -> None:
    grid = np.linspace(max(10.0, float(np.nanmin(values)) - 0.8), min(43.0, float(np.nanmax(values)) + 0.8), 260)
    density = kde_1d(values, grid, bw_adjust=0.86)
    if density.max() > 0:
        density = density / density.max() * width
    if side == "left":
        ax.fill_betweenx(grid, x - density, x, facecolor=mpl.colors.to_rgba(color, 0.28), edgecolor=color, lw=1.1)
    else:
        ax.fill_betweenx(grid, x, x + density, facecolor=mpl.colors.to_rgba(color, 0.28), edgecolor=color, lw=1.1)


def figure_q2_tau_raincloud() -> None:
    policy = pd.read_csv(RESULTS / "q2_final_policies.csv")
    tau_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for rho in (80, 90):
        archive = np.load(STABILITY / f"q2_tau_measurement_rho{rho}.npz")
        q95 = np.quantile(archive["tau"], 0.95, axis=0)
        tau_data[rho] = (q95, archive["bmi"])

    fig, ax = plt.subplots(figsize=(9.6, 5.55))
    fig.subplots_adjust(left=0.10, right=0.975, bottom=0.17, top=0.95)
    ax.axhspan(10, 12, color="#DDEFD9", alpha=0.48, zorder=0)
    ax.axhspan(12, 25, color="#F9EDC8", alpha=0.40, zorder=0)
    ax.axhspan(25, 42, color="#F5D9D5", alpha=0.42, zorder=0)
    ax.axhline(12, color=GREEN, lw=1.0, ls=(0, (4, 3)))
    ax.axhline(25, color=RED, lw=1.15, ls=(0, (5, 3)))

    cuts = [-np.inf, 31, 33.5, 36, np.inf]
    x_positions = np.arange(1, 5)
    rng = np.random.default_rng(SEED + 41)
    for group, x in enumerate(x_positions, start=1):
        for rho, side, offset, color in [(80, "left", -0.055, BLUE), (90, "right", 0.055, ORANGE)]:
            q95, bmi = tau_data[rho]
            mask = (bmi >= cuts[group - 1]) & (bmi < cuts[group])
            values = q95[mask]
            _draw_half_violin(ax, values, x, side, color)
            point_x = x + offset + rng.normal(0, 0.018, values.size)
            ax.scatter(
                point_x,
                values,
                s=11,
                facecolor=mpl.colors.to_rgba(color, 0.34),
                edgecolor=mpl.colors.to_rgba("white", 0.8),
                linewidth=0.2,
                rasterized=True,
                zorder=3,
            )
            box_x = x - 0.09 if rho == 80 else x + 0.09
            bp = ax.boxplot(values, positions=[box_x], widths=0.075, patch_artist=True, showfliers=False, whis=(5, 95), zorder=4)
            for box in bp["boxes"]:
                box.set(facecolor="white", edgecolor=color, linewidth=1.0)
            for key in ("whiskers", "caps", "medians"):
                for item in bp[key]:
                    item.set(color=color, linewidth=1.0)

        for rho, star_x, color in [(0.8, x - 0.19, BLUE), (0.9, x + 0.19, ORANGE)]:
            row = policy[(policy["rho"] == rho) & (policy["group"] == group)].iloc[0]
            t = float(row["recommended_week"])
            ax.scatter(star_x, t, marker="*", s=125, color=color, edgecolor=INK, linewidth=0.55, zorder=6)
            if rho == 0.9 and group == 4:
                ax.text(star_x + 0.04, t + 0.5, "统一时点超窗\n转个体化复检", color=RED, fontsize=7.8, va="bottom")

    labels = ["BMI<31\n$n=123$", "31≤BMI<33.5\n$n=82$", "33.5≤BMI<36\n$n=42$", "BMI≥36\n$n=20$"]
    ax.set_xticks(x_positions, labels)
    ax.set_xlim(0.55, 4.65)
    ax.set_ylim(10, 42)
    ax.set_ylabel("个体 95% 保守达标时点（周）")
    ax.set_xlabel("固定 BMI 分组")
    ax.text(4.62, 11.5, "低风险边界", color=GREEN, fontsize=7.8, ha="right")
    ax.text(4.62, 25.5, "25 周常规窗口", color=RED, fontsize=7.8, ha="right")
    style_axis(ax, grid_axis="y")
    handles = [
        Line2D([0], [0], color=BLUE, lw=6, alpha=0.5, label=r"$\rho=0.80$"),
        Line2D([0], [0], color=ORANGE, lw=6, alpha=0.5, label=r"$\rho=0.90$"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="white", markeredgecolor=INK, markersize=10, label="组推荐时点"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=3, columnspacing=1.5)
    save_figure(fig, "fig05_q2_tau_raincloud")


def _node_intervals(counts: list[int], top: float = 0.86, bottom: float = 0.10, gap: float = 0.035) -> list[tuple[float, float]]:
    scale = (top - bottom - gap * (len(counts) - 1)) / sum(counts)
    result: list[tuple[float, float]] = []
    cursor = top
    for count in counts:
        height = count * scale
        result.append((cursor - height, cursor))
        cursor -= height + gap
    return result


def _flow_patch(x0: float, x1: float, a: tuple[float, float], b: tuple[float, float], color: str) -> PathPatch:
    y0_low, y0_high = a
    y1_low, y1_high = b
    c0 = x0 + (x1 - x0) * 0.42
    c1 = x0 + (x1 - x0) * 0.58
    vertices = [
        (x0, y0_low),
        (c0, y0_low),
        (c1, y1_low),
        (x1, y1_low),
        (x1, y1_high),
        (c1, y1_high),
        (c0, y0_high),
        (x0, y0_high),
        (x0, y0_low),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(MplPath(vertices, codes), facecolor=mpl.colors.to_rgba(color, 0.43), edgecolor=mpl.colors.to_rgba(color, 0.70), lw=0.65)


def figure_q2_policy_alluvial() -> None:
    counts = [123, 82, 42, 20]
    intervals = _node_intervals(counts)
    x_nodes = [0.10, 0.50, 0.90]
    stage_labels = ["基线 BMI 分组", r"平衡方案 $\rho=0.80$", r"保守方案 $\rho=0.90$"]
    group_text = ["BMI<31", "31≤BMI<33.5", "33.5≤BMI<36", "BMI≥36"]
    time80 = ["12周+6天", "14周", "15周+4天", "19周+6天"]
    time90 = ["16周+2天", "17周+6天", "20周+2天", "个体化复检"]

    fig, ax = plt.subplots(figsize=(10.0, 4.55))
    fig.subplots_adjust(left=0.035, right=0.97, bottom=0.08, top=0.92)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    node_w = 0.025

    for stage in range(2):
        for idx, color in enumerate(GROUP_COLORS):
            ax.add_patch(_flow_patch(x_nodes[stage] + node_w, x_nodes[stage + 1], intervals[idx], intervals[idx], color))

    for stage, x in enumerate(x_nodes):
        ax.text(x + node_w / 2, 0.955, stage_labels[stage], ha="center", va="bottom", fontsize=10.2, fontweight="bold", color=INK)
        for idx, ((y0, y1), color) in enumerate(zip(intervals, GROUP_COLORS)):
            edge = RED if stage == 2 and idx == 3 else INK
            rect = Rectangle((x, y0), node_w, y1 - y0, facecolor=color, edgecolor=edge, lw=1.1, zorder=4)
            ax.add_patch(rect)
            center = (y0 + y1) / 2
            if stage == 0:
                text = f"{group_text[idx]}  ·  {counts[idx]}人（{counts[idx] / 267:.1%}）"
                ax.text(x - 0.012, center, text, ha="right", va="center", fontsize=8.2, color=INK)
            elif stage == 1:
                ax.text(x + 0.037, center, time80[idx], ha="left", va="center", fontsize=8.3, color=INK, bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.7))
            else:
                ax.text(x + 0.037, center, time90[idx], ha="left", va="center", fontsize=8.3, color=RED if idx == 3 else INK)

    ax.annotate(
        "可靠度提高",
        xy=(0.83, 0.925),
        xytext=(0.58, 0.925),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.35),
        color=ORANGE,
        ha="center",
        va="center",
        fontsize=8.4,
    )
    ax.text(0.49, 0.025, "流带宽度与孕妇人数成比例；同一流带表示相同 BMI 组在两档可靠度下的排程变化", ha="center", fontsize=8.0, color=MUTED)
    save_figure(fig, "fig06_q2_policy_alluvial")


def figure_q2_optimization_sensitivity() -> None:
    costs = pd.read_csv(RESULTS / "q2_k_cost_curves.csv")
    sens = pd.read_csv(RESULTS / "q2_one_factor_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.65), gridspec_kw={"width_ratios": [1.0, 1.08]})
    fig.subplots_adjust(left=0.09, right=0.965, bottom=0.15, top=0.91, wspace=0.30)

    ax = axes[0]
    for rho, color, marker in [(0.8, BLUE, "o"), (0.9, ORANGE, "s")]:
        sub = costs[(costs["mode"] == "measurement") & (costs["rho"] == rho) & (costs["tau_type"] == "bootstrap_median")].sort_values("groups")
        x = sub["groups"].to_numpy(float)
        y = sub["fraction_of_max_reduction"].to_numpy(float) * 100
        marginal = np.r_[0, np.diff(y)]
        sizes = 45 + 3.1 * marginal
        ax.plot(x, y, color=color, lw=1.65, alpha=0.92)
        ax.scatter(x, y, s=sizes, marker=marker, color=color, edgecolor="white", linewidth=0.65, zorder=3, label=fr"$\rho={rho:.2f}$")
        selected = sub[sub["groups"] == 4].iloc[0]
        ax.scatter(4, selected["fraction_of_max_reduction"] * 100, marker="*", s=190, color=GOLD, edgecolor=INK, linewidth=0.8, zorder=5)
        ax.text(4.08, selected["fraction_of_max_reduction"] * 100 - (4.2 if rho == 0.8 else -1.4), f"{selected['fraction_of_max_reduction'] * 100:.2f}%", color=color, fontsize=8.0)
    ax.axvspan(4.5, 6.25, color="#ECEEF2", alpha=0.65, zorder=0)
    ax.text(5.35, 16, "新增切点的\n边际收益下降", ha="center", color=MUTED, fontsize=8.0)
    ax.set_xticks(range(1, 7))
    ax.set_xlim(0.8, 6.2)
    ax.set_ylim(-2, 104)
    ax.set_xlabel("连续 BMI 分组数 $K$")
    ax.set_ylabel("相对最大可实现损失降幅（%）")
    ax.set_title("(a) 分组数—决策损失帕累托前沿")
    style_axis(ax)
    ax.legend(loc="lower right")

    ax2 = axes[1]
    hard = sens[(sens["mode"] == "hard") & (sens["rho"] == 0.8) & (sens["threshold"].isin([0.035, 0.04, 0.045]))].drop_duplicates("threshold").sort_values("threshold")
    thresholds = hard["threshold"].to_numpy(float) * 100
    time_matrix = np.vstack([np.asarray(ast.literal_eval(x), dtype=float) for x in hard["times"]])
    for group in range(4):
        values = time_matrix[:, group]
        ax2.plot(thresholds, values, color=GROUP_COLORS[group], lw=1.85, marker="o", ms=5.2)
        ax2.text(4.54, values[-1], f"第{group + 1}组  {values[-1]:.2f}周", color=GROUP_COLORS[group], fontsize=7.7, va="center")
    ax2.axvline(4.0, color=INK, lw=0.9, ls=(0, (4, 3)))
    ax2.axhline(12, color=GREEN, lw=0.9, ls=(0, (3, 3)))
    ax2.fill_between([3.45, 4.55], 10, 12, color="#DDEFD9", alpha=0.26)
    ax2.set_xlim(3.45, 4.84)
    ax2.set_ylim(9.5, 19.4)
    ax2.set_xticks([3.5, 4.0, 4.5])
    ax2.set_xlabel("Y 染色体达标阈值（%）")
    ax2.set_ylabel("点事件组时点（周）")
    ax2.set_title("(b) 阈值变化的组时点斜率图")
    style_axis(ax2)
    ax2.text(4.02, 9.85, "主阈值", color=INK, fontsize=7.7, ha="left")
    save_figure(fig, "fig07_q2_optimization_sensitivity")


def _read_two_level_summary(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, header=[0, 1])
    data.columns = pd.MultiIndex.from_tuples(
        [("model", "name"), *list(data.columns[1:])]
    )
    return data


def figure_q3_increment() -> None:
    aft = _read_two_level_summary(RESULTS / "q3_aft_cv_seed_summary.csv")
    aft_inc = pd.read_csv(RESULTS / "q3_aft_paired_increment.csv")
    ml_inc = pd.read_csv(RESULTS / "q3_ml_increment.csv")
    ml_sum = _read_two_level_summary(RESULTS / "q3_ml_seed_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), gridspec_kw={"width_ratios": [1.0, 1.22]})
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.16, top=0.91, wspace=0.31)
    ax = axes[0]
    names = aft[("model", "name")].astype(str)
    x = aft[("nll", "mean")].astype(float)
    y = aft[("mean_brier", "mean")].astype(float)
    xerr = aft[("nll", "std")].astype(float)
    yerr = aft[("mean_brier", "std")].astype(float)
    label_map = {
        "bmi_only": "BMI-only",
        "bmi_age_height": "BMI+年龄+身高",
        "bmi_full": "BMI 全因素",
        "weight_height_full": "体重+身高全因素",
    }
    for name, xi, yi, xe, ye in zip(names, x, y, xerr, yerr):
        selected = name == "bmi_only"
        color = TEAL if selected else ORANGE
        ax.errorbar(xi, yi, xerr=xe, yerr=ye, fmt="*" if selected else "o", ms=11 if selected else 6, color=color, ecolor=mpl.colors.to_rgba(color, 0.55), capsize=2.5, lw=1.0, zorder=3)
        label_offsets = {
            "bmi_only": (7, -13),
            "bmi_age_height": (8, 7),
            "bmi_full": (8, 17),
            "weight_height_full": (-6, -18),
        }
        ax.annotate(
            label_map.get(name, name),
            xy=(xi, yi),
            xytext=label_offsets.get(name, (7, 8)),
            textcoords="offset points",
            color=color if selected else INK,
            fontsize=7.4,
            ha="left" if name != "weight_height_full" else "right",
            va="center",
            arrowprops=dict(arrowstyle="-", color=mpl.colors.to_rgba(color, 0.65), lw=0.55)
            if name in {"bmi_full", "weight_height_full"}
            else None,
        )
    base_x = float(x[names == "bmi_only"].iloc[0])
    base_y = float(y[names == "bmi_only"].iloc[0])
    ax.annotate("左下方同时降低两类误差", xy=(base_x - 0.002, base_y - 0.0005), xytext=(0.735, 0.109), arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.0), color=TEAL, fontsize=7.8)
    ax.set_xlabel("样本外负对数似然 NLL（越小越好）")
    ax.set_ylabel("多时点 Brier 分数（越小越好）")
    ax.set_title("(a) 多因素 AFT 的误差平面")
    ax.set_xlim(0.67, 0.82)
    ax.set_ylim(0.0995, 0.1115)
    style_axis(ax)

    rows: list[dict[str, float | str]] = []
    base_metric = {"nll": 0.6908544076848926, "mean_brier": 0.102167476650364}
    aft_names = {"bmi_age_height": "AFT：+年龄+身高", "bmi_full": "AFT：+全部因素", "weight_height_full": "AFT：体重+身高替代"}
    metric_names = {"nll": "NLL", "mean_brier": "Brier"}
    for _, row in aft_inc.iterrows():
        base = base_metric[row["metric"]]
        rows.append(
            {
                "label": f"{aft_names[row['candidate']]} · {metric_names[row['metric']]}",
                "mean": 100 * row["improvement_mean"] / base,
                "low": 100 * row["improvement_q025"] / base,
                "high": 100 * row["improvement_q975"] / base,
                "family": "统计时间模型",
            }
        )
    ml_baseline = {
        "roc_auc": float(ml_sum.loc[ml_sum[("model", "name")] == "rf_bmi", ("roc_auc", "mean")].iloc[0]),
        "pr_auc": float(ml_sum.loc[ml_sum[("model", "name")] == "rf_bmi", ("pr_auc", "mean")].iloc[0]),
        "brier": float(ml_sum.loc[ml_sum[("model", "name")] == "rf_bmi", ("brier", "mean")].iloc[0]),
    }
    ml_metric_names = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier"}
    for _, row in ml_inc[ml_inc["model_type"] == "rf"].iterrows():
        base = ml_baseline[row["metric"]]
        rows.append(
            {
                "label": f"随机森林：全特征 · {ml_metric_names[row['metric']]}",
                "mean": 100 * row["improvement_mean"] / base,
                "low": 100 * row["q025"] / base,
                "high": 100 * row["q975"] / base,
                "family": "访视层机器学习",
            }
        )
    frame = pd.DataFrame(rows)
    y_pos = np.arange(len(frame))[::-1]
    ax2 = axes[1]
    for yp, (_, row) in zip(y_pos, frame.iterrows()):
        color = ORANGE if row["family"] == "统计时间模型" else TEAL
        ax2.errorbar(row["mean"], yp, xerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]], fmt="o", ms=5.4, color=color, ecolor=color, capsize=2.5, lw=1.2)
        ax2.text(row["high"] + 0.7, yp, f"{row['mean']:+.1f}%", color=color, va="center", fontsize=7.4)
    ax2.axvline(0, color=INK, lw=0.9, ls=(0, (4, 3)))
    ax2.axvspan(-20, 0, color="#F6E4DF", alpha=0.34, zorder=0)
    ax2.axvspan(0, 25, color="#DDEFEA", alpha=0.34, zorder=0)
    ax2.set_yticks(y_pos, frame["label"])
    ax2.set_xlabel("相对基准的样本外改善（%，向右为改善）")
    ax2.set_title("(b) 配对增量及其 95% 区间")
    ax2.set_xlim(-20, 25)
    style_axis(ax2, grid_axis="x")
    ax2.text(-19.2, len(frame) - 0.35, "时间分布主线", color=ORANGE, fontsize=8.0)
    ax2.text(8.0, 2.55, "非线性辅助线", color=TEAL, fontsize=8.0)
    save_figure(fig, "fig08_q3_increment_evidence")


def figure_q4_upset() -> None:
    combos = pd.read_csv(RESULTS / "q4_label_combinations.csv")
    desired = ["normal", "T18", "T13+T18", "T13", "T21", "T18+T21", "T13+T21"]
    combos["order"] = combos["label_combination"].map({name: i for i, name in enumerate(desired)})
    combos = combos.sort_values("order").reset_index(drop=True)
    x = np.arange(len(combos))
    counts = combos["records"].to_numpy(int)
    sets = ["T13", "T18", "T21"]
    set_sizes = [int(sum(count for combo, count in zip(combos["label_combination"], counts) if target in combo.split("+"))) for target in sets]

    fig = plt.figure(figsize=(9.7, 5.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.23, 1.0], height_ratios=[0.64, 0.36], left=0.08, right=0.975, bottom=0.11, top=0.95, wspace=0.08, hspace=0.06)
    ax_summary = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_set = fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_bar)

    heights = np.sqrt(counts)
    bar_colors = ["#9AA3AE" if name == "normal" else TARGET_COLORS.get(name, MAGENTA) for name in combos["label_combination"]]
    for i, name in enumerate(combos["label_combination"]):
        if "+" in name:
            bar_colors[i] = GOLD if name == "T13+T18" else PURPLE
    bars = ax_bar.bar(x, heights, color=bar_colors, edgecolor=INK, linewidth=0.6, width=0.70)
    for rect, value in zip(bars, counts):
        ax_bar.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.35, str(value), ha="center", va="bottom", fontsize=8.2, color=INK)
    tick_counts = [0, 5, 10, 25, 50, 100, 250, 538]
    ax_bar.set_yticks(np.sqrt(tick_counts), [str(v) for v in tick_counts])
    ax_bar.set_ylabel("交集记录数（平方根尺度）")
    ax_bar.tick_params(axis="x", labelbottom=False)
    ax_bar.set_ylim(0, np.sqrt(counts.max()) + 2.4)
    style_axis(ax_bar, grid_axis="y")

    ax_summary.axis("off")
    ax_summary.text(0.52, 0.73, "67 / 605", ha="center", va="center", fontsize=19, color=RED, fontweight="bold")
    ax_summary.text(0.52, 0.58, "总体异常记录", ha="center", fontsize=8.5, color=INK)
    ax_summary.text(0.52, 0.42, "11.1%", ha="center", fontsize=14, color=ORANGE, fontweight="bold")
    ax_summary.text(0.52, 0.29, "记录级异常率", ha="center", fontsize=8.2, color=MUTED)

    y_rows = np.arange(3)
    ax_set.barh(y_rows, set_sizes, color=[CYAN, ORANGE, PURPLE], edgecolor=INK, linewidth=0.6, height=0.58)
    for yv, value in zip(y_rows, set_sizes):
        ax_set.text(value - 0.8, yv, str(value), ha="right", va="center", color="white", fontsize=8.2, fontweight="bold")
    ax_set.set_yticks(y_rows, sets)
    ax_set.invert_xaxis()
    ax_set.set_xlabel("集合规模")
    ax_set.spines[["top", "right", "left"]].set_visible(False)
    ax_set.grid(axis="x", color="#D8DEE8", lw=0.5, alpha=0.5)
    ax_set.tick_params(axis="y", length=0)

    ax_matrix.set_ylim(-0.6, 2.6)
    ax_matrix.set_yticks(y_rows, sets)
    ax_matrix.tick_params(axis="y", labelleft=False, length=0)
    ax_matrix.set_xticks(x, ["无异常", "T18", "T13∩T18", "T13", "T21", "T18∩T21", "T13∩T21"], rotation=0)
    for idx, combo in enumerate(combos["label_combination"]):
        active = []
        components = combo.split("+") if combo != "normal" else []
        for row, target in enumerate(sets):
            is_active = target in components
            ax_matrix.scatter(idx, row, s=52 if is_active else 30, color=INK if is_active else "#E2E5EA", zorder=3)
            if is_active:
                active.append(row)
        if len(active) >= 2:
            ax_matrix.plot([idx, idx], [min(active), max(active)], color=INK, lw=1.35, zorder=2)
    for row in y_rows:
        ax_matrix.axhspan(row - 0.46, row + 0.46, color="#F6F7F9" if row % 2 == 0 else "white", zorder=0)
    ax_matrix.spines[["top", "right", "left"]].set_visible(False)
    ax_matrix.spines["bottom"].set_color(INK)
    ax_matrix.grid(False)
    save_figure(fig, "fig09_q4_label_upset")


def _weighted_pca(data: pd.DataFrame, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = data.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median())
    matrix = x.to_numpy(float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    mean = np.sum(matrix * weights[:, None], axis=0)
    centered = matrix - mean
    scale = np.sqrt(np.sum(centered**2 * weights[:, None], axis=0))
    scale[scale < 1e-10] = 1.0
    standardized = centered / scale
    cov = (standardized * weights[:, None]).T @ standardized
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    scores = standardized @ eigvecs[:, :2]
    ratio = eigvals / eigvals.sum()
    loadings = eigvecs[:, :2] * np.sqrt(eigvals[:2])
    return scores, ratio, loadings, standardized


def figure_q4_weighted_pca() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from run_q4_rebuild import load_bundle

        bundle = load_bundle(DATA_PATH)
    record = bundle.record.copy()
    core = bundle.feature_sets["record"]["core"]
    counts = record.groupby("code")["code"].transform("size").to_numpy(float)
    weights = 1.0 / counts
    scores, ratio, loadings, _ = _weighted_pca(record[core], weights)

    def combo(row: pd.Series) -> str:
        labels = [target for target in ("T13", "T18", "T21") if int(row[f"y_{target}"]) == 1]
        return "+".join(labels) if labels else "normal"

    record["combo"] = record.apply(combo, axis=1)
    record["pc1"] = scores[:, 0]
    record["pc2"] = scores[:, 1]
    record["weight"] = weights
    combo_order = ["normal", "T13", "T18", "T21", "T13+T18", "T18+T21", "T13+T21"]
    combo_colors = {
        "normal": "#A9AFB8",
        "T13": CYAN,
        "T18": ORANGE,
        "T21": PURPLE,
        "T13+T18": GOLD,
        "T18+T21": MAGENTA,
        "T13+T21": "#3B8D6B",
    }

    fig = plt.figure(figsize=(9.65, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.20], height_ratios=[0.20, 1.0], left=0.09, right=0.94, bottom=0.18, top=0.94, wspace=0.05, hspace=0.04)
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    fig.add_subplot(gs[0, 1]).axis("off")

    normal = record["combo"] == "normal"
    abnormal = ~normal
    size = 12 + 34 * record["weight"].to_numpy() / record["weight"].max()
    ax.scatter(record.loc[normal, "pc1"], record.loc[normal, "pc2"], s=size[normal], color=mpl.colors.to_rgba(combo_colors["normal"], 0.38), edgecolors="none", rasterized=True, label=f"无异常（{normal.sum()}）")
    for name in combo_order[1:]:
        mask = record["combo"] == name
        if not mask.any():
            continue
        ax.scatter(record.loc[mask, "pc1"], record.loc[mask, "pc2"], s=size[mask] + 10, color=mpl.colors.to_rgba(combo_colors[name], 0.80), edgecolors="white", linewidths=0.45, rasterized=True, label=f"{name}（{mask.sum()}）")

    xlim = np.quantile(record["pc1"], [0.005, 0.995])
    ylim = np.quantile(record["pc2"], [0.005, 0.995])
    xpad = 0.10 * (xlim[1] - xlim[0])
    ypad = 0.12 * (ylim[1] - ylim[0])
    xlim = (xlim[0] - xpad, xlim[1] + xpad)
    ylim = (ylim[0] - ypad, ylim[1] + ypad)
    gx = np.linspace(*xlim, 100)
    gy = np.linspace(*ylim, 100)
    xx, yy = np.meshgrid(gx, gy)
    for mask, color, linestyle in [(normal, MUTED, (0, (3, 2))), (abnormal, RED, "-")]:
        points = record.loc[mask, ["pc1", "pc2"]].to_numpy().T
        if points.shape[1] > 5:
            density = gaussian_kde(points)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contour(xx, yy, density, levels=[density.max() * 0.25, density.max() * 0.55], colors=[color], linewidths=[0.9, 1.25], linestyles=[linestyle, linestyle], alpha=0.72)

    contribution = np.linalg.norm(loadings, axis=1)
    top_features = np.argsort(contribution)[-7:][::-1]
    feature_names = {
        "z13": "$Z_{13}$",
        "z18": "$Z_{18}$",
        "z21": "$Z_{21}$",
        "zx": "$Z_X$",
        "x_concentration": "X 浓度",
        "gc": "总体 GC",
        "gc_abs_dev": "GC 偏离",
        "gcdev13": "13-GC 偏离",
        "gcdev18": "18-GC 偏离",
        "gcdev21": "21-GC 偏离",
        "unique_ratio": "唯一比对比例",
        "filter_rate": "过滤比例",
        "bmi": "BMI",
        "ga": "孕周",
        "log_unique_reads": "唯一读段（log）",
    }
    arrow_scale = 0.27 * min(xlim[1] - xlim[0], ylim[1] - ylim[0]) / max(np.max(np.abs(loadings[top_features])), 1e-8)
    for idx in top_features:
        dx, dy = loadings[idx] * arrow_scale
        ax.annotate("", xy=(dx, dy), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, alpha=0.80))
        ax.text(dx * 1.07, dy * 1.07, feature_names.get(core[idx], core[idx]), fontsize=7.2, color=INK, ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top")

    ax.axhline(0, color="#CFD4DC", lw=0.65)
    ax.axvline(0, color="#CFD4DC", lw=0.65)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"加权主成分 1（解释 {ratio[0] * 100:.1f}%）")
    ax.set_ylabel(f"加权主成分 2（解释 {ratio[1] * 100:.1f}%）")
    style_axis(ax, grid_axis="")
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, -0.14), ncol=4, columnspacing=1.15, handletextpad=0.35, fontsize=7.3)

    for mask, color, label in [(normal, MUTED, "无异常"), (abnormal, RED, "总体异常")]:
        vals_x = record.loc[mask, "pc1"].to_numpy(float)
        dx = kde_1d(vals_x, gx, weights=record.loc[mask, "weight"].to_numpy(float), bw_adjust=0.95)
        ax_top.plot(gx, dx, color=color, lw=1.45, label=label)
        ax_top.fill_between(gx, 0, dx, color=color, alpha=0.10)
        vals_y = record.loc[mask, "pc2"].to_numpy(float)
        dy = kde_1d(vals_y, gy, weights=record.loc[mask, "weight"].to_numpy(float), bw_adjust=0.95)
        ax_right.plot(dy, gy, color=color, lw=1.45)
        ax_right.fill_betweenx(gy, 0, dy, color=color, alpha=0.10)
    ax_top.legend(loc="upper right", ncol=2)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelleft=False, length=0)
    ax_top.spines[["top", "right", "left"]].set_visible(False)
    ax_top.spines["bottom"].set_color("#C8CED7")
    ax_right.tick_params(axis="y", labelleft=False, length=0)
    ax_right.tick_params(axis="x", labelbottom=False, length=0)
    ax_right.spines[["top", "right", "bottom"]].set_visible(False)
    ax_right.spines["left"].set_color("#C8CED7")
    save_figure(fig, "fig10_q4_weighted_pca")


def figure_q4_score_threshold_landscape() -> None:
    scores = pd.read_csv(Q4_LOCKED / "q4_locked_record_results.csv")
    selection = pd.read_csv(Q4_LOCKED / "q4_locked_model_selection.csv").set_index("target")
    with (Q4_LOCKED / "q4_locked_operational_models.json").open(encoding="utf-8") as handle:
        operational = json.load(handle)
    score_grid = np.linspace(0, 1, 360)
    order = ["ANY", "T13", "T18", "T21"]
    y_base = {target: 3 - idx for idx, target in enumerate(order)}
    fig, ax = plt.subplots(figsize=(10.0, 5.05))
    fig.subplots_adjust(left=0.10, right=0.965, bottom=0.14, top=0.93)

    for target in order:
        sub = scores[scores["target"] == target].copy()
        counts = sub.groupby("code")["code"].transform("size").to_numpy(float)
        weights = 1.0 / counts
        base = y_base[target]
        neg = sub["y"].to_numpy(int) == 0
        pos = ~neg
        d_neg = kde_1d(sub.loc[neg, "locked_score"].to_numpy(float), score_grid, weights=weights[neg], bw_adjust=0.83)
        d_pos = kde_1d(sub.loc[pos, "locked_score"].to_numpy(float), score_grid, weights=weights[pos], bw_adjust=0.83)
        scale = max(d_neg.max(), d_pos.max(), 1e-9)
        d_neg = d_neg / scale * 0.34
        d_pos = d_pos / scale * 0.34
        ax.fill_between(score_grid, base, base - d_neg, color=mpl.colors.to_rgba(BLUE, 0.30), edgecolor=BLUE, lw=1.0)
        ax.fill_between(score_grid, base, base + d_pos, color=mpl.colors.to_rgba(ORANGE, 0.34), edgecolor=ORANGE, lw=1.0)
        ax.plot(score_grid, base - d_neg, color=BLUE, lw=1.05)
        ax.plot(score_grid, base + d_pos, color=ORANGE, lw=1.05)
        rng = np.random.default_rng(SEED + len(target))
        pos_scores = sub.loc[pos, "locked_score"].to_numpy(float)
        ax.scatter(pos_scores, np.full_like(pos_scores, base + 0.37) + rng.normal(0, 0.012, pos_scores.size), s=8, color=mpl.colors.to_rgba(ORANGE, 0.55), edgecolors="none", rasterized=True)
        ax.axhline(base, color="#C7CDD6", lw=0.65)
        row = selection.loc[target]
        if target != "T21":
            entry = operational[target]
            threshold = float(entry.get("operational_threshold", entry.get("threshold")))
            ax.plot([threshold, threshold], [base - 0.38, base + 0.40], color=TARGET_COLORS[target], lw=1.6, ls=(0, (4, 2)))
            ax.scatter(threshold, base, marker="D", s=42, color=TARGET_COLORS[target], edgecolor=INK, linewidth=0.5, zorder=5)
            text = f"阈值 {threshold:.3f}   P={row['precision_w_100seed']:.3f}  R={row['recall_w_100seed']:.3f}  F1={row['f1_w_100seed']:.3f}"
        else:
            text = f"连续分数排序   随机划分 F1={row['f1_w_100seed']:.3f}   时间留出 F1={row['chrono_f1_w']:.3f}"
        ax.text(1.025, base, text, va="center", fontsize=7.8, color=TARGET_COLORS[target], clip_on=False)

    ax.set_xlim(0, 1.36)
    ax.set_ylim(-0.58, 3.58)
    ax.set_yticks([y_base[t] for t in order], ["总体异常 ANY", "T13", "T18", "T21"])
    for tick, target in zip(ax.get_yticklabels(), order):
        tick.set_color(TARGET_COLORS[target])
        tick.set_fontweight("bold")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("折外风险分数")
    style_axis(ax, grid_axis="x")
    handles = [
        Line2D([0], [0], color=ORANGE, lw=6, alpha=0.45, label="真实阳性"),
        Line2D([0], [0], color=BLUE, lw=6, alpha=0.40, label="真实阴性"),
        Line2D([0], [0], marker="D", color=INK, markerfacecolor="white", lw=0, label="最终阈值"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1.03), ncol=3)
    save_figure(fig, "fig11_q4_score_threshold_landscape")


def figure_q4_model_landscape() -> None:
    methods = pd.read_csv(Q4_LOCKED / "q4_locked_method_comparison.csv")
    selection = pd.read_csv(Q4_LOCKED / "q4_locked_model_selection.csv").set_index("target")
    boot = pd.read_csv(Q4_LOCKED / "q4_locked_bootstrap_summary.csv")
    methods = methods[np.isfinite(methods["f1_w_mean"]) & np.isfinite(methods["pr_auc_w_mean"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.05, 5.15), gridspec_kw={"width_ratios": [1.18, 0.82]})
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.92, wspace=0.28)
    ax = axes[0]
    marker_map = {"elastic": "o", "extra_trees": "D", "lightgbm": "s", "logistic": "^", "lda": "P"}
    for _, row in methods.iterrows():
        target = row["target"]
        locked = bool(row["locked"])
        size = 38 + 210 * max(float(row["recall_w_mean"]), 0)
        marker = marker_map.get(row["model"], "o")
        face = TARGET_COLORS[target] if locked else mpl.colors.to_rgba(TARGET_COLORS[target], 0.18)
        ax.scatter(row["pr_auc_w_mean"], row["f1_w_mean"], s=size, marker="*" if locked else marker, facecolor=face, edgecolor=INK if locked else mpl.colors.to_rgba(TARGET_COLORS[target], 0.75), linewidth=0.9 if locked else 0.65, zorder=4 if locked else 2)

    for target, color in TARGET_COLORS.items():
        row = selection.loc[target]
        b_f1 = boot[(boot["target"] == target) & (boot["metric"] == "f1_w")].iloc[0]
        b_pr = boot[(boot["target"] == target) & (boot["metric"] == "pr_auc_w")].iloc[0]
        x0 = float(row["pr_auc_w_100seed"])
        y0 = float(row["f1_w_100seed"])
        ax.errorbar(
            x0,
            y0,
            xerr=[[x0 - b_pr["q025"]], [b_pr["q975"] - x0]],
            yerr=[[y0 - b_f1["q025"]], [b_f1["q975"] - y0]],
            fmt="none",
            ecolor=mpl.colors.to_rgba(color, 0.60),
            capsize=2.2,
            lw=1.0,
            zorder=3,
        )
        offsets = {"ANY": (10, -18), "T13": (10, 11), "T18": (10, 14), "T21": (10, 8)}
        model_cn = {"elastic": "弹性网络", "extra_trees": "极端随机树", "lda": "收缩 LDA"}.get(row["final_model"], row["final_model"])
        ax.annotate(
            f"{target} · {model_cn}",
            xy=(x0, y0),
            xytext=offsets[target],
            textcoords="offset points",
            color=color,
            fontsize=7.5,
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color=mpl.colors.to_rgba(color, 0.68), lw=0.6)
            if target in {"ANY", "T18"}
            else None,
        )

    t13 = methods[methods["target"] == "T13"]
    locked_t13 = t13[t13["locked"]].iloc[0]
    lgb = t13[t13["model"] == "lightgbm"]
    if not lgb.empty:
        row = lgb.iloc[0]
        ax.annotate("T13 非线性增益", xy=(locked_t13["pr_auc_w_mean"], locked_t13["f1_w_mean"]), xytext=(row["pr_auc_w_mean"] + 0.025, row["f1_w_mean"] - 0.03), arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=1.2), color=CYAN, fontsize=7.7)

    ax.set_xlabel("孕妇等权 PR-AUC")
    ax.set_ylabel("孕妇等权 F1")
    ax.set_title("(a) 统计模型与机器学习候选的性能景观")
    ax.set_xlim(0.08, 0.56)
    ax.set_ylim(0.015, 0.56)
    style_axis(ax)
    model_handles = [Line2D([0], [0], marker=marker, color="none", markeredgecolor=INK, markerfacecolor="white", markersize=6, label=name) for name, marker in [("弹性网络", "o"), ("极端随机树", "D"), ("LightGBM", "s"), ("质量逻辑模型", "^"), ("收缩 LDA", "P")]]
    model_handles.append(Line2D([0], [0], marker="*", color="none", markeredgecolor=INK, markerfacecolor=GOLD, markersize=9, label="锁定方案"))
    ax.legend(handles=model_handles, loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.35, fontsize=7.2)

    ax2 = axes[1]
    x_pair = [0, 1]
    for offset, (target, color) in enumerate(TARGET_COLORS.items()):
        row = selection.loc[target]
        values = [float(row["f1_w_100seed"]), float(row["chrono_f1_w"])]
        ax2.plot(x_pair, values, color=color, lw=2.0, marker="o", ms=6.0, zorder=3)
        b_f1 = boot[(boot["target"] == target) & (boot["metric"] == "f1_w")].iloc[0]
        ax2.errorbar(0, values[0], yerr=[[values[0] - b_f1["q025"]], [b_f1["q975"] - values[0]]], fmt="none", ecolor=mpl.colors.to_rgba(color, 0.58), capsize=3, lw=1.1)
        ax2.text(-0.04, values[0], f"{target}  {values[0]:.3f}", color=color, ha="right", va="center", fontsize=7.8)
        ax2.text(1.04, values[1], f"{values[1]:.3f}", color=color, ha="left", va="center", fontsize=7.8)
    ax2.axhspan(0, 0.20, color="#F2E8E6", alpha=0.45)
    ax2.set_xticks(x_pair, ["100 组随机划分", "时间顺序留出"])
    ax2.set_ylabel("F1")
    ax2.set_title("(b) 随机划分与后续批次的外推对照")
    ax2.set_xlim(-0.35, 1.34)
    ax2.set_ylim(0, 0.66)
    style_axis(ax2, grid_axis="y")
    ax2.text(0.52, 0.585, "T13 在后续批次中上升\nT18 的批次敏感性更高", ha="center", fontsize=8.0, color=INK)
    save_figure(fig, "fig12_q4_model_stability")


def _draw_horizontal_raincloud(
    ax: plt.Axes,
    values: np.ndarray,
    y: float,
    color: str,
    rng: np.random.Generator,
    width: float = 0.32,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    span = max(float(np.ptp(values)), 0.01)
    grid = np.linspace(float(values.min() - 0.08 * span), float(values.max() + 0.08 * span), 260)
    density = kde_1d(values, grid, bw_adjust=0.86)
    density = density / max(float(density.max()), 1e-12) * width
    ax.fill_between(
        grid,
        y,
        y + density,
        color=mpl.colors.to_rgba(color, 0.30),
        edgecolor=color,
        linewidth=1.0,
        zorder=2,
    )
    jitter = y - 0.08 - rng.random(values.size) * 0.20
    ax.scatter(
        values,
        jitter,
        s=5.2,
        facecolor=mpl.colors.to_rgba(color, 0.18),
        edgecolor="none",
        rasterized=True,
        zorder=1,
    )
    q025, q25, median, q75, q975 = np.quantile(values, [0.025, 0.25, 0.5, 0.75, 0.975])
    ax.plot([q025, q975], [y - 0.01, y - 0.01], color=color, lw=1.25, zorder=4)
    ax.plot([q025, q025], [y - 0.055, y + 0.035], color=color, lw=1.0, zorder=4)
    ax.plot([q975, q975], [y - 0.055, y + 0.035], color=color, lw=1.0, zorder=4)
    ax.add_patch(
        Rectangle(
            (q25, y - 0.075),
            q75 - q25,
            0.13,
            facecolor="white",
            edgecolor=color,
            linewidth=1.05,
            zorder=5,
        )
    )
    ax.plot([median, median], [y - 0.075, y + 0.055], color=INK, lw=1.25, zorder=6)
    return float(q025), float(median), float(q975)


def _parse_bootstrap_vectors(series: pd.Series, expected: int) -> np.ndarray:
    rows = []
    for item in series:
        values = np.asarray(ast.literal_eval(str(item)), dtype=float)
        if values.size == expected and np.all(np.isfinite(values)):
            rows.append(values)
    if not rows:
        raise ValueError("未找到可用的 Bootstrap 向量。")
    return np.stack(rows)


def _fixed_policy_replicate_times(rho: float) -> tuple[np.ndarray, np.ndarray]:
    rho_key = int(round(rho * 100))
    archive = np.load(STABILITY / f"q2_tau_measurement_rho{rho_key}.npz")
    tau = np.asarray(archive["tau"], dtype=float)
    bmi = np.asarray(archive["bmi"], dtype=float)
    group_ids = np.searchsorted(np.array([31.0, 33.5, 36.0]), bmi, side="right")
    result = np.zeros((tau.shape[0], 4), dtype=float)
    for group in range(4):
        values = tau[:, group_ids == group]
        rank = int(np.ceil(rho * values.shape[1])) - 1
        result[:, group] = np.maximum(10.0, np.partition(values, rank, axis=1)[:, rank])
    return result, bmi


def figure_bootstrap_decision_stability() -> None:
    q1_boot = pd.read_csv(STABILITY / "q1_cluster_bootstrap.csv")
    q1_boot = q1_boot.loc[q1_boot["success"].astype(bool)].copy()
    q1_boot["beta_after_18_total"] = q1_boot["beta_ga_c"] + q1_boot["beta_ga_hinge_18"]
    q2_boot = pd.read_csv(STABILITY / "q2_bootstrap_measurement.csv")
    q2_boot = q2_boot.loc[q2_boot["success"].astype(bool)].copy()
    cuts = {
        0.80: _parse_bootstrap_vectors(q2_boot["rho80_cuts"], 3),
        0.90: _parse_bootstrap_vectors(q2_boot["rho90_cuts"], 3),
    }
    assignment_stability = pd.read_csv(RESULTS / "q2_group_assignment_stability.csv")
    policy = pd.read_csv(RESULTS / "q2_final_policies.csv")
    replicate_times = {
        0.80: _fixed_policy_replicate_times(0.80)[0],
        0.90: _fixed_policy_replicate_times(0.90)[0],
    }

    fig = plt.figure(figsize=(7.20, 5.75))
    grid = GridSpec(
        2,
        2,
        figure=fig,
        left=0.145,
        right=0.965,
        bottom=0.075,
        top=0.895,
        wspace=0.30,
        hspace=0.34,
        height_ratios=[0.95, 1.05],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[1, 1])
    c_grid = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=grid[1, 0],
        width_ratios=[1.0, 0.045],
        hspace=0.14,
        wspace=0.08,
    )
    ax_c80 = fig.add_subplot(c_grid[0, 0])
    ax_c90 = fig.add_subplot(c_grid[1, 0], sharex=ax_c80)
    cax = fig.add_subplot(c_grid[:, 1])
    fig.suptitle(
        "Bootstrap 决策稳定性传递：参数  →  切点  →  归组  →  排程",
        fontsize=11.8,
        y=0.967,
    )

    parameter_specs = [
        ("beta_ga_c", r"18 周前孕周效应  $\beta_1$", BLUE),
        ("beta_after_18_total", r"18 周后总斜率  $\beta_1+\beta_4$", PURPLE),
        ("beta_first_bmi_c", r"基线 BMI 效应  $\beta_2$", ORANGE),
        ("beta_delta_bmi", r"孕期内 BMI 变化  $\beta_3$", TEAL),
    ]
    rng_a = np.random.default_rng(SEED + 131)
    ax_a.axvline(0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=0)
    y_positions = np.arange(len(parameter_specs))[::-1]
    for y, (column, _label, color) in zip(y_positions, parameter_specs):
        _low, median, _high = _draw_horizontal_raincloud(
            ax_a,
            q1_boot[column].to_numpy(float),
            float(y),
            color,
            rng_a,
        )
        ax_a.scatter(median, y - 0.01, s=23, color=INK, edgecolor="white", linewidth=0.45, zorder=7)
    ax_a.set_yticks(y_positions, [item[1] for item in parameter_specs])
    ax_a.set_xlim(-0.082, 0.092)
    ax_a.set_ylim(-0.40, 3.45)
    ax_a.set_xlabel("对数几率尺度系数")
    ax_a.set_title("(a) 参数 Bootstrap 雨云分布")
    ax_a.text(
        0.01,
        0.985,
        "500 次；须线为 95% CI；方框为四分位区间",
        transform=ax_a.transAxes,
        fontsize=7.8,
        color=MUTED,
        ha="left",
        va="top",
    )
    style_axis(ax_a, grid_axis="x")

    official_cuts = np.array([31.0, 33.5, 36.0])
    rho_style = {0.80: (BLUE, 1.0), 0.90: (ORANGE, -1.0)}
    rng_b = np.random.default_rng(SEED + 132)
    for cut in official_cuts:
        ax_b.axvline(cut, color=MUTED, lw=0.85, ls=(0, (4, 3)), alpha=0.75, zorder=0)
    for cut_index, y_base in enumerate([2.0, 1.0, 0.0]):
        for rho, (color, direction) in rho_style.items():
            values = cuts[rho][:, cut_index]
            center = y_base + direction * 0.13
            x_grid = np.linspace(27.8, 38.2, 340)
            density = kde_1d(values, x_grid, bw_adjust=0.82)
            density = density / max(float(density.max()), 1e-12) * 0.24
            ax_b.fill_between(
                x_grid,
                center,
                center + direction * density,
                color=mpl.colors.to_rgba(color, 0.30),
                edgecolor=color,
                linewidth=1.0,
                zorder=2,
            )
            ax_b.scatter(
                values,
                center + rng_b.normal(0, 0.025, values.size),
                s=4.4,
                facecolor=mpl.colors.to_rgba(color, 0.15),
                edgecolor="none",
                rasterized=True,
                zorder=1,
            )
            q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
            ax_b.plot([q025, q975], [center, center], color=color, lw=1.25, zorder=4)
            ax_b.scatter(median, center, s=24, color=color, edgecolor="white", linewidth=0.45, zorder=5)
            ax_b.text(
                median,
                center + direction * 0.26,
                f"{median:.3f}",
                color=color,
                fontsize=7.5,
                ha="center",
                va="bottom" if direction > 0 else "top",
            )
    for cut in official_cuts:
        ax_b.text(cut, 2.52, f"{cut:g}", color=INK, fontsize=7.5, ha="center", va="bottom")
    ax_b.set_yticks([2, 1, 0], [r"$c_1$", r"$c_2$", r"$c_3$"])
    ax_b.set_ylim(-0.58, 2.62)
    ax_b.set_xlim(27.8, 38.2)
    ax_b.set_xlabel("BMI 切点")
    ax_b.set_title("(b) 最优 BMI 切点分布")
    ax_b.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=5, alpha=0.55, label=r"$\rho=0.80$"),
            Line2D([0], [0], color=ORANGE, lw=5, alpha=0.55, label=r"$\rho=0.90$"),
            Line2D([0], [0], color=MUTED, lw=1.0, ls="--", label="正式舍入切点"),
        ],
        loc="upper right",
        ncol=1,
        fontsize=7.4,
    )
    style_axis(ax_b, grid_axis="x")

    bmi_grid = np.linspace(27.0, 41.0, 141)
    x_edges = np.r_[bmi_grid - 0.05, bmi_grid[-1] + 0.05]
    y_edges = np.arange(5) - 0.5
    heat_cmap = LinearSegmentedColormap.from_list(
        "assignment_probability",
        ["#F7FAFC", "#C8E1EC", "#59A9C2", "#244D78"],
    )
    summary = assignment_stability.groupby("rho")["match_final_fraction"].agg(["mean", "median"])
    heat_mesh = None
    for ax, rho in [(ax_c80, 0.80), (ax_c90, 0.90)]:
        draw = cuts[rho]
        group_draws = (bmi_grid[None, :, None] >= draw[:, None, :]).sum(axis=2)
        probability = np.stack([(group_draws == group).mean(axis=0) for group in range(4)])
        heat_mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            probability,
            cmap=heat_cmap,
            vmin=0,
            vmax=1,
            shading="flat",
            rasterized=True,
        )
        for cut in official_cuts:
            ax.axvline(cut, color="white", lw=1.1, ls=(0, (4, 2)), alpha=0.95)
            ax.axvline(cut, color=INK, lw=0.45, ls=(0, (4, 2)), alpha=0.55)
        ax.set_ylim(3.5, -0.5)
        ax.set_xlim(27, 41)
        ax.set_yticks(range(4), ["G1", "G2", "G3", "G4"])
        ax.text(
            0.012,
            0.96,
            rf"$\rho={rho:.2f}$   平均/中位一致率 {summary.loc[rho, 'mean']:.3f}/{summary.loc[rho, 'median']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.7,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
        ax.tick_params(direction="out", width=0.65, length=2.6)
        for spine in ax.spines.values():
            spine.set_color(INK)
            spine.set_linewidth(0.75)
    ax_c80.set_title("(c) BMI × 归组概率", pad=6)
    ax_c80.tick_params(labelbottom=False)
    ax_c90.set_xlabel("BMI")
    ax_c80.set_ylabel("最终分组")
    ax_c90.set_ylabel("最终分组")
    colorbar = fig.colorbar(heat_mesh, cax=cax)
    colorbar.ax.set_title("概率", fontsize=7.8, pad=4)
    colorbar.ax.tick_params(labelsize=7.4)

    ax_d.axhspan(10, 12, color="#DDEFD9", alpha=0.48, zorder=0)
    ax_d.axhspan(12, 25, color="#F9EDC8", alpha=0.34, zorder=0)
    ax_d.axhspan(25, 36.5, color="#F5D9D5", alpha=0.38, zorder=0)
    ax_d.axhline(12, color=GREEN, lw=0.85, ls=(0, (4, 3)), zorder=1)
    ax_d.axhline(25, color=RED, lw=1.05, ls=(0, (5, 3)), zorder=1)
    rng_d = np.random.default_rng(SEED + 133)
    group_positions = np.arange(1, 5)
    for rho, offset, color in [(0.80, -0.14, BLUE), (0.90, 0.14, ORANGE)]:
        for group_index, x in enumerate(group_positions):
            values = replicate_times[rho][:, group_index]
            position = x + offset
            violin = ax_d.violinplot(
                values,
                positions=[position],
                widths=0.27,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=0.22,
            )
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.25)
                body.set_linewidth(0.9)
            ax_d.scatter(
                position + rng_d.normal(0, 0.022, values.size),
                values,
                s=4.2,
                facecolor=mpl.colors.to_rgba(color, 0.11),
                edgecolor="none",
                rasterized=True,
                zorder=2,
            )
            q025, q25, median, q75, q975 = np.quantile(values, [0.025, 0.25, 0.5, 0.75, 0.975])
            ax_d.plot([position, position], [q025, q975], color=color, lw=1.15, zorder=4)
            ax_d.add_patch(
                Rectangle(
                    (position - 0.045, q25),
                    0.09,
                    q75 - q25,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=5,
                )
            )
            ax_d.plot([position - 0.045, position + 0.045], [median, median], color=INK, lw=1.1, zorder=6)
            row = policy.loc[(policy["rho"] == rho) & (policy["group"] == group_index + 1)].iloc[0]
            recommendation = float(row["recommended_week"])
            ax_d.scatter(
                position,
                recommendation,
                marker="D",
                s=45,
                color=color,
                edgecolor=INK,
                linewidth=0.55,
                zorder=8,
            )
            short_label = str(row["recommended_week_day"]).replace("周+0天", "周")
            label_offset = -0.48 if rho == 0.80 else 0.46
            ax_d.text(
                position,
                recommendation + label_offset,
                short_label,
                ha="center",
                va="top" if rho == 0.80 else "bottom",
                color=color,
                fontsize=7.5,
            )
    ax_d.annotate(
        "超过 25 周\n转为个体化复检",
        xy=(4.14, float(policy.loc[(policy["rho"] == 0.90) & (policy["group"] == 4), "recommended_week"].iloc[0])),
        xytext=(3.05, 32.9),
        color=RED,
        fontsize=8.0,
        ha="left",
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.0, "connectionstyle": "arc3,rad=-0.16"},
    )
    group_n = policy.loc[policy["rho"] == 0.80].sort_values("group")["n"].astype(int).to_numpy()
    ax_d.set_xticks(group_positions, [f"G{i}\n$n={n}$" for i, n in enumerate(group_n, start=1)])
    ax_d.set_xlim(0.55, 4.48)
    ax_d.set_ylim(9.5, 36.5)
    ax_d.set_xlabel("固定 BMI 分组")
    ax_d.set_ylabel("组推荐孕周")
    ax_d.set_title("(d) 组时点分布与 25 周边界")
    ax_d.text(4.42, 24.55, "25 周常规检测窗口", color=RED, fontsize=7.7, ha="right", va="top")
    ax_d.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=6, alpha=0.45, label=r"$\rho=0.80$"),
            Line2D([0], [0], color=ORANGE, lw=6, alpha=0.45, label=r"$\rho=0.90$"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=INK, markersize=6, label="正式推荐点"),
        ],
        loc="upper left",
        ncol=1,
        columnspacing=1.0,
        handletextpad=0.4,
        fontsize=7.0,
    )
    style_axis(ax_d, grid_axis="y")

    save_figure(fig, "fig13_bootstrap_decision_stability")


def write_manifest() -> None:
    manifest = {
        "seed": SEED,
        "data": str(DATA_PATH),
        "figures": {
            "fig02_q1_joint_density": "孕周—Y 浓度的六边形密度、BMI 效应曲线与边缘分布",
            "fig03_q1_response_surface": "模型响应面、观测点云与孕周×BMI组达标率三维柱阵",
            "fig04_q1_parameter_ml_evidence": "混合模型参数重抽样区间与预测误差平面",
            "fig05_q2_tau_raincloud": "两档可靠度下个体保守时点的配对雨云图",
            "fig06_q2_policy_alluvial": "BMI 分组到两档推荐时点的守恒流图",
            "fig07_q2_optimization_sensitivity": "分组数帕累托前沿与阈值斜率图",
            "fig08_q3_increment_evidence": "多因素 AFT 误差平面与配对增量区间",
            "fig09_q4_label_upset": "女胎多标签交集 UpSet 图",
            "fig10_q4_weighted_pca": "孕妇等权 PCA 双标图与边缘密度",
            "fig11_q4_score_threshold_landscape": "折外风险分数的正负样本镜像密度与阈值",
            "fig12_q4_model_stability": "候选模型性能景观、Bootstrap 区间与时间留出对照",
            "fig13_bootstrap_decision_stability": "参数、切点、归组与分组排程的 Bootstrap 稳定性传递",
        },
        "primary_sources": [
            str(RESULTS / "q1_final_coefficients.csv"),
            str(STABILITY / "q2_tau_measurement_rho80.npz"),
            str(STABILITY / "q1_cluster_bootstrap.csv"),
            str(STABILITY / "q2_bootstrap_measurement.csv"),
            str(STABILITY / "q2_tau_measurement_rho90.npz"),
            str(RESULTS / "q2_final_policies.csv"),
            str(RESULTS / "q3_aft_paired_increment.csv"),
            str(Q4_LOCKED / "q4_locked_record_results.csv"),
            str(Q4_LOCKED / "q4_locked_method_comparison.csv"),
            str(Q4_LOCKED / "q4_locked_bootstrap_summary.csv"),
        ],
    }
    (OUT_DIR / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    prepared = prepare_data(DATA_PATH)
    visits = prepared.male_visits.copy()
    figure_q1_joint_density(visits)
    figure_q1_surface(visits)
    figure_q1_evidence()
    figure_q2_tau_raincloud()
    figure_q2_policy_alluvial()
    figure_q2_optimization_sensitivity()
    figure_q3_increment()
    figure_q4_upset()
    figure_q4_weighted_pca()
    figure_q4_score_threshold_landscape()
    figure_q4_model_landscape()
    figure_bootstrap_decision_stability()
    write_manifest()
    print(f"Generated 12 publication figures in {OUT_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr
from scipy.stats import norm


SEED = 20250904
Y_THRESHOLD = 0.04
EPS = 1e-12

MALE_SHEET = "男胎检测数据"
FEMALE_SHEET = "女胎检测数据"

QUALITY_COLUMNS = [
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数",
    "GC含量",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
    "被过滤掉读段数的比例",
]


def parse_ga(value: object) -> float:
    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d+)w(?:\+(\d+))?", text)
    if not match:
        raise ValueError(f"Unrecognized gestational age: {value!r}")
    weeks = int(match.group(1))
    days = int(match.group(2) or 0)
    if days < 0 or days > 6:
        raise ValueError(f"Invalid gestational days: {value!r}")
    return weeks + days / 7.0


def parse_date(value: object) -> pd.Timestamp:
    if isinstance(value, (int, np.integer)):
        return pd.to_datetime(str(int(value)), format="%Y%m%d")
    if isinstance(value, float) and value.is_integer():
        return pd.to_datetime(str(int(value)), format="%Y%m%d")
    return pd.to_datetime(value)


def gravidity_numeric(value: object) -> float:
    text = str(value).strip()
    if text in {"≥3", ">=3", "3及以上"}:
        return 3.0
    return float(text)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if np.isscalar(value) else False:
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def week_day(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    weeks = int(np.floor(value))
    days = int(np.round(7 * (value - weeks)))
    if days == 7:
        weeks += 1
        days = 0
    return f"{weeks}周+{days}天"


@dataclass
class PreparedData:
    male_raw: pd.DataFrame
    female_raw: pd.DataFrame
    male_visits: pd.DataFrame
    male_baseline: pd.DataFrame
    sigma_tech: float
    sigma_tech_strict: float
    technical_summary: dict[str, Any]


def pooled_technical_sd(
    frame: pd.DataFrame, key: Sequence[str], value_col: str
) -> tuple[float, int, int, list[float]]:
    pooled_ss = 0.0
    pooled_df = 0
    repeated = 0
    pair_diffs: list[float] = []
    for _, group in frame.groupby(list(key), dropna=False, sort=False):
        if len(group) <= 1:
            continue
        repeated += 1
        values = group[value_col].to_numpy(float)
        pooled_ss += float(np.square(values - values.mean()).sum())
        pooled_df += len(values) - 1
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                pair_diffs.append(abs(float(values[i] - values[j])))
    sigma = float(np.sqrt(pooled_ss / pooled_df))
    return sigma, pooled_df, repeated, pair_diffs


def prepare_data(data_path: Path) -> PreparedData:
    male = pd.read_excel(data_path, sheet_name=MALE_SHEET)
    female = pd.read_excel(data_path, sheet_name=FEMALE_SHEET)
    male.columns = male.columns.str.strip()
    female.columns = female.columns.str.strip()

    for frame in (male, female):
        frame["ga"] = frame["检测孕周"].map(parse_ga)
        frame["date_norm"] = frame["检测日期"].map(parse_date)
        frame["gravidity_num"] = frame["怀孕次数"].map(gravidity_numeric)
        frame["ivf_iui"] = frame["IVF妊娠"].ne("自然受孕").astype(int)
        frame["ivf"] = frame["IVF妊娠"].str.contains("IVF", regex=False).astype(int)
        frame["iui"] = frame["IVF妊娠"].str.contains("IUI", regex=False).astype(int)

    visit_key = ["孕妇代码", "检测抽血次数", "ga"]
    strict_key = ["孕妇代码", "date_norm", "检测抽血次数", "ga"]
    sigma, tech_df, repeated, pair_diffs = pooled_technical_sd(
        male, visit_key, "Y染色体浓度"
    )
    sigma_strict, tech_df_strict, repeated_strict, _ = pooled_technical_sd(
        male, strict_key, "Y染色体浓度"
    )

    ordered = male.sort_values(["孕妇代码", "ga", "date_norm", "序号"])
    agg: dict[str, tuple[str, str]] = {
        "date_norm": ("date_norm", "min"),
        "y": ("Y染色体浓度", "median"),
        "bmi": ("孕妇BMI", "median"),
        "weight": ("体重", "median"),
        "age": ("年龄", "first"),
        "height": ("身高", "first"),
        "gravidity": ("gravidity_num", "first"),
        "parity": ("生产次数", "first"),
        "ivf_iui": ("ivf_iui", "first"),
        "ivf": ("ivf", "first"),
        "iui": ("iui", "first"),
        "n_tech": ("Y染色体浓度", "size"),
        "y_mean": ("Y染色体浓度", "mean"),
        "y_min": ("Y染色体浓度", "min"),
        "y_max": ("Y染色体浓度", "max"),
    }
    for source in QUALITY_COLUMNS:
        safe = {
            "原始读段数": "raw_reads",
            "在参考基因组上比对的比例": "map_rate",
            "重复读段的比例": "duplicate_rate",
            "唯一比对的读段数": "unique_reads",
            "GC含量": "gc",
            "13号染色体的GC含量": "gc13",
            "18号染色体的GC含量": "gc18",
            "21号染色体的GC含量": "gc21",
            "被过滤掉读段数的比例": "filter_rate",
        }[source]
        agg[safe] = (source, "median")
    visits = (
        ordered.groupby(visit_key, as_index=False, dropna=False)
        .agg(**agg)
        .sort_values(["孕妇代码", "ga", "date_norm"])
        .reset_index(drop=True)
    )
    visits["y_range"] = visits["y_max"] - visits["y_min"]

    first_rows = (
        visits.sort_values(["孕妇代码", "ga", "date_norm"])
        .groupby("孕妇代码", sort=False)
        .first()
        .reset_index()
    )
    baseline_cols = [
        "孕妇代码",
        "ga",
        "date_norm",
        "bmi",
        "weight",
        "age",
        "height",
        "gravidity",
        "parity",
        "ivf_iui",
        "ivf",
        "iui",
    ]
    baseline = first_rows[baseline_cols].rename(
        columns={
            "date_norm": "first_date",
            "ga": "first_ga",
            "bmi": "first_bmi",
            "weight": "first_weight",
        }
    )
    visits = visits.merge(
        baseline[["孕妇代码", "first_ga", "first_bmi", "first_weight"]],
        on="孕妇代码",
        how="left",
        validate="many_to_one",
    )
    visits["delta_bmi"] = visits["bmi"] - visits["first_bmi"]

    technical_summary = {
        "male_rows": len(male),
        "male_women": int(male["孕妇代码"].nunique()),
        "male_visits": len(visits),
        "female_rows": len(female),
        "female_women": int(female["孕妇代码"].nunique()),
        "repeat_groups": repeated,
        "repeat_df": tech_df,
        "strict_repeat_groups": repeated_strict,
        "strict_repeat_df": tech_df_strict,
        "sigma_tech": sigma,
        "sigma_tech_strict": sigma_strict,
        "pairwise_abs_diff_median": float(np.median(pair_diffs)),
        "y_range": [float(male["Y染色体浓度"].min()), float(male["Y染色体浓度"].max())],
        "ga_range": [float(visits["ga"].min()), float(visits["ga"].max())],
    }
    return PreparedData(
        male_raw=male,
        female_raw=female,
        male_visits=visits,
        male_baseline=baseline,
        sigma_tech=sigma,
        sigma_tech_strict=sigma_strict,
        technical_summary=technical_summary,
    )


MEDIAN_SE_FACTORS = {1: 1.0, 2: 2 ** -0.5, 3: 0.67017}


def median_se_factor(replicates: int) -> float:
    if replicates in MEDIAN_SE_FACTORS:
        return MEDIAN_SE_FACTORS[replicates]
    return math.sqrt(math.pi / (2 * replicates))


def classify_visit_states(
    visits: pd.DataFrame,
    sigma_tech: float,
    *,
    threshold: float = Y_THRESHOLD,
    mode: str = "hard",
    eta: float = 0.025,
    error_multiplier: float = 1.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    result = visits.copy()
    factors = result["n_tech"].map(lambda x: median_se_factor(int(x))).to_numpy(float)
    result["se_tech"] = sigma_tech * factors * error_multiplier
    y = result["y"].to_numpy(float)
    if mode == "hard":
        states = (y >= threshold).astype(int)
    elif mode == "credible":
        z = norm.ppf(1 - eta)
        lower = y - z * result["se_tech"].to_numpy(float)
        upper = y + z * result["se_tech"].to_numpy(float)
        states = np.where(lower >= threshold, 1, np.where(upper < threshold, 0, -1))
    elif mode == "perturbed":
        if rng is None:
            raise ValueError("rng is required for perturbed states")
        simulated = y + rng.normal(0, result["se_tech"].to_numpy(float))
        result["y_perturbed"] = simulated
        states = (simulated >= threshold).astype(int)
    else:
        raise ValueError(f"Unknown state mode: {mode}")
    result["state"] = states
    return result


def construct_intervals(
    state_visits: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    exclude_conflicts: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_index = baseline.set_index("孕妇代码")
    for code, group in state_visits.groupby("孕妇代码", sort=False):
        group = group.sort_values(["ga", "date_norm"]).reset_index(drop=True)
        states = group["state"].to_numpy(int)
        positive = np.flatnonzero(states == 1)
        negative = np.flatnonzero(states == 0)
        conflict = False
        if len(positive):
            first_positive = int(positive[0])
            conflict = bool(np.any(states[first_positive + 1 :] == 0))
            before_negative = negative[negative < first_positive]
            if len(before_negative):
                left = float(group.iloc[int(before_negative[-1])]["ga"])
                right = float(group.iloc[first_positive]["ga"])
                censor_type = "interval"
            else:
                left = 0.0
                right = float(group.iloc[first_positive]["ga"])
                censor_type = "left"
        elif len(negative):
            left = float(group.iloc[int(negative[-1])]["ga"])
            right = np.inf
            censor_type = "right"
        else:
            left = np.nan
            right = np.nan
            censor_type = "uninformative"
        if exclude_conflicts and conflict:
            continue
        base = baseline_index.loc[code]
        rows.append(
            {
                "code": code,
                "left": left,
                "right": right,
                "type": censor_type,
                "state_conflict": int(conflict),
                "n_visits": len(group),
                "n_uncertain": int(np.sum(states == -1)),
                "first_bmi": float(base["first_bmi"]),
                "first_weight": float(base["first_weight"]),
                "age": float(base["age"]),
                "height": float(base["height"]),
                "gravidity": float(base["gravidity"]),
                "parity": float(base["parity"]),
                "ivf_iui": int(base["ivf_iui"]),
                "ivf": int(base["ivf"]),
                "iui": int(base["iui"]),
                "first_ga": float(base["first_ga"]),
            }
        )
    return pd.DataFrame(rows)


def logdiffexp(log_large: np.ndarray, log_small: np.ndarray) -> np.ndarray:
    delta = np.minimum(log_small - log_large, -np.finfo(float).eps)
    return log_large + np.log1p(-np.exp(delta))


def aft_logcdf(z: np.ndarray, distribution: str) -> np.ndarray:
    if distribution == "lognormal":
        return log_ndtr(z)
    if distribution == "loglogistic":
        return -np.logaddexp(0.0, -z)
    if distribution == "weibull":
        ez = np.exp(np.clip(z, -700, 50))
        return np.log(np.clip(-np.expm1(-ez), 1e-300, 1.0))
    raise ValueError(distribution)


def aft_logsurvival(z: np.ndarray, distribution: str) -> np.ndarray:
    if distribution == "lognormal":
        return log_ndtr(-z)
    if distribution == "loglogistic":
        return -np.logaddexp(0.0, z)
    if distribution == "weibull":
        return -np.exp(np.clip(z, -700, 50))
    raise ValueError(distribution)


def aft_error_quantile(probability: float, distribution: str) -> float:
    if distribution == "lognormal":
        return float(norm.ppf(probability))
    if distribution == "loglogistic":
        return float(np.log(probability / (1 - probability)))
    if distribution == "weibull":
        return float(np.log(-np.log(1 - probability)))
    raise ValueError(distribution)


@dataclass
class AFTFit:
    distribution: str
    feature_cols: list[str]
    coefficients: np.ndarray
    sigma: float
    means: np.ndarray
    scales: np.ndarray
    neg_loglik: float
    success: bool
    message: str
    n: int
    ridge: float = 0.0

    @property
    def parameter_count(self) -> int:
        return len(self.coefficients) + 1

    @property
    def aic(self) -> float:
        return 2 * self.neg_loglik + 2 * self.parameter_count

    @property
    def bic(self) -> float:
        return 2 * self.neg_loglik + self.parameter_count * np.log(self.n)

    def standardized_features(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.feature_cols:
            return np.empty((len(frame), 0))
        raw = frame[self.feature_cols].to_numpy(float)
        return (raw - self.means) / self.scales

    def linear_predictor(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.standardized_features(frame)
        design = np.column_stack([np.ones(len(frame)), x])
        return design @ self.coefficients

    def cdf(self, time: float | np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        time_array = np.asarray(time, dtype=float)
        eta = self.linear_predictor(frame)
        z = (np.log(time_array) - eta) / self.sigma
        return np.exp(aft_logcdf(z, self.distribution))

    def quantile(self, probability: float, frame: pd.DataFrame) -> np.ndarray:
        q = aft_error_quantile(probability, self.distribution)
        return np.exp(self.linear_predictor(frame) + self.sigma * q)

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "feature_cols": self.feature_cols,
            "coefficients": self.coefficients,
            "sigma": self.sigma,
            "means": self.means,
            "scales": self.scales,
            "neg_loglik": self.neg_loglik,
            "aic": self.aic,
            "bic": self.bic,
            "success": self.success,
            "message": self.message,
            "n": self.n,
            "ridge": self.ridge,
        }


def interval_loglik_contributions(
    params: np.ndarray,
    x: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    distribution: str,
) -> np.ndarray:
    beta = params[:-1]
    sigma = float(np.exp(params[-1]))
    eta = np.column_stack([np.ones(len(x)), x]) @ beta
    left_censored = left == 0
    right_censored = np.isinf(right)
    interval_censored = ~(left_censored | right_censored)
    result = np.empty(len(left), dtype=float)
    if np.any(left_censored):
        z_right = (np.log(right[left_censored]) - eta[left_censored]) / sigma
        result[left_censored] = aft_logcdf(z_right, distribution)
    if np.any(right_censored):
        z_left = (np.log(left[right_censored]) - eta[right_censored]) / sigma
        result[right_censored] = aft_logsurvival(z_left, distribution)
    if np.any(interval_censored):
        z_left = (np.log(left[interval_censored]) - eta[interval_censored]) / sigma
        z_right = (np.log(right[interval_censored]) - eta[interval_censored]) / sigma
        log_left = aft_logcdf(z_left, distribution)
        log_right = aft_logcdf(z_right, distribution)
        result[interval_censored] = logdiffexp(log_right, log_left)
    return result


def fit_aft(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    distribution: str = "lognormal",
    nonnegative_features: Iterable[str] = (),
    ridge: float = 0.0,
    starts: Sequence[np.ndarray] | None = None,
) -> AFTFit:
    usable = frame.loc[
        frame["type"].isin(["left", "interval", "right"])
    ].copy()
    feature_cols = list(feature_cols)
    if feature_cols:
        raw = usable[feature_cols].to_numpy(float)
        means = np.nanmean(raw, axis=0)
        scales = np.nanstd(raw, axis=0, ddof=0)
        scales = np.where(scales < 1e-8, 1.0, scales)
        raw = np.where(np.isfinite(raw), raw, means)
        x = (raw - means) / scales
    else:
        means = np.empty(0)
        scales = np.empty(0)
        x = np.empty((len(usable), 0))
    left = usable["left"].to_numpy(float)
    right = usable["right"].to_numpy(float)
    p = 1 + len(feature_cols)

    def objective(params: np.ndarray) -> float:
        values = interval_loglik_contributions(
            params, x, left, right, distribution
        )
        if not np.all(np.isfinite(values)):
            return 1e100
        penalty = ridge * float(np.square(params[1:p]).sum()) / 2
        return -float(values.sum()) + penalty

    bounds: list[tuple[float, float]] = [(-2.0, 6.0)]
    nonnegative = set(nonnegative_features)
    for name in feature_cols:
        bounds.append((0.0, 4.0) if name in nonnegative else (-4.0, 4.0))
    bounds.append((-4.0, 2.0))
    if starts is None:
        base = np.zeros(p + 1)
        base[0] = 2.2
        base[-1] = np.log(0.5)
        starts = [
            base,
            base + np.r_[0.2, np.zeros(p - 1), -0.3],
            base + np.r_[-0.2, np.zeros(p - 1), 0.3],
        ]
    fits = [
        minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1500, "ftol": 1e-11},
        )
        for start in starts
    ]
    fit = min(fits, key=lambda item: item.fun)
    coefficients = fit.x[:-1].copy()
    sigma = float(np.exp(fit.x[-1]))
    return AFTFit(
        distribution=distribution,
        feature_cols=feature_cols,
        coefficients=coefficients,
        sigma=sigma,
        means=means,
        scales=scales,
        neg_loglik=float(fit.fun),
        success=bool(fit.success),
        message=str(fit.message),
        n=len(usable),
        ridge=ridge,
    )


def evaluate_aft_loglik(fit: AFTFit, frame: pd.DataFrame) -> np.ndarray:
    usable = frame.loc[
        frame["type"].isin(["left", "interval", "right"])
    ].copy()
    x = fit.standardized_features(usable)
    params = np.r_[fit.coefficients, np.log(fit.sigma)]
    return interval_loglik_contributions(
        params,
        x,
        usable["left"].to_numpy(float),
        usable["right"].to_numpy(float),
        fit.distribution,
    )


def shuffled_group_folds(
    groups: Sequence[Any],
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups_array = np.asarray(groups)
    unique = np.unique(groups_array)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    bins = np.array_split(shuffled, n_splits)
    result = []
    for test_groups in bins:
        test_mask = np.isin(groups_array, test_groups)
        result.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return result


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(values * weights) / np.sum(weights))


def per_group_row_weights(groups: Sequence[Any]) -> np.ndarray:
    series = pd.Series(np.asarray(groups))
    counts = series.value_counts()
    return series.map(lambda x: 1.0 / counts[x]).to_numpy(float).copy()


def exact_upper_quantile(values: np.ndarray, probability: float) -> float:
    finite = np.asarray(values, dtype=float)
    order = np.sort(finite)
    rank = int(np.ceil(probability * len(order))) - 1
    rank = min(max(rank, 0), len(order) - 1)
    return float(order[rank])


@dataclass
class DPSegment:
    start: int
    end: int
    n: int
    bmi_min: float
    bmi_max: float
    raw_time: float
    operational_time: float
    coverage: float
    cost: float


@dataclass
class DPPolicy:
    groups: int
    alpha: float
    min_size: int
    total_cost: float
    segments: list[DPSegment]
    cutpoints: list[float]
    sorted_order: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": self.groups,
            "alpha": self.alpha,
            "min_size": self.min_size,
            "total_cost": self.total_cost,
            "cutpoints": self.cutpoints,
            "segments": [segment.__dict__ for segment in self.segments],
        }


def exact_dp_policy(
    bmi: np.ndarray,
    tau: np.ndarray,
    *,
    groups: int,
    alpha: float,
    min_size: int,
    lower_week: float = 10.0,
    upper_week: float = 25.0,
) -> DPPolicy:
    bmi = np.asarray(bmi, dtype=float)
    tau = np.asarray(tau, dtype=float)
    order = np.argsort(bmi, kind="stable")
    b = bmi[order]
    t = tau[order]
    n = len(b)
    cost = np.full((n, n), np.inf)
    group_time = np.full((n, n), np.nan)
    raw_time = np.full((n, n), np.nan)
    allowed_start = np.ones(n, dtype=bool)
    allowed_start[1:] = b[1:] > b[:-1]
    for start in range(n):
        if start > 0 and not allowed_start[start]:
            continue
        for end in range(start + min_size - 1, n):
            if end < n - 1 and b[end] == b[end + 1]:
                continue
            values = t[start : end + 1]
            q = exact_upper_quantile(values, alpha)
            operational = max(lower_week, q)
            if operational > upper_week or not np.isfinite(operational):
                continue
            residual = values - operational
            check = residual * (alpha - (residual < 0).astype(float))
            cost[start, end] = float(np.sum(check))
            raw_time[start, end] = q
            group_time[start, end] = operational
    dp = np.full((groups, n), np.inf)
    previous = np.full((groups, n), -1, dtype=int)
    dp[0, :] = cost[0, :]
    for g in range(1, groups):
        min_end = (g + 1) * min_size - 1
        for end in range(min_end, n):
            for prev_end in range(g * min_size - 1, end):
                start = prev_end + 1
                candidate = dp[g - 1, prev_end] + cost[start, end]
                if candidate < dp[g, end]:
                    dp[g, end] = candidate
                    previous[g, end] = prev_end
    if not np.isfinite(dp[groups - 1, n - 1]):
        raise ValueError(f"No feasible DP policy for K={groups}, alpha={alpha}")
    segments: list[DPSegment] = []
    end = n - 1
    for g in range(groups - 1, -1, -1):
        prev_end = int(previous[g, end]) if g > 0 else -1
        start = prev_end + 1
        values = t[start : end + 1]
        operational = float(group_time[start, end])
        segments.append(
            DPSegment(
                start=start,
                end=end,
                n=end - start + 1,
                bmi_min=float(b[start]),
                bmi_max=float(b[end]),
                raw_time=float(raw_time[start, end]),
                operational_time=operational,
                coverage=float(np.mean(values <= operational)),
                cost=float(cost[start, end]),
            )
        )
        end = prev_end
    segments.reverse()
    cuts = [
        float((segments[i].bmi_max + segments[i + 1].bmi_min) / 2)
        for i in range(len(segments) - 1)
    ]
    return DPPolicy(
        groups=groups,
        alpha=alpha,
        min_size=min_size,
        total_cost=float(dp[groups - 1, n - 1]),
        segments=segments,
        cutpoints=cuts,
        sorted_order=order,
    )


def rounded_policy_table(
    bmi: np.ndarray,
    tau: np.ndarray,
    fit: AFTFit,
    baseline: pd.DataFrame,
    *,
    cutpoints: Sequence[float],
    alpha: float,
    rounding: float = 0.5,
) -> tuple[pd.DataFrame, list[float]]:
    rounded = [round(float(c) / rounding) * rounding for c in cutpoints]
    rounded = sorted(set(rounded))
    bins = [-np.inf] + rounded + [np.inf]
    group_id = pd.cut(
        bmi, bins=bins, right=False, labels=False, include_lowest=True
    ).astype(int)
    rows: list[dict[str, Any]] = []
    for group in sorted(np.unique(group_id)):
        mask = group_id == group
        values = np.asarray(tau)[mask]
        time = max(10.0, exact_upper_quantile(values, alpha))
        subset = baseline.loc[mask].copy()
        probabilities = fit.cdf(time, subset)
        shortfall = np.maximum(values - time, 0)
        tail_n = max(1, int(np.ceil(0.10 * len(values))))
        cvar = float(np.sort(shortfall)[-tail_n:].mean())
        rows.append(
            {
                "group": int(group + 1),
                "bmi_left": float(np.min(np.asarray(bmi)[mask])),
                "bmi_right": float(np.max(np.asarray(bmi)[mask])),
                "n": int(mask.sum()),
                "recommended_week": time,
                "week_day": week_day(time),
                "coverage": float(np.mean(values <= time)),
                "mean_attainment_probability": float(np.mean(probabilities)),
                "uncovered_n": int(np.sum(values > time)),
                "out_of_window_n": int(np.sum(values > 25)),
                "cvar90_shortfall": cvar,
            }
        )
    return pd.DataFrame(rows), rounded


def determined_binary_at_time(
    intervals: pd.DataFrame, time: float
) -> tuple[np.ndarray, np.ndarray]:
    left = intervals["left"].to_numpy(float)
    right = intervals["right"].to_numpy(float)
    positive = right <= time
    negative = left >= time
    known = positive | negative
    labels = positive[known].astype(int)
    return known, labels


def support_status(values: np.ndarray) -> pd.DataFrame:
    array = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "minimum": [float(np.min(array))],
            "q01": [float(np.quantile(array, 0.01))],
            "q05": [float(np.quantile(array, 0.05))],
            "median": [float(np.median(array))],
            "q95": [float(np.quantile(array, 0.95))],
            "q99": [float(np.quantile(array, 0.99))],
            "maximum": [float(np.max(array))],
        }
    )

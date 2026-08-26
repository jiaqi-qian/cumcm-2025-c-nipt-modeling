from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nipt_core import (
    classify_visit_states,
    construct_intervals,
    exact_dp_policy,
    fit_aft,
    prepare_data,
    to_jsonable,
    write_json,
)


def solve_case(
    data,
    *,
    case: str,
    mode: str,
    threshold: float,
    eta: float,
    error_multiplier: float,
) -> list[dict]:
    states = classify_visit_states(
        data.male_visits,
        data.sigma_tech,
        mode=mode,
        threshold=threshold,
        eta=eta,
        error_multiplier=error_multiplier,
    )
    intervals = construct_intervals(states, data.male_baseline)
    fits = [
        fit_aft(
            intervals,
            ["first_bmi"],
            distribution=distribution,
            nonnegative_features=["first_bmi"],
        )
        for distribution in ["lognormal", "loglogistic", "weibull"]
    ]
    chosen = min(fits, key=lambda item: item.aic)
    baseline = data.male_baseline.sort_values("孕妇代码").reset_index(drop=True)
    bmi = baseline["first_bmi"].to_numpy(float)
    common = {
        "case": case,
        "mode": mode,
        "threshold": threshold,
        "eta": eta,
        "error_multiplier": error_multiplier,
        "n_left": int((intervals["type"] == "left").sum()),
        "n_interval": int((intervals["type"] == "interval").sum()),
        "n_right": int((intervals["type"] == "right").sum()),
        "n_uninformative": int(
            (intervals["type"] == "uninformative").sum()
        ),
        "state_conflicts": int(intervals["state_conflict"].sum()),
        "uncertain_visits": int(intervals["n_uncertain"].sum()),
        "chosen_distribution": chosen.distribution,
        "aic": chosen.aic,
        "bic": chosen.bic,
        "intercept": float(chosen.coefficients[0]),
        "beta_bmi_std": float(chosen.coefficients[1]),
        "beta_bmi_raw": float(chosen.coefficients[1] / chosen.scales[0]),
        "time_ratio_per_bmi": float(
            np.exp(chosen.coefficients[1] / chosen.scales[0])
        ),
        "sigma": chosen.sigma,
    }
    rows = []
    for rho in [0.80, 0.90]:
        tau = chosen.quantile(rho, baseline)
        try:
            policy = exact_dp_policy(
                bmi,
                tau,
                groups=4,
                alpha=rho,
                min_size=20,
            )
            cuts = policy.cutpoints
            times = [item.operational_time for item in policy.segments]
            coverage = [item.coverage for item in policy.segments]
            cost = policy.total_cost
            feasible = True
        except Exception:
            cuts = []
            times = []
            coverage = []
            cost = np.nan
            feasible = False
        rows.append(
            {
                **common,
                "rho": rho,
                "feasible_k4": feasible,
                "cutpoints": json.dumps(cuts),
                "times": json.dumps(times),
                "coverage": json.dumps(coverage),
                "total_cost": cost,
                "individuals_after_25": int(np.sum(tau > 25)),
                "tau_median": float(np.median(tau)),
                "tau_q90": float(np.quantile(tau, 0.90)),
                "tau_max": float(np.max(tau)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("附件.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nipt_solution/outputs/sensitivity"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = prepare_data(args.data)

    cases: list[dict] = []
    for threshold in [0.035, 0.040, 0.045]:
        cases.append(
            {
                "case": f"hard_threshold_{threshold:.3f}",
                "mode": "hard",
                "threshold": threshold,
                "eta": 0.025,
                "error_multiplier": 1.0,
            }
        )
        cases.append(
            {
                "case": f"credible_threshold_{threshold:.3f}",
                "mode": "credible",
                "threshold": threshold,
                "eta": 0.025,
                "error_multiplier": 1.0,
            }
        )
    for eta in [0.01, 0.025, 0.05, 0.10]:
        cases.append(
            {
                "case": f"credible_eta_{eta:.3f}",
                "mode": "credible",
                "threshold": 0.04,
                "eta": eta,
                "error_multiplier": 1.0,
            }
        )
    for multiplier in [0.75, 1.0, 1.25, 1.5]:
        cases.append(
            {
                "case": f"credible_error_x{multiplier:.2f}",
                "mode": "credible",
                "threshold": 0.04,
                "eta": 0.025,
                "error_multiplier": multiplier,
            }
        )

    unique: dict[tuple, dict] = {}
    for case in cases:
        key = (
            case["case"],
            case["mode"],
            case["threshold"],
            case["eta"],
            case["error_multiplier"],
        )
        unique[key] = case
    rows = []
    for case in unique.values():
        rows.extend(solve_case(data, **case))
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "q2_one_factor_sensitivity.csv", index=False)
    manifest = {
        "cases": len(unique),
        "rows": len(table),
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output / "sensitivity_manifest.json", manifest)
    print(json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

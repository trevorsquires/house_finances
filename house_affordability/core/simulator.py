from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from house_affordability.validation.checks import validate_inputs

from .engine import _monthly_return_params, run_financial_path
from .inputs import SimulationInputs


@dataclass
class SimulationResult:
    paths: pd.DataFrame
    run_metrics: pd.DataFrame
    percentile_runs: Dict[str, pd.DataFrame]
    summary: Dict[str, float]


def _draw_returns(sim_inputs: SimulationInputs, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    stock_mean, stock_vol = _monthly_return_params(
        sim_inputs.market.stock_return_annual, sim_inputs.market.stock_vol_annual
    )
    home_mean, home_vol = _monthly_return_params(
        sim_inputs.market.home_appreciation_annual, sim_inputs.market.home_vol_annual
    )
    corr = float(np.clip(sim_inputs.market.stock_home_correlation, -0.99, 0.99))
    cov = corr * stock_vol * home_vol
    cov_matrix = np.array([[stock_vol**2, cov], [cov, home_vol**2]])
    mean_vector = np.array([stock_mean, home_mean])
    draws = rng.multivariate_normal(
        mean_vector, cov_matrix, size=(sim_inputs.simulation.num_runs, sim_inputs.simulation.months)
    )
    return draws[:, :, 0], draws[:, :, 1]


def simulate(sim_inputs: SimulationInputs) -> SimulationResult:
    validate_inputs(sim_inputs)

    settings = sim_inputs.simulation
    rng = np.random.default_rng(settings.seed)

    stock_returns, home_returns = _draw_returns(sim_inputs, rng)

    path_records = []
    run_records = []

    for run_idx in range(settings.num_runs):
        path, metrics = run_financial_path(
            sim_inputs,
            stock_returns[run_idx],
            home_returns[run_idx],
            run_idx=run_idx,
            rng=rng,
            allow_random_job_loss=True,
        )
        path_records.extend(path)
        run_records.append(metrics)

    paths_df = pd.DataFrame(path_records).set_index(["run", "month"])
    run_metrics_df = pd.DataFrame(run_records).set_index("run")

    percentile_runs = _select_percentile_runs(paths_df, run_metrics_df)

    summary = {
        "probability_default": run_metrics_df["defaulted"].mean(),
        "probability_underwater": run_metrics_df["ever_underwater"].mean(),
        "expected_final_net_worth": run_metrics_df["final_net_worth"].mean(),
    }

    return SimulationResult(paths=paths_df, run_metrics=run_metrics_df, percentile_runs=percentile_runs, summary=summary)


def _select_percentile_runs(
    paths_df: pd.DataFrame, run_metrics_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Pick representative runs nearest to 25th/50th/75th percentile final net worth."""
    percentiles = {"25th": 25, "50th": 50, "75th": 75}
    percentile_runs = {}

    for label, pct in percentiles.items():
        target = np.percentile(run_metrics_df["final_net_worth"], pct)
        idx = (run_metrics_df["final_net_worth"] - target).abs().idxmin()
        percentile_runs[label] = paths_df.loc[idx].reset_index().assign(run=idx)

    return percentile_runs

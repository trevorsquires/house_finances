from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from house_affordability.validation.checks import validate_inputs

from .budget import monthly_income, monthly_non_housing_expense, recurring_housing_costs
from .inputs import SimulationInputs
from .mortgage import amortization_schedule


@dataclass
class SimulationResult:
    paths: pd.DataFrame
    run_metrics: pd.DataFrame
    percentile_runs: Dict[str, pd.DataFrame]
    summary: Dict[str, float]


def _monthly_return_params(annual_mean: float, annual_vol: float) -> Tuple[float, float]:
    return annual_mean / 12.0, annual_vol / np.sqrt(12)


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

    principal = sim_inputs.property.loan_principal
    amort_df = amortization_schedule(principal, sim_inputs.mortgage, horizon_months=settings.months)
    recurring_housing = recurring_housing_costs(sim_inputs.property)

    stock_returns, home_returns = _draw_returns(sim_inputs, rng)

    base_expenses = sim_inputs.household.non_housing_expenses_monthly
    job_loss = sim_inputs.job_loss

    path_records = []
    run_records = []

    for run_idx in range(settings.num_runs):
        cash = sim_inputs.household.savings_buffer - sim_inputs.property.down_payment - sim_inputs.property.closing_costs
        invested = 0.0
        property_value = sim_inputs.property.purchase_price

        defaulted = False
        underwater_flag = False
        job_lost = False

        for month_idx in range(settings.months):
            # Job loss check at start of each year
            if job_loss.enabled and not job_lost and month_idx % 12 == 0:
                if rng.random() < job_loss.annual_probability:
                    job_lost = True

            monthly_income_cash, monthly_stock_comp = monthly_income(
                sim_inputs.household,
                job_lost=job_lost,
                replacement_income_ratio=job_loss.replacement_income_ratio if job_loss.enabled else 1.0,
            )
            if job_lost and job_loss.stock_comp_stops:
                monthly_stock_comp = 0.0

            prev_cash = cash
            schedule_row = amort_df.iloc[month_idx]
            housing_payment = schedule_row["payment"]
            mortgage_balance = schedule_row["ending_balance"]
            non_housing = monthly_non_housing_expense(base_expenses, sim_inputs.household.inflation_annual, month_idx)
            total_essential_outflow = housing_payment + recurring_housing + non_housing + sim_inputs.household.debt_payments_monthly

            cash_after_essentials = prev_cash + monthly_income_cash - total_essential_outflow
            month_default = cash_after_essentials < 0
            defaulted = defaulted or month_default

            # Optional investing from cash only if there is wiggle room
            investable_from_cash = max(0.0, cash_after_essentials)
            contribution = min(sim_inputs.household.stock_contribution_monthly, investable_from_cash)
            cash = cash_after_essentials - contribution
            net_cash_flow = cash - prev_cash

            # Equity compensation is treated as immediate investment
            invested += monthly_stock_comp + contribution

            # Apply market moves
            stock_r = stock_returns[run_idx, month_idx]
            home_r = home_returns[run_idx, month_idx]
            invested = max(0.0, invested * (1 + stock_r))
            property_value = max(0.0, property_value * (1 + home_r))

            equity = property_value - mortgage_balance
            net_worth = cash + invested + equity
            underwater = equity < 0
            underwater_flag = underwater_flag or underwater

            path_records.append(
                {
                    "run": run_idx,
                    "month": month_idx + 1,
                    "cash": cash,
                    "invested": invested,
                    "property_value": property_value,
                    "mortgage_balance": mortgage_balance,
                    "equity": equity,
                    "net_worth": net_worth,
                    "monthly_income_cash": monthly_income_cash,
                    "monthly_stock_comp": monthly_stock_comp,
                    "housing_payment": housing_payment + recurring_housing,
                    "non_housing_expense": non_housing,
                    "debt_payments": sim_inputs.household.debt_payments_monthly,
                    "invested_contribution": contribution,
                    "net_cash_flow": net_cash_flow,
                    "defaulted": month_default,
                    "underwater": underwater,
                    "job_lost": job_lost,
                }
            )

        run_records.append(
            {
                "run": run_idx,
                "final_net_worth": net_worth,
                "ending_cash": cash,
                "ending_equity": equity,
                "defaulted": defaulted,
                "ever_underwater": underwater_flag,
            }
        )

    paths_df = pd.DataFrame(path_records).set_index(["run", "month"])
    run_metrics_df = pd.DataFrame(run_records).set_index("run")

    percentile_runs = _select_percentile_runs(paths_df, run_metrics_df)

    summary = {
        "probability_default": run_metrics_df["defaulted"].mean(),
        "probability_underwater": run_metrics_df["ever_underwater"].mean(),
        "median_final_net_worth": run_metrics_df["final_net_worth"].median(),
        "expected_monthly_cash_flow": paths_df["net_cash_flow"].mean(),
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
        percentile_runs[label] = paths_df.loc[idx].reset_index()

    return percentile_runs

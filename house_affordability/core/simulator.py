from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from house_affordability.validation.checks import validate_inputs

from .budget import monthly_income, monthly_non_housing_expense, recurring_housing_costs
from .inputs import SimulationInputs
from .mortgage import amortization_schedule
from .taxes import apply_tax


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
    federal_rate = sim_inputs.household.federal_tax_rate
    state_rate = sim_inputs.household.state_tax_rate

    path_records = []
    run_records = []

    for run_idx in range(settings.num_runs):
        cash = sim_inputs.household.cash_on_hand - sim_inputs.property.down_payment - sim_inputs.property.closing_costs
        invested_cash = 0.0
        stock_price = sim_inputs.household.initial_stock_price
        vested_units = 0.0
        # Track unvested tranches with semiannual vest cadence (6 months)
        unvested_tranches = []

        property_value = sim_inputs.property.purchase_price

        defaulted = False
        underwater_flag = False
        job_lost = False
        negative_months = 0
        current_negative_streak = 0
        longest_negative_streak = 0
        min_liquidity = cash + invested_cash + vested_units * stock_price

        for month_idx in range(settings.months):
            # Job loss check at start of each year
            if job_loss.enabled and not job_lost and month_idx % 12 == 0:
                if rng.random() < job_loss.annual_probability:
                    job_lost = True
                    # Forfeit any remaining unvested RSUs when job is lost
                    unvested_tranches = []

            monthly_income_cash, _ = monthly_income(
                sim_inputs.household,
                job_lost=job_lost,
                replacement_income_ratio=job_loss.replacement_income_ratio if job_loss.enabled else 1.0,
                month_index=month_idx,
            )
            net_income_cash, tax_on_income = apply_tax(monthly_income_cash, federal_rate, state_rate)
            prev_cash = cash
            schedule_row = amort_df.iloc[month_idx]
            housing_payment = schedule_row["payment"]
            mortgage_balance = schedule_row["ending_balance"]
            non_housing = monthly_non_housing_expense(base_expenses, sim_inputs.household.inflation_annual, month_idx)
            total_essential_outflow = housing_payment + recurring_housing + non_housing + sim_inputs.household.debt_payments_monthly
            essential_net = net_income_cash - total_essential_outflow

            liquidated_from_stock = 0.0
            liquidated_from_invested = 0.0
            month_default = False

            # Pay essentials; if negative, draw down investments first, then savings
            cash = prev_cash
            if essential_net >= 0:
                cash += essential_net
            else:
                deficit = -essential_net
                take_from_invested = min(deficit, invested_cash)
                invested_cash -= take_from_invested
                deficit -= take_from_invested
                liquidated_from_invested = take_from_invested

                if deficit > 0 and vested_units > 0 and stock_price > 0:
                    available_stock_value = vested_units * stock_price
                    sell_value = min(deficit, available_stock_value)
                    units_to_sell = sell_value / stock_price
                    vested_units -= units_to_sell
                    deficit -= sell_value
                    liquidated_from_stock = sell_value

                take_from_cash = min(deficit, cash)
                cash -= take_from_cash
                deficit -= take_from_cash

                month_default = deficit > 0
                if month_default:
                    cash -= deficit  # reflect the remaining shortfall
                defaulted = defaulted or month_default

            if essential_net < 0:
                negative_months += 1
                current_negative_streak += 1
                longest_negative_streak = max(longest_negative_streak, current_negative_streak)
            else:
                current_negative_streak = 0

            # Optional investing from cash only if there is wiggle room
            investable_from_cash = max(0.0, cash)
            contribution = min(sim_inputs.household.stock_contribution_monthly, investable_from_cash)
            cash -= contribution
            invested_cash += contribution
            net_cash_flow = cash - prev_cash

            liquidity_before_market = cash + invested_cash + vested_units * stock_price
            min_liquidity = min(min_liquidity, liquidity_before_market)

            # Apply market moves to invested cash and stock price
            stock_r = stock_returns[run_idx, month_idx]
            home_r = home_returns[run_idx, month_idx]
            invested_cash = max(0.0, invested_cash * (1 + stock_r))
            stock_price = max(0.01, stock_price * (1 + stock_r))  # avoid zero
            property_value = max(0.0, property_value * (1 + home_r))
            liquidity_after_market = cash + invested_cash + vested_units * stock_price
            min_liquidity = min(min_liquidity, liquidity_after_market)

            # RSU grant at end of each year if job not lost (assumption)
            if (month_idx + 1) % 12 == 0 and not (job_lost and job_loss.stock_comp_stops):
                growth = (1 + sim_inputs.household.salary_growth_annual) ** ((month_idx + 1) / 12.0)
                annual_stock_comp = sim_inputs.household.stock_comp_annual * growth
                grant_units = annual_stock_comp / stock_price if stock_price > 0 else 0.0
                vest_events = max(1, sim_inputs.household.stock_vesting_months // 6)
                units_per_event = grant_units / vest_events if vest_events else 0.0
                unvested_tranches.append(
                    {"months_elapsed": 0, "events_remaining": vest_events, "units_per_event": units_per_event}
                )

            # Advance vesting (semiannual cadence)
            vested_this_month = 0.0
            tax_on_vesting = 0.0
            for tranche in unvested_tranches:
                tranche["months_elapsed"] += 1
                if tranche["events_remaining"] > 0 and tranche["months_elapsed"] % 6 == 0:
                    vested_amount = tranche["units_per_event"]
                    gross_value = vested_amount * stock_price
                    net_value, tax_value = apply_tax(gross_value, federal_rate, state_rate)
                    net_units = net_value / stock_price if stock_price > 0 else 0.0
                    vested_units += net_units
                    tranche["events_remaining"] -= 1
                    vested_this_month += net_units
                    tax_on_vesting += tax_value
            # Drop fully vested tranches
            unvested_tranches = [t for t in unvested_tranches if t["events_remaining"] > 0]

            unvested_units = sum(t["units_per_event"] * t["events_remaining"] for t in unvested_tranches)

            equity = property_value - mortgage_balance
            vested_value = vested_units * stock_price
            unvested_value = unvested_units * stock_price
            net_worth = cash + invested_cash + vested_value + unvested_value + equity
            underwater = equity < 0
            underwater_flag = underwater_flag or underwater

            path_records.append(
                {
                    "run": run_idx,
                    "month": month_idx + 1,
                    "cash": cash,
                    "invested_cash": invested_cash,
                    "stock_price": stock_price,
                    "vested_units": vested_units,
                    "unvested_units": unvested_units,
                    "vested_value": vested_value,
                    "unvested_value": unvested_value,
                    "property_value": property_value,
                    "mortgage_balance": mortgage_balance,
                    "equity": equity,
                    "net_worth": net_worth,
                    "monthly_income_cash": monthly_income_cash,
                    "housing_payment": housing_payment + recurring_housing,
                    "non_housing_expense": non_housing,
                    "debt_payments": sim_inputs.household.debt_payments_monthly,
                    "invested_contribution": contribution,
                    "net_cash_flow": net_cash_flow,
                    "defaulted": month_default,
                    "underwater": underwater,
                    "job_lost": job_lost,
                    "liquidity": liquidity_after_market,
                    "liquidated_from_invested": liquidated_from_invested,
                    "liquidated_from_stock": liquidated_from_stock,
                    "vested_this_month": vested_this_month,
                    "tax_paid_income": tax_on_income,
                    "tax_paid_rsu": tax_on_vesting,
                }
            )

        run_records.append(
            {
                "run": run_idx,
                "final_net_worth": net_worth,
                "ending_cash": cash,
                "ending_equity": equity,
                "ending_vested_value": vested_value,
                "ending_unvested_value": unvested_value,
                "defaulted": defaulted,
                "ever_underwater": underwater_flag,
                "negative_months_pct": negative_months / settings.months,
                "longest_negative_streak": longest_negative_streak,
                "min_liquidity": min_liquidity,
            }
        )

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

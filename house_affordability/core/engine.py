from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .budget import monthly_income, monthly_non_housing_expense, recurring_housing_costs
from .inputs import SimulationInputs
from .mortgage import amortization_schedule
from .taxes import apply_tax


def _monthly_return_params(annual_mean: float, annual_vol: float) -> Tuple[float, float]:
    """Convert annual return inputs to monthly drift/vol."""
    return annual_mean / 12.0, annual_vol / np.sqrt(12)


def _salary_for_month(sim_inputs: SimulationInputs, scenario: Any, month_idx: int, job_lost: bool) -> float:
    """Return annual salary for the month, applying growth and scenario adjustments."""
    growth = (1 + sim_inputs.household.salary_growth_annual) ** (month_idx / 12.0)
    base_salary = sim_inputs.household.base_salary_annual * growth

    if scenario and getattr(scenario, "job_loss_month", None):
        job_loss_start = scenario.job_loss_month - 1
        rehire_start = job_loss_start + scenario.unemployment_months
        if job_lost:
            return base_salary * scenario.replacement_income_ratio
        if scenario.rehire_salary_annual is not None and month_idx >= rehire_start:
            rehire_growth = (1 + sim_inputs.household.salary_growth_annual) ** ((month_idx - rehire_start) / 12.0)
            return scenario.rehire_salary_annual * rehire_growth
    return base_salary


def _stock_comp_for_year(sim_inputs: SimulationInputs, scenario: Any, month_idx: int, job_lost: bool) -> float:
    """Return annual stock comp for grants at year boundaries, applying growth and scenario adjustments."""
    growth = (1 + sim_inputs.household.salary_growth_annual) ** (month_idx / 12.0)
    base_comp = sim_inputs.household.stock_comp_annual * growth

    if scenario and getattr(scenario, "job_loss_month", None):
        job_loss_start = scenario.job_loss_month - 1
        rehire_start = job_loss_start + scenario.unemployment_months
        if job_lost:
            return 0.0
        if scenario.rehire_salary_annual is not None and month_idx >= rehire_start:
            rehire_growth = (1 + sim_inputs.household.salary_growth_annual) ** ((month_idx - rehire_start) / 12.0)
            salary_ratio = (
                scenario.rehire_salary_annual / sim_inputs.household.base_salary_annual
                if sim_inputs.household.base_salary_annual
                else 1.0
            )
            return sim_inputs.household.stock_comp_annual * salary_ratio * rehire_growth
    return base_comp


def _non_housing_for_month(sim_inputs: SimulationInputs, scenario: Any, month_idx: int) -> float:
    """Return non-housing expense for month, applying inflation and optional shock."""
    if scenario and getattr(scenario, "expense_shock_month", None) and scenario.expense_new_monthly is not None:
        if month_idx + 1 >= scenario.expense_shock_month:
            months_since_shock = month_idx - (scenario.expense_shock_month - 1)
            return monthly_non_housing_expense(
                scenario.expense_new_monthly, sim_inputs.household.inflation_annual, months_since_shock
            )
    return monthly_non_housing_expense(
        sim_inputs.household.non_housing_expenses_monthly, sim_inputs.household.inflation_annual, month_idx
    )


def run_financial_path(
    sim_inputs: SimulationInputs,
    stock_returns: np.ndarray,
    home_returns: np.ndarray,
    run_idx: int,
    scenario: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
    allow_random_job_loss: bool = True,
) -> tuple[list[dict], dict]:
    """Single path evolution shared by Monte Carlo and deterministic scenarios."""
    settings = sim_inputs.simulation
    amort_df = amortization_schedule(sim_inputs.property.loan_principal, sim_inputs.mortgage, horizon_months=settings.months)
    recurring_housing = recurring_housing_costs(sim_inputs.property)

    cash = sim_inputs.household.cash_on_hand - sim_inputs.property.down_payment - sim_inputs.property.closing_costs
    invested_cash = 0.0
    stock_price = sim_inputs.household.initial_stock_price
    vested_units = 0.0
    unvested_tranches = []
    property_value = sim_inputs.property.purchase_price

    job_lost = False
    defaulted = False
    underwater_flag = False
    negative_months = 0
    current_negative_streak = 0
    longest_negative_streak = 0
    min_liquidity = cash

    path_records: list[dict] = []

    for month_idx in range(settings.months):
        month_one_based = month_idx + 1

        # Job loss / rehire transitions
        if scenario and getattr(scenario, "job_loss_month", None):
            job_loss_start = scenario.job_loss_month
            rehire_start = job_loss_start + scenario.unemployment_months
            if month_one_based == job_loss_start:
                job_lost = True
                unvested_tranches = []  # forfeit unvested RSUs
            if job_lost and month_one_based >= rehire_start:
                job_lost = False
        elif allow_random_job_loss and sim_inputs.job_loss.enabled and not job_lost and month_idx % 12 == 0:
            if rng and rng.random() < sim_inputs.job_loss.annual_probability:
                job_lost = True
                unvested_tranches = []

        # Income
        if scenario and getattr(scenario, "job_loss_month", None):
            annual_salary = _salary_for_month(sim_inputs, scenario, month_idx, job_lost)
            monthly_income_cash = annual_salary / 12.0 + sim_inputs.household.other_income_monthly
        else:
            monthly_income_cash, _ = monthly_income(
                sim_inputs.household,
                job_lost=job_lost,
                replacement_income_ratio=sim_inputs.job_loss.replacement_income_ratio
                if sim_inputs.job_loss.enabled
                else 1.0,
                month_index=month_idx,
            )

        net_income_cash, tax_on_income = apply_tax(
            monthly_income_cash, sim_inputs.household.federal_tax_rate, sim_inputs.household.state_tax_rate
        )

        schedule_row = amort_df.iloc[month_idx]
        housing_payment = schedule_row["payment"]
        mortgage_balance = schedule_row["ending_balance"]

        non_housing = _non_housing_for_month(sim_inputs, scenario, month_idx)
        total_essential_outflow = housing_payment + recurring_housing + non_housing + sim_inputs.household.debt_payments_monthly

        prev_cash = cash
        essential_net = net_income_cash - total_essential_outflow

        liquidated_from_stock = 0.0
        liquidated_from_invested = 0.0
        dipped_into_liquidity = essential_net < 0

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

            if deficit > 0:
                defaulted = True
                cash -= deficit  # reflect the remaining shortfall

        # One-time expense shock
        if scenario and getattr(scenario, "one_time_expense_month", None) and scenario.one_time_expense_amount:
            if month_one_based == scenario.one_time_expense_month:
                dipped_into_liquidity = True
                expense_deficit = scenario.one_time_expense_amount
                take_from_cash = min(expense_deficit, cash)
                cash -= take_from_cash
                expense_deficit -= take_from_cash

                take_from_invested = min(expense_deficit, invested_cash)
                invested_cash -= take_from_invested
                expense_deficit -= take_from_invested

                if expense_deficit > 0 and vested_units > 0 and stock_price > 0:
                    available_stock_value = vested_units * stock_price
                    sell_value = min(expense_deficit, available_stock_value)
                    units_to_sell = sell_value / stock_price
                    vested_units -= units_to_sell
                    expense_deficit -= sell_value
                    liquidated_from_stock += sell_value

                if expense_deficit > 0:
                    defaulted = True
                    cash -= expense_deficit

        if dipped_into_liquidity:
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
        stock_r = float(stock_returns[month_idx])
        home_r = float(home_returns[month_idx])
        invested_cash = max(0.0, invested_cash * (1 + stock_r))
        stock_price = max(0.01, stock_price * (1 + stock_r))  # avoid zero
        property_value = max(0.0, property_value * (1 + home_r))
        liquidity_after_market = cash + invested_cash + vested_units * stock_price
        min_liquidity = min(min_liquidity, liquidity_after_market)

        # RSU grant at end of each year
        if (month_idx + 1) % 12 == 0:
            if scenario and getattr(scenario, "job_loss_month", None):
                if not job_lost:
                    annual_stock_comp = _stock_comp_for_year(sim_inputs, scenario, month_idx + 1, job_lost=False)
                    grant_units = annual_stock_comp / stock_price if stock_price > 0 else 0.0
                    vest_events = max(1, sim_inputs.household.stock_vesting_months // 6)
                    units_per_event = grant_units / vest_events if vest_events else 0.0
                    unvested_tranches.append(
                        {"months_elapsed": 0, "events_remaining": vest_events, "units_per_event": units_per_event}
                    )
            else:
                if not (job_lost and sim_inputs.job_loss.stock_comp_stops):
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
                net_value, tax_value = apply_tax(gross_value, sim_inputs.household.federal_tax_rate, sim_inputs.household.state_tax_rate)
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
                "month": month_one_based,
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
                "defaulted": defaulted,
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

    run_metrics = {
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

    return path_records, run_metrics

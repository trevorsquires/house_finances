from __future__ import annotations

from .inputs import (
    HouseholdInputs,
    JobLossSettings,
    MarketAssumptions,
    MortgageInputs,
    PropertyInputs,
    SimulationInputs,
    SimulationSettings,
)


def default_market_assumptions() -> MarketAssumptions:
    return MarketAssumptions(
        stock_return_annual=0.07,
        stock_vol_annual=0.15,
        home_appreciation_annual=0.03,
        home_vol_annual=0.08,
        stock_home_correlation=0.1,
    )


def base_scenario() -> SimulationInputs:
    """Provide a reasonable starting point for the UI."""
    property_inputs = PropertyInputs(
        purchase_price=350_000,
        down_payment=70_000,
        closing_costs=8_000,
        property_tax_monthly=400,
        insurance_monthly=180,
        hoa_monthly=100,
    )

    mortgage_inputs = MortgageInputs(
        annual_rate=0.05,
        term_years=30,
        loan_type="fixed",
        arm_fixed_years=7,
        arm_adjusted_rate=0.07,
    )

    household_inputs = HouseholdInputs(
        base_salary_annual=100_000,
        stock_comp_annual=20_000,
        stock_vesting_months=48,
        initial_stock_price=100.0,
        federal_tax_rate=0.25,
        state_tax_rate=0.075,
        non_housing_expenses_monthly=2500,
        debt_payments_monthly=400,
        stock_contribution_monthly=500,
        savings_buffer=6_000,
        inflation_annual=0.025,
        other_income_monthly=0.0,
    )

    return SimulationInputs(
        property=property_inputs,
        mortgage=mortgage_inputs,
        household=household_inputs,
        market=default_market_assumptions(),
        job_loss=JobLossSettings(enabled=False, annual_probability=0.05, replacement_income_ratio=0.7),
        simulation=SimulationSettings(months=120, num_runs=500, seed=42),
    )

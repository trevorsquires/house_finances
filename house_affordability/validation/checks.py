from __future__ import annotations

from house_affordability.core.inputs import (
    HouseholdInputs,
    JobLossSettings,
    MarketAssumptions,
    MortgageInputs,
    PropertyInputs,
    SimulationInputs,
    SimulationSettings,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_property(inputs: PropertyInputs) -> None:
    _require(inputs.purchase_price > 0, "Purchase price must be positive.")
    _require(inputs.down_payment >= 0, "Down payment cannot be negative.")
    _require(inputs.closing_costs >= 0, "Closing costs cannot be negative.")
    _require(inputs.down_payment <= inputs.purchase_price, "Down payment exceeds purchase price.")
    _require(inputs.property_tax_monthly >= 0, "Property tax must be non-negative.")
    _require(inputs.insurance_monthly >= 0, "Insurance must be non-negative.")
    _require(inputs.hoa_monthly >= 0, "HOA must be non-negative.")


def validate_mortgage(inputs: MortgageInputs) -> None:
    _require(inputs.annual_rate >= 0, "Interest rate cannot be negative.")
    _require(inputs.term_years > 0, "Term must be positive.")
    _require(inputs.loan_type.lower() in {"fixed", "arm"}, "Loan type must be 'fixed' or 'arm'.")
    if inputs.loan_type.lower() == "arm":
        _require(inputs.arm_fixed_years > 0, "ARM fixed period must be positive.")
        _require(inputs.arm_adjusted_rate is None or inputs.arm_adjusted_rate >= 0, "ARM adjusted rate cannot be negative.")


def validate_household(inputs: HouseholdInputs) -> None:
    _require(inputs.base_salary_annual >= 0, "Base salary cannot be negative.")
    _require(inputs.stock_comp_annual >= 0, "Stock compensation cannot be negative.")
    _require(inputs.stock_vesting_months > 0, "Stock vesting duration must be positive.")
    _require(inputs.initial_stock_price > 0, "Initial stock price must be positive.")
    _require(inputs.non_housing_expenses_monthly >= 0, "Expenses cannot be negative.")
    _require(inputs.debt_payments_monthly >= 0, "Debt payments cannot be negative.")
    _require(inputs.stock_contribution_monthly >= 0, "Stock contribution cannot be negative.")
    _require(inputs.cash_on_hand >= 0, "Cash on hand cannot be negative.")
    _require(inputs.inflation_annual >= 0, "Inflation cannot be negative.")
    _require(0 <= inputs.federal_tax_rate < 1, "Federal tax rate must be between 0 and 1.")
    _require(0 <= inputs.state_tax_rate < 1, "State tax rate must be between 0 and 1.")
    _require(inputs.stock_vesting_months % 6 == 0, "Stock vesting months must align to 6-month cadence.")


def validate_market(inputs: MarketAssumptions) -> None:
    _require(inputs.stock_vol_annual >= 0, "Stock volatility cannot be negative.")
    _require(inputs.home_vol_annual >= 0, "Home volatility cannot be negative.")
    _require(-1.0 <= inputs.stock_home_correlation <= 1.0, "Correlation must be between -1 and 1.")


def validate_job_loss(inputs: JobLossSettings) -> None:
    _require(0 <= inputs.annual_probability <= 1, "Job loss probability must be between 0 and 1.")
    _require(0 <= inputs.replacement_income_ratio <= 2, "Replacement income ratio must be between 0 and 2.")


def validate_simulation(inputs: SimulationSettings) -> None:
    _require(inputs.months > 0, "Simulation horizon must be positive.")
    _require(inputs.num_runs > 0, "Number of runs must be positive.")
    _require(inputs.seed is None or isinstance(inputs.seed, int), "Seed must be an integer or None.")


def validate_inputs(sim_inputs: SimulationInputs) -> None:
    validate_property(sim_inputs.property)
    validate_mortgage(sim_inputs.mortgage)
    validate_household(sim_inputs.household)
    validate_market(sim_inputs.market)
    validate_job_loss(sim_inputs.job_loss)
    validate_simulation(sim_inputs.simulation)

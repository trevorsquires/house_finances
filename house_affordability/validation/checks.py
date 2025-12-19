from __future__ import annotations

from house_affordability.core.inputs import (
    HouseholdInputs,
    MortgageInputs,
    PropertyInputs,
    SimulationInputs,
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


def validate_household(inputs: HouseholdInputs) -> None:
    _require(inputs.base_salary_annual >= 0, "Base salary cannot be negative.")
    _require(inputs.stock_comp_annual >= 0, "Stock compensation cannot be negative.")
    _require(inputs.non_housing_expenses_monthly >= 0, "Expenses cannot be negative.")
    _require(inputs.debt_payments_monthly >= 0, "Debt payments cannot be negative.")
    _require(inputs.stock_contribution_monthly >= 0, "Stock contribution cannot be negative.")
    _require(inputs.savings_buffer >= 0, "Savings buffer cannot be negative.")
    _require(inputs.inflation_annual >= 0, "Inflation cannot be negative.")


def validate_inputs(sim_inputs: SimulationInputs) -> None:
    validate_property(sim_inputs.property)
    validate_mortgage(sim_inputs.mortgage)
    validate_household(sim_inputs.household)

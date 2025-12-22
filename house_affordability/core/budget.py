from __future__ import annotations

from .inputs import HouseholdInputs, PropertyInputs


def monthly_non_housing_expense(base: float, inflation_annual: float, month_index: int) -> float:
    """Apply monthly inflation compounding to baseline expenses."""
    monthly_inflation = inflation_annual / 12.0
    return base * ((1 + monthly_inflation) ** month_index)


def recurring_housing_costs(property_inputs: PropertyInputs) -> float:
    """Property tax, insurance, HOA combined."""
    return (
        property_inputs.property_tax_monthly
        + property_inputs.insurance_monthly
        + property_inputs.hoa_monthly
    )


def monthly_income(
    household: HouseholdInputs, job_lost: bool, replacement_income_ratio: float, month_index: int
) -> tuple[float, float]:
    """Return (salary_income, stock_comp_income). RSUs are handled separately."""
    growth = (1 + household.salary_growth_annual) ** (month_index / 12.0)
    salary = household.monthly_salary() * growth * (replacement_income_ratio if job_lost else 1.0)
    return salary + household.other_income_monthly, 0.0

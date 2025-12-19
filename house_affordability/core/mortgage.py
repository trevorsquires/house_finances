from __future__ import annotations

import pandas as pd

from .inputs import MortgageInputs


def monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
    monthly_rate = annual_rate / 12.0
    if term_months <= 0:
        return 0.0
    if monthly_rate == 0:
        return principal / term_months
    factor = (1 + monthly_rate) ** term_months
    return principal * monthly_rate * factor / (factor - 1)


def amortization_schedule(
    principal: float, mortgage: MortgageInputs, horizon_months: int | None = None
) -> pd.DataFrame:
    term_months = mortgage.term_years * 12
    horizon = min(horizon_months or term_months, term_months)

    balance = principal
    current_rate = mortgage.annual_rate
    months_remaining = term_months
    payment = monthly_payment(balance, current_rate, months_remaining)
    records = []

    arm_reset_month = mortgage.arm_fixed_years * 12 + 1 if mortgage.loan_type.lower() == "arm" else None
    for month in range(1, horizon + 1):
        if arm_reset_month and month == arm_reset_month:
            current_rate = mortgage.arm_adjusted_rate or (mortgage.annual_rate + 0.01)
            months_remaining = term_months - (month - 1)
            payment = monthly_payment(balance, current_rate, months_remaining)

        interest = balance * (current_rate / 12.0)
        principal_paid = min(payment - interest, balance)
        ending_balance = max(balance - principal_paid, 0.0)

        records.append(
            {
                "month": month,
                "payment": payment,
                "interest": interest,
                "principal": principal_paid,
                "ending_balance": ending_balance,
                "rate": current_rate,
            }
        )

        balance = ending_balance
        months_remaining = max(months_remaining - 1, 0)

    return pd.DataFrame.from_records(records).set_index("month")

from __future__ import annotations


def apply_tax(amount: float, federal_rate: float, state_rate: float) -> tuple[float, float]:
    """Return (net_amount, tax_paid) using combined federal + state rates."""
    total_rate = max(0.0, min(federal_rate + state_rate, 0.99))
    tax_paid = amount * total_rate
    return amount - tax_paid, tax_paid

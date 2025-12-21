from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PropertyInputs:
    purchase_price: float
    down_payment: float
    closing_costs: float = 0.0
    property_tax_monthly: float = 0.0
    insurance_monthly: float = 0.0
    hoa_monthly: float = 0.0

    @property
    def loan_principal(self) -> float:
        return max(self.purchase_price - self.down_payment, 0.0)


@dataclass
class MortgageInputs:
    annual_rate: float
    term_years: int
    loan_type: str = "fixed"  # "fixed" or "arm"
    arm_fixed_years: int = 5
    arm_adjusted_rate: Optional[float] = None

    def resolved_arm_rate(self) -> float:
        """Fallback to a modest increase if an adjusted rate is not provided."""
        return self.arm_adjusted_rate if self.arm_adjusted_rate is not None else self.annual_rate + 0.01


@dataclass
class HouseholdInputs:
    base_salary_annual: float
    stock_comp_annual: float = 0.0
    stock_vesting_months: int = 48  # standard 4-year
    initial_stock_price: float = 100.0
    non_housing_expenses_monthly: float = 2000.0
    debt_payments_monthly: float = 0.0
    stock_contribution_monthly: float = 0.0
    cash_on_hand: float = 0.0
    inflation_annual: float = 0.02
    other_income_monthly: float = 0.0
    federal_tax_rate: float = 0.25
    state_tax_rate: float = 0.075

    def monthly_salary(self) -> float:
        return self.base_salary_annual / 12.0

    def total_tax_rate(self) -> float:
        return self.federal_tax_rate + self.state_tax_rate


@dataclass
class MarketAssumptions:
    stock_return_annual: float = 0.07
    stock_vol_annual: float = 0.15
    home_appreciation_annual: float = 0.03
    home_vol_annual: float = 0.08
    stock_home_correlation: float = 0.0


@dataclass
class JobLossSettings:
    enabled: bool = False
    annual_probability: float = 0.05
    replacement_income_ratio: float = 0.7
    stock_comp_stops: bool = True


@dataclass
class SimulationSettings:
    months: int = 120  # 10 years by default
    num_runs: int = 500
    seed: Optional[int] = 42


@dataclass
class SimulationInputs:
    property: PropertyInputs
    mortgage: MortgageInputs
    household: HouseholdInputs
    market: MarketAssumptions = field(default_factory=MarketAssumptions)
    job_loss: JobLossSettings = field(default_factory=JobLossSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)

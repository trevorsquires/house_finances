from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .inputs import SimulationInputs
from .engine import _monthly_return_params, run_financial_path


@dataclass
class WhatIfScenario:
    job_loss_month: Optional[int] = None  # 1-based month when job loss starts
    unemployment_months: int = 0
    replacement_income_ratio: float = 0.0
    rehire_salary_annual: Optional[float] = None
    expense_shock_month: Optional[int] = None  # 1-based
    expense_new_monthly: Optional[float] = None
    one_time_expense_month: Optional[int] = None  # 1-based
    one_time_expense_amount: Optional[float] = None


@dataclass
class WhatIfResult:
    path: pd.DataFrame
    summary: Dict[str, float]


def run_what_if(sim_inputs: SimulationInputs, scenario: WhatIfScenario) -> WhatIfResult:
    """Deterministic scenario run using mean market returns and user shocks."""
    settings = sim_inputs.simulation
    stock_mean, _ = _monthly_return_params(sim_inputs.market.stock_return_annual, sim_inputs.market.stock_vol_annual)
    home_mean, _ = _monthly_return_params(sim_inputs.market.home_appreciation_annual, sim_inputs.market.home_vol_annual)

    stock_returns = np.full(settings.months, stock_mean)
    home_returns = np.full(settings.months, home_mean)

    path_records, run_metrics = run_financial_path(
        sim_inputs,
        stock_returns,
        home_returns,
        run_idx=0,
        scenario=scenario,
        rng=None,
        allow_random_job_loss=False,
    )

    path_df = pd.DataFrame(path_records).drop(columns=["run"]).set_index("month")

    summary = {
        "defaulted": bool(run_metrics["defaulted"]),
        "ever_underwater": bool(run_metrics["ever_underwater"]),
        "longest_negative_streak": run_metrics["longest_negative_streak"],
        "min_liquidity": run_metrics["min_liquidity"],
        "final_net_worth": run_metrics["final_net_worth"],
    }

    return WhatIfResult(path=path_df, summary=summary)

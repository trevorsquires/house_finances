import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # /mount/src/house_finances
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from __future__ import annotations

import streamlit as st

from house_affordability.core.inputs import (
    HouseholdInputs,
    JobLossSettings,
    MarketAssumptions,
    MortgageInputs,
    PropertyInputs,
    SimulationInputs,
    SimulationSettings,
)
from house_affordability.core.scenarios import base_scenario
from house_affordability.core.simulator import simulate
from house_affordability.validation.checks import validate_inputs


st.set_page_config(page_title="Can I Afford This House?", layout="wide")


def sidebar_inputs() -> SimulationInputs:
    defaults = base_scenario()
    st.sidebar.header("Home purchase")
    purchase_price = st.sidebar.number_input(
        "Purchase price", min_value=50_000, max_value=5_000_000, value=int(defaults.property.purchase_price), step=10_000
    )
    down_payment = st.sidebar.number_input(
        "Down payment", min_value=0, max_value=int(purchase_price), value=int(defaults.property.down_payment), step=5_000
    )
    closing_costs = st.sidebar.number_input(
        "Closing costs", min_value=0, max_value=200_000, value=int(defaults.property.closing_costs), step=1_000
    )
    property_tax = st.sidebar.number_input(
        "Property tax (monthly)", min_value=0, max_value=10_000, value=int(defaults.property.property_tax_monthly), step=50
    )
    insurance = st.sidebar.number_input(
        "Home insurance (monthly)", min_value=0, max_value=5_000, value=int(defaults.property.insurance_monthly), step=25
    )
    hoa = st.sidebar.number_input(
        "HOA (monthly)", min_value=0, max_value=5_000, value=int(defaults.property.hoa_monthly), step=25
    )

    st.sidebar.header("Mortgage")
    loan_type = st.sidebar.selectbox("Loan type", options=["fixed", "arm"], index=0 if defaults.mortgage.loan_type == "fixed" else 1)
    interest_rate = st.sidebar.slider(
        "Interest rate (annual %)", min_value=0.0, max_value=15.0, value=defaults.mortgage.annual_rate * 100, step=0.1
    )
    term_years = st.sidebar.slider("Term (years)", min_value=10, max_value=40, value=defaults.mortgage.term_years, step=5)
    arm_fixed_years = defaults.mortgage.arm_fixed_years
    arm_adjusted_rate = defaults.mortgage.arm_adjusted_rate or defaults.mortgage.annual_rate + 0.01
    if loan_type == "arm":
        arm_fixed_years = st.sidebar.slider("ARM fixed period (years)", min_value=1, max_value=10, value=defaults.mortgage.arm_fixed_years)
        arm_adjusted_rate = st.sidebar.slider(
            "ARM adjusted rate (annual %)", min_value=0.0, max_value=20.0, value=arm_adjusted_rate * 100, step=0.1
        )

    st.sidebar.header("Household")
    base_salary = st.sidebar.number_input(
        "Base salary (annual)", min_value=0, max_value=1_000_000, value=int(defaults.household.base_salary_annual), step=5_000
    )
    stock_comp = st.sidebar.number_input(
        "Stock compensation (annual)", min_value=0, max_value=1_000_000, value=int(defaults.household.stock_comp_annual), step=5_000
    )
    non_housing = st.sidebar.number_input(
        "Non-housing expenses (monthly)", min_value=0, max_value=20_000, value=int(defaults.household.non_housing_expenses_monthly), step=100
    )
    debt = st.sidebar.number_input(
        "Debt obligations (monthly)", min_value=0, max_value=20_000, value=int(defaults.household.debt_payments_monthly), step=50
    )
    stock_contribution = st.sidebar.number_input(
        "Desired stock investment (monthly)", min_value=0, max_value=20_000, value=int(defaults.household.stock_contribution_monthly), step=50
    )
    savings = st.sidebar.number_input(
        "Savings buffer (cash on hand)", min_value=0, max_value=2_000_000, value=int(defaults.household.savings_buffer), step=5_000
    )
    inflation = st.sidebar.slider(
        "Expense inflation (annual %)", min_value=0.0, max_value=10.0, value=defaults.household.inflation_annual * 100, step=0.1
    )

    st.sidebar.header("Simulation")
    horizon_years = st.sidebar.slider("Horizon (years)", min_value=1, max_value=30, value=defaults.simulation.months // 12)
    num_runs = st.sidebar.slider("Simulations", min_value=100, max_value=2000, value=defaults.simulation.num_runs, step=50)

    with st.sidebar.expander("Advanced settings"):
        st.subheader("Market assumptions")
        stock_return = st.slider(
            "Stock return (annual %)", min_value=-10.0, max_value=20.0, value=defaults.market.stock_return_annual * 100, step=0.5
        )
        stock_vol = st.slider(
            "Stock volatility (annual %)", min_value=0.0, max_value=50.0, value=defaults.market.stock_vol_annual * 100, step=1.0
        )
        home_return = st.slider(
            "Home appreciation (annual %)", min_value=-5.0, max_value=15.0, value=defaults.market.home_appreciation_annual * 100, step=0.5
        )
        home_vol = st.slider(
            "Home volatility (annual %)", min_value=0.0, max_value=30.0, value=defaults.market.home_vol_annual * 100, step=0.5
        )
        correlation = st.slider(
            "Stock/home correlation", min_value=-1.0, max_value=1.0, value=defaults.market.stock_home_correlation, step=0.05
        )

        st.subheader("Job loss")
        enable_job_loss = st.checkbox("Account for job loss", value=defaults.job_loss.enabled)
        job_loss_prob = defaults.job_loss.annual_probability * 100
        replacement_income = defaults.job_loss.replacement_income_ratio * 100
        if enable_job_loss:
            job_loss_prob = st.slider("Annual probability (%)", min_value=0.0, max_value=50.0, value=job_loss_prob, step=1.0)
            replacement_income = st.slider(
                "Replacement salary (% of base)", min_value=0.0, max_value=100.0, value=replacement_income, step=1.0
            )

    property_inputs = PropertyInputs(
        purchase_price=float(purchase_price),
        down_payment=float(down_payment),
        closing_costs=float(closing_costs),
        property_tax_monthly=float(property_tax),
        insurance_monthly=float(insurance),
        hoa_monthly=float(hoa),
    )

    mortgage_inputs = MortgageInputs(
        annual_rate=interest_rate / 100.0,
        term_years=int(term_years),
        loan_type=loan_type,
        arm_fixed_years=int(arm_fixed_years),
        arm_adjusted_rate=(arm_adjusted_rate / 100.0) if loan_type == "arm" else None,
    )

    household_inputs = HouseholdInputs(
        base_salary_annual=float(base_salary),
        stock_comp_annual=float(stock_comp),
        non_housing_expenses_monthly=float(non_housing),
        debt_payments_monthly=float(debt),
        stock_contribution_monthly=float(stock_contribution),
        savings_buffer=float(savings),
        inflation_annual=inflation / 100.0,
    )

    market_inputs = MarketAssumptions(
        stock_return_annual=stock_return / 100.0,
        stock_vol_annual=stock_vol / 100.0,
        home_appreciation_annual=home_return / 100.0,
        home_vol_annual=home_vol / 100.0,
        stock_home_correlation=correlation,
    )

    job_loss_inputs = JobLossSettings(
        enabled=enable_job_loss,
        annual_probability=job_loss_prob / 100.0,
        replacement_income_ratio=replacement_income / 100.0,
    )

    simulation_settings = SimulationSettings(months=horizon_years * 12, num_runs=int(num_runs))

    return SimulationInputs(
        property=property_inputs,
        mortgage=mortgage_inputs,
        household=household_inputs,
        market=market_inputs,
        job_loss=job_loss_inputs,
        simulation=simulation_settings,
    )


def render_percentile_runs(result):
    st.subheader("Representative scenarios")
    for label, df in result.percentile_runs.items():
        st.markdown(f"**{label} percentile net worth path**")
        chart_data = df.set_index("month")[["net_worth", "cash", "equity", "invested"]]
        st.line_chart(chart_data, height=240)


def render_summary(result):
    st.subheader("Risk dashboard")
    cols = st.columns(4)
    cols[0].metric("Default probability", f"{result.summary['probability_default']*100:.1f}%")
    cols[1].metric("Underwater probability", f"{result.summary['probability_underwater']*100:.1f}%")
    cols[2].metric("Median final net worth", f"${result.summary['median_final_net_worth']:,.0f}")
    cols[3].metric("Expected monthly cash flow", f"${result.summary['expected_monthly_cash_flow']:,.0f}")

    st.markdown("### Final net worth distribution")
    st.bar_chart(result.run_metrics["final_net_worth"])


def main():
    st.title("Can I Afford This House?")
    st.write(
        "Simulate household finances over time with mortgage payments, market swings, and optional job loss to evaluate how resilient a home purchase is."
    )

    sim_inputs = sidebar_inputs()

    run_btn = st.button("Run simulation", type="primary")
    if not run_btn:
        st.info("Adjust inputs in the sidebar and click **Run simulation**.")
        return

    try:
        validate_inputs(sim_inputs)
        result = simulate(sim_inputs)
    except Exception as exc:  # Streamlit friendly error surface
        st.error(f"Unable to run simulation: {exc}")
        return

    render_summary(result)
    render_percentile_runs(result)

    st.markdown("### Raw outputs")
    st.dataframe(result.run_metrics.reset_index(), use_container_width=True)


if __name__ == "__main__":
    main()

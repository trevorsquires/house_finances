from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import pandas as pd
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
from house_affordability.core.mortgage import monthly_payment
from house_affordability.core.scenarios import base_scenario
from house_affordability.core.simulator import simulate
from house_affordability.core.what_if import WhatIfScenario, run_what_if
from house_affordability.validation.checks import validate_inputs



st.set_page_config(page_title="Can I Afford This House?", layout="wide")


def sidebar_inputs() -> SimulationInputs:
    defaults = base_scenario()
    with st.sidebar.expander("House & Loan", expanded=False):
        st.markdown("### Purchase")
        purchase_price = st.number_input(
            "Purchase price", min_value=50_000, max_value=5_000_000, value=int(defaults.property.purchase_price), step=10_000
        )
        down_payment = st.number_input(
            "Down payment", min_value=0, max_value=int(purchase_price), value=int(defaults.property.down_payment), step=5_000
        )
        closing_costs = st.number_input(
            "Closing costs", min_value=0, max_value=200_000, value=int(defaults.property.closing_costs), step=1_000
        )
        property_tax = st.number_input(
            "Property tax (monthly)", min_value=0, max_value=10_000, value=int(defaults.property.property_tax_monthly), step=50
        )
        insurance = st.number_input(
            "Home insurance (monthly)", min_value=0, max_value=5_000, value=int(defaults.property.insurance_monthly), step=25
        )
        hoa = st.number_input(
            "HOA (monthly)", min_value=0, max_value=5_000, value=int(defaults.property.hoa_monthly), step=25
        )

        st.markdown("### Mortgage")
        loan_type = st.selectbox("Loan type", options=["fixed", "arm"], index=0 if defaults.mortgage.loan_type == "fixed" else 1)
        interest_rate = st.number_input(
            "Interest rate (annual %)",
            min_value=0.0,
            max_value=15.0,
            value=float(defaults.mortgage.annual_rate * 100),
            step=0.01,
            format="%.3f",
        )
        term_years = st.slider("Term (years)", min_value=10, max_value=40, value=defaults.mortgage.term_years, step=5)
        arm_fixed_years = defaults.mortgage.arm_fixed_years
        arm_adjusted_rate = defaults.mortgage.arm_adjusted_rate or defaults.mortgage.annual_rate + 0.01
        if loan_type == "arm":
            arm_fixed_years = st.slider("ARM fixed period (years)", min_value=1, max_value=10, value=defaults.mortgage.arm_fixed_years)
            arm_adjusted_rate = st.slider(
                "ARM adjusted rate (annual %)", min_value=0.0, max_value=20.0, value=arm_adjusted_rate * 100, step=0.1
            )

    with st.sidebar.expander("Personal Finances", expanded=False):
        base_salary = st.number_input(
            "Base salary (annual)", min_value=0, max_value=1_000_000, value=int(defaults.household.base_salary_annual), step=5_000
        )
        stock_comp = st.number_input(
            "Stock compensation (annual)", min_value=0, max_value=1_000_000, value=int(defaults.household.stock_comp_annual), step=5_000
        )
        vest_years = st.slider(
            "RSU vesting duration (years)", min_value=1, max_value=6, value=defaults.household.stock_vesting_months // 12
        )
        non_housing = st.number_input(
            "Non-housing expenses (monthly)", min_value=0, max_value=20_000, value=int(defaults.household.non_housing_expenses_monthly), step=100
        )
        debt = st.number_input(
            "Debt obligations (monthly)", min_value=0, max_value=20_000, value=int(defaults.household.debt_payments_monthly), step=50
        )
        stock_contribution = st.number_input(
            "Desired stock investment (monthly)", min_value=0, max_value=20_000, value=int(defaults.household.stock_contribution_monthly), step=50
        )
        savings = st.number_input(
            "Cash on hand (pre-closing)", min_value=0, max_value=2_000_000, value=int(defaults.household.cash_on_hand), step=5_000
        )
        salary_growth = st.slider(
            "Salary growth (annual %)", min_value=0.0, max_value=10.0, value=defaults.household.salary_growth_annual * 100, step=0.1
        )
        federal_tax = st.slider(
            "Federal tax rate (%)", min_value=0.0, max_value=50.0, value=defaults.household.federal_tax_rate * 100, step=0.5
        )
        state_tax = st.slider(
            "State tax rate (%)", min_value=0.0, max_value=20.0, value=defaults.household.state_tax_rate * 100, step=0.5
        )

    with st.sidebar.expander("Simulation & Advanced Settings", expanded=False):
        horizon_years = st.slider("Horizon (years)", min_value=1, max_value=30, value=defaults.simulation.months // 12)
        num_runs = st.slider("Simulations", min_value=100, max_value=2000, value=defaults.simulation.num_runs, step=50)
        inflation = st.slider(
            "Expense inflation (annual %)", min_value=0.0, max_value=10.0, value=defaults.household.inflation_annual * 100, step=0.1
        )

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
        initial_stock_price = st.slider(
            "RSU reference price ($)", min_value=1.0, max_value=1000.0, value=defaults.household.initial_stock_price, step=1.0
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
        stock_vesting_months=int(vest_years * 12),
        initial_stock_price=float(initial_stock_price),
        non_housing_expenses_monthly=float(non_housing),
        debt_payments_monthly=float(debt),
        stock_contribution_monthly=float(stock_contribution),
        cash_on_hand=float(savings),
        salary_growth_annual=salary_growth / 100.0,
        inflation_annual=inflation / 100.0,
        federal_tax_rate=federal_tax / 100.0,
        state_tax_rate=state_tax / 100.0,
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


def deterministic_snapshot(sim_inputs: SimulationInputs) -> dict:
    principal = sim_inputs.property.loan_principal
    p_and_i = monthly_payment(principal, sim_inputs.mortgage.annual_rate, sim_inputs.mortgage.term_years * 12)
    housing_fixed = p_and_i + sim_inputs.property.property_tax_monthly + sim_inputs.property.insurance_monthly + sim_inputs.property.hoa_monthly
    gross_income = sim_inputs.household.monthly_salary() + sim_inputs.household.other_income_monthly
    federal_tax = gross_income * sim_inputs.household.federal_tax_rate
    state_tax = gross_income * sim_inputs.household.state_tax_rate
    after_tax_income = gross_income - federal_tax - state_tax
    expenses = {
        "Federal tax": federal_tax,
        "State tax": state_tax,
        "Mortgage P&I": p_and_i,
        "Property tax": sim_inputs.property.property_tax_monthly,
        "Insurance": sim_inputs.property.insurance_monthly,
        "HOA": sim_inputs.property.hoa_monthly,
        "Non-housing": sim_inputs.household.non_housing_expenses_monthly,
        "Debt": sim_inputs.household.debt_payments_monthly,
        "Stock contribution": sim_inputs.household.stock_contribution_monthly,
    }
    total_out = sum(expenses.values())
    leftover = gross_income - total_out
    post_close_cash = sim_inputs.household.cash_on_hand - sim_inputs.property.down_payment - sim_inputs.property.closing_costs
    monthly_gap = max(-leftover, 0)
    runway_months = (post_close_cash / monthly_gap) if monthly_gap > 0 else None
    front_end = housing_fixed / gross_income if gross_income else 0.0
    back_end = (housing_fixed + sim_inputs.household.debt_payments_monthly) / gross_income if gross_income else 0.0
    loan_to_value = principal / sim_inputs.property.purchase_price if sim_inputs.property.purchase_price else 0.0
    reserves_months = post_close_cash / (housing_fixed + sim_inputs.household.debt_payments_monthly) if (housing_fixed + sim_inputs.household.debt_payments_monthly) > 0 else None

    return {
        "income_after_tax": after_tax_income,
        "gross_income": gross_income,
        "expenses": expenses,
        "leftover": leftover,
        "housing_fixed": housing_fixed,
        "front_end": front_end,
        "back_end": back_end,
        "ltv": loan_to_value,
        "reserves_months": reserves_months,
        "post_close_cash": post_close_cash,
        "runway_months": runway_months,
        "months_to_first_vest": 6,  # semiannual cadence assumption
    }


def render_deterministic(snapshot: dict) -> None:
    st.markdown("#### Cash flow math")
    rows = [
        ("Gross monthly pay", snapshot["gross_income"]),
        ("Federal tax", -snapshot["expenses"]["Federal tax"]),
        ("State tax", -snapshot["expenses"]["State tax"]),
        ("Mortgage P&I", -snapshot["expenses"]["Mortgage P&I"]),
        ("Property tax", -snapshot["expenses"]["Property tax"]),
        ("Insurance", -snapshot["expenses"]["Insurance"]),
        ("HOA", -snapshot["expenses"]["HOA"]),
        ("Non-housing", -snapshot["expenses"]["Non-housing"]),
        ("Debt", -snapshot["expenses"]["Debt"]),
        ("Stock contribution", -snapshot["expenses"]["Stock contribution"]),
        ("Monthly surplus / deficit", snapshot["leftover"]),
    ]
    df = pd.DataFrame(rows, columns=["Line item", "Amount"])
    df["Amount"] = df["Amount"].map(lambda x: f"${x:,.0f}")
    styled = df.style.set_properties(subset=pd.IndexSlice[df.index[-1], :], **{"font-weight": "bold"})
    st.table(styled)


def render_lender_indicators(snapshot: dict, sim_inputs: SimulationInputs) -> None:
    st.subheader("Lender indicators")
    indicators = [
        ("Front-end ratio", snapshot["front_end"] * 100, "<= 31%", snapshot["front_end"] <= 0.31),
        ("Back-end DTI", snapshot["back_end"] * 100, "<= 43%", snapshot["back_end"] <= 0.43),
        ("Loan-to-value (LTV)", snapshot["ltv"] * 100, "<= 80%", snapshot["ltv"] <= 0.80),
        (
            "Cash reserves (months of housing+debt)",
            snapshot["reserves_months"] if snapshot["reserves_months"] is not None else 0,
            ">= 3 mo",
            (snapshot["reserves_months"] or 0) >= 3,
        ),
    ]
    df = pd.DataFrame(indicators, columns=["Heuristic", "Value", "Target", "Satisfies"])
    def fmt_val(row):
        target = str(row["Target"]).lower()
        if "month" in target or "mo" in target:
            return f"{row['Value']:.1f} mo"
        return f"{row['Value']:.1f}%"
    df["Value"] = df.apply(fmt_val, axis=1)
    df["Satisfies"] = df["Satisfies"].map(lambda x: "Yes" if x else "No")
    st.table(df)
    st.caption("Targets are illustrative; lender guidelines vary.")


def render_percentile_runs(result):
    st.subheader("Representative scenarios")
    scenario_labels = {
        "25th": "Downside Scenario (25th percentile net worth)",
        "50th": "Average Scenario (50th percentile net worth)",
        "75th": "Upside Scenario (75th percentile net worth)",
    }
    rows = []
    for label, df in result.percentile_runs.items():
        run_id = df["run"].iloc[0] if "run" in df.columns else None
        metrics = result.run_metrics.loc[run_id] if run_id is not None else None
        if metrics is not None:
            rows.append(
                (
                    scenario_labels.get(label, label),
                    f"{metrics['negative_months_pct']*100:.1f}%",
                    f"${metrics['min_liquidity']:,.0f}",
                    f"{metrics['longest_negative_streak']:.0f} mo",
                )
            )
    if rows:
        st.markdown("**Cash flow stress by scenario**")
        table_df = pd.DataFrame(
            rows,
            columns=[
                "Scenario",
                "% months cash-flow negative",
                "Lowest liquidity (cash + liquid investments)",
                "Longest negative streak",
            ],
        )
        st.table(table_df)

    for label, df in result.percentile_runs.items():
        title = scenario_labels.get(label, f"{label} percentile net worth path")
        st.markdown(f"**{title}**")
        chart_data = df.set_index("month")[["net_worth", "cash", "equity", "invested_cash", "vested_value"]]
        st.line_chart(chart_data, height=240)


def render_summary(result):
    st.subheader("Risk dashboard")
    cols = st.columns(3)
    cols[0].metric("Default probability", f"{result.summary['probability_default']*100:.1f}%")
    cols[1].metric("Underwater probability", f"{result.summary['probability_underwater']*100:.1f}%")
    cols[2].metric("Expected final net worth", f"${result.summary['expected_final_net_worth']:,.0f}")


def main():
    st.title("Can I Afford This House?")
    st.write(
        "Simulate household finances over time with mortgage payments, market swings, and optional job loss to evaluate how resilient a home purchase is."
    )

    sim_inputs = sidebar_inputs()

    tab_calc, tab_lender, tab_sim, tab_whatif = st.tabs(
        ["Monthly calculations", "Lender indicators", "Simulation", "What-if scenarios"]
    )

    with tab_calc:
        snap = deterministic_snapshot(sim_inputs)
        render_deterministic(snap)

    with tab_lender:
        snap = deterministic_snapshot(sim_inputs)
        render_lender_indicators(snap, sim_inputs)

    with tab_sim:
        run_btn = st.button("Run simulation", type="primary")
        if not run_btn:
            st.info("Adjust inputs in the sidebar and click **Run simulation**.")
        else:
            try:
                validate_inputs(sim_inputs)
                result = simulate(sim_inputs)
            except Exception as exc:  # Streamlit friendly error surface
                st.error(f"Unable to run simulation: {exc}")
            else:
                render_summary(result)
                render_percentile_runs(result)

                st.markdown("### Raw outputs")
                st.dataframe(result.run_metrics.reset_index(), use_container_width=True)

    with tab_whatif:
        st.subheader("Build custom shocks")
        st.caption("Deterministic path using mean market returns with user-defined shocks.")
        with st.form(key="what_if_form"):
            with st.expander("Job loss shock", expanded=False):
                jl_col1, jl_col2, jl_col3, jl_col4 = st.columns(4)
                job_loss_enabled = jl_col1.checkbox("Include job loss", value=False)
                job_loss_month = jl_col2.number_input(
                    "Month of job loss",
                    min_value=1,
                    max_value=sim_inputs.simulation.months,
                    value=12,
                    step=1,
                )
                unemployment_months = jl_col3.number_input(
                    "Unemployment duration (months)", min_value=0, max_value=sim_inputs.simulation.months, value=6, step=1
                )
                replacement_income = jl_col4.slider(
                    "Replacement income during unemployment (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=5.0,
                )
                rehire_salary = st.number_input(
                    "New job annual salary (after rehire)",
                    min_value=0,
                    max_value=5_000_000,
                    value=int(sim_inputs.household.base_salary_annual),
                    step=5_000,
                )

            with st.expander("Expense shock", expanded=False):
                exp_col1, exp_col2 = st.columns(2)
                expense_enabled = exp_col1.checkbox("Include expense change", value=False)
                expense_month = exp_col1.number_input(
                    "Month of expense change",
                    min_value=1,
                    max_value=sim_inputs.simulation.months,
                    value=12,
                    step=1,
                )
                expense_new = exp_col2.number_input(
                    "New monthly non-housing expense",
                    min_value=0,
                    max_value=50_000,
                    value=int(sim_inputs.household.non_housing_expenses_monthly),
                    step=100,
                )

            with st.expander("One-time expense shock", expanded=False):
                ote_col1, ote_col2 = st.columns(2)
                one_time_enabled = ote_col1.checkbox("Include one-time expense", value=False)
                one_time_month = ote_col1.number_input(
                    "Month of one-time expense",
                    min_value=1,
                    max_value=sim_inputs.simulation.months,
                    value=6,
                    step=1,
                )
                one_time_amount = ote_col2.number_input(
                    "One-time expense amount",
                    min_value=0,
                    max_value=1_000_000,
                    value=10_000,
                    step=1_000,
                )

            run_scenario_btn = st.form_submit_button("Run what-if scenario", type="primary")

        if run_scenario_btn:
            scenario = WhatIfScenario(
                job_loss_month=int(job_loss_month) if job_loss_enabled else None,
                unemployment_months=int(unemployment_months) if job_loss_enabled else 0,
                replacement_income_ratio=(replacement_income / 100.0) if job_loss_enabled else 0.0,
                rehire_salary_annual=float(rehire_salary) if job_loss_enabled else None,
                expense_shock_month=int(expense_month) if expense_enabled else None,
                expense_new_monthly=float(expense_new) if expense_enabled else None,
                one_time_expense_month=int(one_time_month) if one_time_enabled else None,
                one_time_expense_amount=float(one_time_amount) if one_time_enabled else None,
            )
            try:
                result = run_what_if(sim_inputs, scenario)
            except Exception as exc:
                st.error(f"Unable to run what-if scenario: {exc}")
            else:
                metrics_cols = st.columns(3)
                metrics_cols[0].metric("Lowest liquidity", f"${result.summary['min_liquidity']:,.0f}")
                metrics_cols[1].metric("Longest negative streak", f"{result.summary['longest_negative_streak']} mo")
                metrics_cols[2].metric("Defaulted?", "Yes" if result.summary["defaulted"] else "No")

                st.markdown("**Monthly cash flow and liquidity**")
                chart_df = result.path[["net_cash_flow", "liquidity"]]
                st.line_chart(chart_df, height=280)

                st.markdown("**Scenario path details**")
                st.dataframe(result.path.reset_index(), use_container_width=True)


if __name__ == "__main__":
    main()

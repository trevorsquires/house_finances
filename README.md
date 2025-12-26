# Can I Afford This House?

Prototype Streamlit app that simulates US household finances for a home purchase. Core financial logic is pure Python; the UI is a thin Streamlit layer.

## What it does
- Enter purchase, mortgage, and household details to see monthly cash flow and lender-style ratios.
- Run Monte Carlo simulations to gauge default/underwater probabilities and view percentile scenarios.
- Try deterministic what-if shocks (job loss, expense changes, one-time costs) to see liquidity impacts.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run house_affordability/app/app.py
```

## Project layout

```
house_affordability/
├── core/          # pure financial logic
├── validation/    # input validation rules
└── app/           # Streamlit UI
```

Defaults model a 10-year (120-month) horizon with reasonable US-centric assumptions for stock and housing returns. Toggle advanced settings in the UI for job loss and other knobs.

# Bayesian Causal Inference — Halloween Promo (Simulated)

This example simulates a quick-service restaurant (QSR) **Halloween promo** and fits a
**Bayesian hierarchical model** to estimate causal lift, showing:

- How to **set a prior** on expected lift (e.g., +10%)
- How to **simulate panel data** across stores
- How to **fit a Bayesian model** (with PyMC) that updates beliefs as data arrives
- How to **summarize posteriors** (overall lift and segment-specific lift)
- How to sanity-check **MDE/power** using posterior probabilities (e.g., P(lift > 5%))

> This is a *toy* but realistic workflow you can adapt to real data.

## Quickstart

```bash
# 1) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate     # macOS/Linux
# or: python -m venv .venv && .venv\Scripts\activate  # Windows

# 2) Install dependencies
pip install -r requirements.txt

# 3) Simulate data (writes data/halloween_panel.csv)
python simulate_data.py

# 4) Fit Bayesian model (prints summaries + saves plots to artifacts/)
python bayesian_bci_example.py

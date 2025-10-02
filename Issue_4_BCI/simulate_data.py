import os
import numpy as np
import pandas as pd

np.random.seed(42)

# --- Config ---
n_stores = 120
days_pre, days_post = 21, 14            # 3 weeks pre, 2 weeks promo
treat_frac, urban_frac = 0.50, 0.45     # half treated, 45% urban

# Baseline log-sales (per-store) by segment; multiplicative noise later
mu_urban, mu_suburban = 3.65, 3.80      # ~38.5 vs ~44.7 units baseline means
sigma_store = 0.25                       # store heterogeneity (log scale)
sigma_obs = 0.20                         # observation noise (log scale)

# True promo lift (multiplicative), heterogeneity by segment
lift_urban = 0.04    # +4%
lift_suburban = 0.14 # +14%

# --- Stores & segments ---
stores = pd.DataFrame({
    "store_id": np.arange(n_stores),
    "segment": np.where(np.random.rand(n_stores) < urban_frac, "urban", "suburban"),
})
stores["treated"] = (np.random.rand(n_stores) < treat_frac).astype(int)

# Store-level baseline (log scale)
store_baseline = np.where(
    stores["segment"].eq("urban"),
    np.random.normal(mu_urban, sigma_store, size=n_stores),
    np.random.normal(mu_suburban, sigma_store, size=n_stores),
)
stores["log_base"] = store_baseline

# --- Calendar ---
pre_dates  = pd.date_range("2025-10-01", periods=days_pre,  freq="D")
post_dates = pd.date_range("2025-10-22", periods=days_post, freq="D")

def simulate_period(dates, label):
    rows = []
    for _, s in stores.iterrows():
        for d in dates:
            # Baseline with small weekday bumps (Fri/Sat/Sun)
            weekday = d.weekday()  # 0=Mon
            weekday_bump = {4: 0.05, 5: 0.10, 6: 0.08}.get(weekday, 0.0)
            log_mu = s["log_base"] + weekday_bump

            # Promo only if treated & post
            promo = 0.0
            if label == "post" and s["treated"] == 1:
                promo = np.log1p(lift_urban if s["segment"] == "urban" else lift_suburban)

            # Observed log-sales with noise
            y = np.random.normal(log_mu + promo, sigma_obs)

            rows.append({
                "date": d.date(),
                "store_id": int(s["store_id"]),
                "segment": s["segment"],
                "treated": int(s["treated"]),
                "period": label,
                "baseline_sales": float(np.exp(log_mu)),
                "observed_sales": float(np.exp(y)),
            })
    return pd.DataFrame(rows)

pre_df  = simulate_period(pre_dates,  "pre")
post_df = simulate_period(post_dates, "post")
panel = pd.concat([pre_df, post_df], ignore_index=True)

os.makedirs("data", exist_ok=True)
panel.to_csv("data/halloween_panel.csv", index=False)
print("Wrote data/halloween_panel.csv  rows:", len(panel))

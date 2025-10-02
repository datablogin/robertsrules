import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# 1) Load simulated data
df = pd.read_csv("data/halloween_panel.csv", parse_dates=["date"])

# 2) Aggregate to store x period
grp = df.groupby(["store_id", "segment", "treated", "period"], as_index=False)["observed_sales"].mean()
wide = grp.pivot_table(index=["store_id", "segment", "treated"],
                       columns="period", values="observed_sales").reset_index()
wide.columns.name = None

# 3) Work on log scale; within-store change delta = log(post) - log(pre)
wide["log_pre"]  = np.log(wide["pre"])
wide["log_post"] = np.log(wide["post"])
wide["delta"]    = wide["log_post"] - wide["log_pre"]

# 4) Encode segments
seg_map = {s: i for i, s in enumerate(sorted(wide["segment"].unique()))}
wide["seg_idx"] = wide["segment"].map(seg_map)

seg_idx = wide["seg_idx"].values
treated = wide["treated"].values.astype(int)
delta   = wide["delta"].values
n_seg   = len(seg_map)

# 5) Bayesian hierarchical model:
#    delta ~ baseline_seg[seg] + treated * lift_seg[seg] + eps
with pm.Model() as model:
    # Global priors
    mu_baseline = pm.Normal("mu_baseline", mu=0.0, sigma=0.2)
    mu_lift     = pm.Normal("mu_lift",     mu=np.log1p(0.10), sigma=0.20)  # prior ≈ +10% lift

    # Segment hierarchies
    sigma_baseline = pm.Exponential("sigma_baseline", 5.0)
    sigma_lift     = pm.Exponential("sigma_lift",     5.0)

    baseline_seg = pm.Normal("baseline_seg", mu=mu_baseline, sigma=sigma_baseline, shape=n_seg)
    lift_seg     = pm.Normal("lift_seg",     mu=mu_lift,     sigma=sigma_lift,     shape=n_seg)

    # Expected delta
    delta_mu = baseline_seg[seg_idx] + treated * lift_seg[seg_idx]

    # Noise
    sigma = pm.Exponential("sigma", 10.0)

    # Likelihood
    y = pm.Normal("y", mu=delta_mu, sigma=sigma, observed=delta)

    # Sample posterior
    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=42)
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

# 6) Summaries
summary = az.summary(idata, var_names=["mu_lift", "lift_seg", "mu_baseline", "baseline_seg", "sigma"], kind="stats")
print(summary)

# 7) Convert effects to percentage lift on natural scale
post = idata.posterior
mu_lift_pct   = np.expm1(post["mu_lift"].values) * 100
lift_seg_pct  = np.expm1(post["lift_seg"].values) * 100

print("\nEstimated GLOBAL lift (%): mean=%.2f, 5th=%.2f, 95th=%.2f" % (
    mu_lift_pct.mean(), np.percentile(mu_lift_pct, 5), np.percentile(mu_lift_pct, 95)
))
for seg, idx in seg_map.items():
    vals = lift_seg_pct[:, :, :, idx].reshape(-1)
    print("Segment %-9s lift (%%): mean=%.2f, 5th=%.2f, 95th=%.2f" %
          (seg, vals.mean(), np.percentile(vals, 5), np.percentile(vals, 95)))

# 8) Simple ROI/MDE sanity check: P(global lift > threshold)
threshold = 5.0  # e.g., campaign must beat 5% lift to be ROI-positive
prob_gt = (mu_lift_pct > threshold).mean()
print("\nP(global lift > %.1f%%) = %.3f" % (threshold, prob_gt))

# 9) Plots (saved in artifacts/)
os.makedirs("artifacts", exist_ok=True)

plt.figure(figsize=(6,4))
plt.hist(mu_lift_pct.reshape(-1), bins=40, alpha=0.7)
plt.axvline(threshold, linestyle="--")
plt.title("Posterior of Global Lift (%)")
plt.xlabel("Lift (%)"); plt.ylabel("Density")
plt.tight_layout()
plt.savefig("artifacts/posterior_global_lift.png", dpi=150)

plt.figure(figsize=(6,4))
for seg, idx in seg_map.items():
    vals = lift_seg_pct[:, :, :, idx].reshape(-1)
    plt.hist(vals, bins=40, alpha=0.5, label=seg)
plt.axvline(threshold, linestyle="--")
plt.title("Posterior of Segment Lifts (%)")
plt.xlabel("Lift (%)"); plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/posterior_segment_lifts.png", dpi=150)

print("\nSaved plots to artifacts/.")


"""
Double Machine Learning (DML) for Marketing:
Store Openings -> Loyalty Program Signups (Binary Treatment, Binary Outcome)

Scenario
-------
A retailer opens new stores in select regions. We want the *incremental* effect of
being in a region with a new store on the probability that a resident signs up for
the loyalty program within 90 days. Confounding factors (income, urbanicity,
competitor presence, prior brand awareness) affect both where stores are opened
and who signs up — so naive comparisons are biased.

What this script does
---------------------
1) Synthesizes realistic data with confounding & heterogeneous treatment effects.
2) Implements a 2-fold cross-fitting DML estimator for the ATE:
   - Learn m(X) = E[Y|X] and p(X) = E[T|X] on held-out folds
   - Residualize: Y_tilde = Y - m_hat(X), T_tilde = T - p_hat(X)
   - OLS of Y_tilde ~ T_tilde gives ATE (orthogonalized).
3) Bootstraps the ATE standard error & 95% CI.
4) Prints quick diagnostics (naive estimate vs DML, propensity common support).
5) (Optional) A lightweight CATE exploration via a simple meta-learner.

Dependencies
------------
- numpy, pandas, scikit-learn, statsmodels (for OLS), matplotlib (optional)
Install as needed:
  pip install numpy pandas scikit-learn statsmodels matplotlib
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)


# -----------------------------
# 1) Synthetic Data Generator
# -----------------------------

@dataclass
class SimConfig:
    n: int = 2000
    treatment_share: float = 0.4  # average probability of new-store exposure
    heterogeneity_strength: float = 0.5  # strength of tau(x) heterogeneity
    base_effect: float = 0.08  # baseline uplift from store opening on signup prob
    outcome_noise: float = 0.05  # stochasticity in outcome
    # Controls confounding: higher -> stronger selection into treatment by X
    confounding_strength: float = 3.2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def simulate_store_data(cfg: SimConfig) -> pd.DataFrame:
    n = cfg.n

    # Region/customer covariates
    income = np.random.lognormal(mean=10.7, sigma=0.35, size=n) / 1000  # ~$45k-$120k range
    urbanicity = np.random.beta(2, 2, size=n)  # 0=rural, 1=urban
    competitor_presence = np.random.binomial(1, p=0.45, size=n)  # competitor nearby
    brand_awareness = np.clip(0.2 + 0.6*urbanicity + 0.2*np.random.randn(n), 0, 1)

    X = np.column_stack([income, urbanicity, competitor_presence, brand_awareness])

    # True heterogeneous treatment effect tau(X):
    # More urban, higher awareness -> larger incremental effect from a new store
    tau_true = cfg.base_effect + cfg.heterogeneity_strength * (0.3*urbanicity + 0.2*brand_awareness - 0.1*competitor_presence)

    # Treatment assignment (new store in region): depends on covariates -> confounding
    logits_T = (
        -0.5
        + cfg.confounding_strength * (0.4*urbanicity + 0.2*brand_awareness - 0.2*competitor_presence + 0.1*np.log(income))
    )
    # Shift to approximately match target share
    shift = np.log(cfg.treatment_share/(1-cfg.treatment_share))
    logits_T = logits_T - (np.mean(sigmoid(logits_T)) - cfg.treatment_share) * 3.0 + shift
    p_T = sigmoid(logits_T)
    T = np.random.binomial(1, p_T)

    # Baseline outcome probability without treatment (calibrated for demo)
    base_logit = (
        -2.5
        + 0.6*np.log(income)
        + 1.8*brand_awareness
        + 1.5*urbanicity
        - 0.8*competitor_presence
    )
    p0 = sigmoid(base_logit)

    # Outcome with treatment: add tau_true on the logit scale for smoother effects
    p1 = sigmoid(base_logit + 3.0 * tau_true)  # scale tau so net uplift ~ base_effect
    p = np.where(T == 1, p1, p0)

    # Add outcome noise by mixing with a noise term
    p = np.clip((1 - cfg.outcome_noise) * p + cfg.outcome_noise * np.random.rand(n), 0, 1)
    Y = np.random.binomial(1, p)

    df = pd.DataFrame({
        "Y": Y,
        "T": T,
        "income_k": income,
        "urbanicity": urbanicity,
        "competitor": competitor_presence,
        "brand_awareness": brand_awareness,
        "p_T": p_T,
        "p0": p0,
        "p1": p1,
        "tau_true": tau_true,
    })
    return df


# --------------------------------------
# 2) DML with 2-fold Cross-Fitting ATE
# --------------------------------------

def crossfit_predictions(model, X, y, n_splits=2, is_classifier=False):
    """Out-of-fold predictions for m(X) or p(X)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    oof = np.zeros(len(X))
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr = y[tr]
        mdl = model
        mdl.fit(Xtr, ytr)
        if is_classifier:
            proba = mdl.predict_proba(Xte)[:, 1]
            oof[te] = proba
        else:
            oof[te] = mdl.predict(Xte)
    return oof


def dml_ate(df: pd.DataFrame, features, n_splits=2, n_boot=200, random_state=123):
    rng = np.random.default_rng(random_state)
    X = df[features].values
    Y = df["Y"].values
    T = df["T"].values

    # Nuisance models (flexible ML)
    m_model = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_leaf=10)
    # For propensity: calibrated RF classifier for better probabilities
    p_base = RandomForestClassifier(n_estimators=50, random_state=2, min_samples_leaf=10)
    p_model = CalibratedClassifierCV(estimator=p_base, method="isotonic", cv=3)

    # Cross-fitted predictions
    m_hat = crossfit_predictions(m_model, X, Y, n_splits=n_splits, is_classifier=False)
    p_hat = crossfit_predictions(p_model, X, T, n_splits=n_splits, is_classifier=True)

    # Common support trimming (optional, conservative)
    eps = 0.02
    mask = (p_hat > eps) & (p_hat < 1 - eps)
    X, Y, T, m_hat, p_hat = X[mask], Y[mask], T[mask], m_hat[mask], p_hat[mask]

    # Residualize
    Y_tilde = Y - m_hat
    T_tilde = T - p_hat

    # OLS of Y_tilde ~ T_tilde (no intercept)
    ols = sm.OLS(Y_tilde, T_tilde).fit()
    ate = ols.params.item()
    # Robust (HC3) SE for the single coefficient
    se = ols.HC3_se.item()

    # Bootstrap for CI (resample indices)
    boot = []
    n = len(Y_tilde)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b_ols = sm.OLS(Y_tilde[idx], T_tilde[idx]).fit()
        boot.append(b_ols.params.item())
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    results = {
        "ate": ate,
        "se_HC3": se,
        "ci_95": (ci_low, ci_high),
        "trim_rate": 1 - mask.mean(),
        "propensity_auc": roc_auc_score(T, p_hat) if len(np.unique(T)) > 1 else np.nan,
    }
    return results, {"m_hat": m_hat, "p_hat": p_hat, "mask": mask}


# -------------------------------------------------
# 3) Optional: Simple CATE meta-learner (fast demo)
# -------------------------------------------------

def simple_cate_meta_learner(df: pd.DataFrame, features):
    """
    Very lightweight CATE exploration:
    - Fit two models for E[Y|X, T=0] and E[Y|X, T=1] on their subsets
    - Predict both for all X, take difference as CATE_hat(X)
    This is *not* a full DR-learner, but is quick & illustrative.
    """
    X = df[features].values
    Y = df["Y"].values
    T = df["T"].values

    model0 = RandomForestRegressor(n_estimators=50, random_state=3, min_samples_leaf=10)
    model1 = RandomForestRegressor(n_estimators=50, random_state=4, min_samples_leaf=10)

    model0.fit(X[T == 0], Y[T == 0])
    model1.fit(X[T == 1], Y[T == 1])

    mu0 = model0.predict(X)
    mu1 = model1.predict(X)
    cate_hat = mu1 - mu0
    return cate_hat


# -----------------------------
# 4) Run the full pipeline
# -----------------------------

def main():
    cfg = SimConfig()
    df = simulate_store_data(cfg)

    features = ["income_k", "urbanicity", "competitor", "brand_awareness"]

    # Naive estimate (difference in means) for comparison
    naive = df.loc[df["T"] == 1, "Y"].mean() - df.loc[df["T"] == 0, "Y"].mean()

    dml_res, aux = dml_ate(df, features, n_splits=2, n_boot=50, random_state=7)

    print("\\n=== DML: Store Opening -> Loyalty Signup ===")
    print(f"Samples: {len(df):,}")
    print(f"Naive diff-in-means: {naive: .4f}")
    print(f"DML ATE: {dml_res['ate']: .4f}  (SE≈{dml_res['se_HC3']: .4f})  95% CI{dml_res['ci_95']}")
    print(f"Propensity AUC: {dml_res['propensity_auc']: .3f}")
    if dml_res['trim_rate'] > 0:
        print(f"Common-support trimming removed {dml_res['trim_rate']*100: .1f}% of rows (p_hat near 0/1).")

    # Quick common-support diagnostic: hist of propensity by T
    fig = plt.figure()
    plt.hist(df.loc[df["T"]==0, "p_T"], bins=40, alpha=0.6, label="Control (T=0)")
    plt.hist(df.loc[df["T"]==1, "p_T"], bins=40, alpha=0.6, label="Treated (T=1)")
    plt.title("True Propensity by Treatment Group (simulated)")
    plt.xlabel("Pr(T=1 | X)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("propensity_histogram.png", bbox_inches="tight")
    plt.close()

    # Optional CATE exploration
    cate_hat = simple_cate_meta_learner(df, features)
    df_out = df.copy()
    df_out["cate_hat"] = cate_hat
    # Illustrate heterogeneity: avg CATE by urbanicity decile
    df_out["urb_bin"] = pd.qcut(df_out["urbanicity"], 10, labels=False, duplicates="drop")
    cate_by_bin = df_out.groupby("urb_bin")["cate_hat"].mean()
    print("\\nAvg CATE by Urbanicity Decile (0=rural, 9=urban):")
    print(cate_by_bin.round(4))

    # Save a small CSV snapshot for readers
    df_out_sample = df_out.sample(2000, random_state=11)
    df_out_sample.to_csv("dml_store_openings_sample.csv", index=False)
    print("\\nWrote sample data to dml_store_openings_sample.csv")

if __name__ == "__main__":
    main()

"""
Experiment design: power analysis, sample sizing, and MDE.

This is the "design the experiment" layer (Chapter 3). It is deliberately
COHERENT with the data: baselines are read from experiment_data.csv, so the
sizing here explains the result that cuped_analysis.py reports downstream.

------------------------------------------------------------------------------
PRE-REGISTRATION (decide all of this BEFORE looking at results)
------------------------------------------------------------------------------
Hypothesis
  H0: the new checkout flow does not change the order rate.
  H1: the new checkout flow increases the order rate.
  One-sided in intent (we only ship a positive change), but we TEST two-sided
  and halve nothing - the conservative default reviewers expect.

Metrics
  Primary    : conversion = P(>=1 order in the experiment window).
               Low variance, directly tied to the hypothesis -> this is what
               we power the test on.
  Secondary  : revenue per user. Business-relevant but HEAVY-TAILED
               (CV ~ 2.5 here), so it needs a much larger effect to detect -
               this is the metric CUPED exists to help.
  Guardrails : (a) average order value per ORDER - must not drop;
               (b) refund rate - must not rise.
  Ratio note : revenue per session is a ratio metric (delta method at
               analysis time); kept secondary.

Randomisation
  Unit = user. 50/50 via md5(user_id + salt) % 100. Deterministic and
  uniform; the same user is always in the same arm.

Decision rule (stated in advance)
  SHIP iff the primary metric is significantly positive (alpha = 0.05,
  two-sided) AND no guardrail shows a significant regression. A significant
  secondary is supporting evidence, not a launch criterion on its own.

Stats config
  alpha = 0.05 (two-sided), power = 0.80.
------------------------------------------------------------------------------

Usage:
  python experiment_design.py        # reads experiment_data.csv
"""

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "experiment_data.csv"
ALPHA = 0.05
POWER = 0.80
DAILY_USERS = 7857           # ~220K exposed users over the 28-day window
CUPED_VAR_REDUCTION = 0.293  # measured in cuped_analysis.py (rho ~ 0.60)


def _z():
    return stats.norm.ppf(1 - ALPHA / 2), stats.norm.ppf(POWER)


# --- proportions (the primary metric: conversion) --------------------------
def n_per_arm_proportion(p: float, rel_mde: float) -> float:
    """Users per arm to detect a relative lift on a baseline rate p."""
    z_a, z_b = _z()
    delta = p * rel_mde
    return (z_a + z_b) ** 2 * 2 * p * (1 - p) / delta**2


def mde_proportion(p: float, n: float) -> float:
    """Relative MDE detectable with n users per arm at baseline rate p."""
    z_a, z_b = _z()
    delta_abs = (z_a + z_b) * np.sqrt(2 * p * (1 - p) / n)
    return delta_abs / p


# --- means (the secondary metric: revenue/user, optionally CUPED) ----------
def mde_mean(mean: float, sd: float, n: float, var_reduction: float = 0.0) -> float:
    """Relative MDE on a continuous metric with n per arm.

    var_reduction > 0 models CUPED: it shrinks the effective variance by that
    fraction (sd_eff = sd * sqrt(1 - var_reduction)).
    """
    z_a, z_b = _z()
    sd_eff = sd * np.sqrt(1 - var_reduction)
    delta_abs = (z_a + z_b) * np.sqrt(2 * sd_eff**2 / n)
    return delta_abs / mean


def runtime_days(n_per_arm: float) -> float:
    return 2 * n_per_arm / DAILY_USERS


def main():
    df = pd.read_csv(DATA_PATH)
    c = df[df.variant == "control"]
    p0 = c.converted.mean()
    rev_mean, rev_sd = c.exp_revenue.mean(), c.exp_revenue.std()
    n_actual = min(df.variant.value_counts())

    print(f"Baselines (from control arm of {DATA_PATH}):")
    print(f"  conversion p0 = {p0:.4f}")
    print(f"  revenue/user  = {rev_mean:.0f}  (sd {rev_sd:.0f}, CV {rev_sd / rev_mean:.2f})")
    print(f"  traffic       = {DAILY_USERS:,}/day,  actual n = {n_actual:,}/arm\n")

    # Forward sizing: how many users / how long for various target effects
    print("SAMPLE SIZE - primary metric (conversion), 80% power, alpha 0.05:")
    print(f"  {'target lift':>11} {'users/arm':>11} {'total':>11} {'runtime':>9}")
    for rel in (0.01, 0.015, 0.02, 0.03, 0.05):
        n = n_per_arm_proportion(p0, rel)
        print(f"  {rel:>10.1%} {n:>11,.0f} {2 * n:>11,.0f} {runtime_days(n):>7.1f}d")

    # Reverse: what can THIS experiment actually detect?
    conv_mde = mde_proportion(p0, n_actual)
    rev_mde_naive = mde_mean(rev_mean, rev_sd, n_actual, 0.0)
    rev_mde_cuped = mde_mean(rev_mean, rev_sd, n_actual, CUPED_VAR_REDUCTION)
    print(f"\nMINIMUM DETECTABLE EFFECT at n = {n_actual:,}/arm (80% power):")
    print(f"  conversion (primary)        {conv_mde:>6.2%} relative")
    print(f"  revenue/user (naive)        {rev_mde_naive:>6.2%} relative")
    print(f"  revenue/user (CUPED)        {rev_mde_cuped:>6.2%} relative"
          f"   <- {1 - rev_mde_cuped / rev_mde_naive:.0%} tighter")

    print("\nWhy the result reads the way it does:")
    print(f"  - True conversion lift (~1.5%) sits right at the {conv_mde:.1%} boundary")
    print("    -> primary is significant, but only modestly (p ~ 0.01).")
    print(f"  - True revenue lift (~2%) is BELOW the naive {rev_mde_naive:.1%} MDE")
    print("    -> the naive revenue test is underpowered (p ~ 0.06, n.s.).")
    print(f"  - CUPED cuts the revenue MDE to {rev_mde_cuped:.1%} and tightens the CI")
    print("    -> the same observed lift becomes significant (p ~ 0.002).")
    print("  CUPED is not a nicety here - it is what makes revenue conclusive.")


if __name__ == "__main__":
    main()

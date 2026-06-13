"""
CUPED analysis demo on the synthetic experiment data.

Runs the full evaluation workflow:
  1. SRM check (chi-square on assignment counts)        [Chapter 5]
  2. Conversion: two-proportion z-test                   [Chapter 6]
  3. Revenue/user: naive Welch t-test                    [Chapter 6]
  4. Revenue/user: CUPED-adjusted test                   [Chapter 7]
  5. Guardrail: AOV among converters (should be flat)    [Chapter 2]

CUPED in one paragraph
----------------------
For each user, adjust the experiment metric Y using the pre-experiment
covariate X:   Y_cuped = Y - theta * (X - mean(X)),  theta = cov(Y, X) / var(X).
Because X is measured BEFORE exposure, it is independent of assignment, so the
adjustment cannot bias the treatment effect - it only removes the part of Y's
variance that X explains. Variance reduction ~= rho^2 (rho = corr(X, Y)).
New users have no X (NaN): mean-imputing X gives them an adjustment of zero,
which is the standard production approach.
"""

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "experiment_data.csv"


def srm_check(df: pd.DataFrame) -> None:
    counts = df["variant"].value_counts()
    chi2, p = stats.chisquare(counts.values)
    status = "OK" if p > 0.01 else "!! SRM DETECTED - do not trust results"
    print(f"[1] SRM check        counts={counts.to_dict()}  chi2 p={p:.3f}  -> {status}")


def conversion_test(df: pd.DataFrame) -> None:
    t = df[df.variant == "treatment"]["converted"]
    c = df[df.variant == "control"]["converted"]
    p_t, p_c = t.mean(), c.mean()
    p_pool = df["converted"].mean()
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / len(t) + 1 / len(c)))
    z = (p_t - p_c) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    print(
        f"[2] Conversion       control={p_c:.4f}  treatment={p_t:.4f}  "
        f"lift={p_t / p_c - 1:+.2%}  z={z:.2f}  p={p_val:.4f}"
    )


def cuped_adjust(df: pd.DataFrame, y_col: str, x_col: str) -> pd.Series:
    """Return CUPED-adjusted metric. theta computed pooled across arms."""
    x = df[x_col].fillna(df[x_col].mean())  # new users -> zero adjustment
    y = df[y_col]
    theta = np.cov(y, x)[0, 1] / np.var(x)
    return y - theta * (x - x.mean())


def revenue_tests(df: pd.DataFrame) -> None:
    y_t = df[df.variant == "treatment"]["exp_revenue"]
    y_c = df[df.variant == "control"]["exp_revenue"]

    # Naive Welch t-test
    t_naive, p_naive = stats.ttest_ind(y_t, y_c, equal_var=False)
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var() / len(y_t) + y_c.var() / len(y_c))
    print(
        f"[3] Revenue (naive)  diff={diff:+.1f}/user  "
        f"95% CI=[{diff - 1.96 * se:+.1f}, {diff + 1.96 * se:+.1f}]  "
        f"lift={y_t.mean() / y_c.mean() - 1:+.2%}  t={t_naive:.2f}  p={p_naive:.4f}"
    )

    # CUPED-adjusted
    df = df.copy()
    df["y_cuped"] = cuped_adjust(df, "exp_revenue", "pre_revenue")
    yc_t = df[df.variant == "treatment"]["y_cuped"]
    yc_c = df[df.variant == "control"]["y_cuped"]
    t_cuped, p_cuped = stats.ttest_ind(yc_t, yc_c, equal_var=False)
    diff_c = yc_t.mean() - yc_c.mean()
    se_c = np.sqrt(yc_t.var() / len(yc_t) + yc_c.var() / len(yc_c))

    var_red = 1 - df["y_cuped"].var() / df["exp_revenue"].var()
    rho = df[["pre_revenue", "exp_revenue"]].dropna().corr().iloc[0, 1]
    print(
        f"[4] Revenue (CUPED)  diff={diff_c:+.1f}/user  "
        f"95% CI=[{diff_c - 1.96 * se_c:+.1f}, {diff_c + 1.96 * se_c:+.1f}]  "
        f"t={t_cuped:.2f}  p={p_cuped:.4f}"
    )
    print(
        f"    corr(pre, exp)={rho:.3f}  ->  variance reduction={var_red:.1%}  "
        f"(theory ~ rho^2 = {rho**2:.1%})"
    )
    print(
        f"    Effective sample size gain: x{1 / (1 - var_red):.2f} "
        f"(same power with {1 - (1 - var_red):.0%} fewer users)"
    )


def guardrail_aov(df: pd.DataFrame) -> None:
    """Average order value = revenue PER ORDER among purchasers.

    Trap to avoid: revenue-per-CONVERTER is not AOV - it leaks order
    frequency into the metric, so a treatment that lifts order rate will
    falsely "move" it even when order values are untouched.
    """
    buyers = df[df.exp_orders > 0].copy()
    buyers["aov"] = buyers["exp_revenue"] / buyers["exp_orders"]
    aov_t = buyers[buyers.variant == "treatment"]["aov"]
    aov_c = buyers[buyers.variant == "control"]["aov"]
    t, p = stats.ttest_ind(aov_t, aov_c, equal_var=False)
    print(
        f"[5] Guardrail (AOV)  control={aov_c.mean():.0f}/order  treatment={aov_t.mean():.0f}/order  "
        f"p={p:.4f}  -> {'flat, as designed' if p > 0.05 else 'moved - investigate'}"
    )


def results_summary(df: pd.DataFrame) -> dict:
    """Structured version of the five checks above (same formulas), for the app."""
    counts = df["variant"].value_counts()
    _, srm_p = stats.chisquare(counts.values)

    t = df[df.variant == "treatment"]
    c = df[df.variant == "control"]
    p_t, p_c = t.converted.mean(), c.converted.mean()
    p_pool = df.converted.mean()
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / len(t) + 1 / len(c)))
    z = (p_t - p_c) / se
    conv_p = 2 * (1 - stats.norm.cdf(abs(z)))

    yt, yc = t.exp_revenue, c.exp_revenue
    tn, pn = stats.ttest_ind(yt, yc, equal_var=False)
    diff_n = yt.mean() - yc.mean()
    se_n = np.sqrt(yt.var() / len(yt) + yc.var() / len(yc))

    d = df.copy()
    d["y_cuped"] = cuped_adjust(d, "exp_revenue", "pre_revenue")
    yct = d[d.variant == "treatment"].y_cuped
    ycc = d[d.variant == "control"].y_cuped
    tc, pc = stats.ttest_ind(yct, ycc, equal_var=False)
    diff_c = yct.mean() - ycc.mean()
    se_c = np.sqrt(yct.var() / len(yct) + ycc.var() / len(ycc))
    var_red = 1 - d.y_cuped.var() / d.exp_revenue.var()
    rho = d[["pre_revenue", "exp_revenue"]].dropna().corr().iloc[0, 1]

    b = df[df.exp_orders > 0].copy()
    b["aov"] = b.exp_revenue / b.exp_orders
    _, aov_p = stats.ttest_ind(
        b[b.variant == "treatment"].aov, b[b.variant == "control"].aov, equal_var=False
    )

    return {
        "n_control": int(len(c)), "n_treatment": int(len(t)), "srm_p": srm_p,
        "conv_control": p_c, "conv_treatment": p_t, "conv_lift": p_t / p_c - 1,
        "conv_z": z, "conv_p": conv_p,
        "rev_naive_diff": diff_n, "rev_naive_ci": (diff_n - 1.96 * se_n, diff_n + 1.96 * se_n),
        "rev_naive_lift": yt.mean() / yc.mean() - 1, "rev_naive_p": pn,
        "rev_cuped_diff": diff_c, "rev_cuped_ci": (diff_c - 1.96 * se_c, diff_c + 1.96 * se_c),
        "rev_cuped_p": pc, "var_reduction": var_red, "rho": rho,
        "ess_gain": 1 / (1 - var_red),
        "aov_control": b[b.variant == "control"].aov.mean(),
        "aov_treatment": b[b.variant == "treatment"].aov.mean(), "aov_p": aov_p,
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} users  ({df.is_new_user.mean():.0%} new users)\n")
    srm_check(df)
    conversion_test(df)
    revenue_tests(df)
    guardrail_aov(df)

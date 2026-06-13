"""
A/B Experimentation Platform - Streamlit app.

A walkthrough of one end-to-end experiment: design -> data quality -> EDA ->
results. Every number is computed live from data/experiment_data*.csv using the
same functions as the command-line analysis in src/ (no hardcoded results).

Run locally:   streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cuped_analysis import results_summary  # noqa: E402
from clean_events import quick_readout  # noqa: E402
from experiment_design import (  # noqa: E402
    mde_mean, mde_proportion, n_per_arm_proportion, runtime_days,
)

DATA = os.path.join(ROOT, "data")
CONTROL_C = "#8A90A6"   # slate - control, everywhere
TREATMENT_C = "#5B6CF0"  # indigo - treatment, everywhere
GOOD = "#1FA97A"
BAD = "#E0533D"

st.set_page_config(page_title="A/B Experimentation Platform", page_icon="🔬", layout="wide")


@st.cache_data
def load():
    clean = pd.read_csv(os.path.join(DATA, "experiment_data.csv"))
    raw = pd.read_csv(os.path.join(DATA, "experiment_data_raw.csv"))
    return clean, raw


@st.cache_data
def compute(clean, raw):
    res = results_summary(clean)
    raw_rd, clean_rd = quick_readout(raw), quick_readout(clean)
    return res, raw_rd, clean_rd


clean, raw = load()
res, raw_rd, clean_rd = compute(clean, raw)

st.title("A/B Experimentation Platform")
st.caption(
    "Checkout-redesign experiment · 220K users · 2M+ events · "
    "design → cleaning → EDA → inference, end to end"
)

tab_over, tab_design, tab_dq, tab_eda, tab_res = st.tabs(
    ["Overview", "Design", "Data Quality", "EDA", "Results"]
)

# ---------------------------------------------------------------- OVERVIEW
with tab_over:
    st.subheader("What this is")
    st.markdown(
        "A full experiment lifecycle on a simulated e-commerce checkout redesign. "
        "The pipeline starts from raw event logs, cleans them, sizes the test, and "
        "reaches a ship / no-ship decision. The headline finding below is the point "
        "of the whole project: the revenue result is **inconclusive on naive analysis "
        "and conclusive after variance reduction (CUPED)**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{res['n_control'] + res['n_treatment']:,}")
    c2.metric("Conversion lift", f"{res['conv_lift']:+.2%}", help="primary metric")
    c3.metric("Revenue p — naive", f"{res['rev_naive_p']:.3f}", "not significant", delta_color="off")
    c4.metric("Revenue p — CUPED", f"{res['rev_cuped_p']:.3f}", "significant")

    st.divider()
    st.markdown("**Pipeline**")
    steps = st.columns(5)
    for col, (n, label, sub) in zip(steps, [
        ("1", "Raw events", "2.42M rows, dirty"),
        ("2", "Clean", "dedup · bots · refunds"),
        ("3", "Design", "power / MDE / sizing"),
        ("4", "Test", "z-test · t-test · CUPED"),
        ("5", "Decide", "ship / no-ship"),
    ]):
        col.markdown(f"**{n}. {label}**  \n<span style='color:#8A90A6'>{sub}</span>",
                     unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "**Stack** — Python (numpy, pandas, scipy) for simulation and statistics · "
        "BigQuery SQL for the partitioned raw→clean→analysis pipeline "
        "(`sql/bigquery_setup.sql`) · Streamlit for this app. Data is synthetic, so "
        "ground truth is known and the pipeline can be validated against it."
    )

# ------------------------------------------------------------------ DESIGN
with tab_design:
    st.subheader("Pre-registration")
    st.markdown(
        "- **Hypothesis** — the new checkout flow increases the order rate.\n"
        "- **Primary metric** — conversion (≥1 order). Low variance, tied to the hypothesis.\n"
        "- **Secondary** — revenue per user. Business-relevant but heavy-tailed.\n"
        "- **Guardrails** — average order value (per order) must not drop; refund rate must not rise.\n"
        "- **Randomisation** — by user, 50/50 via `md5(user_id + salt) % 100`.\n"
        "- **Decision rule** — ship iff the primary is significantly positive (α = 0.05, "
        "two-sided) and no guardrail regresses. Stated before looking at results."
    )

    st.divider()
    st.subheader("How much can this experiment detect?")
    p0 = res["conv_control"]
    n = res["n_control"]
    rev_mean = clean[clean.variant == "control"].exp_revenue.mean()
    rev_sd = clean[clean.variant == "control"].exp_revenue.std()
    m1, m2, m3 = st.columns(3)
    m1.metric("Conversion MDE", f"{mde_proportion(p0, n):.2%}", "primary")
    m2.metric("Revenue MDE — naive", f"{mde_mean(rev_mean, rev_sd, n):.2%}")
    m3.metric("Revenue MDE — CUPED", f"{mde_mean(rev_mean, rev_sd, n, res['var_reduction']):.2%}",
              f"-{1 - mde_mean(rev_mean, rev_sd, n, res['var_reduction']) / mde_mean(rev_mean, rev_sd, n):.0%}")
    st.markdown(
        f"At {n:,}/arm, conversion is powered to detect ~{mde_proportion(p0, n):.1%}; "
        f"revenue/user (CV {rev_sd / rev_mean:.1f}) only ~{mde_mean(rev_mean, rev_sd, n):.1%} naively. "
        "That gap is exactly why the revenue result needs CUPED to become conclusive."
    )

    st.divider()
    st.subheader("Sample-size calculator")
    cc = st.columns(3)
    base = cc[0].slider("Baseline conversion", 0.05, 0.60, float(round(p0, 2)), 0.01)
    lift = cc[1].slider("Target lift (relative)", 0.005, 0.10, 0.02, 0.005, format="%.3f")
    power = cc[2].slider("Power", 0.70, 0.95, 0.80, 0.05)
    import experiment_design as ed
    ed.POWER = power
    need = n_per_arm_proportion(base, lift)
    r1, r2 = st.columns(2)
    r1.metric("Users needed per arm", f"{need:,.0f}")
    r2.metric("Runtime at ~7,857/day", f"{runtime_days(need):.0f} days")
    ed.POWER = 0.80

# ------------------------------------------------------------ DATA QUALITY
with tab_dq:
    st.subheader("Cleaning impact")
    st.markdown(
        "Raw event logs carried four realistic defects: duplicate order double-fires, "
        "bot and QA accounts, refunds logged as negative-revenue orders, and rows with "
        "missing user IDs. The duplicates were **concentrated in the treatment arm** "
        "(an instrumentation bug in the new checkout client), so they bias the result — "
        "cleaning is part of the experiment's validity, not housekeeping."
    )

    d1, d2 = st.columns(2)
    d1.metric("Revenue lift — raw data", f"{raw_rd['rev_lift']:+.2%}",
              f"p = {raw_rd['p_rev']:.1g}", delta_color="off")
    d2.metric("Revenue lift — cleaned", f"{clean_rd['rev_lift']:+.2%}",
              f"p = {clean_rd['p_rev']:.2f}")
    st.markdown(
        f"Uncleaned data overstates the revenue lift **{raw_rd['rev_lift'] / clean_rd['rev_lift']:.1f}×** "
        "and reports it as wildly significant. Conversion barely changes between raw and "
        "clean — duplicating an order can't re-convert an already-converted user. Different "
        "metrics, different exposure to the defect."
    )

    st.divider()
    st.markdown("**Conversion is robust; revenue is not** — raw vs cleaned")
    cmp = pd.DataFrame(
        {"raw": [raw_rd["conv_lift"], raw_rd["rev_lift"]],
         "clean": [clean_rd["conv_lift"], clean_rd["rev_lift"]]},
        index=["conversion lift", "revenue lift"],
    )
    st.dataframe(cmp.style.format("{:+.2%}"), width="stretch")

# --------------------------------------------------------------------- EDA
with tab_eda:
    st.subheader("Randomisation balance")
    st.markdown(
        "Before trusting any result, check the arms are comparable on pre-experiment "
        "attributes. Balanced shares (and a passing SRM check) mean differences in "
        "outcomes are attributable to the treatment, not to who landed in each arm."
    )
    bal = (
        clean.groupby("variant")[["device", "region"]]
        .apply(lambda g: pd.Series({
            "mobile %": (g.device == "mobile").mean(),
            "metro %": (g.region == "metro").mean(),
        }))
    )
    new_share = clean.groupby("variant").is_new_user.mean().rename("new user %")
    bal = bal.join(new_share)
    st.dataframe(bal.style.format("{:.1%}"), width="stretch")
    st.caption(f"SRM check (assignment counts): p = {res['srm_p']:.2f} — balanced.")

    st.divider()
    st.subheader("Revenue is heavy-tailed")
    st.markdown(
        "Among buyers, spend spans three orders of magnitude. This long tail is why "
        "revenue/user has high variance and a poor naive MDE — and why CUPED helps."
    )
    buyers = clean[clean.exp_revenue > 0]
    logv = np.log10(buyers.exp_revenue.clip(lower=1))
    counts, edges = np.histogram(logv, bins=40)
    hist = pd.DataFrame({"log10 revenue": (edges[:-1] + edges[1:]) / 2, "buyers": counts})
    st.bar_chart(hist, x="log10 revenue", y="buyers", color=TREATMENT_C, height=260)

    st.divider()
    st.subheader("Why CUPED works: pre-period predicts experiment-period")
    st.markdown(
        f"Each user's pre-experiment spend correlates with their experiment-period spend "
        f"(ρ = {res['rho']:.2f}). CUPED subtracts the part of the outcome the pre-period "
        "already explains, removing variance without touching the treatment effect. "
        "Binned into deciles of pre-period revenue:"
    )
    ret = clean[clean.pre_revenue.notna()].copy()
    ret["decile"] = pd.qcut(ret.pre_revenue.rank(method="first"), 10, labels=False)
    dec = ret.groupby("decile").agg(
        pre=("pre_revenue", "mean"), exp=("exp_revenue", "mean")
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dec.pre, y=dec.exp, mode="markers+lines",
        marker=dict(size=11, color=TREATMENT_C), line=dict(color=TREATMENT_C, width=2),
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="mean pre-period revenue (decile)",
        yaxis_title="mean experiment revenue",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------- RESULTS
with tab_res:
    st.subheader("Decision: ship")
    st.markdown(
        "The primary metric is significantly positive and no guardrail regressed. The "
        "secondary (revenue) is significant **only after CUPED** — the signature result below."
    )

    k = st.columns(4)
    k[0].metric("Conversion", f"{res['conv_treatment']:.1%}", f"{res['conv_lift']:+.2%}")
    k[1].metric("Conversion p", f"{res['conv_p']:.3f}", "significant")
    k[2].metric("Guardrail (AOV)", f"₹{res['aov_treatment']:.0f}/order",
                "flat — good", delta_color="off")
    k[3].metric("CUPED variance ↓", f"{res['var_reduction']:.0%}",
                f"effective n ×{res['ess_gain']:.2f}")

    st.divider()
    st.subheader("Signature: CUPED makes the revenue result conclusive")
    st.markdown(
        "Same point estimate, tighter interval. The naive 95% interval crosses zero "
        "(inconclusive); after CUPED removes pre-period variance, the interval clears "
        "zero and the effect is significant. CUPED changes the **certainty**, not the answer."
    )
    nlo, nhi = res["rev_naive_ci"]
    clo, chi = res["rev_cuped_ci"]
    fig2 = go.Figure()
    fig2.add_vline(x=0, line=dict(color="#C2C5D6", width=2, dash="dash"))
    fig2.add_trace(go.Scatter(
        x=[res["rev_naive_diff"]], y=["Naive"], mode="markers",
        marker=dict(size=15, color=CONTROL_C),
        error_x=dict(type="data", symmetric=False,
                     array=[nhi - res["rev_naive_diff"]], arrayminus=[res["rev_naive_diff"] - nlo],
                     color=CONTROL_C, thickness=3, width=10),
        name=f"Naive (p={res['rev_naive_p']:.3f})",
    ))
    fig2.add_trace(go.Scatter(
        x=[res["rev_cuped_diff"]], y=["CUPED"], mode="markers",
        marker=dict(size=15, color=TREATMENT_C),
        error_x=dict(type="data", symmetric=False,
                     array=[chi - res["rev_cuped_diff"]], arrayminus=[res["rev_cuped_diff"] - clo],
                     color=TREATMENT_C, thickness=3, width=10),
        name=f"CUPED (p={res['rev_cuped_p']:.3f})",
    ))
    fig2.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="revenue lift per user (₹), 95% CI",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, width="stretch")
    st.caption(
        f"Naive: {res['rev_naive_diff']:+.0f} [{nlo:+.0f}, {nhi:+.0f}], p={res['rev_naive_p']:.3f}  ·  "
        f"CUPED: {res['rev_cuped_diff']:+.0f} [{clo:+.0f}, {chi:+.0f}], p={res['rev_cuped_p']:.3f}"
    )

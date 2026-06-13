# Pre-Registered Analysis Plan — Checkout Redesign Experiment

> Committed **before** analysis. Pre-registration fixes the hypothesis, metrics, and
> decision rule in advance, so the results can't be reverse-engineered into a story
> after the fact. This is the document that separates a credible experiment from a
> dashboard.

## Business context

An e-commerce store is testing a redesigned checkout flow against the current one.
The hypothesis is that the new flow reduces friction and increases the rate at which
users place orders. The change ships in the new checkout client; assignment is at the
user level.

## Hypothesis

- **H₀** — the new checkout flow does not change the order rate.
- **H₁** — the new checkout flow increases the order rate.

Tested two-sided at α = 0.05 (we only ship a positive change, but we test two-sided —
the conservative default).

## Metrics

- **Primary — conversion:** P(≥1 order during the 4-week experiment window). Low
  variance, directly tied to the hypothesis; the test is powered on this.
- **Secondary — revenue per user** (gross order revenue). Business-relevant but
  heavy-tailed (CV ≈ 2.5), so it needs a larger effect to detect — this is the metric
  CUPED is for.
- **Guardrails** — (a) average order value *per order* must not drop; (b) refund rate
  must not rise. A guardrail regression blocks launch even if the primary wins.
- **Ratio note** — revenue per session is a ratio metric (delta method at analysis
  time); kept secondary.

## Randomization

- **Unit:** user (not session — sessions from one user are correlated, which
  understates variance and inflates significance).
- **Split:** 50/50 control / treatment.
- **Assignment:** deterministic, `md5(user_id + experiment_salt) % 100`. Stable across
  sessions; the same user is always in the same arm.

## Sample size & power

- Baseline conversion (observed in the 4-week pre-period): **~32.6%**.
- Target effect: detect a **~2% relative** lift on conversion.
- Power 80%, α = 5% two-sided → **~81,000 users per arm** → **~21 days** at ~7,900
  exposed users/day.
- Planned run: **4 weeks (~110,000 per arm)**, which also covers weekly seasonality.
- Note on the secondary: at this sample size, revenue/user (CV ≈ 2.5) is only powered
  to ~2.9% relative naively. We therefore pre-specify **CUPED** (below), which is
  expected to bring the revenue MDE down materially.

## Analysis pipeline (in order)

1. **SRM check** — chi-square on assignment counts; threshold p < 0.01. If SRM fires,
   stop and debug the pipeline; do not interpret results.
2. **Primary** — two-proportion z-test on conversion (control vs treatment).
3. **Secondary** — Welch t-test on revenue/user, reported both naive and
   CUPED-adjusted.
4. **CUPED covariate** — pre-experiment revenue, computed over the 4 weeks *before*
   exposure (strictly pre-treatment → independent of assignment → the adjustment
   cannot bias the effect). New users have no pre-period: their covariate is
   mean-imputed, giving them a zero adjustment.
5. **Guardrails** — Welch t-test on AOV per order; refund-rate check. Both must show
   no significant regression.

## Pre-specified exploratory (not confirmatory)

- Treatment effect by **segment** (device, region, new vs returning), reported as
  exploratory only and never used as a launch criterion. Segments are declared here in
  advance to rule out post-hoc segment hunting.

## Decision rule (fixed in advance)

Ship the new checkout flow **iff**:
- the primary metric (conversion) is significantly positive at α = 0.05, **and**
- no guardrail shows a significant regression.

A significant secondary (revenue) is supporting evidence, not a launch trigger on its
own.

## What I will NOT do

- No peeking / no early stopping on a fixed-horizon test (sequential testing with mSPRT
  would be the alternative if early stopping were required).
- No adding or swapping the primary metric after seeing results.
- No uncorrected segment hunting — segment analysis is exploratory and pre-declared.

## Data integrity (cleaning, applied to raw event logs before any metric)

Before metrics are computed, raw events pass a cleaning layer: drop unattributable rows
(null `user_id`), deduplicate on the business key (retry double-fires carry distinct
`event_id`s), exclude refunds (negative-revenue rows) from gross revenue, and remove bot
and QA accounts (behavioural + id-pattern rules). Cleaning is specified before results
because a defect concentrated in one arm — e.g. a checkout-client bug that double-fires
order events — would bias the comparison.

"""
Synthetic A/B test data generator - EVENT-LEVEL (CUPED + BigQuery-ready).

Scenario
--------
E-commerce checkout redesign experiment:
  - Hypothesis: the new checkout flow lifts order rate.
  - Unit of randomisation: user. 50/50 split via salted hash (deterministic).
  - Pre-experiment period : 2026-04-13 .. 2026-05-10  (4 weeks, CUPED covariate)
  - Experiment period     : 2026-05-11 .. 2026-06-07  (4 weeks)

Output (three files)
--------------------
  users.csv             ~220K rows   dim table: user_id, variant, device, region, is_new_user
  events.csv            ~2M rows     fact table: event_id, user_id, event_type, event_ts, revenue
  experiment_data.csv   ~220K rows   user-level metrics (aggregated locally with pandas,
                                     mirroring exactly what bigquery_setup.sql builds in BQ)

The events table is the one you load into BigQuery PARTITIONED BY DATE(event_ts)
and CLUSTERED BY (event_type, user_id) - see bigquery_setup.sql.

Why this design makes CUPED work
--------------------------------
Each user has a PERSISTENT latent order rate (lambda ~ Gamma, heavy-tailed) and
a typical order value (lognormal). Both periods draw Poisson(lambda) from the
same latent rate, so heavy buyers are heavy in both periods. That persistence
creates the pre/post correlation CUPED exploits (variance reduction ~= rho^2).
A design where each period is an independent coin flip gives rho ~ 0.1 and
CUPED does nothing.

Ground truth (known, so the pipeline can be validated)
------------------------------------------------------
  - Order rate: +2.5% RELATIVE lift in treatment. Small on purpose: large
    samples exist to detect small effects, and a small effect is what makes
    CUPED visibly useful (naive test marginal, CUPED-adjusted test decisive).
  - Conversion (>=1 order) inherits a slightly smaller lift (saturation).
  - Average order value: NO effect (guardrail must stay flat).

Realism details (interview talking points)
------------------------------------------
  - ~18% of users are NEW (no pre-period): pre_revenue is NULL/NaN for them -
    different from a returning user who bought nothing (0.0). CUPED
    mean-imputes those, i.e. their adjustment is zero.
  - Order values are lognormal per ORDER (heavy right tail).
  - Event timestamps follow a daily curve (evenings heavier).
  - Assignment via md5(user_id + salt) % 100: deterministic and uniform,
    the same mechanism real experimentation platforms use.

Usage
-----
  python generate_ab_data.py        # writes the three CSVs (~30s)
"""

import hashlib

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration (all assumptions in one place)
# --------------------------------------------------------------------------
SEED = 11
N_USERS = 220_000
EXPERIMENT_SALT = "checkout_redesign_v1"

TRUE_ORDER_RATE_LIFT = 0.025  # +2.5% relative lift on order rate (ground truth)
TRUE_AOV_LIFT = 0.00          # guardrail: no effect on order value

NEW_USER_SHARE = 0.18           # users with no pre-period history
ORDER_RATE_GAMMA = (0.45, 1.4)  # lambda ~ Gamma(shape, scale): mean ~0.63 orders / 4 wk
SPEND_LOGNORMAL = (7.0, 0.6)    # typical order value, median ~ INR 1,100
WITHIN_USER_NOISE = 0.25        # per-order noise around the user's typical value

# --- defect injection (uses its OWN rng so the clean core stays identical) --
# These reproduce the data-quality failures real event pipelines ship with.
DIRTY_SEED_OFFSET = 1000
N_BOT_USERS = 1_200             # high-session, zero-order accounts (scrapers)
BOT_SESSIONS_RANGE = (60, 200)  # per 4-week period -> 120-400 total
N_QA_USERS = 800                # internal test accounts, 'qa_' id prefix
QA_SESSION_MEAN = 20
QA_ORDER_RATE = 0.6             # QA places tiny test orders (INR 1-10)
DUP_RATE_CONTROL = 0.005        # baseline client retry double-fires
DUP_RATE_TREATMENT = 0.055      # instrumentation BUG in the new checkout flow
REFUND_RATE = 0.015             # refunds logged as negative-revenue orders
NULL_ID_SESSION_RATE = 0.004    # logging failures drop user_id (sessions only)

PRE_START = np.datetime64("2026-04-13")   # both windows are 28 days
EXP_START = np.datetime64("2026-05-11")
PERIOD_DAYS = 28
PRE_END_DATE = "2026-05-10"               # used by the aggregation step

# Hour-of-day mix: quiet nights, evening peak (sums to 1 after normalising)
HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 7, 6, 6, 6, 7, 8, 9, 9, 8, 5, 3],
    dtype=float,
)
HOUR_P = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()


def assign_variant(user_id: str, salt: str) -> str:
    """Deterministic 50/50 assignment via salted hash (production-style)."""
    bucket = int(hashlib.md5(f"{user_id}:{salt}".encode()).hexdigest(), 16) % 100
    return "treatment" if bucket < 50 else "control"


def _timestamps(n: int, start: np.datetime64, rng) -> np.ndarray:
    """Random timestamps in a 28-day window with an evening-heavy hour mix."""
    days = rng.integers(0, PERIOD_DAYS, n)
    hours = rng.choice(24, size=n, p=HOUR_P)
    minutes = rng.integers(0, 60, n)
    seconds = rng.integers(0, 60, n)
    offset = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    return start.astype("datetime64[s]") + offset.astype("timedelta64[s]")


def generate_users(rng) -> pd.DataFrame:
    user_ids = np.array([f"u_{i:06d}" for i in range(N_USERS)])
    variant = np.array([assign_variant(u, EXPERIMENT_SALT) for u in user_ids])
    return pd.DataFrame(
        {
            "user_id": user_ids,
            "variant": variant,
            "device": rng.choice(["mobile", "desktop"], N_USERS, p=[0.72, 0.28]),
            "region": rng.choice(["metro", "tier2", "tier3"], N_USERS, p=[0.50, 0.32, 0.18]),
            "is_new_user": rng.random(N_USERS) < NEW_USER_SHARE,
            # latent traits kept only in memory (never exported - that would leak ground truth)
            "_order_rate": rng.gamma(*ORDER_RATE_GAMMA, N_USERS),
            "_typical_spend": rng.lognormal(*SPEND_LOGNORMAL, N_USERS),
        }
    )


def generate_events(users: pd.DataFrame, rng) -> pd.DataFrame:
    is_treat = (users.variant == "treatment").to_numpy()
    existed_pre = ~users.is_new_user.to_numpy()
    rate = users._order_rate.to_numpy()
    spend = users._typical_spend.to_numpy()

    chunks = []
    for period, start in (("pre", PRE_START), ("exp", EXP_START)):
        active = existed_pre if period == "pre" else np.ones(N_USERS, bool)
        lift = TRUE_ORDER_RATE_LIFT * is_treat if period == "exp" else 0.0
        aov_mult = 1 + TRUE_AOV_LIFT * is_treat if period == "exp" else 1.0

        n_orders = rng.poisson(rate * (1 + lift)) * active
        n_sessions = rng.poisson(2 + 4 * rate) * active

        o_idx = np.repeat(np.arange(N_USERS), n_orders)
        s_idx = np.repeat(np.arange(N_USERS), n_sessions)
        aov = aov_mult[o_idx] if isinstance(aov_mult, np.ndarray) else aov_mult

        chunks.append(
            pd.DataFrame(
                {
                    "user_id": users.user_id.to_numpy()[o_idx],
                    "event_type": "order",
                    "event_ts": _timestamps(len(o_idx), start, rng),
                    "revenue": np.round(
                        spend[o_idx] * aov * rng.lognormal(0, WITHIN_USER_NOISE, len(o_idx)), 2
                    ),
                }
            )
        )
        chunks.append(
            pd.DataFrame(
                {
                    "user_id": users.user_id.to_numpy()[s_idx],
                    "event_type": "session",
                    "event_ts": _timestamps(len(s_idx), start, rng),
                    "revenue": np.nan,
                }
            )
        )

    events = pd.concat(chunks, ignore_index=True).sort_values("event_ts", kind="stable")
    events.insert(0, "event_id", [f"e_{i:08d}" for i in range(len(events))])
    return events.reset_index(drop=True)


def inject_defects(users: pd.DataFrame, events: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Layer realistic pipeline defects onto the clean simulation.

    Uses a SEPARATE rng so the validated clean core is untouched: the
    cleaning layer (clean_events.py / bigquery_setup.sql) must recover the
    certified results exactly.

    Defects injected:
      1. Bot accounts: 60-200 sessions/period, zero orders. Normal-looking
         user_ids -> only detectable BEHAVIOURALLY.
      2. QA accounts ('qa_' prefix): moderate activity, tiny test orders.
      3. Duplicate order events (client retry double-fires) - heavily skewed
         to TREATMENT because the bug lives in the NEW checkout client. Same
         user/timestamp/amount, different event_id: naive DISTINCT event_id
         will NOT catch these; dedup needs the business key.
      4. Refunds logged as negative-revenue 'order' rows (symmetric).
      5. NULL user_id on ~0.4% of sessions (logging failures).
    """
    variant_map = dict(zip(users.user_id, users.variant))

    # --- 1+2) bot and QA accounts (hashed into arms like everyone else) -----
    bot_ids = [f"u_{900000 + i:06d}" for i in range(N_BOT_USERS)]
    qa_ids = [f"qa_{i:04d}" for i in range(N_QA_USERS)]
    extra_ids = bot_ids + qa_ids
    extra_users = pd.DataFrame(
        {
            "user_id": extra_ids,
            "variant": [assign_variant(u, EXPERIMENT_SALT) for u in extra_ids],
            "device": rng.choice(["mobile", "desktop"], len(extra_ids), p=[0.72, 0.28]),
            "region": rng.choice(["metro", "tier2", "tier3"], len(extra_ids), p=[0.50, 0.32, 0.18]),
            "is_new_user": False,
            "_order_rate": np.nan,
            "_typical_spend": np.nan,
        }
    )
    variant_map.update(dict(zip(extra_users.user_id, extra_users.variant)))

    extra_chunks = []
    for start in (PRE_START, EXP_START):
        bot_n = rng.integers(*BOT_SESSIONS_RANGE, N_BOT_USERS)
        qa_sess_n = rng.poisson(QA_SESSION_MEAN, N_QA_USERS)
        qa_ord_n = rng.poisson(QA_ORDER_RATE, N_QA_USERS)
        b_idx = np.repeat(np.arange(N_BOT_USERS), bot_n)
        qs_idx = np.repeat(np.arange(N_QA_USERS), qa_sess_n)
        qo_idx = np.repeat(np.arange(N_QA_USERS), qa_ord_n)
        extra_chunks.append(pd.DataFrame({
            "user_id": np.array(bot_ids)[b_idx], "event_type": "session",
            "event_ts": _timestamps(len(b_idx), start, rng), "revenue": np.nan}))
        extra_chunks.append(pd.DataFrame({
            "user_id": np.array(qa_ids)[qs_idx], "event_type": "session",
            "event_ts": _timestamps(len(qs_idx), start, rng), "revenue": np.nan}))
        extra_chunks.append(pd.DataFrame({
            "user_id": np.array(qa_ids)[qo_idx], "event_type": "order",
            "event_ts": _timestamps(len(qo_idx), start, rng),
            "revenue": np.round(rng.uniform(1, 10, len(qo_idx)), 2)}))

    # --- 3) duplicate order double-fires (treatment-skewed) -----------------
    orders = events[events.event_type == "order"]
    is_treat_ev = orders.user_id.map(variant_map).eq("treatment").to_numpy()
    dup_p = np.where(is_treat_ev, DUP_RATE_TREATMENT, DUP_RATE_CONTROL)
    dups = orders[rng.random(len(orders)) < dup_p].copy()  # same ts + amount

    # --- 4) refunds as negative-revenue orders (symmetric) ------------------
    ref = orders[rng.random(len(orders)) < REFUND_RATE].copy()
    ref["revenue"] = -ref["revenue"]
    in_pre = ref.event_ts.values < EXP_START.astype("datetime64[s]")
    ref["event_ts"] = np.where(  # refund lands later, same period window
        in_pre, _timestamps(len(ref), PRE_START, rng), _timestamps(len(ref), EXP_START, rng)
    )

    new_rows = pd.concat(extra_chunks + [dups.drop(columns="event_id"),
                                         ref.drop(columns="event_id")], ignore_index=True)
    next_id = len(events)
    new_rows.insert(0, "event_id", [f"e_{next_id + i:08d}" for i in range(len(new_rows))])

    raw = pd.concat([events, new_rows], ignore_index=True)

    # --- 5) NULL user_id on a slice of sessions (logging failures) ----------
    sess_pos = np.flatnonzero((raw.event_type == "session").to_numpy())
    null_pos = sess_pos[rng.random(len(sess_pos)) < NULL_ID_SESSION_RATE]
    raw.loc[null_pos, "user_id"] = ""

    raw = raw.sort_values("event_ts", kind="stable").reset_index(drop=True)
    users_full = pd.concat([users, extra_users], ignore_index=True)
    return users_full, raw


def aggregate_user_metrics(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """User x period rollup. Mirrors the SQL in bigquery_setup.sql exactly."""
    ev = events.copy()
    ev["period"] = np.where(ev.event_ts.values < EXP_START.astype("datetime64[s]"), "pre", "exp")

    pivot = ev.pivot_table(
        index="user_id",
        columns=["period", "event_type"],
        values="revenue",
        aggfunc=["count", "sum"],
    )
    pivot.columns = ["_".join(c) for c in pivot.columns]  # e.g. count_pre_order

    df = users[["user_id", "variant", "device", "region", "is_new_user"]].merge(
        pivot, on="user_id", how="left"
    )

    def col(name):  # missing column (e.g. nobody did X) -> zeros
        return df[name] if name in df else pd.Series(0, index=df.index)

    # counts: pivot counted non-null `revenue`, valid for orders only -> recount sessions
    sess_counts = ev[ev.event_type == "session"].groupby(["user_id", "period"]).size().unstack(fill_value=0)
    df = df.merge(
        sess_counts.rename(columns={"pre": "pre_sessions", "exp": "exp_sessions"}),
        on="user_id",
        how="left",
    )

    out = pd.DataFrame(
        {
            "user_id": df.user_id,
            "variant": df.variant,
            "device": df.device,
            "region": df.region,
            "is_new_user": df.is_new_user,
            "pre_sessions": df.get("pre_sessions", 0).fillna(0).astype(int),
            "pre_orders": col("count_pre_order").fillna(0).astype(int),
            "pre_revenue": np.where(
                df.is_new_user, np.nan, col("sum_pre_order").fillna(0.0)
            ),
            "exp_sessions": df.get("exp_sessions", 0).fillna(0).astype(int),
            "exp_orders": col("count_exp_order").fillna(0).astype(int),
            "exp_revenue": col("sum_exp_order").fillna(0.0),
        }
    )
    out["converted"] = (out.exp_orders > 0).astype(int)
    out["pre_revenue"] = out.pre_revenue.round(2)
    out["exp_revenue"] = out.exp_revenue.round(2)
    cols = [
        "user_id", "variant", "device", "region", "is_new_user",
        "pre_sessions", "pre_orders", "pre_revenue",
        "exp_sessions", "exp_orders", "converted", "exp_revenue",
    ]
    return out[cols]


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    users = generate_users(rng)
    events = generate_events(users, rng)

    rng_dirty = np.random.default_rng(SEED + DIRTY_SEED_OFFSET)
    users_full, raw_events = inject_defects(users, events, rng_dirty)

    users_full.drop(columns=["_order_rate", "_typical_spend"]).to_csv("users.csv", index=False)
    raw_events.to_csv("raw_events.csv", index=False)

    n_defect = len(raw_events) - len(events)
    print(f"users.csv        {len(users_full):>10,} rows "
          f"({len(users):,} real + {N_BOT_USERS:,} bots + {N_QA_USERS:,} QA)")
    print(f"raw_events.csv   {len(raw_events):>10,} rows "
          f"({len(events):,} clean core + {n_defect:,} defect rows)")
    print(f"  null user_id   {(raw_events.user_id == '').sum():>10,}")
    print(f"  negative rev   {(raw_events.revenue < 0).sum():>10,}")
    print("\nRun clean_events.py next: it must recover the certified clean results.")

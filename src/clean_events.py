"""
Data cleaning layer + raw-vs-clean impact analysis.

This is the pandas MIRROR of the SQL staging layer in bigquery_setup.sql -
same four rules, same order, same thresholds. Run either; results match.

Cleaning rules (each maps to a real pipeline failure mode):
  R1  Drop events with NULL/empty user_id        (logging failures - unattributable)
  R2  Deduplicate on the BUSINESS KEY
      (user_id, event_type, event_ts, revenue)   (client retry double-fires have
                                                   DIFFERENT event_ids, so
                                                   DISTINCT event_id won't work)
  R3  Drop negative-revenue rows                 (refunds mis-logged as orders;
                                                   metric = gross order revenue,
                                                   refunds analysed separately)
  R4  Exclude flagged accounts                   (QA: 'qa_' prefix;
                                                   bots: >=100 sessions, 0 orders)

Why cleaning is part of EXPERIMENT methodology, not housekeeping:
the duplicate double-fires live in the NEW checkout client, so they
inflate TREATMENT revenue only. Uncleaned data reports a large,
highly significant revenue win that is mostly an instrumentation bug.

Usage:
  python clean_events.py     # reads users.csv + raw_events.csv
                             # writes experiment_data.csv (clean)
                             #        experiment_data_raw.csv (uncleaned)
"""

import numpy as np
import pandas as pd
from scipy import stats

from generate_ab_data import aggregate_user_metrics

BOT_SESSION_THRESHOLD = 100   # total sessions across both periods
DATA_USERS = "users.csv"
DATA_RAW = "raw_events.csv"


def clean(users: pd.DataFrame, raw: pd.DataFrame):
    """Apply R1-R4. Returns (population, events_clean, report_lines, dup_diag)."""
    report = [("raw events", len(raw), "")]

    # R1: unattributable rows ------------------------------------------------
    ev = raw[raw.user_id.notna() & (raw.user_id != "")]
    report.append(("R1 null user_id", len(ev) - len(raw), "logging failures"))

    # R2: dedup on business key (keep first-seen event_id) --------------------
    before = len(ev)
    ev = ev.sort_values("event_id", kind="stable")
    dup_mask = ev.duplicated(["user_id", "event_type", "event_ts", "revenue"], keep="first")
    dupes = ev[dup_mask]
    ev = ev[~dup_mask]
    report.append(("R2 duplicate events", len(ev) - before, "retry double-fires"))

    # diagnostic: duplicate rate by arm (the smoking gun)
    dup_by_arm = (
        dupes[dupes.event_type == "order"]
        .merge(users[["user_id", "variant"]], on="user_id")
        .groupby("variant").size()
    )
    ord_by_arm = (
        raw[(raw.event_type == "order") & (raw.revenue > 0)]
        .merge(users[["user_id", "variant"]], on="user_id")
        .groupby("variant").size()
    )
    dup_diag = (dup_by_arm / ord_by_arm * 100).round(2)

    # R3: negative revenue (refunds mis-logged as orders) ---------------------
    before = len(ev)
    ev = ev[ev.revenue.isna() | (ev.revenue >= 0)]
    report.append(("R3 negative revenue", len(ev) - before, "refunds excluded from gross"))

    # R4: flagged accounts (QA by prefix, bots by behaviour) ------------------
    act = ev.groupby("user_id").agg(
        sessions=("event_type", lambda s: (s == "session").sum()),
        orders=("event_type", lambda s: (s == "order").sum()),
    )
    bots = act[(act.sessions >= BOT_SESSION_THRESHOLD) & (act.orders == 0)].index
    qa = users.user_id[users.user_id.str.startswith("qa_")]
    excluded = set(bots) | set(qa)
    before = len(ev)
    ev = ev[~ev.user_id.isin(excluded)]
    report.append((
        "R4 bot/QA accounts",
        len(ev) - before,
        f"{len(bots)} bots (behaviour) + {len(qa)} QA (prefix)",
    ))
    report.append(("clean events", len(ev), ""))

    population = users[~users.user_id.isin(excluded)]
    return population, ev, report, dup_diag


def quick_readout(metrics: pd.DataFrame) -> dict:
    """Conversion z-test + revenue/user Welch t-test, compactly."""
    t = metrics[metrics.variant == "treatment"]
    c = metrics[metrics.variant == "control"]
    pt, pc = t.converted.mean(), c.converted.mean()
    pp = metrics.converted.mean()
    z = (pt - pc) / np.sqrt(pp * (1 - pp) * (1 / len(t) + 1 / len(c)))
    _, p_rev = stats.ttest_ind(t.exp_revenue, c.exp_revenue, equal_var=False)
    return {
        "users": len(metrics),
        "conv_lift": pt / pc - 1,
        "p_conv": 2 * (1 - stats.norm.cdf(abs(z))),
        "rev_lift": t.exp_revenue.mean() / c.exp_revenue.mean() - 1,
        "p_rev": p_rev,
    }


if __name__ == "__main__":
    users = pd.read_csv(DATA_USERS)
    raw = pd.read_csv(DATA_RAW, parse_dates=["event_ts"])

    population, ev_clean, report, dup_diag = clean(users, raw)

    print("CLEANING FUNNEL")
    for name, n, note in report:
        note = f"   ({note})" if note else ""
        print(f"  {name:<22} {n:>12,}{note}")
    print("\nDuplicate ORDER rate by arm (the smoking gun):")
    for arm, pct in dup_diag.items():
        print(f"  {arm:<10} {pct:>5.2f}%")

    raw_metrics = aggregate_user_metrics(users, raw)
    clean_metrics = aggregate_user_metrics(population, ev_clean)
    raw_metrics.to_csv("experiment_data_raw.csv", index=False)
    clean_metrics.to_csv("experiment_data.csv", index=False)

    r, c = quick_readout(raw_metrics), quick_readout(clean_metrics)
    print("\nIMPACT OF CLEANING ON THE EXPERIMENT READOUT")
    print(f"  {'':<14}{'users':>9}  {'conv lift':>9} {'p':>8}  {'rev lift':>9} {'p':>10}")
    print(f"  {'RAW':<14}{r['users']:>9,}  {r['conv_lift']:>+9.2%} {r['p_conv']:>8.4f}  "
          f"{r['rev_lift']:>+9.2%} {r['p_rev']:>10.2g}")
    print(f"  {'CLEAN':<14}{c['users']:>9,}  {c['conv_lift']:>+9.2%} {c['p_conv']:>8.4f}  "
          f"{c['rev_lift']:>+9.2%} {c['p_rev']:>10.2g}")
    print(f"\n  Raw data overstates the revenue lift {r['rev_lift'] / c['rev_lift']:.1f}x.")
    print("  Driver: duplicate order events concentrated in the treatment arm")
    print("  (instrumentation bug in the new checkout client).")
    print("\nWrote experiment_data.csv (clean) and experiment_data_raw.csv.")
    print("Next: python cuped_analysis.py  (must reproduce the certified numbers)")

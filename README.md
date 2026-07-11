# A/B Experimentation Platform

An end-to-end controlled experiment on a simulated e-commerce checkout redesign —
from raw event logs to a ship / no-ship decision. Built to demonstrate the full
analyst workflow: **data cleaning, EDA, experiment design, and hypothesis testing**,
with a partitioned BigQuery pipeline and an interactive Streamlit app.

> **Live demo:** https://ab-experiment-platform.streamlit.app/

## Headline result

The revenue effect is **inconclusive on naive analysis and conclusive after variance
reduction (CUPED)** — same point estimate, tighter confidence interval.

| Metric | Result | p-value |
|---|---|---|
| Conversion (primary) | +1.56% | 0.011 — significant |
| Revenue / user — naive | +1.98%, 95% CI [−0.7, +34.9] | 0.059 — **not** significant |
| Revenue / user — CUPED | +2.75%, 95% CI [+8.7, +38.6] | 0.002 — significant |
| Guardrail (AOV / order) | flat | 0.76 — no regression |

CUPED reduced revenue variance **29%** (ρ = 0.60 between pre- and experiment-period
revenue), equivalent to a **1.41× larger sample**. At this sample size the secondary
metric was underpowered; CUPED is what made it usable.

## What it demonstrates

- **Data cleaning as experiment validity.** Raw logs carry duplicate order
  double-fires, bot/QA accounts, refunds as negative revenue, and null user IDs.
  The duplicates are concentrated in the treatment arm (an instrumentation bug in
  the new checkout client), so uncleaned data overstates the revenue lift **3.6×**
  and reports it as wildly significant. Cleaning is methodology, not housekeeping.
- **EDA that feeds the method.** Randomisation balance + SRM check establish the
  arms are comparable; the heavy-tailed revenue distribution (CV ≈ 2.5) explains
  the poor naive MDE; the pre-vs-experiment relationship motivates CUPED.
- **Experiment design.** Pre-registered hypothesis, metric hierarchy
  (primary / secondary / guardrail), randomisation scheme, decision rule, and a
  power / MDE analysis sized from the real baselines.
- **Hypothesis testing.** SRM (chi-square), two-proportion z-test, Welch t-test,
  CUPED-adjusted test, and a guardrail test — with the metric-definition trap of
  AOV-per-converter vs AOV-per-order handled explicitly.
- **Warehouse engineering.** A BigQuery pipeline that ingests 2.4M raw events into
  a table **partitioned by event date and clustered by `(event_type, user_id)`**,
  with a raw → clean → analysis flow and a partition-pruning cost demo.

## Partition pruning (measured on BigQuery)

The same 7-day windowed query, run on an unpartitioned vs. a partitioned + clustered
copy of the 2.4M-row events table:

| Table | Bytes scanned |
|---|---|
| `events_flat` (unpartitioned) | 20.48 MB |
| `events_partitioned` (partitioned by day, clustered) | 2.78 MB |

**7.4× fewer bytes scanned** for an identical result. BigQuery bills by bytes
scanned, so at production scale (billions of rows) this is a direct cost and latency
win. At 2M rows partitioning isn't strictly necessary — the table is built the way it
would be at 2B rows, and the pruning benefit is measured rather than assumed.

<img width="436" height="260" alt="image" src="https://github.com/user-attachments/assets/d3c8920e-fd8d-4a99-8414-2ebdcbc674fb" />
<img width="397" height="263" alt="image" src="https://github.com/user-attachments/assets/1a482909-b83b-4c82-9c4a-d9b70c25c5a5" />

 screenshot 1 = events_flat (20.48 MB), screenshot 2 = events_partitioned (2.78 MB).
  
## Repo structure

```
app.py                      Streamlit app (Overview / Design / Data Quality / EDA / Results)
analysis_plan.md             pre-registered plan — committed before results
requirements.txt
.streamlit/config.toml       theme
src/
  generate_ab_data.py        synthetic event generation + defect injection
  clean_events.py            cleaning rules + raw-vs-clean impact
  experiment_design.py       power / MDE / sample sizing
  cuped_analysis.py          SRM, z-test, t-test, CUPED, guardrail
sql/
  bigquery_setup.sql         partitioned raw→clean→analysis pipeline + pruning demo
data/
  experiment_data.csv        user-level metrics, cleaned   (used by the app)
  experiment_data_raw.csv    user-level metrics, uncleaned (used by the app)
  raw_events.csv.gz          2.4M raw events               (pipeline reproduction)
  users.csv                  user dimension                (pipeline reproduction)
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads the two CSVs in `data/` and computes every number live — nothing is
hardcoded. The command-line analysis prints the same numbers:

```bash
python src/cuped_analysis.py        # the five statistical checks
python src/experiment_design.py     # sizing table + MDE analysis
```

## Reproduce the data pipeline

```bash
python src/generate_ab_data.py      # -> users.csv, raw_events.csv (dirty)
python src/clean_events.py          # -> experiment_data.csv (clean) + _raw.csv
```

The data is **synthetic with a known injected effect** (+2.5% order rate), so the
pipeline can be validated against ground truth, and defects are injected with a
separate random stream so cleaning provably recovers the certified result.

For the warehouse path, `sql/bigquery_setup.sql` loads `raw_events.csv.gz` into a
partitioned BigQuery table, runs the same four cleaning rules in SQL, and builds an
analysis table row-for-row identical to `experiment_data.csv`.

## Stack

Python (numpy · pandas · scipy) · BigQuery SQL · Streamlit · Plotly

## Design notes

- **Why synthetic data:** real experiment data is never public, and a known ground
  truth is what lets the pipeline be validated end to end.
- **Why the app reads a CSV, not live BigQuery:** keeps the deployed demo free,
  credential-free, and reproducible. The BigQuery work is shown in `sql/` plus the
  measured pruning result above; a live connection is a documented optional extension.
- **Honest scope:** at 2M rows partitioning isn't strictly necessary — the table is
  built the way it would be at 2B rows, then the pruning benefit is measured.

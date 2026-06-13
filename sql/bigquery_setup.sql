-- ===========================================================================
-- bigquery_setup.sql
-- A/B testing pipeline on BigQuery: RAW -> CLEAN staging -> analysis table
-- Project: product-analytics-portfolio   Dataset: experimentation
--
-- Pipeline:  events_raw (2.42M, dirty)
--              -> data-quality audit (section 3)
--              -> excluded_users (section 4)
--              -> events_clean (2.06M, section 5)
--              -> user_period_metrics (222K -> 220K users, section 6)
--              -> raw-vs-clean impact comparison (section 7)
--
-- Everything fits in the BigQuery free tier (10 GB storage, 1 TB query/month;
-- load jobs are free). Full pipeline uses well under 1 GB of query volume.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 0) One-time setup (Cloud Shell / local bq CLI)
-- ---------------------------------------------------------------------------
--   bq mk --location=US product-analytics-portfolio:experimentation
-- Upload users.csv and raw_events.csv.gz to Cloud Shell, create the tables in
-- section 1, then run the load commands in section 2 (bq reads .gz natively).

-- ---------------------------------------------------------------------------
-- 1) DDL
-- ---------------------------------------------------------------------------
-- Dim table (~222K rows incl. bot/QA accounts - we don't know which yet;
-- finding them is the cleaning layer's job). Small table -> deliberately NOT
-- partitioned: partitioning a 10 MB dim adds overhead for zero benefit.
CREATE OR REPLACE TABLE `product-analytics-portfolio.experimentation.users`
(
  user_id      STRING NOT NULL,
  variant      STRING NOT NULL,            -- 'control' / 'treatment'
  device       STRING,
  region       STRING,
  is_new_user  BOOL
);

-- RAW fact table. Note user_id is NULLABLE: the raw layer is PERMISSIVE by
-- design - it must accept whatever the pipeline ships (including logging
-- failures) so nothing is silently lost at ingest. Constraints are enforced
-- at the CLEAN layer instead. Saying this out loud is a senior-level answer.
CREATE OR REPLACE TABLE `product-analytics-portfolio.experimentation.events_raw`
(
  event_id    STRING NOT NULL,
  user_id     STRING,                      -- NULLABLE on purpose (raw layer)
  event_type  STRING NOT NULL,             -- 'order' / 'session'
  event_ts    TIMESTAMP NOT NULL,
  revenue     NUMERIC                      -- NULL for sessions; can be negative!
)
PARTITION BY DATE(event_ts)                -- prune by date BEFORE billing
CLUSTER BY event_type, user_id             -- low-cardinality filter col first
OPTIONS (require_partition_filter = TRUE); -- cost policy: no accidental scans

-- ---------------------------------------------------------------------------
-- 2) Load (free; rows are routed to daily partitions automatically)
-- ---------------------------------------------------------------------------
--   bq load --source_format=CSV --skip_leading_rows=1 \
--     product-analytics-portfolio:experimentation.users users.csv
--
--   bq load --source_format=CSV --skip_leading_rows=1 \
--     product-analytics-portfolio:experimentation.events_raw raw_events.csv.gz

-- ---------------------------------------------------------------------------
-- 3) DATA QUALITY AUDIT - quantify each defect before touching anything
-- ---------------------------------------------------------------------------
-- 3a. Defect inventory. Expected: ~8.7K null ids, ~3.7K negative revenue,
--     ~7.7K duplicate rows.
WITH win AS (
  SELECT * FROM `product-analytics-portfolio.experimentation.events_raw`
  WHERE DATE(event_ts) BETWEEN '2026-04-13' AND '2026-06-07'
),
keyed AS (
  SELECT user_id, event_type, event_ts, revenue, COUNT(*) AS n
  FROM win WHERE user_id IS NOT NULL
  GROUP BY 1, 2, 3, 4
)
SELECT
  (SELECT COUNT(*) FROM win)                              AS raw_rows,
  (SELECT COUNTIF(user_id IS NULL) FROM win)              AS null_user_id,
  (SELECT COUNTIF(revenue < 0) FROM win)                  AS negative_revenue,
  (SELECT SUM(n) - COUNT(*) FROM keyed)                   AS duplicate_rows;

-- 3b. THE SMOKING GUN: duplicate ORDER rate by experiment arm.
--     Duplicates are a client retry double-fire; the bug ships with the NEW
--     checkout flow, so they concentrate in treatment. Expected: ~0.5%
--     control vs ~5.2% treatment. A symmetric defect dilutes; an ASYMMETRIC
--     defect biases - this is why cleaning is experiment methodology.
WITH keyed AS (
  SELECT user_id, event_ts, revenue, COUNT(*) AS n
  FROM `product-analytics-portfolio.experimentation.events_raw`
  WHERE DATE(event_ts) BETWEEN '2026-04-13' AND '2026-06-07'
    AND event_type = 'order' AND revenue > 0 AND user_id IS NOT NULL
  GROUP BY 1, 2, 3
)
SELECT
  u.variant,
  SUM(k.n)              AS order_events,
  SUM(k.n) - COUNT(*)   AS duplicates,
  ROUND(SAFE_DIVIDE(SUM(k.n) - COUNT(*), SUM(k.n)) * 100, 2) AS dup_pct
FROM keyed k
JOIN `product-analytics-portfolio.experimentation.users` u USING (user_id)
GROUP BY u.variant
ORDER BY u.variant;

-- ---------------------------------------------------------------------------
-- 4) Flagged accounts: QA by id pattern, bots by BEHAVIOUR
-- ---------------------------------------------------------------------------
-- Bots have normal-looking ids, so the rule must be behavioural:
-- >=100 sessions across the study window with zero (positive) orders.
-- Real heavy users top out around ~50 sessions here, so false-positive
-- risk is negligible - state your threshold AND its justification.
CREATE OR REPLACE TABLE
  `product-analytics-portfolio.experimentation.excluded_users` AS
WITH activity AS (
  SELECT
    user_id,
    COUNTIF(event_type = 'session')                  AS sessions,
    COUNTIF(event_type = 'order' AND revenue > 0)    AS orders
  FROM `product-analytics-portfolio.experimentation.events_raw`
  WHERE DATE(event_ts) BETWEEN '2026-04-13' AND '2026-06-07'
    AND user_id IS NOT NULL
  GROUP BY user_id
)
SELECT
  user_id,
  CASE WHEN STARTS_WITH(user_id, 'qa_') THEN 'qa_account'
       ELSE 'bot_behaviour' END AS reason
FROM activity
WHERE STARTS_WITH(user_id, 'qa_')
   OR (sessions >= 100 AND orders = 0);
-- Expect ~2,000 rows: 800 qa_account + ~1,200 bot_behaviour.

-- ---------------------------------------------------------------------------
-- 5) CLEAN layer - four rules, same order as clean_events.py (parity!)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE
  `product-analytics-portfolio.experimentation.events_clean`
PARTITION BY DATE(event_ts)
CLUSTER BY event_type, user_id
AS
WITH attributed AS (                       -- R1: drop unattributable rows
  SELECT *
  FROM `product-analytics-portfolio.experimentation.events_raw`
  WHERE DATE(event_ts) BETWEEN '2026-04-13' AND '2026-06-07'
    AND user_id IS NOT NULL
),
deduped AS (                               -- R2: dedup on the BUSINESS KEY.
  -- Retry double-fires carry DIFFERENT event_ids, so DISTINCT event_id
  -- catches nothing. Key = (user, type, timestamp, amount); keep the
  -- first-seen event_id for determinism.
  SELECT *
  FROM attributed
  QUALIFY ROW_NUMBER() OVER (
            PARTITION BY user_id, event_type, event_ts, revenue
            ORDER BY event_id) = 1
)
SELECT d.*                                 -- R3: refunds out of gross revenue
FROM deduped d                             -- R4: drop flagged accounts
LEFT JOIN `product-analytics-portfolio.experimentation.excluded_users` x
       USING (user_id)
WHERE (d.revenue IS NULL OR d.revenue >= 0)
  AND x.user_id IS NULL;
-- Expect ~2,055,016 rows (from 2,418,031 raw).
-- R3 note: analysis metric is GROSS order revenue; refunds are excluded and
-- would be analysed separately. Netting them is an alternative defensible
-- choice - what matters is that the decision is explicit and documented.

-- ---------------------------------------------------------------------------
-- 6) Analysis table (mirrors aggregate_user_metrics() in Python exactly)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE
  `product-analytics-portfolio.experimentation.user_period_metrics` AS
WITH population AS (                       -- analysis population: real users
  SELECT u.*
  FROM `product-analytics-portfolio.experimentation.users` u
  LEFT JOIN `product-analytics-portfolio.experimentation.excluded_users` x
         USING (user_id)
  WHERE x.user_id IS NULL
),
agg AS (
  SELECT
    user_id,
    COUNTIF(event_type = 'session' AND DATE(event_ts) <= '2026-05-10') AS pre_sessions,
    COUNTIF(event_type = 'order'   AND DATE(event_ts) <= '2026-05-10') AS pre_orders,
    SUM(IF(event_type = 'order' AND DATE(event_ts) <= '2026-05-10', revenue, 0)) AS pre_revenue_raw,
    COUNTIF(event_type = 'session' AND DATE(event_ts) >= '2026-05-11') AS exp_sessions,
    COUNTIF(event_type = 'order'   AND DATE(event_ts) >= '2026-05-11') AS exp_orders,
    SUM(IF(event_type = 'order' AND DATE(event_ts) >= '2026-05-11', revenue, 0)) AS exp_revenue_raw
  FROM `product-analytics-portfolio.experimentation.events_clean`
  WHERE DATE(event_ts) BETWEEN '2026-04-13' AND '2026-06-07'
  GROUP BY user_id
)
SELECT
  p.user_id, p.variant, p.device, p.region, p.is_new_user,
  COALESCE(a.pre_sessions, 0) AS pre_sessions,
  COALESCE(a.pre_orders, 0)   AS pre_orders,
  -- New users have NO pre-period: NULL, not 0. A returning user who bought
  -- nothing is a true 0. CUPED mean-imputes NULLs (zero adjustment), so the
  -- distinction must survive this layer.
  IF(p.is_new_user, NULL, ROUND(COALESCE(a.pre_revenue_raw, 0), 2)) AS pre_revenue,
  COALESCE(a.exp_sessions, 0) AS exp_sessions,
  COALESCE(a.exp_orders, 0)   AS exp_orders,
  IF(COALESCE(a.exp_orders, 0) > 0, 1, 0) AS converted,
  ROUND(COALESCE(a.exp_revenue_raw, 0), 2) AS exp_revenue
FROM population p
LEFT JOIN agg a USING (user_id);           -- LEFT JOIN: zero-event users stay

-- Sanity vs the local pipeline (must match experiment_data.csv exactly):
-- control 110,100 / 32.62% | treatment 109,900 / 33.13%
SELECT variant, COUNT(*) AS users, ROUND(AVG(converted), 4) AS conv_rate,
       ROUND(AVG(exp_revenue), 1) AS rev_per_user
FROM `product-analytics-portfolio.experimentation.user_period_metrics`
GROUP BY variant ORDER BY variant;

-- ---------------------------------------------------------------------------
-- 7) IMPACT OF CLEANING on the experiment readout (the README table)
-- ---------------------------------------------------------------------------
-- Expected: raw layer shows ~+7.1% revenue/user lift (instrumentation bug);
-- clean layer shows ~+2.0% (the true effect, certified by CUPED downstream).
WITH raw_win AS (
  SELECT user_id, event_type, event_ts, revenue
  FROM `product-analytics-portfolio.experimentation.events_raw`
  WHERE DATE(event_ts) BETWEEN '2026-04-13' AND '2026-06-07'
),
raw_per_user AS (
  SELECT u.variant, u.user_id,
         COALESCE(SUM(IF(e.event_type = 'order'
                         AND DATE(e.event_ts) >= '2026-05-11', e.revenue, 0)), 0) AS rev
  FROM `product-analytics-portfolio.experimentation.users` u
  LEFT JOIN raw_win e USING (user_id)
  GROUP BY 1, 2
),
clean_per_user AS (
  SELECT variant, user_id, exp_revenue AS rev
  FROM `product-analytics-portfolio.experimentation.user_period_metrics`
)
SELECT 'raw'   AS layer, variant, COUNT(*) AS users, ROUND(AVG(rev), 1) AS rev_per_user
FROM raw_per_user GROUP BY 2
UNION ALL
SELECT 'clean' AS layer, variant, COUNT(*), ROUND(AVG(rev), 1)
FROM clean_per_user GROUP BY 2
ORDER BY layer DESC, variant;

-- ---------------------------------------------------------------------------
-- 8) PARTITION PRUNING DEMO (screenshot for the README)
-- ---------------------------------------------------------------------------
-- 8a. Inspect the partitions: 56 daily partitions, ~40-45K rows each.
SELECT partition_id, total_rows, ROUND(total_logical_bytes / 1e6, 1) AS mb
FROM `product-analytics-portfolio.experimentation.INFORMATION_SCHEMA.PARTITIONS`
WHERE table_name = 'events_raw'
ORDER BY partition_id;

-- 8b. Pruned: final week -> scans 7 of 56 partitions (~1/8 of the bytes).
--     Note the editor's upfront estimate, then Job info -> bytes processed.
SELECT DATE(event_ts) AS d, COUNT(*) AS events, ROUND(SUM(revenue), 0) AS revenue
FROM `product-analytics-portfolio.experimentation.events_raw`
WHERE DATE(event_ts) BETWEEN '2026-06-01' AND '2026-06-07'
GROUP BY d ORDER BY d;

-- 8c. Control group: an UNPARTITIONED copy, then the IDENTICAL query.
--     Same result, ~8x the bytes billed. That pair of numbers is the demo.
CREATE OR REPLACE TABLE
  `product-analytics-portfolio.experimentation.events_raw_unpartitioned` AS
SELECT * FROM `product-analytics-portfolio.experimentation.events_raw`
WHERE DATE(event_ts) BETWEEN '2026-04-13' AND '2026-06-07';

SELECT DATE(event_ts) AS d, COUNT(*) AS events, ROUND(SUM(revenue), 0) AS revenue
FROM `product-analytics-portfolio.experimentation.events_raw_unpartitioned`
WHERE DATE(event_ts) BETWEEN '2026-06-01' AND '2026-06-07'
GROUP BY d ORDER BY d;

-- 8d. Clustering bonus: adding event_type = 'order' on the clustered table
--     reads even fewer blocks. The editor's estimate does NOT account for
--     clustering - only post-run "bytes processed" shows it. Knowing that
--     quirk is itself a senior-level detail.
SELECT DATE(event_ts) AS d, COUNT(*) AS orders, ROUND(SUM(revenue), 0) AS revenue
FROM `product-analytics-portfolio.experimentation.events_raw`
WHERE DATE(event_ts) BETWEEN '2026-06-01' AND '2026-06-07'
  AND event_type = 'order'
GROUP BY d ORDER BY d;

-- ---------------------------------------------------------------------------
-- 9) Hand-off to Python / Streamlit
-- ---------------------------------------------------------------------------
-- Export user_period_metrics as CSV (console Export, or bq extract). It is
-- row-for-row identical to the locally built experiment_data.csv, so
-- cuped_analysis.py and the Streamlit app run on either source unchanged.

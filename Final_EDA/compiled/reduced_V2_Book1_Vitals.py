# V2_Book1_Vitals
# Functional cells: 24 of 52 code cells (100 total)
# Source: V2_Book1_Vitals.ipynb
# =============================================================================

# ========================================
# CELL 1
# ========================================

# ---------------------------------
# Imports and Variable Declarations
# ---------------------------------

import datetime
from dateutil.relativedelta import relativedelta
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize a Spark session for distributed data processing
spark = SparkSession.builder.getOrCreate()

# Ensure date/time comparisons use Central Time
spark.conf.set("spark.sql.session.timeZone", "America/Chicago")

# We never hard-code "dev", "test" or "prod", so the line below sets the trgt_cat
# catalog for any tables you write out or when reading your own intermediate tables.

# Define target catalog for SQL based on the environment variable
trgt_cat = os.environ.get('trgt_cat')

# Use the general “prod” catalog so you don’t need to prefix every IDP table
spark.sql('USE CATALOG prod;')

# ========================================
# CELL 2
# ========================================

# CELL 1 - NORMALIZE AND CLEAN RAW VITALS
# ========================================

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_vitals_raw AS

WITH
  -- Pull raw vitals from pat_enc_enh
  raw AS (
    SELECT
      pe.PAT_ID,
      CAST(pe.CONTACT_DATE AS TIMESTAMP) AS MEAS_TS,
      pe.WEIGHT       AS WEIGHT_RAW,
      pe.BP_SYSTOLIC  AS SBP_RAW,
      pe.BP_DIASTOLIC AS DBP_RAW,
      pe.PULSE        AS PULSE_RAW,
      pe.BMI          AS BMI_RAW,
      pe.TEMPERATURE  AS TEMP_RAW,
      pe.RESPIRATIONS AS RESP_RAW,
      CAST(pe.CONTACT_DATE AS DATE) AS MEAS_DATE
    FROM clarity_cur.pat_enc_enh pe
    WHERE pe.CONTACT_DATE >= DATE '2021-07-01'  -- Clarity data availability starts here
  ),

  -- Normalize & parse units
  norm AS (
    SELECT
      PAT_ID,
      MEAS_TS,
      MEAS_DATE,
      CAST(WEIGHT_RAW AS DOUBLE)/16.0 AS WEIGHT_LB,
      CAST(WEIGHT_RAW AS DOUBLE)       AS WEIGHT_OZ,
      CAST(SBP_RAW   AS DOUBLE)       AS BP_SYSTOLIC,
      CAST(DBP_RAW   AS DOUBLE)       AS BP_DIASTOLIC,
      CAST(PULSE_RAW AS DOUBLE)       AS PULSE,
      CAST(BMI_RAW   AS DOUBLE)       AS BMI,
      CAST(TEMP_RAW  AS DOUBLE)       AS TEMPERATURE,
      CAST(RESP_RAW  AS DOUBLE)       AS RESP_RATE
    FROM raw
  )

-- Apply plausibility filters
SELECT
  PAT_ID, 
  MEAS_TS, 
  MEAS_DATE,
  CASE WHEN WEIGHT_LB    BETWEEN 50 AND 800 THEN WEIGHT_LB    END AS WEIGHT_LB,
  CASE WHEN BP_SYSTOLIC  BETWEEN 60 AND 280 THEN BP_SYSTOLIC  END AS BP_SYSTOLIC,
  CASE WHEN BP_DIASTOLIC BETWEEN 30 AND 180 THEN BP_DIASTOLIC END AS BP_DIASTOLIC,
  CASE WHEN PULSE        BETWEEN 20 AND 250 THEN PULSE        END AS PULSE,
  CASE WHEN BMI          BETWEEN 10 AND 100 THEN BMI          END AS BMI,
  CASE WHEN WEIGHT_LB    BETWEEN 50 AND 800 THEN WEIGHT_OZ    END AS WEIGHT_OZ,
  CASE WHEN TEMPERATURE  BETWEEN 95 AND 105 THEN TEMPERATURE  END AS TEMPERATURE,
  CASE WHEN RESP_RATE    BETWEEN 8 AND 40   THEN RESP_RATE    END AS RESP_RATE
FROM norm
""")

print("✓ Raw vitals normalized and cleaned")

# ========================================
# CELL 3
# ========================================

# CELL 2 - CALCULATE WEIGHT AND BP PATTERNS
# ==========================================

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_vitals_patterns AS

WITH
  cohort AS (
    SELECT DISTINCT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
  ),

  -- Weight history with lag calculations
  -- Get all weights from past 12 months to calculate trends and detect rapid changes
  weight_history AS (
    SELECT
      c.PAT_ID,
      c.END_DTTM,
      v.WEIGHT_OZ,
      v.MEAS_DATE,
      DATEDIFF(c.END_DTTM, v.MEAS_DATE) AS DAYS_BEFORE_END,
      LAG(v.WEIGHT_OZ) OVER (PARTITION BY c.PAT_ID, c.END_DTTM ORDER BY v.MEAS_DATE) AS PREV_WEIGHT_OZ
    FROM cohort c
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_vitals_raw v
      ON v.PAT_ID = c.PAT_ID
      AND v.MEAS_DATE >= DATE_SUB(c.END_DTTM, 365)
      AND v.MEAS_DATE < c.END_DTTM
      AND v.WEIGHT_OZ IS NOT NULL
  ),

  -- Calculate rapid weight loss
  -- For each measurement within 60 days of END_DTTM, calculate % loss vs. previous measurement
  -- This captures acute drops between consecutive clinic visits that might indicate cancer cachexia
  weight_changes AS (
    SELECT
      PAT_ID,
      END_DTTM,
      WEIGHT_OZ,
      MEAS_DATE,
      DAYS_BEFORE_END,
      PREV_WEIGHT_OZ,
      CASE 
        WHEN DAYS_BEFORE_END <= 60 AND PREV_WEIGHT_OZ IS NOT NULL AND PREV_WEIGHT_OZ > 0
        THEN ((PREV_WEIGHT_OZ - WEIGHT_OZ) / PREV_WEIGHT_OZ) * 100
      END AS WEIGHT_LOSS_PCT
    FROM weight_history
  ),

  -- Weight patterns aggregation
  weight_patterns AS (
    SELECT
      PAT_ID,
      END_DTTM,
      COUNT(*) AS WEIGHT_MEASUREMENT_COUNT_12M,
      STDDEV(WEIGHT_OZ) AS WEIGHT_VOLATILITY_12M,
      REGR_SLOPE(WEIGHT_OZ, DAYS_BEFORE_END) AS WEIGHT_TRAJECTORY_SLOPE,
      REGR_R2(WEIGHT_OZ, DAYS_BEFORE_END) AS WEIGHT_TRAJECTORY_R2,
      MIN(WEIGHT_OZ) AS MIN_WEIGHT_12M,
      MAX(WEIGHT_OZ) AS MAX_WEIGHT_12M,
      MAX(WEIGHT_LOSS_PCT) AS MAX_WEIGHT_LOSS_PCT_60D  -- Maximum loss between any consecutive measurements in last 60 days
    FROM weight_changes
    GROUP BY PAT_ID, END_DTTM
  ),

  -- BP history
  bp_history AS (
    SELECT
      c.PAT_ID,
      c.END_DTTM,
      v.BP_SYSTOLIC,
      v.BP_DIASTOLIC,
      v.BP_SYSTOLIC - v.BP_DIASTOLIC AS PULSE_PRESSURE,
      v.MEAS_DATE
    FROM cohort c
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_vitals_raw v
      ON v.PAT_ID = c.PAT_ID
      AND v.MEAS_DATE >= DATE_SUB(c.END_DTTM, 180)
      AND v.MEAS_DATE < c.END_DTTM
      AND v.BP_SYSTOLIC IS NOT NULL
      AND v.BP_DIASTOLIC IS NOT NULL
  ),

  -- BP variability
  bp_variability AS (
    SELECT
      PAT_ID,
      END_DTTM,
      COUNT(*) AS BP_MEASUREMENT_COUNT_6M,
      STDDEV(BP_SYSTOLIC) AS SBP_VARIABILITY_6M,
      STDDEV(BP_DIASTOLIC) AS DBP_VARIABILITY_6M,
      STDDEV(PULSE_PRESSURE) AS PULSE_PRESSURE_VARIABILITY_6M,
      AVG(PULSE_PRESSURE) AS AVG_PULSE_PRESSURE_6M
    FROM bp_history
    GROUP BY PAT_ID, END_DTTM
  )

-- Join patterns for each patient-month
SELECT 
  c.PAT_ID,
  c.END_DTTM,
  -- Weight pattern features
  wp.WEIGHT_MEASUREMENT_COUNT_12M,
  wp.WEIGHT_VOLATILITY_12M,
  wp.WEIGHT_TRAJECTORY_SLOPE,
  wp.WEIGHT_TRAJECTORY_R2,
  wp.MIN_WEIGHT_12M,
  wp.MAX_WEIGHT_12M,
  wp.MAX_WEIGHT_LOSS_PCT_60D,
  -- BP pattern features
  bpv.BP_MEASUREMENT_COUNT_6M,
  bpv.SBP_VARIABILITY_6M,
  bpv.DBP_VARIABILITY_6M,
  bpv.PULSE_PRESSURE_VARIABILITY_6M,
  bpv.AVG_PULSE_PRESSURE_6M
FROM cohort c
LEFT JOIN weight_patterns wp 
  ON c.PAT_ID = wp.PAT_ID AND c.END_DTTM = wp.END_DTTM
LEFT JOIN bp_variability bpv
  ON c.PAT_ID = bpv.PAT_ID AND c.END_DTTM = bpv.END_DTTM
""")

print("✓ Weight and BP patterns calculated")

# ========================================
# CELL 4
# ========================================

# =========================================================================
# CELL 3 - GET LATEST VITAL VALUES
# =========================================================================
# Purpose: Extract the most recent vital signs for each patient at each snapshot
# Creates a wide table with one row per patient-month containing:
# - Latest vital measurements before the snapshot date
# - Historical values at 6 and 12 months prior (for trend calculation)
# - Measurement dates (for recency features)

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_vitals_latest AS

WITH
  -- Base cohort: all patient-month combinations we need features for
  cohort AS (
    SELECT DISTINCT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
  ),
  
  -- Alias for our cleaned vitals table (improves readability)
  v AS (
    SELECT * FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals_raw
  ),

  -- ================================================================
  -- LATEST WEIGHT
  -- ================================================================
  -- Weight is critical for CRC (weight loss is a key symptom)
  -- We track the exact date for recency calculations
  weight_latest AS (
    SELECT PAT_ID, END_DTTM, WEIGHT_OZ, WEIGHT_LB, MEAS_DATE AS WEIGHT_DATE
    FROM (
      SELECT
        c.PAT_ID, 
        c.END_DTTM, 
        v.WEIGHT_OZ,    -- Keep ounces for precision
        v.WEIGHT_LB,    -- Keep pounds for readability
        v.MEAS_DATE,
        ROW_NUMBER() OVER (
          PARTITION BY c.PAT_ID, c.END_DTTM 
          ORDER BY v.MEAS_DATE DESC
        ) AS rn
      FROM cohort c
      JOIN v ON v.PAT_ID = c.PAT_ID 
        AND v.MEAS_DATE < c.END_DTTM 
        AND v.WEIGHT_OZ IS NOT NULL
    ) t WHERE rn = 1
  ),

  -- ================================================================
  -- WEIGHT 6 MONTHS AGO
  -- ================================================================
  -- Find weight closest to 6 months before snapshot
  -- Used to calculate 6-month weight change (important for cachexia detection)
  weight_6m_ago AS (
    SELECT PAT_ID, END_DTTM, WEIGHT_OZ AS WEIGHT_OZ_6M
    FROM (
      SELECT
        c.PAT_ID, 
        c.END_DTTM, 
        v.WEIGHT_OZ,
        -- Find measurement closest to exactly 180 days ago
        ROW_NUMBER() OVER (
          PARTITION BY c.PAT_ID, c.END_DTTM 
          -- Order by distance from target date (180 days ago)
          ORDER BY ABS(DATEDIFF(v.MEAS_DATE, DATE_SUB(c.END_DTTM, 180)))
        ) AS rn
      FROM cohort c
      JOIN v ON v.PAT_ID = c.PAT_ID 
        -- Look in window: 150-210 days before snapshot (±30 day tolerance)
        AND v.MEAS_DATE < DATE_SUB(c.END_DTTM, 150)   -- At least 5 months ago
        AND v.MEAS_DATE >= DATE_SUB(c.END_DTTM, 210)  -- At most 7 months ago
        AND v.WEIGHT_OZ IS NOT NULL
    ) t WHERE rn = 1
  ),

  -- ================================================================
  -- WEIGHT 12 MONTHS AGO
  -- ================================================================
  -- Find weight closest to 12 months before snapshot
  -- Used for annual weight change calculation
  weight_12m_ago AS (
    SELECT PAT_ID, END_DTTM, WEIGHT_OZ AS WEIGHT_OZ_12M
    FROM (
      SELECT
        c.PAT_ID, 
        c.END_DTTM, 
        v.WEIGHT_OZ,
        ROW_NUMBER() OVER (
          PARTITION BY c.PAT_ID, c.END_DTTM 
          -- Find closest to exactly 365 days ago
          ORDER BY ABS(DATEDIFF(v.MEAS_DATE, DATE_SUB(c.END_DTTM, 365)))
        ) AS rn
      FROM cohort c
      JOIN v ON v.PAT_ID = c.PAT_ID 
        -- Look in window: 335-395 days before (±30 day tolerance)
        AND v.MEAS_DATE < DATE_SUB(c.END_DTTM, 335)   -- At least 11 months ago
        AND v.MEAS_DATE >= DATE_SUB(c.END_DTTM, 395)  -- At most 13 months ago
        AND v.WEIGHT_OZ IS NOT NULL
    ) t WHERE rn = 1
  ),

  -- ================================================================
  -- BMI (LATEST AND HISTORICAL)
  -- ================================================================
  -- BMI is often calculated at visits, so we track it separately from weight
  bmi_latest AS (
    SELECT PAT_ID, END_DTTM, BMI, MEAS_DATE AS BMI_DATE
    FROM (
      SELECT
        c.PAT_ID, c.END_DTTM, v.BMI, v.MEAS_DATE,
        ROW_NUMBER() OVER (
          PARTITION BY c.PAT_ID, c.END_DTTM 
          ORDER BY v.MEAS_DATE DESC
        ) AS rn
      FROM cohort c
      JOIN v ON v.PAT_ID = c.PAT_ID 
        AND v.MEAS_DATE < c.END_DTTM 
        AND v.BMI IS NOT NULL
    ) t WHERE rn = 1
  ),

  -- BMI 6 months ago (for trend analysis)
  bmi_6m_ago AS (
    SELECT PAT_ID, END_DTTM, BMI AS BMI_6M
    FROM (
      SELECT
        c.PAT_ID, c.END_DTTM, v.BMI,
        ROW_NUMBER() OVER (
          PARTITION BY c.PAT_ID, c.END_DTTM 
          ORDER BY ABS(DATEDIFF(v.MEAS_DATE, DATE_SUB(c.END_DTTM, 180)))
        ) AS rn
      FROM cohort c
      JOIN v ON v.PAT_ID = c.PAT_ID 
        AND v.MEAS_DATE < DATE_SUB(c.END_DTTM, 150)
        AND v.MEAS_DATE >= DATE_SUB(c.END_DTTM, 210)
        AND v.BMI IS NOT NULL
    ) t WHERE rn = 1
  ),

  -- BMI 12 months ago
  bmi_12m_ago AS (
    SELECT PAT_ID, END_DTTM, BMI AS BMI_12M
    FROM (
      SELECT
        c.PAT_ID, c.END_DTTM, v.BMI,
        ROW_NUMBER() OVER (
          PARTITION BY c.PAT_ID, c.END_DTTM 
          ORDER BY ABS(DATEDIFF(v.MEAS_DATE, DATE_SUB(c.END_DTTM, 365)))
        ) AS rn
      FROM cohort c
      JOIN v ON v.PAT_ID = c.PAT_ID 
        AND v.MEAS_DATE < DATE_SUB(c.END_DTTM, 335)
        AND v.MEAS_DATE >= DATE_SUB(c.END_DTTM, 395)
        AND v.BMI IS NOT NULL
    ) t WHERE rn = 1
  ),

  -- ================================================================
  -- BLOOD PRESSURE (LATEST ONLY)
  -- ================================================================
  -- Both systolic and diastolic must be present for valid BP reading
  bp_latest AS (
    SELECT PAT_ID, END_DTTM, BP_SYSTOLIC, BP_DIASTOLIC, MEAS_DATE AS BP_DATE
    FROM (
      SELECT
        c.PAT_ID, c.END_DTTM, v.BP_SYSTOLIC, v.BP_DIASTOLIC, v.MEAS_DATE,
        ROW_NUMBER() OVER (
          PARTITION BY c.PAT_ID, c.END_DTTM 
          ORDER BY v.MEAS_DATE DESC
        ) AS rn
      FROM cohort c
      JOIN v ON v.PAT_ID = c.PAT_ID 
        AND v.MEAS_DATE < c.END_DTTM 
        AND v.BP_SYSTOLIC IS NOT NULL    -- Both components required
        AND v.BP_DIASTOLIC IS NOT NULL
    ) t WHERE rn = 1
  ),

  -- ================================================================
  -- OTHER VITAL SIGNS (LATEST ONLY)
  -- ================================================================
  -- Pulse, temperature, and respiratory rate are less critical for CRC
  -- but useful for general health assessment
  
  pulse_latest AS (
    SELECT PAT_ID, END_DTTM, PULSE, MEAS_DATE AS PULSE_DATE
    FROM (
      SELECT
        c.PAT_ID, c.END_DTTM, v.PULSE, v.MEAS_DATE,
        ROW_NUMBER() OVER (
          PARTITION BY c.PAT_ID, c.END_DTTM 
          ORDER BY v.MEAS_DATE DESC
        ) AS rn
      FROM cohort c
      JOIN v ON v.PAT_ID = c.PAT_ID 
        AND v.MEAS_DATE < c.END_DTTM 
        AND v.PULSE IS NOT NULL
    ) t WHERE rn = 1
  ),

  temperature_latest AS (
    SELECT PAT_ID, END_DTTM, TEMPERATURE, MEAS_DATE AS TEMP_DATE
    FROM (
      SELECT
        c.PAT_ID, c.END_DTTM, v.TEMPERATURE, v.MEAS_DATE,
        ROW_NUMBER() OVER (
          PARTITION BY c.PAT_ID, c.END_DTTM 
          ORDER BY v.MEAS_DATE DESC
        ) AS rn
      FROM cohort c
      JOIN v ON v.PAT_ID = c.PAT_ID 
        AND v.MEAS_DATE < c.END_DTTM 
        AND v.TEMPERATURE IS NOT NULL
    ) t WHERE rn = 1
  ),

  resp_rate_latest AS (
    SELECT PAT_ID, END_DTTM, RESP_RATE, MEAS_DATE AS RESP_DATE
    FROM (
      SELECT
        c.PAT_ID, c.END_DTTM, v.RESP_RATE, v.MEAS_DATE,
        ROW_NUMBER() OVER (
          PARTITION BY c.PAT_ID, c.END_DTTM 
          ORDER BY v.MEAS_DATE DESC
        ) AS rn
      FROM cohort c
      JOIN v ON v.PAT_ID = c.PAT_ID 
        AND v.MEAS_DATE < c.END_DTTM 
        AND v.RESP_RATE IS NOT NULL
    ) t WHERE rn = 1
  )

-- ================================================================
-- FINAL ASSEMBLY
-- ================================================================
-- Combine all CTEs using LEFT JOINs to preserve all patient-months
-- even if they have no vital measurements
SELECT 
  c.PAT_ID,
  c.END_DTTM,
  
  -- Weight measurements (critical for CRC detection)
  w.WEIGHT_OZ,
  w.WEIGHT_LB,
  w.WEIGHT_DATE,
  w6.WEIGHT_OZ_6M,    -- For 6-month change
  w12.WEIGHT_OZ_12M,  -- For 12-month change
  
  -- BMI measurements
  b.BMI,
  b.BMI_DATE,
  b6.BMI_6M,          -- For 6-month change
  b12.BMI_12M,        -- For 12-month change
  
  -- Blood pressure
  bp.BP_SYSTOLIC,
  bp.BP_DIASTOLIC,
  bp.BP_DATE,
  
  -- Other vitals
  p.PULSE,
  p.PULSE_DATE,
  t.TEMPERATURE,
  t.TEMP_DATE,
  rr.RESP_RATE,
  rr.RESP_DATE
  
FROM cohort c
-- LEFT JOINs ensure we keep all patient-months even with missing vitals
LEFT JOIN weight_latest w ON c.PAT_ID = w.PAT_ID AND c.END_DTTM = w.END_DTTM
LEFT JOIN weight_6m_ago w6 ON c.PAT_ID = w6.PAT_ID AND c.END_DTTM = w6.END_DTTM
LEFT JOIN weight_12m_ago w12 ON c.PAT_ID = w12.PAT_ID AND c.END_DTTM = w12.END_DTTM
LEFT JOIN bmi_latest b ON c.PAT_ID = b.PAT_ID AND c.END_DTTM = b.END_DTTM
LEFT JOIN bmi_6m_ago b6 ON c.PAT_ID = b6.PAT_ID AND c.END_DTTM = b6.END_DTTM
LEFT JOIN bmi_12m_ago b12 ON c.PAT_ID = b12.PAT_ID AND c.END_DTTM = b12.END_DTTM
LEFT JOIN bp_latest bp ON c.PAT_ID = bp.PAT_ID AND c.END_DTTM = bp.END_DTTM
LEFT JOIN pulse_latest p ON c.PAT_ID = p.PAT_ID AND c.END_DTTM = p.END_DTTM
LEFT JOIN temperature_latest t ON c.PAT_ID = t.PAT_ID AND c.END_DTTM = t.END_DTTM
LEFT JOIN resp_rate_latest rr ON c.PAT_ID = rr.PAT_ID AND c.END_DTTM = rr.END_DTTM
""")

print("✓ Latest vital values extracted")

# ========================================
# CELL 5
# ========================================

# =========================================================================
# CELL 4 - FINAL ASSEMBLY WITH CALCULATED FEATURES
# =========================================================================
# Purpose: Create the final vitals feature table with:
# - Raw vital values
# - Recency features (days since last measurement)
# - Change features (weight/BMI trajectories)
# - Clinical flags (hypertension, obesity, cachexia risk)
# - Pattern features from Cell 2 (volatility, trajectory)

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_vitals AS

SELECT
  l.PAT_ID,
  l.END_DTTM,
  
  -- ================================================================
  -- RAW VITAL VALUES
  -- ================================================================
  -- Keep the actual measurements for downstream use
  l.WEIGHT_OZ,
  l.WEIGHT_LB,
  l.BP_SYSTOLIC,
  l.BP_DIASTOLIC,
  l.PULSE,
  l.BMI,
  l.TEMPERATURE,
  l.RESP_RATE,
  
  -- ================================================================
  -- RECENCY FEATURES
  -- ================================================================
  -- How many days since last measurement?
  -- Stale measurements may be less predictive
  DATEDIFF(l.END_DTTM, l.WEIGHT_DATE) AS DAYS_SINCE_WEIGHT,
  DATEDIFF(l.END_DTTM, l.BP_DATE) AS DAYS_SINCE_SBP,      -- Same date for both
  DATEDIFF(l.END_DTTM, l.BP_DATE) AS DAYS_SINCE_DBP,      -- BP components
  DATEDIFF(l.END_DTTM, l.PULSE_DATE) AS DAYS_SINCE_PULSE,
  DATEDIFF(l.END_DTTM, l.BMI_DATE) AS DAYS_SINCE_BMI,
  DATEDIFF(l.END_DTTM, l.TEMP_DATE) AS DAYS_SINCE_TEMPERATURE,
  DATEDIFF(l.END_DTTM, l.RESP_DATE) AS DAYS_SINCE_RESP_RATE,
  
  -- ================================================================
  -- WEIGHT CHANGE FEATURES
  -- ================================================================
  -- Unintentional weight loss is a key CRC symptom
  
  -- 6-month weight change percentage
  CASE 
    WHEN l.WEIGHT_OZ IS NOT NULL AND l.WEIGHT_OZ_6M IS NOT NULL 
    THEN ROUND(
      ((l.WEIGHT_OZ - l.WEIGHT_OZ_6M) / NULLIF(l.WEIGHT_OZ_6M, 0)) * 100, 
      2  -- Round to 2 decimal places
    )
  END AS WEIGHT_CHANGE_PCT_6M,
  
  -- 12-month weight change percentage  
  CASE 
    WHEN l.WEIGHT_OZ IS NOT NULL AND l.WEIGHT_OZ_12M IS NOT NULL 
    THEN ROUND(
      ((l.WEIGHT_OZ - l.WEIGHT_OZ_12M) / NULLIF(l.WEIGHT_OZ_12M, 0)) * 100, 
      2
    )
  END AS WEIGHT_CHANGE_PCT_12M,
  
-- ================================================================
  -- WEIGHT PATTERN FEATURES (from Cell 5 analysis)
  -- ================================================================
  -- These capture weight trajectory and volatility
  p.WEIGHT_MEASUREMENT_COUNT_12M,                        -- Engagement indicator
  ROUND(p.WEIGHT_VOLATILITY_12M, 2) AS WEIGHT_VOLATILITY_12M,  -- Stability
  ROUND(p.WEIGHT_TRAJECTORY_SLOPE, 4) AS WEIGHT_TRAJECTORY_SLOPE, -- Trend direction
  ROUND(p.WEIGHT_TRAJECTORY_R2, 4) AS WEIGHT_TRAJECTORY_R2,      -- Trend consistency
  ROUND(p.MAX_WEIGHT_LOSS_PCT_60D, 2) AS MAX_WEIGHT_LOSS_PCT_60D, -- Rapid loss: max loss between consecutive measurements in last 60 days
  
  -- ================================================================
  -- BMI TRAJECTORY FEATURES
  -- ================================================================
  -- BMI changes can indicate cachexia or recovery
  
  -- Absolute BMI change (not percentage)
  CASE 
    WHEN l.BMI IS NOT NULL AND l.BMI_6M IS NOT NULL 
    THEN ROUND(l.BMI - l.BMI_6M, 2)
  END AS BMI_CHANGE_6M,
  
  CASE 
    WHEN l.BMI IS NOT NULL AND l.BMI_12M IS NOT NULL 
    THEN ROUND(l.BMI - l.BMI_12M, 2)
  END AS BMI_CHANGE_12M,
  
  -- BMI category transitions (important for risk stratification)
  CASE 
    WHEN l.BMI_12M >= 30 AND l.BMI < 30 THEN 1 ELSE 0
  END AS BMI_LOST_OBESE_STATUS,         -- Was obese, now not
  
  CASE 
    WHEN l.BMI_12M >= 25 AND l.BMI < 25 THEN 1 ELSE 0
  END AS BMI_LOST_OVERWEIGHT_STATUS,    -- Was overweight, now normal
  
  -- ================================================================
  -- BP VARIABILITY FEATURES (from Cell 2)
  -- ================================================================
  -- High BP variability associated with cardiovascular risk
  p.BP_MEASUREMENT_COUNT_6M,
  ROUND(p.SBP_VARIABILITY_6M, 2) AS SBP_VARIABILITY_6M,
  ROUND(p.DBP_VARIABILITY_6M, 2) AS DBP_VARIABILITY_6M,
  ROUND(p.PULSE_PRESSURE_VARIABILITY_6M, 2) AS PULSE_PRESSURE_VARIABILITY_6M,
  ROUND(p.AVG_PULSE_PRESSURE_6M, 2) AS AVG_PULSE_PRESSURE_6M,
  
  -- ================================================================
  -- CALCULATED CARDIOVASCULAR FEATURES
  -- ================================================================
  
  -- Pulse pressure: difference between systolic and diastolic
  -- Wide pulse pressure (>60) can indicate arterial stiffness
  CASE 
    WHEN l.BP_SYSTOLIC IS NOT NULL AND l.BP_DIASTOLIC IS NOT NULL 
    THEN l.BP_SYSTOLIC - l.BP_DIASTOLIC
  END AS PULSE_PRESSURE,
  
  -- Mean arterial pressure: average pressure during cardiac cycle
  -- MAP = DBP + 1/3(SBP - DBP) = (2*DBP + SBP)/3
  CASE 
    WHEN l.BP_SYSTOLIC IS NOT NULL AND l.BP_DIASTOLIC IS NOT NULL 
    THEN ROUND((2 * l.BP_DIASTOLIC + l.BP_SYSTOLIC) / 3.0, 1)
  END AS MEAN_ARTERIAL_PRESSURE,
  
  -- ================================================================
  -- CLINICAL FLAGS FOR CRC RISK
  -- ================================================================
  
  -- Significant weight loss flags (5% and 10% thresholds)
  -- 5% unintentional weight loss in 6 months is clinically significant
  CASE 
    WHEN l.WEIGHT_OZ IS NOT NULL AND l.WEIGHT_OZ_6M IS NOT NULL 
         AND ((l.WEIGHT_OZ - l.WEIGHT_OZ_6M) / NULLIF(l.WEIGHT_OZ_6M, 0)) * 100 <= -5 
    THEN 1 ELSE 0 
  END AS WEIGHT_LOSS_5PCT_6M,
  
  -- 10% weight loss is severe and warrants immediate investigation
  CASE 
    WHEN l.WEIGHT_OZ IS NOT NULL AND l.WEIGHT_OZ_6M IS NOT NULL 
         AND ((l.WEIGHT_OZ - l.WEIGHT_OZ_6M) / NULLIF(l.WEIGHT_OZ_6M, 0)) * 100 <= -10 
    THEN 1 ELSE 0 
  END AS WEIGHT_LOSS_10PCT_6M,
  
  -- Rapid weight loss: >5% in 60 days
  CASE 
    WHEN p.MAX_WEIGHT_LOSS_PCT_60D >= 5 THEN 1 ELSE 0
  END AS RAPID_WEIGHT_LOSS_FLAG,
  
  -- Hypertension flags (JNC 8 criteria)
  CASE 
    WHEN l.BP_SYSTOLIC >= 140 OR l.BP_DIASTOLIC >= 90 THEN 1 ELSE 0 
  END AS HYPERTENSION_FLAG,
  
  -- Stage 2 hypertension
  CASE 
    WHEN l.BP_SYSTOLIC >= 160 OR l.BP_DIASTOLIC >= 100 THEN 1 ELSE 0 
  END AS SEVERE_HYPERTENSION_FLAG,
  
  -- Tachycardia: resting heart rate >100 bpm
  CASE 
    WHEN l.PULSE > 100 THEN 1 ELSE 0 
  END AS TACHYCARDIA_FLAG,
  
  -- BMI categories
  CASE 
    WHEN l.BMI < 18.5 THEN 1 ELSE 0 
  END AS UNDERWEIGHT_FLAG,
  
  CASE 
    WHEN l.BMI >= 30 THEN 1 ELSE 0 
  END AS OBESE_FLAG,
  
  -- ================================================================
  -- ADDITIONAL VITAL SIGN FLAGS
  -- ================================================================
  
  -- Fever: >100.4°F (38°C)
  CASE 
    WHEN l.TEMPERATURE > 100.4 THEN 1 ELSE 0 
  END AS FEVER_FLAG,
  
  -- Tachypnea: respiratory rate >20 breaths/min
  CASE 
    WHEN l.RESP_RATE > 20 THEN 1 ELSE 0 
  END AS TACHYPNEA_FLAG,
  
  -- Bradypnea: respiratory rate <12 breaths/min
  CASE 
    WHEN l.RESP_RATE < 12 THEN 1 ELSE 0 
  END AS BRADYPNEA_FLAG,
  
  -- ================================================================
  -- CACHEXIA RISK SCORE
  -- ================================================================
  -- Cancer cachexia: syndrome of weight loss + low BMI
  -- Common in advanced CRC, but can be early sign
  CASE
    -- High risk: BMI <20 AND 5% weight loss
    WHEN l.BMI < 20 
      AND l.WEIGHT_OZ IS NOT NULL 
      AND l.WEIGHT_OZ_6M IS NOT NULL 
      AND ((l.WEIGHT_OZ - l.WEIGHT_OZ_6M) / NULLIF(l.WEIGHT_OZ_6M, 0)) * 100 <= -5
    THEN 2  
    
    -- Moderate risk: (BMI <22 AND weight loss) OR (BMI <20 alone)
    WHEN (l.BMI < 22 
      AND l.WEIGHT_OZ IS NOT NULL 
      AND l.WEIGHT_OZ_6M IS NOT NULL 
      AND ((l.WEIGHT_OZ - l.WEIGHT_OZ_6M) / NULLIF(l.WEIGHT_OZ_6M, 0)) * 100 <= -5)
      OR (l.BMI < 20)
    THEN 1  
    
    -- Low risk: normal BMI or no weight loss
    ELSE 0  
  END AS CACHEXIA_RISK_SCORE
  
FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals_latest l
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_vitals_patterns p
  ON l.PAT_ID = p.PAT_ID AND l.END_DTTM = p.END_DTTM
""")

print("✓ Final vitals features created")

# ========================================
# CELL 6
# ========================================

# CELL 5 - VALIDATE ROW COUNT
# ============================
# Must have exactly 11,449,023 rows matching base cohort

result = spark.sql(f"""
SELECT 
    COUNT(*) as vitals_count,
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals
""")

result.show()
print("\n✓ Row count validation complete")
assert result.collect()[0]['diff'] == 0, "ERROR: Row count mismatch!"

# ========================================
# CELL 7
# ========================================

# CELL 6 - ANALYZE VITAL SIGNS COVERAGE
# ======================================
# Check completeness of each vital sign

spark.sql(f"""
SELECT 
    COUNT(*) as total_rows,
    
    -- Basic vitals coverage
    SUM(CASE WHEN WEIGHT_OZ IS NOT NULL THEN 1 ELSE 0 END) as has_weight,
    ROUND(100.0 * SUM(CASE WHEN WEIGHT_OZ IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as weight_pct,
    
    SUM(CASE WHEN BP_SYSTOLIC IS NOT NULL THEN 1 ELSE 0 END) as has_bp,
    ROUND(100.0 * SUM(CASE WHEN BP_SYSTOLIC IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as bp_pct,
    
    SUM(CASE WHEN PULSE IS NOT NULL THEN 1 ELSE 0 END) as has_pulse,
    ROUND(100.0 * SUM(CASE WHEN PULSE IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as pulse_pct,
    
    SUM(CASE WHEN BMI IS NOT NULL THEN 1 ELSE 0 END) as has_bmi,
    ROUND(100.0 * SUM(CASE WHEN BMI IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as bmi_pct,
    
    SUM(CASE WHEN TEMPERATURE IS NOT NULL THEN 1 ELSE 0 END) as has_temp,
    ROUND(100.0 * SUM(CASE WHEN TEMPERATURE IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as temp_pct,
    
    SUM(CASE WHEN RESP_RATE IS NOT NULL THEN 1 ELSE 0 END) as has_resp,
    ROUND(100.0 * SUM(CASE WHEN RESP_RATE IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as resp_pct

FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals
""").show(truncate=False)

# ========================================
# CELL 8
# ========================================

# CELL 7 - VITAL SIGNS DISTRIBUTIONS
# ===================================
# Check distributions for plausibility

spark.sql(f"""
SELECT 
    -- Weight statistics
    ROUND(MIN(WEIGHT_LB), 1) as weight_min,
    ROUND(PERCENTILE_APPROX(WEIGHT_LB, 0.01), 1) as weight_p1,
    ROUND(PERCENTILE_APPROX(WEIGHT_LB, 0.25), 1) as weight_q1,
    ROUND(PERCENTILE_APPROX(WEIGHT_LB, 0.50), 1) as weight_median,
    ROUND(PERCENTILE_APPROX(WEIGHT_LB, 0.75), 1) as weight_q3,
    ROUND(PERCENTILE_APPROX(WEIGHT_LB, 0.99), 1) as weight_p99,
    ROUND(MAX(WEIGHT_LB), 1) as weight_max,
    
    -- BMI statistics
    ROUND(MIN(BMI), 1) as bmi_min,
    ROUND(PERCENTILE_APPROX(BMI, 0.01), 1) as bmi_p1,
    ROUND(PERCENTILE_APPROX(BMI, 0.25), 1) as bmi_q1,
    ROUND(PERCENTILE_APPROX(BMI, 0.50), 1) as bmi_median,
    ROUND(PERCENTILE_APPROX(BMI, 0.75), 1) as bmi_q3,
    ROUND(PERCENTILE_APPROX(BMI, 0.99), 1) as bmi_p99,
    ROUND(MAX(BMI), 1) as bmi_max,
    
    -- BP Systolic statistics  
    ROUND(MIN(BP_SYSTOLIC), 0) as sbp_min,
    ROUND(PERCENTILE_APPROX(BP_SYSTOLIC, 0.01), 0) as sbp_p1,
    ROUND(PERCENTILE_APPROX(BP_SYSTOLIC, 0.50), 0) as sbp_median,
    ROUND(PERCENTILE_APPROX(BP_SYSTOLIC, 0.99), 0) as sbp_p99,
    ROUND(MAX(BP_SYSTOLIC), 0) as sbp_max

FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals
WHERE WEIGHT_LB IS NOT NULL OR BMI IS NOT NULL OR BP_SYSTOLIC IS NOT NULL
""").show(truncate=False)

# ========================================
# CELL 9
# ========================================

# CELL 8 - WEIGHT CHANGE ANALYSIS
# ================================
# Critical CRC indicator - weight loss patterns

spark.sql(f"""
SELECT 
    -- Weight change coverage
    COUNT(*) as total_rows,
    SUM(CASE WHEN WEIGHT_CHANGE_PCT_6M IS NOT NULL THEN 1 ELSE 0 END) as has_6m_change,
    ROUND(100.0 * SUM(CASE WHEN WEIGHT_CHANGE_PCT_6M IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_with_6m_change,
    
    -- Weight loss prevalence
    SUM(WEIGHT_LOSS_5PCT_6M) as weight_loss_5pct_count,
    ROUND(100.0 * SUM(WEIGHT_LOSS_5PCT_6M) / NULLIF(SUM(CASE WHEN WEIGHT_CHANGE_PCT_6M IS NOT NULL THEN 1 ELSE 0 END), 0), 2) as pct_with_5pct_loss,
    
    SUM(WEIGHT_LOSS_10PCT_6M) as weight_loss_10pct_count,
    ROUND(100.0 * SUM(WEIGHT_LOSS_10PCT_6M) / NULLIF(SUM(CASE WHEN WEIGHT_CHANGE_PCT_6M IS NOT NULL THEN 1 ELSE 0 END), 0), 2) as pct_with_10pct_loss,
    
    SUM(RAPID_WEIGHT_LOSS_FLAG) as rapid_loss_count,
    ROUND(100.0 * SUM(RAPID_WEIGHT_LOSS_FLAG) / COUNT(*), 2) as pct_rapid_loss,
    
    -- Weight trajectory
    AVG(WEIGHT_TRAJECTORY_SLOPE) as avg_weight_slope,
    STDDEV(WEIGHT_TRAJECTORY_SLOPE) as std_weight_slope,
    
    -- Cachexia risk
    SUM(CASE WHEN CACHEXIA_RISK_SCORE = 2 THEN 1 ELSE 0 END) as high_cachexia_risk,
    SUM(CASE WHEN CACHEXIA_RISK_SCORE = 1 THEN 1 ELSE 0 END) as mod_cachexia_risk,
    ROUND(100.0 * SUM(CASE WHEN CACHEXIA_RISK_SCORE > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_any_cachexia_risk
    
FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals
""").show(truncate=False)

# ========================================
# CELL 10
# ========================================

# CELL 9 - CLINICAL FLAG PREVALENCE
# ==================================
# Check prevalence of clinical conditions

spark.sql(f"""
SELECT 
    ROUND(100.0 * SUM(HYPERTENSION_FLAG) / COUNT(*), 2) as hypertension_pct,
    ROUND(100.0 * SUM(SEVERE_HYPERTENSION_FLAG) / COUNT(*), 2) as severe_htn_pct,
    ROUND(100.0 * SUM(TACHYCARDIA_FLAG) / COUNT(*), 2) as tachycardia_pct,
    ROUND(100.0 * SUM(UNDERWEIGHT_FLAG) / COUNT(*), 2) as underweight_pct,
    ROUND(100.0 * SUM(OBESE_FLAG) / COUNT(*), 2) as obese_pct,
    ROUND(100.0 * SUM(FEVER_FLAG) / COUNT(*), 2) as fever_pct,
    ROUND(100.0 * SUM(TACHYPNEA_FLAG) / COUNT(*), 2) as tachypnea_pct,
    ROUND(100.0 * SUM(BRADYPNEA_FLAG) / COUNT(*), 2) as bradypnea_pct,
    
    -- BMI transitions
    ROUND(100.0 * SUM(BMI_LOST_OBESE_STATUS) / COUNT(*), 2) as lost_obese_status_pct,
    ROUND(100.0 * SUM(BMI_LOST_OVERWEIGHT_STATUS) / COUNT(*), 2) as lost_overweight_status_pct
    
FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals
""").show(truncate=False)

# ========================================
# CELL 11
# ========================================

# CELL 10 - DATA FRESHNESS ANALYSIS
# =================================
# How recent are the vital measurements?

spark.sql(f"""
SELECT 
    -- Weight recency
    ROUND(PERCENTILE_APPROX(DAYS_SINCE_WEIGHT, 0.25), 0) as days_since_weight_q1,
    ROUND(PERCENTILE_APPROX(DAYS_SINCE_WEIGHT, 0.50), 0) as days_since_weight_median,
    ROUND(PERCENTILE_APPROX(DAYS_SINCE_WEIGHT, 0.75), 0) as days_since_weight_q3,
    SUM(CASE WHEN DAYS_SINCE_WEIGHT <= 30 THEN 1 ELSE 0 END) as weight_within_30d,
    SUM(CASE WHEN DAYS_SINCE_WEIGHT <= 90 THEN 1 ELSE 0 END) as weight_within_90d,
    SUM(CASE WHEN DAYS_SINCE_WEIGHT <= 365 THEN 1 ELSE 0 END) as weight_within_1yr,
    
    -- BP recency
    ROUND(PERCENTILE_APPROX(DAYS_SINCE_SBP, 0.50), 0) as days_since_bp_median,
    SUM(CASE WHEN DAYS_SINCE_SBP <= 90 THEN 1 ELSE 0 END) as bp_within_90d,
    
    -- BMI recency
    ROUND(PERCENTILE_APPROX(DAYS_SINCE_BMI, 0.50), 0) as days_since_bmi_median,
    SUM(CASE WHEN DAYS_SINCE_BMI <= 90 THEN 1 ELSE 0 END) as bmi_within_90d
    
FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals
WHERE DAYS_SINCE_WEIGHT IS NOT NULL OR DAYS_SINCE_SBP IS NOT NULL
""").show(truncate=False)

# ========================================
# CELL 12
# ========================================

# CELL 11 - CORRELATION WITH CRC OUTCOME
# ======================================
# Check association of key features with CRC events

spark.sql(f"""
WITH outcome_analysis AS (
    SELECT 
        v.*,
        c.FUTURE_CRC_EVENT
    FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals v
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
        ON v.PAT_ID = c.PAT_ID AND v.END_DTTM = c.END_DTTM
    WHERE c.LABEL_USABLE = 1
)
SELECT 
    -- Overall positive rate
    AVG(CAST(FUTURE_CRC_EVENT AS DOUBLE)) * 100 as overall_crc_rate,
    
    -- Weight loss association
    AVG(CASE WHEN WEIGHT_LOSS_5PCT_6M = 1 THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100 as crc_rate_weight_loss_5pct,
    AVG(CASE WHEN WEIGHT_LOSS_10PCT_6M = 1 THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100 as crc_rate_weight_loss_10pct,
    AVG(CASE WHEN RAPID_WEIGHT_LOSS_FLAG = 1 THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100 as crc_rate_rapid_loss,
    
    -- Cachexia association
    AVG(CASE WHEN CACHEXIA_RISK_SCORE = 0 THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100 as crc_rate_cachexia_0,
    AVG(CASE WHEN CACHEXIA_RISK_SCORE = 1 THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100 as crc_rate_cachexia_1,
    AVG(CASE WHEN CACHEXIA_RISK_SCORE = 2 THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100 as crc_rate_cachexia_2,
    
    -- BMI association
    AVG(CASE WHEN UNDERWEIGHT_FLAG = 1 THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100 as crc_rate_underweight,
    AVG(CASE WHEN OBESE_FLAG = 1 THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100 as crc_rate_obese
    
FROM outcome_analysis
""").show(truncate=False)

# ========================================
# CELL 13
# ========================================

# CELL 12 - BP VARIABILITY ANALYSIS
# =================================
# Check the new BP variability features

spark.sql(f"""
SELECT 
    -- Measurement frequency
    AVG(BP_MEASUREMENT_COUNT_6M) as avg_bp_measurements_6m,
    MAX(BP_MEASUREMENT_COUNT_6M) as max_bp_measurements_6m,
    
    -- Variability statistics
    AVG(SBP_VARIABILITY_6M) as avg_sbp_variability,
    PERCENTILE_APPROX(SBP_VARIABILITY_6M, 0.50) as median_sbp_variability,
    PERCENTILE_APPROX(SBP_VARIABILITY_6M, 0.95) as p95_sbp_variability,
    
    AVG(DBP_VARIABILITY_6M) as avg_dbp_variability,
    AVG(PULSE_PRESSURE_VARIABILITY_6M) as avg_pp_variability,
    
    -- Pulse pressure
    AVG(AVG_PULSE_PRESSURE_6M) as avg_pulse_pressure,
    AVG(PULSE_PRESSURE) as avg_current_pulse_pressure,
    
    -- High variability flag (>15 mmHg SBP variability)
    SUM(CASE WHEN SBP_VARIABILITY_6M > 15 THEN 1 ELSE 0 END) as high_bp_variability_count,
    ROUND(100.0 * SUM(CASE WHEN SBP_VARIABILITY_6M > 15 THEN 1 ELSE 0 END) / 
          NULLIF(SUM(CASE WHEN SBP_VARIABILITY_6M IS NOT NULL THEN 1 ELSE 0 END), 0), 2) as pct_high_bp_variability
    
FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals
""").show(truncate=False)

# ========================================
# CELL 14
# ========================================

# CELL 13 - SUMMARY STATISTICS
# =============================
# Final summary of vital features quality

print("=" * 80)
print("VITALS FEATURES SUMMARY")
print("=" * 80)

summary = spark.sql(f"""
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    
    -- Core vitals availability
    ROUND(100.0 * SUM(CASE WHEN WEIGHT_OZ IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as has_weight_pct,
    ROUND(100.0 * SUM(CASE WHEN BMI IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as has_bmi_pct,
    ROUND(100.0 * SUM(CASE WHEN BP_SYSTOLIC IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as has_bp_pct,
    
    -- Enhanced features availability  
    ROUND(100.0 * SUM(CASE WHEN WEIGHT_CHANGE_PCT_6M IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as has_weight_trend_pct,
    ROUND(100.0 * SUM(CASE WHEN SBP_VARIABILITY_6M IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as has_bp_variability_pct,
    
    -- Key risk indicators
    SUM(WEIGHT_LOSS_5PCT_6M) as weight_loss_cases,
    SUM(CASE WHEN CACHEXIA_RISK_SCORE > 0 THEN 1 ELSE 0 END) as cachexia_risk_cases
    
FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals
""").collect()[0]

for key, value in summary.asDict().items():
    if value is not None:
        if 'pct' in key:
            print(f"{key:30s}: {value:>10.1f}%")
        else:
            print(f"{key:30s}: {value:>10,}")

print("=" * 80)
print("✓ Vitals feature engineering complete")

# ========================================
# CELL 15
# ========================================

# CELL 14
df = spark.sql(f'''select * from dev.clncl_ds.herald_eda_train_vitals''')
df.count()

# ========================================
# CELL 16
# ========================================

# CELL 15
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType


# --- % Nulls (all columns) ---
null_pct_long = (
    df.select([
        (F.avg(F.col(c).isNull().cast("int")) * F.lit(100.0)).alias(c)
        for c in df.columns
    ])
    .select(F.explode(F.array(*[
        F.struct(F.lit(c).alias("column"), F.col(c).alias("pct_null"))
        for c in df.columns
    ])).alias("kv"))
    .select("kv.column", "kv.pct_null")
)

# --- Means (numeric columns only) ---
numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

mean_long = (
    df.select([F.avg(F.col(c)).alias(c) for c in numeric_cols])
    .select(F.explode(F.array(*[
        F.struct(F.lit(c).alias("column"), F.col(c).alias("mean"))
        for c in numeric_cols
    ])).alias("kv"))
    .select("kv.column", "kv.mean")
)

# --- Join & present ---
profile = (
    null_pct_long
    .join(mean_long, on="column", how="left")  # non-numerics get mean = null
    .select(
        "column",
        F.round("pct_null", 4).alias("pct_null"),
        F.round("mean", 6).alias("mean")
    )
    .orderBy(F.desc("pct_null"))
)

profile.show(200, truncate=False)

# ========================================
# CELL 17
# ========================================

# CELL 16 Vitals Feature Reduction using PySpark

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np

# Step 1: Load data and remove obvious redundancies
print("Loading vitals data and removing redundancies...")

# Define redundant features to remove upfront
REDUNDANT_FEATURES = [
    'WEIGHT_LB',  # Keep WEIGHT_OZ for precision
    'WEIGHT_DATE', 'BMI_DATE', 'BP_DATE', 'PULSE_DATE', 'TEMP_DATE', 'RESP_DATE',
    'DAYS_SINCE_DBP',  # Same as DAYS_SINCE_SBP
    'MIN_WEIGHT_12M', 'MAX_WEIGHT_12M',  # Captured in volatility
    'BMI_LOST_OBESE_STATUS', 'BMI_LOST_OVERWEIGHT_STATUS',  # Low prevalence
    'BRADYPNEA_FLAG',  # Very rare
]

# Join with outcome data
df_spark = spark.sql(f"""
    SELECT v.*, c.FUTURE_CRC_EVENT
    FROM {trgt_cat}.clncl_ds.herald_eda_train_vitals v
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
        ON v.PAT_ID = c.PAT_ID AND v.END_DTTM = c.END_DTTM
""")

# Remove redundant columns and cache
cols_to_keep = [c for c in df_spark.columns if c not in REDUNDANT_FEATURES]
df_spark = df_spark.select(*cols_to_keep)
df_spark.cache()

total_rows = df_spark.count()
baseline_crc_rate = df_spark.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"Total rows: {total_rows:,}")
print(f"Baseline CRC rate: {baseline_crc_rate:.4f}")

feature_cols = [c for c in df_spark.columns if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT']]
print(f"Features after removing redundant: {len(feature_cols)}")

# ========================================
# CELL 18
# ========================================

# CELL 17 - Step 2: Calculate Risk Ratios for Binary Features
print("\nCalculating risk ratios for binary features...")

binary_features = [col for col in feature_cols if '_FLAG' in col or 'CACHEXIA_RISK_SCORE' in col]
risk_metrics = []

for feat in binary_features:
    if 'CACHEXIA_RISK_SCORE' in feat:
        # Handle the 0-2 score specially - treat high risk (2) as binary
        stats = df_spark.filter(F.col(feat) == 2).agg(
            F.count('*').alias('count'),
            F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
        ).collect()[0]
        
        prevalence = stats['count'] / total_rows
        risk_ratio = stats['crc_rate'] / baseline_crc_rate
        impact = prevalence * abs(np.log2(max(risk_ratio, 1/risk_ratio)))
        
        risk_metrics.append({
            'feature': feat,
            'prevalence': prevalence,
            'crc_rate_with': stats['crc_rate'],
            'risk_ratio': risk_ratio,
            'impact': impact
        })
    else:
        # Standard binary flags
        stats = df_spark.groupBy(feat).agg(
            F.count('*').alias('count'),
            F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
        ).collect()
        
        stats_dict = {row[feat]: {'count': row['count'], 'crc_rate': row['crc_rate']} for row in stats}
        
        prevalence = stats_dict.get(1, {'count': 0})['count'] / total_rows
        rate_with = stats_dict.get(1, {'crc_rate': 0})['crc_rate']
        rate_without = stats_dict.get(0, {'crc_rate': baseline_crc_rate})['crc_rate']
        risk_ratio = rate_with / (rate_without + 1e-10)
        
        if risk_ratio > 0 and prevalence > 0:
            impact = prevalence * abs(np.log2(max(risk_ratio, 1/(risk_ratio + 1e-10))))
        else:
            impact = 0
        
        risk_metrics.append({
            'feature': feat,
            'prevalence': prevalence,
            'crc_rate_with': rate_with,
            'risk_ratio': risk_ratio,
            'impact': impact
        })

risk_df = pd.DataFrame(risk_metrics).sort_values('impact', ascending=False)
print("\nTop features by impact score:")
print(risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']].head(10))

# ========================================
# CELL 19
# ========================================

# CELL 18 - Step 3: Calculate Mutual Information on Stratified Sample
print("\nCalculating Mutual Information...")

# Sample for MI calculation (stratified by outcome)
sample_fraction = min(100000 / total_rows, 1.0)
df_sample = df_spark.sampleBy("FUTURE_CRC_EVENT", 
                               fractions={0: sample_fraction, 1: 1.0},
                               seed=42).toPandas()

print(f"Sampled {len(df_sample):,} rows for MI calculation")

# Calculate MI for all features
from sklearn.feature_selection import mutual_info_classif

X = df_sample[feature_cols].fillna(-999)
y = df_sample['FUTURE_CRC_EVENT']

mi_scores = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop features by Mutual Information:")
print(mi_df.head(15))

# ========================================
# CELL 20
# ========================================

# CELL 19 - Step 4: Apply Clinical Knowledge and Feature Grouping
print("\nApplying clinical knowledge filters...")

# Merge all metrics
feature_importance = mi_df.merge(
    risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']], 
    on='feature', 
    how='left'
)

# Fill NAs for non-flag features
feature_importance['risk_ratio'] = feature_importance['risk_ratio'].fillna(1.0)
feature_importance['impact'] = feature_importance['impact'].fillna(0)

# Clinical must-keep features
MUST_KEEP = [
    'RAPID_WEIGHT_LOSS_FLAG',
    'WEIGHT_LOSS_10PCT_6M', 
    'WEIGHT_CHANGE_PCT_6M',
    'MAX_WEIGHT_LOSS_PCT_60D',
    'CACHEXIA_RISK_SCORE',
    'BMI',
    'BP_SYSTOLIC',
    'UNDERWEIGHT_FLAG',
    'HYPERTENSION_FLAG'
]

# Near-zero variance features to remove
REMOVE = ['RESP_RATE', 'TEMPERATURE']  # Low signal, temperature rarely populated

print(f"Removing {len(REMOVE)} low-signal features")
feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]

# ========================================
# CELL 21
# ========================================

# CELL 20 - Step 5: Select Optimal Representation per Vital Category
print("\nSelecting optimal features per category...")

def select_optimal_vitals(df_importance):
    """Select best representation for each vital category"""
    
    selected = []
    
    # Define feature groups to handle multicollinearity
    groups = {
        'weight_loss': ['WEIGHT_LOSS_5PCT_6M', 'WEIGHT_LOSS_10PCT_6M', 
                       'RAPID_WEIGHT_LOSS_FLAG', 'WEIGHT_CHANGE_PCT_6M',
                       'WEIGHT_CHANGE_PCT_12M', 'MAX_WEIGHT_LOSS_PCT_60D'],
        'weight_trajectory': ['WEIGHT_TRAJECTORY_SLOPE', 'WEIGHT_TRAJECTORY_R2',
                             'WEIGHT_VOLATILITY_12M'],
        'bp_measures': ['BP_SYSTOLIC', 'BP_DIASTOLIC', 'PULSE_PRESSURE', 
                       'MEAN_ARTERIAL_PRESSURE'],
        'bp_variability': ['SBP_VARIABILITY_6M', 'DBP_VARIABILITY_6M',
                          'PULSE_PRESSURE_VARIABILITY_6M'],
        'bmi_change': ['BMI_CHANGE_6M', 'BMI_CHANGE_12M'],
        'recency': ['DAYS_SINCE_WEIGHT', 'DAYS_SINCE_SBP', 'DAYS_SINCE_BMI',
                   'DAYS_SINCE_PULSE', 'DAYS_SINCE_TEMPERATURE']
    }
    
    # Process each group
    for group_name, group_features in groups.items():
        available = df_importance[df_importance['feature'].isin(group_features)]
        
        if group_name == 'weight_loss':
            # Keep multiple for this critical group
            for feat in ['RAPID_WEIGHT_LOSS_FLAG', 'WEIGHT_LOSS_10PCT_6M', 
                        'WEIGHT_CHANGE_PCT_6M', 'MAX_WEIGHT_LOSS_PCT_60D']:
                if feat in available['feature'].values:
                    selected.append(feat)
                    
        elif group_name == 'weight_trajectory':
            # Keep slope only
            if 'WEIGHT_TRAJECTORY_SLOPE' in available['feature'].values:
                selected.append('WEIGHT_TRAJECTORY_SLOPE')
                
        elif group_name == 'bp_measures':
            # Keep systolic and pulse pressure
            for feat in ['BP_SYSTOLIC', 'PULSE_PRESSURE']:
                if feat in available['feature'].values:
                    selected.append(feat)
                    
        elif group_name == 'bp_variability':
            # Keep systolic variability
            if 'SBP_VARIABILITY_6M' in available['feature'].values:
                selected.append('SBP_VARIABILITY_6M')
                
        elif group_name == 'recency':
            # Keep weight recency only
            if 'DAYS_SINCE_WEIGHT' in available['feature'].values:
                selected.append('DAYS_SINCE_WEIGHT')
                
        else:
            # Select top by MI score
            if len(available) > 0:
                best = available.nlargest(1, 'mi_score')['feature'].values[0]
                selected.append(best)
    
    # Add individual high-value features not in groups
    individual_features = ['BMI', 'WEIGHT_OZ', 'PULSE', 'CACHEXIA_RISK_SCORE',
                          'UNDERWEIGHT_FLAG', 'OBESE_FLAG', 'HYPERTENSION_FLAG',
                          'TACHYCARDIA_FLAG', 'FEVER_FLAG']
    
    for feat in individual_features:
        if feat in df_importance['feature'].values and feat not in selected:
            selected.append(feat)
    
    # Ensure must-keep features
    for feat in MUST_KEEP:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    return list(set(selected))

selected_features = select_optimal_vitals(feature_importance)
print(f"Selected {len(selected_features)} features after optimization")

# ========================================
# CELL 22
# ========================================

# CELL 21 - Step 6: Create Composite Features
print("\nCreating composite features...")

df_final = df_spark

# Weight loss severity score (0-3 scale)
df_final = df_final.withColumn('weight_loss_severity',
    F.when(F.col('WEIGHT_LOSS_10PCT_6M') == 1, 3)
     .when(F.col('WEIGHT_LOSS_5PCT_6M') == 1, 2)
     .when(F.col('WEIGHT_CHANGE_PCT_6M') < -2, 1)
     .otherwise(0)
)

# Vital measurement recency score
df_final = df_final.withColumn('vital_recency_score',
    F.when(F.col('DAYS_SINCE_WEIGHT').isNull(), 0)
     .when(F.col('DAYS_SINCE_WEIGHT') <= 30, 3)
     .when(F.col('DAYS_SINCE_WEIGHT') <= 90, 2)
     .when(F.col('DAYS_SINCE_WEIGHT') <= 180, 1)
     .otherwise(0)
)

# Combined cardiovascular risk
df_final = df_final.withColumn('cardiovascular_risk',
    F.when((F.col('HYPERTENSION_FLAG') == 1) & 
           (F.col('OBESE_FLAG') == 1), 2)
     .when((F.col('HYPERTENSION_FLAG') == 1) | 
           (F.col('OBESE_FLAG') == 1), 1)
     .otherwise(0)
)

# Abnormal weight pattern
df_final = df_final.withColumn('abnormal_weight_pattern',
    F.when((F.col('MAX_WEIGHT_LOSS_PCT_60D') > 5) |
           (F.col('WEIGHT_TRAJECTORY_SLOPE') < -0.5), 1)
     .otherwise(0)
)

# BP instability flag
df_final = df_final.withColumn('bp_instability',
    F.when(F.col('SBP_VARIABILITY_6M') > 15, 1).otherwise(0)
)

composite_features = ['weight_loss_severity', 'vital_recency_score', 
                      'cardiovascular_risk', 'abnormal_weight_pattern', 
                      'bp_instability']

selected_features.extend(composite_features)

print(f"Added {len(composite_features)} composite features")
print(f"Final feature count: {len(selected_features)}")

# ========================================
# CELL 23
# ========================================

# CELL 22 - Step 7: Save Reduced Dataset and Validate
print("\n" + "="*60)
print("FINAL SELECTED FEATURES")
print("="*60)

# Remove duplicates and sort
selected_features_final = sorted(list(set(selected_features)))

# Categorize for display
weight_features = [f for f in selected_features_final if 'WEIGHT' in f.upper() or 'weight' in f]
other_features = [f for f in selected_features_final if f not in weight_features and f not in composite_features]

print(f"Weight-related ({len(weight_features)}):")
for feat in sorted(weight_features):
    risk = " [EXTREME RISK]" if 'RAPID' in feat or '10PCT' in feat else ""
    print(f"  - {feat}{risk}")

print(f"\nOther vitals ({len(other_features)}):")
for feat in sorted(other_features):
    print(f"  - {feat}")

print(f"\nComposite ({len(composite_features)}):")
for feat in composite_features:
    print(f"  - {feat}")

# Select final columns and save
final_columns = ['PAT_ID', 'END_DTTM'] + selected_features_final
df_reduced = df_final.select(*[c for c in final_columns if c in df_final.columns])

# Add vit_ prefix to all columns except keys
vit_cols = [col for col in df_reduced.columns if col not in ['PAT_ID', 'END_DTTM']]
for col in vit_cols:
    df_reduced = df_reduced.withColumnRenamed(col, f'vit_{col}' if not col.startswith('vit_') else col)

# Write to final table
output_table = f'{trgt_cat}.clncl_ds.herald_eda_train_vitals_reduced'
df_reduced.write.mode('overwrite').saveAsTable(output_table)

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: ~70")
print(f"Selected features: {len(selected_features_final)}")
print(f"Reduction: {(1 - len(selected_features_final)/70)*100:.1f}%")
print(f"\n✔ Reduced dataset saved to: {output_table}")

# Verify save
row_count = spark.table(output_table).count()
print(f"✔ Verified {row_count:,} rows written to table")

# ========================================
# CELL 24
# ========================================

df_check_spark = spark.sql('select * from dev.clncl_ds.herald_eda_train_vitals_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/df_check.shape[0]


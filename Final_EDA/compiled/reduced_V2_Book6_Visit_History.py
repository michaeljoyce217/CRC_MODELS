# V2_Book6_Visit_History
# Functional cells: 26 of 54 code cells (105 total)
# Source: V2_Book6_Visit_History.ipynb
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
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Initialize a Spark session for distributed data processing
spark = SparkSession.builder.getOrCreate()

# Ensure date/time comparisons use Central Time
spark.conf.set("spark.sql.session.timeZone", "America/Chicago")

# Define target catalog for SQL based on the environment variable
trgt_cat = os.environ.get('trgt_cat')

# Use appropriate Spark catalog based on the target category
spark.sql('USE CATALOG prod;')

# ========================================
# CELL 2
# ========================================

# Cell 1: Create ED/Inpatient encounters with GI symptom flags
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_encounters AS
WITH gi_symptom_enc AS (
    SELECT DISTINCT PRIM_ENC_CSN_ID
    FROM clarity_cur.hsp_acct_dx_list_enh dx
    WHERE dx.CODE RLIKE '^(K92|K59|R19|R50|D50|K62)' -- GI bleeding, bowel changes, abdominal pain, anemia
)

SELECT
  peh.PAT_ID,
  peh.PAT_ENC_CSN_ID,
  peh.HOSP_ADMSN_TIME AS visit_dt,
  peh.ACCT_CLASS,
  peh.ED_EPISODE_ID,
  CASE WHEN ge.PRIM_ENC_CSN_ID IS NOT NULL THEN 1 ELSE 0 END AS GI_SYMPTOM_YN,
  CASE WHEN peh.DISCH_DATE_TIME IS NOT NULL AND peh.HOSP_ADMSN_TIME IS NOT NULL
       THEN DATEDIFF(DATE(peh.DISCH_DATE_TIME), DATE(peh.HOSP_ADMSN_TIME))
       ELSE NULL END AS los_days,
  CASE 
    WHEN peh.ACCT_CLASS = 'Emergency' OR peh.ED_EPISODE_ID IS NOT NULL THEN 'ED'
    WHEN peh.ACCT_CLASS = 'Inpatient' THEN 'INPATIENT'
    ELSE 'OTHER'
  END AS encounter_type
FROM clarity_cur.pat_enc_hsp_har_enh peh
LEFT JOIN gi_symptom_enc ge ON peh.PAT_ENC_CSN_ID = ge.PRIM_ENC_CSN_ID
WHERE peh.HOSP_ADMSN_TIME >= '2021-07-01'  -- CRITICAL: Data quality cutoff
  AND peh.HOSP_ADMSN_TIME IS NOT NULL
  AND peh.ADT_PATIENT_STAT_C <> 1  -- Exclude preadmits
  AND peh.ADMIT_CONF_STAT_C <> 3   -- Exclude canceled
""")

print("✓ ED/Inpatient encounters created (≥2021-07-01 only)")

# Validation query
validation = spark.sql(f"""
SELECT 
  MIN(visit_dt) as earliest_visit,
  MAX(visit_dt) as latest_visit,
  COUNT(*) as total_encounters,
  COUNT(CASE WHEN visit_dt < '2021-07-01' THEN 1 END) as pre_cutoff_count
FROM {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_encounters
""")
validation.show()

# ========================================
# CELL 3
# ========================================

# Cell 2: Create outpatient encounters (including all appointment statuses for no-show tracking)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_op_encounters AS
WITH gi_symptom_dx AS (
  SELECT DISTINCT PAT_ENC_CSN_ID
  FROM clarity_cur.pat_enc_dx_enh dx
  WHERE dx.ICD10_CODE RLIKE '^(K92|K59|R19|R50|D50|K62)' -- GI symptoms
)

SELECT
  pe.PAT_ID,
  pe.PAT_ENC_CSN_ID,
  TO_DATE(SUBSTRING(pe.CONTACT_DATE,1,10)) AS visit_dt,
  ser.SPECIALTY_NAME,
  pe.APPT_STATUS_C,
  pe.APPT_STATUS_NAME,
  CASE 
    WHEN UPPER(ser.SPECIALTY_NAME) IN ('GASTROENTEROLOGY', 'COLON AND RECTAL SURGERY')
    THEN 1 ELSE 0 
  END AS GI_SPECIALTY_YN,
  CASE 
    WHEN pe.VISIT_PROV_ID = pe.PCP_PROV_ID THEN 1 ELSE 0 
  END AS PCP_VISIT_YN,
  CASE 
    WHEN gd.PAT_ENC_CSN_ID IS NOT NULL THEN 1 ELSE 0
  END AS GI_SYMPTOM_YN
FROM clarity_cur.pat_enc_enh pe
LEFT JOIN clarity_cur.clarity_ser_enh ser ON pe.VISIT_PROV_ID = ser.PROV_ID
LEFT JOIN gi_symptom_dx gd ON pe.PAT_ENC_CSN_ID = gd.PAT_ENC_CSN_ID
WHERE pe.CONTACT_DATE >= '2021-07-01'  -- CRITICAL: Data quality cutoff
  AND pe.ENC_TYPE_NAME NOT IN ('Hospital Encounter')
  -- Include ALL appointment statuses (completed, arrived, no-show, etc.)
""")

print("✓ Outpatient encounters created (≥2021-07-01 only)")

# Validation query
validation = spark.sql(f"""
SELECT 
  MIN(visit_dt) as earliest_visit,
  MAX(visit_dt) as latest_visit,
  COUNT(*) as total_encounters,
  COUNT(CASE WHEN visit_dt < '2021-07-01' THEN 1 END) as pre_cutoff_count
FROM {trgt_cat}.clncl_ds.herald_eda_train_op_encounters
""")
validation.show()

# ========================================
# CELL 4
# ========================================

# Cell 3: ED/Inpatient metrics only (much faster)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_metrics AS
SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- ED visits (various time windows)
  SUM(CASE 
    WHEN ei.encounter_type = 'ED' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -3)
     AND ei.visit_dt < c.END_DTTM
    THEN 1 ELSE 0 
  END) AS ED_LAST_90_DAYS,

  SUM(CASE 
    WHEN ei.encounter_type = 'ED' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND ei.visit_dt < c.END_DTTM
    THEN 1 ELSE 0 
  END) AS ED_LAST_12_MONTHS,

  SUM(CASE 
    WHEN ei.encounter_type = 'ED' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
     AND ei.visit_dt < c.END_DTTM
    THEN 1 ELSE 0 
  END) AS ED_LAST_24_MONTHS,

  -- GI symptom-related ED visits
  SUM(CASE 
    WHEN ei.encounter_type = 'ED' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND ei.visit_dt < c.END_DTTM
     AND ei.GI_SYMPTOM_YN = 1
    THEN 1 ELSE 0 
  END) AS GI_ED_LAST_12_MONTHS,

  -- Inpatient visits
  SUM(CASE 
    WHEN ei.encounter_type = 'INPATIENT' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND ei.visit_dt < c.END_DTTM
    THEN 1 ELSE 0 
  END) AS INP_LAST_12_MONTHS,

  SUM(CASE 
    WHEN ei.encounter_type = 'INPATIENT' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
     AND ei.visit_dt < c.END_DTTM
    THEN 1 ELSE 0 
  END) AS INP_LAST_24_MONTHS,

  -- GI symptom-related inpatient visits
  SUM(CASE 
    WHEN ei.encounter_type = 'INPATIENT' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND ei.visit_dt < c.END_DTTM
     AND ei.GI_SYMPTOM_YN = 1
    THEN 1 ELSE 0 
  END) AS GI_INP_LAST_12_MONTHS,

  -- Total inpatient days
  SUM(CASE 
    WHEN ei.encounter_type = 'INPATIENT' 
     AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND ei.visit_dt < c.END_DTTM
    THEN COALESCE(ei.los_days, 0) ELSE 0 
  END) AS TOTAL_INPATIENT_DAYS_12MO

FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_encounters ei
  ON c.PAT_ID = ei.PAT_ID
GROUP BY c.PAT_ID, c.END_DTTM
""")

# ========================================
# CELL 5
# ========================================

# Cell 4: Outpatient metrics only (separate for performance)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_op_metrics AS
SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- Outpatient visits (completed only)
  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND op.visit_dt < c.END_DTTM
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS OUTPATIENT_VISITS_12MO,

  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
     AND op.visit_dt < c.END_DTTM
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS OUTPATIENT_VISITS_24MO,

  -- GI specialty visits (completed only)
  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND op.visit_dt < c.END_DTTM
     AND op.GI_SPECIALTY_YN = 1
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS GI_VISITS_12MO,

  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
     AND op.visit_dt < c.END_DTTM
     AND op.GI_SPECIALTY_YN = 1
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS GI_VISITS_24MO,

  -- PCP visits (completed only)
  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND op.visit_dt < c.END_DTTM
     AND op.PCP_VISIT_YN = 1
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS PCP_VISITS_12MO,

  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
     AND op.visit_dt < c.END_DTTM
     AND op.PCP_VISIT_YN = 1
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS PCP_VISITS_24MO,

  -- GI symptom-related outpatient visits (completed only)
  SUM(CASE 
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND op.visit_dt < c.END_DTTM
     AND op.GI_SYMPTOM_YN = 1
     AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
    THEN 1 ELSE 0 
  END) AS GI_SYMPTOM_OP_VISITS_12MO,

  -- No shows (all appointment types)
  SUM(CASE
    WHEN op.visit_dt >= ADD_MONTHS(c.END_DTTM, -12)
     AND op.visit_dt < c.END_DTTM
     AND op.APPT_STATUS_C = 4
    THEN 1 ELSE 0 
  END) AS NO_SHOWS_12MO

FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_op_encounters op
  ON c.PAT_ID = op.PAT_ID
GROUP BY c.PAT_ID, c.END_DTTM
""")

# ========================================
# CELL 6
# ========================================

# Cell 5: Join the two metrics tables (very fast)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_visit_counts AS
SELECT
  ei.PAT_ID,
  ei.END_DTTM,
  
  -- ED/Inpatient metrics
  ei.ED_LAST_90_DAYS,
  ei.ED_LAST_12_MONTHS,
  ei.ED_LAST_24_MONTHS,
  ei.GI_ED_LAST_12_MONTHS,
  ei.INP_LAST_12_MONTHS,
  ei.INP_LAST_24_MONTHS,
  ei.GI_INP_LAST_12_MONTHS,
  ei.TOTAL_INPATIENT_DAYS_12MO,
  
  -- Outpatient metrics
  om.OUTPATIENT_VISITS_12MO,
  om.OUTPATIENT_VISITS_24MO,
  om.GI_VISITS_12MO,
  om.GI_VISITS_24MO,
  om.PCP_VISITS_12MO,
  om.PCP_VISITS_24MO,
  om.GI_SYMPTOM_OP_VISITS_12MO,
  om.NO_SHOWS_12MO

FROM {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_metrics ei
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_op_metrics om
  ON ei.PAT_ID = om.PAT_ID AND ei.END_DTTM = om.END_DTTM
""")

# ========================================
# CELL 7
# ========================================

# Cell 6: Add recency features using window functions
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_visit_recency AS
WITH all_visits AS (
    -- ED visits
    SELECT 
      c.PAT_ID, c.END_DTTM, 'ED' AS visit_type,
      ei.visit_dt,
      DATEDIFF(c.END_DTTM, DATE(ei.visit_dt)) AS days_since
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_encounters ei
      ON c.PAT_ID = ei.PAT_ID
    WHERE ei.encounter_type = 'ED'
      AND ei.visit_dt < c.END_DTTM
      AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
    
    UNION ALL
    
    -- Inpatient visits  
    SELECT 
      c.PAT_ID, c.END_DTTM, 'INPATIENT' AS visit_type,
      ei.visit_dt,
      DATEDIFF(c.END_DTTM, DATE(ei.visit_dt)) AS days_since
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_ed_inp_encounters ei
      ON c.PAT_ID = ei.PAT_ID
    WHERE ei.encounter_type = 'INPATIENT'
      AND ei.visit_dt < c.END_DTTM
      AND ei.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
    
    UNION ALL
    
    -- GI specialty visits
    SELECT 
      c.PAT_ID, c.END_DTTM, 'GI_SPECIALTY' AS visit_type,
      op.visit_dt,
      DATEDIFF(c.END_DTTM, op.visit_dt) AS days_since
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_op_encounters op
      ON c.PAT_ID = op.PAT_ID
    WHERE op.GI_SPECIALTY_YN = 1
      AND op.APPT_STATUS_NAME IN ('Completed','Arrived')
      AND op.visit_dt < c.END_DTTM
      AND op.visit_dt >= ADD_MONTHS(c.END_DTTM, -24)
),

recent_visits AS (
    SELECT 
      PAT_ID, END_DTTM, visit_type, days_since,
      ROW_NUMBER() OVER (
        PARTITION BY PAT_ID, END_DTTM, visit_type 
        ORDER BY days_since ASC
      ) AS rn
    FROM all_visits
),

pivoted_recency AS (
    SELECT
      PAT_ID, END_DTTM,
      SUM(CASE WHEN visit_type = 'ED' AND rn = 1 THEN days_since END) AS days_since_last_ed,
      SUM(CASE WHEN visit_type = 'INPATIENT' AND rn = 1 THEN days_since END) AS days_since_last_inpatient,
      SUM(CASE WHEN visit_type = 'GI_SPECIALTY' AND rn = 1 THEN days_since END) AS days_since_last_gi
    FROM recent_visits
    WHERE rn = 1
    GROUP BY PAT_ID, END_DTTM
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  pr.days_since_last_ed,
  pr.days_since_last_inpatient,
  pr.days_since_last_gi
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
LEFT JOIN pivoted_recency pr ON c.PAT_ID = pr.PAT_ID AND c.END_DTTM = pr.END_DTTM
""")

# ========================================
# CELL 8
# ========================================

# Cell 7: Create final table with composite features
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_visit_features_final AS
SELECT
  vc.PAT_ID,
  vc.END_DTTM,
  
  -- Basic visit counts
  vc.ED_LAST_90_DAYS,
  vc.ED_LAST_12_MONTHS,
  vc.ED_LAST_24_MONTHS,
  vc.GI_ED_LAST_12_MONTHS,
  vc.INP_LAST_12_MONTHS,
  vc.INP_LAST_24_MONTHS,
  vc.GI_INP_LAST_12_MONTHS,
  vc.TOTAL_INPATIENT_DAYS_12MO,
  vc.OUTPATIENT_VISITS_12MO,
  vc.OUTPATIENT_VISITS_24MO,
  vc.GI_VISITS_12MO,
  vc.GI_VISITS_24MO,
  vc.PCP_VISITS_12MO,
  vc.PCP_VISITS_24MO,
  vc.GI_SYMPTOM_OP_VISITS_12MO,
  vc.NO_SHOWS_12MO,
  
  -- Recency features
  vr.days_since_last_ed,
  vr.days_since_last_inpatient,
  vr.days_since_last_gi,
  
  -- Composite flags
  CASE WHEN COALESCE(vc.ED_LAST_12_MONTHS, 0) >= 3 THEN 1 ELSE 0 END AS frequent_ed_user_flag,
  CASE WHEN COALESCE(vc.INP_LAST_12_MONTHS, 0) >= 2 THEN 1 ELSE 0 END AS frequent_inpatient_flag,
  CASE WHEN COALESCE(vc.TOTAL_INPATIENT_DAYS_12MO, 0) >= 10 THEN 1 ELSE 0 END AS high_inpatient_days_flag,
  CASE WHEN COALESCE(vc.PCP_VISITS_12MO, 0) >= 2 THEN 1 ELSE 0 END AS engaged_primary_care_flag,
  CASE WHEN COALESCE(vc.GI_VISITS_12MO, 0) >= 1 THEN 1 ELSE 0 END AS gi_specialty_engagement_flag,
  CASE WHEN COALESCE(vc.ED_LAST_90_DAYS, 0) >= 1 THEN 1 ELSE 0 END AS recent_ed_use_flag,
  CASE WHEN COALESCE(vr.days_since_last_inpatient, 9999) <= 180 THEN 1 ELSE 0 END AS recent_hospitalization_flag,
  
  -- Healthcare intensity score (0-4)
  LEAST(4,
    CASE WHEN COALESCE(vc.ED_LAST_12_MONTHS, 0) >= 3 THEN 1 ELSE 0 END +
    CASE WHEN COALESCE(vc.INP_LAST_12_MONTHS, 0) >= 2 THEN 1 ELSE 0 END +
    CASE WHEN COALESCE(vc.TOTAL_INPATIENT_DAYS_12MO, 0) >= 10 THEN 1 ELSE 0 END +
    CASE WHEN COALESCE(vc.GI_VISITS_12MO, 0) >= 2 THEN 1 ELSE 0 END
  ) AS healthcare_intensity_score,
  
  -- Care continuity ratio
  CASE WHEN COALESCE(vc.OUTPATIENT_VISITS_12MO, 0) > 0 
       THEN ROUND(COALESCE(vc.PCP_VISITS_12MO, 0) * 1.0 / vc.OUTPATIENT_VISITS_12MO, 2)
       ELSE NULL END AS primary_care_continuity_ratio,
       
  -- Total GI symptom burden across all settings
  COALESCE(vc.GI_ED_LAST_12_MONTHS, 0) + 
  COALESCE(vc.GI_INP_LAST_12_MONTHS, 0) + 
  COALESCE(vc.GI_SYMPTOM_OP_VISITS_12MO, 0) AS total_gi_symptom_visits_12mo

FROM {trgt_cat}.clncl_ds.herald_eda_train_visit_counts vc
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_visit_recency vr
  ON vc.PAT_ID = vr.PAT_ID AND vc.END_DTTM = vr.END_DTTM
""")

# ========================================
# CELL 9
# ========================================

# Cell 8
# Validate row counts
result = spark.sql(f"""
SELECT 
    COUNT(*) as visit_features_count,
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_eda_train_visit_features_final
""")
result.show()
assert result.collect()[0]['diff'] == 0, "ERROR: Row count mismatch!"
print("✓ Row count validation passed")

# ========================================
# CELL 10
# ========================================

# Cell 9: Comprehensive validation and summary for visit features
print("="*80)
print("VISIT HISTORY FEATURES - VALIDATION SUMMARY")
print("="*80)

# Load the data for analysis
df = spark.sql(f"SELECT * FROM {trgt_cat}.clncl_ds.herald_eda_train_visit_features_final").toPandas()

# Basic stats
total_rows = df.shape[0]
print(f"\nTotal observations: {total_rows:,}")
print(f"Unique patients: {df['PAT_ID'].nunique():,}")

# Emergency Department utilization
print("\n--- EMERGENCY DEPARTMENT UTILIZATION ---")
print(f"Any ED visit (90 days): {(df['ED_LAST_90_DAYS'] > 0).mean():.1%}")
print(f"Any ED visit (12mo): {(df['ED_LAST_12_MONTHS'] > 0).mean():.1%}")
print(f"Mean ED visits (12mo): {df['ED_LAST_12_MONTHS'].mean():.2f}")
print(f"Frequent ED users (≥3/year): {df['frequent_ed_user_flag'].mean():.1%}")
print(f"GI-related ED (12mo): {(df['GI_ED_LAST_12_MONTHS'] > 0).mean():.1%}")

# Inpatient utilization
print("\n--- INPATIENT UTILIZATION ---")
print(f"Any hospitalization (12mo): {(df['INP_LAST_12_MONTHS'] > 0).mean():.1%}")
print(f"Mean hospitalizations (12mo): {df['INP_LAST_12_MONTHS'].mean():.3f}")
print(f"Mean inpatient days (12mo): {df['TOTAL_INPATIENT_DAYS_12MO'].mean():.2f}")
print(f"Frequent admissions (≥2/year): {df['frequent_inpatient_flag'].mean():.1%}")
print(f"High inpatient days (≥10/year): {df['high_inpatient_days_flag'].mean():.1%}")

# Outpatient utilization
print("\n--- OUTPATIENT UTILIZATION ---")
print(f"Any outpatient visit (12mo): {(df['OUTPATIENT_VISITS_12MO'] > 0).mean():.1%}")
print(f"Mean outpatient visits (12mo): {df['OUTPATIENT_VISITS_12MO'].mean():.1f}")
print(f"Any PCP visit (12mo): {(df['PCP_VISITS_12MO'] > 0).mean():.1%}")
print(f"Engaged primary care (≥2 PCP/year): {df['engaged_primary_care_flag'].mean():.1%}")
print(f"Primary care continuity ratio: {df['primary_care_continuity_ratio'].mean():.2f}")

# GI-specific utilization
print("\n--- GI-SPECIFIC UTILIZATION ---")
print(f"Any GI specialist visit (12mo): {(df['GI_VISITS_12MO'] > 0).mean():.1%}")
print(f"Mean GI visits when >0: {df[df['GI_VISITS_12MO'] > 0]['GI_VISITS_12MO'].mean():.2f}")
print(f"GI symptom OP visits (12mo): {(df['GI_SYMPTOM_OP_VISITS_12MO'] > 0).mean():.1%}")
print(f"Total GI symptom visits all settings: {(df['total_gi_symptom_visits_12mo'] > 0).mean():.1%}")

# Care patterns
print("\n--- CARE PATTERNS ---")
print(f"No-shows (12mo): {(df['NO_SHOWS_12MO'] > 0).mean():.1%}")
print(f"Mean no-shows when >0: {df[df['NO_SHOWS_12MO'] > 0]['NO_SHOWS_12MO'].mean():.1f}")
print(f"Healthcare intensity score >0: {(df['healthcare_intensity_score'] > 0).mean():.1%}")
print(f"Mean healthcare intensity: {df['healthcare_intensity_score'].mean():.3f}")

print("="*80)

# ========================================
# CELL 11
# ========================================

# Cell 10: Recency patterns analysis
print("RECENCY PATTERNS ANALYSIS")
print("-"*40)

# Handle nulls properly for recency features
print("Days since last encounter (median for those with visits):")
print(f"  ED: {df['days_since_last_ed'].median():.0f} days (n={df['days_since_last_ed'].notna().sum():,})")
print(f"  Inpatient: {df['days_since_last_inpatient'].median():.0f} days (n={df['days_since_last_inpatient'].notna().sum():,})")
print(f"  GI specialist: {df['days_since_last_gi'].median():.0f} days (n={df['days_since_last_gi'].notna().sum():,})")

print("\nRecent healthcare contact:")
print(f"  ED in last 90 days: {df['recent_ed_use_flag'].mean():.1%}")
print(f"  Hospitalization in last 180 days: {df['recent_hospitalization_flag'].mean():.1%}")

# Check for escalating patterns
escalating = df[(df['recent_ed_use_flag'] == 1) & (df['recent_hospitalization_flag'] == 1)]
print(f"\nEscalating acuity (ED→Hospitalization): {len(escalating):,} ({len(escalating)/total_rows:.2%})")

# ========================================
# CELL 12
# ========================================

# Cell 11: Zero inflation analysis (expected for utilization data)
print("ZERO INFLATION ANALYSIS")
print("-"*40)

visit_cols = [col for col in df.columns if col not in ['PAT_ID', 'END_DTTM']]
zero_pcts = (df[visit_cols] == 0).mean().sort_values(ascending=False)

print("Features with >90% zeros (expected for rare events):")
for col in zero_pcts[zero_pcts > 0.90].index[:15]:
    print(f"  {col}: {zero_pcts[col]:.1%} zeros")

print("\nFeatures with <50% zeros (common events):")
for col in zero_pcts[zero_pcts < 0.50].index:
    if 'days_since' not in col and 'ratio' not in col:  # Skip recency and ratio features
        print(f"  {col}: {zero_pcts[col]:.1%} zeros")

# ========================================
# CELL 13
# ========================================

# Cell 12: Utilization patterns analysis
print("UTILIZATION PATTERN ANALYSIS")
print("-"*40)

# High utilizers
high_util = df[df['healthcare_intensity_score'] >= 2]
print(f"High healthcare utilizers (score ≥2): {len(high_util):,} ({len(high_util)/total_rows:.1%})")
if len(high_util) > 0:
    print(f"  Mean ED visits: {high_util['ED_LAST_12_MONTHS'].mean():.1f}")
    print(f"  Mean hospitalizations: {high_util['INP_LAST_12_MONTHS'].mean():.1f}")
    print(f"  Mean outpatient visits: {high_util['OUTPATIENT_VISITS_12MO'].mean():.1f}")

# GI workup intensity
gi_workup = df[(df['GI_VISITS_12MO'] > 0) & (df['total_gi_symptom_visits_12mo'] > 0)]
print(f"\nActive GI workup (specialist + symptoms): {len(gi_workup):,} ({len(gi_workup)/total_rows:.1%})")

# No primary care
no_pcp = df[df['PCP_VISITS_12MO'] == 0]
print(f"\nNo PCP visits (12mo): {len(no_pcp):,} ({len(no_pcp)/total_rows:.1%})")
print(f"  These patients' mean ED visits: {no_pcp['ED_LAST_12_MONTHS'].mean():.2f}")
print(f"  With PCP visits' mean ED visits: {df[df['PCP_VISITS_12MO'] > 0]['ED_LAST_12_MONTHS'].mean():.2f}")

# ========================================
# CELL 14
# ========================================

# Cell 13: Data quality checks
print("DATA QUALITY CHECKS")
print("-"*40)

# Check for outliers
print("Extreme values check:")
outlier_cols = ['ED_LAST_12_MONTHS', 'INP_LAST_12_MONTHS', 'OUTPATIENT_VISITS_12MO', 
                'TOTAL_INPATIENT_DAYS_12MO', 'NO_SHOWS_12MO']

for col in outlier_cols:
    max_val = df[col].max()
    if max_val > df[col].quantile(0.999) * 2:  # Check for extreme outliers
        print(f"  {col}: max={max_val}, 99.9%ile={df[col].quantile(0.999):.0f}")
        extreme_cases = (df[col] > df[col].quantile(0.999) * 2).sum()
        print(f"    Extreme cases (>2x 99.9%ile): {extreme_cases}")

# Check correlations for data consistency
print("\nLogical consistency checks:")
print(f"  ED visits > 0 but GI ED = 0: {((df['ED_LAST_12_MONTHS'] > 0) & (df['GI_ED_LAST_12_MONTHS'] == 0)).mean():.1%}")
print(f"  Inpatient > 0 but days = 0: {((df['INP_LAST_12_MONTHS'] > 0) & (df['TOTAL_INPATIENT_DAYS_12MO'] == 0)).sum()}")
print(f"  GI visits > outpatient visits: {(df['GI_VISITS_12MO'] > df['OUTPATIENT_VISITS_12MO']).sum()}")

print("\n✓ Visit history features validated successfully")

# ========================================
# CELL 15
# ========================================

# Cell 14
df_check = spark.sql(f'''select * from {trgt_cat}.clncl_ds.herald_eda_train_visit_features_final''')
df = df_check.toPandas()
df.isnull().sum()/df.shape[0]

# ========================================
# CELL 16
# ========================================

df.shape

# ========================================
# CELL 17
# ========================================

# Cell 15
df_mean = df.drop(columns=['PAT_ID', 'END_DTTM'])
df_mean.mean()

# ========================================
# CELL 18
# ========================================

# Step 1: Load visit history with outcome and SPLIT column
df_visit = spark.sql(f"""
    SELECT v.*, c.FUTURE_CRC_EVENT, c.SPLIT
    FROM {trgt_cat}.clncl_ds.herald_eda_train_visit_features_final v
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
        ON v.PAT_ID = c.PAT_ID AND v.END_DTTM = c.END_DTTM
""")

# Add visit_ prefix to all feature columns (except PAT_ID, END_DTTM, FUTURE_CRC_EVENT, SPLIT)
feature_cols = [col for col in df_visit.columns
                if col not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT', 'SPLIT']]

for col in feature_cols:
    df_visit = df_visit.withColumnRenamed(col, f"visit_{col.lower()}")

# Ensure continuity ratio is numeric (can be object due to NULL/division edge cases)
if 'visit_primary_care_continuity_ratio' in df_visit.columns:
    df_visit = df_visit.withColumn('visit_primary_care_continuity_ratio',
        F.col('visit_primary_care_continuity_ratio').cast('double'))

# Cache for performance
df_visit.cache()
total_rows = df_visit.count()

# =============================================================================
# CRITICAL: Filter to TRAINING DATA ONLY for feature selection metrics
# This prevents data leakage from validation/test sets into feature selection
# =============================================================================
df_train = df_visit.filter(F.col("SPLIT") == "train")
df_train.cache()

total_train_rows = df_train.count()
baseline_crc_rate = df_train.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"Loaded {total_rows:,} visit history observations (full cohort)")
print(f"Training rows (for feature selection): {total_train_rows:,}")
print(f"Total features: {len(feature_cols)}")
print(f"Baseline CRC rate (train only): {baseline_crc_rate:.4f}")

# ========================================
# CELL 19
# ========================================

# Step 2: Identify binary features (TRAIN DATA ONLY for feature selection)
flag_features = [col for col in df_visit.columns
                if col.endswith('_flag') or col.endswith('_yn')]

risk_metrics = []

print(f"Calculating risk ratios for {len(flag_features)} flag features (using training data only)...")

for feat in flag_features:
    # NOTE: Using df_train to prevent data leakage
    stats = df_train.groupBy(feat).agg(
        F.count('*').alias('count'),
        F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
    ).collect()

    # Parse results
    stats_dict = {row[feat]: {'count': row['count'], 'crc_rate': row['crc_rate']}
                  for row in stats}

    prevalence = stats_dict.get(1, {'count': 0})['count'] / total_train_rows
    rate_with = stats_dict.get(1, {'crc_rate': 0})['crc_rate']
    rate_without = stats_dict.get(0, {'crc_rate': baseline_crc_rate})['crc_rate']
    risk_ratio = rate_with / (rate_without + 1e-10)
    
    # Calculate impact score
    if risk_ratio > 0 and prevalence > 0:
        impact = prevalence * abs(np.log2(max(risk_ratio, 1/(risk_ratio + 1e-10))))
    else:
        impact = 0
    
    risk_metrics.append({
        'feature': feat,
        'prevalence': prevalence,
        'crc_rate_with': rate_with,
        'crc_rate_without': rate_without,
        'risk_ratio': risk_ratio,
        'impact': impact
    })

risk_df = pd.DataFrame(risk_metrics).sort_values('impact', ascending=False)
print("\nTop Risk Ratio Features:")
for _, row in risk_df.iterrows():
    print(f"  {row['feature']}: RR={row['risk_ratio']:.2f}, Prev={row['prevalence']:.1%}, Impact={row['impact']:.4f}")

# ========================================
# CELL 20
# ========================================

# Step 3: Identify count and continuous features
count_features = [col for col in df_visit.columns 
                 if any(x in col.lower() for x in ['_visits', '_days', '_months', 'count', 'score', 'ratio'])
                 and col not in flag_features 
                 and col != 'FUTURE_CRC_EVENT']

days_since_features = [col for col in df_visit.columns if 'days_since' in col.lower()]

# Analyze count distributions
count_stats = []
for feat in count_features:
    stats = df_visit.select(
        F.avg(feat).alias('mean'),
        F.expr(f'percentile_approx({feat}, 0.5)').alias('median'),
        F.expr(f'percentile_approx({feat}, 0.75)').alias('p75'),
        F.expr(f'percentile_approx({feat}, 0.95)').alias('p95'),
        F.max(feat).alias('max'),
        F.sum(F.when(F.col(feat) == 0, 1).otherwise(0)).alias('zero_count'),
        F.corr(feat, 'FUTURE_CRC_EVENT').alias('outcome_corr')
    ).collect()[0]
    
    count_stats.append({
        'feature': feat,
        'mean': stats['mean'],
        'median': stats['median'],
        'zero_pct': stats['zero_count'] / total_rows,
        'p75': stats['p75'],
        'p95': stats['p95'],
        'max': stats['max'],
        'outcome_corr': stats['outcome_corr']
    })

count_df = pd.DataFrame(count_stats).sort_values('outcome_corr', ascending=False, key=abs)
print("\nTop Correlated Count Features:")
for _, row in count_df.iterrows():
    print(f"  {row['feature']}: Corr={row['outcome_corr']:.3f}, Zero%={row['zero_pct']:.1%}")

# Analyze missing patterns in days_since features
missing_stats = []
for feat in days_since_features:
    null_count = df_visit.filter(F.col(feat).isNull()).count()
    missing_stats.append({
        'feature': feat,
        'missing_rate': null_count / total_rows,
        'visit_type': feat.replace('visit_', '').replace('days_since_', '')
    })

missing_df = pd.DataFrame(missing_stats)
print(f"\nVisit types by frequency (from days_since patterns):")
print(missing_df.sort_values('missing_rate').head())

# ========================================
# CELL 21
# ========================================

# Step 4: Create stratified sample (TRAIN DATA ONLY for feature selection)
# NOTE: Using df_train to prevent data leakage from val/test into feature selection
sample_fraction = min(100000 / total_train_rows, 1.0)
df_sample = df_train.sampleBy("FUTURE_CRC_EVENT",
                              fractions={0: sample_fraction, 1: 1.0},
                              seed=42).toPandas()

print(f"\nSampled {len(df_sample):,} rows from TRAINING data for MI calculation")
print(f"Sample CRC rate: {df_sample['FUTURE_CRC_EVENT'].mean():.4f}")

# Prepare features for MI calculation
feature_cols = [c for c in df_sample.columns
               if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT', 'SPLIT']]

# Handle nulls: -999 for days_since (never had), 0 for others
X = df_sample[feature_cols].copy()
for col in X.columns:
    if 'days_since' in col:
        X[col] = X[col].fillna(-999)  # Distinct value for "never"
    else:
        X[col] = X[col].fillna(0)

y = df_sample['FUTURE_CRC_EVENT']

# Calculate MI scores
mi_scores = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop Mutual Information Features:")
for _, row in mi_df.iterrows():
    print(f"  {row['feature']}: MI={row['mi_score']:.4f}")

# ========================================
# CELL 22
# ========================================

# Step 5: Merge all metrics
feature_importance = mi_df.merge(
    risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']], 
    on='feature', 
    how='left'
).merge(
    missing_df[['feature', 'missing_rate']], 
    on='feature', 
    how='left'
)

# Fill NAs for non-flag features
feature_importance['prevalence'] = feature_importance['prevalence'].fillna(
    1 - feature_importance['missing_rate']
)
feature_importance['risk_ratio'] = feature_importance['risk_ratio'].fillna(1.0)
feature_importance['impact'] = feature_importance['impact'].fillna(0)

# Clinical must-keep features
MUST_KEEP = [
    'visit_total_gi_symptom_visits_12mo',  # Direct symptom burden
    'visit_gi_visits_12mo',                 # GI specialist engagement
    'visit_frequent_ed_user_flag',          # Crisis care pattern
    'visit_healthcare_intensity_score',     # Overall complexity
    'visit_gi_ed_last_12_months',          # Acute GI presentations
    'visit_recent_hospitalization_flag',    # Recent severe illness
    'visit_ed_last_12_months'              # ED utilization
]

# Remove low-value features
REMOVE = [col for col in feature_importance['feature'] 
          if 'days_since' in col and feature_importance[
              feature_importance['feature'] == col]['missing_rate'].values[0] > 0.98]

print(f"\nApplying clinical filters:")
print(f"  Must-keep features: {len(MUST_KEEP)}")
print(f"  Removing {len(REMOVE)} very rare features (>98% missing)")

feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]

# ========================================
# CELL 23
# ========================================

# Step 6: Select optimal features

def select_optimal_visit_features(df_importance):
    """Select best representation for each visit type"""
    
    selected = []
    
    # Group features by visit type
    df_importance['visit_type'] = df_importance['feature'].apply(
        lambda x: x.replace('visit_', '').split('_')[0] if 'visit_' in x else 'other'
    )
    
    # High-priority visit types
    high_priority = ['gi', 'ed', 'inp', 'inpatient']
    
    for vtype in df_importance['visit_type'].unique():
        vtype_features = df_importance[df_importance['visit_type'] == vtype]
        
        if vtype in high_priority:
            # Keep top 2 features for high-priority types
            top_features = vtype_features.nlargest(2, 'mi_score')['feature'].tolist()
            selected.extend(top_features)
            
        elif vtype in ['outpatient', 'pcp']:
            # Keep flag if high prevalence, else best MI
            flag_feat = vtype_features[vtype_features['feature'].str.contains('_flag|_visits')]
            if not flag_feat.empty and flag_feat.iloc[0].get('prevalence', 0) > 0.1:
                selected.append(flag_feat.iloc[0]['feature'])
            else:
                selected.append(vtype_features.nlargest(1, 'mi_score')['feature'].values[0])
                
        else:
            # For others, keep best MI feature
            if len(vtype_features) > 0:
                selected.append(vtype_features.nlargest(1, 'mi_score')['feature'].values[0])
    
    # Ensure must-keep features
    for feat in MUST_KEEP:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    # Balance feature types
    flags_selected = [f for f in selected if '_flag' in f]
    counts_selected = [f for f in selected if any(x in f for x in ['_count', '_visits', '_days_12mo'])]
    recency_selected = [f for f in selected if 'days_since' in f]
    
    print(f"\nFeature type balance:")
    print(f"  Flags: {len(flags_selected)}")
    print(f"  Counts/Visits: {len(counts_selected)}")
    print(f"  Recency: {len(recency_selected)}")
    
    return list(set(selected))

selected_features = select_optimal_visit_features(feature_importance)
print(f"\nSelected {len(selected_features)} features after optimization")

# ========================================
# CELL 24
# ========================================

# Step 7: Start with selected features
df_final = df_visit

# Care gap indicators
df_final = df_final.withColumn(
    'visit_gi_symptoms_no_specialist',
    F.when((F.col('visit_total_gi_symptom_visits_12mo') > 0) & 
           (F.col('visit_gi_visits_12mo') == 0), 1).otherwise(0)
)

df_final = df_final.withColumn(
    'visit_frequent_ed_no_pcp',
    F.when((F.col('visit_frequent_ed_user_flag') == 1) & 
           (F.col('visit_pcp_visits_12mo') == 0), 1).otherwise(0)
)

# Acute care reliance score
df_final = df_final.withColumn(
    'visit_acute_care_reliance',
    F.when(F.col('visit_outpatient_visits_12mo') > 0,
           (F.col('visit_ed_last_12_months') + F.col('visit_inp_last_12_months')) / 
           F.col('visit_outpatient_visits_12mo'))
     .otherwise(F.col('visit_ed_last_12_months') + F.col('visit_inp_last_12_months'))
)

# Healthcare complexity categories
df_final = df_final.withColumn(
    'visit_complexity_category',
    F.when(F.col('visit_healthcare_intensity_score') >= 3, 3)  # High
     .when(F.col('visit_healthcare_intensity_score') >= 1, 2)  # Moderate  
     .when((F.col('visit_outpatient_visits_12mo') > 0) | 
           (F.col('visit_ed_last_12_months') > 0), 1)  # Low
     .otherwise(0)  # None
)

# Recent acute care flag
df_final = df_final.withColumn(
    'visit_recent_acute_care',
    F.when((F.col('visit_ed_last_90_days') > 0) | 
           (F.col('visit_recent_hospitalization_flag') == 1), 1).otherwise(0)
)

# Add composite features to selected list
composite_features = [
    'visit_gi_symptoms_no_specialist',
    'visit_frequent_ed_no_pcp', 
    'visit_acute_care_reliance',
    'visit_complexity_category',
    'visit_recent_acute_care'
]
selected_features.extend(composite_features)

print(f"\nAdded {len(composite_features)} composite features")
print(f"Final feature count: {len(set(selected_features))}")

# Print final feature list by category
print("\n" + "="*60)
print("FINAL SELECTED FEATURES")
print("="*60)

selected_features_sorted = sorted(list(set(selected_features)))

# Categorize for display
for category, description in [
    ('gi', 'GI-Specific Features:'),
    ('ed', 'Emergency Department:'),
    ('inp|inpatient', 'Hospitalization:'),
    ('outpatient|pcp', 'Outpatient/Primary Care:'),
    ('intensity|acute|complex', 'Composite/Risk Scores:'),
    ('days_since', 'Recency Features:'),
    ('', 'Other Features:')
]:
    if category:
        cat_features = [f for f in selected_features_sorted if category in f.lower()]
    else:
        cat_features = [f for f in selected_features_sorted 
                       if not any(x in f.lower() for x in 
                       ['gi', 'ed', 'inp', 'outpatient', 'pcp', 'intensity', 
                        'acute', 'complex', 'days_since'])]
    
    if cat_features:
        print(f"\n{description}")
        for feat in cat_features:
            print(f"  - {feat}")

# Select final columns and save
final_columns = ['PAT_ID', 'END_DTTM'] + sorted(list(set(selected_features)))
df_reduced = df_final.select(*final_columns)

# Write to final table
output_table = f'{trgt_cat}.clncl_ds.herald_eda_train_visit_features_reduced'
df_reduced.write.mode('overwrite').option("mergeSchema", "true").saveAsTable(output_table)

# Verify row count preserved
final_count = spark.table(output_table).count()
assert final_count == total_rows, f"Row count mismatch: {final_count} vs {total_rows}"

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: 41")
print(f"Selected features: {len(set(selected_features))}")
print(f"Reduction: {(1 - len(set(selected_features))/41)*100:.1f}%")
print(f"\n✓ Reduced dataset saved to: {output_table}")
print(f"✓ Verified {final_count:,} rows written to table")

# ========================================
# CELL 25
# ========================================

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_eda_train_visit_features_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 26
# ========================================

display(df_check)


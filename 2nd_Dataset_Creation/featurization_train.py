# ===========================================================================
# featurization_train.py
#
# CRC Risk Prediction: Featurization & Training Dataset Pipeline
#
# Builds the full cohort from scratch and engineers 57 selected features
# from Clarity source tables. These features were selected by iterative
# SHAP winnowing in Book 9 (after excluding CEA/CA19-9/FOBT circular
# reasoning features).
#
# Output: {trgt_cat}.clncl_ds.fudgesicle_train
#
# Pipeline stages:
#   1. Configuration & imports
#   2. Base patient identification (outpatient + inpatient encounters)
#   3. Monthly observation grid (one row per patient-month)
#   4. Demographics (age, gender, marital status, race, system tenure)
#   5. PCP status (active primary care provider at observation date)
#   6. Medical exclusions (prior CRC, colectomy, hospice)
#   7. Screening exclusions (VBC registry + internal procedure records)
#   8. Label construction (CRC within 6mo, three-tier negative quality)
#   9. Vitals features (9 features)
#  10. ICD-10 diagnosis features (7 features)
#  11. Lab features (14 features)
#  12. Outpatient medication features (3 features)
#  13. Inpatient medication features (4 features)
#  14. Visit history features (10 features)
#  15. Procedure features (3 features)
#  16. Final join (all features combined, nulls filled)
#  17. Save output table
#  18. Validation & summary
#
# Requires: PySpark (Databricks), numpy, pandas
# ===========================================================================

import os
import datetime
import time
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

# Catalog / schema -- trgt_cat controls where we WRITE intermediate/output tables.
# Source data (Clarity EHR) lives only in prod, so USE CATALOG is always prod.
trgt_cat = os.environ.get('trgt_cat', 'dev')
spark.sql('USE CATALOG prod')
print(f"Read catalog: prod")
print(f"Write catalog: {trgt_cat}")

# Date parameters
# Dynamic cohort window -- recomputed each training run.
# Data has a 1-day processing lag; use end of last fully complete month.
today = datetime.datetime.today()
data_collection_date = today.replace(day=1) - datetime.timedelta(days=1)

label_months = 6                   # Prediction window: CRC within 6 months
min_followup_months = 12           # Minimum follow-up to confirm negatives
total_exclusion_months = max(label_months, min_followup_months)
cohort_months = 24                 # Rolling observation window length

# index_end: latest observation date (must allow full follow-up after it)
# index_start: 24 months before index_end
index_end_dt = data_collection_date - relativedelta(months=total_exclusion_months)
index_start_dt = index_end_dt - relativedelta(months=cohort_months)
index_start = index_start_dt.strftime('%Y-%m-%d')
index_end = index_end_dt.strftime('%Y-%m-%d')

print(f"Training date: {today.strftime('%Y-%m-%d')}")
print(f"Data complete through: {data_collection_date.strftime('%Y-%m-%d')}")
print(f"Cohort window: {index_start} to {index_end} ({cohort_months} months)")

# ICD-10 code pattern for colorectal cancer
# C18 = colon, C19 = rectosigmoid junction, C20 = rectum
crc_icd_regex = r'^(C(?:18|19|20))'

# Output Delta table
OUTPUT_TABLE = f"{trgt_cat}.clncl_ds.fudgesicle_train"

# The 57 features selected by iterative SHAP winnowing in Book 9.
# CEA, CA 19-9, and FOBT/FIT were excluded before selection (circular
# reasoning). See CLAUDE.md for full rationale.
SELECTED_FEATURES = [
    "AGE_GROUP",
    "HAS_PCP_AT_END",
    "IS_FEMALE",
    "IS_MARRIED_PARTNER",
    "RACE_ASIAN",
    "RACE_CAUCASIAN",
    "icd_ANEMIA_FLAG_12MO",
    "icd_BLEED_CNT_12MO",
    "icd_COMBINED_COMORBIDITY_12MO",
    "icd_IRON_DEF_ANEMIA_FLAG_12MO",
    "icd_MALIGNANCY_FLAG_EVER",
    "icd_SYMPTOM_BURDEN_12MO",
    "icd_chronic_gi_pattern",
    "inp_med_inp_gi_hospitalization",
    "inp_med_inp_ibd_meds_recency",
    "inp_med_inp_obstruction_pattern",
    "inp_med_inp_opioid_use_flag",
    "lab_ALBUMIN_DROP_15PCT_FLAG",
    "lab_ALBUMIN_VALUE",
    "lab_ALK_PHOS_VALUE",
    "lab_ANEMIA_GRADE",
    "lab_ANEMIA_SEVERITY_SCORE",
    "lab_AST_VALUE",
    "lab_ESR_VALUE",
    "lab_HEMOGLOBIN_ACCELERATING_DECLINE",
    "lab_HEMOGLOBIN_VALUE",
    "lab_IRON_SATURATION_PCT",
    "lab_PLATELETS_ACCELERATING_RISE",
    "lab_PLATELETS_VALUE",
    "lab_THROMBOCYTOSIS_FLAG",
    "lab_comprehensive_iron_deficiency",
    "months_since_cohort_entry",
    "out_med_broad_abx_recency",
    "out_med_ibd_meds_recency",
    "out_med_ppi_use_flag",
    "proc_blood_transfusion_count_12mo",
    "proc_high_imaging_intensity_flag",
    "proc_mri_abd_pelvis_count_12mo",
    "visit_acute_care_reliance",
    "visit_gi_symptom_op_visits_12mo",
    "visit_healthcare_intensity_score",
    "visit_inp_last_24_months",
    "visit_no_shows_12mo",
    "visit_outpatient_visits_12mo",
    "visit_pcp_visits_12mo",
    "visit_primary_care_continuity_ratio",
    "visit_recency_last_gi",
    "visit_total_gi_symptom_visits_12mo",
    "vit_BMI",
    "vit_BMI_CHANGE_6M",
    "vit_CACHEXIA_RISK_SCORE",
    "vit_MAX_WEIGHT_LOSS_PCT_60D",
    "vit_PULSE",
    "vit_PULSE_PRESSURE",
    "vit_RECENCY_WEIGHT",
    "vit_SBP_VARIABILITY_6M",
    "vit_WEIGHT_TRAJECTORY_SLOPE",
]

print(f"Selected features: {len(SELECTED_FEATURES)}")


# ===========================================================================
# 2. BASE PATIENT IDENTIFICATION
#
# Find all patients with at least one completed encounter (outpatient or
# inpatient) in our integrated health system during the cohort window.
# Outpatient: appointment status Completed (2) or Arrived (6).
# Inpatient: not pre-admit, not canceled, has charges, not a combined account.
# Both: department must be in our integrated system (RPT_GRP_SIX 116001/116002).
# ===========================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW base_patients AS

-- Outpatient encounters
SELECT DISTINCT pe.PAT_ID
FROM clarity_cur.PAT_ENC_ENH pe
JOIN clarity_cur.DEP_LOC_PLOC_SA_ENH dep
  ON dep.department_id = COALESCE(pe.DEPARTMENT_ID, pe.EFFECTIVE_DEPT_ID)
WHERE pe.CONTACT_DATE >= '{index_start}'
  AND pe.CONTACT_DATE < '{index_end}'
  AND pe.APPT_STATUS_C IN (2, 6)
  AND dep.RPT_GRP_SIX IN ('116001', '116002')

UNION

-- Inpatient admissions
SELECT DISTINCT pe.PAT_ID
FROM clarity_cur.PAT_ENC_HSP_HAR_ENH pe
JOIN clarity_cur.DEP_LOC_PLOC_SA_ENH dep
  ON pe.DEPARTMENT_ID = dep.department_id
WHERE DATE(pe.HOSP_ADMSN_TIME) >= '{index_start}'
  AND DATE(pe.HOSP_ADMSN_TIME) < '{index_end}'
  AND pe.ADT_PATIENT_STAT_C <> 1
  AND pe.ADMIT_CONF_STAT_C <> 3
  AND dep.RPT_GRP_SIX IN ('116001', '116002')
  AND pe.TOT_CHGS <> 0
  AND COALESCE(pe.acct_billsts_ha_c, -1) NOT IN (40, 60, 99)
  AND pe.combine_acct_id IS NULL
""")

patient_count = spark.sql("SELECT COUNT(*) AS n FROM base_patients").collect()[0]['n']
print(f"Base patients identified: {patient_count:,}")


# ===========================================================================
# 3. MONTHLY OBSERVATION GRID
#
# Create one observation per patient per month. The observation day within
# each month is assigned deterministically using a hash of (PAT_ID, month),
# so the same patient always gets the same day in the same month across runs.
# This avoids bias from always using the 1st or 15th.
# ===========================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW observation_grid AS

WITH months AS (
  SELECT explode(
    sequence(
      date_trunc('month', DATE('{index_start}')),
      date_trunc('month', DATE('{index_end}')),
      interval 1 month
    )
  ) AS month_start
),

pat_month AS (
  SELECT
    bp.PAT_ID,
    m.month_start,
    day(last_day(m.month_start)) AS dim,
    pmod(
      abs(hash(concat(CAST(bp.PAT_ID AS STRING), '|', CAST(m.month_start AS STRING)))),
      day(last_day(m.month_start))
    ) + 1 AS rnd_day
  FROM base_patients bp
  CROSS JOIN months m
)

SELECT
  PAT_ID,
  date_add(month_start, rnd_day - 1) AS END_DTTM
FROM pat_month
WHERE date_add(month_start, rnd_day - 1) >= DATE('{index_start}')
  AND date_add(month_start, rnd_day - 1) <= DATE('{index_end}')
""")

grid_count = spark.sql("SELECT COUNT(*) AS n FROM observation_grid").collect()[0]['n']
print(f"Observation grid rows: {grid_count:,}")


# ===========================================================================
# 4. DEMOGRAPHICS
#
# Join patient demographics, compute age at observation date, and apply:
#   - Age filter: 45-100 (matches USPSTF CRC screening guidelines)
#   - System tenure: >= 24 months of encounters in Mercy's EHR
#   - Data quality: plausible age, first-seen before observation date,
#     tenure not longer than lifetime
# Also computes AGE_GROUP (ordinal 0-5) and RACE one-hot flags.
# ===========================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW cohort_demographics AS

WITH first_seen AS (
  SELECT PAT_ID, MIN(first_dt) AS first_seen_dt
  FROM (
    SELECT pe.PAT_ID, CAST(pe.CONTACT_DATE AS DATE) AS first_dt
    FROM clarity_cur.PAT_ENC_ENH pe
    UNION ALL
    SELECT ha.PAT_ID, CAST(ha.HOSP_ADMSN_TIME AS DATE) AS first_dt
    FROM clarity_cur.PAT_ENC_HSP_HAR_ENH ha
  ) z
  GROUP BY PAT_ID
)

SELECT
  idx.PAT_ID,
  idx.END_DTTM,
  FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) AS AGE,
  -- AGE_GROUP: ordinal 0-5 (matches Book 0)
  CASE
    WHEN FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) < 50 THEN 1
    WHEN FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) < 55 THEN 2
    WHEN FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) < 65 THEN 3
    WHEN FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) < 75 THEN 4
    ELSE 5
  END AS AGE_GROUP,
  CASE WHEN p.GENDER = 'Female' THEN 1 ELSE 0 END AS IS_FEMALE,
  CASE WHEN p.MARITAL_STATUS IN ('Married', 'Significant other') THEN 1 ELSE 0 END AS IS_MARRIED_PARTNER,
  -- Race one-hot flags (from RACE_BUCKETS in PATIENT_ENH)
  CASE WHEN COALESCE(p.RACE_BUCKETS, 'Unknown') = 'Asian' THEN 1 ELSE 0 END AS RACE_ASIAN,
  CASE WHEN COALESCE(p.RACE_BUCKETS, 'Unknown') = 'Caucasian' THEN 1 ELSE 0 END AS RACE_CAUCASIAN,
  CAST(months_between(idx.END_DTTM, fs.first_seen_dt) AS INT) AS OBS_MONTHS_PRIOR,
  fs.first_seen_dt,
  -- Data quality flag: catches impossible records (age out of range,
  -- first-seen after observation, tenure exceeding lifetime)
  CASE
    WHEN FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) > 100 THEN 0
    WHEN FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) < 0 THEN 0
    WHEN CAST(months_between(idx.END_DTTM, fs.first_seen_dt) AS INT) >
         FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) * 12 THEN 0
    WHEN fs.first_seen_dt > idx.END_DTTM THEN 0
    ELSE 1
  END AS data_quality_flag
FROM observation_grid idx
LEFT JOIN clarity_cur.PATIENT_ENH p ON idx.PAT_ID = p.PAT_ID
LEFT JOIN first_seen fs ON idx.PAT_ID = fs.PAT_ID
WHERE FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) >= 45
  AND FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) <= 100
  AND CAST(months_between(idx.END_DTTM, fs.first_seen_dt) AS INT) >= 24
  AND CASE
        WHEN FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) > 100 THEN 0
        WHEN FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) < 0 THEN 0
        WHEN CAST(months_between(idx.END_DTTM, fs.first_seen_dt) AS INT) >
             FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) * 12 THEN 0
        WHEN fs.first_seen_dt > idx.END_DTTM THEN 0
        ELSE 1
      END = 1
""")

demo_count = spark.sql("SELECT COUNT(*) AS n FROM cohort_demographics").collect()[0]['n']
print(f"After demographics + quality filters: {demo_count:,}")


# ===========================================================================
# 5. PCP STATUS
#
# Determine if each patient has an active Primary Care Provider within
# Mercy's integrated system at their observation date. Checked by joining
# against pat_pcp with effective/termination date ranges, restricted to
# providers flagged as Integrated or Integrated-Regional.
#
# PCP status is both a model feature and a label quality indicator
# (used in the three-tier negative label system).
# ===========================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW cohort_with_pcp AS

SELECT
  c.*,
  CASE WHEN pcp.PAT_ID IS NOT NULL THEN 1 ELSE 0 END AS HAS_PCP_AT_END
FROM cohort_demographics c
LEFT JOIN (
  SELECT DISTINCT p.PAT_ID, c2.END_DTTM
  FROM cohort_demographics c2
  JOIN clarity.pat_pcp p
    ON p.PAT_ID = c2.PAT_ID
    AND c2.END_DTTM BETWEEN p.EFF_DATE AND COALESCE(p.TERM_DATE, '9999-12-31')
  JOIN clarity_cur.clarity_ser_enh ser
    ON p.PCP_PROV_ID = ser.prov_id
    AND ser.RPT_GRP_ELEVEN_NAME IN ('Integrated-Regional', 'Integrated')
) pcp ON c.PAT_ID = pcp.PAT_ID AND c.END_DTTM = pcp.END_DTTM
""")

pcp_count = spark.sql("SELECT SUM(HAS_PCP_AT_END) AS n FROM cohort_with_pcp").collect()[0]['n']
print(f"Patients with PCP: {pcp_count:,}")


# ===========================================================================
# 6. MEDICAL EXCLUSIONS
#
# Remove patient-observations where the patient has already been diagnosed
# with or treated for CRC before the observation date. These patients are
# not screening candidates.
#
# Excluded conditions:
#   - Prior CRC diagnosis (C18, C19, C20)
#   - Prior colectomy (Z90.49)
#   - Colostomy complications (K91.850) -- implies prior surgical intervention
#   - Hospice/palliative care (Z51.5x) -- screening inappropriate
# ===========================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW medical_exclusions AS

SELECT DISTINCT c.PAT_ID, c.END_DTTM
FROM cohort_with_pcp c
JOIN clarity_cur.pat_enc_enh pe
  ON pe.PAT_ID = c.PAT_ID
  AND DATE(pe.CONTACT_DATE) <= c.END_DTTM
JOIN clarity_cur.pat_enc_dx_enh dd
  ON dd.PAT_ENC_CSN_ID = pe.PAT_ENC_CSN_ID
WHERE dd.ICD10_CODE RLIKE '{crc_icd_regex}'
   OR dd.ICD10_CODE IN ('Z90.49', 'K91.850')
   OR dd.ICD10_CODE LIKE 'Z51.5%'
""")

excl_count = spark.sql("SELECT COUNT(*) AS n FROM medical_exclusions").collect()[0]['n']
print(f"Medical exclusion rows: {excl_count:,}")

# Apply exclusions via anti-join
spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_no_exclusions AS

SELECT c.*
FROM cohort_with_pcp c
LEFT ANTI JOIN medical_exclusions e
  ON c.PAT_ID = e.PAT_ID AND c.END_DTTM = e.END_DTTM
""")

after_excl = spark.sql("SELECT COUNT(*) AS n FROM cohort_no_exclusions").collect()[0]['n']
print(f"After medical exclusions: {after_excl:,}")


# ===========================================================================
# 7. SCREENING EXCLUSIONS (DUAL SOURCE)
#
# The model targets UNSCREENED patients. We use two independent data sources:
#
# Source 1 (VBC registry): Administrative table with current screening status.
#   Not timestamped, so this is a patient-level exclusion -- if the VBC table
#   says a patient is currently screened, ALL observations are excluded.
#   This is intentionally over-broad (see cohort_creation_explained.md).
#
# Source 2 (Internal procedures): ORDER_PROC_ENH from July 2021 onward.
#   Each screening type checked against its own validity window:
#     - Colonoscopy: 10 years
#     - CT Colonography: 5 years
#     - Flexible Sigmoidoscopy: 5 years
#     - FIT-DNA (Cologuard): 3 years
#     - FOBT/FIT: 1 year
#
# A patient-observation is excluded if EITHER source indicates screening.
# ===========================================================================

# Internal screening detection -- per-modality validity check
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW internal_screening AS

SELECT
  c.PAT_ID,
  c.END_DTTM,
  MAX(DATE(op.ORDERING_DATE)) AS last_screening_date,
  MAX(
    CASE
      WHEN op.PROC_CODE IN ('45378','45380','45381','45382','45384','45385','45386',
                            '45388','45389','45390','45391','45392','45393','45398')
           OR LOWER(op.PROC_NAME) LIKE '%colonoscopy%'
        THEN 10
      WHEN op.PROC_CODE IN ('74261','74262','74263')
           OR LOWER(op.PROC_NAME) LIKE '%ct colonography%'
           OR LOWER(op.PROC_NAME) LIKE '%virtual colonoscopy%'
        THEN 5
      WHEN op.PROC_CODE IN ('45330','45331','45332','45333','45334','45335','45337',
                            '45338','45339','45340','45341','45342','45345','45346',
                            '45347','45349','45350')
           OR LOWER(op.PROC_NAME) LIKE '%sigmoidoscopy%'
        THEN 5
      WHEN op.PROC_CODE IN ('81528')
           OR LOWER(op.PROC_NAME) LIKE '%cologuard%'
           OR LOWER(op.PROC_NAME) LIKE '%fit-dna%'
        THEN 3
      WHEN op.PROC_CODE IN ('82270','82274','G0328')
           OR LOWER(op.PROC_NAME) LIKE '%fobt%'
           OR LOWER(op.PROC_NAME) LIKE '%fecal occult%'
           OR (LOWER(op.PROC_NAME) LIKE '%fit%' AND LOWER(op.PROC_NAME) LIKE '%test%')
        THEN 1
      ELSE NULL
    END
  ) AS max_valid_years
FROM cohort_no_exclusions c
JOIN clarity_cur.ORDER_PROC_ENH op
  ON op.PAT_ID = c.PAT_ID
  AND DATE(op.ORDERING_DATE) <= c.END_DTTM
  AND DATE(op.ORDERING_DATE) >= DATE('2021-07-01')
WHERE op.RPT_GRP_SIX IN ('116001', '116002')
  AND op.ORDER_STATUS NOT IN ('Canceled', 'Cancelled')
  AND (
    op.PROC_CODE IN ('45378','45380','45381','45382','45384','45385','45386',
                     '45388','45389','45390','45391','45392','45393','45398',
                     '74261','74262','74263',
                     '45330','45331','45332','45333','45334','45335','45337',
                     '45338','45339','45340','45341','45342','45345','45346',
                     '45347','45349','45350',
                     '81528','82270','82274','G0328')
    OR LOWER(op.PROC_NAME) LIKE '%colonoscopy%'
    OR LOWER(op.PROC_NAME) LIKE '%ct colonography%'
    OR LOWER(op.PROC_NAME) LIKE '%virtual colonoscopy%'
    OR LOWER(op.PROC_NAME) LIKE '%sigmoidoscopy%'
    OR LOWER(op.PROC_NAME) LIKE '%cologuard%'
    OR LOWER(op.PROC_NAME) LIKE '%fit-dna%'
    OR LOWER(op.PROC_NAME) LIKE '%fobt%'
    OR LOWER(op.PROC_NAME) LIKE '%fecal occult%'
  )
GROUP BY c.PAT_ID, c.END_DTTM
-- Only keep if at least one modality has a valid (non-expired) screening
HAVING MAX(
  CASE
    WHEN DATE(op.ORDERING_DATE) > DATEADD(YEAR, -10, c.END_DTTM)
     AND (op.PROC_CODE IN ('45378','45380','45381','45382','45384','45385','45386',
                           '45388','45389','45390','45391','45392','45393','45398')
          OR LOWER(op.PROC_NAME) LIKE '%colonoscopy%') THEN 1
    WHEN DATE(op.ORDERING_DATE) > DATEADD(YEAR, -5, c.END_DTTM)
     AND (op.PROC_CODE IN ('74261','74262','74263')
          OR LOWER(op.PROC_NAME) LIKE '%ct colonography%'
          OR LOWER(op.PROC_NAME) LIKE '%virtual colonoscopy%'
          OR LOWER(op.PROC_NAME) LIKE '%sigmoidoscopy%') THEN 1
    WHEN DATE(op.ORDERING_DATE) > DATEADD(YEAR, -3, c.END_DTTM)
     AND (op.PROC_CODE IN ('81528')
          OR LOWER(op.PROC_NAME) LIKE '%cologuard%'
          OR LOWER(op.PROC_NAME) LIKE '%fit-dna%') THEN 1
    WHEN DATE(op.ORDERING_DATE) > DATEADD(YEAR, -1, c.END_DTTM)
     AND (op.PROC_CODE IN ('82270','82274','G0328')
          OR LOWER(op.PROC_NAME) LIKE '%fobt%'
          OR LOWER(op.PROC_NAME) LIKE '%fecal occult%') THEN 1
    ELSE 0
  END
) = 1
""")

# Apply dual screening exclusion:
# Anti-join removes internal screening matches; WHERE clause removes VBC matches
spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_unscreened AS

SELECT c.*
FROM cohort_no_exclusions c
-- Exclude if internal screening is current
LEFT ANTI JOIN internal_screening ise
  ON c.PAT_ID = ise.PAT_ID AND c.END_DTTM = ise.END_DTTM
-- Exclude if VBC says screened (patient-level, not point-in-time)
WHERE c.PAT_ID NOT IN (
  SELECT PAT_ID
  FROM prod.clncl_cur.vbc_colon_cancer_screen
  WHERE COLON_SCREEN_MET_FLAG = 'Y'
    AND COLON_SCREEN_EXCL_FLAG = 'N'
)
""")

unscreened_count = spark.sql("SELECT COUNT(*) AS n FROM cohort_unscreened").collect()[0]['n']
print(f"After screening exclusions: {unscreened_count:,}")


# ===========================================================================
# 8. LABEL CONSTRUCTION
#
# For each remaining patient-observation, determine:
#   FUTURE_CRC_EVENT: Was the patient diagnosed with CRC (C18/C19/C20)
#     within the next 6 months?
#
# For negatives (no CRC seen), we assign label quality via a three-tier system:
#   Tier 1: Patient returned AFTER the 6-month window (months 7-12) -- high confidence
#   Tier 2: Patient returned in months 4-6 AND has active PCP -- medium confidence
#   Tier 3: No return visit BUT has active PCP -- lower confidence
#   Excluded: No return AND no PCP -- too uncertain, dropped from training
#
# LABEL_USABLE = 1 means the observation is suitable for training.
# ===========================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW cohort_labeled AS

WITH future_crc AS (
  -- Search for CRC diagnosis in the 6-month window after observation date
  SELECT DISTINCT
    c.PAT_ID,
    c.END_DTTM,
    FIRST_VALUE(dd.ICD10_CODE) OVER (
      PARTITION BY c.PAT_ID, c.END_DTTM
      ORDER BY pe.CONTACT_DATE, dd.ICD10_CODE
    ) AS ICD10_CODE
  FROM cohort_unscreened c
  JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID
    AND DATE(pe.CONTACT_DATE) > c.END_DTTM
    AND DATE(pe.CONTACT_DATE) <= DATEADD(MONTH, {label_months}, c.END_DTTM)
  JOIN clarity_cur.pat_enc_dx_enh dd
    ON dd.PAT_ENC_CSN_ID = pe.PAT_ENC_CSN_ID
    AND dd.ICD10_CODE RLIKE '{crc_icd_regex}'
),

next_contact AS (
  -- Find earliest return visit within 12 months (for tier assignment)
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    MIN(pe.CONTACT_DATE) AS next_visit_date
  FROM cohort_unscreened c
  JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID
    AND DATE(pe.CONTACT_DATE) > c.END_DTTM
    AND DATE(pe.CONTACT_DATE) <= DATEADD(MONTH, {min_followup_months}, c.END_DTTM)
    AND pe.APPT_STATUS_C IN (2, 6)
  GROUP BY c.PAT_ID, c.END_DTTM
),

patient_first_obs AS (
  -- First observation date per patient (for months_since_cohort_entry)
  SELECT PAT_ID, MIN(END_DTTM) AS first_obs_date
  FROM cohort_unscreened
  GROUP BY PAT_ID
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  c.AGE,
  c.IS_FEMALE,
  c.IS_MARRIED_PARTNER,
  c.HAS_PCP_AT_END,
  c.first_seen_dt,

  -- Binary label: 1 = CRC diagnosed within 6 months
  CASE WHEN fc.PAT_ID IS NOT NULL THEN 1 ELSE 0 END AS FUTURE_CRC_EVENT,

  -- Cancer subtype (for stratified splitting)
  CASE
    WHEN fc.ICD10_CODE RLIKE '^C18' THEN 'C18'
    WHEN fc.ICD10_CODE RLIKE '^C19' THEN 'C19'
    WHEN fc.ICD10_CODE RLIKE '^C20' THEN 'C20'
    ELSE NULL
  END AS ICD10_GROUP,

  -- Months since this patient's first observation in the cohort
  CAST(months_between(c.END_DTTM, pfo.first_obs_date) AS INT) AS months_since_cohort_entry,

  -- Label usability: three-tier system for negative label quality
  -- Positives are always usable. Negatives need follow-up confirmation.
  CASE
    -- Positive: always usable
    WHEN fc.PAT_ID IS NOT NULL THEN 1
    -- Tier 1: return visit after the 6-month window (high confidence negative)
    WHEN fc.PAT_ID IS NULL AND nc.next_visit_date > DATEADD(MONTH, {label_months}, c.END_DTTM) THEN 1
    -- Tier 2: return in months 4-6 AND has PCP (medium confidence)
    WHEN fc.PAT_ID IS NULL AND c.HAS_PCP_AT_END = 1
     AND nc.next_visit_date > DATEADD(MONTH, 4, c.END_DTTM)
     AND nc.next_visit_date <= DATEADD(MONTH, {label_months}, c.END_DTTM) THEN 1
    -- Tier 3: no return but has PCP (lower confidence, but PCP would be notified)
    WHEN fc.PAT_ID IS NULL AND c.HAS_PCP_AT_END = 1 AND nc.next_visit_date IS NULL THEN 1
    -- Excluded: no return AND no PCP -- cannot confirm negative
    ELSE 0
  END AS LABEL_USABLE

FROM cohort_unscreened c
LEFT JOIN future_crc fc ON c.PAT_ID = fc.PAT_ID AND c.END_DTTM = fc.END_DTTM
LEFT JOIN next_contact nc ON c.PAT_ID = nc.PAT_ID AND c.END_DTTM = nc.END_DTTM
LEFT JOIN patient_first_obs pfo ON c.PAT_ID = pfo.PAT_ID
""")

# Keep only observations with reliable labels
spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_usable AS
SELECT * FROM cohort_labeled WHERE LABEL_USABLE = 1
""")

usable_count = spark.sql("SELECT COUNT(*) AS n FROM cohort_usable").collect()[0]['n']
pos_count = spark.sql("SELECT SUM(FUTURE_CRC_EVENT) AS n FROM cohort_usable").collect()[0]['n']
print(f"Usable observations: {usable_count:,}")
print(f"Positive cases: {pos_count:,} ({pos_count/usable_count*100:.2f}%)")


# Create cohort_base: the foundation view that all feature queries join against
spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_base AS
SELECT
  PAT_ID,
  END_DTTM,
  AGE_GROUP,
  IS_FEMALE,
  IS_MARRIED_PARTNER,
  RACE_ASIAN,
  RACE_CAUCASIAN,
  HAS_PCP_AT_END,
  months_since_cohort_entry,
  FUTURE_CRC_EVENT,
  ICD10_GROUP
FROM cohort_usable
""")

base_count = spark.sql("SELECT COUNT(*) AS n FROM cohort_base").collect()[0]['n']
print(f"Cohort base rows: {base_count:,}")


# ===========================================================================
# 9. VITALS FEATURES (9 features)
#
# Source: pat_enc_enh (outpatient encounters)
# Lookback: 12 months before observation date
#
# Features:
#   vit_BMI                      - Latest Body Mass Index (kg/m2)
#   vit_BMI_CHANGE_6M            - BMI change over 6 months
#   vit_PULSE                    - Latest heart rate (bpm)
#   vit_PULSE_PRESSURE           - SBP - DBP (mmHg)
#   vit_RECENCY_WEIGHT           - Days since last weight measurement
#   vit_CACHEXIA_RISK_SCORE      - Wasting risk (0=none, 1=moderate, 2=high)
#   vit_WEIGHT_TRAJECTORY_SLOPE  - Linear regression slope of weight over time
#   vit_MAX_WEIGHT_LOSS_PCT_60D  - Maximum % weight loss in any 60-day window
#   vit_SBP_VARIABILITY_6M       - Systolic BP standard deviation (6 months)
# ===========================================================================

# Step 1: Extract raw vitals with plausibility filters
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW vitals_raw AS

SELECT
  c.PAT_ID,
  c.END_DTTM,
  DATE(pe.CONTACT_DATE) AS MEAS_DATE,
  DATEDIFF(c.END_DTTM, DATE(pe.CONTACT_DATE)) AS DAYS_BEFORE_END,
  CASE WHEN CAST(pe.BP_SYSTOLIC AS DOUBLE) BETWEEN 60 AND 280
       THEN CAST(pe.BP_SYSTOLIC AS DOUBLE) END AS BP_SYSTOLIC,
  CASE WHEN CAST(pe.BP_DIASTOLIC AS DOUBLE) BETWEEN 30 AND 180
       THEN CAST(pe.BP_DIASTOLIC AS DOUBLE) END AS BP_DIASTOLIC,
  CASE WHEN CAST(pe.PULSE AS DOUBLE) BETWEEN 30 AND 220
       THEN CAST(pe.PULSE AS DOUBLE) END AS PULSE,
  CASE WHEN CAST(pe.WEIGHT AS DOUBLE) / 16.0 BETWEEN 50 AND 800
       THEN CAST(pe.WEIGHT AS DOUBLE) END AS WEIGHT_OZ,
  CASE WHEN CAST(pe.BMI AS DOUBLE) BETWEEN 10 AND 80
       THEN CAST(pe.BMI AS DOUBLE) END AS BMI
FROM cohort_base c
JOIN clarity_cur.pat_enc_enh pe
  ON pe.PAT_ID = c.PAT_ID
  AND DATE(pe.CONTACT_DATE) < c.END_DTTM
  AND DATE(pe.CONTACT_DATE) >= DATE_SUB(c.END_DTTM, 365)
  AND DATE(pe.CONTACT_DATE) >= DATE('2021-07-01')
  AND pe.APPT_STATUS_C IN (2, 6)
""")

print("Raw vitals extracted")

# Step 2: Compute all 9 vitals features
spark.sql("""
CREATE OR REPLACE TEMP VIEW vitals_features AS

WITH
-- Most recent vitals
latest_vitals AS (
  SELECT PAT_ID, END_DTTM, BP_SYSTOLIC, BP_DIASTOLIC, PULSE, BMI,
         WEIGHT_OZ, MEAS_DATE
  FROM (
    SELECT PAT_ID, END_DTTM, BP_SYSTOLIC, BP_DIASTOLIC, PULSE, BMI,
           WEIGHT_OZ, MEAS_DATE,
      ROW_NUMBER() OVER (PARTITION BY PAT_ID, END_DTTM ORDER BY MEAS_DATE DESC) AS rn
    FROM vitals_raw
    WHERE (BP_SYSTOLIC IS NOT NULL OR WEIGHT_OZ IS NOT NULL OR BMI IS NOT NULL)
  ) t WHERE rn = 1
),

-- BMI approximately 6 months ago (closest reading in 150-210 day window)
bmi_6m AS (
  SELECT PAT_ID, END_DTTM, BMI AS BMI_6M
  FROM (
    SELECT PAT_ID, END_DTTM, BMI,
      ROW_NUMBER() OVER (
        PARTITION BY PAT_ID, END_DTTM
        ORDER BY ABS(DAYS_BEFORE_END - 180)
      ) AS rn
    FROM vitals_raw
    WHERE BMI IS NOT NULL
      AND DAYS_BEFORE_END BETWEEN 150 AND 210
  ) t WHERE rn = 1
),

-- Weight trajectory: linear regression slope across all 12-month readings
weight_trajectory AS (
  SELECT
    PAT_ID, END_DTTM,
    REGR_SLOPE(WEIGHT_OZ, DAYS_BEFORE_END) AS WEIGHT_TRAJECTORY_SLOPE
  FROM vitals_raw
  WHERE WEIGHT_OZ IS NOT NULL
  GROUP BY PAT_ID, END_DTTM
  HAVING COUNT(*) >= 2
),

-- Maximum weight loss in any consecutive-measurement 60-day window
weight_changes AS (
  SELECT PAT_ID, END_DTTM, WEIGHT_OZ, MEAS_DATE, DAYS_BEFORE_END,
    LAG(WEIGHT_OZ) OVER (PARTITION BY PAT_ID, END_DTTM ORDER BY MEAS_DATE) AS PREV_WEIGHT_OZ,
    LAG(MEAS_DATE) OVER (PARTITION BY PAT_ID, END_DTTM ORDER BY MEAS_DATE) AS PREV_MEAS_DATE
  FROM vitals_raw
  WHERE WEIGHT_OZ IS NOT NULL
),

max_weight_loss AS (
  SELECT PAT_ID, END_DTTM,
    MAX(CASE
      WHEN PREV_WEIGHT_OZ IS NOT NULL AND PREV_WEIGHT_OZ > 0
       AND DATEDIFF(MEAS_DATE, PREV_MEAS_DATE) <= 60
      THEN ((PREV_WEIGHT_OZ - WEIGHT_OZ) / PREV_WEIGHT_OZ) * 100
    END) AS MAX_WEIGHT_LOSS_PCT_60D
  FROM weight_changes
  GROUP BY PAT_ID, END_DTTM
),

-- Systolic BP variability (standard deviation over 6 months)
bp_variability AS (
  SELECT PAT_ID, END_DTTM, STDDEV(BP_SYSTOLIC) AS SBP_VARIABILITY_6M
  FROM vitals_raw
  WHERE BP_SYSTOLIC IS NOT NULL AND DAYS_BEFORE_END <= 180
  GROUP BY PAT_ID, END_DTTM
  HAVING COUNT(*) >= 2
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  ROUND(lv.BMI, 1) AS vit_BMI,
  ROUND(lv.BMI - b6.BMI_6M, 2) AS vit_BMI_CHANGE_6M,
  ROUND(lv.PULSE, 0) AS vit_PULSE,
  ROUND(lv.BP_SYSTOLIC - lv.BP_DIASTOLIC, 0) AS vit_PULSE_PRESSURE,
  DATEDIFF(c.END_DTTM, lv.MEAS_DATE) AS vit_RECENCY_WEIGHT,
  -- Cachexia risk: ordinal 0-2
  CASE
    WHEN lv.BMI < 18.5 AND COALESCE(mwl.MAX_WEIGHT_LOSS_PCT_60D, 0) > 10 THEN 2
    WHEN lv.BMI < 20 OR COALESCE(mwl.MAX_WEIGHT_LOSS_PCT_60D, 0) > 5 THEN 1
    ELSE 0
  END AS vit_CACHEXIA_RISK_SCORE,
  ROUND(wt.WEIGHT_TRAJECTORY_SLOPE, 4) AS vit_WEIGHT_TRAJECTORY_SLOPE,
  ROUND(mwl.MAX_WEIGHT_LOSS_PCT_60D, 2) AS vit_MAX_WEIGHT_LOSS_PCT_60D,
  ROUND(bv.SBP_VARIABILITY_6M, 2) AS vit_SBP_VARIABILITY_6M

FROM cohort_base c
LEFT JOIN latest_vitals lv ON c.PAT_ID = lv.PAT_ID AND c.END_DTTM = lv.END_DTTM
LEFT JOIN bmi_6m b6 ON c.PAT_ID = b6.PAT_ID AND c.END_DTTM = b6.END_DTTM
LEFT JOIN weight_trajectory wt ON c.PAT_ID = wt.PAT_ID AND c.END_DTTM = wt.END_DTTM
LEFT JOIN max_weight_loss mwl ON c.PAT_ID = mwl.PAT_ID AND c.END_DTTM = mwl.END_DTTM
LEFT JOIN bp_variability bv ON c.PAT_ID = bv.PAT_ID AND c.END_DTTM = bv.END_DTTM
""")

print("Vitals features computed (9 features)")


# ===========================================================================
# 10. ICD-10 DIAGNOSIS FEATURES (7 features)
#
# Source: pat_enc_dx_enh (outpatient) + hsp_acct_dx_list_enh (inpatient)
# Lookback: 12 months for most features; "ever" for malignancy flag
#
# Features:
#   icd_MALIGNANCY_FLAG_EVER        - Prior non-CRC malignancy (Z85)
#   icd_IRON_DEF_ANEMIA_FLAG_12MO   - Iron deficiency anemia (D50) in 12mo
#   icd_ANEMIA_FLAG_12MO            - Any anemia (D50-D64) in 12mo
#   icd_BLEED_CNT_12MO              - GI bleeding encounter count in 12mo
#   icd_SYMPTOM_BURDEN_12MO         - Sum of 6 binary symptom flags in 12mo
#   icd_COMBINED_COMORBIDITY_12MO   - Charlson Comorbidity Index (12mo)
#   icd_chronic_gi_pattern          - Chronic GI condition (IBD/diverticular/complexity)
# ===========================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW icd_features AS

WITH all_dx AS (
  -- Outpatient diagnoses
  SELECT c.PAT_ID, c.END_DTTM, dd.ICD10_CODE AS CODE, DATE(pe.CONTACT_DATE) AS DX_DATE
  FROM cohort_base c
  JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID
    AND DATE(pe.CONTACT_DATE) < c.END_DTTM
    AND DATE(pe.CONTACT_DATE) >= DATE('2021-07-01')
    AND pe.APPT_STATUS_C IN (2, 6)
  JOIN clarity_cur.pat_enc_dx_enh dd
    ON dd.PAT_ENC_CSN_ID = pe.PAT_ENC_CSN_ID
  WHERE dd.ICD10_CODE IS NOT NULL

  UNION ALL

  -- Inpatient diagnoses
  SELECT c.PAT_ID, c.END_DTTM, dd.CODE, DATE(hsp.HOSP_ADMSN_TIME) AS DX_DATE
  FROM cohort_base c
  JOIN clarity_cur.PAT_ENC_HSP_HAR_ENH hsp
    ON hsp.PAT_ID = c.PAT_ID
    AND DATE(hsp.HOSP_ADMSN_TIME) < c.END_DTTM
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE('2021-07-01')
    AND hsp.ADT_PATIENT_STAT_C <> 1
    AND hsp.ADMIT_CONF_STAT_C <> 3
  JOIN clarity_cur.hsp_acct_dx_list_enh dd
    ON dd.PRIM_ENC_CSN_ID = hsp.PAT_ENC_CSN_ID
  WHERE dd.CODE IS NOT NULL
),

-- Charlson Comorbidity Index (renamed to COMBINED_COMORBIDITY to match Book 9)
charlson AS (
  SELECT PAT_ID, END_DTTM, SUM(charlson_wt) AS COMBINED_COMORBIDITY_12MO
  FROM (
    SELECT DISTINCT PAT_ID, END_DTTM,
      CASE
        WHEN CODE RLIKE '^(I21|I22)' THEN 1           -- MI
        WHEN CODE RLIKE '^I50' THEN 1                  -- CHF
        WHEN CODE RLIKE '^(I70|I71|I73)' THEN 1        -- PVD
        WHEN CODE RLIKE '^(I60|I61|I62|I63|I64)' THEN 1 -- CVD
        WHEN CODE RLIKE '^(G30|F01|F03)' THEN 1        -- Dementia
        WHEN CODE RLIKE '^J44' THEN 1                  -- COPD
        WHEN CODE RLIKE '^(M05|M06|M32|M33|M34)' THEN 1 -- Rheumatic
        WHEN CODE RLIKE '^(K25|K26|K27|K28)' THEN 1    -- Peptic ulcer
        WHEN CODE RLIKE '^K70' THEN 1                  -- Mild liver
        WHEN CODE RLIKE '^(E10|E11)' THEN 1            -- Diabetes
        WHEN CODE RLIKE '^(E13|E14)' THEN 2            -- Diabetes w/ complications
        WHEN CODE RLIKE '^(G81|G82)' THEN 2            -- Hemiplegia/paraplegia
        WHEN CODE RLIKE '^N18' THEN 2                  -- Renal disease
        WHEN CODE RLIKE '^C(?:0[0-9]|[1-8][0-9]|9[0-7])' THEN 2 -- Any malignancy
        WHEN CODE RLIKE '^(K72|K76)' THEN 3            -- Severe liver
        WHEN CODE RLIKE '^(C78|C79)' THEN 6            -- Metastatic cancer
        WHEN CODE RLIKE '^B2[0-4]' THEN 6              -- AIDS
      END AS charlson_wt
    FROM all_dx
    WHERE DATEDIFF(END_DTTM, DX_DATE) <= 365
  ) t
  WHERE charlson_wt IS NOT NULL
  GROUP BY PAT_ID, END_DTTM
)

SELECT
  c.PAT_ID,
  c.END_DTTM,

  COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^Z85' THEN 1 END), 0) AS icd_MALIGNANCY_FLAG_EVER,
  COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^D50' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0)
    AS icd_IRON_DEF_ANEMIA_FLAG_12MO,
  COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(D5[0-3]|D6[234])' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0)
    AS icd_ANEMIA_FLAG_12MO,
  COALESCE(SUM(CASE WHEN dx.CODE RLIKE '^(K62\\.5|K92\\.[12])' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 ELSE 0 END), 0)
    AS icd_BLEED_CNT_12MO,

  -- Symptom burden: sum of 6 binary symptom flags in 12 months
  (
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(K62\\.5|K92\\.[12])' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^R10' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(K59|R19\\.4)' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^R63\\.4' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^R53' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(D5[0-3]|D6[234])' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0)
  ) AS icd_SYMPTOM_BURDEN_12MO,

  COALESCE(ch.COMBINED_COMORBIDITY_12MO, 0) AS icd_COMBINED_COMORBIDITY_12MO,

  -- Chronic GI pattern: IBD or diverticular disease or GI complexity >= 2
  CASE WHEN (
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(K50|K51)' THEN 1 END), 0) = 1  -- IBD (Crohn's/UC)
    OR COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^K57' THEN 1 END), 0) = 1      -- Diverticular disease
    OR (
      COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^K5[0-9]' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 730 THEN 1 END), 0) +
      COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^K6[0-9]' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 730 THEN 1 END), 0) +
      COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^K9[0-5]' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 730 THEN 1 END), 0)
    ) >= 2  -- GI complexity >= 2 code groups in 24mo
  ) THEN 1 ELSE 0 END AS icd_chronic_gi_pattern

FROM cohort_base c
LEFT JOIN all_dx dx ON c.PAT_ID = dx.PAT_ID AND c.END_DTTM = dx.END_DTTM
LEFT JOIN charlson ch ON c.PAT_ID = ch.PAT_ID AND c.END_DTTM = ch.END_DTTM
GROUP BY c.PAT_ID, c.END_DTTM, ch.COMBINED_COMORBIDITY_12MO
""")

print("ICD10 features computed (7 features)")


# ===========================================================================
# 11. LAB FEATURES (14 features)
#
# Source: order_proc_enh -> order_results -> clarity_component
# Lookback: 24 months for latest values; 36 months for ESR/iron studies
#
# Features:
#   lab_HEMOGLOBIN_VALUE                - Latest hemoglobin (g/dL)
#   lab_PLATELETS_VALUE                 - Latest platelets (K/uL)
#   lab_AST_VALUE                       - Latest AST (U/L)
#   lab_ALK_PHOS_VALUE                  - Latest alkaline phosphatase (U/L)
#   lab_ALBUMIN_VALUE                   - Latest albumin (g/dL)
#   lab_ESR_VALUE                       - Latest ESR (mm/hr)
#   lab_IRON_SATURATION_PCT             - Iron/TIBC * 100 (%)
#   lab_ALBUMIN_DROP_15PCT_FLAG         - Albumin dropped >15% vs 6mo prior
#   lab_ANEMIA_GRADE                    - WHO anemia grade (0-3)
#   lab_ANEMIA_SEVERITY_SCORE           - Composite (0-6): grade + iron def + microcytosis
#   lab_THROMBOCYTOSIS_FLAG             - Platelets > 450 (binary)
#   lab_comprehensive_iron_deficiency   - Lab + ICD iron deficiency composite
#   lab_HEMOGLOBIN_ACCELERATING_DECLINE - Hemoglobin dropping faster recently
#   lab_PLATELETS_ACCELERATING_RISE     - Platelets rising faster recently
# ===========================================================================

# Step 1: Extract latest lab values for each component
spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_latest AS

WITH lab_results AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    cc.NAME AS COMPONENT_NAME,
    TRY_CAST(REGEXP_REPLACE(ores.ORD_VALUE, '[><]', '') AS FLOAT) AS VALUE,
    ores.RESULT_TIME,
    ROW_NUMBER() OVER (
      PARTITION BY c.PAT_ID, c.END_DTTM, cc.NAME
      ORDER BY ores.RESULT_TIME DESC
    ) AS rn
  FROM cohort_base c
  JOIN clarity_cur.order_proc_enh op
    ON op.PAT_ID = c.PAT_ID
    AND DATE(op.ORDERING_DATE) < c.END_DTTM
    AND DATE(op.ORDERING_DATE) >= DATE_SUB(c.END_DTTM, 1095)  -- 3yr for ESR/iron
    AND DATE(op.ORDERING_DATE) >= DATE('2021-07-01')
    AND op.ORDER_STATUS_C IN (3, 5, 10)
  JOIN clarity.order_results ores
    ON ores.ORDER_PROC_ID = op.ORDER_PROC_ID
  JOIN clarity.clarity_component cc
    ON cc.COMPONENT_ID = ores.COMPONENT_ID
  WHERE cc.NAME IN ('HEMOGLOBIN', 'PLATELETS', 'AST', 'ALBUMIN',
                     'ALKALINE PHOSPHATASE', 'MCV', 'FERRITIN',
                     'ESR', 'IRON', 'TIBC')
    AND DATE(ores.RESULT_TIME) < c.END_DTTM
    AND TRY_CAST(REGEXP_REPLACE(ores.ORD_VALUE, '[><]', '') AS FLOAT) IS NOT NULL
)

SELECT
  PAT_ID,
  END_DTTM,
  MAX(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' THEN VALUE END) AS HEMOGLOBIN_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'PLATELETS' THEN VALUE END) AS PLATELETS_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'AST' THEN VALUE END) AS AST_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'ALBUMIN' THEN VALUE END) AS ALBUMIN_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'ALKALINE PHOSPHATASE' THEN VALUE END) AS ALK_PHOS_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'MCV' THEN VALUE END) AS MCV_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'FERRITIN' THEN VALUE END) AS FERRITIN_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'ESR' THEN VALUE END) AS ESR_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'IRON' THEN VALUE END) AS IRON_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'TIBC' THEN VALUE END) AS TIBC_VALUE
FROM lab_results
WHERE rn = 1
GROUP BY PAT_ID, END_DTTM
""")

print("Latest lab values extracted")

# Step 2: Albumin 6 months prior (for drop flag)
spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_albumin_prior AS

WITH albumin_6m AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    TRY_CAST(REGEXP_REPLACE(ores.ORD_VALUE, '[><]', '') AS FLOAT) AS VALUE,
    DATEDIFF(c.END_DTTM, DATE(ores.RESULT_TIME)) AS DAYS_BEFORE,
    ROW_NUMBER() OVER (
      PARTITION BY c.PAT_ID, c.END_DTTM
      ORDER BY ABS(DATEDIFF(c.END_DTTM, DATE(ores.RESULT_TIME)) - 180)
    ) AS rn
  FROM cohort_base c
  JOIN clarity_cur.order_proc_enh op
    ON op.PAT_ID = c.PAT_ID
    AND DATE(op.ORDERING_DATE) < c.END_DTTM
    AND DATE(op.ORDERING_DATE) >= DATE_SUB(c.END_DTTM, 730)
    AND op.ORDER_STATUS_C IN (3, 5, 10)
  JOIN clarity.order_results ores
    ON ores.ORDER_PROC_ID = op.ORDER_PROC_ID
  JOIN clarity.clarity_component cc
    ON cc.COMPONENT_ID = ores.COMPONENT_ID
  WHERE cc.NAME = 'ALBUMIN'
    AND DATEDIFF(c.END_DTTM, DATE(ores.RESULT_TIME)) BETWEEN 150 AND 210
    AND TRY_CAST(REGEXP_REPLACE(ores.ORD_VALUE, '[><]', '') AS FLOAT) IS NOT NULL
)

SELECT PAT_ID, END_DTTM, VALUE AS ALBUMIN_6M_PRIOR
FROM albumin_6m WHERE rn = 1
""")

print("Albumin prior values extracted")

# Step 3: Lab acceleration features
spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_acceleration AS

WITH lab_history AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    cc.NAME AS COMPONENT_NAME,
    TRY_CAST(REGEXP_REPLACE(ores.ORD_VALUE, '[><]', '') AS FLOAT) AS VALUE,
    DATEDIFF(c.END_DTTM, DATE(ores.RESULT_TIME)) AS DAYS_BEFORE
  FROM cohort_base c
  JOIN clarity_cur.order_proc_enh op
    ON op.PAT_ID = c.PAT_ID
    AND DATE(op.ORDERING_DATE) < c.END_DTTM
    AND DATE(op.ORDERING_DATE) >= DATE_SUB(c.END_DTTM, 730)
    AND DATE(op.ORDERING_DATE) >= DATE('2021-07-01')
    AND op.ORDER_STATUS_C IN (3, 5, 10)
  JOIN clarity.order_results ores
    ON ores.ORDER_PROC_ID = op.ORDER_PROC_ID
  JOIN clarity.clarity_component cc
    ON cc.COMPONENT_ID = ores.COMPONENT_ID
  WHERE cc.NAME IN ('HEMOGLOBIN', 'PLATELETS')
    AND DATE(ores.RESULT_TIME) < c.END_DTTM
    AND TRY_CAST(REGEXP_REPLACE(ores.ORD_VALUE, '[><]', '') AS FLOAT) IS NOT NULL
),

time_points AS (
  SELECT PAT_ID, END_DTTM, COMPONENT_NAME,
    MAX(CASE WHEN DAYS_BEFORE <= 30 THEN VALUE END) AS current_value,
    AVG(CASE WHEN DAYS_BEFORE BETWEEN 60 AND 120 THEN VALUE END) AS value_3mo_prior,
    AVG(CASE WHEN DAYS_BEFORE BETWEEN 150 AND 210 THEN VALUE END) AS value_6mo_prior
  FROM lab_history
  GROUP BY PAT_ID, END_DTTM, COMPONENT_NAME
)

SELECT PAT_ID, END_DTTM,
  MAX(CASE
    WHEN COMPONENT_NAME = 'HEMOGLOBIN'
     AND current_value IS NOT NULL AND value_3mo_prior IS NOT NULL AND value_6mo_prior IS NOT NULL
     AND ((current_value - value_3mo_prior) / 3.0) < -0.5
     AND ((current_value - value_3mo_prior) / 3.0) < ((value_3mo_prior - value_6mo_prior) / 3.0)
    THEN 1 ELSE 0 END) AS lab_HEMOGLOBIN_ACCELERATING_DECLINE,
  MAX(CASE
    WHEN COMPONENT_NAME = 'PLATELETS'
     AND current_value IS NOT NULL AND current_value > 450
     AND value_3mo_prior IS NOT NULL AND value_6mo_prior IS NOT NULL
     AND ((current_value - value_3mo_prior) / 3.0) > ((value_3mo_prior - value_6mo_prior) / 3.0)
    THEN 1 ELSE 0 END) AS lab_PLATELETS_ACCELERATING_RISE
FROM time_points
GROUP BY PAT_ID, END_DTTM
""")

print("Lab acceleration features computed")

# Step 4: Combine all lab features
spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_features AS

SELECT
  c.PAT_ID,
  c.END_DTTM,
  ll.HEMOGLOBIN_VALUE AS lab_HEMOGLOBIN_VALUE,
  ll.PLATELETS_VALUE AS lab_PLATELETS_VALUE,
  ll.AST_VALUE AS lab_AST_VALUE,
  ll.ALK_PHOS_VALUE AS lab_ALK_PHOS_VALUE,
  ll.ALBUMIN_VALUE AS lab_ALBUMIN_VALUE,
  ll.ESR_VALUE AS lab_ESR_VALUE,

  -- Iron saturation: iron / TIBC * 100
  CASE WHEN ll.TIBC_VALUE > 0
    THEN ROUND(ll.IRON_VALUE / ll.TIBC_VALUE * 100, 1)
  END AS lab_IRON_SATURATION_PCT,

  -- Albumin drop > 15% from 6mo prior
  CASE WHEN ll.ALBUMIN_VALUE IS NOT NULL AND ap.ALBUMIN_6M_PRIOR IS NOT NULL
       AND ll.ALBUMIN_VALUE < ap.ALBUMIN_6M_PRIOR * 0.85
    THEN 1 ELSE 0
  END AS lab_ALBUMIN_DROP_15PCT_FLAG,

  -- WHO anemia grade: 0=normal, 1=mild, 2=moderate, 3=severe
  CASE
    WHEN ll.HEMOGLOBIN_VALUE IS NULL THEN NULL
    WHEN ll.HEMOGLOBIN_VALUE >= 12.0 THEN 0
    WHEN ll.HEMOGLOBIN_VALUE >= 10.0 THEN 1
    WHEN ll.HEMOGLOBIN_VALUE >= 8.0 THEN 2
    ELSE 3
  END AS lab_ANEMIA_GRADE,

  -- Anemia severity score: grade (0-3) + iron deficiency (+2) + microcytosis (+1)
  COALESCE(
    CASE
      WHEN ll.HEMOGLOBIN_VALUE >= 12.0 THEN 0
      WHEN ll.HEMOGLOBIN_VALUE >= 10.0 THEN 1
      WHEN ll.HEMOGLOBIN_VALUE >= 8.0 THEN 2
      WHEN ll.HEMOGLOBIN_VALUE IS NOT NULL THEN 3
      ELSE 0
    END
    + CASE WHEN ll.FERRITIN_VALUE < 15 AND ll.HEMOGLOBIN_VALUE < 12 THEN 2 ELSE 0 END
    + CASE WHEN ll.MCV_VALUE < 80 THEN 1 ELSE 0 END
  , 0) AS lab_ANEMIA_SEVERITY_SCORE,

  CASE WHEN ll.PLATELETS_VALUE > 450 THEN 1 ELSE 0 END AS lab_THROMBOCYTOSIS_FLAG,

  COALESCE(la.lab_HEMOGLOBIN_ACCELERATING_DECLINE, 0) AS lab_HEMOGLOBIN_ACCELERATING_DECLINE,
  COALESCE(la.lab_PLATELETS_ACCELERATING_RISE, 0) AS lab_PLATELETS_ACCELERATING_RISE,

  -- Lab-only iron deficiency (combined with ICD flag in final join)
  CASE
    WHEN (ll.HEMOGLOBIN_VALUE < 12 AND ll.MCV_VALUE < 80) THEN 1
    WHEN (ll.FERRITIN_VALUE < 30 AND ll.HEMOGLOBIN_VALUE < 13) THEN 1
    ELSE 0
  END AS lab_iron_deficiency_labs_only

FROM cohort_base c
LEFT JOIN lab_latest ll ON c.PAT_ID = ll.PAT_ID AND c.END_DTTM = ll.END_DTTM
LEFT JOIN lab_albumin_prior ap ON c.PAT_ID = ap.PAT_ID AND c.END_DTTM = ap.END_DTTM
LEFT JOIN lab_acceleration la ON c.PAT_ID = la.PAT_ID AND c.END_DTTM = la.END_DTTM
""")

print("Lab features combined (14 features)")


# ===========================================================================
# 12. OUTPATIENT MEDICATION FEATURES (3 features)
#
# Source: order_med_enh (outpatient prescriptions)
# Lookback: 12 months
#
# Features:
#   out_med_ppi_use_flag         - Any proton pump inhibitor use
#   out_med_broad_abx_recency   - Days since last broad-spectrum antibiotic
#   out_med_ibd_meds_recency    - Days since last IBD medication
# ===========================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW outpatient_med_features AS

WITH outpatient_meds AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    LOWER(med.GENERIC_NAME) AS generic_name,
    DATE(ome.ORDER_START_TIME) AS rx_date,
    DATEDIFF(c.END_DTTM, DATE(ome.ORDER_START_TIME)) AS days_since
  FROM cohort_base c
  JOIN clarity_cur.order_med_enh ome
    ON ome.PAT_ID = c.PAT_ID
    AND DATE(ome.ORDER_START_TIME) < c.END_DTTM
    AND DATE(ome.ORDER_START_TIME) >= DATE_SUB(c.END_DTTM, 365)
    AND DATE(ome.ORDER_START_TIME) >= DATE('2021-07-01')
    AND ome.ORDERING_MODE_C <> 2  -- Exclude inpatient
    AND ome.ORDER_STATUS_C IN (2, 5)  -- Sent or Completed
    AND ome.ORDER_CLASS <> 'Historical Med'
  JOIN clarity.clarity_medication med
    ON med.MEDICATION_ID = ome.MEDICATION_ID
  WHERE med.GENERIC_NAME IS NOT NULL
)

SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- PPI use flag
  COALESCE(MAX(CASE
    WHEN om.generic_name RLIKE '(omeprazole|pantoprazole|lansoprazole|esomeprazole|rabeprazole|dexlansoprazole)'
    THEN 1 END), 0) AS out_med_ppi_use_flag,

  -- Broad-spectrum antibiotics recency (days since last Rx; NULL -> large number in final join)
  MIN(CASE
    WHEN om.generic_name RLIKE '(cephalexin|ceftriaxone|cefdinir|cefuroxime|cefpodoxime|ciprofloxacin|levofloxacin|moxifloxacin|amoxicillin.*clavulanate|piperacillin|meropenem|ertapenem)'
    THEN om.days_since END) AS out_med_broad_abx_recency,

  -- IBD medication recency (5-ASA, immunosuppressants, biologics)
  MIN(CASE
    WHEN om.generic_name RLIKE '(mesalamine|sulfasalazine|balsalazide|olsalazine|azathioprine|mercaptopurine|methotrexate|infliximab|adalimumab|vedolizumab|ustekinumab|tofacitinib|ozanimod)'
    THEN om.days_since END) AS out_med_ibd_meds_recency

FROM cohort_base c
LEFT JOIN outpatient_meds om ON c.PAT_ID = om.PAT_ID AND c.END_DTTM = om.END_DTTM
GROUP BY c.PAT_ID, c.END_DTTM
""")

print("Outpatient medication features computed (3 features)")


# ===========================================================================
# 13. INPATIENT MEDICATION FEATURES (4 features)
#
# Source: order_med_enh (inpatient) + mar_admin_info_enh (administration)
# Lookback: 12 months
#
# Features:
#   inp_med_inp_gi_hospitalization - GI-related hospitalization flag
#   inp_med_inp_ibd_meds_recency  - Days since inpatient IBD medication
#   inp_med_inp_obstruction_pattern - Laxative + opioid same admission
#   inp_med_inp_opioid_use_flag   - Any inpatient opioid
# ===========================================================================

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW inpatient_med_features AS

WITH inpatient_meds AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    LOWER(med.GENERIC_NAME) AS generic_name,
    DATE(mar.TAKEN_TIME) AS admin_date,
    DATEDIFF(c.END_DTTM, DATE(mar.TAKEN_TIME)) AS days_since,
    hsp.PAT_ENC_CSN_ID AS admission_csn
  FROM cohort_base c
  JOIN clarity_cur.PAT_ENC_HSP_HAR_ENH hsp
    ON hsp.PAT_ID = c.PAT_ID
    AND DATE(hsp.HOSP_ADMSN_TIME) < c.END_DTTM
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE_SUB(c.END_DTTM, 365)
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE('2021-07-01')
    AND hsp.ADT_PATIENT_STAT_C <> 1
    AND hsp.ADMIT_CONF_STAT_C <> 3
  JOIN clarity_cur.order_med_enh ome
    ON ome.PAT_ID = c.PAT_ID
    AND ome.PAT_ENC_CSN_ID = hsp.PAT_ENC_CSN_ID
    AND ome.ORDERING_MODE_C = 2  -- Inpatient only
  JOIN prod.clarity_cur.mar_admin_info_enh mar
    ON mar.ORDER_MED_ID = ome.ORDER_MED_ID
    AND mar.TAKEN_TIME IS NOT NULL
    AND UPPER(TRIM(mar.ACTION)) IN (
      'GIVEN', 'PATIENT/FAMILY ADMIN', 'GIVEN-SEE OVERRIDE',
      'NEW BAG', 'BOLUS', 'PUSH', 'APPLIED'
    )
  JOIN clarity.clarity_medication med
    ON med.MEDICATION_ID = ome.MEDICATION_ID
  WHERE med.GENERIC_NAME IS NOT NULL
),

-- Per-admission medication flags
admission_flags AS (
  SELECT
    PAT_ID, END_DTTM, admission_csn,
    MAX(CASE WHEN generic_name RLIKE '(oxycodone|hydromorphone|morphine|fentanyl|hydrocodone|tramadol|methadone|meperidine)'
        THEN 1 ELSE 0 END) AS has_opioid,
    MAX(CASE WHEN generic_name RLIKE '(polyethylene glycol|bisacodyl|senna|docusate|lactulose|magnesium citrate)'
        THEN 1 ELSE 0 END) AS has_laxative,
    MAX(CASE WHEN generic_name RLIKE '(mesalamine|sulfasalazine|azathioprine|mercaptopurine|infliximab|adalimumab|vedolizumab|ustekinumab)'
        THEN 1 ELSE 0 END) AS has_ibd_med,
    MIN(CASE WHEN generic_name RLIKE '(mesalamine|sulfasalazine|azathioprine|mercaptopurine|infliximab|adalimumab|vedolizumab|ustekinumab)'
        THEN days_since END) AS ibd_days_since
  FROM inpatient_meds
  GROUP BY PAT_ID, END_DTTM, admission_csn
)

SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- GI hospitalization flag (admission with laxative or GI bleed meds)
  COALESCE(MAX(af.has_laxative), 0) AS inp_med_inp_gi_hospitalization,

  -- IBD medication recency (days since)
  MIN(af.ibd_days_since) AS inp_med_inp_ibd_meds_recency,

  -- Obstruction pattern: laxative + opioid in same admission
  COALESCE(MAX(CASE WHEN af.has_laxative = 1 AND af.has_opioid = 1 THEN 1 ELSE 0 END), 0)
    AS inp_med_inp_obstruction_pattern,

  -- Inpatient opioid use flag
  COALESCE(MAX(af.has_opioid), 0) AS inp_med_inp_opioid_use_flag

FROM cohort_base c
LEFT JOIN admission_flags af ON c.PAT_ID = af.PAT_ID AND c.END_DTTM = af.END_DTTM
GROUP BY c.PAT_ID, c.END_DTTM
""")

print("Inpatient medication features computed (4 features)")


# ===========================================================================
# 14. VISIT HISTORY FEATURES (10 features)
#
# Source: pat_enc_enh (outpatient), PAT_ENC_HSP_HAR_ENH (inpatient/ED),
#         pat_enc_dx_enh + hsp_acct_dx_list_enh (GI symptom diagnoses)
# Lookback: 12 months for counts; 24 months for GI specialist recency/inpatient
#
# Features:
#   visit_recency_last_gi                - Days since last GI specialist visit
#   visit_pcp_visits_12mo                - PCP visit count (12mo)
#   visit_outpatient_visits_12mo         - Total outpatient visits (12mo)
#   visit_no_shows_12mo                  - No-show count (12mo)
#   visit_gi_symptom_op_visits_12mo      - Outpatient visits with GI symptom dx (12mo)
#   visit_total_gi_symptom_visits_12mo   - All visits with GI symptom dx (12mo)
#   visit_acute_care_reliance            - ED/inpatient to outpatient ratio
#   visit_healthcare_intensity_score     - Weighted visit intensity (op+2*ED+3*inp)
#   visit_inp_last_24_months             - Inpatient admissions in 24 months
#   visit_primary_care_continuity_ratio  - PCP visits / total outpatient visits
# ===========================================================================

spark.sql("""
CREATE OR REPLACE TEMP VIEW visit_features AS

WITH
-- All outpatient encounters in our integrated system (12 months)
outpatient AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    pe.PAT_ENC_CSN_ID,
    DATE(pe.CONTACT_DATE) AS VISIT_DATE,
    pe.APPT_STATUS_C,
    dep.SPECIALTY
  FROM cohort_base c
  JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID
    AND DATE(pe.CONTACT_DATE) < c.END_DTTM
    AND DATE(pe.CONTACT_DATE) >= DATE_SUB(c.END_DTTM, 365)
    AND DATE(pe.CONTACT_DATE) >= DATE('2021-07-01')
  JOIN clarity_cur.dep_loc_ploc_sa_enh dep
    ON dep.department_id = COALESCE(pe.DEPARTMENT_ID, pe.EFFECTIVE_DEPT_ID)
  WHERE dep.RPT_GRP_SIX IN ('116001', '116002')
),

-- Outpatient encounters with GI-related diagnosis codes
gi_symptom_op AS (
  SELECT DISTINCT o.PAT_ID, o.END_DTTM, o.PAT_ENC_CSN_ID
  FROM outpatient o
  JOIN clarity_cur.pat_enc_dx_enh dx
    ON dx.PAT_ENC_CSN_ID = o.PAT_ENC_CSN_ID
  WHERE dx.ICD10_CODE RLIKE '^(K62\\.5|K92|K59|R19|D50|R10|R63\\.4|R53)'
    AND o.APPT_STATUS_C IN (2, 6)
),

-- Inpatient/ED encounters with GI-related diagnosis codes
gi_symptom_acute AS (
  SELECT DISTINCT c.PAT_ID, c.END_DTTM, hsp.PAT_ENC_CSN_ID
  FROM cohort_base c
  JOIN clarity_cur.PAT_ENC_HSP_HAR_ENH hsp
    ON hsp.PAT_ID = c.PAT_ID
    AND DATE(hsp.HOSP_ADMSN_TIME) < c.END_DTTM
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE_SUB(c.END_DTTM, 365)
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE('2021-07-01')
    AND hsp.ADT_PATIENT_STAT_C <> 1
    AND hsp.ADMIT_CONF_STAT_C <> 3
  JOIN clarity_cur.hsp_acct_dx_list_enh dx
    ON dx.PRIM_ENC_CSN_ID = hsp.PAT_ENC_CSN_ID
  WHERE dx.CODE RLIKE '^(K62\\.5|K92|K59|R19|D50|R10|R63\\.4|R53)'
),

-- ED/inpatient encounters (12mo) -- split by type for intensity scoring
acute_care AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    COUNT(DISTINCT hsp.PAT_ENC_CSN_ID) AS ed_inp_visits,
    -- ED visits (ADT_PATIENT_STAT_C = 4 = ED, or emergency department admission)
    COUNT(DISTINCT CASE WHEN hsp.ADT_PATIENT_STAT_C = 4 THEN hsp.PAT_ENC_CSN_ID END) AS ed_visits,
    -- True inpatient (not ED)
    COUNT(DISTINCT CASE WHEN hsp.ADT_PATIENT_STAT_C <> 4 THEN hsp.PAT_ENC_CSN_ID END) AS inp_visits
  FROM cohort_base c
  JOIN clarity_cur.PAT_ENC_HSP_HAR_ENH hsp
    ON hsp.PAT_ID = c.PAT_ID
    AND DATE(hsp.HOSP_ADMSN_TIME) < c.END_DTTM
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE_SUB(c.END_DTTM, 365)
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE('2021-07-01')
    AND hsp.ADT_PATIENT_STAT_C <> 1
    AND hsp.ADMIT_CONF_STAT_C <> 3
  GROUP BY c.PAT_ID, c.END_DTTM
),

-- Inpatient admissions in 24 months (broader lookback)
inp_24mo AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    COUNT(DISTINCT hsp.PAT_ENC_CSN_ID) AS inp_last_24_months
  FROM cohort_base c
  JOIN clarity_cur.PAT_ENC_HSP_HAR_ENH hsp
    ON hsp.PAT_ID = c.PAT_ID
    AND DATE(hsp.HOSP_ADMSN_TIME) < c.END_DTTM
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE_SUB(c.END_DTTM, 730)
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE('2021-07-01')
    AND hsp.ADT_PATIENT_STAT_C NOT IN (1, 4)  -- Not pre-admit, not ED-only
    AND hsp.ADMIT_CONF_STAT_C <> 3
  GROUP BY c.PAT_ID, c.END_DTTM
),

-- Days since last GI specialist visit (24mo lookback for recency)
gi_recency AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    MIN(DATEDIFF(c.END_DTTM, DATE(pe.CONTACT_DATE))) AS days_since_last_gi
  FROM cohort_base c
  JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID
    AND DATE(pe.CONTACT_DATE) < c.END_DTTM
    AND DATE(pe.CONTACT_DATE) >= DATE_SUB(c.END_DTTM, 730)
    AND pe.APPT_STATUS_C IN (2, 6)
  JOIN clarity_cur.dep_loc_ploc_sa_enh dep
    ON dep.department_id = COALESCE(pe.DEPARTMENT_ID, pe.EFFECTIVE_DEPT_ID)
  WHERE UPPER(dep.SPECIALTY) LIKE '%GASTRO%'
  GROUP BY c.PAT_ID, c.END_DTTM
),

-- Aggregate outpatient metrics: visit counts by type
op_metrics AS (
  SELECT
    PAT_ID,
    END_DTTM,
    -- Total completed outpatient visits
    SUM(CASE WHEN APPT_STATUS_C IN (2, 6) THEN 1 ELSE 0 END) AS outpatient_visits_12mo,
    -- PCP visits (primary care, family medicine, internal medicine)
    SUM(CASE WHEN APPT_STATUS_C IN (2, 6)
         AND (UPPER(SPECIALTY) LIKE '%PRIMARY%' OR UPPER(SPECIALTY) LIKE '%FAMILY%'
              OR UPPER(SPECIALTY) LIKE '%INTERNAL MED%')
         THEN 1 ELSE 0 END) AS pcp_visits_12mo,
    -- Appointment no-shows
    SUM(CASE WHEN APPT_STATUS_C IN (3, 4) THEN 1 ELSE 0 END) AS no_shows_12mo
  FROM outpatient
  GROUP BY PAT_ID, END_DTTM
),

-- GI symptom outpatient visit counts
gi_symptom_op_counts AS (
  SELECT PAT_ID, END_DTTM, COUNT(DISTINCT PAT_ENC_CSN_ID) AS gi_symptom_op_visits_12mo
  FROM gi_symptom_op
  GROUP BY PAT_ID, END_DTTM
),

-- GI symptom inpatient/ED visit counts
gi_symptom_acute_counts AS (
  SELECT PAT_ID, END_DTTM, COUNT(DISTINCT PAT_ENC_CSN_ID) AS gi_symptom_acute_visits_12mo
  FROM gi_symptom_acute
  GROUP BY PAT_ID, END_DTTM
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  -- Days since last GI specialist visit (9999 if none in 24mo)
  COALESCE(gr.days_since_last_gi, 9999) AS visit_recency_last_gi,
  COALESCE(om.pcp_visits_12mo, 0) AS visit_pcp_visits_12mo,
  COALESCE(om.outpatient_visits_12mo, 0) AS visit_outpatient_visits_12mo,
  COALESCE(om.no_shows_12mo, 0) AS visit_no_shows_12mo,
  COALESCE(gsop.gi_symptom_op_visits_12mo, 0) AS visit_gi_symptom_op_visits_12mo,
  -- Total GI symptom visits across all settings
  COALESCE(gsop.gi_symptom_op_visits_12mo, 0) + COALESCE(gsac.gi_symptom_acute_visits_12mo, 0)
    AS visit_total_gi_symptom_visits_12mo,
  -- Acute care reliance: ratio of ED/inpatient to outpatient visits
  CASE
    WHEN COALESCE(om.outpatient_visits_12mo, 0) > 0
    THEN ROUND(COALESCE(ac.ed_inp_visits, 0) * 1.0 / om.outpatient_visits_12mo, 3)
    ELSE COALESCE(ac.ed_inp_visits, 0) * 1.0
  END AS visit_acute_care_reliance,
  -- Healthcare intensity: outpatient + 2*ED + 3*inpatient (weighted utilization)
  COALESCE(om.outpatient_visits_12mo, 0)
    + 2 * COALESCE(ac.ed_visits, 0)
    + 3 * COALESCE(ac.inp_visits, 0)
    AS visit_healthcare_intensity_score,
  -- Inpatient admissions in 24 months
  COALESCE(i24.inp_last_24_months, 0) AS visit_inp_last_24_months,
  -- Primary care continuity: PCP share of outpatient visits
  CASE
    WHEN COALESCE(om.outpatient_visits_12mo, 0) > 0
    THEN ROUND(COALESCE(om.pcp_visits_12mo, 0) * 1.0 / om.outpatient_visits_12mo, 3)
    ELSE 0
  END AS visit_primary_care_continuity_ratio

FROM cohort_base c
LEFT JOIN op_metrics om ON c.PAT_ID = om.PAT_ID AND c.END_DTTM = om.END_DTTM
LEFT JOIN gi_symptom_op_counts gsop ON c.PAT_ID = gsop.PAT_ID AND c.END_DTTM = gsop.END_DTTM
LEFT JOIN gi_symptom_acute_counts gsac ON c.PAT_ID = gsac.PAT_ID AND c.END_DTTM = gsac.END_DTTM
LEFT JOIN acute_care ac ON c.PAT_ID = ac.PAT_ID AND c.END_DTTM = ac.END_DTTM
LEFT JOIN inp_24mo i24 ON c.PAT_ID = i24.PAT_ID AND c.END_DTTM = i24.END_DTTM
LEFT JOIN gi_recency gr ON c.PAT_ID = gr.PAT_ID AND c.END_DTTM = gr.END_DTTM
""")

print("Visit features computed (10 features)")


# ===========================================================================
# 15. PROCEDURE FEATURES (3 features)
#
# Source: order_proc_enh (completed orders only)
# Lookback: 12 months
#
# Features:
#   proc_blood_transfusion_count_12mo  - Blood transfusion count (12mo)
#   proc_high_imaging_intensity_flag   - 3+ abdominal/pelvic imaging in 12mo
#   proc_mri_abd_pelvis_count_12mo     - MRI abdomen/pelvis count (12mo)
#
# Note: colonoscopy is deliberately excluded because screened patients
# have already been removed from the cohort.
# ===========================================================================

spark.sql("""
CREATE OR REPLACE TEMP VIEW proc_features AS

WITH proc_raw AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,

    -- Blood transfusion
    CASE
      WHEN op.PROC_CODE IN ('36430','36440','36450','36455','36456','86900','86901')
        OR LOWER(op.PROC_NAME) LIKE '%transfus%'
        OR LOWER(op.PROC_NAME) LIKE '%blood product%'
      THEN 1 ELSE 0
    END AS is_transfusion,

    -- MRI abdomen/pelvis
    CASE
      WHEN op.PROC_CODE IN ('74181','74182','74183')
        OR LOWER(op.PROC_NAME) LIKE '%mri%abd%'
        OR LOWER(op.PROC_NAME) LIKE '%mri%pelv%'
      THEN 1 ELSE 0
    END AS is_mri_abd,

    -- Any abdominal/pelvic imaging (CT or MRI)
    CASE
      WHEN op.PROC_CODE IN ('74150','74160','74170','74176','74177','74178',
                             '74181','74182','74183')
        OR LOWER(op.PROC_NAME) LIKE '%ct%abd%'
        OR LOWER(op.PROC_NAME) LIKE '%ct%pelv%'
        OR LOWER(op.PROC_NAME) LIKE '%mri%abd%'
        OR LOWER(op.PROC_NAME) LIKE '%mri%pelv%'
      THEN 1 ELSE 0
    END AS is_abd_imaging

  FROM cohort_base c
  JOIN clarity_cur.order_proc_enh op
    ON op.PAT_ID = c.PAT_ID
    AND DATE(op.ORDERING_DATE) < c.END_DTTM
    AND DATE(op.ORDERING_DATE) >= DATE_SUB(c.END_DTTM, 365)
    AND DATE(op.ORDERING_DATE) >= DATE('2021-07-01')
    AND op.ORDER_STATUS_C = 5  -- Completed only
    AND op.RPT_GRP_SIX IN ('116001', '116002')
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  COALESCE(SUM(pr.is_transfusion), 0) AS proc_blood_transfusion_count_12mo,
  -- High imaging intensity: 3+ abdominal/pelvic imaging studies in 12 months
  CASE WHEN COALESCE(SUM(pr.is_abd_imaging), 0) >= 3 THEN 1 ELSE 0 END
    AS proc_high_imaging_intensity_flag,
  COALESCE(SUM(pr.is_mri_abd), 0) AS proc_mri_abd_pelvis_count_12mo

FROM cohort_base c
LEFT JOIN proc_raw pr ON c.PAT_ID = pr.PAT_ID AND c.END_DTTM = pr.END_DTTM
GROUP BY c.PAT_ID, c.END_DTTM
""")

print("Procedure features computed (3 features)")


# ===========================================================================
# 16. FINAL JOIN
#
# Join all feature views together and select the final 57 features plus
# identifiers (PAT_ID, END_DTTM, FUTURE_CRC_EVENT, ICD10_GROUP).
#
# Null handling:
#   - Most numeric features: COALESCE to 0
#   - Recency features: COALESCE to 9999 (large number = no recent data)
#   - lab_comprehensive_iron_deficiency: composite of ICD flag + lab criteria
# ===========================================================================

spark.sql("""
CREATE OR REPLACE TEMP VIEW final_features AS

SELECT
  c.PAT_ID,
  c.END_DTTM,
  c.FUTURE_CRC_EVENT,
  c.ICD10_GROUP,

  -- Demographics (6 features)
  c.AGE_GROUP,
  c.IS_FEMALE,
  c.IS_MARRIED_PARTNER,
  c.RACE_ASIAN,
  c.RACE_CAUCASIAN,
  c.HAS_PCP_AT_END,
  COALESCE(c.months_since_cohort_entry, 0) AS months_since_cohort_entry,

  -- Vitals (9 features)
  COALESCE(v.vit_BMI, 0) AS vit_BMI,
  COALESCE(v.vit_BMI_CHANGE_6M, 0) AS vit_BMI_CHANGE_6M,
  COALESCE(v.vit_PULSE, 0) AS vit_PULSE,
  COALESCE(v.vit_PULSE_PRESSURE, 0) AS vit_PULSE_PRESSURE,
  COALESCE(v.vit_RECENCY_WEIGHT, 9999) AS vit_RECENCY_WEIGHT,
  COALESCE(v.vit_CACHEXIA_RISK_SCORE, 0) AS vit_CACHEXIA_RISK_SCORE,
  COALESCE(v.vit_WEIGHT_TRAJECTORY_SLOPE, 0) AS vit_WEIGHT_TRAJECTORY_SLOPE,
  COALESCE(v.vit_MAX_WEIGHT_LOSS_PCT_60D, 0) AS vit_MAX_WEIGHT_LOSS_PCT_60D,
  COALESCE(v.vit_SBP_VARIABILITY_6M, 0) AS vit_SBP_VARIABILITY_6M,

  -- ICD-10 (7 features)
  COALESCE(i.icd_MALIGNANCY_FLAG_EVER, 0) AS icd_MALIGNANCY_FLAG_EVER,
  COALESCE(i.icd_COMBINED_COMORBIDITY_12MO, 0) AS icd_COMBINED_COMORBIDITY_12MO,
  COALESCE(i.icd_IRON_DEF_ANEMIA_FLAG_12MO, 0) AS icd_IRON_DEF_ANEMIA_FLAG_12MO,
  COALESCE(i.icd_ANEMIA_FLAG_12MO, 0) AS icd_ANEMIA_FLAG_12MO,
  COALESCE(i.icd_SYMPTOM_BURDEN_12MO, 0) AS icd_SYMPTOM_BURDEN_12MO,
  COALESCE(i.icd_BLEED_CNT_12MO, 0) AS icd_BLEED_CNT_12MO,
  COALESCE(i.icd_chronic_gi_pattern, 0) AS icd_chronic_gi_pattern,

  -- Labs (14 features)
  COALESCE(l.lab_HEMOGLOBIN_VALUE, 0) AS lab_HEMOGLOBIN_VALUE,
  COALESCE(l.lab_PLATELETS_VALUE, 0) AS lab_PLATELETS_VALUE,
  COALESCE(l.lab_AST_VALUE, 0) AS lab_AST_VALUE,
  COALESCE(l.lab_ALK_PHOS_VALUE, 0) AS lab_ALK_PHOS_VALUE,
  COALESCE(l.lab_ALBUMIN_VALUE, 0) AS lab_ALBUMIN_VALUE,
  COALESCE(l.lab_ESR_VALUE, 0) AS lab_ESR_VALUE,
  COALESCE(l.lab_IRON_SATURATION_PCT, 0) AS lab_IRON_SATURATION_PCT,
  COALESCE(l.lab_ALBUMIN_DROP_15PCT_FLAG, 0) AS lab_ALBUMIN_DROP_15PCT_FLAG,
  COALESCE(l.lab_ANEMIA_GRADE, 0) AS lab_ANEMIA_GRADE,
  COALESCE(l.lab_ANEMIA_SEVERITY_SCORE, 0) AS lab_ANEMIA_SEVERITY_SCORE,
  COALESCE(l.lab_THROMBOCYTOSIS_FLAG, 0) AS lab_THROMBOCYTOSIS_FLAG,
  -- Comprehensive iron deficiency: TRUE if EITHER ICD diagnosis OR lab criteria met
  CASE
    WHEN COALESCE(i.icd_IRON_DEF_ANEMIA_FLAG_12MO, 0) = 1 THEN 1
    WHEN COALESCE(l.lab_iron_deficiency_labs_only, 0) = 1 THEN 1
    ELSE 0
  END AS lab_comprehensive_iron_deficiency,
  COALESCE(l.lab_HEMOGLOBIN_ACCELERATING_DECLINE, 0) AS lab_HEMOGLOBIN_ACCELERATING_DECLINE,
  COALESCE(l.lab_PLATELETS_ACCELERATING_RISE, 0) AS lab_PLATELETS_ACCELERATING_RISE,

  -- Outpatient medications (3 features)
  COALESCE(om.out_med_ppi_use_flag, 0) AS out_med_ppi_use_flag,
  COALESCE(om.out_med_broad_abx_recency, 9999) AS out_med_broad_abx_recency,
  COALESCE(om.out_med_ibd_meds_recency, 9999) AS out_med_ibd_meds_recency,

  -- Inpatient medications (4 features)
  COALESCE(im.inp_med_inp_gi_hospitalization, 0) AS inp_med_inp_gi_hospitalization,
  COALESCE(im.inp_med_inp_ibd_meds_recency, 9999) AS inp_med_inp_ibd_meds_recency,
  COALESCE(im.inp_med_inp_obstruction_pattern, 0) AS inp_med_inp_obstruction_pattern,
  COALESCE(im.inp_med_inp_opioid_use_flag, 0) AS inp_med_inp_opioid_use_flag,

  -- Visits (10 features)
  COALESCE(vis.visit_recency_last_gi, 9999) AS visit_recency_last_gi,
  COALESCE(vis.visit_pcp_visits_12mo, 0) AS visit_pcp_visits_12mo,
  COALESCE(vis.visit_outpatient_visits_12mo, 0) AS visit_outpatient_visits_12mo,
  COALESCE(vis.visit_no_shows_12mo, 0) AS visit_no_shows_12mo,
  COALESCE(vis.visit_gi_symptom_op_visits_12mo, 0) AS visit_gi_symptom_op_visits_12mo,
  COALESCE(vis.visit_total_gi_symptom_visits_12mo, 0) AS visit_total_gi_symptom_visits_12mo,
  COALESCE(vis.visit_acute_care_reliance, 0) AS visit_acute_care_reliance,
  COALESCE(vis.visit_healthcare_intensity_score, 0) AS visit_healthcare_intensity_score,
  COALESCE(vis.visit_inp_last_24_months, 0) AS visit_inp_last_24_months,
  COALESCE(vis.visit_primary_care_continuity_ratio, 0) AS visit_primary_care_continuity_ratio,

  -- Procedures (3 features)
  COALESCE(p.proc_blood_transfusion_count_12mo, 0) AS proc_blood_transfusion_count_12mo,
  COALESCE(p.proc_high_imaging_intensity_flag, 0) AS proc_high_imaging_intensity_flag,
  COALESCE(p.proc_mri_abd_pelvis_count_12mo, 0) AS proc_mri_abd_pelvis_count_12mo

FROM cohort_base c
LEFT JOIN vitals_features v ON c.PAT_ID = v.PAT_ID AND c.END_DTTM = v.END_DTTM
LEFT JOIN icd_features i ON c.PAT_ID = i.PAT_ID AND c.END_DTTM = i.END_DTTM
LEFT JOIN lab_features l ON c.PAT_ID = l.PAT_ID AND c.END_DTTM = l.END_DTTM
LEFT JOIN outpatient_med_features om ON c.PAT_ID = om.PAT_ID AND c.END_DTTM = om.END_DTTM
LEFT JOIN inpatient_med_features im ON c.PAT_ID = im.PAT_ID AND c.END_DTTM = im.END_DTTM
LEFT JOIN visit_features vis ON c.PAT_ID = vis.PAT_ID AND c.END_DTTM = vis.END_DTTM
LEFT JOIN proc_features p ON c.PAT_ID = p.PAT_ID AND c.END_DTTM = p.END_DTTM
""")

final_count = spark.sql("SELECT COUNT(*) AS n FROM final_features").collect()[0]['n']
print(f"Final feature table rows: {final_count:,}")


# ===========================================================================
# 17. SAVE OUTPUT TABLE
#
# Persist the final feature table as a Delta table for downstream training.
# ===========================================================================

spark.sql(f"""
CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
SELECT * FROM final_features
""")

saved_count = spark.table(OUTPUT_TABLE).count()
print(f"Saved to: {OUTPUT_TABLE}")
print(f"Rows: {saved_count:,}")


# ===========================================================================
# 18. VALIDATION & SUMMARY
#
# Sanity checks on the output table:
#   - Null counts per feature (should all be 0 after COALESCE)
#   - Overall positive rate (~0.41% expected)
# ===========================================================================

df_check = spark.table(OUTPUT_TABLE)

# Feature null counts -- should all be 0 after COALESCE in the final join
print("Feature Null Counts (should be 0):")
null_exprs = [
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in SELECTED_FEATURES
]
null_df = df_check.select(null_exprs).toPandas()
total_nulls = null_df.sum(axis=1).values[0]
if total_nulls == 0:
    print("  All features: 0 nulls")
else:
    for col in null_df.columns:
        if null_df[col].values[0] > 0:
            print(f"  {col}: {null_df[col].values[0]} nulls")

# Summary stats
print(f"\nColumn count: {len(df_check.columns)}")
print(f"Total rows: {df_check.count():,}")

# Overall positive rate sanity check (~0.41% expected for this cohort)
pos_rate = df_check.select(F.avg("FUTURE_CRC_EVENT")).collect()[0][0]
print(f"Overall positive rate: {pos_rate*100:.3f}%")

print("\n" + "=" * 60)
print("FEATURIZATION COMPLETE")
print("=" * 60)
print(f"Output: {OUTPUT_TABLE}")
print(f"Features: {len(SELECTED_FEATURES)}")
print(f"Rows: {df_check.count():,}")
print("=" * 60)

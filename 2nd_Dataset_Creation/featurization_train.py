# Databricks notebook source
# MAGIC %md
# MAGIC # Featurization: CRC Risk Prediction (40 Features)
# MAGIC
# MAGIC Builds the full cohort from scratch and engineers 40 selected features from Clarity source tables.
# MAGIC These features were selected by iterative SHAP winnowing (Book 9, iteration 16, test AUPRC=0.1146).
# MAGIC
# MAGIC **Output**: `{trgt_cat}.clncl_ds.fudge_sicle_train`
# MAGIC
# MAGIC **Structure**:
# MAGIC - Cells 1-9: Cohort creation (patients, grid, demographics, PCP, exclusions, screening, labels, splits)
# MAGIC - Cells 10-14: Feature engineering (vitals, ICD10, labs, visits, procedures)
# MAGIC - Cells 15-17: Final join, save, validation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Configuration & Imports

# COMMAND ----------

import os
import datetime
import time
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# Catalog / Schema
# ---------------------------------------------------------------------------
trgt_cat = os.environ.get('trgt_cat', 'dev')
spark.sql(f'USE CATALOG {trgt_cat}')
print(f"Catalog: {trgt_cat}")

# ---------------------------------------------------------------------------
# Date Parameters
# ---------------------------------------------------------------------------
data_collection_date = datetime.datetime(2025, 9, 30)
label_months = 6                   # Predict CRC within 6 months
min_followup_months = 12           # Minimum follow-up to confirm negatives
total_exclusion_months = max(label_months, min_followup_months)

index_start = '2023-01-01'
index_end = (data_collection_date - relativedelta(months=total_exclusion_months)).strftime('%Y-%m-%d')

print(f"Cohort window: {index_start} to {index_end}")

# ---------------------------------------------------------------------------
# ICD Code Pattern for CRC
# ---------------------------------------------------------------------------
crc_icd_regex = r'^(C(?:18|19|20))'  # C18 colon, C19 rectosigmoid, C20 rectum

# ---------------------------------------------------------------------------
# Output Table
# ---------------------------------------------------------------------------
OUTPUT_TABLE = f"{trgt_cat}.clncl_ds.fudge_sicle_train"

# ---------------------------------------------------------------------------
# 40 Selected Features
# ---------------------------------------------------------------------------
SELECTED_FEATURES = [
    "lab_HEMOGLOBIN_ACCELERATING_DECLINE",
    "lab_PLATELETS_ACCELERATING_RISE",
    "vis_visit_recency_last_gi",
    "IS_FEMALE",
    "IS_MARRIED_PARTNER",
    "vit_BP_SYSTOLIC",
    "lab_ALT_AST_RATIO",
    "vit_WEIGHT_OZ",
    "lab_comprehensive_iron_deficiency",
    "lab_PLATELETS_VALUE",
    "lab_THROMBOCYTOSIS_FLAG",
    "vit_MAX_WEIGHT_LOSS_PCT_60D",
    "vit_WEIGHT_CHANGE_PCT_6M",
    "vit_WEIGHT_TRAJECTORY_SLOPE",
    "icd_MALIGNANCY_FLAG_EVER",
    "months_since_cohort_entry",
    "vis_visit_pcp_visits_12mo",
    "vis_visit_outpatient_visits_12mo",
    "vit_vital_recency_score",
    "icd_CHARLSON_SCORE_12MO",
    "vit_RECENCY_WEIGHT",
    "HAS_PCP_AT_END",
    "vis_visit_no_shows_12mo",
    "lab_AST_VALUE",
    "lab_CEA_VALUE",
    "lab_ALK_PHOS_VALUE",
    "vit_SBP_VARIABILITY_6M",
    "vis_visit_gi_symptom_op_visits_12mo",
    "icd_IRON_DEF_ANEMIA_FLAG_12MO",
    "vis_visit_total_gi_symptom_visits_12mo",
    "icd_ANEMIA_FLAG_12MO",
    "vis_visit_gi_symptoms_no_specialist",
    "icd_SYMPTOM_BURDEN_12MO",
    "icd_BLEED_CNT_12MO",
    "proc_total_imaging_count_12mo",
    "proc_ct_abd_pelvis_count_12mo",
    "icd_PAIN_FLAG_12MO",
    "lab_HEMOGLOBIN_VALUE",
    "lab_ALBUMIN_VALUE",
    "vis_visit_acute_care_reliance",
]

print(f"Selected features: {len(SELECTED_FEATURES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Base Patient Identification
# MAGIC
# MAGIC **What This Cell Does**: Identifies all patients with completed encounters in our health system
# MAGIC during the cohort window. Sources both outpatient and inpatient encounters.

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Monthly Observation Grid
# MAGIC
# MAGIC **What This Cell Does**: Creates one observation per patient per month with a deterministic
# MAGIC random day assignment using a hash function for reproducibility.

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Demographics
# MAGIC
# MAGIC **What This Cell Does**: Joins patient demographics, computes age, applies age/quality filters,
# MAGIC and requires 24 months of observability.

# COMMAND ----------

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
  CASE WHEN p.GENDER = 'Female' THEN 1 ELSE 0 END AS IS_FEMALE,
  CASE WHEN p.MARITAL_STATUS IN ('Married', 'Significant other') THEN 1 ELSE 0 END AS IS_MARRIED_PARTNER,
  CAST(months_between(idx.END_DTTM, fs.first_seen_dt) AS INT) AS OBS_MONTHS_PRIOR,
  fs.first_seen_dt,
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: PCP Status
# MAGIC
# MAGIC **What This Cell Does**: Determines if each patient has an active PCP in our integrated
# MAGIC health system at their observation date.

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Medical Exclusions
# MAGIC
# MAGIC **What This Cell Does**: Excludes patient-observations with prior CRC diagnosis,
# MAGIC colectomy, colostomy, or hospice care.

# COMMAND ----------

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

# Apply exclusions
spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_no_exclusions AS

SELECT c.*
FROM cohort_with_pcp c
LEFT ANTI JOIN medical_exclusions e
  ON c.PAT_ID = e.PAT_ID AND c.END_DTTM = e.END_DTTM
""")

after_excl = spark.sql("SELECT COUNT(*) AS n FROM cohort_no_exclusions").collect()[0]['n']
print(f"After medical exclusions: {after_excl:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Screening Exclusions (Dual)
# MAGIC
# MAGIC **What This Cell Does**: Excludes patients who are up-to-date on CRC screening via
# MAGIC both internal procedure tracking and VBC screening registry.

# COMMAND ----------

# Internal screening detection
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

# Apply dual screening exclusion
spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_unscreened AS

SELECT c.*
FROM cohort_no_exclusions c
-- Exclude if internal screening is current
LEFT ANTI JOIN internal_screening ise
  ON c.PAT_ID = ise.PAT_ID AND c.END_DTTM = ise.END_DTTM
-- Exclude if VBC says screened
WHERE c.PAT_ID NOT IN (
  SELECT PAT_ID
  FROM prod.clncl_cur.vbc_colon_cancer_screen
  WHERE COLON_SCREEN_MET_FLAG = 'Y'
    AND COLON_SCREEN_EXCL_FLAG = 'N'
)
""")

unscreened_count = spark.sql("SELECT COUNT(*) AS n FROM cohort_unscreened").collect()[0]['n']
print(f"After screening exclusions: {unscreened_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Label Construction
# MAGIC
# MAGIC **What This Cell Does**: Creates FUTURE_CRC_EVENT label (CRC diagnosis within 6 months)
# MAGIC and LABEL_USABLE filter (three-tier negative label quality system).

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW cohort_labeled AS

WITH future_crc AS (
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

  -- Label
  CASE WHEN fc.PAT_ID IS NOT NULL THEN 1 ELSE 0 END AS FUTURE_CRC_EVENT,

  -- Cancer type (for stratification)
  CASE
    WHEN fc.ICD10_CODE RLIKE '^C18' THEN 'C18'
    WHEN fc.ICD10_CODE RLIKE '^C19' THEN 'C19'
    WHEN fc.ICD10_CODE RLIKE '^C20' THEN 'C20'
    ELSE NULL
  END AS ICD10_GROUP,

  -- Months since cohort entry
  CAST(months_between(c.END_DTTM, pfo.first_obs_date) AS INT) AS months_since_cohort_entry,

  -- Label usability (three-tier system)
  CASE
    WHEN fc.PAT_ID IS NOT NULL THEN 1
    WHEN fc.PAT_ID IS NULL AND nc.next_visit_date > DATEADD(MONTH, {label_months}, c.END_DTTM) THEN 1
    WHEN fc.PAT_ID IS NULL AND c.HAS_PCP_AT_END = 1
     AND nc.next_visit_date > DATEADD(MONTH, 4, c.END_DTTM)
     AND nc.next_visit_date <= DATEADD(MONTH, {label_months}, c.END_DTTM) THEN 1
    WHEN fc.PAT_ID IS NULL AND c.HAS_PCP_AT_END = 1 AND nc.next_visit_date IS NULL THEN 1
    ELSE 0
  END AS LABEL_USABLE

FROM cohort_unscreened c
LEFT JOIN future_crc fc ON c.PAT_ID = fc.PAT_ID AND c.END_DTTM = fc.END_DTTM
LEFT JOIN next_contact nc ON c.PAT_ID = nc.PAT_ID AND c.END_DTTM = nc.END_DTTM
LEFT JOIN patient_first_obs pfo ON c.PAT_ID = pfo.PAT_ID
""")

# Filter to usable labels only
spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_usable AS
SELECT * FROM cohort_labeled WHERE LABEL_USABLE = 1
""")

usable_count = spark.sql("SELECT COUNT(*) AS n FROM cohort_usable").collect()[0]['n']
pos_count = spark.sql("SELECT SUM(FUTURE_CRC_EVENT) AS n FROM cohort_usable").collect()[0]['n']
print(f"Usable observations: {usable_count:,}")
print(f"Positive cases: {pos_count:,} ({pos_count/usable_count*100:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Train/Val/Test Split
# MAGIC
# MAGIC **What This Cell Does**: Patient-level stratified split (70/15/15) preserving cancer type
# MAGIC distribution. Uses random_state=217 for reproducibility.

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Get patient-level labels with cancer type
df_cohort = spark.sql("SELECT * FROM cohort_usable")

patient_labels = df_cohort.groupBy("PAT_ID").agg(
    F.max("FUTURE_CRC_EVENT").alias("is_positive"),
    F.first(F.when(F.col("FUTURE_CRC_EVENT") == 1, F.col("ICD10_GROUP"))).alias("cancer_type")
).toPandas()

# Create multi-class stratification label: 0=negative, 1=C18, 2=C19, 3=C20
cancer_type_map = {'C18': 1, 'C19': 2, 'C20': 3}
patient_labels['strat_label'] = patient_labels.apply(
    lambda row: cancer_type_map.get(row['cancer_type'], 0) if row['is_positive'] == 1 else 0,
    axis=1
)

print(f"Total patients: {len(patient_labels):,}")
print(f"Positive patients: {patient_labels['is_positive'].sum():,}")
print(f"Cancer types: {patient_labels[patient_labels['is_positive']==1]['cancer_type'].value_counts().to_dict()}")

# Split: 70/15/15
np.random.seed(217)

patients_trainval, patients_test = train_test_split(
    patient_labels,
    test_size=0.15,
    stratify=patient_labels['strat_label'],
    random_state=217
)

patients_train, patients_val = train_test_split(
    patients_trainval,
    test_size=0.176,  # 15/85 ≈ 0.176
    stratify=patients_trainval['strat_label'],
    random_state=217
)

train_patients = set(patients_train['PAT_ID'].values)
val_patients = set(patients_val['PAT_ID'].values)
test_patients = set(patients_test['PAT_ID'].values)

# Verify no overlap
assert len(train_patients.intersection(val_patients)) == 0, "TRAIN/VAL overlap!"
assert len(train_patients.intersection(test_patients)) == 0, "TRAIN/TEST overlap!"
assert len(val_patients.intersection(test_patients)) == 0, "VAL/TEST overlap!"

print(f"Train patients: {len(train_patients):,}")
print(f"Val patients:   {len(val_patients):,}")
print(f"Test patients:  {len(test_patients):,}")

# Create SPLIT mapping and join back
train_pdf = pd.DataFrame({'PAT_ID': list(train_patients), 'SPLIT': 'train'})
val_pdf = pd.DataFrame({'PAT_ID': list(val_patients), 'SPLIT': 'val'})
test_pdf = pd.DataFrame({'PAT_ID': list(test_patients), 'SPLIT': 'test'})
split_mapping_pdf = pd.concat([train_pdf, val_pdf, test_pdf], ignore_index=True)

split_mapping_sdf = spark.createDataFrame(split_mapping_pdf)
split_mapping_sdf.createOrReplaceTempView("split_mapping")

spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_base AS
SELECT
  c.PAT_ID,
  c.END_DTTM,
  c.IS_FEMALE,
  c.IS_MARRIED_PARTNER,
  c.HAS_PCP_AT_END,
  c.months_since_cohort_entry,
  c.FUTURE_CRC_EVENT,
  c.ICD10_GROUP,
  sm.SPLIT
FROM cohort_usable c
JOIN split_mapping sm ON c.PAT_ID = sm.PAT_ID
""")

# Verify splits
split_counts = spark.sql("""
  SELECT SPLIT, COUNT(*) AS n, SUM(FUTURE_CRC_EVENT) AS positives,
         ROUND(SUM(FUTURE_CRC_EVENT) / COUNT(*) * 100, 3) AS pos_pct
  FROM cohort_base GROUP BY SPLIT ORDER BY SPLIT
""")
split_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Vitals Features (8 features)
# MAGIC
# MAGIC **What This Cell Does**: Engineers 8 vitals features from pat_enc_enh: latest BP/weight,
# MAGIC weight trajectories, recency scores, and BP variability.

# COMMAND ----------

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
  CASE WHEN CAST(pe.WEIGHT AS DOUBLE) / 16.0 BETWEEN 50 AND 800
       THEN CAST(pe.WEIGHT AS DOUBLE) END AS WEIGHT_OZ
FROM cohort_base c
JOIN clarity_cur.pat_enc_enh pe
  ON pe.PAT_ID = c.PAT_ID
  AND DATE(pe.CONTACT_DATE) < c.END_DTTM
  AND DATE(pe.CONTACT_DATE) >= DATE_SUB(c.END_DTTM, 365)
  AND DATE(pe.CONTACT_DATE) >= DATE('2021-07-01')
  AND pe.APPT_STATUS_C IN (2, 6)
""")

print("Raw vitals extracted")

# COMMAND ----------

# Step 2: Compute all 8 vitals features
spark.sql("""
CREATE OR REPLACE TEMP VIEW vitals_features AS

WITH
-- Latest BP and weight
latest_bp AS (
  SELECT PAT_ID, END_DTTM, BP_SYSTOLIC, MEAS_DATE AS BP_DATE
  FROM (
    SELECT PAT_ID, END_DTTM, BP_SYSTOLIC, MEAS_DATE,
      ROW_NUMBER() OVER (PARTITION BY PAT_ID, END_DTTM ORDER BY MEAS_DATE DESC) AS rn
    FROM vitals_raw
    WHERE BP_SYSTOLIC IS NOT NULL
  ) t WHERE rn = 1
),

latest_weight AS (
  SELECT PAT_ID, END_DTTM, WEIGHT_OZ, MEAS_DATE AS WEIGHT_DATE
  FROM (
    SELECT PAT_ID, END_DTTM, WEIGHT_OZ, MEAS_DATE,
      ROW_NUMBER() OVER (PARTITION BY PAT_ID, END_DTTM ORDER BY MEAS_DATE DESC) AS rn
    FROM vitals_raw
    WHERE WEIGHT_OZ IS NOT NULL
  ) t WHERE rn = 1
),

-- Weight 6 months ago (closest to 180 days prior, within 150-210 day window)
weight_6m AS (
  SELECT PAT_ID, END_DTTM, WEIGHT_OZ AS WEIGHT_OZ_6M
  FROM (
    SELECT PAT_ID, END_DTTM, WEIGHT_OZ,
      ROW_NUMBER() OVER (
        PARTITION BY PAT_ID, END_DTTM
        ORDER BY ABS(DAYS_BEFORE_END - 180)
      ) AS rn
    FROM vitals_raw
    WHERE WEIGHT_OZ IS NOT NULL
      AND DAYS_BEFORE_END BETWEEN 150 AND 210
  ) t WHERE rn = 1
),

-- Weight trajectory (all measurements in 12mo for regression)
weight_trajectory AS (
  SELECT
    PAT_ID,
    END_DTTM,
    REGR_SLOPE(WEIGHT_OZ, DAYS_BEFORE_END) AS WEIGHT_TRAJECTORY_SLOPE
  FROM vitals_raw
  WHERE WEIGHT_OZ IS NOT NULL
  GROUP BY PAT_ID, END_DTTM
  HAVING COUNT(*) >= 2
),

-- Weight changes for max loss in 60 days
weight_changes AS (
  SELECT
    PAT_ID,
    END_DTTM,
    WEIGHT_OZ,
    MEAS_DATE,
    DAYS_BEFORE_END,
    LAG(WEIGHT_OZ) OVER (PARTITION BY PAT_ID, END_DTTM ORDER BY MEAS_DATE) AS PREV_WEIGHT_OZ,
    LAG(MEAS_DATE) OVER (PARTITION BY PAT_ID, END_DTTM ORDER BY MEAS_DATE) AS PREV_MEAS_DATE
  FROM vitals_raw
  WHERE WEIGHT_OZ IS NOT NULL
),

max_weight_loss AS (
  SELECT
    PAT_ID,
    END_DTTM,
    MAX(
      CASE
        WHEN PREV_WEIGHT_OZ IS NOT NULL AND PREV_WEIGHT_OZ > 0
         AND DATEDIFF(MEAS_DATE, PREV_MEAS_DATE) <= 60
        THEN ((PREV_WEIGHT_OZ - WEIGHT_OZ) / PREV_WEIGHT_OZ) * 100
      END
    ) AS MAX_WEIGHT_LOSS_PCT_60D
  FROM weight_changes
  GROUP BY PAT_ID, END_DTTM
),

-- BP variability (6 months)
bp_variability AS (
  SELECT
    PAT_ID,
    END_DTTM,
    STDDEV(BP_SYSTOLIC) AS SBP_VARIABILITY_6M
  FROM vitals_raw
  WHERE BP_SYSTOLIC IS NOT NULL
    AND DAYS_BEFORE_END <= 180
  GROUP BY PAT_ID, END_DTTM
  HAVING COUNT(*) >= 2
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  ROUND(lb.BP_SYSTOLIC, 1) AS vit_BP_SYSTOLIC,
  ROUND(lw.WEIGHT_OZ, 1) AS vit_WEIGHT_OZ,
  DATEDIFF(c.END_DTTM, lw.WEIGHT_DATE) AS vit_RECENCY_WEIGHT,
  CASE
    WHEN lw.WEIGHT_DATE IS NULL THEN 0
    WHEN DATEDIFF(c.END_DTTM, lw.WEIGHT_DATE) <= 30 THEN 3
    WHEN DATEDIFF(c.END_DTTM, lw.WEIGHT_DATE) <= 90 THEN 2
    WHEN DATEDIFF(c.END_DTTM, lw.WEIGHT_DATE) <= 180 THEN 1
    ELSE 0
  END AS vit_vital_recency_score,
  ROUND(wt.WEIGHT_TRAJECTORY_SLOPE, 4) AS vit_WEIGHT_TRAJECTORY_SLOPE,
  ROUND(mwl.MAX_WEIGHT_LOSS_PCT_60D, 2) AS vit_MAX_WEIGHT_LOSS_PCT_60D,
  ROUND(
    CASE
      WHEN lw.WEIGHT_OZ IS NOT NULL AND w6.WEIGHT_OZ_6M IS NOT NULL
      THEN ((lw.WEIGHT_OZ - w6.WEIGHT_OZ_6M) / NULLIF(w6.WEIGHT_OZ_6M, 0)) * 100
    END, 2
  ) AS vit_WEIGHT_CHANGE_PCT_6M,
  ROUND(bv.SBP_VARIABILITY_6M, 2) AS vit_SBP_VARIABILITY_6M

FROM cohort_base c
LEFT JOIN latest_bp lb ON c.PAT_ID = lb.PAT_ID AND c.END_DTTM = lb.END_DTTM
LEFT JOIN latest_weight lw ON c.PAT_ID = lw.PAT_ID AND c.END_DTTM = lw.END_DTTM
LEFT JOIN weight_6m w6 ON c.PAT_ID = w6.PAT_ID AND c.END_DTTM = w6.END_DTTM
LEFT JOIN weight_trajectory wt ON c.PAT_ID = wt.PAT_ID AND c.END_DTTM = wt.END_DTTM
LEFT JOIN max_weight_loss mwl ON c.PAT_ID = mwl.PAT_ID AND c.END_DTTM = mwl.END_DTTM
LEFT JOIN bp_variability bv ON c.PAT_ID = bv.PAT_ID AND c.END_DTTM = bv.END_DTTM
""")

print("Vitals features computed (8 features)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: ICD10 Features (7 features)
# MAGIC
# MAGIC **What This Cell Does**: Engineers 7 ICD10-based features from diagnosis tables:
# MAGIC malignancy flag, Charlson score, anemia flags, symptom burden, bleeding/pain counts.

# COMMAND ----------

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
  SELECT c.PAT_ID, c.END_DTTM, dd.ICD10_CODE AS CODE, DATE(hsp.HOSP_ADMSN_TIME) AS DX_DATE
  FROM cohort_base c
  JOIN clarity_cur.PAT_ENC_HSP_HAR_ENH hsp
    ON hsp.PAT_ID = c.PAT_ID
    AND DATE(hsp.HOSP_ADMSN_TIME) < c.END_DTTM
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE('2021-07-01')
    AND hsp.ADT_PATIENT_STAT_C <> 1
    AND hsp.ADMIT_CONF_STAT_C <> 3
  JOIN clarity_cur.hsp_acct_dx_list_enh dd
    ON dd.PRIM_ENC_CSN_ID = hsp.PAT_ENC_CSN_ID
  WHERE dd.ICD10_CODE IS NOT NULL
),

-- Charlson weights (12mo only)
charlson AS (
  SELECT PAT_ID, END_DTTM, SUM(charlson_wt) AS CHARLSON_SCORE_12MO
  FROM (
    SELECT DISTINCT PAT_ID, END_DTTM,
      CASE
        WHEN CODE RLIKE '^(I21|I22)' THEN 1
        WHEN CODE RLIKE '^I50' THEN 1
        WHEN CODE RLIKE '^(I70|I71|I73)' THEN 1
        WHEN CODE RLIKE '^(I60|I61|I62|I63|I64)' THEN 1
        WHEN CODE RLIKE '^(G30|F01|F03)' THEN 1
        WHEN CODE RLIKE '^J44' THEN 1
        WHEN CODE RLIKE '^(M05|M06|M32|M33|M34)' THEN 1
        WHEN CODE RLIKE '^(K25|K26|K27|K28)' THEN 1
        WHEN CODE RLIKE '^K70' THEN 1
        WHEN CODE RLIKE '^(E10|E11)' THEN 1
        WHEN CODE RLIKE '^(E13|E14)' THEN 2
        WHEN CODE RLIKE '^(G81|G82)' THEN 2
        WHEN CODE RLIKE '^N18' THEN 2
        WHEN CODE RLIKE '^C(?:0[0-9]|[1-8][0-9]|9[0-7])' THEN 2
        WHEN CODE RLIKE '^(K72|K76)' THEN 3
        WHEN CODE RLIKE '^(C78|C79)' THEN 6
        WHEN CODE RLIKE '^B2[0-4]' THEN 6
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

  -- Malignancy flag (ever)
  COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^Z85' THEN 1 END), 0) AS icd_MALIGNANCY_FLAG_EVER,

  -- Iron deficiency anemia (12mo)
  COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^D50' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0)
    AS icd_IRON_DEF_ANEMIA_FLAG_12MO,

  -- Any anemia (12mo)
  COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(D5[0-3]|D6[234])' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0)
    AS icd_ANEMIA_FLAG_12MO,

  -- Bleeding count (12mo)
  COALESCE(SUM(CASE WHEN dx.CODE RLIKE '^(K62\\.5|K92\\.[12])' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 ELSE 0 END), 0)
    AS icd_BLEED_CNT_12MO,

  -- Pain flag (12mo)
  COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^R10' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0)
    AS icd_PAIN_FLAG_12MO,

  -- Symptom burden (sum of 6 binary flags in 12mo)
  (
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(K62\\.5|K92\\.[12])' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^R10' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(K59|R19\\.4)' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^R63\\.4' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^R53' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(D5[0-3]|D6[234])' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0)
  ) AS icd_SYMPTOM_BURDEN_12MO,

  -- Charlson score
  COALESCE(ch.CHARLSON_SCORE_12MO, 0) AS icd_CHARLSON_SCORE_12MO

FROM cohort_base c
LEFT JOIN all_dx dx ON c.PAT_ID = dx.PAT_ID AND c.END_DTTM = dx.END_DTTM
LEFT JOIN charlson ch ON c.PAT_ID = ch.PAT_ID AND c.END_DTTM = ch.END_DTTM
GROUP BY c.PAT_ID, c.END_DTTM, ch.CHARLSON_SCORE_12MO
""")

print("ICD10 features computed (7 features)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12: Labs Features (11 features)
# MAGIC
# MAGIC **What This Cell Does**: Engineers 11 lab features: latest values, ALT/AST ratio,
# MAGIC thrombocytosis flag, acceleration features, and iron deficiency composite.

# COMMAND ----------

# Step 1: Extract latest lab values
spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_latest AS

WITH lab_results AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    cc.NAME AS COMPONENT_NAME,
    TRY_CAST(REGEXP_REPLACE(res.COMPONENT_VALUE, '[><]', '') AS FLOAT) AS VALUE,
    res.COMP_VERIF_DTTM,
    ROW_NUMBER() OVER (
      PARTITION BY c.PAT_ID, c.END_DTTM, cc.NAME
      ORDER BY res.COMP_VERIF_DTTM DESC
    ) AS rn
  FROM cohort_base c
  JOIN clarity_cur.order_proc_enh op
    ON op.PAT_ID = c.PAT_ID
    AND DATE(op.ORDERING_DATE) < c.END_DTTM
    AND DATE(op.ORDERING_DATE) >= DATE_SUB(c.END_DTTM, 730)
    AND DATE(op.ORDERING_DATE) >= DATE('2021-07-01')
    AND op.ORDER_STATUS_C IN (3, 5, 10)
  JOIN clarity.order_results ores
    ON ores.ORDER_PROC_ID = op.ORDER_PROC_ID
  JOIN clarity.res_components res
    ON res.RESULT_ID = ores.RESULT_ID
  JOIN clarity.clarity_component cc
    ON cc.COMPONENT_ID = res.COMPONENT_ID
  WHERE cc.NAME IN ('HEMOGLOBIN', 'PLATELETS', 'AST', 'ALT', 'ALBUMIN',
                     'ALKALINE PHOSPHATASE', 'CEA', 'MCV', 'FERRITIN')
    AND res.COMP_VERIF_DTTM < c.END_DTTM
    AND TRY_CAST(REGEXP_REPLACE(res.COMPONENT_VALUE, '[><]', '') AS FLOAT) IS NOT NULL
)

SELECT
  PAT_ID,
  END_DTTM,
  MAX(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' THEN VALUE END) AS HEMOGLOBIN_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'PLATELETS' THEN VALUE END) AS PLATELETS_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'AST' THEN VALUE END) AS AST_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'ALT' THEN VALUE END) AS ALT_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'ALBUMIN' THEN VALUE END) AS ALBUMIN_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'ALKALINE PHOSPHATASE' THEN VALUE END) AS ALK_PHOS_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'CEA' THEN VALUE END) AS CEA_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'MCV' THEN VALUE END) AS MCV_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'FERRITIN' THEN VALUE END) AS FERRITIN_VALUE
FROM lab_results
WHERE rn = 1
GROUP BY PAT_ID, END_DTTM
""")

print("Latest lab values extracted")

# COMMAND ----------

# Step 2: Lab acceleration features (hemoglobin decline, platelet rise)
spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_acceleration AS

WITH lab_history AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    cc.NAME AS COMPONENT_NAME,
    TRY_CAST(REGEXP_REPLACE(res.COMPONENT_VALUE, '[><]', '') AS FLOAT) AS VALUE,
    DATEDIFF(c.END_DTTM, res.COMP_VERIF_DTTM) AS DAYS_BEFORE
  FROM cohort_base c
  JOIN clarity_cur.order_proc_enh op
    ON op.PAT_ID = c.PAT_ID
    AND DATE(op.ORDERING_DATE) < c.END_DTTM
    AND DATE(op.ORDERING_DATE) >= DATE_SUB(c.END_DTTM, 730)
    AND DATE(op.ORDERING_DATE) >= DATE('2021-07-01')
    AND op.ORDER_STATUS_C IN (3, 5, 10)
  JOIN clarity.order_results ores
    ON ores.ORDER_PROC_ID = op.ORDER_PROC_ID
  JOIN clarity.res_components res
    ON res.RESULT_ID = ores.RESULT_ID
  JOIN clarity.clarity_component cc
    ON cc.COMPONENT_ID = res.COMPONENT_ID
  WHERE cc.NAME IN ('HEMOGLOBIN', 'PLATELETS')
    AND res.COMP_VERIF_DTTM < c.END_DTTM
    AND TRY_CAST(REGEXP_REPLACE(res.COMPONENT_VALUE, '[><]', '') AS FLOAT) IS NOT NULL
),

-- Get values at approximate time points (current, 3mo, 6mo)
time_points AS (
  SELECT
    PAT_ID, END_DTTM, COMPONENT_NAME,
    -- Current value (most recent)
    MAX(CASE WHEN DAYS_BEFORE <= 30 THEN VALUE END) AS current_value,
    -- 3-month prior (60-120 days)
    AVG(CASE WHEN DAYS_BEFORE BETWEEN 60 AND 120 THEN VALUE END) AS value_3mo_prior,
    -- 6-month prior (150-210 days)
    AVG(CASE WHEN DAYS_BEFORE BETWEEN 150 AND 210 THEN VALUE END) AS value_6mo_prior
  FROM lab_history
  GROUP BY PAT_ID, END_DTTM, COMPONENT_NAME
)

SELECT
  PAT_ID,
  END_DTTM,

  -- Hemoglobin accelerating decline
  MAX(CASE
    WHEN COMPONENT_NAME = 'HEMOGLOBIN'
     AND current_value IS NOT NULL
     AND value_3mo_prior IS NOT NULL
     AND value_6mo_prior IS NOT NULL
     AND ((current_value - value_3mo_prior) / 3.0) < -0.5
     AND ((current_value - value_3mo_prior) / 3.0) < ((value_3mo_prior - value_6mo_prior) / 3.0)
    THEN 1 ELSE 0
  END) AS lab_HEMOGLOBIN_ACCELERATING_DECLINE,

  -- Platelets accelerating rise
  MAX(CASE
    WHEN COMPONENT_NAME = 'PLATELETS'
     AND current_value IS NOT NULL
     AND current_value > 450
     AND value_3mo_prior IS NOT NULL
     AND value_6mo_prior IS NOT NULL
     AND ((current_value - value_3mo_prior) / 3.0) > ((value_3mo_prior - value_6mo_prior) / 3.0)
    THEN 1 ELSE 0
  END) AS lab_PLATELETS_ACCELERATING_RISE

FROM time_points
GROUP BY PAT_ID, END_DTTM
""")

print("Lab acceleration features computed")

# COMMAND ----------

# Step 3: Combine all lab features
spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_features AS

SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- Direct values
  ll.HEMOGLOBIN_VALUE AS lab_HEMOGLOBIN_VALUE,
  ll.PLATELETS_VALUE AS lab_PLATELETS_VALUE,
  ll.AST_VALUE AS lab_AST_VALUE,
  ll.ALK_PHOS_VALUE AS lab_ALK_PHOS_VALUE,
  ll.ALBUMIN_VALUE AS lab_ALBUMIN_VALUE,
  ll.CEA_VALUE AS lab_CEA_VALUE,

  -- Derived: ALT/AST ratio
  CASE WHEN ll.AST_VALUE > 0 THEN ROUND(ll.ALT_VALUE / ll.AST_VALUE, 3) END AS lab_ALT_AST_RATIO,

  -- Derived: Thrombocytosis flag
  CASE WHEN ll.PLATELETS_VALUE > 450 THEN 1 ELSE 0 END AS lab_THROMBOCYTOSIS_FLAG,

  -- Acceleration features
  COALESCE(la.lab_HEMOGLOBIN_ACCELERATING_DECLINE, 0) AS lab_HEMOGLOBIN_ACCELERATING_DECLINE,
  COALESCE(la.lab_PLATELETS_ACCELERATING_RISE, 0) AS lab_PLATELETS_ACCELERATING_RISE,

  -- Lab-only iron deficiency component (IDA diagnosis flag added in final join)
  CASE
    WHEN (ll.HEMOGLOBIN_VALUE < 12 AND ll.MCV_VALUE < 80) THEN 1
    WHEN (ll.FERRITIN_VALUE < 30 AND ll.HEMOGLOBIN_VALUE < 13) THEN 1
    ELSE 0
  END AS lab_iron_deficiency_labs_only

FROM cohort_base c
LEFT JOIN lab_latest ll ON c.PAT_ID = ll.PAT_ID AND c.END_DTTM = ll.END_DTTM
LEFT JOIN lab_acceleration la ON c.PAT_ID = la.PAT_ID AND c.END_DTTM = la.END_DTTM
""")

print("Lab features combined (11 features)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 13: Visit Features (9 features)
# MAGIC
# MAGIC **What This Cell Does**: Engineers 9 visit features: PCP/outpatient visit counts, no-shows,
# MAGIC GI symptom visits, GI specialist recency, care gap indicator, acute care reliance.

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TEMP VIEW visit_features AS

WITH
-- Outpatient encounters (12mo)
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

-- GI symptom encounters - outpatient (encounters with GI-related diagnosis)
gi_symptom_op AS (
  SELECT DISTINCT o.PAT_ID, o.END_DTTM, o.PAT_ENC_CSN_ID
  FROM outpatient o
  JOIN clarity_cur.pat_enc_dx_enh dx
    ON dx.PAT_ENC_CSN_ID = o.PAT_ENC_CSN_ID
  WHERE dx.ICD10_CODE RLIKE '^(K62\\.5|K92|K59|R19|D50|R10|R63\\.4|R53)'
    AND o.APPT_STATUS_C IN (2, 6)
),

-- GI symptom encounters - inpatient/ED
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
  WHERE dx.ICD10_CODE RLIKE '^(K62\\.5|K92|K59|R19|D50|R10|R63\\.4|R53)'
),

-- ED/Inpatient encounters (12mo)
acute_care AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    COUNT(DISTINCT hsp.PAT_ENC_CSN_ID) AS ed_inp_visits
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

-- GI specialist recency (any time in lookback)
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

-- Aggregate outpatient metrics
op_metrics AS (
  SELECT
    PAT_ID,
    END_DTTM,

    -- Completed outpatient visits
    SUM(CASE WHEN APPT_STATUS_C IN (2, 6) THEN 1 ELSE 0 END) AS outpatient_visits_12mo,

    -- PCP visits
    SUM(CASE WHEN APPT_STATUS_C IN (2, 6)
         AND (UPPER(SPECIALTY) LIKE '%PRIMARY%' OR UPPER(SPECIALTY) LIKE '%FAMILY%'
              OR UPPER(SPECIALTY) LIKE '%INTERNAL MED%')
         THEN 1 ELSE 0 END) AS pcp_visits_12mo,

    -- No-shows
    SUM(CASE WHEN APPT_STATUS_C IN (3, 4) THEN 1 ELSE 0 END) AS no_shows_12mo,

    -- GI specialist visits
    SUM(CASE WHEN APPT_STATUS_C IN (2, 6) AND UPPER(SPECIALTY) LIKE '%GASTRO%'
         THEN 1 ELSE 0 END) AS gi_visits_12mo

  FROM outpatient
  GROUP BY PAT_ID, END_DTTM
),

-- GI symptom visit counts (outpatient only)
gi_symptom_op_counts AS (
  SELECT PAT_ID, END_DTTM, COUNT(DISTINCT PAT_ENC_CSN_ID) AS gi_symptom_op_visits_12mo
  FROM gi_symptom_op
  GROUP BY PAT_ID, END_DTTM
),

-- GI symptom visit counts (inpatient/ED)
gi_symptom_acute_counts AS (
  SELECT PAT_ID, END_DTTM, COUNT(DISTINCT PAT_ENC_CSN_ID) AS gi_symptom_acute_visits_12mo
  FROM gi_symptom_acute
  GROUP BY PAT_ID, END_DTTM
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  COALESCE(gr.days_since_last_gi, 9999) AS vis_visit_recency_last_gi,
  COALESCE(om.pcp_visits_12mo, 0) AS vis_visit_pcp_visits_12mo,
  COALESCE(om.outpatient_visits_12mo, 0) AS vis_visit_outpatient_visits_12mo,
  COALESCE(om.no_shows_12mo, 0) AS vis_visit_no_shows_12mo,
  COALESCE(gsop.gi_symptom_op_visits_12mo, 0) AS vis_visit_gi_symptom_op_visits_12mo,
  -- Total GI symptom visits (outpatient + inpatient/ED)
  COALESCE(gsop.gi_symptom_op_visits_12mo, 0) + COALESCE(gsac.gi_symptom_acute_visits_12mo, 0)
    AS vis_visit_total_gi_symptom_visits_12mo,
  -- GI symptoms but no specialist
  CASE
    WHEN (COALESCE(gsop.gi_symptom_op_visits_12mo, 0) + COALESCE(gsac.gi_symptom_acute_visits_12mo, 0)) > 0
     AND COALESCE(om.gi_visits_12mo, 0) = 0
    THEN 1 ELSE 0
  END AS vis_visit_gi_symptoms_no_specialist,
  -- Acute care reliance ratio
  CASE
    WHEN COALESCE(om.outpatient_visits_12mo, 0) > 0
    THEN ROUND(COALESCE(ac.ed_inp_visits, 0) * 1.0 / om.outpatient_visits_12mo, 3)
    ELSE COALESCE(ac.ed_inp_visits, 0) * 1.0
  END AS vis_visit_acute_care_reliance

FROM cohort_base c
LEFT JOIN op_metrics om ON c.PAT_ID = om.PAT_ID AND c.END_DTTM = om.END_DTTM
LEFT JOIN gi_symptom_op_counts gsop ON c.PAT_ID = gsop.PAT_ID AND c.END_DTTM = gsop.END_DTTM
LEFT JOIN gi_symptom_acute_counts gsac ON c.PAT_ID = gsac.PAT_ID AND c.END_DTTM = gsac.END_DTTM
LEFT JOIN acute_care ac ON c.PAT_ID = ac.PAT_ID AND c.END_DTTM = ac.END_DTTM
LEFT JOIN gi_recency gr ON c.PAT_ID = gr.PAT_ID AND c.END_DTTM = gr.END_DTTM
""")

print("Visit features computed (9 features)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 14: Procedure Features (2 features)
# MAGIC
# MAGIC **What This Cell Does**: Engineers 2 procedure features: CT abdomen/pelvis count
# MAGIC and total imaging count (CT + MRI) in past 12 months.

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TEMP VIEW proc_features AS

SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- CT abdomen/pelvis count (12mo)
  COALESCE(SUM(CASE
    WHEN op.PROC_CODE IN ('74150','74160','74170','74176','74177','74178')
      OR LOWER(op.PROC_NAME) LIKE '%ct%abd%'
      OR LOWER(op.PROC_NAME) LIKE '%ct%pelv%'
    THEN 1 ELSE 0
  END), 0) AS proc_ct_abd_pelvis_count_12mo,

  -- Total imaging count: CT + MRI abdomen/pelvis (12mo)
  COALESCE(SUM(CASE
    WHEN op.PROC_CODE IN ('74150','74160','74170','74176','74177','74178')
      OR LOWER(op.PROC_NAME) LIKE '%ct%abd%'
      OR LOWER(op.PROC_NAME) LIKE '%ct%pelv%'
      OR op.PROC_CODE IN ('74181','74182','74183')
      OR LOWER(op.PROC_NAME) LIKE '%mri%abd%'
      OR LOWER(op.PROC_NAME) LIKE '%mri%pelv%'
    THEN 1 ELSE 0
  END), 0) AS proc_total_imaging_count_12mo

FROM cohort_base c
LEFT JOIN clarity_cur.order_proc_enh op
  ON op.PAT_ID = c.PAT_ID
  AND DATE(op.ORDERING_DATE) < c.END_DTTM
  AND DATE(op.ORDERING_DATE) >= DATE_SUB(c.END_DTTM, 365)
  AND DATE(op.ORDERING_DATE) >= DATE('2021-07-01')
  AND op.ORDER_STATUS_C = 5  -- Completed
  AND op.RPT_GRP_SIX IN ('116001', '116002')
GROUP BY c.PAT_ID, c.END_DTTM
""")

print("Procedure features computed (2 features)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 15: Final Join
# MAGIC
# MAGIC **What This Cell Does**: Joins all feature views together and selects the final 40 features
# MAGIC plus identifiers. Fills nulls with 0 for numeric features.

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE TEMP VIEW final_features AS

SELECT
  c.PAT_ID,
  c.END_DTTM,
  c.FUTURE_CRC_EVENT,
  c.SPLIT,

  -- Demographics (4)
  c.IS_FEMALE,
  c.IS_MARRIED_PARTNER,
  c.HAS_PCP_AT_END,
  COALESCE(c.months_since_cohort_entry, 0) AS months_since_cohort_entry,

  -- Vitals (8)
  COALESCE(v.vit_BP_SYSTOLIC, 0) AS vit_BP_SYSTOLIC,
  COALESCE(v.vit_WEIGHT_OZ, 0) AS vit_WEIGHT_OZ,
  COALESCE(v.vit_RECENCY_WEIGHT, 9999) AS vit_RECENCY_WEIGHT,
  COALESCE(v.vit_vital_recency_score, 0) AS vit_vital_recency_score,
  COALESCE(v.vit_WEIGHT_TRAJECTORY_SLOPE, 0) AS vit_WEIGHT_TRAJECTORY_SLOPE,
  COALESCE(v.vit_MAX_WEIGHT_LOSS_PCT_60D, 0) AS vit_MAX_WEIGHT_LOSS_PCT_60D,
  COALESCE(v.vit_WEIGHT_CHANGE_PCT_6M, 0) AS vit_WEIGHT_CHANGE_PCT_6M,
  COALESCE(v.vit_SBP_VARIABILITY_6M, 0) AS vit_SBP_VARIABILITY_6M,

  -- ICD10 (7)
  COALESCE(i.icd_MALIGNANCY_FLAG_EVER, 0) AS icd_MALIGNANCY_FLAG_EVER,
  COALESCE(i.icd_CHARLSON_SCORE_12MO, 0) AS icd_CHARLSON_SCORE_12MO,
  COALESCE(i.icd_IRON_DEF_ANEMIA_FLAG_12MO, 0) AS icd_IRON_DEF_ANEMIA_FLAG_12MO,
  COALESCE(i.icd_ANEMIA_FLAG_12MO, 0) AS icd_ANEMIA_FLAG_12MO,
  COALESCE(i.icd_SYMPTOM_BURDEN_12MO, 0) AS icd_SYMPTOM_BURDEN_12MO,
  COALESCE(i.icd_BLEED_CNT_12MO, 0) AS icd_BLEED_CNT_12MO,
  COALESCE(i.icd_PAIN_FLAG_12MO, 0) AS icd_PAIN_FLAG_12MO,

  -- Labs (11)
  COALESCE(l.lab_HEMOGLOBIN_VALUE, 0) AS lab_HEMOGLOBIN_VALUE,
  COALESCE(l.lab_PLATELETS_VALUE, 0) AS lab_PLATELETS_VALUE,
  COALESCE(l.lab_AST_VALUE, 0) AS lab_AST_VALUE,
  COALESCE(l.lab_ALK_PHOS_VALUE, 0) AS lab_ALK_PHOS_VALUE,
  COALESCE(l.lab_ALBUMIN_VALUE, 0) AS lab_ALBUMIN_VALUE,
  COALESCE(l.lab_CEA_VALUE, 0) AS lab_CEA_VALUE,
  COALESCE(l.lab_ALT_AST_RATIO, 0) AS lab_ALT_AST_RATIO,
  COALESCE(l.lab_THROMBOCYTOSIS_FLAG, 0) AS lab_THROMBOCYTOSIS_FLAG,
  CASE
    WHEN COALESCE(i.icd_IRON_DEF_ANEMIA_FLAG_12MO, 0) = 1 THEN 1
    WHEN COALESCE(l.lab_iron_deficiency_labs_only, 0) = 1 THEN 1
    ELSE 0
  END AS lab_comprehensive_iron_deficiency,
  COALESCE(l.lab_HEMOGLOBIN_ACCELERATING_DECLINE, 0) AS lab_HEMOGLOBIN_ACCELERATING_DECLINE,
  COALESCE(l.lab_PLATELETS_ACCELERATING_RISE, 0) AS lab_PLATELETS_ACCELERATING_RISE,

  -- Visits (9)
  COALESCE(vis.vis_visit_recency_last_gi, 9999) AS vis_visit_recency_last_gi,
  COALESCE(vis.vis_visit_pcp_visits_12mo, 0) AS vis_visit_pcp_visits_12mo,
  COALESCE(vis.vis_visit_outpatient_visits_12mo, 0) AS vis_visit_outpatient_visits_12mo,
  COALESCE(vis.vis_visit_no_shows_12mo, 0) AS vis_visit_no_shows_12mo,
  COALESCE(vis.vis_visit_gi_symptom_op_visits_12mo, 0) AS vis_visit_gi_symptom_op_visits_12mo,
  COALESCE(vis.vis_visit_total_gi_symptom_visits_12mo, 0) AS vis_visit_total_gi_symptom_visits_12mo,
  COALESCE(vis.vis_visit_gi_symptoms_no_specialist, 0) AS vis_visit_gi_symptoms_no_specialist,
  COALESCE(vis.vis_visit_acute_care_reliance, 0) AS vis_visit_acute_care_reliance,

  -- Procedures (2)
  COALESCE(p.proc_total_imaging_count_12mo, 0) AS proc_total_imaging_count_12mo,
  COALESCE(p.proc_ct_abd_pelvis_count_12mo, 0) AS proc_ct_abd_pelvis_count_12mo

FROM cohort_base c
LEFT JOIN vitals_features v ON c.PAT_ID = v.PAT_ID AND c.END_DTTM = v.END_DTTM
LEFT JOIN icd_features i ON c.PAT_ID = i.PAT_ID AND c.END_DTTM = i.END_DTTM
LEFT JOIN lab_features l ON c.PAT_ID = l.PAT_ID AND c.END_DTTM = l.END_DTTM
LEFT JOIN visit_features vis ON c.PAT_ID = vis.PAT_ID AND c.END_DTTM = vis.END_DTTM
LEFT JOIN proc_features p ON c.PAT_ID = p.PAT_ID AND c.END_DTTM = p.END_DTTM
""")

final_count = spark.sql("SELECT COUNT(*) AS n FROM final_features").collect()[0]['n']
print(f"Final feature table rows: {final_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 16: Save Output Table
# MAGIC
# MAGIC **What This Cell Does**: Saves the final feature table as a Delta table.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {OUTPUT_TABLE} AS
SELECT * FROM final_features
""")

saved_count = spark.table(OUTPUT_TABLE).count()
print(f"Saved to: {OUTPUT_TABLE}")
print(f"Rows: {saved_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 17: Validation & Summary
# MAGIC
# MAGIC **What This Cell Does**: Validates the output table and prints summary statistics.

# COMMAND ----------

df_check = spark.table(OUTPUT_TABLE)

# Split distribution
print("Split Distribution:")
df_check.groupBy("SPLIT").agg(
    F.count("*").alias("n"),
    F.sum("FUTURE_CRC_EVENT").alias("positives"),
    F.round(F.sum("FUTURE_CRC_EVENT") / F.count("*") * 100, 3).alias("positive_pct")
).orderBy("SPLIT").show()

# Feature null counts (should all be 0 after COALESCE)
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

# Summary stats for numeric features
print(f"\nColumn count: {len(df_check.columns)}")
print(f"Total rows: {df_check.count():,}")

# Positive rate sanity check
pos_rate = df_check.select(F.avg("FUTURE_CRC_EVENT")).collect()[0][0]
print(f"Overall positive rate: {pos_rate*100:.3f}%")

print("\n" + "="*60)
print("FEATURIZATION COMPLETE")
print("="*60)
print(f"Output: {OUTPUT_TABLE}")
print(f"Features: {len(SELECTED_FEATURES)}")
print(f"Rows: {df_check.count():,}")
print("="*60)

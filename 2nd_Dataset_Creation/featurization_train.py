# ===========================================================================
# featurization_train.py
#
# CRC Risk Prediction: Training Dataset Pipeline (Books 0-8 Consolidated)
#
# Builds the full cohort from scratch and engineers 49 selected features
# from Clarity source tables. These features were selected by iterative
# SHAP winnowing (Book 9) cross-referenced with 5-fold CV stability.
#
# CEA, CA 19-9, and FOBT/FIT are excluded (circular reasoning).
#
# Output: {trgt_cat}.clncl_ds.herald_train_wide
#
# Pipeline:
#   Section 1: Configuration
#   Section 2: Book 0 — Cohort Creation → herald_train_final_cohort
#   Section 3: Book 1 — Vitals → herald_train_vitals
#   Section 4: Book 2 — ICD-10 → herald_train_icd10
#   Section 5: Book 4 — Labs → herald_train_labs
#   Section 6: Book 5.2 — Inpatient Meds → herald_train_inpatient_meds
#   Section 7: Book 6 — Visits → herald_train_visits
#   Section 8: Book 7 — Procedures → herald_train_procedures
#   Section 9: Book 8 — Compilation → herald_train_wide
#
# Requires: PySpark (Databricks), numpy, pandas, sklearn
# ===========================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import functions as F


# ===========================================================================
# SECTION 1: CONFIGURATION
# ===========================================================================

trgt_cat = os.environ.get('trgt_cat', 'dev')
spark.sql('USE CATALOG prod')
print(f"Read catalog: prod | Write catalog: {trgt_cat}")

# Fixed study period matching Book 0
index_start = '2023-01-01'
index_end = '2024-09-30'

label_months = 6
min_followup_months = 12
crc_icd_regex = r'^(C(?:18|19|20))'

SELECTED_FEATURES = [
    # Demographics (6)
    "AGE_GROUP",
    "HAS_PCP_AT_END",
    "IS_FEMALE",
    "IS_MARRIED_PARTNER",
    "RACE_CAUCASIAN",
    "RACE_HISPANIC",
    # Temporal (1)
    "months_since_cohort_entry",
    # ICD-10 Diagnoses (6)
    "icd_BLEED_CNT_12MO",
    "icd_FHX_CRC_COMBINED",
    "icd_HIGH_RISK_HISTORY",
    "icd_IRON_DEF_ANEMIA_FLAG_12MO",
    "icd_SYMPTOM_BURDEN_12MO",
    "icd_chronic_gi_pattern",
    # Laboratory Values (11)
    "lab_ALBUMIN_DROP_15PCT_FLAG",
    "lab_ALBUMIN_VALUE",
    "lab_ANEMIA_GRADE",
    "lab_ANEMIA_SEVERITY_SCORE",
    "lab_CRP_6MO_CHANGE",
    "lab_HEMOGLOBIN_ACCELERATING_DECLINE",
    "lab_IRON_SATURATION_PCT",
    "lab_PLATELETS_ACCELERATING_RISE",
    "lab_PLATELETS_VALUE",
    "lab_THROMBOCYTOSIS_FLAG",
    "lab_comprehensive_iron_deficiency",
    # Inpatient Medications (5)
    "inp_med_inp_any_hospitalization",
    "inp_med_inp_gi_bleed_meds_recency",
    "inp_med_inp_ibd_meds_recency",
    "inp_med_inp_laxative_use_flag",
    "inp_med_inp_opioid_use_flag",
    # Visit History (7)
    "visit_gi_symptom_op_visits_12mo",
    "visit_gi_symptoms_no_specialist",
    "visit_no_shows_12mo",
    "visit_outpatient_visits_12mo",
    "visit_primary_care_continuity_ratio",
    "visit_recency_last_gi",
    "visit_total_gi_symptom_visits_12mo",
    # Procedures (2)
    "proc_blood_transfusion_count_12mo",
    "proc_total_imaging_count_12mo",
    # Vitals (11)
    "vit_BMI",
    "vit_MAX_WEIGHT_LOSS_PCT_60D",
    "vit_PULSE",
    "vit_PULSE_PRESSURE",
    "vit_RECENCY_WEIGHT",
    "vit_SBP_VARIABILITY_6M",
    "vit_UNDERWEIGHT_FLAG",
    "vit_WEIGHT_CHANGE_PCT_6M",
    "vit_WEIGHT_OZ",
    "vit_WEIGHT_TRAJECTORY_SLOPE",
    "vit_vital_recency_score",
]

print(f"Target features: {len(SELECTED_FEATURES)}")
print(f"Cohort window: {index_start} to {index_end}")
print("=" * 70)


# ===========================================================================
# SECTION 2: BOOK 0 — COHORT CREATION → herald_train_final_cohort
# ===========================================================================

# --- 2a. Base patient identification ---

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW base_patients AS

SELECT DISTINCT pe.PAT_ID
FROM clarity_cur.PAT_ENC_ENH pe
JOIN clarity_cur.DEP_LOC_PLOC_SA_ENH dep
  ON dep.department_id = COALESCE(pe.DEPARTMENT_ID, pe.EFFECTIVE_DEPT_ID)
WHERE pe.CONTACT_DATE >= '{index_start}'
  AND pe.CONTACT_DATE < '{index_end}'
  AND pe.APPT_STATUS_C IN (2, 6)
  AND dep.RPT_GRP_SIX IN ('116001', '116002')

UNION

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
print(f"Base patients: {patient_count:,}")


# --- 2b. Monthly observation grid ---

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


# --- 2c. Demographics ---

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
  CASE WHEN p.RACE = 'Caucasian' THEN 1 ELSE 0 END AS RACE_CAUCASIAN,
  CASE WHEN p.RACE = 'Hispanic' THEN 1 ELSE 0 END AS RACE_HISPANIC,
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
print(f"After demographics filters: {demo_count:,}")


# --- 2d. PCP status ---

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

print("PCP status computed")


# --- 2e. Medical exclusions ---

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

spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_no_exclusions AS
SELECT c.*
FROM cohort_with_pcp c
LEFT ANTI JOIN medical_exclusions e
  ON c.PAT_ID = e.PAT_ID AND c.END_DTTM = e.END_DTTM
""")

after_excl = spark.sql("SELECT COUNT(*) AS n FROM cohort_no_exclusions").collect()[0]['n']
print(f"After medical exclusions: {after_excl:,}")


# --- 2f. Screening exclusions ---

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW internal_screening AS

SELECT
  c.PAT_ID,
  c.END_DTTM
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

spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_unscreened AS
SELECT c.*
FROM cohort_no_exclusions c
LEFT ANTI JOIN internal_screening ise
  ON c.PAT_ID = ise.PAT_ID AND c.END_DTTM = ise.END_DTTM
WHERE c.PAT_ID NOT IN (
  SELECT PAT_ID
  FROM prod.clncl_cur.vbc_colon_cancer_screen
  WHERE COLON_SCREEN_MET_FLAG = 'Y'
    AND COLON_SCREEN_EXCL_FLAG = 'N'
)
""")

unscreened_count = spark.sql("SELECT COUNT(*) AS n FROM cohort_unscreened").collect()[0]['n']
print(f"After screening exclusions: {unscreened_count:,}")


# --- 2g. Label construction ---

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
  c.RACE_CAUCASIAN,
  c.RACE_HISPANIC,
  c.HAS_PCP_AT_END,
  c.first_seen_dt,
  CASE WHEN fc.PAT_ID IS NOT NULL THEN 1 ELSE 0 END AS FUTURE_CRC_EVENT,
  CASE
    WHEN fc.ICD10_CODE RLIKE '^C18' THEN 'C18'
    WHEN fc.ICD10_CODE RLIKE '^C19' THEN 'C19'
    WHEN fc.ICD10_CODE RLIKE '^C20' THEN 'C20'
    ELSE NULL
  END AS ICD10_GROUP,
  CAST(months_between(c.END_DTTM, pfo.first_obs_date) AS INT) AS months_since_cohort_entry,
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

spark.sql("""
CREATE OR REPLACE TEMP VIEW cohort_base AS
SELECT
  PAT_ID, END_DTTM, AGE, IS_FEMALE, IS_MARRIED_PARTNER,
  RACE_CAUCASIAN, RACE_HISPANIC, HAS_PCP_AT_END,
  months_since_cohort_entry, FUTURE_CRC_EVENT, ICD10_GROUP
FROM cohort_labeled
WHERE LABEL_USABLE = 1
""")

base_count = spark.sql("SELECT COUNT(*) AS n FROM cohort_base").collect()[0]['n']
pos_count = spark.sql("SELECT SUM(FUTURE_CRC_EVENT) AS n FROM cohort_base").collect()[0]['n']
print(f"Cohort base: {base_count:,} rows, {pos_count:,} positives ({pos_count/base_count*100:.3f}%)")


# --- 2h. Train/val/test split ---

print("\nComputing stratified patient-level split (70/15/15)...")

df_cohort = spark.sql("SELECT * FROM cohort_base")

patient_labels = df_cohort.groupBy("PAT_ID").agg(
    F.max("FUTURE_CRC_EVENT").alias("is_positive"),
    F.first(
        F.when(F.col("FUTURE_CRC_EVENT") == 1, F.col("ICD10_GROUP"))
    ).alias("cancer_type")
).toPandas()

cancer_type_map = {'C18': 1, 'C19': 2, 'C20': 3}
patient_labels['strat_label'] = patient_labels.apply(
    lambda row: cancer_type_map.get(row['cancer_type'], 0) if row['is_positive'] == 1 else 0,
    axis=1
)

print(f"Total patients: {len(patient_labels):,}")

np.random.seed(217)

# Step 1: Split off TEST (15%)
patients_trainval, patients_test = train_test_split(
    patient_labels,
    test_size=0.15,
    stratify=patient_labels['strat_label'],
    random_state=217
)

# Step 2: Split TRAIN+VAL into TRAIN (~82%) and VAL (~18%)
patients_train, patients_val = train_test_split(
    patients_trainval,
    test_size=0.176,
    stratify=patients_trainval['strat_label'],
    random_state=217
)

print(f"  TRAIN: {len(patients_train):,} patients ({len(patients_train)/len(patient_labels)*100:.1f}%)")
print(f"  VAL:   {len(patients_val):,} patients ({len(patients_val)/len(patient_labels)*100:.1f}%)")
print(f"  TEST:  {len(patients_test):,} patients ({len(patients_test)/len(patient_labels)*100:.1f}%)")

# Verify no overlap
train_pats = set(patients_train['PAT_ID'].values)
val_pats = set(patients_val['PAT_ID'].values)
test_pats = set(patients_test['PAT_ID'].values)
assert len(train_pats.intersection(val_pats)) == 0, "TRAIN/VAL overlap!"
assert len(train_pats.intersection(test_pats)) == 0, "TRAIN/TEST overlap!"
assert len(val_pats.intersection(test_pats)) == 0, "VAL/TEST overlap!"
print("No patient overlap between splits")

# Create split mapping and join to Spark DataFrame
train_pdf = pd.DataFrame({'PAT_ID': list(train_pats), 'SPLIT': 'train'})
val_pdf = pd.DataFrame({'PAT_ID': list(val_pats), 'SPLIT': 'val'})
test_pdf = pd.DataFrame({'PAT_ID': list(test_pats), 'SPLIT': 'test'})
split_mapping_pdf = pd.concat([train_pdf, val_pdf, test_pdf], ignore_index=True)

split_mapping_sdf = spark.createDataFrame(split_mapping_pdf)
df_with_split = df_cohort.join(split_mapping_sdf, on="PAT_ID", how="left")

null_split = df_with_split.filter(F.col("SPLIT").isNull()).count()
if null_split > 0:
    print(f"WARNING: {null_split} observations have NULL SPLIT!")
else:
    print("All observations have SPLIT assigned")


# --- 2i. Save cohort to herald_train_final_cohort ---

df_with_split.write.mode("overwrite").saveAsTable(
    f"{trgt_cat}.clncl_ds.herald_train_final_cohort"
)

cohort_saved = spark.table(f"{trgt_cat}.clncl_ds.herald_train_final_cohort").count()
print(f"Saved cohort: {cohort_saved:,} rows to {trgt_cat}.clncl_ds.herald_train_final_cohort")

# Print split distribution
spark.table(f"{trgt_cat}.clncl_ds.herald_train_final_cohort").groupBy("SPLIT").agg(
    F.count("*").alias("n_obs"),
    F.countDistinct("PAT_ID").alias("n_patients"),
    F.mean("FUTURE_CRC_EVENT").alias("event_rate")
).orderBy("SPLIT").show()


# ===========================================================================
# SECTION 3: BOOK 1 — VITALS → herald_train_vitals (11 features)
#
# vit_BMI, vit_PULSE, vit_PULSE_PRESSURE, vit_RECENCY_WEIGHT,
# vit_SBP_VARIABILITY_6M, vit_UNDERWEIGHT_FLAG, vit_WEIGHT_CHANGE_PCT_6M,
# vit_WEIGHT_OZ, vit_WEIGHT_TRAJECTORY_SLOPE, vit_MAX_WEIGHT_LOSS_PCT_60D,
# vit_vital_recency_score
# ===========================================================================

print("\n" + "=" * 70)
print("SECTION 3: VITALS")
print("=" * 70)

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

spark.sql("""
CREATE OR REPLACE TEMP VIEW vitals_features AS

WITH
latest_vitals AS (
  SELECT PAT_ID, END_DTTM, BP_SYSTOLIC, BP_DIASTOLIC, PULSE, BMI,
         WEIGHT_OZ, MEAS_DATE, DAYS_BEFORE_END
  FROM (
    SELECT PAT_ID, END_DTTM, BP_SYSTOLIC, BP_DIASTOLIC, PULSE, BMI,
           WEIGHT_OZ, MEAS_DATE, DAYS_BEFORE_END,
      ROW_NUMBER() OVER (PARTITION BY PAT_ID, END_DTTM ORDER BY MEAS_DATE DESC) AS rn
    FROM vitals_raw
    WHERE (BP_SYSTOLIC IS NOT NULL OR WEIGHT_OZ IS NOT NULL OR BMI IS NOT NULL)
  ) t WHERE rn = 1
),

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

weight_trajectory AS (
  SELECT
    PAT_ID, END_DTTM,
    REGR_SLOPE(WEIGHT_OZ, DAYS_BEFORE_END) AS WEIGHT_TRAJECTORY_SLOPE
  FROM vitals_raw
  WHERE WEIGHT_OZ IS NOT NULL
  GROUP BY PAT_ID, END_DTTM
  HAVING COUNT(*) >= 2
),

weight_changes AS (
  SELECT PAT_ID, END_DTTM, WEIGHT_OZ, MEAS_DATE,
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
  lv.BMI AS vit_BMI,
  lv.PULSE AS vit_PULSE,
  CASE WHEN lv.BP_SYSTOLIC IS NOT NULL AND lv.BP_DIASTOLIC IS NOT NULL
    THEN lv.BP_SYSTOLIC - lv.BP_DIASTOLIC END AS vit_PULSE_PRESSURE,
  lv.WEIGHT_OZ AS vit_WEIGHT_OZ,
  CASE WHEN lv.BMI IS NOT NULL AND lv.BMI < 18.5 THEN 1 ELSE 0 END AS vit_UNDERWEIGHT_FLAG,
  -- Ordinal recency encoding (Book 8 transformation)
  CASE
    WHEN lv.DAYS_BEFORE_END IS NULL THEN 0
    WHEN lv.DAYS_BEFORE_END <= 30 THEN 5
    WHEN lv.DAYS_BEFORE_END <= 90 THEN 4
    WHEN lv.DAYS_BEFORE_END <= 180 THEN 3
    WHEN lv.DAYS_BEFORE_END <= 365 THEN 2
    ELSE 1
  END AS vit_RECENCY_WEIGHT,
  CASE
    WHEN lv.WEIGHT_OZ IS NOT NULL AND w6.WEIGHT_OZ_6M IS NOT NULL AND w6.WEIGHT_OZ_6M > 0
    THEN ROUND(((lv.WEIGHT_OZ - w6.WEIGHT_OZ_6M) / w6.WEIGHT_OZ_6M) * 100, 2)
  END AS vit_WEIGHT_CHANGE_PCT_6M,
  wt.WEIGHT_TRAJECTORY_SLOPE AS vit_WEIGHT_TRAJECTORY_SLOPE,
  mwl.MAX_WEIGHT_LOSS_PCT_60D AS vit_MAX_WEIGHT_LOSS_PCT_60D,
  bv.SBP_VARIABILITY_6M AS vit_SBP_VARIABILITY_6M,
  -- Vital recency score: 0-3 based on days since last weight (Book 1)
  CASE
    WHEN lv.DAYS_BEFORE_END IS NULL THEN 0
    WHEN lv.DAYS_BEFORE_END <= 30 THEN 3
    WHEN lv.DAYS_BEFORE_END <= 90 THEN 2
    WHEN lv.DAYS_BEFORE_END <= 180 THEN 1
    ELSE 0
  END AS vit_vital_recency_score

FROM cohort_base c
LEFT JOIN latest_vitals lv ON c.PAT_ID = lv.PAT_ID AND c.END_DTTM = lv.END_DTTM
LEFT JOIN weight_6m w6 ON c.PAT_ID = w6.PAT_ID AND c.END_DTTM = w6.END_DTTM
LEFT JOIN weight_trajectory wt ON c.PAT_ID = wt.PAT_ID AND c.END_DTTM = wt.END_DTTM
LEFT JOIN max_weight_loss mwl ON c.PAT_ID = mwl.PAT_ID AND c.END_DTTM = mwl.END_DTTM
LEFT JOIN bp_variability bv ON c.PAT_ID = bv.PAT_ID AND c.END_DTTM = bv.END_DTTM
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_train_vitals AS
SELECT * FROM vitals_features
""")

vit_count = spark.table(f"{trgt_cat}.clncl_ds.herald_train_vitals").count()
print(f"Vitals saved: {vit_count:,} rows (11 features)")


# ===========================================================================
# SECTION 4: BOOK 2 — ICD-10 → herald_train_icd10 (6 features)
#
# icd_BLEED_CNT_12MO, icd_FHX_CRC_COMBINED, icd_HIGH_RISK_HISTORY,
# icd_IRON_DEF_ANEMIA_FLAG_12MO, icd_SYMPTOM_BURDEN_12MO,
# icd_chronic_gi_pattern
# ===========================================================================

print("\n" + "=" * 70)
print("SECTION 4: ICD-10 DIAGNOSES")
print("=" * 70)

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

  -- Inpatient diagnoses (join via HSP_ACCOUNT_ID, date = DISCH_DATE_TIME)
  SELECT c.PAT_ID, c.END_DTTM, dd.CODE, DATE(hsp.DISCH_DATE_TIME) AS DX_DATE
  FROM cohort_base c
  JOIN clarity_cur.PAT_ENC_HSP_HAR_ENH hsp
    ON hsp.PAT_ID = c.PAT_ID
    AND DATE(hsp.DISCH_DATE_TIME) < c.END_DTTM
    AND DATE(hsp.DISCH_DATE_TIME) >= DATE('2021-07-01')
    AND hsp.ADT_PATIENT_STAT_C <> 1
    AND hsp.ADMIT_CONF_STAT_C <> 3
  JOIN clarity_cur.hsp_acct_dx_list_enh dd
    ON dd.HSP_ACCOUNT_ID = hsp.HSP_ACCOUNT_ID
  WHERE dd.CODE IS NOT NULL
),

-- Structured family history from FAMILY_HX table
family_hx AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    MAX(CASE WHEN fh.MEDICAL_HX_C IN (10404, 20172) THEN 1 ELSE 0 END) AS FHX_CRC_CODED,
    MAX(CASE WHEN dx.CODE RLIKE '^Z80\\.0' THEN 1 ELSE 0 END) AS FHX_DIGESTIVE_CANCER_ICD
  FROM cohort_base c
  LEFT JOIN clarity.family_hx fh
    ON c.PAT_ID = fh.PAT_ID
    AND DATE(fh.CONTACT_DATE) <= c.END_DTTM
  LEFT JOIN all_dx dx
    ON c.PAT_ID = dx.PAT_ID AND c.END_DTTM = dx.END_DTTM
  GROUP BY c.PAT_ID, c.END_DTTM
)

SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- Iron deficiency anemia (D50) in 12 months
  COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^D50' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0)
    AS icd_IRON_DEF_ANEMIA_FLAG_12MO,

  -- GI bleeding encounter count in 12 months
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

  -- HIGH_RISK_HISTORY: IBD, polyps, or prior malignancy (ever)
  CASE WHEN (
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(K50|K51)' THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(D12|K63\\.5)' THEN 1 END), 0) +
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^Z85' THEN 1 END), 0)
  ) >= 1 THEN 1 ELSE 0 END AS icd_HIGH_RISK_HISTORY,

  -- FHX_CRC_COMBINED: structured family_hx OR ICD Z80.0
  GREATEST(
    COALESCE(fhx.FHX_CRC_CODED, 0),
    COALESCE(fhx.FHX_DIGESTIVE_CANCER_ICD, 0)
  ) AS icd_FHX_CRC_COMBINED,

  -- Chronic GI pattern: IBD ever OR diverticular 24mo OR GI complexity >= 2
  CASE WHEN (
    -- IBD ever
    COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^(K50|K51)' THEN 1 END), 0) = 1
    -- Diverticular disease in 24 months
    OR COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^K57' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 730 THEN 1 END), 0) = 1
    -- GI complexity score >= 2 (malabsorption + IBS + hematemesis + bloating + intestinal abscess)
    OR (
      COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^K90' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 730 THEN 1 END), 0) +
      COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^K58' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 730 THEN 1 END), 0) +
      COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^K92\\.0' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
      COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^R14' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 365 THEN 1 END), 0) +
      COALESCE(MAX(CASE WHEN dx.CODE RLIKE '^K63\\.0' AND DATEDIFF(c.END_DTTM, dx.DX_DATE) <= 730 THEN 1 END), 0)
    ) >= 2
  ) THEN 1 ELSE 0 END AS icd_chronic_gi_pattern

FROM cohort_base c
LEFT JOIN all_dx dx ON c.PAT_ID = dx.PAT_ID AND c.END_DTTM = dx.END_DTTM
LEFT JOIN family_hx fhx ON c.PAT_ID = fhx.PAT_ID AND c.END_DTTM = fhx.END_DTTM
GROUP BY c.PAT_ID, c.END_DTTM, fhx.FHX_CRC_CODED, fhx.FHX_DIGESTIVE_CANCER_ICD
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_train_icd10 AS
SELECT * FROM icd_features
""")

icd_count = spark.table(f"{trgt_cat}.clncl_ds.herald_train_icd10").count()
print(f"ICD-10 saved: {icd_count:,} rows (6 features)")


# ===========================================================================
# SECTION 5: BOOK 4 — LABS → herald_train_labs (10 features)
#
# lab_ALBUMIN_DROP_15PCT_FLAG, lab_ALBUMIN_VALUE, lab_ANEMIA_GRADE,
# lab_ANEMIA_SEVERITY_SCORE, lab_CRP_6MO_CHANGE, lab_IRON_SATURATION_PCT,
# lab_PLATELETS_ACCELERATING_RISE, lab_PLATELETS_VALUE,
# lab_THROMBOCYTOSIS_FLAG, lab_HEMOGLOBIN_ACCELERATING_DECLINE
#
# (lab_comprehensive_iron_deficiency computed at compilation in Section 9)
#
# DUAL SOURCE: Outpatient (order_results) + Inpatient (res_components)
# ===========================================================================

print("\n" + "=" * 70)
print("SECTION 5: LABS (dual outpatient + inpatient path)")
print("=" * 70)

# --- 5a. Unified lab results (UNION ALL of outpatient + inpatient) ---

spark.sql("""
CREATE OR REPLACE TEMP VIEW all_lab_results AS

-- Outpatient lab path: order_proc_enh -> order_results -> clarity_component
SELECT
  c.PAT_ID,
  c.END_DTTM,
  cc.NAME AS COMPONENT_NAME,
  TRY_CAST(REGEXP_REPLACE(ores.ORD_VALUE, '[><]', '') AS FLOAT) AS VALUE,
  DATE(ores.RESULT_TIME) AS RESULT_DATE,
  DATEDIFF(c.END_DTTM, DATE(ores.RESULT_TIME)) AS DAYS_BEFORE
FROM cohort_base c
JOIN clarity_cur.order_proc_enh op
  ON op.PAT_ID = c.PAT_ID
  AND DATE(op.ORDERING_DATE) < c.END_DTTM
  AND DATE(op.ORDERING_DATE) >= DATE_SUB(c.END_DTTM, 1095)
  AND DATE(op.ORDERING_DATE) >= DATE('2021-07-01')
  AND op.ORDER_STATUS_C IN (3, 5, 10)
  AND op.LAB_STATUS_C IN (3, 5)
JOIN clarity.order_results ores
  ON ores.ORDER_PROC_ID = op.ORDER_PROC_ID
JOIN clarity.clarity_component cc
  ON cc.COMPONENT_ID = ores.COMPONENT_ID
WHERE cc.NAME IN ('HEMOGLOBIN', 'PLATELETS', 'ALBUMIN', 'MCV', 'FERRITIN',
                   'IRON', 'TIBC', 'CRP')
  AND DATE(ores.RESULT_TIME) < c.END_DTTM
  AND TRY_CAST(REGEXP_REPLACE(ores.ORD_VALUE, '[><]', '') AS FLOAT) IS NOT NULL

UNION ALL

-- Inpatient lab path: order_proc_enh -> clarity_eap -> spec_test_rel ->
--                     res_db_main -> res_components -> clarity_component
SELECT
  peh.PAT_ID,
  c.END_DTTM,
  normalized.COMPONENT_NAME,
  normalized.VALUE,
  normalized.RESULT_DATE,
  normalized.DAYS_BEFORE
FROM (
  SELECT
    op.PAT_ENC_CSN_ID,
    c_inner.PAT_ID AS COHORT_PAT_ID,
    c_inner.END_DTTM,
    CASE
      WHEN TRIM(UPPER(cc.NAME)) IN ('HEMOGLOBIN', 'HEMOGLOBIN POC', 'HEMOGLOBIN VENOUS',
           'HEMOGLOBIN ABG', 'HEMOGLOBIN USED CAPILLARY') THEN 'HEMOGLOBIN'
      WHEN TRIM(UPPER(cc.NAME)) = 'MCV' THEN 'MCV'
      WHEN TRIM(UPPER(cc.NAME)) = 'PLATELETS' THEN 'PLATELETS'
      WHEN TRIM(UPPER(cc.NAME)) = 'ALBUMIN' THEN 'ALBUMIN'
      WHEN TRIM(UPPER(cc.NAME)) IN ('IRON', 'IRON, TOTAL') THEN 'IRON'
      WHEN TRIM(UPPER(cc.NAME)) IN ('TIBC', 'IRON BINDING CAPACITY') THEN 'TIBC'
      WHEN TRIM(UPPER(cc.NAME)) = 'FERRITIN' THEN 'FERRITIN'
      WHEN TRIM(UPPER(cc.NAME)) = 'CRP' THEN 'CRP'
      ELSE NULL
    END AS COMPONENT_NAME,
    CASE
      WHEN TRIM(UPPER(cc.NAME)) = 'CRP' AND UPPER(TRIM(res_comp.COMPONENT_UNITS)) = 'MG/DL'
      THEN TRY_CAST(REGEXP_REPLACE(res_comp.COMPONENT_VALUE, '[><]', '') AS FLOAT) * 10
      ELSE TRY_CAST(REGEXP_REPLACE(res_comp.COMPONENT_VALUE, '[><]', '') AS FLOAT)
    END AS VALUE,
    DATE(res_comp.COMP_VERIF_DTTM) AS RESULT_DATE,
    DATEDIFF(c_inner.END_DTTM, DATE(res_comp.COMP_VERIF_DTTM)) AS DAYS_BEFORE
  FROM clarity_cur.order_proc_enh op
  JOIN clarity.clarity_eap eap ON eap.PROC_ID = op.PROC_ID
  JOIN clarity.spec_test_rel spec ON spec.SPEC_TST_ORDER_ID = op.ORDER_PROC_ID
  JOIN clarity.res_db_main rdm
    ON rdm.RES_SPECIMEN_ID = spec.SPECIMEN_ID
    AND rdm.RES_ORDER_ID = spec.SPEC_TST_ORDER_ID
  JOIN clarity.res_components res_comp
    ON res_comp.RESULT_ID = rdm.RESULT_ID
  JOIN clarity.clarity_component cc
    ON cc.COMPONENT_ID = res_comp.COMPONENT_ID
  JOIN clarity.pat_enc_hsp peh_inner
    ON peh_inner.PAT_ENC_CSN_ID = op.PAT_ENC_CSN_ID
  JOIN cohort_base c_inner
    ON c_inner.PAT_ID = peh_inner.PAT_ID
  WHERE op.ORDER_STATUS_C IN (3, 5, 10)
    AND op.LAB_STATUS_C IN (3, 5)
    AND rdm.RES_VAL_STATUS_C = 9
    AND DATE(res_comp.COMP_VERIF_DTTM) >= DATE('2021-07-01')
    AND DATE(res_comp.COMP_VERIF_DTTM) < c_inner.END_DTTM
    AND DATE(res_comp.COMP_VERIF_DTTM) >= DATE_SUB(c_inner.END_DTTM, 1095)
    AND res_comp.COMPONENT_VALUE IS NOT NULL
    AND TRIM(UPPER(cc.NAME)) IN (
      'HEMOGLOBIN', 'HEMOGLOBIN POC', 'HEMOGLOBIN VENOUS', 'HEMOGLOBIN ABG',
      'HEMOGLOBIN USED CAPILLARY', 'MCV', 'PLATELETS', 'ALBUMIN',
      'IRON', 'IRON, TOTAL', 'TIBC', 'IRON BINDING CAPACITY', 'FERRITIN', 'CRP'
    )
) normalized
JOIN clarity.pat_enc_hsp peh ON peh.PAT_ENC_CSN_ID = normalized.PAT_ENC_CSN_ID
JOIN cohort_base c ON c.PAT_ID = peh.PAT_ID AND c.END_DTTM = normalized.END_DTTM
WHERE normalized.COMPONENT_NAME IS NOT NULL
  AND normalized.VALUE IS NOT NULL
""")

print("Unified lab results created (outpatient + inpatient)")


# --- 5b. Latest lab values (pivot) ---

spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_latest AS

WITH ranked AS (
  SELECT PAT_ID, END_DTTM, COMPONENT_NAME, VALUE,
    ROW_NUMBER() OVER (PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME
                       ORDER BY RESULT_DATE DESC) AS rn
  FROM all_lab_results
  WHERE VALUE IS NOT NULL
    AND (
      (COMPONENT_NAME = 'HEMOGLOBIN' AND VALUE BETWEEN 3 AND 20)
      OR (COMPONENT_NAME = 'PLATELETS' AND VALUE BETWEEN 10 AND 2000)
      OR (COMPONENT_NAME = 'ALBUMIN' AND VALUE BETWEEN 1 AND 6)
      OR (COMPONENT_NAME = 'MCV' AND VALUE BETWEEN 50 AND 150)
      OR (COMPONENT_NAME = 'FERRITIN' AND VALUE BETWEEN 0 AND 10000)
      OR (COMPONENT_NAME = 'IRON' AND VALUE BETWEEN 0 AND 500)
      OR (COMPONENT_NAME = 'TIBC' AND VALUE BETWEEN 100 AND 600)
      OR (COMPONENT_NAME = 'CRP' AND VALUE BETWEEN 0 AND 500)
    )
)

SELECT
  PAT_ID,
  END_DTTM,
  MAX(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' THEN VALUE END) AS HEMOGLOBIN_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'PLATELETS' THEN VALUE END) AS PLATELETS_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'ALBUMIN' THEN VALUE END) AS ALBUMIN_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'MCV' THEN VALUE END) AS MCV_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'FERRITIN' THEN VALUE END) AS FERRITIN_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'IRON' THEN VALUE END) AS IRON_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'TIBC' THEN VALUE END) AS TIBC_VALUE,
  MAX(CASE WHEN COMPONENT_NAME = 'CRP' THEN VALUE END) AS CRP_VALUE
FROM ranked
WHERE rn = 1
GROUP BY PAT_ID, END_DTTM
""")

print("Latest lab values pivoted")


# --- 5c. Albumin 6 months prior ---

spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_albumin_prior AS

WITH albumin_6m AS (
  SELECT
    PAT_ID, END_DTTM, VALUE,
    ROW_NUMBER() OVER (
      PARTITION BY PAT_ID, END_DTTM
      ORDER BY ABS(DAYS_BEFORE - 180)
    ) AS rn
  FROM all_lab_results
  WHERE COMPONENT_NAME = 'ALBUMIN'
    AND VALUE BETWEEN 1 AND 6
    AND DAYS_BEFORE BETWEEN 150 AND 210
)

SELECT PAT_ID, END_DTTM, VALUE AS ALBUMIN_6M_PRIOR
FROM albumin_6m WHERE rn = 1
""")


# --- 5d. CRP 6 months prior ---

spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_crp_prior AS

WITH crp_6m AS (
  SELECT
    PAT_ID, END_DTTM, VALUE,
    ROW_NUMBER() OVER (
      PARTITION BY PAT_ID, END_DTTM
      ORDER BY ABS(DAYS_BEFORE - 180)
    ) AS rn
  FROM all_lab_results
  WHERE COMPONENT_NAME = 'CRP'
    AND VALUE BETWEEN 0 AND 500
    AND DAYS_BEFORE BETWEEN 150 AND 210
)

SELECT PAT_ID, END_DTTM, VALUE AS CRP_6M_PRIOR
FROM crp_6m WHERE rn = 1
""")


# --- 5e. Acceleration features (hemoglobin + platelets) ---

spark.sql("""
CREATE OR REPLACE TEMP VIEW lab_acceleration AS

WITH hgb_time_points AS (
  SELECT PAT_ID, END_DTTM,
    MAX(CASE WHEN DAYS_BEFORE <= 30 THEN VALUE END) AS current_value,
    AVG(CASE WHEN DAYS_BEFORE BETWEEN 15 AND 45 THEN VALUE END) AS value_1mo_prior,
    AVG(CASE WHEN DAYS_BEFORE BETWEEN 60 AND 120 THEN VALUE END) AS value_3mo_prior,
    AVG(CASE WHEN DAYS_BEFORE BETWEEN 150 AND 210 THEN VALUE END) AS value_6mo_prior
  FROM all_lab_results
  WHERE COMPONENT_NAME = 'HEMOGLOBIN'
    AND VALUE BETWEEN 3 AND 20
  GROUP BY PAT_ID, END_DTTM
),

plt_time_points AS (
  SELECT PAT_ID, END_DTTM,
    MAX(CASE WHEN DAYS_BEFORE <= 30 THEN VALUE END) AS current_value,
    AVG(CASE WHEN DAYS_BEFORE BETWEEN 60 AND 120 THEN VALUE END) AS value_3mo_prior,
    AVG(CASE WHEN DAYS_BEFORE BETWEEN 150 AND 210 THEN VALUE END) AS value_6mo_prior
  FROM all_lab_results
  WHERE COMPONENT_NAME = 'PLATELETS'
    AND VALUE BETWEEN 10 AND 2000
  GROUP BY PAT_ID, END_DTTM
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  -- Hemoglobin accelerating decline (Book 4 definition):
  -- Recent velocity < -0.5 g/dL/month AND accelerating
  CASE
    WHEN h.current_value IS NOT NULL
     AND h.value_1mo_prior IS NOT NULL
     AND h.value_3mo_prior IS NOT NULL
     AND h.value_6mo_prior IS NOT NULL
     AND ((h.current_value - h.value_3mo_prior) / 3.0) < -0.5
     AND ((h.current_value - h.value_3mo_prior) / 3.0) < ((h.value_3mo_prior - h.value_6mo_prior) / 3.0)
    THEN 1 ELSE 0
  END AS lab_HEMOGLOBIN_ACCELERATING_DECLINE,
  -- Platelet accelerating rise (Book 4 definition):
  -- Currently elevated (>450) AND accelerating
  CASE
    WHEN p.current_value IS NOT NULL AND p.current_value > 450
     AND p.value_3mo_prior IS NOT NULL AND p.value_6mo_prior IS NOT NULL
     AND ((p.current_value - p.value_3mo_prior) / 3.0) > ((p.value_3mo_prior - p.value_6mo_prior) / 3.0)
    THEN 1 ELSE 0
  END AS lab_PLATELETS_ACCELERATING_RISE
FROM cohort_base c
LEFT JOIN hgb_time_points h ON c.PAT_ID = h.PAT_ID AND c.END_DTTM = h.END_DTTM
LEFT JOIN plt_time_points p ON c.PAT_ID = p.PAT_ID AND c.END_DTTM = p.END_DTTM
""")

print("Acceleration features computed")


# --- 5f. Combine all lab features ---

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_train_labs AS

SELECT
  c.PAT_ID,
  c.END_DTTM,
  ll.PLATELETS_VALUE AS lab_PLATELETS_VALUE,
  ll.ALBUMIN_VALUE AS lab_ALBUMIN_VALUE,

  -- Iron saturation
  CASE WHEN ll.TIBC_VALUE > 0
    THEN ROUND(ll.IRON_VALUE / ll.TIBC_VALUE * 100, 1)
  END AS lab_IRON_SATURATION_PCT,

  -- Albumin drop > 15% from 6mo prior
  CASE WHEN ll.ALBUMIN_VALUE IS NOT NULL AND ap.ALBUMIN_6M_PRIOR IS NOT NULL
       AND ll.ALBUMIN_VALUE < ap.ALBUMIN_6M_PRIOR * 0.85
    THEN 1 ELSE 0
  END AS lab_ALBUMIN_DROP_15PCT_FLAG,

  -- CRP 6-month change
  CASE WHEN ll.CRP_VALUE IS NOT NULL AND cp.CRP_6M_PRIOR IS NOT NULL
    THEN ROUND(ll.CRP_VALUE - cp.CRP_6M_PRIOR, 2)
  END AS lab_CRP_6MO_CHANGE,

  -- Anemia grade (Book 4: 12=normal, 11=mild, 8=moderate, <8=severe)
  CASE
    WHEN ll.HEMOGLOBIN_VALUE IS NULL THEN NULL
    WHEN ll.HEMOGLOBIN_VALUE >= 12.0 THEN 0
    WHEN ll.HEMOGLOBIN_VALUE >= 11.0 THEN 1
    WHEN ll.HEMOGLOBIN_VALUE >= 8.0 THEN 2
    ELSE 3
  END AS lab_ANEMIA_GRADE,

  -- Anemia severity score (Book 4: grade + iron_def*2 + microcytic*1, max 6)
  CASE WHEN ll.HEMOGLOBIN_VALUE IS NOT NULL THEN
    (CASE
      WHEN ll.HEMOGLOBIN_VALUE >= 12.0 THEN 0
      WHEN ll.HEMOGLOBIN_VALUE >= 11.0 THEN 1
      WHEN ll.HEMOGLOBIN_VALUE >= 8.0 THEN 2
      ELSE 3
    END)
    + (CASE WHEN ll.HEMOGLOBIN_VALUE < 12 AND ll.MCV_VALUE < 80
            AND (ll.FERRITIN_VALUE < 30
                 OR (ll.IRON_VALUE IS NOT NULL AND ll.TIBC_VALUE > 0
                     AND (ll.IRON_VALUE / ll.TIBC_VALUE * 100) < 20))
       THEN 2 ELSE 0 END)
    + (CASE WHEN ll.MCV_VALUE < 80 THEN 1 ELSE 0 END)
  END AS lab_ANEMIA_SEVERITY_SCORE,

  CASE WHEN ll.PLATELETS_VALUE > 400 THEN 1 ELSE 0 END AS lab_THROMBOCYTOSIS_FLAG,

  la.lab_PLATELETS_ACCELERATING_RISE,
  la.lab_HEMOGLOBIN_ACCELERATING_DECLINE,

  -- Lab-only iron deficiency (combined with ICD in compilation)
  CASE
    WHEN (ll.HEMOGLOBIN_VALUE < 12 AND ll.MCV_VALUE < 80) THEN 1
    WHEN (ll.FERRITIN_VALUE < 30 AND ll.HEMOGLOBIN_VALUE < 13) THEN 1
    ELSE 0
  END AS lab_iron_deficiency_labs_only

FROM cohort_base c
LEFT JOIN lab_latest ll ON c.PAT_ID = ll.PAT_ID AND c.END_DTTM = ll.END_DTTM
LEFT JOIN lab_albumin_prior ap ON c.PAT_ID = ap.PAT_ID AND c.END_DTTM = ap.END_DTTM
LEFT JOIN lab_crp_prior cp ON c.PAT_ID = cp.PAT_ID AND c.END_DTTM = cp.END_DTTM
LEFT JOIN lab_acceleration la ON c.PAT_ID = la.PAT_ID AND c.END_DTTM = la.END_DTTM
""")

lab_count = spark.table(f"{trgt_cat}.clncl_ds.herald_train_labs").count()
print(f"Labs saved: {lab_count:,} rows (10 features + lab_iron_deficiency_labs_only)")


# ===========================================================================
# SECTION 6: BOOK 5.2 — INPATIENT MEDS → herald_train_inpatient_meds
#            (5 features)
#
# inp_med_inp_opioid_use_flag, inp_med_inp_laxative_use_flag,
# inp_med_inp_ibd_meds_recency, inp_med_inp_gi_bleed_meds_recency,
# inp_med_inp_any_hospitalization
# ===========================================================================

print("\n" + "=" * 70)
print("SECTION 6: INPATIENT MEDICATIONS")
print("=" * 70)

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_train_inpatient_meds AS

WITH inpatient_meds AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    LOWER(med.GENERIC_NAME) AS generic_name,
    DATEDIFF(c.END_DTTM, DATE(mar.TAKEN_TIME)) AS days_since,
    hsp.PAT_ENC_CSN_ID AS admission_csn
  FROM cohort_base c
  JOIN clarity_cur.PAT_ENC_HSP_HAR_ENH hsp
    ON hsp.PAT_ID = c.PAT_ID
    AND DATE(hsp.HOSP_ADMSN_TIME) < c.END_DTTM
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE_SUB(c.END_DTTM, 730)
    AND DATE(hsp.HOSP_ADMSN_TIME) >= DATE('2021-07-01')
    AND hsp.ADT_PATIENT_STAT_C <> 1
    AND hsp.ADMIT_CONF_STAT_C <> 3
  JOIN clarity_cur.order_med_enh ome
    ON ome.PAT_ID = c.PAT_ID
    AND ome.PAT_ENC_CSN_ID = hsp.PAT_ENC_CSN_ID
    AND ome.ORDERING_MODE_C = 2
  JOIN prod.clarity_cur.mar_admin_info_enh mar
    ON mar.ORDER_MED_ID = ome.ORDER_MED_ID
    AND mar.TAKEN_TIME IS NOT NULL
    AND UPPER(TRIM(mar.ACTION)) IN (
      'GIVEN', 'PATIENT/FAMILY ADMIN', 'GIVEN-SEE OVERRIDE',
      'ADMIN BY ANOTHER CLINICIAN (COMMENT)', 'NEW BAG', 'BOLUS', 'PUSH',
      'STARTED BY ANOTHER CLINICIAN', 'BAG SWITCHED',
      'CLINIC SAMPLE ADMINISTERED', 'APPLIED', 'FEEDING STARTED',
      'ACKNOWLEDGED', 'CONTRAST GIVEN', 'NEW BAG-SEE OVERRIDE',
      'BOLUS FROM BAG'
    )
  JOIN clarity.clarity_medication med
    ON med.MEDICATION_ID = ome.MEDICATION_ID
  WHERE med.GENERIC_NAME IS NOT NULL
),

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
        THEN days_since END) AS ibd_days_since,
    MAX(CASE WHEN generic_name RLIKE '(tranexamic|octreotide|vasopressin|pantoprazole|esomeprazole)'
        THEN 1 ELSE 0 END) AS has_gi_bleed_med,
    MIN(CASE WHEN generic_name RLIKE '(tranexamic|octreotide|vasopressin|pantoprazole|esomeprazole)'
        THEN days_since END) AS gi_bleed_days_since,
    MAX(1) AS has_any_key_med
  FROM inpatient_meds
  GROUP BY PAT_ID, END_DTTM, admission_csn
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  COALESCE(MAX(af.has_opioid), 0) AS inp_med_inp_opioid_use_flag,
  COALESCE(MAX(af.has_laxative), 0) AS inp_med_inp_laxative_use_flag,
  -- IBD meds recency (ordinal 0-5)
  CASE
    WHEN MIN(af.ibd_days_since) IS NULL THEN 0
    WHEN MIN(af.ibd_days_since) <= 30 THEN 5
    WHEN MIN(af.ibd_days_since) <= 90 THEN 4
    WHEN MIN(af.ibd_days_since) <= 180 THEN 3
    WHEN MIN(af.ibd_days_since) <= 365 THEN 2
    ELSE 1
  END AS inp_med_inp_ibd_meds_recency,
  -- GI bleed meds recency (ordinal 0-5)
  CASE
    WHEN MIN(af.gi_bleed_days_since) IS NULL THEN 0
    WHEN MIN(af.gi_bleed_days_since) <= 30 THEN 5
    WHEN MIN(af.gi_bleed_days_since) <= 90 THEN 4
    WHEN MIN(af.gi_bleed_days_since) <= 180 THEN 3
    WHEN MIN(af.gi_bleed_days_since) <= 365 THEN 2
    ELSE 1
  END AS inp_med_inp_gi_bleed_meds_recency,
  COALESCE(MAX(af.has_any_key_med), 0) AS inp_med_inp_any_hospitalization

FROM cohort_base c
LEFT JOIN admission_flags af ON c.PAT_ID = af.PAT_ID AND c.END_DTTM = af.END_DTTM
GROUP BY c.PAT_ID, c.END_DTTM
""")

med_count = spark.table(f"{trgt_cat}.clncl_ds.herald_train_inpatient_meds").count()
print(f"Inpatient meds saved: {med_count:,} rows (5 features)")


# ===========================================================================
# SECTION 7: BOOK 6 — VISITS → herald_train_visits (7 features)
#
# visit_outpatient_visits_12mo, visit_gi_symptom_op_visits_12mo,
# visit_total_gi_symptom_visits_12mo, visit_primary_care_continuity_ratio,
# visit_no_shows_12mo, visit_recency_last_gi, visit_gi_symptoms_no_specialist
# ===========================================================================

print("\n" + "=" * 70)
print("SECTION 7: VISIT HISTORY")
print("=" * 70)

spark.sql("""
CREATE OR REPLACE TEMP VIEW visit_features AS

WITH
outpatient AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    pe.PAT_ENC_CSN_ID,
    DATE(pe.CONTACT_DATE) AS VISIT_DATE,
    pe.APPT_STATUS_C,
    CASE WHEN pe.VISIT_PROV_ID = pe.PCP_PROV_ID THEN 1 ELSE 0 END AS IS_PCP_VISIT
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

gi_symptom_op AS (
  SELECT DISTINCT o.PAT_ID, o.END_DTTM, o.PAT_ENC_CSN_ID
  FROM outpatient o
  JOIN clarity_cur.pat_enc_dx_enh dx
    ON dx.PAT_ENC_CSN_ID = o.PAT_ENC_CSN_ID
  WHERE dx.ICD10_CODE RLIKE '^(K62\\.5|K92|K59|R19|D50|R10|R63\\.4|R53)'
    AND o.APPT_STATUS_C IN (2, 6)
),

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
    ON dx.HSP_ACCOUNT_ID = hsp.HSP_ACCOUNT_ID
  WHERE dx.CODE RLIKE '^(K62\\.5|K92|K59|R19|D50|R10|R63\\.4|R53)'
),

-- GI specialist visits (24mo lookback for recency)
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
  JOIN clarity_cur.clarity_ser_enh ser
    ON ser.PROV_ID = pe.VISIT_PROV_ID
  WHERE UPPER(ser.SPECIALTY_NAME) IN ('GASTROENTEROLOGY', 'COLON AND RECTAL SURGERY')
  GROUP BY c.PAT_ID, c.END_DTTM
),

-- GI specialist visit count (12mo, for gi_symptoms_no_specialist)
gi_visits_12mo AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    COUNT(DISTINCT pe.PAT_ENC_CSN_ID) AS gi_visits_12mo
  FROM cohort_base c
  JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID
    AND DATE(pe.CONTACT_DATE) < c.END_DTTM
    AND DATE(pe.CONTACT_DATE) >= DATE_SUB(c.END_DTTM, 365)
    AND pe.APPT_STATUS_C IN (2, 6)
  JOIN clarity_cur.clarity_ser_enh ser
    ON ser.PROV_ID = pe.VISIT_PROV_ID
  WHERE UPPER(ser.SPECIALTY_NAME) IN ('GASTROENTEROLOGY', 'COLON AND RECTAL SURGERY')
  GROUP BY c.PAT_ID, c.END_DTTM
),

-- No-shows (APPT_STATUS_C = 4 only)
no_shows AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    COUNT(*) AS no_shows_12mo
  FROM cohort_base c
  JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID
    AND DATE(pe.CONTACT_DATE) < c.END_DTTM
    AND DATE(pe.CONTACT_DATE) >= DATE_SUB(c.END_DTTM, 365)
    AND DATE(pe.CONTACT_DATE) >= DATE('2021-07-01')
    AND pe.APPT_STATUS_C = 4
  JOIN clarity_cur.dep_loc_ploc_sa_enh dep
    ON dep.department_id = COALESCE(pe.DEPARTMENT_ID, pe.EFFECTIVE_DEPT_ID)
  WHERE dep.RPT_GRP_SIX IN ('116001', '116002')
  GROUP BY c.PAT_ID, c.END_DTTM
),

op_metrics AS (
  SELECT
    PAT_ID,
    END_DTTM,
    SUM(CASE WHEN APPT_STATUS_C IN (2, 6) THEN 1 ELSE 0 END) AS outpatient_visits_12mo,
    SUM(CASE WHEN APPT_STATUS_C IN (2, 6) AND IS_PCP_VISIT = 1
         THEN 1 ELSE 0 END) AS pcp_visits_12mo
  FROM outpatient
  GROUP BY PAT_ID, END_DTTM
),

gi_symptom_op_counts AS (
  SELECT PAT_ID, END_DTTM, COUNT(DISTINCT PAT_ENC_CSN_ID) AS gi_symptom_op_visits_12mo
  FROM gi_symptom_op
  GROUP BY PAT_ID, END_DTTM
),

gi_symptom_acute_counts AS (
  SELECT PAT_ID, END_DTTM, COUNT(DISTINCT PAT_ENC_CSN_ID) AS gi_symptom_acute_visits_12mo
  FROM gi_symptom_acute
  GROUP BY PAT_ID, END_DTTM
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  -- Ordinal GI recency (0-5)
  CASE
    WHEN gr.days_since_last_gi IS NULL THEN 0
    WHEN gr.days_since_last_gi <= 30 THEN 5
    WHEN gr.days_since_last_gi <= 90 THEN 4
    WHEN gr.days_since_last_gi <= 180 THEN 3
    WHEN gr.days_since_last_gi <= 365 THEN 2
    ELSE 1
  END AS visit_recency_last_gi,
  COALESCE(om.outpatient_visits_12mo, 0) AS visit_outpatient_visits_12mo,
  COALESCE(gsop.gi_symptom_op_visits_12mo, 0) AS visit_gi_symptom_op_visits_12mo,
  COALESCE(gsop.gi_symptom_op_visits_12mo, 0) + COALESCE(gsac.gi_symptom_acute_visits_12mo, 0)
    AS visit_total_gi_symptom_visits_12mo,
  -- Primary care continuity
  CASE
    WHEN COALESCE(om.outpatient_visits_12mo, 0) > 0
    THEN ROUND(COALESCE(om.pcp_visits_12mo, 0) * 1.0 / om.outpatient_visits_12mo, 3)
    ELSE 0
  END AS visit_primary_care_continuity_ratio,
  -- No-shows in 12 months
  COALESCE(ns.no_shows_12mo, 0) AS visit_no_shows_12mo,
  -- GI symptoms without specialist
  CASE
    WHEN (COALESCE(gsop.gi_symptom_op_visits_12mo, 0) + COALESCE(gsac.gi_symptom_acute_visits_12mo, 0)) > 0
     AND COALESCE(gv.gi_visits_12mo, 0) = 0
    THEN 1 ELSE 0
  END AS visit_gi_symptoms_no_specialist

FROM cohort_base c
LEFT JOIN op_metrics om ON c.PAT_ID = om.PAT_ID AND c.END_DTTM = om.END_DTTM
LEFT JOIN gi_symptom_op_counts gsop ON c.PAT_ID = gsop.PAT_ID AND c.END_DTTM = gsop.END_DTTM
LEFT JOIN gi_symptom_acute_counts gsac ON c.PAT_ID = gsac.PAT_ID AND c.END_DTTM = gsac.END_DTTM
LEFT JOIN gi_recency gr ON c.PAT_ID = gr.PAT_ID AND c.END_DTTM = gr.END_DTTM
LEFT JOIN gi_visits_12mo gv ON c.PAT_ID = gv.PAT_ID AND c.END_DTTM = gv.END_DTTM
LEFT JOIN no_shows ns ON c.PAT_ID = ns.PAT_ID AND c.END_DTTM = ns.END_DTTM
""")

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_train_visits AS
SELECT * FROM visit_features
""")

visit_count = spark.table(f"{trgt_cat}.clncl_ds.herald_train_visits").count()
print(f"Visits saved: {visit_count:,} rows (7 features)")


# ===========================================================================
# SECTION 8: BOOK 7 — PROCEDURES → herald_train_procedures (2 features)
#
# proc_total_imaging_count_12mo, proc_blood_transfusion_count_12mo
# ===========================================================================

print("\n" + "=" * 70)
print("SECTION 8: PROCEDURES")
print("=" * 70)

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_train_procedures AS

WITH proc_raw AS (
  SELECT
    c.PAT_ID,
    c.END_DTTM,
    CASE
      WHEN op.PROC_CODE IN ('74150','74160','74170','74176','74177','74178',
                             '74181','74182','74183')
        OR LOWER(op.PROC_NAME) LIKE '%ct%abd%'
        OR LOWER(op.PROC_NAME) LIKE '%ct%pelv%'
        OR LOWER(op.PROC_NAME) LIKE '%mri%abd%'
        OR LOWER(op.PROC_NAME) LIKE '%mri%pelv%'
      THEN 1 ELSE 0
    END AS is_abd_imaging,
    CASE
      WHEN op.PROC_CODE IN ('36430','36440','36450','36455','36456','86900','86901')
        OR LOWER(op.PROC_NAME) LIKE '%transfus%'
        OR LOWER(op.PROC_NAME) LIKE '%blood product%'
      THEN 1 ELSE 0
    END AS is_transfusion
  FROM cohort_base c
  JOIN clarity_cur.order_proc_enh op
    ON op.PAT_ID = c.PAT_ID
    AND DATE(op.RESULT_TIME) < c.END_DTTM
    AND DATE(op.RESULT_TIME) >= DATE_SUB(c.END_DTTM, 365)
    AND DATE(op.RESULT_TIME) >= DATE('2021-07-01')
    AND op.ORDER_STATUS_C = 5
    AND op.RESULT_TIME IS NOT NULL
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  COALESCE(SUM(pr.is_abd_imaging), 0) AS proc_total_imaging_count_12mo,
  COALESCE(SUM(pr.is_transfusion), 0) AS proc_blood_transfusion_count_12mo

FROM cohort_base c
LEFT JOIN proc_raw pr ON c.PAT_ID = pr.PAT_ID AND c.END_DTTM = pr.END_DTTM
GROUP BY c.PAT_ID, c.END_DTTM
""")

proc_count = spark.table(f"{trgt_cat}.clncl_ds.herald_train_procedures").count()
print(f"Procedures saved: {proc_count:,} rows (2 features)")


# ===========================================================================
# SECTION 9: BOOK 8 — COMPILATION → herald_train_wide
#
# LEFT JOIN all feature tables, apply AGE_GROUP transformation,
# compute lab_comprehensive_iron_deficiency, SELECT 49 features + metadata
# ===========================================================================

print("\n" + "=" * 70)
print("SECTION 9: COMPILATION")
print("=" * 70)

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_train_wide AS

SELECT
  c.PAT_ID,
  c.END_DTTM,
  c.FUTURE_CRC_EVENT,
  c.ICD10_GROUP,
  c.SPLIT,

  -- Demographics (6)
  CASE
    WHEN c.AGE >= 45 AND c.AGE < 50 THEN 1
    WHEN c.AGE >= 50 AND c.AGE < 55 THEN 2
    WHEN c.AGE >= 55 AND c.AGE < 65 THEN 3
    WHEN c.AGE >= 65 AND c.AGE < 75 THEN 4
    WHEN c.AGE >= 75 THEN 5
    ELSE 0
  END AS AGE_GROUP,
  c.HAS_PCP_AT_END,
  c.IS_FEMALE,
  c.IS_MARRIED_PARTNER,
  c.RACE_CAUCASIAN,
  c.RACE_HISPANIC,

  -- Temporal (1)
  c.months_since_cohort_entry,

  -- Vitals (11)
  v.vit_BMI,
  v.vit_MAX_WEIGHT_LOSS_PCT_60D,
  v.vit_PULSE,
  v.vit_PULSE_PRESSURE,
  v.vit_RECENCY_WEIGHT,
  v.vit_SBP_VARIABILITY_6M,
  v.vit_UNDERWEIGHT_FLAG,
  v.vit_WEIGHT_CHANGE_PCT_6M,
  v.vit_WEIGHT_OZ,
  v.vit_WEIGHT_TRAJECTORY_SLOPE,
  v.vit_vital_recency_score,

  -- ICD-10 (6)
  i.icd_BLEED_CNT_12MO,
  i.icd_FHX_CRC_COMBINED,
  i.icd_HIGH_RISK_HISTORY,
  i.icd_IRON_DEF_ANEMIA_FLAG_12MO,
  i.icd_SYMPTOM_BURDEN_12MO,
  i.icd_chronic_gi_pattern,

  -- Labs (11)
  l.lab_ALBUMIN_DROP_15PCT_FLAG,
  l.lab_ALBUMIN_VALUE,
  l.lab_ANEMIA_GRADE,
  l.lab_ANEMIA_SEVERITY_SCORE,
  l.lab_CRP_6MO_CHANGE,
  l.lab_HEMOGLOBIN_ACCELERATING_DECLINE,
  l.lab_IRON_SATURATION_PCT,
  l.lab_PLATELETS_ACCELERATING_RISE,
  l.lab_PLATELETS_VALUE,
  l.lab_THROMBOCYTOSIS_FLAG,
  -- Comprehensive iron deficiency: ICD D50 diagnosis OR lab criteria
  CASE
    WHEN i.icd_IRON_DEF_ANEMIA_FLAG_12MO = 1 THEN 1
    WHEN l.lab_iron_deficiency_labs_only = 1 THEN 1
    ELSE 0
  END AS lab_comprehensive_iron_deficiency,

  -- Inpatient Medications (5)
  im.inp_med_inp_any_hospitalization,
  im.inp_med_inp_gi_bleed_meds_recency,
  im.inp_med_inp_ibd_meds_recency,
  im.inp_med_inp_laxative_use_flag,
  im.inp_med_inp_opioid_use_flag,

  -- Visit History (7)
  vis.visit_gi_symptom_op_visits_12mo,
  vis.visit_gi_symptoms_no_specialist,
  vis.visit_no_shows_12mo,
  vis.visit_outpatient_visits_12mo,
  vis.visit_primary_care_continuity_ratio,
  vis.visit_recency_last_gi,
  vis.visit_total_gi_symptom_visits_12mo,

  -- Procedures (2)
  p.proc_blood_transfusion_count_12mo,
  p.proc_total_imaging_count_12mo

FROM {trgt_cat}.clncl_ds.herald_train_final_cohort c
LEFT JOIN {trgt_cat}.clncl_ds.herald_train_vitals v
  ON c.PAT_ID = v.PAT_ID AND c.END_DTTM = v.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_train_icd10 i
  ON c.PAT_ID = i.PAT_ID AND c.END_DTTM = i.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_train_labs l
  ON c.PAT_ID = l.PAT_ID AND c.END_DTTM = l.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_train_inpatient_meds im
  ON c.PAT_ID = im.PAT_ID AND c.END_DTTM = im.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_train_visits vis
  ON c.PAT_ID = vis.PAT_ID AND c.END_DTTM = vis.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_train_procedures p
  ON c.PAT_ID = p.PAT_ID AND c.END_DTTM = p.END_DTTM
""")

wide_count = spark.table(f"{trgt_cat}.clncl_ds.herald_train_wide").count()
print(f"Compilation saved: {wide_count:,} rows to {trgt_cat}.clncl_ds.herald_train_wide")


# ===========================================================================
# SECTION 10: VALIDATION
# ===========================================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

df_check = spark.table(f"{trgt_cat}.clncl_ds.herald_train_wide")

# Verify row count matches cohort
cohort_count = spark.table(f"{trgt_cat}.clncl_ds.herald_train_final_cohort").count()
print(f"Cohort rows:      {cohort_count:,}")
print(f"Wide table rows:  {df_check.count():,}")
assert df_check.count() == cohort_count, f"Row count mismatch! Cohort={cohort_count}, Wide={df_check.count()}"

# Verify all 49 features present
present_cols = set(df_check.columns)
missing_features = [f for f in SELECTED_FEATURES if f not in present_cols]
if missing_features:
    print(f"MISSING FEATURES: {missing_features}")
    raise ValueError(f"Missing {len(missing_features)} features: {missing_features}")
else:
    print(f"All {len(SELECTED_FEATURES)} features present")

# Positive rate by split
print("\nPositive rate by split:")
df_check.groupBy("SPLIT").agg(
    F.count("*").alias("n_obs"),
    F.sum("FUTURE_CRC_EVENT").alias("n_positive"),
    F.mean("FUTURE_CRC_EVENT").alias("event_rate")
).orderBy("SPLIT").show()

# Null rates for features
print("Feature null rates:")
for feat in sorted(SELECTED_FEATURES):
    null_count = df_check.filter(F.col(feat).isNull()).count()
    if null_count > 0:
        print(f"  {feat}: {null_count:,} nulls ({null_count/wide_count*100:.1f}%)")

print("\n" + "=" * 70)
print("FEATURIZATION COMPLETE")
print(f"Output: {trgt_cat}.clncl_ds.herald_train_wide")
print(f"Features: {len(SELECTED_FEATURES)}")
print(f"Rows: {wide_count:,}")
print("=" * 70)

# herald_test_train_pipeline.py
# Combined pipeline: Book 0 (Cohort) + Books 2,4-7 (Features) + Book 8 (Compilation)
# All intermediate tables: herald_test_train_*
# Final output: herald_test_train_wide_cleaned
#===============================================================================


################################################################################
# V2_Book0_Cohort_Creation
################################################################################

# V2_Book0_Cohort_Creation
# Functional cells: 26 of 79 code cells (153 total)
# Source: V2_Book0_Cohort_Creation.ipynb
# =============================================================================

# ========================================
# CELL 1
# ========================================


# =====================================================================
# CONFIGURATION AND PARAMETERS
# =====================================================================
"""
This notebook creates a patient-month cohort for CRC risk prediction modeling.

KEY DESIGN DECISIONS:

1. **Patient-Month Observations**: Increases training samples and captures 
   temporal risk evolution patterns.

2. **Deterministic Day Assignment**: Hash-based day assignment ensures 
   reproducibility while maintaining temporal randomization.

3. **Variable Lookback Windows**: Different feature types use different historical 
   windows based on clinical relevance.

4. **Tiered Label Quality**: Three-level approach for negatives based on return 
   visit timing and PCP status.

5. **12-Month Eligibility Window**: Follow-up required to confirm negative labels 
   per clinical ML standards.
"""

import datetime
from dateutil.relativedelta import relativedelta
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "America/Chicago")

# Define target catalog
trgt_cat = os.environ.get('trgt_cat')
spark.sql('USE CATALOG prod;')

# =============================================================================
# TIMING PARAMETERS
# =============================================================================

# Label construction
label_months = 6  # Predict CRC within 6 months (prediction window)
min_followup_months = 12  # Minimum follow-up required to confirm negative labels

# Total exclusion: Use the greater of (label_months + lag_months) or min_followup_months
total_exclusion_months = max(label_months, min_followup_months)  # 12 months

# Current data state
data_collection_date = datetime.datetime(2025, 9, 30)
latest_eligible_date = data_collection_date - relativedelta(months=total_exclusion_months)

# Cohort observation period
# Starting in 2023 to reduce prevalent case proportion
index_start_full = datetime.datetime(2023, 1, 1)
index_end_full = latest_eligible_date

index_start = index_start_full.strftime('%Y-%m-%d')
index_end = index_end_full.strftime('%Y-%m-%d')

print("="*70)
print("STUDY PERIOD CONFIGURATION")
print("="*70)
print(f"Data current through: {data_collection_date.strftime('%Y-%m-%d')}")
print(f"Latest eligible observation: {index_end}")
print(f"  Prediction window: {label_months} months")
print(f"  Minimum follow-up required: {min_followup_months} months")
print(f"  Total exclusion period: {total_exclusion_months} months")
print(f"\nCohort window: {index_start} → {index_end}")
print(f"Duration: {(index_end_full.year - index_start_full.year) * 12 + (index_end_full.month - index_start_full.month)} months")
print("="*70)

# =============================================================================
# LOOKBACK WINDOWS FOR FEATURES
# =============================================================================

lookback_chronic_months = 120      # Chronic conditions: 10 years
lookback_dx_months = 60            # Diagnoses: 5 years
lookback_symptoms_months = 24      # Recent symptoms: 2 years
lookback_labs_months = 24          # Lab results: 2 years
lookback_meds_months = 24          # Medications: 2 years
lookback_utilization_months = 24   # Healthcare use: 2 years

print(f"\nFeature lookback windows:")
print(f"  Chronic conditions/screening: {lookback_chronic_months} months")
print(f"  Diagnoses: {lookback_dx_months} months")
print(f"  Symptoms/Labs/Meds/Utilization: {lookback_symptoms_months} months")

# =============================================================================
# LABEL DEFINITION
# =============================================================================

include_anus = False  # Include C21 (anus) codes
crc_icd_regex = r'^(C(?:18|19|20))' if not include_anus else r'^(C(?:18|19|20|21))'
confirm_repeat_days = 60  # Not used in single_code mode

print(f"\nLabel definition:")
print(f"  ICD-10 pattern: {crc_icd_regex}")
print(f"  Includes: C18 (colon), C19 (rectosigmoid), C20 (rectum)" +
      (", C21 (anus)" if include_anus else ""))

# =============================================================================
# OBSERVABILITY REQUIREMENTS
# =============================================================================

min_obs_months = 24  # Minimum months patient must have been in system

print(f"\nObservability requirements:")
print(f"  Minimum prior system contact: {min_obs_months} months")
print(f"  NOTE: This ensures encounter history but does NOT eliminate prevalent cases")

# =============================================================================
# SCREENING EXCLUSION WINDOWS
# =============================================================================

colonoscopy_standard_months = 120      # 10 years
ct_colonography_months = 60            # 5 years
flexible_sigmoidoscopy_months = 60     # 5 years
fit_dna_months = 36                    # 3 years
fobt_months = 12                       # 1 year

print(f"\nScreening exclusion (unscreened defined as no screening within):")
print(f"  Colonoscopy: {colonoscopy_standard_months} months")
print(f"  CT colonography: {ct_colonography_months} months")
print(f"  Flexible sigmoidoscopy: {flexible_sigmoidoscopy_months} months")
print(f"  FIT-DNA: {fit_dna_months} months")
print(f"  FOBT: {fobt_months} months")

# Label confirmation mode
label_confirm_mode = "single_code"  # Single CRC code sufficient for label

# Legacy aliases for compatibility
start_date = index_start
end_date = index_end
start_timestamp = start_date + ' 00:00:00'
end_timestamp = end_date + ' 00:00:00'

print("\n" + "="*70)
print("CONFIGURATION COMPLETE")
print("="*70)

# ========================================
# CELL 2
# ========================================


# CELL 1
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_cohort_index AS
-- =============================================================================
-- BASE COHORT: Patient-Month Observations with Temporal Structure
-- =============================================================================

WITH
params AS (
  SELECT
    CAST('{index_start}' AS DATE) AS index_start,
    CAST('{index_end}'   AS DATE) AS index_end
),

-- PATIENT IDENTIFICATION

base_patients AS (
  -- Outpatient encounters
  SELECT DISTINCT pe.PAT_ID
  FROM clarity_cur.PAT_ENC_ENH pe
  JOIN clarity_cur.DEP_LOC_PLOC_SA_ENH dep
    ON dep.department_id = COALESCE(pe.DEPARTMENT_ID, pe.EFFECTIVE_DEPT_ID)
  WHERE pe.CONTACT_DATE >= (SELECT index_start FROM params)
    AND pe.CONTACT_DATE <  (SELECT index_end   FROM params)
    AND pe.APPT_STATUS_C IN (2,6)  -- Completed or Arrived
    AND dep.RPT_GRP_SIX IN ('116001','116002')  -- Our health system
  
  UNION
  
  -- Inpatient admissions
  SELECT DISTINCT pe.PAT_ID
  FROM clarity_cur.PAT_ENC_HSP_HAR_ENH pe
  JOIN clarity_cur.DEP_LOC_PLOC_SA_ENH dep
    ON pe.DEPARTMENT_ID = dep.department_id
  WHERE DATE(pe.HOSP_ADMSN_TIME) >= (SELECT index_start FROM params)
    AND DATE(pe.HOSP_ADMSN_TIME) <  (SELECT index_end   FROM params)
    AND pe.ADT_PATIENT_STAT_C <> 1  -- Not preadmit
    AND pe.ADMIT_CONF_STAT_C <> 3   -- Not canceled
    AND dep.RPT_GRP_SIX IN ('116001','116002')
    AND pe.TOT_CHGS <> 0
    AND COALESCE(pe.acct_billsts_ha_c,-1) NOT IN (40,60,99)
    AND pe.combine_acct_id IS NULL
),

-- TEMPORAL GRID: One observation per patient per month

months AS (
  SELECT explode(
    sequence(
      date_trunc('month', (SELECT index_start FROM params)),
      date_trunc('month', (SELECT index_end FROM params)),
      interval 1 month
    )
  ) AS month_start
),

-- Assign each patient a deterministic "random" day within each month
pat_month AS (
  SELECT
    bp.PAT_ID,
    m.month_start,
    day(last_day(m.month_start)) AS dim,
    -- Hash function ensures same day each run (reproducibility)
    pmod(abs(hash(concat(CAST(bp.PAT_ID AS STRING), '|', CAST(m.month_start AS STRING)))),
         day(last_day(m.month_start))) + 1 AS rnd_day
  FROM base_patients bp
  CROSS JOIN months m
),

index_dates AS (
  SELECT
    PAT_ID,
    date_add(month_start, rnd_day - 1) AS END_DTTM
  FROM pat_month
  WHERE date_add(month_start, rnd_day - 1) >= (SELECT index_start FROM params)
    AND date_add(month_start, rnd_day - 1) <= (SELECT index_end   FROM params)  -- Changed < to <=
),

-- OBSERVABILITY: When did we first see each patient?

first_seen AS (
  SELECT PAT_ID, MIN(first_dt) AS first_seen_dt
  FROM (
    SELECT pe.PAT_ID, CAST(pe.CONTACT_DATE AS DATE) AS first_dt
    FROM clarity_cur.PAT_ENC_ENH pe
    UNION ALL
    SELECT ha.PAT_ID, CAST(ha.HOSP_ADMSN_TIME AS DATE) AS first_dt
    FROM clarity_cur.PAT_ENC_HSP_HAR_ENH ha
  ) z
  GROUP BY PAT_ID
),

-- DEMOGRAPHICS WITH QUALITY FLAGS

demog AS (
  SELECT
    idx.PAT_ID,
    idx.END_DTTM,
    p.BIRTH_DATE,
    FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) AS AGE,
    CASE WHEN p.GENDER = 'Female' THEN 1 ELSE 0 END AS IS_FEMALE,
    CASE WHEN p.MARITAL_STATUS IN ('Married','Significant other') THEN 1 ELSE 0 END AS IS_MARRIED_PARTNER,
    
    CASE
      WHEN p.RACE IN ('Unknown/Refused','None') THEN NULL
      WHEN p.RACE IN ('American Indian / Alaska Native','Multi-Racial','Other Pacific Islander','Native Hawaiian')
        THEN 'Other_Small'
      ELSE p.RACE
    END AS RACE_BUCKETS,
    
    fs.first_seen_dt,
    CAST(months_between(idx.END_DTTM, fs.first_seen_dt) AS INT) AS OBS_MONTHS_PRIOR,
    
    -- Data quality flag
    CASE 
      WHEN FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) > 100 THEN 0
      WHEN FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) < 0 THEN 0
      WHEN CAST(months_between(idx.END_DTTM, fs.first_seen_dt) AS INT) > 
           FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) * 12 THEN 0
      WHEN fs.first_seen_dt > idx.END_DTTM THEN 0
      ELSE 1
    END AS data_quality_flag
    
  FROM index_dates idx
  LEFT JOIN clarity_cur.PATIENT_ENH p
    ON idx.PAT_ID = p.PAT_ID
  LEFT JOIN first_seen fs
    ON idx.PAT_ID = fs.PAT_ID
  WHERE FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) >= 45
    AND FLOOR(datediff(idx.END_DTTM, p.BIRTH_DATE) / 365.25) <= 100
),

-- ONE-HOT ENCODE RACE

onehot AS (
  SELECT
    PAT_ID, END_DTTM, AGE, IS_FEMALE, IS_MARRIED_PARTNER, OBS_MONTHS_PRIOR,
    data_quality_flag,
    CASE WHEN RACE_BUCKETS = 'Caucasian' THEN 1 ELSE 0 END AS RACE_CAUCASIAN,
    CASE WHEN RACE_BUCKETS = 'Black or African American' THEN 1 ELSE 0 END AS RACE_BLACK_OR_AFRICAN_AMERICAN,
    CASE WHEN RACE_BUCKETS = 'Hispanic' THEN 1 ELSE 0 END AS RACE_HISPANIC,
    CASE WHEN RACE_BUCKETS = 'Asian' THEN 1 ELSE 0 END AS RACE_ASIAN,
    CASE WHEN RACE_BUCKETS IN ('Other','Other_Small') THEN 1 ELSE 0 END AS RACE_OTHER
  FROM demog
)

SELECT
  *,
  CASE WHEN OBS_MONTHS_PRIOR >= 24 THEN 1 ELSE 0 END AS HAS_FULL_24M_HISTORY,
  CASE
    WHEN AGE BETWEEN 45 AND 49 THEN 'age_45_49'
    WHEN AGE BETWEEN 50 AND 64 THEN 'age_50_64'
    WHEN AGE BETWEEN 65 AND 74 THEN 'age_65_74'
    WHEN AGE >= 75 THEN 'age_75_plus'
  END AS age_group
FROM onehot
WHERE OBS_MONTHS_PRIOR >= {min_obs_months}
  AND data_quality_flag = 1
""")

print("Base cohort index created")

# ========================================
# CELL 3
# ========================================

# CELL 2
# Add PCP status
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_base_with_pcp AS
SELECT 
  c.*,
  CASE 
    WHEN pcp.PAT_ID IS NOT NULL THEN 1 
    ELSE 0 
  END AS HAS_PCP_AT_END
FROM {trgt_cat}.clncl_ds.herald_test_train_cohort_index c
LEFT JOIN (
  SELECT DISTINCT p.PAT_ID, c.END_DTTM
  FROM {trgt_cat}.clncl_ds.herald_test_train_cohort_index c
  JOIN clarity.pat_pcp p
    ON p.PAT_ID = c.PAT_ID
    AND c.END_DTTM BETWEEN p.EFF_DATE AND COALESCE(p.TERM_DATE, '9999-12-31')
  JOIN clarity_cur.clarity_ser_enh ser
    ON p.PCP_PROV_ID = ser.prov_id
    AND ser.RPT_GRP_ELEVEN_NAME IN ('Integrated-Regional','Integrated')
) pcp ON c.PAT_ID = pcp.PAT_ID AND c.END_DTTM = pcp.END_DTTM
""")

# Identify medical exclusions
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_exclusions AS
SELECT DISTINCT c.PAT_ID, c.END_DTTM
FROM {trgt_cat}.clncl_ds.herald_base_with_pcp c
JOIN clarity_cur.pat_enc_enh pe
  ON pe.PAT_ID = c.PAT_ID
  AND DATE(pe.CONTACT_DATE) <= c.END_DTTM
JOIN clarity_cur.pat_enc_dx_enh dd
  ON dd.PAT_ENC_CSN_ID = pe.PAT_ENC_CSN_ID
WHERE dd.ICD10_CODE RLIKE '{crc_icd_regex}'
   OR dd.ICD10_CODE IN ('Z90.49', 'K91.850')
   OR dd.ICD10_CODE LIKE 'Z51.5%'
""")

print("PCP status and exclusions identified")

# ========================================
# CELL 4
# ========================================

# CELL  3
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_with_labels AS
WITH future_crc AS (
  -- Find CRC diagnoses in 6-month prediction window after END_DTTM
  SELECT DISTINCT 
    c.PAT_ID, 
    c.END_DTTM,
    FIRST_VALUE(dd.ICD10_CODE) OVER (
      PARTITION BY c.PAT_ID, c.END_DTTM 
      ORDER BY pe.CONTACT_DATE, dd.ICD10_CODE
    ) AS ICD10_CODE
  FROM {trgt_cat}.clncl_ds.herald_base_with_pcp c
  JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID
    AND DATE(pe.CONTACT_DATE) > c.END_DTTM
    AND DATE(pe.CONTACT_DATE) <= DATEADD(MONTH, {label_months}, c.END_DTTM)
  JOIN clarity_cur.pat_enc_dx_enh dd
    ON dd.PAT_ENC_CSN_ID = pe.PAT_ENC_CSN_ID
    AND dd.ICD10_CODE RLIKE '{crc_icd_regex}'
),
next_contact AS (
  -- Find next contact within 12-month follow-up window
  SELECT 
    c.PAT_ID,
    c.END_DTTM,
    MIN(pe.CONTACT_DATE) as next_visit_date
  FROM {trgt_cat}.clncl_ds.herald_base_with_pcp c
  JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID
    AND DATE(pe.CONTACT_DATE) > c.END_DTTM
    AND DATE(pe.CONTACT_DATE) <= DATEADD(MONTH, {min_followup_months}, c.END_DTTM)
    AND pe.APPT_STATUS_C IN (2,6)
  GROUP BY c.PAT_ID, c.END_DTTM
)
SELECT 
  c.*,
  
  -- Label: CRC diagnosis in 6-month prediction window
  CASE WHEN fc.PAT_ID IS NOT NULL THEN 1 ELSE 0 END AS FUTURE_CRC_EVENT,
  
  -- Diagnostic details (for analysis, not features)
  fc.ICD10_CODE,
  CASE
    WHEN fc.ICD10_CODE RLIKE '^C18' THEN 'C18'
    WHEN fc.ICD10_CODE RLIKE '^C19' THEN 'C19'
    WHEN fc.ICD10_CODE RLIKE '^C20' THEN 'C20'
    WHEN fc.ICD10_CODE RLIKE '^C21' THEN 'C21'
    ELSE NULL
  END AS ICD10_GROUP,
  
  -- DEBUGGING ONLY: These fields used to calculate LABEL_USABLE but excluded from 
  -- final cohort to prevent data leakage. Kept here for validation purposes.
  nc.next_visit_date,
  DATEDIFF(
    COALESCE(nc.next_visit_date, DATEADD(MONTH, {min_followup_months}, c.END_DTTM)),
    c.END_DTTM
  ) AS observable_days,
  
  -- Label quality: Determine if we can confidently assign this label
  CASE
    -- POSITIVE CASES: Always usable (we observed the event)
    WHEN fc.PAT_ID IS NOT NULL THEN 1
    
    -- NEGATIVE TIER 1 (High confidence): Return visit AFTER 6-month prediction window
    -- Rationale: Covers full prediction window, definitively confirms no diagnosis
    WHEN fc.PAT_ID IS NULL 
     AND nc.next_visit_date > DATEADD(MONTH, {label_months}, c.END_DTTM)
     THEN 1
    
    -- NEGATIVE TIER 2 (Medium confidence): Return in months 4-6 AND has PCP
    -- Rationale: Late in prediction window + continuous care relationship
    WHEN fc.PAT_ID IS NULL 
     AND c.HAS_PCP_AT_END = 1
     AND nc.next_visit_date > DATEADD(MONTH, 4, c.END_DTTM)
     AND nc.next_visit_date <= DATEADD(MONTH, {label_months}, c.END_DTTM)
     THEN 1
    
    -- NEGATIVE TIER 3 (Lower confidence): No return visit BUT has PCP
    -- Rationale: PCP relationship implies would document if CRC diagnosed elsewhere
    -- Note: By definition these patients have 12 months elapsed (eligibility requirement)
    WHEN fc.PAT_ID IS NULL 
     AND c.HAS_PCP_AT_END = 1
     AND nc.next_visit_date IS NULL
     THEN 1
    
    -- EXCLUDE: All other cases (no PCP + no return, or early return without PCP)
    ELSE 0
  END AS LABEL_USABLE,
  
  -- Label confidence for stratified validation
  CASE
    WHEN fc.PAT_ID IS NOT NULL THEN 'positive'
    WHEN nc.next_visit_date > DATEADD(MONTH, {label_months}, c.END_DTTM) 
      THEN 'high_confidence_negative'
    WHEN nc.next_visit_date > DATEADD(MONTH, 4, c.END_DTTM) 
     AND c.HAS_PCP_AT_END = 1 
      THEN 'medium_confidence_negative'
    WHEN nc.next_visit_date IS NULL 
     AND c.HAS_PCP_AT_END = 1 
      THEN 'assumed_negative_with_pcp'
    ELSE 'excluded_insufficient_observability'
  END AS LABEL_CONFIDENCE
  
FROM {trgt_cat}.clncl_ds.herald_base_with_pcp c
LEFT JOIN future_crc fc 
  ON c.PAT_ID = fc.PAT_ID AND c.END_DTTM = fc.END_DTTM
LEFT JOIN next_contact nc
  ON c.PAT_ID = nc.PAT_ID AND c.END_DTTM = nc.END_DTTM
LEFT ANTI JOIN {trgt_cat}.clncl_ds.herald_exclusions e
  ON c.PAT_ID = e.PAT_ID AND c.END_DTTM = e.END_DTTM
""")

print("Labels assigned with tiered observability criteria")

# ========================================
# CELL 5
# ========================================

# CELL 4
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_cohort AS
SELECT
  -- Identifiers
  PAT_ID,
  END_DTTM,

  -- Demographics (all known at prediction time)
  AGE,
  IS_FEMALE,
  IS_MARRIED_PARTNER,
  OBS_MONTHS_PRIOR,

  -- Quality flags
  data_quality_flag,

  -- Race (one-hot encoded)
  RACE_CAUCASIAN,
  RACE_BLACK_OR_AFRICAN_AMERICAN,
  RACE_HISPANIC,
  RACE_ASIAN,
  RACE_OTHER,

  -- Derived features
  HAS_FULL_24M_HISTORY,
  age_group,
  HAS_PCP_AT_END,

  -- LABEL
  FUTURE_CRC_EVENT,

  -- Diagnostic info (for analysis only, NOT features)
  ICD10_CODE,
  ICD10_GROUP,

  -- Label quality metadata
  LABEL_USABLE,
  LABEL_CONFIDENCE

  -- NOTE: observable_days and next_visit_date were used to calculate
  -- LABEL_USABLE and LABEL_CONFIDENCE but are excluded here to prevent
  -- data leakage. They contain future information and should not be features.

FROM {trgt_cat}.clncl_ds.herald_with_labels
WHERE LABEL_USABLE = 1  -- ADDED: Explicit filter
""")

print("Training cohort created")

# ========================================
# CELL 6
# ========================================

# CELL 6
# =============================================================================
# DUAL SCREENING EXCLUSION: VBC Table + Supplemental Internal Check
# =============================================================================

# Create comprehensive supplemental screening exclusion table with modality tracking
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_internal_screening_exclusions AS
WITH all_internal_screening AS (
  SELECT
    op.PAT_ID,
    fc.END_DTTM,
    DATE(op.ORDERING_DATE) as procedure_date,
    op.PROC_CODE,
    op.PROC_NAME,
    CASE
      WHEN op.PROC_CODE IN ('45378','45380','45381','45382','45384','45385','45386',
                            '45388','45389','45390','45391','45392','45393','45398')
           OR LOWER(op.PROC_NAME) LIKE '%colonoscopy%'
        THEN 'colonoscopy'
      WHEN op.PROC_CODE IN ('74261','74262','74263')
           OR LOWER(op.PROC_NAME) LIKE '%ct colonography%'
           OR LOWER(op.PROC_NAME) LIKE '%virtual colonoscopy%'
        THEN 'ct_colonography'
      WHEN op.PROC_CODE IN ('45330','45331','45332','45333','45334','45335','45337',
                            '45338','45339','45340','45341','45342','45345','45346',
                            '45347','45349','45350')
           OR LOWER(op.PROC_NAME) LIKE '%sigmoidoscopy%'
        THEN 'flexible_sigmoidoscopy'
      WHEN op.PROC_CODE IN ('81528')
           OR LOWER(op.PROC_NAME) LIKE '%cologuard%'
           OR LOWER(op.PROC_NAME) LIKE '%fit-dna%'
        THEN 'fit_dna'
      WHEN op.PROC_CODE IN ('82270','82274','G0328')
           OR LOWER(op.PROC_NAME) LIKE '%fobt%'
           OR LOWER(op.PROC_NAME) LIKE '%fecal occult%'
           OR (LOWER(op.PROC_NAME) LIKE '%fit%' AND LOWER(op.PROC_NAME) LIKE '%test%')
        THEN 'fobt'
      ELSE 'other'
    END as screening_type,
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
    END as screening_valid_years
  FROM {trgt_cat}.clncl_ds.herald_test_train_cohort fc
  JOIN clarity_cur.ORDER_PROC_ENH op
    ON op.PAT_ID = fc.PAT_ID
    -- ✅ Only join procedures that occurred BEFORE the observation date
    AND DATE(op.ORDERING_DATE) <= fc.END_DTTM
    -- ✅ And only from dates where we trust internal procedure data
    AND DATE(op.ORDERING_DATE) >= DATE('2021-07-01')
  WHERE op.RPT_GRP_SIX IN ('116001','116002')
    AND op.ORDER_STATUS NOT IN ('Canceled', 'Cancelled')
    AND (
      op.PROC_CODE IN ('45378','45380','45381','45382','45384','45385','45386',
                       '45388','45389','45390','45391','45392','45393','45398',
                       '74261','74262','74263',
                       '45330','45331','45332','45333','45334','45335','45337',
                       '45338','45339','45340','45341','45342','45345','45346',
                       '45347','45349','45350',
                       '81528',
                       '82270','82274','G0328')
      OR LOWER(op.PROC_NAME) LIKE '%colonoscopy%'
      OR LOWER(op.PROC_NAME) LIKE '%sigmoidoscopy%'
      OR LOWER(op.PROC_NAME) LIKE '%cologuard%'
      OR LOWER(op.PROC_NAME) LIKE '%fobt%'
      OR LOWER(op.PROC_NAME) LIKE '%fecal occult%'
    )
),
screening_by_type AS (
  SELECT
    PAT_ID,
    END_DTTM,
    screening_type,
    MAX(procedure_date) as last_screening_date,
    MIN(screening_valid_years) as min_valid_years,
    COUNT(*) as screening_count
  FROM all_internal_screening
  WHERE screening_type != 'other'  -- ✅ Filter out unclassified procedures
  GROUP BY PAT_ID, END_DTTM, screening_type
)
SELECT
  PAT_ID,
  END_DTTM,
  MAX(last_screening_date) as last_screening_date,
  MAX(min_valid_years) as max_valid_years,  -- ✅ You correctly use MAX here for conservative exclusion
  
  -- Screening type flags
  MAX(CASE WHEN screening_type = 'colonoscopy' THEN 1 ELSE 0 END) as had_colonoscopy,
  MAX(CASE WHEN screening_type = 'ct_colonography' THEN 1 ELSE 0 END) as had_ct_colonography,
  MAX(CASE WHEN screening_type = 'flexible_sigmoidoscopy' THEN 1 ELSE 0 END) as had_sigmoidoscopy,
  MAX(CASE WHEN screening_type = 'fit_dna' THEN 1 ELSE 0 END) as had_fit_dna,
  MAX(CASE WHEN screening_type = 'fobt' THEN 1 ELSE 0 END) as had_fobt,
  
  -- Most recent date by type
  MAX(CASE WHEN screening_type = 'colonoscopy' THEN last_screening_date END) as last_colonoscopy_date,
  MAX(CASE WHEN screening_type = 'ct_colonography' THEN last_screening_date END) as last_ct_colonography_date,
  MAX(CASE WHEN screening_type = 'flexible_sigmoidoscopy' THEN last_screening_date END) as last_sigmoidoscopy_date,
  MAX(CASE WHEN screening_type = 'fit_dna' THEN last_screening_date END) as last_fit_dna_date,
  MAX(CASE WHEN screening_type = 'fobt' THEN last_screening_date END) as last_fobt_date,
  
  -- Count by type
  MAX(CASE WHEN screening_type = 'colonoscopy' THEN screening_count ELSE 0 END) as colonoscopy_count,
  MAX(CASE WHEN screening_type = 'fobt' THEN screening_count ELSE 0 END) as fobt_count

FROM screening_by_type
GROUP BY PAT_ID, END_DTTM
""")

print("Comprehensive internal screening exclusions identified (all modalities, data from 2021-07-01 onward)")

# =============================================================================
# BUILD FINAL COHORT WITH DUAL SCREENING EXCLUSION
# =============================================================================

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_final_cohort AS
WITH current_screening_status AS (
  SELECT
    PAT_ID,
    COLON_SCREEN_MET_FLAG,
    COLON_SCREEN_EXCL_FLAG,
    CSCOPY_LAST_PROC_DT,
    CT_CSCOPY_LAST_PROC_DT,
    FOBT_LAST_PROC_DT
  FROM prod.clncl_cur.vbc_colon_cancer_screen
  WHERE COLON_SCREEN_EXCL_FLAG = 'N'
),
patient_first_obs AS (
  SELECT PAT_ID, MIN(END_DTTM) as first_obs_date
  FROM {trgt_cat}.clncl_ds.herald_test_train_cohort
  GROUP BY PAT_ID
),
enhanced_cohort AS (
  SELECT
    fc.*,
    CAST(months_between(fc.END_DTTM, pfo.first_obs_date) AS INT) as months_since_cohort_entry,
    COALESCE(cs.COLON_SCREEN_MET_FLAG, 'N') as current_screen_status,
    
    -- VBC screening dates
    cs.CSCOPY_LAST_PROC_DT as vbc_last_colonoscopy_date,
    cs.FOBT_LAST_PROC_DT as vbc_last_fobt_date,
    
    -- Internal screening tracking
    ise.last_screening_date as last_internal_screening_date,
    ise.max_valid_years as screening_valid_years,
    
    -- Screening modality flags
    COALESCE(ise.had_colonoscopy, 0) as had_colonoscopy_in_lookback,
    COALESCE(ise.had_ct_colonography, 0) as had_ct_colonography_in_lookback,
    COALESCE(ise.had_sigmoidoscopy, 0) as had_sigmoidoscopy_in_lookback,
    COALESCE(ise.had_fit_dna, 0) as had_fit_dna_in_lookback,
    COALESCE(ise.had_fobt, 0) as had_fobt_in_lookback,
    
    -- Most recent dates by modality
    ise.last_colonoscopy_date,
    ise.last_ct_colonography_date,
    ise.last_sigmoidoscopy_date,
    ise.last_fit_dna_date,
    ise.last_fobt_date,
    
    -- Screening counts
    COALESCE(ise.colonoscopy_count, 0) as colonoscopy_count,
    COALESCE(ise.fobt_count, 0) as fobt_count,
    
    -- Exclusion logic: per-modality validity check
    -- Each screening type is checked against its OWN validity window.
    -- This avoids a bug where an expired short-lived screening (e.g. FOBT)
    -- could be incorrectly validated by a different modality's longer window.
    CASE
      WHEN ise.last_colonoscopy_date IS NOT NULL
       AND ise.last_colonoscopy_date > DATEADD(YEAR, -10, fc.END_DTTM) THEN 1
      WHEN ise.last_ct_colonography_date IS NOT NULL
       AND ise.last_ct_colonography_date > DATEADD(YEAR, -5, fc.END_DTTM) THEN 1
      WHEN ise.last_sigmoidoscopy_date IS NOT NULL
       AND ise.last_sigmoidoscopy_date > DATEADD(YEAR, -5, fc.END_DTTM) THEN 1
      WHEN ise.last_fit_dna_date IS NOT NULL
       AND ise.last_fit_dna_date > DATEADD(YEAR, -3, fc.END_DTTM) THEN 1
      WHEN ise.last_fobt_date IS NOT NULL
       AND ise.last_fobt_date > DATEADD(YEAR, -1, fc.END_DTTM) THEN 1
      ELSE 0
    END as excluded_by_internal_screening
    
  FROM {trgt_cat}.clncl_ds.herald_test_train_cohort fc
  LEFT JOIN patient_first_obs pfo
    ON fc.PAT_ID = pfo.PAT_ID
  LEFT JOIN current_screening_status cs
    ON fc.PAT_ID = cs.PAT_ID
  LEFT JOIN {trgt_cat}.clncl_ds.herald_internal_screening_exclusions ise
    ON fc.PAT_ID = ise.PAT_ID AND fc.END_DTTM = ise.END_DTTM
)
SELECT *
FROM enhanced_cohort
WHERE NOT (
  current_screen_status = 'Y' 
  OR excluded_by_internal_screening = 1
)
""")

# =============================================================================
# COMPREHENSIVE STATISTICS WITH SCREENING MODALITY BREAKDOWN
# =============================================================================

final_stats = spark.sql(f"""
WITH before_supplemental AS (
  SELECT
    COUNT(*) as obs_before,
    COUNT(DISTINCT PAT_ID) as patients_before
  FROM {trgt_cat}.clncl_ds.herald_test_train_cohort
  WHERE LABEL_USABLE = 1
),
after_vbc_exclusion AS (
  SELECT
    fc.*,
    COALESCE(cs.COLON_SCREEN_MET_FLAG, 'N') as screen_status
  FROM {trgt_cat}.clncl_ds.herald_test_train_cohort fc
  LEFT JOIN prod.clncl_cur.vbc_colon_cancer_screen cs
    ON fc.PAT_ID = cs.PAT_ID
  WHERE (cs.COLON_SCREEN_MET_FLAG = 'N' OR cs.COLON_SCREEN_MET_FLAG IS NULL)
    AND LABEL_USABLE = 1
),
vbc_stats AS (
  SELECT
    COUNT(*) as obs_after_vbc,
    COUNT(DISTINCT PAT_ID) as patients_after_vbc
  FROM after_vbc_exclusion
),
final_stats AS (
  SELECT
    COUNT(*) as final_obs,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    SUM(FUTURE_CRC_EVENT) as positive_cases,
    AVG(FUTURE_CRC_EVENT) * 100 as positive_rate_pct,
    SUM(excluded_by_internal_screening) as excluded_by_supplement
  FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
)
SELECT
  b.obs_before,
  b.patients_before,
  v.obs_after_vbc,
  v.patients_after_vbc,
  b.obs_before - v.obs_after_vbc as excluded_by_vbc_table,
  f.final_obs,
  f.unique_patients,
  v.obs_after_vbc - f.final_obs as excluded_by_supplemental,  -- ← Fixed typo
  f.positive_cases,
  f.positive_rate_pct
FROM before_supplemental b
CROSS JOIN vbc_stats v
CROSS JOIN final_stats f
""").collect()[0]

print("="*70)
print("SCREENING EXCLUSION IMPACT")
print("="*70)
print(f"\nStarting point (label-usable observations):")
print(f"  Observations: {final_stats['obs_before']:,}")
print(f"  Patients: {final_stats['patients_before']:,}")

print(f"\nAfter VBC screening table exclusion:")
print(f"  Observations: {final_stats['obs_after_vbc']:,}")
print(f"  Patients: {final_stats['patients_after_vbc']:,}")
print(f"  Excluded by VBC table: {final_stats['excluded_by_vbc_table']:,}")

print(f"\nAfter supplemental internal screening exclusion (all modalities, data from 2021-07-01 onward):")
print(f"  Final observations: {final_stats['final_obs']:,}")
print(f"  Final patients: {final_stats['unique_patients']:,}")
print(f"  Excluded by supplemental check: {final_stats['excluded_by_supplemental']:,}")
if final_stats['excluded_by_vbc_table'] > 0:
    print(f"  Supplemental capture rate: {final_stats['excluded_by_supplemental']/final_stats['excluded_by_vbc_table']*100:.2f}% additional exclusions")

print(f"\nFinal cohort characteristics:")
print(f"  Positive cases: {final_stats['positive_cases']:,}")
print(f"  Event rate: {final_stats['positive_rate_pct']:.4f}%")

# Add screening modality distribution analysis
screening_modality_stats = spark.sql(f"""
SELECT
  'Colonoscopy' as modality,
  SUM(had_colonoscopy) as patient_count,
  ROUND(AVG(CASE WHEN had_colonoscopy = 1 THEN colonoscopy_count ELSE 0 END), 2) as avg_procedures_per_patient,
  ROUND(SUM(had_colonoscopy) * 100.0 / COUNT(*), 2) as pct_of_excluded
FROM {trgt_cat}.clncl_ds.herald_internal_screening_exclusions

UNION ALL

SELECT
  'FOBT/FIT',
  SUM(had_fobt),
  ROUND(AVG(CASE WHEN had_fobt = 1 THEN fobt_count ELSE 0 END), 2),
  ROUND(SUM(had_fobt) * 100.0 / COUNT(*), 2)
FROM {trgt_cat}.clncl_ds.herald_internal_screening_exclusions

UNION ALL

SELECT
  'FIT-DNA (Cologuard)',
  SUM(had_fit_dna),
  ROUND(AVG(CASE WHEN had_fit_dna = 1 THEN 1 ELSE 0 END), 2),
  ROUND(SUM(had_fit_dna) * 100.0 / COUNT(*), 2)
FROM {trgt_cat}.clncl_ds.herald_internal_screening_exclusions

UNION ALL

SELECT
  'Flexible Sigmoidoscopy',
  SUM(had_sigmoidoscopy),
  ROUND(AVG(CASE WHEN had_sigmoidoscopy = 1 THEN 1 ELSE 0 END), 2),
  ROUND(SUM(had_sigmoidoscopy) * 100.0 / COUNT(*), 2)
FROM {trgt_cat}.clncl_ds.herald_internal_screening_exclusions

UNION ALL

SELECT
  'CT Colonography',
  SUM(had_ct_colonography),
  ROUND(AVG(CASE WHEN had_ct_colonography = 1 THEN 1 ELSE 0 END), 2),
  ROUND(SUM(had_ct_colonography) * 100.0 / COUNT(*), 2)
FROM {trgt_cat}.clncl_ds.herald_internal_screening_exclusions

ORDER BY patient_count DESC
""").toPandas()

print("\n" + "="*70)
print("SCREENING MODALITY DISTRIBUTION (Excluded Patients)")
print("="*70)
print(screening_modality_stats.to_string(index=False))

print("\n" + "="*70)
print("✓ DUAL SCREENING EXCLUSION COMPLETE")
print("="*70)

# ========================================
# CELL 7
# ========================================

# CELL 7
# Screening modality distribution among excluded patients
screening_modality_stats = spark.sql(f"""
SELECT
  'Colonoscopy' as modality,
  SUM(had_colonoscopy) as patient_count,
  ROUND(AVG(colonoscopy_count), 2) as avg_procedures_per_patient,
  ROUND(SUM(had_colonoscopy) * 100.0 / COUNT(*), 2) as pct_of_excluded
FROM {trgt_cat}.clncl_ds.herald_internal_screening_exclusions

UNION ALL

SELECT
  'FOBT/FIT',
  SUM(had_fobt),
  ROUND(AVG(fobt_count), 2),
  ROUND(SUM(had_fobt) * 100.0 / COUNT(*), 2)
FROM {trgt_cat}.clncl_ds.herald_internal_screening_exclusions

UNION ALL

SELECT
  'FIT-DNA (Cologuard)',
  SUM(had_fit_dna),
  ROUND(AVG(CASE WHEN had_fit_dna = 1 THEN 1 ELSE 0 END), 2),
  ROUND(SUM(had_fit_dna) * 100.0 / COUNT(*), 2)
FROM {trgt_cat}.clncl_ds.herald_internal_screening_exclusions

UNION ALL

SELECT
  'Flexible Sigmoidoscopy',
  SUM(had_sigmoidoscopy),
  ROUND(AVG(CASE WHEN had_sigmoidoscopy = 1 THEN 1 ELSE 0 END), 2),
  ROUND(SUM(had_sigmoidoscopy) * 100.0 / COUNT(*), 2)
FROM {trgt_cat}.clncl_ds.herald_internal_screening_exclusions

UNION ALL

SELECT
  'CT Colonography',
  SUM(had_ct_colonography),
  ROUND(AVG(CASE WHEN had_ct_colonography = 1 THEN 1 ELSE 0 END), 2),
  ROUND(SUM(had_ct_colonography) * 100.0 / COUNT(*), 2)
FROM {trgt_cat}.clncl_ds.herald_internal_screening_exclusions

ORDER BY patient_count DESC
""").toPandas()

print("\n" + "="*70)
print("SCREENING MODALITY DISTRIBUTION (Excluded Patients)")
print("="*70)
print(screening_modality_stats.to_string(index=False))

# ========================================
# CELL 8
# ========================================

#  CELL 8
# Detailed quarterly analysis
quarterly_detail = spark.sql(f"""
SELECT 
  DATE_FORMAT(END_DTTM, 'yyyy-Q') as quarter,
  COUNT(*) as obs,
  SUM(FUTURE_CRC_EVENT) as events,
  ROUND(AVG(FUTURE_CRC_EVENT) * 100, 4) as rate_pct
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
GROUP BY DATE_FORMAT(END_DTTM, 'yyyy-Q')
ORDER BY quarter
""").toPandas()

print("\nQuarterly Event Rate Analysis:")
print("="*60)
print(quarterly_detail.to_string(index=False))

# Calculate decline
first_quarter_rate = quarterly_detail.iloc[0]['rate_pct']
last_quarter_rate = quarterly_detail.iloc[-1]['rate_pct']
decline_pct = ((first_quarter_rate - last_quarter_rate) / first_quarter_rate) * 100

print(f"\nDecline Analysis:")
print(f"  First quarter rate: {first_quarter_rate:.4f}%")
print(f"  Last quarter rate: {last_quarter_rate:.4f}%")
print(f"  Total decline: {decline_pct:.1f}%")
print(f"  Ratio: {first_quarter_rate / last_quarter_rate:.2f}x higher in first quarter")

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(quarterly_detail['quarter'], quarterly_detail['rate_pct'], 
         marker='o', linewidth=2, markersize=8)
plt.axhline(y=0.025, color='r', linestyle='--', linewidth=2, 
            label='Expected incident rate (~0.025%)')
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Event Rate (%)', fontsize=12)
plt.title('CRC Event Rate by Quarter: Evidence of Prevalent Case Contamination', 
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# ========================================
# CELL 9
# ========================================

# CELL 9
#  Comprehensive summary
print("="*70)
print("COHORT CREATION SUMMARY")
print("="*70)

summary_stats = spark.sql(f"""
SELECT 
  COUNT(*) as total_obs,
  COUNT(DISTINCT PAT_ID) as unique_patients,
  ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT PAT_ID), 1) as avg_obs_per_patient,
  SUM(FUTURE_CRC_EVENT) as positive_cases,
  ROUND(AVG(FUTURE_CRC_EVENT) * 100, 4) as overall_rate_pct,
  MIN(END_DTTM) as earliest_date,
  MAX(END_DTTM) as latest_date,
  ROUND(AVG(AGE), 1) as avg_age,
  ROUND(AVG(IS_FEMALE) * 100, 1) as pct_female,
  ROUND(AVG(HAS_PCP_AT_END) * 100, 1) as pct_with_pcp,
  ROUND(AVG(OBS_MONTHS_PRIOR), 1) as avg_obs_months
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
""").collect()[0]

print(f"\nCohort Composition:")
print(f"  Total observations: {summary_stats['total_obs']:,}")
print(f"  Unique patients: {summary_stats['unique_patients']:,}")
print(f"  Average observations per patient: {summary_stats['avg_obs_per_patient']}")
print(f"  Date range: {summary_stats['earliest_date']} to {summary_stats['latest_date']}")

print(f"\nOutcome Distribution:")
print(f"  Positive cases: {summary_stats['positive_cases']:,}")
print(f"  Overall event rate: {summary_stats['overall_rate_pct']}%")
print(f"  Class imbalance: 1:{int(100/summary_stats['overall_rate_pct'])}")

print(f"\nPopulation Characteristics:")
print(f"  Average age: {summary_stats['avg_age']} years")
print(f"  Female: {summary_stats['pct_female']}%")
print(f"  Has PCP: {summary_stats['pct_with_pcp']}%")
print(f"  Average prior observability: {summary_stats['avg_obs_months']} months")

print("\n" + "="*70)

# ========================================
# CELL 10
# ========================================

# CELL 10
# =============================================================================
# LABEL CONFIDENCE TIER ANALYSIS
# =============================================================================

print("="*70)
print("LABEL QUALITY TIER BREAKDOWN")
print("="*70)

# Overall distribution by confidence tier
tier_breakdown = spark.sql(f"""
SELECT 
  LABEL_CONFIDENCE,
  COUNT(*) as observations,
  COUNT(DISTINCT PAT_ID) as unique_patients,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct_of_cohort,
  SUM(FUTURE_CRC_EVENT) as positive_cases,
  ROUND(AVG(FUTURE_CRC_EVENT) * 100, 4) as event_rate_pct,
  ROUND(AVG(HAS_PCP_AT_END) * 100, 1) as pct_with_pcp,
  ROUND(AVG(AGE), 1) as avg_age
  -- REMOVED: ROUND(AVG(observable_days), 1) as avg_observable_days
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
GROUP BY LABEL_CONFIDENCE
ORDER BY 
  CASE LABEL_CONFIDENCE
    WHEN 'positive' THEN 1
    WHEN 'high_confidence_negative' THEN 2
    WHEN 'medium_confidence_negative' THEN 3
    WHEN 'assumed_negative_with_pcp' THEN 4
    ELSE 5
  END
""").toPandas()

print("\nDistribution by Label Confidence Tier:")
print(tier_breakdown.to_string(index=False))

# Calculate what percentage of negatives are in each tier
negative_breakdown = spark.sql(f"""
SELECT 
  LABEL_CONFIDENCE,
  COUNT(*) as negative_cases,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct_of_negatives
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
WHERE FUTURE_CRC_EVENT = 0
GROUP BY LABEL_CONFIDENCE
ORDER BY 
  CASE LABEL_CONFIDENCE
    WHEN 'high_confidence_negative' THEN 1
    WHEN 'medium_confidence_negative' THEN 2
    WHEN 'assumed_negative_with_pcp' THEN 3
    ELSE 4
  END
""").toPandas()

print("\n" + "="*60)
print("NEGATIVE LABEL QUALITY BREAKDOWN")
print("="*60)
print(negative_breakdown.to_string(index=False))

# Temporal distribution of label confidence
temporal_confidence = spark.sql(f"""
SELECT 
  DATE_FORMAT(END_DTTM, 'yyyy-Q') as quarter,
  LABEL_CONFIDENCE,
  COUNT(*) as observations
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
GROUP BY DATE_FORMAT(END_DTTM, 'yyyy-Q'), LABEL_CONFIDENCE
ORDER BY quarter, LABEL_CONFIDENCE
""").toPandas()

# Pivot for easier viewing
temporal_pivot = temporal_confidence.pivot(
    index='quarter', 
    columns='LABEL_CONFIDENCE', 
    values='observations'
).fillna(0)

print("\n" + "="*60)
print("LABEL CONFIDENCE BY QUARTER")
print("="*60)
print(temporal_pivot.to_string())

# Key insights
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

total_obs = tier_breakdown['observations'].sum()
total_negatives = tier_breakdown[tier_breakdown['LABEL_CONFIDENCE'] != 'positive']['observations'].sum()
high_conf_neg = tier_breakdown[tier_breakdown['LABEL_CONFIDENCE'] == 'high_confidence_negative']['observations'].values
medium_conf_neg = tier_breakdown[tier_breakdown['LABEL_CONFIDENCE'] == 'medium_confidence_negative']['observations'].values
assumed_neg = tier_breakdown[tier_breakdown['LABEL_CONFIDENCE'] == 'assumed_negative_with_pcp']['observations'].values

print(f"\nTotal observations: {total_obs:,}")
print(f"Positive cases: {tier_breakdown[tier_breakdown['LABEL_CONFIDENCE'] == 'positive']['observations'].values[0]:,}")
print(f"Negative cases: {total_negatives:,}")

if len(high_conf_neg) > 0:
    print(f"\nNegative Label Quality:")
    print(f"  High confidence (return after month 6): {high_conf_neg[0]:,} ({high_conf_neg[0]/total_negatives*100:.1f}%)")
if len(medium_conf_neg) > 0:
    print(f"  Medium confidence (return months 4-6 + PCP): {medium_conf_neg[0]:,} ({medium_conf_neg[0]/total_negatives*100:.1f}%)")
if len(assumed_neg) > 0:
    print(f"  Assumed negative (no return but has PCP): {assumed_neg[0]:,} ({assumed_neg[0]/total_negatives*100:.1f}%)")

print(f"\nPCP Coverage:")
print(f"  Overall: {tier_breakdown['pct_with_pcp'].mean():.1f}%")
print(f"  In assumed negatives: {tier_breakdown[tier_breakdown['LABEL_CONFIDENCE'] == 'assumed_negative_with_pcp']['pct_with_pcp'].values[0] if len(assumed_neg) > 0 else 0:.1f}%")

print("\n" + "="*70)

# ========================================
# CELL 11
# ========================================

# CELL 11
#  CHECK 1: Verify no duplicates
dupe_check = spark.sql(f"""
SELECT 
  COUNT(*) as total_rows,
  COUNT(DISTINCT PAT_ID, END_DTTM) as unique_keys,
  COUNT(*) - COUNT(DISTINCT PAT_ID, END_DTTM) as duplicates
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
""").collect()[0]

print("="*60)
print("DUPLICATE CHECK")
print("="*60)
print(f"Total rows: {dupe_check['total_rows']:,}")
print(f"Unique keys: {dupe_check['unique_keys']:,}")
print(f"Duplicates: {dupe_check['duplicates']:,}")
print(f"Status: {'✓ PASS' if dupe_check['duplicates'] == 0 else '✗ FAIL'}")
print("="*60)

# ========================================
# CELL 12
# ========================================

# CELL 13
# CHECK 3: Age distribution
age_check = spark.sql(f"""
SELECT 
  MIN(AGE) as min_age,
  PERCENTILE(AGE, 0.25) as q1_age,
  PERCENTILE(AGE, 0.5) as median_age,
  PERCENTILE(AGE, 0.75) as q3_age,
  MAX(AGE) as max_age,
  SUM(CASE WHEN AGE < 45 THEN 1 ELSE 0 END) as under_45,
  SUM(CASE WHEN AGE > 100 THEN 1 ELSE 0 END) as over_100
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
""").collect()[0]

print("\n" + "="*60)
print("AGE DISTRIBUTION")
print("="*60)
print(f"Range: {age_check['min_age']} - {age_check['max_age']}")
print(f"Q1: {age_check['q1_age']}")
print(f"Median: {age_check['median_age']}")
print(f"Q3: {age_check['q3_age']}")
print(f"Under 45: {age_check['under_45']:,}")
print(f"Over 100: {age_check['over_100']:,}")
print(f"Status: {'✓ PASS' if age_check['under_45'] == 0 and age_check['over_100'] == 0 else '⚠ WARNING - Check age issues'}")
print("="*60)

# ========================================
# CELL 13
# ========================================

# CELL 14
# CHECK 4: PCP status impact
pcp_impact = spark.sql(f"""
SELECT 
  HAS_PCP_AT_END,
  COUNT(*) as obs,
  AVG(LABEL_USABLE) * 100 as usable_pct,
  SUM(FUTURE_CRC_EVENT) as events,
  AVG(FUTURE_CRC_EVENT) * 100 as crc_rate
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
GROUP BY HAS_PCP_AT_END
""").toPandas()

print("\n" + "="*60)
print("PCP IMPACT ON LABEL USABILITY AND EVENT RATES")
print("="*60)
print(pcp_impact.to_string(index=False))
print("\nInterpretation:")
print("- Patients with PCPs typically have higher detection rates")
print("- This reflects better documentation and follow-up")
print("="*60)

# ========================================
# CELL 14
# ========================================

# CELL 15
# CHECK 5: CRC subtype distribution
crc_distribution = spark.sql(f"""
SELECT 
  ICD10_GROUP,
  COUNT(*) as cases,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
WHERE FUTURE_CRC_EVENT = 1
  AND ICD10_GROUP IS NOT NULL
GROUP BY ICD10_GROUP
ORDER BY cases DESC
""").toPandas()

print("\n" + "="*60)
print("CRC ANATOMICAL DISTRIBUTION")
print("="*60)
print(crc_distribution.to_string(index=False))
print("\nExpected distribution:")
print("  C18 (Colon): ~65-75%")
print("  C20 (Rectum): ~15-20%")
print("  C21 (Anus): ~5-10%")
print("  C19 (Rectosigmoid): ~3-5%")
print("="*60)

# ========================================
# CELL 15
# ========================================

# CELL 16
# CHECK 6: Observability patterns by quarter
obs_check = spark.sql(f"""
SELECT 
  DATE_FORMAT(END_DTTM, 'yyyy-Q') as quarter,
  MIN(OBS_MONTHS_PRIOR) as min_obs_months,
  PERCENTILE(OBS_MONTHS_PRIOR, 0.25) as q1_obs_months,
  AVG(OBS_MONTHS_PRIOR) as avg_obs_months,
  PERCENTILE(OBS_MONTHS_PRIOR, 0.75) as q3_obs_months,
  MAX(OBS_MONTHS_PRIOR) as max_obs_months
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
GROUP BY DATE_FORMAT(END_DTTM, 'yyyy-Q')
ORDER BY quarter
""").toPandas()

print("\n" + "="*60)
print("OBSERVABILITY BY QUARTER")
print("="*60)
print(obs_check.to_string(index=False))
print("\nInterpretation:")
print(f"- All quarters have min={min_obs_months} months (filter working)")
print("- Average observability increases over time (patients more established)")
print("="*60)

# ========================================
# CELL 16
# ========================================

# CELL 17
# CHECK 7: Screening exclusion impact
exclusion_impact = spark.sql(f"""
SELECT 
  'Before exclusions' as stage,
  COUNT(*) as observations,
  COUNT(DISTINCT PAT_ID) as patients,
  AVG(FUTURE_CRC_EVENT) * 100 as crc_rate
FROM {trgt_cat}.clncl_ds.herald_test_train_cohort
WHERE LABEL_USABLE = 1

UNION ALL

SELECT 
  'After exclusions' as stage,
  COUNT(*) as observations,
  COUNT(DISTINCT PAT_ID) as patients,
  AVG(FUTURE_CRC_EVENT) * 100 as crc_rate
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
WHERE LABEL_USABLE = 1
""").toPandas()

print("\n" + "="*60)
print("IMPACT OF SCREENING EXCLUSIONS")
print("="*60)
print(exclusion_impact.to_string(index=False))
print("="*60)

# ========================================
# CELL 17
# ========================================

# CELL 22
# CHECK 12: Executive summary with all key metrics
print("\n" + "="*70)
print("FINAL VALIDATION SUMMARY")
print("="*70)

# Get all key metrics
validation_summary = spark.sql(f"""
SELECT 
  COUNT(*) as total_obs,
  COUNT(DISTINCT PAT_ID) as unique_patients,
  SUM(FUTURE_CRC_EVENT) as positive_cases,
  SUM(LABEL_USABLE) as usable_obs,
  MIN(END_DTTM) as earliest_date,
  MAX(END_DTTM) as latest_date,
  AVG(AGE) as avg_age,
  AVG(IS_FEMALE) * 100 as pct_female,
  AVG(HAS_PCP_AT_END) * 100 as pct_with_pcp
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
""").collect()[0]

print(f"""
COHORT METRICS:
  Total observations: {validation_summary['total_obs']:,}
  Unique patients: {validation_summary['unique_patients']:,}
  Date range: {validation_summary['earliest_date']} to {validation_summary['latest_date']}
  
OUTCOMES:
  Positive cases: {validation_summary['positive_cases']:,}
  Event rate: {(validation_summary['positive_cases']/validation_summary['total_obs']*100):.4f}%
  
DEMOGRAPHICS:
  Average age: {validation_summary['avg_age']:.1f} years
  Female: {validation_summary['pct_female']:.1f}%
  Has PCP: {validation_summary['pct_with_pcp']:.1f}%

All observations in this table are training-ready (LABEL_USABLE = 1)
""")
print("="*70)

# ========================================
# CELL 18
# ========================================

# CELL 23
# COMPREHENSIVE VALIDATION CHECK
print("="*60)
print("COHORT VALIDATION RESULTS")
print("="*60)

# CHECK 1: Table row counts
print("\nTable Row Counts:")
for table in ['herald_test_train_cohort_index', 'herald_base_with_pcp', 
              'herald_test_train_cohort', 'herald_test_train_final_cohort']:
    count = spark.sql(f"SELECT COUNT(*) as n FROM {trgt_cat}.clncl_ds.{table}").collect()[0]['n']
    print(f"  {table}: {count:,} rows")

# CHECK 2: Verify no duplicates
dupe_check = spark.sql(f"""
SELECT 
  COUNT(*) as total_rows,
  COUNT(DISTINCT PAT_ID, END_DTTM) as unique_keys,
  COUNT(*) - COUNT(DISTINCT PAT_ID, END_DTTM) as duplicates
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
""").collect()[0]

print(f"\nDuplicate check: {dupe_check['duplicates']} duplicates found")
if dupe_check['duplicates'] == 0:
    print("  ✓ PASS: No duplicates")
else:
    print("  ✗ FAIL: Duplicates detected!")

# CHECK 3: Verify medical exclusions worked
exclusion_check = spark.sql(f"""
WITH potential_exclusions AS (
  SELECT DISTINCT 
    c.PAT_ID, 
    c.END_DTTM,
    dd.ICD10_CODE
  FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID 
    AND DATE(pe.CONTACT_DATE) <= c.END_DTTM
  JOIN clarity_cur.pat_enc_dx_enh dd
    ON dd.PAT_ENC_CSN_ID = pe.PAT_ENC_CSN_ID
  WHERE dd.ICD10_CODE RLIKE '{crc_icd_regex}'
     OR dd.ICD10_CODE IN ('Z90.49', 'K91.850')
     OR dd.ICD10_CODE LIKE 'Z51.5%'
)
SELECT 
  SUM(CASE WHEN ICD10_CODE RLIKE '{crc_icd_regex}' THEN 1 ELSE 0 END) as prior_crc,
  SUM(CASE WHEN ICD10_CODE IN ('Z90.49', 'K91.850') THEN 1 ELSE 0 END) as colectomy,
  SUM(CASE WHEN ICD10_CODE LIKE 'Z51.5%' THEN 1 ELSE 0 END) as hospice
FROM potential_exclusions
""").collect()[0]

print(f"\nExclusion verification (should all be 0):")
print(f"  Prior CRC: {exclusion_check['prior_crc']}")
print(f"  Colectomy: {exclusion_check['colectomy']}")
print(f"  Hospice: {exclusion_check['hospice']}")
if all(v == 0 for v in [exclusion_check['prior_crc'], exclusion_check['colectomy'], exclusion_check['hospice']]):
    print("  ✓ PASS: All exclusions properly applied")
else:
    print("  ✗ FAIL: Found patients who should be excluded!")

# CHECK 4: Age distribution
age_check = spark.sql(f"""
SELECT 
  MIN(AGE) as min_age,
  PERCENTILE(AGE, 0.5) as median_age,
  MAX(AGE) as max_age,
  SUM(CASE WHEN AGE < 45 THEN 1 ELSE 0 END) as under_45,
  SUM(CASE WHEN AGE > 100 THEN 1 ELSE 0 END) as over_100
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
""").collect()[0]

print(f"\nAge distribution:")
print(f"  Range: {age_check['min_age']} - {age_check['max_age']}")
print(f"  Median: {age_check['median_age']}")
print(f"  Under 45: {age_check['under_45']}")
print(f"  Over 100: {age_check['over_100']}")
if age_check['under_45'] == 0 and age_check['over_100'] == 0:
    print("  ✓ PASS: All ages within expected range")
else:
    print("  ⚠ WARNING: Check age outliers")

# CHECK 5: Label usability by PCP status
pcp_impact = spark.sql(f"""
SELECT 
  HAS_PCP_AT_END,
  COUNT(*) as obs,
  AVG(LABEL_USABLE) * 100 as usable_pct,
  AVG(FUTURE_CRC_EVENT) * 100 as crc_rate
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
GROUP BY HAS_PCP_AT_END
""").toPandas()

print("\nPCP Impact on Label Usability:")
print(pcp_impact.to_string(index=False))

# CHECK 6: Verify column structure
from pyspark.sql.types import *
schema = spark.table(f"{trgt_cat}.clncl_ds.herald_test_train_final_cohort").schema
column_names = [f.name for f in schema.fields]

# Check we HAVE the ICD columns for analysis
has_icd = 'ICD10_CODE' in column_names and 'ICD10_GROUP' in column_names
print(f"\nICD columns present for analysis: {has_icd}")
if has_icd:
    print("  ✓ PASS: ICD columns available for subtype analysis")

# Check we DON'T have future data columns
future_cols = [c for c in column_names if c in ['observable_days', 'next_contact_date']]
print(f"Future data columns that should be excluded: {future_cols if future_cols else 'None ✓'}")
if not future_cols:
    print("  ✓ PASS: No data leakage columns present")

# CHECK 7: CRC Subtype Distribution
crc_distribution = spark.sql(f"""
SELECT 
  ICD10_GROUP,
  COUNT(*) as cases,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
WHERE FUTURE_CRC_EVENT = 1
  AND ICD10_GROUP IS NOT NULL
GROUP BY ICD10_GROUP
ORDER BY cases DESC
""").toPandas()

print("\nCRC Subtype Distribution:")
print(crc_distribution.to_string(index=False))
print("\nExpected distribution:")
print("  C18 (Colon): ~65-75%")
print("  C20 (Rectum): ~15-20%")
print("  C21 (Anus): ~5-10%")
print("  C19 (Rectosigmoid): ~3-5%")

print("\n" + "="*60)
print("VALIDATION COMPLETE")
print("="*60)

# ========================================
# CELL 19
# ========================================

# CELL 24
# Calculate actual quarterly event rates for documentation
quarterly_rates = spark.sql(f"""
SELECT 
  DATE_FORMAT(END_DTTM, 'yyyy-Q') as quarter,
  COUNT(*) as observations,
  SUM(FUTURE_CRC_EVENT) as events,
  ROUND(AVG(FUTURE_CRC_EVENT) * 100, 4) as rate_pct
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
GROUP BY DATE_FORMAT(END_DTTM, 'yyyy-Q')
ORDER BY quarter
""").toPandas()

print("="*70)
print("QUARTERLY EVENT RATES FOR DOCUMENTATION")
print("="*70)
print("\nCopy these values to update the markdown section on declining rates:\n")

for idx, row in quarterly_rates.iterrows():
    print(f"{row['quarter']}: {row['rate_pct']:.2f}% ← "
          f"({row['events']:,} events from {row['observations']:,} observations)")

if len(quarterly_rates) > 0:
    first_rate = quarterly_rates.iloc[0]['rate_pct']
    last_rate = quarterly_rates.iloc[-1]['rate_pct']
    decline_pct = ((first_rate - last_rate) / first_rate) * 100
    ratio = first_rate / last_rate if last_rate > 0 else 0
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"First quarter rate: {first_rate:.4f}%")
    print(f"Last quarter rate: {last_rate:.4f}%")
    print(f"Total decline: {decline_pct:.1f}%")
    print(f"Ratio: {ratio:.2f}x higher in first quarter")
    print(f"Expected incident rate: ~0.025% per 6 months")
    print(f"First quarter is {first_rate/0.025:.1f}x higher than expected incident")
    print("="*70)

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(quarterly_rates['quarter'], quarterly_rates['rate_pct'], 
         marker='o', linewidth=2, markersize=8, color='#2E86AB')
plt.axhline(y=0.025, color='#A23B72', linestyle='--', linewidth=2, 
            label='Expected incident rate (~0.025%)')
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Event Rate (%)', fontsize=12)
plt.title('CRC Event Rate by Quarter: Evidence of Prevalent Case Contamination', 
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("\nUse the values above to update this section in the markdown:")
print("'### The Declining Rate Pattern'")
print("Replace the example quarterly rates with your actual rates.")

# ========================================
# CELL 20
# ========================================

# CELL 25
df = spark.sql(f"SELECT * FROM dev.clncl_ds.herald_test_train_final_cohort")
# exact row count (triggers a full scan)
n_rows = df.count()
print(n_rows)

# ========================================
# CELL 21
# ========================================

# CELL 26
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

df = spark.table("dev.clncl_ds.herald_test_train_final_cohort")

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
# CELL 22
# ========================================

# CELL 27
# SPLIT-1: Load cohort and compute patient-level stratification labels

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the final cohort
df_cohort = spark.table(f"{trgt_cat}.clncl_ds.herald_test_train_final_cohort")

print(f"Total observations: {df_cohort.count():,}")
print(f"Unique patients: {df_cohort.select('PAT_ID').distinct().count():,}")

# Calculate quarters_since_study_start (for later per-quarter analysis)
study_start = "2023-01-01"

df_with_quarter = df_cohort.withColumn(
    "quarters_since_study_start",
    F.floor(F.months_between(F.col("END_DTTM"), F.lit(study_start)) / 3).cast(IntegerType())
)

# Show quarter distribution (informational - not used for splitting)
print("\nQuarter distribution (informational):")
df_with_quarter.groupBy("quarters_since_study_start").agg(
    F.count("*").alias("n_observations"),
    F.countDistinct("PAT_ID").alias("n_patients"),
    F.mean("FUTURE_CRC_EVENT").alias("event_rate")
).orderBy("quarters_since_study_start").show()

# Get patient-level labels with cancer type
print("\nComputing patient-level stratification labels...")
patient_labels = df_with_quarter.groupBy("PAT_ID").agg(
    F.max("FUTURE_CRC_EVENT").alias("is_positive"),
    # Get the cancer type for positive patients (first non-null ICD10_GROUP where event=1)
    F.first(
        F.when(F.col("FUTURE_CRC_EVENT") == 1, F.col("ICD10_GROUP"))
    ).alias("cancer_type")
).toPandas()

# Create multi-class stratification label
# Map: 0=negative, 1=C18, 2=C19, 3=C20, (4=C21 if include_anus=True)
cancer_type_map = {'C18': 1, 'C19': 2, 'C20': 3}
if include_anus:
    cancer_type_map['C21'] = 4

def get_strat_label(row):
    if row['is_positive'] == 0:
        return 0  # Negative
    else:
        return cancer_type_map.get(row['cancer_type'], 0)  # Map cancer type or 0 if unknown

patient_labels['strat_label'] = patient_labels.apply(get_strat_label, axis=1)

# Summary statistics
print(f"\nUnique patients: {len(patient_labels):,}")
print(f"Negative patients (class 0): {(patient_labels['strat_label'] == 0).sum():,}")
print(f"C18 patients (class 1): {(patient_labels['strat_label'] == 1).sum():,}")
print(f"C19 patients (class 2): {(patient_labels['strat_label'] == 2).sum():,}")
print(f"C20 patients (class 3): {(patient_labels['strat_label'] == 3).sum():,}")
if include_anus:
    print(f"C21 patients (class 4): {(patient_labels['strat_label'] == 4).sum():,}")

# Show cancer type distribution among positive patients
positive_patients = patient_labels[patient_labels['is_positive'] == 1]
print(f"\nCancer type distribution (positive patients only):")
print(positive_patients['cancer_type'].value_counts(normalize=True).round(4) * 100)

# ========================================
# CELL 23
# ========================================

# CELL 28
# SPLIT-2: Stratified patient-level split (70/15/15)

np.random.seed(217)  # For reproducibility

# =============================================================================
# STRATIFIED PATIENT-LEVEL SPLIT
# =============================================================================
# Split all patients into TRAIN (70%), VAL (15%), TEST (15%)
# Stratified by cancer type to preserve distribution across splits
#
# Two-step process:
# 1. Split off TEST (15%) from the rest
# 2. Split remaining into TRAIN (70/85 ≈ 82%) and VAL (15/85 ≈ 18%)
# =============================================================================

print("Performing stratified patient-level split (70/15/15)...")
print(f"Total patients: {len(patient_labels):,}")

# Step 1: Split off TEST (15%)
patients_trainval, patients_test = train_test_split(
    patient_labels,
    test_size=0.15,
    stratify=patient_labels['strat_label'],
    random_state=217
)

print(f"\nAfter TEST split:")
print(f"  TRAIN+VAL: {len(patients_trainval):,} patients ({len(patients_trainval)/len(patient_labels)*100:.1f}%)")
print(f"  TEST: {len(patients_test):,} patients ({len(patients_test)/len(patient_labels)*100:.1f}%)")

# Step 2: Split TRAIN+VAL into TRAIN (70%) and VAL (15%)
# 15% of original = 15/85 ≈ 17.6% of remaining
patients_train, patients_val = train_test_split(
    patients_trainval,
    test_size=0.176,  # 15/85 ≈ 0.176
    stratify=patients_trainval['strat_label'],
    random_state=217
)

print(f"\nAfter TRAIN/VAL split:")
print(f"  TRAIN: {len(patients_train):,} patients ({len(patients_train)/len(patient_labels)*100:.1f}%)")
print(f"  VAL: {len(patients_val):,} patients ({len(patients_val)/len(patient_labels)*100:.1f}%)")
print(f"  TEST: {len(patients_test):,} patients ({len(patients_test)/len(patient_labels)*100:.1f}%)")

# Create patient sets for easy lookup
train_patients = set(patients_train['PAT_ID'].values)
val_patients = set(patients_val['PAT_ID'].values)
test_patients = set(patients_test['PAT_ID'].values)

# Verify no overlap
assert len(train_patients.intersection(val_patients)) == 0, "TRAIN/VAL overlap!"
assert len(train_patients.intersection(test_patients)) == 0, "TRAIN/TEST overlap!"
assert len(val_patients.intersection(test_patients)) == 0, "VAL/TEST overlap!"
print("\n✓ No patient overlap between splits")

# =============================================================================
# VERIFY STRATIFICATION PRESERVED
# =============================================================================
print("\n" + "="*70)
print("STRATIFICATION VERIFICATION")
print("="*70)

# Check positive rates
train_pos_rate = patients_train['is_positive'].mean()
val_pos_rate = patients_val['is_positive'].mean()
test_pos_rate = patients_test['is_positive'].mean()
overall_pos_rate = patient_labels['is_positive'].mean()

print(f"\nPositive patient rates:")
print(f"  Overall: {overall_pos_rate:.4%}")
print(f"  TRAIN:   {train_pos_rate:.4%}")
print(f"  VAL:     {val_pos_rate:.4%}")
print(f"  TEST:    {test_pos_rate:.4%}")

# Check cancer type distribution
cancer_types_to_check = ['C18', 'C19', 'C20'] + (['C21'] if include_anus else [])

print("\nCancer type distribution (% of positive patients):")
print("-" * 60)
print(f"{'Cancer Type':<15} {'Overall':>10} {'TRAIN':>10} {'VAL':>10} {'TEST':>10}")
print("-" * 60)

for cancer_type in cancer_types_to_check:
    overall_pct = (patient_labels[patient_labels['is_positive']==1]['cancer_type'] == cancer_type).mean() * 100
    train_pct = (patients_train[patients_train['is_positive']==1]['cancer_type'] == cancer_type).mean() * 100
    val_pct = (patients_val[patients_val['is_positive']==1]['cancer_type'] == cancer_type).mean() * 100
    test_pct = (patients_test[patients_test['is_positive']==1]['cancer_type'] == cancer_type).mean() * 100
    print(f"{cancer_type:<15} {overall_pct:>9.1f}% {train_pct:>9.1f}% {val_pct:>9.1f}% {test_pct:>9.1f}%")

print("-" * 60)

# Show absolute counts
print("\nAbsolute counts by cancer type:")
print("-" * 60)
print(f"{'Cancer Type':<15} {'Overall':>10} {'TRAIN':>10} {'VAL':>10} {'TEST':>10}")
print("-" * 60)

for cancer_type in cancer_types_to_check:
    overall_n = (patient_labels[patient_labels['is_positive']==1]['cancer_type'] == cancer_type).sum()
    train_n = (patients_train[patients_train['is_positive']==1]['cancer_type'] == cancer_type).sum()
    val_n = (patients_val[patients_val['is_positive']==1]['cancer_type'] == cancer_type).sum()
    test_n = (patients_test[patients_test['is_positive']==1]['cancer_type'] == cancer_type).sum()
    print(f"{cancer_type:<15} {overall_n:>10,} {train_n:>10,} {val_n:>10,} {test_n:>10,}")

print("-" * 60)
total_pos = patient_labels['is_positive'].sum()
train_pos = patients_train['is_positive'].sum()
val_pos = patients_val['is_positive'].sum()
test_pos = patients_test['is_positive'].sum()
print(f"{'TOTAL':<15} {total_pos:>10,} {train_pos:>10,} {val_pos:>10,} {test_pos:>10,}")

# ========================================
# CELL 24
# ========================================

# CELL 29
# SPLIT-3: Create SPLIT column and map to observations

# Create patient -> split mapping
train_patients_list = list(train_patients)
val_patients_list = list(val_patients)
test_patients_list = list(test_patients)

# Create mapping DataFrame
train_pdf = pd.DataFrame({'PAT_ID': train_patients_list, 'SPLIT': 'train'})
val_pdf = pd.DataFrame({'PAT_ID': val_patients_list, 'SPLIT': 'val'})
test_pdf = pd.DataFrame({'PAT_ID': test_patients_list, 'SPLIT': 'test'})
split_mapping_pdf = pd.concat([train_pdf, val_pdf, test_pdf], ignore_index=True)

print(f"Split mapping created: {len(split_mapping_pdf):,} patients")

# Convert to Spark DataFrame
split_mapping_sdf = spark.createDataFrame(split_mapping_pdf)

# Join observations with split mapping
df_final = df_with_quarter.join(
    split_mapping_sdf,
    on="PAT_ID",
    how="left"
)

# Verify no nulls in SPLIT column
null_count = df_final.filter(F.col("SPLIT").isNull()).count()
if null_count > 0:
    print(f"WARNING: {null_count} observations have NULL SPLIT!")
else:
    print("✓ All observations have SPLIT assigned")

# Verify split distribution
print("\nSplit distribution (observations):")
df_final.groupBy("SPLIT").agg(
    F.count("*").alias("n_observations"),
    F.countDistinct("PAT_ID").alias("n_patients"),
    F.mean("FUTURE_CRC_EVENT").alias("event_rate")
).orderBy("SPLIT").show()

# ========================================
# CELL 25
# ========================================

# CELL 30
# SPLIT-3b: Verify no patient overlap across splits

print("Patient overlap verification (from final DataFrame):")
train_pats_final = set(df_final.filter(F.col("SPLIT") == "train").select("PAT_ID").distinct().toPandas()["PAT_ID"])
val_pats_final = set(df_final.filter(F.col("SPLIT") == "val").select("PAT_ID").distinct().toPandas()["PAT_ID"])
test_pats_final = set(df_final.filter(F.col("SPLIT") == "test").select("PAT_ID").distinct().toPandas()["PAT_ID"])

print(f"TRAIN patients: {len(train_pats_final):,}")
print(f"VAL patients: {len(val_pats_final):,}")
print(f"TEST patients: {len(test_pats_final):,}")

print(f"\nTRAIN ∩ VAL: {len(train_pats_final.intersection(val_pats_final))} patients (should be 0)")
print(f"TRAIN ∩ TEST: {len(train_pats_final.intersection(test_pats_final))} patients (should be 0)")
print(f"VAL ∩ TEST: {len(val_pats_final.intersection(test_pats_final))} patients (should be 0)")

# Assert no overlap
assert len(train_pats_final.intersection(val_pats_final)) == 0, "TRAIN/VAL overlap!"
assert len(train_pats_final.intersection(test_pats_final)) == 0, "TRAIN/TEST overlap!"
assert len(val_pats_final.intersection(test_pats_final)) == 0, "VAL/TEST overlap!"

print("\n✓ No patient overlap across splits - data leakage prevention verified")

# =============================================================================
# VERIFY CANCER TYPE DISTRIBUTION ACROSS ALL THREE SPLITS
# =============================================================================
print("\n" + "="*70)
print("FINAL CANCER TYPE DISTRIBUTION ACROSS ALL SPLITS")
print("="*70)

# Get positive observations by split with cancer type
cancer_dist_df = df_final.filter(F.col("FUTURE_CRC_EVENT") == 1).groupBy("SPLIT", "ICD10_GROUP").agg(
    F.countDistinct("PAT_ID").alias("patient_count")
).toPandas()

# Get total positive patients by split
totals_by_split = cancer_dist_df.groupby("SPLIT")["patient_count"].sum()

# Include C21 if include_anus=True
cancer_types_final = ['C18', 'C19', 'C20'] + (['C21'] if include_anus else [])

# Display distribution
print("\nCancer type distribution by split (% of positive patients in each split):")
print("-" * 65)
print(f"{'Cancer Type':<12} {'TRAIN':>12} {'VAL':>12} {'TEST':>12} {'OVERALL':>12}")
print("-" * 65)

for cancer_type in cancer_types_final:
    row_data = []
    for split in ['train', 'val', 'test']:
        mask = (cancer_dist_df['SPLIT'] == split) & (cancer_dist_df['ICD10_GROUP'] == cancer_type)
        count = cancer_dist_df.loc[mask, 'patient_count'].sum() if mask.any() else 0
        total = totals_by_split.get(split, 1)
        pct = (count / total * 100) if total > 0 else 0
        row_data.append(pct)

    # Calculate overall
    overall_count = cancer_dist_df[cancer_dist_df['ICD10_GROUP'] == cancer_type]['patient_count'].sum()
    overall_total = cancer_dist_df['patient_count'].sum()
    overall_pct = (overall_count / overall_total * 100) if overall_total > 0 else 0

    print(f"{cancer_type:<12} {row_data[0]:>11.2f}% {row_data[1]:>11.2f}% {row_data[2]:>11.2f}% {overall_pct:>11.2f}%")

print("-" * 65)

# Show absolute counts
print("\nAbsolute patient counts by cancer type and split:")
print("-" * 65)
print(f"{'Cancer Type':<12} {'TRAIN':>12} {'VAL':>12} {'TEST':>12} {'OVERALL':>12}")
print("-" * 65)

for cancer_type in cancer_types_final:
    row_data = []
    for split in ['train', 'val', 'test']:
        mask = (cancer_dist_df['SPLIT'] == split) & (cancer_dist_df['ICD10_GROUP'] == cancer_type)
        count = int(cancer_dist_df.loc[mask, 'patient_count'].sum()) if mask.any() else 0
        row_data.append(count)
    overall_count = int(cancer_dist_df[cancer_dist_df['ICD10_GROUP'] == cancer_type]['patient_count'].sum())
    print(f"{cancer_type:<12} {row_data[0]:>12,} {row_data[1]:>12,} {row_data[2]:>12,} {overall_count:>12,}")

print("-" * 65)
total_by_split = [int(totals_by_split.get(s, 0)) for s in ['train', 'val', 'test']]
print(f"{'TOTAL':<12} {total_by_split[0]:>12,} {total_by_split[1]:>12,} {total_by_split[2]:>12,} {int(cancer_dist_df['patient_count'].sum()):>12,}")

print("\n✓ Cancer type distribution verified across all splits")

# ========================================
# CELL 26
# ========================================

# CELL 31
# SPLIT-4: Save final cohort with SPLIT column

# Drop temporary quarter column before saving
df_to_save = df_final.drop("quarters_since_study_start")

# Materialize the DataFrame to break lazy evaluation chain
# (df_final references the target table in its lineage)
df_to_save = df_to_save.cache()
row_count = df_to_save.count()  # Trigger materialization
print(f"Materialized {row_count:,} rows for saving")

# Save to table (overwrite)
df_to_save.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{trgt_cat}.clncl_ds.herald_test_train_final_cohort")

print(f"✓ Saved cohort with SPLIT column to {trgt_cat}.clncl_ds.herald_test_train_final_cohort")

# Verify save
df_verify = spark.table(f"{trgt_cat}.clncl_ds.herald_test_train_final_cohort")
print(f"\nVerification:")
print(f"  Total rows: {df_verify.count():,}")
print(f"  Columns: {df_verify.columns}")
print(f"\nSplit distribution after save:")
df_verify.groupBy("SPLIT").count().show()

# Unpersist cached DataFrame
df_to_save.unpersist()



################################################################################
# V2_Book2_ICD10
################################################################################

# V2_Book2_ICD10
# Functional cells: 29 of 64 code cells (122 total)
# Source: V2_Book2_ICD10.ipynb
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

import pandas as pd

pd.set_option('display.max_rows', None)        # Show all rows
pd.set_option('display.max_columns', None)     # Show all columns
pd.set_option('display.max_colwidth', None)    # Show full text in each cell
pd.set_option('display.width', None)           # No line-wrapping

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

# =========================================================================
# CELL 1 - EXTRACT ALL RELEVANT ICD-10 CODES FROM MULTIPLE SOURCES
# =========================================================================
# This cell creates a curated table of all diagnosis codes relevant to CRC risk
# We pull from three sources: outpatient encounters, inpatient accounts, and problem lists

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_curated_conditions AS

-- SOURCE 1: Outpatient encounter diagnoses
SELECT
  pe.PAT_ID,
  dd.ICD10_CODE              AS CODE,
  DATE(pe.CONTACT_DATE)      AS CONTACT_DATE,
  pe.PAT_ENC_CSN_ID          AS ENC_ID
FROM clarity_cur.pat_enc_dx_enh dd
JOIN clarity_cur.pat_enc_enh pe
  ON pe.PAT_ENC_CSN_ID = dd.PAT_ENC_CSN_ID
WHERE
  -- Filter for ICD-10 codes relevant to CRC risk
  dd.ICD10_CODE RLIKE '^(D12\\..*|K50\\..*|K51\\..*|D5[0-3]\\..*|D6[234]\\..*|K62\\.5.*|K92\\.[12].*|K59\\.0.*|R10\\..*|R19\\.4.*|R19\\.7.*|R63\\.4.*|R53\\.(1|83).*|Z85\\..*|K63\\.5.*|K62\\.0.*|K62\\.1.*|Z86\\.01.*|E1[01]\\..*|E66\\..*|Z80\\.0.*|Z80\\.2.*|Z80\\.9.*|Z85\\.038.*|Z85\\.048.*|Z83\\.71.*|Z12\\.1[12].*|R19\\.5.*|R19\\.0.*|D37\\.[34].*|K57\\..*)$'
  AND DATE(pe.CONTACT_DATE) >= DATE '2021-07-01'

UNION ALL

-- SOURCE 2: Inpatient hospital account diagnoses
SELECT
  ha.PAT_ID,
  dx.CODE                    AS CODE,
  DATE(ha.DISCH_DATE_TIME)   AS CONTACT_DATE,
  ha.PAT_ENC_CSN_ID          AS ENC_ID
FROM clarity_cur.hsp_acct_dx_list_enh dx
JOIN clarity_cur.pat_enc_hsp_har_enh ha
  ON ha.HSP_ACCOUNT_ID = dx.HSP_ACCOUNT_ID
WHERE
  dx.CODE RLIKE '^(D12\\..*|K50\\..*|K51\\..*|D5[0-3]\\..*|D6[234]\\..*|K62\\.5.*|K92\\.[12].*|K59\\.0.*|R10\\..*|R19\\.4.*|R19\\.7.*|R63\\.4.*|R53\\.(1|83).*|Z85\\..*|K63\\.5.*|K62\\.0.*|K62\\.1.*|Z86\\.01.*|E1[01]\\..*|E66\\..*|Z80\\.0.*|Z80\\.2.*|Z80\\.9.*|Z85\\.038.*|Z85\\.048.*|Z83\\.71.*|Z12\\.1[12].*|R19\\.5.*|R19\\.0.*|D37\\.[34].*|K57\\..*)$'
  AND DATE(ha.DISCH_DATE_TIME) >= DATE '2021-07-01'

UNION ALL

-- SOURCE 3: Problem list history (captures chronic conditions)
SELECT
  phx.PAT_ID,
  phx.HX_PROBLEM_ICD10_CODE  AS CODE,
  DATE(phx.HX_DATE_OF_ENTRY) AS CONTACT_DATE,
  NULL                       AS ENC_ID
FROM clarity_cur.problem_list_hx_enh phx
WHERE
  phx.HX_PROBLEM_ICD10_CODE RLIKE '^(D12\\..*|K50\\..*|K51\\..*|D5[0-3]\\..*|D6[234]\\..*|K62\\.5.*|K92\\.[12].*|K59\\.0.*|R10\\..*|R19\\.4.*|R19\\.7.*|R63\\.4.*|R53\\.(1|83).*|Z85\\..*|K63\\.5.*|K62\\.0.*|K62\\.1.*|Z86\\.01.*|E1[01]\\..*|E66\\..*|Z80\\.0.*|Z80\\.2.*|Z80\\.9.*|Z85\\.038.*|Z85\\.048.*|Z83\\.71.*|Z12\\.1[12].*|R19\\.5.*|R19\\.0.*|D37\\.[34].*|K57\\..*)$'
  AND phx.HX_STATUS = 'Active'
  AND DATE(phx.HX_DATE_OF_ENTRY) >= DATE '2021-07-01'
""")

print("✓ Curated conditions table created")

# ========================================
# CELL 3
# ========================================

# =========================================================================
# CELL 2 - CREATE TIME-WINDOWED VIEWS OF CONDITIONS
# =========================================================================
# This cell takes our curated conditions and creates three temporal views:
# - 12mo: Conditions in the 12 months before each snapshot
# - 24mo: Conditions in the 24 months before each snapshot  
# - ever: Any conditions before the snapshot (lifetime history)

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_conditions_all AS

-- 12-MONTH WINDOW
-- Recent conditions that may indicate active disease or symptoms
SELECT
  c.PAT_ID,
  c.END_DTTM,
  cc.CODE,
  cc.ENC_ID,
  DATE(cc.CONTACT_DATE) AS CONTACT_DATE,
  '12mo' AS WINDOW
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
JOIN {trgt_cat}.clncl_ds.herald_test_train_curated_conditions cc
  ON cc.PAT_ID = c.PAT_ID
  -- Include conditions from 365 days before up to (and including) the snapshot date
 AND DATE(cc.CONTACT_DATE) BETWEEN DATE_SUB(c.END_DTTM, 365) AND c.END_DTTM

UNION ALL

-- 24-MONTH WINDOW  
-- Captures conditions that may have longer-term implications
SELECT
  c.PAT_ID,
  c.END_DTTM,
  cc.CODE,
  cc.ENC_ID,
  DATE(cc.CONTACT_DATE) AS CONTACT_DATE,
  '24mo' AS WINDOW
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
JOIN {trgt_cat}.clncl_ds.herald_test_train_curated_conditions cc
  ON cc.PAT_ID = c.PAT_ID
  -- Include conditions from 730 days before up to (and including) the snapshot date
 AND DATE(cc.CONTACT_DATE) BETWEEN DATE_SUB(c.END_DTTM, 730) AND c.END_DTTM

UNION ALL

-- EVER WINDOW
-- Captures lifetime history (especially important for risk factors like polyps, IBD, family history)
-- FIXED: Changed < to <= for consistency with other windows
SELECT
  c.PAT_ID,
  c.END_DTTM,
  cc.CODE,
  cc.ENC_ID,
  DATE(cc.CONTACT_DATE) AS CONTACT_DATE,
  'ever' AS WINDOW
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
JOIN {trgt_cat}.clncl_ds.herald_test_train_curated_conditions cc
  ON cc.PAT_ID = c.PAT_ID
  -- Any condition on or before the snapshot date (inclusive for consistency)
 AND DATE(cc.CONTACT_DATE) <= c.END_DTTM
""")

print("✓ Time-windowed conditions table created")

# ========================================
# CELL 4
# ========================================

# =========================================================================
# CELL 3 - EXTRACT SYMPTOM FEATURES (BATCH 1)
# =========================================================================
# Purpose: Process acute symptoms that may indicate CRC
# Creates both FLAGS (binary: has condition) and COUNTS (frequency)
# Plus RECENCY features (days since last occurrence)

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_icd10_symptoms AS

WITH 
-- Base cohort from our main training set
cohort AS (
    SELECT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
),

-- ================================================================
-- STEP 1: Aggregate symptoms by patient, snapshot, and time window
-- ================================================================
symptom_counts AS (
    SELECT 
        c.PAT_ID,
        c.END_DTTM,
        
        -- 12-MONTH COUNTS
        -- Count occurrences of each symptom type in past 12 months
        SUM(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^(K62\\.5|K92\\.[12])' 
                 THEN 1 ELSE 0 END) AS BLEED_CNT_12MO,
        SUM(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^R10\\.' 
                 THEN 1 ELSE 0 END) AS PAIN_CNT_12MO,
        SUM(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^(R19\\.4|K59\\.0|R19\\.7)' 
                 THEN 1 ELSE 0 END) AS BOWELCHG_CNT_12MO,
        SUM(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^R63\\.4' 
                 THEN 1 ELSE 0 END) AS WTLOSS_CNT_12MO,
        SUM(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^R53\\.(1|83)' 
                 THEN 1 ELSE 0 END) AS FATIGUE_CNT_12MO,
        SUM(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^(D5[0-3]|D6[234])' 
                 THEN 1 ELSE 0 END) AS ANEMIA_CNT_12MO,
        
        -- 24-MONTH COUNTS
        -- Count occurrences of each symptom type in past 24 months
        SUM(CASE WHEN ca.WINDOW = '24mo' AND ca.CODE RLIKE '^(K62\\.5|K92\\.[12])' 
                 THEN 1 ELSE 0 END) AS BLEED_CNT_24MO,
        SUM(CASE WHEN ca.WINDOW = '24mo' AND ca.CODE RLIKE '^R10\\.' 
                 THEN 1 ELSE 0 END) AS PAIN_CNT_24MO,
        SUM(CASE WHEN ca.WINDOW = '24mo' AND ca.CODE RLIKE '^(R19\\.4|K59\\.0|R19\\.7)' 
                 THEN 1 ELSE 0 END) AS BOWELCHG_CNT_24MO,
        SUM(CASE WHEN ca.WINDOW = '24mo' AND ca.CODE RLIKE '^R63\\.4' 
                 THEN 1 ELSE 0 END) AS WTLOSS_CNT_24MO,
        SUM(CASE WHEN ca.WINDOW = '24mo' AND ca.CODE RLIKE '^R53\\.(1|83)' 
                 THEN 1 ELSE 0 END) AS FATIGUE_CNT_24MO,
        SUM(CASE WHEN ca.WINDOW = '24mo' AND ca.CODE RLIKE '^(D5[0-3]|D6[234])' 
                 THEN 1 ELSE 0 END) AS ANEMIA_CNT_24MO
        
    FROM cohort c
    LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_conditions_all ca
        ON c.PAT_ID = ca.PAT_ID 
        AND c.END_DTTM = ca.END_DTTM
        AND ca.WINDOW IN ('12mo', '24mo')  -- Only recent symptoms
    GROUP BY c.PAT_ID, c.END_DTTM
),

-- ================================================================
-- STEP 2: Get most recent occurrence dates for recency features
-- ================================================================
-- Using FIRST_VALUE with window functions avoids MAX function
symptom_recency AS (
    SELECT DISTINCT
        c.PAT_ID,
        c.END_DTTM,
        
        -- Most recent bleeding event date
        FIRST_VALUE(
            CASE WHEN ca.CODE RLIKE '^(K62\\.5|K92\\.[12])' 
                 THEN ca.CONTACT_DATE END
        ) OVER (
            PARTITION BY c.PAT_ID, c.END_DTTM 
            ORDER BY 
                CASE WHEN ca.CODE RLIKE '^(K62\\.5|K92\\.[12])' 
                     THEN ca.CONTACT_DATE END DESC NULLS LAST
        ) AS BLEED_LATEST,
        
        -- Most recent anemia diagnosis date
        FIRST_VALUE(
            CASE WHEN ca.CODE RLIKE '^(D5[0-3]|D6[234])' 
                 THEN ca.CONTACT_DATE END
        ) OVER (
            PARTITION BY c.PAT_ID, c.END_DTTM 
            ORDER BY 
                CASE WHEN ca.CODE RLIKE '^(D5[0-3]|D6[234])' 
                     THEN ca.CONTACT_DATE END DESC NULLS LAST
        ) AS ANEMIA_LATEST,
        
        -- Most recent pain event date
        FIRST_VALUE(
            CASE WHEN ca.CODE RLIKE '^R10\\.' 
                 THEN ca.CONTACT_DATE END
        ) OVER (
            PARTITION BY c.PAT_ID, c.END_DTTM 
            ORDER BY 
                CASE WHEN ca.CODE RLIKE '^R10\\.' 
                     THEN ca.CONTACT_DATE END DESC NULLS LAST
        ) AS PAIN_LATEST
        
    FROM cohort c
    LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_conditions_all ca
        ON c.PAT_ID = ca.PAT_ID 
        AND c.END_DTTM = ca.END_DTTM
        -- Look at all historical data for recency
        AND ca.WINDOW IN ('12mo', '24mo', 'ever')
)

-- ================================================================
-- STEP 3: Combine counts and recency into final feature set
-- ================================================================
SELECT 
    sc.PAT_ID,
    sc.END_DTTM,
    
    -- 12-MONTH FLAGS (binary: has symptom or not)
    CASE WHEN sc.BLEED_CNT_12MO > 0 THEN 1 ELSE 0 END AS BLEED_FLAG_12MO,
    sc.BLEED_CNT_12MO,
    CASE WHEN sc.PAIN_CNT_12MO > 0 THEN 1 ELSE 0 END AS PAIN_FLAG_12MO,
    sc.PAIN_CNT_12MO,
    CASE WHEN sc.BOWELCHG_CNT_12MO > 0 THEN 1 ELSE 0 END AS BOWELCHG_FLAG_12MO,
    sc.BOWELCHG_CNT_12MO,
    CASE WHEN sc.WTLOSS_CNT_12MO > 0 THEN 1 ELSE 0 END AS WTLOSS_FLAG_12MO,
    sc.WTLOSS_CNT_12MO,
    CASE WHEN sc.FATIGUE_CNT_12MO > 0 THEN 1 ELSE 0 END AS FATIGUE_FLAG_12MO,
    sc.FATIGUE_CNT_12MO,
    CASE WHEN sc.ANEMIA_CNT_12MO > 0 THEN 1 ELSE 0 END AS ANEMIA_FLAG_12MO,
    sc.ANEMIA_CNT_12MO,
    
    -- 24-MONTH FLAGS (binary: has symptom or not)
    CASE WHEN sc.BLEED_CNT_24MO > 0 THEN 1 ELSE 0 END AS BLEED_FLAG_24MO,
    sc.BLEED_CNT_24MO,
    CASE WHEN sc.PAIN_CNT_24MO > 0 THEN 1 ELSE 0 END AS PAIN_FLAG_24MO,
    sc.PAIN_CNT_24MO,
    CASE WHEN sc.BOWELCHG_CNT_24MO > 0 THEN 1 ELSE 0 END AS BOWELCHG_FLAG_24MO,
    sc.BOWELCHG_CNT_24MO,
    CASE WHEN sc.WTLOSS_CNT_24MO > 0 THEN 1 ELSE 0 END AS WTLOSS_FLAG_24MO,
    sc.WTLOSS_CNT_24MO,
    CASE WHEN sc.FATIGUE_CNT_24MO > 0 THEN 1 ELSE 0 END AS FATIGUE_FLAG_24MO,
    sc.FATIGUE_CNT_24MO,
    CASE WHEN sc.ANEMIA_CNT_24MO > 0 THEN 1 ELSE 0 END AS ANEMIA_FLAG_24MO,
    sc.ANEMIA_CNT_24MO,
    
    -- RECENCY FEATURES (days since last occurrence)
    -- NULL if never had the symptom
    DATEDIFF(sc.END_DTTM, sr.BLEED_LATEST) AS DAYS_SINCE_LAST_BLEED,
    DATEDIFF(sc.END_DTTM, sr.ANEMIA_LATEST) AS DAYS_SINCE_LAST_ANEMIA,
    DATEDIFF(sc.END_DTTM, sr.PAIN_LATEST) AS DAYS_SINCE_LAST_PAIN
    
FROM symptom_counts sc
LEFT JOIN symptom_recency sr
    ON sc.PAT_ID = sr.PAT_ID 
    AND sc.END_DTTM = sr.END_DTTM
""")

print("✓ Symptom features created (batch 1)")

# ========================================
# CELL 5
# ========================================

# =========================================================================
# CELL 4 - EXTRACT RISK FACTOR FEATURES (BATCH 2)
# =========================================================================
# This cell processes chronic conditions and risk factors that increase CRC risk
# These use the 'ever' window since lifetime history matters

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_icd10_risk_factors AS

WITH cohort AS (
    SELECT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
),

risk_agg AS (
    SELECT 
        c.PAT_ID,
        c.END_DTTM,
        
        -- POLYPS (major risk factor - adenomatous polyps can become malignant)
        -- D12: Benign neoplasm of colon, rectum, anus
        -- K63.5: Polyp of colon
        SUM(CASE WHEN ca.CODE RLIKE '^(D12|K63\\.5)' THEN 1 ELSE 0 END) AS POLYPS_CNT_EVER,
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^(D12|K63\\.5)' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS POLYPS_FLAG_EVER,
        
        -- INFLAMMATORY BOWEL DISEASE (8-10x increased risk after 8+ years)
        -- K50: Crohn's disease
        -- K51: Ulcerative colitis
        SUM(CASE WHEN ca.CODE RLIKE '^(K50|K51)\\.' THEN 1 ELSE 0 END) AS IBD_CNT_EVER,
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^(K50|K51)\\.' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS IBD_FLAG_EVER,
        
        -- PRIOR MALIGNANCY (increased surveillance/risk)
        -- Z85: Personal history of malignant neoplasm
        SUM(CASE WHEN ca.CODE RLIKE '^Z85\\.' THEN 1 ELSE 0 END) AS MALIGNANCY_CNT_EVER,
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^Z85\\.' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS MALIGNANCY_FLAG_EVER,
        
        -- FAMILY HISTORY FLAGS (genetic risk - no counts needed)
        -- Z80.0: Family history of malignant neoplasm of digestive organs
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^Z80\\.0' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS FHX_CRC_FLAG_EVER,
        
        -- Z80.2: Family history of malignant neoplasm of blood/lymphatic
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^Z80\\.2' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS FHX_ANAL_FLAG_EVER,
        
        -- Z80.9: Family history of malignant neoplasm, unspecified
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^Z80\\.9' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS FHX_GI_MALIG_FLAG_EVER,
        
        -- PERSONAL HISTORY OF ANAL/RECTAL MALIGNANCY
        -- Z85.048: Personal history of malignant neoplasm of anus
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^Z85\\.048' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS PHX_ANUS_MALIG_FLAG_EVER,
        
        -- Z85.038: Personal history of malignant neoplasm of rectum
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^Z85\\.038' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS PHX_RECTUM_MALIG_FLAG_EVER,
        
        -- METABOLIC CONDITIONS (associated with increased CRC risk)
        -- E10/E11: Diabetes (Type 1 and Type 2)
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^E1[01]\\.' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS DIABETES_FLAG_EVER,
        
        -- E66: Obesity
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^E66\\.' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS OBESITY_FLAG_EVER
        
    FROM cohort c
    LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_conditions_all ca
        ON c.PAT_ID = ca.PAT_ID 
        AND c.END_DTTM = ca.END_DTTM
        AND ca.WINDOW = 'ever'  -- Lifetime history for risk factors
    GROUP BY c.PAT_ID, c.END_DTTM
)

SELECT * FROM risk_agg
""")

print("✓ Risk factor features created (batch 2)")

# ========================================
# CELL 6
# ========================================

# =========================================================================
# CELL 5 - EXTRACT OTHER CONDITIONS AND SCREENING (BATCH 3)
# =========================================================================
# This cell processes other relevant conditions and screening history

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_icd10_other AS

WITH cohort AS (
    SELECT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
),

other_agg AS (
    SELECT 
        c.PAT_ID,
        c.END_DTTM,
        ca.WINDOW,
        
        -- DIVERTICULAR DISEASE (associated condition, can cause similar symptoms)
        -- K57: Diverticular disease of intestine
        SUM(CASE WHEN ca.CODE RLIKE '^K57\\.' THEN 1 ELSE 0 END) AS DIVERTICULAR_CNT,
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^K57\\.' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS DIVERTICULAR_FLAG,
        
        -- CRC SCREENING CODES (indicates appropriate screening or diagnostic workup)
        -- Z12.11: Encounter for screening for malignant neoplasm of colon
        -- Z12.12: Encounter for screening for malignant neoplasm of rectum
        CASE WHEN SUM(CASE WHEN ca.CODE RLIKE '^Z12\\.(11|12)' THEN 1 ELSE 0 END) > 0 
             THEN 1 ELSE 0 END AS CRC_SCREEN_FLAG,
        
        -- Most recent screening date (for overdue screening detection)
        MAX(CASE WHEN ca.CODE RLIKE '^Z12\\.(11|12)' 
                 THEN ca.CONTACT_DATE END) AS CRC_SCREEN_LATEST
        
    FROM cohort c
    LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_conditions_all ca
        ON c.PAT_ID = ca.PAT_ID 
        AND c.END_DTTM = ca.END_DTTM
    WHERE ca.WINDOW IN ('12mo', '24mo')  -- Recent timeframe for these conditions
    GROUP BY c.PAT_ID, c.END_DTTM, ca.WINDOW
)

-- Pivot by time window
SELECT 
    PAT_ID,
    END_DTTM,
    
    -- Diverticular disease in past 24 months
    SUM(CASE WHEN WINDOW = '24mo' THEN DIVERTICULAR_FLAG ELSE 0 END) AS DIVERTICULAR_FLAG_24MO,
    SUM(CASE WHEN WINDOW = '24mo' THEN DIVERTICULAR_CNT ELSE 0 END) AS DIVERTICULAR_CNT_24MO,
    
    -- Screening in past 12 and 24 months
    SUM(CASE WHEN WINDOW = '12mo' THEN CRC_SCREEN_FLAG ELSE 0 END) AS CRC_SCREEN_FLAG_12MO,
    SUM(CASE WHEN WINDOW = '24mo' THEN CRC_SCREEN_FLAG ELSE 0 END) AS CRC_SCREEN_FLAG_24MO,
    
    -- Days since last screening code (NULL if never screened)
    DATEDIFF(END_DTTM, MAX(CRC_SCREEN_LATEST)) AS DAYS_SINCE_LAST_CRC_SCREEN_CODE
    
FROM other_agg
GROUP BY PAT_ID, END_DTTM
""")

print("✓ Other condition features created (batch 3)")

# ========================================
# CELL 7
# ========================================

# =========================================================================
# CELL 6 - ENHANCED - EXTRACT ADDITIONAL DIAGNOSTIC FEATURES WITH FAMILY HISTORY
# =========================================================================
# Purpose: Add family history from FAMILY_HX table (using discovered codes), 
# hereditary syndromes, additional GI symptoms, and inflammatory markers

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_icd10_enhanced AS

WITH cohort AS (
    SELECT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
),

-- Extract family history from FAMILY_HX table using specific MEDICAL_HX_C codes
family_history_structured AS (
    SELECT DISTINCT
        c.PAT_ID,
        c.END_DTTM,
        
        -- CRC family history (codes 10404=Colon Cancer, 20172=Rectal Cancer)
        MAX(CASE WHEN fh.MEDICAL_HX_C IN (10404, 20172) THEN 1 ELSE 0 END) AS FHX_CRC_CODED,
        
        -- Polyps family history (codes 20103=Colon Polyps, 20191=Colonic polyp)
        MAX(CASE WHEN fh.MEDICAL_HX_C IN (20103, 20191) THEN 1 ELSE 0 END) AS FHX_POLYPS_CODED,
        
        -- Lynch syndrome/HNPCC (code 103028)
        MAX(CASE WHEN fh.MEDICAL_HX_C = 103028 THEN 1 ELSE 0 END) AS FHX_LYNCH_SYNDROME,
        
        -- Other GI cancers
        -- 20106=Stomach, 20153=Gastric, 20150=Pancreatic, 20173=Liver, 20171=Esophageal
        MAX(CASE WHEN fh.MEDICAL_HX_C IN (20106, 20153, 20150, 20173, 20171) 
                 THEN 1 ELSE 0 END) AS FHX_GI_CANCER_CODED,
        
        -- Any cancer in family (code 600=Cancer, plus major cancer types)
        MAX(CASE WHEN fh.MEDICAL_HX_C IN (600, 10403, 10407, 10405, 20150, 113012) 
                 THEN 1 ELSE 0 END) AS FHX_ANY_CANCER_CODED,
        
        -- First degree relative with CRC
        MAX(CASE 
            WHEN fh.MEDICAL_HX_C IN (10404, 20172) 
            AND UPPER(fh.FAM_RELATION_NAME) IN ('MOTHER','FATHER','BROTHER','SISTER','SON','DAUGHTER')
            THEN 1 ELSE 0 
        END) AS FHX_FIRST_DEGREE_CRC,
        
        -- Age of onset for CRC/polyps (early onset is high risk)
        MIN(CASE 
            WHEN fh.MEDICAL_HX_C IN (10404, 20172, 20103, 20191) 
            AND fh.AGE_OF_ONSET IS NOT NULL 
            THEN fh.AGE_OF_ONSET 
        END) AS FHX_YOUNGEST_CRC_ONSET,
        
        -- Count of family members with CRC/polyps
        COUNT(DISTINCT CASE 
            WHEN fh.MEDICAL_HX_C IN (10404, 20172, 20103, 20191) 
            THEN fh.FAM_RELATION_NAME 
        END) AS FHX_CRC_POLYP_COUNT
        
    FROM cohort c
    LEFT JOIN prod.clarity.family_hx fh
        ON c.PAT_ID = fh.PAT_ID
        AND DATE(fh.CONTACT_DATE) <= c.END_DTTM
    GROUP BY c.PAT_ID, c.END_DTTM
),

-- Extract ICD-10 based conditions including family history codes
icd_conditions AS (
    SELECT 
        c.PAT_ID,
        c.END_DTTM,
        
        -- HEREDITARY/GENETIC RISK FLAGS (lifetime history)
        MAX(CASE WHEN ca.WINDOW = 'ever' AND ca.CODE RLIKE '^Z15\\.0' 
                 THEN 1 ELSE 0 END) AS GENETIC_CANCER_RISK_FLAG,
        
        -- Family history from ICD-10 codes (as backup/supplement)
        MAX(CASE WHEN ca.WINDOW = 'ever' AND ca.CODE RLIKE '^Z80\\.0' 
                 THEN 1 ELSE 0 END) AS FHX_DIGESTIVE_CANCER_ICD,
        MAX(CASE WHEN ca.WINDOW = 'ever' AND ca.CODE RLIKE '^Z83\\.71' 
                 THEN 1 ELSE 0 END) AS FHX_COLONIC_POLYPS_ICD,
        MAX(CASE WHEN ca.WINDOW = 'ever' AND ca.CODE RLIKE '^Z84' 
                 THEN 1 ELSE 0 END) AS FHX_GENETIC_DISORDER_FLAG,
        
        -- GI SYMPTOMS (24-month window for active symptoms)
        MAX(CASE WHEN ca.WINDOW = '24mo' AND ca.CODE RLIKE '^K90' 
                 THEN 1 ELSE 0 END) AS MALABSORPTION_FLAG_24MO,
        MAX(CASE WHEN ca.WINDOW = '24mo' AND ca.CODE RLIKE '^K58\\.0' 
                 THEN 1 ELSE 0 END) AS IBS_DIARRHEA_FLAG_24MO,
        MAX(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^K92\\.0' 
                 THEN 1 ELSE 0 END) AS HEMATEMESIS_FLAG_12MO,
        MAX(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^R14' 
                 THEN 1 ELSE 0 END) AS BLOATING_FLAG_12MO,
        MAX(CASE WHEN ca.WINDOW = '24mo' AND ca.CODE RLIKE '^K63\\.0' 
                 THEN 1 ELSE 0 END) AS INTESTINAL_ABSCESS_FLAG_24MO,
        MAX(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^K59\\.00' 
                 THEN 1 ELSE 0 END) AS CONSTIPATION_FLAG_12MO,
        
        -- INFLAMMATORY CONDITIONS
        MAX(CASE WHEN ca.WINDOW = 'ever' AND ca.CODE RLIKE '^M06' 
                 THEN 1 ELSE 0 END) AS RHEUMATOID_ARTHRITIS_FLAG,
        MAX(CASE WHEN ca.WINDOW = 'ever' AND ca.CODE RLIKE '^L40' 
                 THEN 1 ELSE 0 END) AS PSORIASIS_FLAG,
        MAX(CASE WHEN ca.WINDOW = '24mo' AND ca.CODE RLIKE '^K29' 
                 THEN 1 ELSE 0 END) AS GASTRITIS_FLAG_24MO,
        
        -- DETAILED ANEMIA SUBTYPES
        MAX(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^D50' 
                 THEN 1 ELSE 0 END) AS IRON_DEF_ANEMIA_FLAG_12MO,
        MAX(CASE WHEN ca.WINDOW = '12mo' AND ca.CODE RLIKE '^D64\\.9' 
                 THEN 1 ELSE 0 END) AS ANEMIA_UNSPEC_FLAG_12MO
        
    FROM cohort c
    LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_conditions_all ca
        ON c.PAT_ID = ca.PAT_ID 
        AND c.END_DTTM = ca.END_DTTM
    GROUP BY c.PAT_ID, c.END_DTTM
)

-- FIXED: Changed from FULL OUTER JOIN to LEFT JOIN to prevent including patients outside cohort
-- Combine all features - starting from ICD conditions (which are already joined to cohort)
SELECT 
    ic.PAT_ID,
    ic.END_DTTM,
    
    -- Family history features (combine structured codes and ICD-10)
    GREATEST(
        COALESCE(fh.FHX_CRC_CODED, 0),
        COALESCE(ic.FHX_DIGESTIVE_CANCER_ICD, 0)
    ) AS FHX_CRC_COMBINED,
    
    GREATEST(
        COALESCE(fh.FHX_POLYPS_CODED, 0),
        COALESCE(ic.FHX_COLONIC_POLYPS_ICD, 0)
    ) AS FHX_POLYPS_COMBINED,
    
    -- Direct from family history table
    COALESCE(fh.FHX_LYNCH_SYNDROME, 0) AS FHX_LYNCH_SYNDROME,
    COALESCE(fh.FHX_FIRST_DEGREE_CRC, 0) AS FHX_FIRST_DEGREE_CRC,
    COALESCE(fh.FHX_GI_CANCER_CODED, 0) AS FHX_GI_CANCER,
    COALESCE(fh.FHX_ANY_CANCER_CODED, 0) AS FHX_ANY_CANCER,
    fh.FHX_YOUNGEST_CRC_ONSET,
    COALESCE(fh.FHX_CRC_POLYP_COUNT, 0) AS FHX_CRC_POLYP_COUNT,
    
    -- High-risk family history composite flag
    CASE 
        WHEN fh.FHX_LYNCH_SYNDROME = 1 THEN 1
        WHEN fh.FHX_YOUNGEST_CRC_ONSET < 50 THEN 1
        WHEN fh.FHX_CRC_POLYP_COUNT >= 2 THEN 1
        WHEN fh.FHX_FIRST_DEGREE_CRC = 1 THEN 1
        ELSE 0
    END AS HIGH_RISK_FHX_FLAG,
    
    -- ICD-10 based features
    COALESCE(ic.GENETIC_CANCER_RISK_FLAG, 0) AS GENETIC_CANCER_RISK_FLAG,
    COALESCE(ic.FHX_GENETIC_DISORDER_FLAG, 0) AS FHX_GENETIC_DISORDER_FLAG,
    COALESCE(ic.MALABSORPTION_FLAG_24MO, 0) AS MALABSORPTION_FLAG_24MO,
    COALESCE(ic.IBS_DIARRHEA_FLAG_24MO, 0) AS IBS_DIARRHEA_FLAG_24MO,
    COALESCE(ic.HEMATEMESIS_FLAG_12MO, 0) AS HEMATEMESIS_FLAG_12MO,
    COALESCE(ic.BLOATING_FLAG_12MO, 0) AS BLOATING_FLAG_12MO,
    COALESCE(ic.INTESTINAL_ABSCESS_FLAG_24MO, 0) AS INTESTINAL_ABSCESS_FLAG_24MO,
    COALESCE(ic.CONSTIPATION_FLAG_12MO, 0) AS CONSTIPATION_FLAG_12MO,
    COALESCE(ic.RHEUMATOID_ARTHRITIS_FLAG, 0) AS RHEUMATOID_ARTHRITIS_FLAG,
    COALESCE(ic.PSORIASIS_FLAG, 0) AS PSORIASIS_FLAG,
    COALESCE(ic.GASTRITIS_FLAG_24MO, 0) AS GASTRITIS_FLAG_24MO,
    COALESCE(ic.IRON_DEF_ANEMIA_FLAG_12MO, 0) AS IRON_DEF_ANEMIA_FLAG_12MO,
    COALESCE(ic.ANEMIA_UNSPEC_FLAG_12MO, 0) AS ANEMIA_UNSPEC_FLAG_12MO,
    
    -- Composite scores
    (COALESCE(ic.RHEUMATOID_ARTHRITIS_FLAG, 0) + 
     COALESCE(ic.PSORIASIS_FLAG, 0) + 
     COALESCE(ic.GASTRITIS_FLAG_24MO, 0) + 
     COALESCE(ic.INTESTINAL_ABSCESS_FLAG_24MO, 0)) AS INFLAMMATORY_BURDEN,
     
    (COALESCE(ic.MALABSORPTION_FLAG_24MO, 0) + 
     COALESCE(ic.IBS_DIARRHEA_FLAG_24MO, 0) + 
     COALESCE(ic.HEMATEMESIS_FLAG_12MO, 0) + 
     COALESCE(ic.BLOATING_FLAG_12MO, 0) + 
     COALESCE(ic.INTESTINAL_ABSCESS_FLAG_24MO, 0)) AS GI_COMPLEXITY_SCORE,
     
    -- Bowel pattern
    CASE 
        WHEN ic.CONSTIPATION_FLAG_12MO = 1 AND ic.IBS_DIARRHEA_FLAG_24MO = 1 THEN 'alternating'
        WHEN ic.CONSTIPATION_FLAG_12MO = 1 THEN 'constipation'
        WHEN ic.IBS_DIARRHEA_FLAG_24MO = 1 THEN 'diarrhea'
        ELSE 'normal'
    END AS BOWEL_PATTERN
    
FROM icd_conditions ic
LEFT JOIN family_history_structured fh
    ON ic.PAT_ID = fh.PAT_ID AND ic.END_DTTM = fh.END_DTTM
""")

print("✓ Enhanced ICD-10 features with proper family history created")

# ========================================
# CELL 8
# ========================================

# =========================================================================
# CELL 7 - ASSEMBLE ALL ICD-10 FEATURES INTO TEMP TABLE (CORRECTED)
# =========================================================================

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_icd_10_temp AS

SELECT 
    c.PAT_ID,
    c.END_DTTM,
    
    -- SYMPTOM FEATURES from Cell 3A
    COALESCE(s.BLEED_FLAG_12MO, 0) AS BLEED_FLAG_12MO,
    COALESCE(s.BLEED_CNT_12MO, 0) AS BLEED_CNT_12MO,
    COALESCE(s.BLEED_FLAG_24MO, 0) AS BLEED_FLAG_24MO,
    COALESCE(s.BLEED_CNT_24MO, 0) AS BLEED_CNT_24MO,
    COALESCE(s.PAIN_FLAG_12MO, 0) AS PAIN_FLAG_12MO,
    COALESCE(s.PAIN_CNT_12MO, 0) AS PAIN_CNT_12MO,
    COALESCE(s.PAIN_FLAG_24MO, 0) AS PAIN_FLAG_24MO,
    COALESCE(s.PAIN_CNT_24MO, 0) AS PAIN_CNT_24MO,
    COALESCE(s.BOWELCHG_FLAG_12MO, 0) AS BOWELCHG_FLAG_12MO,
    COALESCE(s.BOWELCHG_CNT_12MO, 0) AS BOWELCHG_CNT_12MO,
    COALESCE(s.BOWELCHG_FLAG_24MO, 0) AS BOWELCHG_FLAG_24MO,
    COALESCE(s.BOWELCHG_CNT_24MO, 0) AS BOWELCHG_CNT_24MO,
    COALESCE(s.WTLOSS_FLAG_12MO, 0) AS WTLOSS_FLAG_12MO,
    COALESCE(s.WTLOSS_CNT_12MO, 0) AS WTLOSS_CNT_12MO,
    COALESCE(s.WTLOSS_FLAG_24MO, 0) AS WTLOSS_FLAG_24MO,
    COALESCE(s.WTLOSS_CNT_24MO, 0) AS WTLOSS_CNT_24MO,
    COALESCE(s.FATIGUE_FLAG_12MO, 0) AS FATIGUE_FLAG_12MO,
    COALESCE(s.FATIGUE_CNT_12MO, 0) AS FATIGUE_CNT_12MO,
    COALESCE(s.FATIGUE_FLAG_24MO, 0) AS FATIGUE_FLAG_24MO,
    COALESCE(s.FATIGUE_CNT_24MO, 0) AS FATIGUE_CNT_24MO,
    COALESCE(s.ANEMIA_FLAG_12MO, 0) AS ANEMIA_FLAG_12MO,
    COALESCE(s.ANEMIA_CNT_12MO, 0) AS ANEMIA_CNT_12MO,
    COALESCE(s.ANEMIA_FLAG_24MO, 0) AS ANEMIA_FLAG_24MO,
    COALESCE(s.ANEMIA_CNT_24MO, 0) AS ANEMIA_CNT_24MO,
    
    -- RISK FACTOR FEATURES from Cell 3B (keep old ICD-based family history for now)
    COALESCE(r.POLYPS_FLAG_EVER, 0) AS POLYPS_FLAG_EVER,
    COALESCE(r.POLYPS_CNT_EVER, 0) AS POLYPS_CNT_EVER,
    COALESCE(r.IBD_FLAG_EVER, 0) AS IBD_FLAG_EVER,
    COALESCE(r.IBD_CNT_EVER, 0) AS IBD_CNT_EVER,
    COALESCE(r.MALIGNANCY_FLAG_EVER, 0) AS MALIGNANCY_FLAG_EVER,
    COALESCE(r.MALIGNANCY_CNT_EVER, 0) AS MALIGNANCY_CNT_EVER,
    COALESCE(r.FHX_CRC_FLAG_EVER, 0) AS FHX_CRC_FLAG_EVER_ICD,  -- Renamed to distinguish
    COALESCE(r.FHX_ANAL_FLAG_EVER, 0) AS FHX_ANAL_FLAG_EVER,
    COALESCE(r.FHX_GI_MALIG_FLAG_EVER, 0) AS FHX_GI_MALIG_FLAG_EVER,
    COALESCE(r.PHX_ANUS_MALIG_FLAG_EVER, 0) AS PHX_ANUS_MALIG_FLAG_EVER,
    COALESCE(r.PHX_RECTUM_MALIG_FLAG_EVER, 0) AS PHX_RECTUM_MALIG_FLAG_EVER,
    COALESCE(r.DIABETES_FLAG_EVER, 0) AS DIABETES_FLAG_EVER,
    COALESCE(r.OBESITY_FLAG_EVER, 0) AS OBESITY_FLAG_EVER,
    
    -- OTHER CONDITIONS from Cell 3C
    COALESCE(o.DIVERTICULAR_FLAG_24MO, 0) AS DIVERTICULAR_FLAG_24MO,
    COALESCE(o.DIVERTICULAR_CNT_24MO, 0) AS DIVERTICULAR_CNT_24MO,
    COALESCE(o.CRC_SCREEN_FLAG_12MO, 0) AS CRC_SCREEN_FLAG_12MO,
    COALESCE(o.CRC_SCREEN_FLAG_24MO, 0) AS CRC_SCREEN_FLAG_24MO,
    
    -- RECENCY FEATURES
    s.DAYS_SINCE_LAST_BLEED,
    s.DAYS_SINCE_LAST_ANEMIA,
    s.DAYS_SINCE_LAST_PAIN,
    o.DAYS_SINCE_LAST_CRC_SCREEN_CODE,
    
    -- ENHANCED FAMILY HISTORY from FAMILY_HX table (Cell 3D)
    COALESCE(e.FHX_CRC_COMBINED, 0) AS FHX_CRC_COMBINED,
    COALESCE(e.FHX_POLYPS_COMBINED, 0) AS FHX_POLYPS_COMBINED,
    COALESCE(e.FHX_LYNCH_SYNDROME, 0) AS FHX_LYNCH_SYNDROME,
    COALESCE(e.FHX_FIRST_DEGREE_CRC, 0) AS FHX_FIRST_DEGREE_CRC,
    COALESCE(e.FHX_GI_CANCER, 0) AS FHX_GI_CANCER,
    COALESCE(e.FHX_ANY_CANCER, 0) AS FHX_ANY_CANCER,
    e.FHX_YOUNGEST_CRC_ONSET,
    COALESCE(e.FHX_CRC_POLYP_COUNT, 0) AS FHX_CRC_POLYP_COUNT,
    COALESCE(e.HIGH_RISK_FHX_FLAG, 0) AS HIGH_RISK_FHX_FLAG,
    
    -- ADDITIONAL GI SYMPTOMS from Cell 3D
    COALESCE(e.GENETIC_CANCER_RISK_FLAG, 0) AS GENETIC_CANCER_RISK_FLAG,
    COALESCE(e.MALABSORPTION_FLAG_24MO, 0) AS MALABSORPTION_FLAG_24MO,
    COALESCE(e.IBS_DIARRHEA_FLAG_24MO, 0) AS IBS_DIARRHEA_FLAG_24MO,
    COALESCE(e.HEMATEMESIS_FLAG_12MO, 0) AS HEMATEMESIS_FLAG_12MO,
    COALESCE(e.BLOATING_FLAG_12MO, 0) AS BLOATING_FLAG_12MO,
    COALESCE(e.CONSTIPATION_FLAG_12MO, 0) AS CONSTIPATION_FLAG_12MO,
    COALESCE(e.IRON_DEF_ANEMIA_FLAG_12MO, 0) AS IRON_DEF_ANEMIA_FLAG_12MO,
    COALESCE(e.INFLAMMATORY_BURDEN, 0) AS INFLAMMATORY_BURDEN,
    COALESCE(e.GI_COMPLEXITY_SCORE, 0) AS GI_COMPLEXITY_SCORE,
    e.BOWEL_PATTERN,
    
    -- COMPOSITE RISK SCORES
    CASE WHEN (COALESCE(s.BLEED_FLAG_12MO, 0) + 
               COALESCE(s.PAIN_FLAG_12MO, 0) + 
               COALESCE(s.BOWELCHG_FLAG_12MO, 0)) >= 2 
         THEN 1 ELSE 0 END AS CRC_SYMPTOM_TRIAD,
    
    CASE WHEN COALESCE(s.ANEMIA_FLAG_12MO, 0) = 1 
          AND COALESCE(s.BLEED_FLAG_12MO, 0) = 1 
         THEN 1 ELSE 0 END AS IDA_WITH_BLEEDING,
    
    (COALESCE(s.BLEED_FLAG_12MO, 0) + COALESCE(s.PAIN_FLAG_12MO, 0) + 
     COALESCE(s.BOWELCHG_FLAG_12MO, 0) + COALESCE(s.WTLOSS_FLAG_12MO, 0) + 
     COALESCE(s.FATIGUE_FLAG_12MO, 0) + COALESCE(s.ANEMIA_FLAG_12MO, 0)) AS SYMPTOM_BURDEN_12MO,
    
    CASE WHEN (COALESCE(r.IBD_FLAG_EVER, 0) + 
               COALESCE(r.POLYPS_FLAG_EVER, 0) + 
               COALESCE(r.MALIGNANCY_FLAG_EVER, 0)) >= 1 
         THEN 1 ELSE 0 END AS HIGH_RISK_HISTORY,
    
    CASE WHEN COALESCE(r.DIABETES_FLAG_EVER, 0) = 1 
          AND COALESCE(r.OBESITY_FLAG_EVER, 0) = 1 
         THEN 1 ELSE 0 END AS METABOLIC_SYNDROME,
    
    -- Updated to use ICD-based FHx (will replace with combined in final)
    CASE WHEN (COALESCE(r.FHX_CRC_FLAG_EVER, 0) + 
               COALESCE(r.FHX_ANAL_FLAG_EVER, 0) + 
               COALESCE(r.FHX_GI_MALIG_FLAG_EVER, 0)) >= 1 
         THEN 1 ELSE 0 END AS ANY_FHX_GI_MALIG

FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_icd10_symptoms s
    ON c.PAT_ID = s.PAT_ID AND c.END_DTTM = s.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_icd10_risk_factors r
    ON c.PAT_ID = r.PAT_ID AND c.END_DTTM = r.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_icd10_other o
    ON c.PAT_ID = o.PAT_ID AND c.END_DTTM = o.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_icd10_enhanced e
    ON c.PAT_ID = e.PAT_ID AND c.END_DTTM = e.END_DTTM
""")

print("✓ ICD-10 temp table assembled with ALL enhanced features including family history")

# ========================================
# CELL 9
# ========================================

# =========================================================================
# CELL 8 - EXTRACT DIAGNOSIS CODES FOR COMORBIDITY SCORING
# =========================================================================
# Purpose: Create a single base table with all ICD-10 codes we need for scoring
# This avoids multiple passes through the large encounter tables

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_dx_for_scoring AS

SELECT
    c.PAT_ID,
    c.END_DTTM,
    dx.ICD10_CODE,
    pe.CONTACT_DATE,
    -- Create separate flags for each window (not nested)
    CASE 
        WHEN DATE(pe.CONTACT_DATE) BETWEEN DATE_SUB(c.END_DTTM, 365) AND c.END_DTTM 
        THEN 1 ELSE 0 
    END AS IN_12MO,
    CASE 
        WHEN DATE(pe.CONTACT_DATE) BETWEEN DATE_SUB(c.END_DTTM, 730) AND c.END_DTTM 
        THEN 1 ELSE 0 
    END AS IN_24MO
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
JOIN clarity_cur.pat_enc_enh pe
    ON pe.PAT_ID = c.PAT_ID
    AND DATE(pe.CONTACT_DATE) BETWEEN DATE_SUB(c.END_DTTM, 730) AND c.END_DTTM
JOIN clarity_cur.pat_enc_dx_enh dx
    ON dx.PAT_ENC_CSN_ID = pe.PAT_ENC_CSN_ID
WHERE dx.ICD10_CODE IS NOT NULL
""")

print("✓ Diagnosis codes extracted for scoring")

# ========================================
# CELL 10
# ========================================

# =========================================================================
# CELL 9 - CALCULATE CHARLSON SCORES (OPTIMIZED)
# =========================================================================
# Purpose: Calculate Charlson scores using pre-extracted diagnosis table
# This avoids re-querying the large clarity tables

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_charlson_scores AS

WITH 
-- Use pre-extracted diagnoses from CELL 10 instead of re-querying
dx_12mo AS (
    SELECT DISTINCT PAT_ID, END_DTTM, ICD10_CODE
    FROM {trgt_cat}.clncl_ds.herald_test_train_dx_for_scoring
    WHERE IN_12MO = 1
),

dx_24mo AS (
    SELECT DISTINCT PAT_ID, END_DTTM, ICD10_CODE
    FROM {trgt_cat}.clncl_ds.herald_test_train_dx_for_scoring
    WHERE IN_24MO = 1
),

-- Apply Charlson weights to 12-month codes
charlson_12mo AS (
    SELECT
        PAT_ID,
        END_DTTM,
        CASE
            WHEN ICD10_CODE RLIKE '^(I21|I22)' THEN 1
            WHEN ICD10_CODE RLIKE '^I50' THEN 1
            WHEN ICD10_CODE RLIKE '^(I70|I71|I73)' THEN 1
            WHEN ICD10_CODE RLIKE '^(I60|I61|I62|I63|I64)' THEN 1
            WHEN ICD10_CODE RLIKE '^(G30|F01|F03)' THEN 1
            WHEN ICD10_CODE RLIKE '^J44' THEN 1
            WHEN ICD10_CODE RLIKE '^(M05|M06|M32|M33|M34)' THEN 1
            WHEN ICD10_CODE RLIKE '^(K25|K26|K27|K28)' THEN 1
            WHEN ICD10_CODE RLIKE '^K70' THEN 1
            WHEN ICD10_CODE RLIKE '^(E10|E11)' THEN 1
            WHEN ICD10_CODE RLIKE '^(E13|E14)' THEN 2
            WHEN ICD10_CODE RLIKE '^(G81|G82)' THEN 2
            WHEN ICD10_CODE RLIKE '^N18' THEN 2
            WHEN ICD10_CODE RLIKE '^C(?:0[0-9]|[1-8][0-9]|9[0-7])' THEN 2
            WHEN ICD10_CODE RLIKE '^(K72|K76)' THEN 3
            WHEN ICD10_CODE RLIKE '^(C78|C79)' THEN 6
            WHEN ICD10_CODE RLIKE '^B2[0-4]' THEN 6
            ELSE 0
        END AS charlson_wt
    FROM dx_12mo
),

-- Apply Charlson weights to 24-month codes  
charlson_24mo AS (
    SELECT
        PAT_ID,
        END_DTTM,
        CASE
            WHEN ICD10_CODE RLIKE '^(I21|I22)' THEN 1
            WHEN ICD10_CODE RLIKE '^I50' THEN 1
            WHEN ICD10_CODE RLIKE '^(I70|I71|I73)' THEN 1
            WHEN ICD10_CODE RLIKE '^(I60|I61|I62|I63|I64)' THEN 1
            WHEN ICD10_CODE RLIKE '^(G30|F01|F03)' THEN 1
            WHEN ICD10_CODE RLIKE '^J44' THEN 1
            WHEN ICD10_CODE RLIKE '^(M05|M06|M32|M33|M34)' THEN 1
            WHEN ICD10_CODE RLIKE '^(K25|K26|K27|K28)' THEN 1
            WHEN ICD10_CODE RLIKE '^K70' THEN 1
            WHEN ICD10_CODE RLIKE '^(E10|E11)' THEN 1
            WHEN ICD10_CODE RLIKE '^(E13|E14)' THEN 2
            WHEN ICD10_CODE RLIKE '^(G81|G82)' THEN 2
            WHEN ICD10_CODE RLIKE '^N18' THEN 2
            WHEN ICD10_CODE RLIKE '^C(?:0[0-9]|[1-8][0-9]|9[0-7])' THEN 2
            WHEN ICD10_CODE RLIKE '^(K72|K76)' THEN 3
            WHEN ICD10_CODE RLIKE '^(C78|C79)' THEN 6
            WHEN ICD10_CODE RLIKE '^B2[0-4]' THEN 6
            ELSE 0
        END AS charlson_wt
    FROM dx_24mo
)

-- Sum distinct weights per patient
SELECT
    COALESCE(c12.PAT_ID, c24.PAT_ID) AS PAT_ID,
    COALESCE(c12.END_DTTM, c24.END_DTTM) AS END_DTTM,
    COALESCE(c12.CHARLSON_SCORE_12MO, 0) AS CHARLSON_SCORE_12MO,
    COALESCE(c24.CHARLSON_SCORE_24MO, 0) AS CHARLSON_SCORE_24MO
FROM (
    SELECT PAT_ID, END_DTTM, SUM(charlson_wt) AS CHARLSON_SCORE_12MO
    FROM (
        SELECT DISTINCT PAT_ID, END_DTTM, charlson_wt
        FROM charlson_12mo
        WHERE charlson_wt > 0
    ) t
    GROUP BY PAT_ID, END_DTTM
) c12
FULL OUTER JOIN (
    SELECT PAT_ID, END_DTTM, SUM(charlson_wt) AS CHARLSON_SCORE_24MO
    FROM (
        SELECT DISTINCT PAT_ID, END_DTTM, charlson_wt
        FROM charlson_24mo
        WHERE charlson_wt > 0
    ) t
    GROUP BY PAT_ID, END_DTTM
) c24
ON c12.PAT_ID = c24.PAT_ID AND c12.END_DTTM = c24.END_DTTM
""")

print("✓ Charlson scores calculated (using pre-extracted diagnoses)")

# ========================================
# CELL 11
# ========================================

# =========================================================================
# CELL 10 - CALCULATE ELIXHAUSER AND CRC RISK SCORES (OPTIMIZED)
# =========================================================================
# Purpose: Calculate two additional scoring systems using pre-extracted diagnoses

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_other_scores AS

WITH 
-- Use pre-extracted diagnoses from CELL 10 instead of re-querying
dx_12mo AS (
    SELECT DISTINCT PAT_ID, END_DTTM, ICD10_CODE
    FROM {trgt_cat}.clncl_ds.herald_test_train_dx_for_scoring
    WHERE IN_12MO = 1
),

dx_24mo AS (
    SELECT DISTINCT PAT_ID, END_DTTM, ICD10_CODE
    FROM {trgt_cat}.clncl_ds.herald_test_train_dx_for_scoring
    WHERE IN_24MO = 1
),

-- Apply Elixhauser weights to 12-month codes
elix_12mo AS (
    SELECT
        PAT_ID,
        END_DTTM,
        CASE
            WHEN ICD10_CODE RLIKE '^(I10|I11)' THEN 1         -- HTN
            WHEN ICD10_CODE RLIKE '^I50' THEN 1               -- CHF  
            WHEN ICD10_CODE RLIKE '^I25' THEN 1               -- CAD
            WHEN ICD10_CODE RLIKE '^N18' THEN 2               -- Renal failure
            WHEN ICD10_CODE RLIKE '^C([0-7][0-9]|8[0-9]|9[0-7])' THEN 1  -- Malignancy
            WHEN ICD10_CODE RLIKE '^K70' THEN 2               -- Liver disease
            WHEN ICD10_CODE RLIKE '^(E10|E11)' THEN 1         -- Diabetes
            WHEN ICD10_CODE RLIKE '^(G30|F01|F03)' THEN 1     -- Dementia
            WHEN ICD10_CODE RLIKE '^(F32|F33)' THEN 1         -- Depression
            WHEN ICD10_CODE RLIKE '^F1[0-9]' THEN 1           -- Substance use
            ELSE 0
        END AS elix_wt
    FROM dx_12mo
),

-- Apply Elixhauser weights to 24-month codes
elix_24mo AS (
    SELECT
        PAT_ID,
        END_DTTM,
        CASE
            WHEN ICD10_CODE RLIKE '^(I10|I11)' THEN 1
            WHEN ICD10_CODE RLIKE '^I50' THEN 1
            WHEN ICD10_CODE RLIKE '^I25' THEN 1
            WHEN ICD10_CODE RLIKE '^N18' THEN 2
            WHEN ICD10_CODE RLIKE '^C([0-7][0-9]|8[0-9]|9[0-7])' THEN 1
            WHEN ICD10_CODE RLIKE '^K70' THEN 2
            WHEN ICD10_CODE RLIKE '^(E10|E11)' THEN 1
            WHEN ICD10_CODE RLIKE '^(G30|F01|F03)' THEN 1
            WHEN ICD10_CODE RLIKE '^(F32|F33)' THEN 1
            WHEN ICD10_CODE RLIKE '^F1[0-9]' THEN 1
            ELSE 0
        END AS elix_wt
    FROM dx_24mo
),

-- Apply CRC risk weights to 12-month codes
crc_risk_12mo AS (
    SELECT
        PAT_ID,
        END_DTTM,
        CASE
            WHEN ICD10_CODE RLIKE '^(K50|K51)' THEN 3         -- IBD
            WHEN ICD10_CODE RLIKE '^D12' THEN 2               -- Polyps
            WHEN ICD10_CODE RLIKE '^E11' THEN 1               -- T2DM
            WHEN ICD10_CODE RLIKE '^E66' THEN 1               -- Obesity
            WHEN ICD10_CODE RLIKE '^Z85' THEN 1               -- Prior malignancy
            WHEN ICD10_CODE RLIKE '^K57' THEN 1               -- Diverticular
            WHEN ICD10_CODE RLIKE '^(Z80\\.0|Z80\\.2|Z80\\.9)' THEN 1  -- Family history
            ELSE 0
        END AS crc_risk_wt
    FROM dx_12mo
),

-- Apply CRC risk weights to 24-month codes
crc_risk_24mo AS (
    SELECT
        PAT_ID,
        END_DTTM,
        CASE
            WHEN ICD10_CODE RLIKE '^(K50|K51)' THEN 3
            WHEN ICD10_CODE RLIKE '^D12' THEN 2
            WHEN ICD10_CODE RLIKE '^E11' THEN 1
            WHEN ICD10_CODE RLIKE '^E66' THEN 1
            WHEN ICD10_CODE RLIKE '^Z85' THEN 1
            WHEN ICD10_CODE RLIKE '^K57' THEN 1
            WHEN ICD10_CODE RLIKE '^(Z80\\.0|Z80\\.2|Z80\\.9)' THEN 1
            ELSE 0
        END AS crc_risk_wt
    FROM dx_24mo
),

-- Aggregate Elixhauser scores
elix_scores AS (
    SELECT
        COALESCE(e12.PAT_ID, e24.PAT_ID) AS PAT_ID,
        COALESCE(e12.END_DTTM, e24.END_DTTM) AS END_DTTM,
        COALESCE(e12.ELIXHAUSER_SCORE_12MO, 0) AS ELIXHAUSER_SCORE_12MO,
        COALESCE(e24.ELIXHAUSER_SCORE_24MO, 0) AS ELIXHAUSER_SCORE_24MO
    FROM (
        SELECT PAT_ID, END_DTTM, SUM(elix_wt) AS ELIXHAUSER_SCORE_12MO
        FROM (
            SELECT DISTINCT PAT_ID, END_DTTM, elix_wt
            FROM elix_12mo
            WHERE elix_wt > 0
        ) t
        GROUP BY PAT_ID, END_DTTM
    ) e12
    FULL OUTER JOIN (
        SELECT PAT_ID, END_DTTM, SUM(elix_wt) AS ELIXHAUSER_SCORE_24MO
        FROM (
            SELECT DISTINCT PAT_ID, END_DTTM, elix_wt
            FROM elix_24mo
            WHERE elix_wt > 0
        ) t
        GROUP BY PAT_ID, END_DTTM
    ) e24
    ON e12.PAT_ID = e24.PAT_ID AND e12.END_DTTM = e24.END_DTTM
),

-- Aggregate CRC risk scores
crc_scores AS (
    SELECT
        COALESCE(cr12.PAT_ID, cr24.PAT_ID) AS PAT_ID,
        COALESCE(cr12.END_DTTM, cr24.END_DTTM) AS END_DTTM,
        COALESCE(cr12.CRC_RISK_SCORE_12MO, 0) AS CRC_RISK_SCORE_12MO,
        COALESCE(cr24.CRC_RISK_SCORE_24MO, 0) AS CRC_RISK_SCORE_24MO
    FROM (
        SELECT PAT_ID, END_DTTM, SUM(crc_risk_wt) AS CRC_RISK_SCORE_12MO
        FROM (
            SELECT DISTINCT PAT_ID, END_DTTM, crc_risk_wt
            FROM crc_risk_12mo
            WHERE crc_risk_wt > 0
        ) t
        GROUP BY PAT_ID, END_DTTM
    ) cr12
    FULL OUTER JOIN (
        SELECT PAT_ID, END_DTTM, SUM(crc_risk_wt) AS CRC_RISK_SCORE_24MO
        FROM (
            SELECT DISTINCT PAT_ID, END_DTTM, crc_risk_wt
            FROM crc_risk_24mo
            WHERE crc_risk_wt > 0
        ) t
        GROUP BY PAT_ID, END_DTTM
    ) cr24
    ON cr12.PAT_ID = cr24.PAT_ID AND cr12.END_DTTM = cr24.END_DTTM
)

-- Combine all scores
SELECT
    COALESCE(e.PAT_ID, c.PAT_ID) AS PAT_ID,
    COALESCE(e.END_DTTM, c.END_DTTM) AS END_DTTM,
    e.ELIXHAUSER_SCORE_12MO,
    e.ELIXHAUSER_SCORE_24MO,
    c.CRC_RISK_SCORE_12MO,
    c.CRC_RISK_SCORE_24MO
FROM elix_scores e
FULL OUTER JOIN crc_scores c
    ON e.PAT_ID = c.PAT_ID AND e.END_DTTM = c.END_DTTM
""")

print("✓ Elixhauser and CRC risk scores calculated (using pre-extracted diagnoses)")

# ========================================
# CELL 12
# ========================================

# =========================================================================
# CELL 11 - COMBINE SCORES AND ADD TEMPORAL FEATURES
# =========================================================================
# Purpose: Join comorbidity scores and create temporal acceleration features

spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_icd_10 AS

SELECT
    t.*,  -- All existing features from temp table
    
    -- Add Charlson and Elixhauser scores only (remove CRC risk score)
    COALESCE(c.CHARLSON_SCORE_12MO, 0) AS CHARLSON_SCORE_12MO,
    COALESCE(c.CHARLSON_SCORE_24MO, 0) AS CHARLSON_SCORE_24MO,
    COALESCE(o.ELIXHAUSER_SCORE_12MO, 0) AS ELIXHAUSER_SCORE_12MO,
    COALESCE(o.ELIXHAUSER_SCORE_24MO, 0) AS ELIXHAUSER_SCORE_24MO,
    
    -- Combined comorbidity burden
    GREATEST(
        COALESCE(c.CHARLSON_SCORE_12MO, 0),
        COALESCE(o.ELIXHAUSER_SCORE_12MO, 0)
    ) AS COMBINED_COMORBIDITY_12MO,
    
    GREATEST(
        COALESCE(c.CHARLSON_SCORE_24MO, 0),
        COALESCE(o.ELIXHAUSER_SCORE_24MO, 0)
    ) AS COMBINED_COMORBIDITY_24MO,
    
    -- TEMPORAL ACCELERATION FEATURES
    -- These capture if symptoms are getting worse (ratio > 0.5 means recent acceleration)
    
    -- Bleeding acceleration
    CASE 
        WHEN t.BLEED_CNT_24MO > 0 
        THEN t.BLEED_CNT_12MO / t.BLEED_CNT_24MO
        ELSE 0 
    END AS BLEED_ACCELERATION,
    
    -- Anemia acceleration  
    CASE 
        WHEN t.ANEMIA_CNT_24MO > 0 
        THEN t.ANEMIA_CNT_12MO / t.ANEMIA_CNT_24MO
        ELSE 0 
    END AS ANEMIA_ACCELERATION,
    
    -- Pain acceleration
    CASE 
        WHEN t.PAIN_CNT_24MO > 0 
        THEN t.PAIN_CNT_12MO / t.PAIN_CNT_24MO
        ELSE 0 
    END AS PAIN_ACCELERATION,
    
    -- Bowel change acceleration
    CASE 
        WHEN t.BOWELCHG_CNT_24MO > 0 
        THEN t.BOWELCHG_CNT_12MO / t.BOWELCHG_CNT_24MO
        ELSE 0 
    END AS BOWELCHG_ACCELERATION,
    
    -- Overall symptom acceleration
    CASE 
        WHEN (t.BLEED_CNT_24MO + t.ANEMIA_CNT_24MO + t.PAIN_CNT_24MO + 
              t.BOWELCHG_CNT_24MO + t.WTLOSS_CNT_24MO + t.FATIGUE_CNT_24MO) > 0
        THEN (t.BLEED_CNT_12MO + t.ANEMIA_CNT_12MO + t.PAIN_CNT_12MO + 
              t.BOWELCHG_CNT_12MO + t.WTLOSS_CNT_12MO + t.FATIGUE_CNT_12MO) /
             (t.BLEED_CNT_24MO + t.ANEMIA_CNT_24MO + t.PAIN_CNT_24MO + 
              t.BOWELCHG_CNT_24MO + t.WTLOSS_CNT_24MO + t.FATIGUE_CNT_24MO)
        ELSE 0 
    END AS SYMPTOM_ACCELERATION,
    
    -- New vs chronic symptoms (1 = all symptoms are new)
    CASE 
        WHEN t.SYMPTOM_BURDEN_12MO > 0 AND 
             (t.BLEED_FLAG_24MO + t.PAIN_FLAG_24MO + t.BOWELCHG_FLAG_24MO + 
              t.WTLOSS_FLAG_24MO + t.FATIGUE_FLAG_24MO + t.ANEMIA_FLAG_24MO -
              t.SYMPTOM_BURDEN_12MO) = 0
        THEN 1 
        ELSE 0 
    END AS ALL_SYMPTOMS_NEW

FROM {trgt_cat}.clncl_ds.herald_test_train_icd_10_temp t
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_charlson_scores c
    ON t.PAT_ID = c.PAT_ID AND t.END_DTTM = c.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_other_scores o
    ON t.PAT_ID = o.PAT_ID AND t.END_DTTM = o.END_DTTM
""")

print("✓ Final ICD-10 table created with acceleration features, CRC risk score removed")

# ========================================
# CELL 13
# ========================================

# =========================================================================
# CELL 12 - VALIDATE ROW COUNT
# =========================================================================
# Ensure we have exactly the same number of rows as our base cohort

result = spark.sql(f"""
SELECT 
    COUNT(*) as icd10_count,
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_test_train_icd_10
""")

result.show()
assert result.collect()[0]['diff'] == 0, "ERROR: Row count mismatch!"
print("✓ Row count validation passed")

# ========================================
# CELL 14
# ========================================

# =========================================================================
# CELL 13 - SYMPTOM PREVALENCE ANALYSIS
# =========================================================================
# Check how common each symptom/condition is in our population

spark.sql(f"""
SELECT 
    -- 12-month symptom prevalence (what % of patients have each symptom)
    ROUND(100.0 * SUM(BLEED_FLAG_12MO) / COUNT(*), 2) as bleed_12mo_pct,
    ROUND(100.0 * SUM(PAIN_FLAG_12MO) / COUNT(*), 2) as pain_12mo_pct,
    ROUND(100.0 * SUM(BOWELCHG_FLAG_12MO) / COUNT(*), 2) as bowel_chg_12mo_pct,
    ROUND(100.0 * SUM(WTLOSS_FLAG_12MO) / COUNT(*), 2) as weight_loss_12mo_pct,
    ROUND(100.0 * SUM(ANEMIA_FLAG_12MO) / COUNT(*), 2) as anemia_12mo_pct,
    
    -- Lifetime risk factor prevalence
    ROUND(100.0 * SUM(POLYPS_FLAG_EVER) / COUNT(*), 2) as polyps_ever_pct,
    ROUND(100.0 * SUM(IBD_FLAG_EVER) / COUNT(*), 2) as ibd_ever_pct,
    
    -- Family history from different sources
    ROUND(100.0 * SUM(FHX_CRC_FLAG_EVER_ICD) / COUNT(*), 2) as fhx_crc_icd_pct,
    ROUND(100.0 * SUM(FHX_CRC_COMBINED) / COUNT(*), 2) as fhx_crc_combined_pct,
    
    -- Composite feature prevalence
    ROUND(100.0 * SUM(CRC_SYMPTOM_TRIAD) / COUNT(*), 2) as symptom_triad_pct,
    ROUND(100.0 * SUM(IDA_WITH_BLEEDING) / COUNT(*), 2) as ida_bleeding_pct,
    
    -- Average symptom burden (0-6 scale)
    ROUND(AVG(SYMPTOM_BURDEN_12MO), 2) as avg_symptom_burden,
    MAX(SYMPTOM_BURDEN_12MO) as max_symptom_burden
    
FROM {trgt_cat}.clncl_ds.herald_test_train_icd_10
""").show(truncate=False)

print("\n✓ Symptom prevalence analysis complete")

# ========================================
# CELL 15
# ========================================

# =========================================================================
# CELL 14 - CRC ASSOCIATION ANALYSIS
# =========================================================================
# Critical validation: Do our features show higher CRC rates?
# We expect to see elevated rates for symptoms and risk factors

spark.sql(f"""
WITH outcome_analysis AS (
    SELECT 
        i.*,
        c.FUTURE_CRC_EVENT
    FROM {trgt_cat}.clncl_ds.herald_test_train_icd_10 i
    JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
        ON i.PAT_ID = c.PAT_ID AND i.END_DTTM = c.END_DTTM
    WHERE c.LABEL_USABLE = 1  -- Only use labeled data for this analysis
)
SELECT 
    -- Baseline CRC rate in the population
    ROUND(AVG(CAST(FUTURE_CRC_EVENT AS DOUBLE)) * 100, 3) as overall_crc_rate_pct,
    
    -- CRC rates for key symptoms (should be elevated)
    ROUND(AVG(CASE WHEN BLEED_FLAG_12MO = 1 
              THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100, 3) as crc_rate_bleeding_pct,
    ROUND(AVG(CASE WHEN ANEMIA_FLAG_12MO = 1 
              THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100, 3) as crc_rate_anemia_pct,
    ROUND(AVG(CASE WHEN IDA_WITH_BLEEDING = 1 
              THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100, 3) as crc_rate_ida_bleeding_pct,
    
    -- CRC rates for risk factors
    ROUND(AVG(CASE WHEN POLYPS_FLAG_EVER = 1 
              THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100, 3) as crc_rate_polyps_pct,
    ROUND(AVG(CASE WHEN IBD_FLAG_EVER = 1 
              THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100, 3) as crc_rate_ibd_pct,
    
    -- CRC rates for composite scores
    ROUND(AVG(CASE WHEN CRC_SYMPTOM_TRIAD = 1 
              THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100, 3) as crc_rate_triad_pct,
    ROUND(AVG(CASE WHEN HIGH_RISK_HISTORY = 1 
              THEN CAST(FUTURE_CRC_EVENT AS DOUBLE) END) * 100, 3) as crc_rate_high_risk_pct
    
FROM outcome_analysis
""").show(truncate=False)

print("\n✓ CRC association analysis complete - higher rates indicate predictive features")

# ========================================
# CELL 16
# ========================================

# Visualize risk elevation for key features
import matplotlib.pyplot as plt

features = ['Baseline', 'Bleeding\n(1.3%)', 'IDA+Bleeding\n(0.7%)', 
            'Symptom Triad\n(1.8%)', 'Anemia\n(6.9%)', 'Polyps\n(2.6%)']
crc_rates = [0.41, 2.56, 1.94, 1.86, 1.34, 1.41]
colors = ['gray', 'darkred', 'red', 'orange', 'gold', 'yellow']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(features, crc_rates, color=colors)

# Add risk ratio labels
for i, (bar, rate) in enumerate(zip(bars, crc_rates)):
    if i > 0:  # Skip baseline
        risk_ratio = rate / crc_rates[0]
        ax.text(rate + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{risk_ratio:.1f}× risk', va='center', fontweight='bold')

ax.axvline(x=0.41, color='black', linestyle='--', linewidth=2, label='Baseline (0.41%)')
ax.set_xlabel('CRC Rate (%)', fontsize=12)
ax.set_title('CRC Risk by Feature Presence\n(Prevalence shown in parentheses)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# ========================================
# CELL 17
# ========================================

# =========================================================================
# CELL 15 - COMORBIDITY SCORE DISTRIBUTIONS
# =========================================================================
# Analyze the distribution of comorbidity scores in the population

spark.sql(f"""
SELECT 
    -- Charlson Comorbidity Index statistics
    ROUND(AVG(CHARLSON_SCORE_12MO), 2) as avg_charlson_12mo,
    PERCENTILE_APPROX(CHARLSON_SCORE_12MO, 0.50) as median_charlson_12mo,
    PERCENTILE_APPROX(CHARLSON_SCORE_12MO, 0.95) as p95_charlson_12mo,
    MAX(CHARLSON_SCORE_12MO) as max_charlson_12mo,
    
    -- Elixhauser score statistics
    ROUND(AVG(ELIXHAUSER_SCORE_12MO), 2) as avg_elixhauser_12mo,
    PERCENTILE_APPROX(ELIXHAUSER_SCORE_12MO, 0.50) as median_elixhauser_12mo,
    PERCENTILE_APPROX(ELIXHAUSER_SCORE_12MO, 0.95) as p95_elixhauser_12mo,
    
    -- Combined comorbidity burden
    ROUND(AVG(COMBINED_COMORBIDITY_12MO), 2) as avg_combined_12mo,
    
    -- Count of high-risk patients
    SUM(CASE WHEN CHARLSON_SCORE_12MO >= 3 THEN 1 ELSE 0 END) as high_charlson_count,
    SUM(CASE WHEN ELIXHAUSER_SCORE_12MO >= 3 THEN 1 ELSE 0 END) as high_elixhauser_count
    
FROM {trgt_cat}.clncl_ds.herald_test_train_icd_10
""").show(truncate=False)

print("\n✓ Comorbidity score analysis complete")

# ========================================
# CELL 18
# ========================================

# CELL 17
df_spark = spark.sql('''select * from dev.clncl_ds.herald_test_train_icd_10''')

# Get column names (excluding ID columns)
feature_cols = [col for col in df_spark.columns 
                if col not in ['PAT_ID', 'END_DTTM']]

# Build SQL query to calculate null percentages
null_check_expressions = [
    f"AVG(CASE WHEN {col} IS NULL THEN 1.0 ELSE 0.0 END) as {col}_null_pct"
    for col in feature_cols
]

# Join all expressions with commas
sql_query = f"""
SELECT 
    {','.join(null_check_expressions)}
FROM dev.clncl_ds.herald_test_train_icd_10
"""

# Execute and display
null_percentages = spark.sql(sql_query)

# Show results (transpose for better readability)
null_percentages.show(vertical=True, truncate=False)

# ========================================
# CELL 19
# ========================================

# CELL 18
# Check for duplicate rows
spark.sql(f"""
SELECT COUNT(*) as total_rows, 
       COUNT(DISTINCT PAT_ID, END_DTTM) as unique_rows,
       COUNT(*) - COUNT(DISTINCT PAT_ID, END_DTTM) as duplicates
FROM {trgt_cat}.clncl_ds.herald_test_train_icd_10
""").show()

# ========================================
# CELL 20
# ========================================

# CELL 19 - CORRECTED
# First, create a temporary view from your Spark dataframe
df_spark.createOrReplaceTempView("icd_features")

# Calculate means for all features
means_result = spark.sql("""
SELECT
    -- Symptom flags (12-month)
    AVG(BLEED_FLAG_12MO) as bleed_flag_12mo_mean,
    AVG(PAIN_FLAG_12MO) as pain_flag_12mo_mean,
    AVG(BOWELCHG_FLAG_12MO) as bowelchg_flag_12mo_mean,
    AVG(WTLOSS_FLAG_12MO) as wtloss_flag_12mo_mean,
    AVG(FATIGUE_FLAG_12MO) as fatigue_flag_12mo_mean,
    AVG(ANEMIA_FLAG_12MO) as anemia_flag_12mo_mean,

    -- Symptom counts (12-month) 
    AVG(BLEED_CNT_12MO) as bleed_cnt_12mo_mean,
    AVG(PAIN_CNT_12MO) as pain_cnt_12mo_mean,
    AVG(BOWELCHG_CNT_12MO) as bowelchg_cnt_12mo_mean,
    AVG(WTLOSS_CNT_12MO) as wtloss_cnt_12mo_mean,
    AVG(FATIGUE_CNT_12MO) as fatigue_cnt_12mo_mean,
    AVG(ANEMIA_CNT_12MO) as anemia_cnt_12mo_mean,

    -- Symptom flags (24-month)
    AVG(BLEED_FLAG_24MO) as bleed_flag_24mo_mean,
    AVG(PAIN_FLAG_24MO) as pain_flag_24mo_mean,
    AVG(BOWELCHG_FLAG_24MO) as bowelchg_flag_24mo_mean,
    AVG(WTLOSS_FLAG_24MO) as wtloss_flag_24mo_mean,
    AVG(FATIGUE_FLAG_24MO) as fatigue_flag_24mo_mean,
    AVG(ANEMIA_FLAG_24MO) as anemia_flag_24mo_mean,

    -- Risk factors
    AVG(POLYPS_FLAG_EVER) as polyps_flag_ever_mean,
    AVG(IBD_FLAG_EVER) as ibd_flag_ever_mean,
    AVG(MALIGNANCY_FLAG_EVER) as malignancy_flag_ever_mean,
    AVG(DIABETES_FLAG_EVER) as diabetes_flag_ever_mean,
    AVG(OBESITY_FLAG_EVER) as obesity_flag_ever_mean,

    -- Family history 
    AVG(FHX_CRC_FLAG_EVER_ICD) as fhx_crc_flag_ever_mean,
    AVG(FHX_CRC_COMBINED) as fhx_crc_combined_mean,
    AVG(FHX_POLYPS_COMBINED) as fhx_polyps_combined_mean,
    AVG(FHX_FIRST_DEGREE_CRC) as fhx_first_degree_crc_mean,
    AVG(HIGH_RISK_FHX_FLAG) as high_risk_fhx_flag_mean,

    -- Other conditions 
    AVG(DIVERTICULAR_FLAG_24MO) as diverticular_flag_24mo_mean,
    AVG(CRC_SCREEN_FLAG_12MO) as crc_screen_flag_12mo_mean,
    AVG(CRC_SCREEN_FLAG_24MO) as crc_screen_flag_24mo_mean,

    -- Additional GI symptoms 
    AVG(IRON_DEF_ANEMIA_FLAG_12MO) as iron_def_anemia_flag_12mo_mean,
    AVG(CONSTIPATION_FLAG_12MO) as constipation_flag_12mo_mean,
    AVG(IBS_DIARRHEA_FLAG_24MO) as ibs_diarrhea_flag_24mo_mean,

    -- Comorbidity scores 
    AVG(CHARLSON_SCORE_12MO) as charlson_score_12mo_mean,
    AVG(CHARLSON_SCORE_24MO) as charlson_score_24mo_mean,
    AVG(ELIXHAUSER_SCORE_12MO) as elixhauser_score_12mo_mean,
    AVG(ELIXHAUSER_SCORE_24MO) as elixhauser_score_24mo_mean,

    -- Composite features 
    AVG(CRC_SYMPTOM_TRIAD) as crc_symptom_triad_mean,
    AVG(IDA_WITH_BLEEDING) as ida_with_bleeding_mean,
    AVG(SYMPTOM_BURDEN_12MO) as symptom_burden_12mo_mean,
    AVG(HIGH_RISK_HISTORY) as high_risk_history_mean,
    AVG(METABOLIC_SYNDROME) as metabolic_syndrome_mean,
    AVG(ANY_FHX_GI_MALIG) as any_fhx_gi_malig_mean,

    -- Burden scores 
    AVG(INFLAMMATORY_BURDEN) as inflammatory_burden_mean,
    AVG(GI_COMPLEXITY_SCORE) as gi_complexity_score_mean,
    AVG(COMBINED_COMORBIDITY_12MO) as combined_comorbidity_12mo_mean,

    -- Acceleration features 
    AVG(BLEED_ACCELERATION) as bleed_acceleration_mean,
    AVG(ANEMIA_ACCELERATION) as anemia_acceleration_mean,
    AVG(SYMPTOM_ACCELERATION) as symptom_acceleration_mean,
    AVG(ALL_SYMPTOMS_NEW) as all_symptoms_new_mean

FROM icd_features
""")

# Display the results
means_result.show(vertical=True, truncate=False)

# ========================================
# CELL 21
# ========================================

# CELL 21
# Step 1: Load ICD-10 diagnosis data and calculate basic statistics

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ICD-10 DIAGNOSIS FEATURE REDUCTION")
print("="*60)

# Load ICD-10 table
df_icd = spark.table("dev.clncl_ds.herald_test_train_icd_10")

# Add icd_ prefix to all columns except keys
icd_cols = [col for col in df_icd.columns if col not in ['PAT_ID', 'END_DTTM']]
for col in icd_cols:
    df_icd = df_icd.withColumnRenamed(col, f'icd_{col}' if not col.startswith('icd_') else col)

# Join with cohort to get outcome variable
df_cohort = spark.sql("""
    SELECT PAT_ID, END_DTTM, FUTURE_CRC_EVENT
    FROM dev.clncl_ds.herald_test_train_final_cohort
""")

df_spark = df_icd.join(
    df_cohort,
    on=['PAT_ID', 'END_DTTM'],
    how='inner'
)

# Cache for performance
df_spark.cache()
total_rows = df_spark.count()
baseline_crc_rate = df_spark.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"\nTotal rows: {total_rows:,}")
print(f"Baseline CRC rate: {baseline_crc_rate:.4f}")

# Calculate coverage for key diagnosis categories
bleeding_coverage = df_spark.filter(F.col('icd_BLEED_FLAG_12MO') == 1).count() / total_rows
anemia_coverage = df_spark.filter(F.col('icd_ANEMIA_FLAG_12MO') == 1).count() / total_rows
symptom_triad_coverage = df_spark.filter(F.col('icd_CRC_SYMPTOM_TRIAD') == 1).count() / total_rows
polyps_coverage = df_spark.filter(F.col('icd_POLYPS_FLAG_EVER') == 1).count() / total_rows
any_symptom_coverage = df_spark.filter(F.col('icd_SYMPTOM_BURDEN_12MO') > 0).count() / total_rows

print(f"\nCoverage rates:")
print(f"Any symptom (12mo): {any_symptom_coverage:.1%}")
print(f"Anemia (12mo): {anemia_coverage:.1%}")
print(f"Bleeding (12mo): {bleeding_coverage:.1%}")
print(f"Symptom triad: {symptom_triad_coverage:.1%}")
print(f"Polyps (ever): {polyps_coverage:.1%}")

# ========================================
# CELL 22
# ========================================

# CELL 22
# Step 2: Calculate Risk Ratios for Binary ICD-10 Features

binary_features = [col for col in df_spark.columns if '_FLAG' in col and col.startswith('icd_')]
risk_metrics = []

print(f"\nCalculating risk ratios for {len(binary_features)} binary features...")

for feat in binary_features:
    stats = df_spark.groupBy(feat).agg(
        F.count('*').alias('count'),
        F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
    ).collect()
    
    # Parse results
    stats_dict = {row[feat]: {'count': row['count'], 'crc_rate': row['crc_rate']} for row in stats}
    
    prevalence = stats_dict.get(1, {'count': 0})['count'] / total_rows if 1 in stats_dict else 0
    rate_with = stats_dict.get(1, {'crc_rate': 0})['crc_rate'] if 1 in stats_dict else 0
    rate_without = stats_dict.get(0, {'crc_rate': baseline_crc_rate})['crc_rate'] if 0 in stats_dict else baseline_crc_rate
    risk_ratio = rate_with / (rate_without + 1e-10)
    
    # Calculate impact
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
print("\nTop features by impact score:")
print(risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']].to_string())

# ========================================
# CELL 23
# ========================================

# CELL 23
# ===========================================

# Step 3: Analyze Count/Score Features and Missing Patterns

print("\nAnalyzing continuous and count features...")

# Separate features by type
count_features = [col for col in df_spark.columns if col.startswith('icd_') and 
                  ('_CNT_' in col or '_COUNT' in col or '_SCORE' in col or '_BURDEN' in col)]
recency_features = [col for col in df_spark.columns if col.startswith('icd_') and 
                    'DAYS_SINCE' in col]
categorical_features = [col for col in df_spark.columns if col.startswith('icd_') and 
                        col in ['icd_BOWEL_PATTERN']]

print(f"Feature types:")
print(f"  - Binary flags: {len(binary_features)}")
print(f"  - Count/score features: {len(count_features)}")
print(f"  - Recency features: {len(recency_features)}")
print(f"  - Categorical features: {len(categorical_features)}")

# Analyze missing patterns for recency features
missing_stats = []
for feat in recency_features:
    missing_rate = df_spark.filter(F.col(feat).isNull()).count() / total_rows
    mean_when_present = df_spark.filter(F.col(feat).isNotNull()).select(F.avg(feat)).collect()[0][0]
    
    missing_stats.append({
        'feature': feat,
        'missing_rate': missing_rate,
        'mean_days_when_present': mean_when_present if mean_when_present else None,
        'symptom': feat.replace('icd_DAYS_SINCE_LAST_', '').replace('icd_DAYS_SINCE_', '')
    })

missing_df = pd.DataFrame(missing_stats)
print(f"\nRecency features by prevalence:")
print(missing_df.sort_values('missing_rate')[['feature', 'missing_rate']].to_string())

# Analyze count feature distributions
count_stats = []
for feat in count_features:
    mean_val = df_spark.select(F.avg(feat)).collect()[0][0]
    max_val = df_spark.select(F.max(feat)).collect()[0][0]
    nonzero_pct = df_spark.filter(F.col(feat) > 0).count() / total_rows
    
    count_stats.append({
        'feature': feat,
        'mean': mean_val,
        'max': max_val,
        'pct_nonzero': nonzero_pct
    })

count_df = pd.DataFrame(count_stats)
print(f"\nCount/score features by prevalence:")
print(count_df.sort_values('pct_nonzero', ascending=False)[['feature', 'mean', 'pct_nonzero']].head(10).to_string())

# ========================================
# CELL 24
# ========================================

# CELL 24
# ===========================================

# Step 4: Calculate Mutual Information Using Stratified Sample

sample_fraction = min(200000 / total_rows, 1.0)

print(f"\nSampling for MI calculation...")
df_sample = df_spark.sampleBy("FUTURE_CRC_EVENT", 
                               fractions={0: sample_fraction, 1: 1.0},
                               seed=42).toPandas()

print(f"Sampled {len(df_sample):,} rows ({len(df_sample)/total_rows*100:.1f}% of total)")
print(f"Sample CRC rate: {df_sample['FUTURE_CRC_EVENT'].mean():.4f}")

# Calculate MI on sample
from sklearn.feature_selection import mutual_info_classif

feature_cols = [c for c in df_sample.columns 
                if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT'] 
                and c.startswith('icd_')]

# Handle categorical features
cat_mask = [c in categorical_features for c in feature_cols]

# Encode categorical features
for cat_feat in categorical_features:
    if cat_feat in df_sample.columns:
        df_sample[cat_feat] = pd.Categorical(df_sample[cat_feat]).codes

print(f"Calculating MI for features...")
X = df_sample[feature_cols].fillna(-999)
y = df_sample['FUTURE_CRC_EVENT']

mi_scores = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop features by Mutual Information:")
print(mi_df.to_string())

# ========================================
# CELL 25
# ========================================

#CELL 25
#===========================================

# Step 5: Apply Clinical Filters for ICD-10 Setting

# Merge all metrics
feature_importance = mi_df.merge(
    risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']], 
    on='feature', 
    how='left'
)

# Add count statistics
if len(count_stats) > 0:
    count_stats_df = pd.DataFrame(count_stats)
    feature_importance = feature_importance.merge(
        count_stats_df[['feature', 'pct_nonzero']], 
        on='feature', 
        how='left'
    )

# Add missing stats for recency features
if len(missing_stats) > 0:
    missing_stats_df = pd.DataFrame(missing_stats)
    feature_importance = feature_importance.merge(
        missing_stats_df[['feature', 'missing_rate']], 
        on='feature', 
        how='left'
    )

# Fill NAs
feature_importance['prevalence'] = feature_importance['prevalence'].fillna(
    feature_importance['pct_nonzero']
)
feature_importance['missing_rate'] = feature_importance['missing_rate'].fillna(0)
feature_importance['risk_ratio'] = feature_importance['risk_ratio'].fillna(1.0)
feature_importance['impact'] = feature_importance['impact'].fillna(0)

# ICD-10-specific critical features
MUST_KEEP = [
    'icd_BLEED_FLAG_12MO',           # Highest risk symptom
    'icd_IDA_WITH_BLEEDING',         # Critical pattern
    'icd_CRC_SYMPTOM_TRIAD',         # Symptom combination
    'icd_ANEMIA_FLAG_12MO',          # Common CRC symptom
    'icd_POLYPS_FLAG_EVER',          # Precursor lesion
    'icd_IBD_FLAG_EVER',             # High-risk condition
    'icd_HIGH_RISK_FHX_FLAG',        # Family history
    'icd_SYMPTOM_BURDEN_12MO',       # Overall symptom count
    'icd_IRON_DEF_ANEMIA_FLAG_12MO'  # Specific anemia type
]

# Remove features with near-zero signal or redundancy
REMOVE = []
for _, row in feature_importance.iterrows():
    feat = row['feature']
    
    # Remove recency features with >99% missing
    if 'DAYS_SINCE' in feat and row.get('missing_rate', 0) > 0.99:
        REMOVE.append(feat)
    
    # Remove 24mo versions if 12mo exists and is better
    if '_24MO' in feat:
        corresponding_12mo = feat.replace('_24MO', '_12MO')
        if corresponding_12mo in feature_importance['feature'].values:
            # Check if 12mo version has better MI score
            mi_12mo = feature_importance[feature_importance['feature'] == corresponding_12mo]['mi_score'].values
            mi_24mo = row['mi_score']
            if len(mi_12mo) > 0 and mi_12mo[0] >= mi_24mo:
                REMOVE.append(feat)
    
    # Remove very rare features with low risk
    if '_FLAG' in feat and feat not in MUST_KEEP:
        if row.get('prevalence', 0) < 0.001 and row.get('risk_ratio', 1) < 2:
            REMOVE.append(feat)

REMOVE = list(set(REMOVE))
print(f"\nRemoving {len(REMOVE)} low-signal or redundant features")
print(f"Examples of removed features: {REMOVE[:5]}")

feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]
print(f"Features remaining after filtering: {len(feature_importance)}")

# ========================================
# CELL 26
# ========================================

# CELL 26
# ===========================================

# Step 6: Select Optimal Features per Diagnosis Category

def select_optimal_icd_features(df_importance):
    """Select best representation for each diagnosis category"""
    
    selected = []
    
    # Group features by clinical category
    for _, row in df_importance.iterrows():
        feat = row['feature']
        
        # SYMPTOMS - keep flags and counts for key symptoms
        if 'BLEED' in feat and ('FLAG_12MO' in feat or 'CNT_12MO' in feat):
            selected.append(feat)
        elif 'ANEMIA' in feat and 'FLAG_12MO' in feat:
            selected.append(feat)
        elif 'IRON_DEF_ANEMIA' in feat:
            selected.append(feat)
        elif any(x in feat for x in ['PAIN_FLAG_12MO', 'BOWELCHG_FLAG_12MO', 'WTLOSS_FLAG_12MO']):
            selected.append(feat)
            
        # COMPOSITE PATTERNS - keep all
        elif any(x in feat for x in ['CRC_SYMPTOM_TRIAD', 'IDA_WITH_BLEEDING', 'SYMPTOM_BURDEN', 
                                      'HIGH_RISK_HISTORY', 'METABOLIC_SYNDROME']):
            selected.append(feat)
            
        # RISK FACTORS - keep key flags
        elif any(x in feat for x in ['POLYPS_FLAG', 'IBD_FLAG', 'MALIGNANCY_FLAG']):
            selected.append(feat)
            
        # FAMILY HISTORY - keep combined and high-risk
        elif any(x in feat for x in ['HIGH_RISK_FHX', 'FHX_CRC_COMBINED', 'FHX_FIRST_DEGREE']):
            selected.append(feat)
            
        # COMORBIDITY SCORES - keep 12-month versions
        elif any(x in feat for x in ['CHARLSON_SCORE_12MO', 'ELIXHAUSER_SCORE_12MO', 
                                      'COMBINED_COMORBIDITY_12MO']):
            selected.append(feat)
            
        # METABOLIC CONDITIONS
        elif any(x in feat for x in ['DIABETES_FLAG', 'OBESITY_FLAG']) and 'EVER' in feat:
            selected.append(feat)
    
    # Ensure must-keep features are included
    for feat in MUST_KEEP:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    # Add top MI features if we have too few
    if len(selected) < 20:
        top_mi = df_importance[~df_importance['feature'].isin(selected)].nlargest(20 - len(selected), 'mi_score')
        selected.extend(top_mi['feature'].tolist())
    
    return list(set(selected))

selected_features = select_optimal_icd_features(feature_importance)
print(f"\nSelected {len(selected_features)} features after diagnosis-category optimization")
print("\nSelected features by category:")

# Print by category for clarity
symptoms = [f for f in selected_features if any(x in f for x in ['BLEED', 'ANEMIA', 'PAIN', 'BOWEL', 'WTLOSS', 'FATIGUE'])]
composites = [f for f in selected_features if any(x in f for x in ['TRIAD', 'IDA_WITH', 'BURDEN', 'METABOLIC'])]
risk_factors = [f for f in selected_features if any(x in f for x in ['POLYPS', 'IBD', 'MALIGNANCY', 'FHX'])]
scores = [f for f in selected_features if any(x in f for x in ['CHARLSON', 'ELIXHAUSER', 'COMORBIDITY'])]

print(f"  Symptoms: {len(symptoms)}")
print(f"  Composites: {len(composites)}")
print(f"  Risk factors: {len(risk_factors)}")
print(f"  Comorbidity scores: {len(scores)}")

# ========================================
# CELL 27
# ========================================

# CELL 27
# Step 7: Create Clinical Composites and Save

df_final = df_spark

# === ICD-10-SPECIFIC COMPOSITE FEATURES ===

# 1. Severe symptom pattern (multiple concerning symptoms)
df_final = df_final.withColumn('icd_severe_symptom_pattern',
    F.when((F.col('icd_BLEED_FLAG_12MO') == 1) & 
           (F.col('icd_ANEMIA_FLAG_12MO') == 1), 1)
    .when((F.col('icd_SYMPTOM_BURDEN_12MO') >= 3), 1)
    .when((F.col('icd_WTLOSS_FLAG_12MO') == 1) & 
          (F.col('icd_ANEMIA_FLAG_12MO') == 1), 1)
    .otherwise(0)
)

# 2. Genetic risk composite (family history patterns)
df_final = df_final.withColumn('icd_genetic_risk_composite',
    F.greatest(
        F.col('icd_HIGH_RISK_FHX_FLAG'),
        F.when((F.col('icd_FHX_CRC_COMBINED') == 1) | 
               (F.col('icd_FHX_FIRST_DEGREE_CRC') == 1), 1).otherwise(0)
    )
)

# 3. Chronic GI pattern (IBD + other chronic conditions)
df_final = df_final.withColumn('icd_chronic_gi_pattern',
    F.when((F.col('icd_IBD_FLAG_EVER') == 1) | 
           (F.col('icd_DIVERTICULAR_FLAG_24MO') == 1) |
           (F.col('icd_GI_COMPLEXITY_SCORE') >= 2), 1).otherwise(0)
)

composite_features = [
    'icd_severe_symptom_pattern',
    'icd_genetic_risk_composite',
    'icd_chronic_gi_pattern'
]

# Add composites to selected features
selected_features.extend(composite_features)
selected_features = sorted(list(set(selected_features)))

print(f"\nAdded {len(composite_features)} composite features")
print(f"Final feature count: {len(selected_features)}")

# === PRINT FINAL FEATURE LIST ===
print("\n" + "="*60)
print("FINAL SELECTED ICD-10 FEATURES")
print("="*60)

for i, feat in enumerate(selected_features, 1):
    # Add description for clarity
    if 'BLEED' in feat:
        desc = " [HIGHEST RISK]"
    elif 'IDA_WITH_BLEEDING' in feat:
        desc = " [CRITICAL PATTERN]"
    elif 'ANEMIA' in feat:
        desc = " [CRC SYMPTOM]"
    elif 'TRIAD' in feat:
        desc = " [SYMPTOM COMBO]"
    elif 'POLYPS' in feat or 'IBD' in feat:
        desc = " [RISK FACTOR]"
    elif 'FHX' in feat or 'genetic' in feat:
        desc = " [FAMILY HISTORY]"
    elif feat in composite_features:
        desc = " [COMPOSITE]"
    elif 'CHARLSON' in feat or 'ELIXHAUSER' in feat:
        desc = " [COMORBIDITY]"
    else:
        desc = ""
    print(f"{i:2d}. {feat:<45} {desc}")

# === SAVE REDUCED DATASET ===
final_columns = ['PAT_ID', 'END_DTTM'] + selected_features
df_reduced = df_final.select(*final_columns)

# Write to final table
output_table = 'dev.clncl_ds.herald_test_train_icd10_reduced'
df_reduced.write.mode('overwrite').saveAsTable(output_table)

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: 116")
print(f"Selected features: {len(selected_features)}")
print(f"Reduction: {(1 - len(selected_features)/116)*100:.1f}%")
print(f"\n✓ Reduced dataset saved to: {output_table}")

# Verify save
row_count = spark.table(output_table).count()
cols_without_prefix = [c for c in selected_features if not c.startswith('icd_')]

print(f"✓ Verified {row_count:,} rows written to table")
if cols_without_prefix:
    print(f"\n⚠ WARNING: These columns missing 'icd_' prefix: {cols_without_prefix}")
else:
    print("✓ All feature columns have 'icd_' prefix for joining")

# ========================================
# CELL 28
# ========================================

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_test_train_icd10_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 29
# ========================================

display(df_check)



################################################################################
# V2_Book4_Labs_Combined
################################################################################

# V2_Book4_Labs_Combined
# Functional cells: 26 of 56 code cells (105 total)
# Source: V2_Book4_Labs_Combined.ipynb
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

# Define target catalog for SQL based on the environment variable
trgt_cat = os.environ.get('trgt_cat')

# Use appropriate Spark catalog based on the target category
spark.sql('USE CATALOG prod;')

# ========================================
# CELL 2
# ========================================

# ---------------------------------
# CELL 1A: Enhanced Inpatient Labs - Basic Labs
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_inpatient_basic_labs AS

WITH
    --------------------------------------------------------------------------
    -- 1) cohort: Our base population
    --------------------------------------------------------------------------
    cohort AS (
        SELECT PAT_ID, END_DTTM
        FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
    ),

    --------------------------------------------------------------------------
    -- 2) basic_labs: CBC, Basic Metabolic Panel, and common tests
    -- Note: Using COMPONENT_NRML_LO and COMPONENT_NRML_HI for reference ranges
    --------------------------------------------------------------------------
    basic_labs AS (
        SELECT
            ec.PAT_ID,
            ec.END_DTTM,
            res_comp.COMP_VERIF_DTTM,
            TRIM(UPPER(cc.NAME)) AS RAW_NAME,
            res_comp.COMPONENT_ABN_C AS ABNORMAL_YN,
            
            -- Map to standardized component names
            CASE
                -- Complete Blood Count Components
                WHEN TRIM(UPPER(cc.NAME)) IN (
                    'HEMOGLOBIN', 'HEMOGLOBIN POC', 'HEMOGLOBIN VENOUS',
                    'HEMOGLOBIN ABG', 'HEMOGLOBIN USED CAPILLARY'
                ) THEN 'HEMOGLOBIN'
                
                WHEN TRIM(UPPER(cc.NAME)) IN ('HEMATOCRIT', 'HEMATOCRIT POC') 
                THEN 'HCT'
                
                WHEN TRIM(UPPER(cc.NAME)) = 'MCV' THEN 'MCV'
                WHEN TRIM(UPPER(cc.NAME)) = 'MCH' THEN 'MCH'
                WHEN TRIM(UPPER(cc.NAME)) = 'MCHC' THEN 'MCHC'
                WHEN TRIM(UPPER(cc.NAME)) = 'PLATELETS' THEN 'PLATELETS'
                
                -- Metabolic Panel
                WHEN TRIM(UPPER(cc.NAME)) = 'ALBUMIN' THEN 'ALBUMIN'
                WHEN TRIM(UPPER(cc.NAME)) = 'HEMOGLOBIN A1C' THEN 'HGBA1C'
                
                -- Lipid Panel
                WHEN TRIM(UPPER(cc.NAME)) IN ('LDL CALCULATED', 'LDL CHOLESTEROL, DIRECT') 
                THEN 'LDL'
                WHEN TRIM(UPPER(cc.NAME)) = 'HDL' THEN 'HDL'
                WHEN TRIM(UPPER(cc.NAME)) IN ('TRIGLYCERIDE', 'TRIGLYCERIDES, DIRECT') 
                THEN 'TRIGLYCERIDES'
                
                ELSE NULL
            END AS COMPONENT_NAME,
            
            -- Extract numeric value, removing qualifiers
            TRY_CAST(REGEXP_REPLACE(res_comp.COMPONENT_VALUE,'[><]','') AS FLOAT) AS COMPONENT_VALUE,
            
            -- Reference range values for later normalization
            TRY_CAST(REGEXP_REPLACE(res_comp.COMPONENT_NRML_LO,'[><]','') AS FLOAT) AS REF_LOW,
            TRY_CAST(REGEXP_REPLACE(res_comp.COMPONENT_NRML_HI,'[><]','') AS FLOAT) AS REF_HIGH

        FROM clarity_cur.order_proc_enh op
        JOIN clarity.clarity_eap eap ON eap.PROC_ID = op.PROC_ID
        JOIN clarity.spec_test_rel spec 
            ON spec.SPEC_TST_ORDER_ID = op.ORDER_PROC_ID
        JOIN clarity.res_db_main rdm 
            ON rdm.RES_SPECIMEN_ID = spec.SPECIMEN_ID
            AND rdm.RES_ORDER_ID = spec.SPEC_TST_ORDER_ID
        JOIN clarity.res_components res_comp 
            ON res_comp.RESULT_ID = rdm.RESULT_ID
        JOIN clarity.clarity_component cc 
            ON cc.COMPONENT_ID = res_comp.COMPONENT_ID
        JOIN clarity.pat_enc_hsp peh 
            ON peh.PAT_ENC_CSN_ID = op.PAT_ENC_CSN_ID
        JOIN cohort ec 
            ON ec.PAT_ID = peh.PAT_ID

        WHERE
            -- Only completed, resulted orders
            op.ORDER_STATUS_C IN (3, 5, 10)
            AND op.LAB_STATUS_C IN (3, 5)
            AND rdm.RES_VAL_STATUS_C = 9
            
            -- Historical data only (no future leakage)
            AND DATE(peh.CONTACT_DATE) >= DATE '2021-07-01'
            AND res_comp.COMPONENT_VALUE IS NOT NULL
            AND res_comp.COMP_VERIF_DTTM < ec.END_DTTM
            
            -- 2-year lookback for basic labs
            AND DATE(res_comp.COMP_VERIF_DTTM) 
                BETWEEN DATE_SUB(ec.END_DTTM, 730) AND ec.END_DTTM
    )

SELECT 
    PAT_ID,
    END_DTTM,
    COMPONENT_NAME,
    COMPONENT_VALUE,
    COMP_VERIF_DTTM,
    ABNORMAL_YN,
    REF_LOW,
    REF_HIGH,
    DATEDIFF(END_DTTM, DATE(COMP_VERIF_DTTM)) AS DAYS_SINCE_LAB
FROM basic_labs
WHERE COMPONENT_NAME IS NOT NULL
    -- Apply outlier filters for data quality
    AND (
        (COMPONENT_NAME = 'HEMOGLOBIN' AND COMPONENT_VALUE BETWEEN 3 AND 20)
        OR (COMPONENT_NAME = 'HCT' AND COMPONENT_VALUE BETWEEN 10 AND 65)
        OR (COMPONENT_NAME = 'MCV' AND COMPONENT_VALUE BETWEEN 50 AND 150)
        OR (COMPONENT_NAME = 'MCH' AND COMPONENT_VALUE BETWEEN 15 AND 45)
        OR (COMPONENT_NAME = 'MCHC' AND COMPONENT_VALUE BETWEEN 25 AND 40)
        OR (COMPONENT_NAME = 'PLATELETS' AND COMPONENT_VALUE BETWEEN 10 AND 2000)
        OR (COMPONENT_NAME = 'ALBUMIN' AND COMPONENT_VALUE BETWEEN 1 AND 6)
        OR COMPONENT_NAME IN ('LDL', 'HDL', 'TRIGLYCERIDES', 'HGBA1C')
    )
''')

print("Basic inpatient labs extracted")
spark.sql(f"SELECT COUNT(*) as row_count, COUNT(DISTINCT PAT_ID) as patients FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_basic_labs").show()

# ========================================
# CELL 3
# ========================================

# ---------------------------------
# CELL 1B: Enhanced Inpatient Labs - Intermediate Table 1
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_inpatient_labs_raw AS

WITH cohort AS (
    SELECT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
)

SELECT
    ec.PAT_ID,
    ec.END_DTTM,
    res_comp.COMP_VERIF_DTTM,
    TRIM(UPPER(cc.NAME)) AS RAW_NAME,
    res_comp.COMPONENT_VALUE AS RAW_VALUE,
    res_comp.COMPONENT_UNITS AS RAW_UNITS,
    res_comp.COMPONENT_ABN_C AS ABNORMAL_YN,
    res_comp.COMPONENT_NRML_LO AS REF_LOW,    -- Fixed: using correct column name
    res_comp.COMPONENT_NRML_HI AS REF_HIGH,   -- Fixed: using correct column name
    
    -- Calculate days since for filtering
    DATEDIFF(ec.END_DTTM, DATE(res_comp.COMP_VERIF_DTTM)) AS DAYS_SINCE_LAB

FROM clarity_cur.order_proc_enh op
JOIN clarity.clarity_eap eap ON eap.PROC_ID = op.PROC_ID
JOIN clarity.spec_test_rel spec ON spec.SPEC_TST_ORDER_ID = op.ORDER_PROC_ID
JOIN clarity.res_db_main rdm 
    ON rdm.RES_SPECIMEN_ID = spec.SPECIMEN_ID
    AND rdm.RES_ORDER_ID = spec.SPEC_TST_ORDER_ID
JOIN clarity.res_components res_comp ON res_comp.RESULT_ID = rdm.RESULT_ID
JOIN clarity.clarity_component cc ON cc.COMPONENT_ID = res_comp.COMPONENT_ID
JOIN clarity.pat_enc_hsp peh ON peh.PAT_ENC_CSN_ID = op.PAT_ENC_CSN_ID
JOIN cohort ec ON ec.PAT_ID = peh.PAT_ID

WHERE op.ORDER_STATUS_C IN (3, 5, 10)
    AND op.LAB_STATUS_C IN (3, 5)
    AND rdm.RES_VAL_STATUS_C = 9
    AND DATE(peh.CONTACT_DATE) >= DATE '2021-07-01'
    AND res_comp.COMPONENT_VALUE IS NOT NULL
    AND res_comp.COMP_VERIF_DTTM < ec.END_DTTM
''')

print("Inpatient raw labs table created")
spark.sql(f"SELECT COUNT(*) as row_count FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_labs_raw").show()

# ========================================
# CELL 4
# ========================================

# ---------------------------------
# CELL 1C: Process Inpatient Labs with Normalization
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_inpatient_labs_processed AS

WITH normalized_labs AS (
    SELECT
        PAT_ID,
        END_DTTM,
        COMP_VERIF_DTTM,
        ABNORMAL_YN,
        REF_LOW,
        REF_HIGH,
        DAYS_SINCE_LAB,
        
        -- Map to standardized component names
        CASE
            -- CBC Components
            WHEN RAW_NAME IN ('HEMOGLOBIN', 'HEMOGLOBIN POC', 'HEMOGLOBIN VENOUS', 
                             'HEMOGLOBIN ABG', 'HEMOGLOBIN USED CAPILLARY') 
                THEN 'HEMOGLOBIN'
            WHEN RAW_NAME IN ('HEMATOCRIT', 'HEMATOCRIT POC') THEN 'HCT'
            WHEN RAW_NAME = 'MCV' THEN 'MCV'
            WHEN RAW_NAME = 'MCH' THEN 'MCH'
            WHEN RAW_NAME = 'MCHC' THEN 'MCHC'
            WHEN RAW_NAME = 'PLATELETS' THEN 'PLATELETS'
            
            -- Iron Studies
            WHEN RAW_NAME IN ('IRON', 'IRON, TOTAL') THEN 'IRON'
            WHEN RAW_NAME IN ('TIBC', 'IRON BINDING CAPACITY') THEN 'TIBC'
            WHEN RAW_NAME IN ('FERRITIN') THEN 'FERRITIN'
            WHEN RAW_NAME IN ('TRANSFERRIN') THEN 'TRANSFERRIN'
            WHEN RAW_NAME = 'TRANSFERRIN SATURATION' THEN 'IRON_SAT'
            
            -- Inflammatory Markers
            WHEN RAW_NAME = 'CRP' THEN 'CRP'
            WHEN RAW_NAME = 'ESR (SEDIMENTATION RATE)' THEN 'ESR'
            
            -- LFTs
            WHEN RAW_NAME = 'ALBUMIN' THEN 'ALBUMIN'
            WHEN RAW_NAME IN ('ALT') THEN 'ALT'
            WHEN RAW_NAME IN ('AST', 'AST (SGOT)') THEN 'AST'
            WHEN RAW_NAME = 'ALKALINE PHOSPHATASE' THEN 'ALK_PHOS'
            WHEN RAW_NAME = 'BILIRUBIN TOTAL' THEN 'BILI_TOTAL'
            WHEN RAW_NAME = 'BILIRUBIN DIRECT' THEN 'BILI_DIRECT'
            WHEN RAW_NAME = 'GGT' THEN 'GGT'
            WHEN RAW_NAME = 'TOTAL PROTEIN' THEN 'TOTAL_PROTEIN'
            
            -- Lipids
            WHEN RAW_NAME IN ('LDL CALCULATED', 'LDL CHOLESTEROL, DIRECT') THEN 'LDL'
            WHEN RAW_NAME = 'HDL' THEN 'HDL'
            WHEN RAW_NAME IN ('TRIGLYCERIDE', 'TRIGLYCERIDES, DIRECT') THEN 'TRIGLYCERIDES'
            
            -- Other
            WHEN RAW_NAME = 'CA 125' THEN 'CA125'
            WHEN RAW_NAME = 'LD (LACTATE DEHYDROGENASE)' THEN 'LDH'
            WHEN RAW_NAME = 'HEMOGLOBIN A1C' THEN 'HGBA1C'
            ELSE NULL
        END AS COMPONENT_NAME,
        
        -- Normalize values with unit conversion
        CASE
            -- CRP conversion from mg/dL to mg/L
            WHEN RAW_NAME = 'CRP' AND UPPER(RAW_UNITS) = 'MG/DL'
                THEN CAST(REGEXP_REPLACE(RAW_VALUE,'[><]','') AS FLOAT) * 10
            -- Handle other numeric values
            ELSE TRY_CAST(REGEXP_REPLACE(RAW_VALUE,'[><]','') AS FLOAT)
        END AS COMPONENT_VALUE
        
    FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_labs_raw
)

-- Apply lookback windows and outlier filters
SELECT *
FROM normalized_labs
WHERE COMPONENT_NAME IS NOT NULL
    AND (
        -- 3-year lookback for slow-changing markers
        (COMPONENT_NAME IN ('CA125', 'FERRITIN', 'CRP', 'ESR', 'LDH')
         AND DAYS_SINCE_LAB <= 1095)
        OR
        -- 2-year lookback for routine labs
        (COMPONENT_NAME NOT IN ('CA125', 'FERRITIN', 'CRP', 'ESR', 'LDH')
         AND DAYS_SINCE_LAB <= 730)
    )
    -- Apply outlier filtering
    AND (
        (COMPONENT_NAME = 'HEMOGLOBIN' AND COMPONENT_VALUE BETWEEN 3 AND 20)
        OR (COMPONENT_NAME = 'HCT' AND COMPONENT_VALUE BETWEEN 10 AND 65)
        OR (COMPONENT_NAME = 'MCV' AND COMPONENT_VALUE BETWEEN 50 AND 150)
        OR (COMPONENT_NAME = 'MCH' AND COMPONENT_VALUE BETWEEN 15 AND 45)
        OR (COMPONENT_NAME = 'MCHC' AND COMPONENT_VALUE BETWEEN 25 AND 40)
        OR (COMPONENT_NAME = 'PLATELETS' AND COMPONENT_VALUE BETWEEN 10 AND 2000)
        OR (COMPONENT_NAME = 'FERRITIN' AND COMPONENT_VALUE BETWEEN 0 AND 10000)
        OR (COMPONENT_NAME = 'IRON' AND COMPONENT_VALUE BETWEEN 0 AND 500)
        OR (COMPONENT_NAME = 'TIBC' AND COMPONENT_VALUE BETWEEN 100 AND 600)
        OR (COMPONENT_NAME = 'CRP' AND COMPONENT_VALUE BETWEEN 0 AND 500)
        OR (COMPONENT_NAME = 'ESR' AND COMPONENT_VALUE BETWEEN 0 AND 200)
        OR (COMPONENT_NAME = 'ALBUMIN' AND COMPONENT_VALUE BETWEEN 1 AND 6)
        OR (COMPONENT_NAME = 'ALT' AND COMPONENT_VALUE BETWEEN 0 AND 2000)
        OR (COMPONENT_NAME = 'AST' AND COMPONENT_VALUE BETWEEN 0 AND 2000)
        OR (COMPONENT_NAME = 'ALK_PHOS' AND COMPONENT_VALUE BETWEEN 0 AND 2000)
        OR (COMPONENT_NAME = 'BILI_TOTAL' AND COMPONENT_VALUE BETWEEN 0 AND 50)
        OR (COMPONENT_NAME = 'GGT' AND COMPONENT_VALUE BETWEEN 0 AND 2000)
        OR (COMPONENT_NAME = 'CA125' AND COMPONENT_VALUE BETWEEN 0 AND 50000)
        OR (COMPONENT_NAME = 'LDH' AND COMPONENT_VALUE BETWEEN 0 AND 5000)
        OR COMPONENT_NAME IN ('LDL', 'HDL', 'TRIGLYCERIDES', 'HGBA1C',
                              'TOTAL_PROTEIN', 'BILI_DIRECT', 'TRANSFERRIN', 'IRON_SAT')
    )
''')

# Validate processed data
validation_df = spark.sql(f'''
SELECT 
    COMPONENT_NAME,
    COUNT(*) as record_count,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    AVG(COMPONENT_VALUE) as avg_value,
    STDDEV(COMPONENT_VALUE) as std_value,
    MIN(COMPONENT_VALUE) as min_value,
    MAX(COMPONENT_VALUE) as max_value
FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_labs_processed
GROUP BY COMPONENT_NAME
ORDER BY record_count DESC
''')

print("Inpatient Lab Processing Summary:")
validation_df.show(50, truncate=False)

# ========================================
# CELL 5
# ========================================

# ---------------------------------
# CELL 2A: Enhanced Outpatient Labs - Raw Extraction
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_outpatient_labs_raw AS

WITH expanded_cohort AS (
    SELECT pat_id, end_dttm
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
)

SELECT
    ec.pat_id AS PAT_ID,
    ec.end_dttm AS END_DTTM,
    ord.result_time AS COMP_VERIF_DTTM,
    TRIM(UPPER(cc.name)) AS RAW_NAME,
    ord.ord_value AS RAW_VALUE,
    TRIM(UPPER(ord.reference_unit)) AS RAW_UNITS,
    ord.reference_low AS REF_LOW,
    ord.reference_high AS REF_HIGH,
    
    -- Calculate abnormal flag based on reference ranges
    CASE 
        WHEN TRY_CAST(REGEXP_REPLACE(ord.ord_value,'[><]','') AS FLOAT) IS NOT NULL
             AND ord.reference_low IS NOT NULL 
             AND ord.reference_high IS NOT NULL
        THEN 
            CASE 
                WHEN TRY_CAST(REGEXP_REPLACE(ord.ord_value,'[><]','') AS FLOAT) < ord.reference_low 
                     OR TRY_CAST(REGEXP_REPLACE(ord.ord_value,'[><]','') AS FLOAT) > ord.reference_high
                THEN 'Y'
                ELSE 'N'
            END
        ELSE NULL
    END AS ABNORMAL_YN,
    
    DATEDIFF(DATE(ec.end_dttm), DATE(ord.result_time)) AS DAYS_SINCE_LAB

FROM clarity_cur.order_proc_enh op
JOIN clarity.order_results ord ON ord.order_proc_id = op.order_proc_id
JOIN clarity.clarity_component cc ON cc.component_id = ord.component_id
JOIN expanded_cohort ec ON ec.pat_id = op.pat_id

WHERE op.order_status_c IN (3,5,10)
    AND op.lab_status_c IN (3,5)
    AND COALESCE(ord.ord_value,'-1') <> '-1'
    AND DATE(ord.result_time) < DATE(ec.end_dttm)
    AND DATE(ord.result_time) >= DATE '2021-07-01'
''')

print("Outpatient raw labs table created")
spark.sql(f"SELECT COUNT(*) as row_count FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_labs_raw").show()

# ========================================
# CELL 6
# ========================================

# ---------------------------------  
# CELL 2B: Process Outpatient Labs with Normalization
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_outpatient_labs_processed AS

WITH normalized_labs AS (
    SELECT
        PAT_ID,
        END_DTTM,
        COMP_VERIF_DTTM,
        ABNORMAL_YN,
        REF_LOW,
        REF_HIGH,
        DAYS_SINCE_LAB,
        
        -- Component name mapping (same as inpatient plus outpatient variations)
        CASE
            -- CBC
            WHEN RAW_NAME IN ('HEMOGLOBIN', 'HEMOGLOBIN POC', 'HEMOGLOBIN ABG',
                             'HEMOGLOBIN VENOUS', 'HEMOGLOBIN G/DL', 'HEMOGLOBIN GM/DL')
                THEN 'HEMOGLOBIN'
            WHEN RAW_NAME IN ('HEMATOCRIT', 'HEMATOCRIT POC', 'HEMATOCRIT ABG', 'HEMATOCRIT, FLD')
                THEN 'HCT'
            WHEN RAW_NAME IN ('MCV', 'MCV FL', 'MCV POC') THEN 'MCV'
            WHEN RAW_NAME IN ('MCH', 'MCH PG', 'MCH POC') THEN 'MCH'
            WHEN RAW_NAME IN ('MCHC', 'MCHC POC') THEN 'MCHC'
            WHEN RAW_NAME LIKE 'PLATELET%' THEN 'PLATELETS'
            
            -- Iron Studies
            WHEN RAW_NAME IN ('IRON', 'IRON, TOTAL') THEN 'IRON'
            WHEN RAW_NAME IN ('TIBC', 'IRON BINDING CAPACITY', 'TOTAL IRON BINDING CAPACITY') THEN 'TIBC'
            WHEN RAW_NAME IN ('FERRITIN', 'FERRITIN NG/ML') THEN 'FERRITIN'
            WHEN RAW_NAME = 'TRANSFERRIN' THEN 'TRANSFERRIN'
            WHEN RAW_NAME = 'TRANSFERRIN SATURATION' THEN 'IRON_SAT'
            
            -- Inflammatory
            WHEN RAW_NAME = 'CRP' THEN 'CRP'
            WHEN RAW_NAME IN ('ESR (SEDIMENTATION RATE)', 'ERYTHROCYTE SEDIMENTATION RATE POC')
                THEN 'ESR'
            
            -- LFTs
            WHEN RAW_NAME IN ('ALBUMIN', 'ALBUMIN (REF LAB)', 'ALBUMIN POC') THEN 'ALBUMIN'
            WHEN RAW_NAME IN ('ALT', 'ALT POC', 'ALT (SGPT)') THEN 'ALT'
            WHEN RAW_NAME IN ('AST', 'AST (SGOT)', 'SGOT') THEN 'AST'
            WHEN RAW_NAME IN ('ALKALINE PHOSPHATASE', 'ALK PHOS', 'ALP') THEN 'ALK_PHOS'
            WHEN RAW_NAME IN ('BILIRUBIN TOTAL', 'BILIRUBIN') THEN 'BILI_TOTAL'
            WHEN RAW_NAME = 'BILIRUBIN DIRECT' THEN 'BILI_DIRECT'
            WHEN RAW_NAME IN ('GGT', 'GAMMA GT', 'GGTP') THEN 'GGT'
            WHEN RAW_NAME IN ('TOTAL PROTEIN', 'PROTEIN, TOTAL') THEN 'TOTAL_PROTEIN'
            
            -- Other
            WHEN RAW_NAME IN ('CA 125', 'CA125', 'CANCER ANTIGEN 125') THEN 'CA125'
            WHEN RAW_NAME = 'LD (LACTATE DEHYDROGENASE)' THEN 'LDH'
            WHEN RAW_NAME IN ('HEMOGLOBIN A1C', 'HGB A1C POC')
                 AND RAW_UNITS IN ('% OF TOTAL HGB','%','%HB','%NGSP','% NGSP')
                THEN 'HGBA1C'
            ELSE NULL
        END AS COMPONENT_NAME,
        
        -- Value normalization
        CASE
            -- CRP conversion
            WHEN RAW_NAME = 'CRP' AND RAW_UNITS = 'MG/DL'
                THEN TRY_CAST(REGEXP_REPLACE(RAW_VALUE,'[><]','') AS FLOAT) * 10
            -- Standard numeric processing
            ELSE TRY_CAST(REGEXP_REPLACE(RAW_VALUE,'[><]','') AS FLOAT)
        END AS COMPONENT_VALUE
        
    FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_labs_raw
)

-- Apply filters and outlier bounds
SELECT *
FROM normalized_labs
WHERE COMPONENT_NAME IS NOT NULL
    AND COMPONENT_VALUE IS NOT NULL
    AND (
        -- Lookback windows
        (COMPONENT_NAME IN ('CA125', 'FERRITIN', 'CRP', 'ESR', 'LDH')
         AND DAYS_SINCE_LAB <= 1095)
        OR
        (COMPONENT_NAME NOT IN ('CA125', 'FERRITIN', 'CRP', 'ESR', 'LDH')
         AND DAYS_SINCE_LAB <= 730)
    )
    -- Apply outlier filters - COMPLETE LIST with bounds for ALL components
    AND (
        (COMPONENT_NAME = 'HEMOGLOBIN' AND COMPONENT_VALUE BETWEEN 3 AND 20)
        OR (COMPONENT_NAME = 'HCT' AND COMPONENT_VALUE BETWEEN 10 AND 65)
        OR (COMPONENT_NAME = 'MCV' AND COMPONENT_VALUE BETWEEN 50 AND 150)
        OR (COMPONENT_NAME = 'MCH' AND COMPONENT_VALUE BETWEEN 15 AND 45)
        OR (COMPONENT_NAME = 'MCHC' AND COMPONENT_VALUE BETWEEN 25 AND 40)
        OR (COMPONENT_NAME = 'PLATELETS' AND COMPONENT_VALUE BETWEEN 10 AND 2000)
        OR (COMPONENT_NAME = 'IRON' AND COMPONENT_VALUE BETWEEN 0 AND 500)
        OR (COMPONENT_NAME = 'TIBC' AND COMPONENT_VALUE BETWEEN 100 AND 600)
        OR (COMPONENT_NAME = 'FERRITIN' AND COMPONENT_VALUE BETWEEN 0 AND 10000)
        OR (COMPONENT_NAME = 'TRANSFERRIN' AND COMPONENT_VALUE BETWEEN 10 AND 600)
        OR (COMPONENT_NAME = 'IRON_SAT' AND COMPONENT_VALUE BETWEEN 0 AND 100)
        OR (COMPONENT_NAME = 'CRP' AND COMPONENT_VALUE BETWEEN 0 AND 500)
        OR (COMPONENT_NAME = 'ESR' AND COMPONENT_VALUE BETWEEN 0 AND 200)
        OR (COMPONENT_NAME = 'ALBUMIN' AND COMPONENT_VALUE BETWEEN 1 AND 6)
        OR (COMPONENT_NAME = 'ALT' AND COMPONENT_VALUE BETWEEN 0 AND 2000)
        OR (COMPONENT_NAME = 'AST' AND COMPONENT_VALUE BETWEEN 0 AND 2000)
        OR (COMPONENT_NAME = 'ALK_PHOS' AND COMPONENT_VALUE BETWEEN 0 AND 2000)
        OR (COMPONENT_NAME = 'BILI_TOTAL' AND COMPONENT_VALUE BETWEEN 0 AND 50)
        OR (COMPONENT_NAME = 'BILI_DIRECT' AND COMPONENT_VALUE BETWEEN 0 AND 50)
        OR (COMPONENT_NAME = 'GGT' AND COMPONENT_VALUE BETWEEN 0 AND 2000)
        OR (COMPONENT_NAME = 'TOTAL_PROTEIN' AND COMPONENT_VALUE BETWEEN 0 AND 20)
        OR (COMPONENT_NAME = 'CA125' AND COMPONENT_VALUE BETWEEN 0 AND 50000)
        OR (COMPONENT_NAME = 'LDH' AND COMPONENT_VALUE BETWEEN 0 AND 5000)
        OR (COMPONENT_NAME = 'HGBA1C' AND COMPONENT_VALUE BETWEEN 2 AND 25)
        OR (COMPONENT_NAME = 'LDL' AND COMPONENT_VALUE BETWEEN 0 AND 1000)
        OR (COMPONENT_NAME = 'HDL' AND COMPONENT_VALUE BETWEEN 0 AND 200)
        OR (COMPONENT_NAME = 'TRIGLYCERIDES' AND COMPONENT_VALUE BETWEEN 0 AND 5000)
    )
''')

# Validation
spark.sql(f'''
SELECT 
    COMPONENT_NAME,
    COUNT(*) as count,
    AVG(COMPONENT_VALUE) as avg_val,
    MIN(COMPONENT_VALUE) as min_val,
    MAX(COMPONENT_VALUE) as max_val
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_labs_processed
GROUP BY COMPONENT_NAME
ORDER BY count DESC
''').show(30, truncate=False)

# ========================================
# CELL 7
# ========================================

# ---------------------------------
# CELL 3: Combine Inpatient and Outpatient Labs
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_combined_labs_all AS

SELECT
    PAT_ID,
    END_DTTM,
    COMPONENT_NAME,
    COMPONENT_VALUE,
    COMP_VERIF_DTTM,
    ABNORMAL_YN,
    REF_LOW,
    REF_HIGH,
    DAYS_SINCE_LAB,
    'INPATIENT' AS SOURCE
FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_labs_processed

UNION ALL

SELECT
    PAT_ID,
    END_DTTM,
    COMPONENT_NAME,
    COMPONENT_VALUE,
    COMP_VERIF_DTTM,
    ABNORMAL_YN,
    REF_LOW,
    REF_HIGH,
    DAYS_SINCE_LAB,
    'OUTPATIENT' AS SOURCE
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_labs_processed
''')

# Quality check
spark.sql(f'''
SELECT 
    SOURCE,
    COUNT(*) as total_records,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    COUNT(DISTINCT COMPONENT_NAME) as unique_labs,
    SUM(CASE WHEN ABNORMAL_YN = 'Y' THEN 1 ELSE 0 END) as abnormal_count
FROM {trgt_cat}.clncl_ds.herald_test_train_combined_labs_all
GROUP BY SOURCE
''').show()

# ========================================
# CELL 8
# ========================================

# ---------------------------------
# CELL 4A: Calculate Iron Saturation and Anemia Classification (FIXED)
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_labs_anemia_features AS

WITH iron_calculations AS (
    -- Get iron and TIBC values on same day to calculate iron saturation
    -- Add ROW_NUMBER to handle multiple tests on same day
    SELECT
        i.PAT_ID,
        i.END_DTTM,
        i.COMPONENT_VALUE AS IRON_VALUE,
        t.COMPONENT_VALUE AS TIBC_VALUE,
        (i.COMPONENT_VALUE / t.COMPONENT_VALUE) * 100 AS CALC_IRON_SAT,
        i.DAYS_SINCE_LAB AS IRON_DAYS,
        ROW_NUMBER() OVER (
            PARTITION BY i.PAT_ID, i.END_DTTM
            ORDER BY i.COMP_VERIF_DTTM DESC, t.COMP_VERIF_DTTM DESC
        ) AS rn_iron
    FROM {trgt_cat}.clncl_ds.herald_test_train_combined_labs_all i
    JOIN {trgt_cat}.clncl_ds.herald_test_train_combined_labs_all t
        ON i.PAT_ID = t.PAT_ID 
        AND i.END_DTTM = t.END_DTTM
        AND DATE(i.COMP_VERIF_DTTM) = DATE(t.COMP_VERIF_DTTM)
    WHERE i.COMPONENT_NAME = 'IRON' 
        AND t.COMPONENT_NAME = 'TIBC'
        AND t.COMPONENT_VALUE > 0
),

iron_calculations_dedup AS (
    -- Keep only most recent iron/TIBC calculation per patient-date
    SELECT * FROM iron_calculations WHERE rn_iron = 1
),

hemoglobin_latest AS (
    -- Get most recent hemoglobin for anemia classification
    SELECT
        PAT_ID,
        END_DTTM,
        FIRST_VALUE(COMPONENT_VALUE) OVER (
            PARTITION BY PAT_ID, END_DTTM 
            ORDER BY COMP_VERIF_DTTM DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS HGB_VALUE,
        FIRST_VALUE(DAYS_SINCE_LAB) OVER (
            PARTITION BY PAT_ID, END_DTTM 
            ORDER BY COMP_VERIF_DTTM DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS HGB_DAYS,
        ROW_NUMBER() OVER (
            PARTITION BY PAT_ID, END_DTTM 
            ORDER BY COMP_VERIF_DTTM DESC
        ) AS rn
    FROM {trgt_cat}.clncl_ds.herald_test_train_combined_labs_all
    WHERE COMPONENT_NAME = 'HEMOGLOBIN'
),

mcv_latest AS (
    -- Get most recent MCV for microcytic classification
    SELECT
        PAT_ID,
        END_DTTM,
        FIRST_VALUE(COMPONENT_VALUE) OVER (
            PARTITION BY PAT_ID, END_DTTM 
            ORDER BY COMP_VERIF_DTTM DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS MCV_VALUE,
        ROW_NUMBER() OVER (
            PARTITION BY PAT_ID, END_DTTM 
            ORDER BY COMP_VERIF_DTTM DESC
        ) AS rn
    FROM {trgt_cat}.clncl_ds.herald_test_train_combined_labs_all
    WHERE COMPONENT_NAME = 'MCV'
),

ferritin_latest AS (
    -- Get most recent ferritin
    SELECT
        PAT_ID,
        END_DTTM,
        FIRST_VALUE(COMPONENT_VALUE) OVER (
            PARTITION BY PAT_ID, END_DTTM 
            ORDER BY COMP_VERIF_DTTM DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS FERRITIN_VALUE,
        ROW_NUMBER() OVER (
            PARTITION BY PAT_ID, END_DTTM 
            ORDER BY COMP_VERIF_DTTM DESC
        ) AS rn
    FROM {trgt_cat}.clncl_ds.herald_test_train_combined_labs_all
    WHERE COMPONENT_NAME = 'FERRITIN'
),

-- Get cohort to ensure all patients are included
cohort AS (
    SELECT DISTINCT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
)

SELECT DISTINCT  -- Add DISTINCT as final safety
    c.PAT_ID,
    c.END_DTTM,
    
    -- Iron saturation (calculated or direct)
    ic.IRON_VALUE,
    ic.TIBC_VALUE,
    ic.CALC_IRON_SAT AS IRON_SATURATION_PCT,
    
    -- Anemia classification (WHO grades)
    h.HGB_VALUE,
    CASE
        WHEN h.HGB_VALUE IS NULL THEN NULL
        WHEN h.HGB_VALUE >= 12 THEN 'NORMAL'  -- Using female cutoff as conservative
        WHEN h.HGB_VALUE >= 11 THEN 'MILD_ANEMIA'
        WHEN h.HGB_VALUE >= 8 THEN 'MODERATE_ANEMIA'
        ELSE 'SEVERE_ANEMIA'
    END AS ANEMIA_GRADE,
    
    -- Iron deficiency anemia pattern (low MCV + low ferritin + low iron sat)
    CASE
        WHEN h.HGB_VALUE < 12 
             AND m.MCV_VALUE < 80 
             AND (f.FERRITIN_VALUE < 30 OR ic.CALC_IRON_SAT < 20)
        THEN 1 ELSE 0
    END AS IRON_DEFICIENCY_ANEMIA_FLAG,
    
    -- Anemia of chronic disease pattern (normal/high ferritin + low iron)
    CASE
        WHEN h.HGB_VALUE < 12 
             AND f.FERRITIN_VALUE >= 30 
             AND ic.CALC_IRON_SAT < 20
        THEN 1 ELSE 0
    END AS CHRONIC_DISEASE_ANEMIA_FLAG,
    
    -- Microcytic anemia flag
    CASE
        WHEN h.HGB_VALUE < 12 AND m.MCV_VALUE < 80 THEN 1 ELSE 0
    END AS MICROCYTIC_ANEMIA_FLAG,
    
    -- Iron deficiency without anemia (early stage)
    CASE
        WHEN h.HGB_VALUE >= 12 
             AND (f.FERRITIN_VALUE < 30 OR ic.CALC_IRON_SAT < 20)
        THEN 1 ELSE 0
    END AS IRON_DEFICIENCY_NO_ANEMIA_FLAG

FROM cohort c  -- Start from cohort
LEFT JOIN hemoglobin_latest h 
    ON c.PAT_ID = h.PAT_ID AND c.END_DTTM = h.END_DTTM AND h.rn = 1
LEFT JOIN iron_calculations_dedup ic  -- Use deduplicated iron calculations
    ON c.PAT_ID = ic.PAT_ID AND c.END_DTTM = ic.END_DTTM
LEFT JOIN mcv_latest m 
    ON c.PAT_ID = m.PAT_ID AND c.END_DTTM = m.END_DTTM AND m.rn = 1
LEFT JOIN ferritin_latest f 
    ON c.PAT_ID = f.PAT_ID AND c.END_DTTM = f.END_DTTM AND f.rn = 1
''')

# ========================================
# CELL 9
# ========================================

# ---------------------------------
# CELL 5: Create Enhanced Pivoted Lab Values with Lipids
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_labs_pivoted_enhanced AS

WITH latest_labs AS (
    -- Get most recent value for each lab per patient-month
    SELECT
        PAT_ID,
        END_DTTM,
        COMPONENT_NAME,
        FIRST_VALUE(COMPONENT_VALUE) OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME 
            ORDER BY COMP_VERIF_DTTM DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS recent_value,
        FIRST_VALUE(DAYS_SINCE_LAB) OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME 
            ORDER BY COMP_VERIF_DTTM DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS recent_days,
        FIRST_VALUE(ABNORMAL_YN) OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME 
            ORDER BY COMP_VERIF_DTTM DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS recent_abnormal,
        ROW_NUMBER() OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME 
            ORDER BY COMP_VERIF_DTTM DESC
        ) AS rn
    FROM {trgt_cat}.clncl_ds.herald_test_train_combined_labs_all
    WHERE COMPONENT_NAME IS NOT NULL
)

SELECT
    PAT_ID,
    END_DTTM,
    
    -- CBC values
    SUM(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' THEN recent_value END) AS HEMOGLOBIN_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'HCT' THEN recent_value END) AS HCT_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'MCV' THEN recent_value END) AS MCV_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'MCH' THEN recent_value END) AS MCH_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'MCHC' THEN recent_value END) AS MCHC_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'PLATELETS' THEN recent_value END) AS PLATELETS_VALUE,
    
    -- Iron studies
    SUM(CASE WHEN COMPONENT_NAME = 'IRON' THEN recent_value END) AS IRON_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'TIBC' THEN recent_value END) AS TIBC_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'FERRITIN' THEN recent_value END) AS FERRITIN_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'TRANSFERRIN' THEN recent_value END) AS TRANSFERRIN_VALUE,
    
    -- Inflammatory markers
    SUM(CASE WHEN COMPONENT_NAME = 'CRP' THEN recent_value END) AS CRP_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'ESR' THEN recent_value END) AS ESR_VALUE,
    
    -- LFTs
    SUM(CASE WHEN COMPONENT_NAME = 'ALBUMIN' THEN recent_value END) AS ALBUMIN_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'ALT' THEN recent_value END) AS ALT_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'AST' THEN recent_value END) AS AST_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'ALK_PHOS' THEN recent_value END) AS ALK_PHOS_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'BILI_TOTAL' THEN recent_value END) AS BILI_TOTAL_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'BILI_DIRECT' THEN recent_value END) AS BILI_DIRECT_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'GGT' THEN recent_value END) AS GGT_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'TOTAL_PROTEIN' THEN recent_value END) AS TOTAL_PROTEIN_VALUE,
    
    -- Lipids (ADDED)
    SUM(CASE WHEN COMPONENT_NAME = 'LDL' THEN recent_value END) AS LDL_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'HDL' THEN recent_value END) AS HDL_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'TRIGLYCERIDES' THEN recent_value END) AS TRIGLYCERIDES_VALUE,
    
    -- Other markers
    SUM(CASE WHEN COMPONENT_NAME = 'CA125' THEN recent_value END) AS CA125_VALUE,

    -- Other
    SUM(CASE WHEN COMPONENT_NAME = 'LDH' THEN recent_value END) AS LDH_VALUE,
    SUM(CASE WHEN COMPONENT_NAME = 'HGBA1C' THEN recent_value END) AS HGBA1C_VALUE,
    
    -- Days since measurements (selected key ones)
    SUM(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' THEN recent_days END) AS HEMOGLOBIN_DAYS,
    SUM(CASE WHEN COMPONENT_NAME = 'FERRITIN' THEN recent_days END) AS FERRITIN_DAYS,
    SUM(CASE WHEN COMPONENT_NAME = 'PLATELETS' THEN recent_days END) AS PLATELETS_DAYS,
    
    -- Abnormal flags (selected key ones)
    SUM(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' AND recent_abnormal = 'Y' THEN 1 ELSE 0 END) AS HEMOGLOBIN_ABNORMAL,
    SUM(CASE WHEN COMPONENT_NAME = 'ALT' AND recent_abnormal = 'Y' THEN 1 ELSE 0 END) AS ALT_ABNORMAL,
    SUM(CASE WHEN COMPONENT_NAME = 'AST' AND recent_abnormal = 'Y' THEN 1 ELSE 0 END) AS AST_ABNORMAL,
    SUM(CASE WHEN COMPONENT_NAME = 'ALK_PHOS' AND recent_abnormal = 'Y' THEN 1 ELSE 0 END) AS ALK_PHOS_ABNORMAL,
    SUM(CASE WHEN COMPONENT_NAME = 'PLATELETS' AND recent_abnormal = 'Y' THEN 1 ELSE 0 END) AS PLATELETS_ABNORMAL
    
FROM latest_labs
WHERE rn = 1
GROUP BY PAT_ID, END_DTTM
''')

print("Enhanced pivoted labs table created with lipids")

# ========================================
# CELL 10
# ========================================

# ---------------------------------
# CELL 7: Enhanced Trends with Platelet Patterns and Velocity Acceleration
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_labs_trends_enhanced AS

WITH lab_values_over_time AS (
    -- Get lab values at different time points for trend calculation
    SELECT
        PAT_ID,
        END_DTTM,
        COMPONENT_NAME,
        COMPONENT_VALUE,
        DAYS_SINCE_LAB,
        COMP_VERIF_DTTM,
        
        -- Current value (most recent)
        FIRST_VALUE(COMPONENT_VALUE) OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME
            ORDER BY COMP_VERIF_DTTM DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS current_value,
        
        -- Value 1 month ago (NEW for acceleration)
        FIRST_VALUE(
            CASE 
                WHEN DAYS_SINCE_LAB BETWEEN 25 AND 35 
                THEN COMPONENT_VALUE 
            END
        ) OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME
            ORDER BY ABS(DAYS_SINCE_LAB - 30)
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS value_1mo_prior,
        
        -- Value 3 months ago
        FIRST_VALUE(
            CASE 
                WHEN DAYS_SINCE_LAB BETWEEN 75 AND 105 
                THEN COMPONENT_VALUE 
            END
        ) OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME
            ORDER BY ABS(DAYS_SINCE_LAB - 90)
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS value_3mo_prior,
        
        -- Value 6 months ago
        FIRST_VALUE(
            CASE 
                WHEN DAYS_SINCE_LAB BETWEEN 150 AND 210 
                THEN COMPONENT_VALUE 
            END
        ) OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME
            ORDER BY ABS(DAYS_SINCE_LAB - 180)
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS value_6mo_prior,
        
        -- Value 9 months ago (NEW for acceleration)
        FIRST_VALUE(
            CASE 
                WHEN DAYS_SINCE_LAB BETWEEN 255 AND 285 
                THEN COMPONENT_VALUE 
            END
        ) OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME
            ORDER BY ABS(DAYS_SINCE_LAB - 270)
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS value_9mo_prior,
        
        -- Value 12 months ago
        FIRST_VALUE(
            CASE 
                WHEN DAYS_SINCE_LAB BETWEEN 330 AND 390 
                THEN COMPONENT_VALUE 
            END
        ) OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME
            ORDER BY ABS(DAYS_SINCE_LAB - 365)
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS value_12mo_prior,
        
        ROW_NUMBER() OVER (
            PARTITION BY PAT_ID, END_DTTM, COMPONENT_NAME
            ORDER BY COMP_VERIF_DTTM DESC
        ) AS rn
        
    FROM {trgt_cat}.clncl_ds.herald_test_train_combined_labs_all
    WHERE COMPONENT_NAME IN (
        'HEMOGLOBIN', 'HCT', 'FERRITIN', 'ALBUMIN',
        'PLATELETS', 'CRP', 'AST', 'ALT'
    )
),

platelet_max_calc AS (
    -- Get highest platelet count in past 12 months
    SELECT
        PAT_ID,
        END_DTTM,
        FIRST_VALUE(COMPONENT_VALUE) OVER (
            PARTITION BY PAT_ID, END_DTTM
            ORDER BY COMPONENT_VALUE DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS PLATELETS_MAX_12MO,
        ROW_NUMBER() OVER (
            PARTITION BY PAT_ID, END_DTTM
            ORDER BY COMPONENT_VALUE DESC
        ) AS rn
    FROM {trgt_cat}.clncl_ds.herald_test_train_combined_labs_all
    WHERE COMPONENT_NAME = 'PLATELETS' 
        AND DAYS_SINCE_LAB <= 365
)

SELECT
    lv.PAT_ID,
    lv.END_DTTM,
    
    -- EXISTING TRENDS: 6-month changes
    SUM(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' THEN current_value - value_6mo_prior END) AS HEMOGLOBIN_6MO_CHANGE,
    SUM(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' AND value_6mo_prior > 0 
        THEN (current_value - value_6mo_prior) / value_6mo_prior * 100 END) AS HEMOGLOBIN_6MO_PCT_CHANGE,
    SUM(CASE WHEN COMPONENT_NAME = 'HCT' THEN current_value - value_6mo_prior END) AS HCT_6MO_CHANGE,
    SUM(CASE WHEN COMPONENT_NAME = 'FERRITIN' THEN current_value - value_6mo_prior END) AS FERRITIN_6MO_CHANGE,
    SUM(CASE WHEN COMPONENT_NAME = 'ALBUMIN' THEN current_value - value_6mo_prior END) AS ALBUMIN_6MO_CHANGE,
    SUM(CASE WHEN COMPONENT_NAME = 'PLATELETS' THEN current_value - value_6mo_prior END) AS PLATELETS_6MO_CHANGE,
    SUM(CASE WHEN COMPONENT_NAME = 'CRP' THEN current_value - value_6mo_prior END) AS CRP_6MO_CHANGE,
    
    -- EXISTING: 3-month changes
    SUM(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' THEN current_value - value_3mo_prior END) AS HEMOGLOBIN_3MO_CHANGE,
    SUM(CASE WHEN COMPONENT_NAME = 'PLATELETS' THEN current_value - value_3mo_prior END) AS PLATELETS_3MO_CHANGE,
    
    -- EXISTING: 12-month changes
    SUM(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' THEN current_value - value_12mo_prior END) AS HEMOGLOBIN_12MO_CHANGE,
    
    -- EXISTING: Rate of change per month (velocity)
    SUM(CASE 
        WHEN COMPONENT_NAME = 'HEMOGLOBIN' AND value_3mo_prior IS NOT NULL
        THEN (current_value - value_3mo_prior) / 3.0
        END) AS HEMOGLOBIN_VELOCITY_PER_MONTH,
        
    SUM(CASE 
        WHEN COMPONENT_NAME = 'PLATELETS' AND value_3mo_prior IS NOT NULL
        THEN (current_value - value_3mo_prior) / 3.0
        END) AS PLATELETS_VELOCITY_PER_MONTH,
    
    -- NEW: VELOCITY ACCELERATION PATTERNS
    -- Hemoglobin acceleration (change in velocity)
    SUM(CASE 
        WHEN COMPONENT_NAME = 'HEMOGLOBIN' 
             AND value_1mo_prior IS NOT NULL 
             AND value_3mo_prior IS NOT NULL 
             AND value_6mo_prior IS NOT NULL
        THEN 
            -- Recent velocity (0-3 months) minus prior velocity (3-6 months)
            ((current_value - value_3mo_prior) / 3.0) - 
            ((value_3mo_prior - value_6mo_prior) / 3.0)
        END) AS HEMOGLOBIN_ACCELERATION,
    
    -- Platelets acceleration
    SUM(CASE 
        WHEN COMPONENT_NAME = 'PLATELETS' 
             AND value_1mo_prior IS NOT NULL 
             AND value_3mo_prior IS NOT NULL 
             AND value_6mo_prior IS NOT NULL
        THEN 
            ((current_value - value_3mo_prior) / 3.0) - 
            ((value_3mo_prior - value_6mo_prior) / 3.0)
        END) AS PLATELETS_ACCELERATION,
    
    -- CRP acceleration
    SUM(CASE 
        WHEN COMPONENT_NAME = 'CRP' 
             AND value_3mo_prior IS NOT NULL 
             AND value_6mo_prior IS NOT NULL 
             AND value_9mo_prior IS NOT NULL
        THEN 
            ((current_value - value_3mo_prior) / 3.0) - 
            ((value_3mo_prior - value_6mo_prior) / 3.0)
        END) AS CRP_ACCELERATION,
    
    -- Ferritin acceleration
    SUM(CASE 
        WHEN COMPONENT_NAME = 'FERRITIN' 
             AND value_3mo_prior IS NOT NULL 
             AND value_6mo_prior IS NOT NULL 
             AND value_9mo_prior IS NOT NULL
        THEN 
            ((current_value - value_3mo_prior) / 3.0) - 
            ((value_3mo_prior - value_6mo_prior) / 3.0)
        END) AS FERRITIN_ACCELERATION,
    
    -- NEW: ACCELERATION FLAGS (rapid worsening patterns)
    -- Hemoglobin accelerating decline
    SUM(CASE 
        WHEN COMPONENT_NAME = 'HEMOGLOBIN'
             AND value_1mo_prior IS NOT NULL 
             AND value_3mo_prior IS NOT NULL 
             AND value_6mo_prior IS NOT NULL
             AND ((current_value - value_3mo_prior) / 3.0) < -0.5  -- Recent velocity < -0.5 g/dL per month
             AND ((current_value - value_3mo_prior) / 3.0) < ((value_3mo_prior - value_6mo_prior) / 3.0)  -- And accelerating
        THEN 1 ELSE 0
        END) AS HEMOGLOBIN_ACCELERATING_DECLINE,
    
    -- Platelets accelerating rise (thrombocytosis progression)
    SUM(CASE 
        WHEN COMPONENT_NAME = 'PLATELETS'
             AND value_3mo_prior IS NOT NULL 
             AND value_6mo_prior IS NOT NULL
             AND current_value > 450  -- Currently elevated
             AND ((current_value - value_3mo_prior) / 3.0) > ((value_3mo_prior - value_6mo_prior) / 3.0)  -- Accelerating
        THEN 1 ELSE 0
        END) AS PLATELETS_ACCELERATING_RISE,
    
    -- CRP accelerating rise (worsening inflammation)
    SUM(CASE 
        WHEN COMPONENT_NAME = 'CRP'
             AND value_3mo_prior IS NOT NULL 
             AND value_6mo_prior IS NOT NULL
             AND current_value > 10  -- Currently elevated
             AND ((current_value - value_3mo_prior) / 3.0) > ((value_3mo_prior - value_6mo_prior) / 3.0)  -- Accelerating
        THEN 1 ELSE 0
        END) AS CRP_ACCELERATING_RISE,
    
    -- NEW: VOLATILITY MEASURES (instability patterns)
    -- Hemoglobin volatility (standard deviation proxy using range)
    SUM(CASE 
        WHEN COMPONENT_NAME = 'HEMOGLOBIN' 
             AND value_3mo_prior IS NOT NULL 
             AND value_6mo_prior IS NOT NULL
             AND value_12mo_prior IS NOT NULL
        THEN 
            GREATEST(current_value, value_3mo_prior, value_6mo_prior, value_12mo_prior) -
            LEAST(current_value, value_3mo_prior, value_6mo_prior, value_12mo_prior)
        END) AS HEMOGLOBIN_VOLATILITY,
    
    -- EXISTING: Significant drop flags
    SUM(CASE WHEN COMPONENT_NAME = 'HEMOGLOBIN' AND value_6mo_prior > 0 
        AND current_value < value_6mo_prior * 0.9 THEN 1 ELSE 0 END) AS HEMOGLOBIN_DROP_10PCT_FLAG,
    SUM(CASE WHEN COMPONENT_NAME = 'ALBUMIN' AND value_6mo_prior > 0 
        AND current_value < value_6mo_prior * 0.85 THEN 1 ELSE 0 END) AS ALBUMIN_DROP_15PCT_FLAG,
    
    -- EXISTING: Platelet pattern flags
    pm.PLATELETS_MAX_12MO,
    
    -- EXISTING: Thrombocytosis flag
    SUM(CASE WHEN COMPONENT_NAME = 'PLATELETS' AND current_value > 450 THEN 1 ELSE 0 END) AS THROMBOCYTOSIS_FLAG,
    
    -- EXISTING: Rising platelets pattern
    SUM(CASE 
        WHEN COMPONENT_NAME = 'PLATELETS' 
             AND current_value > value_3mo_prior 
             AND value_3mo_prior > value_6mo_prior
             AND value_6mo_prior IS NOT NULL
        THEN 1 ELSE 0 
    END) AS PLATELETS_RISING_PATTERN_FLAG,
    
    -- EXISTING: AST/ALT for De Ritis ratio
    SUM(CASE WHEN COMPONENT_NAME = 'AST' THEN current_value END) AS AST_CURRENT,
    SUM(CASE WHEN COMPONENT_NAME = 'ALT' THEN current_value END) AS ALT_CURRENT

FROM lab_values_over_time lv
LEFT JOIN platelet_max_calc pm 
    ON lv.PAT_ID = pm.PAT_ID 
    AND lv.END_DTTM = pm.END_DTTM 
    AND pm.rn = 1
WHERE lv.rn = 1
GROUP BY lv.PAT_ID, lv.END_DTTM, pm.PLATELETS_MAX_12MO
''')

print("Enhanced trends table created with velocity acceleration patterns")

# Validate new acceleration features
spark.sql(f'''
SELECT
    'Acceleration Features' as category,
    SUM(CASE WHEN HEMOGLOBIN_ACCELERATION IS NOT NULL THEN 1 ELSE 0 END) as hgb_accel_coverage,
    AVG(HEMOGLOBIN_ACCELERATION) as avg_hgb_acceleration,
    SUM(HEMOGLOBIN_ACCELERATING_DECLINE) as hgb_accel_decline_count,
    SUM(PLATELETS_ACCELERATING_RISE) as platelets_accel_rise_count,
    SUM(CRP_ACCELERATING_RISE) as crp_accel_rise_count,
    AVG(HEMOGLOBIN_VOLATILITY) as avg_hgb_volatility
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_trends_enhanced
''').show()

# ========================================
# CELL 11
# ========================================

# ---------------------------------
# CELL 8: Final Combined Lab Features with All Enhancements
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_labs_final AS

WITH cohort AS (
    SELECT PAT_ID, END_DTTM, FUTURE_CRC_EVENT
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
)

SELECT
    c.PAT_ID,
    c.END_DTTM,
    c.FUTURE_CRC_EVENT,
    
    -- Basic lab values from pivoted table
    lp.HEMOGLOBIN_VALUE,
    lp.HCT_VALUE,
    lp.MCV_VALUE,
    lp.MCH_VALUE,
    lp.MCHC_VALUE,
    lp.PLATELETS_VALUE,
    lp.IRON_VALUE,
    lp.TIBC_VALUE,
    lp.FERRITIN_VALUE,
    lp.TRANSFERRIN_VALUE,
    lp.CRP_VALUE,
    lp.ESR_VALUE,
    lp.ALBUMIN_VALUE,
    lp.ALT_VALUE,
    lp.AST_VALUE,
    lp.ALK_PHOS_VALUE,
    lp.BILI_TOTAL_VALUE,
    lp.BILI_DIRECT_VALUE,
    lp.GGT_VALUE,
    lp.TOTAL_PROTEIN_VALUE,
    lp.CA125_VALUE,
    lp.LDH_VALUE,
    lp.HGBA1C_VALUE,
    
    -- Lipids
    lp.LDL_VALUE,
    lp.HDL_VALUE,
    lp.TRIGLYCERIDES_VALUE,
    
    -- Days since labs
    lp.HEMOGLOBIN_DAYS,
    lp.PLATELETS_DAYS,
    
    -- Abnormal flags
    lp.HEMOGLOBIN_ABNORMAL,
    lp.PLATELETS_ABNORMAL,
    lp.AST_ABNORMAL,
    lp.ALT_ABNORMAL,
    
    -- Anemia features
    af.ANEMIA_GRADE,
    af.IRON_SATURATION_PCT,
    af.IRON_DEFICIENCY_ANEMIA_FLAG,
    af.CHRONIC_DISEASE_ANEMIA_FLAG,
    af.MICROCYTIC_ANEMIA_FLAG,
    af.IRON_DEFICIENCY_NO_ANEMIA_FLAG,
    
    -- Enhanced lab trends
    lt.HEMOGLOBIN_6MO_CHANGE,
    lt.HEMOGLOBIN_6MO_PCT_CHANGE,
    lt.HEMOGLOBIN_DROP_10PCT_FLAG,
    lt.HEMOGLOBIN_3MO_CHANGE,
    lt.HEMOGLOBIN_12MO_CHANGE,
    lt.HEMOGLOBIN_VELOCITY_PER_MONTH,
    lt.HCT_6MO_CHANGE,
    lt.FERRITIN_6MO_CHANGE,
    lt.ALBUMIN_6MO_CHANGE,
    lt.ALBUMIN_DROP_15PCT_FLAG,
    lt.PLATELETS_6MO_CHANGE,
    lt.PLATELETS_3MO_CHANGE,
    lt.PLATELETS_VELOCITY_PER_MONTH,
    lt.PLATELETS_MAX_12MO,
    lt.THROMBOCYTOSIS_FLAG,
    lt.PLATELETS_RISING_PATTERN_FLAG,
    lt.CRP_6MO_CHANGE,
    
    -- NEW: Acceleration features from enhanced trends
    lt.HEMOGLOBIN_ACCELERATION,
    lt.PLATELETS_ACCELERATION,
    lt.CRP_ACCELERATION,
    lt.FERRITIN_ACCELERATION,
    lt.HEMOGLOBIN_ACCELERATING_DECLINE,
    lt.PLATELETS_ACCELERATING_RISE,
    lt.CRP_ACCELERATING_RISE,
    lt.HEMOGLOBIN_VOLATILITY,
    
    -- Calculated ratios
    CASE 
        WHEN lp.AST_VALUE > 0 
        THEN lp.ALT_VALUE / lp.AST_VALUE 
    END AS ALT_AST_RATIO,
    
    -- De Ritis Ratio (AST/ALT)
    CASE 
        WHEN lt.ALT_CURRENT > 0 
        THEN lt.AST_CURRENT / lt.ALT_CURRENT 
    END AS DE_RITIS_RATIO,
    
    CASE 
        WHEN lp.BILI_TOTAL_VALUE > 0 
        THEN lp.BILI_DIRECT_VALUE / lp.BILI_TOTAL_VALUE 
    END AS BILI_DIRECT_RATIO,
    
    -- Non-HDL Cholesterol
    CASE 
        WHEN lp.HDL_VALUE IS NOT NULL AND lp.LDL_VALUE IS NOT NULL AND lp.TRIGLYCERIDES_VALUE IS NOT NULL
        THEN lp.LDL_VALUE + (lp.TRIGLYCERIDES_VALUE / 5.0)
    END AS NON_HDL_CHOLESTEROL,
    
    -- Triglycerides/HDL Ratio (insulin resistance marker)
    CASE 
        WHEN lp.HDL_VALUE > 0 
        THEN lp.TRIGLYCERIDES_VALUE / lp.HDL_VALUE 
    END AS TG_HDL_RATIO,
    
    -- Hemoglobin trajectory classification
    CASE 
        WHEN lt.HEMOGLOBIN_12MO_CHANGE < -2 THEN 'RAPID_DECLINE'
        WHEN lt.HEMOGLOBIN_12MO_CHANGE < -1 THEN 'MODERATE_DECLINE'
        WHEN lt.HEMOGLOBIN_12MO_CHANGE < 0 THEN 'MILD_DECLINE'
        ELSE 'STABLE_OR_RISING'
    END AS HGB_TRAJECTORY,
    
    -- Combined anemia severity score (0-6 scale)
    (CASE 
        WHEN af.ANEMIA_GRADE = 'SEVERE_ANEMIA' THEN 3
        WHEN af.ANEMIA_GRADE = 'MODERATE_ANEMIA' THEN 2
        WHEN af.ANEMIA_GRADE = 'MILD_ANEMIA' THEN 1
        ELSE 0 
    END) +
    (CASE WHEN af.IRON_DEFICIENCY_ANEMIA_FLAG = 1 THEN 2 ELSE 0 END) +
    (CASE WHEN lp.MCV_VALUE < 80 THEN 1 ELSE 0 END) AS ANEMIA_SEVERITY_SCORE

FROM cohort c
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_labs_pivoted_enhanced lp
    ON c.PAT_ID = lp.PAT_ID AND c.END_DTTM = lp.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_labs_anemia_features af
    ON c.PAT_ID = af.PAT_ID AND c.END_DTTM = af.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_labs_trends_enhanced lt
    ON c.PAT_ID = lt.PAT_ID AND c.END_DTTM = lt.END_DTTM
''')

print("Final lab features table created with all enhancements")

# Extended validation
final_stats = spark.sql(f'''
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT PAT_ID, END_DTTM) as unique_keys,
    COUNT(*) - COUNT(DISTINCT PAT_ID, END_DTTM) as duplicates,
    
    -- Basic coverage
    SUM(CASE WHEN HEMOGLOBIN_VALUE IS NOT NULL THEN 1 ELSE 0 END) as has_hemoglobin,

    -- New acceleration features coverage
    SUM(CASE WHEN HEMOGLOBIN_ACCELERATION IS NOT NULL THEN 1 ELSE 0 END) as has_hgb_acceleration,
    SUM(CASE WHEN HEMOGLOBIN_ACCELERATING_DECLINE = 1 THEN 1 ELSE 0 END) as hgb_accel_decline,
    SUM(CASE WHEN PLATELETS_ACCELERATING_RISE = 1 THEN 1 ELSE 0 END) as platelets_accel_rise,
    
    -- Trajectory and severity features
    SUM(CASE WHEN HGB_TRAJECTORY = 'RAPID_DECLINE' THEN 1 ELSE 0 END) as rapid_decline,
    SUM(CASE WHEN ANEMIA_SEVERITY_SCORE >= 4 THEN 1 ELSE 0 END) as severe_anemia_combo
    
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final
''')
print("\nValidation Results:")
final_stats.show()

# ========================================
# CELL 12
# ========================================

# ---------------------------------
# CELL 9: CRC Association Analysis for Key Lab Features
# ---------------------------------

# Analyze associations with CRC outcome
association_query = f'''
SELECT
    'SEVERE_ANEMIA' as feature,
    SUM(CASE WHEN ANEMIA_GRADE = 'SEVERE_ANEMIA' THEN 1 ELSE 0 END) as feature_present,
    SUM(CASE WHEN ANEMIA_GRADE = 'SEVERE_ANEMIA' AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN ANEMIA_GRADE = 'SEVERE_ANEMIA' THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN ANEMIA_GRADE != 'SEVERE_ANEMIA' OR ANEMIA_GRADE IS NULL THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final

UNION ALL

SELECT
    'IRON_DEFICIENCY_ANEMIA' as feature,
    SUM(IRON_DEFICIENCY_ANEMIA_FLAG) as feature_present,
    SUM(CASE WHEN IRON_DEFICIENCY_ANEMIA_FLAG = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN IRON_DEFICIENCY_ANEMIA_FLAG = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN IRON_DEFICIENCY_ANEMIA_FLAG = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final

UNION ALL

SELECT
    'HEMOGLOBIN_DROP_10PCT' as feature,
    SUM(HEMOGLOBIN_DROP_10PCT_FLAG) as feature_present,
    SUM(CASE WHEN HEMOGLOBIN_DROP_10PCT_FLAG = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN HEMOGLOBIN_DROP_10PCT_FLAG = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN HEMOGLOBIN_DROP_10PCT_FLAG = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final

UNION ALL

SELECT
    'ALBUMIN_DROP_15PCT' as feature,
    SUM(ALBUMIN_DROP_15PCT_FLAG) as feature_present,
    SUM(CASE WHEN ALBUMIN_DROP_15PCT_FLAG = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN ALBUMIN_DROP_15PCT_FLAG = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN ALBUMIN_DROP_15PCT_FLAG = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final
'''

associations_df = spark.sql(association_query)

# Calculate risk ratios - FIXED: Use actual total_rows
total_rows = spark.table(f"{trgt_cat}.clncl_ds.herald_test_train_labs_final").count()
associations_pd = associations_df.toPandas()
associations_pd['risk_ratio'] = associations_pd['crc_rate_with_feature'] / associations_pd['crc_rate_without_feature']
associations_pd['coverage_pct'] = (associations_pd['feature_present'] / total_rows) * 100  

print("\n========== CRC ASSOCIATION ANALYSIS ==========")
print(associations_pd.sort_values('risk_ratio', ascending=False).to_string())

# ========================================
# CELL 13
# ========================================

# ---------------------------------
# CELL 10: Comprehensive Data Quality Report
# ---------------------------------

print("=" * 80)
print("HERALD LABS FEATURE ENGINEERING - DATA QUALITY REPORT")
print("=" * 80)

# 1. Overall coverage
coverage_stats = spark.sql(f'''
SELECT
    COUNT(*) as total_rows,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    SUM(CASE WHEN HEMOGLOBIN_VALUE IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 100 as hemoglobin_coverage_pct,
    SUM(CASE WHEN IRON_VALUE IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 100 as iron_coverage_pct,
    SUM(CASE WHEN ALT_VALUE IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 100 as alt_coverage_pct
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final
''').toPandas()

print("\n1. LAB COVERAGE STATISTICS")
print("-" * 40)
for col in coverage_stats.columns:
    print(f"{col}: {coverage_stats[col].values[0]:,.2f}")

# 2. Anemia distribution
print("\n2. ANEMIA CLASSIFICATION DISTRIBUTION")
print("-" * 40)
spark.sql(f'''
SELECT 
    ANEMIA_GRADE,
    COUNT(*) as count,
    COUNT(*) / (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final WHERE ANEMIA_GRADE IS NOT NULL) * 100 as pct
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final
WHERE ANEMIA_GRADE IS NOT NULL
GROUP BY ANEMIA_GRADE
ORDER BY count DESC
''').show()

# 3. High-risk combinations
print("\n3. HIGH-RISK LAB COMBINATIONS")
print("-" * 40)
spark.sql(f'''
SELECT
    CASE
        WHEN IRON_DEFICIENCY_ANEMIA_FLAG = 1 AND HEMOGLOBIN_DROP_10PCT_FLAG = 1
        THEN 'Iron Def Anemia + Hgb Drop'
        WHEN ANEMIA_GRADE IN ('MODERATE_ANEMIA', 'SEVERE_ANEMIA') AND HEMOGLOBIN_DROP_10PCT_FLAG = 1
        THEN 'Moderate/Severe Anemia + Hgb Drop'
        ELSE 'Other'
    END as risk_pattern,
    COUNT(*) as count,
    AVG(FUTURE_CRC_EVENT) * 100 as crc_rate_pct,
    AVG(FUTURE_CRC_EVENT) / 0.062 as risk_multiplier
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final
GROUP BY 1
HAVING risk_pattern != 'Other'
ORDER BY crc_rate_pct DESC
''').show()

print("\n" + "=" * 80)
print("SUMMARY OF USABLE LAB FEATURES FOR MODEL:")
print("=" * 80)
print("""
HIGH-VALUE FEATURES (Strong CRC Association):
- IRON_DEFICIENCY_ANEMIA_FLAG, ANEMIA_GRADE
- HEMOGLOBIN_DROP_10PCT_FLAG, HEMOGLOBIN_6MO_CHANGE

MODERATE-VALUE FEATURES:
- Iron studies: IRON_VALUE, TIBC_VALUE, IRON_SATURATION_PCT
- LFTs: ALT_VALUE, AST_VALUE, ALK_PHOS_VALUE, BILI_TOTAL_VALUE
- Inflammatory: CRP_VALUE, ESR_VALUE
- Nutritional: ALBUMIN_VALUE, ALBUMIN_DROP_15PCT_FLAG

ADDITIONAL MARKERS:
- CA125_VALUE (limited coverage but potentially valuable)

DATA QUALITY NOTES:
- Good coverage for CBC and basic metabolic labs (~60-70%)
- Iron studies have lower coverage but critical for anemia classification
""")

# ========================================
# CELL 14
# ========================================

# ---------------------------------
# CELL 11: Enhanced CRC Association Analysis (Including Acceleration)
# ---------------------------------

association_query = f'''
-- Add acceleration features to association analysis
SELECT 'HGB_ACCEL_DECLINE' as feature,
    SUM(HEMOGLOBIN_ACCELERATING_DECLINE) as feature_present,
    SUM(CASE WHEN HEMOGLOBIN_ACCELERATING_DECLINE = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN HEMOGLOBIN_ACCELERATING_DECLINE = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN HEMOGLOBIN_ACCELERATING_DECLINE = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final

UNION ALL

SELECT 'PLATELETS_ACCEL_RISE' as feature,
    SUM(PLATELETS_ACCELERATING_RISE) as feature_present,
    SUM(CASE WHEN PLATELETS_ACCELERATING_RISE = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN PLATELETS_ACCELERATING_RISE = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN PLATELETS_ACCELERATING_RISE = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final

UNION ALL

SELECT 'CRP_ACCEL_RISE' as feature,
    SUM(CRP_ACCELERATING_RISE) as feature_present,
    SUM(CASE WHEN CRP_ACCELERATING_RISE = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN CRP_ACCELERATING_RISE = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN CRP_ACCELERATING_RISE = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final

UNION ALL

SELECT 'HGB_RAPID_TRAJECTORY' as feature,
    SUM(CASE WHEN HGB_TRAJECTORY = 'RAPID_DECLINE' THEN 1 ELSE 0 END) as feature_present,
    SUM(CASE WHEN HGB_TRAJECTORY = 'RAPID_DECLINE' AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN HGB_TRAJECTORY = 'RAPID_DECLINE' THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN HGB_TRAJECTORY != 'RAPID_DECLINE' OR HGB_TRAJECTORY IS NULL THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final

UNION ALL

SELECT 'SEVERE_ANEMIA_COMBO' as feature,
    SUM(CASE WHEN ANEMIA_SEVERITY_SCORE >= 4 THEN 1 ELSE 0 END) as feature_present,
    SUM(CASE WHEN ANEMIA_SEVERITY_SCORE >= 4 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN ANEMIA_SEVERITY_SCORE >= 4 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN ANEMIA_SEVERITY_SCORE < 4 OR ANEMIA_SEVERITY_SCORE IS NULL THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_test_train_labs_final
'''

# Run existing associations plus new ones
all_associations = spark.sql(association_query)
associations_pd = all_associations.toPandas()
associations_pd['risk_ratio'] = associations_pd['crc_rate_with_feature'] / associations_pd['crc_rate_without_feature']
associations_pd['coverage_pct'] = (associations_pd['feature_present'] / total_rows) * 100

print("\n========== ENHANCED CRC ASSOCIATION ANALYSIS WITH ACCELERATION ==========")
print(associations_pd.sort_values('risk_ratio', ascending=False).to_string())

# ========================================
# CELL 15
# ========================================

# CELL 12
df = spark.sql('''select * from dev.clncl_ds.herald_test_train_labs_final''')
df.count()

# ========================================
# CELL 16
# ========================================

# CELL 13
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

df = spark.table("dev.clncl_ds.herald_test_train_final_cohort")

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

# CELL 14
# Step 1: Load laboratory data and calculate basic statistics

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("LABORATORY FEATURE REDUCTION")
print("="*60)

# Load labs table first and drop its FUTURE_CRC_EVENT column
df_labs = spark.table("dev.clncl_ds.herald_test_train_labs_final")
df_labs = df_labs.drop("FUTURE_CRC_EVENT")

# Load cohort with FUTURE_CRC_EVENT and SPLIT column
df_cohort = spark.sql("""
    SELECT PAT_ID, END_DTTM, FUTURE_CRC_EVENT, SPLIT
    FROM dev.clncl_ds.herald_test_train_final_cohort
""")

# Join the tables
df_spark = df_labs.join(
    df_cohort,
    on=['PAT_ID', 'END_DTTM'],
    how='inner'
)

# Rename columns to add lab_ prefix (excluding SPLIT)
lab_cols = [col for col in df_spark.columns if col not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT', 'SPLIT']]
for col in lab_cols:
    df_spark = df_spark.withColumnRenamed(col, f'lab_{col}' if not col.startswith('lab_') else col)

# Cache for performance
df_spark.cache()
total_rows = df_spark.count()

# =============================================================================
# CRITICAL: Filter to TRAINING DATA ONLY for feature selection metrics
# This prevents data leakage from validation/test sets into feature selection
# =============================================================================
df_train = df_spark.filter(F.col("SPLIT") == "train")
df_train.cache()

total_train_rows = df_train.count()
baseline_crc_rate = df_train.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"\nTotal rows (full cohort): {total_rows:,}")
print(f"Training rows (for feature selection): {total_train_rows:,}")
print(f"Baseline CRC rate (train only): {baseline_crc_rate:.4f}")

# Calculate coverage for key lab categories
hemoglobin_coverage = df_spark.filter(F.col('lab_HEMOGLOBIN_VALUE').isNotNull()).count() / total_rows
iron_coverage = df_spark.filter(F.col('lab_IRON_VALUE').isNotNull()).count() / total_rows
acceleration_coverage = df_spark.filter(F.col('lab_HEMOGLOBIN_ACCELERATION').isNotNull()).count() / total_rows

print(f"Hemoglobin coverage: {hemoglobin_coverage:.1%}")
print(f"Iron studies coverage: {iron_coverage:.1%}")
print(f"Acceleration features coverage: {acceleration_coverage:.2%}")

# ========================================
# CELL 18
# ========================================

# CELL 15
#===============================================
# Step 2: Calculate Risk Ratios for Binary Lab Features (TRAIN DATA ONLY)

binary_features = [col for col in df_spark.columns if '_FLAG' in col and col.startswith('lab_')]
risk_metrics = []

print(f"\nCalculating risk ratios for {len(binary_features)} binary features (using training data only)...")

for feat in binary_features:
    # NOTE: Using df_train to prevent data leakage
    stats = df_train.groupBy(feat).agg(
        F.count('*').alias('count'),
        F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
    ).collect()

    # Parse results - handle None values
    stats_dict = {row[feat]: {'count': row['count'], 'crc_rate': row['crc_rate']} for row in stats if row[feat] is not None}

    prevalence = stats_dict.get(1, {'count': 0})['count'] / total_train_rows if 1 in stats_dict else 0
    rate_with = stats_dict.get(1, {'crc_rate': 0})['crc_rate'] if 1 in stats_dict else 0
    rate_without = stats_dict.get(0, {'crc_rate': baseline_crc_rate})['crc_rate'] if 0 in stats_dict else baseline_crc_rate
    risk_ratio = rate_with / (rate_without + 1e-10)
    
    # Calculate impact
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
print("\nBinary features by impact score:")
print(risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']].to_string())

# ========================================
# CELL 19
# ========================================

# CELL 16
#===============================================
# Step 3: Analyze Continuous Features and Missing Patterns

print("\nAnalyzing continuous features and missing patterns...")

# Separate features by type
continuous_features = [col for col in df_spark.columns if col.startswith('lab_') and 
                       '_VALUE' in col or '_CHANGE' in col or '_VELOCITY' in col or 
                       '_ACCELERATION' in col or '_PCT' in col or '_RATIO' in col]
categorical_features = [col for col in df_spark.columns if col.startswith('lab_') and 
                        col in ['lab_ANEMIA_GRADE', 'lab_HGB_TRAJECTORY']]

print(f"Feature types:")
print(f"  - Binary flags: {len(binary_features)}")
print(f"  - Continuous features: {len(continuous_features)}")
print(f"  - Categorical features: {len(categorical_features)}")

# Analyze missing patterns
missing_stats = []
for feat in continuous_features:
    missing_rate = df_spark.filter(F.col(feat).isNull()).count() / total_rows
    
    # Get mean for non-null values
    mean_val = df_spark.select(F.avg(feat)).collect()[0][0]
    
    missing_stats.append({
        'feature': feat,
        'missing_rate': missing_rate,
        'mean_value': mean_val,
        'lab_type': feat.replace('lab_', '').split('_')[0]
    })

missing_df = pd.DataFrame(missing_stats)
print(f"\nFeatures with highest coverage (least missing):")
print(missing_df.nsmallest(10, 'missing_rate')[['feature', 'missing_rate']].to_string())

# ========================================
# CELL 20
# ========================================

# CELL 17
#===============================================
# Step 4: Calculate Mutual Information Using Stratified Sample (TRAIN DATA ONLY)

# Take stratified sample for MI calculation
# NOTE: Using df_train to prevent data leakage from val/test into feature selection
sample_fraction = min(200000 / total_train_rows, 1.0)  # Larger sample for labs

print(f"\nSampling for MI calculation (using training data only)...")
df_sample = df_train.sampleBy("FUTURE_CRC_EVENT",
                               fractions={0: sample_fraction, 1: 1.0},  # Keep all positive cases
                               seed=42).toPandas()

print(f"Sampled {len(df_sample):,} rows from TRAINING data ({len(df_sample)/total_train_rows*100:.1f}% of train)")
print(f"Sample CRC rate: {df_sample['FUTURE_CRC_EVENT'].mean():.4f}")

# Calculate MI on sample
from sklearn.feature_selection import mutual_info_classif

feature_cols = [c for c in df_sample.columns 
                if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT'] 
                and c.startswith('lab_')]

# Handle categorical features differently
cat_mask = [c in categorical_features for c in feature_cols]

# Encode categorical features
for cat_feat in categorical_features:
    if cat_feat in df_sample.columns:
        df_sample[cat_feat] = pd.Categorical(df_sample[cat_feat]).codes

print(f"Calculating MI for {len(feature_cols)} features...")
X = df_sample[feature_cols].fillna(-999)
y = df_sample['FUTURE_CRC_EVENT']

mi_scores = mutual_info_classif(X, y, discrete_features=cat_mask, n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop features by Mutual Information:")
print(mi_df.to_string())

# ========================================
# CELL 21
# ========================================

# CELL 18
# Step 5: Apply Clinical Filters for Laboratory Setting

# Merge all metrics
feature_importance = mi_df.merge(
    risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']], 
    on='feature', 
    how='left'
).merge(
    missing_df[['feature', 'missing_rate']], 
    on='feature', 
    how='left'
)

# Fill NAs
feature_importance['prevalence'] = feature_importance['prevalence'].fillna(
    1 - feature_importance['missing_rate']
)
feature_importance['risk_ratio'] = feature_importance['risk_ratio'].fillna(1.0)
feature_importance['impact'] = feature_importance['impact'].fillna(0)

# Lab-specific critical features
MUST_KEEP = [
    'lab_HEMOGLOBIN_ACCELERATING_DECLINE',  # Extreme risk despite low coverage
    'lab_PLATELETS_ACCELERATING_RISE',      # Paraneoplastic syndrome marker
    'lab_IRON_DEFICIENCY_ANEMIA_FLAG',      # Classic CRC pattern
    'lab_ANEMIA_SEVERITY_SCORE',            # Composite severity
    'lab_HEMOGLOBIN_VALUE',                 # Core biomarker
    'lab_HEMOGLOBIN_DROP_10PCT_FLAG',       # Significant change
    'lab_ALBUMIN_DROP_15PCT_FLAG'           # Nutritional/chronic disease
]

# Pre-specified removals for near-zero or redundant features
REMOVE = []

# Add features with extremely low signal
for _, row in feature_importance.iterrows():
    feat = row['feature']
    
    # More lenient for labs due to selective ordering
    if row.get('missing_rate', 0) > 0.998 and feat not in MUST_KEEP:
        # Keep acceleration features despite high missingness if risk is extreme
        if 'ACCELERATION' not in feat and 'ACCELERATING' not in feat:
            REMOVE.append(feat)
    
    # Remove low-impact continuous features with high missingness
    if '_VALUE' in feat or '_CHANGE' in feat:
        if (row.get('missing_rate', 0) > 0.95 and 
            row.get('mi_score', 0) < 0.001 and 
            feat not in MUST_KEEP):
            REMOVE.append(feat)

# Remove highly correlated redundant features (keep the one with better coverage)
redundant_pairs = [
    ('lab_HCT_VALUE', 'lab_HCT_6MO_CHANGE'),  # Keep value, remove change
    ('lab_MCH_VALUE', 'lab_MCHC_VALUE'),       # Keep MCH, remove MCHC
    ('lab_BILI_TOTAL_VALUE', 'lab_BILI_DIRECT_VALUE'),  # Keep total
]

for remove_feat, keep_feat in redundant_pairs:
    if remove_feat not in MUST_KEEP:
        REMOVE.append(remove_feat)

REMOVE = list(set(REMOVE))  # Remove duplicates
print(f"\nRemoving {len(REMOVE)} low-signal or redundant features")
print(f"Examples of removed features: {REMOVE[:5]}")

feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]
print(f"Features remaining after filtering: {len(feature_importance)}")

# ========================================
# CELL 22
# ========================================

# CELL 19
# Step 6: Select Optimal Features per Lab Type - BALANCED APPROACH

def select_optimal_lab_features(df_importance):
    """Select best representation for each laboratory test with balanced criteria"""
    
    selected = []
    
    # Extract lab type (first part after lab_ prefix)
    df_importance['lab_type'] = df_importance['feature'].str.replace('lab_', '').str.split('_').str[0]
    
    # Group by lab type and select intelligently
    for lab in df_importance['lab_type'].unique():
        lab_features = df_importance[df_importance['lab_type'] == lab]
        
        # Balanced selection rules - not too strict, not too loose
        if lab in ['HEMOGLOBIN', 'HGB']:
            # Keep value and top 2 risk indicators
            value_feat = [f for f in lab_features['feature'] if '_VALUE' in f]
            if value_feat:
                selected.append(value_feat[0])
            
            # Get best change/risk features
            risk_feat = lab_features[lab_features['feature'].str.contains('DROP|ACCELERATING|TRAJECTORY|6MO_CHANGE')]
            if not risk_feat.empty:
                top_risk = risk_feat.nlargest(2, 'mi_score')['feature'].tolist()
                selected.extend(top_risk)
                    
        elif lab == 'PLATELETS':
            # Keep value and best flag
            value_feat = lab_features[lab_features['feature'].str.contains('_VALUE')]
            if not value_feat.empty:
                selected.append(value_feat['feature'].values[0])
            flag_feat = lab_features[lab_features['feature'].str.contains('THROMBOCYTOSIS|ACCELERATING')]
            if not flag_feat.empty:
                selected.append(flag_feat.nlargest(1, 'mi_score')['feature'].values[0])
                    
        elif lab == 'CA125':
            # Keep top CA125 feature if available
            if len(lab_features) > 0:
                top_markers = lab_features.nlargest(min(1, len(lab_features)), 'mi_score')['feature'].tolist()
                selected.extend(top_markers)

        elif lab == 'ANEMIA':
            # Keep both score and flags
            anemia_features = lab_features.nlargest(min(2, len(lab_features)), 'mi_score')['feature'].tolist()
            selected.extend(anemia_features)
                
        elif lab == 'ALBUMIN':
            # Keep value and drop flag
            for feat in lab_features['feature']:
                if 'DROP' in feat or '_VALUE' in feat:
                    selected.append(feat)
                    
        elif lab in ['IRON', 'FERRITIN', 'CRP', 'ESR', 'ALT', 'AST', 'ALK']:
            # Keep best feature from each important lab
            if len(lab_features) > 0:
                best_feature = lab_features.nlargest(1, 'mi_score')['feature'].values[0]
                selected.append(best_feature)
    
    # Ensure critical features are included
    CRITICAL_FEATURES = [
        'lab_HEMOGLOBIN_ACCELERATING_DECLINE',
        'lab_PLATELETS_ACCELERATING_RISE',
        'lab_IRON_DEFICIENCY_ANEMIA_FLAG',
        'lab_ANEMIA_SEVERITY_SCORE',
        'lab_HEMOGLOBIN_VALUE',
        'lab_HEMOGLOBIN_DROP_10PCT_FLAG',
        'lab_ALBUMIN_DROP_15PCT_FLAG',
        'lab_THROMBOCYTOSIS_FLAG',
        'lab_PLATELETS_VALUE'
    ]
    
    for feat in CRITICAL_FEATURES:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    return list(set(selected))

selected_features = select_optimal_lab_features(feature_importance)

# Less aggressive filtering - only remove if BOTH high missing AND low impact
final_selected = []
for feat in selected_features:
    feat_data = feature_importance[feature_importance['feature'] == feat]
    if not feat_data.empty:
        missing_rate = feat_data['missing_rate'].values[0] if 'missing_rate' in feat_data.columns else 0
        risk_ratio = feat_data['risk_ratio'].values[0] if 'risk_ratio' in feat_data.columns else 1
        mi_score = feat_data['mi_score'].values[0] if 'mi_score' in feat_data.columns else 0
        
        # Only remove if very poor on ALL metrics
        if missing_rate > 0.99 and risk_ratio < 2 and mi_score < 0.0001:
            print(f"  Removing {feat}: missing={missing_rate:.3f}, risk={risk_ratio:.2f}, MI={mi_score:.5f}")
        else:
            final_selected.append(feat)

selected_features = final_selected
print(f"\nSelected {len(selected_features)} features after lab-type optimization")
print("\nFinal selected features:")
for i, feat in enumerate(sorted(selected_features), 1):
    print(f"{i:2d}. {feat}")

# ========================================
# CELL 23
# ========================================

# CELL 20
# Step 7: Create Clinical Composite Features and Save

df_final = df_spark

# === LAB-SPECIFIC COMPOSITE FEATURES ===

# 1. Iron deficiency pattern (comprehensive)
df_final = df_final.withColumn('lab_comprehensive_iron_deficiency',
    F.when((F.col('lab_IRON_DEFICIENCY_ANEMIA_FLAG') == 1) | 
           ((F.col('lab_HEMOGLOBIN_VALUE') < 12) & (F.col('lab_MCV_VALUE') < 80)) |
           ((F.col('lab_FERRITIN_VALUE') < 30) & (F.col('lab_HEMOGLOBIN_VALUE') < 13)), 1).otherwise(0)
)

# 2. Metabolic dysfunction pattern (LFTs + albumin)
df_final = df_final.withColumn('lab_metabolic_dysfunction',
    F.when((F.col('lab_ALT_ABNORMAL') == 1) | 
           (F.col('lab_AST_ABNORMAL') == 1) |
           (F.col('lab_ALK_PHOS_VALUE') > 150) |  # Using threshold instead of abnormal flag
           (F.col('lab_ALBUMIN_DROP_15PCT_FLAG') == 1), 1).otherwise(0)
)

# 3. Inflammatory burden (CRP + platelets + ESR)
df_final = df_final.withColumn('lab_inflammatory_burden',
    F.when((F.col('lab_CRP_VALUE') > 10) | 
           (F.col('lab_THROMBOCYTOSIS_FLAG') == 1) |
           (F.col('lab_ESR_VALUE') > 30), 1).otherwise(0)
)

# 4. Progressive anemia pattern
df_final = df_final.withColumn('lab_progressive_anemia',
    F.when((F.col('lab_HGB_TRAJECTORY') == 'RAPID_DECLINE') | 
           (F.col('lab_HGB_TRAJECTORY') == 'MODERATE_DECLINE') |
           (F.col('lab_HEMOGLOBIN_ACCELERATING_DECLINE') == 1), 1).otherwise(0)
)

composite_features = [
    'lab_comprehensive_iron_deficiency',
    'lab_metabolic_dysfunction',
    'lab_inflammatory_burden',
    'lab_progressive_anemia'
]

# Add composites to selected features
selected_features.extend(composite_features)
selected_features = sorted(list(set(selected_features)))  # Remove duplicates and sort

print(f"\nAdded {len(composite_features)} composite features")
print(f"Final feature count: {len(selected_features)}")

# Ordinal-encode categorical features (must happen AFTER composite features use them as strings)
if 'lab_ANEMIA_GRADE' in selected_features:
    df_final = df_final.withColumn('lab_ANEMIA_GRADE',
        F.when(F.col('lab_ANEMIA_GRADE') == 'SEVERE_ANEMIA', 3)
         .when(F.col('lab_ANEMIA_GRADE') == 'MODERATE_ANEMIA', 2)
         .when(F.col('lab_ANEMIA_GRADE') == 'MILD_ANEMIA', 1)
         .when(F.col('lab_ANEMIA_GRADE') == 'NORMAL', 0)
         .otherwise(F.lit(None).cast('int')))

if 'lab_HGB_TRAJECTORY' in selected_features:
    df_final = df_final.withColumn('lab_HGB_TRAJECTORY',
        F.when(F.col('lab_HGB_TRAJECTORY') == 'RAPID_DECLINE', 3)
         .when(F.col('lab_HGB_TRAJECTORY') == 'MODERATE_DECLINE', 2)
         .when(F.col('lab_HGB_TRAJECTORY') == 'MILD_DECLINE', 1)
         .when(F.col('lab_HGB_TRAJECTORY') == 'STABLE_OR_RISING', 0)
         .otherwise(F.lit(None).cast('int')))

print("✓ Ordinal-encoded lab_ANEMIA_GRADE and lab_HGB_TRAJECTORY")
print(f"Final feature count: {len(selected_features)}")

# === PRINT FINAL FEATURE LIST ===
print("\n" + "="*60)
print("FINAL SELECTED LABORATORY FEATURES")
print("="*60)

for i, feat in enumerate(selected_features, 1):
    # Add description for clarity
    if 'ACCELERATING' in feat or 'ACCELERATION' in feat:
        desc = " [EXTREME RISK - RARE]"
    elif 'CA125' in feat:
        desc = " [OVARIAN MARKER]"
    elif 'IRON_DEFICIENCY' in feat or 'ANEMIA' in feat:
        desc = " [CRC BIOMARKER]"
    elif 'HEMOGLOBIN' in feat or 'HGB' in feat:
        desc = " [BLEEDING MARKER]"
    elif feat in composite_features:
        desc = " [COMPOSITE]"
    elif 'ALBUMIN' in feat or 'metabolic' in feat:
        desc = " [NUTRITIONAL/METABOLIC]"
    else:
        desc = ""
    print(f"{i:2d}. {feat:<45} {desc}")

# === SAVE REDUCED DATASET ===
final_columns = ['PAT_ID', 'END_DTTM'] + selected_features
df_reduced = df_final.select(*final_columns)

# Write to final table
output_table = 'dev.clncl_ds.herald_test_train_labs_reduced'
df_reduced.write.mode('overwrite').option('mergeSchema', 'true').saveAsTable(output_table)

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: 93")
print(f"Selected features: {len(selected_features)}")
print(f"Reduction: {(1 - len(selected_features)/93)*100:.1f}%")
print(f"\n✓ Reduced dataset saved to: {output_table}")

# Verify save and check all columns have lab_ prefix
row_count = spark.table(output_table).count()
cols_without_prefix = [c for c in selected_features if not c.startswith('lab_')]

print(f"✓ Verified {row_count:,} rows written to table")
if cols_without_prefix:
    print(f"\n⚠ WARNING: These columns missing 'lab_' prefix: {cols_without_prefix}")
else:
    print("✓ All feature columns have 'lab_' prefix for joining")

# ========================================
# CELL 24
# ========================================

# CELL 20A: Validate Composite Feature Logic
print("\n" + "="*60)
print("COMPOSITE FEATURE VALIDATION")
print("="*60)

# Validate comprehensive iron deficiency logic
print("\nComprehensive Iron Deficiency Composite:")
print("The dual hemoglobin thresholds (12 g/dL for overt anemia with")
print("microcytosis, 13 g/dL for iron depletion with low ferritin)")
print("capture the spectrum from early iron deficiency to established anemia.")

validation_query = f'''
SELECT 
    r.lab_comprehensive_iron_deficiency,
    COUNT(*) as count,
    AVG(c.FUTURE_CRC_EVENT) * 100 as crc_rate_pct
FROM dev.clncl_ds.herald_test_train_labs_reduced r
JOIN dev.clncl_ds.herald_test_train_final_cohort c
    ON r.PAT_ID = c.PAT_ID 
    AND r.END_DTTM = c.END_DTTM
GROUP BY r.lab_comprehensive_iron_deficiency
'''

spark.sql(validation_query).show()

# ========================================
# CELL 25
# ========================================

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_test_train_labs_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 26
# ========================================

display(df_check)



################################################################################
# V2_Book5_1_Medications_Outpatient
################################################################################

# V2_Book5_1_Medications_Outpatient
# Functional cells: 27 of 64 code cells (123 total)
# Source: V2_Book5_1_Medications_Outpatient.ipynb
# =============================================================================

# ========================================
# CELL 1
# ========================================

# ---------------------------------
# Imports and Variable Declarations
# ---------------------------------

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# Initialize a Spark session for distributed data processing
spark = SparkSession.builder.getOrCreate()

# Ensure date/time comparisons use Central Time
spark.conf.set("spark.sql.session.timeZone", "America/Chicago")

# Define target catalog for SQL based on the environment variable
trgt_cat = os.environ.get('trgt_cat')

# Use appropriate Spark catalog based on the target CATEGORY
spark.sql('USE CATALOG prod;')

# ========================================
# CELL 2
# ========================================

# Cell 1: Create medication grouper category map
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_med_grouper_category_map
USING DELTA
AS
WITH id_map AS (
  SELECT * FROM VALUES
    -- IRON SUPPLEMENTATION
    ('IRON_SUPPLEMENTATION', 103100859),
    ('IRON_SUPPLEMENTATION', 1060000647),
    ('IRON_SUPPLEMENTATION', 1060000646),
    ('IRON_SUPPLEMENTATION', 1060000650),

    -- PPI USE
    ('PPI_USE', 1030101319),
    ('PPI_USE', 103101245),
    ('PPI_USE', 1060033801),
    ('PPI_USE', 1060038401),

    -- NSAID/ASA USE
    ('NSAID_ASA_USE', 1060031001),
    ('NSAID_ASA_USE', 1060000523),
    ('NSAID_ASA_USE', 1060015801),
    ('NSAID_ASA_USE', 1060046101),
    ('NSAID_ASA_USE', 1060028401),
    ('NSAID_ASA_USE', 1060028501),
    ('NSAID_ASA_USE', 103100721),

    -- STATIN USE
    ('STATIN_USE', 1232000017),
    ('STATIN_USE', 1765734),
    ('STATIN_USE', 1765928),
    ('STATIN_USE', 1765229),
    ('STATIN_USE', 1765232),
    ('STATIN_USE', 1765246),
    ('STATIN_USE', 1765249),
    ('STATIN_USE', 1765260),
    ('STATIN_USE', 1765261),
    ('STATIN_USE', 1765262),
    ('STATIN_USE', 1765263),
    ('STATIN_USE', 1765264),
    ('STATIN_USE', 1754575),
    ('STATIN_USE', 1754577),
    ('STATIN_USE', 1754603),
    ('STATIN_USE', 1754584),
    ('STATIN_USE', 1754588),
    ('STATIN_USE', 1754592),
    ('STATIN_USE', 1754593),
    ('STATIN_USE', 1754594),
    ('STATIN_USE', 1754595),
    ('STATIN_USE', 1765241),
    ('STATIN_USE', 1765239),
    ('STATIN_USE', 1765240),
    ('STATIN_USE', 1765253),
    ('STATIN_USE', 1765254),
    ('STATIN_USE', 1030107485),
    ('STATIN_USE', 103103567),
    ('STATIN_USE', 103103088),
    ('STATIN_USE', 105100537),
    ('STATIN_USE', 1060048301),

    -- METFORMIN USE
    ('METFORMIN_USE', 103101190),
    ('METFORMIN_USE', 1060036201),
    ('METFORMIN_USE', 1060040001),
    ('METFORMIN_USE', 1060000704),
    ('METFORMIN_USE', 1060040101),
    ('METFORMIN_USE', 1765323),
    ('METFORMIN_USE', 1765276),
    ('METFORMIN_USE', 1765279),
    ('METFORMIN_USE', 1765282),
    ('METFORMIN_USE', 1765284),
    ('METFORMIN_USE', 1765289),
    ('METFORMIN_USE', 1765292),
    ('METFORMIN_USE', 1765320),
    ('METFORMIN_USE', 1765328),
    ('METFORMIN_USE', 1765331),
    ('METFORMIN_USE', 1765334),
    ('METFORMIN_USE', 1765336),
    ('METFORMIN_USE', 1765339)
  AS t(category_key, grouper_id)
)
SELECT
  m.category_key,
  gi.GROUPER_ID,
  gi.GROUPER_NAME
FROM id_map m
LEFT JOIN clarity.grouper_items gi
  ON gi.GROUPER_ID = m.grouper_id;
''')

# ========================================
# CELL 3
# ========================================

# Cell 2: Create medication ID category map (for medications not in groupers)
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_med_id_category_map
USING DELTA
AS
SELECT
  UPPER(TRIM(CATEGORY_KEY))   AS CATEGORY_KEY,
  CAST(MEDICATION_ID AS BIGINT) AS MEDICATION_ID,
  UPPER(TRIM(GEN_NAME))       AS GEN_NAME,
  SOURCE_TAG,
  NOTES
FROM VALUES
  -- ===== LAXATIVE_USE =====
  ('LAXATIVE_USE', 229101, 'PRUCALOPRIDE 2 MG TABLET',                  'regex_v1_vetted', '5-HT4 agonist'),
  ('LAXATIVE_USE', 229100, 'PRUCALOPRIDE 1 MG TABLET',                  'regex_v1_vetted', '5-HT4 agonist'),
  ('LAXATIVE_USE', 179778, 'PSYLLIUM HUSK 3.5 GRAM ORAL POWDER PACKET', 'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 221195, 'PSYLLIUM HUSK 0.4 GRAM CAPSULE',            'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 87600,  'PSYLLIUM HUSK 0.52 GRAM CAPSULE',           'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 81953,  'CALCIUM POLYCARBOPHIL 500 MG CHEWABLE TABLET','regex_v1_vetted','bulk/fiber'),
  ('LAXATIVE_USE', 80942,  'CALCIUM POLYCARBOPHIL 625 MG TABLET',       'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 80695,  'METHYLCELLULOSE (LAXATIVE) 500 MG TABLET',  'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 197631, 'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 39610,  'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 93851,  'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 80411,  'LACTULOSE 10 GRAM ORAL PACKET',             'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 82328,  'LACTULOSE 20 GRAM ORAL PACKET',             'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 197575, 'LACTULOSE 20 GRAM/30 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 95550,  'LUBIPROSTONE 24 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 178676, 'LUBIPROSTONE 8 MCG CAPSULE',                'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 70603,  'LUBIPROSTONE 24 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 214922, 'PSYLLIUM HUSK 3 GRAM/3 GRAM ORAL POWDER',   'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 202759, 'LINACLOTIDE 145 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 202760, 'LINACLOTIDE 290 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 221253, 'LINACLOTIDE 72 MCG CAPSULE',                'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 221416, 'PLECANATIDE 3 MG TABLET',                   'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 222935, 'NALDEMEDINE 0.2 MG TABLET',                 'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 212755, 'NALOXEGOL 12.5 MG TABLET',                  'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 212698, 'NALOXEGOL 25 MG TABLET',                    'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 219542, 'METHYLNALTREXONE 150 MG TABLET',            'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 178822, 'METHYLNALTREXONE 12 MG/0.6 ML SC SOLUTION', 'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 197993, 'METHYLNALTREXONE 12 MG/0.6 ML SC SYRINGE',  'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 197990, 'METHYLNALTREXONE 8 MG/0.4 ML SC SYRINGE',   'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 2572,   'DOCUSATE SODIUM 100 MG TABLET',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 212170, 'DOCUSATE SODIUM 50 MG CAPSULE',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2085,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2568,   'DOCUSATE SODIUM 50 MG CAPSULE',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2567,   'DOCUSATE SODIUM 250 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 1815,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 4372,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 11878,  'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 39752,  'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
  ('LAXATIVE_USE', 30015,  'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
  ('LAXATIVE_USE', 7159,   'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
-- Polyethylene glycol products (very commonly used)
('LAXATIVE_USE', 25424, 'POLYETHYLENE GLYCOL 3350 17 GRAM ORAL POWDER PACKET', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 24984, 'POLYETHYLENE GLYCOL 3350 17 GRAM/DOSE ORAL POWDER', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 156129, 'POLYETHYLENE GLYCOL 3350 ORAL', 'discovery_v2', 'osmotic'),
-- Bisacodyl products
('LAXATIVE_USE', 1080, 'BISACODYL 10 MG RECTAL SUPPOSITORY', 'discovery_v2', 'stimulant'),
('LAXATIVE_USE', 13632, 'BISACODYL 5 MG TABLET,DELAYED RELEASE', 'discovery_v2', 'stimulant'),
('LAXATIVE_USE', 83533, 'BISACODYL 5 MG TABLET', 'discovery_v2', 'stimulant'),
-- Magnesium products
('LAXATIVE_USE', 79944, 'MAGNESIUM HYDROXIDE 400 MG/5 ML ORAL SUSPENSION', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 4711, 'MAGNESIUM CITRATE ORAL SOLUTION', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 155118, 'MAGNESIUM CITRATE ORAL', 'discovery_v2', 'osmotic'),
-- Senna combinations (beyond what you have)
('LAXATIVE_USE', 40926, 'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TABLET', 'discovery_v2', 'stimulant combo'),
-- Alvimopan (opioid antagonist for postop ileus)
('LAXATIVE_USE', 179661, 'ALVIMOPAN 12 MG CAPSULE', 'discovery_v2', 'peripherally acting opioid antagonist'),
-- Add these to ANTIDIARRHEAL_USE
('ANTIDIARRHEAL_USE', 4560, 'LOPERAMIDE 2 MG CAPSULE', 'discovery_v2', 'antimotility'),
('ANTIDIARRHEAL_USE', 4562, 'LOPERAMIDE 2 MG TABLET', 'discovery_v2', 'antimotility'),
('ANTIDIARRHEAL_USE', 189801, 'RIFAXIMIN 550 MG TABLET', 'discovery_v2', 'antibiotic for IBS-D/HE'),
('ANTIDIARRHEAL_USE', 189817, 'RIFAXIMIN 550 MG TABLET (XIFAXAN)', 'discovery_v2', 'antibiotic for IBS-D/HE'),
('ANTIDIARRHEAL_USE', 81342, 'BISMUTH SUBSALICYLATE 262 MG CHEWABLE TABLET', 'discovery_v2', 'bismuth compound'),
('ANTIDIARRHEAL_USE', 80560, 'BISMUTH SUBSALICYLATE 262 MG/15 ML ORAL SUSPENSION', 'discovery_v2', 'bismuth compound'),
('ANTIDIARRHEAL_USE', 89923, 'ALOSETRON 0.5 MG TABLET', 'discovery_v2', '5-HT3 antagonist for IBS-D'),
('ANTIDIARRHEAL_USE', 79963, 'ALOSETRON 1 MG TABLET', 'discovery_v2', '5-HT3 antagonist for IBS-D'),

  -- ===== ANTIDIARRHEAL_USE =====
  ('ANTIDIARRHEAL_USE', 2516,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG TABLET',             'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 88489,  'CHOLESTYRAMINE-ASPARTAME 4 GRAM ORAL POWDER FOR SUSP IN A PACKET', 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 183910, 'COLESEVELAM 3.75 GRAM ORAL POWDER PACKET',                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 16388,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 37482,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 80288,  'COLESEVELAM 625 MG TABLET',                                 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176079, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 82389,  'COLESTIPOL 5 GRAM ORAL PACKET',                             'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 21994,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216207, 'ELUXADOLINE 75 MG TABLET',                                  'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 9589,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM ORAL POWDER',            'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 89845,  'COLESTIPOL 5 GRAM ORAL GRANULES',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176078, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 28857,  'COLESEVELAM 625 MG TABLET',                                 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216198, 'ELUXADOLINE 100 MG TABLET',                                 'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 82555,  'COLESTIPOL 1 GRAM TABLET',                                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 9588,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM POWDER FOR SUSP IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176054, 'CHOLESTYRAMINE-ASPARTAME 4 GRAM ORAL POWDER',               'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 13898,  'COLESTIPOL 1 GRAM TABLET',                                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216086, 'ELUXADOLINE 75 MG TABLET',                                  'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 2515,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG/5 ML ORAL LIQUID',   'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 216087, 'ELUXADOLINE 100 MG TABLET',                                 'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 183915, 'COLESEVELAM 3.75 GRAM ORAL POWDER PACKET',                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 252231, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 252230, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216374, 'ELUXADOLINE ORAL',                                          'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 4553,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG TABLET',             'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 154828, 'DIPHENOXYLATE-ATROPINE ORAL',                               'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 146632, 'CHOLESTYRAMINE-ASPARTAME ORAL',                             'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147147, 'COLESTIPOL ORAL',                                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 6767,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM ORAL POWDER',            'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 160917, 'CHOLESTYRAMINE (WITH SUGAR) ORAL',                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 166274, 'COLESEVELAM ORAL',                                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 89844,  'COLESTIPOL 7.5 GRAM ORAL PACKET',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147146, 'COLESTIPOL ORAL',                                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 148928, 'DIPHENOXYLATE-ATROPINE ORAL',                               'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 12057,  'COLESTIPOL 7.5 GRAM ORAL PACKET',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 197067, 'CHOLESTYRAMINE (WITH SUGAR) ORAL',                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 146631, 'CHOLESTYRAMINE ORAL',                                       'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 6766,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM POWDER FOR SUSP IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147144, 'COLESEVELAM ORAL',                                          'regex_v1_vetted', 'bile-acid binder'),

  -- ===== ANTISPASMODIC_USE =====
  ('ANTISPASMODIC_USE', 1738,    'CHLORDIAZEPOXIDE-CLIDINIUM 5 MG-2.5 MG CAPSULE', 'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 2418,    'DICYCLOMINE 10 MG CAPSULE',                      'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 205840,  'DICYCLOMINE 10 MG/5 ML ORAL SOLUTION',           'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 2420,    'DICYCLOMINE 20 MG TABLET',                       'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 144535,  'DICYCLOMINE ORAL',                               'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 148732,  'DICYCLOMINE ORAL',                               'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 4434,    'CHLORDIAZEPOXIDE-CLIDINIUM 5 MG-2.5 MG CAPSULE', 'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 146851,  'CHLORDIAZEPOXIDE-CLIDINIUM ORAL',                'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 154497,  'CHLORDIAZEPOXIDE-CLIDINIUM ORAL',                'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 254015,  'DICYCLOMINE 40 MG TABLET',                       'regex_v1_vetted', 'GI anticholinergic'),

  -- ===== B12_FOLATE_USE =====
  ('B12_FOLATE_USE', 4396,   'LEUCOVORIN CALCIUM 15 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 4397,   'LEUCOVORIN CALCIUM 25 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 212837, 'LEUCOVORIN 4 MG-PYRIDOXAL PHOSPHATE 50 MG-MECOBALAMIN 2 MG TABLET','regex_v1_vetted','combo w/ folinic acid & mecobalamin'),
  ('B12_FOLATE_USE', 4398,   'LEUCOVORIN CALCIUM 5 MG TABLET',                       'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 4395,   'LEUCOVORIN CALCIUM 10 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 242499, 'METHYLFOLATE CALCIUM 25,000 MCG DFE-MECOBALAMIN 2,000 MCG CAPSULE','regex_v1_vetted','high-dose methylfolate + mecobalamin'),
  ('B12_FOLATE_USE', 243320, 'B12 5,000 MCG-METHYLFOLATE 1,360 MCG DFE-B6 2.5 MG CHEWABLE TABLET','regex_v1_vetted','high-dose B12 + methylfolate combo'),
  ('B12_FOLATE_USE', 154414, 'LEUCOVORIN CALCIUM ORAL',                             'regex_v1_vetted', 'folinic acid oral'),
  ('B12_FOLATE_USE', 15370,  'LEUCOVORIN CALCIUM 10 MG/ML INJECTION SOLUTION',      'regex_v1_vetted', 'folinic acid injection'),
  ('B12_FOLATE_USE', 216354, 'COQ10-ALA-RESVERATROL-LEUCOVORIN-B6-MECOBALAMIN-VITC-D3 ORAL','regex_v1_vetted','multi-ingredient incl. leucovorin & mecobalamin'),
  ('B12_FOLATE_USE', 4394,   'LEUCOVORIN CALCIUM 50 MG SOLUTION FOR INJECTION',     'regex_v1_vetted', 'folinic acid injection'),
  ('B12_FOLATE_USE', 4392,   'LEUCOVORIN CALCIUM 100 MG SOLUTION FOR INJECTION',    'regex_v1_vetted', 'folinic acid injection'),

-- ===== IBD_MED_USE =====
('IBD_MEDICATIONS', 6496, 'PREDNISONE 20 MG TABLET', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 6494, 'PREDNISONE 10 MG TABLET', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 14887, 'METHYLPREDNISOLONE 4 MG TABLETS IN A DOSE PACK', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 82000, 'BUDESONIDE DR - ER 3 MG CAPSULE,DELAYED,EXTENDED RELEASE', 'discovery_v2', 'GI-specific steroid'),
('IBD_MEDICATIONS', 167329, 'MESALAMINE 1.2 GRAM TABLET,DELAYED RELEASE', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 94087, 'MESALAMINE 1,000 MG RECTAL SUPPOSITORY', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 10535, 'MESALAMINE 4 GRAM/60 ML ENEMA', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 7562, 'SULFASALAZINE 500 MG TABLET', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 9183, 'AZATHIOPRINE 50 MG TABLET', 'discovery_v2', 'immunosuppressant'),
('IBD_MEDICATIONS', 4973, 'METHOTREXATE SODIUM 2.5 MG TABLET', 'discovery_v2', 'immunosuppressant'),
('IBD_MEDICATIONS', 80729, 'INFLIXIMAB 100 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'anti-TNF biologic'),
('IBD_MEDICATIONS', 225598, 'ADALIMUMAB 40 MG/0.4 ML SUBCUTANEOUS PEN KIT', 'discovery_v2', 'anti-TNF biologic'),
('IBD_MEDICATIONS', 209779, 'VEDOLIZUMAB 300 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'α4β7 integrin blocker'),
('IBD_MEDICATIONS', 184073, 'USTEKINUMAB 90 MG/ML SUBCUTANEOUS SYRINGE', 'discovery_v2', 'IL-12/23 inhibitor'),
('IBD_MEDICATIONS', 217804, 'TOFACITINIB ER 11 MG TABLET,EXTENDED RELEASE 24 HR', 'discovery_v2', 'JAK inhibitor'),

-- ===== HEMORRHOID/RECTAL_MED_USE =====
('HEMORRHOID_RECTAL_MEDS', 3738, 'HYDROCORTISONE ACETATE 25 MG RECTAL SUPPOSITORY', 'discovery_v2', 'rectal steroid'),
('HEMORRHOID_RECTAL_MEDS', 6589, 'HYDROCORTISONE 1 %-PRAMOXINE 1 % RECTAL FOAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 77665, 'HYDROCORTISONE-PRAMOXINE 2.5 %-1 % RECTAL CREAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 80624, 'HYDROCORTISONE 1 %-PRAMOXINE 1 % RECTAL FOAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 82892, 'HYDROCORTISONE-PRAMOXINE 1 %-1 % RECTAL CREAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 10210, 'HYDROCORTISONE 100 MG/60 ML ENEMA', 'discovery_v2', 'rectal steroid'),
('HEMORRHOID_RECTAL_MEDS', 35164, 'LIDOCAINE 3 %-HYDROCORTISONE 0.5 % RECTAL CREAM', 'discovery_v2', 'anesthetic combo'),

-- ===== GI_BLEEDING_MED_USE =====
('GI_BLEEDING_MEDS', 230527, 'TRANEXAMIC ACID 1,000 MG/100 ML IV PIGGYBACK', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 196970, 'TRANEXAMIC ACID 1,000 MG/10 ML INTRAVENOUS SOLUTION', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 190130, 'TRANEXAMIC ACID 650 MG TABLET', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 211947, 'VASOPRESSIN 20 UNIT/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'vasoconstrictor'),
('GI_BLEEDING_MEDS', 25122, 'OCTREOTIDE ACETATE 100 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'somatostatin analog'),
('GI_BLEEDING_MEDS', 25121, 'OCTREOTIDE ACETATE 50 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'somatostatin analog'),

-- ===== CHRONIC_OPIOD_MED_USE =====
('CHRONIC_OPIOID_USE', 3037, 'FENTANYL 50 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 34505, 'HYDROCODONE 5 MG-ACETAMINOPHEN 325 MG TABLET', 'discovery_v2', 'opioid combo'),
('CHRONIC_OPIOID_USE', 10814, 'OXYCODONE 5 MG TABLET', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 301537, 'MORPHINE 4 MG/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 14632, 'TRAMADOL 50 MG TABLET', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 27905, 'FENTANYL 25 MCG/HR TRANSDERMAL PATCH', 'discovery_v2', 'long-acting opioid'),
('CHRONIC_OPIOID_USE', 20920, 'MORPHINE ER 15 MG TABLET,EXTENDED RELEASE', 'discovery_v2', 'long-acting opioid'),
('CHRONIC_OPIOID_USE', 211843, 'OXYCODONE ER 10 MG TABLET,CRUSH RESISTANT,EXTENDED RELEASE', 'discovery_v2', 'long-acting opioid'),

-- ===== BROAD_SPECTRUM_ANTIBIOTIC_USE =====
('BROAD_SPECTRUM_ANTIBIOTICS', 9500, 'CEPHALEXIN 500 MG CAPSULE', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 25119, 'CIPROFLOXACIN 500 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 82091, 'LEVOFLOXACIN 500 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 87765, 'MOXIFLOXACIN 400 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 177634, 'CEFAZOLIN 2 GRAM/20 ML INTRAVENOUS SYRINGE', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 79742, 'CEFTRIAXONE 1 GRAM SOLUTION FOR INJECTION', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 9621, 'CLINDAMYCIN HCL 300 MG CAPSULE', 'discovery_v2', 'lincosamide'),
('BROAD_SPECTRUM_ANTIBIOTICS', 83455, 'PIPERACILLIN-TAZOBACTAM 3.375 GRAM IV PIGGYBACK', 'discovery_v2', 'beta-lactam combo'),
('BROAD_SPECTRUM_ANTIBIOTICS', 83077, 'ERTAPENEM 1 GRAM SOLUTION FOR INJECTION', 'discovery_v2', 'carbapenem'),
('BROAD_SPECTRUM_ANTIBIOTICS', 80713, 'MEROPENEM 1 GRAM INTRAVENOUS SOLUTION', 'discovery_v2', 'carbapenem'),

-- ===== HORMON_THERAPY_USE =====
('HORMONE_THERAPY', 82101, 'ESTRADIOL 0.01% VAGINAL CREAM', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 9967, 'ESTRADIOL 1 MG TABLET', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 80522, 'CONJUGATED ESTROGENS 0.625 MG/GRAM VAGINAL CREAM', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 11498, 'TAMOXIFEN 20 MG TABLET', 'discovery_v2', 'SERM'),
('HORMONE_THERAPY', 81433, 'RALOXIFENE 60 MG TABLET', 'discovery_v2', 'SERM'),
('HORMONE_THERAPY', 7784, 'TESTOSTERONE CYPIONATE 200 MG/ML INTRAMUSCULAR OIL', 'discovery_v2', 'testosterone'),
('HORMONE_THERAPY', 194705, 'TESTOSTERONE 1.62% TRANSDERMAL GEL', 'discovery_v2', 'testosterone'),

-- ===== CHEMOTHERAPY_AGENT_USE =====
('CHEMOTHERAPY_AGENTS', 78342, 'FLUOROURACIL 5 % TOPICAL CREAM', 'discovery_v2', '5-FU'),
('CHEMOTHERAPY_AGENTS', 79057, 'CAPECITABINE 500 MG TABLET', 'discovery_v2', '5-FU prodrug'),
('CHEMOTHERAPY_AGENTS', 80994, 'CAPECITABINE 150 MG TABLET', 'discovery_v2', '5-FU prodrug'),
('CHEMOTHERAPY_AGENTS', 89931, 'BEVACIZUMAB 25 MG/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'VEGF inhibitor'),
('CHEMOTHERAPY_AGENTS', 10631, 'MITOMYCIN 40 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'alkylating agent')

AS t(category_key, medication_id, gen_name, source_tag, notes);
''')

# ========================================
# CELL 4
# ========================================

# Cell 3: Create unpivoted outpatient medications table
# This cell extracts outpatient medication orders from the EHR and categorizes them
# based on our expanded medication mappings for CRC risk prediction
# CRITICAL: Only includes orders from July 1, 2021 forward (data availability constraint)

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds_unpivoted AS

WITH
cohort AS (
  -- Get all patient-month observations from our cohort
  SELECT
    CAST(PAT_ID AS STRING)          AS PAT_ID,
    END_DTTM
  FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
),

map_med AS (
  -- Load direct medication ID to category mappings
  -- These are medications identified through discovery queries
  SELECT
    UPPER(TRIM(CATEGORY_KEY))       AS CATEGORY,
    CAST(MEDICATION_ID AS BIGINT)   AS MEDICATION_ID,
    UPPER(TRIM(GEN_NAME))           AS GEN_NAME
  FROM {trgt_cat}.clncl_ds.herald_test_train_med_id_category_map
),

map_grp AS (
  -- Load grouper-based medication category mappings
  -- Groupers are pre-defined medication sets maintained by the institution
  SELECT
    UPPER(TRIM(CATEGORY_KEY))       AS CATEGORY,
    CAST(GROUPER_ID AS BIGINT)      AS GROUPER_ID,
    UPPER(TRIM(GROUPER_NAME))       AS GROUPER_NAME
  FROM {trgt_cat}.clncl_ds.herald_test_train_med_grouper_category_map
),

-- Compiled membership: expand groupers to individual medications
-- This creates the GROUPER_ID → MEDICATION_ID mapping
grp_med_members AS (
  SELECT DISTINCT
    CAST(itm.GROUPER_ID AS BIGINT)                AS GROUPER_ID,
    CAST(med.MEDICATION_ID AS BIGINT)             AS MEDICATION_ID
  FROM map_grp mg
  JOIN clarity.grouper_items itm
    ON itm.GROUPER_ID = mg.GROUPER_ID
  JOIN clarity.grouper_compiled_rec_list rec
    ON rec.base_grouper_id = itm.GROUPER_ID
  JOIN clarity.clarity_medication med
    ON med.MEDICATION_ID = rec.grouper_records_numeric_id
),

-- Extract outpatient medication orders
-- Key differences from inpatient: uses ORDER data not MAR (medication administration)
-- CRITICAL: Enforces July 1, 2021 data availability constraint
orders_outpatient AS (
  SELECT
    CAST(ome.PAT_ID AS STRING)                    AS PAT_ID,
    CAST(ome.MEDICATION_ID AS BIGINT)             AS MEDICATION_ID,
    UPPER(TRIM(ome.GENERIC_NAME))                 AS RAW_GENERIC,
    CAST(ome.ORDER_START_TIME AS TIMESTAMP)       AS ORDER_TIME,
    CAST(ome.ORDER_MED_ID AS BIGINT)              AS ORDER_MED_ID
  FROM clarity_cur.order_med_enh ome
  WHERE ome.ORDERING_MODE_C <> 2                  -- Exclude inpatient orders
    AND ome.ORDER_START_TIME IS NOT NULL
    AND DATE(ome.ORDER_START_TIME) >= '2021-07-01'  -- Data availability constraint from Book0
    AND ome.ORDER_CLASS <> 'Historical Med'       -- Exclude historical reconciliation
    AND ome.ORDER_STATUS_C IN (2, 5)              -- Sent (2) or Completed (5) orders only
),

-- Match medications via direct medication ID mapping
hits_med AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    mm.CATEGORY,
    om.MEDICATION_ID,
    om.RAW_GENERIC,
    om.ORDER_TIME
  FROM cohort c
  LEFT JOIN orders_outpatient om
    ON om.PAT_ID = c.PAT_ID
   AND DATE(om.ORDER_TIME) <  c.END_DTTM          -- Before prediction point
   AND DATE(om.ORDER_TIME) >= ADD_MONTHS(c.END_DTTM, -24)  -- Within 24-month lookback
  JOIN map_med mm
    ON mm.MEDICATION_ID = om.MEDICATION_ID
),

-- Match medications via grouper membership
hits_grp AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    mg.CATEGORY,
    om.MEDICATION_ID,
    om.RAW_GENERIC,
    om.ORDER_TIME
  FROM cohort c
  LEFT JOIN orders_outpatient om
    ON om.PAT_ID = c.PAT_ID
   AND DATE(om.ORDER_TIME) <  c.END_DTTM
   AND DATE(om.ORDER_TIME) >= ADD_MONTHS(c.END_DTTM, -24)
  JOIN grp_med_members gm
    ON gm.MEDICATION_ID = om.MEDICATION_ID
  JOIN map_grp mg
    ON mg.GROUPER_ID = gm.GROUPER_ID
),

-- Combine all medication category matches
hits_all AS (
  SELECT * FROM hits_med
  UNION ALL
  SELECT * FROM hits_grp
),

-- Deduplicate: Keep most recent order per medication per day
-- This handles cases where same medication ordered multiple times in a day
ranked AS (
  SELECT
    PAT_ID,
    END_DTTM,
    CATEGORY,
    MEDICATION_ID,
    RAW_GENERIC,
    ORDER_TIME,
    DATEDIFF(END_DTTM, DATE(ORDER_TIME)) AS DAYS_SINCE_MED,  -- Days from order to prediction
    ROW_NUMBER() OVER (
      PARTITION BY PAT_ID, END_DTTM, CATEGORY, MEDICATION_ID, DATE(ORDER_TIME)
      ORDER BY ORDER_TIME DESC
    ) AS rn
  FROM hits_all
)

-- Final output: one row per patient-observation-category-medication-day
SELECT
  c.PAT_ID,
  c.END_DTTM,
  r.CATEGORY,
  r.MEDICATION_ID,
  r.RAW_GENERIC,
  r.ORDER_TIME,
  r.DAYS_SINCE_MED
FROM cohort c
LEFT JOIN ranked r
  ON r.PAT_ID   = c.PAT_ID
 AND r.END_DTTM = c.END_DTTM
WHERE r.rn = 1 OR r.rn IS NULL;
''')

print("Outpatient medications unpivoted table created with July 2021 data constraint")

# ========================================
# CELL 5
# ========================================

# Grouper Expansion Validation

# Verify that grouper-based medication categories are properly expanding to individual medications

# Cell 3A: Validate grouper expansion logic
print("="*70)
print("GROUPER EXPANSION VALIDATION")
print("="*70)

# Check how many medications each grouper category expands to
grouper_expansion = spark.sql(f"""
WITH grp_med_members AS (
  SELECT DISTINCT
    CAST(itm.GROUPER_ID AS BIGINT) AS GROUPER_ID,
    CAST(med.MEDICATION_ID AS BIGINT) AS MEDICATION_ID
  FROM {trgt_cat}.clncl_ds.herald_test_train_med_grouper_category_map mg
  JOIN clarity.grouper_items itm
    ON itm.GROUPER_ID = mg.GROUPER_ID
  JOIN clarity.grouper_compiled_rec_list rec
    ON rec.base_grouper_id = itm.GROUPER_ID
  JOIN clarity.clarity_medication med
    ON med.MEDICATION_ID = rec.grouper_records_numeric_id
)
SELECT 
  mg.CATEGORY_KEY,
  mg.GROUPER_NAME,
  COUNT(DISTINCT gm.MEDICATION_ID) as n_medications_expanded
FROM {trgt_cat}.clncl_ds.herald_test_train_med_grouper_category_map mg
LEFT JOIN grp_med_members gm 
  ON mg.GROUPER_ID = gm.GROUPER_ID
GROUP BY mg.CATEGORY_KEY, mg.GROUPER_NAME
ORDER BY n_medications_expanded DESC
""")

display(grouper_expansion)

# Validate that groupers actually returned medications
grouper_check = grouper_expansion.collect()
for row in grouper_check:
    if row['n_medications_expanded'] == 0:
        print(f"WARNING: Grouper '{row['GROUPER_NAME']}' expanded to 0 medications!")
    else:
        print(f"✓ {row['CATEGORY_KEY']}: {row['n_medications_expanded']} medications from grouper")

print("="*70)

# ========================================
# CELL 6
# ========================================

# Cell 3B: Validate no temporal leakage and proper date filtering
print("="*70)
print("TEMPORAL INTEGRITY VALIDATION")
print("="*70)

# Check 1: Verify no future leakage (ORDER_TIME should always be < END_DTTM)
leakage_check = spark.sql(f"""
SELECT COUNT(*) as future_leakage_violations
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds_unpivoted
WHERE ORDER_TIME >= END_DTTM
""").collect()[0]

print(f"\n1. Future Leakage Check:")
print(f"   Violations (ORDER_TIME >= END_DTTM): {leakage_check['future_leakage_violations']:,}")
if leakage_check['future_leakage_violations'] == 0:
    print("   ✓ PASS: No temporal leakage detected")
else:
    print("   ✗ FAIL: Temporal leakage detected!")

# Check 2: Verify 24-month lookback window
window_check = spark.sql(f"""
SELECT 
  MIN(DAYS_SINCE_MED) as min_days_back,
  MAX(DAYS_SINCE_MED) as max_days_back,
  AVG(DAYS_SINCE_MED) as avg_days_back
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds_unpivoted
WHERE ORDER_TIME IS NOT NULL
""").collect()[0]

print(f"\n2. Lookback Window Validation:")
print(f"   Min days back: {window_check['min_days_back']}")
print(f"   Max days back: {window_check['max_days_back']}")
print(f"   Avg days back: {window_check['avg_days_back']:.1f}")
expected_max = 24 * 30 + 30  # ~750 days (24 months + buffer)
if window_check['max_days_back'] <= expected_max:
    print(f"   ✓ PASS: Max within 24-month window (≤{expected_max} days)")
else:
    print(f"   ✗ FAIL: Medications beyond 24-month window detected!")

# Check 3: Verify July 2021 data constraint
date_constraint_check = spark.sql(f"""
SELECT 
  MIN(DATE(ORDER_TIME)) as earliest_order_date,
  COUNT(*) as total_medication_orders,
  SUM(CASE WHEN DATE(ORDER_TIME) < '2021-07-01' THEN 1 ELSE 0 END) as pre_july_2021_orders
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds_unpivoted
WHERE ORDER_TIME IS NOT NULL
""").collect()[0]

print(f"\n3. July 2021 Data Constraint Check:")
print(f"   Earliest order date: {date_constraint_check['earliest_order_date']}")
print(f"   Pre-July 2021 orders: {date_constraint_check['pre_july_2021_orders']:,}")
if date_constraint_check['pre_july_2021_orders'] == 0:
    print("   ✓ PASS: All orders from July 1, 2021 or later")
else:
    print(f"   ✗ FAIL: {date_constraint_check['pre_july_2021_orders']:,} orders before July 2021!")

print("="*70)

# ========================================
# CELL 7
# ========================================

# Cell 4: Create final pivoted outpatient medications table using window functions
# This cell transforms the unpivoted medication data into features for modeling
# Creates 3 types of features per category: flag (ever used), days_since (recency), count (frequency)

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds AS

WITH
  cohort AS (
    -- Base cohort with all patient-month observations
    SELECT CAST(PAT_ID AS STRING) AS PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
  ),

  unpvt AS (
    -- Clean and filter the unpivoted medication data
    SELECT CAST(PAT_ID AS STRING) AS PAT_ID,
           END_DTTM,
           UPPER(TRIM(CATEGORY)) AS CATEGORY,
           CAST(DAYS_SINCE_MED AS INT) AS DAYS_SINCE_MED
    FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds_unpivoted
    WHERE CATEGORY IS NOT NULL
  ),

  -- RECENCY FEATURES: Find most recent medication use per category
  ranked_meds AS (
    SELECT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      DAYS_SINCE_MED,
      ROW_NUMBER() OVER (
        PARTITION BY PAT_ID, END_DTTM, CATEGORY 
        ORDER BY DAYS_SINCE_MED ASC  -- Smallest value = most recent
      ) AS rn_most_recent
    FROM unpvt
  ),

  most_recent_meds AS (
    SELECT 
      PAT_ID,
      END_DTTM,
      CATEGORY,
      DAYS_SINCE_MED
    FROM ranked_meds
    WHERE rn_most_recent = 1
  ),

  -- FREQUENCY FEATURES: Count total medication orders per category
  med_counts AS (
    SELECT DISTINCT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      COUNT(*) OVER (
        PARTITION BY PAT_ID, END_DTTM, CATEGORY
      ) AS med_count
    FROM unpvt
  ),

  -- BINARY FLAGS: Has patient ever used this medication category?
  med_flags AS (
    SELECT DISTINCT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      1 AS has_med_flag
    FROM unpvt
  ),

  -- Pivot recency features (days since last use)
  pivot_days AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Existing categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN DAYS_SINCE_MED END) AS iron_use_days_since,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN DAYS_SINCE_MED END) AS b12_or_folate_use_days_since,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN DAYS_SINCE_MED END) AS laxative_use_days_since,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN DAYS_SINCE_MED END) AS antidiarrheal_use_days_since,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN DAYS_SINCE_MED END) AS antispasmodic_use_days_since,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN DAYS_SINCE_MED END) AS ppi_use_days_since,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN DAYS_SINCE_MED END) AS nsaid_asa_use_days_since,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN DAYS_SINCE_MED END) AS statin_use_days_since,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN DAYS_SINCE_MED END) AS metformin_use_days_since,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN DAYS_SINCE_MED END) AS ibd_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN DAYS_SINCE_MED END) AS hemorrhoid_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN DAYS_SINCE_MED END) AS gi_bleed_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN DAYS_SINCE_MED END) AS opioid_use_days_since,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN DAYS_SINCE_MED END) AS broad_abx_days_since,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN DAYS_SINCE_MED END) AS hormone_therapy_days_since,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN DAYS_SINCE_MED END) AS chemo_agents_days_since
    FROM most_recent_meds
    GROUP BY PAT_ID, END_DTTM
  ),

  -- Pivot frequency features (count of orders in 2 years)
  pivot_counts AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Existing categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN med_count END) AS iron_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN med_count END) AS b12_or_folate_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN med_count END) AS laxative_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN med_count END) AS antidiarrheal_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN med_count END) AS antispasmodic_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN med_count END) AS ppi_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN med_count END) AS nsaid_asa_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN med_count END) AS statin_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN med_count END) AS metformin_use_count_2yr,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN med_count END) AS ibd_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN med_count END) AS hemorrhoid_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN med_count END) AS gi_bleed_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN med_count END) AS opioid_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN med_count END) AS broad_abx_count_2yr,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN med_count END) AS hormone_therapy_count_2yr,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN med_count END) AS chemo_agents_count_2yr
    FROM med_counts
    GROUP BY PAT_ID, END_DTTM
  ),

  -- Pivot binary flags (ever used in 2 years)
  pivot_flags AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Existing categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN has_med_flag END) AS iron_use_flag,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN has_med_flag END) AS b12_or_folate_use_flag,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN has_med_flag END) AS laxative_use_flag,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN has_med_flag END) AS antidiarrheal_use_flag,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN has_med_flag END) AS antispasmodic_use_flag,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN has_med_flag END) AS ppi_use_flag,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN has_med_flag END) AS nsaid_asa_use_flag,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN has_med_flag END) AS statin_use_flag,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN has_med_flag END) AS metformin_use_flag,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN has_med_flag END) AS ibd_meds_flag,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN has_med_flag END) AS hemorrhoid_meds_flag,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN has_med_flag END) AS gi_bleed_meds_flag,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN has_med_flag END) AS opioid_use_flag,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN has_med_flag END) AS broad_abx_flag,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN has_med_flag END) AS hormone_therapy_flag,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN has_med_flag END) AS chemo_agents_flag
    FROM med_flags
    GROUP BY PAT_ID, END_DTTM
  )

-- Final assembly: Join all feature types
-- COALESCE ensures 0 for flags/counts when no medication found (instead of NULL)
SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- Existing medication features
  COALESCE(pf.iron_use_flag, 0) AS iron_use_flag,
  pd.iron_use_days_since,
  COALESCE(pc.iron_use_count_2yr, 0) AS iron_use_count_2yr,

  COALESCE(pf.b12_or_folate_use_flag, 0) AS b12_or_folate_use_flag,
  pd.b12_or_folate_use_days_since,
  COALESCE(pc.b12_or_folate_use_count_2yr, 0) AS b12_or_folate_use_count_2yr,

  COALESCE(pf.laxative_use_flag, 0) AS laxative_use_flag,
  pd.laxative_use_days_since,
  COALESCE(pc.laxative_use_count_2yr, 0) AS laxative_use_count_2yr,

  COALESCE(pf.antidiarrheal_use_flag, 0) AS antidiarrheal_use_flag,
  pd.antidiarrheal_use_days_since,
  COALESCE(pc.antidiarrheal_use_count_2yr, 0) AS antidiarrheal_use_count_2yr,

  COALESCE(pf.antispasmodic_use_flag, 0) AS antispasmodic_use_flag,
  pd.antispasmodic_use_days_since,
  COALESCE(pc.antispasmodic_use_count_2yr, 0) AS antispasmodic_use_count_2yr,

  COALESCE(pf.ppi_use_flag, 0) AS ppi_use_flag,
  pd.ppi_use_days_since,
  COALESCE(pc.ppi_use_count_2yr, 0) AS ppi_use_count_2yr,

  COALESCE(pf.nsaid_asa_use_flag, 0) AS nsaid_asa_use_flag,
  pd.nsaid_asa_use_days_since,
  COALESCE(pc.nsaid_asa_use_count_2yr, 0) AS nsaid_asa_use_count_2yr,

  COALESCE(pf.statin_use_flag, 0) AS statin_use_flag,
  pd.statin_use_days_since,
  COALESCE(pc.statin_use_count_2yr, 0) AS statin_use_count_2yr,

  COALESCE(pf.metformin_use_flag, 0) AS metformin_use_flag,
  pd.metformin_use_days_since,
  COALESCE(pc.metformin_use_count_2yr, 0) AS metformin_use_count_2yr,

  -- New medication category features
  COALESCE(pf.ibd_meds_flag, 0) AS ibd_meds_flag,
  pd.ibd_meds_days_since,
  COALESCE(pc.ibd_meds_count_2yr, 0) AS ibd_meds_count_2yr,

  COALESCE(pf.hemorrhoid_meds_flag, 0) AS hemorrhoid_meds_flag,
  pd.hemorrhoid_meds_days_since,
  COALESCE(pc.hemorrhoid_meds_count_2yr, 0) AS hemorrhoid_meds_count_2yr,

  COALESCE(pf.gi_bleed_meds_flag, 0) AS gi_bleed_meds_flag,
  pd.gi_bleed_meds_days_since,
  COALESCE(pc.gi_bleed_meds_count_2yr, 0) AS gi_bleed_meds_count_2yr,

  COALESCE(pf.opioid_use_flag, 0) AS opioid_use_flag,
  pd.opioid_use_days_since,
  COALESCE(pc.opioid_use_count_2yr, 0) AS opioid_use_count_2yr,

  COALESCE(pf.broad_abx_flag, 0) AS broad_abx_flag,
  pd.broad_abx_days_since,
  COALESCE(pc.broad_abx_count_2yr, 0) AS broad_abx_count_2yr,

  COALESCE(pf.hormone_therapy_flag, 0) AS hormone_therapy_flag,
  pd.hormone_therapy_days_since,
  COALESCE(pc.hormone_therapy_count_2yr, 0) AS hormone_therapy_count_2yr,

  COALESCE(pf.chemo_agents_flag, 0) AS chemo_agents_flag,
  pd.chemo_agents_days_since,
  COALESCE(pc.chemo_agents_count_2yr, 0) AS chemo_agents_count_2yr

FROM cohort c
LEFT JOIN pivot_flags pf ON c.PAT_ID = pf.PAT_ID AND c.END_DTTM = pf.END_DTTM
LEFT JOIN pivot_days pd ON c.PAT_ID = pd.PAT_ID AND c.END_DTTM = pd.END_DTTM  
LEFT JOIN pivot_counts pc ON c.PAT_ID = pc.PAT_ID AND c.END_DTTM = pc.END_DTTM;
''')

# ========================================
# CELL 8
# ========================================

# Cell 5: Validate row count matches cohort and examine medication prevalence
# Critical validation: ensures every cohort observation has medication features

# Row count validation
result = spark.sql(f"""
SELECT 
    COUNT(*) as outpatient_meds_count,
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds
""")

result.show()
assert result.collect()[0]['diff'] == 0, "ERROR: Row count mismatch!"
print("✓ Row count validation passed")

# Calculate prevalence for all medication categories
# This helps validate that categories are capturing expected populations
df_summary = spark.sql(f'''
SELECT 
  COUNT(*) as total_rows,
  COUNT(DISTINCT PAT_ID) as unique_patients,
  
  -- Existing categories prevalence
  ROUND(AVG(iron_use_flag), 4) as iron_prevalence,
  ROUND(AVG(ppi_use_flag), 4) as ppi_prevalence,
  ROUND(AVG(statin_use_flag), 4) as statin_prevalence,
  ROUND(AVG(laxative_use_flag), 4) as laxative_prevalence,
  ROUND(AVG(antidiarrheal_use_flag), 4) as antidiarrheal_prevalence,
  ROUND(AVG(metformin_use_flag), 4) as metformin_prevalence,
  ROUND(AVG(nsaid_asa_use_flag), 4) as nsaid_prevalence,
  ROUND(AVG(b12_or_folate_use_flag), 4) as b12_prevalence,
  ROUND(AVG(antispasmodic_use_flag), 4) as antispasmodic_prevalence,
  
  -- New categories prevalence
  ROUND(AVG(ibd_meds_flag), 4) as ibd_meds_prevalence,
  ROUND(AVG(hemorrhoid_meds_flag), 4) as hemorrhoid_prevalence,
  ROUND(AVG(gi_bleed_meds_flag), 4) as gi_bleed_prevalence,
  ROUND(AVG(opioid_use_flag), 4) as opioid_prevalence,
  ROUND(AVG(broad_abx_flag), 4) as broad_abx_prevalence,
  ROUND(AVG(hormone_therapy_flag), 4) as hormone_prevalence,
  ROUND(AVG(chemo_agents_flag), 4) as chemo_prevalence
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds
''')

print("\n=== OUTPATIENT MEDICATION PREVALENCE ===")
display(df_summary)

# Check correlation with CRC outcome for validation
print("\n=== CHECKING ASSOCIATION WITH CRC OUTCOME ===")
outcome_check = spark.sql(f'''
SELECT 
  'Laxatives' as medication_category,
  AVG(CASE WHEN m.laxative_use_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_med,
  AVG(CASE WHEN m.laxative_use_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_med
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
UNION ALL

SELECT 
  'IBD Medications' as medication_category,
  AVG(CASE WHEN m.ibd_meds_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.ibd_meds_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
UNION ALL

SELECT 
  'Hemorrhoid Meds' as medication_category,
  AVG(CASE WHEN m.hemorrhoid_meds_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.hemorrhoid_meds_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
''')

display(outcome_check)

# ========================================
# CELL 9
# ========================================

# Cell 6: Comprehensive prevalence analysis with CRC association
# Examines medication prevalence and their association with CRC outcome

df_summary = spark.sql(f'''
WITH prevalence AS (
  SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    
    -- Calculate prevalence for all categories
    ROUND(AVG(iron_use_flag), 4) as iron_prevalence,
    ROUND(AVG(ppi_use_flag), 4) as ppi_prevalence,
    ROUND(AVG(statin_use_flag), 4) as statin_prevalence,
    ROUND(AVG(laxative_use_flag), 4) as laxative_prevalence,
    ROUND(AVG(antidiarrheal_use_flag), 4) as antidiarrheal_prevalence,
    ROUND(AVG(metformin_use_flag), 4) as metformin_prevalence,
    ROUND(AVG(nsaid_asa_use_flag), 4) as nsaid_prevalence,
    ROUND(AVG(b12_or_folate_use_flag), 4) as b12_prevalence,
    ROUND(AVG(antispasmodic_use_flag), 4) as antispasmodic_prevalence,
    ROUND(AVG(ibd_meds_flag), 4) as ibd_prevalence,
    ROUND(AVG(hemorrhoid_meds_flag), 4) as hemorrhoid_prevalence,
    ROUND(AVG(gi_bleed_meds_flag), 4) as gi_bleed_prevalence,
    ROUND(AVG(opioid_use_flag), 4) as opioid_prevalence,
    ROUND(AVG(broad_abx_flag), 4) as broad_abx_prevalence,
    ROUND(AVG(hormone_therapy_flag), 4) as hormone_prevalence,
    ROUND(AVG(chemo_agents_flag), 4) as chemo_prevalence
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds
),
crc_association AS (
  SELECT 
    -- Calculate CRC rates for each medication category
    ROUND(AVG(CASE WHEN m.iron_use_flag = 1 THEN c.FUTURE_CRC_EVENT END), 5) as iron_crc_rate,
    ROUND(AVG(CASE WHEN m.laxative_use_flag = 1 THEN c.FUTURE_CRC_EVENT END), 5) as laxative_crc_rate,
    ROUND(AVG(CASE WHEN m.antidiarrheal_use_flag = 1 THEN c.FUTURE_CRC_EVENT END), 5) as antidiarrheal_crc_rate,
    ROUND(AVG(CASE WHEN m.ibd_meds_flag = 1 THEN c.FUTURE_CRC_EVENT END), 5) as ibd_crc_rate,
    ROUND(AVG(CASE WHEN m.hemorrhoid_meds_flag = 1 THEN c.FUTURE_CRC_EVENT END), 5) as hemorrhoid_crc_rate,
    ROUND(AVG(c.FUTURE_CRC_EVENT), 5) as overall_crc_rate
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT * FROM prevalence, crc_association
''')

display(df_summary)
print("\n=== KEY INSIGHTS ===")
print("Look for medications with CRC rates significantly higher than baseline")
print("These are potential risk indicators for your model")

# ========================================
# CELL 10
# ========================================

# Cell 7: Analyze temporal patterns and recency effects
# Understanding how medication timing relates to CRC risk

temporal_analysis = spark.sql(f'''
WITH recency_bands AS (
  SELECT 
    PAT_ID,
    END_DTTM,
    -- Create recency bands for key medications
    CASE 
      WHEN laxative_use_days_since IS NULL THEN 'Never'
      WHEN laxative_use_days_since <= 30 THEN '0-30 days'
      WHEN laxative_use_days_since <= 90 THEN '31-90 days'
      WHEN laxative_use_days_since <= 180 THEN '91-180 days'
      WHEN laxative_use_days_since <= 365 THEN '181-365 days'
      ELSE '365+ days'
    END as laxative_recency,
    
    CASE 
      WHEN iron_use_days_since IS NULL THEN 'Never'
      WHEN iron_use_days_since <= 30 THEN '0-30 days'
      WHEN iron_use_days_since <= 90 THEN '31-90 days'
      WHEN iron_use_days_since <= 180 THEN '91-180 days'
      WHEN iron_use_days_since <= 365 THEN '181-365 days'
      ELSE '365+ days'
    END as iron_recency,
    
    CASE 
      WHEN antidiarrheal_use_days_since IS NULL THEN 'Never'
      WHEN antidiarrheal_use_days_since <= 30 THEN '0-30 days'
      WHEN antidiarrheal_use_days_since <= 90 THEN '31-90 days'
      WHEN antidiarrheal_use_days_since <= 180 THEN '91-180 days'
      WHEN antidiarrheal_use_days_since <= 365 THEN '181-365 days'
      ELSE '365+ days'
    END as antidiarrheal_recency
    
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds
)
SELECT 
  -- Laxative recency analysis
  laxative_recency,
  COUNT(*) as n_observations,
  ROUND(AVG(c.FUTURE_CRC_EVENT), 5) as crc_rate,
  ROUND(AVG(c.FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort), 2) as relative_risk
FROM recency_bands r
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON r.PAT_ID = c.PAT_ID AND r.END_DTTM = c.END_DTTM
GROUP BY laxative_recency
ORDER BY 
  CASE laxative_recency
    WHEN '0-30 days' THEN 1
    WHEN '31-90 days' THEN 2
    WHEN '91-180 days' THEN 3
    WHEN '181-365 days' THEN 4
    WHEN '365+ days' THEN 5
    WHEN 'Never' THEN 6
  END
''')

display(temporal_analysis)
print("\nTemporal patterns show how recency of medication use correlates with CRC risk")
print("Recent use of GI medications may indicate active symptoms")

# ========================================
# CELL 11
# ========================================

# Cell 8: Analyze medication combinations and polypharmacy patterns
# Identifies high-risk medication combinations

combo_analysis = spark.sql(f'''
WITH med_combinations AS (
  SELECT 
    PAT_ID,
    END_DTTM,
    -- Common GI symptom medication combinations
    CASE WHEN laxative_use_flag = 1 AND antidiarrheal_use_flag = 1 
         THEN 1 ELSE 0 END as alternating_bowel_pattern,
    
    CASE WHEN laxative_use_flag = 1 AND hemorrhoid_meds_flag = 1 
         THEN 1 ELSE 0 END as constipation_with_hemorrhoids,
    
    CASE WHEN iron_use_flag = 1 AND ppi_use_flag = 1 
         THEN 1 ELSE 0 END as iron_with_ppi,
    
    CASE WHEN ibd_meds_flag = 1 OR 
              (laxative_use_flag = 1 AND antidiarrheal_use_flag = 1 AND antispasmodic_use_flag = 1)
         THEN 1 ELSE 0 END as complex_gi_pattern,
    
    -- Total medication burden
    iron_use_flag + b12_or_folate_use_flag + laxative_use_flag + 
    antidiarrheal_use_flag + antispasmodic_use_flag + ppi_use_flag + 
    nsaid_asa_use_flag + statin_use_flag + metformin_use_flag +
    ibd_meds_flag + hemorrhoid_meds_flag + gi_bleed_meds_flag +
    opioid_use_flag + broad_abx_flag + hormone_therapy_flag + 
    chemo_agents_flag as total_med_categories
    
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds
)
SELECT 
  'Alternating Bowel (Lax+Antidiarr)' as pattern,
  SUM(alternating_bowel_pattern) as n_patients,
  ROUND(AVG(alternating_bowel_pattern), 4) as prevalence,
  ROUND(AVG(CASE WHEN alternating_bowel_pattern = 1 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_with,
  ROUND(AVG(CASE WHEN alternating_bowel_pattern = 0 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_without
FROM med_combinations m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM

UNION ALL

SELECT 
  'Constipation with Hemorrhoids',
  SUM(constipation_with_hemorrhoids),
  ROUND(AVG(constipation_with_hemorrhoids), 4),
  ROUND(AVG(CASE WHEN constipation_with_hemorrhoids = 1 THEN c.FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN constipation_with_hemorrhoids = 0 THEN c.FUTURE_CRC_EVENT END), 5)
FROM med_combinations m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM

UNION ALL

SELECT 
  'Iron with PPI',
  SUM(iron_with_ppi),
  ROUND(AVG(iron_with_ppi), 4),
  ROUND(AVG(CASE WHEN iron_with_ppi = 1 THEN c.FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN iron_with_ppi = 0 THEN c.FUTURE_CRC_EVENT END), 5)
FROM med_combinations m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM

UNION ALL

SELECT 
  'Complex GI Pattern',
  SUM(complex_gi_pattern),
  ROUND(AVG(complex_gi_pattern), 4),
  ROUND(AVG(CASE WHEN complex_gi_pattern = 1 THEN c.FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN complex_gi_pattern = 0 THEN c.FUTURE_CRC_EVENT END), 5)
FROM med_combinations m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
''')

display(combo_analysis)
print("\nMedication combinations reveal complex symptom patterns")
print("Alternating bowel patterns and complex GI symptoms are particularly concerning")

# ========================================
# CELL 12
# ========================================

# Cell 9: Analyze medication frequency patterns and treatment intensity
# High frequency use may indicate persistent symptoms

frequency_analysis = spark.sql(f'''
WITH frequency_categories AS (
  SELECT 
    PAT_ID,
    END_DTTM,
    
    -- Categorize laxative use frequency
    CASE 
      WHEN laxative_use_count_2yr = 0 THEN 'None'
      WHEN laxative_use_count_2yr = 1 THEN 'Single use'
      WHEN laxative_use_count_2yr BETWEEN 2 AND 5 THEN '2-5 times'
      WHEN laxative_use_count_2yr BETWEEN 6 AND 12 THEN '6-12 times'
      WHEN laxative_use_count_2yr > 12 THEN 'Chronic (>12)'
    END as laxative_frequency,
    
    -- Categorize iron supplementation frequency
    CASE 
      WHEN iron_use_count_2yr = 0 THEN 'None'
      WHEN iron_use_count_2yr = 1 THEN 'Single use'
      WHEN iron_use_count_2yr BETWEEN 2 AND 5 THEN '2-5 times'
      WHEN iron_use_count_2yr BETWEEN 6 AND 12 THEN '6-12 times'
      WHEN iron_use_count_2yr > 12 THEN 'Chronic (>12)'
    END as iron_frequency,
    
    -- Calculate treatment intensity score
    CASE 
      WHEN laxative_use_count_2yr > 12 OR antidiarrheal_use_count_2yr > 12 THEN 'High'
      WHEN laxative_use_count_2yr > 6 OR antidiarrheal_use_count_2yr > 6 THEN 'Moderate'
      WHEN laxative_use_count_2yr > 0 OR antidiarrheal_use_count_2yr > 0 THEN 'Low'
      ELSE 'None'
    END as gi_treatment_intensity
    
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds
)
SELECT 
  laxative_frequency,
  COUNT(*) as n_observations,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct_of_cohort,
  ROUND(AVG(c.FUTURE_CRC_EVENT), 5) as crc_rate,
  ROUND(AVG(c.FUTURE_CRC_EVENT) / NULLIF((SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort), 0), 2) as relative_risk
FROM frequency_categories f
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON f.PAT_ID = c.PAT_ID AND f.END_DTTM = c.END_DTTM
GROUP BY laxative_frequency
ORDER BY 
  CASE laxative_frequency
    WHEN 'None' THEN 1
    WHEN 'Single use' THEN 2
    WHEN '2-5 times' THEN 3
    WHEN '6-12 times' THEN 4
    WHEN 'Chronic (>12)' THEN 5
  END
''')

display(frequency_analysis)
print("\nFrequency analysis reveals dose-response relationship")
print("Chronic use (>12 times) shows highest risk")

# ========================================
# CELL 13
# ========================================

# Cell 10: Age-stratified medication patterns
# CRC risk varies by age, as do medication patterns

age_analysis = spark.sql(f'''
WITH age_cohorts AS (
  SELECT 
    m.*,
    c.AGE,
    c.FUTURE_CRC_EVENT,
    CASE 
      WHEN c.AGE < 50 THEN '45-49'
      WHEN c.AGE < 60 THEN '50-59'
      WHEN c.AGE < 70 THEN '60-69'
      WHEN c.AGE < 80 THEN '70-79'
      ELSE '80+'
    END as age_group
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT 
  age_group,
  COUNT(*) as n_obs,
  
  -- Prevalence by age
  ROUND(AVG(laxative_use_flag), 3) as laxative_prev,
  ROUND(AVG(iron_use_flag), 3) as iron_prev,
  ROUND(AVG(antidiarrheal_use_flag), 3) as antidiarrheal_prev,
  ROUND(AVG(ibd_meds_flag), 3) as ibd_prev,
  ROUND(AVG(hemorrhoid_meds_flag), 3) as hemorrhoid_prev,
  
  -- CRC rates by medication and age
  ROUND(AVG(CASE WHEN laxative_use_flag = 1 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_with_laxative,
  ROUND(AVG(CASE WHEN iron_use_flag = 1 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_with_iron,
  ROUND(AVG(FUTURE_CRC_EVENT), 5) as overall_crc_rate
  
FROM age_cohorts
GROUP BY age_group
ORDER BY age_group
''')

display(age_analysis)
print("\nAge-stratified analysis shows how medication-CRC associations vary by age")
print("Younger patients with GI medications may have higher relative risk")

# ========================================
# CELL 14
# ========================================

# Cell 11: Analyze potentially protective vs risk-increasing medications
# Some medications may be protective (ASA, statins) while others indicate risk

protective_analysis = spark.sql(f'''
WITH medication_effects AS (
  SELECT 
    m.*,
    c.FUTURE_CRC_EVENT,
    
    -- Potentially protective medications
    CASE WHEN nsaid_asa_use_flag = 1 OR statin_use_flag = 1 OR metformin_use_flag = 1 
         THEN 1 ELSE 0 END as has_protective_meds,
    
    -- Risk indicator medications
    CASE WHEN laxative_use_flag = 1 OR antidiarrheal_use_flag = 1 OR 
              iron_use_flag = 1 OR hemorrhoid_meds_flag = 1 OR gi_bleed_meds_flag = 1
         THEN 1 ELSE 0 END as has_risk_meds,
    
    -- Combined profile
    CASE 
      WHEN nsaid_asa_use_flag = 1 AND laxative_use_flag = 0 THEN 'ASA only'
      WHEN nsaid_asa_use_flag = 0 AND laxative_use_flag = 1 THEN 'Laxative only'
      WHEN nsaid_asa_use_flag = 1 AND laxative_use_flag = 1 THEN 'Both ASA and Laxative'
      ELSE 'Neither'
    END as asa_laxative_profile
    
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT 
  'Potentially Protective Meds' as medication_group,
  SUM(has_protective_meds) as n_with_meds,
  ROUND(AVG(has_protective_meds), 3) as prevalence,
  ROUND(AVG(CASE WHEN has_protective_meds = 1 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_with,
  ROUND(AVG(CASE WHEN has_protective_meds = 0 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_without,
  ROUND(AVG(CASE WHEN has_protective_meds = 0 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN has_protective_meds = 1 THEN FUTURE_CRC_EVENT END), 0), 2) as protective_ratio
FROM medication_effects

UNION ALL

SELECT 
  'Risk Indicator Meds',
  SUM(has_risk_meds),
  ROUND(AVG(has_risk_meds), 3),
  ROUND(AVG(CASE WHEN has_risk_meds = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN has_risk_meds = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN has_risk_meds = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN has_risk_meds = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM medication_effects

UNION ALL

SELECT 
  'ASA only',
  SUM(CASE WHEN asa_laxative_profile = 'ASA only' THEN 1 ELSE 0 END),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'ASA only' THEN 1 ELSE 0 END), 3),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'ASA only' THEN FUTURE_CRC_EVENT END), 5),
  NULL,
  NULL
FROM medication_effects

UNION ALL

SELECT 
  'Laxative only',
  SUM(CASE WHEN asa_laxative_profile = 'Laxative only' THEN 1 ELSE 0 END),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'Laxative only' THEN 1 ELSE 0 END), 3),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'Laxative only' THEN FUTURE_CRC_EVENT END), 5),
  NULL,
  NULL
FROM medication_effects

UNION ALL

SELECT 
  'Both ASA and Laxative',
  SUM(CASE WHEN asa_laxative_profile = 'Both ASA and Laxative' THEN 1 ELSE 0 END),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'Both ASA and Laxative' THEN 1 ELSE 0 END), 3),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'Both ASA and Laxative' THEN FUTURE_CRC_EVENT END), 5),
  NULL,
  NULL
FROM medication_effects
''')

display(protective_analysis)
print("\nProtective medication analysis shows complex interactions")
print("ASA may be protective, but when combined with laxatives may indicate higher risk")

# ========================================
# CELL 15
# ========================================

# Cell 12: Analyze GI medication combinations (important for CRC)
spark.sql(f'''
SELECT 
  COUNT(*) as total_patients,
  
  -- Single GI medications
  SUM(CASE WHEN iron_use_flag = 1 THEN 1 ELSE 0 END) as iron_only,
  SUM(CASE WHEN laxative_use_flag = 1 THEN 1 ELSE 0 END) as laxative_only,
  SUM(CASE WHEN antidiarrheal_use_flag = 1 THEN 1 ELSE 0 END) as antidiarrheal_only,
  SUM(CASE WHEN ppi_use_flag = 1 THEN 1 ELSE 0 END) as ppi_only,
  
  -- Key combinations
  SUM(CASE WHEN iron_use_flag = 1 AND laxative_use_flag = 1 THEN 1 ELSE 0 END) as iron_and_laxative,
  SUM(CASE WHEN iron_use_flag = 1 AND ppi_use_flag = 1 THEN 1 ELSE 0 END) as iron_and_ppi,
  SUM(CASE WHEN laxative_use_flag = 1 AND antidiarrheal_use_flag = 1 THEN 1 ELSE 0 END) as laxative_and_antidiarrheal,
  SUM(CASE WHEN ppi_use_flag = 1 AND iron_use_flag = 1 AND laxative_use_flag = 1 THEN 1 ELSE 0 END) as gi_triad,
  
  -- Any GI medication
  SUM(CASE WHEN (iron_use_flag + laxative_use_flag + antidiarrheal_use_flag + 
                 antispasmodic_use_flag + ppi_use_flag) > 0 THEN 1 ELSE 0 END) as any_gi_med,
  
  -- Multiple GI medications
  SUM(CASE WHEN (iron_use_flag + laxative_use_flag + antidiarrheal_use_flag + 
                 antispasmodic_use_flag + ppi_use_flag) >= 2 THEN 1 ELSE 0 END) as multiple_gi_meds
  
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds
''').show()

# ========================================
# CELL 16
# ========================================

# Cell 13: Convert to pandas for detailed statistics
df_spark = spark.sql('''SELECT * FROM dev.clncl_ds.herald_test_train_outpatient_meds''')
df = df_spark.toPandas()

print("Shape:", df.shape)
print("\nNull rates:")
print(df.isnull().sum()/df.shape[0])

# ========================================
# CELL 17
# ========================================

# Cell 14: Calculate mean values for all features
df_check = df.drop(columns=['PAT_ID', 'END_DTTM'], axis=1)
print("Mean values for outpatient medication features:")
print(df_check.mean())

# ========================================
# CELL 18
# ========================================

# Cell 15: Final summary statistics
print("=" * 80)
print("OUTPATIENT MEDICATIONS FEATURE ENGINEERING SUMMARY")
print("=" * 80)

summary = spark.sql(f"""
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    
    -- Core medication coverage
    ROUND(100.0 * AVG(iron_use_flag), 1) as iron_pct,
    ROUND(100.0 * AVG(laxative_use_flag), 1) as laxative_pct,
    ROUND(100.0 * AVG(ppi_use_flag), 1) as ppi_pct,
    ROUND(100.0 * AVG(statin_use_flag), 1) as statin_pct,
    
    -- Average prescription counts
    ROUND(AVG(iron_use_count_2yr), 2) as avg_iron_count,
    ROUND(AVG(laxative_use_count_2yr), 2) as avg_laxative_count,
    ROUND(AVG(ppi_use_count_2yr), 2) as avg_ppi_count,
    ROUND(AVG(statin_use_count_2yr), 2) as avg_statin_count
    
FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds
""").collect()[0]

for key, value in summary.asDict().items():
    if value is not None:
        if 'pct' in key:
            print(f"{key:30s}: {value:>10.1f}%")
        elif 'count' in key and key != 'total_rows':
            print(f"{key:30s}: {value:>10.2f}")
        else:
            print(f"{key:30s}: {value:>10,}")

print("=" * 80)
print("✓ Outpatient medications feature engineering complete")

# ========================================
# CELL 19
# ========================================

# Step 1: Calculate basic statistics using PySpark
print("Calculating feature statistics using PySpark...")

# Join with outcome data
df_spark = spark.sql("""
    SELECT m.*, c.FUTURE_CRC_EVENT
    FROM dev.clncl_ds.herald_test_train_outpatient_meds m
    JOIN dev.clncl_ds.herald_test_train_final_cohort c
        ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
""")

# Cache for performance
df_spark.cache()
total_rows = df_spark.count()
baseline_crc_rate = df_spark.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"Total rows: {total_rows:,}")
print(f"Baseline CRC rate: {baseline_crc_rate:.4f}")

# ========================================
# CELL 20
# ========================================

# Step 2: Calculate Risk Ratios for Flag Features (Fast in PySpark)
flag_features = [col for col in df_spark.columns if '_flag' in col]
risk_metrics = []

for feat in flag_features:
    stats = df_spark.groupBy(feat).agg(
        F.count('*').alias('count'),
        F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
    ).collect()
    
    # Parse results
    stats_dict = {row[feat]: {'count': row['count'], 'crc_rate': row['crc_rate']} for row in stats}
    
    prevalence = stats_dict.get(1, {'count': 0})['count'] / total_rows
    rate_with = stats_dict.get(1, {'crc_rate': 0})['crc_rate']
    rate_without = stats_dict.get(0, {'crc_rate': baseline_crc_rate})['crc_rate']
    risk_ratio = rate_with / (rate_without + 1e-10)
    
    # Handle edge cases for impact calculation
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
print("\nTop features by impact score:")
print(risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']])

# ========================================
# CELL 21
# ========================================

# Step 3: Analyze Feature Types and Information Content
print("\nAnalyzing feature types and information content...")

# Separate features by type
flag_features = [col for col in df_spark.columns if '_flag' in col]
count_features = [col for col in df_spark.columns if '_count' in col]
days_since_features = [col for col in df_spark.columns if '_days_since' in col]

print(f"Feature types:")
print(f"  - Flag features: {len(flag_features)}")
print(f"  - Count features: {len(count_features)}")
print(f"  - Days_since features: {len(days_since_features)}")

# For days_since: These are NULL when medication never given
missing_stats = []
for feat in days_since_features:
    # NULL means never had medication - this is information, not missing data
    never_had = df_spark.filter(F.col(feat).isNull()).count() / total_rows
    
    missing_stats.append({
        'feature': feat,
        'missing_rate': never_had,  # Keep as missing_rate for compatibility
        'medication': feat.replace('_days_since', '')
    })

missing_df = pd.DataFrame(missing_stats)
print(f"\nMedications ranked by usage (from days_since nulls):")
print(missing_df.sort_values('missing_rate').head(10))

# ========================================
# CELL 22
# ========================================

# Step 4: Sample for Mutual Information
sample_fraction = min(100000 / total_rows, 1.0)
df_sample = df_spark.sampleBy("FUTURE_CRC_EVENT", 
                               fractions={0: sample_fraction, 1: 1.0},
                               seed=42).toPandas()

print(f"\nSampled {len(df_sample):,} rows for MI calculation ({len(df_sample)/total_rows*100:.1f}%)")
print(f"Sample CRC rate: {df_sample['FUTURE_CRC_EVENT'].mean():.4f}")

# Calculate MI on sample
from sklearn.feature_selection import mutual_info_classif

feature_cols = [c for c in df_sample.columns if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT']]
X = df_sample[feature_cols].fillna(-999)
y = df_sample['FUTURE_CRC_EVENT']

mi_scores = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop features by Mutual Information:")
print(mi_df);

# ========================================
# CELL 23
# ========================================

# Step 5: Feature Selection Logic - FIXED
# First merge all metrics to create feature_importance
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

# Clinical knowledge filters
MUST_KEEP = [
    'hemorrhoid_meds_flag',
    'hemorrhoid_meds_days_since',  
    'iron_use_flag',
    'laxative_use_flag',
    'antidiarrheal_use_flag',
    'ppi_use_flag',
    'statin_use_flag'
]

# Remove near-zero variance features
REMOVE = ['b12_or_folate_use_flag', 'b12_or_folate_use_days_since', 
          'b12_or_folate_use_count_2yr',
          'chemo_agents_flag', 'chemo_agents_days_since', 
          'chemo_agents_count_2yr']

print(f"\nRemoving {len(REMOVE)} pre-specified near-zero variance features")
feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]

# ========================================
# CELL 24
# ========================================

# Step 6: Select Best Feature per Medication 
def select_optimal_features(df_importance):
    """Select best representation for each medication with balance"""
    
    selected = []
    df_importance['medication'] = df_importance['feature'].str.split('_').str[0]
    
    for med in df_importance['medication'].unique():
        med_features = df_importance[df_importance['medication'] == med]
        
        if med in ['hemorrhoid']:
            # Keep both for extreme risk
            selected.extend(med_features[
                med_features['feature'].str.contains('flag|days_since')
            ]['feature'].tolist())
            
        elif med in ['iron', 'laxative', 'antidiarrheal']:
            # Keep flag for high-risk GI
            flag_feat = med_features[med_features['feature'].str.contains('_flag')]
            if not flag_feat.empty:
                selected.append(flag_feat.iloc[0]['feature'])
                
        elif med in ['ppi', 'statin', 'metformin']:
            # For common meds, keep flag if high prevalence
            flag_feat = med_features[med_features['feature'].str.contains('_flag')]
            if not flag_feat.empty and flag_feat.iloc[0].get('prevalence', 0) > 0.05:
                selected.append(flag_feat.iloc[0]['feature'])
            else:
                # Otherwise best MI feature
                if len(med_features) > 0:
                    selected.append(med_features.nlargest(1, 'mi_score')['feature'].values[0])
        else:
            # For others, keep best MI feature
            if len(med_features) > 0:
                selected.append(med_features.nlargest(1, 'mi_score')['feature'].values[0])
    
    # Ensure must-keep features
    for feat in MUST_KEEP:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    return list(set(selected))

# CALL THE FUNCTION AND ASSIGN RESULT
selected_features = select_optimal_features(feature_importance)
print(f"\nSelected {len(selected_features)} features after optimization")

# ========================================
# CELL 25
# ========================================

# Step 7: Create Composite Features and Save
df_final = df_spark

# GI symptom composite
df_final = df_final.withColumn('gi_symptom_meds',
    F.when((F.col('laxative_use_flag') == 1) | 
           (F.col('antidiarrheal_use_flag') == 1) | 
           (F.col('antispasmodic_use_flag') == 1), 1).otherwise(0)
)

# Alternating bowel pattern
df_final = df_final.withColumn('alternating_bowel',
    F.when((F.col('laxative_use_flag') == 1) & 
           (F.col('antidiarrheal_use_flag') == 1), 1).otherwise(0)
)

# GI bleeding pattern
df_final = df_final.withColumn('gi_bleeding_pattern',
    F.when((F.col('iron_use_flag') == 1) & 
           (F.col('ppi_use_flag') == 1), 1).otherwise(0)
)

# Hemorrhoid risk score (exponential decay)
df_final = df_final.withColumn('hemorrhoid_risk_score',
    F.when(F.col('hemorrhoid_meds_days_since').isNull(), 0)
     .otherwise(30 * F.exp(-F.col('hemorrhoid_meds_days_since') / 30))
)

composite_features = ['gi_symptom_meds', 'alternating_bowel', 'gi_bleeding_pattern', 'hemorrhoid_risk_score']
selected_features.extend(composite_features)

print(f"\nAdded {len(composite_features)} composite features")
print(f"Final feature count: {len(selected_features)}")

# Print final feature list
print("\n" + "="*60)
print("FINAL SELECTED FEATURES")
print("="*60)
selected_features_sorted = sorted(list(set(selected_features)))  # Remove duplicates and sort
for i, feat in enumerate(selected_features_sorted, 1):
    # Add description for clarity
    if 'hemorrhoid' in feat:
        desc = " [EXTREME RISK]"
    elif 'iron' in feat or 'laxative' in feat or 'antidiarrheal' in feat:
        desc = " [HIGH RISK]"
    elif feat in composite_features:
        desc = " [COMPOSITE]"
    elif 'ppi' in feat or 'statin' in feat:
        desc = " [COMMON/IMPORTANT]"
    else:
        desc = ""
    print(f"{i:2d}. {feat:<35} {desc}")

# Select final columns and save
final_columns = ['PAT_ID', 'END_DTTM'] + sorted(list(set(selected_features)))
df_reduced = df_final.select(*final_columns)

# Add icd_ prefix to all columns except keys
out_med_cols = [col for col in df_reduced.columns if col not in ['PAT_ID', 'END_DTTM']]
for col in out_med_cols:
    df_reduced = df_reduced.withColumnRenamed(col, f'out_med_{col}' if not col.startswith('out_med_') else col)

# Write to final table
output_table = 'dev.clncl_ds.herald_test_train_outpatient_meds_reduced'
df_reduced.write.mode('overwrite').saveAsTable(output_table)

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: 48")
print(f"Selected features: {len(set(selected_features))}")
print(f"Reduction: {(1 - len(set(selected_features))/48)*100:.1f}%")
print(f"\n✓ Reduced dataset saved to: {output_table}")

# Verify save
row_count = spark.table(output_table).count()
print(f"✓ Verified {row_count:,} rows written to table")

# ========================================
# CELL 26
# ========================================

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_test_train_outpatient_meds_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 27
# ========================================

display(df_check)



################################################################################
# V2_Book5_2_Medications_Inpatient
################################################################################

# V2_Book5_2_Medications_Inpatient
# Functional cells: 22 of 56 code cells (107 total)
# Source: V2_Book5_2_Medications_Inpatient.ipynb
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

# Initialize a Spark session for distributed data processing
spark = SparkSession.builder.getOrCreate()

# Ensure date/time comparisons use Central Time
spark.conf.set("spark.sql.session.timeZone", "America/Chicago")

# Define target catalog for SQL based on the environment variable
trgt_cat = os.environ.get('trgt_cat')

# Use appropriate Spark catalog based on the target CATEGORY
spark.sql('USE CATALOG prod;')

# ========================================
# CELL 2
# ========================================

# Cell 1: Create medication grouper category map
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_med_grouper_category_map
USING DELTA
AS
WITH id_map AS (
  SELECT * FROM VALUES
    -- IRON SUPPLEMENTATION
    ('IRON_SUPPLEMENTATION', 103100859),
    ('IRON_SUPPLEMENTATION', 1060000647),
    ('IRON_SUPPLEMENTATION', 1060000646),
    ('IRON_SUPPLEMENTATION', 1060000650),

    -- PPI USE
    ('PPI_USE', 1030101319),
    ('PPI_USE', 103101245),
    ('PPI_USE', 1060033801),
    ('PPI_USE', 1060038401),

    -- NSAID/ASA USE
    ('NSAID_ASA_USE', 1060031001),
    ('NSAID_ASA_USE', 1060000523),
    ('NSAID_ASA_USE', 1060015801),
    ('NSAID_ASA_USE', 1060046101),
    ('NSAID_ASA_USE', 1060028401),
    ('NSAID_ASA_USE', 1060028501),
    ('NSAID_ASA_USE', 103100721),

    -- STATIN USE
    ('STATIN_USE', 1232000017),('STATIN_USE', 1765734),('STATIN_USE', 1765928),
    ('STATIN_USE', 1765229),('STATIN_USE', 1765232),('STATIN_USE', 1765246),('STATIN_USE', 1765249),
    ('STATIN_USE', 1765260),('STATIN_USE', 1765261),('STATIN_USE', 1765262),('STATIN_USE', 1765263),('STATIN_USE', 1765264),
    ('STATIN_USE', 1754575),('STATIN_USE', 1754577),('STATIN_USE', 1754603),('STATIN_USE', 1754584),
    ('STATIN_USE', 1754588),('STATIN_USE', 1754592),('STATIN_USE', 1754593),('STATIN_USE', 1754594),('STATIN_USE', 1754595),
    ('STATIN_USE', 1765241),('STATIN_USE', 1765239),('STATIN_USE', 1765240),('STATIN_USE', 1765253),('STATIN_USE', 1765254),
    ('STATIN_USE', 1030107485),('STATIN_USE', 103103567),('STATIN_USE', 103103088),('STATIN_USE', 105100537),('STATIN_USE', 1060048301),

    -- METFORMIN USE
    ('METFORMIN_USE', 103101190),('METFORMIN_USE', 1060036201),('METFORMIN_USE', 1060040001),
    ('METFORMIN_USE', 1060000704),('METFORMIN_USE', 1060040101),('METFORMIN_USE', 1765323),
    ('METFORMIN_USE', 1765276),('METFORMIN_USE', 1765279),('METFORMIN_USE', 1765282),('METFORMIN_USE', 1765284),
    ('METFORMIN_USE', 1765289),('METFORMIN_USE', 1765292),('METFORMIN_USE', 1765320),('METFORMIN_USE', 1765328),
    ('METFORMIN_USE', 1765331),('METFORMIN_USE', 1765334),('METFORMIN_USE', 1765336),('METFORMIN_USE', 1765339)
  AS t(category_key, grouper_id)
)
SELECT
  m.category_key,
  gi.GROUPER_ID,
  gi.GROUPER_NAME
FROM id_map m
LEFT JOIN clarity.grouper_items gi
  ON gi.GROUPER_ID = m.grouper_id;
''')

# ========================================
# CELL 3
# ========================================

# Cell 2: Create medication ID category map (for medications not in groupers)
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_med_id_category_map
USING DELTA
AS
SELECT
  UPPER(TRIM(CATEGORY_KEY))   AS CATEGORY_KEY,
  CAST(MEDICATION_ID AS BIGINT) AS MEDICATION_ID,
  UPPER(TRIM(GEN_NAME))       AS GEN_NAME,
  SOURCE_TAG,
  NOTES
FROM VALUES
  -- ===== LAXATIVE_USE =====
  ('LAXATIVE_USE', 229101, 'PRUCALOPRIDE 2 MG TABLET',                  'regex_v1_vetted', '5-HT4 agonist'),
  ('LAXATIVE_USE', 229100, 'PRUCALOPRIDE 1 MG TABLET',                  'regex_v1_vetted', '5-HT4 agonist'),
  ('LAXATIVE_USE', 179778, 'PSYLLIUM HUSK 3.5 GRAM ORAL POWDER PACKET', 'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 221195, 'PSYLLIUM HUSK 0.4 GRAM CAPSULE',            'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 87600,  'PSYLLIUM HUSK 0.52 GRAM CAPSULE',           'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 81953,  'CALCIUM POLYCARBOPHIL 500 MG CHEWABLE TABLET','regex_v1_vetted','bulk/fiber'),
  ('LAXATIVE_USE', 80942,  'CALCIUM POLYCARBOPHIL 625 MG TABLET',       'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 80695,  'METHYLCELLULOSE (LAXATIVE) 500 MG TABLET',  'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 197631, 'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 39610,  'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 93851,  'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 80411,  'LACTULOSE 10 GRAM ORAL PACKET',             'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 82328,  'LACTULOSE 20 GRAM ORAL PACKET',             'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 197575, 'LACTULOSE 20 GRAM/30 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 95550,  'LUBIPROSTONE 24 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 178676, 'LUBIPROSTONE 8 MCG CAPSULE',                'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 70603,  'LUBIPROSTONE 24 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 214922, 'PSYLLIUM HUSK 3 GRAM/3 GRAM ORAL POWDER',   'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 202759, 'LINACLOTIDE 145 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 202760, 'LINACLOTIDE 290 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 221253, 'LINACLOTIDE 72 MCG CAPSULE',                'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 221416, 'PLECANATIDE 3 MG TABLET',                   'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 222935, 'NALDEMEDINE 0.2 MG TABLET',                 'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 212755, 'NALOXEGOL 12.5 MG TABLET',                  'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 212698, 'NALOXEGOL 25 MG TABLET',                    'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 219542, 'METHYLNALTREXONE 150 MG TABLET',            'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 178822, 'METHYLNALTREXONE 12 MG/0.6 ML SC SOLUTION', 'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 197993, 'METHYLNALTREXONE 12 MG/0.6 ML SC SYRINGE',  'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 197990, 'METHYLNALTREXONE 8 MG/0.4 ML SC SYRINGE',   'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 2572,   'DOCUSATE SODIUM 100 MG TABLET',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 212170, 'DOCUSATE SODIUM 50 MG CAPSULE',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2085,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2568,   'DOCUSATE SODIUM 50 MG CAPSULE',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2567,   'DOCUSATE SODIUM 250 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 1815,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 4372,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 11878,  'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 39752,  'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
  ('LAXATIVE_USE', 30015,  'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
  ('LAXATIVE_USE', 7159,   'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
-- Polyethylene glycol products (very commonly used)
('LAXATIVE_USE', 25424, 'POLYETHYLENE GLYCOL 3350 17 GRAM ORAL POWDER PACKET', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 24984, 'POLYETHYLENE GLYCOL 3350 17 GRAM/DOSE ORAL POWDER', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 156129, 'POLYETHYLENE GLYCOL 3350 ORAL', 'discovery_v2', 'osmotic'),
-- Bisacodyl products
('LAXATIVE_USE', 1080, 'BISACODYL 10 MG RECTAL SUPPOSITORY', 'discovery_v2', 'stimulant'),
('LAXATIVE_USE', 13632, 'BISACODYL 5 MG TABLET,DELAYED RELEASE', 'discovery_v2', 'stimulant'),
('LAXATIVE_USE', 83533, 'BISACODYL 5 MG TABLET', 'discovery_v2', 'stimulant'),
-- Magnesium products
('LAXATIVE_USE', 79944, 'MAGNESIUM HYDROXIDE 400 MG/5 ML ORAL SUSPENSION', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 4711, 'MAGNESIUM CITRATE ORAL SOLUTION', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 155118, 'MAGNESIUM CITRATE ORAL', 'discovery_v2', 'osmotic'),
-- Senna combinations (beyond what you have)
('LAXATIVE_USE', 40926, 'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TABLET', 'discovery_v2', 'stimulant combo'),
-- Alvimopan (opioid antagonist for postop ileus)
('LAXATIVE_USE', 179661, 'ALVIMOPAN 12 MG CAPSULE', 'discovery_v2', 'peripherally acting opioid antagonist'),
-- Add these to ANTIDIARRHEAL_USE
('ANTIDIARRHEAL_USE', 4560, 'LOPERAMIDE 2 MG CAPSULE', 'discovery_v2', 'antimotility'),
('ANTIDIARRHEAL_USE', 4562, 'LOPERAMIDE 2 MG TABLET', 'discovery_v2', 'antimotility'),
('ANTIDIARRHEAL_USE', 189801, 'RIFAXIMIN 550 MG TABLET', 'discovery_v2', 'antibiotic for IBS-D/HE'),
('ANTIDIARRHEAL_USE', 189817, 'RIFAXIMIN 550 MG TABLET (XIFAXAN)', 'discovery_v2', 'antibiotic for IBS-D/HE'),
('ANTIDIARRHEAL_USE', 81342, 'BISMUTH SUBSALICYLATE 262 MG CHEWABLE TABLET', 'discovery_v2', 'bismuth compound'),
('ANTIDIARRHEAL_USE', 80560, 'BISMUTH SUBSALICYLATE 262 MG/15 ML ORAL SUSPENSION', 'discovery_v2', 'bismuth compound'),
('ANTIDIARRHEAL_USE', 89923, 'ALOSETRON 0.5 MG TABLET', 'discovery_v2', '5-HT3 antagonist for IBS-D'),
('ANTIDIARRHEAL_USE', 79963, 'ALOSETRON 1 MG TABLET', 'discovery_v2', '5-HT3 antagonist for IBS-D'),

  -- ===== ANTIDIARRHEAL_USE =====
  ('ANTIDIARRHEAL_USE', 2516,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG TABLET',             'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 88489,  'CHOLESTYRAMINE-ASPARTAME 4 GRAM ORAL POWDER FOR SUSP IN A PACKET', 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 183910, 'COLESEVELAM 3.75 GRAM ORAL POWDER PACKET',                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 16388,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 37482,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 80288,  'COLESEVELAM 625 MG TABLET',                                 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176079, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 82389,  'COLESTIPOL 5 GRAM ORAL PACKET',                             'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 21994,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216207, 'ELUXADOLINE 75 MG TABLET',                                  'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 9589,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM ORAL POWDER',            'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 89845,  'COLESTIPOL 5 GRAM ORAL GRANULES',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176078, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 28857,  'COLESEVELAM 625 MG TABLET',                                 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216198, 'ELUXADOLINE 100 MG TABLET',                                 'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 82555,  'COLESTIPOL 1 GRAM TABLET',                                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 9588,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM POWDER FOR SUSP IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176054, 'CHOLESTYRAMINE-ASPARTAME 4 GRAM ORAL POWDER',               'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 13898,  'COLESTIPOL 1 GRAM TABLET',                                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216086, 'ELUXADOLINE 75 MG TABLET',                                  'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 2515,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG/5 ML ORAL LIQUID',   'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 216087, 'ELUXADOLINE 100 MG TABLET',                                 'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 183915, 'COLESEVELAM 3.75 GRAM ORAL POWDER PACKET',                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 252231, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 252230, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216374, 'ELUXADOLINE ORAL',                                          'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 4553,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG TABLET',             'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 154828, 'DIPHENOXYLATE-ATROPINE ORAL',                               'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 146632, 'CHOLESTYRAMINE-ASPARTAME ORAL',                             'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147147, 'COLESTIPOL ORAL',                                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 6767,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM ORAL POWDER',            'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 160917, 'CHOLESTYRAMINE (WITH SUGAR) ORAL',                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 166274, 'COLESEVELAM ORAL',                                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 89844,  'COLESTIPOL 7.5 GRAM ORAL PACKET',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147146, 'COLESTIPOL ORAL',                                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 148928, 'DIPHENOXYLATE-ATROPINE ORAL',                               'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 12057,  'COLESTIPOL 7.5 GRAM ORAL PACKET',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 197067, 'CHOLESTYRAMINE (WITH SUGAR) ORAL',                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 146631, 'CHOLESTYRAMINE ORAL',                                       'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 6766,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM POWDER FOR SUSP IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147144, 'COLESEVELAM ORAL',                                          'regex_v1_vetted', 'bile-acid binder'),

  -- ===== ANTISPASMODIC_USE =====
  ('ANTISPASMODIC_USE', 1738,    'CHLORDIAZEPOXIDE-CLIDINIUM 5 MG-2.5 MG CAPSULE', 'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 2418,    'DICYCLOMINE 10 MG CAPSULE',                      'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 205840,  'DICYCLOMINE 10 MG/5 ML ORAL SOLUTION',           'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 2420,    'DICYCLOMINE 20 MG TABLET',                       'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 144535,  'DICYCLOMINE ORAL',                               'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 148732,  'DICYCLOMINE ORAL',                               'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 4434,    'CHLORDIAZEPOXIDE-CLIDINIUM 5 MG-2.5 MG CAPSULE', 'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 146851,  'CHLORDIAZEPOXIDE-CLIDINIUM ORAL',                'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 154497,  'CHLORDIAZEPOXIDE-CLIDINIUM ORAL',                'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 254015,  'DICYCLOMINE 40 MG TABLET',                       'regex_v1_vetted', 'GI anticholinergic'),

  -- ===== B12_FOLATE_USE =====
  ('B12_FOLATE_USE', 4396,   'LEUCOVORIN CALCIUM 15 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 4397,   'LEUCOVORIN CALCIUM 25 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 212837, 'LEUCOVORIN 4 MG-PYRIDOXAL PHOSPHATE 50 MG-MECOBALAMIN 2 MG TABLET','regex_v1_vetted','combo w/ folinic acid & mecobalamin'),
  ('B12_FOLATE_USE', 4398,   'LEUCOVORIN CALCIUM 5 MG TABLET',                       'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 4395,   'LEUCOVORIN CALCIUM 10 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 242499, 'METHYLFOLATE CALCIUM 25,000 MCG DFE-MECOBALAMIN 2,000 MCG CAPSULE','regex_v1_vetted','high-dose methylfolate + mecobalamin'),
  ('B12_FOLATE_USE', 243320, 'B12 5,000 MCG-METHYLFOLATE 1,360 MCG DFE-B6 2.5 MG CHEWABLE TABLET','regex_v1_vetted','high-dose B12 + methylfolate combo'),
  ('B12_FOLATE_USE', 154414, 'LEUCOVORIN CALCIUM ORAL',                             'regex_v1_vetted', 'folinic acid oral'),
  ('B12_FOLATE_USE', 15370,  'LEUCOVORIN CALCIUM 10 MG/ML INJECTION SOLUTION',      'regex_v1_vetted', 'folinic acid injection'),
  ('B12_FOLATE_USE', 216354, 'COQ10-ALA-RESVERATROL-LEUCOVORIN-B6-MECOBALAMIN-VITC-D3 ORAL','regex_v1_vetted','multi-ingredient incl. leucovorin & mecobalamin'),
  ('B12_FOLATE_USE', 4394,   'LEUCOVORIN CALCIUM 50 MG SOLUTION FOR INJECTION',     'regex_v1_vetted', 'folinic acid injection'),
  ('B12_FOLATE_USE', 4392,   'LEUCOVORIN CALCIUM 100 MG SOLUTION FOR INJECTION',    'regex_v1_vetted', 'folinic acid injection'),

-- ===== IBD_MED_USE =====
('IBD_MEDICATIONS', 6496, 'PREDNISONE 20 MG TABLET', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 6494, 'PREDNISONE 10 MG TABLET', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 14887, 'METHYLPREDNISOLONE 4 MG TABLETS IN A DOSE PACK', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 82000, 'BUDESONIDE DR - ER 3 MG CAPSULE,DELAYED,EXTENDED RELEASE', 'discovery_v2', 'GI-specific steroid'),
('IBD_MEDICATIONS', 167329, 'MESALAMINE 1.2 GRAM TABLET,DELAYED RELEASE', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 94087, 'MESALAMINE 1,000 MG RECTAL SUPPOSITORY', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 10535, 'MESALAMINE 4 GRAM/60 ML ENEMA', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 7562, 'SULFASALAZINE 500 MG TABLET', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 9183, 'AZATHIOPRINE 50 MG TABLET', 'discovery_v2', 'immunosuppressant'),
('IBD_MEDICATIONS', 4973, 'METHOTREXATE SODIUM 2.5 MG TABLET', 'discovery_v2', 'immunosuppressant'),
('IBD_MEDICATIONS', 80729, 'INFLIXIMAB 100 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'anti-TNF biologic'),
('IBD_MEDICATIONS', 225598, 'ADALIMUMAB 40 MG/0.4 ML SUBCUTANEOUS PEN KIT', 'discovery_v2', 'anti-TNF biologic'),
('IBD_MEDICATIONS', 209779, 'VEDOLIZUMAB 300 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'α4β7 integrin blocker'),
('IBD_MEDICATIONS', 184073, 'USTEKINUMAB 90 MG/ML SUBCUTANEOUS SYRINGE', 'discovery_v2', 'IL-12/23 inhibitor'),
('IBD_MEDICATIONS', 217804, 'TOFACITINIB ER 11 MG TABLET,EXTENDED RELEASE 24 HR', 'discovery_v2', 'JAK inhibitor'),

-- ===== HEMORRHOID/RECTAL_MED_USE =====
('HEMORRHOID_RECTAL_MEDS', 3738, 'HYDROCORTISONE ACETATE 25 MG RECTAL SUPPOSITORY', 'discovery_v2', 'rectal steroid'),
('HEMORRHOID_RECTAL_MEDS', 6589, 'HYDROCORTISONE 1 %-PRAMOXINE 1 % RECTAL FOAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 77665, 'HYDROCORTISONE-PRAMOXINE 2.5 %-1 % RECTAL CREAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 80624, 'HYDROCORTISONE 1 %-PRAMOXINE 1 % RECTAL FOAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 82892, 'HYDROCORTISONE-PRAMOXINE 1 %-1 % RECTAL CREAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 10210, 'HYDROCORTISONE 100 MG/60 ML ENEMA', 'discovery_v2', 'rectal steroid'),
('HEMORRHOID_RECTAL_MEDS', 35164, 'LIDOCAINE 3 %-HYDROCORTISONE 0.5 % RECTAL CREAM', 'discovery_v2', 'anesthetic combo'),

-- ===== GI_BLEEDING_MED_USE =====
('GI_BLEEDING_MEDS', 230527, 'TRANEXAMIC ACID 1,000 MG/100 ML IV PIGGYBACK', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 196970, 'TRANEXAMIC ACID 1,000 MG/10 ML INTRAVENOUS SOLUTION', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 190130, 'TRANEXAMIC ACID 650 MG TABLET', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 211947, 'VASOPRESSIN 20 UNIT/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'vasoconstrictor'),
('GI_BLEEDING_MEDS', 25122, 'OCTREOTIDE ACETATE 100 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'somatostatin analog'),
('GI_BLEEDING_MEDS', 25121, 'OCTREOTIDE ACETATE 50 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'somatostatin analog'),

-- ===== CHRONIC_OPIOD_MED_USE =====
('CHRONIC_OPIOID_USE', 3037, 'FENTANYL 50 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 34505, 'HYDROCODONE 5 MG-ACETAMINOPHEN 325 MG TABLET', 'discovery_v2', 'opioid combo'),
('CHRONIC_OPIOID_USE', 10814, 'OXYCODONE 5 MG TABLET', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 301537, 'MORPHINE 4 MG/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 14632, 'TRAMADOL 50 MG TABLET', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 27905, 'FENTANYL 25 MCG/HR TRANSDERMAL PATCH', 'discovery_v2', 'long-acting opioid'),
('CHRONIC_OPIOID_USE', 20920, 'MORPHINE ER 15 MG TABLET,EXTENDED RELEASE', 'discovery_v2', 'long-acting opioid'),
('CHRONIC_OPIOID_USE', 211843, 'OXYCODONE ER 10 MG TABLET,CRUSH RESISTANT,EXTENDED RELEASE', 'discovery_v2', 'long-acting opioid'),

-- ===== BROAD_SPECTRUM_ANTIBIOTIC_USE =====
('BROAD_SPECTRUM_ANTIBIOTICS', 9500, 'CEPHALEXIN 500 MG CAPSULE', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 25119, 'CIPROFLOXACIN 500 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 82091, 'LEVOFLOXACIN 500 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 87765, 'MOXIFLOXACIN 400 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 177634, 'CEFAZOLIN 2 GRAM/20 ML INTRAVENOUS SYRINGE', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 79742, 'CEFTRIAXONE 1 GRAM SOLUTION FOR INJECTION', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 9621, 'CLINDAMYCIN HCL 300 MG CAPSULE', 'discovery_v2', 'lincosamide'),
('BROAD_SPECTRUM_ANTIBIOTICS', 83455, 'PIPERACILLIN-TAZOBACTAM 3.375 GRAM IV PIGGYBACK', 'discovery_v2', 'beta-lactam combo'),
('BROAD_SPECTRUM_ANTIBIOTICS', 83077, 'ERTAPENEM 1 GRAM SOLUTION FOR INJECTION', 'discovery_v2', 'carbapenem'),
('BROAD_SPECTRUM_ANTIBIOTICS', 80713, 'MEROPENEM 1 GRAM INTRAVENOUS SOLUTION', 'discovery_v2', 'carbapenem'),

-- ===== HORMON_THERAPY_USE =====
('HORMONE_THERAPY', 82101, 'ESTRADIOL 0.01% VAGINAL CREAM', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 9967, 'ESTRADIOL 1 MG TABLET', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 80522, 'CONJUGATED ESTROGENS 0.625 MG/GRAM VAGINAL CREAM', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 11498, 'TAMOXIFEN 20 MG TABLET', 'discovery_v2', 'SERM'),
('HORMONE_THERAPY', 81433, 'RALOXIFENE 60 MG TABLET', 'discovery_v2', 'SERM'),
('HORMONE_THERAPY', 7784, 'TESTOSTERONE CYPIONATE 200 MG/ML INTRAMUSCULAR OIL', 'discovery_v2', 'testosterone'),
('HORMONE_THERAPY', 194705, 'TESTOSTERONE 1.62% TRANSDERMAL GEL', 'discovery_v2', 'testosterone'),

-- ===== CHEMOTHERAPY_AGENT_USE =====
('CHEMOTHERAPY_AGENTS', 78342, 'FLUOROURACIL 5 % TOPICAL CREAM', 'discovery_v2', '5-FU'),
('CHEMOTHERAPY_AGENTS', 79057, 'CAPECITABINE 500 MG TABLET', 'discovery_v2', '5-FU prodrug'),
('CHEMOTHERAPY_AGENTS', 80994, 'CAPECITABINE 150 MG TABLET', 'discovery_v2', '5-FU prodrug'),
('CHEMOTHERAPY_AGENTS', 89931, 'BEVACIZUMAB 25 MG/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'VEGF inhibitor'),
('CHEMOTHERAPY_AGENTS', 10631, 'MITOMYCIN 40 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'alkylating agent')

AS t(category_key, medication_id, gen_name, source_tag, notes);
''')

# ========================================
# CELL 4
# ========================================

# Cell 3: Create unpivoted inpatient medications table
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds_unpivoted AS

WITH
cohort AS (
  SELECT
    CAST(PAT_ID AS STRING)          AS PAT_ID,
    END_DTTM
  FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
),

map_med AS (
  SELECT
    UPPER(TRIM(CATEGORY_KEY))       AS CATEGORY,
    CAST(MEDICATION_ID AS BIGINT)   AS MEDICATION_ID,
    UPPER(TRIM(GEN_NAME))           AS GEN_NAME
  FROM {trgt_cat}.clncl_ds.herald_test_train_med_id_category_map
),

map_grp AS (
  SELECT
    UPPER(TRIM(CATEGORY_KEY))       AS CATEGORY,
    CAST(GROUPER_ID AS BIGINT)      AS GROUPER_ID,
    UPPER(TRIM(GROUPER_NAME))       AS GROUPER_NAME
  FROM {trgt_cat}.clncl_ds.herald_test_train_med_grouper_category_map
),

-- Compiled membership: site's link GROUPER → MEDICATION
grp_med_members AS (
  SELECT DISTINCT
    CAST(itm.GROUPER_ID AS BIGINT)                AS GROUPER_ID,
    CAST(med.MEDICATION_ID AS BIGINT)             AS MEDICATION_ID
  FROM map_grp mg
  JOIN clarity.grouper_items itm
    ON itm.GROUPER_ID = mg.GROUPER_ID
  JOIN clarity.grouper_compiled_rec_list rec
    ON rec.base_grouper_id = itm.GROUPER_ID
  JOIN clarity.clarity_medication med
    ON med.MEDICATION_ID = rec.grouper_records_numeric_id
),

-- Inpatient MAR data (medications actually given)
-- CRITICAL: Constrained by clarity_cur data availability from 2021-07-01
orders_mar AS (
  SELECT
    CAST(ome.PAT_ID AS STRING)                    AS PAT_ID,
    CAST(ome.MEDICATION_ID AS BIGINT)             AS MEDICATION_ID,
    UPPER(TRIM(ome.GENERIC_NAME))                 AS RAW_GENERIC,
    CAST(mar.TAKEN_TIME AS TIMESTAMP)             AS TAKEN_TIME,
    CAST(ome.ORDER_MED_ID AS BIGINT)              AS ORDER_MED_ID
  FROM clarity_cur.order_med_enh ome
  JOIN prod.clarity_cur.mar_admin_info_enh mar
    ON mar.ORDER_MED_ID = ome.ORDER_MED_ID
  WHERE ome.ORDERING_MODE_C = 2                   -- Inpatient orders only
    AND mar.TAKEN_TIME IS NOT NULL
    AND DATE(mar.TAKEN_TIME) >= DATE('2021-07-01')  -- Data availability cutoff
    AND UPPER(TRIM(mar.ACTION)) IN (
      'GIVEN','PATIENT/FAMILY ADMIN','GIVEN-SEE OVERRIDE',
      'ADMIN BY ANOTHER CLINICIAN (COMMENT)','NEW BAG','BOLUS','PUSH',
      'STARTED BY ANOTHER CLINICIAN','BAG SWITCHED',
      'CLINIC SAMPLE ADMINISTERED','APPLIED','FEEDING STARTED',
      'ACKNOWLEDGED','CONTRAST GIVEN','NEW BAG-SEE OVERRIDE',
      'BOLUS FROM BAG'
    )
),

-- MedID map hits
hits_med AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    mm.CATEGORY,
    om.MEDICATION_ID,
    om.RAW_GENERIC,
    om.TAKEN_TIME
  FROM cohort c
  LEFT JOIN orders_mar om
    ON om.PAT_ID = c.PAT_ID
   AND DATE(om.TAKEN_TIME) <  c.END_DTTM
   AND DATE(om.TAKEN_TIME) >= GREATEST(
         ADD_MONTHS(c.END_DTTM, -24),
         DATE('2021-07-01')  -- Respect data availability
       )
  JOIN map_med mm
    ON mm.MEDICATION_ID = om.MEDICATION_ID
),

-- Grouper map hits
hits_grp AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    mg.CATEGORY,
    om.MEDICATION_ID,
    om.RAW_GENERIC,
    om.TAKEN_TIME
  FROM cohort c
  LEFT JOIN orders_mar om
    ON om.PAT_ID = c.PAT_ID
   AND DATE(om.TAKEN_TIME) <  c.END_DTTM
   AND DATE(om.TAKEN_TIME) >= GREATEST(
         ADD_MONTHS(c.END_DTTM, -24),
         DATE('2021-07-01')  -- Respect data availability
       )
  JOIN grp_med_members gm
    ON gm.MEDICATION_ID = om.MEDICATION_ID
  JOIN map_grp mg
    ON mg.GROUPER_ID = gm.GROUPER_ID
),

hits_all AS (
  SELECT * FROM hits_med
  UNION ALL
  SELECT * FROM hits_grp
),

ranked AS (
  SELECT
    PAT_ID,
    END_DTTM,
    CATEGORY,
    MEDICATION_ID,
    RAW_GENERIC,
    TAKEN_TIME,
    DATEDIFF(END_DTTM, DATE(TAKEN_TIME)) AS DAYS_SINCE_MED,
    ROW_NUMBER() OVER (
      PARTITION BY PAT_ID, END_DTTM, CATEGORY, MEDICATION_ID, DATE(TAKEN_TIME)
      ORDER BY TAKEN_TIME DESC
    ) AS rn
  FROM hits_all
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  r.CATEGORY,
  r.MEDICATION_ID,
  r.RAW_GENERIC,
  r.TAKEN_TIME,
  r.DAYS_SINCE_MED
FROM cohort c
LEFT JOIN ranked r
  ON r.PAT_ID   = c.PAT_ID
 AND r.END_DTTM = c.END_DTTM
WHERE r.rn = 1 OR r.rn IS NULL;
''')

# ========================================
# CELL 5
# ========================================

# Cell 4: Create final pivoted inpatient medications table using window functions
# This transforms unpivoted inpatient MAR data into modeling features
# All features prefixed with "inp_" to distinguish from outpatient

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds AS

WITH
  cohort AS (
    SELECT CAST(PAT_ID AS STRING) AS PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
  ),

  unpvt AS (
    SELECT CAST(PAT_ID AS STRING) AS PAT_ID,
           END_DTTM,
           UPPER(TRIM(CATEGORY)) AS CATEGORY,
           CAST(DAYS_SINCE_MED AS INT) AS DAYS_SINCE_MED
    FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds_unpivoted
    WHERE CATEGORY IS NOT NULL
  ),

  -- Use window functions to get most recent medication per category
  ranked_meds AS (
    SELECT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      DAYS_SINCE_MED,
      ROW_NUMBER() OVER (
        PARTITION BY PAT_ID, END_DTTM, CATEGORY 
        ORDER BY DAYS_SINCE_MED ASC
      ) AS rn_most_recent
    FROM unpvt
  ),

  -- Get most recent (minimum days_since) per category
  most_recent_meds AS (
    SELECT 
      PAT_ID,
      END_DTTM,
      CATEGORY,
      DAYS_SINCE_MED
    FROM ranked_meds
    WHERE rn_most_recent = 1
  ),

  -- Count medications per category using window functions
  med_counts AS (
    SELECT DISTINCT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      COUNT(*) OVER (
        PARTITION BY PAT_ID, END_DTTM, CATEGORY
      ) AS med_count
    FROM unpvt
  ),

  -- Create binary flags using window functions
  med_flags AS (
    SELECT DISTINCT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      1 AS has_med_flag
    FROM unpvt
  ),

  -- Pivot most recent days - NOTE: Using "inp_" prefix for inpatient
  pivot_days AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Original categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN DAYS_SINCE_MED END) AS inp_iron_use_days_since,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN DAYS_SINCE_MED END) AS inp_b12_or_folate_use_days_since,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN DAYS_SINCE_MED END) AS inp_laxative_use_days_since,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN DAYS_SINCE_MED END) AS inp_antidiarrheal_use_days_since,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN DAYS_SINCE_MED END) AS inp_antispasmodic_use_days_since,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN DAYS_SINCE_MED END) AS inp_ppi_use_days_since,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN DAYS_SINCE_MED END) AS inp_nsaid_asa_use_days_since,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN DAYS_SINCE_MED END) AS inp_statin_use_days_since,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN DAYS_SINCE_MED END) AS inp_metformin_use_days_since,
      -- New categories for CRC risk
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN DAYS_SINCE_MED END) AS inp_ibd_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN DAYS_SINCE_MED END) AS inp_hemorrhoid_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN DAYS_SINCE_MED END) AS inp_gi_bleed_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN DAYS_SINCE_MED END) AS inp_opioid_use_days_since,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN DAYS_SINCE_MED END) AS inp_broad_abx_days_since,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN DAYS_SINCE_MED END) AS inp_hormone_therapy_days_since,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN DAYS_SINCE_MED END) AS inp_chemo_agents_days_since
    FROM most_recent_meds
    GROUP BY PAT_ID, END_DTTM
  ),

  -- Pivot counts
  pivot_counts AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Original categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN med_count END) AS inp_iron_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN med_count END) AS inp_b12_or_folate_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN med_count END) AS inp_laxative_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN med_count END) AS inp_antidiarrheal_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN med_count END) AS inp_antispasmodic_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN med_count END) AS inp_ppi_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN med_count END) AS inp_nsaid_asa_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN med_count END) AS inp_statin_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN med_count END) AS inp_metformin_use_count_2yr,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN med_count END) AS inp_ibd_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN med_count END) AS inp_hemorrhoid_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN med_count END) AS inp_gi_bleed_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN med_count END) AS inp_opioid_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN med_count END) AS inp_broad_abx_count_2yr,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN med_count END) AS inp_hormone_therapy_count_2yr,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN med_count END) AS inp_chemo_agents_count_2yr
    FROM med_counts
    GROUP BY PAT_ID, END_DTTM
  ),

  -- Pivot flags
  pivot_flags AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Original categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN has_med_flag END) AS inp_iron_use_flag,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN has_med_flag END) AS inp_b12_or_folate_use_flag,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN has_med_flag END) AS inp_laxative_use_flag,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN has_med_flag END) AS inp_antidiarrheal_use_flag,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN has_med_flag END) AS inp_antispasmodic_use_flag,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN has_med_flag END) AS inp_ppi_use_flag,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN has_med_flag END) AS inp_nsaid_asa_use_flag,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN has_med_flag END) AS inp_statin_use_flag,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN has_med_flag END) AS inp_metformin_use_flag,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN has_med_flag END) AS inp_ibd_meds_flag,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN has_med_flag END) AS inp_hemorrhoid_meds_flag,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN has_med_flag END) AS inp_gi_bleed_meds_flag,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN has_med_flag END) AS inp_opioid_use_flag,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN has_med_flag END) AS inp_broad_abx_flag,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN has_med_flag END) AS inp_hormone_therapy_flag,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN has_med_flag END) AS inp_chemo_agents_flag
    FROM med_flags
    GROUP BY PAT_ID, END_DTTM
  )

SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- Original medication features with inp_ prefix
  COALESCE(pf.inp_iron_use_flag, 0) AS inp_iron_use_flag,
  pd.inp_iron_use_days_since,
  COALESCE(pc.inp_iron_use_count_2yr, 0) AS inp_iron_use_count_2yr,

  COALESCE(pf.inp_b12_or_folate_use_flag, 0) AS inp_b12_or_folate_use_flag,
  pd.inp_b12_or_folate_use_days_since,
  COALESCE(pc.inp_b12_or_folate_use_count_2yr, 0) AS inp_b12_or_folate_use_count_2yr,

  COALESCE(pf.inp_laxative_use_flag, 0) AS inp_laxative_use_flag,
  pd.inp_laxative_use_days_since,
  COALESCE(pc.inp_laxative_use_count_2yr, 0) AS inp_laxative_use_count_2yr,

  COALESCE(pf.inp_antidiarrheal_use_flag, 0) AS inp_antidiarrheal_use_flag,
  pd.inp_antidiarrheal_use_days_since,
  COALESCE(pc.inp_antidiarrheal_use_count_2yr, 0) AS inp_antidiarrheal_use_count_2yr,

  COALESCE(pf.inp_antispasmodic_use_flag, 0) AS inp_antispasmodic_use_flag,
  pd.inp_antispasmodic_use_days_since,
  COALESCE(pc.inp_antispasmodic_use_count_2yr, 0) AS inp_antispasmodic_use_count_2yr,

  COALESCE(pf.inp_ppi_use_flag, 0) AS inp_ppi_use_flag,
  pd.inp_ppi_use_days_since,
  COALESCE(pc.inp_ppi_use_count_2yr, 0) AS inp_ppi_use_count_2yr,

  COALESCE(pf.inp_nsaid_asa_use_flag, 0) AS inp_nsaid_asa_use_flag,
  pd.inp_nsaid_asa_use_days_since,
  COALESCE(pc.inp_nsaid_asa_use_count_2yr, 0) AS inp_nsaid_asa_use_count_2yr,

  COALESCE(pf.inp_statin_use_flag, 0) AS inp_statin_use_flag,
  pd.inp_statin_use_days_since,
  COALESCE(pc.inp_statin_use_count_2yr, 0) AS inp_statin_use_count_2yr,

  COALESCE(pf.inp_metformin_use_flag, 0) AS inp_metformin_use_flag,
  pd.inp_metformin_use_days_since,
  COALESCE(pc.inp_metformin_use_count_2yr, 0) AS inp_metformin_use_count_2yr,

  -- New medication category features with inp_ prefix
  COALESCE(pf.inp_ibd_meds_flag, 0) AS inp_ibd_meds_flag,
  pd.inp_ibd_meds_days_since,
  COALESCE(pc.inp_ibd_meds_count_2yr, 0) AS inp_ibd_meds_count_2yr,

  COALESCE(pf.inp_hemorrhoid_meds_flag, 0) AS inp_hemorrhoid_meds_flag,
  pd.inp_hemorrhoid_meds_days_since,
  COALESCE(pc.inp_hemorrhoid_meds_count_2yr, 0) AS inp_hemorrhoid_meds_count_2yr,

  COALESCE(pf.inp_gi_bleed_meds_flag, 0) AS inp_gi_bleed_meds_flag,
  pd.inp_gi_bleed_meds_days_since,
  COALESCE(pc.inp_gi_bleed_meds_count_2yr, 0) AS inp_gi_bleed_meds_count_2yr,

  COALESCE(pf.inp_opioid_use_flag, 0) AS inp_opioid_use_flag,
  pd.inp_opioid_use_days_since,
  COALESCE(pc.inp_opioid_use_count_2yr, 0) AS inp_opioid_use_count_2yr,

  COALESCE(pf.inp_broad_abx_flag, 0) AS inp_broad_abx_flag,
  pd.inp_broad_abx_days_since,
  COALESCE(pc.inp_broad_abx_count_2yr, 0) AS inp_broad_abx_count_2yr,

  COALESCE(pf.inp_hormone_therapy_flag, 0) AS inp_hormone_therapy_flag,
  pd.inp_hormone_therapy_days_since,
  COALESCE(pc.inp_hormone_therapy_count_2yr, 0) AS inp_hormone_therapy_count_2yr,

  COALESCE(pf.inp_chemo_agents_flag, 0) AS inp_chemo_agents_flag,
  pd.inp_chemo_agents_days_since,
  COALESCE(pc.inp_chemo_agents_count_2yr, 0) AS inp_chemo_agents_count_2yr

FROM cohort c
LEFT JOIN pivot_flags pf ON c.PAT_ID = pf.PAT_ID AND c.END_DTTM = pf.END_DTTM
LEFT JOIN pivot_days pd ON c.PAT_ID = pd.PAT_ID AND c.END_DTTM = pd.END_DTTM  
LEFT JOIN pivot_counts pc ON c.PAT_ID = pc.PAT_ID AND c.END_DTTM = pc.END_DTTM;
''')

# ========================================
# CELL 6
# ========================================

# Cell 5: Validate row count matches cohort and examine inpatient medication prevalence
# Important: Inpatient prevalence will be much lower than outpatient (only during hospitalizations)

# Row count validation - CRITICAL CHECK
result = spark.sql(f"""
SELECT 
    COUNT(*) as inpatient_meds_count,
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds
""")

result.show()
assert result.collect()[0]['diff'] == 0, "ERROR: Row count mismatch!"
print("✓ Row count validation passed")

# Validate medication date ranges respect data constraints
date_validation = spark.sql(f'''
SELECT 
  MIN(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME))) as min_days_back,
  MAX(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME))) as max_days_back,
  PERCENTILE(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME)), 0.95) as p95_days_back,
  PERCENTILE(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME)), 0.05) as p05_days_back,
  MIN(DATE(u.TAKEN_TIME)) as earliest_med_date,
  MAX(DATE(u.TAKEN_TIME)) as latest_med_date
FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds_unpivoted u
  ON m.PAT_ID = u.PAT_ID AND m.END_DTTM = u.END_DTTM
WHERE u.TAKEN_TIME IS NOT NULL
''')

print("\n=== MEDICATION DATE RANGE VALIDATION ===")
date_stats = date_validation.collect()[0]
print(f"Days lookback range: {date_stats['min_days_back']} to {date_stats['max_days_back']} days")
print(f"P05 to P95: {date_stats['p05_days_back']} to {date_stats['p95_days_back']} days")
print(f"Medication dates: {date_stats['earliest_med_date']} to {date_stats['latest_med_date']}")
print(f"Expected: max ~730 days (24 months), earliest ~2021-07-01")

# Verify no medications before data availability date
if date_stats['earliest_med_date'] < datetime.date(2021, 7, 1):
    print("⚠ WARNING: Found medications before 2021-07-01 data availability cutoff!")
else:
    print("✓ All medications respect 2021-07-01 data availability constraint")

# Calculate prevalence for all medication categories
df_summary = spark.sql(f'''
SELECT 
  COUNT(*) as total_rows,
  COUNT(DISTINCT PAT_ID) as unique_patients,
  
  -- Original categories prevalence
  ROUND(AVG(inp_iron_use_flag), 4) as inp_iron_prevalence,
  ROUND(AVG(inp_ppi_use_flag), 4) as inp_ppi_prevalence,
  ROUND(AVG(inp_statin_use_flag), 4) as inp_statin_prevalence,
  ROUND(AVG(inp_laxative_use_flag), 4) as inp_laxative_prevalence,
  ROUND(AVG(inp_antidiarrheal_use_flag), 4) as inp_antidiarrheal_prevalence,
  ROUND(AVG(inp_metformin_use_flag), 4) as inp_metformin_prevalence,
  ROUND(AVG(inp_nsaid_asa_use_flag), 4) as inp_nsaid_prevalence,
  ROUND(AVG(inp_b12_or_folate_use_flag), 4) as inp_b12_prevalence,
  ROUND(AVG(inp_antispasmodic_use_flag), 4) as inp_antispasmodic_prevalence,
  
  -- New categories prevalence (expect higher for acute care medications)
  ROUND(AVG(inp_ibd_meds_flag), 4) as inp_ibd_meds_prevalence,
  ROUND(AVG(inp_hemorrhoid_meds_flag), 4) as inp_hemorrhoid_prevalence,
  ROUND(AVG(inp_gi_bleed_meds_flag), 4) as inp_gi_bleed_prevalence,
  ROUND(AVG(inp_opioid_use_flag), 4) as inp_opioid_prevalence,
  ROUND(AVG(inp_broad_abx_flag), 4) as inp_broad_abx_prevalence,
  ROUND(AVG(inp_hormone_therapy_flag), 4) as inp_hormone_prevalence,
  ROUND(AVG(inp_chemo_agents_flag), 4) as inp_chemo_prevalence
FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds
''')

print("\n=== INPATIENT MEDICATION PREVALENCE ===")
print("Note: Lower overall prevalence expected (only during hospitalizations)")
display(df_summary)

# Check association with CRC outcome for key inpatient medications
print("\n=== CHECKING ASSOCIATION WITH CRC OUTCOME (INPATIENT) ===")
outcome_check = spark.sql(f'''
SELECT 
  'Inp Laxatives' as medication_category,
  AVG(CASE WHEN m.inp_laxative_use_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_med,
  AVG(CASE WHEN m.inp_laxative_use_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_med
FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
UNION ALL

SELECT 
  'Inp GI Bleeding Meds' as medication_category,
  AVG(CASE WHEN m.inp_gi_bleed_meds_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.inp_gi_bleed_meds_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
UNION ALL

SELECT 
  'Inp Opioids' as medication_category,
  AVG(CASE WHEN m.inp_opioid_use_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.inp_opioid_use_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM

UNION ALL

SELECT 
  'Inp Broad Spectrum Abx' as medication_category,
  AVG(CASE WHEN m.inp_broad_abx_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.inp_broad_abx_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
''')

display(outcome_check)

# Special validation: Check patients with any hospitalization
hosp_check = spark.sql(f'''
SELECT 
  COUNT(*) as total_observations,
  SUM(CASE WHEN 
    inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
    inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
    inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
    inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
    inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
    inp_chemo_agents_flag > 0 
    THEN 1 ELSE 0 END) as obs_with_any_inp_med,
  ROUND(100.0 * SUM(CASE WHEN 
    inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
    inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
    inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
    inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
    inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
    inp_chemo_agents_flag > 0 
    THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_with_hospitalization
FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds
''')

print("\n=== HOSPITALIZATION CHECK ===")
print("Patients with any inpatient medication (proxy for hospitalization):")
display(hosp_check)

# ========================================
# CELL 7
# ========================================

# Cell 6: Comprehensive inpatient medication analysis
# Focus on hospitalization patterns and acute care indicators

inp_summary = spark.sql(f'''
WITH hospitalization_metrics AS (
  SELECT 
    PAT_ID,
    END_DTTM,
    -- Any inpatient medication indicates hospitalization
    CASE WHEN 
      inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
      inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
      inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
      inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
      inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
      inp_chemo_agents_flag > 0 
    THEN 1 ELSE 0 END as was_hospitalized,
    
    -- GI-related hospitalization
    CASE WHEN 
      inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_antispasmodic_use_flag +
      inp_gi_bleed_meds_flag + inp_hemorrhoid_meds_flag > 0
    THEN 1 ELSE 0 END as gi_related_hosp,
    
    -- Count total unique medication categories during hospitalizations
    inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
    inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
    inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
    inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
    inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
    inp_chemo_agents_flag as inpatient_med_diversity
    
  FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds
)
SELECT 
  COUNT(*) as total_observations,
  SUM(was_hospitalized) as obs_with_hospitalization,
  ROUND(AVG(was_hospitalized), 3) as hospitalization_rate,
  SUM(gi_related_hosp) as gi_related_hospitalizations,
  ROUND(AVG(gi_related_hosp), 3) as gi_hosp_rate,
  ROUND(AVG(inpatient_med_diversity), 2) as avg_med_categories_when_hosp,
  
  -- CRC rates by hospitalization status
  ROUND(AVG(CASE WHEN h.was_hospitalized = 1 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_hospitalized,
  ROUND(AVG(CASE WHEN h.was_hospitalized = 0 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_not_hospitalized,
  ROUND(AVG(CASE WHEN h.gi_related_hosp = 1 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_gi_hosp
  
FROM hospitalization_metrics h
JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
  ON h.PAT_ID = c.PAT_ID AND h.END_DTTM = c.END_DTTM
''')

display(inp_summary)
print("\nHospitalization is a strong risk indicator")
print("GI-related hospitalizations show particularly high CRC risk")

# ========================================
# CELL 8
# ========================================

# Cell 7: Analyze acute GI events during hospitalization
# GI bleeding, obstruction patterns, and emergency treatments

acute_gi_analysis = spark.sql(f'''
WITH acute_patterns AS (
  SELECT 
    m.*,
    c.FUTURE_CRC_EVENT,
    
    -- Acute GI bleeding pattern
    CASE WHEN inp_gi_bleed_meds_flag = 1 OR 
              (inp_iron_use_flag = 1 AND inp_ppi_use_flag = 1)
         THEN 1 ELSE 0 END as acute_gi_bleed_pattern,
    
    -- Obstruction/ileus pattern
    CASE WHEN inp_laxative_use_flag = 1 AND inp_opioid_use_flag = 1 AND inp_antispasmodic_use_flag = 1
         THEN 1 ELSE 0 END as obstruction_pattern,
    
    -- Sepsis/severe infection pattern
    CASE WHEN inp_broad_abx_flag = 1 AND inp_opioid_use_flag = 1
         THEN 1 ELSE 0 END as severe_infection_pattern,
    
    -- IBD flare pattern
    CASE WHEN inp_ibd_meds_flag = 1 OR 
              (inp_antidiarrheal_use_flag = 1 AND inp_antispasmodic_use_flag = 1)
         THEN 1 ELSE 0 END as ibd_flare_pattern
         
  FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT 
  'Acute GI Bleeding Pattern' as acute_event,
  SUM(acute_gi_bleed_pattern) as n_events,
  ROUND(AVG(acute_gi_bleed_pattern), 4) as prevalence,
  ROUND(AVG(CASE WHEN acute_gi_bleed_pattern = 1 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_with_event,
  ROUND(AVG(CASE WHEN acute_gi_bleed_pattern = 0 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_without_event,
  ROUND(AVG(CASE WHEN acute_gi_bleed_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN acute_gi_bleed_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2) as risk_ratio
FROM acute_patterns

UNION ALL

SELECT 
  'Obstruction/Ileus Pattern',
  SUM(obstruction_pattern),
  ROUND(AVG(obstruction_pattern), 4),
  ROUND(AVG(CASE WHEN obstruction_pattern = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN obstruction_pattern = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN obstruction_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN obstruction_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM acute_patterns

UNION ALL

SELECT 
  'Severe Infection Pattern',
  SUM(severe_infection_pattern),
  ROUND(AVG(severe_infection_pattern), 4),
  ROUND(AVG(CASE WHEN severe_infection_pattern = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN severe_infection_pattern = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN severe_infection_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN severe_infection_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM acute_patterns

UNION ALL

SELECT 
  'IBD Flare Pattern',
  SUM(ibd_flare_pattern),
  ROUND(AVG(ibd_flare_pattern), 4),
  ROUND(AVG(CASE WHEN ibd_flare_pattern = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN ibd_flare_pattern = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN ibd_flare_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN ibd_flare_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM acute_patterns

ORDER BY risk_ratio DESC
''')

display(acute_gi_analysis)
print("\nAcute GI events during hospitalization are strong CRC predictors")
print("Obstruction patterns show particularly high risk ratios")

# ========================================
# CELL 9
# ========================================

# Cell 8: Direct comparison of inpatient vs outpatient medication patterns
# Understanding care setting differences

setting_comparison = spark.sql(f'''
WITH combined_data AS (
  SELECT 
    o.PAT_ID,
    o.END_DTTM,
    c.FUTURE_CRC_EVENT,
    
    -- Outpatient medications
    o.laxative_use_flag as out_laxative,
    o.iron_use_flag as out_iron,
    o.antidiarrheal_use_flag as out_antidiarrheal,
    o.gi_bleed_meds_flag as out_gi_bleed,
    o.ibd_meds_flag as out_ibd,
    
    -- Inpatient medications
    i.inp_laxative_use_flag as inp_laxative,
    i.inp_iron_use_flag as inp_iron,
    i.inp_antidiarrheal_use_flag as inp_antidiarrheal,
    i.inp_gi_bleed_meds_flag as inp_gi_bleed,
    i.inp_ibd_meds_flag as inp_ibd,
    
    -- Care patterns
    CASE 
      WHEN o.laxative_use_flag = 1 AND i.inp_laxative_use_flag = 1 THEN 'Both settings'
      WHEN o.laxative_use_flag = 1 AND i.inp_laxative_use_flag = 0 THEN 'Outpatient only'
      WHEN o.laxative_use_flag = 0 AND i.inp_laxative_use_flag = 1 THEN 'Inpatient only'
      ELSE 'Neither'
    END as laxative_care_pattern,
    
    CASE 
      WHEN o.iron_use_flag = 1 AND i.inp_iron_use_flag = 1 THEN 'Both settings'
      WHEN o.iron_use_flag = 1 AND i.inp_iron_use_flag = 0 THEN 'Outpatient only'
      WHEN o.iron_use_flag = 0 AND i.inp_iron_use_flag = 1 THEN 'Inpatient only'
      ELSE 'Neither'
    END as iron_care_pattern
    
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds o
  JOIN {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds i
    ON o.PAT_ID = i.PAT_ID AND o.END_DTTM = i.END_DTTM
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
    ON o.PAT_ID = c.PAT_ID AND o.END_DTTM = c.END_DTTM
)
SELECT 
  laxative_care_pattern as care_pattern,
  'Laxatives' as medication,
  COUNT(*) as n_observations,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct_of_cohort,
  ROUND(AVG(FUTURE_CRC_EVENT), 5) as crc_rate,
  ROUND(AVG(FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort), 2) as relative_risk
FROM combined_data
GROUP BY laxative_care_pattern

UNION ALL

SELECT 
  iron_care_pattern,
  'Iron',
  COUNT(*),
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2),
  ROUND(AVG(FUTURE_CRC_EVENT), 5),
  ROUND(AVG(FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort), 2)
FROM combined_data
GROUP BY iron_care_pattern

ORDER BY medication, care_pattern
''')

display(setting_comparison)
print("\nMedications in 'Both settings' indicate severe/chronic conditions")
print("Inpatient-only medications may indicate acute events")

# ========================================
# CELL 10
# ========================================

# Cell 9: Analyze temporal distance from last hospitalization
# Recent hospitalizations are strong risk indicators

hosp_recency = spark.sql(f'''
WITH hospitalization_recency AS (
  SELECT 
    m.*,
    c.FUTURE_CRC_EVENT,
    
    -- Find most recent hospitalization (minimum days since any inpatient med)
    LEAST(
      COALESCE(inp_iron_use_days_since, 9999),
      COALESCE(inp_ppi_use_days_since, 9999),
      COALESCE(inp_laxative_use_days_since, 9999),
      COALESCE(inp_antidiarrheal_use_days_since, 9999),
      COALESCE(inp_gi_bleed_meds_days_since, 9999),
      COALESCE(inp_opioid_use_days_since, 9999),
      COALESCE(inp_broad_abx_days_since, 9999)
    ) as days_since_any_hosp,
    
    -- Most recent GI-related hospitalization
    LEAST(
      COALESCE(inp_laxative_use_days_since, 9999),
      COALESCE(inp_antidiarrheal_use_days_since, 9999),
      COALESCE(inp_gi_bleed_meds_days_since, 9999),
      COALESCE(inp_hemorrhoid_meds_days_since, 9999)
    ) as days_since_gi_hosp
    
  FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
),
recency_bands AS (
  SELECT 
    *,
    CASE 
      WHEN days_since_any_hosp = 9999 THEN 'Never hospitalized'
      WHEN days_since_any_hosp <= 30 THEN '0-30 days'
      WHEN days_since_any_hosp <= 90 THEN '31-90 days'
      WHEN days_since_any_hosp <= 180 THEN '91-180 days'
      WHEN days_since_any_hosp <= 365 THEN '181-365 days'
      ELSE '365+ days'
    END as hosp_recency_band,
    
    CASE 
      WHEN days_since_gi_hosp = 9999 THEN 'No GI hosp'
      WHEN days_since_gi_hosp <= 90 THEN '0-90 days'
      WHEN days_since_gi_hosp <= 365 THEN '91-365 days'
      ELSE '365+ days'
    END as gi_hosp_recency_band
  FROM hospitalization_recency
)
SELECT 
  hosp_recency_band,
  COUNT(*) as n_observations,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct_of_cohort,
  ROUND(AVG(FUTURE_CRC_EVENT), 5) as crc_rate,
  ROUND(AVG(FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort), 2) as relative_risk
FROM recency_bands
GROUP BY hosp_recency_band
ORDER BY 
  CASE hosp_recency_band
    WHEN '0-30 days' THEN 1
    WHEN '31-90 days' THEN 2
    WHEN '91-180 days' THEN 3
    WHEN '181-365 days' THEN 4
    WHEN '365+ days' THEN 5
    WHEN 'Never hospitalized' THEN 6
  END
''')

display(hosp_recency)
print("\nRecent hospitalizations (<90 days) show highest CRC risk")
print("This temporal pattern is crucial for risk stratification")

# ========================================
# CELL 11
# ========================================

# Cell 10: Preview feature importance based on univariate analysis
# Ranks medication features by their association with CRC

feature_importance = spark.sql(f'''
WITH feature_associations AS (
  -- Calculate association metrics for each feature
  SELECT 'out_laxative' as feature, 
         AVG(laxative_use_flag) as prevalence,
         AVG(CASE WHEN laxative_use_flag = 1 THEN c.FUTURE_CRC_EVENT END) as crc_rate_with,
         AVG(CASE WHEN laxative_use_flag = 0 THEN c.FUTURE_CRC_EVENT END) as crc_rate_without
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'out_iron',
         AVG(iron_use_flag),
         AVG(CASE WHEN iron_use_flag = 1 THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN iron_use_flag = 0 THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'out_antidiarrheal',
         AVG(antidiarrheal_use_flag),
         AVG(CASE WHEN antidiarrheal_use_flag = 1 THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN antidiarrheal_use_flag = 0 THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_test_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'inp_gi_bleed_meds',
         AVG(inp_gi_bleed_meds_flag),
         AVG(CASE WHEN inp_gi_bleed_meds_flag = 1 THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN inp_gi_bleed_meds_flag = 0 THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'inp_any_hospitalization',
         AVG(CASE WHEN inp_iron_use_flag + inp_ppi_use_flag + inp_laxative_use_flag + 
                       inp_antidiarrheal_use_flag + inp_gi_bleed_meds_flag + inp_opioid_use_flag > 0 
             THEN 1 ELSE 0 END),
         AVG(CASE WHEN inp_iron_use_flag + inp_ppi_use_flag + inp_laxative_use_flag + 
                       inp_antidiarrheal_use_flag + inp_gi_bleed_meds_flag + inp_opioid_use_flag > 0 
             THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN inp_iron_use_flag + inp_ppi_use_flag + inp_laxative_use_flag + 
                       inp_antidiarrheal_use_flag + inp_gi_bleed_meds_flag + inp_opioid_use_flag = 0 
             THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_test_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT 
  feature,
  ROUND(prevalence, 3) as prevalence,
  ROUND(crc_rate_with, 5) as crc_rate_with_feature,
  ROUND(crc_rate_without, 5) as crc_rate_without_feature,
  ROUND(crc_rate_with / NULLIF(crc_rate_without, 0), 2) as risk_ratio,
  ROUND(crc_rate_with - crc_rate_without, 5) as risk_difference,
  -- Simple importance score based on prevalence and risk ratio
  ROUND((prevalence * (crc_rate_with / NULLIF(crc_rate_without, 0) - 1)), 4) as importance_score
FROM feature_associations
ORDER BY importance_score DESC
''')

display(feature_importance)
print("\nFeature importance preview shows most predictive medication features")
print("Hospitalization and GI symptom medications rank highest")

# ========================================
# CELL 12
# ========================================

# Cell 11: Convert to pandas for detailed statistics
df_spark = spark.sql('''SELECT * FROM dev.clncl_ds.herald_test_train_inpatient_meds''')
df = df_spark.toPandas()

print("Shape:", df.shape)
print("\nNull rates:")
print(df.isnull().sum()/df.shape[0])

# ========================================
# CELL 13
# ========================================

# Cell 12: Calculate mean values for all features
df_check = df.drop(columns=['PAT_ID', 'END_DTTM'], axis=1)
print("Mean values for intpatient medication features:")
print(df_check.mean())

# ========================================
# CELL 14
# ========================================

# Step 1: Load inpatient data and calculate basic statistics

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("INPATIENT MEDICATION FEATURE REDUCTION")
print("="*60)

# Join with outcome data
df_spark = spark.sql("""
    SELECT m.*, c.FUTURE_CRC_EVENT
    FROM dev.clncl_ds.herald_test_train_inpatient_meds m
    JOIN dev.clncl_ds.herald_test_train_final_cohort c
        ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
""")

# Check column names - they should already have inp_ prefix
print("Sample column names:")
print([col for col in df_spark.columns if col not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT']][:5])

# Cache for performance
df_spark.cache()
total_rows = df_spark.count()
baseline_crc_rate = df_spark.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"\nTotal rows: {total_rows:,}")
print(f"Baseline CRC rate: {baseline_crc_rate:.4f}")

# Calculate hospitalization rate (any inpatient medication)
hosp_rate = df_spark.filter(
    (F.col('inp_iron_use_flag') == 1) | 
    (F.col('inp_ppi_use_flag') == 1) | 
    (F.col('inp_laxative_use_flag') == 1) |
    (F.col('inp_opioid_use_flag') == 1)
).count() / total_rows

print(f"Hospitalization rate (any inpatient med): {hosp_rate:.1%}")

# ========================================
# CELL 15
# ========================================

# Step 2: Calculate Risk Ratios for Inpatient Flag Features

flag_features = [col for col in df_spark.columns if '_flag' in col and col.startswith('inp_')]
risk_metrics = []

print(f"\nCalculating risk ratios for {len(flag_features)} flag features...")

for feat in flag_features:
    stats = df_spark.groupBy(feat).agg(
        F.count('*').alias('count'),
        F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
    ).collect()
    
    # Parse results
    stats_dict = {row[feat]: {'count': row['count'], 'crc_rate': row['crc_rate']} for row in stats}
    
    prevalence = stats_dict.get(1, {'count': 0})['count'] / total_rows
    rate_with = stats_dict.get(1, {'crc_rate': 0})['crc_rate']
    rate_without = stats_dict.get(0, {'crc_rate': baseline_crc_rate})['crc_rate']
    risk_ratio = rate_with / (rate_without + 1e-10)
    
    # Handle edge cases for impact calculation
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
print("\nTop 10 features by impact score (prevalence × log risk ratio):")
print(risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']].head(10).to_string())

# ========================================
# CELL 16
# ========================================

# Step 3: Analyze Feature Types and Information Content

print("\nAnalyzing feature types and information content...")

# Separate features by type
flag_features = [col for col in df_spark.columns if '_flag' in col and col.startswith('inp_')]
count_features = [col for col in df_spark.columns if '_count' in col and col.startswith('inp_')]
days_since_features = [col for col in df_spark.columns if '_days_since' in col and col.startswith('inp_')]

print(f"Feature types:")
print(f"  - Flag features: {len(flag_features)}")
print(f"  - Count features: {len(count_features)}")
print(f"  - Days_since features: {len(days_since_features)}")

# For days_since: Create missing_df for compatibility with Step 5
missing_stats = []
for feat in days_since_features:
    # NULL means never had medication - this is information, not missing data
    never_had = df_spark.filter(F.col(feat).isNull()).count() / total_rows
    
    missing_stats.append({
        'feature': feat,
        'missing_rate': never_had,  # Use missing_rate for consistency
        'medication': feat.replace('inp_', '').replace('_days_since', '')
    })

missing_df = pd.DataFrame(missing_stats)
print(f"\nMedications ranked by usage (from days_since nulls):")
print(missing_df.sort_values('missing_rate')[['feature', 'missing_rate']])

# ========================================
# CELL 17
# ========================================

# Step 4: Sample for Mutual Information Calculation

# Take stratified sample for MI calculation
sample_fraction = min(200000 / total_rows, 1.0)  # Larger sample for inpatient

print(f"\nSampling for MI calculation...")
df_sample = df_spark.sampleBy("FUTURE_CRC_EVENT", 
                               fractions={0: sample_fraction, 1: 1.0},  # Keep all positive cases
                               seed=42).toPandas()

print(f"Sampled {len(df_sample):,} rows ({len(df_sample)/total_rows*100:.1f}% of total)")
print(f"Sample CRC rate: {df_sample['FUTURE_CRC_EVENT'].mean():.4f}")

# Calculate MI on sample
from sklearn.feature_selection import mutual_info_classif

feature_cols = [c for c in df_sample.columns 
                if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT'] 
                and c.startswith('inp_')]

print(f"Calculating MI for {len(feature_cols)} features...")
X = df_sample[feature_cols].fillna(-999)
y = df_sample['FUTURE_CRC_EVENT']

mi_scores = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop features by Mutual Information:")
print(mi_df.to_string())

# ========================================
# CELL 18
# ========================================

# Step 5: Feature Selection Logic for Inpatient 

# Merge all metrics (now using correct missing_df)
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

# Inpatient-specific critical features
MUST_KEEP = [
    'inp_gi_bleed_meds_flag',       # GI bleeding in hospital - critical
    'inp_iron_use_flag',             # Acute bleeding management
    'inp_laxative_use_flag',         # Post-op/obstruction
    'inp_opioid_use_flag',           # Surgery/pain indicator
    'inp_hemorrhoid_meds_flag',      # Rare but high risk
    'inp_gi_bleed_meds_days_since',  # Temporal pattern important
    'inp_ppi_use_flag',              # Common in hospital
    'inp_broad_abx_flag'             # Infection marker
]

# Note: Hemorrhoid medications kept despite extreme rarity (0.07% vs 0.2% outpatient)
# When hemorrhoids are treated inpatient, they likely represent severe bleeding/thrombosis
# requiring urgent care - making this a clinically significant signal despite low prevalence

# Pre-specified removals for near-zero features
REMOVE = ['inp_b12_or_folate_use_flag', 'inp_b12_or_folate_use_days_since', 
          'inp_b12_or_folate_use_count_2yr',  # Essentially never given inpatient
          'inp_chemo_agents_flag', 'inp_chemo_agents_days_since', 
          'inp_chemo_agents_count_2yr']  # Extremely rare inpatient

# Add features with extremely low signal
for _, row in feature_importance.iterrows():
    feat = row['feature']
    
    # More lenient for inpatient due to lower overall prevalence
    if row.get('missing_rate', 0) > 0.999 and feat not in MUST_KEEP:
        REMOVE.append(feat)
    
    # Remove extremely rare with low risk (adjusted for inpatient)
    if '_flag' in feat:
        # Lower threshold for inpatient
        if (row.get('prevalence', 0) < 0.00001 and row.get('risk_ratio', 1) < 3 and feat not in MUST_KEEP):
            REMOVE.append(feat)

REMOVE = list(set(REMOVE))  # Remove duplicates
print(f"\nRemoving {len(REMOVE)} near-zero signal features")
print(f"Examples of removed features: {REMOVE[:5]}")

feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]
print(f"Features remaining after filtering: {len(feature_importance)}")

# ========================================
# CELL 19
# ========================================

# Step 6: Select Best Feature per Medication for Inpatient

def select_optimal_inpatient_features(df_importance):
    """Select best representation for each inpatient medication"""
    
    selected = []
    
    # Extract medication name (remove inp_ prefix for grouping)
    df_importance['medication'] = df_importance['feature'].str.replace('inp_', '').str.split('_').str[0]
    
    # Group by medication
    for med in df_importance['medication'].unique():
        med_features = df_importance[df_importance['medication'] == med]
        
        # Inpatient-specific selection rules
        if med in ['hemorrhoid']:
            # Keep both flag and recency if available
            for feat in med_features['feature']:
                if '_flag' in feat or '_days_since' in feat:
                    selected.append(feat)
                    
        elif med in ['gi', 'iron', 'laxative', 'opioid']:
            # Critical acute care indicators - keep flag
            flag_features = med_features[med_features['feature'].str.contains('_flag')]
            if not flag_features.empty:
                selected.append(flag_features.nlargest(1, 'mi_score')['feature'].values[0])
            
            # Also keep gi_bleed days_since if it's gi_bleed
            if med == 'gi':
                days_features = med_features[med_features['feature'].str.contains('_days_since')]
                if not days_features.empty:
                    selected.append(days_features.iloc[0]['feature'])
                    
        elif med in ['broad', 'antidiarrheal', 'ppi']:
            # Acute event markers - keep flag only
            flag_features = med_features[med_features['feature'].str.contains('_flag')]
            if not flag_features.empty:
                selected.append(flag_features.nlargest(1, 'mi_score')['feature'].values[0])
                
        else:
            # For chronic meds (statins, metformin), keep best MI feature
            if len(med_features) > 0:
                best_feature = med_features.nlargest(1, 'mi_score')['feature'].values[0]
                selected.append(best_feature)
    
    # Ensure must-keep features are included
    for feat in MUST_KEEP:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    return list(set(selected))  # Remove duplicates

selected_features = select_optimal_inpatient_features(feature_importance)
print(f"\nSelected {len(selected_features)} features after medication-level optimization")
print("\nSample of selected features:")
for feat in sorted(selected_features)[:10]:
    print(f"  - {feat}")

# ========================================
# CELL 20
# ========================================

# Step 7: Create Inpatient-Specific Composite Features and Save

df_final = df_spark

# === INPATIENT-SPECIFIC COMPOSITE FEATURES ===

# 1. Acute GI bleeding pattern (iron + PPI during admission)
df_final = df_final.withColumn('inp_acute_gi_bleeding',
    F.when((F.col('inp_iron_use_flag') == 1) & 
           (F.col('inp_ppi_use_flag') == 1), 1).otherwise(0)
)

# 2. Obstruction/ileus pattern (laxatives + opioids)
df_final = df_final.withColumn('inp_obstruction_pattern',
    F.when((F.col('inp_laxative_use_flag') == 1) & 
           (F.col('inp_opioid_use_flag') == 1), 1).otherwise(0)
)

# 3. Severe infection/sepsis pattern (antibiotics + opioids)
df_final = df_final.withColumn('inp_severe_infection',
    F.when((F.col('inp_broad_abx_flag') == 1) & 
           (F.col('inp_opioid_use_flag') == 1), 1).otherwise(0)
)

# 4. Any hospitalization indicator (critical feature)
df_final = df_final.withColumn('inp_any_hospitalization',
    F.when((F.col('inp_iron_use_flag') == 1) | 
           (F.col('inp_ppi_use_flag') == 1) |
           (F.col('inp_laxative_use_flag') == 1) |
           (F.col('inp_opioid_use_flag') == 1) |
           (F.col('inp_broad_abx_flag') == 1), 1).otherwise(0)
)

# 5. GI-specific hospitalization
df_final = df_final.withColumn('inp_gi_hospitalization',
    F.when((F.col('inp_laxative_use_flag') == 1) | 
           (F.col('inp_antidiarrheal_use_flag') == 1) |
           (F.col('inp_gi_bleed_meds_flag') == 1), 1).otherwise(0)
)

composite_features = [
    'inp_acute_gi_bleeding',
    'inp_obstruction_pattern', 
    'inp_severe_infection',
    'inp_any_hospitalization',
    'inp_gi_hospitalization'
]

# Add composites to selected features
selected_features.extend(composite_features)
selected_features = sorted(list(set(selected_features)))  # Remove duplicates and sort

print(f"\nAdded {len(composite_features)} composite features")
print(f"Final feature count: {len(selected_features)}")

# === PRINT FINAL FEATURE LIST ===
print("\n" + "="*60)
print("FINAL SELECTED INPATIENT FEATURES")
print("="*60)

for i, feat in enumerate(selected_features, 1):
    # Add description for clarity
    if 'hemorrhoid' in feat:
        desc = " [RARE BUT EXTREME RISK]"
    elif 'gi_bleed' in feat:
        desc = " [ACUTE BLEEDING]"
    elif 'any_hospitalization' in feat:
        desc = " [KEY INDICATOR]"
    elif feat in composite_features:
        desc = " [COMPOSITE]"
    elif 'opioid' in feat:
        desc = " [PAIN/SURGERY]"
    elif 'iron' in feat or 'laxative' in feat:
        desc = " [HIGH RISK]"
    elif 'ppi' in feat or 'broad_abx' in feat:
        desc = " [COMMON ACUTE]"
    else:
        desc = ""
    print(f"{i:2d}. {feat:<40} {desc}")

# === SAVE REDUCED DATASET ===
final_columns = ['PAT_ID', 'END_DTTM'] + selected_features
df_reduced = df_final.select(*final_columns)

# Add icd_ prefix to all columns except keys
inp_med_cols = [col for col in df_reduced.columns if col not in ['PAT_ID', 'END_DTTM']]
for col in inp_med_cols:
    df_reduced = df_reduced.withColumnRenamed(col, f'inp_med_{col}' if not col.startswith('inp_med_') else col)

# Write to final table
output_table = 'dev.clncl_ds.herald_test_train_inpatient_meds_reduced'
df_reduced.write.mode('overwrite').saveAsTable(output_table)

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: 48")
print(f"Selected features: {len(selected_features)}")
print(f"Reduction: {(1 - len(selected_features)/48)*100:.1f}%")
print(f"\n✓ Reduced dataset saved to: {output_table}")

# Verify save and check all columns have inp_ prefix
row_count = spark.table(output_table).count()
cols_without_prefix = [c for c in selected_features if not c.startswith('inp_')]

print(f"✓ Verified {row_count:,} rows written to table")
if cols_without_prefix:
    print(f"\n⚠ WARNING: These columns missing 'inp_' prefix: {cols_without_prefix}")
else:
    print("✓ All feature columns have 'inp_' prefix for joining")

# ========================================
# CELL 21
# ========================================

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_test_train_inpatient_meds_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 22
# ========================================

display(df_check)



################################################################################
# V2_Book6_Visit_History
################################################################################

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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_ed_inp_encounters AS
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
FROM {trgt_cat}.clncl_ds.herald_test_train_ed_inp_encounters
""")
validation.show()

# ========================================
# CELL 3
# ========================================

# Cell 2: Create outpatient encounters (including all appointment statuses for no-show tracking)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_op_encounters AS
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
FROM {trgt_cat}.clncl_ds.herald_test_train_op_encounters
""")
validation.show()

# ========================================
# CELL 4
# ========================================

# Cell 3: ED/Inpatient metrics only (much faster)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_ed_inp_metrics AS
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

FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_ed_inp_encounters ei
  ON c.PAT_ID = ei.PAT_ID
GROUP BY c.PAT_ID, c.END_DTTM
""")

# ========================================
# CELL 5
# ========================================

# Cell 4: Outpatient metrics only (separate for performance)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_op_metrics AS
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

FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_op_encounters op
  ON c.PAT_ID = op.PAT_ID
GROUP BY c.PAT_ID, c.END_DTTM
""")

# ========================================
# CELL 6
# ========================================

# Cell 5: Join the two metrics tables (very fast)
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_visit_counts AS
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

FROM {trgt_cat}.clncl_ds.herald_test_train_ed_inp_metrics ei
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_op_metrics om
  ON ei.PAT_ID = om.PAT_ID AND ei.END_DTTM = om.END_DTTM
""")

# ========================================
# CELL 7
# ========================================

# Cell 6: Add recency features using window functions
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_visit_recency AS
WITH all_visits AS (
    -- ED visits
    SELECT 
      c.PAT_ID, c.END_DTTM, 'ED' AS visit_type,
      ei.visit_dt,
      DATEDIFF(c.END_DTTM, DATE(ei.visit_dt)) AS days_since
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
    JOIN {trgt_cat}.clncl_ds.herald_test_train_ed_inp_encounters ei
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
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
    JOIN {trgt_cat}.clncl_ds.herald_test_train_ed_inp_encounters ei
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
    FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
    JOIN {trgt_cat}.clncl_ds.herald_test_train_op_encounters op
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
FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
LEFT JOIN pivoted_recency pr ON c.PAT_ID = pr.PAT_ID AND c.END_DTTM = pr.END_DTTM
""")

# ========================================
# CELL 8
# ========================================

# Cell 7: Create final table with composite features
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_visit_features_final AS
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

FROM {trgt_cat}.clncl_ds.herald_test_train_visit_counts vc
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_visit_recency vr
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
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_test_train_visit_features_final
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
df = spark.sql(f"SELECT * FROM {trgt_cat}.clncl_ds.herald_test_train_visit_features_final").toPandas()

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
df_check = spark.sql(f'''select * from {trgt_cat}.clncl_ds.herald_test_train_visit_features_final''')
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
    FROM {trgt_cat}.clncl_ds.herald_test_train_visit_features_final v
    JOIN {trgt_cat}.clncl_ds.herald_test_train_final_cohort c
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
output_table = f'{trgt_cat}.clncl_ds.herald_test_train_visit_features_reduced'
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

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_test_train_visit_features_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 26
# ========================================

display(df_check)



################################################################################
# V2_Book7_Procedures
################################################################################

# V2_Book7_Procedures
# Functional cells: 20 of 41 code cells (77 total)
# Source: V2_Book7_Procedures.ipynb
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

# Define target catalog for SQL based on the environment variable
trgt_cat = os.environ.get('trgt_cat')

# Use appropriate Spark catalog based on the target category
spark.sql('USE CATALOG prod;')

# ========================================
# CELL 2
# ========================================

# Cell 1
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_proc_category_map
AS
WITH proc_map AS (
    SELECT * FROM VALUES
        -- CT abdomen/pelvis (internal codes)
        ('CT_ABDOMEN_PELVIS', 'CT1000'),  -- CT ABDOMEN PELVIS W CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1002'),  -- CT ABDOMEN PELVIS WO CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1048'),  -- CT CHEST ABDOMEN PELVIS W CONT
        ('CT_ABDOMEN_PELVIS', 'CT1051'),  -- CT CHEST ABDOMEN PELVIS WO CONT
        ('CT_ABDOMEN_PELVIS', 'CT1291'),  -- CTA CHEST ABD PELVIS W AND/OR WO CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1226'),  -- CTA ABD PELVIS W AND/OR WO CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1001'),  -- CT ABDOMEN PELVIS W WO CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1173'),  -- CT PELVIS WO CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1009'),  -- CT ABDOMEN WO CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1008'),  -- CT ABDOMEN W WO CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1004'),  -- CT ABDOMEN W CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1171'),  -- CT PELVIS W CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1003'),  -- CT ABDOMEN PELVIS W WO CHEST W CONT
        ('CT_ABDOMEN_PELVIS', 'CT1053'),  -- CT CHEST ABDOMEN W CONTRAST
        ('CT_ABDOMEN_PELVIS', 'CT1049'),  -- CT CHEST ABDOMEN PELVIS W WO CONT
        ('CT_ABDOMEN_PELVIS', 'CT1055'),  -- CT CHEST ABDOMEN WO CONT
        ('CT_ABDOMEN_PELVIS', 'CT1006'),  -- CT ABDOMEN W WO CONT PELVIS W CONT
        ('CT_ABDOMEN_PELVIS', 'CT1172'),  -- CT PELVIS W WO CONTRAST
        
        -- MRI abdomen/pelvis (internal codes)
        ('MRI_ABDOMEN_PELVIS', 'MR1139'),  -- MRI ABDOMEN W WO CONTRAST
        ('MRI_ABDOMEN_PELVIS', 'MR1284'),  -- MRI PELVIS W WO CONTRAST
        ('MRI_ABDOMEN_PELVIS', 'MR1285'),  -- MRI PELVIS WO CONTRAST
        ('MRI_ABDOMEN_PELVIS', 'MR1140'),  -- MRI ABDOMEN WO CONTRAST
        ('MRI_ABDOMEN_PELVIS', 'MR1136'),  -- MRI ABDOMEN PELVIS W WO CONT
        ('MRI_ABDOMEN_PELVIS', 'MR1137'),  -- MRI ABDOMEN PELVIS WO CONTRAST
        ('MRI_ABDOMEN_PELVIS', 'MR1138'),  -- MRI ABDOMEN W CONTRAST
        ('MRI_ABDOMEN_PELVIS', 'MR9007'),  -- MRI ABDOMEN
        ('MRI_ABDOMEN_PELVIS', 'MR1283'),  -- MRI PELVIS W CONTRAST
        
        -- Upper GI procedures (internal codes)
        ('UPPER_GI_PROC', 'GI1012'),  -- UPPER ENDOSCOPY REPORT
        ('UPPER_GI_PROC', 'GI2'),     -- EGD
        ('UPPER_GI_PROC', 'GI1014'),  -- UPPER ENDOSCOPIC ULTRASOUND REPORT
        ('UPPER_GI_PROC', 'GI17'),    -- ENDOSCOPY, UPPER GI
        
        -- Blood transfusions (internal codes)
        ('BLOOD_TRANSFUSION', 'NUR1169'),  -- TRANSFUSE RED BLOOD CELLS
        ('BLOOD_TRANSFUSION', 'NUR1170'),  -- TRANSFUSE RED BLOOD CELLS IN ML
        ('BLOOD_TRANSFUSION', 'NUR1188'),  -- TRANSFUSE EMERGENCY RELEASE RBC
        ('BLOOD_TRANSFUSION', 'NUR1189'),  -- TRANSFUSE EMERGENCY RELEASE RBC IN ML
        
        -- Anal procedures (internal codes)
        ('ANAL_PROCEDURE', 'PRO105'),  -- ANOSCOPY
        
        -- Hemorrhoid procedures (internal codes)
        ('HEMORRHOID_PROC', 'GI1002'),   -- LIGATION HEMORRHOID
        ('HEMORRHOID_PROC', 'PRO163')    -- HEMORRHOID PROCEDURE
        
        -- Note: Colonoscopy excluded as successful screening already removed from cohort
        -- Note: GI biopsies minimal as successful biopsies would have excluded patient
        
    AS t(CATEGORY, PROC_CODE)
)
SELECT * FROM proc_map
''')

print("Procedure code mapping table created successfully")

# ========================================
# CELL 3
# ========================================

# Cell 2
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_proc_unpivoted AS

WITH
    cohort AS (
        SELECT PAT_ID, END_DTTM
        FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
    ),
    
    -- Data availability constraint
    params AS (
        SELECT CAST('2021-07-01' AS TIMESTAMP) AS data_cutoff
    ),

    proc_filtered AS (
        SELECT
            c.PAT_ID,
            c.END_DTTM,
            m.CATEGORY,
            op.RESULT_TIME AS PROC_DATE,
            op.PROC_CODE,
            op.PROC_NAME
        FROM {trgt_cat}.clncl_ds.herald_test_train_proc_category_map m
        JOIN clarity_cur.order_proc_enh op
          ON op.PROC_CODE = m.PROC_CODE
        JOIN cohort c
          ON c.PAT_ID = op.PAT_ID
        CROSS JOIN params p
        WHERE
            op.RESULT_TIME IS NOT NULL
            AND op.ORDER_STATUS_C IN (5)  -- Completed orders
            AND op.RESULT_TIME <= CAST(c.END_DTTM AS TIMESTAMP)
            AND op.RESULT_TIME >= p.data_cutoff  -- Enforce July 2021 cutoff
            AND op.RESULT_TIME BETWEEN
                 CAST(DATE_SUB(CAST(c.END_DTTM AS DATE), 730) AS TIMESTAMP)
             AND CAST(c.END_DTTM AS TIMESTAMP)
    )

SELECT
    PAT_ID,
    END_DTTM,
    CATEGORY,
    PROC_DATE,
    PROC_CODE,
    PROC_NAME
FROM proc_filtered
""")

# Validate extraction
print("="*70)
print("PROCEDURE EXTRACTION VALIDATION")
print("="*70)

extraction_stats = spark.sql(f"""
SELECT 
    COUNT(*) as total_procedures,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    COUNT(DISTINCT CATEGORY) as unique_categories,
    MIN(PROC_DATE) as earliest_proc,
    MAX(PROC_DATE) as latest_proc
FROM {trgt_cat}.clncl_ds.herald_test_train_proc_unpivoted
""").collect()[0]

print(f"\nTotal procedures captured: {extraction_stats['total_procedures']:,}")
print(f"Unique patients: {extraction_stats['unique_patients']:,}")
print(f"Unique categories: {extraction_stats['unique_categories']}")
print(f"Date range: {extraction_stats['earliest_proc']} to {extraction_stats['latest_proc']}")
print(f"\n✓ All procedures are >= 2021-07-01 (data availability boundary)")

# Show breakdown by category
print("\nProcedure breakdown by category:")
spark.sql(f"""
SELECT 
    CATEGORY,
    COUNT(*) as procedure_count,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    ROUND(COUNT(DISTINCT PAT_ID) * 100.0 / 
          (SELECT COUNT(DISTINCT PAT_ID) 
           FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort), 2) as pct_of_cohort
FROM {trgt_cat}.clncl_ds.herald_test_train_proc_unpivoted
GROUP BY CATEGORY
ORDER BY procedure_count DESC
""").show()

print("="*70)

# ========================================
# CELL 4
# ========================================

# Cell 3
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_iron_infusions AS

WITH
    cohort AS (
        SELECT PAT_ID, END_DTTM
        FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
    ),
    
    -- Data availability constraint
    params AS (
        SELECT CAST('2021-07-01' AS TIMESTAMP) AS data_cutoff
    ),
    
    iron_admin AS (
        SELECT
            mai.PAT_ID,
            mai.TAKEN_TIME AS admin_date,
            mai.DSPLY_NM AS med_name
        FROM clarity_cur.mar_admin_info_enh mai
        CROSS JOIN params p
        WHERE mai.MAR_ACTION_C = 1  -- Given
            AND (UPPER(mai.DSPLY_NM) LIKE '%IRON%DEXTRAN%'
                 OR UPPER(mai.DSPLY_NM) LIKE '%IRON%SUCROSE%'
                 OR UPPER(mai.DSPLY_NM) LIKE '%FERRIC%CARBOXYMALTOSE%'
                 OR UPPER(mai.DSPLY_NM) LIKE '%FERRIC%GLUCONATE%'
                 OR UPPER(mai.DSPLY_NM) LIKE '%FERUMOXYTOL%'
                 OR UPPER(mai.DSPLY_NM) LIKE '%VENOFER%'
                 OR UPPER(mai.DSPLY_NM) LIKE '%INJECTAFER%'
                 OR UPPER(mai.DSPLY_NM) LIKE '%FERAHEME%'
                 OR (UPPER(mai.DSPLY_NM) LIKE '%IRON%' 
                     AND (UPPER(mai.DSPLY_NM) LIKE '%IV%' 
                          OR UPPER(mai.DSPLY_NM) LIKE '%INJECT%'
                          OR UPPER(mai.DSPLY_NM) LIKE '%INFUS%')))
            AND mai.TAKEN_TIME IS NOT NULL
            AND mai.TAKEN_TIME >= p.data_cutoff  -- Enforce July 2021 cutoff
    )

SELECT
    c.PAT_ID,
    c.END_DTTM,
    
    -- Iron infusion counts
    COUNT(CASE 
        WHEN ia.admin_date >= DATE_SUB(c.END_DTTM, 365)
         AND ia.admin_date < c.END_DTTM
        THEN 1 END) AS iron_infusions_12mo,
        
    COUNT(CASE 
        WHEN ia.admin_date >= DATE_SUB(c.END_DTTM, 730)
         AND ia.admin_date < c.END_DTTM
        THEN 1 END) AS iron_infusions_24mo,
        
    -- Days since last iron infusion
    MIN(CASE 
        WHEN ia.admin_date < c.END_DTTM
        THEN DATEDIFF(c.END_DTTM, ia.admin_date)
        END) AS days_since_iron_infusion,
        
    -- Flag for any iron infusion
    MAX(CASE 
        WHEN ia.admin_date >= DATE_SUB(c.END_DTTM, 730)
         AND ia.admin_date < c.END_DTTM
        THEN 1 ELSE 0 END) AS iron_infusion_flag

FROM cohort c
LEFT JOIN iron_admin ia ON c.PAT_ID = ia.PAT_ID
GROUP BY c.PAT_ID, c.END_DTTM
""")

# Check iron infusion coverage
print("="*70)
print("IRON INFUSION EXTRACTION VALIDATION")
print("="*70)

iron_stats = spark.sql(f"""
SELECT 
    COUNT(*) as total_obs,
    SUM(iron_infusion_flag) as obs_with_iron,
    ROUND(AVG(iron_infusion_flag) * 100, 2) as pct_with_iron,
    ROUND(AVG(iron_infusions_12mo), 2) as avg_infusions_12mo,
    ROUND(AVG(iron_infusions_24mo), 2) as avg_infusions_24mo
FROM {trgt_cat}.clncl_ds.herald_test_train_iron_infusions
""").collect()[0]

print(f"\nTotal observations: {iron_stats['total_obs']:,}")
print(f"Observations with iron infusion: {iron_stats['obs_with_iron']:,}")
print(f"Coverage rate: {iron_stats['pct_with_iron']}%")
print(f"Average infusions (12mo): {iron_stats['avg_infusions_12mo']}")
print(f"Average infusions (24mo): {iron_stats['avg_infusions_24mo']}")
print(f"\n✓ All iron infusions are >= 2021-07-01 (data availability boundary)")
print("="*70)

# ========================================
# CELL 5
# ========================================

# CELL 4 - Verification Queries for Data Quality

print("="*70)
print("DATA QUALITY VERIFICATION CHECKS")
print("="*70)

# Check 1: Impact of July 2021 cutoff
print("\n1. Impact of Data Availability Boundary")
print("-" * 60)

cutoff_impact = spark.sql("""
WITH all_procs AS (
    SELECT 
        op.PAT_ID,
        op.RESULT_TIME,
        CASE WHEN op.RESULT_TIME >= CAST('2021-07-01' AS TIMESTAMP) 
             THEN 'After cutoff' 
             ELSE 'Before cutoff' END as period
    FROM clarity_cur.order_proc_enh op
    JOIN dev.clncl_ds.herald_test_train_proc_category_map m
      ON op.PROC_CODE = m.PROC_CODE
    WHERE op.ORDER_STATUS_C = 5
      AND op.RESULT_TIME >= CAST('2021-01-01' AS TIMESTAMP)
      AND op.RESULT_TIME <= CAST('2024-12-31' AS TIMESTAMP)
)
SELECT 
    period,
    COUNT(*) as procedure_count,
    COUNT(DISTINCT PAT_ID) as unique_patients
FROM all_procs
GROUP BY period
ORDER BY period
""").toPandas()

print(cutoff_impact.to_string(index=False))
if len(cutoff_impact) > 1:
    before = cutoff_impact[cutoff_impact['period'] == 'Before cutoff']['procedure_count'].values[0]
    after = cutoff_impact[cutoff_impact['period'] == 'After cutoff']['procedure_count'].values[0]
    print(f"\nExcluded {before:,} procedures from before July 2021 ({before/(before+after)*100:.1f}% of total)")

# Check 2: Patient overlap between cohort and procedures
print("\n2. Patient Coverage Analysis")
print("-" * 60)

patient_overlap = spark.sql("""
SELECT 
    COUNT(DISTINCT fc.PAT_ID) as cohort_patients,
    COUNT(DISTINCT pp.PAT_ID) as proc_patients,
    COUNT(DISTINCT CASE WHEN pp.PAT_ID IS NULL THEN fc.PAT_ID END) as patients_no_procs,
    ROUND(COUNT(DISTINCT pp.PAT_ID) * 100.0 / COUNT(DISTINCT fc.PAT_ID), 2) as pct_with_procs
FROM dev.clncl_ds.herald_test_train_final_cohort fc
LEFT JOIN dev.clncl_ds.herald_test_train_proc_unpivoted pp
  ON fc.PAT_ID = pp.PAT_ID
""").collect()[0]

print(f"Total cohort patients: {patient_overlap['cohort_patients']:,}")
print(f"Patients with procedures: {patient_overlap['proc_patients']:,}")
print(f"Patients with no procedures: {patient_overlap['patients_no_procs']:,}")
print(f"Coverage rate: {patient_overlap['pct_with_procs']}%")

# Check 3: Internal procedure code validation
print("\n3. Internal Procedure Code Reference")
print("-" * 60)

code_reference = spark.sql("""
SELECT 
    m.CATEGORY,
    COUNT(DISTINCT m.PROC_CODE) as unique_codes,
    SLICE(COLLECT_SET(m.PROC_CODE), 1, 3) as example_codes
FROM dev.clncl_ds.herald_test_train_proc_category_map m
GROUP BY m.CATEGORY
ORDER BY m.CATEGORY
""").toPandas()

print(code_reference.to_string(index=False))
print("\nNote: These are internal system codes specific to this health system")

# Check 4: Temporal distribution of procedures
print("\n4. Temporal Distribution of Captured Procedures")
print("-" * 60)

temporal_dist = spark.sql("""
SELECT 
    YEAR(PROC_DATE) as year,
    QUARTER(PROC_DATE) as quarter,
    COUNT(*) as procedures,
    COUNT(DISTINCT PAT_ID) as patients
FROM dev.clncl_ds.herald_test_train_proc_unpivoted
GROUP BY YEAR(PROC_DATE), QUARTER(PROC_DATE)
ORDER BY year, quarter
""").toPandas()

print(temporal_dist.to_string(index=False))

# Check 5: Validate no colonoscopy procedures
print("\n5. Colonoscopy Exclusion Verification")
print("-" * 60)

colonoscopy_check = spark.sql("""
SELECT COUNT(*) as colonoscopy_count
FROM dev.clncl_ds.herald_test_train_proc_unpivoted
WHERE LOWER(PROC_NAME) LIKE '%colonoscopy%'
   OR PROC_CODE IN ('45378', '45380', '45381', '45382', '45384', '45385')
""").collect()[0]['colonoscopy_count']

if colonoscopy_check == 0:
    print("✓ PASS: No colonoscopy procedures found (as intended)")
else:
    print(f"⚠ WARNING: Found {colonoscopy_check} colonoscopy procedures - investigate!")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

# ========================================
# CELL 6
# ========================================

# Cell 5
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_test_train_patient_procs AS

WITH
    cohort AS (
        SELECT PAT_ID, END_DTTM
        FROM {trgt_cat}.clncl_ds.herald_test_train_final_cohort
    ),

    unpvt AS (
        SELECT
            PAT_ID,
            END_DTTM,
            CATEGORY,
            PROC_DATE,
            DATEDIFF(END_DTTM, PROC_DATE) AS DAYS_SINCE_PROC
        FROM {trgt_cat}.clncl_ds.herald_test_train_proc_unpivoted
    ),

    -- Most recent procedure dates by category
    recent_procedures AS (
        SELECT 
            PAT_ID, 
            END_DTTM, 
            CATEGORY, 
            PROC_DATE,
            DAYS_SINCE_PROC,
            ROW_NUMBER() OVER (
                PARTITION BY PAT_ID, END_DTTM, CATEGORY 
                ORDER BY PROC_DATE DESC
            ) AS rn
        FROM unpvt
    ),

    -- Days since most recent by category
    procedure_recency AS (
        SELECT
            PAT_ID, 
            END_DTTM,
            MIN(CASE WHEN CATEGORY = 'CT_ABDOMEN_PELVIS' AND rn = 1 
                THEN DAYS_SINCE_PROC END) AS days_since_last_ct,
            MIN(CASE WHEN CATEGORY = 'MRI_ABDOMEN_PELVIS' AND rn = 1 
                THEN DAYS_SINCE_PROC END) AS days_since_last_mri,
            MIN(CASE WHEN CATEGORY = 'UPPER_GI_PROC' AND rn = 1 
                THEN DAYS_SINCE_PROC END) AS days_since_last_upper_gi,
            MIN(CASE WHEN CATEGORY = 'BLOOD_TRANSFUSION' AND rn = 1 
                THEN DAYS_SINCE_PROC END) AS days_since_last_transfusion,
            MIN(CASE WHEN CATEGORY IN ('ANAL_PROCEDURE', 'HEMORRHOID_PROC') AND rn = 1 
                THEN DAYS_SINCE_PROC END) AS days_since_last_anal_proc
        FROM recent_procedures
        WHERE rn = 1
        GROUP BY PAT_ID, END_DTTM
    ),

    pivoted AS (
        SELECT
            PAT_ID,
            END_DTTM,

            -- CT counts
            COUNT(CASE WHEN CATEGORY = 'CT_ABDOMEN_PELVIS' 
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 365) THEN 1 END) AS ct_abd_pelvis_count_12mo,
            COUNT(CASE WHEN CATEGORY = 'CT_ABDOMEN_PELVIS' 
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 730) THEN 1 END) AS ct_abd_pelvis_count_24mo,

            -- MRI counts
            COUNT(CASE WHEN CATEGORY = 'MRI_ABDOMEN_PELVIS' 
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 365) THEN 1 END) AS mri_abd_pelvis_count_12mo,
            COUNT(CASE WHEN CATEGORY = 'MRI_ABDOMEN_PELVIS' 
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 730) THEN 1 END) AS mri_abd_pelvis_count_24mo,
            
            -- Upper GI counts  
            COUNT(CASE WHEN CATEGORY = 'UPPER_GI_PROC' 
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 365) THEN 1 END) AS upper_gi_count_12mo,
            COUNT(CASE WHEN CATEGORY = 'UPPER_GI_PROC' 
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 730) THEN 1 END) AS upper_gi_count_24mo,

            -- Blood transfusion counts
            COUNT(CASE WHEN CATEGORY = 'BLOOD_TRANSFUSION' 
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 365) THEN 1 END) AS blood_transfusion_count_12mo,
            COUNT(CASE WHEN CATEGORY = 'BLOOD_TRANSFUSION' 
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 730) THEN 1 END) AS blood_transfusion_count_24mo,
                       
            -- Anal/hemorrhoid procedure counts
            COUNT(CASE WHEN CATEGORY IN ('ANAL_PROCEDURE', 'HEMORRHOID_PROC')
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 365) THEN 1 END) AS anal_proc_count_12mo,
            COUNT(CASE WHEN CATEGORY IN ('ANAL_PROCEDURE', 'HEMORRHOID_PROC')
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 730) THEN 1 END) AS anal_proc_count_24mo,

            -- Total imaging burden (CT + MRI combined)
            COUNT(CASE WHEN CATEGORY IN ('CT_ABDOMEN_PELVIS', 'MRI_ABDOMEN_PELVIS') 
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 365) THEN 1 END) AS total_imaging_count_12mo,
            COUNT(CASE WHEN CATEGORY IN ('CT_ABDOMEN_PELVIS', 'MRI_ABDOMEN_PELVIS') 
                       AND PROC_DATE >= DATE_SUB(END_DTTM, 730) THEN 1 END) AS total_imaging_count_24mo

        FROM unpvt
        GROUP BY PAT_ID, END_DTTM
    )

SELECT
    c.PAT_ID,
    c.END_DTTM,

    -- Procedure counts
    COALESCE(p.ct_abd_pelvis_count_12mo, 0) AS ct_abd_pelvis_count_12mo,
    COALESCE(p.ct_abd_pelvis_count_24mo, 0) AS ct_abd_pelvis_count_24mo,
    COALESCE(p.mri_abd_pelvis_count_12mo, 0) AS mri_abd_pelvis_count_12mo,
    COALESCE(p.mri_abd_pelvis_count_24mo, 0) AS mri_abd_pelvis_count_24mo,
    COALESCE(p.upper_gi_count_12mo, 0) AS upper_gi_count_12mo,
    COALESCE(p.upper_gi_count_24mo, 0) AS upper_gi_count_24mo,
    COALESCE(p.blood_transfusion_count_12mo, 0) AS blood_transfusion_count_12mo,
    COALESCE(p.blood_transfusion_count_24mo, 0) AS blood_transfusion_count_24mo,
    COALESCE(p.anal_proc_count_12mo, 0) AS anal_proc_count_12mo,
    COALESCE(p.anal_proc_count_24mo, 0) AS anal_proc_count_24mo,
    COALESCE(p.total_imaging_count_12mo, 0) AS total_imaging_count_12mo,
    COALESCE(p.total_imaging_count_24mo, 0) AS total_imaging_count_24mo,
    
    -- Iron infusions from MAR
    COALESCE(ii.iron_infusions_12mo, 0) AS iron_infusions_12mo,
    COALESCE(ii.iron_infusions_24mo, 0) AS iron_infusions_24mo,
    
    -- Recency features
    pr.days_since_last_ct,
    pr.days_since_last_mri,
    pr.days_since_last_upper_gi,
    pr.days_since_last_transfusion,
    pr.days_since_last_anal_proc,
    ii.days_since_iron_infusion,
    
    -- Composite features (intermediate features for CRC model, NOT final risk scores)
    
    -- High diagnostic intensity (multiple imaging studies)
    CASE WHEN COALESCE(p.total_imaging_count_12mo, 0) >= 2 
         THEN 1 ELSE 0 END AS high_imaging_intensity_flag,

    -- Transfusion history (severe bleeding)
    CASE WHEN COALESCE(p.blood_transfusion_count_24mo, 0) >= 1 
         THEN 1 ELSE 0 END AS transfusion_history_flag,
         
    -- Iron infusion history (chronic anemia)
    COALESCE(ii.iron_infusion_flag, 0) AS iron_infusion_flag,
         
    -- Anal pathology history
    CASE WHEN COALESCE(p.anal_proc_count_24mo, 0) >= 1 
         THEN 1 ELSE 0 END AS anal_pathology_flag,

    -- Comprehensive GI workup (upper GI + imaging)
    CASE WHEN COALESCE(p.upper_gi_count_12mo, 0) >= 1 
              AND COALESCE(p.total_imaging_count_12mo, 0) >= 1
         THEN 1 ELSE 0 END AS comprehensive_gi_workup_flag,

    -- Procedure intensity count (0-4 scale) - counts different procedure TYPES
    -- This is an INPUT FEATURE for the model, NOT the final CRC risk score
    LEAST(4,
        CASE WHEN COALESCE(p.total_imaging_count_12mo, 0) >= 2 THEN 1 ELSE 0 END +
        CASE WHEN COALESCE(p.upper_gi_count_12mo, 0) >= 1 THEN 1 ELSE 0 END +
        CASE WHEN COALESCE(p.blood_transfusion_count_12mo, 0) >= 1 THEN 1 ELSE 0 END +
        CASE WHEN COALESCE(ii.iron_infusions_12mo, 0) >= 1 THEN 1 ELSE 0 END
    ) AS procedure_intensity_count,  -- Changed from diagnostic_burden_score
    
    -- Severe anemia composite (transfusion OR iron infusion)
    CASE WHEN COALESCE(p.blood_transfusion_count_24mo, 0) >= 1 
              OR COALESCE(ii.iron_infusions_24mo, 0) >= 1
         THEN 1 ELSE 0 END AS severe_anemia_treatment_flag,  -- Changed from severe_anemia_flag
         
    -- Recent diagnostic activity (within 6 months)
    CASE WHEN COALESCE(pr.days_since_last_ct, 9999) <= 180 
              OR COALESCE(pr.days_since_last_mri, 9999) <= 180
         THEN 1 ELSE 0 END AS recent_diagnostic_activity_flag

FROM cohort c
LEFT JOIN pivoted p
    ON c.PAT_ID = p.PAT_ID AND c.END_DTTM = p.END_DTTM
LEFT JOIN procedure_recency pr
    ON c.PAT_ID = pr.PAT_ID AND c.END_DTTM = pr.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_test_train_iron_infusions ii
    ON c.PAT_ID = ii.PAT_ID AND c.END_DTTM = ii.END_DTTM
""")

print("Final patient procedures table created successfully")
print("Note: All features are INPUTS to the CRC risk model, not risk scores themselves")

# ========================================
# CELL 7
# ========================================

# Cell 6
# Comprehensive validation
result = spark.sql(f"""
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    
    -- Coverage rates
    AVG(CASE WHEN ct_abd_pelvis_count_24mo > 0 THEN 1 ELSE 0 END) as pct_with_ct,
    AVG(CASE WHEN mri_abd_pelvis_count_24mo > 0 THEN 1 ELSE 0 END) as pct_with_mri,
    AVG(CASE WHEN upper_gi_count_24mo > 0 THEN 1 ELSE 0 END) as pct_with_upper_gi,
    AVG(CASE WHEN blood_transfusion_count_24mo > 0 THEN 1 ELSE 0 END) as pct_with_transfusion,
    AVG(CASE WHEN iron_infusions_24mo > 0 THEN 1 ELSE 0 END) as pct_with_iron_infusion,
    AVG(CASE WHEN anal_proc_count_24mo > 0 THEN 1 ELSE 0 END) as pct_with_anal_proc,
    
    -- High-risk flags (UPDATED COLUMN NAMES)
    AVG(transfusion_history_flag) as pct_transfusion_history,
    AVG(iron_infusion_flag) as pct_iron_infusion,
    AVG(severe_anemia_treatment_flag) as pct_severe_anemia_treatment,  -- UPDATED
    AVG(high_imaging_intensity_flag) as pct_high_imaging,
    AVG(comprehensive_gi_workup_flag) as pct_comprehensive_workup,
    
    -- Average counts (UPDATED COLUMN NAMES)
    AVG(total_imaging_count_12mo) as avg_imaging_12mo,
    AVG(procedure_intensity_count) as avg_procedure_intensity  -- UPDATED
    
FROM {trgt_cat}.clncl_ds.herald_test_train_patient_procs
""")
result.show()

print("\n✓ Procedures feature engineering complete")
print("Key features captured:")
print("- CT/MRI imaging patterns (diagnostic workup)")
print("- Blood transfusions (acute bleeding indicator)")
print("- Iron infusions (chronic anemia treatment)")
print("- Upper GI procedures (symptom evaluation)")
print("- Anal/hemorrhoid procedures (bleeding source)")
print("Note: Colonoscopy excluded as successful screening already removed from cohort")
print("\nAll features are INPUTS to the CRC/anal cancer risk model")

# ========================================
# CELL 8
# ========================================

# Cell 7
df_spark = spark.sql('''SELECT * FROM dev.clncl_ds.herald_test_train_patient_procs''')
df = df_spark.toPandas()
df.isnull().sum()/df.shape[0]

# ========================================
# CELL 9
# ========================================

display(df_spark)

# ========================================
# CELL 10
# ========================================

# Cell 8
df2 = df.drop(['PAT_ID','END_DTTM'], axis=1)
df2.mean()

# ========================================
# CELL 11
# ========================================

df2.shape

# ========================================
# CELL 12
# ========================================

# Step 1: Load procedure data and calculate basic statistics

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PROCEDURE FEATURE REDUCTION")
print("="*60)

# Load procedures table and join with cohort for outcome
df_procs = spark.table("dev.clncl_ds.herald_test_train_patient_procs")

# Load cohort with FUTURE_CRC_EVENT
df_cohort = spark.sql("""
    SELECT PAT_ID, END_DTTM, FUTURE_CRC_EVENT
    FROM dev.clncl_ds.herald_test_train_final_cohort
""")

# Join to get outcome variable
df_spark = df_procs.join(
    df_cohort,
    on=['PAT_ID', 'END_DTTM'],
    how='inner'
)

# Add proc_ prefix to all columns except keys
proc_cols = [col for col in df_spark.columns if col not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT']]
for col in proc_cols:
    df_spark = df_spark.withColumnRenamed(col, f'proc_{col}' if not col.startswith('proc_') else col)

# Cache for performance
df_spark.cache()
total_rows = df_spark.count()
baseline_crc_rate = df_spark.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"\nTotal rows: {total_rows:,}")
print(f"Baseline CRC rate: {baseline_crc_rate:.4f}")

# Calculate coverage for key procedure categories
transfusion_coverage = df_spark.filter(F.col('proc_transfusion_history_flag') == 1).count() / total_rows
iron_coverage = df_spark.filter(F.col('proc_iron_infusion_flag') == 1).count() / total_rows
severe_anemia_coverage = df_spark.filter(F.col('proc_severe_anemia_treatment_flag') == 1).count() / total_rows
ct_coverage = df_spark.filter(F.col('proc_ct_abd_pelvis_count_24mo') > 0).count() / total_rows

print(f"\nCoverage rates:")
print(f"CT imaging (24mo): {ct_coverage:.1%}")
print(f"Transfusion history: {transfusion_coverage:.1%}")
print(f"Iron infusion: {iron_coverage:.1%}")
print(f"Severe anemia treatment: {severe_anemia_coverage:.1%}")

# ========================================
# CELL 13
# ========================================

# Step 2: Calculate Risk Ratios for Binary Procedure Features

binary_features = [col for col in df_spark.columns if '_flag' in col and col.startswith('proc_')]
risk_metrics = []

print(f"\nCalculating risk ratios for {len(binary_features)} binary features...")

for feat in binary_features:
    stats = df_spark.groupBy(feat).agg(
        F.count('*').alias('count'),
        F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
    ).collect()
    
    # Parse results
    stats_dict = {row[feat]: {'count': row['count'], 'crc_rate': row['crc_rate']} for row in stats}
    
    prevalence = stats_dict.get(1, {'count': 0})['count'] / total_rows
    rate_with = stats_dict.get(1, {'crc_rate': 0})['crc_rate']
    rate_without = stats_dict.get(0, {'crc_rate': baseline_crc_rate})['crc_rate']
    risk_ratio = rate_with / (rate_without + 1e-10)
    
    # Calculate impact
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
print("\nTop features by impact score (prevalence × log risk ratio):")
print(risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']].to_string())

# ========================================
# CELL 14
# ========================================

# Step 3: Analyze Continuous Features and Missing Patterns

print("\nAnalyzing continuous features and missing patterns...")

# Separate features by type
count_features = [col for col in df_spark.columns if col.startswith('proc_') and 
                  ('_count' in col or 'count_' in col)]
recency_features = [col for col in df_spark.columns if col.startswith('proc_') and 
                    'days_since' in col]
composite_features = [col for col in df_spark.columns if col.startswith('proc_') and 
                      col not in count_features + recency_features + binary_features and
                      col not in ['proc_PAT_ID', 'proc_END_DTTM']]

print(f"Feature types:")
print(f"  - Count features: {len(count_features)}")
print(f"  - Recency features: {len(recency_features)}")
print(f"  - Binary flags: {len(binary_features)}")
print(f"  - Composite features: {len(composite_features)}")

# Analyze missing patterns for recency features (high missing = rare procedure)
missing_stats = []
for feat in recency_features:
    missing_rate = df_spark.filter(F.col(feat).isNull()).count() / total_rows
    mean_when_present = df_spark.filter(F.col(feat).isNotNull()).select(F.avg(feat)).collect()[0][0]
    
    missing_stats.append({
        'feature': feat,
        'missing_rate': missing_rate,
        'mean_days_when_present': mean_when_present if mean_when_present else None,
        'procedure_type': feat.replace('proc_days_since_last_', '').replace('proc_days_since_', '')
    })

missing_df = pd.DataFrame(missing_stats)
print(f"\nRecency features by procedure frequency (less missing = more common):")
print(missing_df.sort_values('missing_rate')[['feature', 'missing_rate', 'mean_days_when_present']].to_string())

# Analyze count feature distributions
count_stats = []
for feat in count_features:
    mean_val = df_spark.select(F.avg(feat)).collect()[0][0]
    max_val = df_spark.select(F.max(feat)).collect()[0][0]
    nonzero_pct = df_spark.filter(F.col(feat) > 0).count() / total_rows
    
    count_stats.append({
        'feature': feat,
        'mean': mean_val,
        'max': max_val,
        'pct_nonzero': nonzero_pct
    })

count_df = pd.DataFrame(count_stats)
print(f"\nCount features by prevalence:")
print(count_df.sort_values('pct_nonzero', ascending=False)[['feature', 'mean', 'pct_nonzero']].to_string())

# ========================================
# CELL 15
# ========================================

#===========================================

# Step 4: Calculate Mutual Information Using Stratified Sample

# Take stratified sample for MI calculation
sample_fraction = min(200000 / total_rows, 1.0)

print(f"\nSampling for MI calculation...")
df_sample = df_spark.sampleBy("FUTURE_CRC_EVENT", 
                               fractions={0: sample_fraction, 1: 1.0},
                               seed=42).toPandas()

print(f"Sampled {len(df_sample):,} rows ({len(df_sample)/total_rows*100:.1f}% of total)")
print(f"Sample CRC rate: {df_sample['FUTURE_CRC_EVENT'].mean():.4f}")

# Calculate MI on sample
from sklearn.feature_selection import mutual_info_classif

feature_cols = [c for c in df_sample.columns 
                if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT'] 
                and c.startswith('proc_')]

print(f"Calculating MI for {len(feature_cols)} features...")
X = df_sample[feature_cols].fillna(-999)
y = df_sample['FUTURE_CRC_EVENT']

mi_scores = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop features by Mutual Information:")
print(mi_df.to_string())

# ========================================
# CELL 16
# ========================================

#===========================================

# Step 5: Apply Clinical Filters for Procedure Setting

# Merge all metrics
feature_importance = mi_df.merge(
    risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']], 
    on='feature', 
    how='left'
)

# Add count statistics
count_stats_df = pd.DataFrame(count_stats)
count_stats_df['feature'] = count_stats_df['feature'].apply(lambda x: x)  # Ensure same format
feature_importance = feature_importance.merge(
    count_stats_df[['feature', 'pct_nonzero']], 
    on='feature', 
    how='left'
)

# Add missing stats for recency features
missing_stats_df = pd.DataFrame(missing_stats)
feature_importance = feature_importance.merge(
    missing_stats_df[['feature', 'missing_rate']], 
    on='feature', 
    how='left'
)

# Fill NAs
feature_importance['prevalence'] = feature_importance['prevalence'].fillna(
    feature_importance['pct_nonzero']
)
feature_importance['missing_rate'] = feature_importance['missing_rate'].fillna(0)
feature_importance['risk_ratio'] = feature_importance['risk_ratio'].fillna(1.0)
feature_importance['impact'] = feature_importance['impact'].fillna(0)

# Procedure-specific critical features
MUST_KEEP = [
    'proc_severe_anemia_treatment_flag',     # Objective bleeding
    'proc_blood_transfusion_count_12mo',     # Acute bleeding
    'proc_iron_infusions_12mo',              # Chronic anemia
    'proc_procedure_intensity_count',        # Overall activity
    'proc_total_imaging_count_12mo',         # Diagnostic workup
    'proc_high_imaging_intensity_flag',      # Multiple studies
    'proc_comprehensive_gi_workup_flag'      # Complete evaluation
]

# Remove features with near-zero signal
REMOVE = []
for _, row in feature_importance.iterrows():
    feat = row['feature']
    
    # Remove recency features for rare procedures (>99% missing)
    if 'days_since' in feat and row.get('missing_rate', 0) > 0.99:
        REMOVE.append(feat)
    
    # Remove 24-month counts if 12-month version exists and is better
    if '_24mo' in feat:
        corresponding_12mo = feat.replace('_24mo', '_12mo')
        if corresponding_12mo in feature_importance['feature'].values:
            # Check if 12mo version has better MI score
            mi_12mo = feature_importance[feature_importance['feature'] == corresponding_12mo]['mi_score'].values
            mi_24mo = row['mi_score']
            if len(mi_12mo) > 0 and mi_12mo[0] >= mi_24mo:
                REMOVE.append(feat)

REMOVE = list(set(REMOVE))
print(f"\nRemoving {len(REMOVE)} low-signal or redundant features")
print(f"Examples of removed features: {REMOVE[:5]}")

feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]
print(f"Features remaining after filtering: {len(feature_importance)}")

# ========================================
# CELL 17
# ========================================

#===========================================

# Step 6: Select Optimal Features per Procedure Type

def select_optimal_proc_features(df_importance):
    """Select best representation for each procedure type"""
    
    selected = []
    
    # Extract procedure type from feature name
    df_importance['proc_type'] = df_importance['feature'].apply(
        lambda x: x.replace('proc_', '').split('_')[0] if 'proc_' in x else ''
    )
    
    # Group by procedure type
    for proc in df_importance['proc_type'].unique():
        if not proc:  # Skip empty
            continue
            
        proc_features = df_importance[df_importance['proc_type'] == proc]
        
        # Procedure-specific selection rules
        if proc in ['blood', 'transfusion']:
            # Keep count and flag for transfusions
            for feat in proc_features['feature']:
                if '12mo' in feat or '_flag' in feat:
                    selected.append(feat)
                    
        elif proc == 'iron':
            # Keep iron infusion count and flag
            for feat in proc_features['feature']:
                if '12mo' in feat or '_flag' in feat:
                    selected.append(feat)
                    
        elif proc in ['ct', 'mri']:
            # Keep 12-month count for imaging
            count_12mo = proc_features[proc_features['feature'].str.contains('12mo')]
            if not count_12mo.empty:
                selected.append(count_12mo.nlargest(1, 'mi_score')['feature'].values[0])
                
        elif proc == 'total':
            # Keep total imaging count
            imaging_12mo = proc_features[proc_features['feature'].str.contains('imaging_count_12mo')]
            if not imaging_12mo.empty:
                selected.append(imaging_12mo['feature'].values[0])
                
        elif proc in ['severe', 'high', 'comprehensive', 'recent', 'anal']:
            # Keep all composite/flag features
            for feat in proc_features['feature']:
                if '_flag' in feat:
                    selected.append(feat)
                    
        elif proc == 'procedure':
            # Keep intensity count
            intensity = proc_features[proc_features['feature'].str.contains('intensity')]
            if not intensity.empty:
                selected.append(intensity['feature'].values[0])
                
        elif proc == 'upper':
            # Keep upper GI count
            upper_12mo = proc_features[proc_features['feature'].str.contains('12mo')]
            if not upper_12mo.empty:
                selected.append(upper_12mo['feature'].values[0])
    
    # Ensure must-keep features are included
    for feat in MUST_KEEP:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    return list(set(selected))

selected_features = select_optimal_proc_features(feature_importance)
print(f"\nSelected {len(selected_features)} features after procedure-type optimization")
print("\nSelected features:")
for feat in sorted(selected_features):
    print(f"  - {feat}")

# ========================================
# CELL 18
# ========================================


# Step 7: Create Clinical Composite Features and Save

df_final = df_spark

# === PROCEDURE-SPECIFIC COMPOSITE FEATURES ===

# 1. Anemia treatment intensity (combines frequency and types)
df_final = df_final.withColumn('proc_anemia_treatment_intensity',
    F.least(F.lit(3),
        F.col('proc_blood_transfusion_count_12mo') + 
        F.col('proc_iron_infusions_12mo')
    )
)

# 2. Diagnostic cascade (multiple imaging without resolution)
df_final = df_final.withColumn('proc_diagnostic_cascade',
    F.when((F.col('proc_total_imaging_count_12mo') >= 2) & 
           (F.col('proc_upper_gi_count_12mo') >= 1), 1).otherwise(0)
)

# 3. Acute bleeding pattern (recent transfusion)
df_final = df_final.withColumn('proc_acute_bleeding_pattern',
    F.when((F.col('proc_blood_transfusion_count_12mo') >= 1) & 
           (F.col('proc_days_since_last_transfusion') <= 90), 1)
    .when(F.col('proc_blood_transfusion_count_12mo') >= 2, 1)
    .otherwise(0)
)

composite_features = [
    'proc_anemia_treatment_intensity',
    'proc_diagnostic_cascade',
    'proc_acute_bleeding_pattern'
]

# Add composites to selected features
selected_features.extend(composite_features)
selected_features = sorted(list(set(selected_features)))

print(f"\nAdded {len(composite_features)} composite features")
print(f"Final feature count: {len(selected_features)}")

# === PRINT FINAL FEATURE LIST ===
print("\n" + "="*60)
print("FINAL SELECTED PROCEDURE FEATURES")
print("="*60)

for i, feat in enumerate(selected_features, 1):
    # Add description for clarity
    if 'transfusion' in feat:
        desc = " [ACUTE BLEEDING]"
    elif 'iron' in feat:
        desc = " [CHRONIC ANEMIA]"
    elif 'severe_anemia' in feat:
        desc = " [OBJECTIVE BLEEDING]"
    elif 'imaging' in feat:
        desc = " [DIAGNOSTIC WORKUP]"
    elif feat in composite_features:
        desc = " [COMPOSITE]"
    elif 'intensity' in feat:
        desc = " [CARE PATTERN]"
    elif 'upper_gi' in feat:
        desc = " [SYMPTOM EVALUATION]"
    else:
        desc = ""
    print(f"{i:2d}. {feat:<45} {desc}")

# === SAVE REDUCED DATASET ===
final_columns = ['PAT_ID', 'END_DTTM'] + selected_features
df_reduced = df_final.select(*final_columns)

# Add proc_ prefix to all columns except keys
proc_cols = [col for col in df_reduced.columns if col not in ['PAT_ID', 'END_DTTM']]
for col in proc_cols:
    df_reduced = df_reduced.withColumnRenamed(col, f'proc_{col}' if not col.startswith('proc_') else col)

# Write to final table
output_table = 'dev.clncl_ds.herald_test_train_procedures_reduced'
df_reduced.write.mode('overwrite').saveAsTable(output_table)

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: 33")
print(f"Selected features: {len(selected_features)}")
print(f"Reduction: {(1 - len(selected_features)/33)*100:.1f}%")
print(f"\n✓ Reduced dataset saved to: {output_table}")

# Verify save
row_count = spark.table(output_table).count()
cols_without_prefix = [c for c in selected_features if not c.startswith('proc_')]

print(f"✓ Verified {row_count:,} rows written to table")
if cols_without_prefix:
    print(f"\n⚠ WARNING: These columns missing 'proc_' prefix: {cols_without_prefix}")
else:
    print("✓ All feature columns have 'proc_' prefix for joining")

# ========================================
# CELL 19
# ========================================

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_test_train_procedures_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 20
# ========================================

display(df_check)



################################################################################
# V2_Book8_Compilation
################################################################################

# V2_Book8_Compilation
# Functional cells: 13 of 27 code cells (47 total)
# Source: V2_Book8_Compilation.ipynb
# =============================================================================

# ========================================
# CELL 1
# ========================================

import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import os
import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional
import warnings

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "America/Chicago")

# Define target catalog for SQL based on the environment variable
trgt_cat = os.environ.get('trgt_cat')

# Use appropriate Spark catalog based on the target category
spark.sql('USE CATALOG prod;')

print("Spark session initialized successfully")
print(f"Spark version: {spark.version}")
print(f"Timezone: America/Chicago")
print(f"Current catalog: dev")
print(f"Current database: clncl_ds")
print(f"Current time: {datetime.datetime.now()}")

# ========================================
# CELL 2
# ========================================

# Adaptive Query Execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.minPartitionSize", "1MB")
spark.conf.set("spark.sql.adaptive.coalescePartitions.initialPartitionNum", "10000")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")

print("✓ Adaptive Query Execution configured")

# Shuffle and Join Optimization
spark.conf.set("spark.sql.shuffle.partitions", "256")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "52428800")  # 50MB
spark.conf.set("spark.sql.broadcastTimeout", "600")
spark.conf.set("spark.sql.cbo.enabled", "true")
spark.conf.set("spark.sql.cbo.joinReorder.enabled", "true")

print("✓ Shuffle and join optimization configured")

# Delta Lake Optimization
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
spark.conf.set("spark.databricks.delta.merge.repartitionBeforeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

print("✓ Delta Lake optimization configured")

# Memory and Execution
spark.conf.set("spark.sql.files.maxPartitionBytes", "67108864")  # 64MB
spark.conf.set("spark.sql.files.openCostInBytes", "4194304")  # 4MB
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "5000")
spark.conf.set("spark.sql.codegen.wholeStage", "true")
spark.conf.set("spark.sql.codegen.hugeMethodLimit", "32768")

print("✓ Memory and execution optimization configured")
print("=" * 80)
print("✓ All runtime optimizations applied successfully")

# ========================================
# CELL 3
# ========================================

# Define the reduced tables
reduced_tables = {
    'icd10': 'dev.clncl_ds.herald_test_train_icd10_reduced',
    'labs': 'dev.clncl_ds.herald_test_train_labs_reduced',
    'outpatient_meds': 'dev.clncl_ds.herald_test_train_outpatient_meds_reduced',
    'inpatient_meds': 'dev.clncl_ds.herald_test_train_inpatient_meds_reduced',
    'visit_features': 'dev.clncl_ds.herald_test_train_visit_features_reduced',
    'procedures': 'dev.clncl_ds.herald_test_train_procedures_reduced'
}

# Get columns from each table
for name, table in reduced_tables.items():
    cols = spark.table(table).columns
    # Remove PAT_ID and END_DTTM from count
    feature_cols = [c for c in cols if c not in ['PAT_ID', 'END_DTTM']]
    print(f"\n{name.upper()}: {len(feature_cols)} features")
    print(f"  Columns: {', '.join(sorted(feature_cols))}")

print("\n" + "="*80)
print("✓ Verification complete - review columns before proceeding to Step 1")
print("="*80)

# ========================================
# CELL 4
# ========================================

print("="*80)
print("CREATING WIDE TABLE FROM REDUCED FEATURES")
print("="*80)

spark.sql("""
-- Replace the hardcoded SELECT with dynamic selection
CREATE OR REPLACE TABLE dev.clncl_ds.herald_test_train_wide AS
SELECT
    c.* EXCEPT (LABEL_CONFIDENCE, current_screen_status,
                vbc_last_colonoscopy_date, vbc_last_fobt_date,
                last_internal_screening_date, last_colonoscopy_date,
                last_ct_colonography_date, last_sigmoidoscopy_date,
                last_fit_dna_date, last_fobt_date,
                fobt_count, had_fobt_in_lookback),
    i.* EXCEPT (PAT_ID, END_DTTM),
    l.* EXCEPT (PAT_ID, END_DTTM),
    om.* EXCEPT (PAT_ID, END_DTTM),
    im.* EXCEPT (PAT_ID, END_DTTM),
    vis.* EXCEPT (PAT_ID, END_DTTM),
    p.* EXCEPT (PAT_ID, END_DTTM)
FROM dev.clncl_ds.herald_test_train_final_cohort AS c
LEFT JOIN dev.clncl_ds.herald_test_train_icd10_reduced AS i USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_test_train_labs_reduced AS l USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_test_train_outpatient_meds_reduced AS om USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_test_train_inpatient_meds_reduced AS im USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_test_train_visit_features_reduced AS vis USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_test_train_procedures_reduced AS p USING (PAT_ID, END_DTTM)
""")

print("✓ Wide table created: dev.clncl_ds.herald_test_train_wide")

# ========================================
# CELL 5
# ========================================

print("="*80)
print("TRANSFORMING FEATURES TO PREVENT MEMORIZATION")
print("="*80)

from pyspark.sql import functions as F
from pyspark.sql.functions import when, col

# Load the wide table
df = spark.table("dev.clncl_ds.herald_test_train_wide")

# ============================================================================
# 1. REMOVE PATIENT-SPECIFIC TEMPORAL IDENTIFIERS
# ============================================================================
print("\n1. Removing patient-specific temporal features...")

# These act as patient fingerprints when combined with other features
features_to_drop = [
    'MONTHS_SINCE_COHORT_ENTRY',  # Primary memorization culprit
    'OBS_MONTHS_PRIOR'             # Another patient identifier
]

# Check which features exist before dropping
existing_to_drop = [f for f in features_to_drop if f in df.columns]
if existing_to_drop:
    df = df.drop(*existing_to_drop)
    print(f"   Dropped: {', '.join(existing_to_drop)}")
else:
    print("   No features to drop (already removed)")

# ============================================================================
# 2. BIN ALL TEMPORAL FEATURES (_DAYS_SINCE)
# ============================================================================
print("\n2. Binning temporal features to prevent exact-day memorization...")

# REVISED TRANSFORMATION CODE - Replace the binning section with:

print("\n2. Binning temporal features with ORDINAL encoding...")

# Find all _DAYS_SINCE features
days_since_cols = [c for c in df.columns if 'DAYS_SINCE' in c.upper()]
print(f"   Found {len(days_since_cols)} temporal features to bin")

for col_name in days_since_cols:
    # Create ordinal encoded feature (0-5 scale preserves ordering)
    binned_col_name = col_name.replace('_DAYS_SINCE', '_RECENCY').replace('_days_since', '_recency')
    
    df = df.withColumn(
        binned_col_name,
        F.when(F.col(col_name).isNull(), 0)  # Never = 0
        .when(F.col(col_name) <= 30, 5)      # Very recent = 5 (highest)
        .when(F.col(col_name) <= 90, 4)      # Recent = 4
        .when(F.col(col_name) <= 180, 3)     # Moderate = 3
        .when(F.col(col_name) <= 365, 2)     # Distant = 2
        .otherwise(1)                         # Very distant = 1 (lowest)
    )
    
    # Drop only the original continuous column
    df = df.drop(col_name)

print(f"   Replaced {len(days_since_cols)} temporal features with ordinal versions")

# ============================================================================
# 3. TRANSFORM CONTINUOUS PATIENT CHARACTERISTICS
# ============================================================================
print("\n3. Transforming patient characteristics...")

# AGE - Convert to ordinal age groups
df = df.withColumn('AGE_GROUP',
    F.when((F.col('AGE') >= 45) & (F.col('AGE') < 50), 1)  # 45-49
    .when((F.col('AGE') >= 50) & (F.col('AGE') < 55), 2)   # 50-54
    .when((F.col('AGE') >= 55) & (F.col('AGE') < 65), 3)   # 55-64
    .when((F.col('AGE') >= 65) & (F.col('AGE') < 75), 4)   # 65-74
    .when(F.col('AGE') >= 75, 5)                            # 75+
    .otherwise(0))  # Should never happen
df = df.drop('AGE')

# WEIGHT_OZ - Convert to quartiles (ordinal)
if 'WEIGHT_OZ' in df.columns:
    weight_percentiles = df.select(
        F.expr('percentile_approx(WEIGHT_OZ, 0.25)').alias('p25'),
        F.expr('percentile_approx(WEIGHT_OZ, 0.50)').alias('p50'),
        F.expr('percentile_approx(WEIGHT_OZ, 0.75)').alias('p75')
    ).collect()[0]
    
    df = df.withColumn('WEIGHT_QUARTILE',
        F.when(F.col('WEIGHT_OZ') <= weight_percentiles['p25'], 1)
        .when(F.col('WEIGHT_OZ') <= weight_percentiles['p50'], 2)
        .when(F.col('WEIGHT_OZ') <= weight_percentiles['p75'], 3)
        .otherwise(4))
    df = df.drop('WEIGHT_OZ')

# BMI - Convert to ordinal clinical categories
if 'BMI' in df.columns:
    df = df.withColumn('BMI_CATEGORY',
        F.when(F.col('BMI') < 18.5, 1)                       # Underweight
        .when((F.col('BMI') >= 18.5) & (F.col('BMI') < 25), 2)  # Normal
        .when((F.col('BMI') >= 25) & (F.col('BMI') < 30), 3)    # Overweight
        .when(F.col('BMI') >= 30, 4)                         # Obese
        .otherwise(0))
    df = df.drop('BMI')

# ============================================================================
# 4. KEEP BUT MONITOR quarters_since_study_start
# ============================================================================
print("\n4. Keeping quarters_since_study_start for prevalent case adjustment")
print("   (Will monitor for memorization in model evaluation)")

# ============================================================================
# SAVE TRANSFORMED TABLE
# ============================================================================
df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("dev.clncl_ds.herald_test_train_wide_transformed")

print("\n" + "="*80)
print("TRANSFORMATION COMPLETE")
print("="*80)

# Verify transformation
final_cols = df.columns
temporal_remaining = [c for c in final_cols if 'DAYS_SINCE' in c.upper()]
print(f"\nFinal column count: {len(final_cols)}")
print(f"Remaining temporal features: {len(temporal_remaining)}")
if temporal_remaining:
    print("  WARNING: These temporal features remain:", temporal_remaining[:5])

print("\n✓ Transformed table saved: dev.clncl_ds.herald_test_train_wide_transformed")
print("  Ready for preprocessing and feature selection")

# ========================================
# CELL 6
# ========================================

# Load the wide table
df = spark.table("dev.clncl_ds.herald_test_train_wide_transformed")

print("="*70)
print("ADDING TEMPORAL FEATURE FOR PREVALENT CASE ADJUSTMENT")
print("="*70)

# Define study start date from your cohort creation
STUDY_START_DATE = '2023-01-01'

# Add quarters_since_study_start using PySpark
df = df.withColumn(
    'quarters_since_study_start',
    F.floor(
        F.months_between(F.col('END_DTTM'), F.lit(STUDY_START_DATE)) / 3
    ).cast('integer')
)

# Verify the feature captures the expected pattern
stats = df.agg(
    F.min('END_DTTM').alias('min_date'),
    F.max('END_DTTM').alias('max_date'),
    F.min('quarters_since_study_start').alias('min_quarter'),
    F.max('quarters_since_study_start').alias('max_quarter')
).collect()[0]

print(f"\nStudy start date: {STUDY_START_DATE}")
print(f"Data date range: {stats['min_date']} to {stats['max_date']}")
print(f"Quarters in dataset: {stats['min_quarter']} to {stats['max_quarter']}")

# Show event rate decline by quarter
quarter_analysis = df.groupBy('quarters_since_study_start').agg(
    F.count('*').alias('Total_Obs'),
    F.sum('FUTURE_CRC_EVENT').alias('CRC_Events'),
    F.avg('FUTURE_CRC_EVENT').alias('Event_Rate')
).orderBy('quarters_since_study_start')

print("\n" + "="*70)
print("EVENT RATE BY QUARTER (Confirming Prevalent Case Pattern)")
print("="*70)
quarter_analysis.show()

# Get first and last quarter rates for decline calculation
quarter_rates = quarter_analysis.select(
    'quarters_since_study_start', 
    'Event_Rate'
).orderBy('quarters_since_study_start').collect()

if len(quarter_rates) >= 2:
    first_quarter_rate = quarter_rates[0]['Event_Rate']
    last_quarter_rate = quarter_rates[-1]['Event_Rate']
    decline_pct = ((first_quarter_rate - last_quarter_rate) / first_quarter_rate) * 100 if first_quarter_rate != 0 else 0
    
    print(f"Event rate decline from Q{quarter_rates[0]['quarters_since_study_start']} to Q{quarter_rates[-1]['quarters_since_study_start']}: {decline_pct:.1f}%")
    print(f"This {decline_pct:.0f}% decline reflects prevalent case clearance over time")

print("\n✓ Feature 'quarters_since_study_start' added to df_spark")
print("="*70)

# Cache the updated dataframe for performance
df = df.cache()

# ========================================
# CELL 7
# ========================================

stats = spark.sql("""
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT PAT_ID) as unique_patients,
        SUM(FUTURE_CRC_EVENT) as positive_cases,
        100.0 * AVG(FUTURE_CRC_EVENT) as positive_rate
    FROM dev.clncl_ds.herald_test_train_wide
""").collect()[0]

print("="*60)
print("WIDE TABLE STATISTICS")
print("="*60)
print(f"  Total rows: {stats['total_rows']:,}")
print(f"  Unique patients: {stats['unique_patients']:,}")
print(f"  Positive cases: {stats['positive_cases']:,}")
print(f"  Positive rate: {stats['positive_rate']:.3f}%")
print(f"  Imbalance ratio: 1:{int(stats['total_rows']/stats['positive_cases'])}")
print("="*60)

# ========================================
# CELL 8
# ========================================

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType
import pandas as pd
import numpy as np

print("="*80)
print("PREPROCESSING: FEATURE QUALITY CHECKS")
print("="*80)

# Exclude identifiers, target, and outcome-related diagnosis columns
# ICD10_CODE and ICD10_GROUP are the diagnosis codes for the CRC outcome - NOT features!
exclude_cols = ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT', 'SPLIT', 'ICD10_CODE', 'ICD10_GROUP']
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"\nStarting with {len(feature_cols)} features")

# =============================================================================
# CHECK 1: NEAR-ZERO VARIANCE (CONSTANT FEATURES)
# =============================================================================
print("\n" + "="*80)
print("CHECK 1: NEAR-ZERO VARIANCE (ESSENTIALLY CONSTANT)")
print("="*80)

# Get numeric columns only
numeric_cols = [f.name for f in df.schema.fields 
                if isinstance(f.dataType, NumericType) 
                and f.name in feature_cols]

# Calculate variance and distinct counts
variance_stats = []
for col in numeric_cols:
    stats = df.select(
        F.variance(F.col(col)).alias('var'),
        F.countDistinct(F.col(col)).alias('n_distinct'),
        F.count(F.col(col)).alias('n_non_null')
    ).collect()[0]
    
    variance_stats.append({
        'feature': col,
        'variance': stats['var'] if stats['var'] is not None else 0,
        'n_distinct': stats['n_distinct'],
        'n_non_null': stats['n_non_null']
    })

variance_df = pd.DataFrame(variance_stats)

# Only flag truly constant features (1 distinct value when non-null exists)
near_zero_var = variance_df[
    (variance_df['n_distinct'] == 1) & (variance_df['n_non_null'] > 0)
].sort_values('variance')

print(f"\nFound {len(near_zero_var)} constant features:")
if len(near_zero_var) > 0:
    print(near_zero_var.to_string(index=False))
else:
    print("None found")

features_to_remove = set(near_zero_var['feature'].tolist())

# =============================================================================
# CHECK 2: PERFECT CORRELATIONS (WITH STRATIFIED SAMPLING)
# =============================================================================
print("\n" + "="*80)
print("CHECK 2: PERFECT CORRELATIONS (|ρ| >= 0.999)")
print("="*80)

# Only check numeric columns that haven't been flagged for removal
remaining_numeric = [c for c in numeric_cols if c not in features_to_remove]

print(f"\nCalculating correlations for {len(remaining_numeric)} numeric features...")
print("Using stratified sample to ensure adequate positive case representation")

# Get class counts
total_rows = df.count()
positive_count = df.filter(F.col('FUTURE_CRC_EVENT') == 1).count()
negative_count = total_rows - positive_count

print(f"\nDataset composition:")
print(f"  Total rows: {total_rows:,}")
print(f"  Positive cases: {positive_count:,} ({positive_count/total_rows*100:.3f}%)")
print(f"  Negative cases: {negative_count:,}")

# Stratified sample: all positives + sample of negatives
target_sample_size = 100000 if total_rows > 100000 else total_rows

# Calculate how many negatives to sample
if positive_count >= target_sample_size:
    # If we have more positives than target, just sample everything proportionally
    sample_fraction = target_sample_size / total_rows
    pdf = df.select(['FUTURE_CRC_EVENT'] + remaining_numeric).sample(False, sample_fraction, seed=42).toPandas()
else:
    # Take all positives + sample negatives to reach target
    negatives_needed = target_sample_size - positive_count
    negative_sample_fraction = 1.0 if negatives_needed >= negative_count else negatives_needed / negative_count
    
    # Get all positives
    positives_df = df.filter(F.col('FUTURE_CRC_EVENT') == 1).select(['FUTURE_CRC_EVENT'] + remaining_numeric)
    
    # Sample negatives
    negatives_df = df.filter(F.col('FUTURE_CRC_EVENT') == 0).select(['FUTURE_CRC_EVENT'] + remaining_numeric).sample(False, negative_sample_fraction, seed=42)
    
    # Combine
    sampled_df = positives_df.union(negatives_df)
    pdf = sampled_df.toPandas()

print(f"\nSample composition:")
print(f"  Total sampled: {len(pdf):,}")
print(f"  Positive cases: {(pdf['FUTURE_CRC_EVENT'] == 1).sum():,} ({(pdf['FUTURE_CRC_EVENT'] == 1).sum()/len(pdf)*100:.3f}%)")
print(f"  Negative cases: {(pdf['FUTURE_CRC_EVENT'] == 0).sum():,}")

# Calculate correlation matrix (excluding FUTURE_CRC_EVENT)
corr_matrix = pdf[remaining_numeric].corr()

# Find perfect correlations (excluding diagonal)
perfect_corrs = []
checked_pairs = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        corr_val = corr_matrix.iloc[i, j]
        
        # Use numpy.abs to avoid PySpark function conflict
        if pd.notna(corr_val) and np.abs(corr_val) >= 0.999:
            pair = tuple(sorted([col1, col2]))
            if pair not in checked_pairs:
                perfect_corrs.append({
                    'feature_1': col1,
                    'feature_2': col2,
                    'correlation': corr_val,
                    'to_remove': col2  # Remove second in alphabetical order
                })
                checked_pairs.add(pair)

if len(perfect_corrs) > 0:
    perfect_corr_df = pd.DataFrame(perfect_corrs)
    print(f"\nFound {len(perfect_corrs)} pairs of perfectly correlated features:")
    print(perfect_corr_df[['feature_1', 'feature_2', 'correlation']].to_string(index=False))
    
    features_to_remove.update(perfect_corr_df['to_remove'].tolist())
else:
    print("\nNone found")

# =============================================================================
# SUMMARY AND CREATE CLEANED TABLE
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nStarting features: {len(feature_cols)}")
print(f"Features flagged for removal: {len(features_to_remove)}")
print(f"  - Constant features: {len(near_zero_var)}")
print(f"  - Perfect correlations: {len(perfect_corrs)}")
print(f"Final feature count: {len(feature_cols) - len(features_to_remove)}")

print("\nNote: High missingness NOT used as removal criterion")
print("Reason: Rare events - missingness patterns can be highly predictive")

if len(features_to_remove) > 0:
    print("\nFeatures being removed:")
    for feat in sorted(features_to_remove):
        print(f"  - {feat}")
    
    # Create cleaned table
    # Keep identifiers, target, SPLIT (for downstream filtering), plus clean features
    # Explicitly exclude ICD10_CODE and ICD10_GROUP (outcome-related, not features)
    keep_cols = ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT', 'SPLIT'] + \
                [c for c in feature_cols if c not in features_to_remove]

    df_cleaned = df.select(keep_cols)

    df_cleaned.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("dev.clncl_ds.herald_test_train_wide_cleaned")

    print(f"\n✓ Cleaned table created: dev.clncl_ds.herald_test_train_wide_cleaned")
    print(f"  Columns: {len(keep_cols)} ({len(keep_cols) - 4} features + 2 IDs + 1 target + 1 split)")
else:
    print("\n✓ No features removed - original table is clean")
    print("  You can proceed with dev.clncl_ds.herald_test_train_wide")

print("="*80)
print("PREPROCESSING COMPLETE")
print("="*80)

# ========================================
# CELL 9
# ========================================

print("="*80)
print("MISSINGNESS AUDIT (SOP COMPLIANCE II.4.a)")
print("="*80)

from pyspark.sql import functions as F
from functools import reduce
from operator import add

# Load the cleaned table
df_audit = spark.table("dev.clncl_ds.herald_test_train_wide_cleaned")

# Get feature columns (exclude identifiers and target)
exclude_cols = ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT', 'SPLIT']
feature_cols = [c for c in df_audit.columns if c not in exclude_cols]

total_rows = df_audit.count()
positive_rows = df_audit.filter(F.col('FUTURE_CRC_EVENT') == 1).count()

print(f"\nDataset: {total_rows:,} total rows | {positive_rows:,} positive cases")
print(f"Features audited: {len(feature_cols)}")

# Calculate missingness for each feature
print("\n" + "-"*80)
print("FEATURE MISSINGNESS SUMMARY")
print("-"*80)

missingness_stats = []

for col_name in feature_cols:
    # Overall missingness
    overall_missing = df_audit.filter(F.col(col_name).isNull()).count()
    overall_pct = (overall_missing / total_rows) * 100

    # Missingness in positive cases (diseased cohort per SOP)
    positive_missing = df_audit.filter(
        (F.col('FUTURE_CRC_EVENT') == 1) & (F.col(col_name).isNull())
    ).count()
    positive_pct = (positive_missing / positive_rows) * 100 if positive_rows > 0 else 0

    missingness_stats.append({
        'feature': col_name,
        'overall_missing_pct': overall_pct,
        'positive_missing_pct': positive_pct,
        'flag': 'HIGH' if positive_pct > 30 else ('REVIEW' if positive_pct > 5 else '')
    })

# Convert to pandas for display
import pandas as pd
miss_df = pd.DataFrame(missingness_stats)
miss_df = miss_df.sort_values('positive_missing_pct', ascending=False)

# Show features with >5% missingness in positive cases
high_miss = miss_df[miss_df['positive_missing_pct'] > 5]
print(f"\nFeatures with >5% missingness in positive cases: {len(high_miss)}")
if len(high_miss) > 0:
    print(high_miss[['feature', 'overall_missing_pct', 'positive_missing_pct', 'flag']].to_string(index=False))
else:
    print("  None - all features have <5% missingness in positive cases")

# Show features with >30% missingness
very_high_miss = miss_df[miss_df['positive_missing_pct'] > 30]
print(f"\nFeatures with >30% missingness in positive cases: {len(very_high_miss)}")
if len(very_high_miss) > 0:
    print("  NOTE: Per SOP II.4.b, these would typically be removed.")
    print("  DECISION: Retained because missingness is informative for rare event prediction.")
    print(very_high_miss[['feature', 'overall_missing_pct', 'positive_missing_pct']].to_string(index=False))
else:
    print("  None")

# Summary statistics
print("\n" + "-"*80)
print("MISSINGNESS DISTRIBUTION")
print("-"*80)
print(f"  Features with 0% missing:     {len(miss_df[miss_df['overall_missing_pct'] == 0])}")
print(f"  Features with <5% missing:    {len(miss_df[miss_df['overall_missing_pct'] < 5])}")
print(f"  Features with 5-30% missing:  {len(miss_df[(miss_df['overall_missing_pct'] >= 5) & (miss_df['overall_missing_pct'] < 30)])}")
print(f"  Features with >30% missing:   {len(miss_df[miss_df['overall_missing_pct'] >= 30])}")

# Row-level missingness (SOP II.4.a also asks for cases/rows)
print("\n" + "-"*80)
print("ROW-LEVEL MISSINGNESS")
print("-"*80)

# Count missing values per row
from pyspark.sql.functions import sum as spark_sum, when, lit # Correct import

# Create expression to count nulls per row
null_countper_row = reduce(add, [when(F.col(c).isNull(), lit(1)).otherwise(lit(0)) for c in feature_cols])
null_count_expr = spark_sum(null_countper_row) # Corrected: changed spar_sum to spark_sum

row_miss = df_audit.withColumn(
    'null_count',
    reduce(add, [when(F.col(c).isNull(), lit(1)).otherwise(lit(0)) for c in feature_cols])
)

row_miss_stats = row_miss.agg(
    F.avg('null_count').alias('avg_nulls_per_row'),
    F.max('null_count').alias('max_nulls_per_row'),
    F.min('null_count').alias('min_nulls_per_row'),
    F.expr('percentile_approx(null_count, 0.5)').alias('median_nulls_per_row'),
    F.expr('percentile_approx(null_count, 0.95)').alias('p95_nulls_per_row')
).collect()[0]

print(f"  Average nulls per row:  {row_miss_stats['avg_nulls_per_row']:.1f} / {len(feature_cols)} features")
print(f"  Median nulls per row:   {row_miss_stats['median_nulls_per_row']} / {len(feature_cols)} features")
print(f"  95th percentile:        {row_miss_stats['p95_nulls_per_row']} / {len(feature_cols)} features")
print(f"  Max nulls in any row:   {row_miss_stats['max_nulls_per_row']} / {len(feature_cols)} features")

print("\n" + "="*80)
print("✓ MISSINGNESS AUDIT COMPLETE (SOP II.4.a)")
print("="*80)

# ========================================
# CELL 10
# ========================================

print("="*80)
print("FINAL FEATURE SET READY FOR MODELING")
print("="*80)

# Validate the cleaned table
final_df = spark.table("dev.clncl_ds.herald_test_train_wide_cleaned")

# Get comprehensive statistics
stats = final_df.agg(
    F.count('*').alias('total_rows'),
    F.countDistinct('PAT_ID').alias('unique_patients'),
    F.sum('FUTURE_CRC_EVENT').alias('positive_cases'),
    (F.avg('FUTURE_CRC_EVENT') * 100).alias('positive_rate'),
    (F.avg('IS_FEMALE') * 100).alias('pct_female'),
    (F.avg('HAS_PCP_AT_END') * 100).alias('pct_with_pcp')).collect()[0]

print(f"\nTable: dev.clncl_ds.herald_test_train_wide_cleaned")
print(f"\nDataset Statistics:")
print(f"  Total observations: {stats['total_rows']:,}")
print(f"  Unique patients: {stats['unique_patients']:,}")
print(f"  Positive cases: {stats['positive_cases']:,}")
print(f"  Positive rate: {stats['positive_rate']:.3f}%")
print(f"  Class imbalance: 1:{int(stats['total_rows']/stats['positive_cases'])}")

print(f"\nDemographics:")
print(f"  Female: {stats['pct_female']:.1f}%")
print(f"  Has PCP: {stats['pct_with_pcp']:.1f}%")

# Feature composition
total_cols = len(final_df.columns)
print(f"\nFeature Composition:")
print(f"  Total columns: {total_cols}")
print(f"  Features: 170")
print(f"  Identifiers: 2 (PAT_ID, END_DTTM)")
print(f"  Target: 1 (FUTURE_CRC_EVENT)")

print("\n" + "="*80)
print("NEXT STEPS: HIERARCHICAL CLUSTERING AND SHAP-BASED SELECTION")
print("="*80)

print("""
Your feature set is ready for modeling. Recommended workflow:

1. TRAIN/VAL/TEST SPLIT
   - Temporal split by END_DTTM (e.g., 60/20/20)
   - Patient-level stratification (not observation-level)
   - Preserve class balance across splits

2. HIERARCHICAL CLUSTERING
   - Use correlation as distance metric
   - Identify redundant feature groups
   - Will help with initial feature selection

3. XGBOOST WITH SHAP ITERATION
   - Start with stratified sample (500K rows, all positives)
   - scale_pos_weight for class imbalance (1:595)
   - Calculate SHAP values separately on positive/negative classes
   - Iteratively remove low-importance features
   - Target 50-75 features for final model

4. HANDLING PCP OBSERVABILITY BIAS
   - DO NOT add care gap or interaction features
   - Instead, evaluate model performance stratified by HAS_PCP_AT_END:
     * Report metrics separately for PCP vs non-PCP patients
     * Consider separate calibration curves by PCP status
     * Document differential performance in deployment guidance
   - This approach acknowledges bias through evaluation rather than
     encoding it into features

5. EXPECTED RESULTS
   - With 7,574 positive cases and 170 features: 45 cases/feature (good)
   - SHAP-based reduction should get you to 30-50 final features
   - Higher performance on PCP patients is expected and acceptable
   - Focus on calibration within each subgroup

KEY DECISION: We are NOT adding care gap, temporal, or bias interaction 
features. The domain-specific features from upstream notebooks capture 
clinical patterns. Additional derived features risk encoding observability 
bias rather than actual risk. Let SHAP discover important interactions.
""")

print("="*80)
print("✓ FEATURE ENGINEERING COMPLETE - READY FOR MODELING")
print("="*80)

# ========================================
# CELL 11
# ========================================

df_check_spark = spark.sql('select * from dev.clncl_ds.herald_test_train_wide_cleaned')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 12
# ========================================

display(df_check)

# ========================================
# CELL 13
# ========================================

df_check.shape


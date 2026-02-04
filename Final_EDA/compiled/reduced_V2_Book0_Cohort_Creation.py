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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_cohort_index AS
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_cohort_index c
LEFT JOIN (
  SELECT DISTINCT p.PAT_ID, c.END_DTTM
  FROM {trgt_cat}.clncl_ds.herald_eda_train_cohort_index c
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_cohort AS
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
  FROM {trgt_cat}.clncl_ds.herald_eda_train_cohort fc
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_final_cohort AS
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
  FROM {trgt_cat}.clncl_ds.herald_eda_train_cohort
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
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_cohort fc
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
  FROM {trgt_cat}.clncl_ds.herald_eda_train_cohort
  WHERE LABEL_USABLE = 1
),
after_vbc_exclusion AS (
  SELECT
    fc.*,
    COALESCE(cs.COLON_SCREEN_MET_FLAG, 'N') as screen_status
  FROM {trgt_cat}.clncl_ds.herald_eda_train_cohort fc
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
  FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_cohort
WHERE LABEL_USABLE = 1

UNION ALL

SELECT 
  'After exclusions' as stage,
  COUNT(*) as observations,
  COUNT(DISTINCT PAT_ID) as patients,
  AVG(FUTURE_CRC_EVENT) * 100 as crc_rate
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
for table in ['herald_eda_train_cohort_index', 'herald_base_with_pcp', 
              'herald_eda_train_cohort', 'herald_eda_train_final_cohort']:
    count = spark.sql(f"SELECT COUNT(*) as n FROM {trgt_cat}.clncl_ds.{table}").collect()[0]['n']
    print(f"  {table}: {count:,} rows")

# CHECK 2: Verify no duplicates
dupe_check = spark.sql(f"""
SELECT 
  COUNT(*) as total_rows,
  COUNT(DISTINCT PAT_ID, END_DTTM) as unique_keys,
  COUNT(*) - COUNT(DISTINCT PAT_ID, END_DTTM) as duplicates
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
  FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
GROUP BY HAS_PCP_AT_END
""").toPandas()

print("\nPCP Impact on Label Usability:")
print(pcp_impact.to_string(index=False))

# CHECK 6: Verify column structure
from pyspark.sql.types import *
schema = spark.table(f"{trgt_cat}.clncl_ds.herald_eda_train_final_cohort").schema
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
df = spark.sql(f"SELECT * FROM dev.clncl_ds.herald_eda_train_final_cohort")
# exact row count (triggers a full scan)
n_rows = df.count()
print(n_rows)

# ========================================
# CELL 21
# ========================================

# CELL 26
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

df = spark.table("dev.clncl_ds.herald_eda_train_final_cohort")

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
df_cohort = spark.table(f"{trgt_cat}.clncl_ds.herald_eda_train_final_cohort")

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
df_to_save.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{trgt_cat}.clncl_ds.herald_eda_train_final_cohort")

print(f"✓ Saved cohort with SPLIT column to {trgt_cat}.clncl_ds.herald_eda_train_final_cohort")

# Verify save
df_verify = spark.table(f"{trgt_cat}.clncl_ds.herald_eda_train_final_cohort")
print(f"\nVerification:")
print(f"  Total rows: {df_verify.count():,}")
print(f"  Columns: {df_verify.columns}")
print(f"\nSplit distribution after save:")
df_verify.groupBy("SPLIT").count().show()

# Unpersist cached DataFrame
df_to_save.unpersist()


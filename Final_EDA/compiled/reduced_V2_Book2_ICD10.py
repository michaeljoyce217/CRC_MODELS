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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_curated_conditions AS

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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_conditions_all AS

-- 12-MONTH WINDOW
-- Recent conditions that may indicate active disease or symptoms
SELECT
  c.PAT_ID,
  c.END_DTTM,
  cc.CODE,
  cc.ENC_ID,
  DATE(cc.CONTACT_DATE) AS CONTACT_DATE,
  '12mo' AS WINDOW
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
JOIN {trgt_cat}.clncl_ds.herald_eda_curated_conditions cc
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
JOIN {trgt_cat}.clncl_ds.herald_eda_curated_conditions cc
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
JOIN {trgt_cat}.clncl_ds.herald_eda_curated_conditions cc
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_icd10_symptoms AS

WITH 
-- Base cohort from our main training set
cohort AS (
    SELECT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
    LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_conditions_all ca
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
    LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_conditions_all ca
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_icd10_risk_factors AS

WITH cohort AS (
    SELECT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
    LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_conditions_all ca
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_icd10_other AS

WITH cohort AS (
    SELECT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
    LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_conditions_all ca
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_icd10_enhanced AS

WITH cohort AS (
    SELECT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
    LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_conditions_all ca
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_icd_10_temp AS

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

FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_icd10_symptoms s
    ON c.PAT_ID = s.PAT_ID AND c.END_DTTM = s.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_icd10_risk_factors r
    ON c.PAT_ID = r.PAT_ID AND c.END_DTTM = r.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_icd10_other o
    ON c.PAT_ID = o.PAT_ID AND c.END_DTTM = o.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_icd10_enhanced e
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_dx_for_scoring AS

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
FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_charlson_scores AS

WITH 
-- Use pre-extracted diagnoses from CELL 10 instead of re-querying
dx_12mo AS (
    SELECT DISTINCT PAT_ID, END_DTTM, ICD10_CODE
    FROM {trgt_cat}.clncl_ds.herald_eda_dx_for_scoring
    WHERE IN_12MO = 1
),

dx_24mo AS (
    SELECT DISTINCT PAT_ID, END_DTTM, ICD10_CODE
    FROM {trgt_cat}.clncl_ds.herald_eda_dx_for_scoring
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_other_scores AS

WITH 
-- Use pre-extracted diagnoses from CELL 10 instead of re-querying
dx_12mo AS (
    SELECT DISTINCT PAT_ID, END_DTTM, ICD10_CODE
    FROM {trgt_cat}.clncl_ds.herald_eda_dx_for_scoring
    WHERE IN_12MO = 1
),

dx_24mo AS (
    SELECT DISTINCT PAT_ID, END_DTTM, ICD10_CODE
    FROM {trgt_cat}.clncl_ds.herald_eda_dx_for_scoring
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_icd_10 AS

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

FROM {trgt_cat}.clncl_ds.herald_eda_train_icd_10_temp t
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_charlson_scores c
    ON t.PAT_ID = c.PAT_ID AND t.END_DTTM = c.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_other_scores o
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
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_eda_train_icd_10
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
    
FROM {trgt_cat}.clncl_ds.herald_eda_train_icd_10
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
    FROM {trgt_cat}.clncl_ds.herald_eda_train_icd_10 i
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
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
    
FROM {trgt_cat}.clncl_ds.herald_eda_train_icd_10
""").show(truncate=False)

print("\n✓ Comorbidity score analysis complete")

# ========================================
# CELL 18
# ========================================

# CELL 17
df_spark = spark.sql('''select * from dev.clncl_ds.herald_eda_train_icd_10''')

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
FROM dev.clncl_ds.herald_eda_train_icd_10
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_icd_10
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
df_icd = spark.table("dev.clncl_ds.herald_eda_train_icd_10")

# Add icd_ prefix to all columns except keys
icd_cols = [col for col in df_icd.columns if col not in ['PAT_ID', 'END_DTTM']]
for col in icd_cols:
    df_icd = df_icd.withColumnRenamed(col, f'icd_{col}' if not col.startswith('icd_') else col)

# Join with cohort to get outcome variable
df_cohort = spark.sql("""
    SELECT PAT_ID, END_DTTM, FUTURE_CRC_EVENT
    FROM dev.clncl_ds.herald_eda_train_final_cohort
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
output_table = 'dev.clncl_ds.herald_eda_train_icd10_reduced'
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

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_eda_train_icd10_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 29
# ========================================

display(df_check)


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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_inpatient_basic_labs AS

WITH
    --------------------------------------------------------------------------
    -- 1) cohort: Our base population
    --------------------------------------------------------------------------
    cohort AS (
        SELECT PAT_ID, END_DTTM
        FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
spark.sql(f"SELECT COUNT(*) as row_count, COUNT(DISTINCT PAT_ID) as patients FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_basic_labs").show()

# ========================================
# CELL 3
# ========================================

# ---------------------------------
# CELL 1B: Enhanced Inpatient Labs - Intermediate Table 1
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_inpatient_labs_raw AS

WITH cohort AS (
    SELECT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
spark.sql(f"SELECT COUNT(*) as row_count FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_labs_raw").show()

# ========================================
# CELL 4
# ========================================

# ---------------------------------
# CELL 1C: Process Inpatient Labs with Normalization
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_inpatient_labs_processed AS

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
        
    FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_labs_raw
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_labs_processed
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_outpatient_labs_raw AS

WITH expanded_cohort AS (
    SELECT pat_id, end_dttm
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
spark.sql(f"SELECT COUNT(*) as row_count FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_labs_raw").show()

# ========================================
# CELL 6
# ========================================

# ---------------------------------  
# CELL 2B: Process Outpatient Labs with Normalization
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_outpatient_labs_processed AS

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
        
    FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_labs_raw
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_labs_processed
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_combined_labs_all AS

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
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_labs_processed

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
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_labs_processed
''')

# Quality check
spark.sql(f'''
SELECT 
    SOURCE,
    COUNT(*) as total_records,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    COUNT(DISTINCT COMPONENT_NAME) as unique_labs,
    SUM(CASE WHEN ABNORMAL_YN = 'Y' THEN 1 ELSE 0 END) as abnormal_count
FROM {trgt_cat}.clncl_ds.herald_eda_train_combined_labs_all
GROUP BY SOURCE
''').show()

# ========================================
# CELL 8
# ========================================

# ---------------------------------
# CELL 4A: Calculate Iron Saturation and Anemia Classification (FIXED)
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_labs_anemia_features AS

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
    FROM {trgt_cat}.clncl_ds.herald_eda_train_combined_labs_all i
    JOIN {trgt_cat}.clncl_ds.herald_eda_train_combined_labs_all t
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
    FROM {trgt_cat}.clncl_ds.herald_eda_train_combined_labs_all
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
    FROM {trgt_cat}.clncl_ds.herald_eda_train_combined_labs_all
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
    FROM {trgt_cat}.clncl_ds.herald_eda_train_combined_labs_all
    WHERE COMPONENT_NAME = 'FERRITIN'
),

-- Get cohort to ensure all patients are included
cohort AS (
    SELECT DISTINCT PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_labs_pivoted_enhanced AS

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
    FROM {trgt_cat}.clncl_ds.herald_eda_train_combined_labs_all
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_labs_trends_enhanced AS

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
        
    FROM {trgt_cat}.clncl_ds.herald_eda_train_combined_labs_all
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
    FROM {trgt_cat}.clncl_ds.herald_eda_train_combined_labs_all
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_trends_enhanced
''').show()

# ========================================
# CELL 11
# ========================================

# ---------------------------------
# CELL 8: Final Combined Lab Features with All Enhancements
# ---------------------------------

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_labs_final AS

WITH cohort AS (
    SELECT PAT_ID, END_DTTM, FUTURE_CRC_EVENT
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_labs_pivoted_enhanced lp
    ON c.PAT_ID = lp.PAT_ID AND c.END_DTTM = lp.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_labs_anemia_features af
    ON c.PAT_ID = af.PAT_ID AND c.END_DTTM = af.END_DTTM
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_labs_trends_enhanced lt
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
    
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final

UNION ALL

SELECT
    'IRON_DEFICIENCY_ANEMIA' as feature,
    SUM(IRON_DEFICIENCY_ANEMIA_FLAG) as feature_present,
    SUM(CASE WHEN IRON_DEFICIENCY_ANEMIA_FLAG = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN IRON_DEFICIENCY_ANEMIA_FLAG = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN IRON_DEFICIENCY_ANEMIA_FLAG = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final

UNION ALL

SELECT
    'HEMOGLOBIN_DROP_10PCT' as feature,
    SUM(HEMOGLOBIN_DROP_10PCT_FLAG) as feature_present,
    SUM(CASE WHEN HEMOGLOBIN_DROP_10PCT_FLAG = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN HEMOGLOBIN_DROP_10PCT_FLAG = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN HEMOGLOBIN_DROP_10PCT_FLAG = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final

UNION ALL

SELECT
    'ALBUMIN_DROP_15PCT' as feature,
    SUM(ALBUMIN_DROP_15PCT_FLAG) as feature_present,
    SUM(CASE WHEN ALBUMIN_DROP_15PCT_FLAG = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN ALBUMIN_DROP_15PCT_FLAG = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN ALBUMIN_DROP_15PCT_FLAG = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final
'''

associations_df = spark.sql(association_query)

# Calculate risk ratios - FIXED: Use actual total_rows
total_rows = spark.table(f"{trgt_cat}.clncl_ds.herald_eda_train_labs_final").count()
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final
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
    COUNT(*) / (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final WHERE ANEMIA_GRADE IS NOT NULL) * 100 as pct
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final

UNION ALL

SELECT 'PLATELETS_ACCEL_RISE' as feature,
    SUM(PLATELETS_ACCELERATING_RISE) as feature_present,
    SUM(CASE WHEN PLATELETS_ACCELERATING_RISE = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN PLATELETS_ACCELERATING_RISE = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN PLATELETS_ACCELERATING_RISE = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final

UNION ALL

SELECT 'CRP_ACCEL_RISE' as feature,
    SUM(CRP_ACCELERATING_RISE) as feature_present,
    SUM(CASE WHEN CRP_ACCELERATING_RISE = 1 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN CRP_ACCELERATING_RISE = 1 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN CRP_ACCELERATING_RISE = 0 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final

UNION ALL

SELECT 'HGB_RAPID_TRAJECTORY' as feature,
    SUM(CASE WHEN HGB_TRAJECTORY = 'RAPID_DECLINE' THEN 1 ELSE 0 END) as feature_present,
    SUM(CASE WHEN HGB_TRAJECTORY = 'RAPID_DECLINE' AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN HGB_TRAJECTORY = 'RAPID_DECLINE' THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN HGB_TRAJECTORY != 'RAPID_DECLINE' OR HGB_TRAJECTORY IS NULL THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final

UNION ALL

SELECT 'SEVERE_ANEMIA_COMBO' as feature,
    SUM(CASE WHEN ANEMIA_SEVERITY_SCORE >= 4 THEN 1 ELSE 0 END) as feature_present,
    SUM(CASE WHEN ANEMIA_SEVERITY_SCORE >= 4 AND FUTURE_CRC_EVENT = 1 THEN 1 ELSE 0 END) as crc_with_feature,
    AVG(CASE WHEN ANEMIA_SEVERITY_SCORE >= 4 THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_feature,
    AVG(CASE WHEN ANEMIA_SEVERITY_SCORE < 4 OR ANEMIA_SEVERITY_SCORE IS NULL THEN FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_feature
FROM {trgt_cat}.clncl_ds.herald_eda_train_labs_final
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
df = spark.sql('''select * from dev.clncl_ds.herald_eda_train_labs_final''')
df.count()

# ========================================
# CELL 16
# ========================================

# CELL 13
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
df_labs = spark.table("dev.clncl_ds.herald_eda_train_labs_final")
df_labs = df_labs.drop("FUTURE_CRC_EVENT")

# Load cohort with FUTURE_CRC_EVENT and SPLIT column
df_cohort = spark.sql("""
    SELECT PAT_ID, END_DTTM, FUTURE_CRC_EVENT, SPLIT
    FROM dev.clncl_ds.herald_eda_train_final_cohort
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
output_table = 'dev.clncl_ds.herald_eda_train_labs_reduced'
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
FROM dev.clncl_ds.herald_eda_train_labs_reduced r
JOIN dev.clncl_ds.herald_eda_train_final_cohort c
    ON r.PAT_ID = c.PAT_ID 
    AND r.END_DTTM = c.END_DTTM
GROUP BY r.lab_comprehensive_iron_deficiency
'''

spark.sql(validation_query).show()

# ========================================
# CELL 25
# ========================================

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_eda_train_labs_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 26
# ========================================

display(df_check)


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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_proc_category_map
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_proc_unpivoted AS

WITH
    cohort AS (
        SELECT PAT_ID, END_DTTM
        FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
        FROM {trgt_cat}.clncl_ds.herald_eda_train_proc_category_map m
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_proc_unpivoted
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
           FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort), 2) as pct_of_cohort
FROM {trgt_cat}.clncl_ds.herald_eda_train_proc_unpivoted
GROUP BY CATEGORY
ORDER BY procedure_count DESC
""").show()

print("="*70)

# ========================================
# CELL 4
# ========================================

# Cell 3
spark.sql(f"""
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_iron_infusions AS

WITH
    cohort AS (
        SELECT PAT_ID, END_DTTM
        FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
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
FROM {trgt_cat}.clncl_ds.herald_eda_train_iron_infusions
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
    JOIN dev.clncl_ds.herald_eda_train_proc_category_map m
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
FROM dev.clncl_ds.herald_eda_train_final_cohort fc
LEFT JOIN dev.clncl_ds.herald_eda_train_proc_unpivoted pp
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
FROM dev.clncl_ds.herald_eda_train_proc_category_map m
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
FROM dev.clncl_ds.herald_eda_train_proc_unpivoted
GROUP BY YEAR(PROC_DATE), QUARTER(PROC_DATE)
ORDER BY year, quarter
""").toPandas()

print(temporal_dist.to_string(index=False))

# Check 5: Validate no colonoscopy procedures
print("\n5. Colonoscopy Exclusion Verification")
print("-" * 60)

colonoscopy_check = spark.sql("""
SELECT COUNT(*) as colonoscopy_count
FROM dev.clncl_ds.herald_eda_train_proc_unpivoted
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
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_patient_procs AS

WITH
    cohort AS (
        SELECT PAT_ID, END_DTTM
        FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
    ),

    unpvt AS (
        SELECT
            PAT_ID,
            END_DTTM,
            CATEGORY,
            PROC_DATE,
            DATEDIFF(END_DTTM, PROC_DATE) AS DAYS_SINCE_PROC
        FROM {trgt_cat}.clncl_ds.herald_eda_train_proc_unpivoted
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
LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_iron_infusions ii
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
    
FROM {trgt_cat}.clncl_ds.herald_eda_train_patient_procs
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
df_spark = spark.sql('''SELECT * FROM dev.clncl_ds.herald_eda_train_patient_procs''')
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
df_procs = spark.table("dev.clncl_ds.herald_eda_train_patient_procs")

# Load cohort with FUTURE_CRC_EVENT
df_cohort = spark.sql("""
    SELECT PAT_ID, END_DTTM, FUTURE_CRC_EVENT
    FROM dev.clncl_ds.herald_eda_train_final_cohort
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
output_table = 'dev.clncl_ds.herald_eda_train_procedures_reduced'
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

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_eda_train_procedures_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 20
# ========================================

display(df_check)


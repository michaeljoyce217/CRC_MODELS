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
    'vitals': 'dev.clncl_ds.herald_eda_train_vitals_reduced',
    'icd10': 'dev.clncl_ds.herald_eda_train_icd10_reduced',
    'labs': 'dev.clncl_ds.herald_eda_train_labs_reduced',
    'outpatient_meds': 'dev.clncl_ds.herald_eda_train_outpatient_meds_reduced',
    'inpatient_meds': 'dev.clncl_ds.herald_eda_train_inpatient_meds_reduced',
    'visit_features': 'dev.clncl_ds.herald_eda_train_visit_features_reduced',
    'procedures': 'dev.clncl_ds.herald_eda_train_procedures_reduced'
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
CREATE OR REPLACE TABLE dev.clncl_ds.herald_eda_train_wide AS
SELECT
    c.* EXCEPT (LABEL_CONFIDENCE, current_screen_status,
                vbc_last_colonoscopy_date, vbc_last_fobt_date,
                last_internal_screening_date, last_colonoscopy_date,
                last_ct_colonography_date, last_sigmoidoscopy_date,
                last_fit_dna_date, last_fobt_date,
                fobt_count, had_fobt_in_lookback),
    v.* EXCEPT (PAT_ID, END_DTTM),
    i.* EXCEPT (PAT_ID, END_DTTM),
    l.* EXCEPT (PAT_ID, END_DTTM),
    om.* EXCEPT (PAT_ID, END_DTTM),
    im.* EXCEPT (PAT_ID, END_DTTM),
    vis.* EXCEPT (PAT_ID, END_DTTM),
    p.* EXCEPT (PAT_ID, END_DTTM)
FROM dev.clncl_ds.herald_eda_train_final_cohort AS c
LEFT JOIN dev.clncl_ds.herald_eda_train_vitals_reduced AS v USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_eda_train_icd10_reduced AS i USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_eda_train_labs_reduced AS l USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_eda_train_outpatient_meds_reduced AS om USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_eda_train_inpatient_meds_reduced AS im USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_eda_train_visit_features_reduced AS vis USING (PAT_ID, END_DTTM)
LEFT JOIN dev.clncl_ds.herald_eda_train_procedures_reduced AS p USING (PAT_ID, END_DTTM)
""")

print("✓ Wide table created: dev.clncl_ds.herald_eda_train_wide")

# ========================================
# CELL 5
# ========================================

print("="*80)
print("TRANSFORMING FEATURES TO PREVENT MEMORIZATION")
print("="*80)

from pyspark.sql import functions as F
from pyspark.sql.functions import when, col

# Load the wide table
df = spark.table("dev.clncl_ds.herald_eda_train_wide")

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
df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("dev.clncl_ds.herald_eda_train_wide_transformed")

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

print("\n✓ Transformed table saved: dev.clncl_ds.herald_eda_train_wide_transformed")
print("  Ready for preprocessing and feature selection")

# ========================================
# CELL 6
# ========================================

# Load the wide table
df = spark.table("dev.clncl_ds.herald_eda_train_wide_transformed")

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
    FROM dev.clncl_ds.herald_eda_train_wide
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

    df_cleaned.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("dev.clncl_ds.herald_eda_train_wide_cleaned")

    print(f"\n✓ Cleaned table created: dev.clncl_ds.herald_eda_train_wide_cleaned")
    print(f"  Columns: {len(keep_cols)} ({len(keep_cols) - 4} features + 2 IDs + 1 target + 1 split)")
else:
    print("\n✓ No features removed - original table is clean")
    print("  You can proceed with dev.clncl_ds.herald_eda_train_wide")

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
df_audit = spark.table("dev.clncl_ds.herald_eda_train_wide_cleaned")

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
final_df = spark.table("dev.clncl_ds.herald_eda_train_wide_cleaned")

# Get comprehensive statistics
stats = final_df.agg(
    F.count('*').alias('total_rows'),
    F.countDistinct('PAT_ID').alias('unique_patients'),
    F.sum('FUTURE_CRC_EVENT').alias('positive_cases'),
    (F.avg('FUTURE_CRC_EVENT') * 100).alias('positive_rate'),
    (F.avg('IS_FEMALE') * 100).alias('pct_female'),
    (F.avg('HAS_PCP_AT_END') * 100).alias('pct_with_pcp')).collect()[0]

print(f"\nTable: dev.clncl_ds.herald_eda_train_wide_cleaned")
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

df_check_spark = spark.sql('select * from dev.clncl_ds.herald_eda_train_wide_cleaned')
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


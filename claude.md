# MASTER PROMPT: CRC Prediction Feature Selection Methodology Improvement

## Project Overview

This project is improving the **feature selection methodology** for a **colorectal cancer (CRC) risk prediction model** with highly imbalanced data (250:1 negative:positive ratio). The model predicts CRC diagnosis within 6 months for unscreened patients.

**Current Status**: All notebooks complete. Rerunning Books 0-9 after removing circular-reasoning features (CEA, CA 19-9, FOBT/FIT) from Book 4. Training scripts (`train.py`, `train_optuna.py`, `featurization_train.py`) ready; feature lists will be updated after Book 9 rerun.

---

## FAILED APPROACHES (DO NOT REPEAT)

This section documents approaches that were tried and FAILED during Book 9 development. Do not repeat these mistakes.

### Failed Approach 1: Aggressive XGBoost Parameters

**What we tried**: Using standard XGBoost parameters (max_depth=4, gamma=1.0, subsample=0.5, learning_rate=0.01)

**What happened**: Models collapsed after a few iterations. Iteration 2 had 894 trees, iteration 3 had only 29 trees. The model couldn't learn anything, early stopping kicked in immediately.

**Why it failed**: With 250:1 class imbalance, aggressive parameters cause overfitting and unstable models. When features are removed, the model destabilizes completely.

**Correct approach**: Use VERY conservative parameters from original methodology (max_depth=2, gamma=2.0, subsample=0.3, learning_rate=0.005).

---

### Failed Approach 2: Percentile-Based Thresholds for Removal

**What we tried**: "Remove features below 15th percentile SHAP" or "two-tier: 15th percentile standard, 5th percentile for last-in-cluster"

**What happened**: When many features have SHAP values of exactly 0, the 15th percentile IS 0. Nothing is < 0, so nothing gets removed. Pipeline stalls after 1-2 iterations with "No features meet removal criteria."

**Why it failed**: Percentile thresholds break when SHAP values cluster at 0. The threshold becomes 0, and the condition `shap < 0` is never true.

**Correct approach**: Use the original multi-criteria approach: feature must meet 2+ of 3 criteria (near-zero SHAP < 0.0002, negative-biased ratio < 0.15, bottom 8%). This is more robust.

---

### Failed Approach 3: Rank-Based Removal ("Remove Bottom N")

**What we tried**: "Remove bottom 15% of features by rank each iteration" regardless of their actual SHAP values.

**What happened**: Blindly removed features that might have been valuable, destabilizing the model.

**Why it failed**: Not all "bottom" features are useless. Some may have low SHAP but still be important. The original methodology's multi-criteria approach ensures only truly useless features are removed.

**Correct approach**: Require features to meet 2+ of 3 criteria before removal.

---

### Failed Approach 4: Using CURRENT Cluster Sizes for Removal Caps

**What we tried**: Recalculating cluster sizes each iteration based on remaining features.

**What happened**: As features were removed, clusters shrank, making the removal caps more restrictive. An 8-feature cluster (cap=3) becomes a 5-feature cluster (cap=0) after removing 3 features.

**Why it failed**: The cluster protection rules became so restrictive that nothing could be removed after a few iterations.

**Correct approach**: Use ORIGINAL cluster sizes from Phase 1 (stored in selection_df). The original methodology reads from the cluster CSV file each iteration, always using original sizes.

---

### Failed Approach 5: Silhouette-Based Clustering Threshold Selection (Unconstrained)

**What we tried**: Let silhouette optimization pick the threshold automatically.

**What happened**: Silhouette optimization picked threshold ~0.5, creating 87 clusters for 172 features (average 2 features per cluster). With MIN_PER_CLUSTER=1, you're locked into keeping 87 features minimum.

**Why it failed**: Too many tiny clusters = no room for Phase 2 winnowing. Cluster protection blocks almost all removals.

**Correct approach**: Constrain threshold selection to [0.6, 0.85] range, targeting 40-70 clusters.

---

### Failed Approach 6: Fixed Clustering Threshold (0.75 or 0.7)

**What we tried**: Hardcoding a specific threshold value.

**What happened**: Different datasets may have different correlation structures. A fixed threshold that worked before may not work now.

**Why it failed**: The original methodology had fewer features with a different correlation structure. Blindly copying the threshold is wrong.

**Correct approach**: Use smart selection within constrained range [0.6, 0.85], targeting a specific cluster count (40-70), using silhouette as a tiebreaker.

---

### Summary: The Original Methodology Works

The **original CRC_ITER1_MODEL-PREVALENCE.py** was a manual iterative process that worked successfully. Key elements:

1. **Very conservative XGBoost** (max_depth=2, learning_rate=0.005, etc.)
2. **Multi-criteria removal** (2+ of 3: near-zero, neg-biased, bottom 8%)
3. **Cluster-specific caps** using ORIGINAL sizes
4. **High-importance cluster protection** (top 20% max_shap = max 1 removal)
5. **Global cap** of 30 removals per iteration

The automated Book 9 now implements this exact logic.

---

## Directory Structure

```
CRC_MODELS/
├── 2nd_Dataset_Creation/           # TRANSFORMED notebooks (working versions)
│   ├── V2_Book0_Cohort_Creation.py # Base cohort with SGKF splits (COMPLETE)
│   ├── V2_Book1_Vitals.py          # Train-only feature selection (COMPLETE)
│   ├── V2_Book2_ICD10.py           # Train-only feature selection (COMPLETE)
│   ├── V2_Book3_Social_Factors.py  # No changes needed (all features excluded)
│   ├── V2_Book4_Labs_Combined.py   # Labs - CEA/CA19-9/FOBT removed (COMPLETE)
│   ├── V2_Book5_1_Medications_Outpatient.py  # Train-only feature selection (COMPLETE)
│   ├── V2_Book5_2_Medications_Inpatient.py   # Train-only feature selection (COMPLETE)
│   ├── V2_Book6_Visit_History.py   # Train-only feature selection (COMPLETE)
│   ├── V2_Book7_Procedures.py      # Train-only feature selection (COMPLETE)
│   ├── V2_Book8_Compilation.py     # No changes needed (just joins tables)
│   ├── V2_Book9_Feature_Selection.py  # Feature selection pipeline (COMPLETE)
│   ├── featurization_train.py      # Production training featurization (standalone)
│   ├── train.py                    # Conservative XGBoost training (standalone)
│   └── train_optuna.py             # Optuna hyperparameter tuning (standalone)
├── docs/
│   ├── presentation_design.md      # HTML presentation design doc
│   └── book4_cea_fobt_removal_guide.md  # Cell-by-cell Book 4 change guide
├── Original_2nd_Dataset_Creation/  # ORIGINAL notebooks (reference/backup)
│   └── (same file structure)
├── Original_Methodology/           # Original clustering/SHAP notebooks (analyzed)
│   ├── CORRELATION_HIERARCHICAL_FEATURE_CLUSTERING.py
│   └── CRC_ITER1_MODEL-PREVALENCE.py
└── Prompts/                        # Additional prompts/documentation
```

## Completed Work Summary

### Phase 1: Book 0 - Cohort Creation with Stratified Patient Split (COMPLETE)

Added train/val/test split assignments using **stratified patient-level random split**:
- **TRAIN (70%)**: Randomly selected patients, stratified by cancer type
- **VAL (15%)**: Randomly selected patients, stratified by cancer type
- **TEST (15%)**: Randomly selected patients, stratified by cancer type

**Stratification classes:**
- 0 = Negative (no CRC diagnosis)
- 1 = C18 (colon cancer)
- 2 = C19 (rectosigmoid junction cancer)
- 3 = C20 (rectal cancer)

Key guarantees:
- NO patient appears in multiple splits (all observations from one patient go to same split)
- **Cancer type distribution (C18/C19/C20) preserved** across all splits
- **Similar positive rates** across train/val/test (no population bias)
- Random seed 217 for reproducibility

**Why stratified random split (not temporal)?** A temporal split (Q6 patients → TEST) created population bias: patients active in Q6 had different characteristics than those who exited earlier, causing 0.7% positive rate in train vs 0.28% in test (2.5x difference). The stratified random split ensures balanced populations. Temporal analysis can still be done by evaluating per-quarter performance within each split.

### Phase 2: Books 1-8 - Train-Only Feature Selection (COMPLETE)

All feature engineering notebooks now filter on `SPLIT='train'` for feature selection metrics:

| Book | Status | Changes |
|------|--------|---------|
| Book 1 (Vitals) | COMPLETE | df_train for risk ratios & MI |
| Book 2 (ICD10) | COMPLETE | df_train for risk ratios & MI |
| Book 3 (Social) | NO CHANGES | All features excluded due to data quality |
| Book 4 (Labs) | COMPLETE | df_train for risk ratios & MI; CEA/CA19-9/FOBT removed |
| Book 5.1 (Outpatient Meds) | COMPLETE | df_train for risk ratios & MI |
| Book 5.2 (Inpatient Meds) | COMPLETE | df_train for risk ratios & MI |
| Book 6 (Visit History) | COMPLETE | df_train for risk ratios & MI |
| Book 7 (Procedures) | COMPLETE | df_train for risk ratios & MI |
| Book 8 (Compilation) | NO CHANGES | Just joins reduced tables |

### Phase 3: Original Methodology Analysis (COMPLETE)

Analyzed two notebooks in Original_Methodology folder:

**1. CORRELATION_HIERARCHICAL_FEATURE_CLUSTERING.py** (465 lines)
- Spearman correlation with distance = 1 - |correlation|
- Average linkage hierarchical clustering
- Fixed threshold 0.7 (correlation > 0.3)
- Outputs cluster assignments to CSV

**2. CRC_ITER1_MODEL-PREVALENCE.py** (1873 lines)
- SHAP calculation separate for positive/negative cases
- Combined importance: `(importance_pos * 2 + importance_neg) / 3`
- Removal criteria: near-zero (<0.0002), neg-biased ratio (<0.15), bottom 8%
- Features must meet 2+ criteria for removal
- Iterative removal with cluster preservation rules

### Issues Identified in Original Methodology

| Issue | Problem | Impact |
|-------|---------|--------|
| **SHAP Weighting** | 2:1 positive weight for 250:1 imbalance | Negatives dominate ~99% of combined importance |
| **Fixed Threshold** | 0.7 hardcoded for all features | May be suboptimal; no data-driven justification |
| **No Train-Only Split** | Clustering on full dataset | Potential data leakage (now fixed in Books 1-8) |

---

## Phase 4: New Feature Selection Pipeline (COMPLETE)

### Pipeline Overview

Create `V2_Book9_Feature_Selection.py` using a **Hybrid Two-Phase Approach**:

| Phase | Method | Features | Purpose |
|-------|--------|----------|---------|
| **Phase 1** | Cluster-Based Reduction | 171 → ~70-80 | Remove redundant/correlated features |
| **Phase 2** | Iterative SHAP Winnowing | ~70-80 → Final | Fine-tune with 20-25 removals per iteration |

### Input
- Wide feature table from Book 8 compilation (~171 features)
- Train/val/test split assignments from Book 0

### Output
- Final reduced feature set (determined by validation gate, not arbitrary target)
- Cluster assignments with justification
- SHAP importance rankings per iteration
- Iteration tracking CSV with overfitting metrics

---

### PHASE 1: Cluster-Based Reduction

**Goal**: Remove redundant features by clustering correlated features and selecting representatives.

```
Step 1.1: Load Data
├── Load wide feature table from Book 8
├── Filter to SPLIT='train' only for correlation computation
└── Verify features loaded

Step 1.2: Compute Correlation Matrix
├── Spearman correlation (handles non-linear relationships)
├── Distance matrix: distance = 1 - |correlation|
└── Compute on TRAINING DATA ONLY

Step 1.3: Smart Threshold Selection
├── Test thresholds: 0.50 to 0.90 in 0.05 increments
├── Constrain to [0.60, 0.85] range
├── Target 40-70 clusters
├── Use silhouette optimization within constraints
└── CHECKPOINT: Save clusters

Step 1.4: Train Baseline Model (CONSERVATIVE PARAMS)
├── XGBoost with very conservative hyperparameters:
│   ├── max_depth: 2 (very shallow)
│   ├── gamma: 2.0 (high min loss reduction)
│   ├── subsample: 0.3 (low row sampling)
│   ├── colsample_bytree: 0.3 (low column sampling)
│   ├── reg_alpha: 5.0 (L1 regularization)
│   ├── reg_lambda: 50.0 (L2 regularization)
│   ├── learning_rate: 0.005 (very slow)
│   └── early_stopping_rounds: 150
└── CHECKPOINT: Save baseline model and metrics

Step 1.5: Compute SHAP with 2:1 Positive Weighting
├── TreeExplainer on baseline model
├── 2:1 weighting for positive cases
├── SHAP_Ratio = importance_pos / importance_neg
└── CHECKPOINT: Save SHAP values

Step 1.6: Select Cluster Representatives
├── For each cluster, keep top features by SHAP_Ratio
├── Drop at most 1-2 per cluster (conservative)
└── CHECKPOINT: Save selection_df with ORIGINAL cluster sizes

Step 1.7: Phase 1 Validation Gate
├── PASS if: val_auprc_drop < 10%
└── CHECKPOINT: Save phase1_complete
```

---

### PHASE 2: Iterative SHAP Winnowing (Original Methodology)

**Based on CRC_ITER1_MODEL-PREVALENCE.py** - the proven manual approach, now automated.

**Removal Criteria**: Feature must meet **AT LEAST 2 of 3** criteria:
1. Near-zero SHAP importance (< 0.0002)
2. Negative-biased ratio (< 0.15)
3. Bottom 8% by SHAP

**Cluster-Specific Removal Caps** (using ORIGINAL cluster sizes from Phase 1):
- Singleton (1 feature): max 1 removal
- Small (2-3 features): max 2 removals, leave at least 1
- Medium (4-7 features): max 2 removals, leave at least 3
- Large (8+ features): max 3 removals, leave at least 5
- High-importance clusters (top 20% by max SHAP): max 1 removal

**Global Cap**: 30 removals per iteration

```
For each iteration:

Step 2.1: Train Model (Conservative XGBoost)
├── Same conservative params as Phase 1
├── Track train, val, AND test AUPRC
└── CHECKPOINT: Save model and metrics

Step 2.2: Compute SHAP with 2:1 Positive Weighting
├── SHAP_Combined = weighted importance
├── SHAP_Ratio = importance_pos / importance_neg
└── CHECKPOINT: Save SHAP values

Step 2.3: Identify Removal Candidates (2+ of 3 Criteria)
├── Calculate features meeting each criterion:
│   ├── Near-zero SHAP (< 0.0002)
│   ├── Negative-biased ratio (< 0.15)
│   └── Bottom 8%
├── Feature must meet 2+ criteria to be flagged
├── Apply cluster-specific caps (using ORIGINAL sizes)
├── Apply high-importance cluster protection
├── Cap at 30 removals
└── Clinical must-keep features always protected

Step 2.4: Stop Conditions
├── Would go below MIN_FEATURES_THRESHOLD (25): STOP
├── No features meet 2+ criteria: STOP
└── Otherwise: Remove features and continue

Step 2.5: Log & Checkpoint
├── Track: n_features, train/val/test AUPRC per iteration
└── CHECKPOINT: Save iteration state
```

---

### Key Parameters (Matching Original)

**XGBoost (Conservative)**:
```python
max_depth = 2
min_child_weight = 50
gamma = 2.0
subsample = 0.3
colsample_bytree = 0.3
colsample_bylevel = 0.5
reg_alpha = 5.0
reg_lambda = 50.0
learning_rate = 0.005
early_stopping_rounds = 150
```

**Removal Criteria**:
```python
ZERO_SHAP_THRESHOLD = 0.0005     # Raised from 0.0002 to continue winnowing past 65 features
NEG_BIAS_RATIO_THRESHOLD = 0.25  # Raised from 0.15
BOTTOM_PERCENTILE = 12           # Raised from 8
MAX_REMOVALS_EARLY = 10          # Cap for iterations 1-5 (was 30, caused crashes)
MAX_REMOVALS_LATE = 5            # Cap for iterations 6+ (finer control)
LATE_PHASE_ITERATION = 5         # Switch to finer control after this iteration
MIN_FEATURES_THRESHOLD = 25
```

**Clustering**:
```python
MIN_CLUSTERING_THRESHOLD = 0.60
MAX_CLUSTERING_THRESHOLD = 0.85
TARGET_CLUSTER_RANGE = (40, 70)
```

---

### Expected Outcome

```
Starting:     ~170 features
After Phase 1: ~100-130 features (40-70 clusters)
After Phase 2: 25-50 features (iterates until 2+ criteria not met)

Each iteration removes 10-30 features that meet 2+ of:
  - Near-zero SHAP (< 0.0002)
  - Negative-biased (ratio < 0.15)
  - Bottom 8%

Stops when remaining features are all "valuable" (don't meet criteria)
or reaches MIN_FEATURES_THRESHOLD (25).
```

The actual stopping point depends on the data - we let the validation gate decide.

---

## Circular Reasoning Exclusions (Book 4)

CEA (Carcinoembryonic Antigen), CA 19-9, and FOBT/FIT were removed from the entire Book 4 labs pipeline. These features create circular reasoning in a model designed for early CRC identification:

- **CEA / CA 19-9**: Tumor markers ordered almost exclusively when a clinician already suspects malignancy. Including them means the model detects the doctor's suspicion, not independent signal.
- **FOBT/FIT**: CRC screening tests. A positive result *is* the detection mechanism, not a predictor of future disease.

All other lab features (CBC, metabolic panel, liver enzymes, iron studies, etc.) are routine tests ordered for many clinical reasons and remain appropriate. CA125 was preserved (ovarian cancer marker, not CRC-specific).

See `docs/book4_cea_fobt_removal_guide.md` for the cell-by-cell change log.

---

## Training Scripts (Standalone Python)

Three standalone Python scripts in `2nd_Dataset_Creation/` run outside Databricks notebooks:

| Script | Purpose | Model Name |
|--------|---------|------------|
| `featurization_train.py` | Production featurization pipeline with dynamic cohort window | N/A (produces feature table) |
| `train.py` | Conservative XGBoost with 40 features | `crc_risk_xgboost_40features` |
| `train_optuna.py` | Optuna hyperparameter tuning (50 trials) | `crc_risk_xgboost_40features_tuned` |

**Note:** Feature lists in all three scripts will need updating after Book 9 rerun produces a new final feature set (the CEA/FOBT removal may change which features survive selection).

`train.py` and `train_optuna.py` can run concurrently — they use different model names, run names, and temp file paths.

`featurization_train.py` uses a dynamic cohort window: end of last complete month (1-day data lag) → 12-month exclusion buffer → 24-month rolling observation window.

---

## Important Technical Context

- **Platform**: Databricks with PySpark
- **Data**: Can't query directly (siloed environment) - work from code
- **Table**: `{trgt_cat}.clncl_ds.herald_eda_train_final_cohort`
- **Class Imbalance**: 250:1 (0.41% positive rate)
- **Prediction Window**: 6 months
- **Cohort Period**: Jan 2023 - Sept 2024
- **Random Seed**: 217 (for all random operations)

### Catalog Pattern (`trgt_cat` vs `prod`)

**CRITICAL: `USE CATALOG` is ALWAYS `prod`.** Full source data (Clarity EHR tables) lives only in `prod`. The `dev` and `test` catalogs do NOT have complete data. Every notebook/script must set:

```python
trgt_cat = os.environ.get('trgt_cat', 'dev')
spark.sql('USE CATALOG prod')
```

- **Reading source data**: Unqualified references (e.g., `clarity_cur.PAT_ENC_ENH`) resolve against the current catalog, which is always `prod`.
- **Writing output tables**: Use `{trgt_cat}` (e.g., `{trgt_cat}.clncl_ds.fudgesicle_train`). This allows dev/prod separation for our own outputs.
- **Reading our own tables**: Use `{trgt_cat}` (e.g., `spark.table(f"{trgt_cat}.clncl_ds.fudgesicle_train")`).

**NEVER use `USE CATALOG {trgt_cat}`** -- this would read source data from dev/test which have incomplete data and will produce wrong results (e.g., 2,762 rows instead of 830K).

## User Preferences

- Linear, readable code (no nested functions for now)
- Discuss changes before implementing
- Keep all logic in single notebooks (no separate utility files)
- Add progress print statements for long-running operations
- Preserve all original functionality unless explicitly changing
- Add "What This Cell Does" / "Conclusion" markdown cells

## Key Technical Decisions

1. **Stratified Patient-Level Split**: 70/15/15 train/val/test split with multi-class stratification by cancer type (C18/C19/C20) - ensures balanced populations across splits
2. **3-Fold CV**: For computational efficiency with ~171 features
3. **Linear Code Style**: No nested functions - keep readable for debugging
4. **Documentation**: "What This Cell Does" + "Conclusion" markdown cells
5. **Dynamic Clustering Threshold**: Silhouette-based instead of fixed 0.7
6. **SHAP Weighting**: 2:1 for positive cases (model handles imbalance via scale_pos_weight)
7. **End Goal**: predict_proba -> isotonic calibration -> 0-100 risk score
8. **Resumable Pipeline**: Granular checkpoints allow stopping/resuming at any step

---

## Statistical Methods Rationale (SOP II.3.b Deviation)

The SOP specifies ChiSquared (for categorical features) and ANOVA (for numerical features) as statistical methods for feature selection. This pipeline uses **Risk Ratios** and **Mutual Information** instead. This is a deliberate methodological choice, not an oversight.

**Why p-value-based tests are inappropriate at this scale:** With N=831,000 observations, traditional hypothesis tests (ChiSquared, ANOVA) become meaningless. At this sample size, even trivially small effects achieve statistical significance (p < 0.001). A feature with Risk Ratio of 1.01—clinically irrelevant—would pass a ChiSquared test with flying colors simply because the massive sample size crushes sampling noise. The p-value answers "is this association unlikely due to chance?" but at N=831K, the answer is always "yes" for any non-zero effect, regardless of whether that effect matters clinically.

**Why Risk Ratios and Mutual Information are superior for this use case:** Risk Ratios directly quantify *effect size*—a 3.6× risk elevation for rapid weight loss is clinically interpretable and actionable, while "p < 0.001" conveys no magnitude. Mutual Information captures the *information content* of features, including non-linear relationships that correlation-based tests miss. Both metrics scale appropriately with clinical relevance rather than sample size. The SOP lists Mutual Information as an acceptable method (II.3.b.i), and Risk Ratios are the standard effect size measure in epidemiology for categorical outcomes.

**Bottom line:** ChiSquared and ANOVA would produce compliance checkboxes, not better feature selection. The current methodology identifies features that matter clinically, not just features that achieve arbitrary significance thresholds.

---

## Commands to Resume

```bash
# Navigate to project
cd /Users/michaeljoyce/Desktop/CLAUDE_CODE/CRC_MODELS

# Start Claude Code
claude
```

## Running the Pipeline

1. **Upload notebooks to Databricks** (Books 0-9 in order)
2. **Run Books 0-8** to create the wide feature table with SPLIT column
3. **Run Book 9** for feature selection:
   - Phase 1: Cluster-based reduction (~171 → ~70-80 features)
   - Phase 2: Iterative SHAP winnowing (~70-80 → final)
   - Checkpoints saved after each step (kill anytime, resume on re-run)
4. **Outputs** in `feature_selection_outputs/`:
   - `final_features.txt` - Feature list
   - `final_features.py` - Importable Python list
   - `final_model.pkl` - Trained model
   - `iteration_tracking.csv` - Metrics per iteration

---

## Reference: Pattern Applied to Books 1-8

```python
# 1. Add SPLIT to the cohort join
df_cohort = spark.sql("""
    SELECT PAT_ID, END_DTTM, FUTURE_CRC_EVENT, SPLIT
    FROM dev.clncl_ds.herald_eda_train_final_cohort
""")

# 2. Create training-only dataframe for feature selection
df_train = df_spark.filter(F.col("SPLIT") == "train")
df_train.cache()

# 3. Use df_train for all feature selection metrics
baseline_crc_rate = df_train.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]
# Risk ratios calculated on df_train
# MI scores calculated on df_train sample
```

## Reference: SGKF Implementation (Book 0)

```python
from sklearn.model_selection import StratifiedGroupKFold

# Create multi-class stratification label: 0=negative, 1=C18, 2=C19, 3=C20
cancer_type_map = {'C18': 1, 'C19': 2, 'C20': 3}
patient_labels['strat_label'] = patient_labels.apply(
    lambda row: cancer_type_map.get(row['cancer_type'], 0) if row['is_positive'] == 1 else 0,
    axis=1
)

sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=217)
y = patient_labels['strat_label'].values  # Multi-class: 0=neg, 1=C18, 2=C19, 3=C20
groups = patient_labels['PAT_ID'].values
# Takes first fold split: ~67% train, ~33% val
train_idx, val_idx = next(sgkf.split(X_dummy, y, groups))
```

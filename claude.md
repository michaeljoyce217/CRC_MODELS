# MASTER PROMPT: CRC Prediction Feature Selection Methodology Improvement

## Project Overview

This project is improving the **feature selection methodology** for a **colorectal cancer (CRC) risk prediction model** with highly imbalanced data (250:1 negative:positive ratio). The model predicts CRC diagnosis within 6 months for unscreened patients.

**Current Status**: All notebooks complete. Ready to run pipeline in Databricks.

## Directory Structure

```
METHODOLOGY_IMBALANCED/
├── 2nd_Dataset_Creation/           # TRANSFORMED notebooks (working versions)
│   ├── V2_Book0_Cohort_Creation.py # Base cohort with SGKF splits (COMPLETE)
│   ├── V2_Book1_Vitals.py          # Train-only feature selection (COMPLETE)
│   ├── V2_Book2_ICD10.py           # Train-only feature selection (COMPLETE)
│   ├── V2_Book3_Social_Factors.py  # No changes needed (all features excluded)
│   ├── V2_Book4_Labs_Combined.py   # Train-only feature selection (COMPLETE)
│   ├── V2_Book5_1_Medications_Outpatient.py  # Train-only feature selection (COMPLETE)
│   ├── V2_Book5_2_Medications_Inpatient.py   # Train-only feature selection (COMPLETE)
│   ├── V2_Book6_Visit_History.py   # Train-only feature selection (COMPLETE)
│   ├── V2_Book7_Procedures.py      # Train-only feature selection (COMPLETE)
│   ├── V2_Book8_Compilation.py     # No changes needed (just joins tables)
│   └── V2_Book9_Feature_Selection.py  # Feature selection pipeline (COMPLETE)
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
| Book 4 (Labs) | COMPLETE | df_train for risk ratios & MI |
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

### PHASE 1: Cluster-Based Reduction (Preserve Signal Diversity)

**Goal**: Remove redundant features while PRESERVING CLUSTER DIVERSITY. Each cluster represents a different type of predictive signal - losing a cluster means losing that signal type entirely.

**Critical Principle**: Be CONSERVATIVE with cluster reduction. It's better to keep a few extra correlated features than to lose an entire signal type.

```
Step 1.1: Load Data
├── Load wide feature table from Book 8
├── Filter to SPLIT='train' only for correlation computation
└── Verify features loaded

Step 1.2: Compute Correlation Matrix
├── Spearman correlation (handles non-linear relationships)
├── Distance matrix: distance = 1 - |correlation|
├── Remove zero-variance features first (undefined correlation)
└── Compute on TRAINING DATA ONLY

Step 1.3: Dynamic Threshold Selection
├── Test thresholds: 0.50 to 0.90 in 0.05 increments
├── Select threshold via silhouette score optimization
└── CHECKPOINT: Save clusters

Step 1.4: Train Baseline Model
├── XGBoost with scale_pos_weight for imbalance
├── Early stopping on validation set
└── CHECKPOINT: Save baseline model and metrics

Step 1.5: Compute SHAP with 2:1 Positive Weighting
├── TreeExplainer on baseline model
├── 2:1 weighting for positive cases
├── SHAP_Ratio = importance_pos / importance_neg
└── SHAP_Ratio > 1 means feature helps identify POSITIVES (what we want!)

Step 1.6: Select Cluster Representatives (ADAPTIVE - DROP AT MOST 1-2)
├── For each cluster:
│   ├── Size 1-2: Keep ALL features (too small to safely reduce)
│   ├── Size 3-4: Keep top (size - 1), drop at most 1
│   └── Size 5+: Keep top (size - 2), drop at most 2
├── Sort by SHAP_Ratio descending (prefer positive-predictive features)
└── This is CONSERVATIVE - preserves signal diversity

Step 1.7: Phase 1 Validation Gate
├── PASS if: val_auprc_drop < 10%
├── If FAIL: Keep more per cluster
└── CHECKPOINT: Save phase1_complete
```

**WHY ADAPTIVE CLUSTER SELECTION**: A cluster of 9 features represents 9 correlated signals. Keeping only 1-2 throws away 7 features that might have been useful representatives of sub-patterns. Dropping at most 1-2 per cluster preserves diversity while still removing obvious redundancy.

---

### PHASE 2: Surgical SHAP Winnowing (Preserve Cluster Diversity)

**Goal**: Remove ONLY features that contribute essentially NOTHING to positive class prediction, while preserving cluster diversity.

**Critical Principle for Rare Event Prediction (0.4% positive rate)**:
- The goal is identifying the rare POSITIVE cases, not just optimizing AUPRC
- Each cluster represents a different TYPE of predictive signal
- Losing a cluster entirely = losing that signal type = potentially missing positives
- Only remove features with truly near-zero contribution to positive identification

```
For each iteration:

Step 2.1: Train Model & Evaluate
├── XGBoost with scale_pos_weight
├── Track train, val, AND test AUPRC (test for post-hoc analysis only)
└── CHECKPOINT: Save model and metrics

Step 2.2: Compute SHAP with 2:1 Positive Weighting
├── SHAP_Combined = weighted importance
├── SHAP_Ratio = importance_pos / importance_neg
└── CHECKPOINT: Save SHAP values

Step 2.3: SURGICAL Removal (Respect Cluster Structure)
├── DYNAMIC THRESHOLD: 15th percentile of SHAP_Combined
│   (adapts to actual distribution, not arbitrary fixed value)
├── For each feature below threshold:
│   ├── Skip if clinical must-keep
│   ├── Skip if removing would leave cluster below MIN_PER_CLUSTER (1)
│   └── Only then add to removal candidates
├── This ensures:
│   ├── Every cluster keeps at least 1 representative
│   ├── Only features with near-zero signal are removed
│   └── Cluster diversity (signal types) is preserved
└── Cap at MAX_REMOVALS_PER_ITERATION (25)

Step 2.4: Simple Stop Conditions
├── Would go below MIN_FEATURES_THRESHOLD (30): STOP
├── No features meet removal criteria: STOP (all remaining have signal)
├── NOTE: Do NOT stop based on val AUPRC changes
│   └── Track metrics for post-hoc sweet spot analysis instead
└── Let it run to completion, then examine iteration_tracking.csv

Step 2.5: Log & Checkpoint
├── Track: n_features, train_auprc, val_auprc, test_auprc per iteration
├── After completion: plot n_features vs val/test AUPRC to find sweet spot
└── CHECKPOINT: Save iteration state
```

**WHY THIS APPROACH**:
1. Complex multi-criteria removal (2+ of 3 criteria, top 50% protected, etc.) was too strict - often removed ZERO features
2. Comparing to baseline was wrong - early iterations naturally have high train-val gap that should DECREASE as features are removed
3. The real goal is preserving signal diversity while removing true noise
4. Post-hoc analysis of the metrics curve is better than trying to automate the stopping point

---

### Metrics Tracking Strategy (Post-Hoc Analysis)

| Metric | Purpose | Usage |
|--------|---------|-------|
| **Train AUPRC** | Overfitting indicator | Track per iteration |
| **Val AUPRC** | Model quality | Track per iteration |
| **Test AUPRC** | Generalization | Track per iteration (for analysis only) |
| **Train-Val Gap** | Overfitting severity | Should DECREASE as features removed |

**Key Principle**: Track ALL metrics but DON'T use them for automated stopping. Run the full sweep to MIN_FEATURES_THRESHOLD, then examine `iteration_tracking.csv` to find the sweet spot where val and test AUPRC are maximized.

**Why not automated stopping?** Early iterations often have HIGH train-val gap (overfitting). The point of removing features is to REDUCE this gap. Stopping when gap is "too high" defeats the purpose. Let the process run and analyze the curve afterward.

---

### Checkpoint System (Resumable Iterations)

**Goal**: Stop anytime and resume without starting over.

```
checkpoints/
├── step1_2_correlation.pkl      # After correlation matrix computed
├── step1_3_clusters.pkl         # After clustering complete
├── step1_4_baseline_model.pkl   # After baseline model trained
├── step1_5_shap_phase1.pkl      # After Phase 1 SHAP computed
├── step1_7_phase1_complete.pkl  # After Phase 1 validation gate
├── step2_iter1_model.pkl        # After iteration 1 model
├── step2_iter1_shap.pkl         # After iteration 1 SHAP
├── step2_iter1_complete.pkl     # After iteration 1 complete
├── step2_iter2_model.pkl        # ...and so on
└── iteration_tracking.csv       # Running log (always updated)
```

**On notebook startup:**
1. Scan for existing checkpoints
2. Display: "Found checkpoint at Phase 2, Iteration 3. Resume? [Y/n]"
3. If resume: Load checkpoint and continue
4. If fresh: Clear checkpoints directory and start over

**You can kill the notebook anytime** - just re-run and it picks up from the last checkpoint.

---

### Key Improvements Over Original

| Aspect | Original | New |
|--------|----------|-----|
| **SHAP Weighting** | 2:1 | 2:1 (same; model handles imbalance via scale_pos_weight) |
| **Clustering Threshold** | Fixed 0.7 | Dynamic via silhouette score |
| **Cluster Representative Selection** | Keep 1 per cluster | ADAPTIVE: drop at most 1-2 per cluster (preserves diversity) |
| **Phase 2 Removal Logic** | Complex multi-criteria (2+ of 3) | SURGICAL: dynamic threshold + cluster protection |
| **SHAP Threshold** | Fixed 0.0002 | Dynamic 15th percentile (adapts to data) |
| **Stop Criterion** | Automated gates | Run full sweep, analyze post-hoc for sweet spot |
| **Cluster Protection** | Implicit | Explicit: MIN_PER_CLUSTER = 1 (never lose signal type) |
| **Resumability** | None (start over) | Granular checkpoints after each step |

### Critical Lessons Learned

1. **Cluster diversity = signal diversity**: Each cluster represents a different type of predictive signal. Losing a cluster entirely means losing that signal type. Be CONSERVATIVE with reduction.

2. **Rare event prediction requires surgical precision**: With 0.4% positive rate, every feature that helps identify positives matters. Don't blindly drop "bottom N" - only remove features with truly ZERO contribution.

3. **Dynamic thresholds > fixed values**: A fixed threshold like 0.002 is arbitrary. The 15th percentile adapts to actual SHAP distribution each iteration.

4. **Post-hoc analysis > automated stopping**: Complex stopping rules (gap thresholds, val drop thresholds) often trigger prematurely or at wrong times. Better to run the full sweep and examine the metrics curve to find the sweet spot.

5. **Protect cluster structure explicitly**: Every cluster should keep at least 1 representative. This is a hard constraint, not a soft preference.

---

### Expected Outcome

```
Starting:     171 features
After Phase 1: ~70-80 features (cluster representatives)
After Phase 2: ~40-60 features (validation-gated)

Iterations breakdown:
  Phase 2, Iter 1: 75 → 55 features (-20)
  Phase 2, Iter 2: 55 → 35 features (-20)
  Phase 2, Iter 3: STOP (val_auprc_drop > 5%)
  Final: Revert to 55 features
```

The actual stopping point depends on the data - we let the validation gate decide.

---

## Important Technical Context

- **Platform**: Databricks with PySpark
- **Data**: Can't query directly (siloed environment) - work from code
- **Table**: `{trgt_cat}.clncl_ds.herald_eda_train_final_cohort`
- **Class Imbalance**: 250:1 (0.41% positive rate)
- **Prediction Window**: 6 months
- **Cohort Period**: Jan 2023 - Sept 2024
- **Random Seed**: 217 (for all random operations)

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
cd /Users/michaeljoyce/Desktop/CLAUDE_CODE/METHODOLOGY_IMBALANCED

# Start Claude Code
claude

# Paste this prompt to resume:
"I'm continuing work on the CRC feature selection methodology improvement.
Please read MASTER_PROMPT.md for full context. All notebooks are complete.
V2_Book9_Feature_Selection.py implements the hybrid two-phase feature
selection pipeline with dynamic clustering and iterative SHAP winnowing."
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

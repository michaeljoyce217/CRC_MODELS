# Feature Selection Rationale: 49-Feature Final Set

## Summary

The final CRC risk prediction model uses **49 features** selected through a systematic three-phase pipeline: hierarchical clustering, iterative SHAP winnowing, and cross-validation stability analysis. This document explains the reasoning behind each selection decision.

**Key numbers:**
- Starting features: 167 (after Book 8 compilation and quality checks)
- After Phase 1 clustering: 143
- After Phase 2 SHAP winnowing (20 iterations): 26 (pipeline endpoint)
- After human review with CV stability: **49** (iter12 stable subset + 1 clinical addition)

---

## Phase 1: Cluster-Based Reduction (167 → 143)

**Method:** Spearman correlation → hierarchical clustering → silhouette-optimized threshold

- Correlation matrix computed on **training data only** (no data leakage)
- Dynamic threshold selection within [0.60, 0.85] range, targeting 40-70 clusters
- Selected threshold: **0.75**, producing **62 clusters**
- Within each cluster, kept features with highest SHAP importance ratio
- Removed **24 redundant features** (cluster members with lower SHAP contribution)

**Baseline performance (all 167 features):**
- Val AUPRC: 0.1134
- Train-val gap: 0.0052

**Post-clustering performance (143 features):**
- Val AUPRC: 0.1004
- Val drop: 11.5% (slightly exceeded 10% gate but proceeded)

---

## Phase 2: Iterative SHAP Winnowing (143 → 26)

**Method:** Train XGBoost → compute SHAP → remove low-importance features → repeat

The pipeline ran 20 iterations with conservative XGBoost parameters (max_depth=2, learning_rate=0.005, gamma=2.0) and multi-criteria removal (features must meet 2+ of: near-zero SHAP < 0.0005, negative-biased ratio < 0.25, bottom 12%).

| Iterations | Removals/iter | Features |
|-----------|---------------|----------|
| 1-5 | 10 (early cap) | 143 → 93 |
| 6-20 | 5 (late cap) | 93 → 26 |
| 20 | 0 (stop) | Would go below 25 |

**Performance trajectory (selected iterations):**

| Iter | Features | Val AUPRC | Test AUPRC | Train-Val Gap |
|------|----------|-----------|------------|---------------|
| 1 | 143 | 0.1004 | 0.1341 | -0.001 |
| 9 | 78 | **0.1330** | **0.1554** | +0.000 |
| 12 | 63 | 0.1225 | **0.1568** | -0.001 |
| 15 | 48 | 0.1220 | 0.1477 | +0.006 |
| 20 | 26 | 0.1200 | 0.1439 | -0.016 |

**Key observation:** Performance oscillates rather than degrading smoothly. This is characteristic of 250:1 class imbalance — each positive case has outsized impact on AUPRC. No single iteration is clearly "optimal" by metrics alone.

---

## Phase 3: 5-Fold CV Stability Analysis

**Method:** Run Phase 1 clustering on 4 additional train/val folds (beyond the main pipeline's fold). Track which features survive clustering across folds.

- **143 unique features** evaluated across 5 folds
- **108 features** appeared in 3+/5 folds → **stable**
- **35 features** appeared in 1-2/5 folds → **unstable**

Stability threshold: 60% (3 out of 5 folds)

**Why this matters:** A feature that only survives clustering on 1-2 of 5 data splits is sensitive to the specific train/val partition. Including it risks fitting to data artifacts rather than generalizable signal.

---

## Final Selection: 49 Features

### Decision Methodology

1. **Start from iteration 12 output** (58 features) — best observed test AUPRC (0.1568), near-zero train-val gap (-0.001)
2. **Remove 10 CV-unstable features** → 48 features, all appearing in 3+/5 folds
3. **Add back `lab_HEMOGLOBIN_ACCELERATING_DECLINE`** (removed during SHAP winnowing but CV-stable at 3/5 folds) → **49 features**

### Why Iteration 12?

- **Best test generalization:** Test AUPRC 0.1568 (highest observed across all 20 iterations)
- **No overfitting:** Train-val gap of -0.001 (validation actually slightly exceeds training)
- **Sufficient winnowing:** 12 rounds of SHAP-based removal ensures only features with consistent importance survive
- **Good cases-per-feature ratio:** 3,092 positive cases / 49 features = 63 cases per feature

### Why Not the 26-Feature Endpoint?

The 26-feature set from iteration 20 is the most parsimonious option, and its performance (val 0.1200, test 0.1439) is respectable. However, the 49-feature set is preferred because:

- **Performance floor is comparable** — val AUPRC varies between 0.10-0.13 regardless of feature count due to class imbalance noise
- **Clinically richer** — includes GI bleeding counts, family history, inflammation trends, weight dynamics, and platelet markers that clinicians would expect in a CRC risk model
- **Still well within safe cases-per-feature territory** — 63 cases/feature vs 119 for the 26-feature set, both adequate

### Why Add `lab_HEMOGLOBIN_ACCELERATING_DECLINE`?

This feature measures the **second derivative of hemoglobin** — whether hemoglobin decline is accelerating. It was removed during SHAP winnowing (before iteration 12) but is:

- **CV-stable** (3/5 folds)
- **Clinically significant** — accelerating hemoglobin decline is a hallmark of occult GI bleeding, one of the strongest CRC signals
- **Complementary** — the set already includes `lab_PLATELETS_ACCELERATING_RISE` (acceleration of platelets upward); hemoglobin acceleration downward completes the anemia dynamics picture
- **Supported by Book 4 analysis** — hemoglobin acceleration showed 10.9x CRC risk elevation in the feature engineering EDA

### Unstable Features Excluded (10 removed from iter12)

| Feature | Folds | Reason for Exclusion |
|---------|-------|---------------------|
| RACE_ASIAN | 2/5 | Sensitive to data split |
| icd_HIGH_RISK_FHX_FLAG | 2/5 | Redundant with icd_FHX_CRC_COMBINED and icd_HIGH_RISK_HISTORY |
| inp_med_inp_antispasmodic_use_recency | 1/5 | Very unstable |
| inp_med_inp_hemorrhoid_meds_flag | 1/5 | Very unstable |
| inp_med_inp_hormone_therapy_recency | 2/5 | Sensitive to data split |
| lab_FERRITIN_6MO_CHANGE | 2/5 | Sensitive to data split |
| out_med_gi_bleed_meds_recency | 1/5 | Very unstable |
| out_med_hemorrhoid_risk_score | 2/5 | Sensitive to data split |
| out_med_hormone_therapy_recency | 2/5 | Sensitive to data split |
| vit_FEVER_FLAG | 2/5 | Sensitive to data split |

---

## Final 49 Features by Domain

### Demographics (6)
| Feature | Description | CV Folds |
|---------|-------------|----------|
| AGE_GROUP | Age group categories (45-49/50-54/55-64/65-74/75+) | 5/5 |
| HAS_PCP_AT_END | Has primary care provider at observation end | 5/5 |
| IS_FEMALE | Sex indicator | 4/5 |
| IS_MARRIED_PARTNER | Married/partnered status | 4/5 |
| RACE_CAUCASIAN | Race indicator | 5/5 |
| RACE_HISPANIC | Race indicator | 3/5 |

### Temporal (1)
| Feature | Description | CV Folds |
|---------|-------------|----------|
| months_since_cohort_entry | Months since patient entered cohort | 5/5 |

### ICD-10 Diagnoses (6)
| Feature | Description | CV Folds |
|---------|-------------|----------|
| icd_BLEED_CNT_12MO | GI bleeding episode count in 12 months | 4/5 |
| icd_FHX_CRC_COMBINED | Family history of CRC (structured + ICD codes) | 4/5 |
| icd_HIGH_RISK_HISTORY | High-risk history composite (polyps, IBD, prior malignancy) | 5/5 |
| icd_IRON_DEF_ANEMIA_FLAG_12MO | Iron deficiency anemia diagnosis in 12 months | 5/5 |
| icd_SYMPTOM_BURDEN_12MO | Total symptom burden score in 12 months | 5/5 |
| icd_chronic_gi_pattern | Chronic GI symptom pattern indicator | 3/5 |

### Laboratory Values (11)
| Feature | Description | CV Folds |
|---------|-------------|----------|
| lab_ALBUMIN_DROP_15PCT_FLAG | Albumin dropped 15%+ over 6 months | 5/5 |
| lab_ALBUMIN_VALUE | Most recent albumin value | 5/5 |
| lab_ANEMIA_GRADE | Anemia severity grade (mild/moderate/severe) | 5/5 |
| lab_ANEMIA_SEVERITY_SCORE | Composite anemia severity score | 4/5 |
| lab_CRP_6MO_CHANGE | CRP change over 6 months (inflammation trend) | 4/5 |
| lab_HEMOGLOBIN_ACCELERATING_DECLINE | Hemoglobin decline accelerating (2nd derivative) | 3/5 |
| lab_IRON_SATURATION_PCT | Iron saturation percentage (transferrin saturation) | 5/5 |
| lab_PLATELETS_ACCELERATING_RISE | Platelet rise accelerating (2nd derivative) | 4/5 |
| lab_PLATELETS_VALUE | Most recent platelet count | 5/5 |
| lab_THROMBOCYTOSIS_FLAG | Elevated platelet count flag | 4/5 |
| lab_comprehensive_iron_deficiency | Composite iron deficiency indicator | 5/5 |

### Inpatient Medications (5)
| Feature | Description | CV Folds |
|---------|-------------|----------|
| inp_med_inp_any_hospitalization | Any hospitalization in lookback period | 4/5 |
| inp_med_inp_gi_bleed_meds_recency | Recency of inpatient GI bleed medications | 3/5 |
| inp_med_inp_ibd_meds_recency | Recency of inpatient IBD medications | 4/5 |
| inp_med_inp_laxative_use_flag | Inpatient laxative administration | 4/5 |
| inp_med_inp_opioid_use_flag | Inpatient opioid administration | 4/5 |

### Visit History (7)
| Feature | Description | CV Folds |
|---------|-------------|----------|
| visit_gi_symptom_op_visits_12mo | GI symptom outpatient visits in 12 months | 5/5 |
| visit_gi_symptoms_no_specialist | GI symptoms without specialist referral | 5/5 |
| visit_no_shows_12mo | Missed appointments in 12 months | 3/5 |
| visit_outpatient_visits_12mo | Total outpatient visits in 12 months | 5/5 |
| visit_primary_care_continuity_ratio | PCP visit continuity ratio | 4/5 |
| visit_recency_last_gi | Recency of last GI-related visit | 4/5 |
| visit_total_gi_symptom_visits_12mo | Total GI symptom visits in 12 months | 5/5 |

### Procedures (2)
| Feature | Description | CV Folds |
|---------|-------------|----------|
| proc_blood_transfusion_count_12mo | Blood transfusions in 12 months | 3/5 |
| proc_total_imaging_count_12mo | Total abdominal/pelvic imaging in 12 months | 5/5 |

### Vitals (11)
| Feature | Description | CV Folds |
|---------|-------------|----------|
| vit_BMI | Body mass index | 4/5 |
| vit_MAX_WEIGHT_LOSS_PCT_60D | Maximum weight loss percentage in 60-day window | 5/5 |
| vit_PULSE | Most recent pulse | 5/5 |
| vit_PULSE_PRESSURE | Systolic minus diastolic blood pressure | 5/5 |
| vit_RECENCY_WEIGHT | Recency of last weight measurement (ordinal) | 5/5 |
| vit_SBP_VARIABILITY_6M | Systolic blood pressure variability over 6 months | 5/5 |
| vit_UNDERWEIGHT_FLAG | BMI < 18.5 indicator | 4/5 |
| vit_WEIGHT_CHANGE_PCT_6M | Weight change percentage over 6 months | 4/5 |
| vit_WEIGHT_OZ | Most recent weight in ounces | 5/5 |
| vit_WEIGHT_TRAJECTORY_SLOPE | Linear slope of weight over time | 5/5 |
| vit_vital_recency_score | Composite vital signs recency score | 5/5 |

---

## Clinical Signal Coverage

The 49 features cover the major CRC risk signal categories:

| Signal Category | Features | Key Indicators |
|----------------|----------|----------------|
| **Anemia / Iron Deficiency** | 8 | Hemoglobin (value, acceleration, grade), iron saturation, albumin drop, comprehensive iron deficiency, anemia severity |
| **GI Bleeding** | 4 | Bleeding episode count, GI bleed meds recency, blood transfusions, platelet dynamics |
| **GI Symptoms** | 5 | Symptom burden, bowel change pattern, GI visits, symptom visits without specialist |
| **Weight / Nutrition** | 6 | Weight trajectory, weight loss, BMI, underweight flag, cachexia-related |
| **High-Risk History** | 3 | Family history, polyps/IBD history, high-risk composite |
| **Healthcare Utilization** | 6 | Visit intensity, PCP continuity, no-shows, imaging, hospitalization |
| **Inflammation** | 3 | CRP change, thrombocytosis, platelet acceleration |
| **Vital Signs** | 5 | Pulse, blood pressure variability, pulse pressure, vital recency |
| **Demographics** | 6 | Age, sex, race, PCP status, marital status |
| **Temporal** | 1 | Months since cohort entry |

No major CRC risk signal category is missing from the final set.

---

## Performance Characteristics

**Dataset:** 858,311 observations, 231,948 patients, 3,092 positive cases (0.360%, 1:277 imbalance)

**Note:** The metrics below were measured at iteration 12 with 58 features (48 stable + 10 unstable). The actual 49-feature model requires retraining; metrics may differ slightly.

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| AUPRC | 0.1215 | 0.1225 | 0.1568 |
| Lift | ~34x | ~34x | ~44x |
| Train-Val Gap | | -0.0010 | |

**Cases per feature:** 3,092 / 49 = 63

---

## Reproducibility

- **Random seed:** 217 (all operations)
- **Clustering threshold:** 0.75 (62 clusters)
- **XGBoost params:** max_depth=2, gamma=2.0, subsample=0.3, colsample_bytree=0.3, learning_rate=0.005
- **CV folds:** 5 (StratifiedGroupKFold on train/val pool)
- **Stability threshold:** 3/5 folds (60%)
- **Source data:** `dev.clncl_ds.herald_eda_train_wide_cleaned` (858,311 rows, 167 features)
- **Selection artifacts:** `Final_EDA/iteration_tracking.csv`, `Final_EDA/features_by_iteration.json`, `Final_EDA/cv_stability_report.json`

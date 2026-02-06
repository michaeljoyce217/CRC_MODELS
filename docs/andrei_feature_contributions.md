# Andrei's Feature Research Contributions

**Researcher:** Andrei (Lucem Health)
**Date of Research:** February 2026
**Reference File:** `Final_EDA/RUN_MODELS/CRC Cohort and Features.xlsx`

Andrei delivered an outstanding literature review, compiling 193 candidate features across demographics, vitals, labs, comorbidities, and medications for CRC risk prediction. His work gave us a strong evidence-based foundation to expand the feature set well beyond what we had originally engineered. Several of his suggestions — particularly the CBC differential ratios (NLR, PLR, SII) and the lifestyle risk factors (alcohol/tobacco ICD codes) — filled genuine gaps in our pipeline that we hadn't identified on our own.

This document tracks which of his suggestions were incorporated, which were already present in the pipeline (validating his recommendations), and which we set aside for now with reasoning.

### Model Availability

Features are available in two model variants:
- **Full Data Model** — All features included (Mercy-specific, uses all EHR data sources)
- **FHIR Portable Model** — Excludes features not available in standard US FHIR R4 implementations

| Feature Source | Full Data | FHIR Portable | Notes |
|---|---|---|---|
| Book 4 Labs (new) | All 20 + 6 ratios + 2 composites | All EXCEPT `lab_SYSTEMIC_INFLAMMATION_INDEX` | SII composite uses CRP/ESR (not in FHIR) |
| Book 2 ICD10 (new) | All 12 code groups + 2 composites | All 12 code groups + 2 composites | ICD codes are standard FHIR resources |
| Book 5.1 Meds (new) | H2 blockers (3 features) | None | All `med_*` features excluded from FHIR |
| Book 5.1 Meds (pre-existing) | Aspirin, Opioids, B12/Folate already present | None | Already in pipeline before Andrei's research |

---

## Section 1: Features Added

Andrei's research directly led to ~89 new engineered features across three Books. Several of his suggestions also confirmed that features we'd already built were well-chosen, which is valuable validation.

### Book 4 — Laboratory Features (20 new components + 6 derived ratios)

Andrei's lab recommendations were especially strong. His emphasis on the full CBC differential (enabling NLR, PLR, LMR, and SII calculations) and nutritional markers (B12, Folate, Vitamin D) addressed real blind spots in our pipeline. The BUN/Creatinine ratio for GI bleeding detection was a particularly sharp clinical insight.

**FHIR Portable availability:** All new lab features are FHIR-available EXCEPT `lab_SYSTEMIC_INFLAMMATION_INDEX` (depends on CRP and ESR, which are not universally available in FHIR).

| Feature | Andrei's LOINC/Reference | Engineered Columns | Clinical Rationale | Winnowed? |
|---|---|---|---|---|
| **RDW** (Red Cell Distribution Width) | 788-0, 30385-9 | `lab_RDW_VALUE`, `lab_RDW_ABNORMAL`, `lab_RDW_ELEVATED`, `lab_RDW_6MO_CHANGE`, `lab_RDW_6MO_PCT_CHANGE`, `lab_RDW_3MO_CHANGE`, `lab_RDW_12MO_CHANGE`, `lab_RDW_VELOCITY_PER_MONTH`, `lab_RDW_ACCELERATION`, `lab_RDW_ACCELERATING_RISE`, `lab_RDW_VOLATILITY`, `lab_RDW_TREND` | #1 feature in ColonFlag (Kinar 2016). Marker of red cell heterogeneity from chronic inflammation, iron deficiency, or occult bleeding. | TBD |
| **WBC** (White Blood Cell Count) | 6690-2, 26464-8 | `lab_WBC_VALUE`, `lab_WBC_ABNORMAL`, `lab_WBC_6MO_CHANGE`, `lab_LEUKOCYTOSIS_FLAG`, `lab_LEUKOPENIA_FLAG` | Basic CBC component; leukocytosis associated with systemic inflammation and malignancy. | TBD |
| **Neutrophils** (Absolute) | 751-8, 26499-4 | `lab_NEUTROPHILS_ABS_VALUE`, `lab_NEUTROPHILS_ABS_ABNORMAL` | Required for NLR calculation; neutrophilia linked to tumor microenvironment. | TBD |
| **Lymphocytes** (Absolute) | 731-0, 26474-7 | `lab_LYMPHOCYTES_ABS_VALUE`, `lab_LYMPHOCYTES_ABS_ABNORMAL` | Required for NLR/PLR/LMR; lymphopenia indicates immune suppression. | TBD |
| **Monocytes** (Absolute) | 742-7, 26484-6 | `lab_MONOCYTES_ABS_VALUE` | Required for LMR calculation; monocytes promote tumor angiogenesis. | TBD |
| **Eosinophils** (Absolute) | 711-2, 26449-9 | `lab_EOSINOPHILS_ABS_VALUE`, `lab_EOSINOPHILIA_FLAG` | Tissue eosinophilia associated with CRC prognosis. | TBD |
| **Basophils** (Absolute) | 704-7, 26444-0 | `lab_BASOPHILS_ABS_VALUE` | Completes CBC differential panel. | TBD |
| **Glucose** | 2345-7, 2339-0 | `lab_GLUCOSE_VALUE`, `lab_GLUCOSE_ABNORMAL`, `lab_HYPERGLYCEMIA_FLAG` | Metabolic dysfunction marker; diabetes is a CRC risk factor. | TBD |
| **BUN** (Blood Urea Nitrogen) | 3094-0, 6299-2 | `lab_BUN_VALUE`, `lab_BUN_ABNORMAL` | Required for BUN/Cr ratio (GI bleeding indicator). | TBD |
| **Creatinine** | 2160-0, 38483-4 | `lab_CREATININE_VALUE`, `lab_CREATININE_ABNORMAL`, `lab_CREATININE_6MO_CHANGE` | Kidney function baseline; required for eGFR. | TBD |
| **Calcium** | 17861-6, 49765-1 | `lab_CALCIUM_VALUE`, `lab_CALCIUM_ABNORMAL`, `lab_HYPERCALCEMIA_FLAG` | Hypercalcemia can indicate advanced malignancy (paraneoplastic). | TBD |
| **Sodium** | 2947-0, 2951-2 | `lab_SODIUM_VALUE`, `lab_SODIUM_ABNORMAL` | Electrolyte imbalance from GI losses. | TBD |
| **Potassium** | 2823-3, 6298-4 | `lab_POTASSIUM_VALUE`, `lab_POTASSIUM_ABNORMAL` | Electrolyte imbalance from chronic diarrhea. | TBD |
| **Chloride** | 2075-0, 2069-3 | `lab_CHLORIDE_VALUE` | Electrolyte panel completion. | TBD |
| **CO2/Bicarbonate** | 2028-9, 20565-8 | `lab_CO2_VALUE` | Metabolic panel completion; acid-base status. | TBD |
| **Total Cholesterol** | 2093-3, 50339-1 | `lab_TOTAL_CHOLESTEROL_VALUE`, `lab_TOTAL_CHOLESTEROL_ABNORMAL` | Completes lipid panel; low cholesterol associated with malignancy. | TBD |
| **TSH** | 3016-3, 3015-5 | `lab_TSH_VALUE`, `lab_TSH_ABNORMAL`, `lab_THYROID_DYSFUNCTION_FLAG` | Thyroid dysfunction affects weight, energy, GI motility. | TBD |
| **Vitamin D** (25-OH) | 1989-3 | `lab_VITAMIN_D_VALUE`, `lab_VITAMIN_D_ABNORMAL`, `lab_VITAMIN_D_DEFICIENCY_FLAG` | Vitamin D deficiency associated with increased CRC risk (meta-analyses). | TBD |
| **Vitamin B12** | 2132-9, 16695-9 | `lab_VITAMIN_B12_VALUE`, `lab_VITAMIN_B12_ABNORMAL`, `lab_B12_DEFICIENCY_FLAG` | B12 deficiency causes macrocytic anemia; nutritional status marker. | TBD |
| **Folate** | 2284-8, 2282-2 | `lab_FOLATE_VALUE`, `lab_FOLATE_ABNORMAL`, `lab_FOLATE_DEFICIENCY_FLAG` | Folate deficiency linked to DNA methylation and CRC risk. | TBD |

#### Derived Ratio Features (6)

| Feature | Formula | Engineered Columns | Clinical Rationale | Winnowed? |
|---|---|---|---|---|
| **NLR** (Neutrophil-to-Lymphocyte Ratio) | Neutrophils / Lymphocytes | `lab_NLR_VALUE`, `lab_NLR_ELEVATED`, `lab_NLR_6MO_CHANGE` | Premier systemic inflammation marker; elevated in CRC (meta-analysis: OR 1.8-2.5). | TBD |
| **PLR** (Platelet-to-Lymphocyte Ratio) | Platelets / Lymphocytes | `lab_PLR_VALUE`, `lab_PLR_ELEVATED` | Tumor-associated inflammation and thrombocytosis. | TBD |
| **LMR** (Lymphocyte-to-Monocyte Ratio) | Lymphocytes / Monocytes | `lab_LMR_VALUE` | Low LMR indicates immune suppression; prognostic in CRC. | TBD |
| **SII** (Systemic Immune-Inflammation Index) | Platelets x Neutrophils / Lymphocytes | `lab_SII_VALUE`, `lab_SII_ELEVATED` | Composite inflammation index; superior to NLR alone in some studies. | TBD |
| **BUN/Creatinine Ratio** | BUN / Creatinine | `lab_BUN_CR_RATIO_VALUE`, `lab_BUN_CR_RATIO_ELEVATED` | Classic GI bleeding indicator (ratio >20 suggests upper GI bleed). | TBD |
| **eGFR** | CKD-EPI (creatinine, age, sex) | `lab_EGFR_VALUE`, `lab_CKD_FLAG` | Kidney function staging; CKD is a comorbidity marker. | TBD |

#### Composite Features (2)

| Feature | Components | Engineered Column | Rationale | Winnowed? |
|---|---|---|---|---|
| **Nutritional Deficiency** | B12 <200, Folate <3, Vitamin D <20, or Iron deficiency | `lab_NUTRITIONAL_DEFICIENCY` | Combined nutritional depletion marker. | TBD |
| **Systemic Inflammation Index** | NLR >3, CRP >10, PLR >150, ESR >30 | `lab_SYSTEMIC_INFLAMMATION_INDEX` | Ordinal 0-4 composite of inflammation markers. | TBD |

---

### Book 2 — ICD-10 Comorbidity Features (12 new code groups)

Andrei's comorbidity recommendations rounded out our diagnostic feature set nicely. The alcohol and tobacco use disorder codes were a smart workaround for the social history data quality problems we'd identified in Book 3 — using clinically documented ICD codes avoids Epic's workflow bias entirely. His suggestion of hemorrhoids and IBS as differential-diagnosis features was also a good catch; these conditions generate symptoms that overlap with CRC and help the model distinguish true signal from noise.

**FHIR Portable availability:** All ICD-10 features are available in both models. ICD codes are standard FHIR Condition resources.

| Feature | ICD-10 Codes | Engineered Columns | Clinical Rationale | Winnowed? |
|---|---|---|---|---|
| **Alcohol Use Disorder** | F10.* | `icd_ALCOHOL_USE_DISORDER_FLAG`, `icd_ALCOHOL_USE_DISORDER_CNT` | Known CRC risk factor (IARC Group 1 carcinogen). ICD codes = clinically documented, not self-report. | TBD |
| **Tobacco Use Disorder** | F17.* | `icd_TOBACCO_USE_DISORDER_FLAG`, `icd_TOBACCO_USE_DISORDER_CNT` | Known CRC risk factor. ICD codes avoid Epic social history workflow bias. | TBD |
| **Hemorrhoids** | K64.* | `icd_HEMORRHOIDS_FLAG`, `icd_HEMORRHOIDS_CNT` | Anorectal pathology; confound for rectal bleeding; differential diagnosis. | TBD |
| **IBS** (Irritable Bowel Syndrome) | K58.* | `icd_IBS_FLAG`, `icd_IBS_CNT` | Important negative comparator for bowel symptoms; differential diagnosis. | TBD |
| **Liver Disease** | K70-K77 | `icd_LIVER_DISEASE_FLAG` | Comorbidity associated with metastatic potential and altered metabolism. | TBD |
| **COPD** | J44.* | `icd_COPD_FLAG` | Comorbidity; tobacco-associated; existing in Charlson but not as standalone feature. | TBD |
| **CKD** (Chronic Kidney Disease) | N18.* | `icd_CKD_FLAG` | Comorbidity; renal impairment affects drug metabolism and lab interpretation. | TBD |
| **Heart Failure** | I50.* | `icd_HEART_FAILURE_FLAG` | Comorbidity; cardiovascular disease burden. | TBD |
| **Celiac Disease** | K90.0 | `icd_CELIAC_DISEASE_FLAG` | Malabsorption syndrome; small bowel pathology. | TBD |
| **Malnutrition** | E44.*, E46 | `icd_MALNUTRITION_FLAG` | Nutritional depletion; weight loss differential. | TBD |
| **Nausea/Vomiting** | R11.* | `icd_NAUSEA_VOMITING_FLAG`, `icd_NAUSEA_VOMITING_CNT`, `icd_DAYS_SINCE_LAST_NAUSEA` | GI symptom not previously captured. | TBD |
| **Intestinal Obstruction** | K56.* | `icd_INTESTINAL_OBSTRUCTION_FLAG`, `icd_INTESTINAL_OBSTRUCTION_CNT` | Potential tumor obstruction presentation. | TBD |

---

### Book 5.1 — Medication Features (1 new category + 3 pre-existing)

Andrei correctly identified four medication categories relevant to CRC risk. Three of them — aspirin (in NSAID_ASA_USE), opioids (in CHRONIC_OPIOID_USE), and B12/folate supplements (in B12_FOLATE_USE) — were already in the pipeline, which is good independent validation of both his literature review and our original feature engineering. H2 blockers were a genuine addition.

**FHIR Portable availability:** None. All `med_*` features are excluded from the FHIR Portable model (outpatient medication data requires RxNorm-to-category mapping not standardized in FHIR).

| Feature | Drug Names | Engineered Columns | Status | Winnowed? |
|---|---|---|---|---|
| **H2-Blockers** | Famotidine, Ranitidine, Cimetidine, Nizatidine | `med_h2_blocker_use_flag`, `med_h2_blocker_use_days_since`, `med_h2_blocker_use_count_2yr` | **NEW** — Added via generic name matching | TBD |
| **Aspirin** | Aspirin, ASA | `med_nsaid_asa_use_flag/days_since/count_2yr` | **Pre-existing** — Already in NSAID_ASA_USE category | N/A |
| **Opioids** | Oxycodone, Hydrocodone, Morphine, Tramadol, Fentanyl | `med_opioid_use_flag/days_since/count_2yr` | **Pre-existing** — Already in CHRONIC_OPIOID_USE category | N/A |
| **B12/Folate** | Leucovorin, Methylfolate, Mecobalamin | `med_b12_or_folate_use_flag/days_since/count_2yr` | **Pre-existing** — Already in B12_FOLATE_USE category | N/A |

---

## Section 2: Features We Chose Not to Add (For Now)

Andrei's literature review was thorough — many of the features below are genuinely relevant to CRC biology but didn't fit our pipeline for practical reasons (data quality, population coverage, or redundancy with existing features). We've documented the reasoning so these decisions can be revisited as the model evolves.

### Urinalysis Panel (15 features)
**Andrei's reference:** LOINC 5778-6 (color), 5767-9 (clarity), 5810-7 (specific gravity), 5803-2 (pH), 2887-8 (protein), 5792-7 (glucose), 2514-8 (ketones), 5794-3 (blood/hemoglobin), 5770-3 (bilirubin), 19161-9 (urobilinogen), 5802-4 (nitrite), 5799-2 (leukocyte esterase), 798-9/799-7 (RBCs), 20408-1 (WBCs), 5769-5 (bacteria)

**Why not added now:** Urinalysis results are primarily qualitative (e.g., "trace", "1+", "2+") and use different extraction patterns from quantitative blood labs. Adding UA would require a separate extraction pipeline for qualitative result interpretation. The most CRC-relevant UA finding (hematuria) is already captured indirectly by ICD-10 codes (R31.*) in the diagnosis pipeline.

**Could revisit?** Yes. If urine blood alone proves valuable via ICD analysis, a targeted UA blood extraction could be worthwhile.

---

### Full Thyroid Panel (7 features beyond TSH)
**Andrei's reference:** Free T4 (LOINC 3024-7), Total T4 (3026-2), Free T3 (3051-0), Total T3 (3050-3), Reverse T3 (3052-8), TPO Antibody (8098-6), Thyroglobulin Antibody (11574-1)

**Why not added now:** TSH alone (which IS being added per Andrei's recommendation) is the standard screening test for thyroid dysfunction and captures >95% of the clinical information. The full panel is ordered only when TSH is abnormal, so including it would create highly missing data with strong selection bias.

**Could revisit?** Unlikely to improve on TSH alone for this use case.

---

### Reproductive Hormones (5 features)
**Andrei's reference:** Total Testosterone (LOINC 2986-8), Free Testosterone (2991-8), Estradiol, FSH, LH

**Why not added now:** These labs are ordered for specific clinical indications (infertility, menopause, hypogonadism), creating extreme selection bias with <5% population coverage. While estrogen exposure may have a protective CRC effect in women, this is better captured by age and sex (already in model) than by measured hormone levels.

**Could revisit?** Potentially relevant if gender-stratified CRC models are developed in the future.

---

### LDH Isoenzyme Fractions (5 features)
**Andrei's reference:** LDH1 (LOINC 2536-1), LDH2 (2539-5), LDH3 (2542-9), LDH4 (2545-2), LDH5 (2548-6)

**Why not added now:** LDH isoenzyme fractionation is a specialty test with extremely low population availability (<1%). Total LDH (already in our pipeline) captures the primary signal.

**Could revisit?** No practical value for population-level screening.

---

### C-Peptide and Insulin (2 features)
**Andrei's reference:** C-Peptide (LOINC 1986-9), Insulin (20448-7)

**Why not added now:** Specialty endocrinology tests with very low population availability. HbA1c (already in our pipeline) provides a better population-level measure of glucose control for CRC risk.

**Could revisit?** HbA1c covers this ground well.

---

### Specialty Lipids — ApoB, ApoA1, Lipoprotein(a) (3 features)
**Andrei's reference:** Apolipoprotein B (LOINC 1871-3), Apolipoprotein A1 (1869-7), Lipoprotein(a) (10835-7)

**Why not added now:** Very rarely ordered (<2% population availability). The standard lipid panel (LDL, HDL, Triglycerides, Total Cholesterol — all now in our pipeline) captures the clinically relevant information. These advanced markers are primarily for cardiovascular risk stratification.

**Could revisit?** No practical value for CRC screening.

---

### VLDL Cholesterol (1 feature)
**Andrei's reference:** LOINC 13458-5, 2091-7

**Why not added now:** VLDL is typically calculated as Total Cholesterol - HDL - LDL. Since all three components are now in the pipeline, VLDL provides no additional information to tree-based models. Can be trivially derived with one line of code if needed.

---

### Globulin and Albumin/Globulin Ratio (2 features)
**Andrei's reference:** Globulin (LOINC 2336-6), A/G Ratio (1759-0)

**Why not added now:** Both are linear transformations of Total Protein and Albumin, which are already in the pipeline. XGBoost can learn the relevant splits from the component features directly.

**Could revisit?** Can be added as derived features trivially if the winnowing results suggest it.

---

### MPV — Mean Platelet Volume (1 feature)
**Andrei's reference:** LOINC 32623-1, 28542-9

**Why not added now:** MPV is inconsistently reported across different hematology analyzers and institutions. The clinical signal is better captured by our extensive platelet count trends (VALUE, 6MO_CHANGE, VELOCITY, ACCELERATION, MAX_12MO, RISING_PATTERN, THROMBOCYTOSIS_FLAG).

**Could revisit?** Yes, if data quality analysis shows consistent MPV reporting at Mercy.

---

### Rare CBC Components — Nucleated RBCs, Immature Granulocytes (4 features)
**Andrei's reference:** Nucleated RBCs (LOINC 58413-6, 771-6), Immature Granulocytes (71695-1, 53115-2)

**Why not added now:** Not part of the standard automated CBC differential — these are reflex tests triggered by abnormal results, creating extreme selection bias. Population prevalence <1%.

**Could revisit?** Selection bias makes these unsuitable for screening models.

---

### Drug Use Disorders — F11-F19 (multiple features)
**Andrei's reference:** ICD-10 F11.* (opioid), F12.* (cannabis), F13.* (sedative), F14.* (cocaine), F15.* (stimulant), F16.* (hallucinogen), F18.* (inhalant), F19.* (other)

**Why not added now:** These are social/behavioral confounders rather than direct CRC risk factors. Including them could introduce socioeconomic bias. Alcohol (F10.*) and tobacco (F17.*) ARE being added because they have established direct biological CRC risk pathways (IARC carcinogens). That distinction matters.

**Could revisit?** Only in the context of health equity impact analysis.

---

### Aminosalicylates — A07EC (1 feature)
**Andrei's reference:** ATC A07EC (Mesalamine, Sulfasalazine, Balsalazide)

**Why not added now:** Very low population prevalence (IBD-specific medication). The clinical information is already captured by IBD diagnosis codes (K50.*, K51.*) in Book 2.

---

### Acetaminophen / Paracetamol — N02BE (1 feature)
**Andrei's reference:** ATC N02BE01

**Why not added now:** Acetaminophen is ubiquitous as an OTC medication and is inconsistently captured in EHR prescribing data. Unlike opioids (which require prescriptions and ARE being added), acetaminophen presence in the EHR reflects documentation patterns more than actual usage.

---

### H2-Receptor Antagonist Subtypes (already added as H2_BLOCKER)
**Andrei's reference:** ATC A02BA (Famotidine, Ranitidine, etc.)

**Status:** ADDED as `med_H2_BLOCKER_FLAG` — grouped all H2-blockers into a single category rather than tracking individual drugs.

---

### AFP (Alpha-Fetoprotein)
**Andrei's reference:** LOINC 1834-1, 83073-7

**Why not added now:** AFP is primarily a hepatocellular carcinoma and germ cell tumor marker, not a CRC marker. It's ordered for specific clinical suspicions, creating the same circularity problem we identified with CEA (ordered only when cancer is already suspected).

**Could revisit?** No. Tumor markers are excluded from screening models by design.

---

*This document will be updated with winnowing survival data after the pipeline is rerun on Databricks.*

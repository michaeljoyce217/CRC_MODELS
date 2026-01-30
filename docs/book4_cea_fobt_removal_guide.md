# Book 4 Change Guide: Removing CEA, CA 19-9, and FOBT/FIT

## Why These Features Are Excluded

Tumor markers (CEA, CA 19-9) and screening tests (FOBT/FIT) represent circular reasoning in a CRC prediction model designed for early identification:

- **CEA / CA 19-9**: Almost exclusively ordered when a clinician already suspects malignancy. Including them means the model detects the doctor's suspicion, not independent signal.
- **FOBT/FIT**: These are CRC screening tests. A positive result *is* the detection mechanism, not a predictor of future disease.

Including these features defeats the purpose of early intervention -- by the time CEA is ordered or FOBT is positive, the clinical process has already flagged the patient.

All other lab features (CBC, metabolic panel, liver enzymes, iron studies, etc.) are routine tests ordered for many clinical reasons and are appropriate for modeling.

---

## Cell-by-Cell Changes

### 1. Title/Intro Markdown (lines 1-65) -- MD

**Remove:**
- Line 16: Bullet `"**Tumor markers**: CEA elevation in 40-70% of established cancers"`
- Lines 25-26: `"**Missing CEA** suggests low cancer suspicion (not ordered without clinical concern)"`
- Line 47: FOBT reference in `"handling text parsing for FOBT results and managing selective ordering patterns for specialized tests"`

**Add:** A new section explaining why tumor markers and screening tests are deliberately excluded (circular reasoning -- see rationale above).

---

### 2. Cell 1B Markdown (around lines 450-525) -- MD

**Remove:**
- Line 453: `"3 years for tumor markers"` from lookback description
- Line 462: `"Selective ordering patterns: Tumor markers like CEA are rarely ordered without clinical suspicion"`
- Line 472: `"Tumor markers -> 'CEA', 'CA19_9', 'CA125'"`
- Line 473: `"Stool tests -> 'FOBT_FIT'"`
- Line 480: `"Tumor markers (CEA, CA19-9, CA125): 3 years"`
- Line 499: `"3-year lookback for tumor markers"`
- Line 510: `"FOBT text parsing"`
- Line 523: `"Tumor markers: Lower volumes"`

---

### 3. Cell 1B Code (around lines 528-670) -- CODE

**Remove:**
- Lines 585-587: CEA and CA19_9 WHEN clauses from component name mapping
- Lines 595-597: FOBT_FIT WHEN clause
- Lines 618-623: Remove `'CEA', 'CA19_9', 'CA125'` from the 3-year lookback list (keep FERRITIN, CRP, ESR, LDH if they use that window)
- Lines 645-646: CEA and CA19_9 outlier range lines
- Line 649: Remove `'FOBT_FIT'` from component list

---

### 4. Cell 1B Conclusion (around line 677) -- MD

- Line 677: Remove reference to tumor markers getting longer windows

---

### 5. Cell 2A Markdown (around line 700) -- MD

- Line 700: Remove `"FOBT/FIT screening predominantly occurs in outpatient settings"`
- Lines 737-738: Remove tumor marker and FOBT mentions

---

### 6. Cell 2B Markdown (around lines 830-900) -- MD

**Remove:**
- Line 834: FOBT text parsing reference
- Line 840: FOBT/FIT bullet
- Line 855: `"Tumor markers -> 'CEA', 'CA19_9', 'CA125'"`
- Lines 857-860: Entire FOBT/FIT Text Processing section
- Line 865: `"3 years for tumor markers"`
- Lines 875+: FOBT Processing Challenges section
- Lines 891, 898: FOBT success references

---

### 7. Cell 2B Code (around lines 905-1050) -- CODE

**Remove:**
- Lines 956-958: CEA and CA19_9 WHEN clauses
- Lines 967-972: Entire FOBT_FIT WHEN clause
- Lines 976-979: FOBT text result handling logic
- Lines 1008-1011: `'CEA', 'CA19_9', 'CA125'` from 3-year lookback list
- Lines 1037-1038: CEA and CA19_9 outlier range lines
- Line 1045: FOBT_FIT outlier range line

---

### 8. Cell 2B Conclusion (around line 1067) -- MD

- Remove FOBT text parsing mention

---

### 9. Cell 4B: CEA Trends (lines 1555-1705) -- DELETE ENTIRE CELL

Delete the markdown cell before it, the code cell (`herald_eda_train_labs_cea_trends`), and the conclusion markdown after it. This entire cell creates a CEA-specific trends table that is no longer needed.

---

### 10. Cell 5: Pivoted Lab Values (around lines 1870-1900) -- CODE

**Remove these lines from the SELECT:**
- Line 1880: `SUM(CASE WHEN COMPONENT_NAME = 'CEA' THEN recent_value END) AS CEA_VALUE,`
- Line 1881: `SUM(CASE WHEN COMPONENT_NAME = 'CA19_9' THEN recent_value END) AS CA19_9_VALUE,`
- Line 1891: `SUM(CASE WHEN COMPONENT_NAME = 'CEA' THEN recent_days END) AS CEA_DAYS,`
- Line 1896: `SUM(CASE WHEN COMPONENT_NAME = 'CEA' AND recent_abnormal = 'Y' THEN 1 ELSE 0 END) AS CEA_ABNORMAL,`

---

### 11. Cell 6: FOBT Features (lines 1994-2112) -- DELETE ENTIRE CELL

Delete the markdown cell before it, the code cell (`herald_eda_train_labs_fobt_features`), and the conclusion markdown after it. This entire cell creates FOBT-specific features that are no longer needed.

---

### 12. Cell 7: Acceleration Features (around line 2290) -- CODE

- Line 2290: Remove `'CEA'` from the component name list if present

---

### 13. Cell 8: Final Join (around lines 2594-2770) -- CODE

**Remove from SELECT:**
- Lines 2634-2635: `lp.CEA_VALUE,` and `lp.CA19_9_VALUE,`
- Lines 2663-2674: ALL CEA trend columns (`ct.CEA_3MO_CHANGE` through `ct.CEA_COUNT_12MO`)
- Lines 2676-2682: ALL FOBT columns (`ff.FOBT_POSITIVE_12MO` through `ff.DAYS_SINCE_FOBT`)
- Line 2702: `lt.CA19_9_6MO_CHANGE,`

**Remove from JOINs:**
- Line 2766: `LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_labs_cea_trends ct ...`
- Line 2768: `LEFT JOIN {trgt_cat}.clncl_ds.herald_eda_train_labs_fobt_features ff ...`

---

### 14. Cell 8 Conclusion / Validation (around lines 2785-2900) -- CODE + MD

- Line 2785: Remove `has_cea` coverage check
- Lines 2847-2848: Remove CEA and FOBT mentions in markdown
- Lines 2863-2865: Remove CEA/FOBT expected associations
- Lines 2891-2899: Remove CEA and FOBT risk ratio results
- Lines 2936-2960: Remove the `CEA_ELEVATED`, `CEA_RISING_PATTERN`, and `FOBT_POSITIVE` union blocks from the validation query

---

### 15. Validation/Analysis Cells (around lines 2997-3400) -- MD + CODE

Throughout these repeated validation cells:
- Remove all CEA and FOBT mentions from markdown conclusions
- Lines 3120-3123: Remove CEA coverage and FOBT coverage stats from code
- Lines 3146-3158: Remove entire "CEA PATTERN ANALYSIS" code block
- Lines 3167-3172: Remove CEA and FOBT from combination pattern analysis
- Lines 3189-3206: Remove CEA and FOBT from feature lists and coverage notes

---

### 16. Feature Reduction Step 1 (around line 3969) -- CODE

- Lines 3969-3974: Remove CEA coverage calculation and print statement

---

### 17. Feature Reduction Step 2 (risk ratios) -- MD + CODE

- Remove CEA and FOBT from risk ratio analysis markdown and any hardcoded references

---

### 18. Feature Reduction Step 5 (clinical filters, around line 4374) -- MD

- Lines 4380-4416: Remove all tumor marker preservation and FOBT retention language

---

### 19. Feature Reduction Step 6 (selection rules) -- CODE

**Remove from MUST_KEEP (line 4443):**
- `'lab_CEA_ELEVATED_FLAG'`
- `'lab_FOBT_POSITIVE_12MO'`

**Remove from redundant_pairs (line 4480):**
- `('lab_CA19_9_VALUE', 'lab_CA19_9_6MO_CHANGE')`

**Remove from selection logic (lines 4618-4624):**
- The `elif lab in ['CEA', 'CA19', 'CA125']:` branch
- The `elif lab == 'FOBT':` branch

**Remove from secondary must-keep lists (lines 4653-4654):**
- `'lab_CEA_ELEVATED_FLAG'`
- `'lab_FOBT_POSITIVE_12MO'`

---

### 20. Feature Reduction Step 7 (composites) -- CODE

**Remove:**
- Lines 4826-4831: Entire `lab_any_tumor_marker` composite creation
- Line 4839: `'lab_any_tumor_marker'` from composite_features list
- Lines 4858-4863: CEA/CA19/CA125 and FOBT descriptor blocks from print logic

---

### 21. Final Summary Markdown (around line 4950) -- MD

- Lines 5022-5023: Remove FOBT Text Processing Success section
- Lines 5058-5059: Remove CEA and FOBT from feature portfolio
- Line 5070: Remove FOBT text parsing mention

---

## Also Affected (Downstream)

After Book 4 is updated and rerun:

1. **Book 8** -- Will automatically pick up the reduced lab feature set (just joins tables)
2. **Book 9** -- Rerun feature selection without CEA/CA19-9/FOBT features
3. **featurization_train.py** -- Update feature list based on Book 9 output
4. **train.py** -- Update FEATURE_COLS
5. **train_optuna.py** -- Update FEATURE_COLS

The final feature count may change from 40 (likely 39, or a different 40th feature may survive that was previously removed).

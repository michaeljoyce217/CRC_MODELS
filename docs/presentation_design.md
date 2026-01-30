# From Correlation Structure to Clinical Signal

## Presentation Design Document

**Format:** Single self-contained HTML file with left sidebar navigation, scrollable narrative content, lightly interactive elements. No external dependencies.

**Audience:** 150-200 data scientists, ML engineers, data engineers, data architects, data analysts, and leaders. Technical but mixed depth. 10-minute live presentation with document distributed for independent reading.

**Core message:** Deliberate feature space expansion combined with mathematically rigorous selection finds predictive signal that intuition and conventional methods miss.

**Tone:** Professional, intellectually precise, warm (per Mercy ethos: compassionate, honest, hopeful). No victory laps. Let the evidence speak.

---

## Document Architecture

- **Single HTML file**, no external dependencies (CSS, JS, data all inline)
- **Fixed left sidebar** with 6 tab titles, always visible, active tab highlighted
- **Main content area:** centered column (~700px wide), scrollable narrative
- **Typography:** System fonts, ~18px body text, monospace for code snippets
- **Color palette:** Mercy brand colors
  - `#006598` (Mercy Blue) -- sidebar, headings, interactive highlights
  - `#D99871` (warm copper) -- callout box accents
  - Dark text on white content area
- **Interactive elements:** Correlation heatmap toggle, hoverable feature cards, Milky Way PCA visualization
- **Callout boxes:** In-line blue/copper bordered boxes for preemptive answers to skeptic questions

---

## Tab Structure

### Tab 1: "The Hypothesis"

**Opening:** Brief grounding in patient impact -- "Every year, patients in our system are diagnosed with colorectal cancer who could have been identified earlier. This project asks whether the data already contains the signal to find them."

**Narrative:**
- The conventional approach uses 20-30 clinically obvious features
- We deliberately expanded to 171 by engineering acceleration, variability, ratios, and composites across Books 1-7
- Not reckless expansion -- each Book applies an initial cull using risk ratios and mutual information, keeping only features with meaningful effect size
- The hypothesis: some of these engineered features would survive rigorous selection and outperform the obvious ones

**Visual:** Expansion graphic showing raw inputs -> engineered features -> 171 total

**Performance comparison:** Side-by-side metrics from `train_simple.py` (~30 baseline features) vs. the final 40-feature model

**Callout box (risk ratios + MI):** "Why risk ratios and mutual information for the initial cull? At N=831K, any non-zero effect achieves statistical significance. Risk ratios quantify effect size directly -- a 3.6x risk elevation is clinically interpretable. Mutual information captures non-linear relationships. Both scale with clinical relevance rather than sample size."

---

### Tab 2: "The Insight"

**Why fewer features matter:** "More features doesn't mean more signal. With 250:1 class imbalance, the model has very few positive cases to learn from. Every unnecessary feature is an opportunity to memorize noise that won't generalize. Reducing features isn't just about elegance -- it directly reduces overfitting risk. And a smaller feature set can be clinically validated, which matters when the output drives patient care."

**3D Milky Way PCA visualization (Three.js):**
- State 1: Star field disc (Milky Way) tilted at angle to X/Y/Z coordinate axes. "These axes don't describe the structure efficiently."
- State 2: Animate -- two new orthogonal vectors align with the disc plane. Third perpendicular (depth). "These two capture nearly all the structure. The third -- depth -- could be dropped with minimal loss. To find our solar system, you'd need the first two. The third adds only minimal directional information."
- State 3: The connection -- "PCA does this automatically. It finds axes that align with your data's actual structure. But those axes are uninterpretable linear combinations."

**Beat 3:** "Hierarchical clustering on the correlation matrix is motivated by the same intuition -- find and address the correlation structure in your data. But instead of projecting onto synthetic axes, it groups correlated features and lets you keep the most informative originals. The redundancy is reduced. The interpretability is untouched."

**Interactive element:** Correlation heatmap with two states:
- Raw: 171x171 matrix, chaotic color soup
- Clustered: Same data, reordered by cluster. Clean blocks along diagonal.
- Toggle button with animated transition
- Color scale: Mercy Blue -> white -> warm copper
- Cluster boundaries: thin Mercy Blue lines

**Callout box:** "Why not PCA directly? The end goal is a clinical risk score reviewed by physicians. Every feature needs a plain-English explanation for the care team."

---

### Tab 3: "The Method"

**Structure:** Pipeline diagram at top, then each stage explained below.

**Stage 1: Initial Cull (Books 1-7)**
- Brief -- already covered in Tab 1. Risk ratios and MI reduced raw features before the 171 entered the pipeline.

**Stage 2: Correlation Clustering**
- Spearman correlation -> distance matrix -> hierarchical clustering
- Smart threshold selection (silhouette-optimized within constrained range, not hardcoded)
- Assigns every feature to a cluster. Doesn't remove anything -- creates structure that informs Stage 3's guardrails.
- Illustrative dendrogram with ~15-20 features (readable labels, clear cut line)
- Full 171-feature dendrogram below (zoomable/scrollable) for reference

**Stage 3: Iterative SHAP Winnowing**
- Train conservative XGBoost -> compute SHAP -> identify weak features -> remove -> repeat
- Multi-criteria rule: feature must fail on 2 of 3 criteria (near-zero SHAP, negative-biased, bottom 8%) before removal. No single metric can kill a feature.
- Cluster structure from Stage 2 enforces removal caps -- can't strip a cluster bare
- Validation AUPRC tracked every iteration. Line chart: feature count dropping while performance holds stable.

**Callout box 1:** "Why not [standard method]? Most standard approaches either assume linear relationships, don't handle correlated feature groups, or test marginal relevance without accounting for redundancy. This methodology addresses all three."

**Callout box 2:** "Why 2-of-3 criteria? A single threshold is brittle. A feature might score low on one metric due to random variation. Requiring convergent evidence from multiple metrics prevents premature removal."

---

### Tab 4: "The Discoveries"

**Opening line:** "These features weren't hypothesized in advance. They survived because the data supported them."

**Feature cards (hoverable -- default shows name + description, hover reveals SHAP rank + clinical interpretation):**

**Theme 1: "Second derivatives"**
- Hemoglobin accelerating decline
- Platelet accelerating rise
- Concrete example: "A patient with hemoglobin of 11.5 looks different from a patient with hemoglobin of 11.5 that was 13.2 three months ago and 12.4 last month. Same snapshot, different trajectory. The model kept both."

**Theme 2: "Care gaps"**
- GI symptoms without specialist referral
- No-show count
- Acute care reliance ratio
- [Needs concrete example -- similar style to Theme 1]

**Theme 3: "Freshness matters"**
- Hemoglobin value AND acceleration both survived independently
- BP value AND variability both survived
- Vital recency score
- [Needs concrete example]

**Theme 4: "Social signal"**
- Marital/partner status survived 17 rounds against clinical features. It encodes something the labs and vitals don't capture.
- [Needs concrete example]

**Callout box (sparsity):** Near CEA discussion or sparse feature. "Some features are missing for 90%+ of patients. When they're present, it's because a clinician ordered a specific test. The model learns both signals -- the value when present, and the absence itself."

**Callout box (excluded features):** "Tumor markers like CEA and screening tests like FOBT were deliberately excluded. These tests are ordered when CRC is already suspected -- including them would mean the model detects the doctor's suspicion, not independent signal. That defeats the purpose of early identification."

---

### Tab 5: "The Result"

**Keep short.** The discoveries tab was the climax.

- Headline: 171 -> 40 features
- Side-by-side comparison: baseline (~30 features) vs. winnowed 40-feature model (AUPRC, AUROC on test set)
- Validation performance stayed stable through winnowing iterations
- One sentence on what's next: Optuna hyperparameter tuning, isotonic calibration, deployment as 0-100 risk score

**No callout boxes.** Let this tab breathe.

---

### Tab 6: "But What About..."

**Purpose:** Reference tab for post-presentation reading. Accordion-style expandable sections.

**"Why not [standard feature selection method]?"**
Most standard approaches either assume linear relationships, don't handle correlated feature groups, or test marginal relevance without accounting for redundancy. This methodology addresses all three.

**"Does this generalize beyond extreme imbalance?"**
The two-phase structure (decorrelation + iterative winnowing) works at any class ratio. The specific thresholds and conservative XGBoost parameters are tuned for 250:1, but the framework adapts.

**"Isn't 17 iterations of retraining expensive?"**
Each iteration trains one XGBoost model with early stopping. On this dataset, that's minutes per iteration. The total pipeline runs in under an hour -- well within a normal training workflow.

**"How do you know the removed features were actually useless?"**
Validation AUPRC tracked across every iteration. Performance held stable as features were removed. If a valuable feature had been dropped, the validation gate would have caught it.

**"What about features with 90%+ null values?"**
XGBoost handles missingness natively. A feature that's null 90% of the time but highly predictive when present can still survive selection. The model learns from both the value when present and the pattern of absence.

---

## Implementation Notes

### Dependencies to Build
- `train_simple.py` -- baseline model with ~30 conventional features (no engineered derivatives). Needed for the A/B comparison in Tabs 1 and 5.
- Correlation matrix data from Book 9 output (for interactive heatmap)
- Real 171-feature dendrogram from Book 9 output
- SHAP importance values from final model
- Iteration tracking CSV (feature count vs. AUPRC per iteration)

### Interactive Elements
1. **Milky Way PCA visualization** -- Three.js 3D star field with animated axis rotation (Tab 2)
2. **Correlation heatmap toggle** -- Raw vs. clustered with animation (Tab 2)
3. **Feature cards** -- Hoverable with SHAP rank reveal (Tab 4)
4. **Accordion FAQ** -- Expandable question sections (Tab 6)
5. **Dendrogram** -- Clean illustrative (~15 features) + full zoomable (171 features) (Tab 3)

### Technical Decisions
- Three.js via CDN link (keeps file small, requires internet)
- System fonts only (no external loads)
- All data embedded inline as JSON
- Single file, email/share friendly

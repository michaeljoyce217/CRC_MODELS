"""
train_optuna.py - Optuna Hyperparameter Tuning for CRC Risk Prediction (49 Features)

Runs 50 Optuna trials to find optimal XGBoost hyperparameters, then retrains
a final model with the best params, evaluates on train/val/test, computes SHAP
feature importances, and saves outputs to disk.

Search space is centered around moderately conservative defaults but allowed to
explore broadly. The 250:1 class imbalance means some regularization is always
needed, but the ultra-conservative Book 9 winnowing params are not required
now that the 49-feature set is locked in.

Model name: crc_risk_xgboost_49features_tuned

Usage (Databricks):
    Run as a Python script in Databricks. Requires:
    - herald_train_wide table (output of featurization_train.py)
    - Environment variable trgt_cat (defaults to 'dev')

Outputs (saved to same directory as this script):
    - crc_risk_xgboost_49features_tuned.pkl            : Best XGBoost model
    - crc_risk_xgboost_49features_tuned_metrics.json    : Performance metrics + metadata
    - crc_risk_xgboost_49features_tuned_shap.csv        : SHAP feature importances
    - crc_risk_xgboost_49features_tuned_trials.csv      : All Optuna trial results
"""

import os
import json
import time
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import shap
import optuna
from pyspark.sql import SparkSession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 217
MODEL_NAME = "crc_risk_xgboost_49features_tuned"
N_TRIALS = 50

# Catalog pattern: read source from prod, read/write our tables with trgt_cat
trgt_cat = os.environ.get("trgt_cat", "dev")

# Output directory = same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 49 Final Features (hardcoded from Final_EDA/feature_selection_rationale.md)
# ---------------------------------------------------------------------------

FINAL_FEATURES = [
    # Demographics (6)
    "AGE_GROUP",
    "HAS_PCP_AT_END",
    "IS_FEMALE",
    "IS_MARRIED_PARTNER",
    "RACE_CAUCASIAN",
    "RACE_HISPANIC",
    # Temporal (1)
    "months_since_cohort_entry",
    # ICD-10 Diagnoses (6)
    "icd_BLEED_CNT_12MO",
    "icd_FHX_CRC_COMBINED",
    "icd_HIGH_RISK_HISTORY",
    "icd_IRON_DEF_ANEMIA_FLAG_12MO",
    "icd_SYMPTOM_BURDEN_12MO",
    "icd_chronic_gi_pattern",
    # Laboratory Values (11)
    "lab_ALBUMIN_DROP_15PCT_FLAG",
    "lab_ALBUMIN_VALUE",
    "lab_ANEMIA_GRADE",
    "lab_ANEMIA_SEVERITY_SCORE",
    "lab_CRP_6MO_CHANGE",
    "lab_HEMOGLOBIN_ACCELERATING_DECLINE",
    "lab_IRON_SATURATION_PCT",
    "lab_PLATELETS_ACCELERATING_RISE",
    "lab_PLATELETS_VALUE",
    "lab_THROMBOCYTOSIS_FLAG",
    "lab_comprehensive_iron_deficiency",
    # Inpatient Medications (5)
    "inp_med_inp_any_hospitalization",
    "inp_med_inp_gi_bleed_meds_recency",
    "inp_med_inp_ibd_meds_recency",
    "inp_med_inp_laxative_use_flag",
    "inp_med_inp_opioid_use_flag",
    # Visit History (7)
    "visit_gi_symptom_op_visits_12mo",
    "visit_gi_symptoms_no_specialist",
    "visit_no_shows_12mo",
    "visit_outpatient_visits_12mo",
    "visit_primary_care_continuity_ratio",
    "visit_recency_last_gi",
    "visit_total_gi_symptom_visits_12mo",
    # Procedures (2)
    "proc_blood_transfusion_count_12mo",
    "proc_total_imaging_count_12mo",
    # Vitals (11)
    "vit_BMI",
    "vit_MAX_WEIGHT_LOSS_PCT_60D",
    "vit_PULSE",
    "vit_PULSE_PRESSURE",
    "vit_RECENCY_WEIGHT",
    "vit_SBP_VARIABILITY_6M",
    "vit_UNDERWEIGHT_FLAG",
    "vit_WEIGHT_CHANGE_PCT_6M",
    "vit_WEIGHT_OZ",
    "vit_WEIGHT_TRAJECTORY_SLOPE",
    "vit_vital_recency_score",
]

assert len(FINAL_FEATURES) == 49, f"Expected 49 features, got {len(FINAL_FEATURES)}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_stage(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*70}")
    print(f"[{timestamp}] {msg}")
    print(f"{'='*70}")


def print_progress(msg, indent=2):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{' '*indent}[{timestamp}] {msg}")


def evaluate_split(model, X, y, feature_cols, split_name):
    """Evaluate model on a single split. Returns metrics dict."""
    y_pred = model.predict_proba(X[feature_cols])[:, 1]

    auprc = average_precision_score(y, y_pred)
    auroc = roc_auc_score(y, y_pred)
    brier = brier_score_loss(y, y_pred)
    baseline_rate = y.mean()
    lift = auprc / baseline_rate if baseline_rate > 0 else 0

    print(f"  {split_name:<6}  AUPRC: {auprc:.4f}  AUROC: {auroc:.4f}  "
          f"Brier: {brier:.6f}  Lift: {lift:.1f}x  "
          f"(N={len(y):,}, events={int(y.sum()):,}, base={baseline_rate:.4%})")

    return {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "brier": float(brier),
        "baseline_rate": float(baseline_rate),
        "lift": float(lift),
        "n_samples": int(len(y)),
        "n_events": int(y.sum()),
    }


# ===========================================================================
# STEP 1: Load Data
# ===========================================================================

print_stage("STEP 1: LOAD DATA")

spark = SparkSession.builder.getOrCreate()
spark.sql("USE CATALOG prod")

table_name = f"{trgt_cat}.clncl_ds.herald_train_wide"
print_progress(f"Reading table: {table_name}")

start = time.time()
df_pandas = spark.table(table_name).toPandas()
elapsed = time.time() - start
print_progress(f"Loaded {len(df_pandas):,} rows in {elapsed:.1f}s")

# Validate that all 49 features are present
missing = [f for f in FINAL_FEATURES if f not in df_pandas.columns]
if missing:
    raise ValueError(f"Missing {len(missing)} features in table: {missing}")
print_progress(f"All 49 features present")

# Validate required columns
for col in ["PAT_ID", "END_DTTM", "FUTURE_CRC_EVENT", "SPLIT"]:
    if col not in df_pandas.columns:
        raise ValueError(f"Required column '{col}' not found in table")

# Split data
train_mask = df_pandas["SPLIT"] == "train"
val_mask = df_pandas["SPLIT"] == "val"
test_mask = df_pandas["SPLIT"] == "test"

X_train = df_pandas.loc[train_mask]
y_train = df_pandas.loc[train_mask, "FUTURE_CRC_EVENT"]
X_val = df_pandas.loc[val_mask]
y_val = df_pandas.loc[val_mask, "FUTURE_CRC_EVENT"]
X_test = df_pandas.loc[test_mask]
y_test = df_pandas.loc[test_mask, "FUTURE_CRC_EVENT"]

# Compute scale_pos_weight from training data
n_pos = int(y_train.sum())
n_neg = int((y_train == 0).sum())
if n_pos == 0:
    raise ValueError("No positive cases in training data")
scale_pos_weight = n_neg / n_pos

print(f"\n  Split summary:")
print(f"    Train: {len(y_train):>9,} obs, {int(y_train.sum()):>5,} events ({y_train.mean():.4%})")
print(f"    Val:   {len(y_val):>9,} obs, {int(y_val.sum()):>5,} events ({y_val.mean():.4%})")
print(f"    Test:  {len(y_test):>9,} obs, {int(y_test.sum()):>5,} events ({y_test.mean():.4%})")
print(f"    scale_pos_weight: {scale_pos_weight:.1f}")


# ===========================================================================
# STEP 2: Optuna Hyperparameter Search (50 Trials)
# ===========================================================================

print_stage(f"STEP 2: OPTUNA SEARCH ({N_TRIALS} TRIALS)")

# Pre-extract numpy arrays for speed during trial loop
X_train_np = X_train[FINAL_FEATURES].values
y_train_np = y_train.values
X_val_np = X_val[FINAL_FEATURES].values
y_val_np = y_val.values


def objective(trial):
    """Optuna objective: maximize validation AUPRC."""

    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.3, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 0.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "n_estimators": 3000,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_SEED,
        "early_stopping_rounds": 150,
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train_np, y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        verbose=False,
    )

    y_pred_val = model.predict_proba(X_val_np)[:, 1]
    val_auprc = average_precision_score(y_val_np, y_pred_val)

    # Also track train AUPRC for overfitting detection
    y_pred_train = model.predict_proba(X_train_np)[:, 1]
    train_auprc = average_precision_score(y_train_np, y_pred_train)

    trial.set_user_attr("train_auprc", train_auprc)
    trial.set_user_attr("train_val_gap", train_auprc - val_auprc)
    trial.set_user_attr("best_iteration", model.best_iteration)

    return val_auprc


# Suppress Optuna info logs (just show our progress updates)
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    study_name=MODEL_NAME,
)

print_progress(f"Starting {N_TRIALS} trials...")
print_progress(f"Objective: maximize validation AUPRC")
print()

search_start = time.time()

for i in range(N_TRIALS):
    study.optimize(objective, n_trials=1)
    trial = study.trials[-1]
    best_so_far = study.best_value

    print_progress(
        f"Trial {i+1:>3}/{N_TRIALS}  "
        f"val_auprc={trial.value:.4f}  "
        f"train_val_gap={trial.user_attrs['train_val_gap']:+.4f}  "
        f"best_iter={trial.user_attrs['best_iteration']:>4}  "
        f"best_so_far={best_so_far:.4f}"
    )

search_elapsed = time.time() - search_start
print(f"\n  Search complete in {search_elapsed:.1f}s")
print(f"  Best trial: #{study.best_trial.number + 1}")
print(f"  Best val AUPRC: {study.best_value:.4f}")

# Print best hyperparameters
best_params = study.best_trial.params
print(f"\n  Best hyperparameters:")
for k, v in sorted(best_params.items()):
    print(f"    {k}: {v}")


# ===========================================================================
# STEP 3: Retrain Best Model
# ===========================================================================

print_stage("STEP 3: RETRAIN WITH BEST PARAMS")

final_params = {
    **best_params,
    "n_estimators": 3000,
    "scale_pos_weight": scale_pos_weight,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": RANDOM_SEED,
    "early_stopping_rounds": 150,
}

print_progress(f"Training final model with best params...")
start = time.time()

model = XGBClassifier(**final_params)
model.fit(
    X_train[FINAL_FEATURES], y_train,
    eval_set=[(X_val[FINAL_FEATURES], y_val)],
    verbose=False,
)

elapsed = time.time() - start
print_progress(f"Training complete in {elapsed:.1f}s")
print_progress(f"Best iteration: {model.best_iteration} / {model.n_estimators}")
print_progress(f"Best validation AUCPR: {model.best_score:.4f}")


# ===========================================================================
# STEP 4: Evaluate on Train / Val / Test
# ===========================================================================

print_stage("STEP 4: EVALUATE MODEL")

metrics = {}
metrics["train"] = evaluate_split(model, X_train, y_train, FINAL_FEATURES, "Train")
metrics["val"] = evaluate_split(model, X_val, y_val, FINAL_FEATURES, "Val")
metrics["test"] = evaluate_split(model, X_test, y_test, FINAL_FEATURES, "Test")

train_val_gap = metrics["train"]["auprc"] - metrics["val"]["auprc"]
print(f"\n  Train-Val AUPRC gap: {train_val_gap:+.4f}")


# ===========================================================================
# STEP 5: Lift by Quarter (Test Set)
# ===========================================================================

print_stage("STEP 5: LIFT BY QUARTER (TEST SET)")

test_df = df_pandas.loc[test_mask].copy()
test_df["END_DTTM"] = pd.to_datetime(test_df["END_DTTM"])
test_df["quarter"] = test_df["END_DTTM"].dt.to_period("Q").astype(str)
test_df["y_pred"] = model.predict_proba(test_df[FINAL_FEATURES])[:, 1]

quarters = sorted(test_df["quarter"].unique())

print(f"  {'Quarter':<10} {'N':>8} {'Events':>8} {'Base Rate':>10} {'AUPRC':>8} {'Lift':>8}")
print(f"  {'-'*54}")

quarter_metrics = {}
for q in quarters:
    q_mask = test_df["quarter"] == q
    q_y = test_df.loc[q_mask, "FUTURE_CRC_EVENT"]
    q_pred = test_df.loc[q_mask, "y_pred"]
    q_base = q_y.mean()
    if q_y.sum() > 0:
        q_auprc = average_precision_score(q_y, q_pred)
        q_lift = q_auprc / q_base if q_base > 0 else 0
    else:
        q_auprc = float("nan")
        q_lift = float("nan")

    print(f"  {q:<10} {len(q_y):>8,} {int(q_y.sum()):>8} {q_base:>10.4%} {q_auprc:>8.4f} {q_lift:>8.1f}x")

    quarter_metrics[q] = {
        "n_samples": int(len(q_y)),
        "n_events": int(q_y.sum()),
        "base_rate": float(q_base),
        "auprc": float(q_auprc) if not np.isnan(q_auprc) else None,
        "lift": float(q_lift) if not np.isnan(q_lift) else None,
    }


# ===========================================================================
# STEP 6: SHAP Feature Importances
# ===========================================================================

print_stage("STEP 6: SHAP FEATURE IMPORTANCES")

print_progress("Computing SHAP values on validation set...")
start = time.time()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val[FINAL_FEATURES])

# Mean absolute SHAP per feature
shap_importance = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": FINAL_FEATURES,
    "mean_abs_shap": shap_importance,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_df["rank"] = range(1, len(shap_df) + 1)
shap_df["pct_of_total"] = shap_df["mean_abs_shap"] / shap_df["mean_abs_shap"].sum() * 100

elapsed = time.time() - start
print_progress(f"SHAP computation complete in {elapsed:.1f}s")

print(f"\n  Top 15 features by mean |SHAP|:")
print(f"  {'Rank':<6} {'Feature':<45} {'Mean |SHAP|':>12} {'% Total':>8}")
print(f"  {'-'*73}")
for _, row in shap_df.head(15).iterrows():
    print(f"  {int(row['rank']):<6} {row['feature']:<45} {row['mean_abs_shap']:>12.6f} {row['pct_of_total']:>7.1f}%")


# ===========================================================================
# STEP 7: Save Outputs
# ===========================================================================

print_stage("STEP 7: SAVE OUTPUTS")

# Save model
model_path = os.path.join(SCRIPT_DIR, f"{MODEL_NAME}.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
file_size_mb = os.path.getsize(model_path) / 1024 / 1024
print_progress(f"Model saved: {model_path} ({file_size_mb:.2f} MB)")

# Save SHAP importances
shap_path = os.path.join(SCRIPT_DIR, f"{MODEL_NAME}_shap.csv")
shap_df.to_csv(shap_path, index=False)
print_progress(f"SHAP saved: {shap_path}")

# Save trial results
trials_data = []
for trial in study.trials:
    row = {
        "trial": trial.number + 1,
        "val_auprc": trial.value,
        "train_auprc": trial.user_attrs.get("train_auprc"),
        "train_val_gap": trial.user_attrs.get("train_val_gap"),
        "best_iteration": trial.user_attrs.get("best_iteration"),
    }
    row.update(trial.params)
    trials_data.append(row)

trials_df = pd.DataFrame(trials_data).sort_values("val_auprc", ascending=False)
trials_path = os.path.join(SCRIPT_DIR, f"{MODEL_NAME}_trials.csv")
trials_df.to_csv(trials_path, index=False)
print_progress(f"Trials saved: {trials_path}")

# Save metrics JSON
metrics_output = {
    "model_name": MODEL_NAME,
    "timestamp": datetime.now().isoformat(),
    "random_seed": RANDOM_SEED,
    "n_features": len(FINAL_FEATURES),
    "feature_list": FINAL_FEATURES,
    "optuna": {
        "n_trials": N_TRIALS,
        "best_trial": study.best_trial.number + 1,
        "best_val_auprc": float(study.best_value),
        "search_space": {
            "max_depth": [2, 8],
            "min_child_weight": [10, 100],
            "gamma": [0.0, 5.0],
            "subsample": [0.3, 0.8],
            "colsample_bytree": [0.3, 0.8],
            "colsample_bylevel": [0.3, 0.8],
            "reg_alpha": [0.1, 10.0],
            "reg_lambda": [1.0, 100.0],
            "learning_rate": [0.001, 0.05],
        },
    },
    "best_hyperparameters": {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_params.items()},
    "scale_pos_weight": float(scale_pos_weight),
    "best_iteration": int(model.best_iteration),
    "best_score": float(model.best_score),
    "metrics": metrics,
    "train_val_gap": float(train_val_gap),
    "quarter_metrics": quarter_metrics,
    "shap_top_15": {row["feature"]: float(row["mean_abs_shap"]) for _, row in shap_df.head(15).iterrows()},
    "data": {
        "table": table_name,
        "total_rows": int(len(df_pandas)),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
    },
}

metrics_path = os.path.join(SCRIPT_DIR, f"{MODEL_NAME}_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_output, f, indent=2)
print_progress(f"Metrics saved: {metrics_path}")


# ===========================================================================
# Done
# ===========================================================================

print_stage("OPTUNA TRAINING COMPLETE")
print(f"""
  Model:      {MODEL_NAME}
  Features:   {len(FINAL_FEATURES)}
  Trials:     {N_TRIALS}
  Best trial: #{study.best_trial.number + 1}
  Best iter:  {model.best_iteration}

  Best hyperparameters:
    max_depth:        {best_params['max_depth']}
    min_child_weight: {best_params['min_child_weight']}
    gamma:            {best_params['gamma']:.3f}
    subsample:        {best_params['subsample']:.3f}
    colsample_bytree: {best_params['colsample_bytree']:.3f}
    colsample_bylevel:{best_params['colsample_bylevel']:.3f}
    reg_alpha:        {best_params['reg_alpha']:.4f}
    reg_lambda:       {best_params['reg_lambda']:.4f}
    learning_rate:    {best_params['learning_rate']:.5f}

  Performance:
    Train AUPRC: {metrics['train']['auprc']:.4f} ({metrics['train']['lift']:.1f}x lift)
    Val AUPRC:   {metrics['val']['auprc']:.4f} ({metrics['val']['lift']:.1f}x lift)
    Test AUPRC:  {metrics['test']['auprc']:.4f} ({metrics['test']['lift']:.1f}x lift)
    Train-Val gap: {train_val_gap:+.4f}

  Outputs:
    {model_path}
    {metrics_path}
    {shap_path}
    {trials_path}
""")

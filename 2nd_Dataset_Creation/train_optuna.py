# ===========================================================================
# train_optuna.py
#
# CRC Risk Prediction: Optuna Hyperparameter Tuning Pipeline
#
# Uses Optuna TPE sampler to search for optimal XGBoost hyperparameters,
# then trains a final model with the best params. Search space is bounded
# by domain knowledge (250:1 class imbalance, 40 features).
#
# Input:  {trgt_cat}.clncl_ds.fudgesicle_train
# Output: Tuned model registered in MLflow
#
# Pipeline stages:
#   1. Configuration & imports
#   2. Feature list (40 features from Book 9 SHAP winnowing)
#   3. Load data & prepare train/val/test splits
#   4. Optuna search space definition
#   5. Optuna objective function
#   6. Run Optuna study (50 trials)
#   7. Train final model with best params
#   8. Evaluate metrics (AUPRC, AUROC, overfitting check)
#   9. SHAP analysis (feature importance on test sample)
#  10. MLflow logging (params, metrics, model, plots, study summary)
#  11. Register model in MLflow Model Registry
#  12. Summary
#
# Requires: PySpark (Databricks), xgboost, shap, mlflow, scikit-learn,
#           numpy, pandas, matplotlib, optuna
# ===========================================================================

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt

# Suppress Optuna info logs (still shows warnings/errors)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

# Catalog / schema -- trgt_cat controls dev vs prod separation.
trgt_cat = os.environ.get('trgt_cat', 'dev')
spark.sql(f'USE CATALOG {trgt_cat}')
print(f"Catalog: {trgt_cat}")

# Input table (output of featurization_train.py) and MLflow settings
INPUT_TABLE = f"{trgt_cat}.clncl_ds.fudgesicle_train"
EXPERIMENT_NAME = f"/Shared/crc_risk_prediction_{trgt_cat}"
MODEL_NAME = "crc_risk_xgboost_40features_tuned"
RANDOM_SEED = 217
TARGET_COL = "FUTURE_CRC_EVENT"

# Optuna settings
N_OPTUNA_TRIALS = 50
N_ESTIMATORS = 5000
EARLY_STOPPING_ROUNDS = 150

print(f"Input table: {INPUT_TABLE}")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Model name: {MODEL_NAME}")
print(f"Optuna trials: {N_OPTUNA_TRIALS}")


# ===========================================================================
# 2. FEATURE LIST
#
# The 40 features selected by iterative SHAP winnowing in Book 9
# (iteration 16, test AUPRC=0.1146).
# ===========================================================================

FEATURE_COLS = [
    "lab_HEMOGLOBIN_ACCELERATING_DECLINE",
    "lab_PLATELETS_ACCELERATING_RISE",
    "vis_visit_recency_last_gi",
    "IS_FEMALE",
    "IS_MARRIED_PARTNER",
    "vit_BP_SYSTOLIC",
    "lab_ALT_AST_RATIO",
    "vit_WEIGHT_OZ",
    "lab_comprehensive_iron_deficiency",
    "lab_PLATELETS_VALUE",
    "lab_THROMBOCYTOSIS_FLAG",
    "vit_MAX_WEIGHT_LOSS_PCT_60D",
    "vit_WEIGHT_CHANGE_PCT_6M",
    "vit_WEIGHT_TRAJECTORY_SLOPE",
    "icd_MALIGNANCY_FLAG_EVER",
    "months_since_cohort_entry",
    "vis_visit_pcp_visits_12mo",
    "vis_visit_outpatient_visits_12mo",
    "vit_vital_recency_score",
    "icd_CHARLSON_SCORE_12MO",
    "vit_RECENCY_WEIGHT",
    "HAS_PCP_AT_END",
    "vis_visit_no_shows_12mo",
    "lab_AST_VALUE",
    "lab_CEA_VALUE",
    "lab_ALK_PHOS_VALUE",
    "vit_SBP_VARIABILITY_6M",
    "vis_visit_gi_symptom_op_visits_12mo",
    "icd_IRON_DEF_ANEMIA_FLAG_12MO",
    "vis_visit_total_gi_symptom_visits_12mo",
    "icd_ANEMIA_FLAG_12MO",
    "vis_visit_gi_symptoms_no_specialist",
    "icd_SYMPTOM_BURDEN_12MO",
    "icd_BLEED_CNT_12MO",
    "proc_total_imaging_count_12mo",
    "proc_ct_abd_pelvis_count_12mo",
    "icd_PAIN_FLAG_12MO",
    "lab_HEMOGLOBIN_VALUE",
    "lab_ALBUMIN_VALUE",
    "vis_visit_acute_care_reliance",
]

print(f"Feature count: {len(FEATURE_COLS)}")


# ===========================================================================
# 3. LOAD DATA & PREPARE SPLITS
#
# Load the featurized table, split into train/val/test pandas DataFrames,
# and compute scale_pos_weight for class imbalance handling.
# ===========================================================================

print("Loading data from Spark...")
df_all = spark.table(INPUT_TABLE).select(
    [TARGET_COL, "SPLIT"] + FEATURE_COLS
).toPandas()
print(f"Total rows loaded: {len(df_all):,}")

# Split into train / val / test
df_train = df_all[df_all["SPLIT"] == "train"].copy()
df_val = df_all[df_all["SPLIT"] == "val"].copy()
df_test = df_all[df_all["SPLIT"] == "test"].copy()

print(f"Train: {len(df_train):,} rows")
print(f"Val:   {len(df_val):,} rows")
print(f"Test:  {len(df_test):,} rows")

# Prepare X, y arrays
X_train = df_train[FEATURE_COLS].values
y_train = df_train[TARGET_COL].values
X_val = df_val[FEATURE_COLS].values
y_val = df_val[TARGET_COL].values
X_test = df_test[FEATURE_COLS].values
y_test = df_test[TARGET_COL].values

# Class imbalance ratio
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
scale_pos_weight = n_neg / n_pos

print(f"\nTrain positives: {n_pos:,} ({n_pos/len(y_train)*100:.3f}%)")
print(f"Train negatives: {n_neg:,}")
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

# Free the full dataframe
del df_all


# ===========================================================================
# 4. OPTUNA SEARCH SPACE DEFINITION
#
# Intelligent bounds based on domain knowledge:
# - 250:1 class imbalance requires conservative regularization
# - 40 features means moderate column sampling is fine
# - Shallow trees (max_depth 2-5) prevent memorizing rare positives
# - High min_child_weight prevents splits on tiny subgroups
# - Log-scale for learning rate and regularization strengths
#
# Fixed params (not tuned): objective, eval_metric, tree_method,
# scale_pos_weight, n_estimators (early stopping handles tree count).
# ===========================================================================

print("\n" + "=" * 60)
print("OPTUNA SEARCH SPACE")
print("=" * 60)
print(f"{'Parameter':<22} {'Range':<25} {'Scale':<8}")
print("-" * 60)
print(f"{'max_depth':<22} {'[2, 5]':<25} {'int':<8}")
print(f"{'min_child_weight':<22} {'[20, 100] step=10':<25} {'int':<8}")
print(f"{'gamma':<22} {'[1.0, 4.0]':<25} {'float':<8}")
print(f"{'subsample':<22} {'[0.2, 0.6]':<25} {'float':<8}")
print(f"{'colsample_bytree':<22} {'[0.2, 0.5]':<25} {'float':<8}")
print(f"{'colsample_bylevel':<22} {'[0.3, 0.7]':<25} {'float':<8}")
print(f"{'reg_alpha':<22} {'[1.0, 15.0]':<25} {'log':<8}")
print(f"{'reg_lambda':<22} {'[10.0, 100.0]':<25} {'log':<8}")
print(f"{'learning_rate':<22} {'[0.001, 0.02]':<25} {'log':<8}")
print("=" * 60)
print(f"\nFixed: scale_pos_weight={scale_pos_weight:.1f}, "
      f"n_estimators={N_ESTIMATORS}, early_stopping={EARLY_STOPPING_ROUNDS}")


# ===========================================================================
# 5. OPTUNA OBJECTIVE FUNCTION
#
# Each trial samples hyperparameters from the search space, trains an
# XGBoost model with early stopping on validation AUPRC, and returns
# the best validation AUPRC as the objective value (maximized).
# ===========================================================================

def objective(trial):
    """Optuna objective: train XGBoost with trial params, return val AUPRC."""

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "verbosity": 0,
        # Tuned hyperparameters
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 20, 100, step=10),
        "gamma": trial.suggest_float("gamma", 1.0, 4.0),
        "subsample": trial.suggest_float("subsample", 0.2, 0.6),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.5),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 0.7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 15.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 10.0, 100.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.02, log=True),
    }

    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **params,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Validation AUPRC
    y_val_pred = model.predict_proba(X_val)[:, 1]
    val_auprc = average_precision_score(y_val, y_val_pred)

    print(f"  Trial {trial.number:3d}: val_auprc={val_auprc:.6f}  "
          f"trees={model.best_iteration + 1}  "
          f"depth={params['max_depth']}  lr={params['learning_rate']:.4f}")

    return val_auprc


# ===========================================================================
# 6. RUN OPTUNA STUDY
#
# 50 trials with TPE sampler (Bayesian optimization). MedianPruner kills
# unpromising trials early. Seed=217 for reproducibility.
# ===========================================================================

print("\n" + "=" * 60)
print(f"STARTING OPTUNA STUDY ({N_OPTUNA_TRIALS} trials)")
print("=" * 60)

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_SEED),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    study_name="crc_xgboost_tuning",
)

study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

# Results
best_trial = study.best_trial

print("\n" + "=" * 60)
print("OPTUNA STUDY COMPLETE")
print("=" * 60)
print(f"Best trial:      {best_trial.number}")
print(f"Best val AUPRC:  {best_trial.value:.6f}")
print(f"\nBest hyperparameters:")
for k, v in best_trial.params.items():
    print(f"  {k}: {v}")


# ===========================================================================
# 7. TRAIN FINAL MODEL WITH BEST PARAMS
#
# Retrain XGBoost using the best trial's hyperparameters with verbose
# output. Evaluates on both train and val sets for monitoring.
# ===========================================================================

print("\n" + "=" * 60)
print("TRAINING FINAL MODEL WITH BEST PARAMS")
print("=" * 60)

best_params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "scale_pos_weight": scale_pos_weight,
    "random_state": RANDOM_SEED,
    "verbosity": 1,
}
best_params.update(best_trial.params)

print("Final model parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"  n_estimators: {N_ESTIMATORS}")
print(f"  early_stopping_rounds: {EARLY_STOPPING_ROUNDS}")
print()

model = xgb.XGBClassifier(
    n_estimators=N_ESTIMATORS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    **best_params,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100,
)

best_iteration = model.best_iteration
best_score = model.best_score
print(f"\nBest iteration: {best_iteration}")
print(f"Best validation AUPRC: {best_score:.6f}")


# ===========================================================================
# 8. EVALUATE METRICS
#
# Compute AUPRC and AUROC on train, validation, and test sets.
# Check the train-val gap for signs of overfitting (threshold: 0.05).
# Generate PR and ROC curves for the test set.
# ===========================================================================

# Predicted probabilities
y_train_pred = model.predict_proba(X_train)[:, 1]
y_val_pred = model.predict_proba(X_val)[:, 1]
y_test_pred = model.predict_proba(X_test)[:, 1]

# AUPRC
train_auprc = average_precision_score(y_train, y_train_pred)
val_auprc = average_precision_score(y_val, y_val_pred)
test_auprc = average_precision_score(y_test, y_test_pred)

# AUROC
train_auroc = roc_auc_score(y_train, y_train_pred)
val_auroc = roc_auc_score(y_val, y_val_pred)
test_auroc = roc_auc_score(y_test, y_test_pred)

# Overfitting check
train_val_gap_auprc = train_auprc - val_auprc
train_val_gap_auroc = train_auroc - val_auroc

print("=" * 60)
print("MODEL PERFORMANCE (TUNED)")
print("=" * 60)
print(f"{'Metric':<12} {'Train':>10} {'Val':>10} {'Test':>10} {'Gap(T-V)':>10}")
print("-" * 60)
print(f"{'AUPRC':<12} {train_auprc:>10.6f} {val_auprc:>10.6f} {test_auprc:>10.6f} {train_val_gap_auprc:>10.6f}")
print(f"{'AUROC':<12} {train_auroc:>10.6f} {val_auroc:>10.6f} {test_auroc:>10.6f} {train_val_gap_auroc:>10.6f}")
print("=" * 60)

if train_val_gap_auprc > 0.05:
    print("WARNING: Train-Val AUPRC gap > 0.05 - possible overfitting")
else:
    print("Train-Val gap is acceptable (< 0.05)")

# Precision-Recall and ROC curves for the test set
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_pred)
fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PR Curve
axes[0].plot(recall, precision, 'b-', linewidth=2)
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title(f'Precision-Recall Curve (Test AUPRC={test_auprc:.4f})')
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3)
baseline_rate = y_test.mean()
axes[0].axhline(y=baseline_rate, color='r', linestyle='--', label=f'Baseline ({baseline_rate:.4f})')
axes[0].legend()

# ROC Curve
axes[1].plot(fpr, tpr, 'b-', linewidth=2)
axes[1].plot([0, 1], [0, 1], 'r--', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title(f'ROC Curve (Test AUROC={test_auroc:.4f})')
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('/tmp/crc_tuned_model_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Curves saved to /tmp/crc_tuned_model_curves.png")


# ===========================================================================
# 9. SHAP ANALYSIS
#
# Compute SHAP values on a sample of test data (up to 5000 rows) using
# TreeExplainer. Generates a beeswarm summary plot and a bar importance
# plot showing all 40 features.
# ===========================================================================

print("Computing SHAP values...")

# Sample for SHAP (up to 5000 rows from test set)
shap_sample_size = min(5000, len(X_test))
np.random.seed(RANDOM_SEED)
shap_idx = np.random.choice(len(X_test), size=shap_sample_size, replace=False)
X_shap = X_test[shap_idx]

# TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

print(f"SHAP computed on {shap_sample_size:,} test samples")

# Mean absolute SHAP importance
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

print("\nFeature Importance (Mean |SHAP|):")
print("-" * 50)
for i, row in shap_importance.iterrows():
    bar = "#" * int(row['mean_abs_shap'] / shap_importance['mean_abs_shap'].max() * 30)
    print(f"  {i+1:2d}. {row['feature']:<45s} {row['mean_abs_shap']:.6f}  {bar}")

# SHAP beeswarm summary plot
fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=FEATURE_COLS,
    show=False,
    max_display=40,
)
plt.tight_layout()
plt.savefig('/tmp/crc_tuned_shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP summary saved to /tmp/crc_tuned_shap_summary.png")

# SHAP bar plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=FEATURE_COLS,
    plot_type="bar",
    show=False,
    max_display=40,
)
plt.tight_layout()
plt.savefig('/tmp/crc_tuned_shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP bar plot saved to /tmp/crc_tuned_shap_bar.png")


# ===========================================================================
# 10. MLFLOW LOGGING
#
# Log all parameters, metrics, the trained model, feature importance CSV,
# performance plots, and the Optuna study summary to the MLflow experiment.
# ===========================================================================

mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment: {EXPERIMENT_NAME}")

with mlflow.start_run(run_name=f"xgboost_40features_optuna_seed{RANDOM_SEED}") as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # Parameters -- best hyperparameters from Optuna
    mlflow.log_param("n_features", len(FEATURE_COLS))
    mlflow.log_param("random_seed", RANDOM_SEED)
    mlflow.log_param("input_table", INPUT_TABLE)
    mlflow.log_param("n_train_rows", len(X_train))
    mlflow.log_param("n_val_rows", len(X_val))
    mlflow.log_param("n_test_rows", len(X_test))
    mlflow.log_param("n_train_positives", int(n_pos))
    mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 1))
    mlflow.log_param("best_iteration", best_iteration)
    mlflow.log_param("tuning_method", "optuna_tpe")
    mlflow.log_param("n_optuna_trials", N_OPTUNA_TRIALS)
    mlflow.log_param("optuna_best_trial", best_trial.number)

    for k, v in best_params.items():
        mlflow.log_param(f"xgb_{k}", v)
    mlflow.log_param("xgb_n_estimators", N_ESTIMATORS)
    mlflow.log_param("xgb_early_stopping_rounds", EARLY_STOPPING_ROUNDS)

    # Metrics
    mlflow.log_metric("train_auprc", train_auprc)
    mlflow.log_metric("val_auprc", val_auprc)
    mlflow.log_metric("test_auprc", test_auprc)
    mlflow.log_metric("train_auroc", train_auroc)
    mlflow.log_metric("val_auroc", val_auroc)
    mlflow.log_metric("test_auroc", test_auroc)
    mlflow.log_metric("train_val_gap_auprc", train_val_gap_auprc)
    mlflow.log_metric("train_val_gap_auroc", train_val_gap_auroc)

    # Model artifact
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        input_example=pd.DataFrame(X_train[:5], columns=FEATURE_COLS),
    )
    print("Model logged to MLflow")

    # Feature list and importance CSV
    feature_list_str = "\n".join(FEATURE_COLS)
    mlflow.log_text(feature_list_str, "features.txt")

    shap_importance.to_csv("/tmp/crc_tuned_shap_importance.csv", index=False)
    mlflow.log_artifact("/tmp/crc_tuned_shap_importance.csv", "analysis")

    # Optuna study summary -- all trials as CSV
    trials_df = study.trials_dataframe()
    trials_df.to_csv("/tmp/crc_optuna_trials.csv", index=False)
    mlflow.log_artifact("/tmp/crc_optuna_trials.csv", "analysis")
    print(f"Optuna study summary logged ({len(trials_df)} trials)")

    # Plots
    mlflow.log_artifact("/tmp/crc_tuned_model_curves.png", "plots")
    mlflow.log_artifact("/tmp/crc_tuned_shap_summary.png", "plots")
    mlflow.log_artifact("/tmp/crc_tuned_shap_bar.png", "plots")

    print("All artifacts logged")
    print(f"Run URL: {run.info.artifact_uri}")


# ===========================================================================
# 11. REGISTER MODEL
#
# Register the tuned model in the MLflow Model Registry under a separate
# name from the conservative baseline model.
# ===========================================================================

model_uri = f"runs:/{run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

print(f"Model registered: {MODEL_NAME}")
print(f"Version: {registered_model.version}")
print(f"Source: {model_uri}")


# ===========================================================================
# 12. SUMMARY
# ===========================================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE (OPTUNA-TUNED)")
print("=" * 60)
print(f"Model:           {MODEL_NAME}")
print(f"Version:         {registered_model.version}")
print(f"Features:        {len(FEATURE_COLS)}")
print(f"Best iteration:  {best_iteration}")
print(f"Trees:           {best_iteration + 1}")
print(f"Optuna trials:   {N_OPTUNA_TRIALS}")
print(f"Best trial:      {best_trial.number}")
print()
print(f"{'Metric':<12} {'Train':>10} {'Val':>10} {'Test':>10}")
print("-" * 50)
print(f"{'AUPRC':<12} {train_auprc:>10.4f} {val_auprc:>10.4f} {test_auprc:>10.4f}")
print(f"{'AUROC':<12} {train_auroc:>10.4f} {val_auroc:>10.4f} {test_auroc:>10.4f}")
print()
print("Best hyperparameters:")
for k, v in best_trial.params.items():
    if isinstance(v, float):
        print(f"  {k:<22s} {v:.6f}")
    else:
        print(f"  {k:<22s} {v}")
print()
print(f"Note: Compare with conservative baseline model '{MODEL_NAME.replace('_tuned', '')}'")
print(f"      to assess whether tuning improved over hand-tuned params.")
print()
print(f"Input table:     {INPUT_TABLE}")
print(f"Experiment:      {EXPERIMENT_NAME}")
print(f"Run ID:          {run_id}")
print("=" * 60)

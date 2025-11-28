import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

# Import Boosting Libraries
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
TRAIN_PATH = "train_updated.csv"
TEST_PATH  = "test_updated.csv"
SAMPLE_PATH = "sample_submission_updated.csv"

TARGET = "RiskFlag"
ID_COL = "ProfileID"
N_FOLDS = 5
RANDOM_STATE = 42

# ------------------------------------------------------------------------------
# 2. Feature Engineering
# ------------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    epsilon = 1e-6

    # Financial Ratios
    df['LTI'] = df['RequestedSum'] / (df['AnnualEarnings'] + epsilon)
    df['EMI_Est'] = (df['RequestedSum'] * (1 + df['OfferRate']/100)) / (df['RepayPeriod'] + epsilon)
    df['EMI_to_Income'] = df['EMI_Est'] / ((df['AnnualEarnings']/12) + epsilon)

    # Interaction Features
    df['Trust_Age_Ratio'] = df['TrustMetric'] / (df['ApplicantYears'] + epsilon)
    df['Work_Stability'] = df['WorkDuration'] * df['ApplicantYears']

    return df

# ------------------------------------------------------------------------------
# 3. Load & Process Data
# ------------------------------------------------------------------------------
print("Loading Data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

# Apply Feature Engineering
train = engineer_features(train)
test = engineer_features(test)

# Separate Target and ID
y = train[TARGET]
X = train.drop(columns=[TARGET, ID_COL])
X_test = test.drop(columns=[ID_COL])

# Identify Categorical Columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical Columns: {cat_cols}")

# ------------------------------------------------------------------------------
# 4. Preparation for XGBoost (Requires Ordinal Encoding)
# ------------------------------------------------------------------------------
# CatBoost handles strings natively, but XGBoost prefers numbers.
# We create a copy for XGBoost.
X_xgb = X.copy()
X_test_xgb = X_test.copy()

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_xgb[cat_cols] = oe.fit_transform(X[cat_cols])
X_test_xgb[cat_cols] = oe.transform(X_test[cat_cols])

# ------------------------------------------------------------------------------
# 5. Cross-Validation Loop
# ------------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_preds_cat = np.zeros(len(X))
test_preds_cat = np.zeros(len(X_test))

oof_preds_xgb = np.zeros(len(X))
test_preds_xgb = np.zeros(len(X_test))

print(f"\nStarting {N_FOLDS}-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Split Data
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # --- Model 1: CatBoost ---
    # Native support for categorical features (pass names, not indices)
    cb_model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        eval_metric='AUC',
        random_seed=RANDOM_STATE,
        bagging_temperature=0.2,
        od_type='Iter',
        metric_period=100,
        od_wait=50,
        allow_writing_files=False,
        cat_features=cat_cols # Crucial for CatBoost
    )

    cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    # Predict
    oof_preds_cat[val_idx] = cb_model.predict_proba(X_val)[:, 1]
    test_preds_cat += cb_model.predict_proba(X_test)[:, 1] / N_FOLDS

    # --- Model 2: XGBoost ---
    # Uses Ordinal Encoded Data
    X_tr_xgb, X_val_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]

    xgb_model = XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        enable_categorical=False
    )

    xgb_model.fit(
        X_tr_xgb, y_tr,
        eval_set=[(X_val_xgb, y_val)],
        verbose=False
    )

    # Predict
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val_xgb)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test_xgb)[:, 1] / N_FOLDS

    print(f"Fold {fold+1} | CatBoost AUC: {roc_auc_score(y_val, oof_preds_cat[val_idx]):.4f} | XGBoost AUC: {roc_auc_score(y_val, oof_preds_xgb[val_idx]):.4f}")

# ------------------------------------------------------------------------------
# 6. Ensemble and Submission
# ------------------------------------------------------------------------------
print("\n--- Final Results ---")
print(f"Overall CatBoost OOF AUC: {roc_auc_score(y, oof_preds_cat):.5f}")
print(f"Overall XGBoost OOF AUC:  {roc_auc_score(y, oof_preds_xgb):.5f}")

# Weighted Ensemble (CatBoost handles categories better, so usually slightly higher weight)
final_oof = (0.6 * oof_preds_cat) + (0.4 * oof_preds_xgb)
print(f"Ensemble OOF AUC:         {roc_auc_score(y, final_oof):.5f}")

# Final Test Predictions
final_test_preds = (0.6 * test_preds_cat) + (0.4 * test_preds_xgb)

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: final_test_preds
})

# Reindex to match sample submission structure exactly
sample_ids = sample[ID_COL].values
submission = submission.set_index(ID_COL).reindex(sample_ids).reset_index()

submission.to_csv("submission_cat_xgb_ensemble.csv", index=False)
print("\nFile saved: submission_cat_xgb_ensemble.csv")

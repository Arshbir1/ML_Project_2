import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
TRAIN_PATH = "train_updated.csv"
TEST_PATH  = "test_updated.csv"
SAMPLE_PATH = "sample_submission_updated.csv"
TARGET = "RiskFlag"
ID_COL = "ProfileID"
RANDOM_STATE = 42

# ------------------------------------------------------------------------------
# 2. Feature Engineering Function
# ------------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Avoid division by zero
    epsilon = 1e-6

    # --- Ratio Features (Critical for Credit Risk) ---

    # Loan to Income Ratio (LTI): High LTI -> Higher Risk
    df['LTI'] = df['RequestedSum'] / (df['AnnualEarnings'] + epsilon)

    # Monthly Income approximation
    df['MonthlyIncome'] = df['AnnualEarnings'] / 12.0

    # Estimated Monthly EMI (Simplified assumption)
    # Total Repayment = Principal + (Principal * Rate * Years / 100)
    loan_years = df['RepayPeriod'] / 12.0
    total_interest = df['RequestedSum'] * (df['OfferRate'] / 100.0) * loan_years
    total_amount = df['RequestedSum'] + total_interest
    df['EstimatedEMI'] = total_amount / (df['RepayPeriod'] + epsilon)

    # Debt Service Coverage Ratio proxy
    df['EMI_to_Income'] = df['EstimatedEMI'] / (df['MonthlyIncome'] + epsilon)

    # Trust per Year of Age (Older people with low trust might be riskier)
    df['Trust_Per_Year'] = df['TrustMetric'] / (df['ApplicantYears'] + epsilon)

    # Disposable Income Proxy
    df['DisposableIncome'] = df['MonthlyIncome'] - df['EstimatedEMI']

    return df

# ------------------------------------------------------------------------------
# 3. Load and Prepare Data
# ------------------------------------------------------------------------------
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

# Apply Feature Engineering
print("Engineering features...")
train_eng = engineer_features(train)
test_eng = engineer_features(test)

X = train_eng.drop(columns=[TARGET, ID_COL])
y = train_eng[TARGET]
X_test = test_eng.drop(columns=[ID_COL])

# Identify columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical Features: {len(num_cols)}")
print(f"Categorical Features: {len(cat_cols)}")

# ------------------------------------------------------------------------------
# 4. Preprocessing Pipeline
# ------------------------------------------------------------------------------
# RobustScaler handles outliers better than StandardScaler for income/loans
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# ------------------------------------------------------------------------------
# 5. Model Definitions
# ------------------------------------------------------------------------------

# --- Model A: Non-Linear SVM (Approximated) ---
# Standard SVC is O(N^3). For 200k rows, we use Nystroem + LinearSVC (O(N))
# to approximate the RBF kernel map efficiently.
svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('nystroem', Nystroem(kernel='rbf', gamma=0.1, n_components=400, random_state=RANDOM_STATE)),
    ('clf', CalibratedClassifierCV(
        estimator=LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE),
        method='isotonic',
        cv=3
    ))
])

# --- Model B: Deep Neural Network ---
# Deeper architecture (256->128->64) to capture complex patterns
nn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE
    ))
])

# ------------------------------------------------------------------------------
# 6. Training and Validation
# ------------------------------------------------------------------------------
# We use a holdout set for local validation to estimate Kaggle score
from sklearn.model_selection import train_test_split

X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("\nTraining SVM (Nystroem Approximation)...")
svm_pipeline.fit(X_train_sub, y_train_sub)
svm_preds_val = svm_pipeline.predict_proba(X_val_sub)[:, 1]
svm_auc = roc_auc_score(y_val_sub, svm_preds_val)
print(f"SVM Validation AUC: {svm_auc:.5f}")

print("\nTraining Neural Network...")
nn_pipeline.fit(X_train_sub, y_train_sub)
nn_preds_val = nn_pipeline.predict_proba(X_val_sub)[:, 1]
nn_auc = roc_auc_score(y_val_sub, nn_preds_val)
print(f"Neural Network Validation AUC: {nn_auc:.5f}")

# Ensemble (Average)
ensemble_preds_val = (svm_preds_val + nn_preds_val) / 2
ensemble_auc = roc_auc_score(y_val_sub, ensemble_preds_val)
print(f"Ensemble Validation AUC: {ensemble_auc:.5f}")

# ------------------------------------------------------------------------------
# 7. Final Training on Full Data & Submission
# ------------------------------------------------------------------------------
print("\nRetraining on FULL dataset for submission...")

# Train SVM on full data
svm_pipeline.fit(X, y)
test_probs_svm = svm_pipeline.predict_proba(X_test)[:, 1]

# Train NN on full data
nn_pipeline.fit(X, y)
test_probs_nn = nn_pipeline.predict_proba(X_test)[:, 1]

# Ensemble
test_probs_ens = (test_probs_svm * 0.5) + (test_probs_nn * 0.5)

# Create Submission DataFrame
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: test_probs_ens
})

# Ensure strict sorting alignment with sample submission if needed
# (Kaggle usually evaluates based on ID match, but sorting is safer)
sample_ids = sample[ID_COL].values
submission = submission.set_index(ID_COL).reindex(sample_ids).reset_index()

# Check distribution
print("\nPrediction Stats:")
print(submission[TARGET].describe())

# Save
submission.to_csv("submission_improved.csv", index=False)
print("\nSuccess! File saved as 'submission_improved.csv'")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ------------------------------------------------------------------------------
# 1. Config
# ------------------------------------------------------------------------------
TRAIN_PATH = "train_updated.csv"
TEST_PATH  = "test_updated.csv"
SAMPLE_PATH = "sample_submission_updated.csv"
TARGET = "RiskFlag"
ID_COL = "ProfileID"
RANDOM_STATE = 42

# ------------------------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------------------------
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

# ------------------------------------------------------------------------------
# 3. Advanced Feature Engineering
# ------------------------------------------------------------------------------
# Based on analysis, these cols have skewed distributions (power law).
# Log-transforming them helps the Neural Network significantly.
log_cols = ['AnnualEarnings', 'RequestedSum', 'WorkDuration', 'TrustMetric']

# These are the top predictors. We will generate interaction terms for them.
poly_cols = ['ApplicantYears', 'OfferRate', 'DebtFactor']

# Categorical columns for OneHot
cat_cols = ['QualificationLevel', 'WorkCategory', 'RelationshipStatus',
            'OwnsProperty', 'FamilyObligation', 'FundUseCase', 'JointApplicant']

# Define transformations
def log_transform(x):
    return np.log1p(np.maximum(x, 0))

# ------------------------------------------------------------------------------
# 4. Preprocessing Pipelines
# ------------------------------------------------------------------------------

# Pipeline A: Log Transform + Scaling (For skewed data)
log_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(log_transform)),
    ('scaler', StandardScaler())
])

# Pipeline B: Polynomial Features (For top predictors)
# Degree 2 generates interactions: Age^2, Age*Rate, Rate^2, etc.
poly_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

# Pipeline C: Categorical
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine them
preprocessor = ColumnTransformer(
    transformers=[
        ('log', log_transformer, log_cols),
        ('poly', poly_transformer, poly_cols),
        ('cat', cat_transformer, cat_cols)
    ],
    remainder='drop' # Drop other columns to reduce noise
)

# ------------------------------------------------------------------------------
# 5. Prepare Train/Validation Sets
# ------------------------------------------------------------------------------
X = train.drop(columns=[TARGET, ID_COL])
y = train[TARGET]
X_test = test.drop(columns=[ID_COL])

# Split for internal validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

print("Fitting preprocessor...")
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

print(f"Processed Feature Count: {X_train_proc.shape[1]}")

# ------------------------------------------------------------------------------
# 6. Model A: Polynomial Linear SVM
# ------------------------------------------------------------------------------
# Since we added Polynomial features in the pipeline, a LinearSVC here
# effectively acts like a Polynomial Kernel SVM but is much faster.
print("\nTraining Polynomial SVM...")

svm_base = LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=3000, dual=False)
svm_calibrated = CalibratedClassifierCV(estimator=svm_base, method='isotonic', cv=3)

svm_calibrated.fit(X_train_proc, y_train)

svm_probs_val = svm_calibrated.predict_proba(X_val_proc)[:, 1]
print(f"SVM Val AUC: {roc_auc_score(y_val, svm_probs_val):.4f}")

# ------------------------------------------------------------------------------
# 7. Model B: Bagged Neural Networks (Ensemble)
# ------------------------------------------------------------------------------
# Instead of 1 big NN, we train 5 smaller ones on subsets.
# This reduces variance and typically improves generalization.
print("\nTraining Bagged Neural Networks (5 estimators)...")

nn_base = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    random_state=RANDOM_STATE
)

# BaggingClassifier trains 'n_estimators' models on random subsets of data
bagged_nn = BaggingClassifier(
    estimator=nn_base,
    n_estimators=5,
    max_samples=0.7,  # Use 70% of data for each model
    max_features=1.0, # Use all features
    n_jobs=1,         # Set to -1 if running locally for parallelism
    random_state=RANDOM_STATE
)

bagged_nn.fit(X_train_proc, y_train)

nn_probs_val = bagged_nn.predict_proba(X_val_proc)[:, 1]
print(f"Bagged NN Val AUC: {roc_auc_score(y_val, nn_probs_val):.4f}")

# ------------------------------------------------------------------------------
# 8. Blending & Submission
# ------------------------------------------------------------------------------
# Weighted average: Give more weight to NN if it performs better
ensemble_probs_val = (0.4 * svm_probs_val) + (0.6 * nn_probs_val)
print(f"Ensemble Val AUC: {roc_auc_score(y_val, ensemble_probs_val):.4f}")

# Predict on Test Set
print("\nGenerating test predictions...")
svm_test_probs = svm_calibrated.predict_proba(X_test_proc)[:, 1]
nn_test_probs = bagged_nn.predict_proba(X_test_proc)[:, 1]

ensemble_test_probs = (0.4 * svm_test_probs) + (0.6 * nn_test_probs)

# Create Submission
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: ensemble_test_probs
})

# Reorder to match sample submission if needed
sample_ids = sample[ID_COL].values
submission = submission.set_index(ID_COL).reindex(sample_ids).reset_index()

submission.to_csv("submission_bagging_poly.csv", index=False)
print("Saved: submission_bagging_poly.csv")

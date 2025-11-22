# full_pipeline_kaggle_fixed_svd.py
# Robust pipeline (handles sklearn version differences and ensures SVD n_components <= n_features-1).
# Run in Colab / Kaggle / Jupyter.

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from scipy import sparse
import inspect

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------------------
# Config
# -------------------------
TRAIN_PATH = "train_updated.csv"
TEST_PATH  = "test_updated.csv"
SAMPLE_PATH = "sample_submission_updated.csv"

TARGET = "RiskFlag"
IDCOL  = "ProfileID"

# desired SVD components (will be clamped to valid range automatically)
DESIRED_SVD_COMPONENTS = 200
RANDOM_STATE = 42

# -------------------------
# Load
# -------------------------
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

print("Train shape:", train.shape, "Test shape:", test.shape, "Sample shape:", sample.shape)

# -------------------------
# Split
# -------------------------
X = train.drop(columns=[TARGET])
y = train[TARGET].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

all_features = [c for c in X_train.columns if c != IDCOL]
numeric_cols = X_train[all_features].select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [c for c in all_features if c not in numeric_cols]

print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# -------------------------
# Preprocess
# -------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))   # keep sparse
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ],
    remainder="drop"
)

print("Fitting preprocessor...")
preprocessor.fit(X_train[all_features])

print("Transforming datasets...")
X_train_trans = preprocessor.transform(X_train[all_features])
X_val_trans   = preprocessor.transform(X_val[all_features])
X_test_trans  = preprocessor.transform(test[all_features])

# Ensure sparse CSR
if not sparse.issparse(X_train_trans):
    X_train_s = sparse.csr_matrix(X_train_trans)
    X_val_s   = sparse.csr_matrix(X_val_trans)
    X_test_s  = sparse.csr_matrix(X_test_trans)
else:
    X_train_s = sparse.csr_matrix(X_train_trans)
    X_val_s   = sparse.csr_matrix(X_val_trans)
    X_test_s  = sparse.csr_matrix(X_test_trans)

print("Transformed shapes (sparse):", X_train_s.shape, X_val_s.shape, X_test_s.shape)

# -------------------------
# LinearSVC + CalibratedClassifierCV (compatibility handling)
# -------------------------
print("\nTraining LinearSVC + CalibratedClassifierCV...")

base_clf = LinearSVC(max_iter=4000, random_state=RANDOM_STATE)

# instantiate CalibratedClassifierCV using estimator= or base_estimator= depending on sklearn
calib_kwargs = {}
calib_sig = inspect.signature(CalibratedClassifierCV.__init__)
if "estimator" in calib_sig.parameters:
    calib_kwargs["estimator"] = base_clf
elif "base_estimator" in calib_sig.parameters:
    calib_kwargs["base_estimator"] = base_clf
else:
    calib_kwargs["estimator"] = base_clf

calib_kwargs["cv"] = 3
calib_kwargs["method"] = "sigmoid"

try:
    svm_clf = CalibratedClassifierCV(**calib_kwargs)
    svm_clf.fit(X_train_s, y_train)

    svm_val_probs = svm_clf.predict_proba(X_val_s)[:, 1]
    svm_val_pred = (svm_val_probs >= 0.5).astype(int)

    print("SVM-like Validation AUC: {:.4f}".format(roc_auc_score(y_val, svm_val_probs)))
    print("SVM-like Validation ACC: {:.4f}".format(accuracy_score(y_val, svm_val_pred)))
except Exception as e:
    raise RuntimeError("Failed to train calibrated LinearSVC: {}".format(e))

# -------------------------
# TruncatedSVD -> MLP pipeline
# Ensure n_components is valid (<= n_features). Clamp automatically.
# -------------------------
n_features = X_train_s.shape[1]
max_valid = max(1, n_features - 1)  # keep at least 1 component
n_components = min(DESIRED_SVD_COMPONENTS, max_valid)

print("\nRequested SVD components:", DESIRED_SVD_COMPONENTS)
print("Number of features after preprocessing:", n_features)
print("Using n_components for SVD:", n_components)

print("\nReducing dimensionality with TruncatedSVD (n_components={})...".format(n_components))
svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
svd.fit(X_train_s)

X_train_reduced = svd.transform(X_train_s)
X_val_reduced   = svd.transform(X_val_s)
X_test_reduced  = svd.transform(X_test_s)

print("Reduced shapes (dense):", X_train_reduced.shape, X_val_reduced.shape, X_test_reduced.shape)

scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train_reduced)
X_val_nn   = scaler.transform(X_val_reduced)
X_test_nn  = scaler.transform(X_test_reduced)

print("\nTraining MLPClassifier on reduced features...")
mlp = MLPClassifier(hidden_layer_sizes=(128,64),
                    alpha=0.001,
                    max_iter=300,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                    verbose=False)

mlp.fit(X_train_nn, y_train)

mlp_val_probs = mlp.predict_proba(X_val_nn)[:, 1]
mlp_val_pred = (mlp_val_probs >= 0.5).astype(int)

print("MLP Validation AUC: {:.4f}".format(roc_auc_score(y_val, mlp_val_probs)))
print("MLP Validation ACC: {:.4f}".format(accuracy_score(y_val, mlp_val_pred)))

# -------------------------
# Test predictions & submission
# -------------------------
print("\nPredicting on test set and writing submission files...")
svm_test_probs = svm_clf.predict_proba(X_test_s)[:, 1]
mlp_test_probs = mlp.predict_proba(X_test_nn)[:, 1]
ensemble_probs = (svm_test_probs + mlp_test_probs) / 2.0

svm_sub = pd.DataFrame({IDCOL: test[IDCOL], TARGET: svm_test_probs})
nn_sub  = pd.DataFrame({IDCOL: test[IDCOL], TARGET: mlp_test_probs})
ens_sub = pd.DataFrame({IDCOL: test[IDCOL], TARGET: ensemble_probs})

# reorder to match sample submission if possible
try:
    svm_sub = svm_sub.set_index(IDCOL).loc[sample[IDCOL]].reset_index()
    nn_sub  = nn_sub.set_index(IDCOL).loc[sample[IDCOL]].reset_index()
    ens_sub = ens_sub.set_index(IDCOL).loc[sample[IDCOL]].reset_index()
except Exception:
    print("Warning: could not reindex to sample file order — writing in test order instead.")

svm_sub.to_csv("svm_submission.csv", index=False)
nn_sub.to_csv("nn_submission.csv", index=False)
ens_sub.to_csv("ensemble_submission.csv", index=False)

print("Saved files: svm_submission.csv, nn_submission.csv, ensemble_submission.csv")

# -------------------------
# Ensemble validation metrics
# -------------------------
try:
    ensemble_val_probs = (svm_val_probs + mlp_val_probs) / 2.0
    ensemble_val_pred = (ensemble_val_probs >= 0.5).astype(int)
    print("\nEnsemble Validation AUC: {:.4f}".format(roc_auc_score(y_val, ensemble_val_probs)))
    print("Ensemble Validation ACC: {:.4f}".format(accuracy_score(y_val, ensemble_val_pred)))
except Exception:
    print("Could not compute ensemble validation metrics (one model may have failed).")

print("\nDone. If you hit MemoryError or long runtimes, try lowering DESIRED_SVD_COMPONENTS (e.g. 50-100) or train on a subset.")

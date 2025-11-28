# ============================================================
# 1) UPLOAD FILES
# ============================================================

import pandas as pd
import io
from google.colab import files

print("Upload train_updated.csv, test_updated.csv, sample_submission_updated.csv")
uploaded = files.upload()

train_key  = [k for k in uploaded.keys() if "train_updated" in k][0]
test_key   = [k for k in uploaded.keys() if "test_updated" in k][0]
sample_key = [k for k in uploaded.keys() if "sample_submission" in k][0]

train_df = pd.read_csv(io.BytesIO(uploaded[train_key]))
test_df  = pd.read_csv(io.BytesIO(uploaded[test_key]))
sample_df = pd.read_csv(io.BytesIO(uploaded[sample_key]))

print("Train:", train_df.shape, " | Test:", test_df.shape)


# ============================================================
# 2) PREPROCESSING
# ============================================================

X = train_df.drop(["RiskFlag", "ProfileID"], axis=1)
y = train_df["RiskFlag"]

test_ids = test_df["ProfileID"]
X_test = test_df.drop(["ProfileID"], axis=1)

# One Hot Encode
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("X:", X.shape, " | X_test:", X_test.shape)


# ============================================================
# 3) TRAIN/VALIDATION SPLIT
# ============================================================

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ============================================================
# 4) DEFINE MODELS
# ============================================================

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Try XGBoost if installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

models = {}

models["LogisticRegression"] = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("model", LogisticRegression(max_iter=500))
])

models["RandomForest"] = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42))
])

models["GradientBoosting"] = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("model", GradientBoostingClassifier())
])

if HAS_XGB:
    models["XGBoost"] = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42))
    ])

models["NeuralNetwork"] = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("model", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=25,
        random_state=42))
])

models["LinearSVM"] = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("model", CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, random_state=42),
        method="sigmoid",
        cv=3))
])


# ============================================================
# 5) TRAIN MODELS + VALIDATE + SAVE SUBMISSIONS
# ============================================================

probability_files = {}
binary_files = {}

for name, model in models.items():
    print(f"\n Training {name} ...")
    model.fit(X_train, y_train)

    # Validation AUC
    val_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, val_pred)
    print(f"{name} AUC = {auc:.5f}")

    # Fit on FULL training set
    model.fit(X, y)

    # Test predictions (probabilities)
    test_pred = model.predict_proba(X_test)[:, 1]

    # --------------------------
    # SAVE PROBABILITY SUBMISSION
    # --------------------------
    prob_df = pd.DataFrame({
        "ProfileID": test_ids,
        "RiskFlag": test_pred
    })

    prob_filename = f"submission_{name.lower()}_prob.csv"
    prob_df.to_csv(prob_filename, index=False)
    probability_files[name] = prob_filename
    print(f"✔ Saved probability submission: {prob_filename}")

    # --------------------------
    # SAVE BINARY SUBMISSION (0/1)
    # --------------------------
    binary_df = prob_df.copy()
    binary_df["RiskFlag"] = (binary_df["RiskFlag"] >= 0.5).astype(int)

    binary_filename = f"submission_{name.lower()}_binary.csv"
    binary_df.to_csv(binary_filename, index=False)
    binary_files[name] = binary_filename
    print(f"✔ Saved binary submission: {binary_filename}")


print("\n DONE! ALL SUBMISSIONS READY.")

print("\n Probability submissions:")
for name, f in probability_files.items():
    print(f"{name:20s} -> {f}")

print("\n Binary submissions (0/1):")
for name, f in binary_files.items():
    print(f"{name:20s} -> {f}")

print("\n➡ Download any file using: files.download('filename')")

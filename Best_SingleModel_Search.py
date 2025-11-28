import pandas as pd
from google.colab import files
import io

print("Upload train_updated.csv, test_updated.csv, sample_submission_updated.csv")
uploaded = files.upload()

# Automatically detect correct uploaded filenames
train_key  = [k for k in uploaded.keys() if "train_updated" in k][0]
test_key   = [k for k in uploaded.keys() if "test_updated" in k][0]
sample_key = [k for k in uploaded.keys() if "sample_submission" in k][0]

print("Loaded files:")
print(train_key, test_key, sample_key)

# Read the data
train_df = pd.read_csv(io.BytesIO(uploaded[train_key]))
test_df  = pd.read_csv(io.BytesIO(uploaded[test_key]))
sample_submission = pd.read_csv(io.BytesIO(uploaded[sample_key]))

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# -----------------------------
# Preprocessing
# -----------------------------
X = train_df.drop(["RiskFlag", "ProfileID"], axis=1)
y = train_df["RiskFlag"]

test_ids = test_df["ProfileID"]
X_test = test_df.drop(["ProfileID"], axis=1)

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# -----------------------------
# Try multiple models
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
}

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    scores[name] = auc
    print(f"{name} AUC:", auc)

# Select best model
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

print("\nBEST MODEL:", best_model_name, "AUC =", scores[best_model_name])

# Train best model on full data
best_model.fit(X, y)
test_preds = best_model.predict_proba(X_test)[:, 1]

# -----------------------------
# Create Kaggle submission
# -----------------------------
submission = pd.DataFrame({
    "ProfileID": test_ids,
    "RiskFlag": test_preds   # probability 0–1 ✔
})

submission.to_csv("final_submission.csv", index=False)

print("Saved final_submission.csv")

files.download("final_submission.csv")

import pandas as pd
from google.colab import files

IN = "final_submission.csv"           # file created by previous cell
OUT = "final_submission_binary.csv"   # cleaned file for Kaggle

# Read the submission
df = pd.read_csv(IN)

# second column should be 'RiskFlag'
predcol = df.columns[1]

# Ensure numeric probabilities
df[predcol] = pd.to_numeric(df[predcol], errors="coerce").fillna(0.5)

# Threshold to convert probability → 0/1
threshold = 0.5
df[predcol] = (df[predcol] >= threshold).astype(int)

# Save cleaned file with correct header and no index
df.to_csv(OUT, index=False)

print("Saved binary submission to:", OUT)
print(df.head())

# Download to your laptop for Kaggle upload
files.download(OUT)

